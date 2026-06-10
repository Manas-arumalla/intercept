"""Nonlinear Model-Predictive Control (NMPC) guidance via CasADi.

Each replan solves a finite-horizon optimal control problem — minimize terminal miss to the
predicted target plus control effort, subject to point-mass dynamics and an acceleration-magnitude
limit — and applies the first control (receding horizon). Unlike the closed-form laws, MPC handles
**constraints explicitly**: the optional terminal **impact-angle** objective steers the interceptor
to arrive along a chosen direction, which PN cannot do.

To stay real-time-plausible the solver is invoked every ``replan_every`` steps (event/time-
triggered replanning) and the first optimal command is held in between. Requires the optional
``casadi`` dependency (``pip install -e ".[mpc]"``); constructing this law without CasADi raises.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from intercept.guidance.base import EPS, GuidanceLaw

Array = NDArray[np.float64]

try:  # CasADi is an optional dependency
    import casadi as ca

    _HAS_CASADI = True
except ImportError:  # pragma: no cover - exercised only when casadi missing
    _HAS_CASADI = False


class MPCGuidance(GuidanceLaw):
    """Receding-horizon NMPC interceptor guidance (CasADi + IPOPT).

    Parameters
    ----------
    target:
        Name of the target entity to home on.
    a_max:
        Acceleration-magnitude limit applied as a hard constraint (m/s²).
    horizon:
        Maximum prediction horizon (s); the actual horizon is ``min(horizon, t_go)`` so the terminal
        cost aims at the predicted intercept.
    n_steps:
        Number of horizon discretization steps.
    w_terminal, w_effort, w_track:
        Weights for terminal miss, control effort, and along-horizon tracking.
    replan_every:
        Re-solve the OCP every this many control steps; hold the first command in between.
    impact_angle_deg:
        If set, also steer the terminal velocity toward this inertial heading (impact-angle term).
    w_angle:
        Weight for the impact-angle objective.
    """

    def __init__(
        self,
        target: str,
        a_max: float,
        *,
        horizon: float = 2.5,
        n_steps: int = 15,
        w_terminal: float = 10.0,
        w_effort: float = 1e-4,
        w_track: float = 0.0,
        replan_every: int = 5,
        impact_angle_deg: float | None = None,
        w_angle: float = 5.0,
    ) -> None:
        super().__init__(target)
        if not _HAS_CASADI:
            raise ImportError('MPCGuidance requires CasADi. Install with: pip install -e ".[mpc]"')
        if a_max <= 0 or horizon <= 0 or n_steps < 2:
            raise ValueError("require a_max > 0, horizon > 0, n_steps >= 2")
        self.a_max = float(a_max)
        self.horizon = float(horizon)
        self.n_steps = int(n_steps)
        self.w_terminal = float(w_terminal)
        self.w_effort = float(w_effort)
        self.w_track = float(w_track)
        self.replan_every = int(replan_every)
        self.impact_angle_deg = impact_angle_deg
        self.w_angle = float(w_angle)
        self._calls = 0
        self._held_u = np.zeros(2)
        self.last_plan: Array | None = None  # planned interceptor xy path (n_steps+1, 2)

    def reset(self) -> None:
        self._calls = 0
        self._held_u = np.zeros(2)
        self.last_plan = None

    def _solve(self, own_state: Array, target_state: Array) -> Array:
        r = target_state[:2] - own_state[:2]
        rng = float(np.linalg.norm(r))
        v_rel = target_state[2:4] - own_state[2:4]
        vc = -(r @ v_rel) / rng if rng > EPS else 0.0
        t_go = rng / vc if vc > EPS else self.horizon
        T = float(np.clip(t_go, 0.3, self.horizon))
        n = self.n_steps
        dt = T / n

        tgt_pos = target_state[:2]
        tgt_vel = target_state[2:4]

        opti = ca.Opti()
        X = opti.variable(4, n + 1)  # [x, y, vx, vy]
        U = opti.variable(2, n)  # [ax, ay]
        opti.subject_to(X[:, 0] == own_state[:4])

        cost = 0
        for k in range(n):
            # Euler point-mass dynamics.
            xn = X[:, k] + dt * ca.vertcat(X[2, k], X[3, k], U[0, k], U[1, k])
            opti.subject_to(X[:, k + 1] == xn)
            opti.subject_to(U[0, k] ** 2 + U[1, k] ** 2 <= self.a_max**2)
            if self.w_track > 0:
                tgt_k = tgt_pos + tgt_vel * (k * dt)
                cost += self.w_track * ca.sumsqr(X[:2, k] - tgt_k)
        cost += self.w_effort * ca.sumsqr(U)

        tgt_T = tgt_pos + tgt_vel * T
        cost += self.w_terminal * ca.sumsqr(X[:2, n] - tgt_T)
        if self.impact_angle_deg is not None:
            ang = np.radians(self.impact_angle_deg)
            speed = float(np.linalg.norm(own_state[2:4]))
            d_vel = speed * np.array([np.cos(ang), np.sin(ang)])
            # Match the full terminal velocity vector (direction + speed): unlike penalizing only
            # the perpendicular component, this breaks the +d / -d sign ambiguity.
            cost += self.w_angle * ca.sumsqr(X[2:4, n] - d_vel)

        opti.minimize(cost)
        opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "max_iter": 80})
        try:
            sol = opti.solve()
            u0 = np.array(sol.value(U[:, 0])).flatten()
            self.last_plan = np.array(sol.value(X[:2, :])).T
        except RuntimeError:
            # Solver failure: fall back to holding the previous command.
            u0 = self._held_u
        return u0

    def command(self, t: float, own_state: Array, target_state: Array) -> Array:
        if self._calls % self.replan_every == 0:
            self._held_u = self._solve(own_state, target_state)
        self._calls += 1
        return self._held_u


class MPCGuidance3D(GuidanceLaw):
    """3-D receding-horizon NMPC interceptor guidance (CasADi + IPOPT).

    The 3-D analogue of :class:`MPCGuidance`: minimize terminal miss to the predicted 3-D target
    plus control effort, subject to 3-D point-mass dynamics and an acceleration-magnitude limit,
    re-solving every ``replan_every`` steps. An optional ``impact_dir`` (a 3-vector inertial
    approach direction) adds a terminal-velocity-matching objective (3-D impact-angle control).
    """

    def __init__(
        self,
        target: str,
        a_max: float,
        *,
        horizon: float = 2.5,
        n_steps: int = 15,
        w_terminal: float = 10.0,
        w_effort: float = 1e-4,
        w_track: float = 0.0,
        replan_every: int = 5,
        impact_dir: Array | None = None,
        w_angle: float = 5.0,
    ) -> None:
        super().__init__(target)
        if not _HAS_CASADI:
            raise ImportError(
                'MPCGuidance3D requires CasADi. Install with: pip install -e ".[mpc]"'
            )
        if a_max <= 0 or horizon <= 0 or n_steps < 2:
            raise ValueError("require a_max > 0, horizon > 0, n_steps >= 2")
        self.a_max = float(a_max)
        self.horizon = float(horizon)
        self.n_steps = int(n_steps)
        self.w_terminal = float(w_terminal)
        self.w_effort = float(w_effort)
        self.w_track = float(w_track)
        self.replan_every = int(replan_every)
        self.impact_dir = None if impact_dir is None else np.asarray(impact_dir, dtype=float)
        self.w_angle = float(w_angle)
        self._calls = 0
        self._held_u = np.zeros(3)
        self.last_plan: Array | None = None  # planned interceptor xyz path (n_steps+1, 3)

    def reset(self) -> None:
        self._calls = 0
        self._held_u = np.zeros(3)
        self.last_plan = None

    def _solve(self, own_state: Array, target_state: Array) -> Array:
        own = np.asarray(own_state, dtype=float)
        tgt = np.asarray(target_state, dtype=float)
        r = tgt[:3] - own[:3]
        rng = float(np.linalg.norm(r))
        v_rel = tgt[3:6] - own[3:6]
        vc = -(r @ v_rel) / rng if rng > EPS else 0.0
        t_go = rng / vc if vc > EPS else self.horizon
        T = float(np.clip(t_go, 0.3, self.horizon))
        n = self.n_steps
        dt = T / n
        tgt_pos, tgt_vel = tgt[:3], tgt[3:6]

        opti = ca.Opti()
        X = opti.variable(6, n + 1)  # [x, y, z, vx, vy, vz]
        U = opti.variable(3, n)  # [ax, ay, az]
        opti.subject_to(X[:, 0] == own[:6])

        cost = 0
        for k in range(n):
            xn = X[:, k] + dt * ca.vertcat(X[3, k], X[4, k], X[5, k], U[0, k], U[1, k], U[2, k])
            opti.subject_to(X[:, k + 1] == xn)
            opti.subject_to(U[0, k] ** 2 + U[1, k] ** 2 + U[2, k] ** 2 <= self.a_max**2)
            if self.w_track > 0:
                tgt_k = tgt_pos + tgt_vel * (k * dt)
                cost += self.w_track * ca.sumsqr(X[:3, k] - tgt_k)
        cost += self.w_effort * ca.sumsqr(U)

        tgt_T = tgt_pos + tgt_vel * T
        cost += self.w_terminal * ca.sumsqr(X[:3, n] - tgt_T)
        if self.impact_dir is not None:
            speed = float(np.linalg.norm(own[3:6]))
            d = self.impact_dir / (np.linalg.norm(self.impact_dir) or 1.0)
            cost += self.w_angle * ca.sumsqr(X[3:6, n] - speed * d)

        opti.minimize(cost)
        opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "max_iter": 80})
        try:
            sol = opti.solve()
            u0 = np.array(sol.value(U[:, 0])).flatten()
            self.last_plan = np.array(sol.value(X[:3, :])).T
        except RuntimeError:
            u0 = self._held_u
        return u0

    def command(self, t: float, own_state: Array, target_state: Array) -> Array:
        if self._calls % self.replan_every == 0:
            self._held_u = self._solve(own_state, target_state)
        self._calls += 1
        return self._held_u


def has_casadi() -> bool:
    """Whether CasADi is available (so MPC guidance can be constructed)."""
    return _HAS_CASADI
