"""Optimal Guidance Law (OGL) — the linear-quadratic / zero-effort-miss optimal law.

For the linearized intercept problem, minimizing control energy ∫a² dt subject to zero terminal
miss yields the closed-form feedback law

    a = N' · ZEM⊥ / t_go²,        N' = 3,

where ``ZEM = r + v·t_go`` is the zero-effort miss and ``ZEM⊥`` its component perpendicular to the
line of sight. With a known (constant) target acceleration the optimal law gains an extra term
``+ (N'/2) · a_T⊥`` — the same structure as Augmented PN, but here ``N'`` follows from the optimal-
control derivation rather than being chosen. References: Bryson & Ho, *Applied Optimal Control*;
Zarchan, *Tactical and Strategic Missile Guidance* (optimal-guidance chapter).

This is mathematically a sibling of ZEM-PN (``N=3``); it is provided as a distinct, documented
*optimal-control* baseline (with optional target-acceleration augmentation and a tunable effective
navigation ratio) so the benchmark can compare "classical PN" against "optimal guidance" explicitly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from intercept.core import frames3d as f3
from intercept.guidance.base import EPS, GuidanceLaw

Array = NDArray[np.float64]


class OptimalGuidance(GuidanceLaw):
    """Energy-optimal (LQ / ZEM) guidance with optional target-acceleration augmentation.

    Parameters
    ----------
    target:
        Name of the target entity to home on.
    n_prime:
        Effective navigation ratio (3.0 = energy-optimal with a hard terminal-miss constraint;
        larger values penalize miss more heavily / respond faster).
    augment:
        If ``True``, add the optimal target-acceleration feedforward ``(n_prime/2) a_T⊥`` using a
        finite-difference estimate of the target acceleration.
    """

    def __init__(self, target: str, n_prime: float = 3.0, augment: bool = False) -> None:
        super().__init__(target)
        if n_prime <= 0:
            raise ValueError("n_prime must be positive")
        self.n_prime = float(n_prime)
        self.augment = augment
        self._prev_t: float | None = None
        self._prev_vt: Array | None = None
        self._a_t = np.zeros(2)

    def reset(self) -> None:
        self._prev_t = None
        self._prev_vt = None
        self._a_t = np.zeros(2)

    def _target_accel(self, t: float, v_target: Array) -> Array:
        if self._prev_t is not None and self._prev_vt is not None:
            dt = t - self._prev_t
            if dt > EPS:
                self._a_t = (v_target - self._prev_vt) / dt
        self._prev_t = t
        self._prev_vt = v_target.copy()
        return self._a_t

    def command(self, t: float, own_state: Array, target_state: Array) -> Array:
        r, v, rng = self._rel(own_state, target_state)
        a_t = self._target_accel(t, target_state[2:4]) if self.augment else np.zeros(2)
        if rng < EPS:
            return np.zeros(2)

        vc = self._closing_speed(r, v, rng)
        if vc <= EPS:
            return np.zeros(2)
        t_go = rng / vc

        u_los = r / rng
        zem = r + v * t_go
        zem_perp = zem - (zem @ u_los) * u_los
        a = self.n_prime * zem_perp / (t_go * t_go)

        if self.augment:
            perp = self._los_perp(r, rng)
            a = a + 0.5 * self.n_prime * float(a_t @ perp) * perp
        return a


def optimal_guidance(target: str, augment: bool = False) -> OptimalGuidance:
    """Convenience factory for the energy-optimal guidance law (N'=3)."""
    return OptimalGuidance(target, n_prime=3.0, augment=augment)


class OptimalGuidance3D(GuidanceLaw):
    """3-D energy-optimal (LQ / ZEM) guidance: ``a = N' · ZEM⊥ / t_go²`` on 3-D states.

    The 2-D ZEM law generalizes directly with vector kinematics: ``ZEM = r + v·t_go`` and ``ZEM⊥``
    its component perpendicular to the 3-D line of sight. With ``augment`` the optimal target-
    acceleration feedforward ``(N'/2) a_T⊥`` is added (finite-difference estimate). N'=3 is
    energy-optimal under a hard terminal-miss constraint.
    """

    def __init__(self, target: str, n_prime: float = 3.0, augment: bool = False) -> None:
        super().__init__(target)
        if n_prime <= 0:
            raise ValueError("n_prime must be positive")
        self.n_prime = float(n_prime)
        self.augment = augment
        self._prev_t: float | None = None
        self._prev_vt: Array | None = None
        self._a_t = np.zeros(3)

    def reset(self) -> None:
        self._prev_t = None
        self._prev_vt = None
        self._a_t = np.zeros(3)

    def command(self, t: float, own_state: Array, target_state: Array) -> Array:
        v_target = np.asarray(target_state, dtype=float)[3:6]
        if self.augment and self._prev_t is not None and self._prev_vt is not None:
            dt = t - self._prev_t
            if dt > EPS:
                self._a_t = (v_target - self._prev_vt) / dt
        self._prev_t, self._prev_vt = t, v_target.copy()

        r, v = f3.relative_state(own_state, target_state)
        rng = float(np.linalg.norm(r))
        if rng < EPS:
            return np.zeros(3)
        vc = f3.closing_speed(own_state, target_state)
        if vc <= EPS:
            return np.zeros(3)
        t_go = rng / vc

        u_los = r / rng
        zem = r + v * t_go
        zem_perp = zem - (zem @ u_los) * u_los
        a = self.n_prime * zem_perp / (t_go * t_go)
        if self.augment:
            a_t_perp = self._a_t - (self._a_t @ u_los) * u_los
            a = a + 0.5 * self.n_prime * a_t_perp
        return a


def optimal_guidance_3d(target: str, augment: bool = False) -> OptimalGuidance3D:
    """Convenience factory for 3-D energy-optimal guidance (N'=3)."""
    return OptimalGuidance3D(target, n_prime=3.0, augment=augment)
