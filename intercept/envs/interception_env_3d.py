"""3-D Gymnasium environment for learning interception guidance.

The 3-D analogue of :class:`~intercept.envs.interception_env.InterceptionEnv` (ADR-0005 fairness
invariant): the *same* 3-D dynamics, RK4 integrator, and `ParametricScenario3D` sampler the 3-D
classical laws and benchmark use, so a learned 3-D policy drops into an `Engagement` and is compared
fairly. Differences from 2-D are only what dimensionality forces:

* **Action** — a **2-DOF** lateral acceleration in the plane perpendicular to the interceptor's
  velocity (the realizable lateral control in 3-D), spanned by an orthonormal basis ``(e1, e2)`` and
  scaled to ``a_max`` (norm-clipped so it stays within the achievable disc). This holds speed and is
  far more learnable than an unconstrained 3-D acceleration.
* **Observation** — normalized 3-D relative kinematics (rich mode adds the LOS-rate-vector
  magnitude, closing speed, range, and the interceptor's achieved lateral accel).
* **Reward** — potential-based 3-D zero-effort-miss shaping (drive the perpendicular ZEM to zero),
  − effort/time, with a terminal intercept bonus or capped closest-approach penalty.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from intercept.envs.interception_env import POS_SCALE, VEL_SCALE, RewardConfig

try:
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True
except ImportError:  # pragma: no cover
    _HAS_GYM = False

from intercept.benchmark.scenario3d import ParametricScenario3D
from intercept.core.dynamics3d import PointMass3D
from intercept.core.frames import segment_min_distance
from intercept.core.integrators import integrate_rk4

Array = NDArray[np.float64]

ACC_SCALE = 400.0
OBS_DIM_3D = 9
RICH_OBS_DIM_3D = 14
_EPS = 1e-9


def lateral_basis_3d(v: Array) -> tuple[Array, Array]:
    """Two orthonormal vectors spanning the plane perpendicular to velocity ``v``."""
    speed = float(np.linalg.norm(v))
    v_hat = v / speed if speed > _EPS else np.array([1.0, 0.0, 0.0])
    ref = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(v_hat, ref)
    if float(np.linalg.norm(e1)) < 1e-6:  # velocity nearly vertical
        e1 = np.cross(v_hat, np.array([0.0, 1.0, 0.0]))
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(v_hat, e1)
    return e1, e2


def build_observation_3d(own_state: Array, target_state: Array) -> Array:
    """Normalized 3-D relative-kinematics observation: rel pos (3), rel vel (3), own vel (3)."""
    own = np.asarray(own_state, dtype=float)
    tgt = np.asarray(target_state, dtype=float)
    r = (tgt[:3] - own[:3]) / POS_SCALE
    v_rel = (tgt[3:6] - own[3:6]) / VEL_SCALE
    v_own = own[3:6] / VEL_SCALE
    return np.concatenate([r, v_rel, v_own]).astype(np.float32)


def build_observation_3d_rich(own_state: Array, target_state: Array) -> Array:
    """Richer 14-D obs: basic 9-D + LOS-rate vector (3), closing speed, range."""
    own = np.asarray(own_state, dtype=float)
    tgt = np.asarray(target_state, dtype=float)
    r = tgt[:3] - own[:3]
    v_rel = tgt[3:6] - own[3:6]
    rng = float(np.linalg.norm(r))
    closing = -(r @ v_rel) / rng if rng > _EPS else 0.0
    omega = np.cross(r, v_rel) / (rng * rng) if rng > _EPS else np.zeros(3)
    return np.concatenate(
        [
            r / POS_SCALE,
            v_rel / VEL_SCALE,
            own[3:6] / VEL_SCALE,
            omega * 10.0,
            [closing / VEL_SCALE, rng / POS_SCALE],
        ]
    ).astype(np.float32)


def lateral_acceleration_3d(own_state: Array, action: Array, a_max: float) -> Array:
    """Map a 2-D action in ``[-1, 1]²`` to a lateral acceleration in the ⟂-velocity plane."""
    a = np.asarray(action, dtype=float).flatten()[:2]
    mag = float(np.linalg.norm(a))
    if mag > 1.0:
        a = a / mag  # clip to the unit disc (within a_max)
    e1, e2 = lateral_basis_3d(np.asarray(own_state, dtype=float)[3:6])
    return a_max * (a[0] * e1 + a[1] * e2)


PN_BASELINE_N = 4.0


def pn_baseline_action_3d(
    own_state: Array, target_state: Array, a_max: float, N: float = PN_BASELINE_N
) -> Array:
    """True-PN-3D command expressed as a 2-DOF action in the ⟂-velocity ``(e1, e2)`` basis.

    The residual-RL baseline for 3-D: ``a = N·(Ω × v_c)`` (realizable true PN) projected onto the
    same lateral basis :func:`lateral_acceleration_3d` uses and divided by ``a_max`` (clipped to the
    unit disc), so a zero residual reproduces PN-3D — preventing the from-scratch policy collapse on
    the lagged/gravity plant (ADR-0011/0014). ``Ω = (r×v_rel)/|r|²``, ``v_c = v_own − v_tgt``.
    """
    own = np.asarray(own_state, dtype=float)
    tgt = np.asarray(target_state, dtype=float)
    r = tgt[:3] - own[:3]
    v_rel = tgt[3:6] - own[3:6]
    rng2 = float(r @ r)
    if rng2 < 1e-12 or a_max <= 0:
        return np.zeros(2)
    omega = np.cross(r, v_rel) / rng2
    v_c = own[3:6] - tgt[3:6]
    a_pn = N * np.cross(omega, v_c)
    e1, e2 = lateral_basis_3d(own[3:6])
    a = np.array([a_pn @ e1, a_pn @ e2]) / a_max
    mag = float(np.linalg.norm(a))
    return a / mag if mag > 1.0 else a


def apn_baseline_action_3d(
    own_state: Array, target_state: Array, a_max: float, N: float = PN_BASELINE_N
) -> Array:
    """Augmented-PN-3D command as a 2-DOF action in the ⟂-velocity ``(e1, e2)`` basis.

    Adds the APN target-acceleration feedforward ``(N/2)·a_T⊥`` to :func:`pn_baseline_action_3d`,
    where ``a_T`` is the target's achieved lateral acceleration (``state[6:9]`` for an aero target;
    zero for a 6-state point mass, reducing to PN). The stronger residual baseline for the
    unpredictable cases (ADR-0014)."""
    own = np.asarray(own_state, dtype=float)
    tgt = np.asarray(target_state, dtype=float)
    r = tgt[:3] - own[:3]
    rng2 = float(r @ r)
    if rng2 < 1e-12 or a_max <= 0:
        return np.zeros(2)
    v_rel = tgt[3:6] - own[3:6]
    omega = np.cross(r, v_rel) / rng2
    v_c = own[3:6] - tgt[3:6]
    a_t = tgt[6:9] if tgt.shape[0] >= 9 else np.zeros(3)
    a_apn = N * np.cross(omega, v_c) + 0.5 * N * a_t
    e1, e2 = lateral_basis_3d(own[3:6])
    a = np.array([a_apn @ e1, a_apn @ e2]) / a_max
    mag = float(np.linalg.norm(a))
    return a / mag if mag > 1.0 else a


def gravity_feedforward_3d(own_state: Array, gravity: float) -> Array:
    """Gravity-bias feed-forward: the ⟂-velocity acceleration cancelling gravity's perpendicular
    component (gravity acts along −z). Zero when ``gravity == 0``."""
    if gravity == 0.0:
        return np.zeros(3)
    v = np.asarray(own_state, dtype=float)[3:6]
    speed = float(np.linalg.norm(v))
    if speed < _EPS:
        return np.array([0.0, 0.0, gravity])
    v_hat = v / speed
    g_up = np.array([0.0, 0.0, gravity])  # cancel gravity ([0,0,-g])
    return g_up - (g_up @ v_hat) * v_hat


if _HAS_GYM:

    class InterceptionEnv3D(gym.Env):
        """Single-interceptor 3-D guidance environment over a `ParametricScenario3D`."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            scenario: ParametricScenario3D | list[ParametricScenario3D] | None = None,
            reward: RewardConfig | None = None,
            max_steps: int = 3000,
            obs_mode: str = "rich",
            action_mode: str = "absolute",
            residual_scale: float = 0.35,
            pn_N: float = PN_BASELINE_N,
            baseline: str = "pn",
        ) -> None:
            super().__init__()
            if scenario is None:
                scenario = ParametricScenario3D(name="rl3d_default")
            if action_mode not in ("absolute", "residual_pn"):
                raise ValueError("action_mode must be 'absolute' or 'residual_pn'")
            if baseline not in ("pn", "apn"):
                raise ValueError("baseline must be 'pn' or 'apn'")
            self._scenarios = scenario if isinstance(scenario, list) else [scenario]
            self.scenario = self._scenarios[0]
            self.reward_cfg = reward or RewardConfig()
            self.max_steps = max_steps
            self.obs_mode = obs_mode
            self.action_mode = action_mode
            self.residual_scale = float(residual_scale)
            self.pn_N = float(pn_N)
            self._baseline_fn = (
                apn_baseline_action_3d if baseline == "apn" else pn_baseline_action_3d
            )
            self.baseline = baseline
            self._obs_fn = build_observation_3d_rich if obs_mode == "rich" else build_observation_3d
            dim = RICH_OBS_DIM_3D if obs_mode == "rich" else OBS_DIM_3D
            self.observation_space = spaces.Box(-10.0, 10.0, shape=(dim,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            self._interceptor_dyn = PointMass3D()
            self._target_dyn = PointMass3D()
            self.t = 0.0

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            idx = int(self.np_random.integers(len(self._scenarios)))
            self.scenario = self._scenarios[idx]
            spec = self.scenario.sample(self.np_random)
            self.interceptor_state = spec.interceptor_state.copy()
            self.target_state = spec.target_state.copy()
            self.target_ctrl = spec.target_controller
            self._interceptor_dyn = spec.interceptor_dynamics
            self._target_dyn = spec.target_dynamics
            self.a_max = spec.interceptor_a_max
            self.gravity = float(getattr(self._interceptor_dyn, "gravity", 0.0))
            self.dt = spec.dt
            self.kill_radius = spec.kill_radius
            self.t = 0.0
            self.steps = 0
            rel0 = self.target_state[:3] - self.interceptor_state[:3]
            self.prev_range = float(np.linalg.norm(rel0))
            self.min_range = self.prev_range
            self.prev_zem = self._zem_perp()
            return self._obs_fn(self.interceptor_state, self.target_state), {}

        def _zem_perp(self) -> float:
            r = self.target_state[:3] - self.interceptor_state[:3]
            v = self.target_state[3:6] - self.interceptor_state[3:6]
            rng = float(np.linalg.norm(r))
            if rng < 1e-6:
                return 0.0
            closing = -(r @ v) / rng
            if closing <= 1e-3:
                return rng
            t_go = min(rng / closing, 15.0)
            zem = r + v * t_go
            u = r / rng
            return float(np.linalg.norm(zem - (zem @ u) * u))

        def step(self, action: Array):
            act = np.asarray(action, dtype=float).flatten()[:2]
            if self.action_mode == "residual_pn":
                base = self._baseline_fn(
                    self.interceptor_state, self.target_state, self.a_max, self.pn_N
                )
                act = base + self.residual_scale * act  # lateral_accel norm-clips to the disc
            cmd = lateral_acceleration_3d(
                self.interceptor_state, act, self.a_max
            ) + gravity_feedforward_3d(self.interceptor_state, self.gravity)
            u_int = self._interceptor_dyn.saturate(cmd)
            world = {"target": self.target_state, "interceptor": self.interceptor_state}
            u_tgt = (
                np.asarray(self.target_ctrl(self.t, self.target_state, world), dtype=float)
                if self.target_ctrl is not None
                else np.zeros(3)
            )

            rel0 = self.target_state[:3] - self.interceptor_state[:3]
            self.interceptor_state = integrate_rk4(
                self._interceptor_dyn, self.t, self.interceptor_state, u_int, self.dt
            )
            self.target_state = integrate_rk4(
                self._target_dyn, self.t, self.target_state, u_tgt, self.dt
            )
            self.t += self.dt
            self.steps += 1

            rel1 = self.target_state[:3] - self.interceptor_state[:3]
            rng = float(np.linalg.norm(rel1))
            seg_d = segment_min_distance(rel0, rel1)
            cfg = self.reward_cfg
            a_mag2 = float(np.sum(np.square(np.asarray(action, dtype=float).flatten()[:2])))
            zem = self._zem_perp()
            reward = cfg.k_zem * (self.prev_zem - zem) - cfg.k_effort * a_mag2 - cfg.k_time
            self.prev_zem = zem

            terminated = truncated = False
            if seg_d <= self.kill_radius:
                reward += cfg.r_hit
                terminated = True
            elif self.t > 0.5 and rng > self.prev_range:
                reward -= cfg.k_miss * min(seg_d, cfg.miss_cap)
                terminated = True
            elif self.t >= self.scenario.t_max or self.steps >= self.max_steps:
                reward -= cfg.k_miss * min(seg_d, cfg.miss_cap)
                truncated = True

            self.min_range = min(self.min_range, seg_d)
            info = {
                "range": rng,
                "intercepted": seg_d <= self.kill_radius,
                "min_range": self.min_range,
            }
            self.prev_range = rng
            return (
                self._obs_fn(self.interceptor_state, self.target_state),
                float(reward),
                terminated,
                truncated,
                info,
            )


def has_gym() -> bool:
    """Whether Gymnasium is available (so the 3-D env can be constructed)."""
    return _HAS_GYM
