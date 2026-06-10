"""Gymnasium environment for learning interception guidance.

The environment wraps the *same* `PointMass2D` dynamics, `RK4` integrator, and scenario sampler the
classical laws and the benchmark use, so a learned policy can later be dropped into an `Engagement`
and compared fairly (the fairness invariant, ADR-0003 / ADR-0005). Training and evaluation share one
physics core; only the control source differs.

Design choices that make the problem well-posed (matching how guidance actually works):

* **Action** — a single scalar lateral acceleration, applied **perpendicular to the interceptor's
  velocity** and scaled to ``a_max`` (so speed is held constant, like a real seeker turning). This
  is the natural guidance control and far more learnable than an unconstrained 2-D acceleration.
* **Observation** — normalized relative kinematics (built by :func:`build_observation`, reused by
  the policy bridge so training and deployment see identical inputs).
* **Reward** — drive the **line-of-sight rate to zero** (the parallel-navigation principle) plus a
  range-closing term, − effort/time, with a terminal intercept bonus or capped closest-approach
  penalty. The shaping says *what* to optimize, not *how* — no PN formula is injected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True
except ImportError:  # pragma: no cover
    _HAS_GYM = False

from intercept.benchmark.scenario import ParametricScenario
from intercept.core.dynamics import PointMass2D
from intercept.core.frames import segment_min_distance
from intercept.core.integrators import integrate_rk4

Array = NDArray[np.float64]

POS_SCALE = 5000.0
VEL_SCALE = 1000.0
OBS_DIM = 6
RICH_OBS_DIM = 11
ACC_SCALE = 400.0


def build_observation(own_state: Array, target_state: Array) -> Array:
    """Normalized relative-kinematics observation shared by the env and the policy bridge.

    Components: relative position (2), relative velocity (2), own velocity (2) — each normalized.
    """
    own = np.asarray(own_state, dtype=float)
    tgt = np.asarray(target_state, dtype=float)
    r = (tgt[:2] - own[:2]) / POS_SCALE
    v_rel = (tgt[2:4] - own[2:4]) / VEL_SCALE
    v_own = own[2:4] / VEL_SCALE
    return np.concatenate([r, v_rel, v_own]).astype(np.float32)


def build_observation_rich(own_state: Array, target_state: Array) -> Array:
    """Richer observation adding the guidance-relevant scalars a real seeker/autopilot would use:
    LOS rate, closing speed, range, and the interceptor's own achieved lateral acceleration (the
    autopilot-lag state). This conditions the policy to learn lead/PN-like behavior under gravity
    and lag, where the minimal observation does not. (11-D.)
    """
    own = np.asarray(own_state, dtype=float)
    tgt = np.asarray(target_state, dtype=float)
    r = tgt[:2] - own[:2]
    v_rel = tgt[2:4] - own[2:4]
    rng = float(np.linalg.norm(r))
    closing = -(r @ v_rel) / rng if rng > 1e-9 else 0.0
    los_rate = (r[0] * v_rel[1] - r[1] * v_rel[0]) / (rng * rng) if rng > 1e-9 else 0.0
    own_acc = own[4:6] if own.shape[0] >= 6 else np.zeros(2)
    return np.concatenate(
        [
            r / POS_SCALE,
            v_rel / VEL_SCALE,
            own[2:4] / VEL_SCALE,
            [los_rate * 10.0, closing / VEL_SCALE, rng / POS_SCALE],
            own_acc / ACC_SCALE,
        ]
    ).astype(np.float32)


def lateral_acceleration(own_state: Array, action_scalar: float, a_max: float) -> Array:
    """Map a scalar action in ``[-1, 1]`` to a lateral acceleration ⟂ to the velocity."""
    v = np.asarray(own_state, dtype=float)[2:4]
    speed = float(np.linalg.norm(v))
    perp = np.array([-v[1], v[0]]) / speed if speed > 1e-6 else np.array([0.0, 1.0])
    return float(np.clip(action_scalar, -1.0, 1.0)) * a_max * perp


PN_BASELINE_N = 4.0


def pn_baseline_scalar(
    own_state: Array, target_state: Array, a_max: float, N: float = PN_BASELINE_N
) -> float:
    """Pure-PN lateral command expressed as a scalar in ``[-1, 1]`` — the residual-RL baseline.

    Pure PN (``a = N·Vc·λ̇`` applied ⟂ to the interceptor's velocity) uses *exactly* the same
    lateral parameterization as :func:`lateral_acceleration`, so passing this scalar reproduces pure
    PN. In **residual** mode the policy outputs a correction added on top, so even a zero action is
    competent PN — which is what prevents the from-scratch policy collapse on the lagged/gravity
    plant (the deferred failure; see ADR-0011). Returns 0 when the geometry is degenerate.
    """
    own = np.asarray(own_state, dtype=float)
    tgt = np.asarray(target_state, dtype=float)
    r = tgt[:2] - own[:2]
    v = tgt[2:4] - own[2:4]
    rng = float(np.linalg.norm(r))
    if rng < 1e-6 or a_max <= 0:
        return 0.0
    vc = -(r @ v) / rng  # closing speed (signed)
    lam_dot = (r[0] * v[1] - r[1] * v[0]) / (rng * rng)  # LOS rate
    return float(np.clip(N * vc * lam_dot / a_max, -1.0, 1.0))


def apn_baseline_scalar(
    own_state: Array, target_state: Array, a_max: float, N: float = PN_BASELINE_N
) -> float:
    """Augmented-PN baseline scalar — pure PN plus a target-acceleration feedforward.

    Adds the APN term ``(N/2)·a_T⊥`` to :func:`pn_baseline_scalar`, where ``a_T`` is the target's
    achieved lateral acceleration. For an aero/realistic target that acceleration is carried in the
    state (``state[4:6]`` in 2-D), so no finite differencing is needed; for a bare point-mass target
    (4-state) the feedforward is zero and this reduces to pure PN. Projected onto the same
    ⟂-velocity axis as the residual action — the stronger baseline for residual RL (ADR-0011).
    """
    own = np.asarray(own_state, dtype=float)
    tgt = np.asarray(target_state, dtype=float)
    base = pn_baseline_scalar(own, tgt, a_max, N)
    if tgt.shape[0] < 6 or a_max <= 0:
        return base
    v = own[2:4]
    speed = float(np.linalg.norm(v))
    if speed < 1e-6:
        return base
    perp = np.array([-v[1], v[0]]) / speed
    a_t = tgt[4:6]  # target achieved lateral acceleration
    aug = 0.5 * N * float(a_t @ perp) / a_max
    return float(np.clip(base + aug, -1.0, 1.0))


def gravity_feedforward(own_state: Array, gravity: float) -> Array:
    """Gravity-bias feed-forward: the ⟂-velocity acceleration that cancels gravity's perpendicular
    component (standard autopilot gravity compensation). Lets the policy learn only the *guidance*
    correction on top, instead of also fighting a constant bias. Zero when ``gravity == 0`` (L0)."""
    if gravity == 0.0:
        return np.zeros(2)
    v = np.asarray(own_state, dtype=float)[2:4]
    speed = float(np.linalg.norm(v))
    if speed < 1e-6:
        return np.array([0.0, gravity])
    v_hat = v / speed
    g_up = np.array([0.0, gravity])  # cancel gravity (which acts as [0, -g])
    return g_up - (g_up @ v_hat) * v_hat


@dataclass
class RewardConfig:
    """Weights for the shaped reward (tuned for point-mass interception).

    The dominant term is **potential-based zero-effort-miss (ZEM) shaping**: rewarding reduction of
    the predicted *perpendicular* miss. This is dense (a clear gradient even for near-head-on shots
    where the LOS rate is tiny) and telescopes to ``zem_initial − zem_final`` (so it cannot blow up
    per-step), directly teaching the policy to *lead* the target — which a range-closing or
    LOS-rate reward fails to bootstrap when the intercept bonus is too sparse to discover.
    """

    mode: str = "zem"  # "zem" (ZEM-potential) or "pn_shaping" (LOS-rate nulling, scaled)
    k_zem: float = 1.0  # [zem] reward per metre of perpendicular ZEM reduced
    k_effort: float = 0.0  # penalty on squared action
    k_time: float = 0.05  # per-step time penalty (encourages prompt intercept)
    r_hit: float = 200.0  # terminal intercept bonus
    k_miss: float = 0.5  # terminal miss penalty per metre of closest approach
    miss_cap: float = 300.0  # cap on the closest-approach used for the miss penalty
    # --- pn_shaping mode (bounded, well-scaled for large/fast engagements) ---
    k_progress: float = 8.0  # reward per unit of normalized range closed (Δrange / POS_SCALE)
    k_los: float = 3.0  # penalty per rad/s of |LOS rate| (rewards the parallel-nav behavior)


if _HAS_GYM:

    class InterceptionEnv(gym.Env):
        """Single-interceptor guidance environment over a `ParametricScenario`.

        Parameters
        ----------
        scenario:
            Engagement distribution, or a list of them; ``reset`` samples a fresh geometry/target
            (choosing a scenario uniformly at random when a list is given — a mixed curriculum that
            counters catastrophic forgetting).
        reward:
            Reward weights.
        max_steps:
            Hard episode cap (in addition to the scenario's ``t_max``).
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            scenario: ParametricScenario | list[ParametricScenario] | None = None,
            reward: RewardConfig | None = None,
            max_steps: int = 2000,
            obs_mode: str = "basic",
            action_mode: str = "absolute",
            residual_scale: float = 0.5,
            pn_N: float = PN_BASELINE_N,
            baseline: str = "pn",
            opponent=None,
            opponent_factory=None,
            sensor=None,
            estimator_factory=None,
        ) -> None:
            super().__init__()
            if scenario is None:
                scenario = ParametricScenario(name="rl_default")
            self._scenarios = scenario if isinstance(scenario, list) else [scenario]
            self.scenario = self._scenarios[0]
            self.reward_cfg = reward or RewardConfig()
            self.max_steps = max_steps
            self.obs_mode = obs_mode
            if action_mode not in ("absolute", "residual_pn"):
                raise ValueError("action_mode must be 'absolute' or 'residual_pn'")
            if baseline not in ("pn", "apn"):
                raise ValueError("baseline must be 'pn' or 'apn'")
            self.action_mode = action_mode
            self.residual_scale = float(residual_scale)
            self.pn_N = float(pn_N)
            self._baseline_fn = apn_baseline_scalar if baseline == "apn" else pn_baseline_scalar
            self.baseline = baseline
            # Optional opponent controller (t, target_state, world)->accel that overrides the
            # scenario's scripted maneuver — e.g. a learned RLEvader, enabling self-play.
            # ``opponent_factory`` (0-arg, returns a controller) is called *each reset* — e.g. to
            # sample from a pool of past opponents for fictitious / population self-play.
            self.opponent = opponent
            self.opponent_factory = opponent_factory
            # Optional noisy sensing: the policy *observes* a filtered estimate (seeker-on-
            # interceptor radar + EKF), while reward/intercept use truth — i.e. train RL on
            # estimated observations (ADR-0005 follow-up). None => perfect-state observations.
            self.sensor = sensor
            self.estimator_factory = estimator_factory
            self._estimating = sensor is not None and estimator_factory is not None
            self._estimator = None
            self._est_target = None
            self._obs_fn = build_observation_rich if obs_mode == "rich" else build_observation
            dim = RICH_OBS_DIM if obs_mode == "rich" else OBS_DIM
            self.observation_space = spaces.Box(-10.0, 10.0, shape=(dim,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

            self._interceptor_dyn = PointMass2D()
            self._target_dyn = PointMass2D()
            self._reset_state()

        def _reset_state(self) -> None:
            self.interceptor_state = np.zeros(4)
            self.target_state = np.zeros(4)
            self.target_ctrl = None
            self.a_max = self.scenario.interceptor_a_max
            self.gravity = 0.0
            self.dt = self.scenario.dt
            self.kill_radius = self.scenario.kill_radius
            self.t = 0.0
            self.steps = 0
            self.prev_range = 0.0
            self.min_range = 0.0

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            idx = int(self.np_random.integers(len(self._scenarios)))
            self.scenario = self._scenarios[idx]
            spec = self.scenario.sample(self.np_random)
            self.interceptor_state = spec.interceptor_state.copy()
            self.target_state = spec.target_state.copy()
            if self.opponent_factory is not None:
                self.target_ctrl = self.opponent_factory()  # fresh opponent sampled per reset
            elif self.opponent is not None:
                self.target_ctrl = self.opponent
            else:
                self.target_ctrl = spec.target_controller
            # Use the scenario's actual plants, so the policy can train on any fidelity level
            # (L0 point-mass or L2 aero with gravity/drag/g-limit/lag).
            self._interceptor_dyn = spec.interceptor_dynamics
            self._target_dyn = spec.target_dynamics
            self.a_max = spec.interceptor_a_max
            self.gravity = float(getattr(self._interceptor_dyn, "gravity", 0.0))
            self.dt = spec.dt
            self.kill_radius = spec.kill_radius
            self.t = 0.0
            self.steps = 0
            r0 = self.target_state[:2] - self.interceptor_state[:2]
            self.prev_range = float(np.linalg.norm(r0))
            self.min_range = self.prev_range
            self.prev_zem = self._zem_perp()
            self._estimator = None
            self._est_target = self.target_state.copy()
            if self._estimating:
                self._sense(init=True)
            return self._obs_fn(self.interceptor_state, self._obs_target()), {}

        def _obs_target(self) -> Array:
            """Target state the *policy* sees: the filtered estimate if sensing, else truth."""
            return self._est_target if self._estimating else self.target_state

        def _sense(self, init: bool = False) -> None:
            """Noisy seeker measurement (from the interceptor) + EKF → updated target estimate."""
            sp = self.interceptor_state[:2]
            z = self.sensor.measure(sp, self.target_state[:2], self.np_random)
            if init or self._estimator is None:
                invert = getattr(self.sensor, "invert", None)
                pos0 = (
                    np.asarray(invert(sp, z), float)[:2]
                    if invert is not None
                    else np.asarray(sp, float).copy()
                )
                x0 = np.concatenate([pos0, np.zeros(4)])
                p0 = np.diag([50.0**2, 50.0**2, 300.0**2, 300.0**2, 100.0**2, 100.0**2])
                self._estimator = self.estimator_factory(x0, p0)
            else:
                self._estimator.predict(self.dt)
                self._estimator.update(z, self.sensor, sp)
            self._est_target = np.asarray(self._estimator.target_state(), float)

        def _zem_perp(self) -> float:
            """Predicted perpendicular zero-effort miss (the quantity to drive to zero)."""
            r = self.target_state[:2] - self.interceptor_state[:2]
            v = self.target_state[2:4] - self.interceptor_state[2:4]
            rng = float(np.linalg.norm(r))
            if rng < 1e-6:
                return 0.0
            closing = -(r @ v) / rng
            if closing <= 1e-3:
                return rng  # receding/parallel: episode is ending; use range as the miss proxy
            t_go = min(rng / closing, 15.0)
            zem = r + v * t_go
            u = r / rng
            return float(np.linalg.norm(zem - (zem @ u) * u))

        def _los_rate(self) -> float:
            r = self.target_state[:2] - self.interceptor_state[:2]
            v = self.target_state[2:4] - self.interceptor_state[2:4]
            rng2 = float(r @ r)
            if rng2 < 1e-9:
                return 0.0
            return float(r[0] * v[1] - r[1] * v[0]) / rng2

        def step(self, action: Array):
            a = float(np.asarray(action, dtype=float).flatten()[0])
            # Residual mode: the learned action is a *correction* on a pure-PN baseline, so a zero
            # action is already competent PN (no policy collapse on the lagged/gravity plant).
            if self.action_mode == "residual_pn":
                base = self._baseline_fn(
                    self.interceptor_state, self._obs_target(), self.a_max, self.pn_N
                )
                a_eff = float(np.clip(base + self.residual_scale * a, -1.0, 1.0))
            else:
                a_eff = a
            cmd = lateral_acceleration(
                self.interceptor_state, a_eff, self.a_max
            ) + gravity_feedforward(self.interceptor_state, self.gravity)
            u_int = self._interceptor_dyn.saturate(cmd)
            world = {"target": self.target_state, "interceptor": self.interceptor_state}
            u_tgt = (
                np.asarray(self.target_ctrl(self.t, self.target_state, world), dtype=float)
                if self.target_ctrl is not None
                else np.zeros(2)
            )

            rel0 = self.target_state[:2] - self.interceptor_state[:2]
            self.interceptor_state = integrate_rk4(
                self._interceptor_dyn, self.t, self.interceptor_state, u_int, self.dt
            )
            self.target_state = integrate_rk4(
                self._target_dyn, self.t, self.target_state, u_tgt, self.dt
            )
            self.t += self.dt
            self.steps += 1

            rel1 = self.target_state[:2] - self.interceptor_state[:2]
            rng = float(np.linalg.norm(rel1))
            seg_d = segment_min_distance(rel0, rel1)  # closest approach within the step
            cfg = self.reward_cfg
            if cfg.mode == "pn_shaping":
                # Bounded, well-scaled dense reward (robust at large ranges / high speeds):
                # reward closing (normalized) and penalize LOS rate (rewards parallel navigation).
                reward = (
                    cfg.k_progress * (self.prev_range - rng) / POS_SCALE
                    - cfg.k_los * abs(self._los_rate())
                    - cfg.k_effort * a * a
                    - cfg.k_time
                )
                miss_pen = cfg.k_miss * min(seg_d, cfg.miss_cap) / POS_SCALE
            else:
                zem = self._zem_perp()
                reward = cfg.k_zem * (self.prev_zem - zem) - cfg.k_effort * a * a - cfg.k_time
                self.prev_zem = zem
                miss_pen = cfg.k_miss * min(seg_d, cfg.miss_cap)

            terminated = False
            truncated = False
            if seg_d <= self.kill_radius:
                reward += cfg.r_hit
                terminated = True
            elif self.t > 0.5 and rng > self.prev_range:
                # Range has begun to increase => closest approach passed => miss.
                reward -= miss_pen
                terminated = True
            elif self.t >= self.scenario.t_max or self.steps >= self.max_steps:
                reward -= miss_pen
                truncated = True

            self.min_range = min(self.min_range, seg_d)
            info = {
                "range": rng,
                "intercepted": seg_d <= self.kill_radius,
                "min_range": self.min_range,
            }
            self.prev_range = rng
            if self._estimating:
                self._sense()  # update the seeker/EKF estimate
            obs = self._obs_fn(self.interceptor_state, self._obs_target())
            return obs, float(reward), terminated, truncated, info


def has_gym() -> bool:
    """Whether Gymnasium is available (so the env can be constructed)."""
    return _HAS_GYM
