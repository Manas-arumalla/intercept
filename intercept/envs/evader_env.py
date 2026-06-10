"""Adversarial-RL evader environment: learn to *evade* a pursuing interceptor.

The mirror image of :class:`~intercept.envs.interception_env.InterceptionEnv` — here the **agent is
the target**, maximizing the interceptor's miss distance (and survival time) against a fixed pursuit
law (True PN). This is a *learned* adversary that reacts to the pursuer every step, complementing
open-loop scripted maneuvers and the game-theoretic anti-LOS `optimal_evader` (game-theory.md):
self-play-like difficulty without a second learner. Same plant for both (fairness invariant); the
interceptor is faster, so capture is the default and the evader must *work* to maximize the miss.

* **Action** — scalar lateral acceleration ⟂ the evader's velocity, scaled to its g-limit.
* **Observation** — normalized relative kinematics *from the evader's view* (pursuer-relative
  position/velocity + own velocity).
* **Reward** — grow the interceptor's predicted zero-effort miss (opposite of the interceptor's
  objective) + a small survival bonus; a big penalty if caught, a miss bonus if it escapes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from intercept.core.aero import G0, AeroMissile2D
from intercept.core.frames import segment_min_distance
from intercept.core.integrators import integrate_rk4
from intercept.envs.interception_env import POS_SCALE, VEL_SCALE, lateral_acceleration
from intercept.guidance.pn import true_pn

try:
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True
except ImportError:  # pragma: no cover
    _HAS_GYM = False

Array = NDArray[np.float64]


def build_evader_observation(evader_state: Array, pursuer_state: Array) -> Array:
    """Normalized relative kinematics from the evader's view: pursuer-relative pos/vel + own vel."""
    e = np.asarray(evader_state, dtype=float)
    p = np.asarray(pursuer_state, dtype=float)
    r = (p[:2] - e[:2]) / POS_SCALE  # vector toward the pursuer
    v_rel = (p[2:4] - e[2:4]) / VEL_SCALE
    v_own = e[2:4] / VEL_SCALE
    return np.concatenate([r, v_rel, v_own]).astype(np.float32)


@dataclass
class EvaderReward:
    """Reward weights for the evader (maximize the interceptor's miss / survival)."""

    k_zem: float = 1.0  # reward per metre of predicted-miss (ZEM) increase
    k_survive: float = 0.02  # per-step survival bonus
    r_caught: float = 200.0  # penalty for being intercepted
    k_escape: float = 1.0  # bonus per metre of closest-approach achieved on escape
    miss_cap: float = 400.0  # cap on the rewarded closest approach


if _HAS_GYM:

    class EvaderEnv(gym.Env):
        """Single-evader environment: the agent flees a faster True-PN interceptor."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            reward: EvaderReward | None = None,
            *,
            interceptor_speed: float = 1000.0,
            evader_speed: float = 700.0,
            interceptor_a_max: float = 40 * G0,
            evader_a_max: float = 35 * G0,
            pursuer_N: float = 4.0,
            dt: float = 0.01,
            t_max: float = 16.0,
            kill_radius: float = 20.0,
            range_min: float = 5000.0,
            range_max: float = 8000.0,
            max_steps: int = 2000,
            pursuer_factory=None,
        ) -> None:
            super().__init__()
            self.reward_cfg = reward or EvaderReward()
            self.interceptor_speed = interceptor_speed
            self.evader_speed = evader_speed
            self.interceptor_a_max = interceptor_a_max
            self.evader_a_max = evader_a_max
            self.pursuer_N = pursuer_N
            self.dt = dt
            self.t_max = t_max
            self.kill_radius = kill_radius
            self.range_min = range_min
            self.range_max = range_max
            self.max_steps = max_steps
            self.observation_space = spaces.Box(-10.0, 10.0, shape=(6,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
            self._int_dyn = AeroMissile2D(a_max=interceptor_a_max, tau=0.2)
            self._eva_dyn = AeroMissile2D(a_max=evader_a_max, tau=0.3)
            # The pursuing interceptor's guidance — True PN by default, or an injected controller
            # (e.g. a learned RLGuidance) for self-play. Must chase the entity named "evader".
            self._pursuit = (
                pursuer_factory() if pursuer_factory is not None else true_pn("evader", N=pursuer_N)
            )

        def _zem_perp(self) -> float:
            """Interceptor's predicted perpendicular zero-effort miss on the evader (maximized)."""
            r = self.evader_state[:2] - self.int_state[:2]
            v = self.evader_state[2:4] - self.int_state[2:4]
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

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            rng = self.np_random
            downrange = float(rng.uniform(self.range_min, self.range_max))
            offset = float(rng.uniform(-1500.0, 1500.0))
            # Interceptor at origin aimed (lead) at the evader; evader heads roughly back toward it.
            evader_pos = np.array([downrange, offset])
            aim = evader_pos / np.linalg.norm(evader_pos)
            heading = np.arctan2(-aim[1], -aim[0]) + float(rng.uniform(-0.3, 0.3))
            ev_vel = self.evader_speed * np.array([np.cos(heading), np.sin(heading)])
            self.int_state = self._int_dyn.initial_state([0.0, 0.0], self.interceptor_speed * aim)
            self.evader_state = self._eva_dyn.initial_state(evader_pos, ev_vel)
            if hasattr(self._pursuit, "reset"):
                self._pursuit.reset()
            self.t = 0.0
            self.steps = 0
            r0 = self.evader_state[:2] - self.int_state[:2]
            self.prev_range = float(np.linalg.norm(r0))
            self.min_range = self.prev_range
            self.prev_zem = self._zem_perp()
            return build_evader_observation(self.evader_state, self.int_state), {}

        def step(self, action: Array):
            a = float(np.asarray(action, dtype=float).flatten()[0])
            u_eva = self._eva_dyn.saturate(
                lateral_acceleration(self.evader_state, a, self.evader_a_max)
            )
            # Interceptor pursues with True PN (homing on the evader).
            u_int = self._int_dyn.saturate(
                np.asarray(
                    self._pursuit(self.t, self.int_state, {"evader": self.evader_state}),
                    dtype=float,
                )
            )

            rel0 = self.evader_state[:2] - self.int_state[:2]
            self.int_state = integrate_rk4(self._int_dyn, self.t, self.int_state, u_int, self.dt)
            self.evader_state = integrate_rk4(
                self._eva_dyn, self.t, self.evader_state, u_eva, self.dt
            )
            self.t += self.dt
            self.steps += 1

            rel1 = self.evader_state[:2] - self.int_state[:2]
            rng = float(np.linalg.norm(rel1))
            seg_d = segment_min_distance(rel0, rel1)
            self.min_range = min(self.min_range, seg_d)
            cfg = self.reward_cfg
            zem = self._zem_perp()
            reward = cfg.k_zem * (zem - self.prev_zem) + cfg.k_survive
            self.prev_zem = zem

            terminated = truncated = False
            if seg_d <= self.kill_radius:
                reward -= cfg.r_caught
                terminated = True
            elif self.t > 0.5 and rng > self.prev_range:
                # Closest approach passed: the interceptor missed — the evader escaped.
                reward += cfg.k_escape * min(self.min_range, cfg.miss_cap)
                terminated = True
            elif self.t >= self.t_max or self.steps >= self.max_steps:
                reward += cfg.k_escape * min(self.min_range, cfg.miss_cap)
                truncated = True

            self.prev_range = rng
            info = {
                "range": rng,
                "intercepted": seg_d <= self.kill_radius,
                "min_range": self.min_range,
            }
            return (
                build_evader_observation(self.evader_state, self.int_state),
                float(reward),
                terminated,
                truncated,
                info,
            )


def has_gym() -> bool:
    """Whether Gymnasium is available (so the evader env can be constructed)."""
    return _HAS_GYM
