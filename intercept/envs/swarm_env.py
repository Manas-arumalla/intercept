"""Centralized multi-interceptor swarm environment — *learned* cooperative target allocation.

A single policy observes the whole engagement (every interceptor relative to every threat) and emits
each decision step an ``N×M`` preference matrix; each living interceptor is assigned its highest-
preference living threat (a learned weapon-target assignment) and homes on it with True PN. The team
reward rewards intercepts and penalizes leakers — so the policy must learn to *spread out and
cooperate* (not all chase the same threat), the job the Hungarian allocator does analytically. This
allows a comparison of **learned coordination vs. the optimization baseline** on identical dynamics
(fairness invariant) at realistic comparable speeds (threats ~Mach 2, interceptors ~Mach 3 — a
~1.45x closing edge). PN does the guidance, so there is no from-scratch-control collapse; the
learned part is purely the cooperative allocation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from intercept.core.aero import G0
from intercept.core.dynamics import PointMass2D
from intercept.core.integrators import integrate_rk4
from intercept.guidance.pn import true_pn

try:
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True
except ImportError:  # pragma: no cover
    _HAS_GYM = False

Array = NDArray[np.float64]
POS_SCALE, VEL_SCALE = 12000.0, 1200.0
INT_SPEED, THREAT_SPEED = 1000.0, 700.0  # ~Mach 3 vs ~Mach 2 (realistic ~1.45x)
INT_AMAX, THREAT_AMAX = 45 * G0, 25 * G0


def has_gym() -> bool:
    return _HAS_GYM


@dataclass
class SwarmReward:
    k_hit: float = 30.0  # per threat intercepted
    k_leak: float = 40.0  # per threat that reaches the defended point
    k_close: float = 2.0  # dense: team closing on assigned threats (Δ min-range / POS_SCALE)
    k_time: float = 0.02  # per-step time penalty


def sample_swarm(rng: np.random.Generator, n_int: int, n_threat: int):
    """Threats inbound from a fan toward the origin; interceptors from a battery near it."""
    threats = []
    azis = np.linspace(40.0, 140.0, n_threat) + rng.uniform(-5, 5, n_threat)
    for j in range(n_threat):
        az = np.radians(azis[j])
        rngm = rng.uniform(8000.0, 11000.0)
        pos = rngm * np.array([np.cos(az), np.sin(az)])
        aim = -pos / np.linalg.norm(pos)
        threats.append(np.array([*pos, *(THREAT_SPEED * aim)]))
    centroid = np.mean([t[:2] for t in threats], axis=0)
    ints = []
    for _ in range(n_int):
        p = rng.uniform(-600, 600, 2)
        aim = (centroid - p) / np.linalg.norm(centroid - p)
        ints.append(np.array([*p, *(INT_SPEED * aim)]))
    return ints, threats


if _HAS_GYM:

    class CentralizedSwarmEnv(gym.Env):
        """N-interceptor vs M-threat area defense; the policy emits an N×M allocation each step."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            n_int: int = 4,
            n_threat: int = 4,
            reward: SwarmReward | None = None,
            dt: float = 0.05,
            t_max: float = 30.0,
            kill_radius: float = 60.0,
            leak_radius: float = 250.0,
            reassign_every: int = 4,
            weave_g: float = 6.0,
        ) -> None:
            super().__init__()
            self.n_int, self.n_threat = n_int, n_threat
            self.reward_cfg = reward or SwarmReward()
            self.dt, self.t_max = dt, t_max
            self.kill_radius, self.leak_radius = kill_radius, leak_radius
            self.reassign_every = reassign_every
            self.weave_g = weave_g
            self._idyn = PointMass2D(a_max=INT_AMAX)
            self._tdyn = PointMass2D(a_max=THREAT_AMAX)
            obs_dim = n_int * n_threat * 4 + n_int + n_threat
            self.observation_space = spaces.Box(-10.0, 10.0, shape=(obs_dim,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(n_int * n_threat,), dtype=np.float32)
            self._reset_state()

        def _reset_state(self) -> None:
            self.ipos = np.zeros((self.n_int, 4))
            self.tpos = np.zeros((self.n_threat, 4))
            self.alive_i = np.ones(self.n_int, bool)
            self.alive_t = np.ones(self.n_threat, bool)
            self.t = 0.0
            self.steps = 0
            self.assign = np.zeros(self.n_int, int)
            self._tphase = np.zeros(self.n_threat)
            self.n_hits = 0
            self.n_leaks = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            ints, threats = sample_swarm(self.np_random, self.n_int, self.n_threat)
            self._reset_state()
            self.ipos = np.array(ints)
            self.tpos = np.array(threats)
            self._tphase = self.np_random.uniform(0, 2 * np.pi, self.n_threat)
            self.prev_minrange = self._total_min_range()
            return self._obs(), {}

        def _total_min_range(self) -> float:
            tot = 0.0
            for j in range(self.n_threat):
                if not self.alive_t[j]:
                    continue
                d = [
                    np.linalg.norm(self.tpos[j, :2] - self.ipos[i, :2])
                    for i in range(self.n_int)
                    if self.alive_i[i]
                ]
                tot += min(d) if d else 0.0
            return tot

        def _obs(self) -> Array:
            parts = []
            for i in range(self.n_int):
                for j in range(self.n_threat):
                    r = (self.tpos[j, :2] - self.ipos[i, :2]) / POS_SCALE
                    v = (self.tpos[j, 2:4] - self.ipos[i, 2:4]) / VEL_SCALE
                    live = self.alive_i[i] and self.alive_t[j]
                    parts.append(np.concatenate([r, v]) if live else np.zeros(4))
            obs = np.concatenate(
                [np.concatenate(parts), self.alive_i.astype(float), self.alive_t.astype(float)]
            )
            return obs.astype(np.float32)

        def _threat_accel(self, j: int) -> Array:
            """Inbound threat with a light horizontal weave (its own evasion)."""
            v = self.tpos[j, 2:4]
            s = float(np.linalg.norm(v))
            if s < 1e-6:
                return np.zeros(2)
            perp = np.array([-v[1], v[0]]) / s
            return self.weave_g * G0 * np.sin(1.5 * self.t + self._tphase[j]) * perp

        def step(self, action: Array):
            pref = np.asarray(action, float).reshape(self.n_int, self.n_threat)
            if self.steps % self.reassign_every == 0:
                for i in range(self.n_int):
                    live = np.where(self.alive_t)[0]
                    if live.size:
                        self.assign[i] = live[int(np.argmax(pref[i, live]))]

            world_t = {f"T{j}": self.tpos[j] for j in range(self.n_threat)}
            for i in range(self.n_int):
                if not self.alive_i[i]:
                    continue
                j = self.assign[i]
                if not self.alive_t[j]:
                    continue
                law = true_pn(f"T{j}", N=4.0)
                u = self._idyn.saturate(np.asarray(law(self.t, self.ipos[i], world_t), float))
                self.ipos[i] = integrate_rk4(self._idyn, self.t, self.ipos[i], u, self.dt)
            for j in range(self.n_threat):
                if not self.alive_t[j]:
                    continue
                u = self._tdyn.saturate(self._threat_accel(j))
                self.tpos[j] = integrate_rk4(self._tdyn, self.t, self.tpos[j], u, self.dt)

            self.t += self.dt
            self.steps += 1
            reward = -self.reward_cfg.k_time
            # intercepts
            for j in range(self.n_threat):
                if not self.alive_t[j]:
                    continue
                for i in range(self.n_int):
                    if (
                        self.alive_i[i]
                        and np.linalg.norm(self.tpos[j, :2] - self.ipos[i, :2]) < self.kill_radius
                    ):
                        self.alive_t[j] = False
                        self.n_hits += 1
                        reward += self.reward_cfg.k_hit
                        break
            # leaks (threat reached the defended point)
            for j in range(self.n_threat):
                if self.alive_t[j] and np.linalg.norm(self.tpos[j, :2]) < self.leak_radius:
                    self.alive_t[j] = False
                    self.n_leaks += 1
                    reward += -self.reward_cfg.k_leak

            cur = self._total_min_range()
            reward += self.reward_cfg.k_close * (self.prev_minrange - cur) / POS_SCALE
            self.prev_minrange = cur

            terminated = not self.alive_t.any()
            truncated = self.t >= self.t_max
            info = {"hits": self.n_hits, "leaks": self.n_leaks}
            return self._obs(), float(reward), terminated, truncated, info
