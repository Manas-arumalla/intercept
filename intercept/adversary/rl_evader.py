"""Deploy a trained adversarial-RL evader as a target `Controller`.

`RLEvader` wraps a policy trained in :class:`~intercept.envs.evader_env.EvaderEnv` and exposes it as
a :data:`~intercept.core.entities.Controller` ``(t, own, world) -> control`` on the *target*, so the
learned evader drops into an ordinary `Engagement` and faces any guidance law — the adversarial
counterpart to :func:`~intercept.adversary.optimal_evader` and the scripted maneuvers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from intercept.envs.evader_env import build_evader_observation
from intercept.envs.interception_env import lateral_acceleration

Array = NDArray[np.float64]


class _Policy(Protocol):
    def predict(self, observation: Array, deterministic: bool = ...) -> tuple[Array, object]: ...


class RLEvader:
    """Trained evader policy as a target controller (lateral accel ⟂ its velocity).

    Parameters
    ----------
    pursuer:
        Name of the pursuing interceptor in the world snapshot (the threat to evade).
    policy:
        Trained policy with ``predict(obs, deterministic) -> (action, state)`` (e.g. an SB3 model).
    a_max:
        The evader's lateral-acceleration scale (m/s²); matches its plant g-limit used in training.
    obs_norm:
        Optional observation normalizer (e.g. frozen ``VecNormalize.normalize_obs``).
    """

    def __init__(
        self,
        pursuer: str,
        policy: _Policy,
        a_max: float,
        *,
        deterministic: bool = True,
        obs_norm: Callable[[Array], Array] | None = None,
    ) -> None:
        self.pursuer = pursuer
        self.policy = policy
        self.a_max = float(a_max)
        self.deterministic = deterministic
        self.obs_norm = obs_norm

    def __call__(self, t: float, own: Array, world: Mapping[str, Array]) -> Array:
        if self.pursuer not in world:
            return np.zeros(2)
        obs = build_evader_observation(own, world[self.pursuer])
        if self.obs_norm is not None:
            obs = self.obs_norm(obs)
        action, _ = self.policy.predict(obs, deterministic=self.deterministic)
        scalar = float(np.asarray(action, dtype=float).flatten()[0])
        return lateral_acceleration(own, scalar, self.a_max)
