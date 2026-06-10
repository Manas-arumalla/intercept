"""RL policy guidance: deploy a trained policy as a `Controller`.

`RLGuidance` wraps any object exposing a Stable-Baselines3-style ``predict(obs, deterministic)``
(an SB3 model, or a CleanRL/torch policy adapter) and turns it into a guidance law conforming to
the `Controller` contract. It builds the observation with the *same* :func:`build_observation` used
during training and scales the ``[-1, 1]²`` action to the interceptor's ``a_max`` — so the learned
agent runs inside the ordinary `Engagement` and is benchmarked against the classical/optimal/MPC
laws on identical dynamics and geometries.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from intercept.envs.interception_env import (
    PN_BASELINE_N,
    apn_baseline_scalar,
    build_observation,
    build_observation_rich,
    gravity_feedforward,
    lateral_acceleration,
    pn_baseline_scalar,
)

Array = NDArray[np.float64]


class Policy(Protocol):
    """Minimal policy interface (matches Stable-Baselines3 ``model.predict``)."""

    def predict(self, observation: Array, deterministic: bool = ...) -> tuple[Array, object]: ...


class RLGuidance:
    """Wrap a trained policy as a guidance `Controller`.

    Parameters
    ----------
    target:
        Name of the target entity in the world snapshot.
    policy:
        Object with ``predict(obs, deterministic) -> (action, state)`` (e.g. an SB3 model).
    a_max:
        Acceleration scale applied to the scalar ``[-1, 1]`` action (m/s²).
    deterministic:
        Use the policy's deterministic action (default) vs. sampling.
    obs_norm:
        Optional observation normalizer (raw obs -> normalized obs), e.g.
        ``VecNormalize.normalize_obs`` — required when the policy was trained with VecNormalize so
        deployment sees the *same* normalized inputs as training.

    The policy outputs a single scalar; it is mapped to a lateral acceleration perpendicular to the
    interceptor velocity (matching the training env), so the learned law has exactly the lateral
    control authority the classical laws have.
    """

    def __init__(
        self,
        target: str,
        policy: Policy,
        a_max: float,
        deterministic: bool = True,
        obs_norm: Callable[[Array], Array] | None = None,
        obs_mode: str = "basic",
        gravity: float = 0.0,
        action_mode: str = "absolute",
        residual_scale: float = 0.5,
        pn_N: float = PN_BASELINE_N,
        baseline: str = "pn",
        recurrent: bool = False,
    ) -> None:
        self.target = target
        self.policy = policy
        self.a_max = float(a_max)
        self.deterministic = deterministic
        self.obs_norm = obs_norm
        self.gravity = float(gravity)
        self.action_mode = action_mode
        self.residual_scale = float(residual_scale)
        self.pn_N = float(pn_N)
        self.baseline = baseline
        self._baseline_fn = apn_baseline_scalar if baseline == "apn" else pn_baseline_scalar
        self.recurrent = recurrent
        self._obs_fn = build_observation_rich if obs_mode == "rich" else build_observation
        self._lstm_state = None  # recurrent hidden state, carried across an engagement
        self._prev_t: float | None = None

    def __call__(self, t: float, own_state: Array, world: Mapping[str, Array]) -> Array:
        target_state = world[self.target]
        obs = self._obs_fn(own_state, target_state)
        if self.obs_norm is not None:
            obs = self.obs_norm(obs)
        if self.recurrent:
            # New engagement (time reset) => clear the LSTM state and flag an episode start.
            episode_start = self._prev_t is None or t <= self._prev_t
            if episode_start:
                self._lstm_state = None
            action, self._lstm_state = self.policy.predict(
                obs,
                state=self._lstm_state,
                episode_start=np.array([episode_start]),
                deterministic=self.deterministic,
            )
            self._prev_t = t
        else:
            action, _ = self.policy.predict(obs, deterministic=self.deterministic)
        scalar = float(np.asarray(action, dtype=float).flatten()[0])
        # Mirror the training-time action parameterization (residual on a PN/APN baseline).
        if self.action_mode == "residual_pn":
            base = self._baseline_fn(own_state, target_state, self.a_max, self.pn_N)
            scalar = float(np.clip(base + self.residual_scale * scalar, -1.0, 1.0))
        return lateral_acceleration(own_state, scalar, self.a_max) + gravity_feedforward(
            own_state, self.gravity
        )


class RLGuidance3D:
    """Deploy a trained **3-D** policy as a guidance `Controller`.

    Mirrors :class:`RLGuidance` for three dimensions: builds the 3-D observation used in training
    (:func:`build_observation_3d`/``_rich``) and maps the policy's 2-DOF ``[-1, 1]²`` action to a
    lateral acceleration in the plane perpendicular to the interceptor's velocity
    (:func:`lateral_acceleration_3d`), plus gravity feed-forward — so the learned 3-D law runs in an
    ordinary `Engagement` against the 3-D classical/optimal/SMG/MPC laws on identical dynamics.
    Supports recurrent (LSTM) policies via per-engagement hidden-state threading.
    """

    def __init__(
        self,
        target: str,
        policy: Policy,
        a_max: float,
        deterministic: bool = True,
        obs_norm: Callable[[Array], Array] | None = None,
        obs_mode: str = "rich",
        gravity: float = 0.0,
        recurrent: bool = False,
        action_mode: str = "absolute",
        residual_scale: float = 0.35,
        pn_N: float = 4.0,
        baseline: str = "pn",
    ) -> None:
        from intercept.envs.interception_env_3d import (
            apn_baseline_action_3d,
            build_observation_3d,
            build_observation_3d_rich,
            gravity_feedforward_3d,
            lateral_acceleration_3d,
            pn_baseline_action_3d,
        )

        self.target = target
        self.policy = policy
        self.a_max = float(a_max)
        self.deterministic = deterministic
        self.obs_norm = obs_norm
        self.gravity = float(gravity)
        self.recurrent = recurrent
        self.action_mode = action_mode
        self.residual_scale = float(residual_scale)
        self.pn_N = float(pn_N)
        self._lateral = lateral_acceleration_3d
        self._grav_ff = gravity_feedforward_3d
        self._pn_base = apn_baseline_action_3d if baseline == "apn" else pn_baseline_action_3d
        self._obs_fn = build_observation_3d_rich if obs_mode == "rich" else build_observation_3d
        self._lstm_state = None
        self._prev_t: float | None = None

    def __call__(self, t: float, own_state: Array, world: Mapping[str, Array]) -> Array:
        target_state = world[self.target]
        obs = self._obs_fn(own_state, target_state)
        if self.obs_norm is not None:
            obs = self.obs_norm(obs)
        if self.recurrent:
            episode_start = self._prev_t is None or t <= self._prev_t
            if episode_start:
                self._lstm_state = None
            action, self._lstm_state = self.policy.predict(
                obs,
                state=self._lstm_state,
                episode_start=np.array([episode_start]),
                deterministic=self.deterministic,
            )
            self._prev_t = t
        else:
            action, _ = self.policy.predict(obs, deterministic=self.deterministic)
        act = np.asarray(action, dtype=float).flatten()[:2]
        if self.action_mode == "residual_pn":
            act = (
                self._pn_base(own_state, target_state, self.a_max, self.pn_N)
                + self.residual_scale * act
            )
        return self._lateral(own_state, act, self.a_max) + self._grav_ff(own_state, self.gravity)
