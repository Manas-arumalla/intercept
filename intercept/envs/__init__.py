"""RL environment bridges.

Gymnasium envs (single-interceptor guidance in 2-D/3-D and an adversarial evader) wrapping the
*same* :class:`~intercept.core.Engagement` / dynamics used for classical evaluation — the shared
core is what makes the classical-vs-learned comparison fair (ADR-0005).
"""

from intercept.envs.evader_env import EvaderReward, build_evader_observation
from intercept.envs.interception_env import (
    OBS_DIM,
    PN_BASELINE_N,
    POS_SCALE,
    RICH_OBS_DIM,
    VEL_SCALE,
    RewardConfig,
    apn_baseline_scalar,
    build_observation,
    build_observation_rich,
    has_gym,
    lateral_acceleration,
    pn_baseline_scalar,
)
from intercept.envs.interception_env_3d import (
    apn_baseline_action_3d,
    build_observation_3d,
    build_observation_3d_rich,
    gravity_feedforward_3d,
    lateral_acceleration_3d,
    pn_baseline_action_3d,
)

__all__ = [
    "build_observation",
    "build_observation_rich",
    "lateral_acceleration",
    "pn_baseline_scalar",
    "apn_baseline_scalar",
    "PN_BASELINE_N",
    "RewardConfig",
    "has_gym",
    "OBS_DIM",
    "RICH_OBS_DIM",
    "POS_SCALE",
    "VEL_SCALE",
    "build_observation_3d",
    "build_observation_3d_rich",
    "lateral_acceleration_3d",
    "gravity_feedforward_3d",
    "pn_baseline_action_3d",
    "apn_baseline_action_3d",
    "build_evader_observation",
    "EvaderReward",
]

if has_gym():
    from intercept.envs.evader_env import EvaderEnv  # noqa: F401
    from intercept.envs.interception_env import InterceptionEnv  # noqa: F401
    from intercept.envs.interception_env_3d import InterceptionEnv3D  # noqa: F401
    from intercept.envs.swarm_env import CentralizedSwarmEnv, SwarmReward  # noqa: F401

    __all__ += [
        "InterceptionEnv",
        "InterceptionEnv3D",
        "EvaderEnv",
        "CentralizedSwarmEnv",
        "SwarmReward",
    ]
