"""Tests for the RL interception environment and the policy bridge.

Gym-dependent tests skip if Gymnasium is absent. We validate the Gymnasium API, observation
construction/normalization, determinism, that the shaped reward rewards interception, and that a
hand-coded PN-like scalar "policy" drives both the env and an `Engagement` (via `RLGuidance`) to a
hit. The action is a single scalar lateral acceleration (perpendicular to the interceptor velocity).
"""

import numpy as np
import pytest

from intercept.benchmark import ParametricScenario
from intercept.core import Engagement, Entity, PointMass2D
from intercept.envs import OBS_DIM, RewardConfig, build_observation, has_gym, lateral_acceleration
from intercept.guidance.rl_policy import RLGuidance

gym_required = pytest.mark.skipif(not has_gym(), reason="Gymnasium not installed")


# --- observation & action mapping ------------------------------------------


def test_build_observation_shape_and_dtype():
    own = np.array([0.0, 0.0, 600.0, 0.0])
    tgt = np.array([5000.0, 1000.0, -300.0, 0.0])
    obs = build_observation(own, tgt)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert obs[0] == pytest.approx(1.0)  # relative x normalized by POS_SCALE=5000
    assert obs[1] == pytest.approx(0.2)


def test_lateral_acceleration_is_perpendicular_to_velocity():
    own = np.array([0.0, 0.0, 600.0, 0.0])  # velocity along +x
    a = lateral_acceleration(own, 0.5, a_max=200.0)
    assert np.allclose(a, [0.0, 100.0])  # +90 deg from +x, scaled by 0.5*200
    assert np.dot(a, own[2:4]) == pytest.approx(0.0)


# --- RLGuidance bridge ------------------------------------------------------


class _ConstPolicy:
    def __init__(self, scalar):
        self._a = np.array([scalar], dtype=float)

    def predict(self, observation, deterministic=True):
        return self._a, None


class _PNPolicy:
    """Hand-coded PN-like scalar policy: command proportional to the LOS rate (from the obs)."""

    def __init__(self, gain=10.0):
        self.gain = gain

    def predict(self, observation, deterministic=True):
        o = np.asarray(observation, dtype=float)
        r, v = o[:2], o[2:4]  # normalized relative position / velocity
        cross = r[0] * v[1] - r[1] * v[0]  # proportional to LOS rate
        return np.array([np.clip(self.gain * cross, -1.0, 1.0)]), None


def test_rl_guidance_scales_action():
    g = RLGuidance("target", _ConstPolicy(0.5), a_max=200.0)
    u = g(0.0, np.array([0.0, 0.0, 600.0, 0.0]), {"target": np.array([1000.0, 0.0, 0.0, 0.0])})
    assert np.allclose(u, [0.0, 100.0])  # lateral, perpendicular to +x velocity


def test_rl_guidance_clips_action():
    g = RLGuidance("target", _ConstPolicy(5.0), a_max=100.0)  # out-of-range scalar
    u = g(0.0, np.array([0.0, 0.0, 600.0, 0.0]), {"target": np.array([1000.0, 0.0, 0.0, 0.0])})
    assert np.allclose(u, [0.0, 100.0])  # clipped to 1.0 then scaled


def test_rl_guidance_pn_policy_intercepts_crossing():
    g = RLGuidance("target", _PNPolicy(gain=12.0), a_max=300.0)
    interceptor = Entity(
        "interceptor",
        PointMass2D(a_max=300.0),
        np.array([0.0, 0.0, 700.0, 0.0]),
        controller=g,
        role="interceptor",
    )
    target = Entity("target", PointMass2D(), np.array([4000.0, 1200.0, -250.0, 0.0]), role="target")
    res = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=20.0,
        kill_radius=20.0,
    ).run()
    assert res.intercepted


# --- environment (Gymnasium) -----------------------------------------------


@gym_required
def test_env_reset_and_step_api():
    from intercept.envs import InterceptionEnv

    env = InterceptionEnv(ParametricScenario(name="t", target_heading_deg=180.0))
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    obs2, reward, terminated, truncated, info2 = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs2)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "range" in info2


@gym_required
def test_env_action_space_is_scalar():
    from intercept.envs import InterceptionEnv

    env = InterceptionEnv(ParametricScenario(name="t"))
    assert env.action_space.shape == (1,)


@gym_required
def test_env_reset_is_seed_deterministic():
    from intercept.envs import InterceptionEnv

    env = InterceptionEnv(ParametricScenario(name="t"))
    o1, _ = env.reset(seed=123)
    o2, _ = env.reset(seed=123)
    assert np.allclose(o1, o2)
    o3, _ = env.reset(seed=124)
    assert not np.allclose(o1, o3)


@gym_required
def test_env_passes_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env

    from intercept.envs import InterceptionEnv

    check_env(InterceptionEnv(ParametricScenario(name="t")), skip_render_check=True)


@gym_required
def test_env_pn_policy_earns_intercept_bonus():
    from intercept.envs import InterceptionEnv

    scenario = ParametricScenario(
        name="cross",
        interceptor_speed=700,
        interceptor_a_max=300,
        target_speed=250,
        range_min=3000,
        range_max=3500,
        offset_min=-800,
        offset_max=800,
        target_heading_deg=110,
        dt=0.02,
        t_max=15.0,
        kill_radius=25.0,
    )
    env = InterceptionEnv(scenario, RewardConfig())
    policy = _PNPolicy(gain=12.0)
    obs, _ = env.reset(seed=3)
    total, intercepted = 0.0, False
    for _ in range(2000):
        action, _ = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        intercepted = intercepted or info["intercepted"]
        if terminated or truncated:
            break
    assert intercepted
    assert total > 50.0  # the terminal intercept bonus dominates a successful episode
