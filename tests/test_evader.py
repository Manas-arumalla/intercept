"""Tests for the adversarial-RL evader environment and the RLEvader deploy controller."""

from __future__ import annotations

import numpy as np
import pytest

from intercept.envs.evader_env import EvaderReward, build_evader_observation, has_gym

pytestmark = pytest.mark.skipif(not has_gym(), reason="gymnasium not installed")


def test_build_evader_observation_shape_and_sign():
    evader = np.array([5000.0, 0.0, -700.0, 0.0])
    pursuer = np.array([0.0, 0.0, 1000.0, 0.0])
    obs = build_evader_observation(evader, pursuer)
    assert obs.shape == (6,)
    # Relative position points from evader toward pursuer (pursuer is behind/−x of the evader).
    assert obs[0] < 0.0


def test_evader_reward_defaults():
    r = EvaderReward()
    assert r.r_caught > 0 and r.k_zem > 0 and r.miss_cap > 0


def test_evader_env_reset_step_shapes():
    from intercept.envs import EvaderEnv

    env = EvaderEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (6,) and env.action_space.shape == (1,)
    obs, r, term, trunc, info = env.step(np.array([0.5], dtype=np.float32))
    assert obs.shape == (6,) and np.isfinite(r) and "min_range" in info


def test_evading_increases_min_range_vs_straight():
    """A hard-turning evader forces a larger closest approach than flying straight into PN."""
    from intercept.envs import EvaderEnv

    def run(action):
        env = EvaderEnv()
        env.reset(seed=3)
        for _ in range(env.max_steps):
            _, _, term, trunc, info = env.step(np.array([action], dtype=np.float32))
            if term or trunc:
                return info["min_range"]
        return info["min_range"]

    assert run(1.0) > run(0.0)  # max-turn evasion beats a straight (caught) target


class _MockEvaderPolicy:
    def predict(self, obs, deterministic=True):
        return np.array([0.6], dtype=np.float32), None


def test_rl_evader_emits_perp_acceleration():
    from intercept.adversary.rl_evader import RLEvader

    ev = RLEvader("interceptor", _MockEvaderPolicy(), a_max=300.0)
    own = np.array([5000.0, 0.0, -700.0, 0.0])
    world = {"interceptor": np.array([0.0, 0.0, 1000.0, 0.0])}
    a = ev(0.0, own, world)
    assert a.shape == (2,)
    assert abs(float(a @ own[2:4])) < 1e-6  # ⟂ to the evader's velocity
    assert np.linalg.norm(a) <= 300.0 + 1e-6
    # No pursuer in the world => no command.
    assert np.allclose(ev(0.0, own, {}), 0.0)
