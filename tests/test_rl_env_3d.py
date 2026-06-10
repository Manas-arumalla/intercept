"""Tests for the 3-D RL environment, action/observation helpers, and the 3-D deploy wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from intercept.envs.interception_env_3d import (
    build_observation_3d,
    build_observation_3d_rich,
    gravity_feedforward_3d,
    has_gym,
    lateral_acceleration_3d,
)

pytestmark = pytest.mark.skipif(not has_gym(), reason="gymnasium not installed")

G0 = 9.80665


def _own(vx=600.0, vy=0.0, vz=50.0):
    return np.array([0.0, 0.0, 1000.0, vx, vy, vz, 0.0, 0.0, 0.0])


def test_lateral_acceleration_3d_perp_and_clipped():
    own = _own()
    a_max = 300.0
    for action in ([1.0, 0.0], [0.0, 1.0], [0.7, -0.7], [2.0, 2.0]):
        a = lateral_acceleration_3d(own, np.array(action), a_max)
        assert abs(float(a @ own[3:6])) < 1e-9  # perpendicular to velocity
        assert np.linalg.norm(a) <= a_max + 1e-6  # within the achievable disc


def test_observation_3d_shapes():
    own, tgt = _own(), np.array([4000.0, 800.0, 2000.0, -250.0, 30.0, -10.0, 0, 0, 0])
    assert build_observation_3d(own, tgt).shape == (9,)
    assert build_observation_3d_rich(own, tgt).shape == (14,)


def test_gravity_feedforward_3d_perp_and_off():
    own = _own()
    g = gravity_feedforward_3d(own, G0)
    assert abs(float(g @ own[3:6])) < 1e-9  # ⟂ velocity
    assert g[2] > 0.0  # opposes gravity (which is along −z)
    assert np.allclose(gravity_feedforward_3d(own, 0.0), 0.0)  # off when gravity == 0


def test_env3d_reset_step_and_zero_action_runs():
    from intercept.benchmark import ParametricScenario3D
    from intercept.envs import InterceptionEnv3D

    sc = ParametricScenario3D(
        name="t",
        model="aero",
        interceptor_a_max=40 * G0,
        target_a_max=22 * G0,
        range_min=7000,
        range_max=7000,
        offset_min=500,
        offset_max=500,
        alt_min=3500,
        alt_max=3500,
        maneuver={"type": "weave", "g": 10, "frequency": 0.3},
    )
    env = InterceptionEnv3D(sc, obs_mode="rich")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (14,) and env.action_space.shape == (2,)
    r = 0.0
    for _ in range(env.max_steps):
        obs, r, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        if term or trunc:
            break
    assert np.isfinite(r)


class _MockPolicy2D:
    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return np.array([0.3, -0.4], dtype=np.float32), state


def test_rlguidance3d_emits_perp_acceleration():
    from intercept.guidance.rl_policy import RLGuidance3D

    g = RLGuidance3D("target", _MockPolicy2D(), a_max=300.0, obs_mode="rich", gravity=G0)
    own = _own()
    tgt = np.array([4000.0, 0.0, 1000.0, -250.0, 0.0, 0.0, 0, 0, 0])
    a = g(0.0, own, {"target": tgt})
    # Command minus gravity feed-forward is the pure lateral term, which is ⟂ velocity.
    lateral = a - gravity_feedforward_3d(own, G0)
    assert abs(float(lateral @ own[3:6])) < 1e-6
    assert a.shape == (3,)


def test_pn_baseline_action_3d_reproduces_true_pn_3d():
    from intercept.envs.interception_env_3d import pn_baseline_action_3d
    from intercept.guidance import true_pn_3d

    own = _own(vx=900.0, vy=20.0, vz=120.0)
    tgt = np.array([6000.0, 1500.0, 2500.0, -700.0, 40.0, -20.0, 0, 0, 0])
    a_max = 5000.0  # large enough that the PN command is not norm-clipped, so equality holds
    act = pn_baseline_action_3d(own, tgt, a_max)
    applied = lateral_acceleration_3d(own, act, a_max)  # baseline command vector
    pn_cmd = true_pn_3d("target").command(0.0, own, tgt)  # the tested True-PN-3D law
    # PN is ⟂ to closing velocity ≈ ⟂ velocity; the baseline = PN projected on the lateral plane.
    v_hat = own[3:6] / np.linalg.norm(own[3:6])
    pn_perp = pn_cmd - (pn_cmd @ v_hat) * v_hat
    assert np.allclose(applied, pn_perp, atol=1e-6)


def test_env3d_residual_zero_intercepts_where_absolute_saturated_misses():
    from intercept.benchmark import ParametricScenario3D
    from intercept.envs import InterceptionEnv3D

    kw = dict(
        model="aero",
        interceptor_speed=1000.0,
        interceptor_a_max=40 * G0,
        target_speed=700.0,
        target_a_max=22 * G0,
        target_azimuth_deg=200.0,
        target_elevation_deg=-2.0,
        range_min=7000,
        range_max=7000,
        offset_min=500,
        offset_max=500,
        alt_min=3500,
        alt_max=3500,
        dt=0.01,
        t_max=20.0,
        kill_radius=20.0,
        maneuver={"type": "weave", "g": 12, "frequency": 0.3},
    )
    sc = ParametricScenario3D(name="t", **kw)

    def _run(env, action):
        env.reset(seed=0)
        hit = False
        for _ in range(env.max_steps):
            _, _, term, trunc, info = env.step(np.array(action, dtype=np.float32))
            hit = info["intercepted"]
            if term or trunc:
                break
        return hit

    res_env = InterceptionEnv3D(sc, obs_mode="rich", action_mode="residual_pn")
    abs_env = InterceptionEnv3D(sc, obs_mode="rich", action_mode="absolute")
    assert _run(res_env, [0.0, 0.0])  # zero residual == PN-3D == intercept
    assert not _run(abs_env, [1.0, 1.0])  # constant saturated action (collapse mode) misses


def test_env3d_invalid_action_mode():
    from intercept.benchmark import ParametricScenario3D
    from intercept.envs import InterceptionEnv3D

    with pytest.raises(ValueError):
        InterceptionEnv3D(ParametricScenario3D(name="t"), action_mode="bogus")


def test_apn_baseline_action_3d_adds_target_accel():
    from intercept.envs.interception_env_3d import (
        apn_baseline_action_3d,
        pn_baseline_action_3d,
    )

    own = _own(vx=900.0, vy=20.0, vz=120.0)
    a_max = 5000.0
    tgt6 = np.array([6000.0, 1500.0, 2500.0, -700.0, 40.0, -20.0])  # point mass: no accel state
    assert np.allclose(
        apn_baseline_action_3d(own, tgt6, a_max), pn_baseline_action_3d(own, tgt6, a_max)
    )  # reduces to PN
    tgt9 = np.concatenate([tgt6, [0.0, 150.0, 60.0]])  # aero: has lateral accel
    assert not np.allclose(
        apn_baseline_action_3d(own, tgt9, a_max), pn_baseline_action_3d(own, tgt9, a_max)
    )  # feedforward changes it
