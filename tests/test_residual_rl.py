"""Tests for the residual-PN action parameterization (the novel RL hybrid mechanism).

These verify the *mechanism* deterministically (no training): the pure-PN baseline scalar matches
the tested pure-PN law, a zero residual reproduces pure PN inside the env, and a constant-zero
"policy" in residual mode intercepts where a constant-saturated policy (the documented from-scratch
failure mode) does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from intercept.envs.interception_env import (
    apn_baseline_scalar,
    has_gym,
    lateral_acceleration,
    pn_baseline_scalar,
)
from intercept.guidance import pure_pn

pytestmark = pytest.mark.skipif(not has_gym(), reason="gymnasium not installed")


def test_pn_baseline_scalar_matches_pure_pn_law():
    own = np.array([0.0, 0.0, 300.0, 0.0])
    tgt = np.array([4000.0, 800.0, -250.0, 60.0])
    a_max = 300.0
    scalar = pn_baseline_scalar(own, tgt, a_max, N=4.0)
    cmd = lateral_acceleration(own, scalar, a_max)  # residual-baseline command vector
    # Reference: the tested pure-PN law, projected onto the same ⟂-velocity axis and clipped.
    ref_vec = pure_pn("target", N=4.0).command(0.0, own, tgt)
    v = own[2:4]
    perp = np.array([-v[1], v[0]]) / np.linalg.norm(v)
    ref_scalar = float(np.clip((ref_vec @ perp) / a_max, -1.0, 1.0))
    assert scalar == pytest.approx(ref_scalar, abs=1e-9)
    assert np.allclose(cmd, ref_scalar * a_max * perp)


def test_pn_baseline_scalar_degenerate_safe():
    own = np.array([0.0, 0.0, 300.0, 0.0])
    assert pn_baseline_scalar(own, own.copy(), 0.0) == 0.0  # a_max == 0
    coincident = np.array([0.0, 0.0, -100.0, 0.0])
    assert pn_baseline_scalar(own, coincident, 300.0) == 0.0  # zero range


def _crossing_env(action_mode):
    from intercept.benchmark import ParametricScenario
    from intercept.core import G0
    from intercept.envs import InterceptionEnv

    sc = ParametricScenario(
        name="t",
        model="aero",
        interceptor_speed=1000.0,
        interceptor_a_max=40 * G0,
        target_speed=700.0,
        target_a_max=25 * G0,
        interceptor_tau=0.2,
        target_tau=0.3,
        target_heading_deg=150.0,
        offset_min=1500.0,
        offset_max=1500.0,
        range_min=7000.0,
        range_max=7000.0,
        dt=0.01,
        t_max=18.0,
        kill_radius=20.0,
    )
    return InterceptionEnv(sc, obs_mode="rich", action_mode=action_mode)


def test_residual_zero_action_reproduces_pn_and_intercepts():
    """A zero residual = pure PN, which intercepts the aero crossing target (no collapse)."""
    env = _crossing_env("residual_pn")
    env.reset(seed=0)
    intercepted = False
    for _ in range(env.max_steps):
        _, _, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
        intercepted = info["intercepted"]
        if term or trunc:
            break
    assert intercepted


def test_absolute_constant_saturated_action_does_not_intercept():
    """The documented from-scratch failure mode (constant saturated action) misses the same shot —
    this is exactly what residual mode fixes."""
    env = _crossing_env("absolute")
    env.reset(seed=0)
    intercepted = False
    for _ in range(env.max_steps):
        _, _, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
        intercepted = info["intercepted"]
        if term or trunc:
            break
    assert not intercepted


def test_invalid_action_mode_rejected():
    from intercept.benchmark import ParametricScenario
    from intercept.envs import InterceptionEnv

    with pytest.raises(ValueError):
        InterceptionEnv(ParametricScenario(name="t"), action_mode="bogus")
    with pytest.raises(ValueError):
        InterceptionEnv(ParametricScenario(name="t"), baseline="bogus")


def test_apn_baseline_adds_target_accel_feedforward():
    """APN baseline = PN baseline for a point-mass target; adds (N/2)·a_T⊥ for an aero target."""
    own = np.array([0.0, 0.0, 300.0, 0.0])
    tgt4 = np.array([4000.0, 800.0, -250.0, 60.0])  # 4-state: no accel available
    assert apn_baseline_scalar(own, tgt4, 300.0, N=4.0) == pn_baseline_scalar(own, tgt4, 300.0, 4.0)

    a_t = np.array([0.0, 120.0])  # target pulling +y lateral accel
    tgt6 = np.array([4000.0, 800.0, -250.0, 60.0, a_t[0], a_t[1]])
    a_max, N = 300.0, 4.0
    base = pn_baseline_scalar(own, tgt6, a_max, N)
    perp = np.array([-own[3], own[2]]) / np.linalg.norm(own[2:4])
    expected = float(np.clip(base + 0.5 * N * (a_t @ perp) / a_max, -1.0, 1.0))
    assert apn_baseline_scalar(own, tgt6, a_max, N) == pytest.approx(expected, abs=1e-9)


class _MockRecurrentPolicy:
    """Records how it is called so we can assert RLGuidance threads LSTM state correctly."""

    def __init__(self):
        self.calls = []
        self._step = 0

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        self.calls.append((state, None if episode_start is None else bool(episode_start[0])))
        self._step += 1
        return np.array([0.0], dtype=np.float32), ("hidden", self._step)


def test_recurrent_rlguidance_threads_and_resets_state():
    from intercept.guidance.rl_policy import RLGuidance

    pol = _MockRecurrentPolicy()
    g = RLGuidance(
        "target",
        pol,
        a_max=300.0,
        obs_mode="rich",
        recurrent=True,
        action_mode="residual_pn",
        baseline="apn",
    )
    own = np.array([0.0, 0.0, 300.0, 0.0, 0.0, 0.0])
    tgt = np.array([4000.0, 0.0, -250.0, 0.0, 0.0, 0.0])
    world = {"target": tgt}
    for t in (0.0, 0.01, 0.02):
        g(t, own, world)
    # First call: fresh episode (state None, episode_start True); later calls thread the state.
    assert pol.calls[0] == (None, True)
    assert pol.calls[1][1] is False and pol.calls[1][0] == ("hidden", 1)
    assert pol.calls[2][1] is False and pol.calls[2][0] == ("hidden", 2)
    # A time reset (new engagement) clears the state and flags a new episode start.
    g(0.0, own, world)
    assert pol.calls[3] == (None, True)


def test_interception_env_opponent_override():
    """The opponent override replaces the scenario maneuver (enables self-play)."""
    from intercept.benchmark import ParametricScenario
    from intercept.envs import InterceptionEnv

    def opponent(t, own, world):
        return np.array([1.0, 2.0])

    env = InterceptionEnv(ParametricScenario(name="t"), opponent=opponent)
    env.reset(seed=0)
    assert env.target_ctrl is opponent  # the override is used, not the scenario controller


def test_interception_env_opponent_factory_per_reset():
    """opponent_factory is invoked each reset (enables pooled / fictitious-play opponents)."""
    from intercept.benchmark import ParametricScenario
    from intercept.envs import InterceptionEnv

    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return lambda t, own, world: np.zeros(2)

    env = InterceptionEnv(ParametricScenario(name="t"), opponent_factory=factory)
    env.reset(seed=0)
    env.reset(seed=1)
    assert calls["n"] == 2  # a fresh opponent sampled on each reset


def test_interception_env_estimated_observations():
    """With a sensor+estimator the obs come from the EKF estimate (not truth); a zero-residual
    (PN-on-estimate) policy still intercepts with a decent seeker."""
    from intercept.benchmark import ParametricScenario
    from intercept.envs import InterceptionEnv
    from intercept.estimation import EKF, nca_model
    from intercept.sensors import Radar

    sc = ParametricScenario(
        name="t",
        interceptor_speed=1000,
        target_speed=700,
        interceptor_a_max=250,
        range_min=5000,
        range_max=8000,
    )
    env = InterceptionEnv(
        sc,
        obs_mode="basic",
        action_mode="residual_pn",
        sensor=Radar(sigma_range=20.0, sigma_bearing=0.005),
        estimator_factory=lambda x0, p0: EKF(lambda d: nca_model(d, 50.0, ndim=2), x0, p0),
    )
    assert env._estimating
    obs, _ = env.reset(seed=0)
    assert np.all(np.isfinite(obs))
    hit = False
    for _ in range(env.max_steps):
        obs, r, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
        assert np.all(np.isfinite(obs))
        if term or trunc:
            hit = info["intercepted"]
            break
    assert hit  # PN on the EKF estimate still brings it home
