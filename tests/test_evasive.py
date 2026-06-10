"""Tests for aggressive/reactive evasive maneuvers and the aero realistic-scenario wiring."""

import numpy as np
import pytest

from intercept.adversary import hard_turn, random_telegraph, reactive_break
from intercept.benchmark import ParametricScenario, run_montecarlo, summarize
from intercept.core import G0, AeroMissile2D
from intercept.guidance import true_pn


def test_hard_turn_is_perpendicular_and_bounded():
    c = hard_turn(accel=300.0, sign=1.0)
    own = np.array([0.0, 0.0, 200.0, 0.0])  # velocity +x
    u = c(0.0, own, {})
    assert np.allclose(u, [0.0, 300.0])  # perpendicular to +x
    assert np.dot(u, own[2:4]) == pytest.approx(0.0)


def test_random_telegraph_is_seeded_and_switches():
    rng = np.random.default_rng(0)
    c = random_telegraph(accel=200.0, mean_switch=0.5, rng=rng)
    own = np.array([0.0, 0.0, 300.0, 0.0])
    signs = [np.sign(c(t, own, {})[1]) for t in np.linspace(0, 5, 200)]
    assert set(np.unique(signs)).issubset({-1.0, 1.0})
    assert len(set(signs)) == 2  # it does flip sign over 5 s

    # Same seed => identical sequence (reproducible Monte-Carlo).
    c2 = random_telegraph(accel=200.0, mean_switch=0.5, rng=np.random.default_rng(0))
    signs2 = [np.sign(c2(t, own, {})[1]) for t in np.linspace(0, 5, 200)]
    assert signs == signs2


def test_reactive_break_triggers_only_inside_range():
    c = reactive_break("interceptor", accel=300.0, trigger_range=500.0)
    own = np.array([0.0, 0.0, 200.0, 0.0])
    far = {"interceptor": np.array([2000.0, 0.0, 0.0, 0.0])}
    near = {"interceptor": np.array([300.0, 0.0, 0.0, 0.0])}
    assert np.allclose(c(0.0, own, far), [0.0, 0.0])  # outside trigger => coasts
    assert np.linalg.norm(c(0.0, own, near)) == pytest.approx(300.0)  # inside => max-g break


def test_reactive_break_uses_base_outside_trigger():
    base = hard_turn(accel=100.0)
    c = reactive_break("interceptor", accel=300.0, trigger_range=500.0, base=base)
    own = np.array([0.0, 0.0, 200.0, 0.0])
    far = {"interceptor": np.array([2000.0, 0.0, 0.0, 0.0])}
    assert np.linalg.norm(c(0.0, own, far)) == pytest.approx(100.0)  # base active outside trigger


def test_aero_scenario_builds_and_runs():
    scen = ParametricScenario(
        name="aero_test",
        model="aero",
        interceptor_speed=1000,
        interceptor_a_max=40 * G0,
        target_speed=700,
        target_a_max=25 * G0,
        range_min=6000,
        range_max=8000,
        offset_min=-1000,
        offset_max=1000,
        target_heading_deg=170,
        dt=0.01,
        t_max=20.0,
        kill_radius=20.0,
        maneuver={"type": "telegraph", "g": 20, "mean_switch": 0.8},
    )
    spec = scen.sample(np.random.default_rng(0))
    assert isinstance(spec.interceptor_dynamics, AeroMissile2D)
    assert spec.interceptor_state.shape == (6,)  # aero is 6-state
    res = spec.build(lambda tgt: true_pn(tgt, N=4.0)).run()
    assert res.duration > 0.0  # it runs to a terminal condition


def test_realistic_scenario_is_harder_for_pn():
    # A fast, high-g, randomly-jinking aero target should be intercepted *less* reliably than a
    # slow straight one — i.e. realism makes PN's job genuinely hard (not a guaranteed catch).
    common = dict(
        model="aero",
        interceptor_speed=1000,
        interceptor_a_max=40 * G0,
        range_min=6000,
        range_max=8000,
        offset_min=-1500,
        offset_max=1500,
        target_heading_deg=170,
        dt=0.01,
        t_max=22.0,
        kill_radius=20.0,
    )
    easy = ParametricScenario(name="easy", target_speed=300, target_a_max=5 * G0, **common)
    hard = ParametricScenario(
        name="hard",
        target_speed=800,
        target_a_max=30 * G0,
        maneuver={"type": "telegraph", "g": 25, "mean_switch": 0.6},
        **common,
    )
    pe = run_montecarlo(easy, lambda tgt: true_pn(tgt, N=4.0), n_trials=30, seed=1)
    ph = run_montecarlo(hard, lambda tgt: true_pn(tgt, N=4.0), n_trials=30, seed=1)
    assert summarize(ph).p_intercept < summarize(pe).p_intercept
