"""Tests for the benchmark layer: metrics, scenario reproducibility, and Monte-Carlo fairness."""

import math

import numpy as np
import pytest

from intercept.benchmark import (
    ParametricScenario,
    run_montecarlo,
    summarize,
    wilson_interval,
)
from intercept.benchmark.metrics import MetricSummary
from intercept.guidance import true_pn, zem_pn

# --- metrics ---------------------------------------------------------------


def test_wilson_interval_bounds_and_symmetry():
    lo, hi = wilson_interval(5, 10)
    assert 0.0 < lo < 0.5 < hi < 1.0
    # Symmetric around 0.5 for a 50% rate.
    assert math.isclose(0.5 - lo, hi - 0.5, rel_tol=1e-9)


def test_wilson_interval_edge_cases():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(10, 10)
    assert hi == pytest.approx(1.0, abs=1e-9)
    assert lo < 1.0  # Wilson keeps a lower bound below 1 even at 100% observed
    lo0, hi0 = wilson_interval(0, 10)
    assert lo0 == pytest.approx(0.0, abs=1e-9)
    assert hi0 > 0.0


def test_summarize_empty():
    s = summarize([])
    assert isinstance(s, MetricSummary)
    assert s.n_trials == 0 and s.p_intercept == 0.0


# --- scenario reproducibility ---------------------------------------------


def test_scenario_sample_is_seed_reproducible():
    scen = ParametricScenario(
        name="t", range_min=3000, range_max=5000, offset_min=-1000, offset_max=1000
    )
    a = scen.sample(np.random.default_rng(42))
    b = scen.sample(np.random.default_rng(42))
    assert np.allclose(a.interceptor_state, b.interceptor_state)
    assert np.allclose(a.target_state, b.target_state)
    # Different seed => (almost surely) different geometry.
    c = scen.sample(np.random.default_rng(43))
    assert not np.allclose(a.target_state, c.target_state)


def test_scenario_at_is_deterministic():
    scen = ParametricScenario(name="t")
    s1 = scen.at(4000.0, 500.0)
    s2 = scen.at(4000.0, 500.0)
    assert np.allclose(s1.target_state, s2.target_state)
    assert s1.target_state[0] == 4000.0 and s1.target_state[1] == 500.0


# --- Monte-Carlo fairness & determinism -----------------------------------


def _scen():
    return ParametricScenario(
        name="mc",
        interceptor_speed=700,
        interceptor_a_max=150,
        target_speed=250,
        range_min=3000,
        range_max=4000,
        offset_min=-400,
        offset_max=400,
        target_heading_deg=180,
        t_max=30.0,
    )


def test_montecarlo_is_deterministic():
    scen = _scen()
    r1 = run_montecarlo(scen, lambda tgt: true_pn(tgt, N=4), n_trials=8, seed=0)
    r2 = run_montecarlo(scen, lambda tgt: true_pn(tgt, N=4), n_trials=8, seed=0)
    assert [r.miss_distance for r in r1] == [r.miss_distance for r in r2]


def test_montecarlo_fairness_same_initial_conditions_across_algorithms():
    # The same seed must yield identical sampled engagements regardless of guidance law:
    # the *target* trajectory (which does not depend on the interceptor here) must match per trial.
    scen = _scen()
    r_pn = run_montecarlo(scen, lambda tgt: true_pn(tgt, N=4), n_trials=6, seed=7)
    r_zem = run_montecarlo(scen, lambda tgt: zem_pn(tgt, N=4), n_trials=6, seed=7)
    for a, b in zip(r_pn, r_zem, strict=True):
        # initial target state identical
        assert np.allclose(a.states["target"][0], b.states["target"][0])


def test_montecarlo_validation():
    with pytest.raises(ValueError):
        run_montecarlo(_scen(), lambda tgt: true_pn(tgt), n_trials=0)


def test_summarize_on_real_run_has_sane_metrics():
    scen = _scen()
    results = run_montecarlo(scen, lambda tgt: true_pn(tgt, N=4), n_trials=20, seed=1)
    s = summarize(results)
    assert s.n_trials == 20
    assert 0.0 <= s.p_intercept <= 1.0
    assert s.p_intercept_lo <= s.p_intercept <= s.p_intercept_hi
    assert s.miss_median >= 0.0
    assert s.effort_mean >= 0.0


# --- paired-bootstrap significance -----------------------------------------


def test_paired_bootstrap_detects_and_rejects_difference():
    from intercept.benchmark import paired_bootstrap

    rng = np.random.default_rng(0)
    # A clearly better than B (paired): a intercepts 90%, b 60% on the same trials.
    a = (rng.random(200) < 0.9).astype(float)
    b = (rng.random(200) < 0.6).astype(float)
    res = paired_bootstrap(a, b, n_boot=2000, rng=np.random.default_rng(1), metric="p_intercept")
    assert res.diff > 0 and res.significant and res.p_value < 0.05
    assert res.ci_lo > 0.0  # CI excludes zero

    # Identical performance => not significant.
    same = (rng.random(200) < 0.8).astype(float)
    res2 = paired_bootstrap(same, same.copy(), n_boot=2000, rng=np.random.default_rng(2))
    assert res2.diff == 0.0 and not res2.significant


def test_compare_intercept_pairs_results_by_seed():
    from intercept.benchmark import ParametricScenario, compare_intercept, run_montecarlo
    from intercept.guidance import pure_pn, true_pn

    sc = ParametricScenario(
        name="S2",
        interceptor_a_max=100.0,
        target_heading_deg=90.0,
        range_min=4000,
        range_max=4000,
        offset_min=1500,
        offset_max=1500,
    )
    ra = run_montecarlo(sc, lambda t: true_pn(t, N=4.0), n_trials=40, seed=7)
    rb = run_montecarlo(sc, lambda t: pure_pn(t, N=4.0), n_trials=40, seed=7)
    cmp = compare_intercept(ra, rb, n_boot=2000, rng=np.random.default_rng(3))
    assert cmp.n_pairs == 40 and cmp.metric == "p_intercept"
    assert -1.0 <= cmp.diff <= 1.0


def test_gain_sensitivity_higher_n_helps_true_pn_on_jink():
    """Gain sweep: True PN's capture of an unpredictable jink improves with higher N."""
    from intercept.benchmark import ParametricScenario, run_montecarlo
    from intercept.core import G0
    from intercept.guidance import true_pn

    sc = ParametricScenario(
        name="jink",
        model="aero",
        interceptor_speed=1000.0,
        interceptor_a_max=40 * G0,
        target_speed=700.0,
        target_a_max=25 * G0,
        interceptor_tau=0.2,
        target_tau=0.3,
        target_heading_deg=165.0,
        offset_min=-1200,
        offset_max=1200,
        range_min=6000,
        range_max=9000,
        dt=0.01,
        t_max=18.0,
        kill_radius=20.0,
        maneuver={"type": "telegraph", "g": 22, "mean_switch": 0.7},
    )

    def p_int(N):
        res = run_montecarlo(sc, lambda t: true_pn(t, N=N), n_trials=40, seed=4)
        return sum(r.intercepted for r in res) / len(res)

    low, high = p_int(2.0), p_int(6.0)
    assert high > low + 0.2  # a higher navigation constant clearly helps on the jink
