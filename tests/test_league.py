"""Tests for the INTERCEPT League Bradley-Terry / Elo rating fit."""

from __future__ import annotations

import numpy as np
import pytest

from intercept.benchmark import bradley_terry, bradley_terry_bootstrap, elo_expected_score


def test_bt_recovers_known_ordering():
    # A beats B 8/10, B beats C 8/10, A beats C 9/10 -> ordering A > B > C.
    names = ["A", "B", "C"]
    wins = np.array([[0, 8, 9], [2, 0, 8], [1, 2, 0]], dtype=float)
    elo = bradley_terry(names, wins)
    assert elo["A"] > elo["B"] > elo["C"]
    # Mean anchored at 1500.
    assert np.mean(list(elo.values())) == pytest.approx(1500.0, abs=1e-6)


def test_bt_expected_score_matches_observed_rate():
    # Two players, 75/25: fitted Elo gap should reproduce ~0.75 expected score (within smoothing).
    names = ["X", "Y"]
    wins = np.array([[0, 75], [25, 0]], dtype=float)
    elo = bradley_terry(names, wins, eps=0.0)
    assert elo_expected_score(elo["X"], elo["Y"]) == pytest.approx(0.75, abs=0.01)


def test_bt_undefeated_is_finite_and_top():
    # An undefeated participant must get a large but FINITE rating (smoothing).
    names = ["champ", "mid", "low"]
    wins = np.array([[0, 10, 10], [0, 0, 7], [0, 3, 0]], dtype=float)
    elo = bradley_terry(names, wins)
    assert np.isfinite(elo["champ"])
    assert elo["champ"] > elo["mid"] > elo["low"]


def test_bt_handles_disjoint_pairs_via_smoothing_guard():
    # Pairs that never played get no pseudo-counts; ratings stay finite and ordered within groups.
    names = ["a", "b", "c", "d"]
    wins = np.zeros((4, 4))
    wins[0, 1] = 9
    wins[1, 0] = 1  # a > b
    wins[2, 3] = 9
    wins[3, 2] = 1  # c > d
    elo = bradley_terry(names, wins)
    assert all(np.isfinite(v) for v in elo.values())
    assert elo["a"] > elo["b"] and elo["c"] > elo["d"]


def test_bt_shape_validation():
    with pytest.raises(ValueError):
        bradley_terry(["a", "b"], np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# Bootstrap CI tests
# ---------------------------------------------------------------------------


def _outcomes(wins_a: int, total: int) -> dict[tuple[str, str], list[bool]]:
    """Helper: fixed win record for A vs B."""
    return {("A", "B"): [True] * wins_a + [False] * (total - wins_a)}


def test_bootstrap_ci_contains_point_estimate():
    """The 95 % CI should bracket the point-estimate Elo in almost all cases."""
    outcomes = _outcomes(30, 40)  # A wins 75 %
    names = ["A", "B"]
    ci = bradley_terry_bootstrap(names, outcomes, n_replicates=500, rng=np.random.default_rng(0))
    # Compute the point estimate
    wins = np.array([[0, 30], [10, 0]], float)
    elo = bradley_terry(names, wins)
    for nm in names:
        lo, hi = ci[nm]
        assert lo <= elo[nm] <= hi, f"{nm}: point {elo[nm]:.1f} not in [{lo:.1f}, {hi:.1f}]"


def test_bootstrap_ci_dominant_player_ranked_high():
    """A player who wins all matches should have a CI lower bound well above 1500."""
    outcomes = {("A", "B"): [True] * 30, ("A", "C"): [True] * 30, ("B", "C"): [True] * 20}
    names = ["A", "B", "C"]
    ci = bradley_terry_bootstrap(names, outcomes, n_replicates=500, rng=np.random.default_rng(1))
    assert ci["A"][0] > 1500, "Dominant player CI lower bound should exceed league mean"


def test_bootstrap_ci_interval_width_shrinks_with_more_matches():
    """More matches → narrower CI (law of large numbers)."""
    names = ["A", "B"]
    ci_small = bradley_terry_bootstrap(
        names, _outcomes(8, 10), n_replicates=500, rng=np.random.default_rng(3)
    )
    ci_large = bradley_terry_bootstrap(
        names, _outcomes(80, 100), n_replicates=500, rng=np.random.default_rng(4)
    )
    width_small = ci_small["A"][1] - ci_small["A"][0]
    width_large = ci_large["A"][1] - ci_large["A"][0]
    assert width_large < width_small, "More data should produce a narrower CI"


def test_bootstrap_ci_evenly_matched_intervals_overlap():
    """When two players are evenly matched, their CIs should substantially overlap."""
    outcomes = {("A", "B"): [True] * 20 + [False] * 20}
    names = ["A", "B"]
    ci = bradley_terry_bootstrap(names, outcomes, n_replicates=500, rng=np.random.default_rng(5))
    # With a ~50/50 record both CIs should contain 1500 (the mean)
    for nm in names:
        lo, hi = ci[nm]
        assert lo <= 1500.0 <= hi, f"{nm} CI [{lo:.1f}, {hi:.1f}] should straddle 1500"
