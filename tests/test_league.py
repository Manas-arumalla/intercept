"""Tests for the INTERCEPT League Bradley-Terry / Elo rating fit."""

from __future__ import annotations

import numpy as np
import pytest

from intercept.benchmark import bradley_terry, elo_expected_score


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
