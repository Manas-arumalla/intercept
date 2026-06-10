"""Tests for multi-agent: weapon-target assignment (Hungarian) and the N-vs-M swarm engagement."""

import numpy as np
import pytest

from intercept.core import (  # noqa: F401 (Engagement used indirectly)
    Engagement,
    Entity,
    PointMass2D,
)
from intercept.guidance import true_pn
from intercept.multiagent import (
    MultiEngagement,
    cost_matrix,
    weapon_target_assignment,
)


def _interceptor(name, pos, vel, a_max=200.0):
    return Entity(name, PointMass2D(a_max=a_max), np.array([*pos, *vel]), role="interceptor")


def _target(name, pos, vel, ctrl=None):
    return Entity(name, PointMass2D(), np.array([*pos, *vel]), controller=ctrl, role="target")


# --- weapon-target assignment ---------------------------------------------


def test_wta_prefers_nearest_pairing():
    # Two interceptors at left; two targets; the optimal assignment pairs each to the closer one.
    ints = [np.array([0.0, 0.0, 300.0, 0.0]), np.array([0.0, 1000.0, 300.0, 0.0])]
    tgts = [np.array([3000.0, 0.0, -200.0, 0.0]), np.array([3000.0, 1000.0, -200.0, 0.0])]
    amap = weapon_target_assignment(ints, tgts)
    assert amap == {0: 0, 1: 1}


def test_wta_assignment_is_valid_and_complete():
    ints = [np.array([0.0, float(i) * 500, 300.0, 0.0]) for i in range(3)]
    tgts = [np.array([4000.0, float(j) * 500, -200.0, 0.0]) for j in range(3)]
    amap = weapon_target_assignment(ints, tgts)
    assert set(amap.keys()) == {0, 1, 2}
    assert sorted(amap.values()) == [0, 1, 2]  # a perfect matching for equal counts


def test_wta_surplus_interceptors_all_assigned():
    # 3 interceptors, 2 targets => every interceptor still gets an assignment (redundant coverage).
    ints = [np.array([0.0, float(i) * 400, 300.0, 0.0]) for i in range(3)]
    tgts = [np.array([4000.0, 0.0, -200.0, 0.0]), np.array([4000.0, 1500.0, -200.0, 0.0])]
    amap = weapon_target_assignment(ints, tgts)
    assert len(amap) == 3
    assert all(j in (0, 1) for j in amap.values())


def test_cost_matrix_shape_and_uncatchable():
    ints = [np.array([0.0, 0.0, 600.0, 0.0])]
    tgts = [
        np.array([3000.0, 0.0, -200.0, 0.0]),  # closing => finite cost
        np.array([3000.0, 0.0, 900.0, 0.0]),
    ]  # fleeing faster => uncatchable (big cost)
    C = cost_matrix(ints, tgts, speeds=[600.0])
    assert C.shape == (1, 2)
    assert C[0, 0] < 1e6 < C[0, 1]


# --- multi-agent engagement -----------------------------------------------


def test_multiengagement_intercepts_all_when_outnumbering():
    # 3 fast interceptors vs 2 straight threats => both threats intercepted.
    ints = [_interceptor(f"i{i}", (0.0, i * 800.0), (700.0, 0.0), a_max=300) for i in range(3)]
    tgts = [
        _target("t0", (5000.0, 200.0), (-250.0, 0.0)),
        _target("t1", (5000.0, 1600.0), (-250.0, 0.0)),
    ]
    res = MultiEngagement(
        ints,
        tgts,
        lambda name: true_pn(name, N=4.0),
        dt=0.01,
        t_max=30.0,
        kill_radius=25.0,
        reassign_every=25,
    ).run()
    assert res.n_killed == 2
    assert res.leakers == 0
    assert len(res.times) == res.tracks["i0"].shape[0]


def test_multiengagement_logs_and_roles():
    ints = [_interceptor("i0", (0.0, 0.0), (700.0, 0.0), a_max=300)]
    tgts = [_target("t0", (4000.0, 300.0), (-250.0, 0.0))]
    res = MultiEngagement(
        ints, tgts, lambda name: true_pn(name, N=4.0), dt=0.01, t_max=30.0, kill_radius=25.0
    ).run()
    assert res.roles["i0"] == "interceptor" and res.roles["t0"] == "target"
    assert res.n_interceptors == 1 and res.n_targets == 1
    assert res.tracks["t0"].shape[1] == 4  # point-mass state logged


# --- global kill-probability WTA -------------------------------------------


def test_kill_probability_bounds_and_uncatchable():
    from intercept.multiagent import kill_probability

    intc = np.array([0.0, 0.0, 800.0, 0.0])
    near = np.array([900.0, 0.0, -100.0, 0.0])
    p = kill_probability(intc, near, 800.0)
    assert 0.0 < p <= 0.95
    # A faster target fleeing directly away is uncatchable => zero kill probability.
    fleeing = np.array([100.0, 0.0, 2000.0, 0.0])
    assert kill_probability(intc, fleeing, 800.0) == 0.0


def test_expected_leakers_matches_survival_product():
    from intercept.multiagent import expected_leakers

    P = np.array([[0.8, 0.4], [0.5, 0.0]])
    # interceptor 0 -> target 0, interceptor 1 -> target 0 (both on target 0); target 1 unassigned.
    assignment = {0: 0, 1: 0}
    # survival(0) = (1-0.8)(1-0.5) = 0.1 ; survival(1) = 1 (unassigned) => 1.1
    assert expected_leakers(assignment, P) == pytest.approx(0.1 + 1.0)


def test_kill_prob_wta_reduces_expected_leakers_via_surplus():
    from intercept.multiagent import (
        expected_leakers,
        kill_probability_matrix,
        weapon_target_assignment,
    )

    intc = [
        np.array([0.0, 0.0, 800.0, 0.0]),
        np.array([0.0, 300.0, 800.0, 0.0]),
        np.array([0.0, -300.0, 800.0, 0.0]),
    ]
    tgt = [np.array([900.0, 0.0, -100.0, 0.0]), np.array([5500.0, 0.0, -120.0, 0.0])]
    P = kill_probability_matrix(intc, tgt)
    a_time = weapon_target_assignment(intc, tgt, objective="time")
    a_kp = weapon_target_assignment(intc, tgt, objective="kill_prob")
    assert len(a_kp) == 3 and set(a_kp.values()) <= {0, 1}  # valid complete assignment
    # The kill-prob objective spends the surplus interceptor to cut expected leakers.
    assert expected_leakers(a_kp, P) < expected_leakers(a_time, P)


def test_wta_invalid_objective():
    with pytest.raises(ValueError):
        weapon_target_assignment([np.zeros(4)], [np.zeros(4)], objective="bogus")
