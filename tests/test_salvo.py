"""Tests for impact-time-control (salvo) guidance — synchronized simultaneous arrival."""

from __future__ import annotations

import numpy as np
import pytest

from intercept.core import Engagement, Entity, PointMass2D
from intercept.guidance import ImpactTimeGuidance, impact_time_guidance, true_pn

TARGET0 = np.array([5000.0, 0.0, -100.0, 0.0])
LAUNCHES = [
    np.array([0.0, 0.0, 600.0, 0.0]),
    np.array([0.0, -1500.0, 580.0, 80.0]),
    np.array([800.0, 2000.0, 560.0, -60.0]),
    np.array([-500.0, 1200.0, 600.0, 40.0]),
]


def _arrival(launch, factory):
    interceptor = Entity(
        "interceptor",
        PointMass2D(a_max=300.0),
        launch.copy(),
        controller=factory(),
        role="interceptor",
    )
    target = Entity("target", PointMass2D(), TARGET0.copy(), role="target")
    res = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=60.0,
        kill_radius=20.0,
    ).run()
    return res.intercepted, res.intercept_time


def test_impact_time_validation():
    with pytest.raises(ValueError):
        ImpactTimeGuidance("t", t_impact=-1.0)
    with pytest.raises(ValueError):
        ImpactTimeGuidance("t", t_impact=10.0, N=1.0)


def test_salvo_synchronizes_arrival():
    """A battery from different ranges all arrives near the commanded time (tight spread)."""
    pn = [_arrival(L, lambda: true_pn("target", N=4.0)) for L in LAUNCHES]
    pn_times = [t for hit, t in pn if hit]
    assert len(pn_times) == len(LAUNCHES)
    pn_spread = max(pn_times) - min(pn_times)

    t_impact = max(pn_times) + 1.6  # feasible: above slowest natural arrival
    salvo = [_arrival(L, lambda tI=t_impact: impact_time_guidance("target", tI)) for L in LAUNCHES]
    assert all(hit for hit, _ in salvo)  # every interceptor still hits
    salvo_times = [t for _, t in salvo]
    spread = max(salvo_times) - min(salvo_times)
    assert spread < 0.5  # synchronized to a fraction of a second
    assert spread < pn_spread  # far tighter than uncoordinated PN
    assert abs(float(np.mean(salvo_times)) - t_impact) < 0.6  # arrives near the commanded time


def test_impact_time_reduces_to_pn_when_no_spare_time():
    """If t_impact is at/below the natural time-to-go, ITCG just flies PN (no delay bias)."""
    # A very small t_impact => e_t <= 0 from the start => pure PN command.
    own = np.array([0.0, 0.0, 600.0, 0.0])
    tgt = np.array([4000.0, 600.0, -200.0, 0.0])
    itcg = ImpactTimeGuidance("target", t_impact=0.5, N=4.0)
    pn = true_pn("target", N=4.0)
    assert np.allclose(itcg.command(0.0, own, tgt), pn.command(0.0, own, tgt))
