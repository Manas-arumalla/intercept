"""Tests for pincer coverage guidance (branch-covering interceptor pair)."""

from __future__ import annotations

import numpy as np
import pytest

from intercept.adversary import surprise_break
from intercept.core import Engagement, Entity
from intercept.core.aero import G0, AeroMissile2D
from intercept.guidance import PincerGuidance, pincer_pair, true_pn
from intercept.multiagent.swarm import MultiEngagement


def test_pincer_validation():
    with pytest.raises(ValueError):
        PincerGuidance("t", true_pn("t"), side=0.5)
    with pytest.raises(ValueError):
        PincerGuidance("t", true_pn("t"), side=1.0, beta=1.5)
    with pytest.raises(ValueError):
        PincerGuidance("t", true_pn("t"), side=1.0, r_split=1000.0, r_merge=2000.0)


def test_pincer_offsets_outside_merge_and_equals_base_inside():
    own = np.array([0.0, 0.0, 900.0, 0.0])
    base = true_pn("target", N=4.0)
    pin = PincerGuidance(
        "target", true_pn("target", N=4.0), +1.0, beta=0.2, r_split=5000.0, r_merge=2500.0
    )
    far = np.array([8000.0, 200.0, -700.0, 0.0])  # range > r_split: full offset
    assert not np.allclose(pin.command(0.0, own, far), base.command(0.0, own, far))
    near = np.array([2000.0, 50.0, -700.0, 0.0])  # range < r_merge: plain base law
    assert np.allclose(pin.command(0.0, own, near), base.command(0.0, own, near))


def test_pincer_converges_on_straight_target():
    idyn = AeroMissile2D(a_max=40 * G0, tau=0.2)
    edyn = AeroMissile2D(a_max=30 * G0, tau=0.3)
    tpos = np.array([8000.0, 200.0])
    aim = tpos / np.linalg.norm(tpos)
    g = PincerGuidance("target", true_pn("target", N=4.0), +1.0, beta=0.2)
    intc = Entity(
        "interceptor",
        idyn,
        idyn.initial_state([0.0, 0.0], 1000 * aim),
        controller=g,
        role="interceptor",
    )
    tgt = Entity("target", edyn, edyn.initial_state(tpos, [-700.0, 0.0]), role="target")
    res = Engagement(
        [intc, tgt],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=16.0,
        kill_radius=20.0,
    ).run()
    assert res.intercepted  # the split-then-merge still brings it home


def test_pincer_pair_covers_the_branch_a_redundant_pair_cannot():
    """The headline: vs a 30g surprise break a redundant True-PN pair fails on BOTH branches; the
    pincer pair catches each break with the interceptor covering that side (pure geometry)."""
    idyn = lambda: AeroMissile2D(a_max=40 * G0, tau=0.2)  # noqa: E731
    edyn = lambda: AeroMissile2D(a_max=30 * G0, tau=0.3)  # noqa: E731

    def run(sign, pincer):
        tp = np.array([8000.0, 200.0])
        aim = tp / np.linalg.norm(tp)
        laws = (
            list(
                pincer_pair(
                    "T0", lambda: true_pn("T0", N=4.0), beta=0.2, r_split=4000.0, r_merge=1200.0
                )
            )
            if pincer
            else [true_pn("T0", N=4.0), true_pn("T0", N=4.0)]
        )
        ints = [
            Entity(
                f"I{i}",
                idyn(),
                idyn().initial_state([0.0, (-1) ** i * 150.0], 1000 * aim),
                role="interceptor",
            )
            for i in range(2)
        ]
        tgt = Entity(
            "T0",
            edyn(),
            edyn().initial_state(tp, [-700.0, 0.0]),
            controller=surprise_break("I0", 30 * G0, 1800.0, sign),
            role="target",
        )
        eng = MultiEngagement(
            ints,
            [tgt],
            lambda tn, L=laws: L.pop(0) if L else true_pn(tn),
            dt=0.01,
            t_max=16.0,
            kill_radius=20.0,
            reassign_every=10**9,
        )
        return eng.run()

    for sign in (+1.0, -1.0):
        assert len(run(sign, pincer=False).kills) == 0  # redundant pair: defeated on both branches
        assert len(run(sign, pincer=True).kills) == 1  # pincer pair: each branch is covered
