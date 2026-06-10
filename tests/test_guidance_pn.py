"""Tests for Proportional Navigation and Augmented PN.

Validates the closed-form behavior (zero LOS-rate => zero command), that PN intercepts a crossing
target with modest control authority where pure pursuit lags, that the variants behave sensibly,
and that APN beats True PN against a constantly-maneuvering target.
"""

import numpy as np
import pytest

from intercept.adversary import scripted
from intercept.core import Engagement, Entity, PointMass2D
from intercept.guidance import AugmentedPN, ProportionalNavigation, true_pn
from intercept.guidance.apn import AugmentedPN as APN


def _make(
    interceptor_guidance,
    target_state,
    target_ctrl=None,
    a_max=100.0,
    speed=600.0,
    interceptor_heading=None,
    dt=0.005,
    t_max=30.0,
    kill_radius=10.0,
):
    target = Entity(
        "target",
        PointMass2D(),
        np.array(target_state, float),
        controller=target_ctrl,
        role="target",
    )
    if interceptor_heading is None:
        # aim straight at the target's initial position
        tp = np.array(target_state, float)[:2]
        interceptor_heading = tp / np.linalg.norm(tp)
    v0 = speed * np.asarray(interceptor_heading, float)
    interceptor = Entity(
        "interceptor",
        PointMass2D(a_max=a_max),
        np.array([0.0, 0.0, v0[0], v0[1]]),
        controller=interceptor_guidance,
        role="interceptor",
    )
    return Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=dt,
        t_max=t_max,
        kill_radius=kill_radius,
    ).run()


def test_zero_los_rate_gives_zero_command():
    # Pursuer and target on a perfect collision course (radial closing) => lambda_dot = 0.
    pn = true_pn("target", N=4.0)
    own = np.array([0.0, 0.0, 100.0, 0.0])
    tgt = np.array([500.0, 0.0, -100.0, 0.0])  # head-on along x
    cmd = pn.command(0.0, own, tgt)
    assert np.allclose(cmd, [0.0, 0.0], atol=1e-9)


def test_invalid_params():
    with pytest.raises(ValueError):
        ProportionalNavigation("target", N=-1.0)
    with pytest.raises(ValueError):
        ProportionalNavigation("target", variant="bogus")
    with pytest.raises(ValueError):
        AugmentedPN("target", N=0.0)


def test_pn_intercepts_crossing_target_with_low_authority():
    # Crossing target; PN should connect with modest authority (where pure pursuit lags).
    res = _make(
        true_pn("target", N=4.0),
        target_state=[4000.0, 1500.0, -250.0, 0.0],
        a_max=100.0,
        speed=600.0,
        kill_radius=10.0,
    )
    assert res.intercepted, f"PN failed to intercept: reason={res.reason}, miss={res.miss_distance}"


def test_pn_command_direction_leads_toward_target_motion():
    # Pursuer +x, target ahead crossing +y => PN should command +y (lead the crossing).
    pn = true_pn("target", N=4.0)
    own = np.array([0.0, 0.0, 300.0, 0.0])
    tgt = np.array([1000.0, 0.0, 0.0, 100.0])
    cmd = pn.command(0.0, own, tgt)
    assert cmd[1] > 0.0


def test_zem_and_true_pn_agree_for_nonmaneuvering():
    # For a non-maneuvering target the two forms should produce similar miss distances.
    tgt = [4000.0, 1200.0, -250.0, 0.0]
    r_true = _make(ProportionalNavigation("target", N=4, variant="true"), tgt, a_max=150.0)
    r_zem = _make(ProportionalNavigation("target", N=4, variant="zem"), tgt, a_max=150.0)
    assert r_true.intercepted and r_zem.intercepted


def test_apn_beats_pn_against_maneuvering_target():
    # A target pulling a constant lateral acceleration (hard turn) is the case APN is designed for.
    # Use a tiny kill radius so the engagement runs to closest approach and we compare the true
    # terminal miss distance (otherwise a generous kill radius hides the difference).
    target_state = [4000.0, 1000.0, -250.0, 0.0]
    maneuver = scripted.step_maneuver(accel=90.0, t_start=0.0)
    r_pn = _make(
        true_pn("target", N=4.0),
        target_state,
        target_ctrl=maneuver,
        a_max=150.0,
        speed=600.0,
        kill_radius=0.5,
    )
    r_apn = _make(
        APN("target", N=4.0),
        target_state,
        target_ctrl=maneuver,
        a_max=150.0,
        speed=600.0,
        kill_radius=0.5,
    )
    # APN's target-acceleration feedforward should yield a smaller terminal miss than plain PN.
    assert r_apn.miss_distance < r_pn.miss_distance


def test_apn_reduces_to_pn_on_first_call():
    # With no previous velocity sample, the feedforward term is zero => equals True PN.
    apn = AugmentedPN("target", N=4.0)
    pn = true_pn("target", N=4.0)
    own = np.array([0.0, 0.0, 300.0, 0.0])
    tgt = np.array([1000.0, 200.0, -100.0, 50.0])
    assert np.allclose(apn.command(0.0, own, tgt), pn.command(0.0, own, tgt))


def test_reset_clears_apn_state():
    apn = AugmentedPN("target", N=4.0)
    own = np.array([0.0, 0.0, 300.0, 0.0])
    apn.command(0.0, own, np.array([1000.0, 0.0, 0.0, 0.0]))
    apn.command(0.1, own, np.array([1000.0, 0.0, 0.0, 50.0]))
    apn.reset()
    assert apn._prev_t is None and apn._prev_vt is None
    assert np.allclose(apn._a_t, [0.0, 0.0])
