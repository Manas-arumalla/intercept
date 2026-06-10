"""Tests for the advanced 3-D evasion maneuvers and the realistic complex-trajectory engagement."""

from __future__ import annotations

import numpy as np

from intercept.adversary import combine, serpentine3d, terminal_spiral


def _state(pos, vel):
    return np.array([*pos, *vel, 0.0, 0.0, 0.0])


# --- maneuver primitives ---------------------------------------------------


def test_serpentine3d_perpendicular_and_bounded():
    ctrl = serpentine3d(accel=100.0, frequency=0.2, tilt=0.5)
    own = _state([0, 0, 5000], [600.0, 0.0, 0.0])
    mags = []
    for t in np.linspace(0.0, 5.0, 40):
        a = ctrl(t, own, {})
        assert abs(float(a @ own[3:6])) < 1e-9  # perpendicular to velocity
        mags.append(float(np.linalg.norm(a)))
    assert max(mags) <= 100.0 + 1e-9  # magnitude never exceeds the amplitude


def test_terminal_spiral_intensifies_as_pursuer_closes():
    ctrl = terminal_spiral(
        "pursuer", base_accel=5.0, max_accel=50.0, trigger_range=1000.0, rate=2.0
    )
    own = _state([0, 0, 5000], [600.0, 0.0, 0.0])
    far = {"pursuer": _state([5000.0, 0, 5000], [-600.0, 0, 0])}  # 5 km away
    near = {"pursuer": _state([100.0, 0, 5000], [-600.0, 0, 0])}  # 100 m away
    a_far = float(np.linalg.norm(ctrl(0.3, own, far)))
    a_near = float(np.linalg.norm(ctrl(0.3, own, near)))
    assert a_far == 5.0  # baseline beyond the trigger
    assert a_near > 45.0  # ramped toward max in the endgame
    assert a_near > a_far


def test_terminal_spiral_no_pursuer_falls_back_to_base():
    ctrl = terminal_spiral(
        "pursuer", base_accel=7.0, max_accel=50.0, trigger_range=1000.0, rate=2.0
    )
    own = _state([0, 0, 5000], [600.0, 0.0, 0.0])
    assert np.isclose(float(np.linalg.norm(ctrl(0.0, own, {}))), 7.0)


def test_combine_sums_commands():
    a = serpentine3d(accel=30.0, frequency=0.1, tilt=0.3)
    b = serpentine3d(accel=20.0, frequency=0.4, tilt=0.7)
    c = combine(a, b)
    own = _state([0, 0, 5000], [500.0, 10.0, -5.0])
    expected = a(1.3, own, {}) + b(1.3, own, {})
    assert np.allclose(c(1.3, own, {}), expected)


# --- realistic engagement (speed parity) -----------------------------------


def test_advanced_engagement_intercepts_without_speed_cheat():
    from experiments.p14_advanced_evasion import NOMINAL, build

    eng, _ = build(NOMINAL)
    res = eng.run()
    assert res.intercepted

    vi = np.linalg.norm(res.states["interceptor"][:, 3:6], axis=1)
    vt = np.linalg.norm(res.states["target"][:, 3:6], axis=1)
    # The interceptor must NOT win on raw speed: closing-speed edge at the merge stays modest,
    # and the threat is genuinely fast (comparable peak speeds).
    merge_edge = vi[-1] / vt[-1] - 1.0
    assert merge_edge < 0.6  # realistic SAM-vs-supersonic edge, not 2-3x
    assert vt.max() > 0.8 * vi.max()  # threat top speed comparable to interceptor
