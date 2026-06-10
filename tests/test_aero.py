"""Tests for the higher-fidelity AeroMissile2D dynamics (gravity, drag, g-limit, autopilot lag)."""

import numpy as np
import pytest

from intercept.core import G0, AeroMissile2D, Engagement, Entity
from intercept.core.integrators import integrate_rk4
from intercept.guidance import true_pn


def test_dimensions_and_initial_state():
    d = AeroMissile2D()
    assert d.state_dim == 6 and d.control_dim == 2
    s = d.initial_state([10.0, 20.0], [300.0, 0.0])
    assert np.allclose(s, [10.0, 20.0, 300.0, 0.0, 0.0, 0.0])


def test_gravity_and_parasitic_drag():
    d = AeroMissile2D(gravity=G0, k_drag=1e-5, k_induced=0.0, thrust=0.0)
    state = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0])
    dx = d.derivative(0.0, state, np.zeros(2))
    # vy' = -gravity; vx' = -k_drag * V^2 (drag opposes +x motion)
    assert dx[3] == pytest.approx(-G0)
    assert dx[2] == pytest.approx(-1e-5 * 100.0**2)


def test_autopilot_lag_first_order():
    d = AeroMissile2D(a_max=1e6, tau=0.5)  # huge g-limit so command isn't clipped
    state = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0])  # velocity +x
    cmd = np.array([0.0, 50.0])  # purely lateral (perp to +x)
    dx = d.derivative(0.0, state, cmd)
    # a_dot = (cmd_perp - a_achieved)/tau = (50 - 0)/0.5 = 100 on the y-accel state
    assert dx[5] == pytest.approx(50.0 / 0.5)


def test_g_limit_saturation():
    d = AeroMissile2D(a_max=100.0)
    assert np.isclose(np.linalg.norm(d.saturate(np.array([300.0, 400.0]))), 100.0)


def test_induced_drag_bleeds_speed_when_maneuvering():
    # Pulling lateral g adds along-track deceleration (induced drag) beyond parasitic drag.
    d = AeroMissile2D(gravity=0.0, k_drag=1e-5, k_induced=3e-4)
    v = np.array([0.0, 0.0, 200.0, 0.0])
    coasting = d.derivative(0.0, np.concatenate([v, [0.0, 0.0]]), np.zeros(2))
    turning = d.derivative(0.0, np.concatenate([v, [0.0, 300.0]]), np.zeros(2))
    # Along-track (x) deceleration is stronger while pulling 300 m/s^2 of lateral accel.
    assert turning[2] < coasting[2] < 0.0


def test_ballistic_arc_without_control():
    # A coasting body under gravity follows a downward arc.
    d = AeroMissile2D(gravity=G0, k_drag=0.0, k_induced=0.0)
    s = d.initial_state([0.0, 1000.0], [200.0, 0.0])
    for _ in range(100):  # 1 s at dt=0.01
        s = integrate_rk4(d, 0.0, s, np.zeros(2), 0.01)
    assert s[1] < 1000.0  # fell under gravity
    assert s[3] == pytest.approx(-G0 * 1.0, rel=1e-3)  # vy ~ -g*t


def test_pn_intercepts_nonmaneuvering_aero_target():
    # Sanity: PN still works against a (gravity-falling, drag-decelerating) aero target.
    plant = AeroMissile2D(a_max=40 * G0, tau=0.15)
    interceptor = Entity(
        "interceptor",
        plant,
        plant.initial_state([0.0, 0.0], [900.0, 100.0]),
        controller=true_pn("target", N=4.0),
        role="interceptor",
    )
    tgt = AeroMissile2D(a_max=1.0, tau=0.2)  # essentially non-maneuvering
    target = Entity(
        "target", tgt, tgt.initial_state([6000.0, 2000.0], [-600.0, 0.0]), role="target"
    )
    res = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.005,
        t_max=20.0,
        kill_radius=20.0,
    ).run()
    assert res.intercepted


def test_validation():
    with pytest.raises(ValueError):
        AeroMissile2D(a_max=-1.0)
    with pytest.raises(ValueError):
        AeroMissile2D(tau=0.0)
