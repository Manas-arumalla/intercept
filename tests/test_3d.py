"""Tests for the 3-D extension: dynamics, LOS geometry, and 3-D PN/APN interception."""

from collections.abc import Mapping

import numpy as np
import pytest

from intercept.core import G0, AeroMissile3D, Engagement, Entity, PointMass3D
from intercept.core import frames3d as f3
from intercept.guidance import (
    augmented_pn_3d,
    optimal_guidance_3d,
    sliding_mode_3d,
    true_pn_3d,
)

# --- dynamics --------------------------------------------------------------


def test_pointmass3d_dims_and_accessors():
    d = PointMass3D()
    assert d.state_dim == 6 and d.control_dim == 3
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.allclose(d.position(s), [1, 2, 3])
    assert np.allclose(d.velocity(s), [4, 5, 6])


def test_pointmass3d_double_integrator():
    d = PointMass3D()
    dx = d.derivative(0.0, np.array([0, 0, 0, 1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.5]))
    assert np.allclose(dx, [1.0, 2.0, 3.0, 0.0, 0.0, 0.5])


def test_aero3d_gravity_along_minus_z_and_lag():
    d = AeroMissile3D(a_max=1e6, tau=0.5, gravity=G0, k_drag=0.0, k_induced=0.0)
    s = d.initial_state([0, 0, 1000.0], [200.0, 0.0, 0.0])
    dx = d.derivative(0.0, s, np.array([0.0, 0.0, 50.0]))  # vertical (⟂ velocity) command
    assert dx[5] == pytest.approx(-G0 + 0.0)  # vz' has gravity (no achieved accel yet)
    assert dx[8] == pytest.approx(50.0 / 0.5)  # az' autopilot lag toward command


# --- 3-D geometry ----------------------------------------------------------


def test_los_rate_vector_matches_2d_zcomponent():
    p = np.array([0, 0, 0, 0.0, 0, 0])
    t = np.array([100.0, 0, 0, 0.0, 50.0, 0.0])  # moving +y
    omega = f3.los_rate_vector(p, t)
    # Ω = r×v/|r|^2 = (100,0,0)×(0,50,0)/100^2 = (0,0,5000)/10000 = (0,0,0.5)
    assert np.allclose(omega, [0.0, 0.0, 0.5])


def test_closing_speed_3d_head_on():
    p = np.array([0, 0, 0, 50.0, 0, 0])
    t = np.array([1000.0, 0, 0, -50.0, 0, 0])
    assert f3.closing_speed(p, t) == pytest.approx(100.0)


# --- 3-D interception ------------------------------------------------------


def _barrel_roll_3d(amplitude: float, rate: float):
    """Target maneuver: lateral accel of fixed magnitude rotating about the velocity axis."""

    def controller(t: float, own: np.ndarray, world: Mapping[str, np.ndarray]) -> np.ndarray:
        v = own[3:6]
        s = float(np.linalg.norm(v))
        if s < 1e-9:
            return np.zeros(3)
        v_hat = v / s
        ref = np.array([0.0, 0.0, 1.0])
        e1 = np.cross(v_hat, ref)
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(v_hat, np.array([0.0, 1.0, 0.0]))
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(v_hat, e1)
        return amplitude * (np.cos(rate * t) * e1 + np.sin(rate * t) * e2)

    return controller


def test_true_pn_3d_intercepts_maneuvering_target():
    plant = PointMass3D(a_max=50 * G0)
    interceptor = Entity(
        "interceptor",
        plant,
        np.array([0, 0, 0, 700.0, 0.0, 100.0]),
        controller=true_pn_3d("target", N=4.0),
        role="interceptor",
    )
    target = Entity(
        "target",
        PointMass3D(),
        np.array([5000.0, 1500.0, 2000.0, -250.0, 0.0, 0.0]),
        controller=_barrel_roll_3d(amplitude=80.0, rate=1.0),
        role="target",
    )
    res = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=25.0,
        kill_radius=20.0,
    ).run()
    assert res.intercepted
    # Trajectory is genuinely 3-D (interceptor leaves the initial plane).
    assert np.ptp(res.states["interceptor"][:, 2]) > 100.0


def test_augmented_pn_3d_intercepts_aero_target():
    plant = AeroMissile3D(a_max=45 * G0, tau=0.2)
    interceptor = Entity(
        "interceptor",
        plant,
        plant.initial_state([0, 0, 0], [1000.0, 0.0, 150.0]),
        controller=augmented_pn_3d("target", N=4.0),
        role="interceptor",
    )
    tgt = AeroMissile3D(a_max=20 * G0, tau=0.3)
    target = Entity(
        "target",
        tgt,
        tgt.initial_state([7000.0, 1000.0, 3000.0], [-700.0, 50.0, 0.0]),
        controller=_barrel_roll_3d(amplitude=120.0, rate=0.8),
        role="target",
    )
    res = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=25.0,
        kill_radius=25.0,
    ).run()
    assert res.intercepted


def test_pn3d_validation():
    with pytest.raises(ValueError):
        true_pn_3d("target", N=0.0)


# --- 3-D benchmark ---------------------------------------------------------


def test_scenario3d_samples_3d_spec():
    from intercept.benchmark import ParametricScenario3D
    from intercept.core import AeroMissile3D

    sc = ParametricScenario3D(
        name="b",
        model="aero",
        interceptor_a_max=40 * G0,
        maneuver={"type": "barrel_roll", "g": 15, "rate": 1.2},
    )
    spec = sc.sample(np.random.default_rng(0))
    assert isinstance(spec.interceptor_dynamics, AeroMissile3D)
    assert spec.interceptor_state.shape == (9,) and spec.target_state.shape == (9,)
    assert spec.target_controller is not None  # barrel-roll controller built
    assert spec.target_state[2] > 0.0  # target starts at altitude (z>0)


def test_make_maneuver_3d_rejects_unknown():
    from intercept.benchmark import make_maneuver_3d

    assert make_maneuver_3d(None) is None
    assert make_maneuver_3d({"type": "straight"}) is None
    with pytest.raises(ValueError):
        make_maneuver_3d({"type": "bogus"})


def test_optimal_and_sliding_mode_3d_intercept_barrel_roll():
    """3-D optimal (OGL) and sliding-mode laws recover the barrel-roll where True PN fails."""
    from intercept.benchmark import ParametricScenario3D, run_montecarlo

    sc = ParametricScenario3D(
        name="b2",
        model="aero",
        interceptor_speed=1000.0,
        interceptor_a_max=35 * G0,
        target_speed=700.0,
        target_a_max=30 * G0,
        interceptor_tau=0.25,
        target_tau=0.3,
        range_min=8000.0,
        range_max=9000.0,
        offset_min=-500.0,
        offset_max=500.0,
        alt_min=3000.0,
        alt_max=4000.0,
        target_azimuth_deg=200.0,
        target_elevation_deg=-2.0,
        dt=0.01,
        t_max=26.0,
        kill_radius=20.0,
        maneuver={"type": "barrel_roll", "g": 20, "rate": 1.4},
    )
    ogl = run_montecarlo(sc, lambda t: optimal_guidance_3d(t, augment=True), n_trials=12, seed=3)
    smg = run_montecarlo(sc, lambda t: sliding_mode_3d(t, eta=300.0), n_trials=12, seed=3)
    assert sum(r.intercepted for r in ogl) >= 11
    assert sum(r.intercepted for r in smg) >= 11


def test_pn3d_validation_optimal_sliding():
    with pytest.raises(ValueError):
        optimal_guidance_3d("target").__class__("target", n_prime=0.0)
    with pytest.raises(ValueError):
        sliding_mode_3d("target").__class__("target", N=0.0)


def test_benchmark3d_apn_beats_true_pn_on_barrel_roll():
    """3-D analogue of the realism result: the barrel-roll defeats True PN-3D; APN-3D recovers."""
    from intercept.benchmark import ParametricScenario3D, run_montecarlo

    common = dict(
        model="aero",
        interceptor_speed=1000.0,
        interceptor_a_max=35 * G0,
        target_speed=700.0,
        target_a_max=30 * G0,
        interceptor_tau=0.25,
        target_tau=0.3,
        range_min=8000.0,
        range_max=9000.0,
        offset_min=-500.0,
        offset_max=500.0,
        alt_min=3000.0,
        alt_max=4000.0,
        target_azimuth_deg=200.0,
        target_elevation_deg=-2.0,
        dt=0.01,
        t_max=26.0,
        kill_radius=20.0,
        maneuver={"type": "barrel_roll", "g": 20, "rate": 1.4},
    )
    sc = ParametricScenario3D(name="b2", **common)
    pn = run_montecarlo(sc, lambda t: true_pn_3d(t, N=4.0), n_trials=12, seed=3)
    apn = run_montecarlo(sc, lambda t: augmented_pn_3d(t, N=4.0), n_trials=12, seed=3)
    assert sum(r.intercepted for r in apn) >= 11  # APN-3D catches it
    assert sum(r.intercepted for r in pn) <= 3  # True PN-3D largely fails
