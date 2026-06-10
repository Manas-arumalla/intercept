"""Tests for L3 realism: ISA atmosphere, transonic drag, propulsion/mass, q-limited turning."""

import numpy as np
import pytest

from intercept.adversary import barrel_roll, weave
from intercept.core import Engagement, Entity, RealisticMissile2D, RealisticMissile3D
from intercept.core.atmosphere import drag_coefficient, isa, mach
from intercept.guidance import AugmentedPN, augmented_pn_3d

# --- atmosphere ------------------------------------------------------------


def test_isa_sea_level():
    rho, a, p, temp = isa(0.0)
    assert rho == pytest.approx(1.225, rel=0.01)
    assert a == pytest.approx(340.3, rel=0.01)
    assert p == pytest.approx(101325.0, rel=0.01)
    assert temp == pytest.approx(288.15, rel=0.001)


def test_isa_density_and_sound_decrease_with_altitude():
    rho0, a0, _, _ = isa(0.0)
    rho10, a10, _, _ = isa(10000.0)
    assert rho10 < 0.5 * rho0  # ~0.41 kg/m^3 at 10 km
    assert a10 < a0  # colder => slower sound


def test_drag_coefficient_transonic_peak():
    assert drag_coefficient(1.1) > drag_coefficient(0.3)  # transonic rise
    assert drag_coefficient(1.1) > drag_coefficient(3.0)  # supersonic relaxation


def test_mach_helper():
    assert mach(340.3, 0.0) == pytest.approx(1.0, rel=0.01)


# --- propulsion / mass -----------------------------------------------------


def test_thrust_schedule_and_mass_burnoff():
    d = RealisticMissile2D()  # boost 3 s, sustain 8 s
    assert d.thrust(1.0) == d.thrust_boost
    assert d.thrust(5.0) == d.thrust_sustain
    assert d.thrust(20.0) == 0.0
    assert d.mass(0.0) == pytest.approx(d.mass0)
    assert d.mass(100.0) == pytest.approx(d.dry_mass)  # fully burned
    assert d.dry_mass < d.mass(2.0) < d.mass0  # burning


# --- physics-derived turn limit -------------------------------------------


def test_available_g_drops_with_altitude_and_low_speed():
    d = RealisticMissile2D()
    fast_low = d.initial_state([0.0, 1000.0], [900.0, 0.0])  # fast, low altitude
    fast_high = d.initial_state([0.0, 15000.0], [900.0, 0.0])  # fast, high altitude
    slow_low = d.initial_state([0.0, 1000.0], [150.0, 0.0])  # slow, low altitude
    g_fast_low = d.max_lateral_accel(fast_low, t=5.0)
    assert d.max_lateral_accel(fast_high, t=5.0) < g_fast_low  # thinner air => less lift
    assert d.max_lateral_accel(slow_low, t=5.0) < g_fast_low  # low q => less lift


# --- realistic interception ------------------------------------------------


def test_realistic_2d_intercept():
    idyn = RealisticMissile2D()
    tdyn = RealisticMissile2D.target()
    tgt = np.array([7000.0, 4500.0])
    aim = tgt / np.linalg.norm(tgt)
    interceptor = Entity(
        "interceptor",
        idyn,
        idyn.initial_state([0.0, 0.0], 150.0 * aim),
        controller=AugmentedPN("target", N=4.0),
        role="interceptor",
    )
    target = Entity(
        "target",
        tdyn,
        tdyn.initial_state([7000.0, 4500.0], [-650.0, -80.0]),
        controller=weave(amplitude=12 * 9.81, frequency=0.3),
        role="target",
    )
    res = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.005,
        t_max=25.0,
        kill_radius=20.0,
    ).run()
    assert res.intercepted
    # No speed-cheat: the interceptor's peak speed is comparable to the target's, not 3x.
    peak = np.linalg.norm(res.states["interceptor"][:, 2:4], axis=1).max()
    assert 700.0 < peak < 1400.0


def test_realistic_3d_intercept():
    idyn = RealisticMissile3D()
    tdyn = RealisticMissile3D.target()
    interceptor = Entity(
        "interceptor",
        idyn,
        idyn.initial_state([0, 0, 0], [200.0, 0.0, 200.0]),
        controller=augmented_pn_3d("target", N=4.0),
        role="interceptor",
    )
    t0 = tdyn.initial_state([7000.0, 1500.0, 5000.0], [-650.0, -40.0, -60.0])
    target = Entity(
        "target", tdyn, t0, controller=barrel_roll(accel=10 * 9.81, rate=0.8), role="target"
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


def test_validation():
    with pytest.raises(ValueError):
        RealisticMissile2D(mass=10.0, dry_mass=20.0)  # dry > launch
    with pytest.raises(ValueError):
        RealisticMissile2D(cl_max=0.0)


def test_benchmark_realistic_model_builds_l3_and_intercepts():
    """The ParametricScenario `model="realistic"` wires the L3 plant into the benchmark core."""
    from intercept.benchmark import ParametricScenario, run_montecarlo
    from intercept.core import G0

    sc = ParametricScenario(
        name="L3_bench",
        model="realistic",
        interceptor_speed=900.0,
        interceptor_a_max=45 * G0,
        target_speed=700.0,
        target_a_max=35 * G0,
        interceptor_tau=0.18,
        target_tau=0.3,
        target_heading_deg=150.0,
        offset_min=-1500,
        offset_max=1500,
        range_min=8000,
        range_max=11000,
        dt=0.01,
        t_max=25.0,
        kill_radius=20.0,
        maneuver={"type": "weave", "g": 18, "frequency": 0.3},
    )
    spec = sc.sample(np.random.default_rng(0))
    assert isinstance(spec.interceptor_dynamics, RealisticMissile2D)
    assert isinstance(spec.target_dynamics, RealisticMissile2D)
    assert spec.interceptor_state.shape == (6,)  # [x, y, vx, vy, ax, ay]
    res = run_montecarlo(sc, lambda t: AugmentedPN(t, N=4.0), n_trials=8, seed=7)
    assert sum(r.intercepted for r in res) >= 6  # L3 APN catches the realistic weave
