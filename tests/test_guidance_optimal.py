"""Tests for optimal (OGL), sliding-mode (SMG), and MPC guidance laws."""

import numpy as np
import pytest

from intercept.adversary import scripted
from intercept.core import Engagement, Entity, PointMass2D
from intercept.guidance import (
    OptimalGuidance,
    SlidingModeGuidance,
    optimal_guidance,
    sliding_mode,
    zem_pn,
)
from intercept.guidance.mpc import MPCGuidance, has_casadi


def _engage(
    guidance,
    target_state,
    *,
    target_ctrl=None,
    a_max=120.0,
    speed=650.0,
    kill_radius=10.0,
    dt=0.01,
    t_max=30.0,
):
    tp = np.array(target_state, float)[:2]
    aim = tp / np.linalg.norm(tp)
    interceptor = Entity(
        "interceptor",
        PointMass2D(a_max=a_max),
        np.array([0.0, 0.0, speed * aim[0], speed * aim[1]]),
        controller=guidance,
        role="interceptor",
    )
    target = Entity(
        "target",
        PointMass2D(),
        np.array(target_state, float),
        controller=target_ctrl,
        role="target",
    )
    return Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=dt,
        t_max=t_max,
        kill_radius=kill_radius,
    ).run()


# --- OGL -------------------------------------------------------------------


def test_ogl_equals_zem_pn_n3():
    ogl = OptimalGuidance("target", n_prime=3.0)
    zem = zem_pn("target", N=3.0)
    own = np.array([0.0, 0.0, 600.0, 0.0])
    tgt = np.array([3000.0, 800.0, -250.0, 30.0])
    assert np.allclose(ogl.command(0.0, own, tgt), zem.command(0.0, own, tgt))


def test_ogl_intercepts_crossing():
    res = _engage(
        optimal_guidance("target"), [4000.0, 1500.0, -250.0, 0.0], a_max=120.0, speed=600.0
    )
    assert res.intercepted


def test_ogl_augment_reduces_miss_vs_maneuver():
    target_state = [4000.0, 1000.0, -250.0, 0.0]
    maneuver = scripted.step_maneuver(accel=90.0)
    plain = _engage(
        OptimalGuidance("target", augment=False),
        target_state,
        target_ctrl=maneuver,
        a_max=150.0,
        kill_radius=0.5,
    )
    aug = _engage(
        OptimalGuidance("target", augment=True),
        target_state,
        target_ctrl=maneuver,
        a_max=150.0,
        kill_radius=0.5,
    )
    assert aug.miss_distance < plain.miss_distance


def test_ogl_validation():
    with pytest.raises(ValueError):
        OptimalGuidance("target", n_prime=-1.0)


# --- SMG -------------------------------------------------------------------


def test_smg_zero_command_on_collision_course():
    smg = SlidingModeGuidance("target")
    own = np.array([0.0, 0.0, 100.0, 0.0])
    tgt = np.array([500.0, 0.0, -100.0, 0.0])  # zero LOS rate
    assert np.allclose(smg.command(0.0, own, tgt), [0.0, 0.0], atol=1e-9)


def test_smg_intercepts_crossing():
    res = _engage(
        sliding_mode("target", eta=60.0), [4000.0, 1500.0, -250.0, 0.0], a_max=120.0, speed=600.0
    )
    assert res.intercepted


def test_smg_robust_to_maneuvering_target():
    # Sliding mode is designed for robustness to unknown target maneuvers.
    res = _engage(
        sliding_mode("target", eta=100.0),
        [4000.0, 800.0, -250.0, 0.0],
        target_ctrl=scripted.weave(amplitude=80.0, frequency=0.4),
        a_max=250.0,
        speed=700.0,
        kill_radius=15.0,
    )
    assert res.miss_distance < 20.0


def test_smg_validation():
    with pytest.raises(ValueError):
        SlidingModeGuidance("target", boundary=0.0)


# --- MPC (requires CasADi) -------------------------------------------------

casadi_required = pytest.mark.skipif(not has_casadi(), reason="CasADi not installed")


@casadi_required
def test_mpc_intercepts_crossing():
    res = _engage(
        MPCGuidance("target", a_max=150.0, replan_every=8),
        [3500.0, 1000.0, -250.0, 0.0],
        a_max=150.0,
        speed=650.0,
        kill_radius=15.0,
        t_max=25.0,
    )
    assert res.intercepted


def _terminal_heading(res) -> float:
    v = res.states["interceptor"][-1, 2:4]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _angle_err(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


@casadi_required
def test_mpc_impact_angle_steers_terminal_heading():
    # The impact-angle objective should pull the terminal heading toward the requested direction,
    # measurably closer than the same MPC without the objective (a robust relative check).
    desired = -35.0  # approach descending
    geom = [3200.0, 700.0, -220.0, 0.0]
    plain = _engage(
        MPCGuidance("target", a_max=220.0, replan_every=6),
        geom,
        a_max=220.0,
        speed=650.0,
        kill_radius=25.0,
        t_max=25.0,
    )
    angled = _engage(
        MPCGuidance(
            "target",
            a_max=220.0,
            replan_every=6,
            impact_angle_deg=desired,
            w_angle=8.0,
            w_terminal=10.0,
        ),
        geom,
        a_max=220.0,
        speed=650.0,
        kill_radius=25.0,
        t_max=25.0,
    )
    err_plain = _angle_err(_terminal_heading(plain), desired)
    err_angled = _angle_err(_terminal_heading(angled), desired)
    # The objective measurably steers the terminal heading toward the requested direction.
    # (Miss/angle trade-off and a full intercept are tuned with a longer horizon in the demo.)
    assert err_angled < err_plain


@casadi_required
def test_mpc_validation():
    with pytest.raises(ValueError):
        MPCGuidance("target", a_max=-1.0)


# --- 3-D MPC ---------------------------------------------------------------


@casadi_required
def test_mpc3d_intercepts_maneuvering_target():
    from intercept.adversary import barrel_roll
    from intercept.core import G0, AeroMissile3D
    from intercept.guidance.mpc import MPCGuidance3D

    plant = AeroMissile3D(a_max=45 * G0, tau=0.2)
    interceptor = Entity(
        "interceptor",
        plant,
        plant.initial_state([0, 0, 0], [1000.0, 0.0, 150.0]),
        controller=MPCGuidance3D("target", a_max=45 * G0, horizon=3.0, replan_every=8),
        role="interceptor",
    )
    tgt = AeroMissile3D(a_max=20 * G0, tau=0.3)
    target = Entity(
        "target",
        tgt,
        tgt.initial_state([7000.0, 1000.0, 3000.0], [-700.0, 40.0, 0.0]),
        controller=barrel_roll(accel=12 * G0, rate=0.9),
        role="target",
    )
    res = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.02,
        t_max=25.0,
        kill_radius=25.0,
    ).run()
    assert res.intercepted


@casadi_required
def test_mpc3d_validation():
    from intercept.guidance.mpc import MPCGuidance3D

    with pytest.raises(ValueError):
        MPCGuidance3D("target", a_max=-1.0)
