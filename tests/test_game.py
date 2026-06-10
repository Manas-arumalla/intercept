"""Tests for game-theoretic guidance: Apollonius circle, intercept point, optimal evasion."""

import numpy as np
import pytest

from intercept.adversary import optimal_evader, straight
from intercept.core import Engagement, Entity, PointMass2D
from intercept.guidance import ApolloniusGuidance, apollonius_circle, true_pn
from intercept.guidance.game import intercept_point


def _engage(
    interceptor_guidance,
    target_state,
    *,
    target_ctrl=None,
    a_max=200.0,
    speed=700.0,
    kill_radius=15.0,
    dt=0.01,
    t_max=40.0,
):
    tp = np.array(target_state, float)[:2]
    aim = tp / np.linalg.norm(tp)
    interceptor = Entity(
        "interceptor",
        PointMass2D(a_max=a_max),
        np.array([0.0, 0.0, speed * aim[0], speed * aim[1]]),
        controller=interceptor_guidance,
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


# --- Apollonius circle -----------------------------------------------------


def test_apollonius_circle_ratio_property():
    # Every point on the circle must satisfy |X-E|/|X-P| = alpha.
    p = np.array([0.0, 0.0])
    e = np.array([1000.0, 300.0])
    alpha = 0.4
    center, radius = apollonius_circle(p, e, alpha)
    for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        x = center + radius * np.array([np.cos(th), np.sin(th)])
        ratio = np.linalg.norm(x - e) / np.linalg.norm(x - p)
        assert ratio == pytest.approx(alpha, rel=1e-6)


def test_apollonius_circle_validation():
    with pytest.raises(ValueError):
        apollonius_circle(np.zeros(2), np.ones(2), 1.0)
    with pytest.raises(ValueError):
        apollonius_circle(np.zeros(2), np.ones(2), 0.0)


def test_apollonius_interior_is_evader_dominant():
    # The circle center is inside; check it is closer in time to the evader than the pursuer.
    p = np.array([0.0, 0.0])
    e = np.array([1000.0, 0.0])
    alpha = 0.5
    center, _ = apollonius_circle(p, e, alpha)
    # time-to-reach center: evader |c-e|/v_e vs pursuer |c-p|/v_p, with v_e/v_p = alpha
    t_e = np.linalg.norm(center - e) / alpha
    t_p = np.linalg.norm(center - p) / 1.0
    assert t_e < t_p  # interior favors the evader


# --- intercept point -------------------------------------------------------


def test_intercept_point_head_on():
    # Pursuer at origin (speed 300), evader at (1000,0) closing at 100 m/s along -x.
    p = np.array([0.0, 0.0])
    e = np.array([1000.0, 0.0])
    v = np.array([-100.0, 0.0])
    pt = intercept_point(p, e, v, pursuer_speed=300.0)
    # closing speed 400 over 1000 m => t=2.5 s; evader reaches x=1000-250=750
    assert pt is not None
    assert pt[0] == pytest.approx(750.0, rel=1e-3)
    assert pt[1] == pytest.approx(0.0, abs=1e-6)


def test_intercept_point_uncatchable_returns_none():
    # Evader faster and running directly away => no intercept.
    p = np.array([0.0, 0.0])
    e = np.array([100.0, 0.0])
    v = np.array([300.0, 0.0])
    assert intercept_point(p, e, v, pursuer_speed=200.0) is None


# --- ApolloniusGuidance ----------------------------------------------------


def test_apollonius_guidance_intercepts_crossing():
    res = _engage(ApolloniusGuidance("target"), [4000.0, 1500.0, -250.0, 0.0], a_max=150.0)
    assert res.intercepted


def test_apollonius_guidance_zero_velocity_is_safe():
    g = ApolloniusGuidance("target")
    assert np.allclose(
        g.command(0.0, np.array([0.0, 0.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0, 0.0])), [0.0, 0.0]
    )


# --- optimal evader --------------------------------------------------------


def test_optimal_evader_turns_away_from_pursuer():
    ev = optimal_evader("interceptor", gain=6.0)
    # Evader at origin moving +x; pursuer ahead at +x => evader should command a turn (nonzero).
    own = np.array([0.0, 0.0, 200.0, 0.0])
    world = {"interceptor": np.array([1000.0, 0.0, 0.0, 0.0])}
    cmd = ev(0.0, own, world)
    assert np.linalg.norm(cmd) > 0.0
    # Pursuer directly behind (-x): already fleeing => near-zero command.
    world_behind = {"interceptor": np.array([-1000.0, 0.0, 0.0, 0.0])}
    assert np.linalg.norm(ev(0.0, own, world_behind)) == pytest.approx(0.0, abs=1e-6)


def test_optimal_evader_delays_capture_vs_straight():
    # A fleeing (optimal) evader should be caught later than a non-fleeing straight target.
    target_state = [3000.0, 0.0, 250.0, 0.0]  # target initially moving +x (away-ish)
    r_straight = _engage(
        true_pn("target", N=4.0),
        [3000.0, 0.0, -250.0, 0.0],
        target_ctrl=straight(),
        a_max=300.0,
        speed=700.0,
    )
    r_evader = _engage(
        true_pn("target", N=4.0),
        target_state,
        target_ctrl=optimal_evader("interceptor"),
        a_max=300.0,
        speed=700.0,
    )
    assert r_straight.intercepted and r_evader.intercepted
    assert r_evader.intercept_time > r_straight.intercept_time
