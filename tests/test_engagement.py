"""Tests for the engagement loop: termination logic, determinism, logging, and intercept."""

import numpy as np

from intercept.core import Engagement, Entity, PointMass2D
from intercept.core.engagement import TerminationReason
from intercept.core.entities import zero_controller


def _straight(name, state, role):
    return Entity(name=name, dynamics=PointMass2D(), state=np.array(state, dtype=float), role=role)


def test_head_on_collision_intercepts():
    # Two bodies closing head-on at the same y => they collide.
    interceptor = _straight("interceptor", [0.0, 0.0, 100.0, 0.0], "interceptor")
    target = _straight("target", [1000.0, 0.0, -100.0, 0.0], "target")
    result = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=20.0,
        kill_radius=5.0,
    ).run()
    assert result.intercepted
    assert result.reason == TerminationReason.INTERCEPT
    assert result.miss_distance <= 5.0
    assert result.intercept_time is not None
    # Closing at 200 m/s over 1000 m => ~5 s.
    assert abs(result.intercept_time - 5.0) < 0.1


def test_parallel_miss_records_closest_approach():
    # Bodies pass with a fixed perpendicular offset and never get within kill radius.
    interceptor = _straight("interceptor", [0.0, 0.0, 100.0, 0.0], "interceptor")
    target = _straight("target", [1000.0, 200.0, -100.0, 0.0], "target")
    result = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=20.0,
        kill_radius=5.0,
    ).run()
    assert not result.intercepted
    assert result.reason == TerminationReason.MISS
    assert np.isclose(result.miss_distance, 200.0, atol=1.0)


def test_timeout_when_separating():
    # Moving apart from the start => never intercept, ends on recede/timeout.
    interceptor = _straight("interceptor", [0.0, 0.0, -100.0, 0.0], "interceptor")
    target = _straight("target", [100.0, 0.0, 100.0, 0.0], "target")
    result = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=2.0,
        kill_radius=5.0,
        stop_on_recede=True,
    ).run()
    assert not result.intercepted


def test_determinism():
    def make():
        i = _straight("interceptor", [0.0, 0.0, 100.0, 10.0], "interceptor")
        t = _straight("target", [1000.0, 50.0, -100.0, 0.0], "target")
        return Engagement([i, t], interceptor="interceptor", target="target", dt=0.01).run()

    r1, r2 = make(), make()
    assert np.allclose(r1.states["interceptor"], r2.states["interceptor"])
    assert r1.miss_distance == r2.miss_distance


def test_logs_have_consistent_shapes():
    i = _straight("interceptor", [0.0, 0.0, 100.0, 0.0], "interceptor")
    t = _straight("target", [500.0, 0.0, -100.0, 0.0], "target")
    result = Engagement([i, t], interceptor="interceptor", target="target", dt=0.01).run()
    n = len(result.times)
    assert result.states["interceptor"].shape == (n, 4)
    assert result.controls["interceptor"].shape == (n, 2)
    assert result.control_effort("interceptor") >= 0.0


def test_zero_controller_coasts():
    c = zero_controller(2)
    assert np.allclose(c(0.0, np.zeros(4), {}), [0.0, 0.0])
