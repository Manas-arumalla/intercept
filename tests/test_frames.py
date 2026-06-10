"""Tests for engagement geometry: range, LOS angle/rate, closing speed, ZEM."""

import numpy as np

from intercept.core import (
    closing_speed,
    los_angle,
    los_rate,
    range_to,
    relative_state,
    zero_effort_miss,
)


def test_range_and_relative_state():
    p = np.array([0.0, 0.0, 0.0, 0.0])
    t = np.array([3.0, 4.0, 1.0, -1.0])
    r, v = relative_state(p, t)
    assert np.allclose(r, [3.0, 4.0])
    assert np.allclose(v, [1.0, -1.0])
    assert range_to(p, t) == 5.0


def test_los_angle():
    p = np.array([0.0, 0.0, 0.0, 0.0])
    t = np.array([0.0, 10.0, 0.0, 0.0])
    assert np.isclose(los_angle(p, t), np.pi / 2)


def test_closing_speed_head_on():
    # Target approaching pursuer head-on at 100 m/s closing.
    p = np.array([0.0, 0.0, 50.0, 0.0])
    t = np.array([1000.0, 0.0, -50.0, 0.0])
    assert np.isclose(closing_speed(p, t), 100.0)


def test_los_rate_zero_on_collision_course():
    # Pure radial approach along the LOS => zero LOS rotation rate (PN commands zero).
    p = np.array([0.0, 0.0, 100.0, 0.0])
    t = np.array([500.0, 0.0, -100.0, 0.0])
    assert np.isclose(los_rate(p, t), 0.0)


def test_los_rate_nonzero_for_crossing():
    p = np.array([0.0, 0.0, 0.0, 0.0])
    t = np.array([100.0, 0.0, 0.0, 50.0])  # moving +y, perpendicular to LOS
    # lambda_dot = (rx*vy - ry*vx)/|r|^2 = (100*50 - 0)/100^2 = 0.5 rad/s
    assert np.isclose(los_rate(p, t), 0.5)


def test_zero_effort_miss():
    p = np.array([0.0, 0.0, 10.0, 0.0])
    t = np.array([100.0, 0.0, 10.0, 0.0])  # same velocity => constant relative position
    zem = zero_effort_miss(p, t, t_go=5.0)
    assert np.allclose(zem, [100.0, 0.0])
