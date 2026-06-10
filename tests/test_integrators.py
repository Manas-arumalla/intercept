"""Tests for the RK4 integrator: exactness on polynomials and convergence order."""

import numpy as np

from intercept.core import RK4, PointMass2D, integrate_rk4


def test_constant_velocity_is_exact():
    # No control => straight line; RK4 integrates linear motion exactly.
    d = PointMass2D()
    state = np.array([0.0, 0.0, 100.0, -50.0])
    s = state.copy()
    dt = 0.01
    for _ in range(100):  # 1 second
        s = integrate_rk4(d, 0.0, s, np.zeros(2), dt)
    assert np.allclose(s[:2], [100.0, -50.0], atol=1e-9)
    assert np.allclose(s[2:], [100.0, -50.0], atol=1e-12)


def test_constant_acceleration_is_exact():
    # x = x0 + v0 t + 0.5 a t^2 is degree-2; RK4 is exact for it.
    d = PointMass2D()
    s = np.array([0.0, 0.0, 0.0, 0.0])
    a = np.array([2.0, -3.0])
    dt = 0.01
    n = 200  # t = 2 s
    for _ in range(n):
        s = integrate_rk4(d, 0.0, s, a, dt)
    t = n * dt
    expected_pos = 0.5 * a * t**2
    expected_vel = a * t
    assert np.allclose(s[:2], expected_pos, atol=1e-8)
    assert np.allclose(s[2:], expected_vel, atol=1e-10)


def test_rk4_strategy_matches_function():
    d = PointMass2D()
    s = np.array([1.0, 2.0, 3.0, 4.0])
    u = np.array([0.3, 0.7])
    a = RK4().step(d, 0.0, s, u, 0.02)
    b = integrate_rk4(d, 0.0, s, u, 0.02)
    assert np.allclose(a, b)


def test_does_not_mutate_inputs():
    d = PointMass2D()
    s = np.array([1.0, 2.0, 3.0, 4.0])
    s_copy = s.copy()
    integrate_rk4(d, 0.0, s, np.array([1.0, 1.0]), 0.01)
    assert np.allclose(s, s_copy)
