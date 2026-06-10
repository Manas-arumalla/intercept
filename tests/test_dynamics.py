"""Tests for point-mass dynamics: derivative correctness, saturation, drag, validation."""

import numpy as np
import pytest

from intercept.core import PointMass2D


def test_dimensions_and_labels():
    d = PointMass2D()
    assert d.state_dim == 4
    assert d.control_dim == 2
    assert d.state_labels == ("x", "y", "vx", "vy")
    assert d.control_labels == ("ax", "ay")


def test_derivative_double_integrator():
    d = PointMass2D()
    state = np.array([1.0, 2.0, 3.0, -4.0])
    control = np.array([0.5, -0.25])
    dx = d.derivative(0.0, state, control)
    # position derivative is velocity; velocity derivative is acceleration (no drag).
    assert np.allclose(dx, [3.0, -4.0, 0.5, -0.25])


def test_position_velocity_accessors():
    d = PointMass2D()
    state = np.array([10.0, 20.0, 1.0, 2.0])
    assert np.allclose(d.position(state), [10.0, 20.0])
    assert np.allclose(d.velocity(state), [1.0, 2.0])


def test_saturation_clips_magnitude():
    d = PointMass2D(a_max=10.0)
    u = d.saturate(np.array([30.0, 40.0]))  # magnitude 50 -> clip to 10
    assert np.isclose(np.linalg.norm(u), 10.0)
    assert np.allclose(u, [6.0, 8.0])


def test_saturation_passes_small_commands():
    d = PointMass2D(a_max=10.0)
    u = d.saturate(np.array([3.0, 4.0]))  # magnitude 5 < 10
    assert np.allclose(u, [3.0, 4.0])


def test_drag_opposes_velocity():
    d = PointMass2D(drag_coeff=0.1)
    state = np.array([0.0, 0.0, 10.0, 0.0])
    dx = d.derivative(0.0, state, np.array([0.0, 0.0]))
    assert dx[2] == pytest.approx(-1.0)  # -k*vx = -0.1*10


def test_invalid_params():
    with pytest.raises(ValueError):
        PointMass2D(a_max=-1.0)
    with pytest.raises(ValueError):
        PointMass2D(drag_coeff=-0.5)
