"""Infrared / electro-optical sensor: bearing (angle) only — the classic angles-only problem.

``z = [bearing]`` with ``bearing = atan2(r_y, r_x)``. A single bearing does not fix range, so the
target position is *not* observable from one look; observability requires interceptor maneuver
(parallax) and a good initial range guess. Used to contrast with radar and to demonstrate
angles-only UKF tracking.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from intercept.sensors.base import Sensor, wrap_to_pi

Array = NDArray[np.float64]


class IRSeeker(Sensor):
    """Bearing-only (angles-only) sensor with Gaussian noise.

    Parameters
    ----------
    sigma_bearing:
        Bearing noise standard deviation (rad).
    """

    dim = 1

    def __init__(self, sigma_bearing: float = 0.005) -> None:
        if sigma_bearing <= 0:
            raise ValueError("noise standard deviation must be positive")
        self.sigma_bearing = float(sigma_bearing)
        self.R = np.array([[sigma_bearing**2]])

    def h(self, sensor_pos: Array, target_pos: Array) -> Array:
        r = np.asarray(target_pos, dtype=float)[:2] - np.asarray(sensor_pos, dtype=float)[:2]
        return np.array([np.arctan2(r[1], r[0])])

    def jacobian(self, sensor_pos: Array, target_pos: Array) -> Array:
        r = np.asarray(target_pos, dtype=float)[:2] - np.asarray(sensor_pos, dtype=float)[:2]
        rng2 = max(float(r[0] ** 2 + r[1] ** 2), 1e-12)
        return np.array([[-r[1] / rng2, r[0] / rng2]])

    def residual(self, z_a: Array, z_b: Array) -> Array:
        return np.array([wrap_to_pi(float(z_a[0]) - float(z_b[0]))])
