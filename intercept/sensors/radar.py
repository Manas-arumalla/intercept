"""Radar sensor: measures range and bearing to the target (range-azimuth).

``z = [range, bearing]`` where ``range = |r|`` and ``bearing = atan2(r_y, r_x)`` for the relative
vector ``r = target_pos − sensor_pos``. Range + bearing makes the target position fully observable
from a single look (unlike angles-only), so it is the default for the estimation-coupled study.
The measurement is nonlinear in position, so it needs an EKF/UKF (not a linear KF).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from intercept.sensors.base import Sensor, wrap_to_pi

Array = NDArray[np.float64]


class Radar(Sensor):
    """Range + bearing sensor with Gaussian noise.

    Parameters
    ----------
    sigma_range:
        Range noise standard deviation (m).
    sigma_bearing:
        Bearing noise standard deviation (rad).
    """

    dim = 2

    def __init__(self, sigma_range: float = 20.0, sigma_bearing: float = 0.01) -> None:
        if sigma_range <= 0 or sigma_bearing <= 0:
            raise ValueError("noise standard deviations must be positive")
        self.sigma_range = float(sigma_range)
        self.sigma_bearing = float(sigma_bearing)
        self.R = np.diag([sigma_range**2, sigma_bearing**2])

    def h(self, sensor_pos: Array, target_pos: Array) -> Array:
        r = np.asarray(target_pos, dtype=float)[:2] - np.asarray(sensor_pos, dtype=float)[:2]
        return np.array([np.hypot(r[0], r[1]), np.arctan2(r[1], r[0])])

    def jacobian(self, sensor_pos: Array, target_pos: Array) -> Array:
        r = np.asarray(target_pos, dtype=float)[:2] - np.asarray(sensor_pos, dtype=float)[:2]
        rng = float(np.hypot(r[0], r[1]))
        rng = max(rng, 1e-6)
        rng2 = rng * rng
        return np.array(
            [
                [r[0] / rng, r[1] / rng],
                [-r[1] / rng2, r[0] / rng2],
            ]
        )

    def residual(self, z_a: Array, z_b: Array) -> Array:
        d = np.asarray(z_a, dtype=float) - np.asarray(z_b, dtype=float)
        d[1] = wrap_to_pi(d[1])
        return d

    def invert(self, sensor_pos: Array, z: Array) -> Array:
        """Recover an approximate target position from a measurement (for tracker init)."""
        rng, bearing = float(z[0]), float(z[1])
        sp = np.asarray(sensor_pos, dtype=float)[:2]
        return sp + rng * np.array([np.cos(bearing), np.sin(bearing)])
