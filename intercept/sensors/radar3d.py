"""3-D radar sensor: range, azimuth, and elevation to the target.

``z = [range, azimuth, elevation]`` for the relative vector ``r = target_pos − sensor_pos``:

* ``range  = |r|``
* ``azimuth   = atan2(r_y, r_x)``                (bearing in the horizontal plane)
* ``elevation = atan2(r_z, sqrt(r_x² + r_y²))``  (angle above the horizontal plane)

Range + two angles makes the 3-D target position fully observable. Azimuth and elevation are
angle-valued, so :meth:`residual` wraps those two components to ``[-π, π]``. The analytic Jacobian
is provided (for the EKF); the UKF needs only :meth:`h`. ``pos_dim = 3`` so the dimension-generic
EKF/UKF slice three position components from the tracker state.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from intercept.sensors.base import Sensor, wrap_to_pi

Array = NDArray[np.float64]


class Radar3D(Sensor):
    """Range + azimuth + elevation sensor with Gaussian noise (3-D analogue of :class:`Radar`)."""

    pos_dim = 3

    def __init__(
        self, sigma_range: float = 20.0, sigma_az: float = 0.01, sigma_el: float = 0.01
    ) -> None:
        if sigma_range <= 0 or sigma_az <= 0 or sigma_el <= 0:
            raise ValueError("sigma_range, sigma_az, sigma_el must be positive")
        self.sigma_range = float(sigma_range)
        self.sigma_az = float(sigma_az)
        self.sigma_el = float(sigma_el)
        self.dim = 3
        self.R = np.diag([sigma_range**2, sigma_az**2, sigma_el**2])

    def h(self, sensor_pos: Array, target_pos: Array) -> Array:
        r = np.asarray(target_pos, dtype=float)[:3] - np.asarray(sensor_pos, dtype=float)[:3]
        rng = float(np.linalg.norm(r))
        rho_h = float(np.hypot(r[0], r[1]))
        az = np.arctan2(r[1], r[0])
        el = np.arctan2(r[2], rho_h)
        return np.array([rng, az, el])

    def jacobian(self, sensor_pos: Array, target_pos: Array) -> Array:
        r = np.asarray(target_pos, dtype=float)[:3] - np.asarray(sensor_pos, dtype=float)[:3]
        x, y, z = r
        rng = float(np.linalg.norm(r))
        rho_h2 = x * x + y * y
        rho_h = float(np.sqrt(rho_h2)) or 1e-9
        rng = rng or 1e-9
        rho_h2 = rho_h2 or 1e-9
        return np.array(
            [
                [x / rng, y / rng, z / rng],
                [-y / rho_h2, x / rho_h2, 0.0],
                [-z * x / (rng * rng * rho_h), -z * y / (rng * rng * rho_h), rho_h / (rng * rng)],
            ]
        )

    def residual(self, z_a: Array, z_b: Array) -> Array:
        d = np.asarray(z_a, dtype=float) - np.asarray(z_b, dtype=float)
        d[1] = wrap_to_pi(d[1])  # azimuth
        d[2] = wrap_to_pi(d[2])  # elevation
        return d

    def invert(self, sensor_pos: Array, z: Array) -> Array:
        """Recover a target position from a measurement (e.g. to seed a tracker)."""
        rng, az, el = float(z[0]), float(z[1]), float(z[2])
        sp = np.asarray(sensor_pos, dtype=float)[:3]
        direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        return sp + rng * direction
