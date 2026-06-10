"""Estimator interface for target tracking.

An estimator maintains a Gaussian belief ``(x, P)`` over the 6-D target state
``[x, y, vx, vy, ax, ay]``. ``predict(dt)`` propagates it with a motion model; ``update(z, sensor,
sensor_pos)`` corrects it with a (possibly nonlinear) measurement. The tracked target state is
exposed as a 4-D ``[x, y, vx, vy]`` so it slots directly into the guidance laws, which are unaware
whether they are fed truth or an estimate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from intercept.sensors.base import Sensor

Array = NDArray[np.float64]


class Estimator(ABC):
    """Recursive Bayesian target-state estimator (Gaussian belief)."""

    x: Array  # state mean (6,)
    P: Array  # state covariance (6, 6)

    @abstractmethod
    def predict(self, dt: float) -> None:
        """Propagate the belief forward by ``dt`` using the motion model."""

    @abstractmethod
    def update(self, z: Array, sensor: Sensor, sensor_pos: Array) -> None:
        """Correct the belief with measurement ``z`` taken from ``sensor_pos``."""

    @property
    def _ndim(self) -> int:
        """Spatial dimension inferred from the ``[pos, vel, acc]`` state (2 or 3)."""
        return int(self.x.shape[0]) // 3

    @property
    def position(self) -> Array:
        """Estimated target position (``[x, y]`` in 2-D, ``[x, y, z]`` in 3-D)."""
        return self.x[: self._ndim].copy()

    @property
    def velocity(self) -> Array:
        """Estimated target velocity (``[vx, vy]`` in 2-D, ``[vx, vy, vz]`` in 3-D)."""
        n = self._ndim
        return self.x[n : 2 * n].copy()

    def target_state(self) -> Array:
        """Estimated ``[pos, vel]`` target state for guidance laws (4-D in 2-D, 6-D in 3-D)."""
        n = self._ndim
        return np.concatenate([self.x[:n], self.x[n : 2 * n]])

    def nees(self, true_state6: Array) -> float:
        """Normalized estimation error squared vs. a 6-D ground-truth state (filter consistency)."""
        e = np.asarray(true_state6, dtype=float) - self.x
        return float(e @ np.linalg.solve(self.P, e))
