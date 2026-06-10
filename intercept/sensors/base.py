"""Sensor interface: a measurement model mapping (sensor position, target position) -> measurement.

Sensors are defined in terms of the **target position** only (the kinematic observable); the
estimator places the position-Jacobian into the full state Jacobian, so sensors are decoupled from
the tracker's state dimension. Noise is drawn from an **explicit** ``numpy.random.Generator`` passed
in at measure time, so an engagement seeded per Monte-Carlo trial stays reproducible (ADR-0003).

Angle-valued measurement components require care: residuals must be wrapped to ``[-π, π]``.
Subclasses override :meth:`residual` when a component is an angle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def wrap_to_pi(angle: float | Array) -> float | Array:
    """Wrap an angle (or array of angles) to ``[-π, π]``."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


class Sensor(ABC):
    """Measurement model with additive Gaussian noise.

    Attributes
    ----------
    dim:
        Measurement dimension.
    R:
        Measurement-noise covariance (``dim × dim``).
    pos_dim:
        Dimension of the target position the sensor observes (2 for planar, 3 for 3-D). The
        estimator slices the first ``pos_dim`` state components as position, so sensors stay
        decoupled from the tracker's full state dimension.
    """

    dim: int
    R: Array
    pos_dim: int = 2

    @abstractmethod
    def h(self, sensor_pos: Array, target_pos: Array) -> Array:
        """Noise-free measurement of ``target_pos`` from ``sensor_pos``."""

    @abstractmethod
    def jacobian(self, sensor_pos: Array, target_pos: Array) -> Array:
        """Jacobian ``∂h/∂target_pos`` (shape ``dim × 2``)."""

    def residual(self, z_a: Array, z_b: Array) -> Array:
        """Measurement residual ``z_a - z_b`` (override to wrap angle components)."""
        return np.asarray(z_a, dtype=float) - np.asarray(z_b, dtype=float)

    def measure(self, sensor_pos: Array, target_pos: Array, rng: np.random.Generator) -> Array:
        """Return a noisy measurement: ``h(...) + N(0, R)``."""
        z = self.h(sensor_pos, target_pos)
        noise = rng.multivariate_normal(np.zeros(self.dim), self.R)
        return z + noise
