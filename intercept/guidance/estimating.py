"""Estimating guidance: close the sense → estimate → guide loop.

:class:`EstimatingGuidance` is a :data:`~intercept.core.entities.Controller` that wraps a sensor,
an estimator, and an underlying guidance law. Each step it takes a noisy measurement of the *true*
target (from the world snapshot), runs the estimator, and feeds the **estimated** target state to
the guidance law — which is otherwise unchanged and unaware it is no longer seeing truth. This is
what enables the estimation-coupled study: guidance performance as a function of sensor noise and
filter choice.

The sensor's RNG is supplied at construction and fixed per Monte-Carlo trial, so an engagement
remains reproducible despite per-step measurement noise (the determinism half of ADR-0003).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from intercept.core.entities import Controller
from intercept.estimation.base import Estimator
from intercept.sensors.base import Sensor

Array = NDArray[np.float64]

#: Builds an estimator from an initial state/covariance: ``factory(x0, P0) -> Estimator``.
EstimatorFactory = Callable[[Array, Array], Estimator]


class EstimatingGuidance:
    """Wrap a guidance law with a sensor + estimator (perfect interceptor self-state).

    Parameters
    ----------
    target:
        Name of the target entity in the world snapshot.
    sensor:
        Measurement model (e.g. :class:`~intercept.sensors.radar.Radar`).
    estimator_factory:
        Builds the estimator once the first measurement initializes ``(x0, P0)``.
    guidance:
        The underlying guidance law (a callable on ``(t, own_state, world)``), bound to ``target``.
    rng:
        Generator for sensor noise (fixed per trial for reproducibility).
    init_pos_std, init_vel_std, init_acc_std:
        Initial 1-σ uncertainties for the estimator's position/velocity/acceleration states.
    platform_error:
        Optional ``error(t) -> position-error vector`` modeling the interceptor's INS
        self-localization error (e.g. :class:`~intercept.estimation.ins.INSError`). The seeker still
        measures the *true* relative geometry, but the filter places the target using the *believed*
        (error-corrupted) platform position — so the estimate inherits the platform's nav error.
        ``None`` (default) = perfect interceptor self-state.
    """

    def __init__(
        self,
        target: str,
        sensor: Sensor,
        estimator_factory: EstimatorFactory,
        guidance: Controller,
        rng: np.random.Generator,
        *,
        init_pos_std: float = 50.0,
        init_vel_std: float = 300.0,
        init_acc_std: float = 100.0,
        platform_error: Callable[[float], Array] | None = None,
    ) -> None:
        self.target = target
        self.sensor = sensor
        self.estimator_factory = estimator_factory
        self.guidance = guidance
        self.rng = rng
        self.init_pos_std = init_pos_std
        self.init_vel_std = init_vel_std
        self.init_acc_std = init_acc_std
        self.platform_error = platform_error
        self.estimator: Estimator | None = None
        self._last_t: float | None = None

    @property
    def _ndim(self) -> int:
        """Spatial dimension the sensor observes (2 for planar, 3 for 3-D)."""
        return int(getattr(self.sensor, "pos_dim", 2))

    def _initialize(self, sensor_pos: Array, z: Array) -> None:
        n = self._ndim
        invert = getattr(self.sensor, "invert", None)
        if invert is not None:
            pos0 = np.asarray(invert(sensor_pos, z), dtype=float)[:n]
        else:  # angles-only: fall back to a nominal guess at the sensor position
            pos0 = np.asarray(sensor_pos, dtype=float)[:n].copy()
        x0 = np.concatenate([pos0, np.zeros(2 * n)])
        diag = [self.init_pos_std**2] * n + [self.init_vel_std**2] * n + [self.init_acc_std**2] * n
        self.estimator = self.estimator_factory(x0, np.diag(diag))

    def __call__(self, t: float, own_state: Array, world: Mapping[str, Array]) -> Array:
        n = self._ndim
        true_sensor_pos = np.asarray(own_state, dtype=float)[:n]
        true_target_pos = np.asarray(world[self.target], dtype=float)[:n]
        # The seeker measures the TRUE relative geometry (physics)...
        z = self.sensor.measure(true_sensor_pos, true_target_pos, self.rng)
        # ...but the filter places the target using the BELIEVED platform position (INS error).
        believed_sensor_pos = true_sensor_pos
        if self.platform_error is not None:
            believed_sensor_pos = true_sensor_pos + np.asarray(self.platform_error(t), dtype=float)

        if self.estimator is None:
            self._initialize(believed_sensor_pos, z)
        else:
            dt = t - (self._last_t if self._last_t is not None else t)
            if dt > 0:
                self.estimator.predict(dt)
            self.estimator.update(z, self.sensor, believed_sensor_pos)
        self._last_t = t

        assert self.estimator is not None
        est_world = dict(world)
        est_world[self.target] = self.estimator.target_state()
        return np.asarray(self.guidance(t, own_state, est_world), dtype=float)

    @property
    def estimator_state(self) -> Array:
        """Current estimator state (zeros of the right size if not yet initialized)."""
        if self.estimator is not None:
            return self.estimator.x.copy()
        return np.zeros(3 * self._ndim)
