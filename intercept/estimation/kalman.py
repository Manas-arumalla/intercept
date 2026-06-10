"""Extended and Unscented Kalman Filters for target tracking.

The motion model is linear (constant-velocity / -acceleration), so the *prediction* step is an
exact Kalman predict for both filters; they differ only in how they handle the **nonlinear**
range/bearing measurement:

* :class:`EKF` linearizes the measurement via its analytic Jacobian (Joseph-form covariance update
  for numerical stability).
* :class:`UKF` propagates sigma points through the measurement function (unscented transform),
  avoiding Jacobians and handling stronger nonlinearity better — at higher cost.

Both consume any :class:`~intercept.sensors.base.Sensor` and respect its angle-aware ``residual``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from intercept.estimation.base import Estimator
from intercept.sensors.base import Sensor

Array = NDArray[np.float64]

#: A motion model builder: ``dt -> (F, Q)``.
ModelFn = Callable[[float], tuple[Array, Array]]


def _gaussian_loglik(y: Array, S: Array) -> float:
    """Log-likelihood of innovation ``y`` under ``N(0, S)`` (used for IMM mode weighting)."""
    sign, logdet = np.linalg.slogdet(2.0 * np.pi * S)
    return float(-0.5 * (y @ np.linalg.solve(S, y) + logdet))


class EKF(Estimator):
    """Extended Kalman Filter (linear predict + EKF measurement update)."""

    def __init__(self, model_fn: ModelFn, x0: Array, P0: Array) -> None:
        self.model_fn = model_fn
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.last_loglik: float = 0.0

    def predict(self, dt: float) -> None:
        F, Q = self.model_fn(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: Array, sensor: Sensor, sensor_pos: Array) -> None:
        z = np.asarray(z, dtype=float)
        n = self.x.shape[0]
        d = sensor.pos_dim
        zpred = sensor.h(sensor_pos, self.x[:d])
        H = np.zeros((sensor.dim, n))
        H[:, :d] = sensor.jacobian(sensor_pos, self.x[:d])
        y = sensor.residual(z, zpred)
        S = H @ self.P @ H.T + sensor.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        # Joseph-form covariance update (symmetric, positive-definite preserving).
        ImKH = np.eye(n) - K @ H
        self.P = ImKH @ self.P @ ImKH.T + K @ sensor.R @ K.T
        self.last_loglik = _gaussian_loglik(y, S)


class UKF(Estimator):
    """Unscented Kalman Filter (linear predict + unscented measurement update)."""

    def __init__(
        self,
        model_fn: ModelFn,
        x0: Array,
        P0: Array,
        *,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self.model_fn = model_fn
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        n = self.x.shape[0]
        self.lam = alpha**2 * (n + kappa) - n
        c = n + self.lam
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * c))
        self.Wc = np.full(2 * n + 1, 1.0 / (2 * c))
        self.Wm[0] = self.lam / c
        self.Wc[0] = self.lam / c + (1.0 - alpha**2 + beta)
        self._c = c
        self.last_loglik: float = 0.0

    def _sigma_points(self) -> Array:
        n = self.x.shape[0]
        # Symmetric sqrt via Cholesky of the scaled covariance (with a small jitter for safety).
        try:
            A = np.linalg.cholesky(self._c * self.P)
        except np.linalg.LinAlgError:
            A = np.linalg.cholesky(self._c * self.P + 1e-9 * np.eye(n))
        pts = np.zeros((2 * n + 1, n))
        pts[0] = self.x
        for i in range(n):
            pts[1 + i] = self.x + A[:, i]
            pts[1 + n + i] = self.x - A[:, i]
        return pts

    def predict(self, dt: float) -> None:
        # Motion is linear => exact Kalman predict (equivalent to propagating sigma points).
        F, Q = self.model_fn(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: Array, sensor: Sensor, sensor_pos: Array) -> None:
        z = np.asarray(z, dtype=float)
        n = self.x.shape[0]
        d = sensor.pos_dim
        pts = self._sigma_points()
        # Reference measurement for angle-safe residual arithmetic.
        zref = sensor.h(sensor_pos, self.x[:d])
        dZ = np.array([sensor.residual(sensor.h(sensor_pos, p[:d]), zref) for p in pts])
        z_mean_offset = self.Wm @ dZ
        zpred = zref + z_mean_offset

        S = np.array(sensor.R, dtype=float).copy()
        Pxz = np.zeros((n, sensor.dim))
        for i, p in enumerate(pts):
            dz = sensor.residual(sensor.h(sensor_pos, p[:d]), zpred)
            dx = p - self.x
            S += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)

        K = Pxz @ np.linalg.inv(S)
        y = sensor.residual(z, zpred)
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T
        self.last_loglik = _gaussian_loglik(y, S)
