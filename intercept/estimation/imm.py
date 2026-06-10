"""Interacting Multiple Model (IMM) estimator.

Runs a bank of filters with different motion models (here: nearly-constant-velocity and
nearly-constant-acceleration), maintains a probability over which model is active, and mixes their
estimates each step through a Markov mode-transition matrix. This is the standard tool for
maneuvering-target tracking: the quiescent model gives low noise when the target flies straight,
and the maneuver model takes over (its probability rises) when the target turns.

Algorithm (Blom & Bar-Shalom): interaction/mixing → model-conditioned predict+update → mode-
probability update from each filter's measurement likelihood → moment-matched combined estimate.
All sub-filters share the 6-D state, so mixing is a plain weighted combination.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from intercept.estimation.base import Estimator
from intercept.estimation.kalman import EKF
from intercept.estimation.models import nca_model, ncv_model
from intercept.sensors.base import Sensor

Array = NDArray[np.float64]


class IMM(Estimator):
    """Interacting Multiple Model estimator over a bank of sub-filters.

    Parameters
    ----------
    filters:
        Sub-filters (e.g. an NCV-:class:`EKF` and an NCA-:class:`EKF`), each with its own model.
    transition:
        Markov mode-transition matrix ``Π`` (``M × M``); ``Π[i, j]`` = P(model j | model i).
    mu0:
        Initial mode probabilities (defaults to uniform).
    """

    def __init__(
        self,
        filters: Sequence[EKF],
        transition: Array,
        mu0: Array | None = None,
    ) -> None:
        self.filters = list(filters)
        m = len(self.filters)
        self.Pi = np.array(transition, dtype=float)
        if self.Pi.shape != (m, m):
            raise ValueError("transition matrix must be (M, M)")
        self.mu = np.full(m, 1.0 / m) if mu0 is None else np.array(mu0, dtype=float)
        self.mu = self.mu / self.mu.sum()
        self._cbar = self.mu.copy()
        self._n = int(np.asarray(self.filters[0].x).shape[0])  # state dimension (2-D or 3-D)
        self.x = np.zeros(self._n)
        self.P = np.eye(self._n)
        self._combine()

    def _combine(self) -> None:
        """Moment-match the bank into a single Gaussian ``(x, P)``."""
        self.x = sum(mu_j * f.x for mu_j, f in zip(self.mu, self.filters, strict=True))
        P = np.zeros((self._n, self._n))
        for mu_j, f in zip(self.mu, self.filters, strict=True):
            dx = f.x - self.x
            P += mu_j * (f.P + np.outer(dx, dx))
        self.P = P

    def predict(self, dt: float) -> None:
        m = len(self.filters)
        # Interaction / mixing.
        self._cbar = self.Pi.T @ self.mu  # predicted mode probabilities
        mu_ij = (self.Pi * self.mu[:, None]) / np.maximum(self._cbar[None, :], 1e-12)
        mixed_x = [sum(mu_ij[i, j] * self.filters[i].x for i in range(m)) for j in range(m)]
        for j, f in enumerate(self.filters):
            P0 = np.zeros((self._n, self._n))
            for i in range(m):
                dx = self.filters[i].x - mixed_x[j]
                P0 += mu_ij[i, j] * (self.filters[i].P + np.outer(dx, dx))
            f.x = mixed_x[j].copy()
            f.P = P0
        # Model-conditioned prediction.
        for f in self.filters:
            f.predict(dt)
        self._combine()

    def update(self, z: Array, sensor: Sensor, sensor_pos: Array) -> None:
        logliks = np.empty(len(self.filters))
        for j, f in enumerate(self.filters):
            f.update(z, sensor, sensor_pos)
            logliks[j] = f.last_loglik
        # Mode-probability update (stabilized by subtracting the max log-likelihood).
        likelihood = np.exp(logliks - logliks.max())
        self.mu = self._cbar * likelihood
        self.mu = self.mu / max(self.mu.sum(), 1e-300)
        self._combine()

    @property
    def mode_probabilities(self) -> Array:
        """Current probability of each model being active."""
        return self.mu.copy()


def make_cv_ca_imm(
    x0: Array,
    P0: Array,
    *,
    q_cv: float = 1.0,
    q_ca: float = 200.0,
    p_stay: float = 0.95,
    ndim: int = 2,
) -> IMM:
    """Build a standard 2-model IMM (NCV + NCA) with a symmetric sticky transition matrix.

    Parameters
    ----------
    x0, P0:
        Shared initial state and covariance for both sub-filters (length ``3·ndim``).
    q_cv, q_ca:
        Process-noise spectral densities for the quiescent / maneuver models.
    p_stay:
        Probability of remaining in the same mode between steps (diagonal of ``Π``).
    ndim:
        Spatial dimension (2 for planar, 3 for 3-D); selects the model dimensionality.
    """
    cv = EKF(lambda dt: ncv_model(dt, q_cv, ndim=ndim), x0, P0)
    ca = EKF(lambda dt: nca_model(dt, q_ca, ndim=ndim), x0, P0)
    off = 1.0 - p_stay
    transition = np.array([[p_stay, off], [off, p_stay]])
    return IMM([cv, ca], transition)
