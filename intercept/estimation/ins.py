"""Inertial-navigation (INS) platform error model.

A seeker mounted on the interceptor measures the *true* relative geometry (range/bearing), but to
place the target in the world frame the filter must add the platform's **own** position — which the
interceptor knows only through its INS, with error. `INSError` models that self-localization error
as a per-trial constant bias plus a linear drift (a first-order stand-in for accelerometer/gyro bias
integrated over the short engagement). It is sampled once at construction from an injected RNG, so
an engagement stays reproducible (no sampling in the loop — the determinism half of ADR-0003).

Used by :class:`~intercept.guidance.estimating.EstimatingGuidance` via its ``platform_error`` hook:
the measurement uses the true platform position (physics), the filter update uses the INS-corrupted
one, so the target estimate inherits the platform's navigation error.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class INSError:
    """Platform self-position error: ``error(t) = bias + drift · t`` (metres).

    Parameters
    ----------
    ndim:
        Spatial dimension (2 or 3).
    rng:
        Generator used **once** to sample the per-trial bias and drift.
    bias_std:
        1-σ of the initial constant position bias (m).
    drift_rate:
        1-σ of the per-axis drift velocity (m/s) — the integrated-bias growth over the engagement.
    """

    def __init__(
        self,
        ndim: int,
        rng: np.random.Generator,
        *,
        bias_std: float = 20.0,
        drift_rate: float = 2.0,
    ) -> None:
        self.ndim = int(ndim)
        self.bias = rng.normal(0.0, bias_std, self.ndim)
        self.drift = rng.normal(0.0, drift_rate, self.ndim)

    def __call__(self, t: float) -> Array:
        """Position error vector at time ``t`` (s)."""
        return self.bias + self.drift * float(t)
