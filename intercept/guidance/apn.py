"""Augmented Proportional Navigation (APN).

APN adds a target-acceleration feedforward term to True PN, compensating a maneuvering
(accelerating) target that ordinary PN lags:

    a = N · Vc · λ̇  +  (N / 2) · a_T⊥

where ``a_T⊥`` is the target acceleration projected perpendicular to the LOS. Reference:
Zarchan, *Tactical and Strategic Missile Guidance*. (The closely related *Augmented Ideal PN*,
Cho & Kim, IEEE TAES 2016, has a rigorous nonlinear inverse-optimality basis; a future variant.)

Target acceleration is not part of the kinematic state, so this implementation **estimates it by
finite differencing** the observed target velocity between calls. That keeps APN self-contained and
explicit about information: with perfect state knowledge the estimate is essentially exact for
piecewise-constant maneuvers; once an estimator is inserted, the same code consumes the
estimated target velocity instead. On the first call (no previous sample) the feedforward is zero,
so APN reduces to True PN.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from intercept.guidance.base import EPS, GuidanceLaw

Array = NDArray[np.float64]


class AugmentedPN(GuidanceLaw):
    """Augmented Proportional Navigation with a finite-difference target-acceleration estimate.

    Parameters
    ----------
    target:
        Name of the target entity to home on.
    N:
        Navigation constant, typically 3–5.
    """

    def __init__(self, target: str, N: float = 4.0) -> None:
        super().__init__(target)
        if N <= 0:
            raise ValueError("navigation constant N must be positive")
        self.N = float(N)
        self._prev_t: float | None = None
        self._prev_vt: Array | None = None
        self._a_t: Array = np.zeros(2)

    def reset(self) -> None:
        self._prev_t = None
        self._prev_vt = None
        self._a_t = np.zeros(2)

    def _update_target_accel(self, t: float, v_target: Array) -> Array:
        """Estimate target acceleration via backward finite difference of its velocity."""
        if self._prev_t is not None and self._prev_vt is not None:
            dt = t - self._prev_t
            if dt > EPS:
                self._a_t = (v_target - self._prev_vt) / dt
        self._prev_t = t
        self._prev_vt = v_target.copy()
        return self._a_t

    def command(self, t: float, own_state: Array, target_state: Array) -> Array:
        r, v, rng = self._rel(own_state, target_state)
        a_t = self._update_target_accel(t, target_state[2:4])
        if rng < EPS:
            return np.zeros(2)

        vc = self._closing_speed(r, v, rng)
        lam_dot = self._los_rate(r, v, rng)
        perp = self._los_perp(r, rng)

        pn_term = self.N * vc * lam_dot * perp
        a_t_perp = float(a_t @ perp) * perp  # target accel component perpendicular to LOS
        apn_term = 0.5 * self.N * a_t_perp
        return pn_term + apn_term
