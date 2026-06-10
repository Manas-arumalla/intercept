"""3-D Proportional Navigation (realizable true PN) and Augmented PN.

Generalizes the planar laws using the LOS angular-velocity *vector* ``Ω = (r×v)/|r|²``:

    True PN (3-D):   a = N · (Ω × v_rel)          — perpendicular to the relative velocity
    Augmented PN:    a = N · (Ω × v_rel) + (N/2)·a_T⊥   — + target-acceleration feedforward ⟂ LOS

Driving ``Ω → 0`` holds a constant inertial LOS direction (a 3-D collision course). The commanded
acceleration vector drops straight into the 3-D plants (`PointMass3D` applies it directly;
`AeroMissile3D` takes its lateral component, g-limits, and lags it). References: Zarchan; Shneydor
(3-D / realizable true PN). Conforms to the `Controller` contract on 3-D states.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from intercept.core import frames3d as f3
from intercept.guidance.base import EPS, GuidanceLaw

Array = NDArray[np.float64]


class ProportionalNavigation3D(GuidanceLaw):
    """3-D realizable true Proportional Navigation: ``a = N · (Ω × v_rel)``."""

    def __init__(self, target: str, N: float = 4.0) -> None:
        super().__init__(target)
        if N <= 0:
            raise ValueError("navigation constant N must be positive")
        self.N = float(N)

    def command(self, t: float, own_state: Array, target_state: Array) -> Array:
        r, _ = f3.relative_state(own_state, target_state)
        if float(np.linalg.norm(r)) < EPS:
            return np.zeros(3)
        omega = f3.los_rate_vector(own_state, target_state)
        # Closing velocity (pursuer − target); a = N·(Ω × V_c) turns toward the target's motion.
        v_closing = own_state[3:6] - target_state[3:6]
        return self.N * np.cross(omega, v_closing)


class AugmentedPN3D(GuidanceLaw):
    """3-D Augmented PN: True-PN term plus a target-acceleration feedforward perpendicular to LOS.

    Target acceleration is estimated by finite-differencing the observed target velocity.
    """

    def __init__(self, target: str, N: float = 4.0) -> None:
        super().__init__(target)
        if N <= 0:
            raise ValueError("navigation constant N must be positive")
        self.N = float(N)
        self._prev_t: float | None = None
        self._prev_vt: Array | None = None
        self._a_t = np.zeros(3)

    def reset(self) -> None:
        self._prev_t = None
        self._prev_vt = None
        self._a_t = np.zeros(3)

    def command(self, t: float, own_state: Array, target_state: Array) -> Array:
        v_target = np.asarray(target_state, dtype=float)[3:6]
        if self._prev_t is not None and self._prev_vt is not None:
            dt = t - self._prev_t
            if dt > EPS:
                self._a_t = (v_target - self._prev_vt) / dt
        self._prev_t, self._prev_vt = t, v_target.copy()

        r, _ = f3.relative_state(own_state, target_state)
        rng = float(np.linalg.norm(r))
        if rng < EPS:
            return np.zeros(3)
        omega = f3.los_rate_vector(own_state, target_state)
        v_closing = own_state[3:6] - target_state[3:6]
        pn = self.N * np.cross(omega, v_closing)

        u_los = r / rng
        a_t_perp = self._a_t - (self._a_t @ u_los) * u_los  # target accel ⟂ LOS
        return pn + 0.5 * self.N * a_t_perp


def true_pn_3d(target: str, N: float = 4.0) -> ProportionalNavigation3D:
    """Convenience factory for 3-D True PN."""
    return ProportionalNavigation3D(target, N=N)


def augmented_pn_3d(target: str, N: float = 4.0) -> AugmentedPN3D:
    """Convenience factory for 3-D Augmented PN."""
    return AugmentedPN3D(target, N=N)
