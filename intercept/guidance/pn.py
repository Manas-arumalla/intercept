"""Proportional Navigation (PN) guidance — the canonical homing baseline.

PN commands acceleration proportional to the line-of-sight (LOS) rotation rate, steering the
interceptor to null the LOS rate and hold a constant bearing (constant-bearing-decreasing-range),
which produces a collision course. References: Zarchan, *Tactical and Strategic Missile Guidance*;
Siouris, *Missile Guidance and Control Systems*; Shneydor, *Missile Guidance and Pursuit*.

Variants implemented (``variant=``):

* ``"true"`` — **True PN**: ``a = N · Vc · λ̇`` applied perpendicular to the **LOS**.
  ``Vc`` is closing speed, ``λ̇`` the LOS rate, ``N`` the navigation constant (typically 3–5).
* ``"pure"`` — **Pure PN**: same magnitude, applied perpendicular to the interceptor's **velocity**.
* ``"zem"`` — **Zero-Effort-Miss PN**: ``a = N · ZEM⊥ / t_go²`` where ``ZEM = r + v·t_go`` is the
  predicted miss with no further acceleration and ``t_go = |r| / Vc``. Equivalent to True PN for a
  non-maneuvering target, and the natural form to augment for maneuvering targets (see ``apn``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from intercept.guidance.base import EPS, GuidanceLaw

Array = NDArray[np.float64]

_VARIANTS = ("true", "pure", "zem")


class ProportionalNavigation(GuidanceLaw):
    """Proportional Navigation guidance law.

    Parameters
    ----------
    target:
        Name of the target entity to home on.
    N:
        Navigation constant (effective navigation ratio), typically 3–5.
    variant:
        ``"true"`` (perpendicular to LOS), ``"pure"`` (perpendicular to own velocity), or
        ``"zem"`` (zero-effort-miss form).
    """

    def __init__(self, target: str, N: float = 4.0, variant: str = "true") -> None:
        super().__init__(target)
        if N <= 0:
            raise ValueError("navigation constant N must be positive")
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}")
        self.N = float(N)
        self.variant = variant

    def command(self, t: float, own_state: Array, target_state: Array) -> Array:
        r, v, rng = self._rel(own_state, target_state)
        if rng < EPS:
            return np.zeros(2)
        vc = self._closing_speed(r, v, rng)

        if self.variant == "zem":
            # Time-to-go estimate from current closing geometry.
            if vc <= EPS:
                return np.zeros(2)
            t_go = rng / vc
            zem = r + v * t_go
            u_los = r / rng
            zem_perp = zem - (zem @ u_los) * u_los  # component perpendicular to LOS
            return self.N * zem_perp / (t_go * t_go)

        lam_dot = self._los_rate(r, v, rng)
        a_mag = self.N * vc * lam_dot
        if self.variant == "true":
            perp = self._los_perp(r, rng)
        else:  # "pure": perpendicular to the interceptor's own velocity
            v_own = own_state[2:4]
            s = float(np.linalg.norm(v_own))
            u = (v_own / s) if s > EPS else (r / rng)
            perp = np.array([-u[1], u[0]])
        return a_mag * perp


def true_pn(target: str, N: float = 4.0) -> ProportionalNavigation:
    """Convenience factory for True PN."""
    return ProportionalNavigation(target, N=N, variant="true")


def pure_pn(target: str, N: float = 4.0) -> ProportionalNavigation:
    """Convenience factory for Pure PN."""
    return ProportionalNavigation(target, N=N, variant="pure")


def zem_pn(target: str, N: float = 4.0) -> ProportionalNavigation:
    """Convenience factory for Zero-Effort-Miss PN."""
    return ProportionalNavigation(target, N=N, variant="zem")
