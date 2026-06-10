"""Sliding-Mode Guidance (SMG).

Treats the line-of-sight rate ``λ̇`` as the sliding variable: ``s = λ̇``. Driving ``s → 0`` enforces
the parallel-navigation (constant-bearing) condition, i.e. a collision course, robustly against an
*unknown* maneuvering target. The command is

    a = (N · Vc · λ̇ + η · sat(λ̇ / Φ)) ⟂ LOS,

where the first term is the equivalent control (PN-like) and the second is the robust switching
term. A boundary layer ``Φ`` with a ``tanh`` (saturation) approximation of ``sign(·)`` suppresses
the chattering ideal sliding mode would induce. Choosing ``η`` above the target's lateral-
acceleration bound guarantees the sliding surface is reached despite the maneuver. Reference:
the sliding-mode-guidance literature; Shtessel et al., *Sliding Mode Control and Observation*.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from intercept.core import frames3d as f3
from intercept.guidance.base import EPS, GuidanceLaw

Array = NDArray[np.float64]


class SlidingModeGuidance(GuidanceLaw):
    """Sliding-mode guidance on the LOS-rate sliding surface, with a boundary layer.

    Parameters
    ----------
    target:
        Name of the target entity to home on.
    N:
        Navigation ratio for the equivalent-control term.
    eta:
        Switching gain (m/s²); set above the target's expected lateral-acceleration bound.
    boundary:
        Boundary-layer half-width ``Φ`` (rad/s) for the ``tanh`` approximation of ``sign``
        (chattering suppression). Smaller = closer to ideal sliding mode but more chattering.
    """

    def __init__(
        self, target: str, N: float = 4.0, eta: float = 60.0, boundary: float = 0.01
    ) -> None:
        super().__init__(target)
        if N <= 0 or eta < 0 or boundary <= 0:
            raise ValueError("require N > 0, eta >= 0, boundary > 0")
        self.N = float(N)
        self.eta = float(eta)
        self.boundary = float(boundary)

    def command(self, t: float, own_state: Array, target_state: Array) -> Array:
        r, v, rng = self._rel(own_state, target_state)
        if rng < EPS:
            return np.zeros(2)
        vc = self._closing_speed(r, v, rng)
        lam_dot = self._los_rate(r, v, rng)
        perp = self._los_perp(r, rng)

        equivalent = self.N * vc * lam_dot
        switching = self.eta * np.tanh(lam_dot / self.boundary)
        return (equivalent + switching) * perp


def sliding_mode(target: str, eta: float = 60.0) -> SlidingModeGuidance:
    """Convenience factory for sliding-mode guidance."""
    return SlidingModeGuidance(target, eta=eta)


class SlidingModeGuidance3D(GuidanceLaw):
    """3-D sliding-mode guidance on the LOS-rate *vector* ``Ω = (r×v)/|r|²`` (sliding surface).

    Generalizes the planar law: the equivalent control is the realizable-true-PN term
    ``N · (Ω × v_c)`` and the robust switching term is ``η · tanh(|Ω|/Φ)`` along the same
    direction (the unit of ``Ω × v_c``, which drives ``Ω → 0``). Choosing ``η`` above the target's
    lateral-acceleration bound guarantees the surface is reached despite an unknown 3-D maneuver;
    the boundary layer ``Φ`` suppresses chattering. ``v_c`` is the closing (pursuer−target) vel.
    """

    def __init__(
        self, target: str, N: float = 4.0, eta: float = 60.0, boundary: float = 0.01
    ) -> None:
        super().__init__(target)
        if N <= 0 or eta < 0 or boundary <= 0:
            raise ValueError("require N > 0, eta >= 0, boundary > 0")
        self.N = float(N)
        self.eta = float(eta)
        self.boundary = float(boundary)

    def command(self, t: float, own_state: Array, target_state: Array) -> Array:
        r, _ = f3.relative_state(own_state, target_state)
        if float(np.linalg.norm(r)) < EPS:
            return np.zeros(3)
        omega = f3.los_rate_vector(own_state, target_state)
        v_c = np.asarray(own_state, dtype=float)[3:6] - np.asarray(target_state, dtype=float)[3:6]
        cross_dir = np.cross(omega, v_c)  # PN turning direction (⟂ v_c, rotation plane)
        equivalent = self.N * cross_dir
        norm = float(np.linalg.norm(cross_dir))
        if norm < EPS:
            return equivalent
        switching = (
            self.eta * np.tanh(float(np.linalg.norm(omega)) / self.boundary) * (cross_dir / norm)
        )
        return equivalent + switching


def sliding_mode_3d(target: str, eta: float = 60.0) -> SlidingModeGuidance3D:
    """Convenience factory for 3-D sliding-mode guidance."""
    return SlidingModeGuidance3D(target, eta=eta)
