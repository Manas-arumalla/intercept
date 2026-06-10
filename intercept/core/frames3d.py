"""3-D engagement geometry: range, closing speed, and the LOS angular-velocity *vector*.

In 3-D the line-of-sight rotation is a vector ``Ω = (r × v) / (r·r)`` (the 2-D scalar λ̇ is its
z-component). 3-D Proportional Navigation commands acceleration ``∝ Ω × v_rel`` (perpendicular to
the relative velocity), generalizing the planar law. Operates on ``[x,y,z,vx,vy,vz,…]`` states.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _pos(state: Array) -> Array:
    return np.asarray(state, dtype=float)[:3]


def _vel(state: Array) -> Array:
    return np.asarray(state, dtype=float)[3:6]


def relative_state(pursuer: Array, target: Array) -> tuple[Array, Array]:
    """Relative position ``r = target − pursuer`` and relative velocity ``v`` (both ℝ³)."""
    return _pos(target) - _pos(pursuer), _vel(target) - _vel(pursuer)


def range_to(pursuer: Array, target: Array) -> float:
    r, _ = relative_state(pursuer, target)
    return float(np.linalg.norm(r))


def closing_speed(pursuer: Array, target: Array) -> float:
    """Closing speed ``Vc = −d(range)/dt`` (m/s); positive when range decreasing."""
    r, v = relative_state(pursuer, target)
    rng = float(np.linalg.norm(r))
    return 0.0 if rng < 1e-9 else float(-(r @ v) / rng)


def los_rate_vector(pursuer: Array, target: Array) -> Array:
    """Line-of-sight angular-velocity vector ``Ω = (r × v) / |r|²`` (rad/s)."""
    r, v = relative_state(pursuer, target)
    rr = float(r @ r)
    return np.zeros(3) if rr < 1e-12 else np.cross(r, v) / rr


def los_unit(pursuer: Array, target: Array) -> Array:
    """Unit line-of-sight vector from pursuer to target."""
    r, _ = relative_state(pursuer, target)
    n = float(np.linalg.norm(r))
    return r / n if n > 1e-9 else np.zeros(3)


def zero_effort_miss(pursuer: Array, target: Array, t_go: float) -> Array:
    """Zero-effort miss vector ``ZEM = r + v·t_go`` (ℝ³)."""
    r, v = relative_state(pursuer, target)
    return r + v * float(t_go)
