"""3-D evasive maneuvers — genuinely three-dimensional, visually striking targets.

* :func:`barrel_roll` — lateral acceleration of fixed magnitude rotating about the velocity axis,
  producing a helical / corkscrew flight path (a classic hard-to-intercept 3-D maneuver).
* :func:`weave3d` — sinusoidal lateral acceleration along a fixed lateral axis (a 3-D weave).
* :func:`serpentine3d` — a bang-bang S-weave whose maneuver plane is *tilted* out of the
  horizontal, so the ground track snakes while the altitude porpoises (a 3-D serpentine).
* :func:`terminal_spiral` — **closed-loop**: a corkscrew that *intensifies* as the pursuer closes,
  ramping from a gentle midcourse weave to a tight max-g spiral in the endgame. This models the
  spiraling / weaving terminal maneuver real maneuvering-reentry vehicles and sea-skimming
  anti-ship missiles use to defeat endgame interception — yet a predictive law (APN/OGL) that
  estimates the target's acceleration can still run it down. The plant clips every command to its
  physics-derived turn limit, so the spiral genuinely costs the target energy (induced drag).
* :func:`combine` — sum several maneuver controllers into one commanded-acceleration vector
  (e.g. a lofted cruise bias + a serpentine + a terminal spiral); the plant clips the total.

All command acceleration perpendicular to the target's velocity (the plant g-limits it).
References: Zarchan, *Tactical and Strategic Missile Guidance* (maneuvering-target evasion);
Shneydor, *Missile Guidance and Pursuit* (weave/spiral geometries).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from intercept.core.entities import Controller

Array = NDArray[np.float64]
_EPS = 1e-9


def _lateral_basis(v: Array) -> tuple[Array, Array]:
    """Two orthonormal vectors spanning the plane perpendicular to velocity ``v``."""
    speed = float(np.linalg.norm(v))
    v_hat = v / speed if speed > _EPS else np.array([1.0, 0.0, 0.0])
    ref = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(v_hat, ref)
    if float(np.linalg.norm(e1)) < 1e-6:  # velocity nearly vertical
        e1 = np.cross(v_hat, np.array([0.0, 1.0, 0.0]))
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(v_hat, e1)
    return e1, e2


def barrel_roll(accel: float, rate: float) -> Controller:
    """Helical corkscrew: lateral accel of magnitude ``accel`` rotating at ``rate`` rad/s."""

    def controller(t: float, own: Array, world: Mapping[str, Array]) -> Array:
        e1, e2 = _lateral_basis(own[3:6])
        return accel * (np.cos(rate * t) * e1 + np.sin(rate * t) * e2)

    return controller


def weave3d(accel: float, frequency: float, axis: str = "e1") -> Controller:
    """Sinusoidal 3-D weave along one lateral basis axis (``e1`` or ``e2``)."""
    w = 2.0 * np.pi * frequency

    def controller(t: float, own: Array, world: Mapping[str, Array]) -> Array:
        e1, e2 = _lateral_basis(own[3:6])
        direction = e1 if axis == "e1" else e2
        return accel * np.sin(w * t) * direction

    return controller


def serpentine3d(accel: float, frequency: float, tilt: float = 0.5) -> Controller:
    """Tilted S-weave: sinusoidal lateral accel in a plane rotated ``tilt`` rad toward ``e2``.

    With ``tilt=0`` this is a pure horizontal weave; with ``tilt`` near ``π/2`` it porpoises in
    altitude. Intermediate tilts snake in 3-D (the ground track weaves while the height oscillates).
    """
    w = 2.0 * np.pi * frequency

    def controller(t: float, own: Array, world: Mapping[str, Array]) -> Array:
        e1, e2 = _lateral_basis(own[3:6])
        direction = np.cos(tilt) * e1 + np.sin(tilt) * e2
        return accel * np.sin(w * t) * direction

    return controller


def terminal_spiral(
    pursuer_name: str,
    base_accel: float,
    max_accel: float,
    trigger_range: float,
    rate: float,
    base: Controller | None = None,
) -> Controller:
    """Corkscrew that intensifies as the pursuer closes (closed-loop terminal evasion).

    Beyond ``trigger_range`` the target flies ``base`` (default: the gentle ``base_accel`` spiral).
    Inside it, the spiral magnitude ramps linearly from ``base_accel`` to ``max_accel`` as range
    → 0 — the tightening terminal spiral that exploits the interceptor's autopilot lag and finite
    turn rate. The acceleration always rotates at ``rate`` rad/s in the plane perpendicular to the
    target's velocity, so the flight path is a true 3-D helix.

    Parameters
    ----------
    pursuer_name:
        Name of the interceptor entity to sense in the world snapshot.
    base_accel, max_accel:
        Spiral magnitude (m/s²) far from / at the merge; the plant clips to the turn limit.
    trigger_range:
        Range (m) at which the spiral begins to tighten.
    rate:
        Angular rate of the corkscrew (rad/s).
    base:
        Controller flown before the trigger (defaults to the gentle spiral itself).
    """

    def controller(t: float, own: Array, world: Mapping[str, Array]) -> Array:
        e1, e2 = _lateral_basis(own[3:6])
        spiral = np.cos(rate * t) * e1 + np.sin(rate * t) * e2
        if pursuer_name not in world:
            return base(t, own, world) if base is not None else base_accel * spiral
        rng = float(np.linalg.norm(np.asarray(world[pursuer_name], dtype=float)[:3] - own[:3]))
        if rng > trigger_range:
            return base(t, own, world) if base is not None else base_accel * spiral
        c = max(0.0, min(1.0, (trigger_range - rng) / trigger_range))
        return (base_accel + (max_accel - base_accel) * c) * spiral

    return controller


def combine(*controllers: Controller) -> Controller:
    """Sum several maneuver controllers into one commanded-acceleration vector (the plant clips)."""

    def controller(t: float, own: Array, world: Mapping[str, Array]) -> Array:
        total = np.zeros(3)
        for c in controllers:
            total = total + np.asarray(c(t, own, world), dtype=float)
        return total

    return controller
