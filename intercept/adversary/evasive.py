"""Aggressive and *reactive* evasive maneuvers — the hard adversaries.

Where :mod:`intercept.adversary.scripted` provides gentle open-loop maneuvers, these are the
high-g and closed-loop evasions that make interception genuinely difficult against a realistic
:class:`~intercept.core.aero.AeroMissile2D` target (whose induced drag bleeds speed when it pulls
g — so evasion has a real energy cost). All command lateral acceleration perpendicular to the
target's velocity, magnitude in m/s² (the plant clips to the target's own g-limit).

* :func:`hard_turn` — sustained max-g break (a constant high-g turn / spiral).
* :func:`random_telegraph` — bang-bang jink with sign flips at random (exponential) intervals,
  using an explicit seeded RNG so Monte-Carlo trials stay reproducible. Unpredictable by design.
* :func:`reactive_break` — **closed-loop**: flies a baseline maneuver until the interceptor closes
  inside a trigger range, then executes a max-g break *away* from the interceptor (a last-ditch
  defensive break that exploits the interceptor's autopilot lag and g-limit — the classic way a
  maneuverable target defeats proportional navigation in the endgame).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from intercept.core.entities import Controller

Array = NDArray[np.float64]
_EPS = 1e-9


def _perp(state: Array) -> Array:
    """Unit vector perpendicular (+90°) to the entity's velocity; zero if at rest."""
    v = state[2:4]
    s = float(np.linalg.norm(v))
    if s < _EPS:
        return np.zeros(2)
    return np.array([-v[1], v[0]]) / s


def hard_turn(accel: float, sign: float = 1.0) -> Controller:
    """Sustained constant-magnitude lateral break (a max-g turn / spiral)."""

    def controller(t: float, own: Array, world: Mapping[str, Array]) -> Array:
        return float(np.sign(sign)) * accel * _perp(own)

    return controller


def random_telegraph(accel: float, mean_switch: float, rng: np.random.Generator) -> Controller:
    """Bang-bang lateral jink; sign flips at random exponential intervals (seeded, reproducible)."""
    state = {
        "sign": 1.0 if rng.random() < 0.5 else -1.0,
        "next": float(rng.exponential(mean_switch)),
    }

    def controller(t: float, own: Array, world: Mapping[str, Array]) -> Array:
        if t >= state["next"]:
            state["sign"] = -state["sign"]
            state["next"] = t + float(rng.exponential(mean_switch))
        return state["sign"] * accel * _perp(own)

    return controller


def reactive_break(
    pursuer_name: str,
    accel: float,
    trigger_range: float,
    base: Controller | None = None,
) -> Controller:
    """Fly ``base`` until the pursuer is within ``trigger_range``, then break max-g away from it.

    Parameters
    ----------
    pursuer_name:
        Name of the interceptor entity to sense in the world snapshot.
    accel:
        Break acceleration magnitude (m/s²); the plant clips it to the target's g-limit.
    trigger_range:
        Range (m) at which the defensive break is triggered.
    base:
        Controller used before the break (defaults to coasting). Could be a gentle weave/cruise.
    """

    def controller(t: float, own: Array, world: Mapping[str, Array]) -> Array:
        if pursuer_name not in world:
            return base(t, own, world) if base is not None else np.zeros(2)
        rel = np.asarray(world[pursuer_name], dtype=float)[:2] - own[:2]
        if float(np.linalg.norm(rel)) > trigger_range:
            return base(t, own, world) if base is not None else np.zeros(2)
        perp = _perp(own)
        if np.linalg.norm(perp) < _EPS:
            return np.zeros(2)
        # Break toward the side away from the pursuer (defeat its lead with a late LOS-rate spike).
        away = own[:2] - np.asarray(world[pursuer_name], dtype=float)[:2]
        sign = 1.0 if float(perp @ away) >= 0.0 else -1.0
        return sign * accel * perp

    return controller


def surprise_break(
    pursuer_name: str,
    accel: float,
    trigger_range: float,
    sign: float,
    base: Controller | None = None,
) -> Controller:
    """Like :func:`reactive_break`, but the break *direction* is supplied by the caller.

    A pursuer cannot predict ``sign`` (sample it per trial from an injected RNG) — the adversary
    for branch-coverage studies: a lone interceptor must commit to one lead, a coordinated pair
    can cover both turn directions.
    """

    def controller(t: float, own: Array, world: Mapping[str, Array]) -> Array:
        if pursuer_name not in world:
            return base(t, own, world) if base is not None else np.zeros(2)
        rel = np.asarray(world[pursuer_name], dtype=float)[:2] - own[:2]
        if float(np.linalg.norm(rel)) > trigger_range:
            return base(t, own, world) if base is not None else np.zeros(2)
        perp = _perp(own)
        if np.linalg.norm(perp) < _EPS:
            return np.zeros(2)
        return float(np.sign(sign)) * accel * perp

    return controller
