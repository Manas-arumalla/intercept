"""Game-theoretic optimal evasion for the simple-motion pursuit-evasion game.

Against a faster pursuer in the simple-motion (constant-speed) differential game, capture is
inevitable, and the evader's optimal *game-of-degree* strategy — the one that maximizes time-to-
capture / closest-approach distance — is to flee **directly along the line of sight, away from the
pursuer** (Isaacs). This is a genuinely different, harder adversary than the open-loop scripted
maneuvers (weave/jink): it reacts to the pursuer's position every step.

`optimal_evader` returns a `Controller` that turns the evader's velocity toward the anti-LOS
direction (away from the pursuer), commanding lateral acceleration from the heading error.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from intercept.core.entities import Controller

Array = NDArray[np.float64]
_EPS = 1e-9


def optimal_evader(pursuer_name: str, gain: float = 6.0) -> Controller:
    """Evader that flees directly away from the pursuer (optimal simple-motion evasion).

    Parameters
    ----------
    pursuer_name:
        Name of the pursuer entity in the world snapshot (the threat to flee from).
    gain:
        Heading-correction gain for turning the velocity toward the anti-LOS (escape) direction.
    """

    def controller(t: float, own: Array, world: Mapping[str, Array]) -> Array:
        v = own[2:4]
        speed = float(np.linalg.norm(v))
        if speed < _EPS or pursuer_name not in world:
            return np.zeros(2)
        # Desired heading: directly away from the pursuer (anti-line-of-sight).
        away = own[:2] - np.asarray(world[pursuer_name], dtype=float)[:2]
        if np.linalg.norm(away) < _EPS:
            return np.zeros(2)
        desired = np.arctan2(away[1], away[0])
        current = np.arctan2(v[1], v[0])
        err = (desired - current + np.pi) % (2 * np.pi) - np.pi
        a_lat = gain * speed * err
        v_hat = v / speed
        return a_lat * np.array([-v_hat[1], v_hat[0]])

    return controller
