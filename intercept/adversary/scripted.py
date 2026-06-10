"""Scripted (open-loop) evasive maneuvers for target entities.

These conform to the :data:`intercept.core.entities.Controller` contract and command lateral
acceleration *perpendicular to the target's own velocity* (so they turn without changing speed in
the ideal point-mass model). They provide the difficulty curriculum that separates a real
guidance benchmark from a "does it hit a straight line" toy:

* :func:`straight` — no maneuver (constant velocity).
* :func:`weave` — sinusoidal lateral acceleration (the classic PN-stressing maneuver).
* :func:`step_maneuver` — constant lateral acceleration after a trigger time (a hard turn / jink).
* :func:`bang_bang` — sign-alternating constant-magnitude lateral acceleration (square-wave jink).

Game-theoretic and RL evaders arrive in P6; these scripted laws are the P1 baseline adversaries.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from intercept.core.entities import Controller

Array = NDArray[np.float64]
_EPS = 1e-9


def _perp_of_velocity(state: Array) -> Array:
    """Unit vector perpendicular (+90 deg) to the entity's velocity; zero if at rest."""
    v = state[2:4]
    s = float(np.linalg.norm(v))
    if s < _EPS:
        return np.zeros(2)
    return np.array([-v[1], v[0]]) / s


def straight() -> Controller:
    """Constant-velocity target (no control)."""

    def controller(t: float, own: Array, world: Mapping[str, Array]) -> Array:
        return np.zeros(2)

    return controller


def weave(amplitude: float, frequency: float, phase: float = 0.0) -> Controller:
    """Sinusoidal weave: lateral accel ``A·sin(2π f t + φ)`` perpendicular to velocity.

    Parameters
    ----------
    amplitude:
        Peak lateral acceleration (m/s²).
    frequency:
        Weave frequency (Hz).
    phase:
        Phase offset (rad).
    """

    w = 2.0 * np.pi * frequency

    def controller(t: float, own: Array, world: Mapping[str, Array]) -> Array:
        return amplitude * np.sin(w * t + phase) * _perp_of_velocity(own)

    return controller


def step_maneuver(accel: float, t_start: float = 0.0) -> Controller:
    """Constant lateral accel of magnitude ``accel`` beginning at ``t_start`` (a hard turn)."""

    def controller(t: float, own: Array, world: Mapping[str, Array]) -> Array:
        if t < t_start:
            return np.zeros(2)
        return accel * _perp_of_velocity(own)

    return controller


def bang_bang(accel: float, period: float, t_start: float = 0.0) -> Controller:
    """Square-wave jink: lateral accel of magnitude ``accel`` flipping sign every ``period/2``."""

    def controller(t: float, own: Array, world: Mapping[str, Array]) -> Array:
        if t < t_start:
            return np.zeros(2)
        sign = 1.0 if (((t - t_start) // (period / 2.0)) % 2 == 0) else -1.0
        return sign * accel * _perp_of_velocity(own)

    return controller
