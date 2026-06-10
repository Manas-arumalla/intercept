"""Numerical integrators for advancing :class:`~intercept.core.dynamics.Dynamics`.

We default to fixed-step RK4: deterministic, seed-free, and accurate enough for point-mass
engagements while keeping Monte-Carlo runs reproducible. The control is held constant across
the step (zero-order hold), matching a digital guidance computer running at a fixed rate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from intercept.core.dynamics import Dynamics

Array = NDArray[np.float64]


def integrate_rk4(dynamics: Dynamics, t: float, state: Array, control: Array, dt: float) -> Array:
    """Advance ``state`` by ``dt`` with classical RK4 (control held constant over the step)."""
    x = np.asarray(state, dtype=float)
    u = np.asarray(control, dtype=float)
    k1 = dynamics.derivative(t, x, u)
    k2 = dynamics.derivative(t + 0.5 * dt, x + 0.5 * dt * k1, u)
    k3 = dynamics.derivative(t + 0.5 * dt, x + 0.5 * dt * k2, u)
    k4 = dynamics.derivative(t + dt, x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class Integrator(ABC):
    """Strategy interface for stepping dynamics forward in time."""

    @abstractmethod
    def step(self, dynamics: Dynamics, t: float, state: Array, control: Array, dt: float) -> Array:
        """Return the state after advancing by ``dt`` from time ``t``."""


class RK4(Integrator):
    """Classical fixed-step 4th-order Runge-Kutta integrator."""

    def step(self, dynamics: Dynamics, t: float, state: Array, control: Array, dt: float) -> Array:
        return integrate_rk4(dynamics, t, state, control, dt)
