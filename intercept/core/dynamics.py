"""Dynamics models.

A ``Dynamics`` exposes a continuous-time state derivative ``f(t, state, control)``; the
integrator (see :mod:`intercept.core.integrators`) advances it. Keeping dynamics as pure
``derivative`` functions lets us swap integrators and reuse the same model for both forward
simulation and (later) optimization/autodiff backends.

Fidelity ladder (see ADR-0002): we start with a 2D point-mass and climb toward 3D and
autopilot-lag / saturation variants without changing the interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class Dynamics(ABC):
    """Continuous-time dynamics interface: ``dx/dt = f(t, x, u)``.

    Conventions
    -----------
    * ``state`` and ``control`` are 1-D float arrays; their layout is model-specific and
      documented by :pyattr:`state_labels` / :pyattr:`control_labels`.
    * Implementations must be pure (no hidden mutation of inputs) so the same model is safe
      to reuse across integrators, rollouts, and parallel Monte-Carlo trials.
    """

    #: Human-readable names for each state component (length == :pyattr:`state_dim`).
    state_labels: tuple[str, ...] = ()
    #: Human-readable names for each control component (length == :pyattr:`control_dim`).
    control_labels: tuple[str, ...] = ()

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state vector."""

    @property
    @abstractmethod
    def control_dim(self) -> int:
        """Dimension of the control vector."""

    @abstractmethod
    def derivative(self, t: float, state: Array, control: Array) -> Array:
        """Return ``dx/dt`` at time ``t`` for the given ``state`` and ``control``."""

    def saturate(self, control: Array) -> Array:
        """Return the control actually applied by the plant (default: identity, no limits).

        Models with finite control authority override this; the engagement logs the saturated
        command so effort metrics reflect what the plant could actually deliver, not unbounded
        guidance commands (which diverge as range -> 0)."""
        return np.asarray(control, dtype=float)

    # --- convenience accessors (override in subclasses that have these concepts) ---

    def position(self, state: Array) -> Array:
        """Extract the position sub-vector from a state (default: first 2 components)."""
        return np.asarray(state, dtype=float)[:2]

    def velocity(self, state: Array) -> Array:
        """Extract the velocity sub-vector from a state (default: components 2:4)."""
        return np.asarray(state, dtype=float)[2:4]


class PointMass2D(Dynamics):
    """2-D point mass with acceleration command and optional acceleration saturation.

    State  ``[x, y, vx, vy]``  (m, m, m/s, m/s)
    Control ``[ax, ay]``       (m/s^2) — commanded inertial acceleration.

    This is the L0 fidelity model: the commanded acceleration is applied directly (no
    autopilot lag). Speed is *not* held constant; lateral vs. longitudinal acceleration is a
    property of the guidance law, not the plant. Optional ``a_max`` clips the command
    magnitude to model finite control authority.

    Parameters
    ----------
    a_max:
        Maximum acceleration magnitude in m/s^2 (``None`` = unlimited). Often expressed as a
        multiple of g elsewhere; here it is raw m/s^2.
    drag_coeff:
        Optional linear drag ``-k * v`` (1/s). Defaults to 0 (no drag) for the ideal model.
    """

    state_labels = ("x", "y", "vx", "vy")
    control_labels = ("ax", "ay")

    def __init__(self, a_max: float | None = None, drag_coeff: float = 0.0) -> None:
        if a_max is not None and a_max <= 0:
            raise ValueError("a_max must be positive or None")
        if drag_coeff < 0:
            raise ValueError("drag_coeff must be non-negative")
        self.a_max = a_max
        self.drag_coeff = drag_coeff

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def control_dim(self) -> int:
        return 2

    def saturate(self, control: Array) -> Array:
        """Clip the commanded acceleration to ``a_max`` magnitude (no-op if unlimited)."""
        u = np.asarray(control, dtype=float)
        if self.a_max is None:
            return u
        mag = float(np.linalg.norm(u))
        if mag > self.a_max and mag > 0.0:
            return u * (self.a_max / mag)
        return u

    def derivative(self, t: float, state: Array, control: Array) -> Array:
        x = np.asarray(state, dtype=float)
        u = self.saturate(control)
        vx, vy = x[2], x[3]
        ax = u[0] - self.drag_coeff * vx
        ay = u[1] - self.drag_coeff * vy
        return np.array([vx, vy, ax, ay], dtype=float)
