"""Three-dimensional dynamics — because real engagements are 3-D.

Two models mirroring the 2-D ones, with position/velocity in ℝ³:

* ``PointMass3D`` (L0): state ``[x,y,z,vx,vy,vz]``, control ``[ax,ay,az]`` — direct acceleration.
* ``AeroMissile3D`` (L2): state ``[x,y,z,vx,vy,vz,ax,ay,az]`` — gravity (along −z), parasitic +
  induced drag, hard g-limit, first-order autopilot lag; only the lateral (⟂-velocity) command is
  achievable as lift. Same physics as :class:`~intercept.core.aero.AeroMissile2D`, in 3-D.

Both override :meth:`position`/:meth:`velocity` to slice ℝ³, so the (dimension-agnostic)
:class:`~intercept.core.engagement.Engagement` loop, segment-distance intercept test, and metrics
all work in 3-D unchanged. Guidance reads ``state[:3]`` / ``state[3:6]`` via the 3-D frames helpers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from intercept.core.aero import G0
from intercept.core.dynamics import Dynamics

Array = NDArray[np.float64]


class PointMass3D(Dynamics):
    """3-D point mass with direct acceleration command and optional g-limit / drag / gravity."""

    state_labels = ("x", "y", "z", "vx", "vy", "vz")
    control_labels = ("ax", "ay", "az")

    def __init__(
        self, a_max: float | None = None, drag_coeff: float = 0.0, gravity: float = 0.0
    ) -> None:
        if a_max is not None and a_max <= 0:
            raise ValueError("a_max must be positive or None")
        if drag_coeff < 0:
            raise ValueError("drag_coeff must be non-negative")
        self.a_max = a_max
        self.drag_coeff = drag_coeff
        self.gravity = gravity

    @property
    def state_dim(self) -> int:
        return 6

    @property
    def control_dim(self) -> int:
        return 3

    def position(self, state: Array) -> Array:
        return np.asarray(state, dtype=float)[:3]

    def velocity(self, state: Array) -> Array:
        return np.asarray(state, dtype=float)[3:6]

    def saturate(self, control: Array) -> Array:
        u = np.asarray(control, dtype=float)
        if self.a_max is None:
            return u
        mag = float(np.linalg.norm(u))
        return u * (self.a_max / mag) if mag > self.a_max and mag > 0 else u

    def derivative(self, t: float, state: Array, control: Array) -> Array:
        x = np.asarray(state, dtype=float)
        u = self.saturate(control)
        v = x[3:6]
        accel = u - self.drag_coeff * v + np.array([0.0, 0.0, -self.gravity])
        return np.concatenate([v, accel])


class AeroMissile3D(Dynamics):
    """3-D planar-equivalent 3-DOF missile: gravity, drag, induced drag, g-limit, autopilot lag."""

    state_labels = ("x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az")
    control_labels = ("acx", "acy", "acz")

    def __init__(
        self,
        a_max: float = 40.0 * G0,
        tau: float = 0.2,
        gravity: float = G0,
        k_drag: float = 8e-6,
        k_induced: float = 3e-4,
        thrust: float = 0.0,
    ) -> None:
        if a_max <= 0 or tau <= 0:
            raise ValueError("a_max and tau must be positive")
        if min(gravity, k_drag, k_induced) < 0 or thrust < 0:
            raise ValueError("gravity, drag, and thrust coefficients must be non-negative")
        self.a_max = float(a_max)
        self.tau = float(tau)
        self.gravity = float(gravity)
        self.k_drag = float(k_drag)
        self.k_induced = float(k_induced)
        self.thrust = float(thrust)

    @property
    def state_dim(self) -> int:
        return 9

    @property
    def control_dim(self) -> int:
        return 3

    def position(self, state: Array) -> Array:
        return np.asarray(state, dtype=float)[:3]

    def velocity(self, state: Array) -> Array:
        return np.asarray(state, dtype=float)[3:6]

    def initial_state(self, position: Array, velocity: Array) -> Array:
        p = np.asarray(position, dtype=float)
        v = np.asarray(velocity, dtype=float)
        return np.array([p[0], p[1], p[2], v[0], v[1], v[2], 0.0, 0.0, 0.0])

    def saturate(self, control: Array) -> Array:
        u = np.asarray(control, dtype=float)
        mag = float(np.linalg.norm(u))
        return u * (self.a_max / mag) if mag > self.a_max and mag > 0 else u

    def _lateral_command(self, u: Array, v_hat: Array) -> Array:
        perp = u - (u @ v_hat) * v_hat
        mag = float(np.linalg.norm(perp))
        return perp * (self.a_max / mag) if mag > self.a_max and mag > 0 else perp

    def derivative(self, t: float, state: Array, control: Array) -> Array:
        x = np.asarray(state, dtype=float)
        u = np.asarray(control, dtype=float)
        v = x[3:6]
        a_ach = x[6:9]
        speed = float(np.linalg.norm(v))
        v_hat = v / speed if speed > 1e-9 else np.array([1.0, 0.0, 0.0])

        u_perp = self._lateral_command(u, v_hat)
        a_dot = (u_perp - a_ach) / self.tau

        decel = self.thrust - self.k_drag * speed * speed - self.k_induced * float(a_ach @ a_ach)
        a_long = decel * v_hat
        a_grav = np.array([0.0, 0.0, -self.gravity])
        accel = a_ach + a_long + a_grav
        return np.concatenate([v, accel, a_dot])
