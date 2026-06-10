"""Higher-fidelity planar missile dynamics (fidelity level L2; see ADR-0002, ADR-0006).

`AeroMissile2D` is a 3-DOF point-mass model with the physics that make interception *hard and
realistic* — the things the idealized `PointMass2D` (L0) omits:

* **Gravity** — trajectories arc; sustained turns must fight gravity.
* **Parasitic drag** ``∝ V²`` — speed bleeds over the engagement (no free constant-speed cruise).
* **Induced drag** ``∝ a_lat²`` — *pulling g costs energy*: a hard-maneuvering target slows down,
  the central realistic trade-off (an evader that jinks hard loses the speed it needs to escape).
* **Bounded lateral acceleration** (a hard g-limit) — the missile cannot turn arbitrarily fast.
* **First-order autopilot lag** ``τ`` — commanded acceleration is *not* achieved instantly; the
  response delay forces the guidance to lead and makes terminal precision genuinely difficult.

Only the **lateral** (perpendicular-to-velocity) component of a guidance command is achievable as
body lift; the along-velocity axis is governed by thrust − drag − gravity. This is the standard
Zarchan 3-DOF kinematic missile, written in Cartesian state so it drops in behind the same
`Dynamics` interface as L0 — guidance, estimation, RL, and the benchmark are unchanged.

State ``[x, y, vx, vy, ax, ay]`` (position, velocity, *achieved* lateral acceleration — the extra
two states carry the autopilot lag). Control ``[acx, acy]`` is the commanded acceleration vector
(from any guidance law); its lateral component is g-limited and low-pass filtered into the achieved
acceleration. Guidance/estimation read only ``state[:4]`` and are oblivious to the extra states.

Reference: Zarchan, *Tactical and Strategic Missile Guidance* (3-DOF engagement modeling).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from intercept.core.dynamics import Dynamics

Array = NDArray[np.float64]

G0 = 9.80665  # standard gravity (m/s²)


class AeroMissile2D(Dynamics):
    """Planar 3-DOF missile: gravity, parasitic + induced drag, g-limit, autopilot lag.

    Parameters
    ----------
    a_max:
        Lateral-acceleration (g-limit) bound in m/s² (e.g. ``40 * G0`` for a 40 g interceptor).
    tau:
        Autopilot first-order time constant (s); larger = more sluggish response.
    gravity:
        Gravitational acceleration (m/s²); set 0 to disable.
    k_drag:
        Parasitic-drag coefficient (1/m): along-track deceleration is ``k_drag · V²``.
    k_induced:
        Induced-drag coefficient (s²/m): extra deceleration ``k_induced · |a_lat|²`` from pulling g.
    thrust:
        Along-velocity thrust acceleration (m/s²); default 0 (coasting / sustainer-off).
    """

    state_labels = ("x", "y", "vx", "vy", "ax", "ay")
    control_labels = ("acx", "acy")

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
        return 6

    @property
    def control_dim(self) -> int:
        return 2

    def initial_state(self, position: Array, velocity: Array) -> Array:
        """Build a full 6-state from position/velocity (achieved acceleration starts at zero)."""
        p = np.asarray(position, dtype=float)
        v = np.asarray(velocity, dtype=float)
        return np.array([p[0], p[1], v[0], v[1], 0.0, 0.0])

    def saturate(self, control: Array) -> Array:
        """Clip the commanded acceleration magnitude to the g-limit (for effort logging)."""
        u = np.asarray(control, dtype=float)
        mag = float(np.linalg.norm(u))
        if mag > self.a_max and mag > 0.0:
            return u * (self.a_max / mag)
        return u

    def _lateral_command(self, u: Array, v_hat: Array) -> Array:
        """Perpendicular-to-velocity component of the command, clipped to the g-limit."""
        perp = u - (u @ v_hat) * v_hat
        mag = float(np.linalg.norm(perp))
        if mag > self.a_max and mag > 0.0:
            perp = perp * (self.a_max / mag)
        return perp

    def derivative(self, t: float, state: Array, control: Array) -> Array:
        x = np.asarray(state, dtype=float)
        u = np.asarray(control, dtype=float)
        v = x[2:4]
        a_ach = x[4:6]
        speed = float(np.linalg.norm(v))
        v_hat = v / speed if speed > 1e-9 else np.array([1.0, 0.0])

        # Autopilot lag: achieved lateral accel chases the (g-limited) lateral command.
        u_perp = self._lateral_command(u, v_hat)
        a_dot = (u_perp - a_ach) / self.tau

        # Along-track: thrust − parasitic drag (∝V²) − induced drag (∝ a_lat²).
        decel = self.thrust - self.k_drag * speed * speed - self.k_induced * float(a_ach @ a_ach)
        a_long = decel * v_hat
        a_grav = np.array([0.0, -self.gravity])

        accel = a_ach + a_long + a_grav
        return np.array([v[0], v[1], accel[0], accel[1], a_dot[0], a_dot[1]])
