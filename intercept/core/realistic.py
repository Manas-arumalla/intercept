"""L3 realistic aero-propulsive missile dynamics (no prescribed g-limit or constant drag).

Everything that was a hand-set constant at L2 now *emerges from physics* (ADR-0008):

* **Propulsion** — a rocket **boost → sustain → coast** thrust schedule with **propellant mass
  burn-off**, so the missile accelerates on the boost and decelerates (drag) when it coasts; its
  acceleration capability ``T/m`` grows as it lightens.
* **Mach-dependent drag** through the **ISA standard atmosphere**: ``D = ½ρ(h)V² · S · Cd(M)`` with
  the transonic drag rise; plus **lift-induced drag** ``∝ C_L²`` (so turning costs energy, exactly).
* **Lift / dynamic-pressure-limited turning** — the achievable lateral acceleration is
  ``min(structural limit, q·S·C_Lmax / m)``. Turn capability *falls* at low speed / high altitude
  (low ``q``): the interceptor cannot simply pull arbitrary g — a real, no-cheat constraint.
* First-order **autopilot lag**, gravity, and (only) the lateral command is achievable as lift.

State ``[pos(n), vel(n), a_lat(n)]`` (n = 2 or 3; the last n are the achieved lateral acceleration
for the autopilot lag). Control is the commanded acceleration vector. Altitude is the last position
component (y in 2-D, z in 3-D), consistent with gravity along −that axis. Reads `state[:n]` /
`state[n:2n]` so all guidance/estimation work unchanged. Reference: standard "modified point-mass"
aeroballistic modeling (Zarchan; aeroprediction practice).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from intercept.core.atmosphere import drag_coefficient, isa
from intercept.core.dynamics import Dynamics

Array = NDArray[np.float64]
G0 = 9.80665


class _RealisticMissile(Dynamics):
    """Dimension-generic L3 aero-propulsive point mass (subclassed for 2-D / 3-D)."""

    ndim: int = 2

    def __init__(
        self,
        *,
        mass: float = 180.0,  # launch mass (kg)
        dry_mass: float = 105.0,  # burnout mass (kg) — 75 kg propellant
        ref_area: float = 0.05,  # aerodynamic reference area (m²)
        cl_max: float = 6.0,  # max lift coefficient (body + fins at high alpha)
        induced_k: float = 0.15,  # lift-induced drag factor (Cd_i = k·C_L²)
        g_struct: float = 45.0 * G0,  # structural load-factor limit (m/s²)
        tau: float = 0.18,  # autopilot lag (s)
        gravity: float = G0,
        thrust_boost: float = 42000.0,  # ~3 s boost to ~Mach 3.5+
        t_boost: float = 3.0,
        thrust_sustain: float = 8000.0,
        t_sustain: float = 8.0,
    ) -> None:
        if mass <= 0 or dry_mass <= 0 or dry_mass > mass:
            raise ValueError("require 0 < dry_mass <= mass")
        if ref_area <= 0 or cl_max <= 0 or tau <= 0 or g_struct <= 0:
            raise ValueError("ref_area, cl_max, tau, g_struct must be positive")
        self.mass0 = float(mass)
        self.dry_mass = float(dry_mass)
        self.S = float(ref_area)
        self.cl_max = float(cl_max)
        self.induced_k = float(induced_k)
        self.g_struct = float(g_struct)
        self.tau = float(tau)
        self.gravity = float(gravity)
        self.thrust_boost = float(thrust_boost)
        self.t_boost = float(t_boost)
        self.thrust_sustain = float(thrust_sustain)
        self.t_sustain = float(t_sustain)
        self._impulse = self.thrust_boost * self.t_boost + self.thrust_sustain * self.t_sustain

    @property
    def state_dim(self) -> int:
        return 3 * self.ndim

    @property
    def control_dim(self) -> int:
        return self.ndim

    def position(self, state: Array) -> Array:
        return np.asarray(state, dtype=float)[: self.ndim]

    def velocity(self, state: Array) -> Array:
        return np.asarray(state, dtype=float)[self.ndim : 2 * self.ndim]

    def initial_state(self, position: Array, velocity: Array) -> Array:
        return np.concatenate(
            [np.asarray(position, float), np.asarray(velocity, float), np.zeros(self.ndim)]
        )

    # --- propulsion schedule ---
    def thrust(self, t: float) -> float:
        if t < self.t_boost:
            return self.thrust_boost
        if t < self.t_boost + self.t_sustain:
            return self.thrust_sustain
        return 0.0

    def mass(self, t: float) -> float:
        if self._impulse <= 0:
            return self.mass0
        if t <= 0:
            delivered = 0.0
        elif t < self.t_boost:
            delivered = self.thrust_boost * t
        elif t < self.t_boost + self.t_sustain:
            delivered = self.thrust_boost * self.t_boost + self.thrust_sustain * (t - self.t_boost)
        else:
            delivered = self._impulse
        frac = min(delivered / self._impulse, 1.0)
        return self.mass0 - (self.mass0 - self.dry_mass) * frac

    def max_lateral_accel(self, state: Array, t: float = 0.0) -> float:
        """Achievable lateral acceleration = min(structural, lift/dynamic-pressure limit) (m/s²)."""
        x = np.asarray(state, dtype=float)
        v = x[self.ndim : 2 * self.ndim]
        speed = float(np.linalg.norm(v))
        rho = isa(float(x[self.ndim - 1]))[0]
        q = 0.5 * rho * speed * speed
        a_aero = q * self.S * self.cl_max / self.mass(t)
        return min(self.g_struct, a_aero)

    def saturate(self, control: Array) -> Array:
        u = np.asarray(control, dtype=float)
        mag = float(np.linalg.norm(u))
        return u * (self.g_struct / mag) if mag > self.g_struct and mag > 0 else u

    def _grav_vec(self) -> Array:
        g = np.zeros(self.ndim)
        g[self.ndim - 1] = -self.gravity
        return g

    def derivative(self, t: float, state: Array, control: Array) -> Array:
        n = self.ndim
        x = np.asarray(state, dtype=float)
        u = np.asarray(control, dtype=float)
        pos, v, a_ach = x[:n], x[n : 2 * n], x[2 * n : 3 * n]
        speed = float(np.linalg.norm(v))
        v_hat = v / speed if speed > 1e-9 else np.eye(n)[0]

        rho, a_sound, _, _ = isa(float(pos[n - 1]))
        m = self.mass(t)
        q = 0.5 * rho * speed * speed
        mach = speed / a_sound if a_sound > 0 else 0.0

        # Lateral command: perpendicular to velocity, clipped to the physics-derived turn limit.
        a_lat_max = min(self.g_struct, q * self.S * self.cl_max / m) if q > 0 else self.g_struct
        perp = u - (u @ v_hat) * v_hat
        pmag = float(np.linalg.norm(perp))
        if pmag > a_lat_max and pmag > 0:
            perp = perp * (a_lat_max / pmag)
        a_dot = (perp - a_ach) / self.tau

        # Drag: zero-lift (Mach) + lift-induced (from the achieved lateral accel).
        cl = (m * float(np.linalg.norm(a_ach)) / (q * self.S)) if q > 1e-6 else 0.0
        cd = drag_coefficient(mach) + self.induced_k * cl * cl
        drag = q * self.S * cd
        a_long = (self.thrust(t) - drag) / m

        accel = a_ach + a_long * v_hat + self._grav_vec()
        return np.concatenate([v, accel, a_dot])


class RealisticMissile2D(_RealisticMissile):
    """L3 aero-propulsive missile in the vertical plane (x downrange, y altitude)."""

    ndim = 2
    state_labels = ("x", "y", "vx", "vy", "ax", "ay")
    control_labels = ("acx", "acy")

    @classmethod
    def target(cls, **kw) -> RealisticMissile2D:
        """Preset: a fast, maneuverable threat (sustainer holds speed, ~35 g structural)."""
        params = dict(
            mass=320.0,
            dry_mass=300.0,
            ref_area=0.1,
            cl_max=2.5,
            induced_k=0.2,
            g_struct=35.0 * G0,
            tau=0.3,
            thrust_boost=0.0,
            t_boost=0.0,
            thrust_sustain=2600.0,
            t_sustain=40.0,
        )
        params.update(kw)
        return cls(**params)


class RealisticMissile3D(_RealisticMissile):
    """L3 aero-propulsive missile in 3-D (z is altitude)."""

    ndim = 3
    state_labels = ("x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az")
    control_labels = ("acx", "acy", "acz")

    @classmethod
    def target(cls, **kw) -> RealisticMissile3D:
        """Preset: a fast, maneuverable 3-D threat."""
        params = dict(
            mass=320.0,
            dry_mass=300.0,
            ref_area=0.1,
            cl_max=2.5,
            induced_k=0.2,
            g_struct=35.0 * G0,
            tau=0.3,
            thrust_boost=0.0,
            t_boost=0.0,
            thrust_sustain=2600.0,
            t_sustain=40.0,
        )
        params.update(kw)
        return cls(**params)
