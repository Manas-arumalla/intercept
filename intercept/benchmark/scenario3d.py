"""3-D engagement scenarios for the Monte-Carlo benchmark.

The 2-D :class:`~intercept.benchmark.scenario.ParametricScenario` proved the fairness invariant;
this extends the *same* idea to 3-D so the benchmark runner, metrics, and statistics (which are
dimension-agnostic — they operate on an :class:`EngagementResult`) compare 3-D guidance laws on
identical 3-D geometries. Only the geometry sampler and the plant/maneuver builders change.

The interceptor starts at the origin; the target starts at ``(x, y, z)`` = (downrange, cross-range,
altitude) sampled from the given ranges, flying at ``target_speed`` along an inertial heading set by
``target_azimuth_deg`` (in the x–y plane, 180° ⇒ toward −x) and ``target_elevation_deg`` (climb +).
``interceptor_aim="lead"`` points the interceptor straight at the target's initial position (so it
leads upward to altitude); ``"downrange"`` points it along +x. Fidelity follows ``model``:
``point_mass`` (L0 ``PointMass3D``), ``aero`` (L2 ``AeroMissile3D``), ``realistic`` (L3
``RealisticMissile3D``). Gravity acts along −z for the aero/realistic plants.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from intercept.adversary import barrel_roll, serpentine3d, terminal_spiral, weave3d
from intercept.benchmark.scenario import EngagementSpec, _accel
from intercept.core import AeroMissile3D, Dynamics, PointMass3D
from intercept.core.entities import Controller
from intercept.core.realistic import RealisticMissile3D

Array = NDArray[np.float64]

G = 9.80665


def make_maneuver_3d(
    spec: Mapping[str, object] | None,
    pursuer: str = "interceptor",
) -> Controller | None:
    """Build a 3-D target controller from a maneuver spec dict (``None``/``straight`` => coasting).

    Accelerations are given as ``accel`` (m/s²) or ``g`` (multiples of g₀). Supported ``type``s:
    ``weave`` (3-D sinusoidal weave along a lateral axis), ``barrel_roll`` (helical corkscrew),
    ``serpentine`` (tilted 3-D S-weave), ``spiral`` (closed-loop intensifying terminal spiral).
    """
    if not spec:
        return None
    kind = str(spec.get("type", "straight"))
    if kind == "straight":
        return None
    if kind == "weave":
        return weave3d(_accel(spec, 50.0), float(spec["frequency"]), str(spec.get("axis", "e1")))
    if kind == "barrel_roll":
        return barrel_roll(_accel(spec, 100.0), float(spec.get("rate", 1.0)))
    if kind == "serpentine":
        return serpentine3d(
            _accel(spec, 100.0), float(spec["frequency"]), float(spec.get("tilt", 0.5))
        )
    if kind == "spiral":
        return terminal_spiral(
            pursuer,
            _accel(spec, 80.0),
            float(spec.get("max_accel", _accel(spec, 80.0) * 4.0)),
            float(spec.get("trigger_range", 2500.0)),
            float(spec.get("rate", 1.5)),
        )
    raise ValueError(f"unknown 3-D maneuver type: {kind!r}")


@dataclass
class ParametricScenario3D:
    """A distribution over 3-D engagement geometries plus a 3-D target maneuver."""

    name: str
    description: str = ""
    interceptor_speed: float = 900.0
    interceptor_a_max: float = 40.0 * G
    target_speed: float = 700.0
    range_min: float = 6000.0
    range_max: float = 9000.0
    offset_min: float = -1500.0
    offset_max: float = 1500.0
    alt_min: float = 2000.0
    alt_max: float = 5000.0
    target_azimuth_deg: float = 180.0
    target_elevation_deg: float = 0.0
    interceptor_aim: str = "lead"
    maneuver: dict | None = None
    kill_radius: float = 20.0
    dt: float = 0.01
    t_max: float = 25.0
    model: str = "aero"  # "point_mass" (L0), "aero" (L2), "realistic" (L3)
    gravity: float = G
    drag: float = 8e-6
    induced: float = 3e-4
    interceptor_tau: float = 0.2
    target_tau: float = 0.3
    target_a_max: float = 1000.0

    def _target_velocity(self) -> Array:
        az = np.radians(self.target_azimuth_deg)
        el = np.radians(self.target_elevation_deg)
        return self.target_speed * np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )

    def _interceptor_velocity(self, target_pos: Array) -> Array:
        if self.interceptor_aim == "downrange":
            aim = np.array([1.0, 0.0, 0.0])
        else:  # "lead": aim straight at the target's initial position
            aim = target_pos / np.linalg.norm(target_pos)
        return self.interceptor_speed * aim

    def _dynamics_pair(self) -> tuple[Dynamics, Dynamics]:
        if self.model == "point_mass":
            return PointMass3D(a_max=self.interceptor_a_max), PointMass3D()
        if self.model == "aero":
            i = AeroMissile3D(
                a_max=self.interceptor_a_max,
                tau=self.interceptor_tau,
                gravity=self.gravity,
                k_drag=self.drag,
                k_induced=self.induced,
            )
            t = AeroMissile3D(
                a_max=self.target_a_max,
                tau=self.target_tau,
                gravity=self.gravity,
                k_drag=self.drag,
                k_induced=self.induced,
            )
            return i, t
        if self.model == "realistic":
            i = RealisticMissile3D(
                g_struct=self.interceptor_a_max, tau=self.interceptor_tau, gravity=self.gravity
            )
            t = RealisticMissile3D.target(
                g_struct=self.target_a_max, tau=self.target_tau, gravity=self.gravity
            )
            return i, t
        raise ValueError(f"unknown model: {self.model!r}")

    @staticmethod
    def _state(dyn: Dynamics, pos: Array, vel: Array) -> Array:
        if hasattr(dyn, "initial_state"):
            return dyn.initial_state(pos, vel)
        return np.concatenate([np.asarray(pos, float), np.asarray(vel, float)])

    def _spec(self, target_pos: Array) -> EngagementSpec:
        vt = self._target_velocity()
        vi = self._interceptor_velocity(target_pos)
        i_dyn, t_dyn = self._dynamics_pair()
        return EngagementSpec(
            interceptor_state=self._state(i_dyn, np.zeros(3), vi),
            target_state=self._state(t_dyn, target_pos, vt),
            interceptor_dynamics=i_dyn,
            target_dynamics=t_dyn,
            interceptor_a_max=self.interceptor_a_max,
            target_controller=make_maneuver_3d(self.maneuver),
            kill_radius=self.kill_radius,
            dt=self.dt,
            t_max=self.t_max,
        )

    def sample(self, rng: np.random.Generator) -> EngagementSpec:
        """Sample a randomized 3-D engagement (Monte-Carlo trial)."""
        target_pos = np.array(
            [
                rng.uniform(self.range_min, self.range_max),
                rng.uniform(self.offset_min, self.offset_max),
                rng.uniform(self.alt_min, self.alt_max),
            ]
        )
        return self._spec(target_pos)


def load_scenario_3d(path: str | Path) -> ParametricScenario3D:
    """Load a :class:`ParametricScenario3D` from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return ParametricScenario3D(**data)
