"""Engagement scenarios: parametric geometry + maneuver, with YAML (de)serialization.

A :class:`ParametricScenario` defines a *distribution* over engagement geometries (initial range,
cross-range offset, target heading) plus a target maneuver. Given a ``numpy`` RNG it samples a
concrete :class:`EngagementSpec`; the guidance law is injected separately at build time. This
separation is the benchmark's **fairness invariant**: the same scenario and the same per-trial
seed produce *identical* initial conditions and target behavior regardless of which guidance law
is under test (see ADR-0003).

Distances are metres, speeds m/s, accelerations m/s² (1 g ≈ 9.81), angles degrees.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from intercept.adversary import evasive, scripted
from intercept.core import AeroMissile2D, Dynamics, Engagement, Entity, PointMass2D
from intercept.core.entities import Controller
from intercept.core.realistic import RealisticMissile2D

Array = NDArray[np.float64]

G = 9.80665  # standard gravity (m/s²), for expressing maneuvers/limits in g


@dataclass
class EngagementSpec:
    """A fully-specified single engagement (everything except the interceptor's guidance law).

    Carries the actual plant objects (``*_dynamics``) so the same spec works for any fidelity level
    (``PointMass2D`` L0 or ``AeroMissile2D`` L2); states are already the correct length for them.
    """

    interceptor_state: Array
    target_state: Array
    interceptor_dynamics: Dynamics
    target_dynamics: Dynamics
    interceptor_a_max: float  # kept for convenience (e.g. RL action scaling)
    target_controller: Controller | None
    kill_radius: float
    dt: float
    t_max: float

    def build(self, guidance_factory) -> Engagement:
        """Construct an :class:`Engagement`, injecting the interceptor guidance via a factory.

        ``guidance_factory(target_name) -> Controller`` must return a *fresh* controller (stateful
        laws like APN keep per-engagement memory), so a new one is built for every trial.
        """
        interceptor = Entity(
            "interceptor",
            self.interceptor_dynamics,
            self.interceptor_state,
            controller=guidance_factory("target"),
            role="interceptor",
        )
        target = Entity(
            "target",
            self.target_dynamics,
            self.target_state,
            controller=self.target_controller,
            role="target",
        )
        return Engagement(
            [interceptor, target],
            interceptor="interceptor",
            target="target",
            dt=self.dt,
            t_max=self.t_max,
            kill_radius=self.kill_radius,
        )


def _accel(spec: Mapping[str, object], default: float = 0.0) -> float:
    """Resolve an acceleration magnitude from a maneuver spec — ``accel`` (m/s²) or ``g`` × g₀."""
    if "accel" in spec:
        return float(spec["accel"])  # type: ignore[arg-type]
    if "g" in spec:
        return float(spec["g"]) * G  # type: ignore[arg-type]
    return default


def make_maneuver(
    spec: Mapping[str, object] | None,
    rng: np.random.Generator | None = None,
    pursuer: str = "interceptor",
) -> Controller | None:
    """Build a target controller from a maneuver spec dict (``None``/``straight`` => coasting).

    Accelerations may be given as ``accel`` (m/s²) or ``g`` (multiples of gravity). Reactive and
    telegraph maneuvers need the per-trial ``rng`` / ``pursuer`` name for reproducibility.
    """
    if not spec:
        return None
    kind = str(spec.get("type", "straight"))
    if kind == "straight":
        return None
    if kind == "weave":
        return scripted.weave(
            _accel(spec, 50.0), float(spec["frequency"]), float(spec.get("phase", 0.0))
        )
    if kind == "step":
        return scripted.step_maneuver(_accel(spec), float(spec.get("t_start", 0.0)))
    if kind == "bang_bang":
        return scripted.bang_bang(
            _accel(spec), float(spec["period"]), float(spec.get("t_start", 0.0))
        )
    if kind == "hard_turn":
        return evasive.hard_turn(_accel(spec), float(spec.get("sign", 1.0)))
    if kind == "telegraph":
        if rng is None:
            raise ValueError("telegraph maneuver requires an rng")
        return evasive.random_telegraph(_accel(spec), float(spec.get("mean_switch", 1.0)), rng)
    if kind == "reactive":
        base = make_maneuver(spec.get("base"), rng, pursuer)  # type: ignore[arg-type]
        return evasive.reactive_break(pursuer, _accel(spec), float(spec["trigger_range"]), base)
    raise ValueError(f"unknown maneuver type: {kind!r}")


@dataclass
class ParametricScenario:
    """A distribution over engagement geometries plus a target maneuver.

    The interceptor starts at the origin. The target starts down-range at ``x ∈ [range_min,
    range_max]`` with cross-range ``y ∈ [offset_min, offset_max]`` and flies at ``target_speed``
    along ``target_heading_deg`` (inertial; 180° = toward −x). ``interceptor_aim="lead"`` points
    the interceptor at the target's initial position; ``"downrange"`` points it along +x.
    """

    name: str
    description: str = ""
    interceptor_speed: float = 1000.0  # ~Mach 3; realistic ~1.45x the target
    interceptor_a_max: float = 250.0
    target_speed: float = 700.0  # ~Mach 2
    range_min: float = 3000.0
    range_max: float = 5000.0
    offset_min: float = -1500.0
    offset_max: float = 1500.0
    target_heading_deg: float = 180.0
    interceptor_aim: str = "lead"
    maneuver: dict | None = None
    kill_radius: float = 10.0
    dt: float = 0.005
    t_max: float = 40.0
    # --- fidelity / physics (L2 when model="aero", L3 when model="realistic"; ADR-0006/0008) ---
    model: str = "point_mass"  # "point_mass" (L0), "aero" (L2), or "realistic" (L3)
    gravity: float = G  # used when model="aero"/"realistic"
    drag: float = 8e-6  # parasitic-drag coeff k_drag (1/m), aero only
    induced: float = 3e-4  # induced-drag coeff k_induced (s²/m), aero only
    interceptor_tau: float = 0.2  # interceptor autopilot lag (s)
    target_tau: float = 0.3  # target autopilot lag (s)
    target_a_max: float = 1000.0  # target g-limit (m/s²); large => effectively unconstrained

    def _interceptor_velocity(self, target_pos: Array) -> Array:
        if self.interceptor_aim == "downrange":
            aim = np.array([1.0, 0.0])
        else:  # "lead": aim at the target's initial position
            aim = target_pos / np.linalg.norm(target_pos)
        return self.interceptor_speed * aim

    def _dynamics_pair(self) -> tuple[Dynamics, Dynamics]:
        if self.model == "aero":
            i = AeroMissile2D(
                a_max=self.interceptor_a_max,
                tau=self.interceptor_tau,
                gravity=self.gravity,
                k_drag=self.drag,
                k_induced=self.induced,
            )
            t = AeroMissile2D(
                a_max=self.target_a_max,
                tau=self.target_tau,
                gravity=self.gravity,
                k_drag=self.drag,
                k_induced=self.induced,
            )
            return i, t
        if self.model == "realistic":
            # L3 aero-propulsive (ADR-0008): a boosting interceptor vs a sustaining threat. The
            # structural g-limit comes from a_max; achievable turn is q/lift-limited by physics.
            i = RealisticMissile2D(
                g_struct=self.interceptor_a_max, tau=self.interceptor_tau, gravity=self.gravity
            )
            t = RealisticMissile2D.target(
                g_struct=self.target_a_max, tau=self.target_tau, gravity=self.gravity
            )
            return i, t
        if self.model == "point_mass":
            return PointMass2D(a_max=self.interceptor_a_max), PointMass2D()
        raise ValueError(f"unknown model: {self.model!r}")

    def _state(self, dyn: Dynamics, pos: Array, vel: Array) -> Array:
        if isinstance(dyn, (AeroMissile2D, RealisticMissile2D)):
            return dyn.initial_state(pos, vel)
        return np.array([pos[0], pos[1], vel[0], vel[1]])

    def _spec(self, target_pos: Array, rng: np.random.Generator) -> EngagementSpec:
        heading = np.radians(self.target_heading_deg)
        vt = self.target_speed * np.array([np.cos(heading), np.sin(heading)])
        vi = self._interceptor_velocity(target_pos)
        i_dyn, t_dyn = self._dynamics_pair()
        return EngagementSpec(
            interceptor_state=self._state(i_dyn, np.array([0.0, 0.0]), vi),
            target_state=self._state(t_dyn, target_pos, vt),
            interceptor_dynamics=i_dyn,
            target_dynamics=t_dyn,
            interceptor_a_max=self.interceptor_a_max,
            target_controller=make_maneuver(self.maneuver, rng),
            kill_radius=self.kill_radius,
            dt=self.dt,
            t_max=self.t_max,
        )

    def sample(self, rng: np.random.Generator) -> EngagementSpec:
        """Sample a randomized engagement (Monte-Carlo trial)."""
        target_pos = np.array(
            [
                rng.uniform(self.range_min, self.range_max),
                rng.uniform(self.offset_min, self.offset_max),
            ]
        )
        return self._spec(target_pos, rng)

    def at(self, downrange: float, crossrange: float) -> EngagementSpec:
        """Deterministic engagement with the target at a fixed position (capture-region sweeps)."""
        return self._spec(np.array([float(downrange), float(crossrange)]), np.random.default_rng(0))


def load_scenario(path: str | Path) -> ParametricScenario:
    """Load a :class:`ParametricScenario` from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return ParametricScenario(**data)


def load_suite(directory: str | Path) -> dict[str, ParametricScenario]:
    """Load every ``*.yaml`` scenario in a directory, keyed by name (sorted by filename)."""
    directory = Path(directory)
    suite: dict[str, ParametricScenario] = {}
    for path in sorted(directory.glob("*.yaml")):
        sc = load_scenario(path)
        suite[sc.name] = sc
    return suite
