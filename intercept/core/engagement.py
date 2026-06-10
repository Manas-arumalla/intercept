"""The engagement simulation loop.

:class:`Engagement` advances a set of :class:`~intercept.core.entities.Entity` objects with a
shared integrator and time step, logs full trajectories, and detects termination: intercept
(range within the kill radius), miss (closest approach passed without intercept), ground
impact, or timeout. The result is a self-contained :class:`EngagementResult` carrying every
quantity the metrics layer needs — miss distance, intercept time, control histories.

Determinism: with fixed ``dt`` and deterministic controllers, a run is fully reproducible.
Stochasticity (sensor noise, randomized initial conditions) is injected by callers/sensors via
an explicit RNG, never sampled inside this loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from intercept.core.entities import Entity
from intercept.core.frames import segment_min_distance
from intercept.core.integrators import RK4, Integrator

Array = NDArray[np.float64]


class TerminationReason(str, Enum):
    """Why an engagement ended."""

    INTERCEPT = "intercept"
    MISS = "miss"
    TIMEOUT = "timeout"
    GROUND = "ground"


@dataclass
class EngagementResult:
    """Outcome and full trajectory log of a single engagement.

    Attributes
    ----------
    times:
        Time stamps, shape ``(T,)``.
    states:
        Per-entity state history, ``{name: array(T, state_dim)}``.
    controls:
        Per-entity control history, ``{name: array(T, control_dim)}``.
    reason:
        Termination reason.
    intercepted:
        ``True`` iff the engagement ended in an intercept.
    miss_distance:
        Minimum interceptor-target range achieved (m). For an intercept this is <= kill radius.
    intercept_time:
        Time of intercept (s), or ``None`` if no intercept.
    closest_approach_time:
        Time at which ``miss_distance`` occurred (s).
    interceptor / target:
        Names of the designated interceptor and target entities.
    """

    times: Array
    states: dict[str, Array]
    controls: dict[str, Array]
    reason: TerminationReason
    intercepted: bool
    miss_distance: float
    intercept_time: float | None
    closest_approach_time: float
    interceptor: str
    target: str

    @property
    def duration(self) -> float:
        return float(self.times[-1]) if len(self.times) else 0.0

    def control_effort(self, name: str) -> float:
        """Integral of squared control magnitude for an entity (proxy for energy / effort)."""
        u = self.controls[name]
        if len(self.times) < 2:
            return 0.0
        sq = np.sum(u**2, axis=1)
        return float(np.trapezoid(sq, self.times))


class Engagement:
    """Configure and run a single engagement between an interceptor and a target.

    Parameters
    ----------
    entities:
        All bodies in the scene. Must include ``interceptor`` and ``target`` by name.
    interceptor, target:
        Names of the two entities whose relative range defines intercept/miss.
    dt:
        Integration / control time step (s).
    t_max:
        Maximum simulated time (s) before declaring a timeout.
    kill_radius:
        Range (m) within which an intercept is declared.
    integrator:
        Integration strategy (defaults to :class:`RK4`).
    ground_level:
        If set, an entity whose ``y`` drops below this triggers a GROUND termination.
    stop_on_recede:
        If ``True`` (default), terminate at closest approach once range begins to increase —
        the standard terminal-homing miss-distance definition.
    """

    def __init__(
        self,
        entities: Sequence[Entity],
        *,
        interceptor: str,
        target: str,
        dt: float = 0.01,
        t_max: float = 60.0,
        kill_radius: float = 5.0,
        integrator: Integrator | None = None,
        ground_level: float | None = None,
        stop_on_recede: bool = True,
    ) -> None:
        names = [e.name for e in entities]
        if len(names) != len(set(names)):
            raise ValueError("Entity names must be unique")
        if interceptor not in names or target not in names:
            raise ValueError("interceptor and target must name entities in the scene")
        if dt <= 0 or t_max <= 0 or kill_radius <= 0:
            raise ValueError("dt, t_max, and kill_radius must be positive")

        self.entities = list(entities)
        self._by_name = {e.name: e for e in self.entities}
        self.interceptor = interceptor
        self.target = target
        self.dt = dt
        self.t_max = t_max
        self.kill_radius = kill_radius
        self.integrator = integrator or RK4()
        self.ground_level = ground_level
        self.stop_on_recede = stop_on_recede

    def _rel_vec(self) -> Array:
        pi = self._by_name[self.interceptor].position
        pt = self._by_name[self.target].position
        return pt - pi

    def _range(self) -> float:
        return float(np.linalg.norm(self._rel_vec()))

    def _world_snapshot(self) -> dict[str, Array]:
        return {e.name: e.state.copy() for e in self.entities}

    def run(self) -> EngagementResult:
        """Simulate until a termination condition is met; return the logged result."""
        times: list[float] = []
        state_log: dict[str, list[Array]] = {e.name: [] for e in self.entities}
        control_log: dict[str, list[Array]] = {e.name: [] for e in self.entities}

        n_steps = int(np.ceil(self.t_max / self.dt))
        t = 0.0
        prev_rel = self._rel_vec()
        prev_range = float(np.linalg.norm(prev_rel))
        min_range = prev_range
        min_range_time = 0.0
        reason = TerminationReason.TIMEOUT
        intercept_time: float | None = None

        for _ in range(n_steps + 1):
            world = self._world_snapshot()

            # Log current state and the control each entity applies from this state. We log the
            # *saturated* command (what the plant actually delivers) so effort metrics stay finite
            # even though guidance commands diverge as range -> 0.
            for e in self.entities:
                state_log[e.name].append(e.state.copy())
                u = e.compute_control(t, world)
                control_log[e.name].append(e.dynamics.saturate(u).copy())
            times.append(t)

            rel = self._rel_vec()
            rng = float(np.linalg.norm(rel))
            # Closest approach *within* the step just taken (tunnelling-proof for fast closing).
            seg_d = segment_min_distance(prev_rel, rel)
            if seg_d < min_range:
                min_range, min_range_time = seg_d, t

            # --- termination checks (evaluated on the just-logged state) ---
            if seg_d <= self.kill_radius:
                reason, intercept_time = TerminationReason.INTERCEPT, t
                break
            if self.ground_level is not None and any(
                e.position[1] < self.ground_level for e in self.entities
            ):
                reason = TerminationReason.GROUND
                break
            if self.stop_on_recede and rng > prev_range and t > 0.0:
                # Range has started increasing -> closest approach passed -> miss.
                reason = TerminationReason.MISS
                break
            prev_range = rng
            prev_rel = rel

            # --- advance all entities by one step (zero-order-hold control) ---
            for e in self.entities:
                u = control_log[e.name][-1]
                e.state = self.integrator.step(e.dynamics, t, e.state, u, self.dt)
            t += self.dt

        times_arr = np.asarray(times, dtype=float)
        states = {n: np.asarray(v, dtype=float) for n, v in state_log.items()}
        controls = {n: np.asarray(v, dtype=float) for n, v in control_log.items()}

        return EngagementResult(
            times=times_arr,
            states=states,
            controls=controls,
            reason=reason,
            intercepted=reason == TerminationReason.INTERCEPT,
            miss_distance=min_range,
            intercept_time=intercept_time,
            closest_approach_time=min_range_time,
            interceptor=self.interceptor,
            target=self.target,
        )
