"""Many-vs-many engagement: N interceptors defending against M threats with live re-assignment.

`MultiEngagement` orchestrates a layered/area defense: it periodically re-solves the weapon-target
assignment (Hungarian, :mod:`.assignment`), points each interceptor's guidance at its assigned
threat, integrates everyone, and detects intercepts (proximity within the step). Killed threats and
expended interceptors freeze; the engagement ends when all threats are down, all interceptors are
spent, or time runs out. Reuses the shared dynamics, guidance, and segment-distance intercept test.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from intercept.core.entities import Controller, Entity
from intercept.core.frames import segment_min_distance
from intercept.core.integrators import RK4, Integrator
from intercept.multiagent.assignment import weapon_target_assignment

Array = NDArray[np.float64]

#: Builds a guidance controller bound to a target name: ``factory(target_name) -> Controller``.
GuidanceFactory = Callable[[str], Controller]


@dataclass
class MultiEngagementResult:
    """Outcome and full multi-agent trajectory log."""

    times: Array
    tracks: dict[str, Array]  # name -> (T, state_dim) history (frozen after death)
    roles: dict[str, str]  # name -> "interceptor" | "target"
    kills: list[tuple[str, str, float]]  # (target, interceptor, time)
    kill_points: list[Array]  # interception positions
    n_targets: int
    n_interceptors: int

    @property
    def n_killed(self) -> int:
        return len(self.kills)

    @property
    def leakers(self) -> int:
        """Threats that were never intercepted."""
        return self.n_targets - self.n_killed


class MultiEngagement:
    """Configure and run an N-interceptor vs M-target engagement with live WTA re-assignment.

    Parameters
    ----------
    interceptors, targets:
        Entities (with dynamics + initial state). Interceptor controllers are set dynamically from
        the assignment; target controllers (evasion) are their own.
    guidance_factory:
        ``factory(target_name) -> Controller`` for the interceptors (e.g. ``lambda t: true_pn(t)``).
    dt, t_max, kill_radius:
        Integration step, horizon, and intercept threshold.
    reassign_every:
        Re-solve the weapon-target assignment every this many steps.
    allocator:
        Optional ``(interceptor_states, target_states, target_names, ndim) -> {i: j}`` override for
        the WTA. When ``None`` (default) the time-minimizing Hungarian assignment is used; pass a
        custom allocator (e.g. :func:`intercept.multiagent.defense.make_value_allocator`) for an
        alternative doctrine. ``target_names`` lets a stateful allocator track threats across
        replans (e.g. accumulating an impact-prediction history).
    """

    def __init__(
        self,
        interceptors: list[Entity],
        targets: list[Entity],
        guidance_factory: GuidanceFactory,
        *,
        dt: float = 0.01,
        t_max: float = 40.0,
        kill_radius: float = 20.0,
        reassign_every: int = 25,
        integrator: Integrator | None = None,
        allocator: Callable[[list[Array], list[Array], list[str], int], dict[int, int]]
        | None = None,
    ) -> None:
        names = [e.name for e in (*interceptors, *targets)]
        if len(names) != len(set(names)):
            raise ValueError("entity names must be unique")
        self.interceptors = list(interceptors)
        self.targets = list(targets)
        self.guidance_factory = guidance_factory
        self.dt = dt
        self.t_max = t_max
        self.kill_radius = kill_radius
        self.reassign_every = reassign_every
        self.integrator = integrator or RK4()
        self.allocator = allocator

    def run(self) -> MultiEngagementResult:
        ints, tgts = self.interceptors, self.targets
        alive_i = {e.name: True for e in ints}
        alive_t = {e.name: True for e in tgts}
        ctrl: dict[str, Controller] = {}  # interceptor name -> current guidance
        assigned: dict[str, str] = {}  # interceptor name -> target name
        prev_pos = {e.name: e.position.copy() for e in (*ints, *tgts)}

        times: list[float] = []
        log: dict[str, list[Array]] = {e.name: [] for e in (*ints, *tgts)}
        kills: list[tuple[str, str, float]] = []
        kill_points: list[Array] = []

        n_steps = int(np.ceil(self.t_max / self.dt))
        t = 0.0
        for step in range(n_steps + 1):
            live_i = [e for e in ints if alive_i[e.name]]
            live_t = [e for e in tgts if alive_t[e.name]]
            if not live_t or not live_i:
                # log final frame then stop
                for e in (*ints, *tgts):
                    log[e.name].append(e.state.copy())
                times.append(t)
                break

            # --- re-assign (Hungarian) periodically ---
            if step % self.reassign_every == 0:
                ndim = int(getattr(live_i[0].dynamics, "control_dim", 2))
                istates = [e.state for e in live_i]
                tstates = [e.state for e in live_t]
                if self.allocator is not None:
                    amap = self.allocator(istates, tstates, [e.name for e in live_t], ndim)
                else:
                    amap = weapon_target_assignment(istates, tstates, ndim=ndim)
                for li, interceptor in enumerate(live_i):
                    tgt_name = live_t[amap[li]].name
                    if assigned.get(interceptor.name) != tgt_name:
                        assigned[interceptor.name] = tgt_name
                        ctrl[interceptor.name] = self.guidance_factory(tgt_name)

            world = {e.name: e.state.copy() for e in (*ints, *tgts)}
            for e in (*ints, *tgts):
                log[e.name].append(e.state.copy())
            times.append(t)

            # --- controls + integrate the living ---
            for interceptor in live_i:
                u = np.asarray(ctrl[interceptor.name](t, interceptor.state, world), dtype=float)
                interceptor.state = self.integrator.step(
                    interceptor.dynamics,
                    t,
                    interceptor.state,
                    interceptor.dynamics.saturate(u),
                    self.dt,
                )
            for tgt in live_t:
                u = (
                    np.asarray(tgt.compute_control(t, world), dtype=float)
                    if tgt.controller is not None
                    else np.zeros(tgt.dynamics.control_dim)
                )
                tgt.state = self.integrator.step(
                    tgt.dynamics, t, tgt.state, tgt.dynamics.saturate(u), self.dt
                )

            # --- intercept detection (closest approach within the step) ---
            for tgt in live_t:
                if not alive_t[tgt.name]:
                    continue
                for interceptor in live_i:
                    if not alive_i[interceptor.name]:
                        continue
                    rel0 = prev_pos[tgt.name] - prev_pos[interceptor.name]
                    rel1 = tgt.position - interceptor.position
                    if segment_min_distance(rel0, rel1) <= self.kill_radius:
                        alive_t[tgt.name] = False
                        alive_i[interceptor.name] = False
                        kills.append((tgt.name, interceptor.name, t))
                        kill_points.append(0.5 * (tgt.position + interceptor.position))
                        break

            for e in (*ints, *tgts):
                prev_pos[e.name] = e.position.copy()
            t += self.dt

        roles = {**{e.name: "interceptor" for e in ints}, **{e.name: "target" for e in tgts}}
        tracks = {n: np.asarray(v, dtype=float) for n, v in log.items()}
        return MultiEngagementResult(
            times=np.asarray(times, dtype=float),
            tracks=tracks,
            roles=roles,
            kills=kills,
            kill_points=kill_points,
            n_targets=len(tgts),
            n_interceptors=len(ints),
        )
