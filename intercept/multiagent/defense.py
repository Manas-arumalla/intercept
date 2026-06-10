"""Asset-value, decoy-aware layered defense — the counter to a coordinated saturation raid.

A naive area defense assigns interceptors to *tracks* (minimize time-to-intercept, one per track,
fire on everything). Against a coordinated raid (:mod:`intercept.adversary.swarm_tactics`) that is
exactly the behavior the attacker exploits: decoys soak up interceptors, and a packed saturation
point exceeds the handling capacity before the cheapest-time logic can prioritize.

This module supplies a **threat-evaluation + value-prioritized allocator** that plugs into
:class:`~intercept.multiagent.swarm.MultiEngagement` via its ``allocator`` hook. The doctrine,
following open-source layered-defense practice:

1. **Impact-point prediction.** Extrapolate each track to its closest approach to the defended
   asset (constant-velocity). The predicted **miss distance** and **time-to-asset** are the
   threat-evaluation features — *no* warhead/lethality data, purely kinematic.
2. **Decoy de-prioritization.** A track whose predicted miss exceeds ``lethal_radius`` does not
   endanger the asset; it gets near-zero value, so the magazine is not spent on it.
3. **Value-prioritized, capacity-aware assignment.** Interceptors are committed to the
   highest-value (soonest-arriving, closest-predicted-miss, catchable) threats first; with fewer
   interceptors than threats, the *least* dangerous tracks are deliberately left — the opposite of
   spreading thin across decoys.

The operating envelope: this helps precisely when the raid contains decoys or over-saturates the
magazine. Against an all-real, within-capacity raid it reduces to the same coverage as time-WTA
(measured in ``experiments/p35_swarm_penetration.py``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from intercept.multiagent.assignment import intercept_time_cost

Array = NDArray[np.float64]
_BIG = 1e9


def predict_closest_approach(
    target_state: Array, defended: Array, ndim: int = 3
) -> tuple[float, float]:
    """Constant-velocity prediction of a track's closest approach to ``defended``.

    Returns ``(time_to_closest, miss_distance)`` in (s, m). Time is clamped to ``≥ 0`` (a receding
    track has its closest approach now). This is the threat-evaluation primitive — impact-point
    prediction on the *current* kinematic state (a maneuvering threat's prediction updates each
    replan, exactly as a real tracker's would)."""
    s = np.asarray(target_state, float)
    d = np.asarray(defended, float)[:ndim]
    p = s[:ndim] - d
    v = s[ndim : 2 * ndim]
    vv = float(v @ v)
    t_star = 0.0 if vv < 1e-9 else max(0.0, -float(p @ v) / vv)
    miss = float(np.linalg.norm(p + v * t_star))
    return t_star, miss


def threat_value(
    target_state: Array,
    defended: Array,
    *,
    lethal_radius: float = 800.0,
    tau_asset: float = 12.0,
    ndim: int = 3,
    miss_override: float | None = None,
) -> float:
    """Asset-relative danger of a track in ``[0, 1]`` (1 = imminent hit, ~0 = decoy/receding).

    Combines two kinematic features: a **lethality gate** on predicted miss (a smooth shoulder at
    ``lethal_radius`` — tracks predicted to miss widely are decoys/non-threats) and an **urgency**
    factor ``exp(−t_to_asset/τ)`` (sooner arrival ⇒ more dangerous, less time to re-engage).

    ``miss_override`` substitutes a tracked miss statistic (e.g. the minimum predicted miss seen so
    far, which is robust to a maneuvering threat's instantaneous heading swings) for the snapshot
    miss — see :func:`make_value_allocator`."""
    t_star, miss = predict_closest_approach(target_state, defended, ndim)
    if miss_override is not None:
        miss = miss_override
    # Smooth lethality gate: ~1 inside lethal_radius, decaying outside (decoys -> ~0).
    gate = 1.0 / (1.0 + (miss / max(lethal_radius, 1e-6)) ** 2)
    urgency = float(np.exp(-t_star / max(tau_asset, 1e-6)))
    return float(gate * urgency)


def value_prioritized_assignment(
    interceptor_states: list[Array],
    target_states: list[Array],
    *,
    defended: Array,
    speeds: list[float] | None = None,
    lethal_radius: float = 800.0,
    tau_asset: float = 12.0,
    ndim: int = 3,
    misses: list[float | None] | None = None,
) -> dict[int, int]:
    """Allocate interceptors to the highest-value, catchable threats (capacity-aware Hungarian).

    The score of pairing interceptor ``i`` with threat ``j`` is ``threat_value(j) · catchability``,
    where catchability is ``exp(−t_intercept/τ)`` (0 if uncatchable). The one-to-one core is the
    Hungarian assignment maximizing the product of pairing scores (on ``−log(score)``); because
    low-value decoys carry near-zero score, with fewer interceptors than threats the optimizer
    **leaves the decoys/wide-miss tracks unengaged** and spends the magazine on real threats. Each
    surplus interceptor then reinforces the highest-residual-value catchable threat (diminishing
    return per existing assignment — a one-look shoot-look-shoot heuristic).

    On an all-real, within-capacity raid every track has comparable value, so this reduces to the
    same coverage as time-WTA (the gain is specific to decoys / over-saturation).
    """
    n, m = len(interceptor_states), len(target_states)
    if n == 0 or m == 0:
        return {}
    speeds = speeds or [
        float(np.linalg.norm(np.asarray(s)[ndim : 2 * ndim])) for s in interceptor_states
    ]
    misses = misses if misses is not None else [None] * m
    value = np.array(
        [
            threat_value(
                t,
                defended,
                lethal_radius=lethal_radius,
                tau_asset=tau_asset,
                ndim=ndim,
                miss_override=mo,
            )
            for t, mo in zip(target_states, misses, strict=True)
        ]
    )
    score = np.zeros((n, m))  # value × catchability (higher = better pairing)
    for i in range(n):
        for j in range(m):
            tic = intercept_time_cost(interceptor_states[i], target_states[j], speeds[i], ndim)
            score[i, j] = 0.0 if tic >= _BIG else value[j] * float(np.exp(-tic / 8.0))

    rows, cols = linear_sum_assignment(-np.log(score + 1e-9))
    assignment = {int(i): int(j) for i, j in zip(rows, cols, strict=True)}
    covered = np.zeros(m)
    for j in assignment.values():
        covered[j] += 1.0
    for i in range(n):  # surplus interceptors reinforce the leakiest real
        if i not in assignment:
            disc = score[i] * (0.35**covered)
            j = int(np.argmax(disc)) if disc.max() > 0 else int(np.argmax(value))
            assignment[i] = j
            covered[j] += 1.0
    return assignment


def make_value_allocator(defended: Array, *, lethal_radius: float = 800.0, tau_asset: float = 12.0):
    """Build a stateful ``MultiEngagement``-compatible asset-value allocator.

    Returns ``allocator(interceptor_states, target_states, target_names, ndim) -> {i: j}`` — a
    drop-in for the ``allocator`` hook. It maintains a **track history**: the minimum predicted miss
    seen for each named threat across replans. A maneuvering real threat's instantaneous heading
    swings make a single-snapshot miss unreliable (a hard weave momentarily looks like it will miss
    — indistinguishable from a near-miss decoy), but as a real threat bores in its *minimum*
    predicted miss collapses toward zero while a decoy's floors at its offset. Using that running
    minimum as the lethality feature is the robust discriminator — and it mirrors real doctrine:
    the picture is ambiguous at long range, so the defender does not confidently classify (or commit
    its full magazine) until tracks have closed enough to resolve.
    """
    best_miss: dict[str, float] = {}

    def allocator(
        interceptor_states: list[Array],
        target_states: list[Array],
        target_names: list[str],
        ndim: int = 3,
    ) -> dict[int, int]:
        misses: list[float | None] = []
        for nm, st in zip(target_names, target_states, strict=True):
            _, miss = predict_closest_approach(st, defended, ndim)
            best_miss[nm] = min(best_miss.get(nm, np.inf), miss)
            misses.append(best_miss[nm])
        return value_prioritized_assignment(
            interceptor_states,
            target_states,
            defended=defended,
            lethal_radius=lethal_radius,
            tau_asset=tau_asset,
            ndim=ndim,
            misses=misses,
        )

    return allocator
