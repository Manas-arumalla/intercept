"""Weapon-Target Assignment (WTA) — which interceptor engages which threat.

Assigning N interceptors to M threats to minimize total engagement cost (or maximize kill
probability) is the classic resource-allocation problem in layered defense. Here we use the
**Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) — optimal in O((N+M)³), and
re-solvable cheaply each replan as the engagement evolves and threats are killed.

Two objectives are supported:

* ``"time"`` (default) — minimize total predicted time-to-intercept.
* ``"kill_prob"`` — maximize the **global kill probability**: assign on a geometry-based
  kill-probability model (sooner / catchable intercepts score higher), and direct **surplus**
  interceptors greedily to the threats most likely to *leak* (shoot-look-shoot / redundant
  coverage), minimizing the expected number of surviving threats. The kill model is a documented
  heuristic (``p_max·exp(−t_intercept/τ)``), not real lethality data (simulation-only scope).

References: Kuhn (Hungarian method); operations-research WTA canon.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from intercept.guidance.game import intercept_point

Array = NDArray[np.float64]
_BIG = 1e9


def intercept_time_cost(
    interceptor_state: Array, target_state: Array, interceptor_speed: float, ndim: int = 2
) -> float:
    """Predicted time-to-intercept (s) of a constant-velocity target; ``_BIG`` if uncatchable.

    ``ndim`` selects the spatial dimension (2 for planar, 3 for 3-D): position is ``state[:ndim]``
    and velocity ``state[ndim:2·ndim]``."""
    p = np.asarray(interceptor_state, dtype=float)[:ndim]
    e = np.asarray(target_state, dtype=float)[:ndim]
    v = np.asarray(target_state, dtype=float)[ndim : 2 * ndim]
    pt = intercept_point(p, e, v, interceptor_speed)
    if pt is None:
        return _BIG
    return float(np.linalg.norm(pt - p) / max(interceptor_speed, 1e-6))


def cost_matrix(
    interceptor_states: list[Array],
    target_states: list[Array],
    speeds: list[float] | None = None,
    ndim: int = 2,
) -> Array:
    """Build the ``(N × M)`` predicted-time-to-intercept cost matrix."""
    n, m = len(interceptor_states), len(target_states)
    speeds = speeds or [
        float(np.linalg.norm(np.asarray(s)[ndim : 2 * ndim])) for s in interceptor_states
    ]
    C = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            C[i, j] = intercept_time_cost(interceptor_states[i], target_states[j], speeds[i], ndim)
    return C


def kill_probability(
    interceptor_state: Array,
    target_state: Array,
    interceptor_speed: float,
    *,
    tau: float = 8.0,
    p_max: float = 0.95,
    ndim: int = 2,
) -> float:
    """Geometry-based single-shot kill probability in ``[0, 1]`` (heuristic, not real data).

    ``p_max·exp(−t_intercept/τ)`` for a catchable target (sooner / favorable geometry ⇒ higher),
    and 0 if the target is uncatchable. ``τ`` is the characteristic engagement time over which kill
    likelihood decays (longer flyouts accumulate evasion / uncertainty)."""
    t = intercept_time_cost(interceptor_state, target_state, interceptor_speed, ndim)
    if t >= _BIG:
        return 0.0
    return float(p_max * np.exp(-t / tau))


def kill_probability_matrix(
    interceptor_states: list[Array],
    target_states: list[Array],
    speeds: list[float] | None = None,
    *,
    tau: float = 8.0,
    p_max: float = 0.95,
    ndim: int = 2,
) -> Array:
    """Build the ``(N × M)`` single-shot kill-probability matrix."""
    n, m = len(interceptor_states), len(target_states)
    speeds = speeds or [
        float(np.linalg.norm(np.asarray(s)[ndim : 2 * ndim])) for s in interceptor_states
    ]
    P = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            P[i, j] = kill_probability(
                interceptor_states[i], target_states[j], speeds[i], tau=tau, p_max=p_max, ndim=ndim
            )
    return P


def expected_leakers(assignment: dict[int, int], kill_prob: Array) -> float:
    """Expected number of surviving threats given an assignment and a kill-probability matrix.

    A threat ``j`` survives all interceptors assigned to it with probability ``∏(1 − P[i, j])``;
    unassigned threats survive with probability 1. The sum over threats is the expected leakers."""
    m = kill_prob.shape[1]
    survival = np.ones(m)
    for i, j in assignment.items():
        survival[j] *= 1.0 - kill_prob[i, j]
    return float(survival.sum())


def weapon_target_assignment(
    interceptor_states: list[Array],
    target_states: list[Array],
    speeds: list[float] | None = None,
    *,
    objective: str = "time",
    tau: float = 8.0,
    p_max: float = 0.95,
    ndim: int = 2,
) -> dict[int, int]:
    """Assign each interceptor to a target (interceptor index -> target index).

    ``objective="time"`` (default) minimizes total time-to-intercept via the Hungarian algorithm;
    surplus interceptors get redundant coverage of their individually-cheapest target.

    ``objective="kill_prob"`` maximizes the **global kill probability**: the one-to-one core is the
    Hungarian solution on ``−log(P_kill)`` (maximizing the product of assigned kill probabilities),
    and each **surplus** interceptor is then placed greedily on the threat where it most reduces the
    **expected leakers** (``survivalⱼ · P[i, j]``, shoot-look-shoot). Threats beyond the interceptor
    count are left for the next replan.
    """
    n, m = len(interceptor_states), len(target_states)
    if n == 0 or m == 0:
        return {}
    if objective not in ("time", "kill_prob"):
        raise ValueError("objective must be 'time' or 'kill_prob'")

    if objective == "time":
        C = cost_matrix(interceptor_states, target_states, speeds, ndim)
        rows, cols = linear_sum_assignment(C)
        assignment = {int(i): int(j) for i, j in zip(rows, cols, strict=True)}
        for i in range(n):  # surplus → cheapest target (redundant coverage)
            if i not in assignment:
                assignment[i] = int(np.argmin(C[i]))
        return assignment

    # objective == "kill_prob": maximize product of assigned P_kill, then minimize expected leakers.
    P = kill_probability_matrix(
        interceptor_states, target_states, speeds, tau=tau, p_max=p_max, ndim=ndim
    )
    rows, cols = linear_sum_assignment(-np.log(P + 1e-9))
    assignment = {int(i): int(j) for i, j in zip(rows, cols, strict=True)}
    survival = np.ones(m)
    for i, j in assignment.items():
        survival[j] *= 1.0 - P[i, j]
    for i in range(n):  # surplus → threat most likely to leak
        if i not in assignment:
            j = int(np.argmax(survival * P[i]))
            assignment[i] = j
            survival[j] *= 1.0 - P[i, j]
    return assignment
