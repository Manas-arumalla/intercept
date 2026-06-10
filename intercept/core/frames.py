"""Engagement geometry: line-of-sight (LOS), range, closing speed, LOS rate, zero-effort-miss.

These are the kinematic quantities every homing guidance law is built on. Implemented in 2-D
for P0; the 3-D generalizations (LOS as a unit vector, LOS rate as ``r x v / |r|^2``) will live
here too as the platform climbs the fidelity ladder. All functions are pure and operate on the
``PointMass2D`` state layout ``[x, y, vx, vy]``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _pos(state: Array) -> Array:
    return np.asarray(state, dtype=float)[:2]


def _vel(state: Array) -> Array:
    return np.asarray(state, dtype=float)[2:4]


def relative_state(pursuer: Array, target: Array) -> tuple[Array, Array]:
    """Return relative position ``r = target_pos - pursuer_pos`` and relative velocity ``v``."""
    r = _pos(target) - _pos(pursuer)
    v = _vel(target) - _vel(pursuer)
    return r, v


def range_to(pursuer: Array, target: Array) -> float:
    """Euclidean range between pursuer and target positions (m)."""
    r, _ = relative_state(pursuer, target)
    return float(np.linalg.norm(r))


def los_angle(pursuer: Array, target: Array) -> float:
    """Line-of-sight angle ``lambda`` (rad) from pursuer to target, measured from +x axis."""
    r, _ = relative_state(pursuer, target)
    return float(np.arctan2(r[1], r[0]))


def closing_speed(pursuer: Array, target: Array) -> float:
    """Closing speed ``Vc = -d(range)/dt`` (m/s). Positive when range is decreasing."""
    r, v = relative_state(pursuer, target)
    rng = float(np.linalg.norm(r))
    if rng == 0.0:
        return 0.0
    # d(range)/dt = (r . v) / |r|;  closing speed is its negation.
    return float(-(r @ v) / rng)


def los_rate(pursuer: Array, target: Array) -> float:
    """Line-of-sight angular rate ``lambda_dot`` (rad/s).

    For planar motion, ``lambda_dot = (r x v) / |r|^2`` using the scalar 2-D cross product
    ``r_x v_y - r_y v_x``. This is the signal Proportional Navigation nulls.
    """
    r, v = relative_state(pursuer, target)
    rng2 = float(r @ r)
    if rng2 == 0.0:
        return 0.0
    cross = float(r[0] * v[1] - r[1] * v[0])
    return cross / rng2


def segment_min_distance(r0: Array, r1: Array) -> float:
    """Minimum distance from the origin to the segment ``r0 → r1``.

    Used to detect intercept *within* an integration step: the relative position moves from ``r0``
    to ``r1`` over the step, so the true closest approach is the segment's nearest point to the
    origin — not just ``min(|r0|, |r1|)``. Without this, fast closing speeds can "tunnel" a
    discrete sampler straight through a small kill radius between steps.
    """
    r0 = np.asarray(r0, dtype=float)
    r1 = np.asarray(r1, dtype=float)
    seg = r1 - r0
    ss = float(seg @ seg)
    if ss < 1e-12:
        return float(np.linalg.norm(r0))
    t = float(np.clip(-(r0 @ seg) / ss, 0.0, 1.0))
    return float(np.linalg.norm(r0 + t * seg))


def zero_effort_miss(pursuer: Array, target: Array, t_go: float) -> Array:
    """Zero-effort-miss (ZEM) vector: predicted miss if neither side accelerates for ``t_go``.

    ``ZEM = r + v * t_go`` where ``r``, ``v`` are relative position/velocity. Used by ZEM-form
    guidance laws (e.g., ``a_cmd = N * ZEM_perp / t_go^2``).
    """
    r, v = relative_state(pursuer, target)
    return r + v * float(t_go)
