"""Game-theoretic / geometric pursuit guidance (pursuit-evasion differential games).

For two constant-speed players the **Apollonius circle** is the locus of points the evader and a
faster pursuer reach at the same time: ``|X − E| / |X − P| = α``, with speed ratio ``α = v_E/v_P <
1``. Its interior is the evader's dominance region; the pursuer can guarantee capture and the
optimal play (simple motion) is a constant-bearing course toward the predicted capture point.
References: Isaacs, *Differential Games*; Weintraub, Pachter & Garcia (ACC 2020); Dorothy et al.,
"One Apollonius Circle is Enough…", *Automatica* 2024.

This module provides the circle geometry (for analysis/visualization) and `ApolloniusGuidance`, a
pursuer that each step predicts the evader's straight-line motion, solves the intercept triangle for
its own speed, and steers (constant-bearing) toward that capture point — geometrically optimal
against a non-maneuvering evader and re-planned each step against a maneuvering one.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from intercept.guidance.base import EPS, GuidanceLaw

Array = NDArray[np.float64]


def apollonius_circle(
    pursuer_pos: Array, evader_pos: Array, speed_ratio: float
) -> tuple[Array, float]:
    """Apollonius circle for ``|X−E|/|X−P| = speed_ratio`` (``α = v_E/v_P``, ``0 < α < 1``).

    Returns ``(center, radius)``. The circle's interior is the evader's dominance region (points it
    reaches before the pursuer). Requires a faster pursuer (``α < 1``).
    """
    if not (0.0 < speed_ratio < 1.0):
        raise ValueError("speed_ratio must be in (0, 1) — pursuer must be faster")
    p = np.asarray(pursuer_pos, dtype=float)[:2]
    e = np.asarray(evader_pos, dtype=float)[:2]
    k2 = speed_ratio**2
    center = (e - k2 * p) / (1.0 - k2)
    radius2 = float(center @ center) - (float(e @ e) - k2 * float(p @ p)) / (1.0 - k2)
    return center, float(np.sqrt(max(radius2, 0.0)))


def intercept_point(
    pursuer_pos: Array, evader_pos: Array, evader_vel: Array, pursuer_speed: float
) -> Array | None:
    """Soonest constant-velocity intercept point of the evader by a pursuer of given speed.

    Solves ``|E + V·t − P| = v_P·t`` for the smallest positive ``t``; returns ``E + V·t`` (or
    ``None`` if no positive solution exists — the pursuer cannot catch a straight-running evader).
    """
    p = np.asarray(pursuer_pos, dtype=float)  # dimension-generic (2-D or 3-D positions)
    e = np.asarray(evader_pos, dtype=float)
    v = np.asarray(evader_vel, dtype=float)
    d = e - p
    a = float(v @ v) - pursuer_speed**2
    b = 2.0 * float(d @ v)
    c = float(d @ d)
    if abs(a) < EPS:  # speeds equal: linear equation b t + c = 0
        if abs(b) < EPS:
            return None
        t = -c / b
        return e + v * t if t > 0 else None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    roots = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]
    positive = sorted(t for t in roots if t > EPS)
    if not positive:
        return None
    return e + v * positive[0]


class ApolloniusGuidance(GuidanceLaw):
    """Geometric pursuit: steer (constant-bearing) toward the predicted intercept point.

    Parameters
    ----------
    target:
        Name of the target/evader entity.
    gain:
        Heading-correction gain for steering the velocity toward the intercept point.
    """

    def __init__(self, target: str, gain: float = 6.0) -> None:
        super().__init__(target)
        self.gain = float(gain)

    def command(self, t: float, own_state: Array, target_state: Array) -> Array:
        p = own_state[:2]
        v = own_state[2:4]
        speed = float(np.linalg.norm(v))
        if speed < EPS:
            return np.zeros(2)
        aim_pt = intercept_point(p, target_state[:2], target_state[2:4], speed)
        if aim_pt is None:
            aim_pt = target_state[:2]  # cannot lead: fall back to pure pursuit of current position
        los = aim_pt - p
        desired = np.arctan2(los[1], los[0])
        current = np.arctan2(v[1], v[0])
        err = (desired - current + np.pi) % (2 * np.pi) - np.pi
        a_lat = self.gain * speed * err
        v_hat = v / speed
        return a_lat * np.array([-v_hat[1], v_hat[0]])
