"""Realistic 3-D threat-trajectory library — the *kinds of paths* attacking missiles fly.

These are public, textbook **kinematic trajectory shapes** (no targeting, sensing, warhead, or
detection-evasion content — see the project scope note): a threat is launched toward a defended
point and flies a characteristic profile via shaping accelerations. They give a swarm of *visibly
different* incoming trajectories to defend against. Each factory returns a
:data:`~intercept.core.entities.Controller` ``(t, own_state, world) -> a`` producing a world-frame
acceleration (m/s²) for a 3-D point-mass/aero threat (state ``[x,y,z,vx,vy,vz,...]``).

Profiles (loosely after how cruise/ballistic/anti-ship threats are described in open sources):

* ``cruise_weave``      — level cruise at altitude with a horizontal S-weave.
* ``sea_skimming``      — very low level flight, then a terminal **pop-up and dive**.
* ``lofted_ballistic``  — a high lofted arc (climb, then a steep ballistic-like descent).
* ``terminal_spiral``   — a corkscrew that tightens/intensifies as it closes (terminal evasive).
* ``diving_jink``       — a descending run broken up by hard lateral jinks.
* ``boost_glide``       — a shallow glide that periodically S-turns (maneuvering glide phase).

All shaping is reproducible/stateless (no RNG in the loop, per ADR-0003); jink switching is a
deterministic function of time.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from intercept.core.entities import Controller

Array = NDArray[np.float64]
_EPS = 1e-9
_UP = np.array([0.0, 0.0, 1.0])


def _frame(vel: Array) -> tuple[Array, Array, Array]:
    """Forward / horizontal-lateral / vertical-ish unit triad from a velocity vector."""
    speed = float(np.linalg.norm(vel))
    fwd = vel / speed if speed > _EPS else np.array([1.0, 0.0, 0.0])
    lat = np.cross(_UP, fwd)  # horizontal, ⟂ to ground track
    nlat = float(np.linalg.norm(lat))
    lat = lat / nlat if nlat > _EPS else np.array([0.0, 1.0, 0.0])
    up = np.cross(fwd, lat)  # completes a right-handed frame
    return fwd, lat, up


def _alt_hold(state: Array, target_alt: float, kp: float = 0.5, kd: float = 1.2) -> float:
    """PD vertical acceleration (m/s²) to drive altitude toward ``target_alt``."""
    z, vz = float(state[2]), float(state[5])
    return kp * (target_alt - z) - kd * vz


def cruise_weave(
    amplitude: float = 120.0, frequency: float = 0.25, cruise_alt: float | None = None
) -> Controller:
    """Level cruise with a horizontal sinusoidal weave (amplitude in m/s²)."""

    def control(t: float, own: Array, world) -> Array:
        _, lat, _ = _frame(np.asarray(own, float)[3:6])
        a = amplitude * np.sin(2 * np.pi * frequency * t) * lat
        if cruise_alt is not None:
            a = a + _alt_hold(own, cruise_alt) * _UP
        return a

    return control


def sea_skimming(
    cruise_alt: float = 60.0,
    popup_range: float = 2500.0,
    climb_g: float = 8.0,
    defended=(0.0, 0.0, 0.0),
) -> Controller:
    """Skim at ``cruise_alt``, then **pop up and dive** within ``popup_range`` of the target."""
    d = np.asarray(defended, float)
    g = 9.80665

    def control(t: float, own: Array, world) -> Array:
        s = np.asarray(own, float)
        ground_rng = float(np.linalg.norm((s[:3] - d)[:2]))
        if ground_rng > popup_range:
            return _alt_hold(s, cruise_alt) * _UP  # skim low
        # Terminal pop-up then dive: climb hard until above a pop apex, then push over.
        pop_apex = cruise_alt + 600.0
        vert = climb_g * g if s[2] < pop_apex and s[5] >= -5.0 else -climb_g * g
        return vert * _UP

    return control


def lofted_ballistic(
    dive_g: float = 6.0, apex_frac: float = 0.45, flight_time: float = 14.0
) -> Controller:
    """A lofted arc: climb for the first ``apex_frac`` of flight, then a steep ballistic dive."""
    g = 9.80665

    def control(t: float, own: Array, world) -> Array:
        if t < apex_frac * flight_time:
            return dive_g * g * _UP  # boost/loft upward
        return -dive_g * g * _UP  # ballistic descent

    return control


def terminal_spiral(
    spin_g: float = 14.0,
    frequency: float = 0.5,
    defended=(0.0, 0.0, 0.0),
    tighten_range: float = 4000.0,
) -> Controller:
    """A corkscrew whose radius tightens (acceleration grows) as it closes on the defended point."""
    d = np.asarray(defended, float)
    g = 9.80665

    def control(t: float, own: Array, world) -> Array:
        s = np.asarray(own, float)
        _, lat, up = _frame(s[3:6])
        rng = float(np.linalg.norm(s[:3] - d))
        gain = spin_g * g * (1.0 + max(0.0, (tighten_range - rng) / tighten_range))
        w = 2 * np.pi * frequency * t
        return gain * (np.cos(w) * lat + np.sin(w) * up)  # rotating lateral accel

    return control


def diving_jink(jink_g: float = 16.0, period: float = 1.1, dive_g: float = 3.0) -> Controller:
    """A descending run broken by hard alternating lateral jinks (deterministic bang-bang)."""
    g = 9.80665

    def control(t: float, own: Array, world) -> Array:
        _, lat, _ = _frame(np.asarray(own, float)[3:6])
        sign = 1.0 if int(t / period) % 2 == 0 else -1.0
        return sign * jink_g * g * lat - dive_g * g * _UP

    return control


def boost_glide(turn_g: float = 7.0, frequency: float = 0.18, glide_g: float = 1.5) -> Controller:
    """A shallow maneuvering glide: gentle periodic S-turns with a slow descent."""
    g = 9.80665

    def control(t: float, own: Array, world) -> Array:
        _, lat, _ = _frame(np.asarray(own, float)[3:6])
        return turn_g * g * np.sin(2 * np.pi * frequency * t) * lat - glide_g * g * _UP

    return control


#: Named threat profiles, for building a diverse swarm.
THREAT_PROFILES: dict[str, Controller] = {
    "cruise-weave": cruise_weave(amplitude=130.0, frequency=0.25, cruise_alt=3500.0),
    "sea-skimming": sea_skimming(cruise_alt=80.0, popup_range=2600.0, climb_g=9.0),
    "lofted-ballistic": lofted_ballistic(dive_g=6.0, apex_frac=0.45, flight_time=14.0),
    "terminal-spiral": terminal_spiral(spin_g=14.0, frequency=0.5),
    "diving-jink": diving_jink(jink_g=16.0, period=1.1),
    "boost-glide": boost_glide(turn_g=7.0, frequency=0.18),
}
