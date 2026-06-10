"""Coordinated swarm-penetration tactics — *how* a raid is organized to defeat a defense.

The threat library (:mod:`.threats`) gives the *shapes* individual missiles fly. This module adds
the **coordination layer**: the public, textbook ways a salvo is organized to get past a layered
defense, expressed as **kinematic raid geometries** (launch ranges, axes, arrival timing, and
decoys). No targeting, sensing, warhead, or detection-evasion content — purely the engagement
geometry a defender's allocation logic must cope with (project scope note).

The four tactics modeled (after open-source descriptions of saturation / salvo doctrine):

* ``simultaneous_tot``  — **time-on-target**: every real threat is timed to arrive in a tight
  window, so the defender cannot engage them sequentially (no time to look-shoot-look).
* ``decoy_screen``      — real threats mixed with **decoys** that are kinematically similar early
  but are *not* aimed at the asset; a defender that engages every track wastes its magazine.
* ``concentrated_axis`` — a **saturation point**: threats packed into one narrow azimuth sector to
  exceed the local target-handling capacity of the defense.
* ``stream_raid``       — **sequential waves**: spaced arrivals that bait and then exhaust the
  magazine (exploiting shoot-look-shoot cycle time).

Each builder returns a :class:`Raid` (the threat entities, which names are decoys, and the defended
point). Decoys are genuinely distinguishable: their velocity is aimed to *miss* the asset by a wide
margin, so an impact-point predictor (see :mod:`intercept.multiagent.defense`) can de-prioritize
them — that is the discriminator the counter exploits, and the benchmark measures whether it helps.

References (public): Wikipedia *Saturation attack* / *Swarming (military)*; cooperative-salvo and
decoy-screen guidance literature (e.g. simultaneous-arrival salvo studies; naval decoy-clustering
against salvo threats). All numbers here are illustrative simulation parameters, not real systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from intercept.adversary.threats import cruise_weave, diving_jink, terminal_spiral
from intercept.core import PointMass3D
from intercept.core.entities import Controller, Entity

Array = NDArray[np.float64]


@dataclass
class Raid:
    """A coordinated raid: threat entities plus the metadata a defender benchmark needs."""

    threats: list[Entity]
    decoys: set[str] = field(default_factory=set)
    defended: Array = field(default_factory=lambda: np.zeros(3))
    tactic: str = ""

    @property
    def real_threats(self) -> list[Entity]:
        """Entities that actually endanger the defended point (decoys excluded)."""
        return [e for e in self.threats if e.name not in self.decoys]


def _aimed_velocity(pos: Array, aim: Array, speed: float) -> Array:
    """Unit velocity of magnitude ``speed`` pointing from ``pos`` toward ``aim``."""
    d = np.asarray(aim, float) - np.asarray(pos, float)
    n = float(np.linalg.norm(d))
    return speed * d / n if n > 1e-9 else np.array([speed, 0.0, 0.0])


def _spawn(name: str, pos: Array, vel: Array, ctrl: Controller | None, amax: float) -> Entity:
    state = np.array([*pos, *vel], dtype=float)
    return Entity(name, PointMass3D(a_max=amax), state, controller=ctrl, role="target")


def _profile(kind: str, defended: Array) -> Controller:
    """A representative maneuvering profile for a real threat (kinematic, deterministic)."""
    if kind == "spiral":
        return terminal_spiral(spin_g=12.0, frequency=0.45, defended=tuple(defended))
    if kind == "jink":
        return diving_jink(jink_g=14.0, period=1.2, dive_g=2.0)
    return cruise_weave(amplitude=130.0, frequency=0.18, cruise_alt=float(defended[2] + 3000.0))


def _decoy_ctrl(amplitude: float = 60.0) -> Controller:
    """A decoy's mild weave (looks like a maneuvering threat; its *heading* gives it away)."""
    return cruise_weave(amplitude=amplitude, frequency=0.2)


# --------------------------------------------------------------------------------------------------
# Tactics. Each returns a Raid; all share realistic comparable speeds (threat ~Mach 2).
# --------------------------------------------------------------------------------------------------
def simultaneous_tot(
    n_real: int = 8,
    *,
    speed: float = 700.0,
    rng_m: float = 11000.0,
    az_span: float = 110.0,
    amax: float = 30 * 9.80665,
    defended: Array | None = None,
) -> Raid:
    """**Time-on-target**: equal launch range + speed ⇒ all real threats arrive together.

    A simultaneous arrival denies the defender the time to engage threats one-by-one; every track
    must be handled in parallel or it leaks."""
    d = np.zeros(3) if defended is None else np.asarray(defended, float)
    azis = np.radians(np.linspace(-az_span / 2, az_span / 2, n_real) + 90.0)
    threats = []
    for j, az in enumerate(azis):
        alt = d[2] + 2500.0 + 400.0 * (j % 3)
        pos = d + np.array([rng_m * np.cos(az), rng_m * np.sin(az), alt - d[2]])
        vel = _aimed_velocity(pos, d, speed)
        kind = ("weave", "jink", "spiral")[j % 3]
        threats.append(_spawn(f"R{j}:{kind}", pos, vel, _profile(kind, d), amax))
    return Raid(threats=threats, decoys=set(), defended=d, tactic="simultaneous-TOT")


def decoy_screen(
    n_real: int = 5,
    n_decoy: int = 7,
    *,
    speed: float = 700.0,
    rng_m: float = 11000.0,
    az_span: float = 120.0,
    decoy_miss: float = 2500.0,
    amax: float = 30 * 9.80665,
    defended: Array | None = None,
) -> Raid:
    """**Decoy screen**: ``n_real`` threats aimed at the asset, ``n_decoy`` decoys aimed to miss.

    Decoys interleave with real threats and fly similar weaves, but their velocity is aimed to pass
    the asset by ``decoy_miss`` metres — so a defender that engages *every* track squanders its
    magazine on chaff, while an impact-point predictor can ignore them."""
    d = np.zeros(3) if defended is None else np.asarray(defended, float)
    n = n_real + n_decoy
    azis = np.radians(np.linspace(-az_span / 2, az_span / 2, n) + 90.0)
    # Interleave decoys among real threats so they cannot be separated by position alone.
    is_decoy = np.zeros(n, dtype=bool)
    is_decoy[np.linspace(0, n - 1, n_decoy).round().astype(int)] = True
    threats, decoys = [], set()
    for j, az in enumerate(azis):
        alt = d[2] + 2500.0 + 350.0 * (j % 4)
        pos = d + np.array([rng_m * np.cos(az), rng_m * np.sin(az), alt - d[2]])
        if is_decoy[j]:
            # Aim past the asset: offset the aim-point laterally by decoy_miss.
            perp = np.array([-np.sin(az), np.cos(az), 0.0])
            aim = d + perp * decoy_miss * (1.0 if j % 2 else -1.0)
            vel = _aimed_velocity(pos, aim, speed)
            nm = f"D{j}:decoy"
            threats.append(_spawn(nm, pos, vel, _decoy_ctrl(), amax))
            decoys.add(nm)
        else:
            vel = _aimed_velocity(pos, d, speed)
            kind = ("weave", "jink", "spiral")[j % 3]
            threats.append(_spawn(f"R{j}:{kind}", pos, vel, _profile(kind, d), amax))
    return Raid(threats=threats, decoys=decoys, defended=d, tactic="decoy-screen")


def concentrated_axis(
    n_real: int = 10,
    *,
    speed: float = 700.0,
    rng_m: float = 11000.0,
    az_span: float = 30.0,
    amax: float = 30 * 9.80665,
    defended: Array | None = None,
) -> Raid:
    """**Saturation point**: threats packed into a narrow ``az_span`` sector from one bearing.

    Concentrating the raid on one axis aims to exceed the *local* handling capacity of the defense
    — more simultaneous threats on one bearing than interceptors that can be brought to bear."""
    d = np.zeros(3) if defended is None else np.asarray(defended, float)
    azis = np.radians(np.linspace(-az_span / 2, az_span / 2, n_real) + 90.0)
    threats = []
    for j, az in enumerate(azis):
        alt = d[2] + 2500.0 + 300.0 * (j % 5)
        pos = d + np.array([rng_m * np.cos(az), rng_m * np.sin(az), alt - d[2]])
        vel = _aimed_velocity(pos, d, speed)
        kind = ("weave", "jink", "spiral")[j % 3]
        threats.append(_spawn(f"R{j}:{kind}", pos, vel, _profile(kind, d), amax))
    return Raid(threats=threats, decoys=set(), defended=d, tactic="concentrated-axis")


def stream_raid(
    n_real: int = 9,
    *,
    waves: int = 3,
    speed: float = 700.0,
    rng_lo: float = 9000.0,
    wave_gap: float = 3500.0,
    az_span: float = 100.0,
    amax: float = 30 * 9.80665,
    defended: Array | None = None,
) -> Raid:
    """**Sequential waves**: ``waves`` echelons spaced by ``wave_gap`` in range (staggered arrival).

    A stream bait-and-exhausts a magazine: each wave draws interceptors, and later waves arrive
    after the first salvo is committed — exploiting shoot-look-shoot cycle time / magazine drain.
    """
    d = np.zeros(3) if defended is None else np.asarray(defended, float)
    per = int(np.ceil(n_real / waves))
    threats, j = [], 0
    for w in range(waves):
        k = min(per, n_real - j)
        if k <= 0:
            break
        azis = np.radians(np.linspace(-az_span / 2, az_span / 2, k) + 90.0)
        for az in azis:
            rng_m = rng_lo + w * wave_gap
            alt = d[2] + 2500.0 + 300.0 * (j % 4)
            pos = d + np.array([rng_m * np.cos(az), rng_m * np.sin(az), alt - d[2]])
            vel = _aimed_velocity(pos, d, speed)
            kind = ("weave", "jink", "spiral")[j % 3]
            threats.append(_spawn(f"R{j}:w{w}:{kind}", pos, vel, _profile(kind, d), amax))
            j += 1
    return Raid(threats=threats, decoys=set(), defended=d, tactic="stream-raid")


#: Named raid builders, for sweeping the tactics in a benchmark.
SWARM_TACTICS = {
    "simultaneous-TOT": simultaneous_tot,
    "decoy-screen": decoy_screen,
    "concentrated-axis": concentrated_axis,
    "stream-raid": stream_raid,
}
