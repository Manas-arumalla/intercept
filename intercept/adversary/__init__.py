"""Adversarial targets / evaders.

``scripted`` (straight, weave, step, bang-bang), ``evasive`` / ``evasive3d`` (hard turn, telegraph
jink, reactive break, barrel-roll, serpentine, terminal spiral), ``optimal_evader`` (anti-LOS
game-theoretic evasion), ``threats`` (realistic 3-D trajectory profiles), and ``swarm_tactics``
(coordinated penetration raids). All conform to the :data:`intercept.core.entities.Controller`
signature.
"""

from intercept.adversary.evasive import hard_turn, random_telegraph, reactive_break, surprise_break
from intercept.adversary.evasive3d import (
    barrel_roll,
    combine,
    serpentine3d,
    terminal_spiral,
    weave3d,
)
from intercept.adversary.optimal_evader import optimal_evader
from intercept.adversary.scripted import bang_bang, step_maneuver, straight, weave
from intercept.adversary.swarm_tactics import (
    SWARM_TACTICS,
    Raid,
    concentrated_axis,
    decoy_screen,
    simultaneous_tot,
    stream_raid,
)
from intercept.adversary.threats import THREAT_PROFILES

__all__ = [
    "THREAT_PROFILES",
    "SWARM_TACTICS",
    "Raid",
    "simultaneous_tot",
    "decoy_screen",
    "concentrated_axis",
    "stream_raid",
    "straight",
    "weave",
    "step_maneuver",
    "bang_bang",
    "optimal_evader",
    "hard_turn",
    "random_telegraph",
    "reactive_break",
    "surprise_break",
    "barrel_roll",
    "weave3d",
    "serpentine3d",
    "terminal_spiral",
    "combine",
]
