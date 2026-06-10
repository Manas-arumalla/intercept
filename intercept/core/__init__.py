"""Simulation core: dynamics, integrators, engagement geometry, entities, and the sim loop.

This layer is deliberately algorithm-agnostic. Guidance laws, estimators, sensors, and
adversaries are injected as plug-ins (see the sibling subpackages) so that every paradigm
runs against *identical* dynamics — the fairness property the benchmark depends on.
"""

from intercept.core.aero import G0, AeroMissile2D
from intercept.core.dynamics import Dynamics, PointMass2D
from intercept.core.dynamics3d import AeroMissile3D, PointMass3D
from intercept.core.engagement import Engagement, EngagementResult, TerminationReason
from intercept.core.entities import Entity
from intercept.core.frames import (
    closing_speed,
    los_angle,
    los_rate,
    range_to,
    relative_state,
    segment_min_distance,
    zero_effort_miss,
)
from intercept.core.integrators import RK4, Integrator, integrate_rk4
from intercept.core.realistic import RealisticMissile2D, RealisticMissile3D

__all__ = [
    "Dynamics",
    "PointMass2D",
    "AeroMissile2D",
    "PointMass3D",
    "AeroMissile3D",
    "RealisticMissile2D",
    "RealisticMissile3D",
    "G0",
    "Integrator",
    "RK4",
    "integrate_rk4",
    "Entity",
    "Engagement",
    "EngagementResult",
    "TerminationReason",
    "range_to",
    "relative_state",
    "los_angle",
    "los_rate",
    "closing_speed",
    "segment_min_distance",
    "zero_effort_miss",
]
