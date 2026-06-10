"""INTERCEPT — a reproducible benchmark and simulation platform comparing classical,
optimal, game-theoretic, and learned missile-interception guidance.

Simulation-only, educational/research project. See README and ``planning/`` for scope and ethics.
"""

__version__ = "0.1.0"

from intercept.core import (  # noqa: F401
    RK4,
    Dynamics,
    Engagement,
    EngagementResult,
    Entity,
    PointMass2D,
    integrate_rk4,
)

__all__ = [
    "__version__",
    "Dynamics",
    "PointMass2D",
    "RK4",
    "integrate_rk4",
    "Entity",
    "Engagement",
    "EngagementResult",
]
