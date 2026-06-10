"""Sensor models (truth -> measurement) with seeded noise.

``radar`` (range + bearing, with an inverse for measurement-to-state) and ``ir_eo`` (angles-only)
share a common ``Sensor`` interface ``measure(true_state, rng) -> Measurement``. Each sensor takes
an explicit ``numpy.random.Generator`` so Monte-Carlo runs stay reproducible.
"""

from intercept.sensors.base import Sensor, wrap_to_pi
from intercept.sensors.ir_eo import IRSeeker
from intercept.sensors.radar import Radar
from intercept.sensors.radar3d import Radar3D

__all__ = ["Sensor", "wrap_to_pi", "Radar", "Radar3D", "IRSeeker"]
