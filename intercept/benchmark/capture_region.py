"""Capture-region analysis.

Sweeps the target's initial position over a grid and records, for each starting point, whether the
interceptor (under a fixed guidance law and a deterministic target maneuver) achieves an intercept,
and the resulting miss distance. The set of starting points that yield an intercept is the
guidance law's **capture region** for that engagement — a standard, visually compelling way to
compare laws (a larger region = more robust).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from intercept.benchmark.montecarlo import GuidanceFactory
from intercept.benchmark.scenario import ParametricScenario

Array = NDArray[np.float64]


@dataclass
class CaptureRegion:
    """Result of a capture-region sweep over a target-position grid.

    ``intercepted`` and ``miss_distance`` are shaped ``(n_cross, n_down)`` so they can be passed
    straight to ``pcolormesh(downrange, crossrange, ...)``.
    """

    downrange: Array
    crossrange: Array
    intercepted: NDArray[np.bool_]
    miss_distance: Array
    scenario_name: str
    algorithm: str

    @property
    def capture_fraction(self) -> float:
        """Fraction of grid cells that resulted in an intercept."""
        return float(self.intercepted.mean())


def compute_capture_region(
    scenario: ParametricScenario,
    guidance_factory: GuidanceFactory,
    *,
    downrange: Array,
    crossrange: Array,
    algorithm: str = "guidance",
) -> CaptureRegion:
    """Run one deterministic engagement per target start (downrange, crossrange); log outcomes."""
    down = np.asarray(downrange, dtype=float)
    cross = np.asarray(crossrange, dtype=float)
    hit = np.zeros((cross.size, down.size), dtype=bool)
    miss = np.zeros((cross.size, down.size), dtype=float)
    for j, cr in enumerate(cross):
        for i, dr in enumerate(down):
            result = scenario.at(dr, cr).build(guidance_factory).run()
            hit[j, i] = result.intercepted
            miss[j, i] = result.miss_distance
    return CaptureRegion(down, cross, hit, miss, scenario.name, algorithm)
