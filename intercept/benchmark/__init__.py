"""Benchmark harness: scenario suite, Monte-Carlo runner, metrics, capture-region sweeps.

The centerpiece of the project. ``scenario`` defines parametric engagement distributions
(YAML-(de)serialized), ``montecarlo`` runs seeded randomized-initial-condition sweeps, ``metrics``
computes miss distance, P(intercept) with Wilson intervals, time-to-intercept, and control effort,
and ``runner`` evaluates the algorithm x scenario grid into result tables and figures. The fairness
invariant holds throughout: all algorithms share dynamics, noise, and per-trial RNG streams.
"""

from intercept.benchmark.capture_region import CaptureRegion, compute_capture_region
from intercept.benchmark.league import bradley_terry, bradley_terry_bootstrap, elo_expected_score
from intercept.benchmark.metrics import (
    MetricSummary,
    PairedComparison,
    compare_intercept,
    paired_bootstrap,
    summarize,
    wilson_interval,
)
from intercept.benchmark.montecarlo import run_montecarlo
from intercept.benchmark.runner import (
    BenchmarkRow,
    format_table,
    run_benchmark,
    write_csv,
)
from intercept.benchmark.scenario import (
    EngagementSpec,
    ParametricScenario,
    load_scenario,
    load_suite,
)
from intercept.benchmark.scenario3d import (
    ParametricScenario3D,
    load_scenario_3d,
    make_maneuver_3d,
)

__all__ = [
    "ParametricScenario",
    "ParametricScenario3D",
    "EngagementSpec",
    "load_scenario",
    "load_suite",
    "load_scenario_3d",
    "make_maneuver_3d",
    "run_montecarlo",
    "summarize",
    "wilson_interval",
    "MetricSummary",
    "paired_bootstrap",
    "compare_intercept",
    "PairedComparison",
    "run_benchmark",
    "BenchmarkRow",
    "write_csv",
    "format_table",
    "compute_capture_region",
    "CaptureRegion",
    "bradley_terry",
    "bradley_terry_bootstrap",
    "elo_expected_score",
]
