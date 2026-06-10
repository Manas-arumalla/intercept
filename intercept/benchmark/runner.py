"""Benchmark runner: evaluate every (algorithm × scenario) cell and tabulate the metrics.

Produces a list of result rows (one per cell) that can be printed, written to CSV, and plotted.
Each algorithm is a guidance *factory* so stateful laws (e.g. APN) get a fresh instance per trial.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from intercept.benchmark.metrics import MetricSummary, summarize
from intercept.benchmark.montecarlo import GuidanceFactory, run_montecarlo
from intercept.benchmark.scenario import ParametricScenario


@dataclass
class BenchmarkRow:
    """Metrics for one (algorithm, scenario) cell."""

    algorithm: str
    scenario: str
    summary: MetricSummary

    def flat(self) -> dict[str, object]:
        return {"algorithm": self.algorithm, "scenario": self.scenario, **self.summary.as_dict()}


def run_benchmark(
    scenarios: Mapping[str, ParametricScenario],
    algorithms: Mapping[str, GuidanceFactory],
    *,
    n_trials: int = 200,
    seed: int = 0,
) -> list[BenchmarkRow]:
    """Run the full algorithm × scenario grid and return one :class:`BenchmarkRow` per cell.

    The same ``seed`` and ``n_trials`` are used for every algorithm within a scenario, so all
    algorithms see identical sampled engagements (the fairness invariant).
    """
    rows: list[BenchmarkRow] = []
    for scen_name, scenario in scenarios.items():
        for algo_name, factory in algorithms.items():
            results = run_montecarlo(scenario, factory, n_trials=n_trials, seed=seed)
            rows.append(BenchmarkRow(algo_name, scen_name, summarize(results)))
    return rows


def write_csv(rows: list[BenchmarkRow], path: str | Path) -> Path:
    """Write benchmark rows to a CSV file (creating parent directories). Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [r.flat() for r in rows]
    fieldnames = list(flat[0].keys()) if flat else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)
    return path


def format_table(rows: list[BenchmarkRow]) -> str:
    """Render a compact fixed-width results table (for console / logs)."""
    header = (
        f"{'scenario':<26}{'algorithm':<20}{'P_int':>7}{'95% CI':>16}"
        f"{'miss_med':>10}{'tti':>8}{'effort':>12}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        s = r.summary
        ci = f"[{s.p_intercept_lo:.2f},{s.p_intercept_hi:.2f}]"
        tti = "  -   " if s.tti_mean != s.tti_mean else f"{s.tti_mean:6.2f}"  # nan check
        lines.append(
            f"{r.scenario:<26}{r.algorithm:<20}{s.p_intercept:>7.2f}{ci:>16}"
            f"{s.miss_median:>10.2f}{tti:>8}{s.effort_mean:>12.0f}"
        )
    return "\n".join(lines)
