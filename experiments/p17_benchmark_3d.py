"""P17 — 3-D Monte-Carlo benchmark: PN vs Augmented PN on genuinely 3-D evasive engagements.

Extends the centerpiece benchmark to three dimensions (ADR-0012). The dimension-agnostic runner,
Wilson-CI metrics, and effort metric are unchanged — only the scenario geometry and maneuvers are
3-D (`ParametricScenario3D`). The suite grades from a gentle crossing up to a closed-loop
*intensifying terminal spiral*, so it shows where True PN (no maneuver feed-forward) drops and
Augmented PN (target-acceleration feed-forward) recovers — the 3-D analogue of the 2-D result.

Run:
    python experiments/p17_benchmark_3d.py [--trials 200] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from intercept.benchmark import (
    ParametricScenario3D,
    format_table,
    run_benchmark,
    run_montecarlo,
    write_csv,
)
from intercept.core import G0
from intercept.guidance import (
    augmented_pn_3d,
    optimal_guidance_3d,
    sliding_mode_3d,
    true_pn_3d,
)
from intercept.viz import plot_pintercept_bars

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"
RESULTS = ROOT / "results"


def suite() -> dict[str, ParametricScenario3D]:
    """A graded 3-D suite (L2 aero). Interceptor authority is deliberately modest (35 g) so the
    hard maneuvers separate the laws rather than being trivially caught."""
    common = dict(
        model="aero",
        interceptor_speed=1000.0,
        interceptor_a_max=35 * G0,
        target_speed=700.0,
        target_a_max=30 * G0,
        interceptor_tau=0.25,
        target_tau=0.3,
        range_min=7000.0,
        range_max=10000.0,
        offset_min=-1800.0,
        offset_max=1800.0,
        alt_min=2500.0,
        alt_max=5500.0,
        dt=0.01,
        t_max=26.0,
        kill_radius=20.0,
    )
    return {
        "B1_crossing": ParametricScenario3D(
            name="B1_crossing", target_azimuth_deg=205.0, target_elevation_deg=-3.0, **common
        ),
        "B2_barrel20g": ParametricScenario3D(
            name="B2_barrel20g",
            target_azimuth_deg=200.0,
            target_elevation_deg=-2.0,
            maneuver={"type": "barrel_roll", "g": 20, "rate": 1.4},
            **common,
        ),
        "B3_serpentine": ParametricScenario3D(
            name="B3_serpentine",
            target_azimuth_deg=200.0,
            target_elevation_deg=-2.0,
            maneuver={"type": "serpentine", "g": 22, "frequency": 0.5, "tilt": 0.6},
            **common,
        ),
        "B4_spiral": ParametricScenario3D(
            name="B4_spiral",
            target_azimuth_deg=200.0,
            target_elevation_deg=-2.0,
            maneuver={
                "type": "spiral",
                "g": 10,
                "max_accel": 28 * G0,
                "trigger_range": 2500.0,
                "rate": 2.6,
            },
            **common,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="3-D Monte-Carlo benchmark")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    algorithms = {
        "True PN-3D (N=4)": lambda t: true_pn_3d(t, N=4.0),
        "Augmented PN-3D": lambda t: augmented_pn_3d(t, N=4.0),
        "Optimal-3D (OGL)": lambda t: optimal_guidance_3d(t, augment=True),
        "Sliding-mode-3D": lambda t: sliding_mode_3d(t, eta=300.0),
    }
    scenarios = suite()
    rows = run_benchmark(scenarios, algorithms, n_trials=args.trials, seed=args.seed)
    print(format_table(rows))
    write_csv(rows, RESULTS / "p17_benchmark_3d.csv")
    plot_pintercept_bars(rows, save_path=FIG / "p17_benchmark_3d.png", show=show)

    print("\nMean control effort (lower = more efficient):")
    for algo, fac in algorithms.items():
        eff = []
        for sc in scenarios.values():
            mc = run_montecarlo(sc, fac, n_trials=args.trials, seed=args.seed)
            eff.append(np.mean([r.control_effort(r.interceptor) for r in mc]))
        print(f"  {algo:18s}: {np.mean(eff):12.0f}")
    print(f"\nFigure: {FIG / 'p17_benchmark_3d.png'}")


if __name__ == "__main__":
    main()
