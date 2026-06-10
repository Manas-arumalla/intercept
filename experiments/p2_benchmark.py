"""P2 demo — the benchmark harness in action.

Runs the shared scenario suite (S1–S5) over a Monte-Carlo of randomized engagements for several
guidance laws on *identical* sampled geometries (fairness invariant), prints the metrics table,
writes a CSV, and renders the probability-of-intercept bar chart plus a capture-region heatmap.

Run:
    python experiments/p2_benchmark.py [--trials 200] [--seed 0] [--no-show]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from intercept.benchmark import (
    compute_capture_region,
    format_table,
    load_suite,
    run_benchmark,
    write_csv,
)
from intercept.guidance import AugmentedPN, pure_pn, true_pn, zem_pn
from intercept.viz import plot_capture_region, plot_pintercept_bars

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"
RESULTS = ROOT / "results"

# Guidance laws under test (factories: target name -> fresh controller per trial). All at the same
# navigation constant N=4 so the comparison isolates the *variant*, not the gain.
ALGORITHMS = {
    "Pure PN (N=4)": lambda tgt: pure_pn(tgt, N=4.0),
    "True PN (N=4)": lambda tgt: true_pn(tgt, N=4.0),
    "ZEM PN (N=4)": lambda tgt: zem_pn(tgt, N=4.0),
    "Augmented PN (N=4)": lambda tgt: AugmentedPN(tgt, N=4.0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 benchmark harness demo")
    parser.add_argument("--trials", type=int, default=200, help="Monte-Carlo trials per cell")
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
    parser.add_argument("--no-show", action="store_true", help="headless (save figures only)")
    args = parser.parse_args()
    show = not args.no_show

    suite = load_suite(SCENARIOS)
    print(f"Loaded {len(suite)} scenarios: {', '.join(suite)}")
    print(
        f"Running {len(ALGORITHMS)} algorithms x {len(suite)} scenarios "
        f"x {args.trials} trials = {len(ALGORITHMS) * len(suite) * args.trials} engagements\n"
    )

    t0 = time.perf_counter()
    rows = run_benchmark(suite, ALGORITHMS, n_trials=args.trials, seed=args.seed)
    dt = time.perf_counter() - t0

    print(format_table(rows))
    print(f"\nCompleted in {dt:.1f} s")

    csv_path = write_csv(rows, RESULTS / "p2_benchmark.csv")
    print(f"Results CSV: {csv_path}")

    # Probability-of-intercept bar chart across the suite.
    plot_pintercept_bars(rows, save_path=FIG / "p2_pintercept_by_scenario.png", show=show)

    # Capture-region heatmap: True PN on the high-offset crossing scenario.
    scen = suite["S5_high_offset_crossing"]
    region = compute_capture_region(
        scen,
        lambda tgt: true_pn(tgt, N=4.0),
        downrange=np.linspace(2000.0, 6000.0, 41),
        crossrange=np.linspace(-3000.0, 3000.0, 41),
        algorithm="True PN (N=4)",
    )
    plot_capture_region(
        region, metric="miss", save_path=FIG / "p2_capture_region_truepn_S5.png", show=show
    )
    print(f"Capture fraction (True PN, S5): {region.capture_fraction * 100:.0f}%")
    print(f"Figures saved to: {GALLERY}")


if __name__ == "__main__":
    main()
