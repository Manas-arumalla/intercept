"""Realistic-engagement benchmark — does the interceptor still win when it can't win on speed?

Runs the realistic L2 scenario suite (supersonic, comparable speeds, high-g evasive/deceptive
targets, autopilot lag, gravity + induced drag) over a seeded Monte-Carlo, comparing guidance laws.
The point: simple True PN, which trivially caught the old slow straight targets, now *misses* the
maneuvering ones — and the prediction-aware laws (Augmented PN, optimal, sliding-mode) recover much
of the gap. Intelligence, not speed, decides the outcome.

Run:
    python experiments/p8_realistic_benchmark.py [--trials 200] [--no-show]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from intercept.benchmark import format_table, load_suite, run_benchmark, write_csv
from intercept.guidance import AugmentedPN, optimal_guidance, sliding_mode, true_pn
from intercept.viz import plot_pintercept_bars

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios" / "realistic"
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"
RESULTS = ROOT / "results"

ALGORITHMS = {
    "True PN (N=4)": lambda tgt: true_pn(tgt, N=4.0),
    "Augmented PN": lambda tgt: AugmentedPN(tgt, N=4.0),
    "Optimal (OGL)": lambda tgt: optimal_guidance(tgt, augment=True),
    "Sliding-mode": lambda tgt: sliding_mode(tgt, eta=300.0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Realistic-engagement benchmark")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    suite = load_suite(SCENARIOS)
    print(f"Realistic suite: {', '.join(suite)}")
    print(f"{len(ALGORITHMS)} laws x {len(suite)} scenarios x {args.trials} trials\n")

    t0 = time.perf_counter()
    rows = run_benchmark(suite, ALGORITHMS, n_trials=args.trials, seed=args.seed)
    print(format_table(rows))
    print(f"\nCompleted in {time.perf_counter() - t0:.1f} s")

    write_csv(rows, RESULTS / "p8_realistic_benchmark.csv")
    plot_pintercept_bars(rows, save_path=FIG / "p8_realistic_benchmark.png", show=show)
    print(f"Figure: {FIG / 'p8_realistic_benchmark.png'}")


if __name__ == "__main__":
    main()
