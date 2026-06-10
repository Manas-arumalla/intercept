"""Animate realistic L2 engagements — fast, supersonic, high-g evasive targets.

Generates, in gallery/:
  * anim_realistic_jink.gif / .png   — True PN vs. an Augmented-PN interceptor against a 25 g
    random-telegraph jinking target (watch PN lag the jinks while APN tracks them)
  * anim_realistic_reactive.gif/.png — interceptor vs. a target that breaks hard when it closes

Run:
    python experiments/animate_realistic.py [--show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from intercept.benchmark import load_suite
from intercept.guidance import AugmentedPN, true_pn
from intercept.viz import animate_comparison, animate_engagement, filmstrip_engagement

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"
SCENARIOS = ROOT / "scenarios" / "realistic"


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate realistic engagements")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    show = args.show

    suite = load_suite(SCENARIOS)

    # Telegraph jink: True PN vs Augmented PN on the SAME sampled geometry/target (seed fixed).
    print("1/2  realistic jink: PN vs APN ...")
    jink = suite["R3_telegraph_jink"]
    results = {}
    for label, factory in {
        "True PN": lambda t: true_pn(t, N=4.0),
        "Augmented PN": lambda t: AugmentedPN(t, N=4.0),
    }.items():
        spec = jink.sample(np.random.default_rng(11))  # same seed => identical target track
        results[label] = spec.build(factory).run()
    animate_comparison(
        results,
        title="Realistic 25 g jink — True PN vs. Augmented PN",
        save_path=ANIM / "anim_realistic_jink.gif",
        show=show,
        tail=40,
    )
    filmstrip_engagement(results["Augmented PN"], save_path=FIG / "anim_realistic_jink.png")
    for label, r in results.items():
        print(f"    {label:14s}: {r.reason.name:9s} miss={r.miss_distance:6.1f} m")

    # Reactive break.
    print("2/2  reactive break ...")
    spec = suite["R4_reactive_break"].sample(np.random.default_rng(5))
    res = spec.build(lambda t: AugmentedPN(t, N=4.0)).run()
    animate_engagement(
        res,
        title="Realistic engagement — target breaks hard at close range",
        save_path=ANIM / "anim_realistic_reactive.gif",
        show=show,
        tail=40,
    )
    filmstrip_engagement(res, save_path=FIG / "anim_realistic_reactive.png")
    print(f"    reactive: {res.reason.name} miss={res.miss_distance:.1f} m")
    print(f"\nDone. Animations in: {GALLERY}")


if __name__ == "__main__":
    main()
