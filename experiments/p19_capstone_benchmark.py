"""P19 — Capstone: one fair benchmark across paradigms, fidelity levels (L0→L3), and dimensions.

This is the project's thesis in a single figure. The *same* seeded Monte-Carlo harness and metrics
(fairness invariant, ADR-0003) evaluate the guidance laws on a maneuvering target across the whole
fidelity ladder and both dimensionalities:

* **2-D** rows at L0 (`PointMass2D`), L2 (`AeroMissile2D`), L3 (`RealisticMissile2D`) vs a hard
  random-telegraph jink, comparing True PN, Augmented PN, Optimal (OGL), and Sliding-mode.
* **3-D** rows at L2/L3 (`AeroMissile3D`/`RealisticMissile3D`) vs a barrel-roll, comparing the 3-D
  PN family (True PN-3D, Augmented PN-3D).

It also reports the **paired-bootstrap significance** of Augmented PN vs True PN per row (the
fairness invariant makes the trials paired), turning the comparison from point estimates into a
statistically grounded statement.

Run:
    python experiments/p19_capstone_benchmark.py [--trials 150] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from intercept.benchmark import (
    ParametricScenario,
    ParametricScenario3D,
    compare_intercept,
    run_montecarlo,
)
from intercept.core import G0
from intercept.guidance import (
    AugmentedPN,
    augmented_pn_3d,
    optimal_guidance,
    optimal_guidance_3d,
    sliding_mode,
    sliding_mode_3d,
    true_pn,
    true_pn_3d,
)

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"
RESULTS = ROOT / "results"

LAWS = ["True PN", "Aug PN", "OGL", "Sliding-mode"]

LAWS_2D = {
    "True PN": lambda t: true_pn(t, N=4.0),
    "Aug PN": lambda t: AugmentedPN(t, N=4.0),
    "OGL": lambda t: optimal_guidance(t, augment=True),
    "Sliding-mode": lambda t: sliding_mode(t, eta=300.0),
}
LAWS_3D = {
    "True PN": lambda t: true_pn_3d(t, N=4.0),
    "Aug PN": lambda t: augmented_pn_3d(t, N=4.0),
    "OGL": lambda t: optimal_guidance_3d(t, augment=True),
    "Sliding-mode": lambda t: sliding_mode_3d(t, eta=300.0),
}


def _row_scenarios() -> list[tuple[str, object, dict]]:
    """Each row: (label, scenario, law-factory dict). 2-D jink ladder + 3-D barrel-roll ladder."""
    jink = {"type": "telegraph", "g": 22, "mean_switch": 0.7}
    c2 = dict(
        interceptor_speed=1000.0,
        interceptor_a_max=40 * G0,
        target_speed=700.0,
        target_a_max=30 * G0,
        target_heading_deg=165.0,
        offset_min=-1200,
        offset_max=1200,
        range_min=6000,
        range_max=9000,
        dt=0.01,
        t_max=18.0,
        kill_radius=20.0,
        maneuver=jink,
    )
    barrel = {"type": "barrel_roll", "g": 20, "rate": 1.4}
    c3 = dict(
        interceptor_speed=1000.0,
        interceptor_a_max=35 * G0,
        target_speed=700.0,
        target_a_max=30 * G0,
        interceptor_tau=0.25,
        target_tau=0.3,
        target_azimuth_deg=200.0,
        target_elevation_deg=-2.0,
        range_min=7000,
        range_max=10000,
        offset_min=-1800,
        offset_max=1800,
        alt_min=2500,
        alt_max=5500,
        dt=0.01,
        t_max=26.0,
        kill_radius=20.0,
        maneuver=barrel,
    )
    return [
        ("2-D | L0 | jink", ParametricScenario(name="c_2d_l0", model="point_mass", **c2), LAWS_2D),
        ("2-D | L2 | jink", ParametricScenario(name="c_2d_l2", model="aero", **c2), LAWS_2D),
        ("2-D | L3 | jink", ParametricScenario(name="c_2d_l3", model="realistic", **c2), LAWS_2D),
        ("3-D | L2 | barrel", ParametricScenario3D(name="c_3d_l2", model="aero", **c3), LAWS_3D),
        (
            "3-D | L3 | barrel",
            ParametricScenario3D(name="c_3d_l3", model="realistic", **c3),
            LAWS_3D,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Capstone cross-paradigm/fidelity/dim benchmark")
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    rows = _row_scenarios()
    P = np.full((len(rows), len(LAWS)), np.nan)
    sig_lines = []
    for i, (label, sc, laws) in enumerate(rows):
        cache = {}
        for j, law in enumerate(LAWS):
            if law not in laws:
                continue
            res = run_montecarlo(sc, laws[law], n_trials=args.trials, seed=args.seed)
            cache[law] = res
            P[i, j] = sum(r.intercepted for r in res) / len(res)
        # Paired-bootstrap: Augmented PN vs True PN on identical trials.
        if "True PN" in cache and "Aug PN" in cache:
            cmp = compare_intercept(
                cache["True PN"], cache["Aug PN"], rng=np.random.default_rng(args.seed)
            )
            verdict = "significant" if cmp.significant else "n.s."
            sig_lines.append(
                f"  {label:20s}: APN-PN = {-cmp.diff:+.2f} "
                f"[{-cmp.ci_hi:+.2f},{-cmp.ci_lo:+.2f}] p={cmp.p_value:.3f} {verdict}"
            )

    print("=" * 72)
    print(f"CAPSTONE benchmark - P(intercept), {args.trials} trials/cell (seed {args.seed})")
    print("=" * 72)
    header = "row".ljust(22) + "".join(f"{law:>14}" for law in LAWS)
    print(header)
    for (label, _, _), prow in zip(rows, P, strict=True):
        cells = "".join(("      n/a    " if np.isnan(v) else f"{v:>14.2f}") for v in prow)
        print(label.ljust(22) + cells)
    print("\nAugmented PN vs True PN (paired bootstrap, +ve => APN better):")
    print("\n".join(sig_lines))
    print("=" * 72)

    # --- heatmap "thesis figure" ---
    fig, ax = plt.subplots(figsize=(9, 6))
    masked = np.ma.masked_invalid(P)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#3a3a3a")
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(LAWS)), LAWS)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    for i in range(len(rows)):
        for j in range(len(LAWS)):
            if not np.isnan(P[i, j]):
                ax.text(
                    j,
                    i,
                    f"{P[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if P[i, j] < 0.6 else "black",
                    fontsize=10,
                )
    ax.set_title("INTERCEPT capstone — P(intercept) across paradigm × fidelity × dimension")
    fig.colorbar(im, ax=ax, label="P(intercept)")
    fig.tight_layout()
    GALLERY.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p19_capstone_benchmark.png", dpi=150)
    print(f"Figure: {FIG / 'p19_capstone_benchmark.png'}")
    plt.show() if show else plt.close(fig)


if __name__ == "__main__":
    main()
