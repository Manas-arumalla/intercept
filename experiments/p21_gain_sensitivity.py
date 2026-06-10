"""P21 — gain-sensitivity sweep: how the navigation constant N trades capture vs. effort.

The benchmark elsewhere uses "best-effort" gains (N=4); this closes that documented gap by sweeping
the navigation constant for True PN and Augmented PN across realistic (L2 aero) evasive scenarios,
reporting P(intercept) (Wilson CI) and mean control effort vs. N. It quantifies the classic PN
trade-off: low N under-responds (misses the maneuver), high N over-responds (chatters, burns
effort), with a broad robust plateau between — so the benchmark's N=4 choice is justified.

Run:
    python experiments/p21_gain_sensitivity.py [--trials 150] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from intercept.benchmark import ParametricScenario, run_montecarlo, wilson_interval
from intercept.core import G0
from intercept.guidance import AugmentedPN, true_pn

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"
RESULTS = ROOT / "results"

N_VALUES = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def scenarios() -> dict[str, ParametricScenario]:
    common = dict(
        model="aero",
        interceptor_speed=1000.0,
        interceptor_a_max=40 * G0,
        target_speed=700.0,
        target_a_max=25 * G0,
        interceptor_tau=0.2,
        target_tau=0.3,
        target_heading_deg=165.0,
        offset_min=-1200,
        offset_max=1200,
        range_min=6000,
        range_max=9000,
        dt=0.01,
        t_max=18.0,
        kill_radius=20.0,
    )
    return {
        "weave_18g": ParametricScenario(
            name="weave_18g", maneuver={"type": "weave", "g": 18, "frequency": 0.35}, **common
        ),
        "jink_22g": ParametricScenario(
            name="jink_22g", maneuver={"type": "telegraph", "g": 22, "mean_switch": 0.7}, **common
        ),
    }


def laws():
    return {
        "True PN": lambda N: lambda t: true_pn(t, N=N),
        "Augmented PN": lambda N: lambda t: AugmentedPN(t, N=N),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Navigation-constant gain-sensitivity sweep")
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    scns = scenarios()
    law_facs = laws()
    # results[(law, scenario)] = list over N of (p_intercept, lo, hi, effort_mean)
    results: dict[tuple[str, str], list[tuple[float, float, float, float]]] = {}
    rows_csv = [["law", "scenario", "N", "p_intercept", "ci_lo", "ci_hi", "effort_mean"]]

    for law_name, make in law_facs.items():
        for scn_name, scn in scns.items():
            series = []
            for N in N_VALUES:
                res = run_montecarlo(scn, make(N), n_trials=args.trials, seed=args.seed)
                hits = sum(r.intercepted for r in res)
                p = hits / len(res)
                lo, hi = wilson_interval(hits, len(res))
                eff = float(np.mean([r.control_effort(r.interceptor) for r in res]))
                series.append((p, lo, hi, eff))
                rows_csv.append(
                    [
                        law_name,
                        scn_name,
                        f"{N:.0f}",
                        f"{p:.3f}",
                        f"{lo:.3f}",
                        f"{hi:.3f}",
                        f"{eff:.0f}",
                    ]
                )
            results[(law_name, scn_name)] = series

    # --- print table ---
    print("=" * 64)
    print(f"GAIN-SENSITIVITY SWEEP - P(intercept) by N ({args.trials} trials/cell)")
    print("=" * 64)
    print("law / scenario".ljust(28) + "".join(f"{'N=' + str(int(n)):>9}" for n in N_VALUES))
    for (law_name, scn_name), series in results.items():
        cells = "".join(f"{p:>9.2f}" for (p, _, _, _) in series)
        print(f"{law_name + ' / ' + scn_name:28s}{cells}")
    print("=" * 64)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "p21_gain_sensitivity.csv", "w", newline="") as f:
        f.write("\n".join(",".join(map(str, r)) for r in rows_csv))

    # --- figure: P(intercept) vs N (left) and effort vs N (right) ---
    fig, (axp, axe) = plt.subplots(1, 2, figsize=(13, 5))
    styles = {"True PN": "-o", "Augmented PN": "--s"}
    colors = {"weave_18g": "#1f77b4", "jink_22g": "#d62728"}
    for (law_name, scn_name), series in results.items():
        p = [s[0] for s in series]
        lo = [s[0] - s[1] for s in series]
        hi = [s[2] - s[0] for s in series]
        eff = [s[3] for s in series]
        label = f"{law_name} · {scn_name}"
        axp.errorbar(
            N_VALUES,
            p,
            yerr=[lo, hi],
            fmt=styles[law_name],
            color=colors[scn_name],
            capsize=3,
            label=label,
        )
        axe.plot(N_VALUES, eff, styles[law_name], color=colors[scn_name], label=label)
    axp.set_xlabel("navigation constant N")
    axp.set_ylabel("P(intercept)  (95% Wilson CI)")
    axp.set_title("Capture vs. navigation constant")
    axp.set_ylim(0, 1.05)
    axp.grid(True, alpha=0.3)
    axp.legend(fontsize=8)
    axe.set_xlabel("navigation constant N")
    axe.set_ylabel("mean control effort")
    axe.set_title("Effort vs. navigation constant")
    axe.grid(True, alpha=0.3)
    axe.legend(fontsize=8)
    fig.tight_layout()
    GALLERY.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p21_gain_sensitivity.png", dpi=150)
    print(f"Figure: {FIG / 'p21_gain_sensitivity.png'}")
    plt.show() if show else plt.close(fig)


if __name__ == "__main__":
    main()
