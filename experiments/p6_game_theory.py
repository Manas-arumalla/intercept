"""P6 demo — pursuit-evasion game theory.

Figure 1 (``p6_apollonius.png``): the Apollonius circle for a 1v1 engagement — the evader's
dominance region and the predicted capture point.

Figure 2 (``p6_evader_robustness.png``): how hard each adversary is. A True-PN interceptor faces a
suite of evaders — straight, weave, jink (step), and the **game-theoretic optimal evader** (flees
along the anti-line-of-sight) — over a Monte-Carlo of geometries; we report probability of intercept
(with Wilson CIs) and mean time-to-intercept. The optimal evader is the hardest, by design.

Run:
    python experiments/p6_game_theory.py [--trials 150] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from intercept.adversary import optimal_evader, step_maneuver, straight, weave
from intercept.benchmark import ParametricScenario
from intercept.benchmark.metrics import summarize
from intercept.benchmark.runner import BenchmarkRow, format_table
from intercept.core import Engagement, Entity, PointMass2D
from intercept.guidance import true_pn
from intercept.viz import plot_apollonius

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"


def figure_apollonius(show: bool) -> None:
    pursuer_speed = 1000.0  # ~Mach 3 pursuer
    evader_vel = np.array([-580.0, 350.0])  # |v_E| ~ 680 (~Mach 2) => alpha ~ 0.68 (realistic)
    plot_apollonius(
        pursuer_pos=np.array([0.0, 0.0]),
        evader_pos=np.array([3500.0, 1200.0]),
        evader_vel=evader_vel,
        pursuer_speed=pursuer_speed,
        save_path=FIG / "p6_apollonius.png",
        show=show,
    )


def run_evader(evader_factory, scenario, n_trials, seed):
    """Monte-Carlo a True-PN interceptor vs a given evader controller over sampled geometries."""
    results = []
    for ss in np.random.SeedSequence(seed).spawn(n_trials):
        rng = np.random.default_rng(ss)
        spec = scenario.sample(rng)
        interceptor = Entity(
            "interceptor",
            PointMass2D(a_max=spec.interceptor_a_max),
            spec.interceptor_state,
            controller=true_pn("target", N=4.0),
            role="interceptor",
        )
        target = Entity(
            "target", PointMass2D(), spec.target_state, controller=evader_factory(), role="target"
        )
        results.append(
            Engagement(
                [interceptor, target],
                interceptor="interceptor",
                target="target",
                dt=spec.dt,
                t_max=spec.t_max,
                kill_radius=spec.kill_radius,
            ).run()
        )
    return summarize(results)


def figure_evader_robustness(show: bool, trials: int) -> None:
    # Head-on baseline so a *straight* target is intercepted reliably/quickly; the maneuvering and
    # game-theoretic evaders then degrade either intercept probability (weave/jink force a miss) or
    # time-to-intercept (the optimal evader flees, delaying capture). Moderate authority.
    scenario = ParametricScenario(
        name="pe",
        interceptor_speed=1000,
        interceptor_a_max=250,
        target_speed=700,
        range_min=4000,
        range_max=6000,
        offset_min=-400,
        offset_max=400,
        target_heading_deg=180,
        kill_radius=15.0,
        dt=0.01,
        t_max=25.0,
    )
    evaders = {
        "straight": straight,
        "weave (6g)": lambda: weave(amplitude=58.8, frequency=0.3),
        "jink (step)": lambda: step_maneuver(accel=60.0, t_start=2.0),
        "optimal (game)": lambda: optimal_evader("interceptor"),
    }
    rows = [
        BenchmarkRow(name, "vs True PN", run_evader(f, scenario, trials, seed=7))
        for name, f in evaders.items()
    ]
    print(format_table(rows))

    # In this pursuer-dominant regime every evader is caught (P_intercept ~ 1.0), so the
    # differentiator is *time-to-intercept*: the game-theoretic optimal evader maximizes it.
    names = [r.algorithm for r in rows]
    ttis = [r.summary.tti_mean for r in rows]
    misses = [r.summary.miss_median for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#7f7f7f", "#ff7f0e", "#8c564b", "#d62728"]
    bars = ax.bar(names, ttis, color=colors)
    for b, m in zip(bars, misses, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"miss {m:.0f} m",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("mean time-to-intercept (s)")
    ax.set_title(
        "Evader robustness vs. True PN — capture time by adversary "
        "(all intercepted; optimal evader delays capture most)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "p6_evader_robustness.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print("\nMean time-to-intercept by evader:")
    for r in rows:
        tti = r.summary.tti_mean
        print(
            f"  {r.algorithm:16s}: P_int={r.summary.p_intercept:.2f}  "
            f"tti={'n/a' if tti != tti else f'{tti:.2f} s'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="P6 pursuit-evasion game theory")
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    print("Figure 1: Apollonius circle")
    figure_apollonius(show)
    print("Figure 2: evader robustness (True PN vs evader suite)")
    figure_evader_robustness(show, args.trials)
    print(f"Figures saved to: {GALLERY}")


if __name__ == "__main__":
    main()
