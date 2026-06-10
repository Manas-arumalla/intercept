"""P14 — advanced complex-trajectory evasion under full L3 physics (speed parity).

Both missiles are :class:`~intercept.core.realistic.RealisticMissile3D` (ISA atmosphere, boost–
sustain–coast propulsion + mass burn-off, Mach + induced drag, **lift / dynamic-pressure-limited
turning** — every g the target pulls is paid for in energy and bounded by physics). The threat
flies the kind of complex trajectory real maneuvering missiles use to defeat interception:

* a **lofted, descending** midcourse (gravity + initial climb angle),
* a **tilted 3-D serpentine** (the ground track snakes while the altitude porpoises), and
* an **intensifying terminal spiral** — a corkscrew that tightens from a gentle weave to a
  near-max-g helix as the interceptor closes (closed-loop on range-to-go).

Speeds are deliberately realistic and *comparable* — the interceptor does **not** out-run the
threat. The interceptor launches slow (~Mach 1.2) and boosts to ~Mach 3, then coasts and is *down*
to ~Mach 2.6 at the merge; the threat is a fast ~Mach 3 missile that bleeds to ~Mach 2 because hard
maneuvering costs energy (induced drag) — leaving only a ~30–40 % closing-speed edge, typical of a
real surface-to-air engagement against a supersonic threat. Shrink the interceptor's motor any
further and it simply *misses*: it is flying on near-minimum energy, not a speed cheat. The
interceptor wins with **Augmented PN** — a predictive law that estimates the target's acceleration
— by flying an efficient lead, not by overtaking. A robustness Monte-Carlo then perturbs the launch
geometry and every maneuver parameter to show the intercept is not a single tuned shot.

A note on what you will see: at supersonic closing speeds a real missile's "spiral" is a long,
*thin* helix — its turn radius is small next to the distance it covers per revolution — so an
equal-aspect 3-D view at engagement scale looks nearly straight. That is the faithful physics; the
maneuvering is made visible here through the cross-range / altitude **projections** and the
**lateral-g time history** (commanded vs. the physics-available turn limit), the way the
maneuvering is actually quantified in the literature.

Run:
    python experiments/p14_advanced_evasion.py [--show] [--trials N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from intercept.adversary import combine, serpentine3d, terminal_spiral
from intercept.benchmark import wilson_interval
from intercept.core import Engagement, Entity
from intercept.core.realistic import G0, RealisticMissile3D
from intercept.guidance import augmented_pn_3d
from intercept.viz import animate_engagement_3d_modern, plot_engagement_3d_modern
from intercept.viz.threed import _BG, _MODERN

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"

# Nominal showcase parameters (the robustness sweep perturbs all of these).
# Speeds are realistic and comparable: interceptor boosts to ~Mach 3 then coasts to ~Mach 2.6 at the
# merge; the ~Mach 3 threat bleeds to ~Mach 2 under hard maneuvering — only a ~30–40 % closing edge.
NOMINAL = dict(
    intc_v=(380.0, 30.0, 160.0),  # launch slow (~Mach 1.2); the booster does the work
    intc_boost=42000.0,
    intc_t_boost=3.0,
    intc_sustain=6000.0,
    intc_t_sustain=8.0,
    intc_S=0.05,
    tgt_p=(16000.0, 1500.0, 6500.0),
    tgt_v=(-950.0, 0.0, -40.0),  # fast incoming threat (~Mach 3)
    tgt_sustain=8000.0,
    tgt_S=0.07,
    tgt_cl=3.0,
    serp_g=18.0,
    serp_f=0.12,
    tilt=0.55,
    sp_base=8.0,
    sp_max=30.0,
    sp_trig=3000.0,
    sp_rate=1.2,
    N=4.0,
)


def build(params: dict) -> tuple[Engagement, RealisticMissile3D]:
    """Build the L3-vs-L3 engagement from a parameter dict; returns (engagement, target plant)."""
    intc = RealisticMissile3D(
        ref_area=params["intc_S"],
        thrust_boost=params["intc_boost"],
        t_boost=params["intc_t_boost"],
        thrust_sustain=params["intc_sustain"],
        t_sustain=params["intc_t_sustain"],
    )
    interceptor = Entity(
        "interceptor",
        intc,
        intc.initial_state([0.0, 0.0, 0.0], list(params["intc_v"])),
        controller=augmented_pn_3d("target", N=params["N"]),
        role="interceptor",
    )
    tgt = RealisticMissile3D.target(
        ref_area=params["tgt_S"],
        cl_max=params["tgt_cl"],
        thrust_sustain=params["tgt_sustain"],
        t_sustain=60.0,
    )
    target = Entity(
        "target",
        tgt,
        tgt.initial_state(list(params["tgt_p"]), list(params["tgt_v"])),
        controller=combine(
            serpentine3d(
                accel=params["serp_g"] * G0, frequency=params["serp_f"], tilt=params["tilt"]
            ),
            terminal_spiral(
                "interceptor",
                base_accel=params["sp_base"] * G0,
                max_accel=params["sp_max"] * G0,
                trigger_range=params["sp_trig"],
                rate=params["sp_rate"],
            ),
        ),
        role="target",
    )
    eng = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=50.0,
        kill_radius=20.0,
    )
    return eng, tgt


def perturb(rng: np.random.Generator) -> dict:
    """Draw a perturbed parameter set around NOMINAL for the robustness Monte-Carlo."""

    def jit(x, frac):
        return x * (1.0 + frac * float(rng.uniform(-1.0, 1.0)))

    p = dict(NOMINAL)
    p["intc_v"] = (jit(380.0, 0.10), 30.0 + rng.uniform(-25, 25), jit(160.0, 0.15))
    p["tgt_p"] = (
        16000.0 + rng.uniform(-1500, 1500),
        1500.0 + rng.uniform(-900, 900),
        6500.0 + rng.uniform(-900, 900),
    )
    p["tgt_v"] = (jit(-950.0, 0.08), rng.uniform(-50, 50), -40.0 + rng.uniform(-30, 30))
    p["serp_g"] = 18.0 + rng.uniform(-6, 6)
    p["serp_f"] = jit(0.12, 0.35)
    p["tilt"] = 0.55 + rng.uniform(-0.35, 0.35)
    p["sp_max"] = 34.0 + rng.uniform(-7, 7)
    p["sp_trig"] = 3000.0 + rng.uniform(-700, 700)
    p["sp_rate"] = 1.2 + rng.uniform(-0.4, 0.4)
    return p


def maneuver_stats(res, tgt: RealisticMissile3D) -> dict:
    """Quantify how complex the realized target trajectory is."""
    p = res.states["target"][:, :3]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    path = float(seg.sum())
    chord = float(np.linalg.norm(p[-1] - p[0])) or 1.0
    a_ach = np.linalg.norm(res.states["target"][:, 6:9], axis=1)
    avail = np.array(
        [tgt.max_lateral_accel(s, t) for s, t in zip(res.states["target"], res.times, strict=True)]
    )
    return dict(
        path=path,
        chord=chord,
        tortuosity=path / chord,
        alt_span=float(p[:, 2].max() - p[:, 2].min()),
        g_peak=float(a_ach.max() / G0),
        g_mean=float(a_ach.mean() / G0),
        g_avail_min=float(avail.min() / G0),
        a_ach=a_ach / G0,
        avail=avail / G0,
    )


def analysis_figure(res, stats: dict, save_path: Path, show: bool) -> None:
    """Dark multi-panel that makes the maneuvering visible: projections + lateral-g history."""
    ip = res.states["interceptor"][:, :3]
    tp = res.states["target"][:, :3]
    ci, ct = _MODERN["interceptor"], _MODERN["target"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor=_BG)
    fig.suptitle(
        "Advanced complex-trajectory engagement — L3 physics, speed parity",
        color="#e6edf3",
        fontsize=14,
    )

    def style(ax, xlabel, ylabel, title):
        ax.set_facecolor(_BG)
        ax.set_title(title, color="#e6edf3", fontsize=11)
        ax.set_xlabel(xlabel, color="#7d8590")
        ax.set_ylabel(ylabel, color="#7d8590")
        ax.tick_params(colors="#55585f", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#2a2e36")
        ax.grid(True, color="#1b1f27", lw=0.6)

    # Top-down: downrange vs cross-range (the serpentine snake).
    ax = axes[0, 0]
    ax.plot(tp[:, 0], tp[:, 1], color=ct, lw=1.8, label="target")
    ax.plot(ip[:, 0], ip[:, 1], color=ci, lw=1.8, label="interceptor")
    style(ax, "downrange x (m)", "cross-range y (m)", "Top-down — serpentine ground track")
    ax.legend(facecolor="#0d1117", labelcolor="#e6edf3", edgecolor="#2a2e36", fontsize=8)

    # Side view: downrange vs altitude (loft / porpoise / dive).
    ax = axes[0, 1]
    ax.plot(tp[:, 0], tp[:, 2], color=ct, lw=1.8)
    ax.plot(ip[:, 0], ip[:, 2], color=ci, lw=1.8)
    style(ax, "downrange x (m)", "altitude z (m)", "Side view — loft / porpoise / dive")

    # Range-to-go vs time.
    ax = axes[1, 0]
    rng = np.linalg.norm(tp - ip, axis=1)
    ax.plot(res.times, rng, color="#c6ff00", lw=1.8)
    if res.intercept_time is not None:
        ax.axvline(res.intercept_time, color="#ff4d6d", ls="--", lw=1.0)
    style(ax, "time (s)", "range to interceptor (m)", "Closing range")

    # Target lateral-g: commanded/achieved vs the physics-available turn limit.
    ax = axes[1, 1]
    ax.plot(res.times, stats["a_ach"], color=ct, lw=1.8, label="achieved g")
    ax.plot(
        res.times, stats["avail"], color="#9aa0a6", lw=1.2, ls="--", label="available turn limit"
    )
    style(ax, "time (s)", "lateral acceleration (g)", "Target g vs. physics limit")
    ax.legend(facecolor="#0d1117", labelcolor="#e6edf3", edgecolor="#2a2e36", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, facecolor=_BG)
    plt.show() if show else plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced complex-trajectory evasion demo")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--trials", type=int, default=60, help="robustness Monte-Carlo trials")
    args = parser.parse_args()

    eng, tgt = build(NOMINAL)
    res = eng.run()
    stats = maneuver_stats(res, tgt)

    print("=" * 64)
    print("SHOWCASE — APN interceptor vs. complex-trajectory threat (L3, speed parity)")
    print("=" * 64)
    print(f"Outcome            : {res.reason.name}")
    print(f"Miss distance      : {res.miss_distance:.2f} m")
    if res.intercept_time is not None:
        print(f"Intercept time     : {res.intercept_time:.2f} s")
    print(f"Target path length : {stats['path']:.0f} m  (tortuosity {stats['tortuosity']:.3f})")
    print(f"Target alt. span   : {stats['alt_span']:.0f} m")
    print(f"Target g  peak/mean: {stats['g_peak']:.1f} / {stats['g_mean']:.1f} g")
    print(f"Turn limit (min)   : {stats['g_avail_min']:.1f} g  (physics-bounded)")
    print("=" * 64)

    title = "INTERCEPT — APN vs. complex-trajectory threat (L3)"
    plot_engagement_3d_modern(
        res, title=title, save_path=FIG / "p14_advanced_modern.png", show=args.show
    )
    animate_engagement_3d_modern(
        res, title=title, save_path=ANIM / "p14_advanced_modern.gif", show=args.show
    )
    analysis_figure(res, stats, FIG / "p14_advanced_analysis.png", args.show)
    from intercept.viz import has_plotly, interactive_engagement_3d

    if has_plotly():
        interactive_engagement_3d(
            res, title=title, show=args.show, save_path=ANIM / "p14_advanced_interactive.html"
        )
    print(f"Saved figures + animation to {GALLERY}")

    # --- Robustness Monte-Carlo: perturb geometry + every maneuver parameter ---
    rng = np.random.default_rng(20260608)
    misses, hits = [], 0
    for _ in range(args.trials):
        r, _ = build(perturb(rng))
        out = r.run()
        hits += int(out.intercepted)
        misses.append(out.miss_distance)
    lo, hi = wilson_interval(hits, args.trials)
    misses = np.array(misses)
    print("-" * 64)
    print(f"ROBUSTNESS ({args.trials} perturbed trials, seed 20260608)")
    print(f"P(intercept)       : {hits / args.trials:.2f}  (95% CI [{lo:.2f}, {hi:.2f}])")
    print(f"Miss median / p90  : {np.median(misses):.1f} / {np.percentile(misses, 90):.1f} m")
    print("-" * 64)


if __name__ == "__main__":
    main()
