"""P34 — pincer coverage: a coordinated pair covers both escape branches of a surprise break.

A ~Mach 2 threat breaks 30 g toward an *unpredictable* side at 1.8 km. That defeats True PN on
both branches — and a **redundant** True-PN pair is no better (identical laws ⇒ identical paths ⇒
perfectly correlated failures). The **pincer pair** (`PincerGuidance`, ADR-0027) splits the approach
geometrically — each interceptor biased to one side via a virtual aim-point that tapers off before
the endgame — so whichever way the target breaks, one of the pair is already leading that branch.
**No target-acceleration estimate is needed**: pure cooperative geometry on plain PN. (Reference:
an Augmented-PN pair solves this regime via acceleration feedforward — the pincer buys the same
robustness without that measurement.)

Outputs: figure (both-branch trajectories + Monte-Carlo bars) and a GIF of the pincer in action.

Run:
    python experiments/p34_pincer.py [--trials 60] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from intercept.adversary import surprise_break, weave
from intercept.core import Entity
from intercept.core.aero import G0, AeroMissile2D
from intercept.guidance import AugmentedPN, pincer_pair, true_pn
from intercept.multiagent.swarm import MultiEngagement

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "gallery" / "figures"
ANIM = ROOT / "gallery" / "animations"
RESULTS = ROOT / "results"
INT_SPEED, TGT_SPEED = 1000.0, 700.0  # ~Mach 3 vs ~Mach 2
INT_AMAX, TGT_AMAX = 40 * G0, 30 * G0
TRIGGER, BREAK_G = 1800.0, 30 * G0
R_SPLIT, R_MERGE = 4000.0, 1200.0
# The split width is matched to the threat's break envelope (ADR-0027 documents this sensitivity):
BETA_BY_VARIANT = {"clean cruise": 0.20, "weaving cruise": 0.22}


def _idyn():
    return AeroMissile2D(a_max=INT_AMAX, tau=0.2)


def _edyn():
    return AeroMissile2D(a_max=TGT_AMAX, tau=0.3)


def _run(laws: list, sign: float, tpos: np.ndarray, weave_g: float = 4.0):
    aim = tpos / np.linalg.norm(tpos)
    ints = [
        Entity(
            f"I{i}",
            _idyn(),
            _idyn().initial_state([0.0, (-1) ** i * 150.0], INT_SPEED * aim),
            role="interceptor",
        )
        for i in range(len(laws))
    ]
    # Threat profile: optionally a light evasive weave on the cruise-in, then the surprise break.
    cruise = weave(amplitude=weave_g * G0, frequency=0.12) if weave_g > 0 else None
    tgt = Entity(
        "T0",
        _edyn(),
        _edyn().initial_state(tpos, [-TGT_SPEED, 0.0]),
        controller=surprise_break("I0", BREAK_G, TRIGGER, sign, base=cruise),
        role="target",
    )
    eng = MultiEngagement(
        ints,
        [tgt],
        lambda tn, L=list(laws): L.pop(0) if L else true_pn(tn),
        dt=0.01,
        t_max=16.0,
        kill_radius=20.0,
        reassign_every=10**9,
    )
    return eng.run()


def _configs(beta: float = 0.20):
    return {
        "single True PN": lambda: [true_pn("T0", N=4.0)],
        "redundant PN ×2": lambda: [true_pn("T0", N=4.0), true_pn("T0", N=4.0)],
        "pincer PN ×2": lambda: list(
            pincer_pair(
                "T0", lambda: true_pn("T0", N=4.0), beta=beta, r_split=R_SPLIT, r_merge=R_MERGE
            )
        ),
        "redundant APN ×2 (ref)": lambda: [AugmentedPN("T0", N=4.0), AugmentedPN("T0", N=4.0)],
    }


def montecarlo(trials: int, seed: int):
    """Both threat regimes: clean cruise (the coverage mechanism) and weaving cruise (stressed)."""
    stats = {}
    for variant, wg in (("clean cruise", 0.0), ("weaving cruise", 4.0)):
        print(f"  --- threat: {variant} (pincer beta={BETA_BY_VARIANT[variant]}) ---")
        for name, make in _configs(beta=BETA_BY_VARIANT[variant]).items():
            wins = 0
            for s in range(trials):
                rng = np.random.default_rng((seed, s))
                sign = 1.0 if rng.random() < 0.5 else -1.0
                tpos = np.array([rng.uniform(7000.0, 9000.0), rng.uniform(-600.0, 600.0)])
                wins += int(len(_run(make(), sign, tpos, weave_g=wg).kills) > 0)
            stats[(name, variant)] = wins / trials
            print(f"  {name:24s}: P(>=1 intercept) = {stats[(name, variant)]:.2f}")
    return stats


def showcase(stats, show: bool) -> None:
    tpos = np.array([8000.0, 200.0])
    mech = _configs(beta=BETA_BY_VARIANT["clean cruise"])
    runs = {s: _run(mech["pincer PN ×2"](), s, tpos, weave_g=0.0) for s in (+1.0, -1.0)}

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0), gridspec_kw={"width_ratios": [2, 2, 1.3]})
    for ax, (sign, res) in zip(axes[:2], runs.items(), strict=False):
        colors = {"I0": "#1f9ede", "I1": "#7a30c9"}
        for nm, role in res.roles.items():
            p = res.tracks[nm][:, :2]
            if role == "interceptor":
                ax.plot(
                    p[:, 0],
                    p[:, 1],
                    "-",
                    color=colors[nm],
                    lw=1.8,
                    label=f"{nm} ({'left' if nm == 'I0' else 'right'}-cover)",
                )
            else:
                ax.plot(p[:, 0], p[:, 1], "k--", lw=1.4, label="target")
        winner = res.kills[0][1] if res.kills else None
        if res.kill_points:
            kp = np.array(res.kill_points)
            ax.plot(
                kp[:, 0],
                kp[:, 1],
                "*",
                color="#c6ff00",
                ms=20,
                mec="k",
                label=f"intercept by {winner}",
            )
        ax.set_title(
            f"break direction {'+' if sign > 0 else '−'} — "
            f"the covering interceptor ({winner}) takes it"
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    ax3 = axes[2]
    configs = ["single True PN", "redundant PN ×2", "pincer PN ×2", "redundant APN ×2 (ref)"]
    short = ["1× PN", "2× PN", "2× pincer", "2× APN"]
    x = np.arange(len(configs))
    clean = [stats[(c, "clean cruise")] for c in configs]
    weav = [stats[(c, "weaving cruise")] for c in configs]
    ax3.bar(x - 0.19, clean, 0.38, color="#1f9ede", label="clean cruise")
    ax3.bar(x + 0.19, weav, 0.38, color="#b46bff", label="weaving cruise")
    for xi, (a, b) in enumerate(zip(clean, weav, strict=True)):
        ax3.text(xi - 0.19, a + 0.02, f"{a:.2f}", ha="center", fontsize=7)
        ax3.text(xi + 0.19, b + 0.02, f"{b:.2f}", ha="center", fontsize=7)
    ax3.set_xticks(x, short, fontsize=8)
    ax3.set_ylim(0, 1.2)
    ax3.set_ylabel("P(≥1 intercept)")
    ax3.set_title("30 g surprise break\n(split matched per envelope)")
    ax3.legend(fontsize=7)
    ax3.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p34_pincer.png", dpi=150)
    plt.show() if show else plt.close(fig)

    # --- GIF: the pair splits, the target breaks, the covering interceptor takes it ---
    res = runs[+1.0]
    tracks = {nm: res.tracks[nm][:, :2] for nm in res.tracks}
    n = max(len(p) for p in tracks.values())
    step = max(1, n // 140)
    figa, ax = plt.subplots(figsize=(8, 5.5), facecolor="#05060a")
    from intercept.viz.animation import style_dark_2d

    style_dark_2d(ax)
    allp = np.vstack(list(tracks.values()))
    ax.set_xlim(allp[:, 0].min() - 300, allp[:, 0].max() + 300)
    ax.set_ylim(allp[:, 1].min() - 500, allp[:, 1].max() + 500)
    ax.set_aspect("equal")
    ax.set_title("Pincer coverage — the pair splits; whichever way it breaks, one is waiting")
    ax.grid(True, alpha=0.25)
    style = {"I0": ("#19e6ff", "-"), "I1": ("#b46bff", "-"), "T0": ("#ff4d6d", "--")}
    lines = {nm: ax.plot([], [], ls, color=c, lw=1.8)[0] for nm, (c, ls) in style.items()}
    heads = {nm: ax.plot([], [], "o", color=c, ms=7)[0] for nm, (c, _) in style.items()}
    (burst,) = ax.plot([], [], "*", color="#c6ff00", ms=22, mec="k")
    kt = res.kills[0][2] if res.kills else None

    def update(f):
        k = min(f * step, n - 1)
        for nm, p in tracks.items():
            kk = min(k, len(p) - 1)
            lines[nm].set_data(p[: kk + 1, 0], p[: kk + 1, 1])
            heads[nm].set_data([p[kk, 0]], [p[kk, 1]])
        if kt is not None and res.times[min(k, len(res.times) - 1)] >= kt:
            kp = res.kill_points[0]
            burst.set_data([kp[0]], [kp[1]])
        return [*lines.values(), *heads.values(), burst]

    anim = FuncAnimation(figa, update, frames=n // step + 1, interval=40, blit=False)
    ANIM.mkdir(parents=True, exist_ok=True)
    anim.save(str(ANIM / "p34_pincer.gif"), writer=PillowWriter(fps=25))
    plt.show() if show else plt.close(figa)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pincer coverage demo")
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    print("=" * 60)
    print("PINCER COVERAGE — 30 g surprise break, unpredictable side")
    print("=" * 60)
    stats = montecarlo(args.trials, args.seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p34_pincer.csv").write_text(
        "config,threat,p_any_intercept\n"
        + "\n".join(f"{n},{v},{p:.3f}" for (n, v), p in stats.items())
    )
    showcase(stats, show)
    print(f"Figure: {FIG / 'p34_pincer.png'}\nGIF: {ANIM / 'p34_pincer.gif'}")


if __name__ == "__main__":
    main()
