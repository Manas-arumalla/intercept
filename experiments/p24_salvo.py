"""P24 — cooperative salvo: many interceptors, one commanded impact time.

A battery launched from different ranges/bearings would, under plain PN, arrive spread out in time.
Impact-Time-Control Guidance (`ImpactTimeGuidance`, ADR-0016) makes every interceptor arrive at the
*same* commanded time ``t_impact`` — a synchronized salvo that saturates the defense at once. Each
interceptor solves the same 1-v-1 engagement against the (shared, coasting) target; the target's
motion is identical across them, so the trajectories overlay into one salvo picture.

Produces a two-panel figure (trajectories + arrival-time comparison) and an animated GIF of the
synchronized arrival.

Run:
    python experiments/p24_salvo.py [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from intercept.adversary import weave
from intercept.core import G0, Engagement, Entity, PointMass2D
from intercept.guidance import ImpactTimeGuidance, true_pn

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"

INT_SPEED = 1050.0  # ~Mach 3.1 interceptor
A_MAX = 392.0  # 40 g
# Incoming threat at ~Mach 2 (700 m/s) flying toward the defended area near the origin — realistic
# comparable speeds (interceptor only ~1.5x the threat).
TARGET0 = np.array([11000.0, 800.0, -700.0, 0.0])
# A spread battery: launch positions near the defended area; each interceptor's velocity is aimed at
# the threat at INT_SPEED.
LAUNCH_POS = [
    np.array([0.0, 0.0]),
    np.array([300.0, -1300.0]),
    np.array([-400.0, 1400.0]),
    np.array([500.0, 900.0]),
]


def _run(pos, factory):
    aim = TARGET0[:2] - np.asarray(pos, float)
    vel = INT_SPEED * aim / np.linalg.norm(aim)
    s0 = np.array([pos[0], pos[1], vel[0], vel[1]])
    interceptor = Entity(
        "interceptor", PointMass2D(a_max=A_MAX), s0, controller=factory(), role="interceptor"
    )
    # Realistic inbound: the threat flies a visible ~8 g serpentine weave — deterministic, so every
    # interceptor's 1-v-1 sim sees the identical target track (the salvo overlay stays valid).
    target = Entity(
        "target",
        PointMass2D(a_max=10 * G0),
        TARGET0.copy(),
        controller=weave(amplitude=8 * G0, frequency=0.1),
        role="target",
    )
    return Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=40.0,
        kill_radius=20.0,
    ).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cooperative salvo (impact-time guidance) demo")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    # Choose a feasible common impact time: a buffer above the slowest natural (PN) arrival.
    pn = [_run(p, lambda: true_pn("target", N=4.0)) for p in LAUNCH_POS]
    pn_times = [r.intercept_time for r in pn]
    t_impact = round(max(pn_times) + 1.8, 1)

    salvo = [
        _run(p, lambda tI=t_impact: ImpactTimeGuidance("target", t_impact=tI, N=4.0, k=0.3))
        for p in LAUNCH_POS
    ]
    salvo_times = [r.intercept_time for r in salvo]

    print("=" * 60)
    print(f"COOPERATIVE SALVO — commanded impact time t_impact = {t_impact:.1f} s")
    print("=" * 60)
    print(
        f"PN arrival times    : {[round(x, 2) for x in pn_times]}  spread "
        f"{max(pn_times) - min(pn_times):.2f} s"
    )
    print(
        f"ITCG arrival times  : {[round(x, 2) for x in salvo_times]}  spread "
        f"{max(salvo_times) - min(salvo_times):.2f} s"
    )
    print("=" * 60)

    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(LAUNCH_POS)))
    tgt_xy = salvo[0].states["target"][:, :2]

    # --- static figure: trajectories + arrival-time bars ---
    fig, (axt, axb) = plt.subplots(1, 2, figsize=(13, 5.5))
    axt.plot(tgt_xy[:, 0], tgt_xy[:, 1], "k--", lw=1.2, label="target")
    for i, r in enumerate(salvo):
        p = r.states["interceptor"][:, :2]
        axt.plot(p[:, 0], p[:, 1], "-", color=colors[i], lw=1.6, label=f"interceptor {i + 1}")
        axt.plot(p[0, 0], p[0, 1], "o", color=colors[i], ms=6)
    ti = int(np.argmin(np.abs(salvo[0].times - salvo[0].closest_approach_time)))
    axt.plot(
        tgt_xy[ti, 0],
        tgt_xy[ti, 1],
        "*",
        color="#c6ff00",
        ms=22,
        mec="k",
        label="simultaneous impact",
    )
    axt.set_title(f"Salvo trajectories — all arrive at t={t_impact:.1f}s")
    axt.set_xlabel("x (m)")
    axt.set_ylabel("y (m)")
    axt.set_aspect("equal", adjustable="datalim")
    axt.legend(fontsize=8)
    axt.grid(True, alpha=0.3)

    x = np.arange(len(LAUNCH_POS))
    axb.bar(x - 0.2, pn_times, 0.4, label="True PN", color="#1f77b4")
    axb.bar(x + 0.2, salvo_times, 0.4, label="ITCG (salvo)", color="#d62728")
    axb.axhline(t_impact, color="#c6ff00", ls="--", lw=1.5, label=f"commanded {t_impact:.1f}s")
    axb.set_title("Arrival time by interceptor")
    axb.set_xlabel("interceptor")
    axb.set_ylabel("intercept time (s)")
    axb.set_xticks(x, [str(i + 1) for i in range(len(LAUNCH_POS))])
    axb.legend(fontsize=8)
    axb.grid(True, alpha=0.3)
    fig.tight_layout()
    GALLERY.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p24_salvo.png", dpi=150)
    plt.show() if show else plt.close(fig)

    # --- animated GIF of the synchronized arrival ---
    n = max(len(r.times) for r in salvo)
    step = max(1, n // 150)
    paths = [r.states["interceptor"][:, :2] for r in salvo]
    figa, ax = plt.subplots(figsize=(7, 6), facecolor="#05060a")
    from intercept.viz.animation import style_dark_2d

    style_dark_2d(ax)
    ax.set_aspect("equal")
    allp = np.vstack([tgt_xy, *paths])
    ax.set_xlim(allp[:, 0].min() - 300, allp[:, 0].max() + 300)
    ax.set_ylim(allp[:, 1].min() - 300, allp[:, 1].max() + 300)
    ax.set_title(f"Cooperative salvo — simultaneous arrival (t={t_impact:.1f}s)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    (tgt_ln,) = ax.plot([], [], "--", color="#ff4d6d", lw=1.4)
    (tgt_hd,) = ax.plot([], [], "s", color="#ff4d6d", ms=7)
    lines = [ax.plot([], [], "-", color=colors[i], lw=1.6)[0] for i in range(len(paths))]
    heads = [ax.plot([], [], "o", color=colors[i], ms=7)[0] for i in range(len(paths))]

    def update(f):
        k = min(f * step, n - 1)
        kt = min(k, len(tgt_xy) - 1)
        tgt_ln.set_data(tgt_xy[: kt + 1, 0], tgt_xy[: kt + 1, 1])
        tgt_hd.set_data([tgt_xy[kt, 0]], [tgt_xy[kt, 1]])
        for p, ln, hd in zip(paths, lines, heads, strict=True):
            kk = min(k, len(p) - 1)
            ln.set_data(p[: kk + 1, 0], p[: kk + 1, 1])
            hd.set_data([p[kk, 0]], [p[kk, 1]])
        return [tgt_ln, tgt_hd, *lines, *heads]

    anim = FuncAnimation(figa, update, frames=n // step + 1, interval=40, blit=False)
    anim.save(str(ANIM / "p24_salvo.gif"), writer=PillowWriter(fps=25))
    plt.show() if show else plt.close(figa)
    print(f"Saved figure + GIF to {GALLERY}")


if __name__ == "__main__":
    main()
