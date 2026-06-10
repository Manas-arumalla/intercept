"""P28 — swarm-vs-swarm showcase: a saturating raid of *diverse* threat trajectories, defended.

A defended point is attacked by a salvo of threats each flying a **different realistic profile**
(`intercept.adversary.threats`): cruise-weave, sea-skimming pop-up, lofted-ballistic, terminal-
spiral, diving-jink, boost-glide. A battery of interceptors defends with Augmented-PN-3D guidance
and Hungarian weapon-target assignment (re-solved live, now 3-D-aware). Realistic comparable speeds
(threats ~Mach 2, interceptors ~Mach 3).

Outputs (gallery/): a labeled static 3-D figure of the raid (each threat coloured/annotated by type)
and a **cinematic** animated GIF (`viz.animate_swarm_3d_modern`).

Run:
    python experiments/p28_swarm_showcase.py [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from intercept.adversary.threats import THREAT_PROFILES
from intercept.core import Entity, PointMass3D
from intercept.core.aero import G0
from intercept.guidance import augmented_pn_3d
from intercept.multiagent.swarm import MultiEngagement
from intercept.viz import animate_swarm_3d_modern
from intercept.viz.threed import _BG, _glow_plot, _set_equal_aspect, _style_dark

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "gallery" / "figures"
ANIM = ROOT / "gallery" / "animations"
RESULTS = ROOT / "results"

THREAT_SPEED, INT_SPEED = 700.0, 1000.0  # ~Mach 2 vs ~Mach 3 (realistic ~1.45x speed ratio)
THREAT_AMAX, INT_AMAX = 30 * G0, 50 * G0
DEFENDED = np.array([0.0, 0.0, 0.0])


def build(seed: int = 7):
    rng = np.random.default_rng(seed)
    profiles = list(THREAT_PROFILES.items())
    # A two-wave saturating raid: wave 1 = all six profiles at 9-11 km; wave 2 = six more at
    # 13.5-15.5 km arriving ~6 s later — 12 diverse threats total against a 12-interceptor battery.
    threats, labels = [], {}
    j = 0
    for wave, (r_lo, r_hi) in enumerate(((9000.0, 11000.0), (13500.0, 15500.0))):
        azis = np.linspace(35.0, 145.0, len(profiles)) + (8.0 if wave else 0.0)
        for (name, ctrl), az_deg in zip(profiles, azis, strict=True):
            az = np.radians(az_deg)
            rngm = float(rng.uniform(r_lo, r_hi))
            alt = float(rng.uniform(2500.0, 5500.0))
            pos = np.array([rngm * np.cos(az), rngm * np.sin(az), alt])
            aim = -pos / np.linalg.norm(pos)
            state = np.array([*pos, *(THREAT_SPEED * aim)])
            nm = f"T{j}:{name}"
            threats.append(
                Entity(nm, PointMass3D(a_max=THREAT_AMAX), state, controller=ctrl, role="target")
            )
            labels[nm] = f"{name} (w{wave + 1})"
            j += 1
    centroid = np.mean([t.state[:3] for t in threats], axis=0)
    interceptors = []
    for i in range(12):
        p = np.array([rng.uniform(-600, 600), rng.uniform(-600, 600), 0.0])
        aim = (centroid - p) / np.linalg.norm(centroid - p)
        interceptors.append(
            Entity(
                f"I{i}",
                PointMass3D(a_max=INT_AMAX),
                np.array([*p, *(INT_SPEED * aim)]),
                role="interceptor",
            )
        )
    eng = MultiEngagement(
        interceptors,
        threats,
        lambda t: augmented_pn_3d(t, N=4.0),
        dt=0.02,
        t_max=36.0,
        kill_radius=50.0,
        reassign_every=20,
    )
    return eng, labels


def static_figure(result, labels, path: Path, show: bool) -> None:
    fig = plt.figure(figsize=(9.5, 8), facecolor=_BG)
    ax = fig.add_subplot(projection="3d")
    _style_dark(ax)
    allpts = np.vstack([s[:, :3] for s in result.tracks.values()])
    _set_equal_aspect(ax, allpts)
    warm = plt.cm.autumn(np.linspace(0.0, 0.75, len(labels)))
    for idx, (nm, prof) in enumerate(labels.items()):
        p = result.tracks[nm][:, :3]
        _glow_plot(ax, p[:, 0], p[:, 1], p[:, 2], warm[idx])
        ax.plot([p[0, 0]], [p[0, 1]], [p[0, 2]], "o", color=warm[idx], ms=5, label=prof)
    for nm, r in result.roles.items():
        if r != "interceptor":
            continue
        p = result.tracks[nm][:, :3]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], "-", color="#19e6ff", lw=1.0, alpha=0.7)
    if result.kill_points:
        kp = np.array(result.kill_points)
        ax.scatter(
            kp[:, 0],
            kp[:, 1],
            kp[:, 2],
            c="#c6ff00",
            marker="*",
            s=140,
            depthshade=False,
            label="intercepts",
        )
    ax.scatter([0], [0], [0], c="#19e6ff", marker="^", s=90, label="defended point")
    kept = result.n_targets - result.leakers
    ax.set_title(
        f"Diverse-threat raid — {kept}/{result.n_targets} intercepted (2 waves, 12 interceptors)",
        color="#e6edf3",
        fontsize=12,
        pad=12,
    )
    ax.legend(
        loc="upper left", fontsize=7, facecolor="#0b0d12", labelcolor="#cdd5df", framealpha=0.6
    )
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=_BG)
    plt.show() if show else plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diverse swarm-vs-swarm showcase")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    eng, labels = build()
    result = eng.run()
    kept = result.n_targets - result.leakers
    print("=" * 60)
    print(
        f"SWARM SHOWCASE — {result.n_targets} diverse threats vs {result.n_interceptors} "
        f"interceptors"
    )
    print("=" * 60)
    for tgt, interceptor, t in result.kills:
        print(f"  {interceptor:4s} intercepted {tgt:20s} at t={t:5.2f}s")
    print(f"  intercepted {kept}/{result.n_targets}   leakers {result.leakers}")
    print("=" * 60)

    static_figure(result, labels, FIG / "p28_swarm_showcase.png", show)
    animate_swarm_3d_modern(
        result,
        title=f"Swarm defense — {kept}/{result.n_targets} diverse threats intercepted",
        save_path=ANIM / "p28_swarm_showcase.gif",
        max_frames=140,
        show=show,
    )  # ≤10 MB so GitHub renders it inline
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p28_swarm_showcase.csv").write_text(
        "target,interceptor,time\n" + "\n".join(f"{t},{i},{tm:.2f}" for t, i, tm in result.kills)
    )
    print(f"Figure: {FIG / 'p28_swarm_showcase.png'}\nAnimation: {ANIM / 'p28_swarm_showcase.gif'}")


if __name__ == "__main__":
    main()
