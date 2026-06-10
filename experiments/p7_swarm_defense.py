"""P7 demo — swarm / area defense: many interceptors vs a salvo of incoming threats.

A battery of interceptors defends a point against a fan of inbound threats. A Hungarian
weapon-target assignment (re-solved as the engagement evolves and threats are killed) pairs each
interceptor to a threat; each then runs Proportional Navigation. Produces a static engagement map
and an animated GIF.

Run:
    python experiments/p7_swarm_defense.py [--n-int 8] [--n-tgt 8] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from intercept.adversary import weave
from intercept.core import Entity, PointMass2D
from intercept.guidance import true_pn
from intercept.multiagent import MultiEngagement
from intercept.viz import animate_swarm, plot_swarm

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"


def build(n_int: int, n_tgt: int) -> MultiEngagement:
    rng = np.random.default_rng(7)
    # Threats inbound from a fan (range ~7 km) toward the defended point at the origin.
    targets = []
    for j in range(n_tgt):
        ang = np.radians(np.linspace(35, 145, n_tgt)[j]) + rng.uniform(-0.03, 0.03)
        rng_m = rng.uniform(6500, 8000)
        pos = rng_m * np.array([np.cos(ang), np.sin(ang)])
        # ~Mach 2 (700 m/s) inbound threats vs ~Mach 3 (1000 m/s) interceptors — realistic ~1.45x
        # edge for area defense.
        vel = -700.0 * np.array([np.cos(ang), np.sin(ang)])  # heading at the defended point
        targets.append(
            Entity(
                f"T{j}",
                PointMass2D(),
                np.array([*pos, *vel]),
                controller=weave(amplitude=6 * 9.81, frequency=0.2),
                role="target",
            )
        )
    # Interceptors launch from a small battery near the origin, fanned toward the threat sector.
    interceptors = []
    centroid = np.mean([t.state[:2] for t in targets], axis=0)
    for i in range(n_int):
        pos = np.array([rng.uniform(-300, 300), rng.uniform(-150, 150)])
        aim = (centroid - pos) / np.linalg.norm(centroid - pos)
        interceptors.append(
            Entity(
                f"I{i}",
                PointMass2D(a_max=392.0),
                np.array([*pos, *(1000.0 * aim)]),
                role="interceptor",
            )
        )
    return MultiEngagement(
        interceptors,
        targets,
        lambda name: true_pn(name, N=4.0),
        dt=0.01,
        t_max=30.0,
        kill_radius=25.0,
        reassign_every=20,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Swarm / area defense demo")
    parser.add_argument("--n-int", type=int, default=8)
    parser.add_argument("--n-tgt", type=int, default=8)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    res = build(args.n_int, args.n_tgt).run()
    print("=" * 56)
    print(f"Interceptors: {res.n_interceptors}   Threats: {res.n_targets}")
    print(f"Intercepted : {res.n_killed}   Leakers: {res.leakers}")
    print("=" * 56)

    plot_swarm(res, save_path=FIG / "p7_swarm_defense.png", show=show)
    animate_swarm(
        res,
        title="Swarm / area defense (Hungarian WTA + PN)",
        save_path=ANIM / "p7_swarm_defense.gif",
        show=show,
    )
    print(f"Figure + animation saved to {GALLERY}")


if __name__ == "__main__":
    main()
