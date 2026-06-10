"""P9 demo — full 3-D engagement (real life is 3-D).

A supersonic interceptor using 3-D Augmented PN runs down a fast target performing a 3-D barrel-roll
(helical corkscrew) evasive maneuver, under realistic L2 physics (gravity, drag, g-limit, autopilot
lag). Produces a static 3-D trajectory plot and a rotating 3-D animated replay (GIF).

Run:
    python experiments/p9_3d_demo.py [--show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from intercept.adversary import barrel_roll
from intercept.core import G0, AeroMissile3D, Engagement, Entity
from intercept.guidance import augmented_pn_3d
from intercept.viz import (
    animate_engagement_3d,
    animate_engagement_3d_modern,
    plot_engagement_3d,
    plot_engagement_3d_modern,
)

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"


def build() -> Engagement:
    plant = AeroMissile3D(a_max=45 * G0, tau=0.2)
    interceptor = Entity(
        "interceptor",
        plant,
        plant.initial_state([0.0, 0.0, 0.0], [1000.0, 0.0, 120.0]),
        controller=augmented_pn_3d("target", N=4.0),
        role="interceptor",
    )
    tgt = AeroMissile3D(a_max=25 * G0, tau=0.3)
    target = Entity(
        "target",
        tgt,
        tgt.initial_state([7500.0, 1500.0, 3000.0], [-750.0, 60.0, 0.0]),
        controller=barrel_roll(accel=18 * G0, rate=1.2),
        role="target",
    )
    return Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=25.0,
        kill_radius=20.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="3-D engagement demo")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    res = build().run()
    print("=" * 56)
    print(f"Outcome        : {res.reason.name}")
    print(f"Miss distance  : {res.miss_distance:.2f} m")
    if res.intercept_time is not None:
        print(f"Intercept time : {res.intercept_time:.2f} s")
    print("=" * 56)

    plot_engagement_3d(
        res,
        title="3-D engagement: APN interceptor vs. barrel-roll evader",
        save_path=FIG / "p9_3d_engagement.png",
        show=args.show,
    )
    animate_engagement_3d(
        res,
        title="3-D: APN vs. barrel-roll evader",
        save_path=ANIM / "p9_3d_engagement.gif",
        show=args.show,
    )

    # Modern / cinematic styling (dark theme, neon glow, intercept flash, orbiting camera).
    plot_engagement_3d_modern(
        res,
        title="INTERCEPT — APN vs. barrel-roll evader",
        save_path=FIG / "p9_3d_engagement_modern.png",
        show=args.show,
    )
    animate_engagement_3d_modern(
        res,
        title="INTERCEPT — APN vs. barrel-roll evader",
        save_path=ANIM / "p9_3d_engagement_modern.gif",
        show=args.show,
    )
    print(f"Saved 3-D figures + animations (classic + modern) to {GALLERY}")

    # Interactive browsable replay (Plotly HTML), if plotly is installed (.[viz]).
    from intercept.viz import has_plotly, interactive_engagement_3d

    if has_plotly():
        interactive_engagement_3d(
            res,
            title="INTERCEPT — interactive 3-D replay",
            save_path=ANIM / "p9_interactive_3d.html",
            show=args.show,
        )
        print(f"Saved interactive HTML replay to {ANIM / 'p9_interactive_3d.html'}")
    else:
        print("(install .[viz] for the interactive Plotly replay)")


if __name__ == "__main__":
    main()
