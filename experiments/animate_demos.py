"""Generate animated engagement replays (GIFs) + filmstrip montages — watch the intercepts happen.

Realistic comparable speeds (interceptor ~Mach 3, target ~Mach 2 — only a ~1.45x closing edge).
Produces, in gallery/:
  * anim_pn_weaving.gif / .png   — True PN runs down a 15 g weaving target
  * anim_pn_vs_pursuit.gif       — PN leads & intercepts a crossing target while pure pursuit lags
  * anim_optimal_evader.gif/.png — PN vs the game-theoretic evader fleeing along the anti-LOS

Run:
    python experiments/animate_demos.py [--show]
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from intercept.adversary import optimal_evader, weave
from intercept.core import Engagement, Entity, PointMass2D
from intercept.core.entities import Controller
from intercept.guidance import true_pn
from intercept.viz import animate_comparison, animate_engagement, filmstrip_engagement

Array = NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"


def pure_pursuit(target_name: str, gain: float = 8.0, a_max: float = 100.0) -> Controller:
    def controller(t: float, own: Array, world: Mapping[str, Array]) -> Array:
        v = own[2:4]
        s = float(np.linalg.norm(v))
        if s < 1e-6:
            return np.zeros(2)
        los = world[target_name][:2] - own[:2]
        err = (np.arctan2(los[1], los[0]) - np.arctan2(v[1], v[0]) + np.pi) % (2 * np.pi) - np.pi
        v_hat = v / s
        return float(np.clip(gain * s * err, -a_max, a_max)) * np.array([-v_hat[1], v_hat[0]])

    return controller


def engage(
    guidance,
    target_state,
    *,
    target_ctrl=None,
    a_max=392.0,
    speed=1000.0,
    kill_radius=20.0,
    dt=0.01,
    t_max=30.0,
    aim=True,
):
    tp = np.array(target_state, float)[:2]
    head = tp / np.linalg.norm(tp) if aim else np.array([1.0, 0.0])
    interceptor = Entity(
        "interceptor",
        PointMass2D(a_max=a_max),
        np.array([0.0, 0.0, speed * head[0], speed * head[1]]),
        controller=guidance,
        role="interceptor",
    )
    target = Entity(
        "target",
        PointMass2D(),
        np.array(target_state, float),
        controller=target_ctrl,
        role="target",
    )
    return Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=dt,
        t_max=t_max,
        kill_radius=kill_radius,
    ).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate engagement animations")
    parser.add_argument("--show", action="store_true", help="pop up interactive windows")
    args = parser.parse_args()
    show = args.show

    # Realistic comparable speeds throughout: ~Mach 3 interceptor (1000 m/s) vs ~Mach 2 target
    # (700 m/s) — only a ~1.45x edge.
    # 1) True PN vs a weaving target.
    print("1/3  PN vs weaving target ...")
    res = engage(
        true_pn("target", N=4.0),
        [7000.0, 1800.0, -700.0, 0.0],
        target_ctrl=weave(amplitude=147.0, frequency=0.3),
        a_max=392.0,
    )
    animate_engagement(
        res,
        title="True PN vs. 15g weaving target (Mach 2)",
        save_path=ANIM / "anim_pn_weaving.gif",
        show=show,
    )
    filmstrip_engagement(res, save_path=FIG / "anim_pn_weaving.png")

    # 2) PN vs pure pursuit on a crossing target (PN leads & hits, pursuit lags the crossing).
    print("2/3  PN vs pure pursuit (crossing) ...")
    tgt = [7000.0, 2600.0, -680.0, 170.0]  # ~700 m/s, crossing
    results = {
        "Pure pursuit": engage(pure_pursuit("target", a_max=250.0), tgt, a_max=250.0),
        "True PN": engage(true_pn("target", N=4.0), tgt, a_max=250.0),
    }
    animate_comparison(
        results,
        title="PN leads & intercepts; pure pursuit lags the crossing",
        save_path=ANIM / "anim_pn_vs_pursuit.gif",
        show=show,
    )

    # 3) PN vs the game-theoretic optimal evader (fleeing tail-chase, realistic 1.45x speed edge).
    print("3/3  PN vs optimal evader ...")
    res_ev = engage(
        true_pn("target", N=4.0),
        [4500.0, 400.0, 700.0, 0.0],
        target_ctrl=optimal_evader("interceptor"),
        a_max=392.0,
        t_max=30.0,
    )
    animate_engagement(
        res_ev,
        title="True PN vs. game-theoretic optimal evader",
        save_path=ANIM / "anim_optimal_evader.gif",
        show=show,
    )
    filmstrip_engagement(res_ev, save_path=FIG / "anim_optimal_evader.png")

    print(f"\nDone. GIFs + filmstrips in: {GALLERY}")


if __name__ == "__main__":
    main()
