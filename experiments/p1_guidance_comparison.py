"""P1 demo — Proportional Navigation baseline vs. the P0 pure-pursuit placeholder, and
True PN vs. Augmented PN against a maneuvering target.

Two figures:
  1. ``p1_pn_vs_pursuit.png``      — same crossing geometry and *low* control authority where pure
     pursuit lags into a large miss but PN holds a constant bearing and intercepts.
  2. ``p1_pn_vs_apn_maneuvering.png`` — a hard-turning target; APN's target-acceleration
     feedforward reduces the terminal miss relative to plain PN.

Run:
    python experiments/p1_guidance_comparison.py [--no-show]
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from intercept.adversary import scripted
from intercept.core import Engagement, Entity, PointMass2D
from intercept.core.entities import Controller
from intercept.guidance import AugmentedPN, true_pn
from intercept.viz import compare_engagements_2d

Array = NDArray[np.float64]
GALLERY = Path(__file__).resolve().parents[1] / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"


def pure_pursuit(target_name: str, gain: float = 8.0, a_max: float = 100.0) -> Controller:
    """The P0 placeholder: steer velocity toward the line of sight (no lead)."""

    def controller(t: float, own: Array, world: Mapping[str, Array]) -> Array:
        v = own[2:4]
        speed = float(np.linalg.norm(v))
        if speed < 1e-6:
            return np.zeros(2)
        los = world[target_name][:2] - own[:2]
        err = (np.arctan2(los[1], los[0]) - np.arctan2(v[1], v[0]) + np.pi) % (2 * np.pi) - np.pi
        a_lat = float(np.clip(gain * speed * err, -a_max, a_max))
        v_hat = v / speed
        return a_lat * np.array([-v_hat[1], v_hat[0]])

    return controller


def run_engagement(
    guidance, target_state, *, target_ctrl=None, a_max, speed, kill_radius, dt=0.005, t_max=30.0
):
    tp = np.array(target_state, float)[:2]
    aim = tp / np.linalg.norm(tp)
    interceptor = Entity(
        "interceptor",
        PointMass2D(a_max=a_max),
        np.array([0.0, 0.0, speed * aim[0], speed * aim[1]]),
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


def figure_pn_vs_pursuit(show: bool) -> None:
    target_state = [6500.0, 2500.0, -700.0, 0.0]  # ~Mach 2 crossing target
    a_max, speed, kr = 200.0, 1000.0, 15.0  # realistic ~1.45x speeds, modest authority
    results = {
        "Pure pursuit": run_engagement(
            pure_pursuit("target", a_max=a_max),
            target_state,
            a_max=a_max,
            speed=speed,
            kill_radius=kr,
        ),
        "True PN (N=4)": run_engagement(
            true_pn("target", N=4.0), target_state, a_max=a_max, speed=speed, kill_radius=kr
        ),
    }
    compare_engagements_2d(
        results,
        title="PN vs. Pure Pursuit — crossing target, modest authority (a_max=200 m/s²)",
        save_path=FIG / "p1_pn_vs_pursuit.png",
        show=show,
    )
    for label, r in results.items():
        print(f"  {label:22s}: {r.reason.name:9s} miss={r.miss_distance:7.2f} m")


def figure_pn_vs_apn(show: bool) -> None:
    target_state = [6500.0, 1800.0, -700.0, 0.0]
    maneuver = scripted.step_maneuver(accel=200.0, t_start=0.0)  # hard, sustained turn (~20 g)
    a_max, speed, kr = 250.0, 1000.0, 0.5  # tiny kill radius => compare terminal miss
    results = {
        "True PN (N=4)": run_engagement(
            true_pn("target", N=4.0),
            target_state,
            target_ctrl=maneuver,
            a_max=a_max,
            speed=speed,
            kill_radius=kr,
        ),
        "Augmented PN (N=4)": run_engagement(
            AugmentedPN("target", N=4.0),
            target_state,
            target_ctrl=maneuver,
            a_max=a_max,
            speed=speed,
            kill_radius=kr,
        ),
    }
    compare_engagements_2d(
        results,
        title="True PN vs. Augmented PN — hard-turning target (terminal miss distance)",
        save_path=FIG / "p1_pn_vs_apn_maneuvering.png",
        show=show,
    )
    for label, r in results.items():
        print(f"  {label:22s}: {r.reason.name:9s} miss={r.miss_distance:7.3f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 guidance comparison")
    parser.add_argument("--no-show", action="store_true", help="headless (save only)")
    args = parser.parse_args()
    show = not args.no_show

    print("Figure 1: PN vs. Pure Pursuit")
    figure_pn_vs_pursuit(show)
    print("Figure 2: True PN vs. Augmented PN (maneuvering target)")
    figure_pn_vs_apn(show)
    print(f"Figures saved to: {GALLERY}")


if __name__ == "__main__":
    main()
