"""P0 demo — first end-to-end engagement.

Exercises the whole simulation core: a faster interceptor runs down a constant-velocity target
using a minimal *pure-pursuit* controller (steer velocity toward the line of sight). Pure
pursuit is a deliberately trivial placeholder — Proportional Navigation and its variants arrive
in P1 (``intercept.guidance.pn``) and will be benchmarked against everything else. The point of
P0 is to validate dynamics + RK4 integration + the engagement loop + plotting, and to produce
the project's first trajectory figure.

Run:
    python experiments/p0_first_engagement.py            # show + save figure
    python experiments/p0_first_engagement.py --no-show  # headless (CI), still saves
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from intercept.core import Engagement, Entity, PointMass2D
from intercept.viz import plot_engagement_2d

Array = NDArray[np.float64]


def pure_pursuit(target_name: str, gain: float = 5.0, a_max: float = 300.0):
    """Minimal pure-pursuit controller (placeholder; real guidance laws land in P1).

    Commands lateral acceleration perpendicular to the interceptor's velocity, proportional to
    the heading error between the velocity vector and the line of sight to the target. This
    turns the velocity toward the target while keeping speed ~constant.
    """

    def controller(t: float, own: Array, world: Mapping[str, Array]) -> Array:
        v = own[2:4]
        speed = float(np.linalg.norm(v))
        if speed < 1e-6:
            return np.zeros(2)
        los = world[target_name][:2] - own[:2]
        desired = np.arctan2(los[1], los[0])
        current = np.arctan2(v[1], v[0])
        err = (desired - current + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        a_lat = float(np.clip(gain * speed * err, -a_max, a_max))
        v_hat = v / speed
        perp = np.array([-v_hat[1], v_hat[0]])  # +90 deg from velocity
        return a_lat * perp

    return controller


def build_engagement() -> Engagement:
    a_max = 600.0  # ~60 g acceleration authority
    plant = PointMass2D(a_max=a_max)

    # Target: ~Mach 2 (650 m/s), inbound toward the defended origin from an offset (a gentle
    # lead-pursuit geometry — near head-on, so the trivial pursuit law can still connect).
    target_pos = np.array([5000.0, 700.0])
    target_vel = -650.0 * target_pos / np.linalg.norm(target_pos)
    target = Entity(
        name="target",
        dynamics=PointMass2D(),
        state=np.array([target_pos[0], target_pos[1], target_vel[0], target_vel[1]]),
        role="target",
    )

    # Interceptor: launched aimed *at the target's initial position* so pure pursuit only has to
    # correct for target motion (a gentle lead-pursuit curve into the intercept). Realistic
    # comparable speeds (interceptor only ~1.45x the target); even so this trivial
    # law connects on a non-maneuvering target, while PN will hold *crossing/maneuvering*
    # targets that pure pursuit cannot — the comparison that motivates the benchmark.
    speed = 950.0
    aim = target_pos / np.linalg.norm(target_pos)
    interceptor = Entity(
        name="interceptor",
        dynamics=plant,
        state=np.array([0.0, 0.0, speed * aim[0], speed * aim[1]]),
        controller=pure_pursuit("target", gain=8.0, a_max=a_max),
        role="interceptor",
    )

    return Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.005,
        t_max=30.0,
        kill_radius=15.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 first-engagement demo")
    parser.add_argument("--no-show", action="store_true", help="do not open a window (headless)")
    parser.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parents[1] / "gallery" / "figures" / "p0_first_engagement.png"
        ),
        help="output figure path",
    )
    args = parser.parse_args()

    result = build_engagement().run()

    print("=" * 60)
    print(f"Outcome           : {result.reason.name}")
    print(f"Intercepted       : {result.intercepted}")
    print(f"Miss distance     : {result.miss_distance:.3f} m")
    if result.intercept_time is not None:
        print(f"Intercept time    : {result.intercept_time:.3f} s")
    print(f"Closest approach  : t = {result.closest_approach_time:.3f} s")
    print(f"Interceptor effort: {result.control_effort('interceptor'):.1f} (m/s^2)^2 s")
    print(f"Duration          : {result.duration:.3f} s ({len(result.times)} steps)")
    print("=" * 60)

    plot_engagement_2d(result, save_path=args.out, show=not args.no_show)
    print(f"Figure saved to: {args.out}")


if __name__ == "__main__":
    main()
