"""P10 demo — L3 realistic aero-propulsive engagement (vertical plane).

A surface-launched interceptor boosts from low speed, climbs through the atmosphere, and uses
Augmented PN to run down a fast, weaving, high-altitude target — under faithful physics: ISA
atmosphere, Mach-dependent drag, rocket boost/sustain/coast with mass burn-off, and turn capability
limited by aerodynamic lift (dynamic pressure). Plots the geometry plus speed / Mach / available-g
time histories that show the realism (no fixed speed or g — they emerge from the flight condition).

Run:
    python experiments/p10_realistic_demo.py [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from intercept.adversary import weave
from intercept.core import Engagement, Entity, RealisticMissile2D
from intercept.core.atmosphere import mach as mach_at
from intercept.guidance import AugmentedPN

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"


def build() -> tuple[Engagement, RealisticMissile2D, RealisticMissile2D]:
    interceptor_dyn = RealisticMissile2D()  # boost-sustain SAM
    target_dyn = RealisticMissile2D.target()  # fast maneuverable threat
    # Launch climbing toward the target; boost provides the speed (initial speed is low).
    tgt_pos = np.array([7000.0, 4500.0])
    aim = tgt_pos / np.linalg.norm(tgt_pos)
    interceptor = Entity(
        "interceptor",
        interceptor_dyn,
        interceptor_dyn.initial_state([0.0, 0.0], 150.0 * aim),
        controller=AugmentedPN("target", N=4.0),
        role="interceptor",
    )
    target = Entity(
        "target",
        target_dyn,
        target_dyn.initial_state([7000.0, 4500.0], [-650.0, -80.0]),
        controller=weave(amplitude=12 * 9.81, frequency=0.3),
        role="target",
    )
    eng = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.005,
        t_max=30.0,
        kill_radius=20.0,
    )
    return eng, interceptor_dyn, target_dyn


def main() -> None:
    parser = argparse.ArgumentParser(description="L3 realistic engagement demo")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    eng, idyn, _ = build()
    res = eng.run()
    t = res.times
    si = res.states["interceptor"]
    st = res.states["target"]
    spd = np.linalg.norm(si[:, 2:4], axis=1)
    mach = np.array([mach_at(s, alt) for s, alt in zip(spd, si[:, 1], strict=True)])
    gmax = np.array([idyn.max_lateral_accel(si[k], t[k]) / 9.80665 for k in range(len(t))])

    print("=" * 56)
    print(f"Outcome        : {res.reason.name}")
    print(f"Miss distance  : {res.miss_distance:.2f} m")
    if res.intercept_time is not None:
        print(f"Intercept time : {res.intercept_time:.2f} s")
    print(f"Interceptor peak speed : {spd.max():.0f} m/s  (Mach {mach.max():.1f})")
    print(f"Available g at launch / intercept : {gmax[0]:.0f} g / {gmax[-1]:.0f} g")
    print("=" * 56)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    ax1.plot(si[:, 0] / 1000, si[:, 1] / 1000, color="#1f77b4", lw=2, label="interceptor")
    ax1.plot(st[:, 0] / 1000, st[:, 1] / 1000, color="#d62728", lw=2, label="target (weaving)")
    ax1.plot(si[0, 0] / 1000, si[0, 1] / 1000, "o", color="#1f77b4")
    ax1.plot(st[0, 0] / 1000, st[0, 1] / 1000, "o", color="#d62728")
    ti = int(np.argmin(np.abs(t - res.closest_approach_time)))
    ax1.plot(
        si[ti, 0] / 1000,
        si[ti, 1] / 1000,
        "*",
        color="#2ca02c",
        ms=18,
        label=f"intercept ({res.miss_distance:.0f} m)",
    )
    ax1.set_xlabel("down-range (km)")
    ax1.set_ylabel("altitude (km)")
    ax1.set_title("L3 engagement — vertical plane")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(t, mach, color="#1f77b4", lw=2, label="interceptor Mach")
    ax2.plot(t, gmax, color="#ff7f0e", lw=2, label="available lateral g")
    ax2.axvspan(0, idyn.t_boost, color="red", alpha=0.08, label="boost")
    ax2.axvspan(
        idyn.t_boost, idyn.t_boost + idyn.t_sustain, color="orange", alpha=0.06, label="sustain"
    )
    ax2.set_xlabel("time (s)")
    ax2.set_title("Realism: Mach & available-g emerge from physics (atmosphere, mass, q)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "p10_realistic_engagement.png", dpi=140)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Figure: {FIG / 'p10_realistic_engagement.png'}")


if __name__ == "__main__":
    main()
