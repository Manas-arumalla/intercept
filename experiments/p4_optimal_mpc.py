"""P4 demo — optimal / sliding-mode / MPC guidance.

Figure 1 (``p4_law_comparison.png``): True PN, Optimal (OGL), Sliding-Mode, and NMPC on the same
weaving-target engagement, overlaid with miss distance and effort.

Figure 2 (``p4_impact_angle.png``): NMPC with a terminal **impact-angle** objective — the same
target struck from several requested approach headings, which closed-form PN cannot do. Each
trajectory is annotated with its requested vs. achieved terminal heading.

Run:
    python experiments/p4_optimal_mpc.py [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from intercept.adversary import scripted
from intercept.core import Engagement, Entity, PointMass2D
from intercept.guidance import optimal_guidance, sliding_mode, true_pn
from intercept.guidance.mpc import MPCGuidance, has_casadi
from intercept.viz import compare_engagements_2d

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"


def _engage(
    guidance,
    target_state,
    *,
    target_ctrl=None,
    a_max=150.0,
    speed=1000.0,
    kill_radius=10.0,
    dt=0.01,
    t_max=30.0,
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


def figure_law_comparison(show: bool) -> None:
    target_state = [6500.0, 1800.0, -700.0, 0.0]  # ~Mach 2 weaving target
    maneuver = scripted.weave(amplitude=150.0, frequency=0.3)  # ~15 g
    a_max = 250.0
    results = {
        "True PN (N=4)": _engage(
            true_pn("target", N=4.0), target_state, target_ctrl=maneuver, a_max=a_max
        ),
        "Optimal (OGL, N'=3)": _engage(
            optimal_guidance("target", augment=True),
            target_state,
            target_ctrl=maneuver,
            a_max=a_max,
        ),
        "Sliding-mode": _engage(
            sliding_mode("target", eta=80.0), target_state, target_ctrl=maneuver, a_max=a_max
        ),
    }
    if has_casadi():
        results["NMPC"] = _engage(
            MPCGuidance("target", a_max=a_max, horizon=3.0, replan_every=5),
            target_state,
            target_ctrl=maneuver,
            a_max=a_max,
        )
    compare_engagements_2d(
        results,
        title="Optimal vs. classical vs. MPC — weaving target (a_max=250 m/s²)",
        save_path=FIG / "p4_law_comparison.png",
        show=show,
    )
    for label, r in results.items():
        eff = r.control_effort("interceptor")
        print(f"  {label:22s}: {r.reason.name:9s} miss={r.miss_distance:6.2f} m  effort={eff:8.0f}")


def figure_impact_angle(show: bool) -> None:
    if not has_casadi():
        print("  (skipped — CasADi not installed)")
        return
    target_state = [6500.0, 0.0, -650.0, 0.0]  # ~Mach 2 inbound target
    desired_angles = [45.0, 15.0, -15.0, -45.0]
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728"]

    for ang, color in zip(desired_angles, colors, strict=True):
        guidance = MPCGuidance(
            "target",
            a_max=300.0,
            horizon=8.0,
            n_steps=30,
            replan_every=5,
            impact_angle_deg=ang,
            w_angle=15.0,
            w_terminal=12.0,
        )
        res = _engage(
            guidance, target_state, a_max=392.0, speed=1000.0, kill_radius=15.0, t_max=30.0
        )
        traj = res.states["interceptor"]
        v_final = traj[-1, 2:4]
        achieved = np.degrees(np.arctan2(v_final[1], v_final[0]))
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            "-",
            color=color,
            lw=1.8,
            label=f"want {ang:+.0f}° → got {achieved:+.0f}° (miss {res.miss_distance:.0f} m)",
        )
        ax.plot(traj[0, 0], traj[0, 1], "o", color=color, ms=6)
        # short arrow showing terminal heading
        vh = v_final / np.linalg.norm(v_final)
        ax.annotate(
            "",
            xy=traj[-1, :2] + vh * 400,
            xytext=traj[-1, :2],
            arrowprops=dict(arrowstyle="->", color=color, lw=2),
        )

    tgt = np.array(target_state[:2])
    ax.plot(tgt[0], tgt[1], "k*", ms=18, label="target")
    ax.plot(0, 0, "ks", ms=8, label="interceptor launch")
    ax.set_title("Impact-angle-constrained NMPC — same target, different approach headings")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "p4_impact_angle.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 optimal/MPC guidance demo")
    parser.add_argument("--no-show", action="store_true", help="headless (save only)")
    args = parser.parse_args()
    show = not args.no_show

    print("Figure 1: law comparison (PN / OGL / SMG / MPC) on a weaving target")
    figure_law_comparison(show)
    print("Figure 2: impact-angle-constrained NMPC")
    figure_impact_angle(show)
    print(f"Figures saved to: {GALLERY}")


if __name__ == "__main__":
    main()
