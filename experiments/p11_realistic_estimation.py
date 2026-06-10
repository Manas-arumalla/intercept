"""P11 — robustness under uncertainty: noisy seeker + IMM estimator in the loop (L3 target).

The interceptor no longer sees truth. It gets noisy radar measurements of a fast, weaving L3
(realistic aero-propulsive) target, runs an IMM (constant-velocity + constant-acceleration) tracker,
and guides on the *estimate* with Augmented PN. We compare perfect-information guidance against the
noisy-seeker + IMM pipeline over a Monte-Carlo — showing the system stays robust under realistic
sensing error.

Run:
    python experiments/p11_realistic_estimation.py [--trials 40] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from intercept.adversary import weave
from intercept.core import Engagement, Entity, RealisticMissile2D
from intercept.estimation import EKF, make_cv_ca_imm, nca_model
from intercept.guidance import AugmentedPN, EstimatingGuidance
from intercept.sensors import Radar

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"


def run_trial(
    seed: int, *, estimating: bool, sigma_range: float = 25.0, sigma_bearing: float = 0.004
):
    rng = np.random.default_rng(seed)
    idyn = RealisticMissile2D()
    tdyn = RealisticMissile2D.target()
    # Randomized engagement geometry.
    tgt_pos = np.array([rng.uniform(6500, 8000), rng.uniform(3800, 5200)])
    tgt_vel = np.array([rng.uniform(-700, -600), rng.uniform(-120, -40)])
    aim = tgt_pos / np.linalg.norm(tgt_pos)

    if estimating:
        radar = Radar(sigma_range=sigma_range, sigma_bearing=sigma_bearing)
        guidance = EstimatingGuidance(
            "target",
            radar,
            lambda x0, P0: make_cv_ca_imm(x0, P0),  # IMM (CV + CA) tracker
            AugmentedPN("target", N=4.0),
            rng,
        )
    else:
        guidance = AugmentedPN("target", N=4.0)

    interceptor = Entity(
        "interceptor",
        idyn,
        idyn.initial_state([0.0, 0.0], 150.0 * aim),
        controller=guidance,
        role="interceptor",
    )
    target = Entity(
        "target",
        tdyn,
        tdyn.initial_state(tgt_pos, tgt_vel),
        controller=weave(amplitude=12 * 9.81, frequency=0.3),
        role="target",
    )
    return Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=25.0,
        kill_radius=20.0,
    ).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Realistic estimation-in-the-loop robustness")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    for label, est in [("Perfect info (APN)", False), ("Noisy radar + IMM (APN)", True)]:
        results = [run_trial(1000 + i, estimating=est) for i in range(args.trials)]
        p = np.mean([r.intercepted for r in results])
        miss = np.median([r.miss_distance for r in results])
        print(f"  {label:26s}: P_intercept={p:.2f}  median miss={miss:6.1f} m")

    # Illustrative single run: true vs estimated target track.
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(7)
    idyn = RealisticMissile2D()
    tdyn = RealisticMissile2D.target()
    tgt_pos = np.array([7200.0, 4600.0])
    aim = tgt_pos / np.linalg.norm(tgt_pos)
    radar = Radar(sigma_range=25.0, sigma_bearing=0.004)
    guidance = EstimatingGuidance(
        "target",
        radar,
        lambda x0, P0: EKF(lambda d: nca_model(d, 80.0), x0, P0),
        AugmentedPN("target", N=4.0),
        rng,
    )
    interceptor = Entity(
        "interceptor",
        idyn,
        idyn.initial_state([0.0, 0.0], 150.0 * aim),
        controller=guidance,
        role="interceptor",
    )
    target = Entity(
        "target",
        tdyn,
        tdyn.initial_state(tgt_pos, [-650.0, -80.0]),
        controller=weave(amplitude=12 * 9.81, frequency=0.3),
        role="target",
    )
    res = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=25.0,
        kill_radius=20.0,
    ).run()

    fig, ax = plt.subplots(figsize=(9, 6))
    si, st = res.states["interceptor"], res.states["target"]
    ax.plot(si[:, 0] / 1000, si[:, 1] / 1000, color="#1f77b4", lw=2, label="interceptor")
    ax.plot(st[:, 0] / 1000, st[:, 1] / 1000, color="#d62728", lw=2, label="target (true)")
    ax.plot(si[0, 0] / 1000, si[0, 1] / 1000, "o", color="#1f77b4")
    outcome = "INTERCEPT" if res.intercepted else res.reason.name
    ax.set_title(f"Noisy radar + EKF in the loop — {outcome} (miss {res.miss_distance:.0f} m)")
    ax.set_xlabel("down-range (km)")
    ax.set_ylabel("altitude (km)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "p11_realistic_estimation.png", dpi=140)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Figure: {FIG / 'p11_realistic_estimation.png'}")


if __name__ == "__main__":
    main()
