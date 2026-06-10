"""P3 demo — tracking and the estimation-coupled guidance study.

Figure 1 (``p3_imm_tracking.png``): a target flies straight then makes a hard turn at t=3 s,
observed by a noisy radar from the origin. Three trackers — EKF(NCV), EKF(NCA), and the IMM —
are compared by position error over time, with the IMM's mode probabilities below. The IMM gets
the best of both: low error while straight (NCV) and fast adaptation through the turn (NCA).

Figure 2 (``p3_guidance_vs_noise.png``): the estimation-coupled study (a verified research gap) —
how guidance performance degrades as seeker noise grows. For a sweep of radar noise levels we run a
Monte-Carlo of EstimatingGuidance(EKF + True PN) engagements and plot miss distance and
probability of intercept vs. noise, against the perfect-information baseline.

Run:
    python experiments/p3_estimation.py [--trials 60] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from intercept.benchmark import ParametricScenario
from intercept.core import Engagement, Entity, PointMass2D
from intercept.estimation import EKF, make_cv_ca_imm, nca_model, ncv_model
from intercept.guidance import EstimatingGuidance, true_pn
from intercept.sensors import Radar

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"


# --------------------------------------------------------------------------- Figure 1: tracking


def _simulate_truth(dt: float, n: int) -> np.ndarray:
    """Target: constant velocity, then a hard left turn (constant accel) from t=3 s."""
    states = np.zeros((n, 6))
    s = np.array([2500.0, -200.0, -180.0, 60.0, 0.0, 0.0])
    for k in range(n):
        t = k * dt
        s = s.copy()
        s[4:6] = np.array([40.0, 90.0]) if t >= 3.0 else np.array([0.0, 0.0])
        s[2:4] = s[2:4] + s[4:6] * dt
        s[:2] = s[:2] + s[2:4] * dt
        states[k] = s
    return states


def figure_tracking(show: bool) -> None:
    dt, n = 0.1, 80
    truth = _simulate_truth(dt, n)
    radar = Radar(sigma_range=15.0, sigma_bearing=0.005)
    sp = np.array([0.0, 0.0])
    rng = np.random.default_rng(0)

    x0 = np.array([truth[0, 0], truth[0, 1], 0, 0, 0, 0])
    P0 = np.diag([100.0, 100.0, 300**2, 300**2, 100**2, 100**2])
    trackers = {
        "EKF (NCV)": EKF(lambda d: ncv_model(d, 1.0), x0.copy(), P0.copy()),
        "EKF (NCA)": EKF(lambda d: nca_model(d, 200.0), x0.copy(), P0.copy()),
        "IMM (NCV+NCA)": make_cv_ca_imm(x0.copy(), P0.copy()),
    }
    errs = {k: [] for k in trackers}
    imm_modes = []
    times = np.arange(n) * dt
    for k in range(n):
        z = radar.measure(sp, truth[k, :2], rng)
        for name, est in trackers.items():
            if k > 0:
                est.predict(dt)
            est.update(z, radar, sp)
            errs[name].append(np.linalg.norm(est.position - truth[k, :2]))
        imm_modes.append(trackers["IMM (NCV+NCA)"].mode_probabilities.copy())
    imm_modes = np.array(imm_modes)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1], sharex=True)
    for name, e in errs.items():
        ax1.plot(times, e, lw=1.8, label=f"{name} (mean {np.mean(e[10:]):.1f} m)")
    ax1.axvline(3.0, color="k", ls="--", alpha=0.5)
    ax1.text(3.05, ax1.get_ylim()[1] * 0.9, "target turns", fontsize=9)
    ax1.set_ylabel("position error (m)")
    ax1.set_title("Tracking a maneuvering target: EKF vs. IMM (noisy radar)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, imm_modes[:, 0], lw=1.8, label="P(NCV / quiescent)")
    ax2.plot(times, imm_modes[:, 1], lw=1.8, label="P(NCA / maneuver)")
    ax2.axvline(3.0, color="k", ls="--", alpha=0.5)
    ax2.set_ylabel("IMM mode prob.")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="center right")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "p3_imm_tracking.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  IMM mean error (post-spinup): {np.mean(errs['IMM (NCV+NCA)'][10:]):.1f} m")


# ----------------------------------------------------- Figure 2: estimation-coupled guidance study


def _run_trial(
    scenario: ParametricScenario, sigma_range: float, sigma_bearing: float, seed_pair: list[int]
) -> tuple[bool, float]:
    rng = np.random.default_rng(np.random.SeedSequence(seed_pair))
    spec = scenario.sample(rng)
    radar = Radar(sigma_range=sigma_range, sigma_bearing=sigma_bearing)
    guidance = EstimatingGuidance(
        "target",
        radar,
        lambda x0, P0: EKF(lambda d: nca_model(d, 50.0), x0, P0),
        true_pn("target", N=4.0),
        rng,
    )
    interceptor = Entity(
        "interceptor",
        PointMass2D(a_max=spec.interceptor_a_max),
        spec.interceptor_state,
        controller=guidance,
        role="interceptor",
    )
    target = Entity(
        "target", PointMass2D(), spec.target_state, controller=spec.target_controller, role="target"
    )
    res = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=spec.dt,
        t_max=spec.t_max,
        kill_radius=spec.kill_radius,
    ).run()
    return res.intercepted, res.miss_distance


def figure_guidance_vs_noise(show: bool, trials: int) -> None:
    scenario = ParametricScenario(
        name="estcoupled",
        interceptor_speed=1000,
        interceptor_a_max=250,
        target_speed=700,
        range_min=6000,
        range_max=8000,
        offset_min=-800,
        offset_max=800,
        target_heading_deg=150,
        kill_radius=10.0,
        dt=0.01,
        t_max=40.0,
    )
    sigma_ranges = [1.0, 10.0, 25.0, 50.0, 100.0, 200.0]
    bearing_scale = 0.0005  # sigma_bearing grows with sigma_range
    p_int, miss_med = [], []
    for i, sr in enumerate(sigma_ranges):
        sb = max(0.001, bearing_scale * sr)
        outcomes = [_run_trial(scenario, sr, sb, [i, t]) for t in range(trials)]
        hits = [o[0] for o in outcomes]
        misses = [o[1] for o in outcomes]
        p_int.append(float(np.mean(hits)))
        miss_med.append(float(np.median(misses)))
        print(f"  sigma_r={sr:6.1f} m: P_int={p_int[-1]:.2f}  miss_med={miss_med[-1]:7.2f} m")

    fig, ax1 = plt.subplots(figsize=(9, 6))
    color1 = "#1f77b4"
    ax1.plot(sigma_ranges, p_int, "o-", color=color1, lw=2, label="P(intercept)")
    ax1.set_xlabel("radar range noise σ_r (m)")
    ax1.set_ylabel("P(intercept)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1.05)
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.plot(sigma_ranges, miss_med, "s--", color=color2, lw=2, label="median miss")
    ax2.set_ylabel("median miss distance (m)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(
        "Estimation-coupled guidance: performance vs. seeker noise\n"
        "(EKF + True PN, radar range+bearing)"
    )
    fig.tight_layout()
    fig.savefig(FIG / "p3_guidance_vs_noise.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 estimation demo")
    parser.add_argument("--trials", type=int, default=60, help="trials per noise level (figure 2)")
    parser.add_argument("--no-show", action="store_true", help="headless (save only)")
    args = parser.parse_args()
    show = not args.no_show

    print("Figure 1: IMM vs EKF tracking of a maneuvering target")
    figure_tracking(show)
    print("Figure 2: estimation-coupled guidance (miss vs seeker noise)")
    figure_guidance_vs_noise(show, args.trials)
    print(f"Figures saved to: {GALLERY}")


if __name__ == "__main__":
    main()
