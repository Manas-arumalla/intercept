"""P18 — 3-D target tracking: UKF/EKF from a noisy 3-D radar (range, azimuth, elevation).

The 2-D estimation suite is extended to three dimensions (ADR-0013): dimension-generic motion
models (`ncv_model`/`nca_model` with `ndim=3`), a dimension-generic EKF/UKF (they slice `pos_dim`
position components from the state), and a `Radar3D` sensor. A 3-D barrel-rolling target is tracked
from noisy angle+range measurements; a nearly-constant-velocity EKF (which cannot represent the
maneuver) is compared with a nearly-constant-acceleration UKF.

Run:
    python experiments/p18_estimation_3d.py [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from intercept.adversary import barrel_roll
from intercept.core import G0, AeroMissile3D
from intercept.core.integrators import integrate_rk4
from intercept.estimation import EKF, UKF, make_cv_ca_imm
from intercept.estimation.models import nca_model, ncv_model
from intercept.sensors import Radar3D

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"


def simulate(seed: int = 0):
    rng = np.random.default_rng(seed)
    plant = AeroMissile3D(a_max=20 * G0, tau=0.3)
    x = plant.initial_state([8000.0, 1500.0, 4000.0], [-700.0, 0.0, -30.0])
    ctrl = barrel_roll(accel=15 * G0, rate=1.3)
    radar = Radar3D(sigma_range=25.0, sigma_az=0.008, sigma_el=0.008)
    sp = np.zeros(3)
    dt, n = 0.05, 150

    z0 = radar.measure(sp, x[:3], rng)
    p0 = radar.invert(sp, z0)
    x0 = np.concatenate([p0, [-700.0, 0.0, -30.0], [0.0, 0.0, 0.0]])
    P0 = np.diag([200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 100.0, 100.0, 100.0]) ** 2
    ekf = EKF(lambda d: ncv_model(d, q=5.0, ndim=3), x0.copy(), P0.copy())
    ukf = UKF(lambda d: nca_model(d, q=200.0, ndim=3), x0.copy(), P0.copy())
    imm = make_cv_ca_imm(x0.copy(), P0.copy(), q_cv=5.0, q_ca=200.0, ndim=3)

    truth, est_ekf, est_ukf, est_imm, meas, times, mu_ca = [], [], [], [], [], [], []
    for k in range(n):
        u = ctrl(k * dt, x, {})
        x = integrate_rk4(plant, k * dt, x, u, dt)
        z = radar.measure(sp, x[:3], rng)
        for f in (ekf, ukf, imm):
            f.predict(dt)
            f.update(z, radar, sp)
        truth.append(x[:3].copy())
        est_ekf.append(ekf.x[:3].copy())
        est_ukf.append(ukf.x[:3].copy())
        est_imm.append(imm.x[:3].copy())
        mu_ca.append(imm.mode_probabilities[1])  # P(maneuver model active)
        meas.append(radar.invert(sp, z))
        times.append(k * dt)
    return (
        np.array(times),
        np.array(truth),
        np.array(est_ekf),
        np.array(est_ukf),
        np.array(est_imm),
        np.array(meas),
        np.array(mu_ca),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="3-D tracking demo")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    t, truth, e_ekf, e_ukf, e_imm, meas, mu_ca = simulate()
    err_ekf = np.linalg.norm(e_ekf - truth, axis=1)
    err_ukf = np.linalg.norm(e_ukf - truth, axis=1)
    err_imm = np.linalg.norm(e_imm - truth, axis=1)
    err_raw = np.linalg.norm(meas - truth, axis=1)
    warm = t > 1.0
    print("=" * 60)
    print("3-D tracking RMSE (after warm-up), barrel-rolling target")
    print(f"  raw radar inversion : {np.sqrt(np.mean(err_raw[warm] ** 2)):6.1f} m")
    print(f"  NCV EKF             : {np.sqrt(np.mean(err_ekf[warm] ** 2)):6.1f} m")
    print(f"  NCA UKF             : {np.sqrt(np.mean(err_ukf[warm] ** 2)):6.1f} m")
    print(f"  CV/CA IMM           : {np.sqrt(np.mean(err_imm[warm] ** 2)):6.1f} m")
    print(f"  IMM P(maneuver) end : {mu_ca[-1]:.2f}")
    print("=" * 60)

    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], "-", color="#222", lw=2, label="truth")
    ax.plot(e_imm[:, 0], e_imm[:, 1], e_imm[:, 2], "--", color="#2ca02c", lw=1.5, label="CV/CA IMM")
    ax.scatter(
        meas[::5, 0],
        meas[::5, 1],
        meas[::5, 2],
        s=6,
        c="#d62728",
        alpha=0.3,
        label="radar (inverted)",
    )
    ax.set_title("3-D barrel-roll: truth vs. IMM estimate")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t, err_raw, color="#d62728", alpha=0.35, lw=1.0, label="raw radar")
    ax2.plot(t, err_ekf, color="#ff7f0e", lw=1.6, label="NCV EKF")
    ax2.plot(t, err_ukf, color="#1f77b4", lw=1.6, label="NCA UKF")
    ax2.plot(t, err_imm, color="#2ca02c", lw=1.6, label="CV/CA IMM")
    ax2.set_title("Position error vs. time")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("position error (m)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.tight_layout()

    GALLERY.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p18_estimation_3d.png", dpi=150)
    print(f"Figure: {FIG / 'p18_estimation_3d.png'}")
    plt.show() if show else plt.close(fig)


if __name__ == "__main__":
    main()
