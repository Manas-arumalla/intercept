"""P26 — advanced 3-D estimation: IMM in the guidance loop + INS platform error.

Two things the estimation layer now does (ADR-0019):

1. **3-D IMM in the guidance loop.** A seeker-on-interceptor `Radar3D` + a dimension-generic CV/CA
   `IMM` feed an Augmented-PN-3D law via `EstimatingGuidance` — the sense->estimate->guide loop
   closed in 3-D against a maneuvering target at realistic comparable speeds.
2. **INS platform error.** The seeker measures the *true* relative geometry, but the filter places
   the target using the interceptor's *believed* position (`INSError`: bias + drift). As the
   platform's navigation error grows, the target estimate — and the miss — degrade gracefully.

Figure: (left) miss distance & P(intercept) vs. INS drift rate; (right) a 3-D intercept with the
true vs. IMM-estimated target track.

Run:
    python experiments/p26_estimation_advanced.py [--trials 16] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from intercept.core import Engagement, Entity, PointMass3D
from intercept.core.aero import G0
from intercept.estimation import INSError, make_cv_ca_imm
from intercept.guidance import EstimatingGuidance, augmented_pn_3d
from intercept.sensors import Radar3D

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "gallery" / "figures"
RESULTS = ROOT / "results"

# Realistic comparable speeds (interceptor ~Mach 3, target ~Mach 2 = ~1.43x); long range so the
# filter has time to converge before intercept.
I0 = np.array([0.0, 0.0, 0.0, 1000.0, 0.0, 60.0])
T0 = np.array([13000.0, 3000.0, 4000.0, -700.0, 40.0, 0.0])
A_MAX = 40 * G0


def _radar() -> Radar3D:
    return Radar3D(sigma_range=20.0, sigma_az=0.004, sigma_el=0.004)


def _run(ins_drift: float, seed: int):
    rng = np.random.default_rng(seed)
    pe = (
        INSError(3, np.random.default_rng(seed + 99), bias_std=40.0, drift_rate=ins_drift)
        if ins_drift > 0
        else None
    )
    guid = EstimatingGuidance(
        "target",
        _radar(),
        lambda x0, P0: make_cv_ca_imm(x0, P0, ndim=3),
        augmented_pn_3d("target", N=4.0),
        rng,
        platform_error=pe,
    )
    intc = Entity(
        "interceptor", PointMass3D(a_max=A_MAX), I0.copy(), controller=guid, role="interceptor"
    )
    tgt = Entity("target", PointMass3D(), T0.copy(), role="target")
    res = Engagement(
        [intc, tgt],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=30.0,
        kill_radius=25.0,
    ).run()
    return res


def _estimated_track(res):
    """Reconstruct the IMM's estimated target track along the actual interceptor path."""
    rng = np.random.default_rng(0)
    radar = _radar()
    ipath = res.states["interceptor"][:, :3]
    tpath = res.states["target"][:, :3]
    times = res.times
    est = None
    last_t = None
    out = []
    for k in range(len(times)):
        sp, tp = ipath[k], tpath[k]
        z = radar.measure(sp, tp, rng)
        if est is None:
            pos0 = np.asarray(radar.invert(sp, z), dtype=float)[:3]
            x0 = np.concatenate([pos0, np.zeros(6)])
            p0 = np.diag([50.0**2] * 3 + [300.0**2] * 3 + [100.0**2] * 3)
            est = make_cv_ca_imm(x0, p0, ndim=3)
        else:
            dt = times[k] - last_t
            if dt > 0:
                est.predict(dt)
            est.update(z, radar, sp)
        last_t = times[k]
        out.append(est.target_state()[:3])
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced 3-D estimation demo")
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    drifts = [0.0, 2.0, 5.0, 10.0, 20.0]
    p_int, miss_med = [], []
    print("=" * 56)
    print("3-D IMM-in-loop vs. INS platform drift (realistic speeds)")
    print("=" * 56)
    for d in drifts:
        rs = [_run(d, s) for s in range(args.trials)]
        p = float(np.mean([r.intercepted for r in rs]))
        mm = float(np.median([r.miss_distance for r in rs]))
        p_int.append(p)
        miss_med.append(mm)
        print(f"  INS drift {d:5.1f} m/s : P(intercept)={p:.2f}  miss_median={mm:7.1f} m")
    print("=" * 56)

    res0 = _run(0.0, 0)
    est_track = _estimated_track(res0)

    fig = plt.figure(figsize=(13, 5.5))
    ax1 = fig.add_subplot(1, 2, 1)
    c = "#1f77b4"
    ax1.plot(drifts, miss_med, "o-", color=c, lw=2, label="median miss")
    ax1.set_xlabel("INS drift rate (m/s)")
    ax1.set_ylabel("median miss distance (m)", color=c)
    ax1.tick_params(axis="y", labelcolor=c)
    ax1.axhline(25.0, ls="--", color="gray", lw=1, label="kill radius (25 m)")
    ax1.set_title("Platform INS error degrades the estimate (and the miss)")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(drifts, p_int, "s--", color="#d62728", lw=2, label="P(intercept)")
    ax2.set_ylabel("P(intercept)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(-0.05, 1.08)
    ax1.legend(loc="center right", fontsize=8)

    axb = fig.add_subplot(1, 2, 2, projection="3d")
    ip = res0.states["interceptor"][:, :3]
    tp = res0.states["target"][:, :3]
    axb.plot(ip[:, 0], ip[:, 1], ip[:, 2], "-", color="#19e6ff", lw=2, label="interceptor")
    axb.plot(tp[:, 0], tp[:, 1], tp[:, 2], "-", color="#ff4d6d", lw=2, label="target (true)")
    axb.plot(
        est_track[:, 0],
        est_track[:, 1],
        est_track[:, 2],
        "--",
        color="#c6ff00",
        lw=1.5,
        label="target (IMM estimate)",
    )
    axb.scatter(*ip[-1], color="#c6ff00", s=80, marker="*", label="intercept")
    axb.set_title("3-D IMM in the guidance loop")
    axb.set_xlabel("x (m)")
    axb.set_ylabel("y (m)")
    axb.set_zlabel("z (m)")
    axb.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p26_estimation_advanced.png", dpi=150)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p26_estimation_advanced.csv").write_text(
        "ins_drift_mps,p_intercept,miss_median_m\n"
        + "\n".join(f"{d},{p:.3f},{m:.1f}" for d, p, m in zip(drifts, p_int, miss_med, strict=True))
    )
    print(f"Figure: {FIG / 'p26_estimation_advanced.png'}")
    plt.show() if show else plt.close(fig)


if __name__ == "__main__":
    main()
