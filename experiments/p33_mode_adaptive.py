"""P33 — IMM-mode-adaptive guidance: the estimator's maneuver belief arbitrates the law.

`ModeAdaptiveGuidance` (ADR-0026) blends a cheap quiescent law (True PN) with an aggressive maneuver
law (Augmented PN) by the IMM's maneuver-mode probability: it flies efficient PN while the target
cruises and hardens automatically the moment the filter *detects* the break. Against a
cruise-then-break threat (realistic ~Mach 3 vs ~Mach 2) it should match APN's intercept rate at a
fraction of the control effort; PN alone misses the break.

Outputs: a 3-panel figure (belief-coloured trajectory, μ timeline, Monte-Carlo P(intercept)+effort)
and a GIF whose interceptor trail recolours blue→red as the guidance hardens.

Run:
    python experiments/p33_mode_adaptive.py [--trials 30] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from intercept.adversary import step_maneuver, weave
from intercept.core import Engagement, Entity
from intercept.core.aero import G0, AeroMissile2D
from intercept.estimation import EKF, make_cv_ca_imm, nca_model
from intercept.guidance import AugmentedPN, EstimatingGuidance, ModeAdaptiveGuidance, true_pn
from intercept.sensors import Radar

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "gallery" / "figures"
ANIM = ROOT / "gallery" / "animations"
RESULTS = ROOT / "results"
INT_SPEED, TGT_SPEED = 1000.0, 700.0  # ~Mach 3 vs ~Mach 2
INT_AMAX, TGT_AMAX = 40 * G0, 25 * G0
CMAP = plt.cm.coolwarm  # blue = PN (quiet), red = APN (maneuver detected)


def _radar() -> Radar:
    return Radar(sigma_range=15.0, sigma_bearing=0.004)


def _ekf(x0, p0):
    return EKF(lambda d: nca_model(d, 50.0), x0, p0)


def _imm(x0, p0):
    return make_cv_ca_imm(x0, p0, p_stay=0.995)


def _adaptive(rng):
    return ModeAdaptiveGuidance(
        "target", _radar(), _imm, true_pn("target", N=4.0), AugmentedPN("target", N=4.0), rng
    )


def _threat(t_break: float, weave_g: float = 8.0, break_g: float = 20.0):
    """Realistic profile: a visible ~8 g serpentine weave on the cruise-in, then a 20 g break.

    The light weave is sub-maneuver (the IMM should stay quiescent through it — the discrimination
    is part of the test); the break is the real threat the maneuver law must answer.
    """
    cruise = weave(amplitude=weave_g * G0, frequency=0.12)
    brk = step_maneuver(accel=break_g * G0, t_start=t_break)

    def controller(t, own, world):
        return cruise(t, own, world) if t < t_break else brk(t, own, world)

    return controller


def _engage(factory, rng_geo, seed, weave_g: float = 8.0):
    rng = np.random.default_rng(seed)
    idyn = AeroMissile2D(a_max=INT_AMAX, tau=0.2)
    edyn = AeroMissile2D(a_max=TGT_AMAX, tau=0.3)
    downrange = rng_geo.uniform(7500.0, 8500.0)
    offset = rng_geo.uniform(-700.0, 700.0)
    # Late break: the target is quiet for most of the flight — the regime this law targets
    # (ADR-0026 documents the early-sustained-break envelope limit separately).
    t_break = rng_geo.uniform(3.5, 4.5)
    guid = factory(rng)
    tpos = np.array([downrange, offset])
    aim = tpos / np.linalg.norm(tpos)
    intc = Entity(
        "interceptor",
        idyn,
        idyn.initial_state([0.0, 0.0], INT_SPEED * aim),
        controller=guid,
        role="interceptor",
    )
    tgt = Entity(
        "target",
        edyn,
        edyn.initial_state([downrange, offset], [-TGT_SPEED, 0.0]),
        controller=_threat(t_break, weave_g=weave_g),
        role="target",
    )
    res = Engagement(
        [intc, tgt],
        interceptor="interceptor",
        target="target",
        dt=0.01,
        t_max=16.0,
        kill_radius=20.0,
    ).run()
    return res, guid, t_break


def montecarlo(trials: int, seed: int, weave_g: float = 8.0):
    laws = {
        "True PN (est)": lambda rng: EstimatingGuidance(
            "target", _radar(), _ekf, true_pn("target", N=4.0), rng
        ),
        "Augmented PN (est)": lambda rng: EstimatingGuidance(
            "target", _radar(), _ekf, AugmentedPN("target", N=4.0), rng
        ),
        "Mode-adaptive (IMM)": _adaptive,
    }
    stats = {}
    for name, fac in laws.items():
        hits, eff = 0, []
        for s in range(trials):
            rng_geo = np.random.default_rng((seed, s))  # identical geometry across laws (paired)
            res, _, _ = _engage(fac, rng_geo, seed=s, weave_g=weave_g)
            hits += int(res.intercepted)
            eff.append(res.control_effort("interceptor"))
        stats[name] = (hits / trials, float(np.mean(eff)))
        print(f"  {name:22s}: P(int)={stats[name][0]:.2f}  effort={stats[name][1]:10.0f}")
    return stats


def _weights(guid) -> np.ndarray:
    mu = np.array(guid.mu_history)
    g = guid.sharpness
    w = mu[:, 1] ** g / (mu[:, 1] ** g + (1 - mu[:, 1]) ** g)
    return np.column_stack([mu[:, 0], w])


def _showcase_engagement():
    """A representative *intercepting* engagement with a mid-flight break (hardening visible).

    Scans a few seeds/offsets and returns the first intercept — presentation only; the Monte-Carlo
    statistics above are the result."""
    idyn = AeroMissile2D(a_max=INT_AMAX, tau=0.2)
    edyn = AeroMissile2D(a_max=TGT_AMAX, tau=0.3)
    t_break = 3.8
    best = None
    for seed in range(7, 27):
        rng = np.random.default_rng(seed)
        tpos = np.array([8000.0, 300.0 - 50.0 * (seed - 7)])
        aim = tpos / np.linalg.norm(tpos)
        guid = _adaptive(rng)
        intc = Entity(
            "interceptor",
            idyn,
            idyn.initial_state([0.0, 0.0], INT_SPEED * aim),
            controller=guid,
            role="interceptor",
        )
        tgt = Entity(
            "target",
            edyn,
            edyn.initial_state(tpos, [-TGT_SPEED, 0.0]),
            controller=_threat(t_break),
            role="target",
        )
        res = Engagement(
            [intc, tgt],
            interceptor="interceptor",
            target="target",
            dt=0.01,
            t_max=16.0,
            kill_radius=20.0,
        ).run()
        if best is None or res.miss_distance < best[0].miss_distance:
            best = (res, guid)
        if res.intercepted:
            return res, guid, t_break
    return best[0], best[1], t_break


def showcase_figure(stats, show: bool) -> None:
    res, guid, t_break = _showcase_engagement()
    tw = _weights(guid)
    ip = res.states["interceptor"][:, :2]
    tp = res.states["target"][:, :2]
    w_t = np.interp(res.times, tw[:, 0], tw[:, 1])

    fig = plt.figure(figsize=(13.5, 5.2))
    ax1 = fig.add_subplot(1, 3, (1, 2))
    pts = ip.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=CMAP, array=w_t[:-1], lw=2.5)
    ax1.add_collection(lc)
    ax1.plot(tp[:, 0], tp[:, 1], "k--", lw=1.2, label="target")
    kb = np.argmin(np.abs(res.times - t_break))
    ax1.plot(tp[kb, 0], tp[kb, 1], "k^", ms=9, label=f"break (t={t_break:.1f}s)")
    outcome = "intercept" if res.intercepted else "closest approach"
    ax1.plot(
        ip[-1, 0],
        ip[-1, 1],
        "*",
        color="#c6ff00",
        ms=18,
        mec="k",
        label=f"{outcome} (miss {res.miss_distance:.0f} m)",
    )
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title(
        "Interceptor trail coloured by the IMM arbitration (blue = PN cruise, red = APN hardened)"
    )
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)
    fig.colorbar(lc, ax=ax1, label="maneuver-law weight w", shrink=0.85)

    ax2 = fig.add_subplot(2, 3, 3)
    mu = np.array(guid.mu_history)
    ax2.plot(mu[:, 0], mu[:, 1], lw=1.5, color="#d6453d", label="P(maneuver mode)")
    ax2.plot(tw[:, 0], tw[:, 1], lw=1.5, color="#7a30c9", ls="--", label="law weight w (γ=3)")
    ax2.axvline(t_break, color="k", ls=":", lw=1, label="true break")
    ax2.set_ylabel("belief / weight")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("IMM detects the maneuver")

    ax3 = fig.add_subplot(2, 3, 6)
    names = list(stats)
    p = [stats[n][0] for n in names]
    e = [stats[n][1] / 1e3 for n in names]
    x = np.arange(len(names))
    ax3.bar(x - 0.18, p, 0.36, color="#1f9ede", label="P(intercept)")
    axr = ax3.twinx()
    axr.bar(x + 0.18, e, 0.36, color="#999999", label="effort (k)")
    ax3.set_xticks(x, ["PN", "APN", "Adaptive"], fontsize=8)
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel("P(intercept)", color="#1f9ede")
    axr.set_ylabel("mean effort ((m/s²)²·s ×10³)", color="#777777")
    ax3.set_title("Monte-Carlo: APN robustness at lower effort")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p33_mode_adaptive.png", dpi=150)
    plt.show() if show else plt.close(fig)

    # --- GIF: trail recolours as the guidance hardens ---
    n = len(res.times)
    step = max(1, n // 140)
    figa, ax = plt.subplots(figsize=(7.5, 5.5), facecolor="#05060a")
    from intercept.viz.animation import style_dark_2d

    style_dark_2d(ax)
    allp = np.vstack([ip, tp])
    ax.set_xlim(allp[:, 0].min() - 300, allp[:, 0].max() + 300)
    ax.set_ylim(allp[:, 1].min() - 400, allp[:, 1].max() + 400)
    ax.set_aspect("equal")
    ax.set_title("Mode-adaptive guidance — PN cruise (blue) hardens to APN (red) on detection")
    ax.grid(True, alpha=0.25)
    lc_anim = LineCollection([], cmap=CMAP, lw=2.5)
    lc_anim.set_clim(0, 1)
    ax.add_collection(lc_anim)
    (tgt_ln,) = ax.plot([], [], "--", color="#ff4d6d", lw=1.4)
    (tgt_hd,) = ax.plot([], [], "s", color="#ff4d6d", ms=6)
    (int_hd,) = ax.plot([], [], "o", color="#7a30c9", ms=7)

    def update(f):
        k = min(f * step, n - 1)
        seg = (
            np.concatenate(
                [ip[: k + 1].reshape(-1, 1, 2)[:-1], ip[: k + 1].reshape(-1, 1, 2)[1:]], axis=1
            )
            if k > 1
            else []
        )
        if k > 1:
            lc_anim.set_segments(seg)
            lc_anim.set_array(w_t[:k])
        tgt_ln.set_data(tp[: k + 1, 0], tp[: k + 1, 1])
        tgt_hd.set_data([tp[k, 0]], [tp[k, 1]])
        int_hd.set_data([ip[k, 0]], [ip[k, 1]])
        return [lc_anim, tgt_ln, tgt_hd, int_hd]

    anim = FuncAnimation(figa, update, frames=n // step + 1, interval=40, blit=False)
    ANIM.mkdir(parents=True, exist_ok=True)
    anim.save(str(ANIM / "p33_mode_adaptive.gif"), writer=PillowWriter(fps=25))
    plt.show() if show else plt.close(figa)


def main() -> None:
    parser = argparse.ArgumentParser(description="IMM-mode-adaptive guidance demo")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument(
        "--weave-g",
        type=float,
        default=8.0,
        help="cruise weave g (8 = visible serpentine; 4 = light/sub-threshold)",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    print("=" * 60)
    print("MODE-ADAPTIVE GUIDANCE — cruise-then-break Monte-Carlo")
    print("=" * 60)
    stats = montecarlo(args.trials, args.seed, weave_g=args.weave_g)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p33_mode_adaptive.csv").write_text(
        "law,p_intercept,mean_effort\n"
        + "\n".join(f"{n},{p:.3f},{e:.0f}" for n, (p, e) in stats.items())
    )
    showcase_figure(stats, show)
    print(f"Figure: {FIG / 'p33_mode_adaptive.png'}\nGIF: {ANIM / 'p33_mode_adaptive.gif'}")


if __name__ == "__main__":
    main()
