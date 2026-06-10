"""P35 — coordinated swarm penetration vs. an asset-value layered defense.

Real swarms do not merely bring *more* missiles — they **coordinate** to defeat the defender's
allocation logic: simultaneous time-on-target, decoy screens, concentrated saturation points, and
sequential waves (``intercept.adversary.swarm_tactics``, grounded in open-source saturation/salvo
doctrine). This experiment pits those tactics against a **limited magazine** of interceptors and
compares two defensive doctrines:

* **naive** — time-minimizing Hungarian WTA: engage every track, nearest-first (the P7 default);
* **asset-value** — impact-point prediction + decoy de-prioritization + value-prioritized,
  capacity-aware allocation (``intercept.multiagent.defense``).

The reported metric is **real-threat leakers** (decoys leaking past the asset is harmless). With
fewer interceptors than tracks, the naive defender squanders shots on chaff and over-saturates; the
asset-value defender spends its magazine on the threats that actually endanger the asset.

Run:
    python experiments/p35_swarm_penetration.py [--trials 40] [--magazine 5] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from intercept.adversary.swarm_tactics import SWARM_TACTICS, Raid, decoy_screen
from intercept.core import Entity, PointMass3D
from intercept.core.aero import G0
from intercept.guidance import augmented_pn_3d
from intercept.multiagent.defense import make_value_allocator
from intercept.multiagent.swarm import MultiEngagement
from intercept.viz.threed import _BG, _glow_plot, _set_equal_aspect, _style_dark

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "gallery" / "figures"
ANIM = ROOT / "gallery" / "animations"
RESULTS = ROOT / "results"

INT_SPEED, INT_AMAX = 1000.0, 50 * G0  # ~Mach 3, 50 g (realistic)
DEFENDED = np.zeros(3)
LETHAL_RADIUS = 800.0  # predicted-miss gate for "endangers the asset"


def _jitter(raid: Raid, rng: np.random.Generator) -> Raid:
    """Rotate the whole raid about the defended point by a random bearing + small position noise.

    Preserves the tactic's internal geometry (relative axes / timing) while varying the engagement,
    so Monte-Carlo trials are independent draws of the *same* tactic."""
    th = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(th), np.sin(th)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    d = raid.defended
    for e in raid.threats:
        p = d + rot @ (e.state[:3] - d) + rng.normal(0, 150.0, 3) * [1, 1, 0.3]
        v = rot @ e.state[3:6]
        e.state = np.array([*p, *v], dtype=float)
    return raid


def _battery(magazine: int, raid: Raid, rng: np.random.Generator) -> list[Entity]:
    """A battery of ``magazine`` interceptors near the defended point, aimed at the raid centre."""
    centroid = np.mean([t.state[:3] for t in raid.threats], axis=0)
    ints = []
    for i in range(magazine):
        p = DEFENDED + np.array([rng.uniform(-500, 500), rng.uniform(-500, 500), 0.0])
        aim = (centroid - p) / np.linalg.norm(centroid - p)
        ints.append(
            Entity(
                f"I{i}",
                PointMass3D(a_max=INT_AMAX),
                np.array([*p, *(INT_SPEED * aim)]),
                role="interceptor",
            )
        )
    return ints


def _engage(raid: Raid, magazine: int, doctrine: str, rng: np.random.Generator):
    ints = _battery(magazine, raid, rng)
    allocator = (
        make_value_allocator(raid.defended, lethal_radius=LETHAL_RADIUS)
        if doctrine == "asset-value"
        else None
    )
    eng = MultiEngagement(
        ints,
        raid.threats,
        lambda t: augmented_pn_3d(t, N=4.0),
        dt=0.02,
        t_max=30.0,
        kill_radius=50.0,
        reassign_every=15,
        allocator=allocator,
    )
    return eng.run()


def _real_leakers(result, raid: Raid) -> int:
    killed = {t for t, _, _ in result.kills}
    return sum(1 for e in raid.real_threats if e.name not in killed)


def montecarlo(trials: int, magazine: int, seed: int = 11):
    """Real-threat leakers for naive vs asset-value across all four tactics (lower is better)."""
    stats: dict[str, dict[str, float]] = {}
    for tactic, builder in SWARM_TACTICS.items():
        out = {"naive": [], "asset-value": []}
        n_real = None
        for s in range(trials):
            for doctrine in ("naive", "asset-value"):
                rng = np.random.default_rng((seed, s))  # same geometry for both doctrines
                raid = _jitter(builder(defended=DEFENDED), rng)
                n_real = len(raid.real_threats)
                res = _engage(raid, magazine, doctrine, np.random.default_rng((seed, s, 1)))
                out[doctrine].append(_real_leakers(res, raid))
        stats[tactic] = {
            "naive": float(np.mean(out["naive"])),
            "asset-value": float(np.mean(out["asset-value"])),
            "n_real": float(n_real),
        }
        print(
            f"  {tactic:18s}: naive {stats[tactic]['naive']:.2f} leak / "
            f"asset-value {stats[tactic]['asset-value']:.2f} leak  (of {n_real} real)"
        )
    return stats


def _panel(ax, result, raid: Raid, title: str) -> None:
    _style_dark(ax)
    allpts = np.vstack([s[:, :3] for s in result.tracks.values()])
    _set_equal_aspect(ax, allpts)
    killed = {t for t, _, _ in result.kills}
    for nm in result.tracks:
        if result.roles.get(nm) != "target":
            continue
        p = result.tracks[nm][:, :3]
        is_decoy = nm in raid.decoys
        col = "#7d8590" if is_decoy else ("#37ff8b" if nm in killed else "#ff4d6d")
        _glow_plot(ax, p[:, 0], p[:, 1], p[:, 2], col)
    for nm, r in result.roles.items():
        if r == "interceptor":
            p = result.tracks[nm][:, :3]
            ax.plot(p[:, 0], p[:, 1], p[:, 2], "-", color="#19e6ff", lw=0.9, alpha=0.7)
    if result.kill_points:
        kp = np.array(result.kill_points)
        ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2], c="#c6ff00", marker="*", s=120, depthshade=False)
    ax.scatter([0], [0], [0], c="#19e6ff", marker="^", s=110)
    leak = _real_leakers(result, raid)
    ax.set_title(
        f"{title}\n{leak} real leaker(s), {len(result.kills)} shots used",
        color="#e6edf3",
        fontsize=11,
        pad=8,
    )


def showcase(magazine: int, show: bool, seed: int = 3) -> None:
    rngA = np.random.default_rng(seed)
    raid_naive = _jitter(decoy_screen(n_real=5, n_decoy=7, defended=DEFENDED), rngA)
    # Rebuild the identical raid for the value run (entities are mutated in place during a sim).
    rngA2 = np.random.default_rng(seed)
    raid_value = _jitter(decoy_screen(n_real=5, n_decoy=7, defended=DEFENDED), rngA2)

    res_naive = _engage(raid_naive, magazine, "naive", np.random.default_rng((seed, 9)))
    res_value = _engage(raid_value, magazine, "asset-value", np.random.default_rng((seed, 9)))

    fig = plt.figure(figsize=(14.5, 6.2), facecolor=_BG)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _panel(ax1, res_naive, raid_naive, f"Naive time-WTA ({magazine} interceptors)")
    _panel(ax2, res_value, raid_value, f"Asset-value defense ({magazine} interceptors)")
    fig.suptitle(
        "Decoy-screen saturation raid — green=killed, red=leaked real threat, grey=decoy",
        color="#cdd5df",
        fontsize=12,
    )
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p35_swarm_penetration.png", dpi=150, facecolor=_BG)
    plt.show() if show else plt.close(fig)
    return res_value, raid_value


def tactics_figure(magazine: int, show: bool, seed: int = 5) -> None:
    """A 2×2 gallery: each penetration tactic's engagement under the asset-value defense."""
    fig = plt.figure(figsize=(13.5, 11.0), facecolor=_BG)
    for k, (tactic, builder) in enumerate(SWARM_TACTICS.items()):
        rng = np.random.default_rng((seed, k))
        raid = _jitter(builder(defended=DEFENDED), rng)
        res = _engage(raid, magazine, "asset-value", np.random.default_rng((seed, k, 1)))
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        _panel(ax, res, raid, tactic)
    fig.suptitle(
        "Coordinated penetration tactics vs. the asset-value defense — "
        "green=killed, red=leaked real threat, grey=decoy",
        color="#cdd5df",
        fontsize=12,
    )
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p35_tactics_gallery.png", dpi=150, facecolor=_BG)
    plt.show() if show else plt.close(fig)


def bars_figure(stats, magazine: int, show: bool) -> None:
    tactics = list(stats)
    x = np.arange(len(tactics))
    naive = [stats[t]["naive"] for t in tactics]
    value = [stats[t]["asset-value"] for t in tactics]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(x - 0.2, naive, 0.4, color="#d1495b", label="naive time-WTA")
    ax.bar(x + 0.2, value, 0.4, color="#2a9d8f", label="asset-value defense")
    for xi, (a, b) in enumerate(zip(naive, value, strict=True)):
        ax.text(xi - 0.2, a + 0.03, f"{a:.1f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, b + 0.03, f"{b:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, [t.replace("-", "\n") for t in tactics], fontsize=8)
    ax.set_ylabel("mean real-threat leakers  (lower = better)")
    ax.set_title(f"Coordinated penetration tactics vs. {magazine}-interceptor magazine")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p35_penetration_bars.png", dpi=150)
    plt.show() if show else plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Swarm penetration vs asset-value defense")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--magazine", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    print(f"Monte-Carlo: {args.trials} trials/tactic, {args.magazine}-interceptor magazine")
    stats = montecarlo(args.trials, args.magazine, args.seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p35_swarm_penetration.csv").write_text(
        "tactic,n_real,naive_leakers,asset_value_leakers\n"
        + "\n".join(
            f"{t},{int(s['n_real'])},{s['naive']:.3f},{s['asset-value']:.3f}"
            for t, s in stats.items()
        )
    )
    bars_figure(stats, args.magazine, show)

    tactics_figure(args.magazine, show)

    res_value, raid_value = showcase(args.magazine, show)
    from intercept.viz import animate_swarm_3d_modern

    animate_swarm_3d_modern(
        res_value,
        title="Asset-value defense vs. a decoy-screen saturation raid",
        save_path=ANIM / "p35_swarm_penetration.gif",
        max_frames=140,
        show=show,
    )
    # Second animation: the simultaneous time-on-target raid (everything arrives at once).
    rng = np.random.default_rng(21)
    from intercept.adversary.swarm_tactics import simultaneous_tot

    raid_tot = _jitter(simultaneous_tot(defended=DEFENDED), rng)
    res_tot = _engage(raid_tot, args.magazine + 3, "asset-value", np.random.default_rng(22))
    animate_swarm_3d_modern(
        res_tot,
        title="Simultaneous time-on-target raid — every threat arrives at once",
        save_path=ANIM / "p35_tot_raid.gif",
        max_frames=140,
        show=show,
    )
    print(
        f"Figures: {FIG / 'p35_penetration_bars.png'}, {FIG / 'p35_swarm_penetration.png'}, "
        f"{FIG / 'p35_tactics_gallery.png'}"
    )
    print(f"Animations: {ANIM / 'p35_swarm_penetration.gif'}, {ANIM / 'p35_tot_raid.gif'}")


if __name__ == "__main__":
    main()
