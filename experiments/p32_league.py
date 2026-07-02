"""P32 — the INTERCEPT League: Elo ratings for guidance laws *and* evaders.

Every seeded engagement is a **match**: the guidance law wins if it intercepts, the evader wins if
it escapes. A round-robin tournament (every law vs every evader over paired geometries, identical
dynamics — the fairness invariant) feeds a **Bradley-Terry** fit that places pursuers and evaders
on **one Elo ladder** — a single, readable skill scale across paradigms (classical, optimal, robust,
learned) and adversaries (scripted, game-theoretic, learned). Realistic comparable speeds
(interceptor ~Mach 3 vs ~Mach 2), L2 aero plant.

Beyond the point-estimate ladder, the experiment produces:

* **95 % bootstrap CI** (1 000 replicates, match-level resampling): ratings ± CI, statistical ties
  flagged where intervals overlap.
* **Scenario-sensitivity table**: the ladder refit on four disjoint subsets — head-on geometry,
  crossing geometry, scripted evaders only, adversarial evaders only — reveals whether a
  law's ranking is robust or scenario-specific.

Outputs: updated leaderboard (CSV + markdown), ladder figure with error bars, rank-stability
heatmap.

Run:
    python experiments/p32_league.py [--trials 40] [--boot 1000] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from intercept.adversary import optimal_evader, random_telegraph, reactive_break, straight, weave
from intercept.adversary.rl_evader import RLEvader
from intercept.benchmark import bradley_terry, bradley_terry_bootstrap, elo_expected_score
from intercept.core import Engagement, Entity
from intercept.core.aero import G0, AeroMissile2D
from intercept.guidance import (
    AugmentedPN,
    optimal_guidance,
    pure_pn,
    sliding_mode,
    true_pn,
    zem_pn,
)
from intercept.guidance.rl_policy import RLGuidance

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
FIG = ROOT / "gallery" / "figures"
RESULTS = ROOT / "results"
INT_SPEED, EVA_SPEED = 1000.0, 700.0  # ~Mach 3 vs ~Mach 2 (realistic ~1.43x)
INT_AMAX, EVA_AMAX = 40 * G0, 30 * G0
DT, T_MAX, KILL = 0.01, 16.0, 20.0

# Evader type labels for scenario sensitivity.
_SCRIPTED = {"straight", "weave 18g", "telegraph jink 22g", "reactive break 25g"}
_ADVERSARIAL = {"optimal (anti-LOS)", "RL evader"}


def _rl_law(path: Path, vec_name: str, **kw):
    """Load a saved policy + VecNormalize stats and return an RLGuidance factory."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from intercept.benchmark import ParametricScenario
    from intercept.envs import InterceptionEnv

    if "Recurrent" in kw.get("label", "") or kw.pop("recurrent_model", False):
        from sb3_contrib import RecurrentPPO

        model = RecurrentPPO.load(str(path), device="cpu")
        kw["recurrent"] = True
    else:
        from stable_baselines3 import PPO

        model = PPO.load(str(path), device="cpu")
    kw.pop("label", None)
    norm = None
    vecp = MODELS / vec_name
    if vecp.exists():
        dummy = DummyVecEnv(
            [lambda: InterceptionEnv(ParametricScenario(name="d"), obs_mode="rich")]
        )
        vec = VecNormalize.load(str(vecp), dummy)
        vec.training = False
        norm = vec.normalize_obs
    return lambda: RLGuidance(
        "target",
        model,
        a_max=INT_AMAX,
        obs_norm=norm,
        obs_mode="rich",
        gravity=G0,
        action_mode="residual_pn",
        **kw,
    )


def build_pursuers() -> dict[str, callable]:
    laws = {
        "Pure PN": lambda: pure_pn("target", N=4.0),
        "True PN": lambda: true_pn("target", N=4.0),
        "ZEM PN": lambda: zem_pn("target", N=4.0),
        "Augmented PN": lambda: AugmentedPN("target", N=4.0),
        "Optimal (OGL)": lambda: optimal_guidance("target", augment=True),
        "Sliding-mode": lambda: sliding_mode("target", eta=300.0),
    }
    p15 = MODELS / "p15_residual_ppo.zip"
    if p15.exists():
        laws["RL residual-PN"] = _rl_law(
            p15, "p15_residual_ppo.vec.pkl", residual_scale=0.35, pn_N=4.0
        )
    p16 = MODELS / "p16_recurrent_residual.zip"
    if p16.exists():
        laws["RL recurrent APN-res"] = _rl_law(
            p16,
            "p16_recurrent_residual.vec.pkl",
            recurrent_model=True,
            residual_scale=0.35,
            pn_N=4.0,
            baseline="apn",
        )
    return laws


def build_evaders() -> dict[str, callable]:
    """Evader factories: ``factory(rng) -> Controller`` (rng for seeded jinks / RL norm)."""
    evaders = {
        "straight": lambda rng: straight(),
        "weave 18g": lambda rng: weave(amplitude=18 * G0, frequency=0.35),
        "telegraph jink 22g": lambda rng: random_telegraph(22 * G0, 0.7, rng),
        "reactive break 25g": lambda rng: reactive_break("interceptor", 25 * G0, 2500.0),
        "optimal (anti-LOS)": lambda rng: optimal_evader("interceptor"),
    }
    p22 = MODELS / "p22_ppo_evader.zip"
    if p22.exists():
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from intercept.envs import EvaderEnv

        model = PPO.load(str(p22), device="cpu")
        norm = None
        vecp = MODELS / "p22_ppo_evader.vec.pkl"
        if vecp.exists():
            vec = VecNormalize.load(str(vecp), DummyVecEnv([lambda: EvaderEnv()]))
            vec.training = False
            norm = vec.normalize_obs
        evaders["RL evader"] = lambda rng: RLEvader(
            "interceptor", model, a_max=35 * G0, obs_norm=norm
        )
    return evaders


def _sample_geoms(n: int, seed: int) -> tuple[list, list[str]]:
    """Sample n engagement geometries; return (states, geo_tags).

    geo_tags are ``"head-on"`` (|lateral offset| < 500 m) or ``"crossing"``
    (|offset| ≥ 500 m), reflecting the dominant component of the engagement
    angle at the starting range.
    """
    rng = np.random.default_rng(seed)
    idyn = AeroMissile2D(a_max=INT_AMAX, tau=0.2)
    edyn = AeroMissile2D(a_max=EVA_AMAX, tau=0.3)
    geoms, geo_tags = [], []
    for _ in range(n):
        downrange = float(rng.uniform(5000.0, 8000.0))
        offset = float(rng.uniform(-1500.0, 1500.0))
        tag = "head-on" if abs(offset) < 500.0 else "crossing"
        geo_tags.append(tag)
        epos = np.array([downrange, offset])
        aim = epos / np.linalg.norm(epos)
        heading = np.arctan2(-aim[1], -aim[0]) + float(rng.uniform(-0.3, 0.3))
        evel = EVA_SPEED * np.array([np.cos(heading), np.sin(heading)])
        geoms.append(
            (idyn.initial_state([0.0, 0.0], INT_SPEED * aim), edyn.initial_state(epos, evel))
        )
    return geoms, geo_tags


def run_league(trials: int, seed: int):
    """Run the round-robin tournament.

    Returns
    -------
    pursuers, evaders, elo, pint, match_log, pairing_outcomes
        ``match_log`` is a list of ``(pursuer, evader, won: bool, geo_tag: str)`` tuples.
        ``pairing_outcomes`` is ``{(pursuer, evader): [bool, ...]}`` for bootstrap use.
    """
    pursuers = build_pursuers()
    evaders = build_evaders()
    geoms, geo_tags = _sample_geoms(trials, seed)
    names = list(pursuers) + list(evaders)
    idx = {nm: i for i, nm in enumerate(names)}
    wins = np.zeros((len(names), len(names)))
    pint: dict[tuple[str, str], float] = {}
    match_log: list[tuple[str, str, bool, str]] = []
    pairing_outcomes: dict[tuple[str, str], list[bool]] = {}

    for lname, lfac in pursuers.items():
        for ename, efac in evaders.items():
            pairing_outcomes[(lname, ename)] = []
            hits = 0
            for k, ((s0, s1), geo_tag) in enumerate(zip(geoms, geo_tags, strict=True)):
                rng = np.random.default_rng((seed, k))
                intc = Entity(
                    "interceptor",
                    AeroMissile2D(a_max=INT_AMAX, tau=0.2),
                    s0.copy(),
                    controller=lfac(),
                    role="interceptor",
                )
                tgt = Entity(
                    "target",
                    AeroMissile2D(a_max=EVA_AMAX, tau=0.3),
                    s1.copy(),
                    controller=efac(rng),
                    role="target",
                )
                res = Engagement(
                    [intc, tgt],
                    interceptor="interceptor",
                    target="target",
                    dt=DT,
                    t_max=T_MAX,
                    kill_radius=KILL,
                ).run()
                won = res.intercepted
                if won:
                    wins[idx[lname], idx[ename]] += 1
                    hits += 1
                else:
                    wins[idx[ename], idx[lname]] += 1
                match_log.append((lname, ename, won, geo_tag))
                pairing_outcomes[(lname, ename)].append(won)
            pint[(lname, ename)] = hits / len(geoms)
            print(f"  {lname:22s} vs {ename:20s}: P(int) = {pint[(lname, ename)]:.2f}")

    elo = bradley_terry(names, wins)
    return pursuers, evaders, elo, pint, match_log, pairing_outcomes


def _scenario_elo(
    names: list[str],
    match_log: list[tuple[str, str, bool, str]],
    pred,
) -> dict[str, float] | None:
    """Refit BT on matches satisfying pred(pursuer, evader, won, geo_tag).

    Returns None if fewer than ``len(names) * 2`` matches pass the filter.
    """
    idx = {nm: i for i, nm in enumerate(names)}
    n = len(names)
    W = np.zeros((n, n))
    for pursuer, evader, won, tag in match_log:
        if not pred(pursuer, evader, won, tag):
            continue
        pi, ei = idx[pursuer], idx[evader]
        if won:
            W[pi, ei] += 1
        else:
            W[ei, pi] += 1
    if W.sum() < n * 2:
        return None
    return bradley_terry(names, W)


def scenario_sensitivity(
    names: list[str],
    match_log: list[tuple[str, str, bool, str]],
) -> dict[str, dict[str, float]]:
    """Refit the BT ladder on four scenario subsets.

    Subsets
    -------
    Head-on     lateral offset < 500 m; threat arrives near straight-in.
    Crossing    lateral offset ≥ 500 m; threat crosses the interceptor's path.
    Scripted    matches against rule-based evaders (straight, weave, jink, break).
    Adversarial matches against intelligent evaders (game-theoretic optimal, RL).
    """
    subsets = {
        "Head-on": lambda p, e, w, t: t == "head-on",
        "Crossing": lambda p, e, w, t: t == "crossing",
        "Scripted": lambda p, e, w, t: e in _SCRIPTED,
        "Adversarial": lambda p, e, w, t: e in _ADVERSARIAL,
    }
    results: dict[str, dict[str, float]] = {}
    for label, pred in subsets.items():
        elo = _scenario_elo(names, match_log, pred)
        if elo is not None:
            results[label] = elo
    return results


def _figure_ladder(board, pursuers, ci, show: bool) -> None:
    """Horizontal bar chart with 95 % bootstrap CI error bars."""
    names = [nm for nm, _ in board][::-1]
    elo = dict(board)
    vals = [elo[nm] for nm in names]
    colors = ["#1f9ede" if nm in pursuers else "#d6453d" for nm in names]

    xerr_lo = [elo[nm] - ci[nm][0] for nm in names]
    xerr_hi = [ci[nm][1] - elo[nm] for nm in names]

    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, vals, color=colors, alpha=0.85, height=0.6)
    ax.errorbar(
        vals,
        y_pos,
        xerr=[xerr_lo, xerr_hi],
        fmt="none",
        color="#333333",
        capsize=4,
        capthick=1.2,
        lw=1.2,
    )
    for b, v, nm in zip(bars, vals, names, strict=True):
        lo, hi = ci[nm]
        ax.text(
            v + 6,
            b.get_y() + b.get_height() / 2,
            f"{v:.0f}  [{lo:.0f}–{hi:.0f}]",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y_pos, names)
    ax.axvline(1500, color="gray", ls="--", lw=1)
    ax.set_xlabel("Elo rating  (Bradley-Terry fit, mean 1500)  ·  error bars = 95 % bootstrap CI")
    ax.set_title(
        "INTERCEPT League — guidance laws (blue) vs evaders (red)\n"
        "error bars show 95 % CI (1 000 bootstrap replicates, match-level resampling)"
    )
    ax.set_xlim(min(vals) - 120, max(vals) + 200)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p32_league.png", dpi=150)
    plt.show() if show else plt.close(fig)


def _figure_rank_stability(
    board, pursuers, sensitivity: dict[str, dict[str, float]], show: bool
) -> None:
    """Heatmap of rank position across scenario subsets.

    Rows = participants (sorted by overall rank, top at top).
    Columns = scenario subset.
    Cell = rank in that subset (1 = best); grey if not enough data.
    """
    all_names = [nm for nm, _ in board]
    subsets = list(sensitivity)
    n_p = len(all_names)
    n_s = len(subsets)
    if n_s == 0:
        return

    rank_matrix = np.full((n_p, n_s), np.nan)
    for j, subset in enumerate(subsets):
        sub_elo = sensitivity[subset]
        sub_board = sorted(sub_elo.items(), key=lambda kv: -kv[1])
        sub_rank = {nm: r + 1 for r, (nm, _) in enumerate(sub_board)}
        for i, nm in enumerate(all_names):
            if nm in sub_rank:
                rank_matrix[i, j] = sub_rank[nm]

    fig, ax = plt.subplots(figsize=(7.0, 0.5 * n_p + 1.8))
    cmap = plt.cm.RdYlGn_r
    cmap.set_bad(color="#cccccc")
    im = ax.imshow(rank_matrix, cmap=cmap, vmin=1, vmax=n_p, aspect="auto")

    ax.set_xticks(range(n_s), subsets, fontsize=9)
    ax.set_yticks(range(n_p), all_names, fontsize=9)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    for i in range(n_p):
        for j in range(n_s):
            v = rank_matrix[i, j]
            txt = f"{int(v)}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")

    plt.colorbar(im, ax=ax, label="rank position  (1 = highest Elo)", shrink=0.6)
    ax.set_title(
        "Rank stability across scenario subsets\n"
        "(colour: green = top-ranked, red = bottom-ranked)",
        pad=30,
        fontsize=10,
    )
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p32_rank_stability.png", dpi=150)
    plt.show() if show else plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="INTERCEPT League — Elo over laws and evaders")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--boot", type=int, default=1000, help="bootstrap replicates")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    print("=" * 64)
    print(f"INTERCEPT LEAGUE — round-robin, {args.trials} paired matches per pairing")
    print("=" * 64)
    pursuers, evaders, elo, pint, match_log, pairing_outcomes = run_league(args.trials, args.seed)
    all_names = list(pursuers) + list(evaders)

    board = sorted(elo.items(), key=lambda kv: -kv[1])

    # --- Bootstrap confidence intervals ---
    print(f"\nBootstrap CI ({args.boot} replicates) ...")
    ci = bradley_terry_bootstrap(
        all_names,
        pairing_outcomes,
        n_replicates=args.boot,
        rng=np.random.default_rng(42),
    )

    # --- Leaderboard ---
    print("\n" + "=" * 64)
    print("LEADERBOARD (Bradley-Terry / Elo, mean 1500)  ·  95 % bootstrap CI")
    print("=" * 64)
    lines_md = [
        "| rank | participant | side | Elo | 95 % CI | statistical tie? |",
        "|---|---|---|---|---|---|",
    ]
    prev_hi = None
    for rank, (nm, r) in enumerate(board, 1):
        side = "guidance" if nm in pursuers else "evader"
        lo, hi = ci[nm]
        tie = prev_hi is not None and lo < prev_hi
        tie_str = "~" if tie else ""
        print(f"  {rank:2d}. {nm:24s} {side:9s} {r:7.0f}  [{lo:.0f}–{hi:.0f}]{tie_str}")
        lines_md.append(
            f"| {rank} | {nm} | {side} | {r:.0f} | [{lo:.0f}, {hi:.0f}] | {'yes' if tie else ''} |"
        )
        prev_hi = hi

    top_law = next(nm for nm, _ in board if nm in pursuers)
    top_eva = next(nm for nm, _ in board if nm in evaders)
    p_top = elo_expected_score(elo[top_law], elo[top_eva])
    print(f"\n  Expected score, {top_law} vs {top_eva}: {p_top:.2f}")

    # --- Scenario sensitivity ---
    print("\nScenario sensitivity ...")
    sensitivity = scenario_sensitivity(all_names, match_log)
    for subset, sub_elo in sensitivity.items():
        sub_board = sorted(sub_elo.items(), key=lambda kv: -kv[1])
        top3 = ", ".join(f"{nm} ({r:.0f})" for nm, r in sub_board[:3])
        print(f"  [{subset}] top-3: {top3}")

    # --- Save results ---
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p32_league.csv").write_text(
        "participant,side,elo,ci_lo,ci_hi\n"
        + "\n".join(
            f"{nm},{'guidance' if nm in pursuers else 'evader'},"
            f"{r:.1f},{ci[nm][0]:.1f},{ci[nm][1]:.1f}"
            for nm, r in board
        )
    )
    (RESULTS / "p32_league.md").write_text("\n".join(lines_md) + "\n")

    # --- Figures ---
    _figure_ladder(board, pursuers, ci, show)
    _figure_rank_stability(board, pursuers, sensitivity, show)

    print(f"\nFigures:  {FIG / 'p32_league.png'},  {FIG / 'p32_rank_stability.png'}")
    print(f"Results:  {RESULTS / 'p32_league.csv'},  {RESULTS / 'p32_league.md'}")


if __name__ == "__main__":
    main()
