"""P32 — the INTERCEPT League: Elo ratings for guidance laws *and* evaders.

Every seeded engagement is a **match**: the guidance law wins if it intercepts, the evader wins if
it escapes. A round-robin tournament (every law vs every evader over paired geometries, identical
dynamics — the fairness invariant) feeds a **Bradley-Terry** fit (`intercept.benchmark.league`)
that places pursuers and evaders on **one Elo ladder** — a single, readable skill scale across
paradigms (classical, optimal, robust, learned) and adversaries (scripted, game-theoretic, learned).
Realistic comparable speeds (interceptor ~Mach 3 vs ~Mach 2), L2 aero plant.

Outputs: a leaderboard (CSV + markdown snippet) and a ladder figure.

Run:
    python experiments/p32_league.py [--trials 40] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from intercept.adversary import optimal_evader, random_telegraph, reactive_break, straight, weave
from intercept.adversary.rl_evader import RLEvader
from intercept.benchmark import bradley_terry, elo_expected_score
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


def _sample_geoms(n: int, seed: int):
    rng = np.random.default_rng(seed)
    idyn = AeroMissile2D(a_max=INT_AMAX, tau=0.2)
    edyn = AeroMissile2D(a_max=EVA_AMAX, tau=0.3)
    out = []
    for _ in range(n):
        downrange = float(rng.uniform(5000.0, 8000.0))
        offset = float(rng.uniform(-1500.0, 1500.0))
        epos = np.array([downrange, offset])
        aim = epos / np.linalg.norm(epos)
        heading = np.arctan2(-aim[1], -aim[0]) + float(rng.uniform(-0.3, 0.3))
        evel = EVA_SPEED * np.array([np.cos(heading), np.sin(heading)])
        out.append(
            (idyn.initial_state([0.0, 0.0], INT_SPEED * aim), edyn.initial_state(epos, evel))
        )
    return out


def run_league(trials: int, seed: int):
    pursuers = build_pursuers()
    evaders = build_evaders()
    geoms = _sample_geoms(trials, seed)
    names = list(pursuers) + list(evaders)
    idx = {nm: i for i, nm in enumerate(names)}
    wins = np.zeros((len(names), len(names)))
    pint = {}
    for lname, lfac in pursuers.items():
        for ename, efac in evaders.items():
            hits = 0
            for k, (s0, s1) in enumerate(geoms):
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
                if res.intercepted:
                    wins[idx[lname], idx[ename]] += 1
                    hits += 1
                else:
                    wins[idx[ename], idx[lname]] += 1
            pint[(lname, ename)] = hits / len(geoms)
            print(f"  {lname:22s} vs {ename:20s}: P(int) = {pint[(lname, ename)]:.2f}")
    elo = bradley_terry(names, wins)
    return pursuers, evaders, elo, pint


def main() -> None:
    parser = argparse.ArgumentParser(description="INTERCEPT League — Elo over laws and evaders")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    print("=" * 64)
    print(f"INTERCEPT LEAGUE — round-robin, {args.trials} paired matches per pairing")
    print("=" * 64)
    pursuers, evaders, elo, pint = run_league(args.trials, args.seed)

    board = sorted(elo.items(), key=lambda kv: -kv[1])
    print("\n" + "=" * 64)
    print("LEADERBOARD (Bradley-Terry fit, Elo scale, mean 1500)")
    print("=" * 64)
    lines_md = ["| rank | participant | side | Elo |", "|---|---|---|---|"]
    for rank, (nm, r) in enumerate(board, 1):
        side = "guidance" if nm in pursuers else "evader"
        print(f"  {rank:2d}. {nm:24s} {side:9s} {r:7.0f}")
        lines_md.append(f"| {rank} | {nm} | {side} | {r:.0f} |")
    top_law = next(nm for nm, _ in board if nm in pursuers)
    top_eva = next(nm for nm, _ in board if nm in evaders)
    p_top = elo_expected_score(elo[top_law], elo[top_eva])
    print(f"\n  Expected score, {top_law} vs {top_eva}: {p_top:.2f}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    names = [nm for nm, _ in board][::-1]
    vals = [elo[nm] for nm in names]
    colors = ["#1f9ede" if nm in pursuers else "#d6453d" for nm in names]
    bars = ax.barh(names, vals, color=colors)
    for b, v in zip(bars, vals, strict=True):
        ax.text(v + 6, b.get_y() + b.get_height() / 2, f"{v:.0f}", va="center", fontsize=9)
    ax.axvline(1500, color="gray", ls="--", lw=1)
    ax.set_xlabel("Elo rating (Bradley-Terry fit, mean 1500)")
    ax.set_title("INTERCEPT League — guidance laws (blue) vs evaders (red) on one ladder")
    ax.set_xlim(min(vals) - 60, max(vals) + 90)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p32_league.png", dpi=150)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p32_league.csv").write_text(
        "participant,side,elo\n"
        + "\n".join(f"{nm},{'guidance' if nm in pursuers else 'evader'},{r:.1f}" for nm, r in board)
    )
    (RESULTS / "p32_league.md").write_text("\n".join(lines_md) + "\n")
    print(f"\nFigure: {FIG / 'p32_league.png'}\nMarkdown: {RESULTS / 'p32_league.md'}")
    plt.show() if show else plt.close(fig)


if __name__ == "__main__":
    main()
