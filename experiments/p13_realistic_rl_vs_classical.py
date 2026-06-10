"""P13 — learned vs. classical on REALISTIC (L2 aero) engagements, held-out.

Loads the PPO policy retrained on the aero plant (p12) and benchmarks it against True PN, Augmented
PN, and Sliding-mode on held-out realistic engagements (supersonic, gravity/drag/g-limit/lag, with
crossing / 18 g weave / 22 g random-telegraph jink targets). Reports intercept rate and effort —
the fair learned-vs-classical comparison on hard, realistic targets.

Run (after p12_train_rl_realistic.py):
    python experiments/p13_realistic_rl_vs_classical.py [--trials 100] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from intercept.benchmark import (
    ParametricScenario,
    format_table,
    run_benchmark,
    run_montecarlo,
    write_csv,
)
from intercept.core import G0
from intercept.guidance import AugmentedPN, sliding_mode, true_pn
from intercept.guidance.rl_policy import RLGuidance
from intercept.viz import plot_pintercept_bars

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"
RESULTS = ROOT / "results"
A_MAX = 40 * G0


def eval_scenarios() -> dict[str, ParametricScenario]:
    common = dict(
        model="aero",
        interceptor_speed=1000.0,
        interceptor_a_max=A_MAX,
        target_speed=700.0,
        target_a_max=25 * G0,
        interceptor_tau=0.2,
        target_tau=0.3,
        range_min=6000.0,
        range_max=9000.0,
        dt=0.01,
        t_max=18.0,
        kill_radius=20.0,
    )
    return {
        "A1_crossing": ParametricScenario(
            name="A1_crossing",
            target_heading_deg=150.0,
            offset_min=-2000,
            offset_max=2000,
            **common,
        ),
        "A2_weave18g": ParametricScenario(
            name="A2_weave18g",
            target_heading_deg=165.0,
            offset_min=-1200,
            offset_max=1200,
            maneuver={"type": "weave", "g": 18, "frequency": 0.35},
            **common,
        ),
        "A3_jink22g": ParametricScenario(
            name="A3_jink22g",
            target_heading_deg=165.0,
            offset_min=-1200,
            offset_max=1200,
            maneuver={"type": "telegraph", "g": 22, "mean_switch": 0.7},
            **common,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Realistic RL vs classical")
    parser.add_argument("--model", default=str(ROOT / "models" / "p12_ppo_realistic.zip"))
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=99, help="held-out seed")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model} — run p12_train_rl_realistic.py first.")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from intercept.envs import InterceptionEnv

    model = PPO.load(args.model, device="cpu")
    obs_norm = None
    vecp = Path(args.model).with_suffix(".vec.pkl")
    if vecp.exists():
        dummy = DummyVecEnv(
            [lambda: InterceptionEnv(ParametricScenario(name="d"), obs_mode="basic")]
        )
        vec = VecNormalize.load(str(vecp), dummy)
        vec.training = False
        obs_norm = vec.normalize_obs

    algorithms = {
        "RL (PPO, aero)": lambda t: RLGuidance(
            t, model, a_max=A_MAX, obs_norm=obs_norm, obs_mode="basic", gravity=G0
        ),
        "True PN (N=4)": lambda t: true_pn(t, N=4.0),
        "Augmented PN": lambda t: AugmentedPN(t, N=4.0),
        "Sliding-mode": lambda t: sliding_mode(t, eta=300.0),
    }
    scenarios = eval_scenarios()
    rows = run_benchmark(scenarios, algorithms, n_trials=args.trials, seed=args.seed)
    print(format_table(rows))
    write_csv(rows, RESULTS / "p13_realistic_rl_vs_classical.csv")
    plot_pintercept_bars(rows, save_path=FIG / "p13_realistic_rl_vs_classical.png", show=show)

    print("\nMean control effort (lower = more efficient):")
    for algo, fac in algorithms.items():
        eff = []
        for sc in scenarios.values():
            res = run_montecarlo(sc, fac, n_trials=args.trials, seed=args.seed)
            eff.append(np.mean([r.control_effort(r.interceptor) for r in res]))
        print(f"  {algo:16s}: {np.mean(eff):12.0f}")
    print(f"\nFigure: {FIG / 'p13_realistic_rl_vs_classical.png'}")


if __name__ == "__main__":
    main()
