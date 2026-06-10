"""P5 — head-to-head: the learned policy vs. classical / optimal guidance.

Loads the trained PPO policy, wraps it as a guidance `Controller`, and benchmarks it against True
PN, Optimal (OGL), and Augmented PN on a **held-out** set of engagement geometries (seeds disjoint
from training) on identical dynamics — the rigorous classical-vs-learned comparison the literature
rarely does cleanly, reporting *both* intercept rate and control effort/efficiency.

Run (after p5_train_rl.py):
    python experiments/p5_rl_vs_classical.py [--model models/p5_ppo_interceptor.zip] [--trials 100]
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
from intercept.guidance import AugmentedPN, optimal_guidance, true_pn
from intercept.guidance.rl_policy import RLGuidance
from intercept.viz import compare_engagements_2d, plot_pintercept_bars

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"
RESULTS = ROOT / "results"

A_MAX = 250.0


def eval_scenarios() -> dict[str, ParametricScenario]:
    common = dict(
        interceptor_speed=1000,
        interceptor_a_max=A_MAX,
        target_speed=700,
        range_min=5000,
        range_max=8000,
        dt=0.02,
        t_max=16.0,
        kill_radius=20.0,
    )
    return {
        "E1_headon": ParametricScenario(
            name="E1_headon", target_heading_deg=180.0, offset_min=-400, offset_max=400, **common
        ),
        "E2_crossing": ParametricScenario(
            name="E2_crossing",
            target_heading_deg=110.0,
            offset_min=-1200,
            offset_max=1200,
            **common,
        ),
        "E3_weaving": ParametricScenario(
            name="E3_weaving",
            target_heading_deg=180.0,
            offset_min=-400,
            offset_max=400,
            maneuver={"type": "weave", "amplitude": 150.0, "frequency": 0.3},
            **common,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RL vs classical comparison")
    parser.add_argument("--model", default=str(ROOT / "models" / "p5_ppo_interceptor.zip"))
    parser.add_argument("--vecnorm", default=str(ROOT / "models" / "p5_vecnormalize.pkl"))
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42, help="held-out eval seed")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model} — run experiments/p5_train_rl.py first.")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from intercept.envs import InterceptionEnv

    model = PPO.load(args.model, device="cpu")

    obs_norm = None
    if Path(args.vecnorm).exists():
        dummy = DummyVecEnv([lambda: InterceptionEnv(ParametricScenario(name="d"))])
        vec = VecNormalize.load(args.vecnorm, dummy)
        vec.training = False
        obs_norm = vec.normalize_obs

    algorithms = {
        "RL (PPO residual)": lambda tgt: RLGuidance(
            tgt,
            model,
            a_max=A_MAX,
            obs_norm=obs_norm,
            action_mode="residual_pn",
            residual_scale=0.5,
        ),
        "True PN (N=4)": lambda tgt: true_pn(tgt, N=4.0),
        "Optimal (OGL)": lambda tgt: optimal_guidance(tgt, augment=True),
        "Augmented PN": lambda tgt: AugmentedPN(tgt, N=4.0),
    }
    scenarios = eval_scenarios()

    rows = run_benchmark(scenarios, algorithms, n_trials=args.trials, seed=args.seed)
    print(format_table(rows))
    write_csv(rows, RESULTS / "p5_rl_vs_classical.csv")
    plot_pintercept_bars(rows, save_path=FIG / "p5_rl_vs_classical_pintercept.png", show=show)

    # Effort/efficiency table (RL's hypothesized edge — Research §8): mean effort per algorithm.
    print("\nMean control effort by algorithm (lower = more efficient):")
    for algo in algorithms:
        efforts = []
        for scen in scenarios.values():
            results = run_montecarlo(scen, algorithms[algo], n_trials=args.trials, seed=args.seed)
            efforts.append(np.mean([r.control_effort(r.interceptor) for r in results]))
        print(f"  {algo:16s}: {np.mean(efforts):10.0f}")

    # Trajectory overlay on one crossing engagement: RL vs True PN.
    cross = scenarios["E2_crossing"]
    overlay = {}
    for label, factory in {
        "RL (PPO residual)": algorithms["RL (PPO residual)"],
        "True PN": algorithms["True PN (N=4)"],
    }.items():
        res = run_montecarlo(cross, factory, n_trials=1, seed=args.seed)[0]
        overlay[label] = res
    compare_engagements_2d(
        overlay,
        title="RL vs. True PN — crossing engagement",
        save_path=FIG / "p5_rl_vs_pn_trajectory.png",
        show=show,
    )
    print(f"\nFigures saved to: {GALLERY}")


if __name__ == "__main__":
    main()
