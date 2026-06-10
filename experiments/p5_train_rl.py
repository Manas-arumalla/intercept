"""P5 — train an RL interception policy (PPO, Stable-Baselines3) with a curriculum.

Trains across progressively harder scenarios (head-on → crossing → weaving), continuing the same
policy, and saves the model for evaluation against the classical/optimal/MPC laws
(``p5_rl_vs_classical.py``). The env shares the engagement physics core, so the learned policy is
later benchmarked on identical dynamics (ADR-0005).

Run:
    python experiments/p5_train_rl.py [--timesteps 450000] [--device auto] [--wandb]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from intercept.benchmark import ParametricScenario
from intercept.envs import InterceptionEnv, RewardConfig

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RUNS = ROOT / "runs"


def curriculum() -> list[tuple[str, object]]:
    """Progressively harder training stages (same physics, increasing difficulty).

    The final stage mixes all geometries (the env samples one per episode) so the policy stays
    competent on the earlier ones — countering catastrophic forgetting.
    """
    common = dict(
        interceptor_speed=1000,
        interceptor_a_max=250,
        target_speed=700,
        range_min=5000,
        range_max=8000,
        dt=0.02,
        t_max=16.0,
        kill_radius=20.0,
    )
    headon = ParametricScenario(
        name="rl_headon", target_heading_deg=180.0, offset_min=-400, offset_max=400, **common
    )
    crossing = ParametricScenario(
        name="rl_crossing", target_heading_deg=110.0, offset_min=-1200, offset_max=1200, **common
    )
    weaving = ParametricScenario(
        name="rl_weaving",
        target_heading_deg=180.0,
        offset_min=-400,
        offset_max=400,
        maneuver={"type": "weave", "amplitude": 150.0, "frequency": 0.3},
        **common,
    )
    return [
        ("headon", headon),
        ("crossing", crossing),
        ("mixed", [headon, crossing, weaving]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO interception policy")
    parser.add_argument(
        "--timesteps", type=int, default=600_000, help="total timesteps (split over stages)"
    )
    parser.add_argument("--device", default="auto", help="torch device (auto/cpu/cuda)")
    parser.add_argument("--n-envs", type=int, default=8, help="parallel envs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--out", default=str(MODELS / "p5_ppo_interceptor.zip"))
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    stages = curriculum()
    per_stage = max(1, args.timesteps // len(stages))
    reward = RewardConfig()
    MODELS.mkdir(parents=True, exist_ok=True)
    vec_path = str(Path(args.out).with_name("p5_vecnormalize.pkl"))

    if args.wandb:
        import wandb

        wandb.init(project="intercept-rl", config=vars(args), sync_tensorboard=True)

    model = None
    obs_rms = ret_rms = None  # carried across stages so normalization stays consistent
    for i, (name, scenario) in enumerate(stages):
        raw = make_vec_env(
            lambda sc=scenario: InterceptionEnv(
                sc, reward, action_mode="residual_pn", residual_scale=0.5
            ),
            n_envs=args.n_envs,
            seed=args.seed + i,
        )
        venv = VecNormalize(raw, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.999)
        if obs_rms is not None:
            venv.obs_rms, venv.ret_rms = obs_rms, ret_rms
        if model is None:
            model = PPO(
                "MlpPolicy",
                venv,
                device=args.device,
                seed=args.seed,
                verbose=1,
                n_steps=1024,
                batch_size=256,
                gae_lambda=0.95,
                gamma=0.999,
                ent_coef=0.0,
                learning_rate=3e-4,
                tensorboard_log=str(RUNS),
            )
        else:
            model.set_env(venv)
        print(f"\n=== Stage {i + 1}/{len(stages)}: {name} ({per_stage} steps) ===")
        model.learn(
            total_timesteps=per_stage,
            reset_num_timesteps=False,
            tb_log_name=f"ppo_{name}",
            progress_bar=False,
        )
        obs_rms, ret_rms = venv.obs_rms, venv.ret_rms
        venv.save(vec_path)  # persist normalization stats for deployment
        venv.close()

    assert model is not None
    model.save(args.out)
    print(f"\nSaved model to {args.out}")
    print(f"Saved VecNormalize stats to {vec_path}")
    if args.wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
