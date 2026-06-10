"""P12 — train an RL interceptor on the realistic L2 aero plant (gravity/drag/g-limit/lag).

The P5 agent learned on the idealized L0 point mass; this retrains PPO against the *realistic*
dynamics and fast, evasive aero targets (curriculum: supersonic crossing → 20 g weave → mixed with
a 25 g random-telegraph jink). Same env/observation/VecNormalize machinery (ADR-0005), only the
plant and adversaries are harder. Saves a separate model so the L0 agent is preserved.

Run:
    python experiments/p12_train_rl_realistic.py [--timesteps 1500000] [--device cpu]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from intercept.benchmark import ParametricScenario
from intercept.core import G0
from intercept.envs import InterceptionEnv, RewardConfig

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RUNS = ROOT / "runs"


def curriculum() -> list[tuple[str, object]]:
    """Aero (L2) training stages of increasing difficulty; final stage mixes them."""
    common = dict(
        model="aero",
        interceptor_speed=1000.0,
        interceptor_a_max=40 * G0,
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
    crossing = ParametricScenario(
        name="rl_aero_crossing",
        target_heading_deg=150.0,
        offset_min=-2000,
        offset_max=2000,
        **common,
    )
    weave = ParametricScenario(
        name="rl_aero_weave",
        target_heading_deg=165.0,
        offset_min=-1200,
        offset_max=1200,
        maneuver={"type": "weave", "g": 18, "frequency": 0.35},
        **common,
    )
    jink = ParametricScenario(
        name="rl_aero_jink",
        target_heading_deg=165.0,
        offset_min=-1200,
        offset_max=1200,
        maneuver={"type": "telegraph", "g": 22, "mean_switch": 0.7},
        **common,
    )
    return [("crossing", crossing), ("weave", weave), ("mixed", [crossing, weave, jink])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO interceptor on realistic L2 dynamics")
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(MODELS / "p12_ppo_realistic.zip"))
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    stages = curriculum()
    per_stage = max(1, args.timesteps // len(stages))
    reward = RewardConfig()
    MODELS.mkdir(parents=True, exist_ok=True)
    vec_path = str(Path(args.out).with_suffix(".vec.pkl"))

    model = None
    obs_rms = ret_rms = None
    for i, (name, scenario) in enumerate(stages):
        raw = make_vec_env(
            lambda sc=scenario: InterceptionEnv(sc, reward, obs_mode="rich"),
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
            tb_log_name=f"ppo_aero_{name}",
            progress_bar=False,
        )
        obs_rms, ret_rms = venv.obs_rms, venv.ret_rms
        venv.save(vec_path)
        venv.close()

    assert model is not None
    model.save(args.out)
    print(f"\nSaved model to {args.out}\nSaved VecNormalize to {vec_path}")


if __name__ == "__main__":
    main()
