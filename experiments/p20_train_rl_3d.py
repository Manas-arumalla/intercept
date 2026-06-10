"""P20 — train an RL interceptor in 3-D, and benchmark it against the 3-D classical laws.

The RL centerpiece, lifted to three dimensions (ADR-0014): a PPO policy with a **2-DOF lateral
action** (in the plane ⟂ the interceptor's velocity) and a 3-D observation, trained over a 3-D aero
curriculum (crossing → 3-D weave → barrel-roll) with VecNormalize. Deployed via `RLGuidance3D` it
runs inside the ordinary `Engagement` and is compared held-out against True PN-3D, Augmented PN-3D,
Optimal-3D, and Sliding-mode-3D on identical 3-D engagements — the fair learned-vs-classical
comparison, now in 3-D.

Run:
    python experiments/p20_train_rl_3d.py --train [--timesteps 1200000] [--device cuda]
    python experiments/p20_train_rl_3d.py --eval  [--trials 100] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from intercept.benchmark import (
    ParametricScenario3D,
    format_table,
    run_benchmark,
    run_montecarlo,
    write_csv,
)
from intercept.core import G0
from intercept.guidance import (
    augmented_pn_3d,
    optimal_guidance_3d,
    sliding_mode_3d,
    true_pn_3d,
)
from intercept.guidance.rl_policy import RLGuidance3D
from intercept.viz import plot_pintercept_bars

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"
RESULTS = ROOT / "results"
A_MAX = 40 * G0
RESIDUAL_SCALE = 0.35  # learned correction can add up to ±0.35·a_max on the PN-3D baseline
RESIDUAL_EFFORT = 0.05  # penalize residual magnitude — "do nothing (= PN-3D)" is the default
PN_N = 4.0
DEFAULT_MODEL = MODELS / "p20_ppo_3d.zip"


def _common() -> dict:
    return dict(
        model="aero",
        interceptor_speed=1000.0,
        interceptor_a_max=A_MAX,
        target_speed=700.0,
        target_a_max=22 * G0,
        interceptor_tau=0.2,
        target_tau=0.3,
        range_min=6000.0,
        range_max=9000.0,
        offset_min=-1500.0,
        offset_max=1500.0,
        alt_min=2500.0,
        alt_max=5000.0,
        dt=0.01,
        t_max=20.0,
        kill_radius=20.0,
    )


def curriculum() -> list[tuple[str, object]]:
    c = _common()
    crossing = ParametricScenario3D(
        name="rl3d_crossing", target_azimuth_deg=200.0, target_elevation_deg=-3.0, **c
    )
    weave = ParametricScenario3D(
        name="rl3d_weave",
        target_azimuth_deg=200.0,
        target_elevation_deg=-2.0,
        maneuver={"type": "weave", "g": 12, "frequency": 0.3},
        **c,
    )
    barrel = ParametricScenario3D(
        name="rl3d_barrel",
        target_azimuth_deg=200.0,
        target_elevation_deg=-2.0,
        maneuver={"type": "barrel_roll", "g": 14, "rate": 1.1},
        **c,
    )
    return [("crossing", crossing), ("weave", weave), ("mixed", [crossing, weave, barrel])]


def eval_scenarios() -> dict[str, ParametricScenario3D]:
    c = _common()
    return {
        "C1_crossing": ParametricScenario3D(
            name="C1_crossing", target_azimuth_deg=205.0, target_elevation_deg=-3.0, **c
        ),
        "C2_weave": ParametricScenario3D(
            name="C2_weave",
            target_azimuth_deg=200.0,
            target_elevation_deg=-2.0,
            maneuver={"type": "weave", "g": 12, "frequency": 0.3},
            **c,
        ),
        "C3_barrel": ParametricScenario3D(
            name="C3_barrel",
            target_azimuth_deg=200.0,
            target_elevation_deg=-2.0,
            maneuver={"type": "barrel_roll", "g": 14, "rate": 1.1},
            **c,
        ),
    }


def _make_env(scenario):
    from intercept.envs import InterceptionEnv3D, RewardConfig

    # Residual-PN-3D parameterization (ADR-0011/0014): the policy learns a correction on the
    # True-PN-3D baseline, so a zero action is already competent PN — no from-scratch collapse.
    return InterceptionEnv3D(
        scenario,
        RewardConfig(k_effort=RESIDUAL_EFFORT),
        obs_mode="rich",
        action_mode="residual_pn",
        residual_scale=RESIDUAL_SCALE,
        pn_N=PN_N,
    )


def train(args) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    stages = curriculum()
    per_stage = max(1, args.timesteps // len(stages))
    MODELS.mkdir(parents=True, exist_ok=True)
    vec_path = str(Path(args.out).with_suffix(".vec.pkl"))
    model = None
    obs_rms = ret_rms = None
    for i, (name, scenario) in enumerate(stages):
        raw = make_vec_env(
            lambda sc=scenario: _make_env(sc), n_envs=args.n_envs, seed=args.seed + i
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
            tb_log_name=f"ppo_3d_{name}",
            progress_bar=False,
        )
        obs_rms, ret_rms = venv.obs_rms, venv.ret_rms
        venv.save(vec_path)
        venv.close()
    assert model is not None
    model.save(args.out)
    print(f"\nSaved 3-D policy to {args.out}\nSaved VecNormalize to {vec_path}")


def _obs_norm_for(model_path: Path):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from intercept.envs import InterceptionEnv3D

    vecp = model_path.with_suffix(".vec.pkl")
    if not vecp.exists():
        return None
    dummy = DummyVecEnv(
        [lambda: InterceptionEnv3D(ParametricScenario3D(name="d"), obs_mode="rich")]
    )
    vec = VecNormalize.load(str(vecp), dummy)
    vec.training = False
    return vec.normalize_obs


def evaluate(args) -> None:
    show = not args.no_show
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path} — run with --train first.")
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device="cpu")
    norm = _obs_norm_for(model_path)
    algorithms = {
        "RL-3D (residual)": lambda t: RLGuidance3D(
            t,
            model,
            a_max=A_MAX,
            obs_norm=norm,
            obs_mode="rich",
            gravity=G0,
            action_mode="residual_pn",
            residual_scale=RESIDUAL_SCALE,
            pn_N=PN_N,
        ),
        "True PN-3D": lambda t: true_pn_3d(t, N=4.0),
        "Augmented PN-3D": lambda t: augmented_pn_3d(t, N=4.0),
        "Optimal-3D": lambda t: optimal_guidance_3d(t, augment=True),
        "Sliding-mode-3D": lambda t: sliding_mode_3d(t, eta=300.0),
    }
    scenarios = eval_scenarios()
    rows = run_benchmark(scenarios, algorithms, n_trials=args.trials, seed=args.seed)
    print(format_table(rows))
    write_csv(rows, RESULTS / "p20_rl_3d.csv")
    plot_pintercept_bars(rows, save_path=FIG / "p20_rl_3d.png", show=show)

    print("\nMean control effort (lower = more efficient):")
    for algo, fac in algorithms.items():
        eff = []
        for sc in scenarios.values():
            mc = run_montecarlo(sc, fac, n_trials=args.trials, seed=args.seed)
            eff.append(np.mean([r.control_effort(r.interceptor) for r in mc]))
        print(f"  {algo:18s}: {np.mean(eff):12.0f}")
    print(f"\nFigure: {FIG / 'p20_rl_3d.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/eval a 3-D RL interceptor")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--timesteps", type=int, default=1_200_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--eval-seed", dest="eval_seed", type=int, default=99)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--out", default=str(DEFAULT_MODEL))
    args = parser.parse_args()
    if not args.train and not args.eval:
        args.eval = Path(args.model).exists()
        args.train = not args.eval
    if args.train:
        train(args)
    if args.eval:
        args.seed = args.eval_seed
        evaluate(args)


if __name__ == "__main__":
    main()
