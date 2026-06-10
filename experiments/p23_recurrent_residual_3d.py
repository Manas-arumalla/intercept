"""P23 — recurrent, APN-baseline residual RL in 3-D (the strongest learned 3-D guidance).

The 3-D analogue of P16: a `sb3_contrib.RecurrentPPO` (LSTM) policy that learns a bounded correction
on an **Augmented-PN-3D** baseline (`baseline="apn"`, `action_mode="residual_pn"`), so it starts
maneuver-aware (target-accel feed-forward) and the LSTM adds memory of the target's recent motion.
Trained on the 3-D aero curriculum, then benchmarked held-out against the 3-D classical laws and the
P20 PN-residual MLP (ablation: what the APN baseline + recurrence add over the plain residual).
See ADR-0014.

Run:
    python experiments/p23_recurrent_residual_3d.py --train [--timesteps 1200000] [--device cuda]
    python experiments/p23_recurrent_residual_3d.py --eval  [--trials 100] [--no-show]
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
RESIDUAL_SCALE = 0.35
RESIDUAL_EFFORT = 0.05
PN_N = 4.0
BASELINE = "apn"
DEFAULT_MODEL = MODELS / "p23_recurrent_residual_3d.zip"


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
        name="r3d_crossing", target_azimuth_deg=200.0, target_elevation_deg=-3.0, **c
    )
    weave = ParametricScenario3D(
        name="r3d_weave",
        target_azimuth_deg=200.0,
        target_elevation_deg=-2.0,
        maneuver={"type": "weave", "g": 12, "frequency": 0.3},
        **c,
    )
    barrel = ParametricScenario3D(
        name="r3d_barrel",
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

    return InterceptionEnv3D(
        scenario,
        RewardConfig(k_effort=RESIDUAL_EFFORT),
        obs_mode="rich",
        action_mode="residual_pn",
        residual_scale=RESIDUAL_SCALE,
        pn_N=PN_N,
        baseline=BASELINE,
    )


def train(args) -> None:
    from sb3_contrib import RecurrentPPO
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
            model = RecurrentPPO(
                "MlpLstmPolicy",
                venv,
                device=args.device,
                seed=args.seed,
                verbose=1,
                n_steps=512,
                batch_size=128,
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
            tb_log_name=f"recurrent_residual_3d_{name}",
            progress_bar=False,
        )
        obs_rms, ret_rms = venv.obs_rms, venv.ret_rms
        venv.save(vec_path)
        venv.close()
    assert model is not None
    model.save(args.out)
    print(f"\nSaved recurrent APN-residual 3-D policy to {args.out}")


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
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    rec = RecurrentPPO.load(str(model_path), device="cpu")
    rec_norm = _obs_norm_for(model_path)
    algorithms = {
        "Recurrent APN-residual": lambda t: RLGuidance3D(
            t,
            rec,
            a_max=A_MAX,
            obs_norm=rec_norm,
            obs_mode="rich",
            gravity=G0,
            action_mode="residual_pn",
            residual_scale=RESIDUAL_SCALE,
            pn_N=PN_N,
            baseline=BASELINE,
            recurrent=True,
        ),
        "True PN-3D": lambda t: true_pn_3d(t, N=4.0),
        "Augmented PN-3D": lambda t: augmented_pn_3d(t, N=4.0),
        "Optimal-3D": lambda t: optimal_guidance_3d(t, augment=True),
        "Sliding-mode-3D": lambda t: sliding_mode_3d(t, eta=300.0),
    }
    # P20 PN-residual MLP for the ablation, if present.
    p20 = MODELS / "p20_ppo_3d.zip"
    if p20.exists():
        try:
            m20 = PPO.load(str(p20), device="cpu")
            algorithms["Residual-PN (MLP)"] = lambda t: RLGuidance3D(
                t,
                m20,
                a_max=A_MAX,
                obs_norm=_obs_norm_for(p20),
                obs_mode="rich",
                gravity=G0,
                action_mode="residual_pn",
                residual_scale=RESIDUAL_SCALE,
                pn_N=PN_N,
                baseline="pn",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"(skipping P20 MLP: {exc})")

    scenarios = eval_scenarios()
    rows = run_benchmark(scenarios, algorithms, n_trials=args.trials, seed=args.seed)
    print(format_table(rows))
    write_csv(rows, RESULTS / "p23_recurrent_residual_3d.csv")
    plot_pintercept_bars(rows, save_path=FIG / "p23_recurrent_residual_3d.png", show=show)

    print("\nMean control effort (lower = more efficient):")
    for algo, fac in algorithms.items():
        eff = []
        for sc in scenarios.values():
            mc = run_montecarlo(sc, fac, n_trials=args.trials, seed=args.seed)
            eff.append(np.mean([r.control_effort(r.interceptor) for r in mc]))
        print(f"  {algo:24s}: {np.mean(eff):12.0f}")
    print(f"\nFigure: {FIG / 'p23_recurrent_residual_3d.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recurrent APN-residual 3-D RL")
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
