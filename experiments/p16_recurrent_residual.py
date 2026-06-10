"""P16 — Recurrent, APN-baseline residual RL: the strongest learned variant on the realistic plant.

Two upgrades over P15's PN-residual MLP, aimed at the *unpredictable* jink it could not beat:

1. **APN baseline** (`baseline="apn"`) — the residual now corrects **Augmented PN** (which already
   feed-forwards the target's measured lateral acceleration) instead of plain PN, so the learned
   part starts from a maneuver-aware command.
2. **Recurrent policy** (`sb3_contrib.RecurrentPPO`, `MlpLstmPolicy`) — an LSTM gives the policy
   *memory* of the target's recent motion, the missing ingredient for anticipating a random
   (telegraph) jink that a feed-forward (memoryless) policy structurally cannot.

Trains on the same aero (L2) curriculum as P15, then benchmarks held-out against True PN, Augmented
PN, Sliding-mode, and the P15 PN-residual MLP (if present) for an ablation.

Run:
    python experiments/p16_recurrent_residual.py --train [--timesteps 800000] [--device cuda]
    python experiments/p16_recurrent_residual.py --eval  [--trials 100] [--no-show]
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
DEFAULT_MODEL = MODELS / "p16_recurrent_residual.zip"


def _common() -> dict:
    return dict(
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


def curriculum() -> list[tuple[str, object]]:
    common = _common()
    crossing = ParametricScenario(
        name="res_crossing", target_heading_deg=150.0, offset_min=-2000, offset_max=2000, **common
    )
    weave = ParametricScenario(
        name="res_weave",
        target_heading_deg=165.0,
        offset_min=-1200,
        offset_max=1200,
        maneuver={"type": "weave", "g": 18, "frequency": 0.35},
        **common,
    )
    jink = ParametricScenario(
        name="res_jink",
        target_heading_deg=165.0,
        offset_min=-1200,
        offset_max=1200,
        maneuver={"type": "telegraph", "g": 22, "mean_switch": 0.7},
        **common,
    )
    return [("crossing", crossing), ("weave", weave), ("mixed", [crossing, weave, jink])]


def eval_scenarios() -> dict[str, ParametricScenario]:
    common = _common()
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


def _obs_norm_for(model_path: Path, obs_mode: str = "rich"):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from intercept.envs import InterceptionEnv

    vecp = model_path.with_suffix(".vec.pkl")
    if not vecp.exists():
        return None
    dummy = DummyVecEnv([lambda: InterceptionEnv(ParametricScenario(name="d"), obs_mode=obs_mode)])
    vec = VecNormalize.load(str(vecp), dummy)
    vec.training = False
    return vec.normalize_obs


def _make_env(scenario):
    from intercept.envs import InterceptionEnv, RewardConfig

    return InterceptionEnv(
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
            tb_log_name=f"recurrent_residual_{name}",
            progress_bar=False,
        )
        obs_rms, ret_rms = venv.obs_rms, venv.ret_rms
        venv.save(vec_path)
        venv.close()

    assert model is not None
    model.save(args.out)
    print(f"\nSaved recurrent residual policy to {args.out}\nSaved VecNormalize to {vec_path}")


def evaluate(args) -> None:
    show = not args.no_show
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path} — run with --train first.")
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    rec_model = RecurrentPPO.load(str(model_path), device="cpu")
    rec_norm = _obs_norm_for(model_path)

    algorithms = {
        "Recurrent APN-residual": lambda t: RLGuidance(
            t,
            rec_model,
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
        "True PN (N=4)": lambda t: true_pn(t, N=4.0),
        "Augmented PN": lambda t: AugmentedPN(t, N=4.0),
        "Sliding-mode": lambda t: sliding_mode(t, eta=300.0),
    }
    # P15 PN-residual MLP, for the ablation (baseline & memory contributions).
    p15 = MODELS / "p15_residual_ppo.zip"
    if p15.exists():
        try:
            m15 = PPO.load(str(p15), device="cpu")
            algorithms["Residual-PN (MLP)"] = lambda t: RLGuidance(
                t,
                m15,
                a_max=A_MAX,
                obs_norm=_obs_norm_for(p15),
                obs_mode="rich",
                gravity=G0,
                action_mode="residual_pn",
                residual_scale=0.35,
                pn_N=PN_N,
                baseline="pn",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"(skipping P15 MLP: {exc})")

    scenarios = eval_scenarios()
    rows = run_benchmark(scenarios, algorithms, n_trials=args.trials, seed=args.seed)
    print(format_table(rows))
    write_csv(rows, RESULTS / "p16_recurrent_residual.csv")
    plot_pintercept_bars(rows, save_path=FIG / "p16_recurrent_residual.png", show=show)

    print("\nMean control effort (lower = more efficient):")
    for algo, fac in algorithms.items():
        eff = []
        for sc in scenarios.values():
            mc = run_montecarlo(sc, fac, n_trials=args.trials, seed=args.seed)
            eff.append(np.mean([r.control_effort(r.interceptor) for r in mc]))
        print(f"  {algo:24s}: {np.mean(eff):12.0f}")
    print(f"\nFigure: {FIG / 'p16_recurrent_residual.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recurrent APN-residual RL guidance")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--timesteps", type=int, default=800_000)
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
