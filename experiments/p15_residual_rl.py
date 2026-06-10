"""P15 — Residual RL guidance: a learned correction on a Proportional-Navigation baseline.

**The idea (and why it's novel here).** From-scratch PPO does not transfer to the realistic (L2
aero) plant: it collapses to a constant saturated action and intercepts ~0–2 % (P12/P13), even
though a hand-coded PN scores ~100 % in the same env. Residual policy learning (Silver et al. 2018;
Johannink et al. 2019) fixes this by parameterizing the action as a *correction* on a competent
analytic baseline. Here the policy outputs a bounded residual added to a **pure-PN** command
(`action_mode="residual_pn"`): a zero action is already PN, so there is no collapse, and learning
only has to discover the *maneuver-anticipation* correction PN lacks against jinking/breaking
targets. To our knowledge this PN-residual parameterization is not a standard published guidance
law — it is a documented hybrid this benchmark contributes.

This trains the residual policy on the same aero curriculum P12 used, then benchmarks it held-out
against True PN, Augmented PN, Sliding-mode — and, if present, the failed from-scratch P12 model —
on identical realistic engagements.

Run:
    python experiments/p15_residual_rl.py --train [--timesteps 600000] [--device cuda] [--n-envs 4]
    python experiments/p15_residual_rl.py --eval  [--trials 100] [--no-show]
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
RESIDUAL_SCALE = 0.35  # the residual can add up to ±0.35·a_max on top of the PN baseline
RESIDUAL_EFFORT = 0.05  # penalty on residual magnitude — "do nothing (= PN)" is the default
PN_N = 4.0
DEFAULT_MODEL = MODELS / "p15_residual_ppo.zip"


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
    """Aero (L2) stages of increasing difficulty; the final stage mixes them."""
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


def _make_env(scenario):
    from intercept.envs import InterceptionEnv, RewardConfig

    # Penalize residual magnitude so the policy defaults to the PN baseline and only spends a
    # correction where it demonstrably helps (prevents fitting harmful residuals into PN).
    reward = RewardConfig(k_effort=RESIDUAL_EFFORT)
    return InterceptionEnv(
        scenario,
        reward,
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
            tb_log_name=f"ppo_residual_{name}",
            progress_bar=False,
        )
        obs_rms, ret_rms = venv.obs_rms, venv.ret_rms
        venv.save(vec_path)
        venv.close()

    assert model is not None
    model.save(args.out)
    print(f"\nSaved residual policy to {args.out}\nSaved VecNormalize to {vec_path}")


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


def evaluate(args) -> None:
    show = not args.no_show
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path} — run with --train first.")
    from stable_baselines3 import PPO

    res_model = PPO.load(str(model_path), device="cpu")
    res_norm = _obs_norm_for(model_path)

    algorithms = {
        "Residual-RL (PN+res)": lambda t: RLGuidance(
            t,
            res_model,
            a_max=A_MAX,
            obs_norm=res_norm,
            obs_mode="rich",
            gravity=G0,
            action_mode="residual_pn",
            residual_scale=RESIDUAL_SCALE,
            pn_N=PN_N,
        ),
        "True PN (N=4)": lambda t: true_pn(t, N=4.0),
        "Augmented PN": lambda t: AugmentedPN(t, N=4.0),
        "Sliding-mode": lambda t: sliding_mode(t, eta=300.0),
    }
    # Include the failed from-scratch policy for contrast, if it's around (best-effort: a stale
    # artifact with a different observation layout shouldn't break the residual eval).
    fs_path = MODELS / "p12_ppo_realistic.zip"
    if fs_path.exists():
        try:
            fs_model = PPO.load(str(fs_path), device="cpu")
            fs_obs_mode = "rich" if fs_model.observation_space.shape == (11,) else "basic"
            fs_norm = _obs_norm_for(fs_path, fs_obs_mode)
            algorithms["From-scratch PPO"] = lambda t: RLGuidance(
                t, fs_model, a_max=A_MAX, obs_norm=fs_norm, obs_mode=fs_obs_mode, gravity=G0
            )
        except Exception as exc:  # noqa: BLE001 — contrast model is optional
            print(f"(skipping from-scratch contrast model: {exc})")

    scenarios = eval_scenarios()
    rows = run_benchmark(scenarios, algorithms, n_trials=args.trials, seed=args.seed)
    print(format_table(rows))
    write_csv(rows, RESULTS / "p15_residual_rl.csv")
    plot_pintercept_bars(rows, save_path=FIG / "p15_residual_rl.png", show=show)

    print("\nMean control effort (lower = more efficient):")
    for algo, fac in algorithms.items():
        eff = []
        for sc in scenarios.values():
            mc = run_montecarlo(sc, fac, n_trials=args.trials, seed=args.seed)
            eff.append(np.mean([r.control_effort(r.interceptor) for r in mc]))
        print(f"  {algo:20s}: {np.mean(eff):12.0f}")
    print(f"\nFigure: {FIG / 'p15_residual_rl.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual RL guidance (PN + learned correction)")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--timesteps", type=int, default=600_000)
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
