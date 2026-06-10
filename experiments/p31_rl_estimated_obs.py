"""P31 — training RL directly on estimated (noisy-sensor) observations.

ADR-0005's policy trained on *perfect* state and only *deployed* on estimates. This closes that gap:
`InterceptionEnv(sensor=..., estimator_factory=...)` feeds the policy a **seeker-on-interceptor
radar + EKF estimate** during training (reward/intercept still use truth), so it learns to act under
sensing noise. We then sweep deploy-time sensor noise and compare, head-to-head via
`EstimatingGuidance`, the **truth-trained** policy vs. this **estimate-trained** policy — does
training on noise help when the seeker is noisy?

Run:
    python experiments/p31_rl_estimated_obs.py --train [--timesteps 700000] [--device cuda]
    python experiments/p31_rl_estimated_obs.py --eval  [--trials 120] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from intercept.benchmark import ParametricScenario
from intercept.core import Engagement, Entity, PointMass2D
from intercept.estimation import EKF, nca_model
from intercept.guidance import EstimatingGuidance, true_pn
from intercept.guidance.rl_policy import RLGuidance
from intercept.sensors import Radar

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
FIG = ROOT / "gallery" / "figures"
RESULTS = ROOT / "results"
A_MAX, INT_SPEED, TGT_SPEED = 250.0, 1000.0, 700.0
SIGMA_TRAIN = 60.0  # train under a moderately-noisy seeker so robustness is learned
MODEL = MODELS / "p31_rl_estimated.zip"
# The clean ablation twin: identical scenario/budget/architecture, but trained on TRUTH observations
# — so the eval isolates the observation source, not scenario familiarity.
MODEL_TRUTH = MODELS / "p31_rl_truth.zip"
P5_MODEL = MODELS / "p5_ppo_interceptor.zip"


def _scenario() -> ParametricScenario:
    # A weaving target stresses the estimator (the EKF lags the maneuver), so seeker noise actually
    # bites — the regime where training on estimates can matter.
    return ParametricScenario(
        name="estobs",
        interceptor_speed=INT_SPEED,
        interceptor_a_max=A_MAX,
        target_speed=TGT_SPEED,
        range_min=5000,
        range_max=8000,
        offset_min=-1500,
        offset_max=1500,
        dt=0.02,
        t_max=16.0,
        kill_radius=20.0,
        target_heading_deg=170.0,
        maneuver={"type": "weave", "amplitude": 120.0, "frequency": 0.3},
    )


def _ekf_factory(x0, p0):
    return EKF(lambda d: nca_model(d, 50.0, ndim=2), x0, p0)


def train(args, truth_twin: bool = False) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    from intercept.envs import InterceptionEnv, RewardConfig

    out = MODEL_TRUTH if truth_twin else MODEL

    def make_env():
        kw = {}
        if not truth_twin:
            kw = dict(
                sensor=Radar(sigma_range=SIGMA_TRAIN, sigma_bearing=0.006),
                estimator_factory=_ekf_factory,
            )
        return InterceptionEnv(
            _scenario(),
            RewardConfig(k_effort=0.02),
            obs_mode="basic",
            action_mode="residual_pn",
            residual_scale=0.5,
            **kw,
        )

    raw = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)
    venv = VecNormalize(raw, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.999)
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
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    MODELS.mkdir(parents=True, exist_ok=True)
    model.save(out)
    venv.save(str(out.with_suffix(".vec.pkl")))
    print(f"Saved {out}")


def _obs_norm(path: Path):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from intercept.envs import InterceptionEnv

    vecp = path.with_suffix(".vec.pkl")
    if path == P5_MODEL:
        vecp = MODELS / "p5_vecnormalize.pkl"
    if not vecp.exists():
        return None
    vec = VecNormalize.load(str(vecp), DummyVecEnv([lambda: InterceptionEnv(_scenario())]))
    vec.training = False
    return vec.normalize_obs


def _p_intercept(make_guidance, sigma, trials, seed):
    idyn = PointMass2D(a_max=A_MAX)
    hits = 0
    for s in range(trials):
        rng = np.random.default_rng(seed + s)
        spec = _scenario().sample(rng)
        radar = Radar(sigma_range=sigma, sigma_bearing=max(1e-4, sigma * 1e-4))
        guid = EstimatingGuidance("target", radar, _ekf_factory, make_guidance(), rng)
        intc = Entity(
            "interceptor", idyn, spec.interceptor_state, controller=guid, role="interceptor"
        )
        tgt = Entity(
            "target",
            spec.target_dynamics,
            spec.target_state,
            controller=spec.target_controller,
            role="target",
        )
        res = Engagement(
            [intc, tgt],
            interceptor="interceptor",
            target="target",
            dt=spec.dt,
            t_max=spec.t_max,
            kill_radius=spec.kill_radius,
        ).run()
        hits += int(res.intercepted)
    return hits / trials


def evaluate(args) -> None:
    show = not args.no_show
    from stable_baselines3 import PPO

    def rl_factory(path):
        model = PPO.load(str(path), device="cpu")
        norm = _obs_norm(path)
        return lambda: RLGuidance(
            "target",
            model,
            a_max=A_MAX,
            obs_norm=norm,
            obs_mode="basic",
            action_mode="residual_pn",
            residual_scale=0.5,
        )

    policies = {"PN (on estimate)": lambda: true_pn("target", N=4.0)}
    if MODEL_TRUTH.exists():  # ablation twin: same scenario/budget, truth observations
        policies["RL truth-trained (twin)"] = rl_factory(MODEL_TRUTH)
    elif P5_MODEL.exists():  # fallback reference if the twin is not trained yet
        policies["RL truth-trained"] = rl_factory(P5_MODEL)
    if MODEL.exists():
        policies["RL estimate-trained (P31)"] = rl_factory(MODEL)

    sigmas = [1.0, 25.0, 50.0, 100.0, 200.0]
    print("=" * 64)
    print(f"Training RL on estimated obs — P(intercept) vs seeker noise, {args.trials} trials")
    print("=" * 64)
    curves = {}
    header = "sigma_range (m)".ljust(26) + "".join(f"{s:>8.0f}" for s in sigmas)
    print(header)
    for name, fac in policies.items():
        ys = [_p_intercept(fac, s, args.trials, args.seed) for s in sigmas]
        curves[name] = ys
        print(f"{name:26s}" + "".join(f"{y:8.2f}" for y in ys))
    print("=" * 64)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, ys in curves.items():
        ax.plot(sigmas, ys, "o-", lw=2, label=name)
    ax.set_xlabel("seeker range noise σ (m)")
    ax.set_ylabel("P(intercept)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("RL trained on estimated obs vs. truth-trained, under seeker noise")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p31_rl_estimated_obs.png", dpi=150)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p31_rl_estimated_obs.csv").write_text(
        "policy,"
        + ",".join(str(s) for s in sigmas)
        + "\n"
        + "\n".join(n + "," + ",".join(f"{y:.3f}" for y in curves[n]) for n in curves)
    )
    print(f"Figure: {FIG / 'p31_rl_estimated_obs.png'}")
    plt.show() if show else plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL on estimated observations")
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--train-truth",
        action="store_true",
        help="train the truth-observation ablation twin (same scenario/budget)",
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--timesteps", type=int, default=700_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if not args.train and not args.train_truth and not args.eval:
        args.eval = MODEL.exists()
        args.train = not args.eval
    if args.train:
        train(args)
    if args.train_truth:
        train(args, truth_twin=True)
    if args.eval:
        evaluate(args)


if __name__ == "__main__":
    main()
