"""P25 — one self-play round: train an interceptor against the *learned* evader.

The adversarial-RL evader (P22) defeats True PN (it escapes ~93% of the time). This closes one round
of the arms race (ADR-0017): train a fresh residual-PN interceptor in `InterceptionEnv` whose
**opponent is that learned evader** (via `InterceptionEnv(opponent=RLEvader(...))`), then check it
**hardens**: does the gen-1 interceptor catch the evader that beat PN? Held-out, it is compared with
True PN and Augmented PN, all facing the same frozen gen-0 evader.

Run:
    python experiments/p25_selfplay.py --train [--timesteps 700000] [--device cuda]
    python experiments/p25_selfplay.py --eval  [--trials 150] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from intercept.adversary.rl_evader import RLEvader
from intercept.benchmark import ParametricScenario
from intercept.core import Engagement, Entity
from intercept.core.aero import G0, AeroMissile2D
from intercept.guidance import AugmentedPN, true_pn
from intercept.guidance.rl_policy import RLGuidance

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"
RESULTS = ROOT / "results"
INT_AMAX, EVA_AMAX = 40 * G0, 35 * G0
INT_SPEED, EVA_SPEED = 1000.0, 700.0
DT, T_MAX, KILL = 0.01, 16.0, 20.0
RESIDUAL_SCALE, PN_N = 0.5, 4.0
# Residual on the APN baseline: PN alone fails against the learned evader (~0.07), so a residual on
# it can't bootstrap; Augmented PN already catches it ~0.71, a competent base for the residual.
BASELINE = "apn"
EVADER_MODEL = MODELS / "p22_ppo_evader.zip"
DEFAULT_MODEL = MODELS / "p25_interceptor_gen1.zip"


def _load_evader():
    """Frozen gen-0 evader (from P22) as an RLEvader opponent factory."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from intercept.envs import EvaderEnv

    if not EVADER_MODEL.exists():
        raise SystemExit(f"Evader model not found: {EVADER_MODEL} — run p22 --train first.")
    model = PPO.load(str(EVADER_MODEL), device="cpu")
    norm = None
    vecp = EVADER_MODEL.with_suffix(".vec.pkl")
    if vecp.exists():
        vec = VecNormalize.load(str(vecp), DummyVecEnv([lambda: EvaderEnv()]))
        vec.training = False
        norm = vec.normalize_obs
    return lambda: RLEvader("interceptor", model, a_max=EVA_AMAX, obs_norm=norm)


def _scenario() -> ParametricScenario:
    # Matches the P22 evader's training profile (so the opponent faces a familiar pursuer geometry).
    return ParametricScenario(
        name="selfplay",
        model="aero",
        interceptor_speed=INT_SPEED,
        interceptor_a_max=INT_AMAX,
        target_speed=EVA_SPEED,
        target_a_max=EVA_AMAX,
        interceptor_tau=0.2,
        target_tau=0.3,
        target_heading_deg=180.0,
        offset_min=-1500,
        offset_max=1500,
        range_min=5000,
        range_max=8000,
        dt=DT,
        t_max=T_MAX,
        kill_radius=KILL,
    )


def train(args) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    from intercept.envs import InterceptionEnv, RewardConfig

    make_evader = _load_evader()
    MODELS.mkdir(parents=True, exist_ok=True)

    def make_env():
        return InterceptionEnv(
            _scenario(),
            RewardConfig(k_effort=0.02),
            obs_mode="rich",
            action_mode="residual_pn",
            residual_scale=RESIDUAL_SCALE,
            pn_N=PN_N,
            baseline=BASELINE,
            opponent=make_evader(),
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
    model.save(args.out)
    venv.save(str(Path(args.out).with_suffix(".vec.pkl")))
    print(f"\nSaved gen-1 interceptor to {args.out}")


def _sample_geoms(n, seed):
    rng = np.random.default_rng(seed)
    idyn, edyn = AeroMissile2D(a_max=INT_AMAX, tau=0.2), AeroMissile2D(a_max=EVA_AMAX, tau=0.3)
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


def evaluate(args) -> None:
    show = not args.no_show
    make_evader = _load_evader()
    interceptors = {
        "True PN": lambda t: true_pn(t, N=4.0),
        "Augmented PN": lambda t: AugmentedPN(t, N=4.0),
    }
    gen1 = Path(args.model)
    if gen1.exists():
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from intercept.envs import InterceptionEnv

        m = PPO.load(str(gen1), device="cpu")
        norm = None
        vecp = gen1.with_suffix(".vec.pkl")
        if vecp.exists():
            vec = VecNormalize.load(
                str(vecp), DummyVecEnv([lambda: InterceptionEnv(_scenario(), obs_mode="rich")])
            )
            vec.training = False
            norm = vec.normalize_obs
        interceptors["Interceptor gen-1 (self-play)"] = lambda t: RLGuidance(
            t,
            m,
            a_max=INT_AMAX,
            obs_norm=norm,
            obs_mode="rich",
            gravity=G0,
            action_mode="residual_pn",
            residual_scale=RESIDUAL_SCALE,
            pn_N=PN_N,
            baseline=BASELINE,
        )
    else:
        print(f"(gen-1 model not found at {gen1}; showing classical only)")

    geoms = _sample_geoms(args.trials, seed=args.seed)
    print("=" * 60)
    print(f"SELF-PLAY ROUND — interceptors vs. the learned gen-0 evader ({args.trials} trials)")
    print("=" * 60)
    p_int = {}
    for name, fac in interceptors.items():
        hits = 0
        for s0, s1 in geoms:
            intc = Entity(
                "interceptor",
                AeroMissile2D(a_max=INT_AMAX, tau=0.2),
                s0.copy(),
                controller=fac("target"),
                role="interceptor",
            )
            tgt = Entity(
                "target",
                AeroMissile2D(a_max=EVA_AMAX, tau=0.3),
                s1.copy(),
                controller=make_evader(),
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
            hits += int(res.intercepted)
        p_int[name] = hits / len(geoms)
        print(f"  {name:32s}: P(intercept) = {p_int[name]:.2f}")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(p_int)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"][: len(names)]
    ax.bar(names, [p_int[n] for n in names], color=colors)
    for i, n in enumerate(names):
        ax.text(i, p_int[n] + 0.02, f"{p_int[n]:.2f}", ha="center")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("P(intercept) vs. learned evader")
    ax.set_title("Self-play round: training against the learned evader hardens the interceptor")
    plt.xticks(rotation=12, ha="right")
    fig.tight_layout()
    GALLERY.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p25_selfplay.png", dpi=150)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p25_selfplay.csv").write_text(
        "interceptor,p_intercept_vs_gen0_evader\n" + "\n".join(f"{n},{p_int[n]:.3f}" for n in names)
    )
    print(f"Figure: {FIG / 'p25_selfplay.png'}")
    plt.show() if show else plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play round: interceptor vs learned evader")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--timesteps", type=int, default=700_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=150)
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
        evaluate(args)


if __name__ == "__main__":
    main()
