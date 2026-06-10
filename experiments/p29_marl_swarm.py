"""P29 — MARL cooperative swarm: *learned* target allocation vs. the Hungarian baseline.

A centralized policy (`CentralizedSwarmEnv`) learns to allocate **3 interceptors over 5 inbound
threats** — under-resourced, so coordination matters: poor allocation lets threats leak. We compare
the learned allocator, head-to-head on **identical seeded scenarios** (fairness invariant), against
(a) the analytic **Hungarian WTA** and (b) a **random** allocator — all sharing the same dynamics,
PN guidance, and realistic comparable speeds. The question: can a learned cooperative policy
match or beat the optimization baseline at minimizing leakers?

Run:
    python experiments/p29_marl_swarm.py --train [--timesteps 1500000] [--device cuda]
    python experiments/p29_marl_swarm.py --eval  [--trials 200] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
FIG = ROOT / "gallery" / "figures"
RESULTS = ROOT / "results"
N_INT, N_THREAT = 3, 5
MODEL = MODELS / "p29_marl_swarm.zip"


def _make_env():
    from intercept.envs import CentralizedSwarmEnv

    return CentralizedSwarmEnv(n_int=N_INT, n_threat=N_THREAT)


def train(args) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    raw = make_vec_env(_make_env, n_envs=args.n_envs, seed=args.seed)
    venv = VecNormalize(raw, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.997)
    model = PPO(
        "MlpPolicy",
        venv,
        device=args.device,
        seed=args.seed,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.997,
        ent_coef=0.005,
        learning_rate=3e-4,
        tensorboard_log=str(RUNS),
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    MODELS.mkdir(parents=True, exist_ok=True)
    model.save(MODEL)
    venv.save(str(MODEL.with_suffix(".vec.pkl")))
    print(f"Saved {MODEL}")


def _hungarian_action(env):
    """One-hot N×M action encoding the Hungarian weapon-target assignment of the current state."""
    from intercept.multiagent.assignment import weapon_target_assignment

    live_i = np.where(env.alive_i)[0]
    live_t = np.where(env.alive_t)[0]
    pref = -np.ones((env.n_int, env.n_threat), dtype=np.float32)
    if live_i.size and live_t.size:
        amap = weapon_target_assignment(
            [env.ipos[i] for i in live_i], [env.tpos[j] for j in live_t], ndim=2
        )
        for li, i in enumerate(live_i):
            pref[i, live_t[amap[li]]] = 1.0
    return pref.reshape(-1)


def _run_episode(env, seed, policy):
    obs, _ = env.reset(seed=seed)
    done = False
    info = {"hits": 0, "leaks": 0}
    while not done:
        a = policy(env, obs)
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
    return info["hits"], info["leaks"]


def evaluate(args) -> None:
    show = not args.no_show
    from intercept.envs import CentralizedSwarmEnv

    policies = {
        "Random": lambda env, obs: env.action_space.sample(),
        "Hungarian WTA": lambda env, obs: _hungarian_action(env),
    }
    if MODEL.exists():
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        model = PPO.load(str(MODEL), device="cpu")
        vecp = MODEL.with_suffix(".vec.pkl")
        norm = None
        if vecp.exists():
            vec = VecNormalize.load(str(vecp), DummyVecEnv([_make_env]))
            vec.training = False
            norm = vec.normalize_obs

        def learned(env, obs):
            o = norm(obs.reshape(1, -1))[0] if norm else obs
            a, _ = model.predict(o, deterministic=True)
            return a

        policies["Learned (MARL)"] = learned
    else:
        print(f"(no model at {MODEL}; comparing Random vs Hungarian only)")

    env = CentralizedSwarmEnv(n_int=N_INT, n_threat=N_THREAT)
    seeds = list(range(1000, 1000 + args.trials))
    print("=" * 56)
    print(f"MARL swarm — {N_INT} interceptors vs {N_THREAT} threats, {args.trials} trials")
    print("=" * 56)
    summary = {}
    for name, pol in policies.items():
        leaks = [_run_episode(env, s, pol)[1] for s in seeds]
        summary[name] = float(np.mean(leaks))
        print(f"  {name:18s}: mean leakers = {summary[name]:.2f} / {N_THREAT}")
    print("=" * 56)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(summary)
    ax.bar(
        names, [summary[n] for n in names], color=["#9aa0a6", "#1f77b4", "#2ca02c"][: len(names)]
    )
    for i, n in enumerate(names):
        ax.text(i, summary[n] + 0.03, f"{summary[n]:.2f}", ha="center")
    ax.set_ylabel(f"mean leakers (of {N_THREAT})")
    ax.set_title(f"Cooperative allocation — learned vs. Hungarian ({N_INT} vs {N_THREAT})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p29_marl_swarm.png", dpi=150)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p29_marl_swarm.csv").write_text(
        "policy,mean_leakers\n" + "\n".join(f"{n},{summary[n]:.3f}" for n in names)
    )
    print(f"Figure: {FIG / 'p29_marl_swarm.png'}")
    plt.show() if show else plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="MARL cooperative swarm")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if not args.train and not args.eval:
        args.eval = MODEL.exists()
        args.train = not args.eval
    if args.train:
        train(args)
    if args.eval:
        evaluate(args)


if __name__ == "__main__":
    main()
