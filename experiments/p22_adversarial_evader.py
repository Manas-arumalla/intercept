"""P22 — adversarial-RL evader: a learned target that evades a pursuing interceptor.

Trains a PPO **evader** (`EvaderEnv`: maximize the interceptor's miss against a True-PN pursuer),
then deploys it via `RLEvader` and measures how hard an adversary it is — P(intercept) and miss
distance against True PN / Augmented PN / Sliding-mode interceptors — compared to a straight target,
a scripted weave, and the game-theoretic `optimal_evader`. A *lower* P(intercept) / *larger* miss
means a harder adversary (ADR-0015; complements the scripted + game-theoretic evaders).

Run:
    python experiments/p22_adversarial_evader.py --train [--timesteps 800000] [--device cuda]
    python experiments/p22_adversarial_evader.py --eval  [--trials 150] [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from intercept.adversary import optimal_evader, weave
from intercept.adversary.rl_evader import RLEvader
from intercept.core import Engagement, Entity
from intercept.core.aero import G0, AeroMissile2D
from intercept.guidance import AugmentedPN, sliding_mode, true_pn

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
GALLERY = ROOT / "gallery"
FIG = GALLERY / "figures"
ANIM = GALLERY / "animations"
RESULTS = ROOT / "results"
INT_SPEED, EVA_SPEED = 1000.0, 700.0
INT_AMAX, EVA_AMAX = 40 * G0, 35 * G0
DT, T_MAX, KILL = 0.01, 16.0, 20.0
DEFAULT_MODEL = MODELS / "p22_ppo_evader.zip"


def train(args) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    from intercept.envs import EvaderEnv

    MODELS.mkdir(parents=True, exist_ok=True)
    raw = make_vec_env(lambda: EvaderEnv(), n_envs=args.n_envs, seed=args.seed)
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
    print(f"\nSaved evader policy to {args.out}")


def _sample_geometries(n: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sample (interceptor_state, evader_state) pairs matching EvaderEnv's reset distribution."""
    rng = np.random.default_rng(seed)
    out = []
    idyn, edyn = AeroMissile2D(a_max=INT_AMAX, tau=0.2), AeroMissile2D(a_max=EVA_AMAX, tau=0.3)
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


def _run(int_state, eva_state, guidance_factory, evader_controller) -> tuple[bool, float]:
    interceptor = Entity(
        "interceptor",
        AeroMissile2D(a_max=INT_AMAX, tau=0.2),
        int_state.copy(),
        controller=guidance_factory("target"),
        role="interceptor",
    )
    target = Entity(
        "target",
        AeroMissile2D(a_max=EVA_AMAX, tau=0.3),
        eva_state.copy(),
        controller=evader_controller,
        role="target",
    )
    res = Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=DT,
        t_max=T_MAX,
        kill_radius=KILL,
    ).run()
    return res.intercepted, res.miss_distance


def evaluate(args) -> None:
    show = not args.no_show
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path} — run with --train first.")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from intercept.envs import EvaderEnv

    model = PPO.load(str(model_path), device="cpu")
    obs_norm = None
    vecp = model_path.with_suffix(".vec.pkl")
    if vecp.exists():
        vec = VecNormalize.load(str(vecp), DummyVecEnv([lambda: EvaderEnv()]))
        vec.training = False
        obs_norm = vec.normalize_obs

    evaders = {
        "straight": lambda: None,
        "weave 18g": lambda: weave(18 * G0, frequency=0.3),
        "optimal (anti-LOS)": lambda: optimal_evader("interceptor"),
        "RL evader": lambda: RLEvader("interceptor", model, a_max=EVA_AMAX, obs_norm=obs_norm),
    }
    interceptors = {
        "True PN": lambda t: true_pn(t, N=4.0),
        "Augmented PN": lambda t: AugmentedPN(t, N=4.0),
        "Sliding-mode": lambda t: sliding_mode(t, eta=300.0),
    }

    geoms = _sample_geometries(args.trials, seed=args.seed)
    print("=" * 70)
    print(f"ADVERSARIAL EVADER - P(intercept) by the interceptor ({args.trials} trials)")
    print("  (lower P / larger miss = harder evader)")
    print("=" * 70)
    print("evader".ljust(22) + "".join(f"{name:>16}" for name in interceptors))
    grid = np.zeros((len(evaders), len(interceptors)))
    miss_grid = np.zeros_like(grid)
    for i, (ename, emake) in enumerate(evaders.items()):
        cells = []
        for j, ifac in enumerate(interceptors.values()):
            outs = [_run(s0, s1, ifac, emake()) for (s0, s1) in geoms]
            p = float(np.mean([o[0] for o in outs]))
            miss = float(np.median([o[1] for o in outs]))
            grid[i, j], miss_grid[i, j] = p, miss
            cells.append(f"{p:>7.2f}/{miss:>6.0f}m")
        print(ename.ljust(22) + "".join(f"{c:>16}" for c in cells))
    print("=" * 70)

    RESULTS.mkdir(parents=True, exist_ok=True)
    hdr = "evader," + ",".join(interceptors) + "\n"
    body = "\n".join(
        list(evaders)[i] + "," + ",".join(f"{grid[i, j]:.3f}" for j in range(len(interceptors)))
        for i in range(len(evaders))
    )
    (RESULTS / "p22_adversarial_evader.csv").write_text(hdr + body)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, cmap="magma_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(interceptors)), list(interceptors))
    ax.set_yticks(range(len(evaders)), list(evaders))
    for i in range(len(evaders)):
        for j in range(len(interceptors)):
            ax.text(
                j,
                i,
                f"{grid[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if grid[i, j] > 0.5 else "black",
            )
    ax.set_title("P(intercept): interceptor (x) vs. evader (y) — lower row = harder evader")
    fig.colorbar(im, ax=ax, label="P(intercept)")
    fig.tight_layout()
    GALLERY.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p22_adversarial_evader.png", dpi=150)
    print(f"Figure: {FIG / 'p22_adversarial_evader.png'}")
    plt.show() if show else plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial-RL evader: train / evaluate")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--timesteps", type=int, default=800_000)
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
