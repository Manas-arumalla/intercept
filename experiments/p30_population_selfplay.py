"""P30 — population self-play: fictitious play against a POOL of past opponents.

P27 showed single-step alternating self-play is non-transitive: interceptor gen-2 mastered evader
gen-1 but *catastrophically forgot* evader gen-0 (0.80 -> 0.18). The fix (ADR-0023) is **fictitious
play**: train the interceptor against a *pool* of past evader generations sampled per episode
(`InterceptionEnv(opponent_factory=...)`), so it must stay good against **all** of them at once.

We then re-evaluate the cross-table and show the pool-trained interceptor catches *both* evader
generations — no forgetting — unlike the latest-opponent gen-2. Identical seeds / dynamics (fair),
realistic speeds (matched within ~1.45x).

Run:
    python experiments/p30_population_selfplay.py --train [--timesteps 800000] [--device cuda]
    python experiments/p30_population_selfplay.py --eval  [--trials 150] [--no-show]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from intercept.guidance import AugmentedPN, true_pn

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import sibling experiment helpers
from p27_converged_selfplay import (  # noqa: E402
    BASELINE,
    EVADER_G0,
    EVADER_G1,
    INT_G1,
    INT_G2,
    PN_N,
    RESIDUAL_SCALE,
    _evader_factory,
    _interceptor_guidance,
    _load,
    _norm,
    _p_intercept,
    _sample_geoms,
    _scenario,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
FIG = ROOT / "gallery" / "figures"
RESULTS = ROOT / "results"
INT_POOL = MODELS / "p30_interceptor_pool.zip"


def _pool_opponent_factory(seed: int = 0, p_gen0: float = 0.5):
    """Per-reset sampler over the evader pool {gen-0, gen-1} for fictitious play.

    Uniform (``p_gen0=0.5``) with un-normalized reward (see ``train``) balances both opponents;
    earlier over-weighting traded gen-1 away. ``p_gen0`` can prioritize the harder opponent."""
    makers = [_evader_factory(EVADER_G0), _evader_factory(EVADER_G1)]
    rng = np.random.default_rng(seed)
    return lambda: makers[0 if rng.random() < p_gen0 else 1]()


def train(args) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    from intercept.envs import InterceptionEnv, RewardConfig

    if not EVADER_G0.exists() or not EVADER_G1.exists():
        raise SystemExit("Need evader gen-0 (P22) and gen-1 (P27) models first.")

    def make_env():
        return InterceptionEnv(
            _scenario(),
            RewardConfig(k_effort=0.02),
            obs_mode="rich",
            action_mode="residual_pn",
            residual_scale=RESIDUAL_SCALE,
            pn_N=PN_N,
            baseline=BASELINE,
            opponent_factory=_pool_opponent_factory(args.seed),
        )

    print("=== Train interceptor vs evader POOL {gen-0, gen-1} (fictitious play) ===")
    raw = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)
    # norm_reward=False: per-episode opponent switching makes returns bimodal (easy gen-1 vs hard
    # gen-0); normalizing that destabilizes PPO. Normalize observations only.
    venv = VecNormalize(raw, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.999)
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
    model.save(INT_POOL)
    venv.save(str(INT_POOL.with_suffix(".vec.pkl")))
    print(f"Saved {INT_POOL}")


def evaluate(args) -> None:
    show = not args.no_show
    from intercept.envs import InterceptionEnv

    interceptors = {
        "True PN": lambda t: true_pn(t, N=4.0),
        "Aug PN": lambda t: AugmentedPN(t, N=4.0),
    }
    for label, path in [("Int gen-1", INT_G1), ("Int gen-2", INT_G2), ("Int POOL", INT_POOL)]:
        if path.exists():
            m = _load(path)
            nm = _norm(path, lambda: InterceptionEnv(_scenario(), obs_mode="rich"))
            interceptors[label] = lambda t, m=m, nm=nm: _interceptor_guidance(m, nm, target=t)

    evaders = {}
    for label, path in [("Eva gen-0", EVADER_G0), ("Eva gen-1", EVADER_G1)]:
        if path.exists():
            evaders[label] = _evader_factory(path)

    geoms = _sample_geoms(args.trials, seed=args.seed)
    rows, cols = list(interceptors), list(evaders)
    table = {}
    print("=" * 64)
    print(f"POPULATION SELF-PLAY cross-table — P(intercept), {args.trials} trials")
    print("=" * 64)
    print("interceptor \\ evader".ljust(22) + "".join(f"{c:>12s}" for c in cols) + f"{'min':>8s}")
    for r in rows:
        vals = [_p_intercept(interceptors[r], evaders[c], geoms) for c in cols]
        table[r] = vals
        print(f"{r:22s}" + "".join(f"{v:12.2f}" for v in vals) + f"{min(vals):8.2f}")
    print("=" * 64)
    print("('min' = worst-case over evader generations — robustness; higher is better)")

    import matplotlib.pyplot as plt

    M = np.array([table[r] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=11)
    ax.set_xlabel("evader generation")
    ax.set_ylabel("interceptor")
    ax.set_title("Population self-play — pool-trained interceptor avoids forgetting")
    fig.colorbar(im, ax=ax, label="P(intercept)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p30_population_selfplay.png", dpi=150)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p30_population_selfplay.csv").write_text(
        "interceptor,"
        + ",".join(cols)
        + "\n"
        + "\n".join(r + "," + ",".join(f"{v:.3f}" for v in table[r]) for r in rows)
    )
    print(f"Figure: {FIG / 'p30_population_selfplay.png'}")
    plt.show() if show else plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Population self-play (fictitious play)")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--timesteps", type=int, default=800_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if not args.train and not args.eval:
        args.eval = INT_POOL.exists()
        args.train = not args.eval
    if args.train:
        train(args)
    if args.eval:
        evaluate(args)


if __name__ == "__main__":
    main()
