"""P27 — converged self-play: alternating evader/interceptor generations (the arms race).

P25 did one round (interceptor gen-1 trained vs the frozen gen-0 evader). This continues the arms
race (ADR-0020): alternately retrain each side against the other's latest policy and tabulate a
**cross-table** of P(intercept) for every interceptor generation vs every evader generation. Because
two-sided learning is non-transitive, the cross-table (not a single number) is the result: it shows
each new generation beating the opponent it trained against, and whether the race is converging.

Generations (reusing P22/P25, adding two):
  evader   gen-0 = P22 (vs True PN)            interceptor gen-1 = P25 (vs evader gen-0)
  evader   gen-1 = trained vs interceptor g1   interceptor gen-2 = trained vs evader g1

Run:
    python experiments/p27_converged_selfplay.py --train  [--timesteps 600000] [--device cuda]
    python experiments/p27_converged_selfplay.py --eval   [--trials 150] [--no-show]
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
FIG = ROOT / "gallery" / "figures"
RESULTS = ROOT / "results"
INT_AMAX, EVA_AMAX = 40 * G0, 35 * G0
INT_SPEED, EVA_SPEED = 1000.0, 700.0
DT, T_MAX, KILL = 0.01, 16.0, 20.0
RESIDUAL_SCALE, PN_N, BASELINE = 0.5, 4.0, "apn"

EVADER_G0 = MODELS / "p22_ppo_evader.zip"
INT_G1 = MODELS / "p25_interceptor_gen1.zip"
EVADER_G1 = MODELS / "p27_evader_gen1.zip"
INT_G2 = MODELS / "p27_interceptor_gen2.zip"


def _norm(model_path: Path, make_env):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    vecp = model_path.with_suffix(".vec.pkl")
    if not vecp.exists():
        return None
    vec = VecNormalize.load(str(vecp), DummyVecEnv([make_env]))
    vec.training = False
    return vec.normalize_obs


def _scenario() -> ParametricScenario:
    return ParametricScenario(
        name="sp",
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


def _interceptor_guidance(model, norm, target="target"):
    return RLGuidance(
        target,
        model,
        a_max=INT_AMAX,
        obs_norm=norm,
        obs_mode="rich",
        gravity=G0,
        action_mode="residual_pn",
        residual_scale=RESIDUAL_SCALE,
        pn_N=PN_N,
        baseline=BASELINE,
    )


def _load(path: Path):
    from stable_baselines3 import PPO

    return PPO.load(str(path), device="cpu")


def _evader_factory(model_path: Path):
    """RLEvader controller from a saved evader model (chases 'interceptor')."""
    from intercept.envs import EvaderEnv

    model = _load(model_path)
    norm = _norm(model_path, lambda: EvaderEnv())
    return lambda: RLEvader("interceptor", model, a_max=EVA_AMAX, obs_norm=norm)


def _pursuer_factory(int_path: Path):
    """RLGuidance pursuer (chases 'evader') from a saved interceptor model (EvaderEnv self-play)."""
    from intercept.envs import InterceptionEnv

    model = _load(int_path)
    norm = _norm(int_path, lambda: InterceptionEnv(_scenario(), obs_mode="rich"))
    return lambda: _interceptor_guidance(model, norm, target="evader")


def _train_ppo(make_env, out: Path, timesteps: int, device: str, seed: int, n_envs: int):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    raw = make_vec_env(make_env, n_envs=n_envs, seed=seed)
    venv = VecNormalize(raw, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.999)
    model = PPO(
        "MlpPolicy",
        venv,
        device=device,
        seed=seed,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.999,
        ent_coef=0.0,
        learning_rate=3e-4,
        tensorboard_log=str(RUNS),
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    MODELS.mkdir(parents=True, exist_ok=True)
    model.save(out)
    venv.save(str(out.with_suffix(".vec.pkl")))
    print(f"Saved {out}")


def train(args) -> None:
    from intercept.envs import EvaderEnv, InterceptionEnv, RewardConfig

    if not INT_G1.exists() or not EVADER_G0.exists():
        raise SystemExit("Need P22 (evader gen-0) and P25 (interceptor gen-1) models first.")

    # 1) evader gen-1: flee the gen-1 interceptor (P25).
    print("\n=== Train evader gen-1 vs interceptor gen-1 ===")
    pursuer = _pursuer_factory(INT_G1)
    _train_ppo(
        lambda: EvaderEnv(pursuer_factory=pursuer),
        EVADER_G1,
        args.timesteps,
        args.device,
        args.seed,
        args.n_envs,
    )

    # 2) interceptor gen-2: catch the gen-1 evader.
    print("\n=== Train interceptor gen-2 vs evader gen-1 ===")
    evader = _evader_factory(EVADER_G1)
    _train_ppo(
        lambda: InterceptionEnv(
            _scenario(),
            RewardConfig(k_effort=0.02),
            obs_mode="rich",
            action_mode="residual_pn",
            residual_scale=RESIDUAL_SCALE,
            pn_N=PN_N,
            baseline=BASELINE,
            opponent=evader(),
        ),
        INT_G2,
        args.timesteps,
        args.device,
        args.seed,
        args.n_envs,
    )


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


def _p_intercept(int_guidance_factory, evader_factory, geoms):
    hits = 0
    for s0, s1 in geoms:
        intc = Entity(
            "interceptor",
            AeroMissile2D(a_max=INT_AMAX, tau=0.2),
            s0.copy(),
            controller=int_guidance_factory("target"),
            role="interceptor",
        )
        tgt = Entity(
            "target",
            AeroMissile2D(a_max=EVA_AMAX, tau=0.3),
            s1.copy(),
            controller=evader_factory(),
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
    return hits / len(geoms)


def evaluate(args) -> None:
    show = not args.no_show
    from intercept.envs import InterceptionEnv

    # Interceptor generations (rows).
    interceptors = {
        "True PN": lambda t: true_pn(t, N=4.0),
        "Aug PN": lambda t: AugmentedPN(t, N=4.0),
    }
    for label, path in [("Int gen-1", INT_G1), ("Int gen-2", INT_G2)]:
        if path.exists():
            m = _load(path)
            nm = _norm(path, lambda: InterceptionEnv(_scenario(), obs_mode="rich"))
            interceptors[label] = lambda t, m=m, nm=nm: _interceptor_guidance(m, nm, target=t)

    # Evader generations (columns).
    evaders = {}
    for label, path in [("Eva gen-0", EVADER_G0), ("Eva gen-1", EVADER_G1)]:
        if path.exists():
            evaders[label] = _evader_factory(path)

    geoms = _sample_geoms(args.trials, seed=args.seed)
    rows, table = list(interceptors), {}
    cols = list(evaders)
    print("=" * 64)
    print(f"CONVERGED SELF-PLAY cross-table — P(intercept), {args.trials} trials")
    print("=" * 64)
    header = "interceptor \\ evader".ljust(22) + "".join(f"{c:>12s}" for c in cols)
    print(header)
    for r in rows:
        vals = [_p_intercept(interceptors[r], evaders[c], geoms) for c in cols]
        table[r] = vals
        print(f"{r:22s}" + "".join(f"{v:12.2f}" for v in vals))
    print("=" * 64)

    import matplotlib.pyplot as plt

    M = np.array([table[r] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="black", fontsize=11)
    ax.set_xlabel("evader generation")
    ax.set_ylabel("interceptor")
    ax.set_title("Self-play arms race — P(intercept) cross-table")
    fig.colorbar(im, ax=ax, label="P(intercept)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "p27_converged_selfplay.png", dpi=150)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p27_converged_selfplay.csv").write_text(
        "interceptor,"
        + ",".join(cols)
        + "\n"
        + "\n".join(r + "," + ",".join(f"{v:.3f}" for v in table[r]) for r in rows)
    )
    print(f"Figure: {FIG / 'p27_converged_selfplay.png'}")
    plt.show() if show else plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Converged self-play (arms race)")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if not args.train and not args.eval:
        args.eval = INT_G2.exists()
        args.train = not args.eval
    if args.train:
        train(args)
    if args.eval:
        evaluate(args)


if __name__ == "__main__":
    main()
