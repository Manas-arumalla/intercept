# Reinforcement-learning guidance (the centerpiece)

Modules: [`intercept/envs/interception_env.py`](../../intercept/envs/interception_env.py),
[`intercept/guidance/rl_policy.py`](../../intercept/guidance/rl_policy.py).
Design contract: [ADR-0005](../adr/0005-rl-env-contract.md).

## Why this is the centerpiece

The project's thesis is a *rigorous* comparison of classical and learned guidance. P5 trains a deep
RL policy to intercept and benchmarks it against PN / OGL / APN on **identical dynamics and
held-out geometries**, reporting both intercept rate **and control effort** — the comparison the
literature rarely standardizes (Research §8, §E).

## Environment  (`InterceptionEnv`, Gymnasium)

Built on the *same* `PointMass2D`, `RK4`, and `ParametricScenario` sampler as the benchmark — there
is no separate "RL physics".

- **Observation** (`build_observation`, shared with the policy bridge): normalized relative position,
  relative velocity, and own velocity (6-D). Identical in training and deployment.
- **Action**: `Box(-1, 1)²` scaled to the interceptor's `a_max` and saturated — the *same* control
  authority the classical laws get.
- **Reward** (`RewardConfig`): dense closing reward + LOS-rate shaping (toward parallel navigation)
  − effort/time penalties, with a terminal intercept bonus / miss penalty. The shaping says *what*
  to optimize (hit, efficiently), not *how* — no PN formula is injected.

## Training  (`experiments/p5_train_rl.py`)

PPO (Stable-Baselines3, `MlpPolicy`), 8 vectorized envs wrapped in **`VecNormalize`** (running
observation + reward standardization — the single most important ingredient here; see below),
**curriculum** head-on → crossing → *mixed* (the env samples a geometry per episode in the mixed
stage, countering catastrophic forgetting), 1.2M timesteps. Optional Weights & Biases (`--wandb`);
TensorBoard to `runs/`. Saves `models/p5_ppo_interceptor.zip` + `models/p5_vecnormalize.pkl` (the
frozen normalization stats, reused at deployment so the policy sees identical inputs).

### What it took to make RL learn

The first attempts intercepted ~2–6%. Four fixes, in order of impact:
1. **`VecNormalize`** — the lead signal (cross-range, LOS rate) is tiny in raw-normalized obs, so the
   network ignored it; standardizing inputs/returns took head-on from 5% → **100%**.
2. **Potential-based ZEM reward** — rewarding reduction of the predicted perpendicular miss gives a
   dense lead gradient even when the LOS rate is small and the intercept bonus is too sparse to find.
3. **1-D lateral action** (⟂ velocity) instead of unconstrained 2-D acceleration.
4. **Tunnelling fix** — `segment_min_distance` (closest approach *within* a step), so fast closing
   can't skip the kill radius between discrete steps (fixed in the env *and* the engagement core).

## Deployment & comparison  (`RLGuidance`, `experiments/p5_rl_vs_classical.py`)

`RLGuidance` wraps the trained policy as an ordinary guidance `Controller`, so it runs inside the
same `Engagement` and `run_benchmark` as every classical law — the fairness invariant (ADR-0003)
applies unchanged. Evaluation uses a **held-out seed** (42, disjoint from training).

### Results (held-out seed 42, 100 trials/scenario)

P(intercept), with median miss (m) and mean control effort (∫‖a‖²dt):

| Scenario | RL (PPO) | True PN | Optimal (OGL) | Augmented PN |
|---|---|---|---|---|
| E1 head-on | **0.72** (17 m) | 1.00 (10 m) | 1.00 (10 m) | 1.00 (10 m) |
| E2 crossing | **0.95** (14 m) | 1.00 (13 m) | 1.00 (12 m) | 1.00 (13 m) |
| E3 weaving | **0.64** (18 m) | 1.00 (10 m) | 1.00 (10 m) | 1.00 (10 m) |
| **mean effort** | **28 100** | 14 000 | 14 800 | 17 200 |

Figures: `gallery/figures/p5_rl_vs_classical_pintercept.png` (grouped bars + Wilson CIs),
`gallery/figures/p5_rl_vs_pn_trajectory.png` (RL and PN fly near-identical interception curves on a crossing
shot — RL 13 m vs PN 18 m on that sample). Source: `results/p5_rl_vs_classical.csv`.

**Reading the numbers.** The learned policy intercepts **64–95%** across the suite (best on
crossing, where the LOS-rate signal is strongest) but does **not** match decades-tuned PN's 100%,
and is **less efficient** here (≈2× the control effort, larger miss). This is the
expected outcome for a point-mass PPO agent trained in minutes — and exactly the value of the
platform: a clean, reproducible, fairly-measured comparison rather than a cherry-picked win. The
literature's stronger learned results use hierarchical/curriculum RL and far more training (a clear
extension path).

## Reading the result

A point-mass PPO agent trained in minutes is **not expected to beat decades-tuned PN on raw
intercept rate** — and the benchmark reports it plainly. The value is the *clean, reproducible
comparison*: where the learned policy is competitive, where it trails, and at what control-effort
cost — exactly the apples-to-apples evaluation that is the project's contribution. Stronger learned
results (hierarchical/curriculum RL reaching ~100% in the literature) are a clear extension path.

## Residual RL on the realistic plant (PN + learned correction)

From-scratch PPO does **not** transfer to the realistic (L2 aero) plant: it collapses to a constant
saturated action and intercepts ~0–2 %, even though a hand-coded PN scores ~100 % in the same env
(the gravity + autopilot-lag exploration problem is ill-conditioned). The fix is **residual policy
learning** (Silver et al. 2018; Johannink et al. 2019): the policy outputs a *bounded correction*
added to a pure-PN baseline (`action_mode="residual_pn"`, `pn_baseline_scalar`), so a zero action is
already competent PN — no collapse — and learning only has to find the maneuver-anticipation
correction. A residual-effort penalty makes "do nothing (= PN)" the default. See
[ADR-0011](../adr/0011-residual-rl-guidance.md); train/eval in `experiments/p15_residual_rl.py`.

Held-out (100 trials/scenario, realistic L2 aero), P(intercept):

| Scenario | Residual-RL (PN+res) | True PN | Augmented PN | Sliding-mode | From-scratch PPO |
|---|---|---|---|---|---|
| crossing | **1.00** | 1.00 | 1.00 | 1.00 | 0.01 |
| weave 18 g | **1.00** | 1.00 | 1.00 | 1.00 | 0.01 |
| jink 22 g | **0.68** | 0.81 | 0.93 | 1.00 | 0.00 |

**Reading it.** The residual parameterization **resolves the collapse**: the learned policy
now *runs* at PN-class competence (1.00 / 1.00 on crossing/weave) where from-scratch PPO is ~0 — that
is the contribution, a learned guidance law viable on the realistic plant. The PN-residual MLP does
**not** beat the robust classical laws: on the *unpredictable* random-telegraph jink it reaches 0.68,
trailing pure PN (0.81) and below Augmented PN (0.93) and sliding-mode (1.00). Expected — a
*feedforward* policy cannot anticipate a random jink. Figure: `gallery/figures/p15_residual_rl.png`.

### Recurrent, APN-baseline residual (P16) — the strongest learned variant

Two upgrades aimed at the jink (`experiments/p16_recurrent_residual.py`): (1) the residual corrects
**Augmented PN** (`baseline="apn"`, which feed-forwards the target's *measured* lateral accel from
its state) instead of plain PN; (2) the policy is **recurrent** (`sb3_contrib.RecurrentPPO`,
`MlpLstmPolicy`), so an LSTM gives it *memory* of the target's recent motion — the missing ingredient
for anticipating a random jink. Held-out (100 trials/scenario), P(intercept):

| Scenario | **Recurrent APN-residual** | Residual-PN MLP | True PN | Augmented PN | Sliding-mode |
|---|---|---|---|---|---|
| crossing | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 |
| weave 18 g | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 |
| jink 22 g | **0.95** | 0.68 | 0.81 | 0.93 | 1.00 |

On the jink the recurrent APN-residual reaches **0.95 — beating True PN (0.81) and Augmented PN
(0.93)** at lower aggregate control effort than APN, and trailing only sliding-mode (1.00). The
ablation is clean: the APN baseline + LSTM memory lift the jink from **0.68 → 0.95**. So a *learned*
guidance law does beat the PN family on the realistic plant's hardest case — short of the
robust sliding-mode, but a genuine learned win. Figure: `gallery/figures/p16_recurrent_residual.png`; CSV:
`results/p16_recurrent_residual.csv`. Both residual variants share the mechanism in
[ADR-0011](../adr/0011-residual-rl-guidance.md).

## 3-D RL (residual-PN-3D)

The same recipe extends to three dimensions ([ADR-0014](../adr/0014-three-dimensional-rl.md)):
`InterceptionEnv3D` (a **2-DOF** lateral action in the ⟂-velocity plane, 3-D observation) and
`RLGuidance3D`. From-scratch 3-D PPO **collapses** exactly as in 2-D (held out 0.00/0.00/0.00, a
constant saturated action); **residual-PN-3D** (the policy corrects a True-PN-3D baseline) resolves
it — held-out **1.00 / 0.91 / 0.98** (crossing / weave / barrel), at parity with the 3-D classical
laws. With this, **every guidance paradigm runs in 3-D**. Figure `gallery/figures/p20_rl_3d.png`,
CSV `results/p20_rl_3d.csv`; experiment `experiments/p20_train_rl_3d.py`.

## Limitations / next

- Perfect-state observation in the RL input (an estimator now closes the loop for the *classical*
  laws in both 2-D and 3-D via `EstimatingGuidance`; coupling it into the RL observation is next).
- Single-agent; MARL swarm/cooperative learning is P7.
- A CleanRL single-file showcase agent (for maximum transparency) is a planned addition alongside
  the SB3 baseline.

## References

- Schulman et al., "Proximal Policy Optimization Algorithms" (2017).
- Raffin et al., "Stable-Baselines3" (JMLR 2021).
- Yan et al., hierarchical PPO guidance+evasion, *Sci. Rep.* 2022 (verified in the research phase).
