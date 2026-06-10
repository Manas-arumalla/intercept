# ADR-0014 — Three-dimensional reinforcement-learning guidance

- **Status:** Accepted
- **Date:** 2026-06-08

## Context

After the optimal, sliding-mode, and MPC laws were lifted to 3-D, **RL was the last guidance
paradigm still 2-D only**. The 2-D `InterceptionEnv` is tightly coupled to planar kinematics (1-D
scalar lateral action, 2-D observation, 2-D gravity feed-forward), so it could not host a 3-D agent.

## Decision

Add a dedicated `intercept.envs.interception_env_3d` (rather than refactor the working 2-D env,
preserving the fairness invariant and avoiding regression risk):

- **Env** `InterceptionEnv3D` over `ParametricScenario3D` — same 3-D dynamics / RK4 / sampler the
  3-D classical laws and benchmark use, with the tunnelling-proof `segment_min_distance` intercept
  test (dimension-agnostic).
- **Action** — a **2-DOF** lateral acceleration in the plane ⟂ the interceptor's velocity, spanned
  by an orthonormal basis `(e1, e2)` and scaled to `a_max` (norm-clipped to the achievable disc).
  This is the realizable lateral control in 3-D (the velocity-perpendicular plane is 2-D) and holds
  speed, far more learnable than an unconstrained 3-D acceleration. Two action modes:
  `"absolute"` (raw) and **`"residual_pn"`** — the policy outputs a bounded correction on a
  True-PN-3D baseline (`pn_baseline_action_3d`, projected into the same `(e1, e2)` basis), so a zero
  action is already competent PN-3D.
- **Observation** — `build_observation_3d` (rel pos/vel + own vel, 9-D) and `_rich` (adds the
  LOS-rate vector, closing speed, range; 14-D).
- **Reward** — the same potential-based ZEM shaping as 2-D, in 3-D; gravity feed-forward cancels the
  ⟂-velocity component of gravity (along −z).
- **Deploy** — `RLGuidance3D` mirrors the training parameterization (obs, 2-DOF action → lateral
  accel, gravity FF) and supports recurrent (LSTM) policies, so the learned 3-D law runs inside an
  ordinary `Engagement` and is benchmarked against True/Augmented PN-3D, Optimal-3D, Sliding-mode-3D.
- **Experiment** `experiments/p20_train_rl_3d.py` — PPO + VecNormalize over a 3-D aero curriculum
  (crossing → 3-D weave → barrel-roll) and a held-out 3-D learned-vs-classical comparison.

## Finding: from-scratch collapses, residual-PN-3D fixes it

A first PPO trained in **absolute** mode reproduced the 2-D failure exactly: held out it **collapsed
to a constant saturated action (`[1, −1]`) ignoring the observation**, intercepting **0/100** on
every 3-D scenario (≈700 m miss) — even though the training return looked healthy (it was ZEM
potential-shaping accumulation, not real intercepts; verified by running the deterministic policy
back in the env: 0/30). The fix is the same residual policy learning that worked in 2-D (ADR-0011):
train in **`residual_pn`** mode (default for `p20`), so the learned part is only a bounded
*correction* on PN-3D and a zero action is already competent — no collapse.

**Result (held-out, 100 trials/scenario):** from-scratch **0.00/0.00/0.00** → residual-PN-3D MLP
**1.00 / 0.91 / 0.98** (crossing / weave / barrel). A **recurrent (LSTM) APN-residual** (P23 —
`baseline="apn"`, `RecurrentPPO`; `apn_baseline_action_3d` feeds the target's measured lateral accel)
then closes the gap to **1.00 / 1.00 / 1.00** at competitive effort — full parity with the 3-D
classical laws, mirroring the 2-D recurrent win (ablation: APN baseline + memory lift the weave
0.91→1.00, barrel 0.98→1.00). Figures `gallery/figures/p20_rl_3d.png`, `gallery/figures/p23_recurrent_residual_3d.png`.

## Consequences

- (+) **Every guidance paradigm now runs in 3-D** (PN/APN, Optimal, Sliding-mode, MPC, RL) on the
  same fair testbed. The 3-D residual RL agent is evaluated held-out against the 3-D classical laws
  (results in the progress log / results digest).
- (+) Reuses the shared machinery (RewardConfig, VecNormalize, the `Controller` contract) and the
  residual idea from ADR-0011; the 2-D env and all 2-D RL work are untouched. Deterministic unit
  tests cover the action/obs/gravity helpers, the PN-3D baseline (a zero residual reproduces PN-3D
  and intercepts where a constant-saturated action misses), an episode rollout, and the wrapper.
- (−) A separate 3-D env duplicates structure from the 2-D one; a future refactor could share a
  dimension-generic base. A recurrent 3-D residual policy (`RLGuidance3D` already threads LSTM
  state) is a natural further step.
- (−) As in 2-D, the learned part is a *correction*: its value shows on the harder maneuvers; on
  easy geometries PN-3D already suffices.
