# ADR-0015 — Adversarial-RL evader

- **Status:** Accepted
- **Date:** 2026-06-09

## Context

The benchmark's adversaries were the open-loop scripted maneuvers (weave/jink/bang-bang) and the
game-theoretic anti-LOS `optimal_evader`. The game-theory note (`docs/algorithms/game-theory.md`)
flagged a **learned (RL) evader** as the natural next adversary — a target that *adapts* to the
pursuer rather than following a fixed rule — but it was deferred. This closes that gap.

## Decision

Add a mirror-image RL setup where the **agent is the target**:

- **`intercept.envs.evader_env.EvaderEnv`** — the agent controls the evader (scalar lateral accel ⟂
  its velocity, scaled to its g-limit) against a fixed **True-PN** pursuer; both share the L2 aero
  plant (fairness invariant), with the interceptor faster so capture is the default and the evader
  must *work*. Observation: normalized relative kinematics from the evader's view. Reward: grow the
  interceptor's predicted zero-effort miss (the opposite of the interceptor's objective) + a survival
  bonus; large penalty if caught, miss-distance bonus on escape. Tunnelling-proof intercept via
  `segment_min_distance`.
- **`intercept.adversary.rl_evader.RLEvader`** — deploys the trained policy as a target `Controller`
  `(t, own, world)`, so it drops into an ordinary `Engagement` against any guidance law (imported by
  path, like `RLGuidance`, to avoid an adversary↔envs import cycle).
- **`experiments/p22_adversarial_evader.py`** — PPO + VecNormalize training, then a held-out grid:
  P(intercept) and miss for the RL evader vs. straight / scripted-weave / `optimal_evader`, against
  True PN / Augmented PN / Sliding-mode interceptors (paired geometries).

## Consequences

- (+) A *learned*, reactive adversary complements the scripted and game-theoretic evaders — a
  self-play-like difficulty source without a second simultaneously-learning agent (the pursuer is
  fixed). Reuses the shared RL machinery (RewardConfig-style weights, VecNormalize, the `Controller`
  contract). Deterministic unit tests cover the obs/reward/env/wrapper and that turning beats flying
  straight; training is a separate experiment.
- **Result (held-out, 150 paired geometries) — P(intercept) by the interceptor:** the RL evader
  **defeats the True PN it trained against (1.00 → 0.07**, median miss ~1.2 km) — a strong learned
  adversary. But it **overfits to that pursuer**: the robust laws still catch it (Augmented PN 0.71,
  sliding-mode 0.78), whereas the analytic anti-LOS `optimal_evader` generalizes (0.00 / 0.08 / 0.31)
  and scripted straight/weave are caught 1.00 by all. Reading the result — a learned adversary exploits
  its *training opponent's* specific weakness, and this independently confirms APN / sliding-mode are
  robust even against learning. Figure `gallery/figures/p22_adversarial_evader.png`.
- (−) The pursuer is a *fixed* True-PN law, not co-adapting; full two-sided self-play (alternating or
  simultaneous learning, a proper differential game) is a further step.
- (−) Like all RL here, the evader is point-mass L2 and trained briefly; it is a benchmark adversary,
  not a claim of optimal evasion (the game-theoretic `optimal_evader` covers the analytic optimum).
