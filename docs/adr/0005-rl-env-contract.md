# ADR-0005 — RL environment contract, reward shaping, and fair learned-vs-classical comparison

- **Status:** Accepted
- **Date:** 2026-06-08

## Context

The project's thesis is a rigorous comparison of classical and *learned* guidance. For that
comparison to be valid, a learned policy must (a) train against the same physics the classical laws
and benchmark use, (b) be evaluated on held-out geometries it never trained on, and (c) be reported
on the same metrics — including control effort/efficiency, where the literature suggests learned
methods have their real edge (Research §8), not just intercept rate.

## Decision

1. **One physics core.** `InterceptionEnv` (Gymnasium) is built on the same `PointMass2D`, `RK4`,
   and `ParametricScenario` sampler as the engagement loop and benchmark. There is no separate
   "RL physics". Training and evaluation differ only in the control source.
2. **Shared observation function.** `build_observation(own, target)` (normalized relative position,
   relative velocity, own velocity) is used **both** by the env and by `RLGuidance`, so the policy
   sees identical inputs in training and deployment. Changing it changes both at once.
3. **Action = scaled acceleration.** `Box(-1, 1)²` scaled to the interceptor's `a_max` and passed
   through the same `saturate`. The learned law therefore has *exactly* the control authority the
   classical laws have on a given scenario — no hidden advantage.
4. **Deployment via the `Controller` contract.** `RLGuidance` wraps a trained policy as an ordinary
   guidance `Controller`, so it runs inside the same `Engagement` and `run_benchmark` as PN/OGL/MPC.
   The benchmark's fairness invariant (ADR-0003) then applies unchanged.
5. **Reward shaping (documented, not hidden).** Dense closing reward + LOS-rate shaping (toward
   parallel navigation) − effort/time penalties, with a terminal intercept bonus or miss penalty.
   The shaping encodes *what* to optimize (hit, efficiently), not *how* (no PN formula is injected).
   Weights live in `RewardConfig`.
6. **Curriculum + held-out evaluation.** Training proceeds head-on → crossing → mixed (the env
   samples a scenario per episode in the mixed stage, countering catastrophic forgetting).
   Evaluation uses a **disjoint seed** so reported geometries were never trained on.

## Consequences

- (+) Learned-vs-classical numbers are directly comparable; differences are attributable to the
  policy, not to mismatched physics, authority, or test sets.
- (+) Any future learner (SAC/TD3, CleanRL, MARL) reuses the same env/observation/bridge.
- (−) Reward shaping is a design choice that influences the result; it is version-controlled and
  documented so reviewers can challenge or ablate it.
- (−) Training uses perfect-state observations. **Update:** at *deployment* the sensing loop closes
  for free — `EstimatingGuidance` wraps any `Controller`, so `EstimatingGuidance(..., RLGuidance(...))`
  feeds the trained policy a noisy-radar→EKF *estimate* of the target. Measured robustness: a
  PN-equivalent policy via this path tracks truth-fed performance with negligible degradation
  (≈52.7→52.8 m miss in a sample geometry); tested in `tests/test_estimation.py`. Training *on*
  estimated observations (vs. deploying on them) remains future work.
- (−) Episodes can be long (up to `t_max/dt`), so wall-clock training is non-trivial; mitigated by
  vectorized envs and an early-termination-at-closest-approach reward.
