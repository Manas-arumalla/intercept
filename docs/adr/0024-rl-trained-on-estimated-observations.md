# ADR-0024 — Training RL directly on estimated (noisy-sensor) observations

- **Status:** Accepted
- **Date:** 2026-06-09

## Context

ADR-0005 trained the RL policy on **perfect state** and only *deployed* it on estimates (via
`EstimatingGuidance` wrapping `RLGuidance`, which is robust at moderate noise). The open gap was
training the policy **directly on estimated observations**, so it can adapt to sensing noise rather
than merely tolerate it.

## Decision

- `InterceptionEnv(sensor=..., estimator_factory=...)`: each step the env takes a noisy
  seeker-on-interceptor radar measurement, runs an EKF, and builds the **observation (and the
  residual-PN baseline) from the filtered estimate** — while **reward and intercept detection use
  truth** (the policy sees noise; the world is scored against ground truth). `None` ⇒ perfect-state observations.
- `experiments/p31_rl_estimated_obs.py`: train a policy under a noisy seeker (σ_range = 60 m), then
  sweep deploy-time seeker noise and compare — head-to-head via `EstimatingGuidance` — the
  **truth-trained** policy (P5) vs. this **estimate-trained** policy, plus a PN-on-estimate baseline.

## Result (weaving target, P(intercept) vs seeker range-noise σ; 120 trials)

| policy \ σ (m) | 1 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| PN (on estimate) | 1.00 | 1.00 | 1.00 | 1.00 | **0.93** |
| RL **truth-trained twin** | 1.00 | 1.00 | 0.97 | 0.90 | 0.69 |
| RL **estimate-trained** | 1.00 | 0.99 | **0.99** | **0.98** | **0.88** |

The ablation is clean — same scenario, budget, and architecture; only the training observation source
differs — and decisive: **training on estimates preserves robustness that truth-training loses**
(σ=200: 0.88 vs 0.69, a 19-point gap; σ=100: 0.98 vs 0.90). Two caveats: (1) a first eval on a
*straight* target showed no separation at all (a good EKF erases the noise; the effect needs a
maneuvering target where the filter lags); (2) plain PN-on-estimate remains the most noise-robust
policy here (0.93 at σ=200) — the ablation isolates the *training-observation* effect, it does not
crown RL. Figure `gallery/figures/p31_rl_estimated_obs.png`.

## Consequences

- (+) Closes the ADR-0005 gap — RL trained end-to-end on the sense→estimate→guide loop, compared
  fairly against truth-trained deployment under noise. Figure `gallery/figures/p31_rl_estimated_obs.png`.
- (−) Single EKF + radar; richer sensing (IMM-in-loop observations, angles-only/IR, clutter, false
  alarms) and recurrent policies over the estimate are further work.
