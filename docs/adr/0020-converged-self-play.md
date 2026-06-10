# ADR-0020 — Converged self-play (alternating arms race + cross-table)

- **Status:** Accepted
- **Date:** 2026-06-09

## Context

ADR-0017 did one self-play round (interceptor gen-1 vs the frozen gen-0 evader). The open item was
*converged* self-play — both sides co-adapting over multiple generations. This ADR adds the
infrastructure and a multi-generation arms race, reported as a **cross-table** (every interceptor
generation vs every evader generation), because two-sided learning is non-transitive and a single
scalar would hide the dynamics.

## Decision

- Add a `pursuer_factory` hook to `EvaderEnv` (mirroring `InterceptionEnv(opponent=...)`), so the
  evader can be trained against a *learned* `RLGuidance` interceptor — the missing half of self-play.
- `experiments/p27_converged_selfplay.py`: generations evader g0 (=P22), interceptor g1 (=P25),
  then **evader g1** (trained vs interceptor g1) and **interceptor g2** (trained vs evader g1).
  Evaluate the full P(intercept) cross-table held-out.

## Result (150 trials, P(intercept) by the interceptor)

| interceptor \ evader | Eva gen-0 | Eva gen-1 |
|---|---|---|
| True PN | 0.07 | 0.91 |
| Augmented PN | 0.71 | 1.00 |
| Interceptor gen-1 | **0.80** | 1.00 |
| Interceptor gen-2 | **0.18** | 1.00 |

**The arms race does not monotonically converge in single-step alternation.** Interceptor gen-2
specialized against evader gen-1 (1.00) but **catastrophically forgot** evader gen-0 (0.80 → 0.18);
and evader gen-1, by overfitting to beat interceptor gen-1, became *more exploitable* by everyone
(True PN catches it 0.91 vs 0.07 for gen-0). Figure `gallery/figures/p27_converged_selfplay.png`.

## Consequences

- (+) A textbook demonstration of self-play non-transitivity / forgetting — the cross-table
  makes it legible, and the reusable `pursuer_factory`/`opponent` hooks form a real self-play harness.
- (−) **Stable convergence needs a population**, not the latest-opponent: fictitious self-play / PSRO
  (train against a *pool* of past generations, track an empirical-best-response cross-table) is the
  proper next step — deliberately scoped out here as a larger effort.
- (−) Each generation is a separate PPO run (compute-heavy); only two new generations were trained.
