# ADR-0023 — Population self-play (fictitious play against a pool)

- **Status:** Accepted
- **Date:** 2026-06-09

## Context

ADR-0020 found single-step alternating self-play **non-transitive**: interceptor gen-2, trained only
against the *latest* evader (gen-1), mastered it (1.00) but **catastrophically forgot** evader gen-0
(0.80 → 0.18). The standard remedy is **fictitious / population self-play** — best-respond to a
*distribution* of past opponents, not the most recent one.

## Decision

- Add an `opponent_factory` hook to `InterceptionEnv` (a 0-arg callable invoked **each reset**), so
  the opponent can be **sampled per episode from a pool**.
- `experiments/p30_population_selfplay.py`: train a fresh interceptor against the pool
  {evader gen-0, evader gen-1} (uniform per-episode sampling), then re-evaluate the full cross-table
  and report the **worst-case (min over evader generations)** — the robustness number a single
  generation sacrifices.

## Result (150 trials, P(intercept) by the interceptor)

| interceptor \ evader | Eva gen-0 | Eva gen-1 | min (robustness) |
|---|---|---|---|
| Int gen-1 (vs gen-0) | 0.80 | 1.00 | 0.80 |
| Int gen-2 (vs gen-1) | **0.18** | 1.00 | 0.18 ← forgot gen-0 |
| **Int POOL (fictitious)** | **0.78** | 0.59 | 0.59 |

**Fictitious play eliminates the catastrophic forgetting:** the pool-trained interceptor handles the
hard gen-0 at **0.78** — vs gen-2's **0.18** (the latest-opponent policy that forgot it), and on par
with gen-1's 0.80. The learned residual costs some performance on the *easy* gen-1 (0.59), so the
worst-case min (0.59) lands between gen-2 (0.18) and the single-round gen-1 (0.80).

A necessary engineering finding along the way: **per-episode opponent switching makes episode returns
bimodal** (easy vs. hard opponent), which **destabilizes PPO reward normalization** — the first runs
collapsed (reward → −250, min 0.16). Disabling reward normalization (`norm_reward=False`, normalize
observations only) fixed it; uniform pooling then balances both opponents. Figure
`gallery/figures/p30_population_selfplay.png`.

## Consequences

- (+) Confirms the textbook fix for the ADR-0020 instability — training against a *pool* recovers the
  forgotten opponent (0.18 → 0.78). The `opponent_factory` hook is reusable for any pooled/curriculum
  opponent, and the bimodal-return / reward-normalization lesson is documented.
- (−) Worst-case min (0.59) still trails the single-round gen-1 (0.80): the residual mildly
  miscalibrates on the easy opponent. A full **PSRO** (meta-game payoff matrix, best-response to the
  Nash mixture, a growing population, per-opponent value heads) is the heavier next step.
