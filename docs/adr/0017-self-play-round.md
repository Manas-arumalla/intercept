# ADR-0017 — Two-sided self-play (one arms-race round)

- **Status:** Accepted
- **Date:** 2026-06-09

## Context

ADR-0015 added a learned **evader** that defeats True PN (it escapes ~93 % of the time) but overfits
to that pursuer. The natural next step flagged there was **two-sided self-play** — both sides
learning. Full co-adapting self-play (alternating training to convergence) is open-ended and can be
unstable/cyclic; this ADR scopes **one clean round of the arms race** as a concrete, interpretable
result rather than a full tournament.

## Decision

- Add an `opponent` override to `InterceptionEnv`: an arbitrary target `Controller` that replaces the
  scenario's scripted maneuver. Passing `RLEvader(<frozen gen-0 evader>)` makes the interceptor train
  **against the learned evader** — the self-play hook (the env step already exposes the interceptor
  in the world snapshot the evader reads).
- `experiments/p25_selfplay.py`: train a fresh **residual-PN interceptor (gen-1)** with the gen-0
  evader as the opponent, then evaluate it held-out against the *same frozen evader* alongside True PN
  and Augmented PN. The question: does training against the learned adversary harden the interceptor?

## Result (held-out, 150 trials, all vs. the frozen gen-0 evader)

P(intercept) of the evader: True PN **0.07**, Augmented PN **0.71**, **gen-1 self-play interceptor
0.80**. Training against the learned adversary **hardens** the interceptor past both classical laws.

A necessary detail: the residual must sit on a **competent** baseline. A first run with the
**PN** baseline (which itself fails at 0.07 against this evader) could not bootstrap — gen-1 stayed at
**0.05** with negative training reward. Switching to the **APN** baseline (already 0.71 vs the evader)
let the residual climb to 0.80 — the same "residual needs a competent base" lesson as ADR-0011.

## Consequences

- (+) A concrete arms-race round on the existing infrastructure (the `opponent` hook + `RLEvader` +
  residual RL), no new training framework — and it demonstrably **hardens the interceptor** against a
  learned adversary that defeats PN. The `opponent` override is general (any controller), reusable for
  future rounds. Figure `gallery/figures/p25_selfplay.png`.
- (−) The gen-0 evader is *frozen and deterministic*, so gen-1 can partly overfit to its exact policy
  (the cost of a single round vs. a randomized opponent pool). It is one round, not converged
  self-play — alternating to a stable equilibrium (or a population/PSRO scheme) remains future work.
- (−) Two-sided learning is inherently non-transitive; a single round shows adaptation, not a global
  "winner." Reported as such.
