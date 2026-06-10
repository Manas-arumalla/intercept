# ADR-0022 — MARL cooperative swarm (learned target allocation)

- **Status:** Accepted
- **Date:** 2026-06-09

## Context

Swarm defense so far coordinated interceptors with the **analytic Hungarian** weapon-target
assignment (ADR-0009/0021). The open item was *learned* cooperation: can a policy discover the
allocation itself? Full decentralized MARL (per-agent policies, PettingZoo/MAPPO) is a large
framework lift; this ADR scopes a tractable, clearly-bounded version that reuses the single-agent
PPO stack.

## Decision

- `intercept.envs.CentralizedSwarmEnv`: one policy observes the whole engagement (every interceptor
  relative to every threat + alive flags) and emits an **N×M preference matrix** each decision step;
  each living interceptor takes its highest-preference living threat (a *learned* WTA) and homes with
  True PN. Team reward = +per intercept, −per leaker, with dense closing shaping. The learned part is
  **purely the cooperative allocation** — PN does the guidance, so there is no from-scratch-control
  collapse to mask the comparison.
- Deliberately **under-resourced — 3 interceptors vs 5 threats** — so allocation matters (some
  threats leak; good coordination minimizes them).
- `experiments/p29_marl_swarm.py` compares, on **identical seeded scenarios** (fairness invariant),
  the learned allocator vs. the **Hungarian WTA** (run in the *same* env via a one-hot action) and a
  **random** allocator, scoring mean leakers.

## Result (200 held-out trials, mean leakers of 5)

| allocator | mean leakers ↓ |
|---|---|
| Random | 0.98 |
| **Learned (MARL)** | **0.69** |
| Hungarian WTA | 0.65 |

The learned cooperative allocator **nearly matches the near-optimal Hungarian baseline (0.69 vs
0.65)** and is far better than random (0.98) — it discovered effective spread-out coordination from
the team reward alone. It does **not beat** the analytic optimum (Hungarian is near-optimal
for this assignment), so the value here is demonstrating *learned* coordination that approaches it,
not a new SOTA. Figure `gallery/figures/p29_marl_swarm.png`.

## Consequences

- (+) A learned cooperative-allocation policy on the existing PPO stack, compared apples-to-apples
  with the optimization baseline on shared dynamics. Figure `gallery/figures/p29_marl_swarm.png`,
  tests in `tests/test_swarm_env.py`.
- (−) **Centralized** training/execution with a fixed N, M and a flat N×M action — not decentralized
  per-agent MARL; true MAPPO/IPPO with PettingZoo and variable team sizes is the further step.
- (−) Coordination is learned over PN guidance (allocation only), not end-to-end joint control.
