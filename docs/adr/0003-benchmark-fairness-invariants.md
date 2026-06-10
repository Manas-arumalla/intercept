# ADR-0003 — Benchmark fairness invariants

- **Status:** Accepted
- **Date:** 2026-06-08

## Context

The project's core contribution is a *rigorous cross-paradigm comparison* of guidance laws
(Research Report §E.1). A comparison is only credible if differences in outcome are attributable
to the algorithm under test and nothing else. Without explicit invariants it is easy to compare
algorithms on subtly different engagements (different sampled geometries, different target
behavior, different RNG draws) and draw invalid conclusions — a failure mode the literature
frequently exhibits.

## Decision

The benchmark enforces the following **fairness invariants**, structurally in code:

1. **Shared dynamics.** All entities in an engagement use the *same* injected `Dynamics`; the
   interceptor's plant (`PointMass2D`, `a_max`) is a property of the scenario, not the guidance
   law. Every law sees identical control authority and integration.
2. **Identical sampled engagements per seed.** Monte-Carlo trial `i` derives its RNG from
   `numpy.random.SeedSequence(seed).spawn(n_trials)[i]`. The scenario samples the engagement
   (initial range, cross-range, target heading, maneuver) from that RNG *before* any guidance runs.
   Therefore trial `i` produces the **same** initial conditions and the **same** target trajectory
   for every algorithm. Algorithms are compared on a matched set of engagements, paired by trial.
3. **No RNG inside the loop.** Stochasticity is sampled only when constructing the engagement, never
   inside `Engagement.run()`. Given a spec, a run is deterministic. (When sensors/noise arrive in
   P3, they will draw from an explicit per-trial RNG fixed at sample time, preserving this.)
4. **Fresh stateful controllers per trial.** Guidance is injected via a *factory*
   `factory(target_name) -> Controller`, called once per trial, so stateful laws (e.g. APN's
   finite-difference memory) never leak state between engagements.
5. **Documented tuning.** Classical laws are given documented, best-effort gains
   (e.g. `N`); RL agents (P5) will be evaluated on **held-out** seeds/scenarios disjoint from
   training. Any cap or truncation (trial count, grid resolution) is logged, never silent.

These are validated by tests (`test_benchmark.py`): scenario sampling is seed-reproducible, the
Monte-Carlo is deterministic, and the same seed yields identical target trajectories across
different guidance laws.

## Consequences

- (+) Differences in P_intercept / miss / effort are causally attributable to the guidance law.
- (+) Results are exactly reproducible from `(scenario, seed, n_trials)`.
- (+) Paired trials enable lower-variance comparisons (same geometry, different law).
- (−) Adding any new stochastic component (sensor noise, wind) must route through the sample-time
  RNG, not the loop — a constraint contributors must respect (documented in `CLAUDE.md`).
- (−) "Best-effort gain tuning" is a judgement call; gains are recorded in configs/experiments so
  reviewers can challenge them.
