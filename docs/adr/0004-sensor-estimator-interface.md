# ADR-0004 — Sensor / estimator interface and estimation-coupled guidance

- **Status:** Accepted
- **Date:** 2026-06-08

## Context

To study guidance under realistic, imperfect information (a verified research gap — guidance
performance *as a function of estimator quality*), P3 introduces sensors and recursive estimators.
The design must (a) not require changing any guidance law, (b) let filters of different state
dimension and type interoperate (for IMM), (c) preserve the benchmark's reproducibility, and
(d) handle angle-valued measurements correctly.

## Decision

1. **Sensors are defined on target position only.** `Sensor.h(sensor_pos, target_pos)` and
   `jacobian(...)` return the measurement and its `∂h/∂target_pos` (shape `dim × 2`). The estimator
   embeds this into the full state Jacobian, so sensors are **decoupled from the tracker's state
   dimension**. Angle components are handled by a sensor-provided `residual` that wraps to `[-π, π]`.
2. **Estimators share a 6-D state** `[x, y, vx, vy, ax, ay]` and expose `predict(dt)` /
   `update(z, sensor, sensor_pos)` plus `target_state() -> [x, y, vx, vy]`. The common dimension lets
   the IMM mix heterogeneous models by plain weighted combination.
3. **Guidance is unchanged.** `EstimatingGuidance` wraps sensor + estimator + law and substitutes the
   estimate into the `world` snapshot the law reads. Truth vs. estimate is invisible to the law —
   the payoff of the `Controller` contract chosen in P0.
4. **Noise RNG is explicit and per-trial.** Sensors draw from a `numpy.random.Generator` fixed at
   engagement construction, so per-step measurement noise stays reproducible (the determinism half
   of [ADR-0003](0003-benchmark-fairness-invariants.md)). This refines, not contradicts, "no RNG
   inside the loop": the loop is deterministic *given* the seeded sensor.
5. **Numerical hygiene.** EKF uses the Joseph-form covariance update; the UKF uses Cholesky sigma
   points with jitter fallback and residual-from-reference angle handling; IMM stabilizes the
   mode-probability update by subtracting the max log-likelihood.

## Consequences

- (+) New sensors/estimators drop in without touching guidance or the engagement loop.
- (+) The estimation-coupled study (guidance vs. seeker noise) and IMM maneuver tracking are now
  first-class, reproducible experiments.
- (+) Angle handling is centralized in the sensor, so every filter inherits correct bearing residuals.
- (−) Fixing the state at 6-D is slightly wasteful for pure-CV tracking and would need revisiting for
  models that require extra states (e.g. coordinated-turn rate ω) — acceptable for now.
- (−) `EstimatingGuidance` assumes perfect interceptor self-state (no own-navigation error); a
  realistic INS error model is future work.
