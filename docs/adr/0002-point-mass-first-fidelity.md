# ADR-0002 — Progressive fidelity: point-mass first, pluggable dynamics

- **Status:** Accepted
- **Date:** 2026-06-07

## Context

Engagement realism spans a wide ladder: 2-D point-mass → 3-D kinematic → autopilot lag + actuator
saturation → full 6-DOF rigid-body aero. Higher fidelity is more realistic but slows algorithm
development and Monte-Carlo throughput, and most published guidance/RL/game-theory results I
build on are themselves idealized point-mass studies (see Research Report §E.5). The benchmark's
*comparative* validity depends on every algorithm seeing the **same** plant, whatever its fidelity.

## Decision

Adopt a **progressive fidelity ladder behind a single `Dynamics` interface:**

1. **L0 — 2-D point-mass, ideal** (`PointMass2D`): direct acceleration command, optional
   saturation/drag. *(Implemented in P0.)*
2. **L1 — 2-D + autopilot lag + acceleration saturation + sensor noise.**
3. **L2 — 3-D point-mass / kinematic.**
4. **L3 (optional, P9) — 3-D with simplified aero / autopilot loops; 6-DOF via JSBSim or a
   MATLAB/Simulink cross-check.**

Dynamics are exposed as pure `derivative(t, state, control)` functions so integrators, rollouts,
optimization/autodiff backends, and RL envs all reuse the same model. Climbing the ladder must not
require changing guidance/estimation/benchmark code — only swapping the injected `Dynamics`.

## Consequences

- (+) Fast algorithm development and high Monte-Carlo throughput at L0/L1; matches the fidelity of
  most reference results, enabling fair reproduction.
- (+) A built-in robustness study: re-run the *same* benchmark at higher fidelity and report which
  conclusions survive autopilot lag / saturation / noise (a verified gap and differentiator).
- (+) Fairness invariant is structural: all paradigms share the injected plant.
- (−) Point-mass omits real aerodynamics/attitude; results are comparative and educational, not
  operational predictions — stated explicitly in the README/scope note.
- (−) Some laws (e.g., 6-DOF autopilot interactions) cannot be studied until L3; accepted as out of
  scope for the core thesis.
