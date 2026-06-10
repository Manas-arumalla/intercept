# ADR-0006 — Realistic engagement fidelity (L2): aero dynamics + aggressive/reactive targets

- **Status:** Accepted
- **Date:** 2026-06-08

## Context

At L0 (`PointMass2D`) the interceptor caught targets largely because it was faster, and targets flew
simple, slow, near-straight paths. That is neither realistic nor technically challenging, and it
made the guidance comparison uninformative (everything hits). Real engagements involve fast,
highly-maneuverable, *deceptive* targets, finite control authority, response lag, and energy limits;
the interceptor must win through prediction, estimation, and control — not speed.

## Decision

Add a higher-fidelity level **L2** behind the existing `Dynamics` interface (ADR-0002), plus
aggressive/reactive adversaries and realistic scenario presets. **No guidance/estimation/RL/benchmark
code changes** — the new model keeps the `[x, y, vx, vy, …]` layout, so everything reads `state[:4]`.

1. **`AeroMissile2D` (L2 plant)** — planar 3-DOF (Zarchan-style), state `[x,y,vx,vy,ax,ay]`:
   - **gravity** (arcing trajectories),
   - **parasitic drag** ∝ V² (speed bleeds; no free cruise),
   - **induced drag** ∝ a_lat² (*pulling g costs energy* — a hard-jinking target slows down),
   - **hard lateral g-limit**, and
   - **first-order autopilot lag** τ (commanded ≠ achieved acceleration; forces lead, defeats naive
     terminal correction). Only the lateral (perpendicular-to-velocity) command is achievable as lift.
2. **Aggressive / reactive maneuvers** (`adversary/evasive.py`): high-g weave, **random-telegraph
   jink** (seeded, unpredictable), sustained hard turn, and a **closed-loop reactive break** that
   senses the interceptor and pulls max-g away inside a trigger range.
3. **Realistic scenario presets** (`scenarios/realistic/`, `model: aero`): comparable supersonic
   speeds (interceptor ~Mach 3.5 vs target ~Mach 2.2–2.4, only ~1.5× — no speed-win), 5–30 g targets,
   autopilot lag, gravity. Accelerations expressible in g.

## Consequences

- (+) **The comparison becomes informative.** Re-benchmark (150 trials/cell): on the unpredictable
  jink True PN falls to **0.56** and on the reactive break to **0.21**, while Augmented PN, Optimal,
  and Sliding-mode recover to **0.79–1.00** (sliding-mode, built for unknown maneuvers, stays at
  1.00). Intelligence/robustness — not speed — now decides the outcome, as intended.
- (+) Physically grounded energy trade-off: a target that jinks hard bleeds speed (induced drag),
  exactly the real dilemma an evader faces.
- (+) Same interfaces ⇒ every guidance law, the estimator stack, and the benchmark work unchanged;
  scenarios choose fidelity via `model: point_mass | aero`.
- (−) Parameters (drag coefficients, τ, g-limits) are representative/textbook-grounded (Zarchan),
  not tuned to a specific vehicle; documented and adjustable per scenario.
- (−) The L0-trained RL policy faces a different plant at L2 (expected degradation); retraining on
  L2 is a follow-up. Classical/optimal/robust laws are model-light and transfer directly.
- (−) Still planar; the **3-D extension** is the next milestone (real engagements are 3-D).
