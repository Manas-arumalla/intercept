# ADR-0012 — Three-dimensional Monte-Carlo benchmark

- **Status:** Accepted
- **Date:** 2026-06-08

## Context

The project has full 3-D dynamics (`PointMass3D`, `AeroMissile3D`, `RealisticMissile3D`), 3-D
guidance (`true_pn_3d`, `augmented_pn_3d`), and 3-D evasion (barrel-roll, weave, serpentine,
intensifying terminal spiral), but the centerpiece **Monte-Carlo benchmark was 2-D only**
(`ParametricScenario` samples a planar geometry). The cross-paradigm fairness comparison — the
project's thesis — therefore did not cover the 3-D engagements the rest of the codebase models.

## Decision

Add `intercept.benchmark.scenario3d.ParametricScenario3D` — a 3-D geometry/maneuver sampler that
emits the **same** `EngagementSpec` the 2-D scenario does. Because the Monte-Carlo runner
(`run_montecarlo`/`run_benchmark`), the Wilson-CI metrics, and the control-effort metric all operate
on a dimension-agnostic `EngagementResult`, **nothing in the runner or metrics changed** — only the
geometry sampler and the plant/maneuver builders are new:

- Geometry: interceptor at the origin; target at `(downrange, cross-range, altitude)` sampled from
  ranges, flying at `target_speed` along an inertial heading set by azimuth + elevation; the
  interceptor leads straight at the target (so it climbs to altitude). Gravity acts along −z for the
  aero/realistic plants.
- Fidelity via `model`: `point_mass` (L0), `aero` (L2), `realistic` (L3) — mirroring the 2-D scenario.
- `make_maneuver_3d`: `weave`, `barrel_roll`, `serpentine`, and the closed-loop `spiral`
  (intensifying terminal corkscrew), accel as `g` or `accel`.
- `experiments/p17_benchmark_3d.py`: a graded suite (crossing → barrel-roll → serpentine → spiral)
  with modest interceptor authority so the hard maneuvers separate the laws.

## Consequences

- (+) The benchmark now spans **2-D and 3-D** on one fair testbed, and the 3-D result mirrors the
  2-D realism finding: on a sustained **barrel-roll**, True PN-3D drops to **0.00** while Augmented
  PN-3D (target-acceleration feed-forward) holds **1.00**; on the **intensifying terminal spiral**,
  True PN **0.18** vs APN **0.82** — at ~1.8× control effort. Figure `gallery/figures/p17_benchmark_3d.png`,
  CSV `results/p17_benchmark_3d.csv`. Tests in `tests/test_3d.py`.
- (+) `EngagementSpec` was already plant-carrying and dimension-agnostic, so the extension is purely
  additive (no risk to the 2-D path).
- **Update:** 3-D **Optimal (OGL-3D)** and **Sliding-mode (SMG-3D)** guidance were subsequently added
  (`optimal_guidance_3d`, `sliding_mode_3d`), so the 3-D benchmark and capstone now compare four laws,
  not just the PN family — on the barrel-roll that defeats True PN-3D (0.00), APN-3D / OGL-3D / SMG-3D
  all recover to 1.00. (MPC and RL guidance remain 2-D — a further follow-up.)
- (−) `ParametricScenario3D` duplicates some geometry/field structure from the 2-D scenario rather
  than sharing a common base; acceptable for clarity, a future refactor could unify them.
