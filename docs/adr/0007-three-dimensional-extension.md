# ADR-0007 — Three-dimensional engagements

- **Status:** Accepted
- **Date:** 2026-06-08

## Context

Real interception is three-dimensional (climb/dive, out-of-plane maneuvers, helical evasion). The
platform was planar (2-D). A 3-D capability is needed for realism and showcase value, ideally
without rewriting the guidance/benchmark machinery.

## Decision

Add 3-D models, geometry, guidance, and visualization **alongside** the 2-D stack, reusing the
dimension-agnostic engagement core. The key enabler: `Engagement` works on whatever
`Dynamics.position()` / `velocity()` return, and the segment-distance intercept test and metrics are
vector-dimension-agnostic — so 3-D needs no engagement changes.

1. **3-D dynamics** (`core/dynamics3d.py`): `PointMass3D` (state `[x,y,z,vx,vy,vz]`) and
   `AeroMissile3D` (`[…,ax,ay,az]`, gravity along −z, parasitic + induced drag, g-limit, autopilot
   lag). Both override `position`/`velocity` to slice ℝ³.
2. **3-D geometry** (`core/frames3d.py`): range, closing speed, and the **LOS angular-velocity
   vector** `Ω = (r×v)/|r|²` (the 2-D scalar λ̇ is its z-component).
3. **3-D guidance** (`guidance/pn3d.py`): realizable true PN `a = N·(Ω × V_c)` (with closing
   velocity `V_c = v_pursuer − v_target`) and Augmented PN with target-acceleration feedforward ⟂
   LOS. Same `Controller` contract.
4. **3-D visualization** (`viz/threed.py`): static mplot3d trajectory plot and a rotating animated
   GIF. **3-D evasion** (`adversary/evasive3d.py`): helical `barrel_roll`, `weave3d`.

## Consequences

- (+) Full 3-D engagements (e.g. APN interceptor vs. a barrel-roll evader under L2 physics)
  intercept correctly and render in 3-D — validated by tests and the `p9_3d_demo` (miss ~16 m).
- (+) Zero changes to `Engagement`, metrics, or the intercept test — the dimension-agnostic core
  (ADR-0002 interface discipline) paid off again.
- (+) 2-D and 3-D coexist; pick by choosing 2-D vs 3-D dynamics + guidance.
- (−) **Sign care:** 3-D PN must use the *closing* velocity (`v_pursuer − v_target`); the naive
  `Ω × v_relative` steers away. Caught by the "intercepts a maneuvering target" test (it failed
  first, then passed after the fix) — a good argument for behavior-level tests, not just unit math.
- (−) The benchmark/scenario suite and the RL env remain 2-D for now; a 3-D scenario schema, 3-D
  Monte-Carlo benchmark, and 3-D RL/MPC are natural follow-ups.
- (−) 3-D sensors/estimation (the EKF/UKF/IMM stack) are still 2-D; extending them is future work.
