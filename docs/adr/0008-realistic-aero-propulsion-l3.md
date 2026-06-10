# ADR-0008 — L3 realism: aero-propulsive physics with no prescribed limits

- **Status:** Accepted
- **Date:** 2026-06-08

## Context

L2 (`AeroMissile2D/3D`) added drag/lag/g-limit but the **g-limit and drag coefficients were
hand-set constants** and **speed was effectively free** (no propulsion/mass). That is the residual
"cheat": an interceptor with a fixed 40 g and constant speed regardless of altitude or energy state.
Real missile capability is dictated by propulsion (a finite-impulse rocket) and aerodynamics
(lift/drag scaling with dynamic pressure and Mach). The design goal was to push realism as far as a point-mass abstraction allows.

## Decision

Add fidelity **L3** (`core/realistic.py` `RealisticMissile2D/3D`, `core/atmosphere.py`) where the
performance-defining quantities **emerge from physics** rather than being prescribed:

1. **ISA standard atmosphere** — density, speed of sound, pressure vs. altitude (troposphere +
   lower stratosphere).
2. **Propulsion** — rocket **boost → sustain → coast** thrust schedule with **propellant mass
   burn-off**; speed is gained on boost and lost to drag on coast; ``T/m`` rises as it lightens.
3. **Mach-dependent zero-lift drag** ``Cd0(M)`` (transonic rise) + **lift-induced drag** ``∝ C_L²``.
   Drag ``D = ½ρ(h)V²·S·Cd``; turning costs energy exactly (induced drag), not by a fudge factor.
4. **Lift / dynamic-pressure-limited turning** — achievable lateral accel
   ``= min(structural, q·S·C_Lmax/m)``. **Turn capability falls at low speed / high altitude.** No
   fixed g.
5. First-order autopilot lag, gravity; only the lateral command is achievable as lift.

Same `[pos, vel, …]` interface ⇒ guidance, estimation, the engagement loop, and metrics are
unchanged. The same model serves 2-D (vertical plane) and 3-D (z = altitude).

## Consequences

- (+) **Physics-driven, not prescribed:** a validated engagement shows the interceptor boosting **150 → 843 m/s (Mach
  2.6)** with available lateral g rising **2 g → 45 g** purely from dynamic pressure — it *cannot*
  maneuver hard when slow/just-launched, exactly as real interceptors can't. Intercepts a weaving
  high-altitude target at 17 m.
- (+) Speeds are comparable (interceptor not 3× the target); success comes from boost timing, lead,
  and energy management — not a speed advantage.
- (+) **Robust under uncertainty:** with a noisy radar + IMM/EKF estimator in the loop, P(intercept)
  on a fast weaving L3 target holds at **0.97** vs 1.00 perfect-info (median miss 16 vs 14 m).
- (+) Realistic sizing is just parameters; presets (`.target()`) and overrides keep it tunable.
- (−) Still a **point-mass** (no 6-DOF attitude/airframe, fin actuators, or thrust-vector control);
  drag/CLmax curves are representative, not a specific airframe or wind-tunnel data.
- (−) L3 is not yet wired into `ParametricScenario`/the Monte-Carlo benchmark or the RL env (used via
  dedicated demos/tests); those integrations are follow-ups.
- (−) Parameters are not validated against flight-test data — this remains a research/engineering
  simulation, not an operational predictor (consistent with the project scope note).
