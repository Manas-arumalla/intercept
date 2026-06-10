# ADR-0010 — Realistic speed parity for showcase engagements (no speed cheat)

- **Status:** Accepted
- **Date:** 2026-06-08

## Context

The project's design principle is that the interceptor must win by **intelligent guidance**
(prediction, lead, robust control), **not** by a raw speed/energy advantage. A first cut of the
advanced complex-trajectory demo (`p14_advanced_evasion`) violated this: with the default
`RealisticMissile3D` propulsion the interceptor peaked at **Mach 4.3** and was still **Mach 4.2 at
the merge**, while the maneuvering threat bled down to **Mach 1.3** — a **+226 %** closing-speed
edge. The interceptor was effectively *out-running* the target, not intercepting it, which defeats
the purpose of the comparison.

Two coupled causes:

1. The interceptor's booster was oversized relative to the engagement, and its sustainer kept it
   fast all the way to the merge.
2. The threat preset had a **weak sustainer** (2.6 kN) that could not overcome drag, so it bled
   energy continuously — worsened by the induced drag of hard maneuvering.

## Decision

Tune the L3 propulsion of **both** bodies so speeds are realistic and **comparable**, and bake the
parameters into `p14`'s `NOMINAL` (the robustness sweep perturbs around them):

- **Interceptor:** launch **slow (~Mach 1.2)** — the booster does the work, as real interceptors do
  — boost to **~Mach 3**, then **coast** (reduced sustainer) so it is **~Mach 2.6 at the merge**.
- **Threat:** a genuinely fast **~Mach 3** missile with a **stronger sustainer** (8 kN) and a sleeker
  reference area, so it *holds* energy and only bleeds to **~Mach 2** as a *consequence* of pulling
  hard g (induced drag) — physically correct, not imposed.
- Net **closing-speed edge at the merge ≈ +37 %** — typical of a real surface-to-air engagement
  against a supersonic threat.

The decisive no-cheat evidence: a propulsion sweep shows that **shrinking the interceptor's motor
any further makes it miss** (edge goes to −82 % and miss distance to hundreds of metres). The
showcase interceptor therefore flies on **near-minimum energy** — it has just enough to make the
intercept via an efficient Augmented-PN lead, and no speed margin to spare.

How the maneuvering is *shown* faithfully: at supersonic speed a real "spiral" is a long, **thin**
helix (turn radius ≪ distance per revolution), so an equal-aspect 3-D view at engagement scale looks
near-straight — faking a tight visible corkscrew would require hundreds of g and would be a cheat.
The complexity is instead made visible the way the literature quantifies it: cross-range /
altitude **projections** and an **achieved-g vs. physics-available-turn-limit** time history.

## Consequences

- (+) The showcase honors the project principle: comparable speeds, intercept by guidance not speed,
  and a target that is genuinely fast and maneuvering (≤ ~17 g, clipped by its physics turn limit).
- (+) Robustness Monte-Carlo (60 randomized trials over geometry + every maneuver parameter) →
  **P(intercept) = 1.00** (95 % CI [0.94, 1.00]), median miss 13.7 m — not a single tuned shot.
- (+) A regression **test asserts the no-cheat property** (`test_advanced_engagement_intercepts_
  without_speed_cheat`: merge edge < 60 %, threat top speed > 0.8 × interceptor's).
- (−) Parameters are tuned for *this* geometry; a different engagement window would need re-tuning to
  keep the interceptor near minimum energy. The robustness sweep bounds, but does not eliminate, that.
- (−) The realistic thin-helix truth means the 3-D oblique view is visually subdued; the analysis
  panel (not the 3-D render) is the primary evidence of trajectory complexity.
