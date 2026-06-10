# ADR-0018 — Project-wide realistic-speed standardization + gallery layout

- **Status:** Accepted
- **Date:** 2026-06-09

## Context

ADR-0010 established "no speed cheat" (interceptor ~1.5× the target, not 3×) for the *realistic*
engagement suite. But several earlier/auxiliary experiments still used cartoonish ratios — the
animations especially read as fake: a slow target trivially run down by a much faster interceptor.
Measured offenders: the cooperative salvo (target 100 m/s vs interceptors 600 → **6×**), `p0` (200 vs
950 → **4.75×**), the base benchmark scenarios (up to **4×**), `animate_demos`/`p7`/`p3`/`p5`/`p6`
(~2.5–2.8×), and the `ParametricScenario` default (600/250 → 2.4×). Since the working animations are
the project's highlight, an inconsistent/easy-looking engagement undercuts the whole artifact.

## Decision

1. **One realistic speed regime everywhere:** interceptor ~Mach 3 (≈1000 m/s, 40 g), target ~Mach 2
   (≈700 m/s) — a **~1.45× edge**, matching the realistic suite. Geometries/ranges widened so the
   harder regime still yields meaningful intercepts; pedagogical contrasts preserved (pure pursuit
   still lags a crossing target; only the game-theoretic evader escapes; p2 keeps partial-capture
   separation between PN variants).
2. **From-scratch RL doesn't survive the harder regime** (`p5` collapsed to 0.00), so `p5` now uses
   the project's **residual** parameterization — consistent with ADR-0011 and reported in full (it
   matches the classical laws and edges them on the crossing shot).
3. **Gallery split by kind:** `gallery/figures/` (static graphs/plots) and `gallery/animations/`
   (GIFs + interactive HTML), so the showcase animations are easy to find.

## Consequences

- (+) Every engagement and animation now depicts a realistically *hard-won* intercept — no slow-target
  optics. Consistent speeds across 2-D/3-D/L0–L3, the benchmark, and the demos.
- (+) The benchmark stays meaningful: e.g. S2 crossing separates Pure PN (0.95) from True/Aug/ZEM PN
  (0.77); S5 high-offset is a 4–12 % capture-region corner.
- (−) Old figures/animations were regenerated; pre-0018 numbers in any cached copy no longer match.
- (−) Test fixtures that construct scenarios with explicit legacy speeds are left as-is (internal,
  not part of the public API) — only the *default* and the showcase experiments were standardized.
