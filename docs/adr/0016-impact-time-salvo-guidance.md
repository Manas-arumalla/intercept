# ADR-0016 — Impact-time-control (cooperative salvo) guidance

- **Status:** Accepted
- **Date:** 2026-06-09

## Context

The multi-agent layer (ADR-0009) assigns interceptors to threats but every interceptor flew its own
fastest intercept, so a battery launched from different ranges arrives **spread out in time**. A
*salvo* — all interceptors arriving simultaneously to saturate the defense — needs **impact-time
control**, the main open cooperative-guidance item. Two earlier feedback-ITCG prototypes failed (a
`sign(λ̇)` bias that chattered, then a lead-angle `t_go` estimate that overshot and never
synchronized) and were removed rather than shipped — so the bar was a *validated* synchronizer.

## Decision

Add `intercept.guidance.impact_time.ImpactTimeGuidance` — biased PN steering to a commanded absolute
impact time (after Jeon, Lee & Tahk 2006), in the corrected form that actually converges:

    a = N·Vc·λ̇·⟂̂_LOS  +  k·Vc·max(0, (t_impact − t) − R/Vc)·⟂̂_v,away

Two fixes over the failed attempts: (1) a **simple `t_go = R/Vc`** (the lead-angle correction caused
overshoot), and (2) a **consistent** bias along the velocity-perpendicular pointing *away from the
LOS* (raising the lead angle to lengthen the path), with a **small gain `k ≈ 0.2`**. The bias
vanishes as the time-error → 0, so PN homes cleanly at the end. The interceptor cannot speed up, so
`t_impact` must be ≥ the slowest member's natural time-to-go.

## Consequences

- (+) **Validated synchronized salvo:** a 4-interceptor battery launched from different ranges/bearings
  arrives within **0.19 s** of the commanded time against a visibly weaving (~8 g serpentine) threat
  (vs **0.57 s** natural spread; 0.01–0.04 s against milder weaves), all
  intercepting — `experiments/p24_salvo.py` (figure `gallery/figures/p24_salvo.png` + animated
  `gallery/animations/p24_salvo.gif`). Tests in `tests/test_salvo.py` (spread < 0.5 s and ≪ PN; reduces to PN
  when there is no spare time).
- (+) Reuses the `Controller` contract and the shared geometry helpers; composes with the existing
  Hungarian WTA (each assigned interceptor can be given the common impact time).
- (−) Tuned and demonstrated for a slow/coasting target with comparable-speed interceptors and a
  feasible `t_impact`; against a fast *maneuvering* target the timing loosens (the bias competes with
  the maneuver). Two-sided coordination (the target reacting) is out of scope here.
- (−) `k` and the feasible `t_impact` window are problem-dependent; the demo picks `t_impact` as a
  buffer above the slowest natural arrival. An automatic feasibility/gain selection is future work.
