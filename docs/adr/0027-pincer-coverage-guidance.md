# ADR-0027 — Pincer coverage guidance (a pair covers both escape branches)

- **Status:** Accepted
- **Date:** 2026-06-10

## Context

A target that breaks hard toward an **unpredictable side** defeats a lone True-PN interceptor —
whichever lead it commits to, the coin-flip goes the other way half the time (and in the hard regime
here, *both* ways: the late 30 g turn outruns PN's correction). Crucially, throwing a **second
identical interceptor** at it buys nothing: identical laws ⇒ identical trajectories ⇒ perfectly
correlated failures. Multi-interceptor value requires *decorrelating* the pair.

## Decision

`intercept.guidance.pincer.PincerGuidance` — wrap any base `GuidanceLaw` to steer at a **virtual
aim-point offset laterally from the target**: ``virtual = target ± β·R·⟂̂_LOS``, full above
``r_split``, tapering to **zero inside ``r_merge``** so each interceptor converges to the *plain*
base law for the endgame (a first version without the taper-to-zero left a guaranteed terminal miss
∝ β — documented bug). ``pincer_pair`` builds the ``(+1, −1)`` pair. No communication needed beyond
the initial side assignment; pure geometry.

## Result (30 g surprise break at 1.8 km, unpredictable side; 60 paired diverse geometries)

Two threat regimes — a clean cruise (the coverage *mechanism*) and a realistic ~4 g weaving cruise
(the stressed case); the split is matched to each envelope (β = 0.20 / 0.22, disclosed):

| configuration | clean cruise | weaving cruise |
|---|---|---|
| single True PN | 0.00 | 0.00 |
| redundant True PN × 2 | 0.00 | 0.00 |
| **pincer True PN × 2** | **1.00** | **0.45** |
| redundant Augmented PN × 2 (reference) | 1.00 | 1.00 |

On the clean cruise the coverage is **perfect, with every break taken by the interceptor covering
that side** — cooperative geometry substituting for acceleration sensing (APN needs a 9-state filter
feeding ``a_T``; the pincer needs only the PN seeker). The weaving cruise measurably erodes it
(1.00 → 0.45): the weave perturbs the split geometry, while APN's feedforward stays robust. Caveats:
(1) with an acceleration estimate available, a law upgrade alone solves this regime; (2) the split
width must be matched to the threat's break envelope — the working β window is narrow (±10 % fails),
analogous to ITCG's gain sensitivity; (3) at a *heavy* visible ~8 g serpentine cruise the static
split fails at every β (≤0.27) — a fixed side-split cannot cover a serpentine corridor (dynamic /
re-assigned splits are future work), while APN holds 1.00. Figure `gallery/figures/p34_pincer.png` (both branches +
grouped Monte-Carlo bars), GIF `gallery/animations/p34_pincer.gif`.

## Consequences

- (+) A second genuinely *cooperative* guidance behavior (after the salvo's impact-time control):
  the pair's value comes from coordination, not redundancy — measured against the exact
  failure-mode it fixes, with the equal-resources control (redundant pair) reported.
- (−) Side assignment is static (set at launch); choosing/re-assigning sides from the estimated
  threat state, three-way splits for 3-D cones, and an automatic β/r_merge selection from the
  expected break envelope are future work.
