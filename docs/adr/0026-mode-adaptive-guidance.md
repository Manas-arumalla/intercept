# ADR-0026 — Mode-adaptive guidance (IMM belief arbitrates the law)

- **Status:** Accepted
- **Date:** 2026-06-10

## Context

Fixed guidance laws sit at fixed points of the robustness/efficiency trade: True PN is cheap but
misses maneuvers; Augmented PN catches them but spends control authority all flight. The IMM in the
estimation stack *already computes* the statistic that distinguishes these regimes — the maneuver-
mode probability — but nothing consumed it for guidance. Estimation-aware **law arbitration** is the
missing composition.

## Decision

`intercept.guidance.adaptive.ModeAdaptiveGuidance`: an interceptor-mounted sensor feeds a CV/CA IMM;
each step the command is a blend of a quiescent law (True PN) and a maneuver law (Augmented PN)
weighted by the maneuver belief with a **sharpness exponent** (``w = μ^γ/(μ^γ+(1−μ)^γ)``, default
γ=3 — commits harder once the belief tilts, no chattering hard switch). Reproducible sensing per
ADR-0003 (injected RNG); ``mu_history`` logged for analysis. Two tuning findings, documented:

- At a 100 Hz seeker the IMM needs a **sticky transition** (``p_stay=0.995`` ⇒ ~2 s dwell) to
  integrate mode evidence; the default 0.95 leaves the belief wandering near 0.5.
- The **sharpness** γ=3 turned the late-break effort savings from −26 % (linear blend) to −50 %.

## Result (weaving cruise → 20 g break, 60 paired jittered geometries, ~Mach 3 vs ~Mach 2)

The threat flies a realistic profile — a light ~4 g evasive weave on the cruise-in, then a sustained
20 g break — so the IMM must *discriminate* the benign weave from the real maneuver:

| law | P(intercept) | mean effort |
|---|---|---|
| True PN (estimated) | 0.52 | 79 k |
| Augmented PN (estimated) | 0.55 | 426 k |
| **Mode-adaptive (IMM)** | **0.62** | **265 k** |

On this harder threat the adaptive law is **the best of the three** — APN loses its edge because its
acceleration feedforward chases the weave, while the IMM blend stays calm through it and hardens for
the break — at **38 % less effort than APN**. (On the simpler straight-cruise variant: PN 0.55 / APN 0.78 / adaptive 0.72 at half APN's effort.
On a *visible* ~8 g serpentine cruise the IMM correctly saturates and adaptive defaults to APN-like
behaviour — 0.92 @ 449k vs APN 1.00 @ 426k — i.e. it spends effort exactly when the threat actually
maneuvers; regimes reproducible via `--weave-g`.) **Operating envelope:** against a *sustained-from-launch* hard maneuver the
~0.8 s detection lag costs the intercept and always-on APN wins — mode-adaptive is for threats that
are quiescent-ish for most of the flight. Figure `gallery/figures/p33_mode_adaptive.png`
(belief-coloured trail + μ timeline + MC bars), GIF `gallery/animations/p33_mode_adaptive.gif`.

## Consequences

- (+) A novel estimation-aware guidance composition using only existing components (IMM + PN + APN),
  with the trade-off measured, the envelope stated, and tests (`tests/test_adaptive.py`: the belief
  actually switches; effort < APN at equal intercept on the favorable case).
- (−) Two laws and one belief dimension; a richer bank (CT model, spiral mode) and an
  effort-optimal arbitration (e.g. risk-sensitive weighting) are future work.
