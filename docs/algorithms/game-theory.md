# Pursuit-evasion game theory (Apollonius circle & optimal evasion)

Modules: [`intercept/guidance/game.py`](../../intercept/guidance/game.py),
[`intercept/adversary/optimal_evader.py`](../../intercept/adversary/optimal_evader.py).

## Apollonius circle

For two constant-speed players, the **Apollonius circle** is the locus of points the evader and a
faster pursuer reach simultaneously: `|X − E| / |X − P| = α`, where `α = v_E / v_P < 1`. Its
**interior is the evader's dominance region** (points the evader reaches first). `apollonius_circle`
returns `(center, radius)`:

```
center = (E − α²·P) / (1 − α²),   radius² = |center|² − (|E|² − α²|P|²)/(1 − α²)
```

`intercept_point(P, E, V, v_P)` solves the intercept triangle `|E + V·t − P| = v_P·t` for the
soonest capture point (or `None` if a straight-running evader cannot be caught). For the optimal
straight-line evader this capture point lies on the Apollonius circle. References: Isaacs,
*Differential Games*; Weintraub, Pachter & Garcia (ACC 2020); Dorothy et al., *Automatica* 2024.

## `ApolloniusGuidance` (geometric pursuer)

Each step it predicts the evader's straight-line motion, solves for the soonest intercept point, and
steers the velocity (constant-bearing) toward it — geometrically optimal against a non-maneuvering
evader, re-planned each step against a maneuvering one. Falls back to pure pursuit if the evader is
momentarily uncatchable. **Validated:** intercepts crossing targets; safe at zero velocity.

## Game-theoretic optimal evader

In the simple-motion game against a faster pursuer, capture is inevitable, and the evader's optimal
*game-of-degree* strategy (maximize time-to-capture) is to **flee directly along the anti-line-of-
sight**, away from the pursuer (`optimal_evader`). This reacts to the pursuer every step — a harder,
smarter adversary than the open-loop scripted weave/jink.

## Results (P6 demo)

- **Apollonius diagram** (`gallery/figures/p6_apollonius.png`): pursuer, evader (with velocity), dominance
  circle, and the predicted capture point on its boundary.
- **Evader robustness vs. True PN** (`gallery/figures/p6_evader_robustness.png`, 150 trials each): in a
  pursuer-dominant regime every adversary is intercepted (P ≈ 1.0), so the differentiator is
  **time-to-intercept** — the game-theoretic optimal evader stretches capture to **≈8.7 s vs ≈3.6 s**
  for straight/weave/jink. A subtle, correct finding: against an acceleration-limited PN, the
  *optimal (simple-motion) evader maximizes capture time*, whereas weave/jink target *miss distance*;
  in a low-authority regime those instead drive intercept probability down. The two notions of
  "hardest adversary" differ — exactly the kind of nuance the platform makes measurable.

## Notes / extensions

- The pursuer's straight-line prediction is exact only for a non-maneuvering evader (re-planned each
  step otherwise). Full HJ-reachability / value-function solutions are a heavier future extension.
- **Adversarial-RL self-play** (training an evader against the interceptor, and co-evolution) is the
  natural next step — the env and `Controller` contract already support an RL evader; deferred to a
  later iteration.
