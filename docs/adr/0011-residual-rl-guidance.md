# ADR-0011 — Residual RL guidance (a learned correction on a PN baseline)

- **Status:** Accepted
- **Date:** 2026-06-08

## Context

From-scratch PPO does not transfer to the realistic (L2 aero) plant. Across basic and rich
observations, gravity feed-forward, a redesigned reward, an entropy bonus, GPU, and up to 1.8 M
steps, the policy **collapses to a constant saturated action that ignores the observation**
(verified by instrumenting the rollout: action ≈ +1.000 while the LOS rate swings), intercepting
~0–2 % — even though a hand-coded `action ∝ LOS-rate` from the same observation scores 40/40 in the
same environment (ADR-0005 / P12 / P13). The lagged (τ ≈ 0.2 s) plant plus gravity makes the
from-scratch exploration problem ill-conditioned: a flailing policy never reaches the sparse
intercept bonus, so it settles into a degenerate fixed action.

I deferred this with a concrete plan (recurrent policy, or imitation/residual warm-start from PN).
This ADR records the chosen fix.

## Decision

Add a **residual action parameterization** to the guidance env and the deployment wrapper. The
policy no longer outputs the full command; it outputs a **bounded correction** added to a
**pure-PN baseline**:

```
a_applied = clip( a_pn(state) + residual_scale · a_policy , −1, +1 ) · a_max · ⟂v   (+ gravity FF)
```

where `a_pn` is pure PN expressed in the env's existing scalar ⟂-velocity parameterization
(`pn_baseline_scalar`, exactly reproducing the tested `pure_pn` law), `residual_scale = 0.5`, and
`a_policy ∈ [−1, 1]` is the learned residual. Implemented as `action_mode="residual_pn"` on
`InterceptionEnv` and mirrored on `RLGuidance` so training and deployment share the parameterization
(the fairness invariant). The effort penalty is charged on the **residual** magnitude, so the policy
is rewarded for small corrections.

This is **residual policy learning** (Silver et al. 2018; Johannink et al. 2019) applied to missile
guidance: the analytic baseline supplies the bulk of the command and a good behavioral prior, so
(a) a zero residual is already competent PN — eliminating the collapse — and (b) learning only has
to discover the *maneuver-anticipation* correction that PN lacks against jinking/breaking targets.
As far as I can find, this specific PN-residual law is not a standard published guidance method; it is a
documented hybrid the benchmark contributes (`experiments/p15_residual_rl.py`, ADR cross-ref in code).

## Outcome (held-out, 100 trials/scenario, realistic L2 aero)

Two variants were trained and benchmarked:

| Scenario | Recurrent APN-residual | Residual-PN (MLP) | True PN | Augmented PN | Sliding-mode | From-scratch PPO |
|---|---|---|---|---|---|---|
| crossing | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| weave 18 g | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| jink 22 g | **0.95** | 0.68 | 0.81 | 0.93 | 1.00 | 0.00 |

- The **PN-residual MLP** (`experiments/p15_residual_rl.py`) resolves the collapse (1.00 / 1.00 /
  0.68 vs from-scratch ~0) but does not beat the classical laws on the jink.
- The **recurrent APN-residual** (`experiments/p16_recurrent_residual.py` — APN baseline + LSTM
  policy) reaches **0.95 on the jink, beating True PN (0.81) and Augmented PN (0.93)** at lower
  aggregate effort than APN, trailing only sliding-mode (1.00). Ablation: APN baseline + memory lift
  the jink 0.68 → 0.95. A genuine *learned* win over the PN family on the hardest realistic case,
  with sliding-mode still leading.

## Consequences

- (+) Resolves the deferred failure on its own terms — a learned policy that *runs* on the realistic
  plant, and (recurrent + APN baseline) actually *beats* PN/APN on the unpredictable jink. Figures
  `gallery/figures/p15_residual_rl.png`, `gallery/figures/p16_recurrent_residual.png`; CSVs in `results/`.
- (+) Reuses all existing machinery (rich obs, VecNormalize, curriculum, the `Controller` contract);
  no change to the engagement core. The mechanism is unit-tested deterministically without training
  (`tests/test_residual_rl.py`, 7): baselines match `pure_pn` / the APN feed-forward formula; a zero
  residual intercepts where a constant-saturated absolute action misses; the recurrent wrapper threads
  and resets LSTM state across episodes.
- (+) `baseline="pn"|"apn"` selects the analytic prior (`pn_baseline_scalar` / `apn_baseline_scalar`);
  the APN baseline feed-forwards the target's achieved lateral accel (from its state) so the learned
  part starts maneuver-aware. `RLGuidance(recurrent=True)` supports LSTM policies for the
  partial-observability of unpredictable maneuvers.
- (−) The learned part is, by construction, a *correction*: on scenarios where PN already succeeds
  the residual's value is marginal; its payoff is on the hard (jink / reactive-break) cases.
- (−) The baseline must be cheap and on the same action manifold (⟂ velocity). A baseline with a
  different control parameterization would need its own scalar projection.
- (−) The recurrent policy still trails the robust sliding-mode on the random jink (0.95 vs 1.00);
  closing that gap (deeper/longer training, OGL baseline, reactive-break scenarios) is future work.
