# Elo Methodology — the INTERCEPT League

## Motivation

A cross-paradigm benchmark table of P(intercept) for every (law, evader) pair is informative but
hard to read at a glance: with six guidance laws and up to six adversaries, the table has up to
36 cells. The fundamental question — *on equal footing, which paradigm is strongest?* — has no
single obvious answer in that format.

I adapted the Bradley-Terry skill-rating model, standard in competitive rankings, to answer it:
treat every seeded engagement as a **match**, aggregate match outcomes across the whole round-robin,
and fit a single latent *strength* parameter to each participant. Reported on the familiar Elo scale
(mean 1500), this puts every guidance law and every evader on one ordered ladder — one number per
participant instead of a full cross-table.

The key property that makes this meaningful: because every match is played on **identical dynamics**
with the **same seeded geometries** and a shared controller contract, the only source of variance
between participants is the algorithm. The Elo gap is an algorithm gap, not a testbed gap.

---

## Pairing protocol

- **Participants:** all enabled guidance laws (pursuers) and all enabled evaders.
- **Round-robin:** every pursuer is matched against every evader.
- **Match count per pairing:** `--trials` (default 40) seeded engagements. The same 40 geometry
  seeds are used for all pairings, so match outcomes are *paired* — pairwise differences cancel
  geometry variance.
- **Outcome rule:** intercept within the kill radius → pursuer wins; time-out or escape → evader
  wins. Draws do not occur (either the engagement terminates with kill or it times out).
- **Dynamics:** L2 (`AeroMissile2D`) for both participants — gravity, drag, g-limit, autopilot lag
  — at realistic comparable speeds (~Mach 3 interceptor, ~Mach 2 evader, ~1.43× closing edge). No
  participant gets a physics advantage.

---

## Bradley-Terry model

The Bradley-Terry model assigns latent strength parameters `π_i > 0` to each participant `i`.
For a match between `i` and `j`, the probability that `i` wins is:

```
P(i beats j) = π_i / (π_i + π_j)
```

Given an observed win-count matrix `W` (`W[i,j]` = number of matches `i` beat `j`), the
maximum-likelihood estimate of `π` is found by the Zermelo / minorization-maximization (MM)
iteration:

```
π_i^{(t+1)} = W_i· / Σ_j  n_{ij} / (π_i^{(t)} + π_j^{(t)})
```

where `W_i· = Σ_j W[i,j]` is participant `i`'s total wins and `n_{ij} = W[i,j] + W[j,i]`
is the total matches played between `i` and `j`. The iteration is run until the maximum
absolute change in `log π` is below `1e-10`, or for at most 10 000 steps; convergence is
typically reached in under 100 iterations.

### Additive smoothing

To keep ratings finite when a participant wins (or loses) every match, a pseudo-count of
`eps = 0.5` wins is added to **both directions** of every pair that actually played — equivalent
to a uniform Dirichlet prior with `alpha = eps + 1`. Pairs that never played receive no
pseudo-counts (leaving the between-group comparison anchor-free if the groups are disjoint, which
does not occur in a full round-robin).

### Elo scale

Bradley-Terry strengths are mapped to the familiar Elo scale via:

```
Elo_i = 1500 + (400 / ln 10) × ln(π_i)
```

This is the exact Elo formula under base-10 logistic function (the BT model and the Elo model are
identical up to parametrisation). The geometric-mean normalisation `Σ ln(π_i) = 0` anchors the
mean rating at exactly 1500. The expected score of `a` against `b` is:

```
E(a, b) = 1 / (1 + 10^{(Elo_b − Elo_a) / 400})
```

which predicts unplayed match-ups without additional fitting.

### Why BT over raw win-rate

Raw win-rate against a fixed opponent pool ranks participants relative to *that pool*, not on an
absolute scale. If the pool changes (e.g. a new evader is added), all ratings shift. BT places
everyone on one latent axis whose origin and scale are fixed by normalisation, making it stable
under pool changes (existing ratings shift only by re-fitting, not by design). It is also
transitive by construction: if `A` tends to beat `B` and `B` beats `C`, the model assigns
`π_A > π_B > π_C`, even if `A` and `C` never met directly.

---

## Bootstrap confidence intervals

A single BT fit on the observed win counts gives a point estimate. To quantify uncertainty, I use a
**non-parametric paired bootstrap** at the match level:

1. For each of 1 000 replicates:
   a. For every (pursuer, evader) pairing, resample its `trials` binary match outcomes *with
      replacement* (preserving the pairing structure — geometry variance is not resampled away).
   b. Aggregate the resampled outcomes into a win-count matrix.
   c. Fit BT and record each participant's Elo rating.
2. Report the 2.5th–97.5th percentile of the 1 000 replicate ratings as the **95 % CI**.

Resampling at the match level (not at the trial-geometry level) is appropriate because:
- Geometry is a nuisance variable we want to average over, not vary.
- The match outcome is the atomic unit of information the BT model consumes.
- Resampling within pairings preserves the paired structure, so the bootstrap estimates sampling
  variance from the binary outcome distribution, not from geometry variation.

**Interpreting CIs.** If two adjacent participants on the ladder have overlapping 95 % CIs they are
marked as a *statistical tie* — the data are consistent with either ordering. A 200-point Elo gap
with non-overlapping CIs is a clear finding; a 50-point gap within the CI width of a single rating
is noise.

---

## Scenario sensitivity

To check whether rankings are robust or scenario-dependent, the BT ladder is refit on four
disjoint subsets of the match log:

| Subset | Filter |
|---|---|
| **Head-on** | engagement geometry with lateral offset < 500 m (roughly straight-in) |
| **Crossing** | lateral offset ≥ 500 m (target crosses the interceptor's path) |
| **Scripted** | matches against rule-based evaders (straight, weave, jink, reactive break) |
| **Adversarial** | matches against intelligent evaders (game-theoretic optimal, RL-trained) |

Each subset is a strict subset of the full match log — no new simulations are run. A subset is
omitted from the heatmap if it contains fewer than `2 × n_participants` matches.

**How to read the rank-stability heatmap.** Rows are participants (sorted by their overall rank);
columns are subsets. The cell value is the participant's rank in that subset (1 = highest rated).
A participant whose row is uniformly green is robustly top-ranked across all scenario types; a
participant whose row is mixed shows scenario-dependent performance — that is itself a finding
worth reporting.

**Planned extension.** A noisy-sensor subset (matching through an EKF/IMM estimation loop rather
than perfect-state observations) would add a fifth column. This requires wiring the sense→estimate
→guide closure into the league runner and is left for a future iteration.

---

## Reproducibility

All match outcomes are seeded: geometry `k` uses `np.random.default_rng((seed, k))`. The BT fit
is deterministic (MM iteration from a ones initialisation). The bootstrap uses
`np.random.default_rng(42)` by default, making the CI values reproducible.

Reproduce the full ladder:

```bash
python experiments/p32_league.py [--trials 40] [--boot 1000]
```

Results are committed under `results/p32_league.csv` (ratings + CIs) and `results/p32_league.md`
(markdown leaderboard with tie flags).

---

## References

- Bradley, R.A. & Terry, M.E. (1952). "Rank analysis of incomplete block designs." *Biometrika* 39.
- Hunter, D.R. (2004). "MM algorithms for generalized Bradley-Terry models." *Ann. Statist.* 32(1).
- Elo, A.E. (1978). *The Rating of Chessplayers, Past and Present.* Arco.
