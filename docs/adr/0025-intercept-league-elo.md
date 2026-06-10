# ADR-0025 — The INTERCEPT League: Bradley-Terry/Elo ratings over laws and evaders

- **Status:** Accepted
- **Date:** 2026-06-10

## Context

With 8+ guidance laws and 6+ adversaries the benchmark's P(intercept) tables grow quadratically and
stop being readable; there is no *single scale* answering "which law is strongest, and which evader
is hardest?". Game-rating systems solve exactly this for chess/Go/LLM arenas — but a search of the
literature found **no prior work rating guidance laws and evaders on a Bradley-Terry/Elo ladder**
(BT/Elo appears only in LLM-arena and forecasting benchmarks; guidance comparisons remain ad-hoc
tables). That makes the League both genuinely useful and a novel benchmark framing.

## Decision

- `intercept.benchmark.league.bradley_terry(names, wins)` — a Zermelo/MM maximum-likelihood
  Bradley-Terry fit with additive smoothing (undefeated participants stay finite), reported on the
  **Elo scale** (BT with base 10 / scale 400 *is* Elo; mean anchored at 1500). Order-free and
  deterministic, unlike sequential Elo updates. Plus `elo_expected_score` for predicted win rates.
- `experiments/p32_league.py` — the tournament: every guidance law (PN family, OGL, sliding-mode,
  and the *learned* residual + recurrent-APN-residual policies) vs every evader (scripted,
  game-theoretic anti-LOS, and the *learned* RL evader), over paired seeded geometries on the
  realistic L2 aero plant (~Mach 3 vs ~Mach 2, no speed cheat). **Intercept ⇒ the law wins; escape ⇒
  the evader wins.** The BT fit places *both sides on one ladder*.

## Result (40 matches per pairing; 8 laws × 6 evaders = 1,920 engagements)

| rank | participant | side | Elo |
|---|---|---|---|
| 1 | Sliding-mode | guidance | 2214 |
| 2 | **optimal (anti-LOS)** | **evader** | **2019** |
| 3 | Optimal (OGL) | guidance | 1917 |
| 4 | Augmented PN | guidance | 1871 |
| 5 | RL recurrent APN-res | guidance | 1681 |
| 6 | reactive break 25g | evader | 1571 |
| 7 | RL evader | evader | 1537 |
| 8 | RL residual-PN | guidance | 1530 |
| 9 | Pure PN | guidance | 1423 |
| 10 | True PN | guidance | 1310 |
| 11 | ZEM PN | guidance | 1310 |
| 12 | telegraph jink 22g | evader | 1180 |
| 13 | straight | evader | 719 |
| 14 | weave 18g | evader | 719 |

Headlines the tables couldn't show at a glance: **sliding-mode is the champion** (robustness wins the
league); the **game-theoretic anti-LOS evader out-rates every law except sliding-mode** — a *target*
is the #2 player; the learned laws sit mid-table (recurrent APN-residual above the PN family, plain
residual at PN level); deterministic scripted evaders are simply food (719). The fitted ratings give
calibrated predictions for unplayed pairings, e.g. expected score sliding-mode vs anti-LOS ≈ 0.75.
Figure `gallery/figures/p32_league.png`.

## Consequences

- (+) One readable leaderboard across paradigms *and* adversaries — and `elo_expected_score`
  predicts P(intercept) for **unplayed pairings**. A genuinely novel benchmark framing for guidance.
- (+) Pure post-processing of the existing fair Monte-Carlo harness — no new simulation assumptions;
  tests cover ordering recovery, smoothing, and the observed-rate ↔ Elo-gap consistency.
- (−) Binary outcomes only (intercept/escape) — miss distance and control effort are not in the
  rating; a margin-aware (Davidson/ordinal BT) extension is future work.
- (−) Ratings are relative to this pool; adding a participant shifts the ladder (standard BT caveat).
