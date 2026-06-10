# Benchmark methodology

How INTERCEPT compares guidance laws rigorously and reproducibly. Module:
[`intercept/benchmark/`](../../intercept/benchmark/).

## Overview

For each **(algorithm × scenario)** cell the harness runs a seeded Monte-Carlo of randomized
engagements and aggregates standardized metrics. All algorithms see *identical* sampled engagements
per seed (the
[fairness invariant](../adr/0003-benchmark-fairness-invariants.md)), so outcome differences are
attributable to the guidance law alone.

```
scenario suite (YAML)  ─┐
guidance factories     ─┤→ run_benchmark ─→ [BenchmarkRow(algorithm, scenario, MetricSummary)]
n_trials, seed         ─┘        │
                                 ├─ format_table / write_csv   → results table + CSV
                                 ├─ plot_pintercept_bars        → P(intercept) bar chart
                                 └─ compute_capture_region/plot → capture-region heatmap
```

## Scenarios

A [`ParametricScenario`](../../intercept/benchmark/scenario.py) is a *distribution* over engagement
geometries plus a target maneuver, (de)serialized to YAML in [`scenarios/`](../../scenarios/):

| Field | Meaning |
|---|---|
| `interceptor_speed`, `interceptor_a_max` | launch speed (m/s) and acceleration limit (m/s²) |
| `target_speed`, `target_heading_deg` | target speed and inertial heading (180° = toward −x) |
| `range_min/max`, `offset_min/max` | sampled down-range and cross-range of the target start |
| `interceptor_aim` | `lead` (aim at target start) or `downrange` (+x) |
| `maneuver` | `null` / `weave` / `step` / `bang_bang` (+ params; accel in m/s², 1 g ≈ 9.81) |
| `kill_radius`, `dt`, `t_max` | intercept threshold, time step, horizon |

`scenario.sample(rng)` draws a randomized engagement (Monte-Carlo); `scenario.at(x, y)` places the
target deterministically (capture-region sweeps).

### Suite (S1–S5)

| Scenario | Geometry | Purpose |
|---|---|---|
| S1 `headon_nonmaneuvering` | head-on, straight | easy baseline (near-100% expected) |
| S2 `crossing` | target crosses the LOS | non-zero LOS rate; lead behavior |
| S3 `tail_chase` | target receding | pursuit / closing-speed limits |
| S4 `weaving_6g` | 6 g sinusoidal weave | PN-stressing maneuver; APN benefit |
| S5 `high_offset_crossing` | large cross-range, oblique | launch envelope / capture region |

## Metrics

[`MetricSummary`](../../intercept/benchmark/metrics.py) per cell:

- **P(intercept)** with a **Wilson score 95% CI** (robust for small samples / extreme rates).
- **Miss distance** — mean, median, 95th percentile (over all trials; closest approach).
- **Time-to-intercept** — mean over successful intercepts.
- **Control effort** — mean of ∫‖a‖² dt for the interceptor (energy proxy).

## Reproducibility

A result is fully determined by `(scenario suite, algorithm set, seed, n_trials)`. Trial `i` uses
`SeedSequence(seed).spawn(n_trials)[i]`; RNG is never sampled inside the engagement loop. The CSV
under `results/` plus the seed reproduces every figure. Validated in `tests/test_benchmark.py`.

## Capture-region analysis

[`compute_capture_region`](../../intercept/benchmark/capture_region.py) sweeps the target's start
position over a grid and records intercept/miss at each cell; the intercepting set is the law's
**capture region** for that engagement (larger = more robust). Rendered as a miss-distance heatmap
with the capture boundary outlined.

## How to run

```bash
python experiments/p2_benchmark.py --trials 200 --seed 0 [--no-show]
```

Produces `results/p2_benchmark.csv`, `gallery/figures/p2_pintercept_by_scenario.png`, and
`gallery/figures/p2_capture_region_truepn_S5.png`.

## Pairwise significance (paired bootstrap)

Wilson intervals bound each algorithm's P(intercept) on its own, but the right question is often
*"is algorithm A significantly better than B?"* The fairness invariant makes that a **paired**
comparison: with the same scenario and seed, trial *i* has identical initial conditions and target
behavior for both algorithms, so the difference is measured trial-by-trial. `paired_bootstrap` (and
the convenience `compare_intercept`) resamples trial indices to produce a percentile CI and a
two-sided p-value for the mean difference (`significant` ⇔ the CI excludes zero). This turns the
benchmark from point estimates into statistically grounded statements — used in the capstone
(`experiments/p19_capstone_benchmark.py`), e.g. on the 2-D L2 jink Augmented PN beats True PN by
+0.09 (95% CI [+0.03, +0.16], p = 0.009).

## Dimensions and fidelity

The same runner/metrics evaluate **2-D and 3-D** (`ParametricScenario` / `ParametricScenario3D`)
across the fidelity ladder **L0→L3** (`model = point_mass | aero | realistic`), because everything
operates on a dimension-agnostic `EngagementResult`. The capstone benchmark sweeps the full
paradigm × fidelity × dimension grid in one figure.

## Limitations (current)

- Target maneuvers are open-loop or closed-loop scripted/game-theoretic; adversarial-RL evaders are
  future work.
- Classical gains were chosen "best-effort" (N=4); `experiments/p21_gain_sensitivity.py` now sweeps
  N ∈ {2..7} and shows N=4 sits on the robust capture/effort plateau (True PN's jink capture rises
  with N; Augmented PN peaks near N≈3 then declines as effort climbs). Figure
  `gallery/figures/p21_gain_sensitivity.png`.
- The 3-D Monte-Carlo benchmark compares True/Augmented PN, Optimal (OGL-3D), and Sliding-mode
  (SMG-3D); 3-D MPC and RL are demonstrated and tested separately but kept out of the heavy sweeps
  for solve/rollout cost.
