# Project Plan — "INTERCEPT": A Research-Grade Missile Interception & Defense Simulation Benchmark

**Name:** `INTERCEPT` — Interception Networks, Tracking, Estimation, Replanning, Control & Pursuit-evasion Toolkit.
**Date:** 2026-06-07 · **Status:** Design document and roadmap
**Builds on:** [00_RESEARCH_REPORT.md](00_RESEARCH_REPORT.md)

---

## 1. Objectives

### 1.1 Primary objective (the thesis)
Build a **reproducible, open-source simulation platform and benchmark** that rigorously compares **classical, optimal, game-theoretic, and learned (RL)** missile-interception guidance on **identical, configurable engagement geometries** with **shared metrics and Monte-Carlo statistics**. This fills the verified gap (Research §E.1): such cross-paradigm comparisons exist only in scattered single studies on non-shared geometries.

### 1.2 Secondary objectives
- Demonstrate the **full autonomy stack**: sense (sensor models) → estimate (EKF/UKF/IMM) → predict → decide (assignment/guidance) → act → replan.
- Showcase **multi-agent / swarm / cooperative** interception (salvo, WTA) and **adversarial** evaders (scripted → game-theoretic → RL self-play).
- Deliver **publication-quality visualization** and a **reproducible experiment pipeline** (seeds, configs, results tables).
- Produce a project that reads as **research-grade engineering**: clean architecture, tests, docs, ADRs, benchmark report.

### 1.3 Non-goals (explicit scope guard)
- No hardware, no real targeting/sensor data, no munitions/warhead modeling, no detection-evasion or operational tooling. Simulation + public textbook algorithms only.
- Not a high-fidelity 6-DOF aero simulator first (6-DOF is an *optional later extension*, not a pillar).

### 1.4 Success criteria
- ≥4 guidance paradigms (PN/APN baseline, OGL/optimal, MPC, RL) + game-theoretic 1v1, benchmarked on a shared scenario suite with Monte-Carlo P_intercept, miss distance, effort, and capture regions.
- Estimation-coupled study (guidance vs estimator noise) and at least one adversarial-evader comparison.
- One-command reproducible benchmark producing the headline figures; CI green; docs complete.

---

## 2. Architecture

**Design principles:** plug-in interfaces (swap dynamics/guidance/estimator/sensor independently), config-driven scenarios (YAML), deterministic+seeded, separation of *simulation core* from *algorithms* from *experiment/analysis*.

```
 ┌──────────────────────────────────────────────────┐
 │ Scenario (YAML) │
 │ geometry · entities · sensors · noise · seeds │
 └───────────────────────┬──────────────────────────┘
 │
 ┌──────────────────────────▼───────────────────────────┐
 │ Simulation Core │
 │ ┌────────────┐ ┌──────────────┐ ┌───────────────┐ │
 │ │ Dynamics │ │ Integrator │ │ Engagement / │ │
 │ │ (2D/3D PM, │ │ (RK4/RK45) │ │ event manager │ │
 │ │ autopilot │ │ │ │ (intercept, │ │
 │ │ lag,6DOF*)│ └──────────────┘ │ miss, ground)│ │
 │ └────────────┘ └───────────────┘ │
 └───────┬───────────────┬───────────────┬───────────────┘
 │ true state │ measurements │ commands
 ┌─────────────▼──┐ ┌────────▼─────────┐ ┌─▼────────────────┐
 │ Sensors │ │ Estimation / │ │ Guidance / │
 │ radar/IR/EO │──▶│ Tracking │──▶│ Control │
 │ noise, clutter │ │ EKF/UKF/IMM/PF, │ │ PN·APN·OGL·SMG· │
 │ │ │ fusion(CI), pred │ │ MPC·RL·Game· │
 └────────────────┘ └──────────────────┘ │ cooperative/WTA │
 └──────────────────┘
 ▲ │
 │ Adversary (evader) │
 └──── scripted / game-theoretic / RL ◀──┘

 ┌──────────────────────────────────────────────────────────────────┐
 │ Experiment & Analysis layer (outside core) │
 │ Monte-Carlo runner · benchmark harness · metrics · capture-region │
 │ sweeps · plotting · report generation · RL training (Gym env) │
 └──────────────────────────────────────────────────────────────────┘
```

**Key interfaces (abstract base classes):**
- `Dynamics`: `step(state, control, dt) -> state`; implementations: `PointMass2D`, `PointMass3D`, `PointMass3DWithLag`, (`SixDOF` optional).
- `Sensor`: `measure(true_state, rng) -> Measurement` (radar: range/range-rate/angles; IR/EO: angles-only; noise+clutter).
- `Estimator`: `predict`, `update(measurement)` → `(x̂, P)`; implementations: EKF, UKF, IMM, PF.
- `Guidance`: `command(estimate, own_state) -> acceleration`; implementations per paradigm.
- `Adversary`: `maneuver(state) -> acceleration`; scripted/game/RL.
- `Allocator` (multi-agent): `assign(interceptors, threats) -> assignment`.

**The RL bridge:** a thin `gymnasium.Env` / `pettingzoo` wrapper around the Simulation Core so RL training and classical evaluation use the *same* dynamics — guaranteeing a fair comparison (a key correctness property of the benchmark).

---

## 3. Simulation Design

- **Frames & state:** inertial Cartesian; relative LOS computed for guidance. 2D first (x, y, v, γ), then 3D (position, velocity, optional Euler/quaternion for 6-DOF).
- **Fidelity ladder (progressive):**
 1. **L0 — 2D point-mass, ideal:** instantaneous acceleration command, no lag/noise. (Algorithm development & first comparisons.)
 2. **L1 — 2D + autopilot lag + acceleration saturation + sensor noise.** (Realism that breaks idealized results — Research §E.5.)
 3. **L2 — 3D point-mass/kinematic** (same ladder of imperfections).
 4. **L3 (optional) — 3D with simplified aero / autopilot loops**; **6-DOF via JSBSim** as a stretch goal / MATLAB-Simulink cross-check.
- **Time stepping:** fixed-step RK4 (deterministic, seeded); RK45 option for accuracy studies. Guidance/estimator update rates configurable (decouple control rate from integration rate).
- **Engagement manager:** detects intercept (miss < kill radius), miss (range increasing past closest approach), ground/boundary, timeout; logs the full trajectory.
- **Determinism:** every run takes an explicit seed; configs hashed and stored with results.

---

## 4. Algorithms (Implementation Roster)

Tiered by the roadmap (see §10). ✅ = pillar, ◇ = secondary, ★ = stretch/novel.

| Area | Classical (baseline/control group) | Novel / research |
|---|---|---|
| **Guidance** ✅ | PN, Pure/True PN, ZEM-PN, APN, OGL/LQ | AIPN (inverse-optimal), Sliding-Mode, **NMPC (acados)**, **RL (PPO/SAC/TD3)** ★ |
| **Estimation** ✅ | EKF, UKF | IMM (CV/CA/CT), Particle filter ◇, LSTM trajectory predictor ★ |
| **Sensor fusion** ◇ | Single-sensor radar | Radar+IR/EO fusion, Covariance Intersection |
| **Multi-agent** ◇ | Hungarian WTA | Consensus salvo / fixed-time cooperative guidance ★ |
| **Swarm** ★ | Boids / potential-field defense | MARL (MADDPG/QMIX via PettingZoo) ★ |
| **Game theory** ◇ | — | Apollonius-circle 1v1, reach-avoid / HJ reachability ★ |
| **Adversary** ✅ | Straight, weave, jink, bang-bang | Game-theoretic optimal evader, **adversarial-RL self-play** ★ |
| **Replanning** ◇ | — | Event-triggered NMPC, receding horizon |

**Library choices:** NumPy/SciPy (core), `python-control` (LQR/Riccati for OGL), `FilterPy` (estimation baselines), **CasADi + acados** (MPC/trajopt), **Gymnasium + PettingZoo + Stable-Baselines3/CleanRL** (RL), `hj_reachability` (reachability, optional), PyVista/Plotly/Matplotlib (viz). MATLAB/Simulink optional for a 6-DOF reference cross-check.

---

## 5. Benchmarking Framework

**The centerpiece.** A declarative benchmark harness that runs every registered algorithm against a shared **scenario suite** and emits a standardized results table + figures.

- **Scenario suite (versioned, in `scenarios/`):**
 - `S1_headon_nonmaneuvering` · `S2_crossing` · `S3_tail_chase` · `S4_weaving_target(3g/6g/9g)` · `S5_high_offset` · `S6_noisy_seeker` · `S7_salvo_multi_interceptor` · `S8_swarm_area_defense` · `S9_adversarial_evader`.
- **Harness:** `benchmark run --algos all --scenarios all --trials 1000 --seed 0` → for each (algorithm × scenario): Monte-Carlo over randomized initial conditions, collect metrics, write `results/<timestamp>/...` (Parquet/CSV + config hash + figures).
- **Fairness guarantees (correctness-critical):** all paradigms share the *same* dynamics, sensor noise, and scenario RNG stream; RL agents evaluated on *held-out* seeds/scenarios; classical laws given best-effort gain tuning (documented).
- **Capture-region module:** grid-sweep initial range × aspect angle (× target-g), render success/failure boundary per algorithm.
- **Ablation hooks:** toggle estimator type, noise level, autopilot lag, fidelity level — to produce the "which conclusions survive higher fidelity" study (Research §E.5).

---

## 6. Evaluation Metrics

Primary (per Research §B):
- **Miss distance** (mean, median, CEP) · **P_intercept / P_k** (Monte-Carlo success rate ± Wilson CI) · **Time-to-intercept / flight time** · **Control effort** (∫a² dt, peak g) · **Capture-region area** · **Robustness curve** (P_intercept vs target-g / noise σ / latency).
Secondary:
- Computational cost (wall-clock per decision), sample efficiency (RL), terminal impact-angle error (constrained guidance), salvo time-spread (cooperative).

**Reporting standard:** every headline claim accompanied by N trials, seed, CI, and the scenario id. No single-run claims.

---

## 7. Visualization Strategy

The "visually impressive" requirement is a first-class goal, not an afterthought.
- **2D engagement replays** (Matplotlib animation / FuncAnimation): trajectories, LOS lines, acceleration vectors, miss point.
- **3D engagement** (PyVista or Plotly 3D): interceptor/target tubes, intercept burst, camera fly-through; export MP4/GIF for the README.
- **Apollonius-circle / dominance-region** overlays (game theory) — distinctive, elegant.
- **Capture-region heatmaps** per algorithm (the headline benchmark figure).
- **Monte-Carlo dashboards:** miss-distance distributions, P_intercept-vs-g robustness curves, effort vs accuracy scatter (the "RL's real edge" plot).
- **Swarm animations:** many-vs-many, color-coded assignments.
- **Optional interactive dashboard** (Streamlit/Plotly Dash): pick algorithm + scenario, run live, see metrics. Strong recruiter demo.
- **README hero GIF** + a `gallery/` of figures.

---

## 8. Documentation Structure

(Per the continuous-documentation requirement — this is a deliverable, not overhead.)
```
docs/
 index.md # landing / quickstart
 research/ # this research report + literature notes, BibTeX
 theory/ # derivations: PN, APN, OGL, SMG, MPC, game theory, RL formulation
 architecture/ # system design, interfaces, data flow
 adr/ # Architecture Decision Records (0001-*.md, one per decision)
 algorithms/ # per-algorithm docs: math, params, usage, references
 benchmarks/ # methodology, scenario definitions, how to reproduce
 results/ # generated benchmark report(s), figures, tables
 experiments/ # experiment log (date, hypothesis, config, outcome)
 progress/ # PROGRESS.md / devlog, milestone checkpoints
 tutorials/ # "add a new guidance law", "train an RL agent"
```
- **MkDocs (Material)** for a polished docs site (GitHub Pages). Math via KaTeX.
- **ADRs**: numbered, immutable, one decision each (e.g., "0001 — Python primary stack", "0002 — point-mass first").
- **Experiment tracker:** results tables + config hashes in-repo; optional MLflow/Weights & Biases for RL runs (logged to `docs/experiments/`).
- **Progress log + checkpoints:** `docs/progress/PROGRESS.md` updated each work session; **git tags** `v0.1-pn-baseline`, `v0.2-estimation`, … at each milestone; CHANGELOG.md (Keep-a-Changelog).

---

## 9. Repository Structure

```
intercept/
├── README.md # thesis, hero GIF, quickstart, results teaser, ethics note
├── LICENSE # MIT or Apache-2.0
├── CITATION.cff # make it citable
├── pyproject.toml # pip-installable, pinned deps, ruff/black/mypy config
├── CHANGELOG.md
├──.github/workflows/ci.yml # lint + type + tests + (smoke) benchmark
├── intercept/ # the package
│ ├── core/ # dynamics, integrators, engagement manager, frames
│ ├── sensors/ # radar, ir_eo, noise, clutter
│ ├── estimation/ # ekf, ukf, imm, pf, fusion, predictors
│ ├── guidance/ # pn, apn, ogl, smg, mpc, rl_policy, game, base
│ ├── multiagent/ # wta, cooperative, swarm
│ ├── adversary/ # scripted, game_theoretic, rl_evader
│ ├── envs/ # gymnasium + pettingzoo wrappers (the RL bridge)
│ ├── benchmark/ # harness, scenario loader, metrics, montecarlo
│ └── viz/ # plotting, animation, dashboard
├── scenarios/ # YAML scenario suite (versioned)
├── configs/ # algorithm + experiment configs
├── experiments/ # runnable scripts producing paper figures
├── notebooks/ # exploratory + tutorial notebooks
├── tests/ # unit + property + regression (golden trajectories)
├── results/ # generated (gitignored except curated headline figures)
├── docs/ # (see §8)
└── matlab/ # optional 6-DOF / Simulink cross-check
```

---

## 10. Roadmap (Phases & Milestones)

Each phase ends with a **git tag**, updated **PROGRESS.md**, tests green, and a short **checkpoint report** in `docs/progress/`. Phases are deliverable-complete on their own (de-risked, no big-bang).

| Phase | Milestone (tag) | Deliverables | Verifies |
|---|---|---|---|
| **P0 — Scaffold** | `v0.1-scaffold` | Repo, package, CI, docs site skeleton, ADR-0001/0002, point-mass 2D dynamics + RK4, engagement manager, first trajectory plot | Project runs; 1 trajectory rendered |
| **P1 — Classical guidance baseline** | `v0.2-pn` | PN/Pure/True/ZEM-PN, APN; 2D ideal; unit tests; first engagement animation | PN hits straight & constant-accel targets; matches `propNav`-style reference |
| **P2 — Benchmark harness + metrics** | `v0.3-bench` | Scenario suite S1–S5, Monte-Carlo runner, metrics, capture-region sweep, headline figures | Reproducible PN-vs-APN benchmark table + capture-region heatmap |
| **P3 — Estimation & sensors** | `v0.4-estimation` | Radar/IR sensor models + noise/clutter; EKF/UKF/IMM; estimation-coupled guidance study | Tracking RMSE plots; guidance-vs-estimator-noise ablation |
| **P4 — Optimal & MPC** | `v0.5-mpc` | OGL/LQ (python-control), Sliding-Mode, NMPC (CasADi/acados), event-triggered replanning | NMPC interceptor in benchmark; constrained impact-angle demo |
| **P5 — RL centerpiece** | `v0.6-rl` | Gymnasium env (RL bridge), PPO/SAC/TD3 agents, curriculum, training logs (MLflow/W&B), RL-vs-classical head-to-head on shared geometries + **effort/efficiency** comparison | RL matches/explains classical on miss + effort (Research §8); held-out eval |
| **P6 — Game theory & adversaries** | `v0.7-game` | Apollonius-circle 1v1, reach-avoid; scripted→game→RL evaders; adversarial self-play; S9 | Theory-vs-learned evader comparison; robustness curves |
| **P7 — Multi-agent / swarm** | `v0.8-swarm` | Hungarian WTA, cooperative/fixed-time salvo guidance, Boids + MARL swarm defense; S7/S8 | Salvo simultaneous-arrival demo; swarm animation |
| **P8 — Polish & release** | `v1.0` | 3D upgrade pass (L2), interactive dashboard, full benchmark report, docs site, CITATION, README hero GIF, optional preprint/blog | One-command full benchmark reproduces all headline figures |
| **P9 — Stretch** | `v1.x` | 6-DOF/JSBSim or MATLAB cross-check, differentiable-sim guidance, POMDP/chance-constrained MPC | As scoped |

**Suggested cadence:** P0–P2 establish a *complete, impressive vertical slice* early (a working benchmarked interceptor with figures) — so the repo is portfolio-worthy from ~Phase 2 onward, and every later phase adds a comparison column.

---

## 11. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Scope creep (12 directions) | High | High | Pillars (✅) vs secondary (◇) vs stretch (★) tiering; each phase ships standalone; cut ★ first |
| RL doesn't converge / unstable | Med | Med | Start single-agent point-mass; curriculum; use proven SB3 PPO/SAC; reward-shaping ADR; fall back to a measured "RL is competitive, not dominant" result |
| MPC real-time/solver pain (acados build) | Med | Med | Prototype with do-mpc/CasADi+IPOPT first; acados only if needed; document install |
| Unfair comparisons undermine the thesis | Med | High | Shared dynamics/seeds/scenarios enforced in code; tune classical laws fairly; held-out RL eval; peer-style review of methodology |
| Adversarial self-play instability | Med | Low | Optional/stretch; scripted+game-theoretic evaders give the comparison even if RL self-play is dropped |
| 6-DOF/aero rabbit hole | Med | Med | Explicitly optional (P9); point-mass + lag is enough for the thesis |
| "Looks like weapons work" perception | Low | Med | Prominent ethics/scope note; defensive/educational framing; no operational content |
| Reproducibility rot | Med | Med | Pinned deps, seeds, config hashing, CI smoke-benchmark, golden-trajectory regression tests |

---

## 12. Validation Methods

- **Unit tests:** dynamics conservation, integrator accuracy (vs analytic), PN closed-form sanity (zero LOS-rate → straight), filter consistency (NEES/NIS).
- **Property tests:** intercept is frame-invariant; seed → deterministic trajectory.
- **Regression tests:** golden trajectories (hash-locked) so refactors can't silently change physics.
- **Cross-validation against references:** PN behavior vs `propNav`/textbook closed-forms; EKF/UKF vs FilterPy; (optional) 6-DOF vs MATLAB.
- **Statistical validation:** Monte-Carlo with confidence intervals; capture-region reproducibility across seeds.
- **Methodology review:** an ADR documenting how fairness is enforced; ideally a `/code-review ultra` pass before `v1.0`.

---

## 13. Expected Outcomes / Deliverables

- A **pip-installable, documented, tested, CI-green open-source platform** with a polished docs site and README hero animation.
- A **standardized benchmark** comparing PN/APN/OGL/SMG/MPC/RL (+game-theoretic 1v1) across a shared scenario suite, with Monte-Carlo statistics, capture regions, and effort/efficiency analysis — **the differentiating contribution**.
- **Reproducible headline figures** (capture-region heatmaps, robustness curves, effort-vs-accuracy scatter, swarm/salvo animations).
- A **technical write-up** (benchmark report + optionally a short paper/blog) suitable for a portfolio centerpiece.
- A clear **development history**: ADRs, progress log, milestone tags, experiment tracker — demonstrating engineering rigor to reviewers.

---

## 14. Key choices

- **Name:** INTERCEPT. **License:** MIT.
- **RL framework:** Stable-Baselines3 for reliable baselines (PPO/RecurrentPPO), with the readable
 showcase agents kept single-file.
- **Experiment tracking:** in-repo CSV result tables for the benchmark (committed), with optional
 Weights & Biases for RL training curves.
- **Docs host:** GitHub Pages via MkDocs Material.
- **Build cadence:** an early vertical slice (core → PN → benchmark) before broadening across
 paradigms, fidelity, and scale.

