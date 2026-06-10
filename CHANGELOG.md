# Changelog

All notable changes to INTERCEPT are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Coordinated swarm penetration tactics + asset-value layered defense (novel)
- `intercept.adversary.swarm_tactics` — four literature-grounded **saturation tactics** as kinematic
  `Raid` builders: simultaneous **time-on-target**, **decoy screen** (real threats interleaved with
  decoys aimed to miss), **concentrated saturation point**, and **sequential waves**.
- `intercept.multiagent.defense` — the **counter**: impact-point prediction (`predict_closest_approach`)
  → decoy de-prioritization (`threat_value`) → value-prioritized, capacity-aware Hungarian allocation
  (`value_prioritized_assignment`). Plugs into `MultiEngagement` via a new `allocator` hook (now also
  passes `target_names` so allocators can be stateful).
- **Stateful discriminator (the core insight):** a single-snapshot impact predictor *cannot* separate
  a hard-maneuvering real threat (predicted miss swings to 1.5–3 km on a weave) from a decoy — so
  `make_value_allocator` tracks each track's **running minimum predicted miss** (collapses for a real
  inbound, floors for a decoy; classical constant-bearing / track-history reasoning).
- **Result (40 trials/tactic, 5-interceptor magazine, real-threat leakers):** asset-value defense
  **eliminates leakage on the decoy screen (naive 1.70 → 0.00)** and is **tied** on the
  three all-real tactics (TOT 3.0/3.1, concentrated 5.0/5.1, stream 4.0/4.0) — discrimination helps
  only against decoys; undecoyed over-saturation is magazine-bound, not algorithm-bound. ADR-0028;
  `experiments/p35_swarm_penetration.py`; 7 tests in `tests/test_swarm_tactics.py`. Figures
  `gallery/figures/p35_penetration_bars.png` + `p35_swarm_penetration.png` +
  `p35_tactics_gallery.png` (2×2, every tactic), GIFs `gallery/animations/p35_swarm_penetration.gif`
  (decoy screen) + `p35_tot_raid.gif` (time-on-target raid).

### Changed — Visible-threat realism pass + a bigger two-wave swarm raid
- **Physics note made actionable:** at ~Mach 2, a "light" weave (4 g @ 0.25 Hz) displaces the path
  by only ~16 m — invisible at engagement scale. Threat cruises upgraded to *visible* serpentines
  (8 g @ 0.1–0.12 Hz ≈ 150–200 m excursions, well within the realistic suite's 20 g envelope):
  - **Salvo (p24):** ITCG still synchronizes against the weaving inbound — **0.19 s spread vs
    0.57 s natural** (0.01–0.04 s on milder weaves); the maneuver measurably loosens the timing.
  - **Mode-adaptive (p33):** dual regime via `--weave-g` — light 4 g: adaptive is the best law
    (0.62 @ 265k, APN 0.55 @ 426k); visible 8 g serpentine: the IMM *correctly* saturates and
    adaptive defaults to APN-like (0.92 @ 449k) — it spends effort exactly when the threat maneuvers.
  - **Pincer (p34):** weaving variant re-matched (β=0.22): clean 1.00 / 4 g weave **0.45** /
    8 g serpentine **fails at every β (≤0.27)** — a static side-split cannot cover a serpentine
    corridor (a clear envelope limit; dynamic splits are future work). APN reference 1.00 throughout.
- **Swarm showcase (p28) scaled up:** a **two-wave raid — 12 threats × 6 profiles vs 12
  interceptors → 12/12 intercepted, 0 leakers**; hero GIF 8.1 MB (GitHub-safe).

### Changed — Cinematic restyle of every 2-D animation + GitHub-safe hero GIF
- All 2-D animations now match the dark/neon 3-D theme: new `viz.animation.style_dark_2d`, neon
  under-glow trails, white-core/halo entity heads, growing intercept flash (shared `_render`, the
  2-D swarm animator, and the salvo/mode-adaptive/pincer GIFs). Every GIF regenerated.
- `p28_swarm_showcase.gif` (README hero) shrunk 10.5 → 7.4 MB (140 frames) so GitHub renders it
  inline (10 MB limit).

### Added — Pincer coverage guidance: a pair covers both escape branches (novel)
- `intercept.guidance.PincerGuidance` / `pincer_pair` — wraps any base law to approach via a virtual
  aim-point offset to one side (`±β·R·⟂̂_LOS`, tapering to zero inside `r_merge` for a clean
  endgame); the pair splits to cover both turn directions. New `surprise_break` evader
  (unpredictable break side, per-trial injected sign). ADR-0027; 4 tests in `tests/test_pincer.py`.
- **Result (30 g surprise break, 60 paired trials, two threat regimes):** single PN and redundant
  PN pair **0.00 in both** (identical laws ⇒ correlated failures); **pincer PN pair 1.00 on a clean
  cruise** (perfect per-branch coverage — geometry substituting for acceleration sensing) and
  **0.52 under a realistic ~4 g weaving cruise** (the weave erodes the split; APN reference 1.00).
  Split matched per envelope (β=0.20/0.18, disclosed); narrow β window documented. Figure
  `gallery/figures/p34_pincer.png` (grouped bars), GIF `gallery/animations/p34_pincer.gif`.

### Added — Mode-adaptive guidance: the IMM belief arbitrates the law (novel)
- `intercept.guidance.ModeAdaptiveGuidance` — the seeker→IMM maneuver-mode probability blends a
  quiescent law (True PN) with a maneuver law (Augmented PN) via a sharpened weight
  (`w = μ^γ/(μ^γ+(1−μ)^γ)`, γ=3). Estimation-aware law arbitration from existing components.
  ADR-0026; tests in `tests/test_adaptive.py`.
- **Result (weaving cruise → 20 g break, 60 paired trials):** PN 0.52 @ 79k, APN 0.55 @ 426k (its
  feedforward chases the weave), **adaptive 0.62 @ 265k — the best law on this realistic threat at
  38 % less effort than APN**. Operating envelope: a sustained-from-launch maneuver favors always-on
  APN (detection lag). Figure `gallery/figures/p33_mode_adaptive.png` (belief-coloured trail + μ
  timeline), GIF `gallery/animations/p33_mode_adaptive.gif`.

### Added — Training RL directly on estimated observations (ADR-0005 gap closed)
- `InterceptionEnv(sensor=..., estimator_factory=...)`: the policy *observes* a seeker-on-interceptor
  radar→EKF estimate during training; reward/intercept use truth. `experiments/p31_rl_estimated_obs.py`
  + a **clean ablation twin** (same scenario/budget, truth observations). ADR-0024; env test added.
- **Result (weave, 120 trials):** estimate-trained holds **0.99/0.98/0.88** at σ=50/100/200 m where
  the truth-trained twin drops to 0.97/0.90/**0.69** — a 19-point robustness gap from training under
  noise. Note: on a straight target the effect vanishes, and PN-on-estimate stays most noise-robust
  overall. Figure `gallery/figures/p31_rl_estimated_obs.png`.

### Added — The INTERCEPT League: Elo ratings for guidance laws and evaders (novel)
- `intercept.benchmark.league` — Bradley-Terry MLE (Zermelo/MM, additive smoothing) reported on the
  Elo scale, + `elo_expected_score`; 5 unit tests. `experiments/p32_league.py` runs the round-robin
  (8 laws × 6 evaders × 40 paired geometries, realistic speeds): **intercept ⇒ law wins, escape ⇒
  evader wins**, both sides rated on **one ladder**. ADR-0025; literature search found no prior
  guidance-law rating ladder.
- **Leaderboard:** Sliding-mode champion (2214); the game-theoretic anti-LOS **evader** is the #2
  player overall (2019, above OGL/APN); recurrent APN-residual (1681) above the PN family
  (1310–1423); scripted evaders bottom (719). Figure `gallery/figures/p32_league.png`; README gets a
  League section + badges.

### Added — Population self-play (fictitious play vs a pool)
- `InterceptionEnv(opponent_factory=...)` — per-reset opponent sampler (pooled / curriculum
  opponents). `experiments/p30_population_selfplay.py` trains the interceptor against a pool
  {evader gen-0, gen-1} and re-evaluates the cross-table. ADR-0023; test in `tests/test_residual_rl.py`.
- **Result:** fictitious play **fixes the ADR-0020 catastrophic forgetting** — the pool interceptor
  catches the hard gen-0 at **0.78** (vs the latest-opponent gen-2's **0.18**); worst-case min 0.59
  (the residual mildly miscalibrates on the easy gen-1). Engineering lesson: per-episode opponent
  switching gives **bimodal returns that destabilize PPO reward normalization** — `norm_reward=False`
  fixed a collapse (reward −250 → +331). Figure `gallery/figures/p30_population_selfplay.png`.

### Added — MARL cooperative swarm (learned target allocation)
- `intercept.envs.CentralizedSwarmEnv` — a centralized policy emits an N×M allocation preference each
  step; each interceptor takes its top living threat and PN-guides it (learned *coordination*, PN
  does guidance → no from-scratch collapse). `experiments/p29_marl_swarm.py` trains it on an
  under-resourced 3-vs-5 defense and compares it **fairly** (identical seeds, same env) to the
  Hungarian WTA and random allocators. ADR-0022; tests in `tests/test_swarm_env.py`.
- **Result (200 trials, mean leakers of 5):** Random **0.98**, **Learned (MARL) 0.69**, Hungarian
  **0.65** — the learned allocator nearly matches the near-optimal analytic baseline (and far beats
  random), discovering cooperative spread from reward alone; it does not beat the optimum (reported
  as such). Figure `gallery/figures/p29_marl_swarm.png`.

### Added — Diverse-threat swarm-vs-swarm showcase (bigger, cinematic)
- `intercept.adversary.threats` — a realistic 3-D **threat-trajectory library** (`THREAT_PROFILES`):
  cruise-weave, sea-skimming pop-up, lofted-ballistic, terminal-spiral, diving-jink, boost-glide
  (public textbook kinematic shapes).
- **3-D weapon-target assignment:** `intercept_point` / `intercept_time_cost` / `cost_matrix` /
  `kill_probability[_matrix]` / `weapon_target_assignment` are now dimension-generic (`ndim`, default
  2); `MultiEngagement` infers it from `dynamics.control_dim`. (Fixes a latent 2-D-only WTA bug that
  made 3-D swarms mis-assign.)
- `viz.animate_swarm_3d_modern` — cinematic multi-entity 3-D replay (glow trails, per-kill flashes,
  defended-point marker, orbit). `experiments/p28_swarm_showcase.py`: **8 threats across 6 profiles
  vs 10 interceptors → 8/8 intercepted, 0 leakers** at realistic speeds. Figure
  `gallery/figures/p28_swarm_showcase.png` + GIF `gallery/animations/p28_swarm_showcase.gif`.
  ADR-0021; tests in `tests/test_swarm3d.py`.

### Added — Converged self-play (alternating arms race + cross-table)
- `EvaderEnv(pursuer_factory=...)` hook — train the evader against a *learned* interceptor (the
  missing half of self-play). `experiments/p27_converged_selfplay.py` trains evader gen-1 (vs
  interceptor gen-1) and interceptor gen-2 (vs evader gen-1) and tabulates the P(intercept)
  cross-table. ADR-0020; figure `gallery/figures/p27_converged_selfplay.png`.
- **Result:** the race is **non-transitive** — interceptor gen-2 beats evader gen-1 (1.00) but
  *catastrophically forgets* evader gen-0 (0.80 → 0.18), and evader gen-1 became more exploitable
  (True PN 0.07 → 0.91). Motivates population/PSRO methods (next step).

### Changed — Project-wide realistic-speed standardization (no speed cheat) + gallery reorganization
- **Eliminated all interceptor/target speed cheats.** Every engagement now uses realistic comparable
  speeds — interceptor ~Mach 3 (1000 m/s), target ~Mach 2 (700 m/s), only a **~1.45× edge** (ADR-0018).
  Fixed: the salvo (was 6×), `animate_demos` (~2.7×), `p7` swarm (2.5×), `p0` (4.75×), `p1/p3/p4/p6`,
  the 5 base benchmark scenarios (up to 4×), and the `ParametricScenario` default (600/250 → 1000/700).
  The realistic suite and aero experiments (p12–p25) were already ~1.5×.
- **p5 RL** retrained: from-scratch PPO *collapses* at realistic speeds, so it now learns a **residual**
  on a PN baseline — matches the classical laws (head-on/weave 1.00) and **edges them on the crossing
  shot (0.82 vs 0.76)**. `animate_demos` drops the old cheaty-regime RL clip.
- **Gallery reorganized:** `gallery/figures/` (static graphs) and `gallery/animations/` (GIFs +
  interactive HTML). All 26 experiments' save paths and all README/docs image links updated; every
  figure/animation regenerated at realistic speeds.

### Added — Estimation extras (INS platform error, 3-D IMM in the guidance loop)
- `intercept.estimation.INSError`: interceptor self-localization error (per-trial bias + linear
  drift) for the seeker-on-interceptor platform. `EstimatingGuidance(platform_error=...)` hook — the
  seeker measures true relative geometry, the filter places the target using the *believed* (INS-
  corrupted) platform position, so the estimate inherits the platform's nav error.
- Confirmed **3-D IMM in the guidance loop**: `EstimatingGuidance` + `make_cv_ca_imm(ndim=3)` +
  `Radar3D` + `augmented_pn_3d` closes the 3-D sense→estimate→guide loop and intercepts; INS error
  measurably degrades it. Tests in `tests/test_estimation.py`. (Seeker is already interceptor-mounted,
  i.e. moving-platform sensing.) ADR-0019.
- `experiments/p26_estimation_advanced.py` (figure `gallery/figures/p26_estimation_advanced.png`):
  realistic-speed 3-D IMM-in-loop intercept + INS-drift sweep — median miss grows monotonically
  (≈20→54→80→132→245 m at 0/2/5/10/20 m/s drift).

### Added — Two-sided self-play (one arms-race round)
- `InterceptionEnv(opponent=...)` override — train against an arbitrary target controller (e.g.
  `RLEvader`). `experiments/p25_selfplay.py` trains a gen-1 residual-PN interceptor against the frozen
  learned gen-0 evader and evaluates it (held-out) vs True PN / Augmented PN against that evader.
  ADR-0017; test in `tests/test_residual_rl.py`.
- **Result (held-out, 150 trials vs. the frozen gen-0 evader):** True PN **0.07**, Augmented PN
  **0.71**, **gen-1 self-play interceptor 0.80** — training against the learned adversary hardens the
  interceptor past both classical laws. (One detail: this needs the **APN** baseline; a residual on
  the *failing* PN baseline couldn't bootstrap and stayed at 0.05.) Figure `gallery/p25_selfplay.png`.

### Added — Cooperative salvo (impact-time-control guidance)
- `intercept.guidance.ImpactTimeGuidance` / `impact_time_guidance`: biased PN steering to a commanded
  impact time (corrected ITCG — simple `t_go=R/Vc`, consistent away-from-LOS lead-angle bias, small
  gain). ADR-0016; `experiments/p24_salvo.py` (figure + GIF); tests (`tests/test_salvo.py`, 3).
- **Result:** a 4-interceptor battery from different ranges arrives within **0.14 s** of the commanded
  time (vs **1.28 s** under plain PN), all intercepting — a validated synchronized salvo. Figures
  `gallery/p24_salvo.png`, `gallery/p24_salvo.gif`. (Two earlier feedback-ITCG prototypes that didn't
  synchronize were removed rather than shipped; this corrected form is validated.)

### Added — Recurrent APN-residual 3-D RL (full parity in 3-D)
- `apn_baseline_action_3d` + `baseline="pn"|"apn"` on `InterceptionEnv3D`/`RLGuidance3D` (Augmented-PN-3D
  baseline: PN + target-accel feed-forward from `state[6:9]`). `experiments/p23_recurrent_residual_3d.py`
  trains a `RecurrentPPO` (LSTM) residual on the APN-3D baseline. ADR-0014; test in `tests/test_rl_env_3d.py`.
- **Result (held-out, 100 trials):** recurrent APN-residual reaches **1.00 / 1.00 / 1.00** (crossing /
  weave / barrel) at competitive effort — closing the PN-residual MLP's gap (weave 0.91→1.00, barrel
  0.98→1.00), full parity with the 3-D classical laws. Figure `gallery/p23_recurrent_residual_3d.png`.

### Added — Gallery: interactive replays + evader animation
- `gallery/p14_advanced_interactive.html` — interactive Plotly replay of the flagship no-cheat L3
  complex-trajectory engagement (p14 emits it when `.[viz]` is installed).
- `gallery/anim_rl_evader.gif` — the learned adversarial-RL evader escaping True PN (≈3.1 km miss),
  the animated companion to the p22 hardness grid.
- Audited all gallery figures against current code — every experiment's figure is present and
  up-to-date (p17/p19/p20 reflect the latest laws/eval); only p23's awaits its training run.

### Added — Interactive 3-D dashboard (Plotly)
- `intercept.viz.dashboard.interactive_engagement_3d`: a browsable, self-contained **HTML** replay of
  a 3-D engagement (play/pause + time slider, draggable camera, hover read-outs; dark theme) — the
  interactive complement to the Matplotlib views. `p9_3d_demo.py` emits `gallery/p9_interactive_3d.html`
  when `.[viz]` (Plotly) is installed. Guarded import + CI-safe test (`tests/test_dashboard.py`,
  skips without Plotly). Closes the "interactive dashboard" limitation.

### Added — Estimator in the RL loop (sense→estimate→guide for learned guidance)
- `EstimatingGuidance` (which wraps any `Controller`) now composes with `RLGuidance`, so a trained
  RL policy can be fed a noisy-radar→EKF *estimate* of the target — closing the sensing loop for
  learned guidance (ADR-0005 follow-up), no new code. Robustness: a PN-equivalent policy via this
  path matches truth-fed performance with negligible degradation (~52.7→52.8 m in a sample geometry).
  CI-safe test (mock policy, no model file) in `tests/test_estimation.py`.

### Added — Adversarial-RL evader
- `intercept.envs.evader_env.EvaderEnv`: the agent is the **target**, maximizing the interceptor's
  miss against a fixed True-PN pursuer (lateral action, evader-view observation, grow-the-miss
  reward, tunnelling-proof intercept). `intercept.adversary.rl_evader.RLEvader` deploys the trained
  policy as a target `Controller`. ADR-0015; `experiments/p22_adversarial_evader.py`; tests
  (`tests/test_evader.py`, 5). A learned, reactive adversary complementing the scripted and
  game-theoretic (`optimal_evader`) evaders.
- **Result (held-out, 150 trials):** the RL evader defeats the True PN it trained against
  (**1.00 → 0.07** intercept, ~1.2 km miss) but overfits to it — Augmented PN (0.71) and sliding-mode
  (0.78) still catch it, while the analytic anti-LOS `optimal_evader` generalizes (0.00/0.08/0.31).
  Confirms the robust laws hold up even against a learned adversary. Figure
  `gallery/p22_adversarial_evader.png`.

### Added — Documentation site + CI docs build
- `mkdocs.yml`: an mkdocs-material site (dark/light, math, nav over the index, results, limitations,
  benchmark methodology, six algorithm notes, all 14 ADRs, and the progress log). Build locally with
  `pip install -e ".[docs]" && mkdocs serve`.
- `.github/workflows/docs.yml`: GitHub Pages deploy on push to main; CI (`ci.yml`) gains a `docs`
  job that verifies `mkdocs build`. `site/` git-ignored.

### Added — Global kill-probability weapon-target assignment
- `intercept.multiagent.assignment`: `kill_probability` (geometry heuristic `p_max·exp(−t/τ)`),
  `kill_probability_matrix`, `expected_leakers`, and a new `weapon_target_assignment(...,
  objective="kill_prob")` that maximizes the global kill probability and routes **surplus**
  interceptors to the threats most likely to leak (shoot-look-shoot), minimizing expected survivors.
  Default `objective="time"` is unchanged. Tests (`tests/test_multiagent.py`, +4); ADR-0009 updated.
- **Finding:** with surplus interceptors, the kill-prob objective cuts **expected leakers** (e.g.
  0.58 → 0.47 in a 3-vs-2 near/far case) by covering the leak-prone far threat instead of piling on
  the already-safe near one. *(The 1-to-1 core coincides with the time objective by construction,
  since `P = p_max·exp(−t/τ)`; the gain is in surplus allocation.)*

### Added — Gain-sensitivity sweep
- `experiments/p21_gain_sensitivity.py`: sweeps the navigation constant N ∈ {2..7} for True PN and
  Augmented PN across realistic (L2 aero) weave/jink scenarios, reporting P(intercept) (Wilson CI)
  and mean control effort vs. N. Closes the "best-effort gains" benchmark gap. Test in
  `tests/test_benchmark.py`. Figure `gallery/p21_gain_sensitivity.png`, CSV
  `results/p21_gain_sensitivity.csv`.
- **Finding:** on a steady weave all N intercept; on the *unpredictable* jink True PN climbs with N
  (0.29 → 1.00 from N=2→6, faster response), while Augmented PN peaks near **N≈3 (0.96)** then gently
  declines (the accel feed-forward + high N over-responds) — and effort rises monotonically with N.
  The benchmark's N=4 default sits on the robust plateau.

### Added — 3-D RL guidance (the last paradigm lifted to 3-D)
- `intercept.envs.interception_env_3d`: `InterceptionEnv3D` (Gymnasium) over `ParametricScenario3D`
  — **2-DOF lateral action** in the ⟂-velocity plane, 3-D observation (`build_observation_3d`/`_rich`),
  potential-based ZEM reward, 3-D gravity feed-forward, tunnelling-proof intercept. `RLGuidance3D`
  deploy wrapper (mirrors the parameterization; recurrent-capable). ADR-0014.
- **Residual-PN-3D** (`action_mode="residual_pn"`, `pn_baseline_action_3d`): from-scratch 3-D PPO
  *collapses* (constant saturated action, 0/100 deploy — the 2-D failure, reproduced and diagnosed),
  so the policy learns a bounded correction on a True-PN-3D baseline (a zero action = PN-3D), exactly
  as in ADR-0011.
- `experiments/p20_train_rl_3d.py` (PPO + VecNormalize over a 3-D curriculum, residual mode; held-out
  eval vs True PN-3D / Augmented PN-3D / Optimal-3D / Sliding-mode-3D). Tests (`tests/test_rl_env_3d.py`,
  9). **Every guidance paradigm now runs in 3-D.**
- **Result (held-out, 100 trials/scenario):** from-scratch 3-D PPO collapses (**0.00/0.00/0.00** on
  crossing/weave/barrel, constant saturated action — diagnosed). Residual-PN-3D resolves it:
  **1.00 / 0.91 / 0.98** — competent across all three, at parity with the classical 3-D laws (which
  are 1.00 here, so the learned correction's value is marginal where PN-3D already suffices). Mirrors
  the 2-D residual result. Figure `gallery/p20_rl_3d.png`; CSV `results/p20_rl_3d.csv`.

### Added — 3-D estimating-guidance + scope/limitations doc
- `EstimatingGuidance` and the `Estimator` accessors (`position`/`velocity`/`target_state`) are now
  **dimension-generic**, so the **sense→estimate→guide loop closes in 3-D** (Radar3D + NCA-UKF → True
  PN-3D intercepts a noisy 3-D engagement). Test in `tests/test_estimation.py`. 2-D path unchanged.
- `docs/limitations.md`: a consolidated scope-vs-open-follow-ups account (deliberate
  point-mass/sim-only boundaries vs. genuine gaps). Linked from README and the docs index.
- An impact-time (salvo) guidance prototype was attempted but did not converge robustly and was
  **removed rather than shipped unvalidated**; cooperative/salvo remains a documented follow-up.

### Added — 3-D optimal, sliding-mode & MPC guidance
- `intercept.guidance`: `OptimalGuidance3D`/`optimal_guidance_3d` (3-D energy-optimal ZEM law,
  optional target-accel augmentation) and `SlidingModeGuidance3D`/`sliding_mode_3d` (LOS-rate-vector
  sliding surface + boundary layer). The 3-D benchmark (p17) and capstone (p19) now compare the full
  set — True PN-3D, Augmented PN-3D, Optimal-3D, Sliding-mode-3D — not just the PN family.
- `intercept.guidance.mpc.MPCGuidance3D`: 3-D receding-horizon NMPC (CasADi/IPOPT; 6-state dynamics,
  3-D accel constraint, optional 3-D impact-direction objective). Intercepts a 3-D barrel-roll.
- On the 3-D barrel-roll that defeats True PN-3D (0.00), all of APN-3D / OGL-3D / SMG-3D recover to
  1.00; the terminal-spiral ladder is True PN 0.19 → APN 0.82 → OGL 0.98 → SMG 1.00. Tests
  (`tests/test_3d.py` +2, `tests/test_guidance_optimal.py` +2). Only RL guidance remains 2-D.

### Added — Capstone benchmark + statistical significance
- `intercept.benchmark.metrics`: `paired_bootstrap` + `compare_intercept` + `PairedComparison` —
  paired-bootstrap CI and two-sided p-value for pairwise law comparisons (the fairness invariant
  makes trials paired). Methodology doc updated; tests (`tests/test_benchmark.py`, +2).
- `experiments/p19_capstone_benchmark.py`: a single fair benchmark across **paradigm × fidelity
  (L0→L3) × dimension (2-D/3-D)** as a P(intercept) heatmap with APN-vs-PN significance per row.
  Findings: 2-D L2 jink True PN 0.83 → APN 0.92 (significant, p=0.009) → Sliding-mode 1.00; 3-D L2
  barrel 0.00 → 1.00 (p<0.001). Figure `gallery/p19_capstone_benchmark.png`.

### Added — 3-D estimation (dimension-generic filters + 3-D radar)
- `ncv_model`/`nca_model` take `ndim` (default 2; `ndim=3` → 9-D state); EKF/UKF **and the IMM** are
  now dimension-generic (infer state size from `x0`, position size from `sensor.pos_dim`;
  `make_cv_ca_imm(..., ndim=3)`); `Sensor` gains `pos_dim`. New `sensors/radar3d.py` `Radar3D`
  (range/azimuth/elevation, analytic Jacobian, angle-safe residuals, `invert`). ADR-0013;
  `experiments/p18_estimation_3d.py`; tests (`tests/test_estimation.py`, +4). 2-D behavior unchanged.
- **Result:** on a 3-D barrel-rolling target, the **CV/CA IMM tracks best (~42 m)** vs NCA UKF
  **~52 m** and NCV EKF **~71 m** (at the raw-measurement floor) — the 3-D analogue of the
  IMM-vs-EKF result. Figure `gallery/p18_estimation_3d.png`.

### Added — 3-D Monte-Carlo benchmark
- `intercept.benchmark.scenario3d.ParametricScenario3D` (+ `make_maneuver_3d`, `load_scenario_3d`):
  samples 3-D geometry/maneuvers and emits the same `EngagementSpec`, so the dimension-agnostic
  runner/metrics extend to 3-D unchanged (ADR-0012). `experiments/p17_benchmark_3d.py`; tests in
  `tests/test_3d.py`.
- **Result (200 trials/scenario, L2 aero):** the 3-D realism analogue — on a sustained barrel-roll
  True PN-3D drops to **0.00** while Augmented PN-3D holds **1.00**; on an intensifying terminal
  spiral True PN **0.18** vs APN **0.82** (~1.8× effort). Figure `gallery/p17_benchmark_3d.png`.

### Added — Portfolio polish
- Rewrote `README.md` from "early development / planned" to an accurate, results-forward overview:
  what's built (9 laws / 6 paradigms, L0→L3, 2-D/3-D, multi-agent), headline numbers, embedded hero
  GIFs + benchmark figures, quickstart.
- Added `docs/results.md` — a consolidated results digest (the thesis with numbers + the figure
  behind each finding). Linked from README and the docs index.
- Added `.gitattributes` (LF normalization + binary asset rules) for clean cross-platform git.

### Added — Residual RL guidance (novel hybrid; resolves the deferred realistic-RL failure)
- `InterceptionEnv` + `RLGuidance` gain `action_mode="residual_pn"`: the policy outputs a bounded
  correction added to a pure-PN baseline (`pn_baseline_scalar`), so a zero action is competent PN —
  eliminating the from-scratch policy collapse on the lagged/gravity plant. `residual_scale`, `pn_N`.
- `experiments/p15_residual_rl.py` (train + held-out eval vs True PN / Augmented PN / Sliding-mode,
  and the failed from-scratch PPO for contrast). ADR-0011; tests (`tests/test_residual_rl.py`, 5:
  baseline matches `pure_pn`; zero residual intercepts where a constant-saturated action misses).
- **Result (held-out, 100 trials/scenario, realistic L2 aero):** residual-RL **resolves the
  collapse** — PN-residual MLP P(intercept) **1.00 / 1.00 / 0.68** on crossing / 18 g weave / 22 g
  jink vs **from-scratch PPO 0.01 / 0.01 / 0.00**. Figure `gallery/p15_residual_rl.png`.

### Added — Recurrent, APN-baseline residual RL (a learned win over the PN family)
- `apn_baseline_scalar` + `baseline="pn"|"apn"` on `InterceptionEnv`/`RLGuidance` (APN baseline
  feed-forwards the target's achieved lateral accel from its state). `RLGuidance(recurrent=True)`
  threads/resets LSTM hidden state across an engagement, enabling `sb3_contrib.RecurrentPPO` policies.
- `experiments/p16_recurrent_residual.py` (RecurrentPPO `MlpLstmPolicy` + APN-residual). Tests
  extended (`tests/test_residual_rl.py`, 7: APN feed-forward formula; recurrent state threading/reset).
- **Result (held-out):** the recurrent APN-residual reaches **1.00 / 1.00 / 0.95** (crossing / weave /
  jink) — on the unpredictable jink it **beats True PN (0.81) and Augmented PN (0.93)** at lower
  aggregate effort than APN, trailing only sliding-mode (1.00). Ablation: APN baseline + LSTM memory
  lift the jink 0.68 → 0.95. A genuine learned win over the PN family on the hardest realistic case.
  Figure `gallery/p16_recurrent_residual.png`; CSV `results/p16_recurrent_residual.csv`.

### Added — L3 in the benchmark
- `ParametricScenario` gains `model="realistic"` (L3 `RealisticMissile2D`: boosting interceptor vs
  sustaining threat, q/lift-limited turning), so the Monte-Carlo benchmark and the RL env run on the
  realistic plant. Test in `tests/test_realistic.py`. (3-D Monte-Carlo benchmark + 3-D estimation
  remain documented follow-ups — the scenario class is currently 2-D.)

### Added — Modern 3-D animation + advanced complex-trajectory evasion (L3, speed parity)
- `intercept.viz.threed`: `animate_engagement_3d_modern` + `plot_engagement_3d_modern` — dark theme,
  neon glow trails, growing intercept flash, orbiting camera. `p9_3d_demo.py` also emits the modern
  frame/animation.
- `intercept.adversary.evasive3d`: `serpentine3d` (tilted 3-D S-weave), `terminal_spiral`
  (closed-loop corkscrew that intensifies as the pursuer closes — models maneuvering-reentry /
  sea-skimmer terminal evasion), `combine` (sum maneuvers).
- `experiments/p14_advanced_evasion.py`: both missiles `RealisticMissile3D` (L3); threat flies
  lofted-descent + serpentine + intensifying terminal spiral; APN interceptor. Emits the modern 3-D
  frame/animation, a research-grade analysis panel (top-down serpentine, altitude loft, closing
  range, achieved-g vs physics turn limit), and a robustness Monte-Carlo. Tests
  (`tests/test_evasive3d.py`, 5); ADR-0010; extended `docs/algorithms/realistic-engagements.md`.
- **Realistic speed parity (no speed cheat):** interceptor launches ~Mach 1.2, boosts to ~Mach 3,
  coasts to ~Mach 2.6 at merge; the ~Mach 3 threat bleeds to ~Mach 2 under hard maneuvering — only a
  ~37 % closing edge (was +226 %). Shrinking the interceptor motor further makes it miss
  (near-minimum-energy intercept). Showcase: 12.8 m miss; robustness 60/60 intercept (P=1.00).

### Added — P7 Multi-agent / swarm defense
- `intercept.multiagent`: Hungarian `weapon_target_assignment` (+ time-to-intercept `cost_matrix`)
  and `MultiEngagement` (N-vs-M with live re-assignment, per-step intercept detection, kill logging).
- `intercept.viz`: `plot_swarm`, `animate_swarm`. `experiments/p7_swarm_defense.py`; ADR-0009;
  tests (`tests/test_multiagent.py`, 6). Demo: 8 vs 8 → 8/8 intercepted.

### Added — RL on realistic dynamics (a documented negative result)
- `InterceptionEnv` now uses the scenario's plant (trains on L2 aero); added `build_observation_rich`
  (LOS rate, closing, range, lag state) via `obs_mode="rich"`; `RLGuidance(obs_mode=...)`.
- `experiments/p12_train_rl_realistic.py`, `p13_realistic_rl_vs_classical.py`.
- Added gravity feed-forward + rich observation + `RewardConfig(mode="pn_shaping")` to support RL on
  the realistic plant (all also useful generally).
- **Finding (DEFERRED for a dedicated attempt):** feed-forward PPO did not transfer to the realistic
  plant (~0.01–0.02 intercept vs classical 0.81–1.00). Diagnosis: the policy *collapses to a
  constant saturated action ignoring the observation* on the lagged (τ=0.2 s) dynamics, even though a
  hand-coded `action ∝ LOS-rate` scores 40/40 in the same env. Deferred to a future attempt
  (recurrent/LSTM policy, or imitation/residual warm-start from PN). Realistic robustness meanwhile
  is delivered by APN/OGL/sliding-mode + IMM-in-the-loop, with the negative RL result reported in full.

### Added — L3 realism (aero-propulsive physics)
- `intercept.core.atmosphere`: ISA standard atmosphere + transonic `Cd0(Mach)`.
- `intercept.core.realistic.RealisticMissile2D/3D`: boost-sustain-coast thrust + mass burn-off,
  Mach + induced drag, **lift/dynamic-pressure-limited turning** (no prescribed g), autopilot lag.
- `experiments/p10_realistic_demo.py` (Mach/available-g time history), `p11_realistic_estimation.py`
  (noisy radar + IMM/EKF in the loop). Tests (`tests/test_realistic.py`, 9); ADR-0008.
- Validated: interceptor boosts to Mach 2.6, available g rises 2→45 g from dynamic pressure,
  intercepts a weaving high-altitude target (17 m); noisy-seeker+IMM holds 0.97 P(intercept).

### Added — 3-D extension
- `intercept.core.dynamics3d`: `PointMass3D`, `AeroMissile3D` (gravity/drag/g-limit/lag in 3-D).
- `intercept.core.frames3d`: 3-D LOS geometry incl. the LOS angular-velocity vector `Ω`.
- `intercept.guidance.pn3d`: `ProportionalNavigation3D` (realizable true PN) + `AugmentedPN3D`.
- `intercept.adversary.evasive3d`: helical `barrel_roll`, `weave3d`.
- `intercept.viz.threed`: 3-D trajectory plot + rotating animated GIF. `experiments/p9_3d_demo.py`.
- Tests (`tests/test_3d.py`, 8); ADR-0007. The dimension-agnostic engagement core/metrics needed
  no changes.

### Added — Realism upgrade (L2 aero fidelity)
- `intercept.core.aero.AeroMissile2D`: planar 3-DOF plant — gravity, parasitic + induced drag
  (hard turns bleed speed), hard g-limit, first-order autopilot lag; `[x,y,vx,vy,ax,ay]` state.
- `intercept.adversary.evasive`: `hard_turn`, `random_telegraph` (seeded jink), `reactive_break`
  (closed-loop max-g break away from the interceptor).
- `ParametricScenario`: `model: aero` + aero params + g-specified maneuvers; spec carries plant
  objects. `scenarios/realistic/` suite (R1–R4). `experiments/p8_realistic_benchmark.py`,
  `experiments/animate_realistic.py`. ADR-0006, `docs/algorithms/realistic-engagements.md`.
- Result: on realistic high-g/unpredictable/reactive targets True PN drops to 0.21–0.56 while
  Augmented PN / Optimal / Sliding-mode recover to 0.79–1.00 — intelligence beats speed.

### Added — Animated replays
- `intercept.viz.animation`: `animate_engagement` (GIF/MP4 replay with fading trails + intercept
  burst), `animate_comparison` (overlay several laws vs a shared target), `filmstrip_engagement`
  (static snapshot montage). `experiments/animate_demos.py` generates showcase GIFs in `gallery/`
  (PN vs weaving, PN vs pure-pursuit, RL intercept, PN vs optimal evader); `--show` for live windows.

### Added — P6 Game theory & adversaries (`v0.7-game`)
- `intercept.guidance.game`: `apollonius_circle`, `intercept_point`, `ApolloniusGuidance`
  (geometric constant-bearing pursuer).
- `intercept.adversary.optimal_evader`: game-theoretic optimal evasion (anti-LOS flight).
- `intercept.viz.plot_apollonius`: Apollonius-circle / dominance-region diagram.
- `experiments/p6_game_theory.py`; tests (`tests/test_game.py`, 9); `docs/algorithms/game-theory.md`.
- Result: vs True PN, the optimal evader maximizes capture time (≈8.7 s vs ≈3.6 s) while
  scripted weave/jink target miss distance — distinct notions of adversary difficulty.

### Added — P5 RL centerpiece (`v0.6-rl`)
- `intercept.envs`: Gymnasium `InterceptionEnv` over the engagement core; `build_observation`,
  `lateral_acceleration` (1-D lateral action), potential-based ZEM `RewardConfig`, mixed-curriculum.
- `intercept.guidance.rl_policy.RLGuidance`: deploys a trained policy as a `Controller` (with frozen
  `VecNormalize` obs stats via `obs_norm`).
- `experiments/p5_train_rl.py` (PPO + VecNormalize + curriculum, saves model + normalizer),
  `experiments/p5_rl_vs_classical.py` (held-out comparison + effort table + trajectory overlay).
- Tests (`tests/test_rl_env.py`, 10); ADR-0005; `docs/algorithms/rl-guidance.md`.
- **Result (held-out):** PPO intercepts 64–95% (best on crossing) vs classical 100%, at ~2× effort —
  a clean, reproducible learned-vs-classical comparison.

### Fixed
- Intercept "tunnelling": fast closing could skip the kill radius between discrete steps. Added
  `core.frames.segment_min_distance` (closest approach within a step), now used by both the RL env
  and the `Engagement` core.
- `viz.plot_pintercept_bars`: clamp tiny negative Wilson error-bar values (matplotlib rejects yerr<0).

### Added — P4 Optimal & MPC guidance (`v0.5-mpc`)
- `intercept.guidance`: `OptimalGuidance` (LQ/ZEM energy-optimal, `optimal_guidance`),
  `SlidingModeGuidance` (`sliding_mode`), and `MPCGuidance` (CasADi/IPOPT NMPC with acceleration
  limits, optional impact-angle objective, event-triggered replanning; requires `.[mpc]`).
- `experiments/p4_optimal_mpc.py`; tests (`tests/test_guidance_optimal.py`, 11);
  `docs/algorithms/optimal-mpc-guidance.md`.
- Demo: OGL most accurate, PN most efficient, SMG robust, NMPC most flexible (impact-angle) but
  costliest — the trade-off space the benchmark exposes.

### Added — P3 Sensors & estimation (`v0.4-estimation`)
- `intercept.sensors`: `Sensor` base + `wrap_to_pi`, `Radar` (range+bearing), `IRSeeker` (bearing-only).
- `intercept.estimation`: NCV/NCA motion models, `EKF` (Joseph form), `UKF` (sigma points), `IMM`
  (+`make_cv_ca_imm`), `Estimator` base with NEES.
- `intercept.guidance.EstimatingGuidance`: sense→estimate→guide bridge (guidance laws unchanged).
- `experiments/p3_estimation.py`; tests (`tests/test_estimation.py`, 12); ADR-0004 and
  `docs/algorithms/estimation-tracking.md`.
- Figures: IMM-vs-EKF maneuver tracking (NCV diverges, IMM holds ~9 m); estimation-coupled study
  (P(intercept) flat to σ_r≈50 m, collapses by 200 m).

### Added — P2 Benchmark harness (`v0.3-bench`)
- `intercept.benchmark`: `ParametricScenario` + YAML loader (`load_scenario`/`load_suite`),
  `EngagementSpec`; `wilson_interval` + `summarize`/`MetricSummary`; seeded `run_montecarlo`
  (fairness invariant via `SeedSequence.spawn`); `run_benchmark`/`format_table`/`write_csv`;
  `compute_capture_region`.
- `intercept.viz`: `plot_pintercept_bars` (Wilson-CI bars) and `plot_capture_region` (miss heatmap).
- `scenarios/`: suite S1–S5 (head-on, crossing, tail-chase, 9 g weave, high-offset crossing).
- `experiments/p2_benchmark.py`; tests (`tests/test_benchmark.py`, 9); ADR-0003 and
  `docs/benchmarks/methodology.md`.
- Full run (4 laws × 5 scenarios × 200 trials): True PN ≡ ZEM PN; Pure PN marginally higher capture
  at the low-authority boundary (CIs overlap); APN matches PN capture on the weave at ~2.5× effort.

### Changed
- Control-effort metric now integrates the **saturated** (applied) acceleration via a new
  `Dynamics.saturate` hook, instead of the raw command (which diverged as range → 0).

### Added — P1 Proportional Navigation baseline (`v0.2-pn`)
- `intercept.guidance`: `GuidanceLaw` base + `ProportionalNavigation` (True/Pure/ZEM variants,
  `true_pn`/`pure_pn`/`zem_pn` factories) and `AugmentedPN` (finite-difference target-accel
  feedforward).
- `intercept.adversary.scripted`: `straight`, `weave`, `step_maneuver`, `bang_bang` evaders.
- `intercept.viz.compare_engagements_2d`: multi-law overlay comparison plot.
- `experiments/p1_guidance_comparison.py`: PN-vs-pure-pursuit and PN-vs-APN figures.
- Tests (`tests/test_guidance_pn.py`, 8) and `docs/algorithms/proportional-navigation.md`.
- Result: True PN intercepts a low-authority crossing target (8.2 m) where pure pursuit misses
  (102.9 m); APN reduces terminal miss vs. a maneuvering target (0.94 m → 0.75 m).

### Added — P0 Scaffold (`v0.1-scaffold`)
- Project scaffold: `pyproject.toml`, MIT `LICENSE`, `README.md`, `.gitignore`, CI workflow, `CITATION.cff`.
- Simulation core (`intercept.core`): `Dynamics` interface + `PointMass2D`, `RK4` integrator,
  LOS/engagement geometry helpers (`frames`), `Entity` model, and an `Engagement` simulation
  loop with intercept / miss / timeout / ground termination and full trajectory logging.
- Plug-in subpackage stubs for `sensors`, `estimation`, `guidance`, `multiagent`, `adversary`,
  `envs`, `benchmark`, `viz` (interfaces to be filled per roadmap).
- `intercept.viz.engagement2d` trajectory/closest-approach plotting.
- `experiments/p0_first_engagement.py` — first end-to-end engagement producing a trajectory figure.
- Unit tests for dynamics, integrator accuracy, and engagement geometry/termination.
- Docs: site index, ADR-0001 (Python-primary stack), ADR-0002 (point-mass-first fidelity),
  development progress log.
