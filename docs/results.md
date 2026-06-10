# INTERCEPT — Results summary

A consolidated digest of the project's headline findings, with the figure and experiment behind
each. Every result is reproducible: experiments are seeded and write to `gallery/` (figures) and
`results/` (CSVs). Numbers below are from the development log ([progress](progress/PROGRESS.md)) and
the per-topic algorithm notes ([docs/algorithms/](algorithms/)).

## The thesis

Cross-paradigm guidance comparisons are normally scattered across studies on *different* geometries
and metrics, reporting only miss distance. INTERCEPT puts PN/APN, optimal/geometric, MPC, RL, and
game-theoretic guidance on **identical dynamics, a shared scenario suite, shared metrics, and seeded
Monte-Carlo**, behind one controller contract that guarantees no method gets a physics advantage.
The results below are what that fair comparison reveals.

---

## 1 · Proportional Navigation baseline

True PN intercepts a low-authority crossing target at **8.2 m** where pure pursuit misses by
**102.9 m** — the textbook lead-vs-lag result, on this testbed. Augmented PN reduces terminal miss
against a maneuvering target (**0.94 m → 0.75 m**).
Figures: `gallery/figures/p1_pn_vs_pursuit.png`, `gallery/figures/p1_pn_vs_apn_maneuvering.png`.

## 2 · The benchmark — the centerpiece

4 PN-family laws × 5 scenarios × 200 seeded trials, Wilson-CI metrics, capture-region sweeps. True
PN ≡ ZEM PN exactly; Pure PN is marginally higher at the low-authority boundary (CIs overlap); APN
matches PN capture on the weave at ~2.5× control effort. The control-effort metric integrates the
**saturated (applied)** acceleration, fixing a divergence as range → 0.
Figures: `gallery/figures/p2_pintercept_by_scenario.png`, `gallery/figures/p2_capture_region_truepn_S5.png`;
CSV: `results/p2_benchmark.csv`.

## 3 · Sensing & estimation

With a sense→estimate→guide loop closed, an **IMM** filter holds **~9 m** tracking error through a
target turn where a single-model EKF (constant-velocity) diverges to **~350 m**. Interception is
flat to a range-noise σ ≈ 50 m and collapses by ~200 m.
Figures: `gallery/figures/p3_imm_tracking.png`, `gallery/figures/p3_guidance_vs_noise.png`.

## 4 · Optimal & MPC guidance

The trade-off space the benchmark exposes, vs a weaving target: **OGL** most accurate (miss 0.73 m),
**PN** cheapest (effort ~9.1 k), **sliding-mode** robust (4.22 m), **NMPC** most flexible
(impact-angle constraint) but costliest (effort ~27 k). Impact-angle NMPC hits ±15° accurately
(~9 m) and exposes a genuine miss/angle trade-off at ±45°.
Figures: `gallery/figures/p4_law_comparison.png`, `gallery/figures/p4_impact_angle.png`.

## 5 · Reinforcement learning vs classical

Held out: PPO (VecNormalize + curriculum, potential-based ZEM reward, 1-D lateral action) intercepts
**0.72 / 0.95 / 0.64** on head-on / crossing / weaving vs classical **1.00**, at ~2× control effort.
The value is the clean, reproducible comparison: learned guidance is competitive, not superior, with its control-effort cost reported alongside.
Figures: `gallery/figures/p5_rl_vs_classical_pintercept.png`, `gallery/figures/p5_rl_vs_pn_trajectory.png`;
CSV: `results/p5_rl_vs_classical.csv`.

## 6 · Game theory & adversaries

The "hardest adversary" depends on the metric. A game-theoretic optimal (anti-LOS) evader maximizes
**capture time** (≈8.7 s vs ≈3.6 s for straight/weave/jink) while scripted weave/jink maximize
**miss distance** — two distinct notions of difficulty, both in the suite.
Figures: `gallery/figures/p6_apollonius.png`, `gallery/figures/p6_evader_robustness.png`.

## 7 · Realism: intelligence beats speed (L2)

On realistic evasive engagements, simple guidance fails where smart guidance recovers:

| Scenario | True PN | Augmented PN | Optimal (OGL) | Sliding-mode |
|---|---|---|---|---|
| R1 supersonic crossing | 1.00 | 1.00 | 1.00 | 1.00 |
| R2 weave 20 g | 1.00 | 1.00 | 1.00 | 1.00 |
| **R3 telegraph jink** | **0.56** | 0.79 | 0.83 | **1.00** |
| **R4 reactive break** | **0.21** | 1.00 | 1.00 | 1.00 |

An unpredictable jink halves plain PN; a reactive last-ditch break collapses it to 0.21 — while
prediction (APN), optimality (OGL), and robustness (sliding-mode) recover to 0.79–1.00.
Figure: `gallery/figures/p8_realistic_benchmark.png`.

## 8 · L3 aero-propulsive realism

`RealisticMissile2D/3D`: ISA atmosphere, boost–sustain–coast thrust + propellant burn-off, Mach +
induced drag, and **lift / dynamic-pressure-limited turning** — g-capability *emerges from physics*,
not a prescribed limit. Validated: the interceptor boosts to ~Mach 2.6, available g rises 2 → 45 g
from dynamic pressure, and it intercepts a weaving high-altitude target at **17 m**; with a noisy
seeker + IMM in the loop it holds **0.97** P(intercept).
Figures: `gallery/figures/p10_realistic_engagement.png`, `gallery/figures/p11_realistic_estimation.png`.

## 9 · Multi-agent area defense

8 interceptors vs an 8-threat fan, Hungarian weapon-target assignment on a predicted
time-to-intercept cost matrix with live re-assignment → **8/8 intercepted, 0 leakers**.
Figures: `gallery/figures/p7_swarm_defense.png`, `gallery/animations/p7_swarm_defense.gif`.

## 10 · Advanced complex-trajectory evasion (L3, speed parity)

Both missiles on the full L3 plant. A fast ~Mach 3 threat flies a lofted descent + tilted serpentine
+ an **intensifying closed-loop terminal spiral**. Speeds are realistic and comparable — the
interceptor launches ~Mach 1.2, boosts to ~Mach 3, then coasts to ~Mach 2.6 at the merge, a
**+37 % closing edge** (a propulsion sweep confirms a smaller motor → it *misses*: a near-minimum-energy intercept, not a thrust advantage). The threat pulls up to 17 g, visibly **clipped by its physics turn limit**.

- Showcase: **INTERCEPT, miss 12.8 m, t = 11.5 s.**
- Robustness (60 randomized trials over geometry + every maneuver parameter):
  **P(intercept) = 1.00 (95 % CI [0.94, 1.00]),** median miss 13.7 m.

Why the 3-D view looks near-straight: at supersonic speed a real "spiral" is a long *thin* helix, so
maneuvering is shown via projections + an achieved-g-vs-turn-limit time history (the faithful way),
not a faked corkscrew. Figures: `gallery/figures/p14_advanced_analysis.png`,
`gallery/figures/p14_advanced_modern.png` / `.gif`. See [ADR-0010](adr/0010-realistic-speed-parity-no-cheat.md).

## 11 · Residual RL guidance — resolving the realistic-plant RL failure

From-scratch PPO collapses on the realistic (L2 aero) plant (constant saturated action, ~0–2 %
intercept). **Residual policy learning** — the policy outputs a bounded *correction* on a PN/APN
baseline ([ADR-0011](adr/0011-residual-rl-guidance.md)) — fixes it: a zero action is already PN, so
there is no collapse. Held-out (100 trials/scenario), P(intercept):

| Scenario | Recurrent APN-residual | Residual-PN (MLP) | True PN | Augmented PN | Sliding-mode | From-scratch PPO |
|---|---|---|---|---|---|---|
| crossing | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| weave 18 g | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| jink 22 g | **0.95** | 0.68 | 0.81 | 0.93 | 1.00 | 0.00 |

Two takeaways. (1) The residual parameterization **resolves the from-scratch collapse** — the learned
policy runs at PN-class competence (1.00/1.00) where from-scratch RL is ~0. (2) With an **APN baseline
+ a recurrent (LSTM) policy**, the learned law reaches **0.95 on the unpredictable jink — beating True
PN (0.81) and Augmented PN (0.93)** at lower aggregate effort than APN, trailing only sliding-mode
(1.00). Ablation: the upgrades lift the jink 0.68 → 0.95. A genuine learned win over the PN family on
the realistic plant's hardest case, with sliding-mode still leading. Figures:
`gallery/figures/p15_residual_rl.png`, `gallery/figures/p16_recurrent_residual.png`; CSVs in `results/`.

---

## 12 · 3-D benchmark — intelligence beats speed, in 3-D

The benchmark extends to 3-D ([ADR-0012](adr/0012-three-dimensional-benchmark.md)) via
`ParametricScenario3D` (same `EngagementSpec`, unchanged runner/metrics), comparing **True PN-3D,
Augmented PN-3D, Optimal (OGL-3D), and Sliding-mode (SMG-3D)**. On a graded 3-D suite, a sustained
**barrel-roll defeats True PN-3D (0.00)** while the smarter laws hold **1.00**; the intensifying
**terminal spiral** gives a clean robustness ladder — True PN **0.19** → Augmented PN **0.82** →
Optimal **0.98** → Sliding-mode **1.00** — the 3-D analogue of §7. Figure
`gallery/figures/p17_benchmark_3d.png`.

## 13 · 3-D estimation

Dimension-generic motion models + EKF/UKF/**IMM** and a `Radar3D` (range/az/el)
([ADR-0013](adr/0013-three-dimensional-estimation.md)). Tracking a 3-D barrel-rolling target, the
**CV/CA IMM tracks best (~42 m)** vs the NCA UKF **~52 m** and the NCV EKF **~71 m** (stuck at the
raw-measurement floor) — the 3-D analogue of §3. Figure `gallery/figures/p18_estimation_3d.png`.

## 14 · Capstone — one fair benchmark, with significance

`experiments/p19_capstone_benchmark.py` evaluates the laws across **paradigm × fidelity (L0→L3) ×
dimension** on a maneuvering target, with **paired-bootstrap significance** (the fairness invariant
makes trials paired — `paired_bootstrap`). Highlights (150 trials/cell): 2-D L2 jink True PN **0.83**
→ Augmented PN **0.92** (significant, p=0.009) → Sliding-mode **1.00**; 3-D L2 barrel **0.00 → 1.00**
(p<0.001). Where the interceptor is idealized (L0) or very capable (L3) the difference is
*not* significant, and reported as such. Figure `gallery/figures/p19_capstone_benchmark.png`.

## 15 · 3-D RL guidance (every paradigm now in 3-D)

The RL centerpiece lifted to 3-D ([ADR-0014](adr/0014-three-dimensional-rl.md)): `InterceptionEnv3D`
(2-DOF lateral action, 3-D observation) + `RLGuidance3D`. From-scratch 3-D PPO **collapses** exactly
as in 2-D (held out **0.00/0.00/0.00**, constant saturated action); **residual-PN-3D** resolves it
(**1.00 / 0.91 / 0.98** crossing/weave/barrel), and a **recurrent (LSTM) APN-residual** (P23) closes
the gap to **1.00 / 1.00 / 1.00** at competitive effort — full parity with the 3-D classical laws
(ablation: APN baseline + memory lift weave 0.91→1.00, barrel 0.98→1.00).
So **every guidance paradigm — PN/APN, Optimal, Sliding-mode, MPC, RL — now runs in 3-D**, alongside
EKF/UKF/IMM estimation and a 3-D sense→estimate→guide loop. Figures `gallery/figures/p20_rl_3d.png`,
`gallery/figures/p23_recurrent_residual_3d.png`.

## 16 · Adversarial-RL evader (a learned adversary)

`EvaderEnv` + `RLEvader` ([ADR-0015](adr/0015-adversarial-rl-evader.md)): the agent is the *target*,
trained to maximize the interceptor's miss against a True-PN pursuer. Held-out (150 paired
geometries), P(intercept) by the interceptor:

| Evader | True PN | Augmented PN | Sliding-mode |
|---|---|---|---|
| straight | 1.00 | 1.00 | 1.00 |
| scripted weave 18 g | 1.00 | 1.00 | 1.00 |
| optimal (anti-LOS) | 0.00 | 0.08 | 0.31 |
| **RL evader** | **0.07** | 0.71 | 0.78 |

The RL evader **slashes its training opponent (True PN) from 1.00 → 0.07** (≈1.2 km median miss) — a
strong learned adversary — but **overfits**: the robust laws (APN, sliding-mode) still catch it,
while the analytic anti-LOS evader generalizes across pursuers. Takeaway: a learned adversary
exploits the *specific* law it trained against, and this independently confirms that prediction
(APN) and robustness (sliding-mode) hold up even against learning. Figure
`gallery/figures/p22_adversarial_evader.png`.

## 17 · Cooperative salvo (impact-time-control guidance)

`ImpactTimeGuidance` ([ADR-0016](adr/0016-impact-time-salvo-guidance.md)) steers each interceptor to a
commanded impact time via biased PN. A 4-interceptor battery launched from different ranges/bearings
arrives **within 0.19 s** of the commanded time against a visibly weaving (~8 g serpentine) inbound
threat — vs a **0.57 s** natural spread — all
intercepting: a synchronized salvo that saturates the defense at once. Figures `gallery/figures/p24_salvo.png`,
`gallery/animations/p24_salvo.gif`.

## 18 · Two-sided self-play (one arms-race round)

The learned evader (§16) defeats True PN (0.07) and challenges Augmented PN (0.71). Training a fresh
interceptor **against that evader** (`InterceptionEnv(opponent=RLEvader(...))`,
[ADR-0017](adr/0017-self-play-round.md)) **hardens** it: held-out vs the frozen evader, the gen-1
interceptor intercepts **0.80** — beating both PN (0.07) and APN (0.71). One detail: this works
only with a *competent* (APN) baseline; a residual on the failing PN baseline stayed at 0.05 — the
"residual needs a competent base" lesson again. Figure `gallery/figures/p25_selfplay.png`.

## 19 · Advanced 3-D estimation (IMM-in-loop + INS platform error)

`EstimatingGuidance` + a 3-D CV/CA `IMM` + interceptor-mounted `Radar3D` close the
sense→estimate→guide loop in 3-D against a maneuvering target at realistic speeds (perfect INS:
P(intercept) 1.00, ~20 m miss). An `INSError` platform model (the seeker measures true geometry; the
filter uses the *believed* platform position) degrades the miss **gracefully and monotonically** —
≈20 → 54 → 80 → 132 → 245 m as INS drift grows 0 → 20 m/s
([ADR-0019](adr/0019-ins-platform-error-and-3d-imm-loop.md), figure
`gallery/figures/p26_estimation_advanced.png`).

## 20 · Converged self-play — a non-transitive arms race

Continuing the self-play to a second round and tabulating the full cross-table
([ADR-0020](adr/0020-converged-self-play.md)) exposes the classic instability of single-step
alternation. P(intercept) by the interceptor (150 trials):

| interceptor \ evader | Eva gen-0 | Eva gen-1 |
|---|---|---|
| True PN | 0.07 | 0.91 |
| Augmented PN | 0.71 | 1.00 |
| Interceptor gen-1 | **0.80** | 1.00 |
| Interceptor gen-2 | **0.18** | 1.00 |

Interceptor gen-2 masters the evader it trained against (gen-1, 1.00) but **catastrophically forgets**
gen-0 (0.80 → 0.18); evader gen-1, overfit to beating gen-1, becomes *more* catchable by everyone
(True PN 0.07 → 0.91). The lesson: stable convergence needs **population-based** self-play
(fictitious play / PSRO against a pool of past opponents), not the latest-opponent. Figure
`gallery/figures/p27_converged_selfplay.png`.

## 21 · Diverse-threat swarm-vs-swarm (a saturating raid, defended)

A defended point is attacked by a **two-wave raid of 12 threats across 6 distinct realistic profiles** — cruise-weave,
sea-skimming pop-up, lofted-ballistic, terminal-spiral, diving-jink, boost-glide
([`adversary.threats`](adr/0021-diverse-threat-swarm.md)) — and defended by **12 interceptors** with
Augmented-PN-3D guidance + Hungarian WTA (now 3-D-aware). Result: **12/12 intercepted, 0 leakers** at
realistic comparable speeds (threats ~Mach 2, interceptors ~Mach 3). Building this fixed a latent
2-D-only WTA bug (assignment now dimension-generic). Cinematic
`gallery/animations/p28_swarm_showcase.gif` + labeled `gallery/figures/p28_swarm_showcase.png`.

## 22 · MARL cooperative swarm — learned vs. analytic allocation

A centralized policy (`CentralizedSwarmEnv`, [ADR-0022](adr/0022-marl-cooperative-swarm.md)) learns
to allocate **3 interceptors over 5 inbound threats** (under-resourced, so coordination matters),
compared head-to-head on identical seeds with the Hungarian WTA and random. Mean leakers (200
trials): Random **0.98**, **Learned (MARL) 0.69**, Hungarian **0.65**. The learned allocator
**discovers cooperative spread from reward alone**, nearly matching the near-optimal analytic
baseline and far beating random — it approaches but does not beat the optimum. Figure
`gallery/figures/p29_marl_swarm.png`.

## 23 · Population self-play — fixing the forgetting

The ADR-0020 catastrophic forgetting (interceptor gen-2 forgot evader gen-0: 0.80→0.18) is addressed
by **fictitious play** — training against a *pool* of past evaders sampled per episode
(`InterceptionEnv(opponent_factory=...)`, [ADR-0023](adr/0023-population-self-play.md)). The
pool-trained interceptor recovers the hard gen-0 to **0.78** (vs gen-2's **0.18**), on par with the
single-round gen-1 (0.80); it trades a little on the easy gen-1 (0.59), so worst-case min 0.59 — well
above the forgetting gen-2. An engineering finding worth recording: per-episode opponent switching makes
returns **bimodal**, destabilizing PPO reward normalization (early runs collapsed to −250 reward);
`norm_reward=False` fixed it. Full PSRO (Nash-mixture best response, growing population) is the
heavier next step. Figure `gallery/figures/p30_population_selfplay.png`.

## 24 · The INTERCEPT League — one Elo ladder over laws *and* evaders

Every seeded engagement is a match (intercept ⇒ law wins; escape ⇒ evader wins); a **Bradley-Terry**
fit over the full round-robin (8 laws × 6 evaders × 40 paired geometries, realistic ~Mach 3 vs
~Mach 2) places everything on one Elo scale ([ADR-0025](adr/0025-intercept-league-elo.md)).
**Sliding-mode is the champion (2214)**; the **game-theoretic anti-LOS evader (2019) out-rates every
other law** — the #2 player is a *target*; the learned recurrent APN-residual (1681) sits above the
whole PN family (1310–1423); scripted evaders bottom out (719). The fit predicts unplayed pairings
(`elo_expected_score`, e.g. sliding-mode vs anti-LOS ≈ 0.75). A literature search found no prior
skill-rating ladder for guidance laws — a novel benchmark framing. Figure
`gallery/figures/p32_league.png`, full table `results/p32_league.md`.

## 25 · Training RL on estimated observations — noise-robustness you can measure

`InterceptionEnv(sensor=..., estimator_factory=...)` trains the policy on a seeker→EKF *estimate*
(reward/intercept stay truth) — closing the ADR-0005 gap. The clean ablation (same weave scenario,
budget, architecture; only the observation source differs,
[ADR-0024](adr/0024-rl-trained-on-estimated-observations.md)): as seeker noise grows to σ=200 m, the
**estimate-trained policy holds 0.99/0.98/0.88** while its **truth-trained twin degrades to
0.97/0.90/0.69** — a 19-point robustness gap earned purely by training under noise. Notes: on
a *straight* target the effect vanishes (the EKF erases the noise), and plain PN-on-estimate remains
the most noise-robust policy on this scenario. Figure `gallery/figures/p31_rl_estimated_obs.png`.

## 26 · Mode-adaptive guidance — the estimator's belief flies the missile

`ModeAdaptiveGuidance` ([ADR-0026](adr/0026-mode-adaptive-guidance.md)) blends True PN and Augmented
PN by the IMM's **maneuver-mode probability** (sharpened, γ=3). Against a realistic threat — a light
~4 g weaving cruise, then a sustained 20 g break — the IMM must discriminate the benign weave from
the real maneuver. Monte-Carlo (60 paired geometries, ~Mach 3 vs ~Mach 2): on a *light* 4 g weave, PN 0.52 @ 79k,
APN 0.55 @ 426k (its feedforward chases the weave), **adaptive 0.62 @ 265k — the best law at 38 %
less effort than APN**; on a *visible* 8 g serpentine the IMM correctly saturates and adaptive
defaults to APN-like behaviour (0.92 @ 449k vs APN 1.00 @ 426k) — it spends effort exactly when the
threat actually maneuvers (both regimes reproducible via `--weave-g`). Operating envelope: sustained-from-launch maneuvers favor
always-on APN (detection lag ≈ 0.8 s). Figure `gallery/figures/p33_mode_adaptive.png`, GIF
`gallery/animations/p33_mode_adaptive.gif` (the trail recolours blue→red as the belief hardens).

## 27 · Pincer coverage — cooperative geometry instead of acceleration sensing

A 30 g break toward an **unpredictable side** at 1.8 km defeats True PN on both branches — and a
**redundant** PN pair is exactly as dead (identical laws ⇒ correlated failures: 0.00). The **pincer
pair** ([ADR-0027](adr/0027-pincer-coverage-guidance.md)) splits the approach via side-offset
virtual aim-points that taper off before the endgame. On a clean cruise the coverage is **perfect —
P(≥1) = 1.00, every kill taken by the interceptor covering that side** — the robustness an APN pair
buys with its acceleration feedforward, achieved with plain PN and *no acceleration estimate at
all*. A realistic ~4 g weaving cruise measurably erodes it (1.00 → 0.45; APN stays 1.00); a *heavy* ~8 g serpentine defeats the static split entirely (≤0.27 at
every β — a fixed side-split cannot cover a serpentine corridor), and the split width must be
matched to the break envelope (narrow β window). Figure
`gallery/figures/p34_pincer.png` (both regimes), GIF `gallery/animations/p34_pincer.gif`.

## 28 · Coordinated swarm penetration vs. an asset-value defense

Real swarms don't just bring *more* missiles — they **coordinate** to defeat the defender's logic.
Four literature-grounded tactics ([ADR-0028](adr/0028-swarm-penetration-and-asset-value-defense.md),
`intercept.adversary.swarm_tactics`): simultaneous **time-on-target**, **decoy screens**,
**concentrated saturation points**, and **sequential waves**. The counter
(`intercept.multiagent.defense`) is an **asset-value layered defense**: impact-point prediction →
decoy de-prioritization → value-prioritized, capacity-aware allocation. Against a 5-interceptor
magazine (40 trials/tactic, mean **real-threat leakers**, lower better):

| tactic | naive time-WTA | asset-value defense |
|---|---|---|
| **decoy-screen (5 real + 7 decoys)** | **1.70** | **0.00** |
| simultaneous-TOT (8 real) | 3.00 | 3.08 |
| concentrated-axis (10 real) | 5.00 | 5.12 |
| stream-raid (9 real) | 4.00 | 4.05 |

The asset-value defender **eliminates real-threat leakage against the decoy screen** by spending its
magazine only on threats that actually endanger the asset — and is **tied** on the all-real
tactics (discrimination helps only when there are decoys; undecoyed over-saturation is magazine-bound,
not algorithm-bound). The discriminator had to be **stateful**: a single snapshot can't separate a
hard-maneuvering real threat (predicted miss swings to 1.5–3 km on a weave) from a decoy, so the
defender tracks each track's *running minimum* predicted miss — which collapses for a real inbound
and floors for a decoy (classical constant-bearing / track-history reasoning). Figures
`gallery/figures/p35_penetration_bars.png`, `gallery/figures/p35_swarm_penetration.png`, and
`gallery/figures/p35_tactics_gallery.png` (each tactic's engagement); GIFs
`gallery/animations/p35_swarm_penetration.gif` and `gallery/animations/p35_tot_raid.gif`.

## Open limitations

See [limitations.md](limitations.md) for the full scope-vs-open account. In brief:

- **RL on the realistic plant:** the from-scratch collapse is *resolved* by residual RL (§11), and
  the recurrent APN-residual now *beats* True PN and Augmented PN on the unpredictable jink (0.95),
  trailing only the robust sliding-mode (1.00). Closing that last gap (deeper/longer training, OGL
  baseline, reactive-break scenarios) is future work.
- Models are point-mass (no 6-DOF / thrust-vector control); aero curves are representative
  (Zarchan-grounded), not flight-validated.
- The benchmark Monte-Carlo now supports L3 (`model="realistic"`); the scenario class and the
  estimator suite are still 2-D — a 3-D Monte-Carlo benchmark + 3-D estimation are tracked follow-ups.
