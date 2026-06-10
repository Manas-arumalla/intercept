# Research Report — State of the Art in Missile Interception, Guidance, Tracking & Defense Simulation

**Project:** Research-grade missile interception & defense simulation platform (software-only)
**Audience:** Robotics engineer portfolio / GitHub showcase (recruiters, researchers, technical reviewers)
**Date:** 2026-06-07
**Status:** Research foundation — the state of the art this project builds on
**Method:** Multi-source deep-research harness (5 search angles, 25 sources fetched, 104 claims extracted, 25 adversarially verified by 3-vote panels → 24 confirmed, 1 refuted). Findings tagged by confidence. Where a topic was not independently verified in this batch, it is marked **[textbook-standard]** and rests on canonical references (Zarchan, Siouris, Shneydor, Bar-Shalom, Isaacs) rather than a verified citation.

> **Scope & ethics note.** This is a *simulation-only, educational/research* platform using point-mass and kinematic models with public, textbook-level algorithms. It contains no hardware integration, no real targeting data, no munitions modeling, and no detection-evasion tooling. The defensive/academic framing is intentional and should be stated explicitly in the repo README.

---

## 0. Executive Summary

The field splits cleanly into a **mature classical core** and a **fast-moving modern layer** — which is exactly the structure that makes a *rigorous classical-vs-novel comparison* the most valuable thing this project can offer.

- **Classical core (high confidence, textbook-canonical):** Proportional Navigation (PN) and its augmented/optimal variants are the documented industry baseline (Siouris 2004; Zarchan). Augmented Ideal PN (AIPN) even has a rigorous *inverse-optimality* derivation as a closed-loop nonlinear optimal feedback law (Cho & Kim, IEEE TAES 2016).
- **Modern layer (high confidence):** Game-theoretic pursuit-evasion (Apollonius circles, differential games — Weintraub/Pachter/Garcia 2020; Dorothy et al., *Automatica* 2024), fixed-time **cooperative/salvo** guidance with impact-angle constraints (Zhou et al. 2022), and **reinforcement learning** — notably *hierarchical PPO* that matched classical miss-distance while improving flight time and energy (Yan et al., *Sci. Rep.* 2022).
- **Tooling (high confidence):** The Python ecosystem is strong and lightweight. `propNav` gives a 3-DOF point-mass PN reference (NumPy + Matplotlib only); PettingZoo/Gymnasium provide MARL APIs and an off-the-shelf Pursuit benchmark; CasADi + acados provide the AD + NLP/SQP + real-time NMPC backbone.
- **The gap / opportunity (the project's thesis):** Comparisons today are *scattered across single studies* on *non-shared* geometries and metrics. **There is no clean, reproducible open-source benchmark that pits PN/APN vs optimal/geometric vs game-theoretic vs learned guidance on identical engagement geometries with shared metrics** (miss distance, P_intercept, time-to-intercept, control effort/ΔV, capture region, Monte-Carlo success). Building that benchmark *is* the differentiator.

**One-line thesis for the repo:** *"A reproducible open-source benchmark and simulation platform that rigorously compares classical, optimal, game-theoretic, and learned missile-interception guidance under identical, configurable engagement geometries — with full Monte-Carlo statistics, capture-region analysis, and publication-quality visualization."*

---

## 1. Guidance & Control Algorithms

### How it works
- **Proportional Navigation (PN).** Commands lateral acceleration proportional to the line-of-sight (LOS) rate: `a_cmd = N · Vc · λ̇`, where `N` is the navigation constant (typically 3–5), `Vc` closing velocity, `λ̇` LOS rotation rate. Variants: **True PN** (acceleration ⟂ to LOS), **Pure PN** (⟂ to missile velocity), **ZEM (zero-effort-miss) PN** (`a_cmd = N·ZEM/t_go²`). [verified — propNav implements True/ZEM/Augmented PN]
- **Augmented PN (APN).** Adds a target-acceleration feedforward term: `a_cmd = N·Vc·λ̇ + (N/2)·a_T`, compensating constant-maneuvering targets. [textbook-standard: Zarchan]
- **Augmented Ideal PN (AIPN).** Derived as the *closed-loop nonlinear optimal feedback solution* for maneuvering-target interception, minimizing a range-weighted control-energy index — **without** linearizing kinematics or assuming near-collision course. [verified — Cho & Kim, IEEE TAES 52(2), 2016]
- **Optimal Guidance Law (OGL / LQ).** Linear-quadratic optimal control minimizing terminal miss + control energy; the classic Bryson-Ho result reduces to APN-like forms under linearized kinematics. [textbook-standard: Siouris, Bryson & Ho]
- **Sliding-Mode Guidance (SMG).** Defines a sliding surface on LOS rate (`s = λ̇`) and drives it to zero with a discontinuous/robust control; strong robustness to target maneuvers and model error, at the cost of chattering. [textbook-standard; closely related to terminal-sliding-mode cooperative laws — see §5]
- **Predictive / MPC guidance.** Solves a finite-horizon optimal control problem online and re-solves each step (see §9).
- **Impact-angle / impact-time constrained guidance.** Augments the law with terminal-angle or time-of-arrival constraints (basis for salvo attack — see §5). [verified in cooperative context — Zhou et al. 2022]
- **Geometric / Differential-Geometry (DG) guidance.** Uses curvature/torsion of the engagement geometry; can extend the kinematic boundary vs PN/APN in tail-chase regions. [verified-medium — NPS thesis ADA556639, single study]

### Where used
PN is the de-facto standard in essentially all fielded homing missiles (textbook canon — Siouris, Zarchan, Shneydor). APN/OGL appear in modern interceptors against maneuvering targets. SMG and DG are active research / specialized.

### Advantages / Limitations
| Law | Advantages | Limitations |
|---|---|---|
| PN | Trivial to implement, near-optimal vs non-maneuvering targets, minimal sensing (just LOS rate) | Degrades vs high-g maneuvering targets; sensitive to noise on λ̇ |
| APN | Handles constant target accel | Needs target-acceleration estimate |
| AIPN/OGL | Rigorous optimality, energy-efficient | Needs t_go and target-accel estimates; model-dependent |
| SMG | Robust to maneuver/model error | Chattering; gain tuning |
| MPC | Handles constraints explicitly | Compute cost; needs model |
| DG | Larger envelope in tail-chase | Slower intercepts; mixed results elsewhere; single-study evidence |

### Implementation complexity (software-only)
PN/ZEM/APN: **trivial** (tens of lines, NumPy). OGL/LQ: **low** (solve Riccati / closed-form). SMG: **low-medium** (surface + reaching law + chattering mitigation). MPC: **medium-high** (CasADi/acados). DG: **medium** (curvature computation).

### Computational requirements
PN/APN/OGL/SMG: negligible — closed-form per step, real-time on any CPU. MPC: per-step NLP solve (ms-scale with acados). No GPU needed for any classical law.

### Portfolio / research value
**Very high as the baseline backbone.** PN/APN/OGL are the reference every reviewer expects; implementing them cleanly with derivations signals GNC literacy. AIPN's inverse-optimality and DG give genuine depth beyond a toy. These are the *control group* against which novel methods are measured.

---

## 2. Trajectory Prediction

### How it works
- **Ballistic / aerodynamic propagation.** Integrate equations of motion (gravity, drag, thrust) forward; for ballistic threats this is highly predictable. [textbook-standard]
- **Kalman-family predictors.** Propagate the estimated state with the motion model between measurements (constant-velocity CV, constant-acceleration CA, coordinated-turn CT). [textbook-standard: Bar-Shalom]
- **IMM (Interacting Multiple Model).** Runs a bank of filters (e.g., CV + CA + CT), maintains a model-probability mixture via a Markov transition matrix — the standard for maneuvering-target prediction. [textbook-standard: Bar-Shalom; Blom & Bar-Shalom 1988]
- **Learning-based prediction.** LSTM/Transformer/Gaussian-process predictors for trajectories that defy simple kinematic models. [emerging — see §13/D]

### Where used / Advantages / Limitations
IMM is ubiquitous in air-defense tracking; CV/CA/CT are the workhorses. Advantages: principled uncertainty, fuses with tracking. Limitations: model mismatch on novel maneuvers; learning-based needs data and lacks guarantees.

### Implementation complexity / Compute
CV/CA/CT propagation: **trivial**. IMM: **medium** (bank + mixing logic). Learning-based: **medium-high** (training pipeline, GPU helpful). All inference is cheap; only training needs GPU.

### Portfolio / research value
**High.** IMM is a recognizable "I know real tracking" signal. A clean IMM + an LSTM predictor with a careful comparison (accuracy vs guarantees) is a strong, self-contained sub-study.

---

## 3. Target Tracking & State Estimation

### How it works  **[textbook-standard: Bar-Shalom "Estimation with Applications to Tracking and Navigation"; not independently verified in this batch]**
- **EKF** — linearizes nonlinear measurement/motion models via Jacobians.
- **UKF** — unscented transform (sigma points); better for strong nonlinearity, no Jacobians.
- **Particle filter (PF)** — sample-based, handles non-Gaussian/multimodal posteriors; expensive.
- **IMM filter** — see §2; the maneuvering-target standard.
- **Multi-target data association:** **JPDA** (joint probabilistic data association) and **MHT** (multiple-hypothesis tracking) for clutter/multiple targets.
- **Track-before-detect (TBD)** — integrate raw sensor returns over time before declaring a track (low-SNR targets).

### Advantages / Limitations
EKF: cheap, ubiquitous, can diverge under strong nonlinearity. UKF: more robust, slightly costlier. PF: most general, costly, sample-degeneracy. JPDA: tractable clutter handling; MHT: more powerful, combinatorial cost.

### Implementation complexity / Compute
EKF/UKF: **low** (FilterPy or hand-rolled). PF: **medium**. JPDA: **medium-high**. MHT: **high**. EKF/UKF cheap; PF/MHT scale with particles/hypotheses (CPU-bound, parallelizable).

### Portfolio / research value
**High and recruiter-legible.** A tracking stack (EKF/UKF/IMM with a noise/clutter model feeding the interceptor) demonstrates the full sense→estimate→act loop, which a guidance-only project lacks. Differentiator: study guidance performance *as a function of estimation quality* (rarely done cleanly).

---

## 4. Sensor Fusion

### How it works  **[textbook-standard; not independently verified in this batch]**
- **Measurement models:** radar (range, range-rate, azimuth/elevation), IR/EO (angles-only — bearing, no range). Angles-only tracking is a classic hard problem.
- **Centralized fusion** — all measurements to one filter (optimal if model correct). **Distributed/decentralized fusion** — local tracks fused at a center.
- **Covariance Intersection (CI)** — fuses estimates with unknown cross-correlation without double-counting information (consistent but conservative).

### Advantages / Limitations
Multi-sensor fusion improves observability (e.g., radar range + IR angles). Centralized is optimal but brittle/comms-heavy; distributed is scalable/robust but needs CI to stay consistent.

### Implementation complexity / Compute
Measurement models + centralized EKF/UKF fusion: **low-medium**. Distributed + CI: **medium**. Cheap at runtime.

### Portfolio / research value
**Medium-high.** Radar+IR fusion with CI is a tidy, impressive module that few hobby projects include. Pairs naturally with §3.

---

## 5. Multi-Agent Interception & Cooperative Interceptors

### How it works
- **Weapon-Target Assignment (WTA).** Assign interceptors to threats to maximize kill probability / minimize leakage — an NP-hard assignment/optimization problem; solved with Hungarian algorithm (linear assignment), auction algorithms, MILP, or heuristics. [textbook-standard: operations-research canon]
- **Cooperative / consensus guidance.** Interceptors share state over a communication graph and reach **consensus on time-to-go** for **simultaneous (salvo) arrival**. [verified]
- **Fixed-time distributed cooperative guidance with impact-angle constraint.** LOS-frame decomposition: a *nonsingular terminal sliding mode along the LOS* enforces simultaneous impact, while the *perpendicular-to-LOS* channel achieves fixed-time convergence of angular rate/angle — convergence time bounded *independent of initial conditions*. [verified — Zhou, Wang & Liu, *Proc. IMechE Part G* 2022; corroborated by large replicated literature. Caveat: point-mass kinematic; gain-dependent/conservative bound; autopilot lag often unmodeled]

### Where used / Advantages / Limitations
Salvo/simultaneous-attack and cooperative-defense research; relevant to saturation-attack defense. Advantages: overwhelm/coordinate, terminal-geometry control. Limitations: needs comms/consensus; idealized kinematics in most results.

### Implementation complexity / Compute
WTA (Hungarian/auction): **low-medium** (`scipy.optimize.linear_sum_assignment`). Consensus salvo guidance: **medium** (graph + terminal-sliding-mode law). Cheap at runtime; scales with agent count.

### Portfolio / research value
**Very high and visually compelling.** Multi-interceptor salvo with synchronized impact is a striking demo and a clear robotics/multi-agent-systems signal. Strong novelty when combined with the benchmark.

---

## 6. Swarm Defense Strategies

### How it works
- **Swarm-vs-swarm / area defense:** many cheap interceptors vs many threats; decentralized control rules.
- **Flocking / Boids** (separation, alignment, cohesion) and potential-field methods for decentralized area coverage and interception. [textbook-standard: Reynolds 1987; Olfati-Saber flocking]
- **MARL swarm defense** — learned decentralized policies (see §8). [emerging — verified MARL tooling via PettingZoo; emerging research direction §13]

### Where used / Advantages / Limitations
Counter-swarm / drone-defense research. Advantages: scalable, robust to individual loss, emergent coverage. Limitations: hard to give guarantees; credit assignment (MARL); coordination overhead.

### Implementation complexity / Compute
Boids/potential-field: **low-medium**. MARL swarm: **high** (training, non-stationarity); GPU recommended. Runtime inference cheap.

### Portfolio / research value
**High — strong visual impact.** A swarm defense scenario (rule-based vs learned) makes an excellent headline animation. PettingZoo's built-in Pursuit env gives a ready MARL baseline (caveat: random, not adversarial, evaders).

---

## 7. Game-Theoretic Planning (Pursuit-Evasion)

### How it works
- **Differential games / Isaacs' theory.** Two-player zero-sum games over continuous dynamics; the value function solves the Hamilton-Jacobi-Isaacs (HJI) PDE; saddle-point strategies. Provides outcome analysis *without assuming opponent behavior*. Taxonomy: 1v1, N-pursuer-1-evader, 1-pursuer-M-evader, NvM. [verified — Weintraub, Pachter & Garcia, ACC 2020]
- **Apollonius circle.** For constant-speed simple-motion games with a faster pursuer, the locus of equal-time-to-reach points is a circle; **a single Apollonius circle solves games where payoff depends only on capture location** (its interior is the evader's dominance region). Reformulating as a nonlinear control problem yields a pursuer strategy guaranteeing capture within (an arbitrarily-close neighborhood of) the initial circle against *any* admissible evader. [verified — Dorothy, Maity, Shishika, Von Moll, *Automatica* 2024. Scope: obstacle-free simple motion, constant speeds, faster pursuer, point capture]
- **Reach-avoid games.** Defender prevents attacker from reaching a target region; HJ reachability gives the value/optimal strategies.

### Advantages / Limitations
Advantages: worst-case guarantees, principled adversarial modeling, beautiful geometry. Limitations: analytic tractability needs perfect-information + simplified dynamics; HJI PDE suffers curse of dimensionality (grid-based HJ reachability limited to ~5-6D).

### Implementation complexity / Compute
Apollonius / 1v1 closed-form strategies: **low-medium** (geometry). HJ reachability (e.g., via `hj_reachability`/`optimized_dp`/Level-Set Toolbox): **high**, grid-bound (GPU helps). 

### Portfolio / research value
**Very high research signal.** Game theory is the most academically respected angle here; Apollonius-circle visualizations are elegant and uncommon in hobby projects. Pairs perfectly with an *adversarial RL evader* (§11) for a "theory vs learned" comparison.

---

## 8. Reinforcement Learning for Guidance/Interception

### How it works
- **Continuous-control RL:** **DDPG, TD3, SAC** (off-policy, sample-efficient), **PPO** (on-policy, stable). State = relative kinematics (LOS, range, closing velocity, target accel estimate); action = commanded lateral acceleration; reward shaped on terminal **miss distance** (+ effort/time penalties).
- **Hierarchical RL.** A high-level policy selector activates low-level guidance + evasion sub-policies. A two-layer hierarchical PPO let a missile *simultaneously* guide-to-target and evade an interceptor, hitting **100% test-set success where flat PPO never converged** — and matched classical miss distance (RL 0.441 m vs 0.428 m) while improving flight time (~49.5 s vs ~54.9 s) and energy. [verified — Yan et al., *Sci. Rep.* 12:18888, 2022. Caveat: single self-reported study, ~100-state test set, idealized point-mass, not independently replicated]
- **MARL:** MADDPG (centralized-training/decentralized-execution), QMIX (value factorization) for cooperative/swarm interception. [verified tooling via PettingZoo; emerging research §13]
- **Curriculum learning & sim-to-real:** stage difficulty (non-maneuvering → maneuvering → evasive targets); domain randomization for robustness.

### Advantages / Limitations
Advantages: handles complex/partial-information regimes, can co-optimize effort/time, no hand-derived law. Limitations: no convergence/optimality guarantees, sample-hungry, reward-shaping-sensitive, sim-to-real gap, hard to certify — and **rarely benchmarked fairly against tuned classical laws** (the core opportunity).

### Implementation complexity / Compute
**Medium-high.** Stable-Baselines3 / CleanRL + a custom Gymnasium env. Single-agent PPO/SAC trains in minutes-hours on the local GPU for point-mass; MARL is harder. **GPU recommended (available).**

### Portfolio / research value
**Very high — this is the project's centerpiece spine.** A clean Gymnasium interception env + trained PPO/SAC/TD3 agents benchmarked head-to-head against PN/APN/OGL on identical geometries and effort metrics is exactly the comparison the literature does *not* standardize. High novelty + high recruiter appeal (RL + control).

---

## 9. Optimization-Based Interception (MPC / Trajectory Optimization)

### How it works
- **MPC:** solve a finite-horizon optimal control problem (minimize miss + effort subject to dynamics + acceleration/seeker limits), apply the first control, re-solve each step (receding horizon).
- **Trajectory optimization:** **direct collocation** and **single/multiple shooting** transcribe the OCP into an NLP solved by SQP/interior-point (IPOPT) methods.
- **Real-time iteration (RTI):** one SQP iteration per step for real-time NMPC. [verified — acados implements SQP-type solvers, multiple shooting, HPIPM QP on BLASFEO; CasADi provides the AD backbone]

### Where used / Advantages / Limitations
Modern GNC research, constrained terminal-guidance, impact-angle/time control. Advantages: handles constraints explicitly, optimal-by-construction, replanning-ready. Limitations: needs a model, compute cost, local optima, tuning of horizon/weights.

### Implementation complexity / Compute
**Medium-high.** `do-mpc`/CasADi for prototyping, **acados** for real-time speed (ms-scale solves). CPU-bound; no GPU needed.

### Portfolio / research value
**High — premier control-systems signal.** An NMPC interceptor (acados) benchmarked against PN and RL closes the "classical / optimal / learned" triangle and demonstrates real optimal-control engineering. Strong with reviewers from robotics/controls.

---

## 10. Uncertainty-Aware Decision Making

### How it works  **[textbook-standard; not independently verified in this batch]**
- **POMDPs** — decision-making under partial observability (belief-state MDP); solved approximately (POMCP, DESPOT, SARSOP; `POMDPs.jl`).
- **Robust / chance-constrained control** — guarantee constraints under bounded/probabilistic uncertainty (e.g., keep P[miss < r] ≥ 1−ε).
- **Belief-space planning** — plan over distributions of states rather than point states.

### Advantages / Limitations
Advantages: principled handling of sensing/uncertainty; explicit risk accounting. Limitations: POMDP solves scale poorly; chance constraints add conservatism and complexity.

### Implementation complexity / Compute
POMDP solvers: **high**. Chance-constrained MPC: **medium-high** (extends §9). Compute moderate-heavy.

### Portfolio / research value
**Medium-high (selective).** A chance-constrained MPC layer or a belief-space interception decision is a sophisticated, differentiating add-on; full POMDP solving is likely scope-overflow — better as a focused demonstration than a pillar.

---

## 11. Adversarial Strategies (Evaders & Countermeasures)

### How it works
- **Evasive maneuvers:** weave/sinusoidal (the classic PN-defeating maneuver), barrel-roll, jinking, optimal bang-bang evasion. [textbook-standard: Zarchan; Shneydor]
- **Decoys / countermeasures:** flares/chaff modeled as false tracks feeding the estimator (ties to §3-4 data association).
- **Adversarial RL:** train the *evader* with RL against the interceptor (self-play / co-evolution); a learned missile-maneuvering algorithm can be posed as the adversary. [verified — game-theoretic maneuvering algorithm exists in literature; adversarial-RL framing emerging]

### Advantages / Limitations
Advantages: stress-tests guidance rigorously, prevents overfitting to weak targets, produces a difficulty curriculum. Limitations: self-play instability, non-stationarity, can be compute-heavy.

### Implementation complexity / Compute
Scripted evasions: **trivial-low**. Adversarial-RL self-play: **high** (instability, GPU). 

### Portfolio / research value
**High.** An adversarial evader (scripted → learned) is what turns a "does PN hit a straight target" toy into a *real benchmark*. The "theory-optimal evader (game theory) vs learned evader (RL)" comparison is genuinely novel and publishable-flavored.

---

## 12. Real-Time Replanning

### How it works
- **Receding-horizon / MPC replanning** (see §9): re-solve each step.
- **Event-triggered replanning:** re-plan only when a trigger fires (target maneuver detected, estimate jump) — saves compute.
- **Anytime planners:** return a feasible solution fast, improve with more time (anytime RRT*, anytime A*); relevant for mid-course routing.

### Advantages / Limitations
Advantages: adapts to maneuvering targets / new threats; event-triggering saves compute. Limitations: real-time deadlines, solver warm-starting, stability of frequent replanning.

### Implementation complexity / Compute
Receding-horizon: comes "for free" with MPC (§9). Event-triggered: **low** add-on (trigger logic). Anytime planners: **medium**. Compute as per §9.

### Portfolio / research value
**Medium-high.** Best framed as a *property* of the MPC/RL pillars (e.g., "event-triggered NMPC re-plans on detected target maneuver") rather than a standalone module — demonstrates systems-level real-time thinking.

---

## A. Open-Source Frameworks, Benchmarks & Libraries

| Tool | Role | Notes / confidence |
|---|---|---|
| **`propNav`** (github.com/gedeschaines/propNav) | 3-DOF point-mass PN reference (True/ZEM/Augmented PN, N=4 example) | NumPy + Matplotlib only; RK4 bundled. **Ideal starting reference.** [verified] |
| **`missile-proportional-navigation-python`** (alti3) | PN demo in Python | Secondary reference [verified-listed] |
| **Gymnasium** | Single-agent RL API | Standard; build custom interception env [textbook-standard] |
| **PettingZoo** (Farama) | Multi-agent RL API + built-in **Pursuit** env (8 pursuers/30 evaders/16×16) | Off-the-shelf MARL benchmark. **Caveat: evaders move randomly, not adversarially** [verified] |
| **Stable-Baselines3 / CleanRL** | RL algorithms (PPO/SAC/TD3/DDPG) | SB3 = batteries-included; CleanRL = single-file, readable [textbook-standard] |
| **CasADi** (web.casadi.org) | AD + nonlinear optimization backbone | Backs do-mpc/acados/rockit; interfaces IPOPT/SNOPT/qpOASES/HPIPM [verified] |
| **acados** (docs.acados.org) | Real-time NLP/OCP, SQP, multiple shooting, HPIPM | Python/MATLAB/Octave interfaces; ms-scale NMPC [verified] |
| **do-mpc** | High-level MPC on CasADi | Fast prototyping [verified-adjacent] |
| **python-control** | Classical/modern control (LQR, Riccati, TF) | For OGL/LQ, autopilot loops [textbook-standard] |
| **FilterPy** | Kalman/EKF/UKF/IMM/PF | Estimation stack (§3) [textbook-standard] |
| **JSBSim** | 6-DOF flight dynamics | For optional high-fidelity extension [textbook-standard] |
| **`hj_reachability` / `optimized_dp`** | HJ reachability / differential games | GPU-capable, grid-bound (§7) [textbook-standard] |
| **PyVista / Plotly / Matplotlib** | 3D / interactive / publication viz | Visualization stack [textbook-standard] |

## B. Standard Evaluation Metrics  [partly textbook-standard; Zarchan SGS paper & arXiv 1906.02113 in source set]
- **Miss distance** (primary; closest approach) and **terminal miss CEP**.
- **Probability of intercept / kill (P_k)** — Monte-Carlo success rate over randomized engagements.
- **Time-to-intercept** and **flight time**.
- **Control effort / ΔV / energy** (∫a² dt) — efficiency, *under-used in comparisons*.
- **Capture region / launch envelope** — set of initial geometries from which intercept succeeds (kinematic boundary). 
- **Robustness** — success vs target maneuver level (g), sensor noise, latency.

## C. Validation / Benchmarking Methodology  [textbook-standard: Zarchan Monte-Carlo guidance analysis]
- **Monte-Carlo over engagement geometries** (randomized initial range, aspect angle, target maneuver, noise) → success-rate statistics with confidence intervals.
- **Capture-region sweeps** — grid over initial conditions, map success/failure boundary.
- **Ablation studies** — isolate the effect of each component (e.g., estimation quality → guidance performance).
- **Seeded reproducibility** — fixed RNG seeds, logged configs.

## D. Novel & Emerging Directions (last few years)
- **Learning-based / neural guidance** matching classical accuracy with better effort/time [verified — Yan 2022].
- **Differentiable simulation** — end-to-end gradients through the dynamics for guidance/policy optimization (emerging; sources arXiv 2508.00641, 2508.06520, Wiley RNC 70029 in set).
- **MARL swarm defense** & **game-theoretic deep learning** (deep-RL approximations to differential-game equilibria) [emerging — tandfonline 2024.2355023 in set].
- **Hierarchical RL** for combined guidance+evasion [verified — Yan 2022].

## E. Gaps & Opportunities (the project's differentiators)
1. **No standardized open-source cross-paradigm benchmark.** PN/APN vs optimal/geometric vs game-theoretic vs learned guidance are compared (if at all) on *different* geometries/metrics. **A shared-geometry, shared-metric, Monte-Carlo benchmark is the headline contribution.** [verified as a gap]
2. **Effort/efficiency under-reported.** Most comparisons report miss distance only; control effort/ΔV/time comparisons are rare and revealing (RL's real edge) [verified — Yan 2022].
3. **Estimation-coupled guidance.** Studying guidance performance *as a function of estimator quality* (EKF/UKF/IMM noise) is rarely done cleanly — easy, valuable sub-study.
4. **Theory vs learned adversaries.** Game-theoretic optimal evaders vs RL-learned evaders on the same field — novel and visually compelling.
5. **Fidelity-ladder robustness.** Almost all cited results are idealized point-mass; a *progressive-fidelity* platform that shows which conclusions survive autopilot lag / actuator saturation / sensor noise is genuinely useful.
6. **Reproducibility.** A clean, documented, seeded, pip-installable benchmark with publication-quality plots fills a real community gap.

---

## Confidence Ledger & Caveats
- **High confidence (verified 3-0):** PN/APN canonical & comparison framing (Siouris); AIPN inverse-optimality (Cho & Kim 2016); differential-game taxonomy (Weintraub 2020); Apollonius-circle results (Dorothy 2024, pursuer-strategy 2-1); hierarchical PPO results (Yan 2022); fixed-time cooperative salvo guidance (Zhou 2022); tooling (propNav, PettingZoo, CasADi, acados).
- **Medium confidence (single study):** DG-guidance envelope advantage (NPS thesis ADA556639) — do not over-generalize.
- **Refuted (do NOT rely on):** "PN/APN/DG are within 1–3% vs non-maneuvering noise-free targets / advanced laws give negligible benefit" — refuted 1-2.
- **[textbook-standard], not independently verified this batch:** EKF/UKF/PF/IMM/JPDA/MHT/TBD (§3), sensor fusion / covariance intersection (§4), POMDP/chance-constrained/belief-space (§10), evasion kinematics (§11), metrics/Monte-Carlo methodology (§B/C). These are well-established in Bar-Shalom / Zarchan / Isaacs and safe to build on, but cite primary texts when documenting.
- **Time-sensitivity:** RL / cooperative-guidance / differentiable-sim / tooling are fast-moving — re-check for newer results before publishing.

## Open Questions Carried Into Planning
1. Exact set of standardized engagement geometries & metrics for the benchmark (the core deliverable).
2. How far up the fidelity ladder to climb (point-mass → autopilot lag → 6-DOF?) for fair comparison.
3. Which estimation/fusion stack to standardize on.
4. How to model adversarial evaders (scripted → game-theoretic → RL self-play) for robustness.

## Source List (verified set)
- Siouris, *Missile Guidance and Control Systems* (Springer 2004) — https://link.springer.com/book/10.1007/b97614
- Shneydor, *Missile Guidance and Pursuit* — https://archive.org/details/missileguidancep0000shne
- Zarchan, *Tactical and Strategic Missile Guidance* (7th ed.) — https://www.amazon.com/Tactical-Strategic-Missile-Guidance-Seventh/dp/162410584X
- Cho & Kim, "Optimality of Augmented Ideal PN…", IEEE TAES 52(2) 2016 — https://www.researchgate.net/publication/303671258
- NPS thesis (Differential Geometry guidance), ADA556639 — https://apps.dtic.mil/sti/tr/pdf/ADA556639.pdf
- Weintraub, Pachter & Garcia, "An Introduction to Pursuit-Evasion Differential Games", ACC 2020 — https://arxiv.org/pdf/2003.05013
- Dorothy, Maity, Shishika, Von Moll, "One Apollonius Circle is Enough…", *Automatica* 2024 — https://arxiv.org/pdf/2111.09205
- Yan et al., hierarchical PPO guidance+evasion, *Sci. Rep.* 12:18888 (2022) — https://www.nature.com/articles/s41598-022-21756-6
- Zhou, Wang & Liu, fixed-time cooperative guidance, *Proc. IMechE Part G* (2022) — https://journals.sagepub.com/doi/abs/10.1177/09544100211048043
- propNav — https://github.com/gedeschaines/propNav
- PettingZoo — https://github.com/Farama-Foundation/PettingZoo
- CasADi — https://web.casadi.org/ ; acados — https://docs.acados.org/
- Zarchan, Science & Global Security — https://scienceandglobalsecurity.org/archive/sgs08zarchan.pdf
- Eval/methodology & emerging-direction sources — arXiv 1906.02113, 2511.02526, 2508.00641, 2508.06520; Wiley RNC 70029; Tandfonline 2024.2355023; NASA TM-109057; ScienceDirect S100093611930247X
