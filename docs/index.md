# INTERCEPT — Documentation

**A research-grade, reproducible benchmark and simulation platform comparing classical, optimal, game-theoretic, and learned missile-interception guidance.**

> **Scope & ethics.** Simulation-only, educational/research project using point-mass/kinematic models and public textbook algorithms. No hardware, no real targeting/sensor data, no munitions modeling, no detection-evasion tooling. See the [README](../README.md).

## Start here

- **Research foundation:** [Research Report](../planning/00_RESEARCH_REPORT.md) — fact-checked state of the art across guidance, tracking, estimation, fusion, multi-agent, game theory, RL, optimization, and the identified gaps.
- **Plan & roadmap:** [Project Plan](../planning/01_PROJECT_PLAN.md) — objectives → architecture → benchmark design → phases/milestones → risks → expected outcomes.
- **Results digest:** [results.md](results.md) — headline findings with the figure/experiment behind each.
- **Scope & limitations:** [limitations.md](limitations.md) — deliberate boundaries vs. open follow-ups, stated plainly.
- **Progress log:** [PROGRESS.md](progress/PROGRESS.md) — dated development history and milestone checkpoints.
- **Decisions:** [Architecture Decision Records](adr/) — the reasoning behind each design choice.

## The thesis

Cross-paradigm guidance comparisons today are scattered across single studies on *different* engagement geometries and metrics, and most report only miss distance. INTERCEPT puts **PN/APN, optimal/geometric, game-theoretic, MPC, and RL** guidance on the **same field** — identical dynamics, shared scenario suite, shared metrics, full Monte-Carlo statistics — and makes the comparison **reproducible**.

## Architecture at a glance

```
Scenario (YAML) → Simulation Core (dynamics · RK4 · engagement loop)
   ├── Sensors      → Estimation/Tracking (EKF/UKF/IMM) ┐
   ├── Guidance/Control (PN·APN·OGL·SMG·MPC·RL·Game)    ├→ Engagement → Result/metrics
   └── Adversary (scripted · game-theoretic · RL)        ┘
Experiment & Analysis layer: Monte-Carlo · benchmark harness · capture-region · viz · RL training
```

The core is algorithm-agnostic; guidance laws, estimators, sensors, and adversaries are plug-ins conforming to a single controller/interface contract, so every paradigm runs against *identical* dynamics — the fairness property the benchmark depends on.

## Capabilities

**Simulation core.** `PointMass2D/3D` dynamics, an `RK4` integrator, line-of-sight geometry, the `Entity`/controller contract, and an `Engagement` loop with tunnelling-proof intercept detection. The same loop runs every fidelity level and both dimensions unchanged.

**Guidance, six paradigms.** Proportional Navigation (True/Pure/ZEM) and Augmented PN ([notes](algorithms/proportional-navigation.md)); optimal LQ/ZEM (OGL), sliding-mode, and CasADi NMPC with an impact-angle constraint and event-triggered replanning ([notes](algorithms/optimal-mpc-guidance.md)); Apollonius-circle geometric pursuit and a game-theoretic optimal evader ([notes](algorithms/game-theory.md)); and reinforcement learning ([notes](algorithms/rl-guidance.md)). Every law runs in 2-D and 3-D.

**Fidelity ladder (L0→L3).** `AeroMissile2D/3D` adds gravity, parasitic and induced drag, a g-limit, and autopilot lag; `RealisticMissile2D/3D` adds an ISA atmosphere, boost–sustain–coast propulsion with mass burn-off, Mach-dependent drag, and lift/dynamic-pressure-limited turning, so available g emerges from the physics rather than a prescribed limit ([ADR-0008](adr/0008-realistic-aero-propulsion-l3.md)). On realistic evasive engagements simple PN drops to 0.21–0.56 P(intercept) while prediction and robust laws recover to 0.79–1.00 ([notes](algorithms/realistic-engagements.md)).

**Estimation and sensing.** Radar and IR-EO sensor models with seeded noise; EKF (Joseph), UKF, and IMM filters; and `EstimatingGuidance` closing the sense→estimate→guide loop in 2-D and 3-D ([notes](algorithms/estimation-tracking.md), [ADR-0013](adr/0013-three-dimensional-estimation.md)). The IMM holds ~9 m tracking error through a turn that diverges a single-model EKF to ~350 m.

**The benchmark.** A YAML scenario suite (2-D and 3-D), seeded Monte-Carlo built on the [fairness invariant](adr/0003-benchmark-fairness-invariants.md), Wilson-interval metrics, capture-region sweeps, and paired-bootstrap significance testing ([methodology](benchmarks/methodology.md)). A capstone heatmap spans paradigm × fidelity (L0→L3) × dimension with per-row significance.

**Learned guidance that wins.** From-scratch PPO collapses on the realistic plant; a **residual** parameterization (a bounded correction on a PN/APN baseline) restores it, and the recurrent APN-residual reaches P(intercept) **1.00/1.00/0.95** in 2-D — beating True PN (0.81) and Augmented PN (0.93) on the unpredictable jink — and full parity in 3-D ([ADR-0011](adr/0011-residual-rl-guidance.md), [RL notes](algorithms/rl-guidance.md)).

**Many-vs-many and coordination.** Hungarian weapon-target assignment with live re-assignment, a diverse-threat raid library, cooperative salvo (impact-time) and pincer-coverage guidance, and coordinated penetration tactics countered by an asset-value layered defense ([ADR-0009](adr/0009-multiagent-swarm.md), [ADR-0028](adr/0028-swarm-penetration-and-asset-value-defense.md)).

**The INTERCEPT League.** Every seeded engagement is a match; a Bradley-Terry fit puts every guidance law *and* every evader on one Elo ladder ([ADR-0025](adr/0025-intercept-league-elo.md)). Sliding-mode wins; the game-theoretic evader out-rates every guidance law.

**Quality bar.** 199 unit/property/regression tests passing, `ruff`-clean, fully type-hinted (with `mypy` running in CI). Guidance spans six paradigms; fidelity spans L0→L3; engagements run in 2-D, 3-D, and many-vs-many on one fair benchmark.

**Open follow-ups (see [limitations.md](limitations.md)):** full PSRO self-play (Nash-mixture best response); decentralized MARL (MAPPO/PettingZoo); margin-aware League ratings (ordinal Bradley-Terry over miss distance).
