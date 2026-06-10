# Development history

This is a milestone-level narrative of how INTERCEPT came together — what I built, in what order,
and why. It complements the [CHANGELOG](../../CHANGELOG.md) (release-style entries) and the
[Architecture Decision Records](../adr/) (the reasoning behind each design choice). Every milestone
below is reflected in committed code, tests, figures, and result tables.

The project grew along three axes at once: **paradigm** (PN → optimal → game-theoretic → learned),
**fidelity** (L0 point-mass → L3 aero-propulsive), and **scale** (1-v-1 → many-vs-many). The
through-line is a single fairness invariant — every controller runs against the same injected
dynamics — so each addition slots into one benchmark rather than a new one.

---

## Foundations — the simulation core and a fair contract

I started with the piece everything else depends on: an algorithm-agnostic core. `PointMass2D`
dynamics, an `RK4` integrator, line-of-sight geometry, and an `Engagement` loop that detects
intercepts by closest approach *within* a step (so a fast closing geometry can't tunnel through the
kill radius between samples). The key abstraction is the controller contract
`(t, own_state, world) -> control`: guidance laws, evaders, and allocators all satisfy it and all
share the same dynamics. That contract is what makes every later comparison fair, and it is the
reason the engagement loop never had to change as fidelity and dimension grew
([ADR-0001](../adr/0001-python-primary-stack.md), [ADR-0002](../adr/0002-point-mass-first-fidelity.md)).

## Guidance baselines and the benchmark harness

On that core I implemented the Proportional Navigation family (True / Pure / ZEM) and Augmented PN,
validated against the textbook results: True PN intercepts a low-authority crossing target at 8.2 m
where pure pursuit misses by 103 m, and APN's acceleration feedforward reduces terminal miss against
a maneuvering target. I then built the benchmark harness that is the centerpiece of the project — a
YAML scenario suite, seeded Monte-Carlo with the fairness invariant, Wilson-interval metrics, and
capture-region sweeps — and fixed a subtle bug where the control-effort metric diverged as range → 0
until I switched it to integrate the *saturated* (applied) acceleration
([ADR-0003](../adr/0003-benchmark-fairness-invariants.md)).

## Sensing, estimation, and the closed loop

Next I added the autonomy stack: radar (range + bearing) and IR (angles-only) sensors with seeded
noise, then EKF (Joseph form), UKF, and IMM estimators on a shared state model, and an
`EstimatingGuidance` wrapper that closes the sense → estimate → guide loop by substituting the
estimate into the world snapshot the guidance law reads. The headline here is the IMM holding ~9 m
tracking error through a hard target turn where a single-model EKF diverges to ~350 m
([ADR-0004](../adr/0004-sensor-estimator-interface.md)).

## Optimal, geometric, and model-predictive guidance

I broadened the paradigm coverage with optimal LQ/ZEM guidance (OGL), sliding-mode guidance, and a
CasADi/IPOPT nonlinear MPC that supports an impact-angle constraint and event-triggered replanning.
Run against a weaving target, these expose the trade-off space the benchmark exists to reveal: OGL
most accurate, PN cheapest in control effort, sliding-mode most robust, NMPC most flexible but
costliest ([ADR-0005](../adr/0005-rl-env-contract.md) for the env contract,
[optimal/MPC notes](../algorithms/optimal-mpc-guidance.md)).

## Reinforcement learning, done as a fair comparison

The RL work was the most involved. A Gymnasium environment wraps the engagement core; PPO trains
with VecNormalize and a curriculum, against a potential-based zero-effort-miss reward. The discipline
I held throughout was to report learned guidance as *competitive, not magical*: against the classical
laws it matches on the easy shots and trades control effort for the harder ones. Getting there meant
solving three concrete failure modes — a reward that rewarded pursuit instead of lead, intercept
tunnelling at the simulation step, and observation normalization that was hiding the tiny lead
signal — each documented so the result is reproducible rather than lucky
([ADR-0005](../adr/0005-rl-env-contract.md), [RL notes](../algorithms/rl-guidance.md)).

## Game theory and adversaries

To make the evader side a first-class citizen I added Apollonius-circle dominance geometry, a
constant-bearing geometric pursuer, and a game-theoretic optimal (anti-line-of-sight) evader. A
useful nuance fell out of the comparison: the "hardest" adversary depends on the metric — the
optimal evader maximizes *capture time*, while scripted weave/jink maximize *miss distance*
([ADR-0006](../adr/0006-realistic-fidelity-l2.md), [game-theory notes](../algorithms/game-theory.md)).

## Climbing the fidelity ladder (L2 → L3)

I then raised fidelity behind the same `Dynamics` interface. L2 (`AeroMissile2D/3D`) adds gravity,
parasitic and induced drag, a g-limit, and autopilot lag; L3 (`RealisticMissile2D/3D`) adds an ISA
atmosphere, boost–sustain–coast propulsion with propellant burn-off, Mach-dependent drag, and
lift/dynamic-pressure-limited turning — so available g *emerges from the physics* rather than being
prescribed. The benchmark result that justifies the whole ladder: on realistic evasive engagements
True PN collapses to 0.21–0.56 P(intercept) on a reactive break / unpredictable jink, while
prediction (APN), optimality (OGL), and robustness (sliding-mode) recover to 0.79–1.00 —
intelligence beats a speed margin ([ADR-0006](../adr/0006-realistic-fidelity-l2.md),
[ADR-0008](../adr/0008-realistic-aero-propulsion-l3.md),
[realistic-engagement notes](../algorithms/realistic-engagements.md)).

## Three dimensions, end to end

Because the geometry, engagement loop, and metrics were written dimension-agnostically, extending to
3-D was mostly a matter of adding `PointMass3D` / `AeroMissile3D` / `RealisticMissile3D`, a 3-D
line-of-sight rate (the vector form Ω = r × v / |r|²), and 3-D PN/APN — with the engagement core
unchanged. From there I brought every layer into 3-D: the Monte-Carlo benchmark
(`ParametricScenario3D`), dimension-generic EKF/UKF estimation with `Radar3D`, and 3-D RL. A
barrel-roll evader defeats True PN-3D (0.00) while APN/OGL/SMG hold 1.00
([ADR-0007](../adr/0007-three-dimensional-extension.md),
[ADR-0012](../adr/0012-three-dimensional-benchmark.md),
[ADR-0013](../adr/0013-three-dimensional-estimation.md),
[ADR-0014](../adr/0014-three-dimensional-rl.md)).

## Speed parity — keeping the showcase internally consistent

A first cut of the advanced L3 showcase let the interceptor out-run the target (a +226 % closing-speed
edge), which would have been a win of thrust, not guidance. I re-tuned the propulsion to a realistic
parity (~+37 % closing edge) and added a propulsion sweep proving that a smaller motor turns the win
into a miss — confirming the intercept is a near-minimum-energy *algorithm* win. I then standardized
this speed regime project-wide so no experiment quietly benefits from an unrealistic margin
([ADR-0010](../adr/0010-realistic-speed-parity-no-cheat.md),
[ADR-0018](../adr/0018-realistic-speed-standardization.md)).

## Resolving the realistic-plant RL collapse — a learned win

From-scratch PPO collapses on the lagged, gravity-loaded realistic plant (~0–2 % intercept). The fix
is a **residual** parameterization: the policy outputs a bounded correction on a PN/APN baseline, so
a zero action is already competent and there is nothing to collapse to. With an APN baseline and a
recurrent (LSTM) policy, the learned law reaches 1.00 / 1.00 / 0.95 (crossing / weave / jink) and
*beats* both True PN (0.81) and Augmented PN (0.93) on the unpredictable jink, trailing only
sliding-mode — a genuine learned win on the hardest realistic case
([ADR-0011](../adr/0011-residual-rl-guidance.md), [RL notes](../algorithms/rl-guidance.md)).

## Many-vs-many: assignment, cooperation, and coordination

Area defense brought the scale axis. `MultiEngagement` runs N interceptors against M threats with
live Hungarian weapon-target re-assignment; on top of it I added a global-kill-probability objective
(shoot-look-shoot surplus routing), a diverse-threat raid library (six realistic trajectory
profiles), 3-D-aware assignment, and a cinematic 3-D swarm animator. The cooperative-guidance side
came next: impact-time-control salvo guidance synchronizing a battery's arrival, and pincer coverage
where a decorrelated PN pair covers both branches of an unpredictable break with no acceleration
estimate. Most recently I modeled coordinated *penetration* tactics — time-on-target, decoy screens,
saturation points, sequential waves — and built the asset-value layered defense that counters them,
eliminating real-threat leakage against a decoy screen where a naive time-WTA defender wastes its
magazine on chaff ([ADR-0009](../adr/0009-multiagent-swarm.md),
[ADR-0016](../adr/0016-impact-time-salvo-guidance.md),
[ADR-0021](../adr/0021-diverse-threat-swarm.md),
[ADR-0027](../adr/0027-pincer-coverage-guidance.md),
[ADR-0028](../adr/0028-swarm-penetration-and-asset-value-defense.md)).

## Self-play, MARL, and estimation-aware guidance

The adversarial and learning frontier produced a cluster of focused studies: a learned evader and a
two-sided arms-race round that turns out to be non-transitive (a later interceptor catastrophically
forgets an earlier opponent), population/fictitious play that fixes the forgetting (0.18 → 0.78), a
centralized MARL allocator that nearly matches the Hungarian optimum, RL trained directly on
estimated observations (a measurable robustness gap over a truth-trained twin), and two novel
estimation-aware guidance laws — mode-adaptive guidance, where the IMM's maneuver belief arbitrates
between PN and APN, and the pincer pair above
([ADR-0015](../adr/0015-adversarial-rl-evader.md), [ADR-0017](../adr/0017-self-play-round.md),
[ADR-0020](../adr/0020-converged-self-play.md), [ADR-0022](../adr/0022-marl-cooperative-swarm.md),
[ADR-0023](../adr/0023-population-self-play.md),
[ADR-0024](../adr/0024-rl-trained-on-estimated-observations.md),
[ADR-0026](../adr/0026-mode-adaptive-guidance.md)).

## The INTERCEPT League — one ladder over everything

To summarize the whole comparison in a single artifact, I framed each seeded engagement as a match
and fit a Bradley-Terry model over the full round-robin, placing every guidance law *and* every
evader on one Elo ladder. Sliding-mode wins the league, the game-theoretic evader out-rates every
guidance law (a target is the #2 player overall), and the fit predicts unplayed match-ups. It is the
first skill-rating ladder for guidance laws I am aware of in the literature
([ADR-0025](../adr/0025-intercept-league-elo.md)).

## Where it stands

Guidance spans six paradigms; fidelity spans L0 → L3; engagements run in 2-D, 3-D, and
many-vs-many — all on one fair, seeded benchmark, with 199 tests passing, `ruff`-clean, and fully
type-hinted (`mypy` runs in CI).
The deliberate boundaries and the genuine open follow-ups are catalogued in
[limitations.md](../limitations.md); the headline numbers and the figure behind each live in
[results.md](../results.md).
