# Architecture Decision Records

Numbered, immutable records of significant decisions. Each captures the context, the decision, and
its consequences. Supersede (don't edit) a decision by adding a new ADR that references the old one.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-python-primary-stack.md) | Python as the primary implementation stack | Accepted |
| [0002](0002-point-mass-first-fidelity.md) | Progressive fidelity: point-mass first, pluggable dynamics | Accepted |
| [0003](0003-benchmark-fairness-invariants.md) | Benchmark fairness invariants | Accepted |
| [0004](0004-sensor-estimator-interface.md) | Sensor / estimator interface and estimation-coupled guidance | Accepted |
| [0005](0005-rl-env-contract.md) | RL environment contract, reward shaping, fair learned-vs-classical comparison | Accepted |
| [0006](0006-realistic-fidelity-l2.md) | Realistic engagement fidelity (L2): aero dynamics + aggressive/reactive targets | Accepted |
| [0007](0007-three-dimensional-extension.md) | Three-dimensional engagements (3-D dynamics, geometry, PN/APN, viz) | Accepted |
| [0008](0008-realistic-aero-propulsion-l3.md) | L3 realism: aero-propulsive physics (atmosphere, thrust/mass, q-limited g) | Accepted |
| [0009](0009-multiagent-swarm.md) | Multi-agent / swarm engagements (Hungarian WTA + N-vs-M) | Accepted |
| [0010](0010-realistic-speed-parity-no-cheat.md) | Realistic speed parity for showcase engagements (no speed cheat) | Accepted |
| [0011](0011-residual-rl-guidance.md) | Residual RL guidance (learned correction on a PN baseline) | Accepted |
| [0012](0012-three-dimensional-benchmark.md) | Three-dimensional Monte-Carlo benchmark (ParametricScenario3D) | Accepted |
| [0013](0013-three-dimensional-estimation.md) | Three-dimensional estimation (dimension-generic EKF/UKF + Radar3D) | Accepted |
| [0014](0014-three-dimensional-rl.md) | Three-dimensional RL guidance (InterceptionEnv3D + RLGuidance3D) | Accepted |
| [0015](0015-adversarial-rl-evader.md) | Adversarial-RL evader (EvaderEnv + RLEvader) | Accepted |
| [0016](0016-impact-time-salvo-guidance.md) | Impact-time-control (cooperative salvo) guidance | Accepted |
| [0017](0017-self-play-round.md) | Two-sided self-play (one arms-race round) | Accepted |
| [0018](0018-realistic-speed-standardization.md) | Project-wide realistic-speed standardization + gallery layout | Accepted |
| [0019](0019-ins-platform-error-and-3d-imm-loop.md) | INS platform error + 3-D IMM in the guidance loop | Accepted |
| [0020](0020-converged-self-play.md) | Converged self-play (alternating arms race + cross-table) | Accepted |
| [0021](0021-diverse-threat-swarm.md) | Diverse-threat swarm-vs-swarm + 3-D weapon-target assignment | Accepted |
| [0022](0022-marl-cooperative-swarm.md) | MARL cooperative swarm (learned target allocation) | Accepted |
| [0023](0023-population-self-play.md) | Population self-play (fictitious play vs a pool) | Accepted |
| [0024](0024-rl-trained-on-estimated-observations.md) | Training RL directly on estimated observations | Accepted |
| [0025](0025-intercept-league-elo.md) | INTERCEPT League — Bradley-Terry/Elo over laws and evaders | Accepted |
| [0026](0026-mode-adaptive-guidance.md) | Mode-adaptive guidance (IMM belief arbitrates the law) | Accepted |
| [0027](0027-pincer-coverage-guidance.md) | Pincer coverage guidance (a pair covers both escape branches) | Accepted |
| [0028](0028-swarm-penetration-and-asset-value-defense.md) | Coordinated swarm penetration tactics + asset-value layered defense | Accepted |
