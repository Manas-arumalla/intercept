# Scope & limitations

A precise account of what INTERCEPT does and does not model, split into three groups: **deliberate
scope boundaries** (intentional — not defects), **resolved** engineering limitations, and **open
follow-ups**. Results are only as good as the model, so I state the model's edges plainly.

## Deliberate scope boundaries (by design — will not be "fixed")

These are intentional choices from the project's charter (see [README](../README.md) scope note and
[ADR-0002](adr/0002-point-mass-first-fidelity.md)); changing them would change what the project *is*.

- **Simulation only.** No hardware-in-the-loop, no real targeting/sensor data, no munitions or
  warhead modeling, no detection-evasion tooling. INTERCEPT studies *guidance and autonomy
  algorithms* as a controls/robotics research and teaching artifact.
- **Point-mass / kinematic dynamics (L0–L3).** Bodies are translational point masses; L2/L3 add
  aero, propulsion, g-limits, and autopilot lag, but there is **no 6-DOF rotational dynamics or
  thrust-vector control**. A 6-DOF rung *could* sit above L3 behind the same `Dynamics` interface,
  but it is explicitly out of the current scope.
- **Public textbook algorithms.** Everything traces to open references (Zarchan, Siouris, Shneydor,
  Bar-Shalom, Isaacs, Jeon/Lee/Tahk, Silver/Johannink). No proprietary or operational methods.
- **Representative, not flight-validated, parameters.** Masses, thrust curves, and drag coefficients
  are physically plausible and self-consistent, not tied to any real vehicle.

## Resolved (previously listed as limitations)

- **3-D across every guidance paradigm.** PN/APN, Optimal (OGL-3D), Sliding-mode (SMG-3D), MPC
  (`MPCGuidance3D`), and **RL** (`InterceptionEnv3D` + `RLGuidance3D`) all run in 3-D
  (ADR-0007/0012/0014).
- **3-D estimation + sense→estimate→guide loop.** Dimension-generic EKF/UKF/IMM + `Radar3D`, and
  `EstimatingGuidance` now closes the loop in 3-D (ADR-0013).
- **L3 in the benchmark.** `ParametricScenario(model="realistic")` and `ParametricScenario3D` run the
  Monte-Carlo benchmark on the realistic plant (ADR-0008/0012).
- **Realistic-plant RL.** From-scratch PPO collapses on the lagged/gravity plant; residual-PN
  resolves it in 2-D and 3-D, a recurrent APN-residual **beats** the PN family on the 2-D jink, and
  the recurrent APN-residual 3-D reaches **full parity (1.00/1.00/1.00)** with the classical laws
  (ADR-0011/0014). Every guidance paradigm now runs in 3-D.
- **Statistical rigor.** Pairwise law comparisons use a paired bootstrap (CI + p-value), not just
  point estimates.
- **Cooperative salvo.** `ImpactTimeGuidance` (ADR-0016) synchronizes a battery's arrival to a
  commanded impact time (4 interceptors within 0.14 s vs 1.28 s under PN) — the simultaneous-arrival
  cooperative-guidance item, validated.
- **Gain sensitivity.** `experiments/p21_gain_sensitivity.py` sweeps the navigation constant
  N ∈ {2..7}; N=4 sits on the robust capture/effort plateau (no longer an unjustified "best-effort").
- **Global-kill-probability WTA.** `weapon_target_assignment(objective="kill_prob")` maximizes global
  kill probability and routes surplus interceptors to the most-likely-to-leak threats
  (shoot-look-shoot), cutting expected leakers vs. the time-only assignment.
- **Estimation extras.** 3-D IMM **in the guidance loop**, moving-platform (seeker-on-interceptor)
  sensing, and an `INSError` platform-navigation model (bias + drift) are in place — the seeker
  measures true geometry while the filter places the target using the INS-corrupted platform
  position, so the estimate (and miss) degrade realistically with nav error.

## Open follow-ups (genuine, not yet done)

Ranked roughly by value, and listed openly.

- **Full PSRO self-play.** A learned evader (ADR-0015), one arms-race round (ADR-0017), the
  non-transitive cross-table (ADR-0020), and **population/fictitious play** that fixes the forgetting
  (ADR-0023 — pool-trained interceptor recovers the forgotten opponent 0.18→0.78) are all done. A
  full **PSRO** (meta-game payoff matrix, best-response to the Nash mixture, growing population) for a
  tight equilibrium is the remaining step.
- **MPC/RL in the heavy Monte-Carlo benchmark.** Both are demoed/tested but kept out of the large
  sweeps (IPOPT solves / rollout cost); a budgeted subset run is possible.
- **Training RL directly on estimated observations.** *(Done: 3-D IMM in the guidance loop,
  moving-platform seeker-on-interceptor sensing, and an `INSError` platform-navigation model — see
  the Resolved section. The estimator also closes the loop for RL at deploy time via
  `EstimatingGuidance(RLGuidance(...))`; training a policy directly on estimated observations is the
  remaining piece.)*
- **Decentralized MARL.** A *centralized* learned allocator now exists (`CentralizedSwarmEnv`,
  ADR-0022 — nearly matches Hungarian, 0.69 vs 0.65 leakers); fully decentralized per-agent MARL
  (MAPPO/IPPO via PettingZoo, variable team sizes, partial observability) is the remaining step.
- **Presentation.** *(Done: a published `mkdocs-material` site + CI docs build, and an interactive
  Plotly 3-D replay — `viz.interactive_engagement_3d` → standalone HTML. A live multi-engagement
  Plotly/PyVista dashboard with controls is a possible further step.)*
