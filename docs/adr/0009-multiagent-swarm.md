# ADR-0009 — Multi-agent / swarm engagements (weapon-target assignment + N-vs-M)

- **Status:** Accepted
- **Date:** 2026-06-08

## Context

Layered/area defense is many-vs-many: a battery of interceptors against a salvo of threats. This
needs (a) **resource allocation** — who shoots what — and (b) an engagement loop that runs many
bodies, re-allocates as the situation changes, and scores intercepts/leakers. The 1v1 `Engagement`
doesn't cover this.

## Decision

Add `intercept.multiagent` reusing the existing dynamics, guidance, and segment-distance intercept
test:

1. **Weapon-Target Assignment** (`assignment.py`) — the **Hungarian algorithm**
   (`scipy.optimize.linear_sum_assignment`) on a **predicted time-to-intercept** cost matrix
   (via the existing `intercept_point` solver). Optimal, O((N+M)³), cheap to re-solve each replan.
   Surplus interceptors (N>M) get redundant coverage of their cheapest target; surplus threats
   (M>N) are picked up on later replans as others are killed.
2. **`MultiEngagement`** (`swarm.py`) — N interceptors vs M targets: periodically re-solves the
   assignment, points each interceptor's guidance (via a `guidance_factory(target_name)`) at its
   assignment, integrates all living bodies, and detects intercepts by the closest approach within
   each step (tunnelling-proof). Killed threats / expended interceptors freeze; ends when all
   threats are down, all interceptors spent, or time-out. Logs every track + kill events.
3. **Visualization** (`viz/swarm_plots.py`) — static engagement map and an animated GIF with kill
   bursts.

## Consequences

- (+) Area-defense engagements work and read well: 8 interceptors vs an 8-threat fan → 8/8
  intercepted, with the WTA producing sensible spatial pairings (demo `p7_swarm_defense`).
- (+) Re-assignment makes the defense adaptive (re-engage after kills, pick up leakers when possible).
- (+) Built entirely on the existing interfaces — any dynamics (L0/L2/L3) and any guidance law slot in.
- (−) The default WTA cost is single-target time-to-intercept. **Update:** a global-kill-probability
  objective was since added (`objective="kill_prob"` + `kill_probability` / `expected_leakers`): it
  maximizes the product of assigned kill probabilities and routes surplus interceptors to the
  most-likely-to-leak threats (shoot-look-shoot), cutting expected leakers. **Salvo timing** is now
  addressed by impact-time-control guidance (`ImpactTimeGuidance`, ADR-0016); interceptors are still
  one-shot.
- (−) **Cooperative/salvo guidance** (simultaneous arrival, impact-time control) and **MARL** swarm
  policies are noted as follow-ups; this ADR covers assignment + the N-vs-M engagement core.
