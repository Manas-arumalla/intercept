# ADR-0021 — Diverse-threat swarm-vs-swarm + 3-D weapon-target assignment

- **Status:** Accepted
- **Date:** 2026-06-09

## Context

The P7 swarm used a fan of near-identical inbound threats — visually flat and not showing the
*variety* of trajectories a real layered defense must handle. The ask: a bigger, more impressive
swarm-vs-swarm with **distinct realistic threat profiles** and a cinematic animation. Building it
surfaced a latent bug: the weapon-target assignment (and `intercept_point`) were **2-D-only**
(`state[:2]`), so a 3-D swarm assigned/scored on the ground projection and mostly missed.

## Decision

- **Threat-trajectory library** `intercept.adversary.threats` — public, textbook **kinematic**
  profiles (no targeting/sensing/warhead/evasion-tooling, per scope): `cruise_weave`, `sea_skimming`
  (pop-up + dive), `lofted_ballistic`, `terminal_spiral`, `diving_jink`, `boost_glide`, exposed as
  `THREAT_PROFILES`.
- **Dimension-generic WTA:** `intercept_point`, `intercept_time_cost`, `cost_matrix`,
  `kill_probability[_matrix]`, and `weapon_target_assignment` take an `ndim` (2 default → unchanged
  2-D behavior); `MultiEngagement` infers it from `dynamics.control_dim`.
- **Cinematic swarm animation** `viz.animate_swarm_3d_modern` (dark theme, neon glow trails per
  entity, per-kill intercept flashes, defended-point marker, orbiting camera) for a
  `MultiEngagementResult`. Showcase `experiments/p28_swarm_showcase.py`.

## Consequences

- (+) A defended point survives a **two-wave saturating raid of 12 threats across 6 distinct
  profiles vs. 12 interceptors → 12/12 intercepted, 0 leakers**, at realistic comparable speeds (threats ~Mach 2,
  interceptors ~Mach 3). Static labeled figure `gallery/figures/p28_swarm_showcase.png` + cinematic
  `gallery/animations/p28_swarm_showcase.gif`. Tests in `tests/test_swarm3d.py`.
- (+) The 3-D WTA fix also benefits any future 3-D multi-agent work; 2-D defaults keep all prior
  results/tests intact.
- (−) Threat profiles are open-loop kinematic shapes (not closed-loop terrain-following or homing);
  interceptor coordination is still optimization-based WTA, not learned (that is the MARL follow-up).
