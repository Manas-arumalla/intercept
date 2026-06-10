# ADR-0019 — INS platform error + 3-D IMM in the guidance loop

- **Status:** Accepted
- **Date:** 2026-06-09

## Context

ADR-0013 added dimension-generic EKF/UKF/IMM and `Radar3D`, and `EstimatingGuidance` closed the
sense→estimate→guide loop. Two estimation gaps remained from the open list: (1) running the **IMM**
(not just single-model filters) *inside the 3-D guidance loop*, and (2) modeling the fact that the
seeker is mounted on a **moving platform whose own position is known only through an INS**, with error
— so far the interceptor had perfect self-knowledge.

## Decision

- `intercept.estimation.INSError(ndim, rng, bias_std, drift_rate)` — platform self-localization error
  as a per-trial constant bias plus linear drift (a first-order integrated-IMU-bias stand-in),
  sampled once from an injected RNG (reproducible; no sampling in the loop, per ADR-0003).
- `EstimatingGuidance(platform_error=...)` hook: the seeker measures the **true** relative geometry,
  but the filter places the target in the world frame using the **believed** (INS-corrupted) platform
  position — so the target estimate inherits the platform's navigation error.
- The seeker is already interceptor-mounted (`sensor_pos = own_state[:n]`), i.e. moving-platform
  sensing; `make_cv_ca_imm(ndim=3)` composes with `EstimatingGuidance` for a 3-D IMM-in-loop.

## Consequences

- (+) **3-D IMM closes the loop** and intercepts a maneuvering target at realistic comparable speeds
  (perfect INS: P(intercept) 1.00, ~20 m miss). The estimating loop needs convergence time, so the
  demo uses a long-range engagement (a fast/short engagement starves the filter — documented gotcha).
- (+) **Graceful, physical degradation with INS error:** median miss grows monotonically with drift
  (≈20 → 54 → 80 → 132 → 245 m at 0/2/5/10/20 m/s) — `experiments/p26_estimation_advanced.py`
  (figure `gallery/figures/p26_estimation_advanced.png`), tests in `tests/test_estimation.py`.
- (−) The INS model is a kinematic bias+drift, not a full strapdown IMU error state; coupling it into
  the filter's covariance (treating platform error as a correlated measurement noise) is future work.
