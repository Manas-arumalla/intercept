# ADR-0013 — Three-dimensional estimation (dimension-generic filters + 3-D radar)

- **Status:** Accepted
- **Date:** 2026-06-08

## Context

The estimation suite (P3) was 2-D only: `ncv_model`/`nca_model` hard-coded a 6-D state
`[x,y,vx,vy,ax,ay]`, and the EKF/UKF assumed a 2-D position (`self.x[:2]`, `STATE_DIM`). With the
benchmark now spanning 3-D (ADR-0012), tracking should too — otherwise the sense→estimate→guide
story stops at 2-D.

## Decision

Make the estimators **dimension-generic** and add a 3-D sensor, without disturbing the 2-D path:

- **Motion models** (`models.py`): `ncv_model`/`nca_model` take `ndim` (default 2). The state is
  `[pos(n), vel(n), acc(n)]`; the F/Q are built per-axis, so `ndim=2` reproduces the old matrices
  exactly and `ndim=3` gives the 9-D state `[x,y,z,vx,vy,vz,ax,ay,az]`.
- **Filters** (`kalman.py`): EKF/UKF infer the state dimension from `x0` (`n = x.shape[0]`) and the
  position dimension from the sensor (`sensor.pos_dim`), slicing `self.x[:pos_dim]` for the
  measurement. No `STATE_DIM` dependency remains; 2-D behavior is unchanged.
- **Sensor base**: a `pos_dim` attribute (default 2) decouples sensors from the tracker's state size.
- **`sensors/radar3d.py` `Radar3D`**: range + azimuth + elevation with Gaussian noise, an analytic
  3×3 Jacobian (for the EKF), angle-safe residuals (wrap az/el), and `invert` (measurement → 3-D
  position, e.g. to seed a tracker). `pos_dim = 3`.
- **`experiments/p18_estimation_3d.py`**: a 3-D barrel-rolling target tracked from the noisy 3-D
  radar; a nearly-constant-velocity EKF (cannot represent the maneuver) vs a nearly-constant-
  acceleration UKF.

## Consequences

- (+) Tracking now works in 3-D on the project's own filters. On a barrel-rolling target the NCA UKF
  reaches **~52 m** position RMSE vs **~71 m** for the NCV EKF (which sits at the raw-measurement
  noise floor) — the 3-D analogue of the 2-D IMM-vs-EKF result. Figure
  `gallery/figures/p18_estimation_3d.png`. Tests in `tests/test_estimation.py` (+3).
- (+) The change is backward-compatible: every 2-D estimation test passes unchanged, and `STATE_DIM`
  remains exported for any external 2-D caller.
- (+) The **IMM is dimension-generic too** (`make_cv_ca_imm(..., ndim=3)`): on the 3-D barrel-roll
  it is the *best* tracker (~42 m, beating the NCA UKF ~52 m and NCV EKF ~71 m), mixing the
  NCV/NCA models as the rotating maneuver demands. Added to `p18_estimation_3d.py` and tested.
- (−) `Radar3D` measures from a fixed sensor position; moving-platform / seeker-on-interceptor 3-D
  estimation and an estimator-in-the-3-D-guidance-loop are follow-ups (the 2-D `EstimatingGuidance`
  already shows the pattern).
