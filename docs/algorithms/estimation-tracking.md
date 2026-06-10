# Target tracking & state estimation (sensors, EKF, UKF, IMM)

Modules: [`intercept/sensors/`](./../intercept/sensors/),
[`intercept/estimation/`](./../intercept/estimation/),
[`intercept/guidance/estimating.py`](./../intercept/guidance/estimating.py).

## The sense → estimate → guide loop

Until P3, guidance laws were fed the *true* target state. P3 inserts a sensor and an estimator:
each step the interceptor takes a **noisy measurement** of the true target, a recursive filter
maintains a belief over the target state, and the **estimate** is handed to the (unchanged)
guidance law. Because guidance reads the target from the `world` snapshot, swapping truth → estimate
needs no change to the laws — exactly what the `Controller` contract was designed for.

```
true target ─▶ Sensor.measure(rng) ─▶ z ─▶ Estimator.predict/update ─▶ x̂ ─▶ Guidance ─▶ a_cmd
```

## Sensors ([`sensors/`](./../intercept/sensors/))

A `Sensor` maps (sensor position, target position) → measurement, with additive Gaussian noise
drawn from an explicit RNG (reproducible per trial). Defined on target *position* only, so they are
independent of the tracker's state dimension; angle components use a wrapped `residual`.

| Sensor | Measurement | Notes |
|---|---|---|
| `Radar` | `[range, bearing]` | Position fully observable from one look; nonlinear → EKF/UKF. Has `invert` for tracker init. |
| `IRSeeker` | `[bearing]` | Angles-only: range unobservable from a single look; needs parallax. |

## Motion models ([`estimation/models.py`](./../intercept/estimation/models.py))

Shared 6-D state `[x, y, vx, vy, ax, ay]`:
- **NCV** (nearly-constant-velocity): acceleration uncoupled, low process noise — the quiescent model.
- **NCA** (nearly-constant-acceleration): full CA kinematics, white-noise-jerk process noise — the maneuver model.

Each returns `(F, Q)` for a step `dt` and process spectral density `q`.

## Filters ([`estimation/kalman.py`](./../intercept/estimation/kalman.py), [`imm.py`](./../intercept/estimation/imm.py))

The motion model is linear, so prediction is an exact Kalman predict; the filters differ in the
**nonlinear measurement** update:

- **EKF** — linearizes the measurement via its analytic Jacobian; Joseph-form covariance update.
- **UKF** — unscented transform (sigma points) through the measurement function; no Jacobian, with
 angle-safe residual-from-reference handling of the bearing.
- **IMM** — a bank of filters (NCV + NCA) mixed each step through a Markov mode-transition matrix
 (Blom & Bar-Shalom). Mode probabilities track which model is active; the combined estimate
 moment-matches the bank. `make_cv_ca_imm(..)` builds the standard 2-model IMM.

## Estimating guidance ([`guidance/estimating.py`](./../intercept/guidance/estimating.py))

`EstimatingGuidance(target, sensor, estimator_factory, guidance, rng)` is a `Controller` wrapping
the whole loop. It initializes the estimator from the first measurement (radar `invert`), then each
step measures → predict → update → feeds the estimated target state to the underlying law.

## Validated behavior (P3 tests + demo)

- **Radar Jacobian** matches finite differences; **noise statistics** match the configured σ.
- **EKF & UKF converge** on a noisy constant-velocity target (< 30 m residual); velocity recovered.
- **NEES consistency** — average normalized error squared stays within a few × the state dimension.
- **IMM** mode probability favors the NCA model under a sustained maneuver.
- **EstimatingGuidance** still intercepts a maneuvering target through a noisy seeker.

### Headline figures

- `gallery/figures/p3_imm_tracking.png` — target turns at t=3 s: EKF(NCV) error diverges to ~350 m while
 EKF(NCA) and IMM stay ~9–10 m; the IMM mode probability switches to the maneuver model at the turn.
- `gallery/figures/p3_guidance_vs_noise.png` — **estimation-coupled study**: P(intercept) is flat at 1.0 up
 to radar σ_r ≈ 50 m, falls to 0.75 at 100 m, and collapses to 0.15 at 200 m (median miss 5 → 80 m).
 Guidance is robust to moderate sensing error but has a noise threshold beyond which it fails.

## Limitations / next

- Single-target, single-sensor; no clutter/false-alarms or data association (JPDA/MHT) yet.
- Angles-only (IR) guidance needs an observability-aware initialization (parallax) — demonstrated
 for tracking, not yet wired into the guidance loop.
- Multi-sensor fusion (radar + IR, covariance intersection) is the natural P3+ extension.

## References

- Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking and Navigation*.
- Blom & Bar-Shalom, "The interacting multiple model algorithm…", IEEE TAC 1988.
- Julier & Uhlmann, "Unscented filtering and nonlinear estimation", Proc. IEEE 2004.
