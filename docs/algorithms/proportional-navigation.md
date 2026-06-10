# Proportional Navigation (PN) and Augmented PN (APN)

Module: [`intercept/guidance/pn.py`](./../intercept/guidance/pn.py),
[`intercept/guidance/apn.py`](./../intercept/guidance/apn.py).
The classical guidance baseline.

## Idea

Proportional Navigation commands lateral acceleration proportional to the **line-of-sight (LOS)
rotation rate**. Nulling the LOS rate holds a constant bearing while range decreases
(constant-bearing-decreasing-range), which is a collision course. It is the canonical homing law
in essentially all fielded homing missiles and the reference baseline for this benchmark.

## Equations

Let `r = p_T − p_M` and `v = v_T − v_M` be relative position/velocity, `R = |r|`.

- **Closing speed:** `Vc = −(r · v) / R`
- **LOS rate (planar):** `λ̇ = (r_x v_y − r_y v_x) / R²`
- **Zero-effort miss:** `ZEM = r + v · t_go`, with `t_go = R / Vc`

Variants (selected by `variant=`):

| Variant | Command | Applied perpendicular to |
|---|---|---|
| `"true"` | `a = N · Vc · λ̇` | the **LOS** |
| `"pure"` | `a = N · Vc · λ̇` | the interceptor **velocity** |
| `"zem"` | `a = N · ZEM⊥ / t_go²` | (ZEM⊥ already ⟂ LOS) |

`N` is the navigation constant (typically 3–5; default 4).

**Augmented PN** adds a target-acceleration feedforward to True PN:

```
a = N · Vc · λ̇ + (N / 2) · a_T⊥
```

where `a_T⊥` is the target acceleration perpendicular to the LOS. Because target acceleration is
not a kinematic state, `AugmentedPN` **estimates it by finite-differencing** the observed target
velocity between calls (exact for piecewise-constant maneuvers under perfect information; on the
first call the feedforward is zero, so APN reduces to True PN).

## Usage

```python
from intercept.guidance import true_pn, zem_pn, AugmentedPN
from intercept.core import Entity, PointMass2D

interceptor = Entity("interceptor", PointMass2D(a_max=100.0),
 state=[0, 0, 600, 0], controller=true_pn("target", N=4.0))
```

Any guidance law conforms to the `Controller` contract `(t, own_state, world) -> control`, so it
drops straight into an `Entity` and runs against identical dynamics as every other paradigm.

## Validated behavior (P1 tests + demo)

- **Zero LOS-rate ⇒ zero command:** on a collision course PN coasts straight (`test_zero_los_rate_*`).
- **Constant-bearing intercept of a crossing target at low authority:** with `a_max = 100 m/s²`,
 True PN intercepts (miss ≈ 8 m) where pure pursuit lags into a ~103 m miss.
 Figure: `gallery/figures/p1_pn_vs_pursuit.png`.
- **APN < PN terminal miss vs. a maneuvering target:** against a 90 m/s² sustained turn, APN's
 feedforward reduces terminal miss (≈ 0.94 m → 0.75 m). Figure: `gallery/figures/p1_pn_vs_apn_maneuvering.png`.

## Limitations / notes

- Point-mass, perfect-information. LOS-rate sensitivity to noise and the effect of an
 estimator between sensor and guidance are studied from P3.
- `Vc → 0` (pure crossing, no closing) degenerates the magnitude form; ZEM form guards `Vc ≤ 0`.
- Pure PN can differ from True PN when the interceptor velocity is far from the LOS direction.

## References

- Zarchan, *Tactical and Strategic Missile Guidance* (7th ed.).
- Siouris, *Missile Guidance and Control Systems* (Springer, 2004).
- Shneydor, *Missile Guidance and Pursuit*.
- Cho & Kim, "Optimality of Augmented Ideal PN…", IEEE TAES 52(2), 2016 (basis for a future AIPN variant).
