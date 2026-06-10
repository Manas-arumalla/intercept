# Optimal, sliding-mode & MPC guidance

Modules: [`intercept/guidance/ogl.py`](../../intercept/guidance/ogl.py),
[`smg.py`](../../intercept/guidance/smg.py), [`mpc.py`](../../intercept/guidance/mpc.py).
All conform to the `Controller` contract and run against identical dynamics.

## Optimal Guidance Law (OGL / LQ)  — `OptimalGuidance`

Minimizing control energy ∫a² dt for the linearized intercept (zero terminal miss) gives the
closed-form feedback law

```
a = N' · ZEM⊥ / t_go²,     N' = 3        (+ (N'/2)·a_T⊥ if augmented)
```

with zero-effort miss `ZEM = r + v·t_go`. It is the optimal-control sibling of ZEM-PN (`N=3`); the
implementation lets `N'` be tuned and adds the optimal target-acceleration feedforward. **Validated:**
`OptimalGuidance(N'=3)` reproduces `zem_pn(N=3)` to machine precision; the augmented form reduces
terminal miss against a maneuvering target. Reference: Bryson & Ho; Zarchan.

## Sliding-Mode Guidance (SMG)  — `SlidingModeGuidance`

Takes the LOS rate as the sliding variable `s = λ̇`; driving `s → 0` enforces parallel navigation
robustly against an *unknown* maneuvering target:

```
a = (N · Vc · λ̇  +  η · tanh(λ̇ / Φ)) ⟂ LOS
```

The first term is the equivalent (PN-like) control; the second is the robust switching term, with a
`tanh` boundary layer `Φ` suppressing chattering. Choosing `η` above the target's lateral-accel
bound guarantees reaching the surface. **Validated:** zero command on a collision course; intercepts
crossing and weaving targets; robust to maneuvers. Reference: Shtessel et al., *Sliding Mode Control*.

## Nonlinear MPC  — `MPCGuidance` (requires CasADi)

Each replan solves a finite-horizon OCP (CasADi + IPOPT):

```
min  w_terminal·‖p_M(T) − p̂_T(T)‖²  +  w_effort·Σ‖a‖²   (+ impact-angle term)
s.t. point-mass dynamics,   ‖a_k‖ ≤ a_max,   horizon T = min(horizon, t_go)
```

and applies the first command (receding horizon). The solver runs every `replan_every` steps
(event/time-triggered replanning) with the command held in between — a real-time-plausible cadence.

**Impact-angle objective.** Setting `impact_angle_deg` adds a terminal cost matching the full
velocity vector `v_M(T) ≈ speed·[cosθ, sinθ]` (matching the *vector*, not just the perpendicular
component, avoids the ±θ sign ambiguity). This steers the interceptor to arrive along a chosen
heading — a constraint closed-form PN cannot express. There is a genuine **miss/angle trade-off**:
moderate requested angles are met with small miss; steep angles cost more miss under finite
authority and horizon (shown in `gallery/figures/p4_impact_angle.png`).

## Comparison (P4 demo, weaving target, a_max=150 m/s²)

| Law | Miss (m) | Effort (∫‖a‖² dt) |
|---|---|---|
| True PN (N=4) | 4.74 | 9 112 (cheapest) |
| Optimal (OGL, augmented) | **0.73** (best) | 13 638 |
| Sliding-mode | 4.22 | 11 935 |
| NMPC | 3.95 | 27 423 (most) |

All four intercept the weaver; the augmented OGL is most accurate, PN cheapest, MPC most expensive
(it pays compute + effort for constraint-handling generality). Figures:
`gallery/figures/p4_law_comparison.png`, `gallery/figures/p4_impact_angle.png`.

## 3-D variants

`optimal_guidance_3d` (`OptimalGuidance3D`) and `sliding_mode_3d` (`SlidingModeGuidance3D`)
generalize OGL and sliding-mode to three dimensions using the vector kinematics in
`core.frames3d`: OGL-3D uses the vector zero-effort miss ``ZEM = r + v·t_go`` and its component
perpendicular to the 3-D LOS (``a = N'·ZEM⊥/t_go²``, optional target-accel augmentation); SMG-3D
uses the **LOS-rate vector** ``Ω = (r×v)/|r|²`` as the sliding surface, with equivalent control
``N·(Ω×v_c)`` and switching ``η·tanh(|Ω|/Φ)`` along the same direction. In the 3-D benchmark (P17),
on a terminal spiral they extend the robustness ladder: True PN 0.19 → APN 0.82 → **OGL 0.98 → SMG
1.00**.

`MPCGuidance3D` (CasADi/IPOPT) is the 3-D NMPC: 6-state point-mass dynamics, a 3-D
acceleration-magnitude constraint, terminal-miss + effort cost re-solved every few steps, and an
optional 3-D `impact_dir` terminal-velocity objective. It intercepts a 3-D barrel-rolling target
(tested); like the 2-D NMPC it is kept out of the heavy Monte-Carlo benchmark (IPOPT solves are far
costlier than the closed-form laws). Only RL guidance now remains 2-D.

## Notes / limitations

- MPC uses Euler discretization and assumes constant-velocity target prediction over the horizon;
  IPOPT solves are ms-scale but far costlier than the closed-form laws (hence not in the full
  Monte-Carlo benchmark by default).
- `acados` (compiled SQP/RTI) is a future drop-in for faster real-time MPC; CasADi + IPOPT is the
  pip-installable baseline here.
