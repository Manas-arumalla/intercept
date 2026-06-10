# Realistic engagements (L2 fidelity): physics, evasion, and why guidance must be smart

Modules: [`intercept/core/aero.py`](./../intercept/core/aero.py),
[`intercept/adversary/evasive.py`](./../intercept/adversary/evasive.py),
[`scenarios/realistic/`](./../scenarios/realistic/). Decision:.

## Why

At L0 the interceptor won by being faster against slow, near-straight targets. L2 makes the
engagement *realistic and hard*: comparable supersonic speeds, finite g, response lag, energy
limits, and fast, high-g, **deceptive** targets. The interceptor must now win through prediction,
robustness, and control.

## L2 plant — `AeroMissile2D` (planar 3-DOF, Zarchan-style)

State `[x, y, vx, vy, ax, ay]` (the last two carry autopilot lag). Physics:

| Effect | Model | Why it matters |
|---|---|---|
| Gravity | `−g` on vertical accel | trajectories arc; sustained turns fight gravity |
| Parasitic drag | `−k_drag · V²` along velocity | speed bleeds; no free constant-speed cruise |
| **Induced drag** | `−k_induced · |a_lat|²` along velocity | **pulling g costs energy** — a hard-jinking target slows down |
| g-limit | lateral command clipped to `a_max` | finite turn capability |
| Autopilot lag | `ȧ = (a_cmd⊥ − a)/τ` | commanded ≠ achieved — forces lead, breaks naive terminal correction |

Only the lateral (⟂-velocity) component of a guidance command is achievable as body lift. The
`[x,y,vx,vy,…]` layout means **all existing guidance/estimation/RL/benchmark code is unchanged**
(it reads `state[:4]`); scenarios opt in with `model: aero`.

## Adversaries — `adversary/evasive.py`

- `hard_turn` — sustained max-g break (spiral).
- `random_telegraph(accel, mean_switch, rng)` — bang-bang jink with **random** sign flips (seeded,
 reproducible). Deliberately unpredictable.
- `reactive_break(pursuer, accel, trigger_range, base)` — **closed-loop**: cruises a baseline
 maneuver, then breaks max-g *away* from the interceptor once it closes inside the trigger range —
 a last-ditch defensive break exploiting the interceptor's lag and g-limit.

## Realistic suite (`scenarios/realistic/`)

Interceptor ~Mach 3.5 (40 g, τ=0.2 s) vs target ~Mach 2.2–2.4 (5–30 g, τ=0.3 s) — only ~1.5×
speed, so no speed-win. R1 supersonic crossing (light); R2 20 g weave; R3 25 g random-telegraph
jink; R4 30 g reactive break.

## Result — intelligence beats speed (150 trials/cell, `p8_realistic_benchmark.py`)

P(intercept) by guidance law:

| Scenario | True PN | Augmented PN | Optimal (OGL) | Sliding-mode |
|---|---|---|---|---|
| R1 supersonic crossing | 1.00 | 1.00 | 1.00 | 1.00 |
| R2 weave 20 g | 1.00 | 1.00 | 1.00 | 1.00 |
| **R3 telegraph jink** | **0.56** | 0.79 | 0.83 | **1.00** |
| **R4 reactive break** | **0.21** | 1.00 | 1.00 | 1.00 |

Figure: `gallery/figures/p8_realistic_benchmark.png`. Animations: `gallery/animations/anim_realistic_jink.gif`,
`anim_realistic_reactive.gif`.

**Reading it.** Steady maneuvers (R1/R2) are handled by a capable interceptor. But an *unpredictable*
jink halves plain PN's success (0.56), and a *reactive last-ditch break* collapses it to 0.21 —
while **Augmented PN** (target-acceleration feedforward), **Optimal/OGL**, and especially
**Sliding-mode** (designed for unknown maneuvers) recover to 0.79–1.00. The interceptor now succeeds
through prediction and robustness, not speed — exactly the realism goal.

## Advanced 3-D complex-trajectory evasion (L3, speed parity)

`experiments/p14_advanced_evasion.py` is the most demanding showcase: **both** missiles are
`RealisticMissile3D` (L3 — ISA atmosphere, boost–sustain–coast propulsion + mass burn-off, Mach +
induced drag, lift/dynamic-pressure-limited turning). The threat flies the kind of complex path real
maneuvering missiles use: a **lofted, descending** midcourse, a **tilted 3-D serpentine**
(`serpentine3d` — ground track snakes while altitude porpoises), and an **intensifying terminal
spiral** (`terminal_spiral` — a *closed-loop* corkscrew that tightens from a gentle weave to a
near-max-g helix as the interceptor closes, the way maneuvering-reentry vehicles and sea-skimming
anti-ship missiles defeat endgame interception). `combine` sums these into one commanded vector; the
plant clips it to the physics turn limit, so every g costs energy.

**Realistic speed parity (no speed cheat —).**
Speeds are deliberately *comparable*: the interceptor launches ~Mach 1.2, boosts to ~Mach 3, then
coasts to **~Mach 2.6 at the merge**; the threat is a fast **~Mach 3** missile that bleeds to
**~Mach 2** under hard maneuvering — only a **~37 %** closing edge (a real SAM-vs-supersonic figure).
A propulsion sweep confirms that shrinking the interceptor's motor further makes it *miss*: it flies
on near-minimum energy and wins by an efficient **Augmented-PN lead**, not by out-running the target.

**Showing the maneuvering faithfully.** At supersonic speed a real "spiral" is a long, *thin* helix
(turn radius ≪ distance per revolution), so an equal-aspect 3-D view at engagement scale looks
near-straight — a tight visible corkscrew would need hundreds of g (a cheat). The complexity is
shown the way it is quantified in practice: the **analysis panel** (`gallery/figures/p14_advanced_analysis.png`)
overlays the top-down serpentine, the altitude loft/porpoise, the closing range, and the target's
**achieved g against its physics-available turn limit** (the achieved g is visibly *clipped* by the
limit). The modern 3-D render/animation (`gallery/p14_advanced_modern.{png,gif}`) gives
the cinematic view.

**Result.** Showcase: INTERCEPT, miss **12.8 m**, t = 11.5 s; target peaks **17.2 g**, clipped to
~10.5 g near the merge as it slows and air thins. Robustness Monte-Carlo (60 randomized trials over
geometry + every maneuver parameter): **P(intercept) = 1.00** (95 % CI [0.94, 1.00]), median miss
13.7 m — the intercept is robust, not a single tuned shot.

## Limitations / next

- Parameters are representative (Zarchan-grounded), not vehicle-specific; tunable per scenario.
- L0-trained RL degrades on L2/L3 (different plant); retraining on the realistic plant is the open
 follow-up (RecurrentPPO / imitation warm-start from PN — see the RL notes).
- Speed-parity tuning is geometry-specific (the robustness sweep bounds, but does not eliminate, the
 dependence); a much longer/shorter engagement window would need re-tuning to stay near minimum energy.
