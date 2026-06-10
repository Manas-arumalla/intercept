# ADR-0028 — Coordinated swarm penetration tactics + asset-value layered defense

- **Status:** Accepted
- **Date:** 2026-06-10

## Context

The swarm work so far (P7, ADR-0021) modeled threats as a fan of *independent* trajectories and
defended with time-minimizing Hungarian WTA. But a real saturation raid's danger is not just
*numbers* — it is **coordination** designed to defeat the defender's allocation logic. Open-source
saturation/salvo doctrine (Wikipedia *Saturation attack* / *Swarming (military)*; cooperative-salvo
and naval decoy-screen literature) describes a few recurring tactics. The goal here was (a) to model those
tactics as kinematic raid geometries, and (b) to build and measure a defensive **counter**.

Scope note: these are **engagement-geometry** descriptions only (launch ranges, axes, arrival
timing, decoys) — no targeting, sensing, warhead, or detection-evasion content (charter).

## Decision

**Attacker — `intercept.adversary.swarm_tactics`** (four tactics, each a `Raid` builder):

- `simultaneous_tot` — **time-on-target**: equal range+speed ⇒ all real threats arrive together, so
  the defender cannot engage them sequentially.
- `decoy_screen` — real threats interleaved with **decoys** aimed to miss the asset; a defender that
  engages every track wastes its magazine on chaff.
- `concentrated_axis` — a **saturation point**: threats packed into one narrow azimuth sector to
  exceed local handling capacity.
- `stream_raid` — **sequential waves** that bait and drain the magazine (shoot-look-shoot cycle).

**Defender — `intercept.multiagent.defense`** (the counter, plugged into `MultiEngagement` via a new
`allocator` hook):

1. **Impact-point prediction** — extrapolate each track to its closest approach to the asset
   (`predict_closest_approach`); predicted miss + time-to-asset are the threat-evaluation features.
2. **Decoy de-prioritization** — a smooth lethality gate on predicted miss; wide-miss tracks score
   ~0 (`threat_value`).
3. **Value-prioritized, capacity-aware assignment** — Hungarian on `−log(value·catchability)`, so
   with fewer interceptors than tracks the optimizer *leaves the decoys unengaged*
   (`value_prioritized_assignment`); surplus interceptors reinforce the leakiest real threat.

**The subtlety that drove the design.** A *single-snapshot* impact predictor cannot tell a
hard-maneuvering real threat from a decoy: a weaving/jinking inbound's instantaneous heading swings
make its predicted miss oscillate to 1.5–3 km — overlapping the decoys (measured directly). The fix
is a **stateful** evaluator (`make_value_allocator`) that tracks each named track's **running
minimum predicted miss**: a real threat's minimum collapses toward zero as it bores in, while a
decoy's floors at its offset. This mirrors real fire-control doctrine (constant-bearing /
track-history), including its corollary — the picture is *ambiguous at long range*, so the defender
should not confidently classify or commit its full magazine until tracks have closed.

## Result (40 trials/tactic, 5-interceptor magazine, realistic ~Mach 3 vs ~Mach 2)

Metric = **mean real-threat leakers** (decoys leaking is harmless). Lower is better.

| tactic | naive time-WTA | asset-value defense |
|---|---|---|
| simultaneous-TOT (8 real) | 3.00 | 3.08 |
| **decoy-screen (5 real, 7 decoys)** | **1.70** | **0.00** |
| concentrated-axis (10 real) | 5.00 | 5.12 |
| stream-raid (9 real) | 4.00 | 4.05 |

The asset-value defender **eliminates real-threat leakage against the decoy screen (1.70 → 0.00)**:
it spends all five interceptors on the five real threats instead of squandering ~1.7 shots on chaff.
On the three **all-real** tactics it is statistically tied with time-WTA — the expected "no free
lunch": discrimination only helps when there is something to discriminate; against undecoyed
over-saturation you simply need more magazine. Visuals: `gallery/figures/p35_penetration_bars.png`
(all four tactics), `gallery/figures/p35_swarm_penetration.png` (side-by-side decoy-screen 3-D:
naive wastes shots on grey decoys → 2 leakers; asset-value → 0),
`gallery/figures/p35_tactics_gallery.png` (2×2: every tactic's engagement under the asset-value
defense), GIFs `gallery/animations/p35_swarm_penetration.gif` (decoy screen) and
`gallery/animations/p35_tot_raid.gif` (simultaneous time-on-target raid). Experiment
`experiments/p35_swarm_penetration.py`; tests `tests/test_swarm_tactics.py`.

## Consequences

- (+) A *complete* attacker→defender story: named, literature-grounded penetration tactics and a
  measured counter, with the win (decoys) and the non-win (undecoyed saturation) both reported.
- (+) The `allocator` hook on `MultiEngagement` is reusable for any future doctrine (learned WTA,
  shoot-look-shoot variants) without forking the engagement loop; `target_names` enables stateful
  allocators.
- (−) The discriminator is purely kinematic (track history of predicted miss); a real system fuses
  RCS/IR/range-rate. Decoys here are open-loop kinematic shapes. The all-real saturation cases are
  magazine-bound, not algorithm-bound — closing them needs more interceptors or cooperative
  guidance, not better allocation. Dynamic shoot-look-shoot timing (hold-fire reserves keyed to the
  resolving picture) is the natural follow-up.
