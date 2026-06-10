"""Tests for coordinated swarm-penetration tactics and the asset-value defense counter."""

from __future__ import annotations

import numpy as np

from intercept.adversary.swarm_tactics import (
    SWARM_TACTICS,
    concentrated_axis,
    decoy_screen,
    simultaneous_tot,
    stream_raid,
)
from intercept.multiagent.defense import (
    make_value_allocator,
    predict_closest_approach,
    threat_value,
    value_prioritized_assignment,
)


def test_all_tactics_build_inbound_raids():
    """Every tactic produces threats whose velocity carries them toward the defended point."""
    for name, builder in SWARM_TACTICS.items():
        raid = builder(defended=np.zeros(3))
        assert raid.tactic == name
        assert len(raid.threats) >= 1
        for e in raid.real_threats:  # real threats close on the asset
            t_star, miss = predict_closest_approach(e.state, raid.defended)
            assert t_star > 0.0, name  # not already past
            assert miss < 1500.0, (name, miss)  # aimed at the asset (small predicted miss)


def test_decoy_screen_labels_and_geometry():
    """Decoys are labeled, miss the asset widely, and real threats are aimed at it."""
    raid = decoy_screen(n_real=5, n_decoy=7, defended=np.zeros(3))
    assert len(raid.decoys) == 7
    assert len(raid.real_threats) == 5
    for e in raid.threats:
        _, miss = predict_closest_approach(e.state, raid.defended)
        if e.name in raid.decoys:
            assert miss > 1500.0  # decoys fly wide
        else:
            assert miss < 1000.0  # real threats bore in


def test_threat_value_separates_real_from_decoy():
    """A real (low-miss) track scores far higher than a wide-miss decoy."""
    raid = decoy_screen(n_real=4, n_decoy=4, defended=np.zeros(3))
    real = next(e for e in raid.threats if e.name not in raid.decoys)
    decoy = next(e for e in raid.threats if e.name in raid.decoys)
    assert threat_value(real.state, raid.defended) > 5 * threat_value(decoy.state, raid.defended)


def test_predict_closest_approach_receding_target():
    """A target moving directly away has its closest approach now (t=0)."""
    state = np.array([1000.0, 0.0, 0.0, 300.0, 0.0, 0.0])  # at +x, moving +x (away)
    t_star, miss = predict_closest_approach(state, np.zeros(3))
    assert t_star == 0.0
    assert np.isclose(miss, 1000.0)


def test_value_allocator_prioritizes_real_threats_under_limited_magazine():
    """With fewer interceptors than tracks, the value allocator engages real threats, not decoys.

    Passes the true predicted miss directly so the test is deterministic."""
    raid = decoy_screen(n_real=3, n_decoy=5, defended=np.zeros(3))
    # Interceptors at the origin, fast enough to catch anything.
    istates = [np.array([0.0, 0.0, 0.0, 0.0, 1000.0, 0.0]) for _ in range(3)]
    tstates = [e.state for e in raid.threats]
    misses = [predict_closest_approach(e.state, raid.defended)[1] for e in raid.threats]
    amap = value_prioritized_assignment(istates, tstates, defended=raid.defended, misses=misses)
    real_idx = {i for i, e in enumerate(raid.threats) if e.name not in raid.decoys}
    engaged = set(amap.values())
    # All three interceptors should be committed to real threats (decoys de-prioritized).
    assert engaged <= real_idx or len(engaged & real_idx) >= 3


def test_make_value_allocator_is_stateful_min_miss():
    """The allocator tracks the running minimum predicted miss across calls (track history)."""
    raid = simultaneous_tot(n_real=3, defended=np.zeros(3))
    alloc = make_value_allocator(np.zeros(3))
    istates = [np.array([0.0, 0.0, 0.0, 0.0, 1000.0, 0.0]) for _ in range(3)]
    names = [e.name for e in raid.threats]
    a1 = alloc(istates, [e.state for e in raid.threats], names, 3)
    assert len(a1) == 3
    # A second call with the same tracks returns a valid assignment (no crash, all interceptors).
    a2 = alloc(istates, [e.state for e in raid.threats], names, 3)
    assert set(a2.keys()) == {0, 1, 2}


def test_concentrated_and_stream_counts():
    """Concentrated packs all-real into a narrow sector; stream splits into waves."""
    conc = concentrated_axis(n_real=10, defended=np.zeros(3))
    assert len(conc.real_threats) == 10 and len(conc.decoys) == 0
    stream = stream_raid(n_real=9, waves=3, defended=np.zeros(3))
    assert len(stream.real_threats) == 9
    waves = {e.name.split(":")[1] for e in stream.threats}  # "w0"/"w1"/"w2"
    assert waves == {"w0", "w1", "w2"}
