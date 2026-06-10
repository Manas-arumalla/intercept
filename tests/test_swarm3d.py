"""Tests for the 3-D threat-trajectory library and the dimension-aware swarm/WTA."""

from __future__ import annotations

import numpy as np

from intercept.adversary.threats import THREAT_PROFILES
from intercept.core import Entity, PointMass3D
from intercept.core.aero import G0
from intercept.guidance import augmented_pn_3d
from intercept.multiagent.assignment import intercept_time_cost, weapon_target_assignment
from intercept.multiagent.swarm import MultiEngagement


def test_threat_profiles_return_finite_3d_accel():
    own = np.array([8000.0, 1000.0, 4000.0, -650.0, 50.0, -20.0])
    for name, ctrl in THREAT_PROFILES.items():
        a = np.asarray(ctrl(2.0, own, {}), dtype=float)
        assert a.shape == (3,), name
        assert np.all(np.isfinite(a)), name


def test_intercept_time_cost_is_3d_aware():
    # Interceptor at origin; target offset purely in altitude, closing downward.
    interceptor = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1000.0])  # climbing at 1000 m/s
    target = np.array([0.0, 0.0, 5000.0, 0.0, 0.0, -200.0])  # descending overhead
    t3 = intercept_time_cost(interceptor, target, 1000.0, ndim=3)
    assert np.isfinite(t3) and t3 > 0  # 3-D: catchable overhead
    # In 2-D the altitude is ignored (both at the same ground point) -> degenerate/instant.
    t2 = intercept_time_cost(interceptor, target, 1000.0, ndim=2)
    assert t2 != t3  # dimension actually matters


def test_wta_3d_assignment_distinct():
    states_i = [np.array([0.0, 0.0, 0.0, 0, 0, 1000.0]), np.array([0.0, 0.0, 0.0, 0, 0, 1000.0])]
    states_t = [
        np.array([3000.0, 0.0, 1000.0, -200.0, 0, 0]),
        np.array([0.0, 3000.0, 4000.0, 0, -200.0, 0]),
    ]
    amap = weapon_target_assignment(states_i, states_t, ndim=3)
    assert set(amap.values()) == {0, 1}  # one interceptor per threat


def test_3d_swarm_intercepts_diverse_threats():
    rng = np.random.default_rng(0)
    profiles = list(THREAT_PROFILES.items())[:3]  # 3 diverse threats (fast test)
    threats = []
    for j, (_, ctrl) in enumerate(profiles):
        az = np.radians(70 + 20 * j)
        pos = np.array([9000 * np.cos(az), 9000 * np.sin(az), 3500.0])
        aim = -pos / np.linalg.norm(pos)
        threats.append(
            Entity(
                f"T{j}",
                PointMass3D(a_max=30 * G0),
                np.array([*pos, *(700.0 * aim)]),
                controller=ctrl,
                role="target",
            )
        )
    centroid = np.mean([t.state[:3] for t in threats], axis=0)
    ints = []
    for i in range(3):
        p = np.array([rng.uniform(-400, 400), rng.uniform(-400, 400), 0.0])
        aim = (centroid - p) / np.linalg.norm(centroid - p)
        ints.append(
            Entity(
                f"I{i}",
                PointMass3D(a_max=50 * G0),
                np.array([*p, *(1000.0 * aim)]),
                role="interceptor",
            )
        )
    res = MultiEngagement(
        ints,
        threats,
        lambda t: augmented_pn_3d(t, N=4.0),
        dt=0.02,
        t_max=25.0,
        kill_radius=50.0,
        reassign_every=20,
    ).run()
    assert res.leakers == 0  # all diverse threats intercepted
