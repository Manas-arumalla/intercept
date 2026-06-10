"""Test the centralized MARL swarm env (skipped if gymnasium is absent)."""

from __future__ import annotations

import numpy as np
import pytest

from intercept.envs.swarm_env import has_gym

pytestmark = pytest.mark.skipif(not has_gym(), reason="gymnasium not installed")


def test_swarm_env_shapes_and_bookkeeping():
    from intercept.envs import CentralizedSwarmEnv

    env = CentralizedSwarmEnv(n_int=3, n_threat=5)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3 * 5 * 4 + 3 + 5,)
    assert env.action_space.shape == (3 * 5,)
    done = False
    info = {}
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(r)
        done = term or trunc
    # Every threat ends either intercepted or leaked, and counts never exceed the threat count.
    assert 0 <= info["hits"] <= 5 and 0 <= info["leaks"] <= 5
    assert info["hits"] + info["leaks"] <= 5


def test_hungarian_beats_random_leakers():
    """Sanity: the analytic allocator should be no worse than random on average."""
    from intercept.envs import CentralizedSwarmEnv
    from intercept.multiagent.assignment import weapon_target_assignment

    def hungarian(env):
        li, lt = np.where(env.alive_i)[0], np.where(env.alive_t)[0]
        pref = -np.ones((env.n_int, env.n_threat), dtype=np.float32)
        if li.size and lt.size:
            amap = weapon_target_assignment(
                [env.ipos[i] for i in li], [env.tpos[j] for j in lt], ndim=2
            )
            for k, i in enumerate(li):
                pref[i, lt[amap[k]]] = 1.0
        return pref.reshape(-1)

    def run(pol):
        leaks = []
        for s in range(12):
            env = CentralizedSwarmEnv(n_int=3, n_threat=5)
            env.reset(seed=s)
            done = False
            info = {}
            while not done:
                a = pol(env) if pol else env.action_space.sample()
                _, _, term, trunc, info = env.step(a)
                done = term or trunc
            leaks.append(info["leaks"])
        return float(np.mean(leaks))

    assert run(hungarian) <= run(None) + 1e-9
