"""Seeded Monte-Carlo over randomized engagement geometries.

The **fairness invariant** (ADR-0003): trial ``i`` derives its RNG from a fixed
``SeedSequence(seed).spawn(n)``, so trial ``i`` produces the *same* sampled initial conditions and
target behavior for *every* guidance law. Differences in outcome are therefore attributable to the
guidance law alone, never to luck. RNG is never sampled inside the engagement loop — only here, up
front, when the scenario is sampled.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from intercept.benchmark.scenario import ParametricScenario
from intercept.core.engagement import EngagementResult

#: A guidance factory maps a target-entity name to a fresh controller for one engagement.
GuidanceFactory = Callable[[str], object]


def run_montecarlo(
    scenario: ParametricScenario,
    guidance_factory: GuidanceFactory,
    *,
    n_trials: int = 200,
    seed: int = 0,
) -> list[EngagementResult]:
    """Run ``n_trials`` engagements of ``scenario`` under ``guidance_factory``.

    Parameters
    ----------
    scenario:
        The engagement distribution to sample from.
    guidance_factory:
        ``factory(target_name) -> Controller``; called fresh per trial.
    n_trials:
        Number of Monte-Carlo trials.
    seed:
        Base seed; trial ``i`` uses ``SeedSequence(seed).spawn(n_trials)[i]``.

    Returns
    -------
    list[EngagementResult]:
        One result per trial, in trial order.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    child_seeds = np.random.SeedSequence(seed).spawn(n_trials)
    results: list[EngagementResult] = []
    for ss in child_seeds:
        rng = np.random.default_rng(ss)
        spec = scenario.sample(rng)
        results.append(spec.build(guidance_factory).run())
    return results
