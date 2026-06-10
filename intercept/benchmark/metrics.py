"""Benchmark metrics over a set of engagements.

Standard interception-benchmark quantities (see Research Report §B): probability of intercept
with a **Wilson score confidence interval** (correct for small samples and extreme proportions,
unlike the normal approximation), miss-distance statistics, time-to-intercept (over successful
intercepts), and control effort. All functions are pure and operate on lists of
:class:`~intercept.core.engagement.EngagementResult`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from intercept.core.engagement import EngagementResult


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Parameters
    ----------
    successes, n:
        Number of successes and trials.
    z:
        Standard-normal quantile (1.96 ≈ 95% CI).

    Returns
    -------
    (lo, hi):
        Confidence-interval bounds in ``[0, 1]``. Returns ``(0.0, 1.0)`` for ``n == 0``.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class PairedComparison:
    """Paired bootstrap comparison of two algorithms (``a − b``) on identical seeded trials."""

    metric: str
    n_pairs: int
    mean_a: float
    mean_b: float
    diff: float  # mean(a) − mean(b)
    ci_lo: float
    ci_hi: float
    p_value: float  # two-sided bootstrap p-value, H0: no difference
    significant: bool  # CI excludes zero at the chosen alpha

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    metric: str = "metric",
) -> PairedComparison:
    """Paired bootstrap of the mean difference ``a − b`` for a per-trial metric.

    The benchmark fairness invariant makes trials **paired**: trial *i* has identical initial
    conditions and target behavior for both algorithms (same per-trial seed), so the difference is
    measured trial-by-trial and the only randomness is sampling which trials. We resample trial
    indices ``n_boot`` times, recompute the mean difference, and report a percentile confidence
    interval and a two-sided bootstrap p-value (H0: mean difference = 0; significant ⇔ CI excludes
    zero).

    Parameters
    ----------
    a, b:
        Per-trial metric values for the two algorithms in the *same trial order* (e.g.
        ``intercepted`` as 0/1, or miss distance). Must be equal length.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be equal-length 1-D paired arrays")
    n = a.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    d = a - b
    observed = float(d.mean())
    if n == 0:
        return PairedComparison(
            metric, 0, math.nan, math.nan, math.nan, math.nan, math.nan, 1.0, False
        )
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = (
        float(np.percentile(boot, 100 * alpha / 2)),
        float(np.percentile(boot, 100 * (1 - alpha / 2))),
    )
    frac_below = float(np.mean(boot <= 0.0))
    frac_above = float(np.mean(boot >= 0.0))
    p_value = min(1.0, 2.0 * min(frac_below, frac_above))
    return PairedComparison(
        metric,
        n,
        float(a.mean()),
        float(b.mean()),
        observed,
        lo,
        hi,
        p_value,
        significant=(lo > 0.0 or hi < 0.0),
    )


def compare_intercept(results_a, results_b, **kwargs) -> PairedComparison:
    """Paired-bootstrap comparison of P(intercept) between two algorithms' result lists.

    The two lists must come from the *same scenario and seed* (so trial *i* is the same engagement);
    this holds when both are produced by ``run_montecarlo(..., seed=S)`` with the same ``S``.
    """
    a = [1.0 if r.intercepted else 0.0 for r in results_a]
    b = [1.0 if r.intercepted else 0.0 for r in results_b]
    return paired_bootstrap(a, b, metric="p_intercept", **kwargs)


@dataclass
class MetricSummary:
    """Aggregate metrics for one (algorithm, scenario) cell of the benchmark."""

    n_trials: int
    n_intercept: int
    p_intercept: float
    p_intercept_lo: float
    p_intercept_hi: float
    miss_mean: float
    miss_median: float
    miss_p95: float
    tti_mean: float  # mean time-to-intercept over successful intercepts (nan if none)
    effort_mean: float  # mean interceptor control effort over all trials

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def summarize(results: Sequence[EngagementResult], z: float = 1.96) -> MetricSummary:
    """Aggregate a list of engagement results into a :class:`MetricSummary`.

    Miss-distance statistics are taken over *all* trials (closest approach is defined for every
    engagement); time-to-intercept is averaged over successful intercepts only.
    """
    n = len(results)
    if n == 0:
        return MetricSummary(0, 0, 0.0, 0.0, 1.0, math.nan, math.nan, math.nan, math.nan, math.nan)

    intercepts = [r for r in results if r.intercepted]
    n_hit = len(intercepts)
    p = n_hit / n
    lo, hi = wilson_interval(n_hit, n, z)

    miss = np.array([r.miss_distance for r in results], dtype=float)
    if intercepts:
        tti = np.array([r.intercept_time for r in intercepts], dtype=float)
    else:
        tti = np.array([])
    effort = np.array([r.control_effort(r.interceptor) for r in results], dtype=float)

    return MetricSummary(
        n_trials=n,
        n_intercept=n_hit,
        p_intercept=p,
        p_intercept_lo=lo,
        p_intercept_hi=hi,
        miss_mean=float(np.mean(miss)),
        miss_median=float(np.median(miss)),
        miss_p95=float(np.percentile(miss, 95)),
        tti_mean=float(np.mean(tti)) if tti.size else math.nan,
        effort_mean=float(np.mean(effort)),
    )
