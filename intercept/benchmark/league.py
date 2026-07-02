"""INTERCEPT League — Bradley-Terry / Elo skill ratings for guidance laws *and* evaders.

Benchmark tables answer "what is P(intercept) of law X vs evader Y?" — but with many laws and many
adversaries there is no single scale. The League treats every seeded engagement as a **match**
(intercept ⇒ the guidance law wins; escape ⇒ the evader wins) and fits a **Bradley-Terry model**
over all matches, placing pursuers and evaders on **one latent skill scale**, reported as familiar
**Elo** ratings (BT with base 10 / scale 400 *is* Elo; mean anchored at 1500).

The fit is the classic Zermelo/MM (minorization-maximization) iteration with additive smoothing
(``eps`` pseudo-wins per pairing) so undefeated participants get large-but-finite ratings. Ratings
are order-free (unlike sequential Elo updates) and reproducible.

Reference: Bradley & Terry (1952); Hunter (2004) "MM algorithms for generalized Bradley-Terry
models"; the BT↔Elo equivalence is standard.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def bradley_terry(
    names: list[str], wins: Array, *, eps: float = 0.5, tol: float = 1e-10, max_iter: int = 10_000
) -> dict[str, float]:
    """Fit Bradley-Terry strengths from a pairwise win-count matrix; return Elo-scale ratings.

    Parameters
    ----------
    names:
        Participant names (length ``n``).
    wins:
        ``(n, n)`` matrix; ``wins[i, j]`` = number of matches ``i`` beat ``j``.
    eps:
        Additive smoothing (pseudo-wins added to both directions of every *contested* pairing —
        i.e. pairs that actually played), keeping undefeated participants finite.

    Returns
    -------
    dict
        ``name -> Elo rating`` (BT log-strength × 400/ln 10, mean-anchored at 1500).
    """
    W = np.asarray(wins, dtype=float).copy()
    n = len(names)
    if W.shape != (n, n):
        raise ValueError(f"wins must be ({n},{n}), got {W.shape}")
    played = (W + W.T) > 0
    W[played] += eps
    np.fill_diagonal(W, 0.0)

    p = np.ones(n)
    total_wins = W.sum(axis=1)
    n_ij = W + W.T
    for _ in range(max_iter):
        denom = np.zeros(n)
        for i in range(n):
            mask = n_ij[i] > 0
            denom[i] = float(np.sum(n_ij[i, mask] / (p[i] + p[mask])))
        p_new = np.where(denom > 0, total_wins / np.maximum(denom, 1e-300), p)
        p_new = np.maximum(p_new, 1e-300)
        p_new /= np.exp(np.mean(np.log(p_new)))  # normalize: geometric mean 1
        if np.max(np.abs(np.log(p_new) - np.log(p))) < tol:
            p = p_new
            break
        p = p_new

    elo = 1500.0 + 400.0 / np.log(10.0) * np.log(p)
    return {nm: float(e) for nm, e in zip(names, elo, strict=True)}


def elo_expected_score(r_a: float, r_b: float) -> float:
    """Expected win probability of ``a`` vs ``b`` under the fitted Elo ratings."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def bradley_terry_bootstrap(
    names: list[str],
    pairing_outcomes: dict[tuple[str, str], list[bool]],
    *,
    n_replicates: int = 1000,
    confidence: float = 0.95,
    eps: float = 0.5,
    rng: np.random.Generator | None = None,
) -> dict[str, tuple[float, float]]:
    """Non-parametric bootstrap confidence intervals on Bradley-Terry Elo ratings.

    For each replicate, resamples every (pursuer, evader) pairing's match outcomes
    *with replacement*, rebuilds the win-count matrix, and refits BT. The reported
    interval is the ``(alpha/2, 1-alpha/2)``-percentile across replicates.

    Parameters
    ----------
    names:
        All participant names (length ``n``).
    pairing_outcomes:
        ``{(pursuer_name, evader_name): [True, False, ...]}`` — per-pairing binary
        match outcomes (``True`` = pursuer/guidance-law intercepted).
    n_replicates:
        Bootstrap replicates; 1 000 gives stable 95 % CIs.
    confidence:
        CI level (default 0.95 → 2.5th–97.5th percentile).
    eps:
        Additive smoothing forwarded to :func:`bradley_terry`.
    rng:
        Seeded generator for reproducibility; ``None`` → ``np.random.default_rng()``.

    Returns
    -------
    dict
        ``name -> (ci_lo, ci_hi)`` Elo rating bounds.
    """
    if rng is None:
        rng = np.random.default_rng()
    idx = {nm: i for i, nm in enumerate(names)}
    n = len(names)
    alpha = 1.0 - confidence
    rep_ratings: dict[str, list[float]] = {nm: [] for nm in names}

    for _ in range(n_replicates):
        W = np.zeros((n, n))
        for (pname, ename), results in pairing_outcomes.items():
            if not results:
                continue
            pi = idx.get(pname)
            ei = idx.get(ename)
            if pi is None or ei is None:
                continue
            m = len(results)
            arr = np.asarray(results, dtype=bool)
            sample_idx = rng.integers(0, m, size=m)
            n_wins = int(arr[sample_idx].sum())
            W[pi, ei] += n_wins
            W[ei, pi] += m - n_wins
        elo_rep = bradley_terry(names, W, eps=eps)
        for nm, r in elo_rep.items():
            rep_ratings[nm].append(r)

    lo_pct = alpha / 2 * 100.0
    hi_pct = (1.0 - alpha / 2) * 100.0
    return {
        nm: (
            float(np.percentile(rep_ratings[nm], lo_pct)),
            float(np.percentile(rep_ratings[nm], hi_pct)),
        )
        for nm in names
    }
