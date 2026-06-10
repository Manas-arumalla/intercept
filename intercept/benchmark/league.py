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
