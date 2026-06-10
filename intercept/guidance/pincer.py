"""Pincer coverage guidance: a coordinated pair covers both of the target's escape branches.

A target that will break hard in an **unpredictable direction** defeats a lone interceptor's lead —
whichever side it commits to, a coin-flip break goes the other way half the time. Two interceptors
flying the *same* law are no better: identical states ⇒ identical trajectories ⇒ perfectly
correlated outcomes. The pincer decorrelates them geometrically: each wraps a base law (PN/APN) but
steers at a **virtual aim-point offset laterally from the target**, one to each side:

    virtual_target = target ± β · range · ⟂̂_LOS

The offset decays with range (β·R → 0 at intercept), so each interceptor converges smoothly to its
plain base law while approaching from its *own side* — pre-positioned for one turn branch. Whichever
way the target breaks, one of the pair is already leading that branch. Pure geometry on top of any
`GuidanceLaw`; no communication needed beyond the initial side assignment.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from intercept.guidance.base import EPS, GuidanceLaw

Array = NDArray[np.float64]


class PincerGuidance(GuidanceLaw):
    """Wrap a base law to approach from one side (``side=+1`` left of the LOS, ``-1`` right).

    Parameters
    ----------
    target:
        Name of the target entity.
    base:
        The underlying :class:`GuidanceLaw` instance (e.g. ``true_pn(...)``/``AugmentedPN(...)``)
        bound to the same target.
    side:
        ``+1`` or ``-1`` — which side of the line-of-sight to bias the approach toward.
    beta:
        Lateral aim-point offset as a fraction of current range while split.
    r_split, r_merge:
        The offset is full above ``r_split``, tapers linearly, and is **zero inside ``r_merge``** —
        so each interceptor approaches angled from its side, then converges to the *plain* base law
        for the endgame (a non-zero terminal offset would just buy a guaranteed miss).
    """

    def __init__(
        self,
        target: str,
        base: GuidanceLaw,
        side: float,
        beta: float = 0.2,
        r_split: float = 5000.0,
        r_merge: float = 2500.0,
    ) -> None:
        super().__init__(target)
        if abs(side) != 1.0:
            raise ValueError("side must be +1 or -1")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be in (0, 1)")
        if not 0.0 < r_merge < r_split:
            raise ValueError("require 0 < r_merge < r_split")
        self.base = base
        self.side = float(side)
        self.beta = float(beta)
        self.r_split = float(r_split)
        self.r_merge = float(r_merge)

    def command(self, t: float, own_state: Array, target_state: Array) -> Array:
        tgt = np.asarray(target_state, dtype=float)
        r = tgt[:2] - np.asarray(own_state, dtype=float)[:2]
        rng = float(np.linalg.norm(r))
        if rng < EPS:
            return self.base.command(t, own_state, target_state)
        taper = float(np.clip((rng - self.r_merge) / (self.r_split - self.r_merge), 0.0, 1.0))
        if taper <= 0.0:
            return self.base.command(t, own_state, target_state)
        u = r / rng
        perp = np.array([-u[1], u[0]])  # ⟂ to the LOS
        virtual = tgt.copy()
        virtual[:2] = tgt[:2] + self.side * self.beta * rng * taper * perp
        return self.base.command(t, own_state, virtual)


def pincer_pair(
    target: str, base_factory, beta: float = 0.2, r_split: float = 5000.0, r_merge: float = 2500.0
) -> tuple[GuidanceLaw, GuidanceLaw]:
    """Build the coordinated pair: (left-covering, right-covering) wraps of ``base_factory()``."""
    return (
        PincerGuidance(target, base_factory(), +1.0, beta=beta, r_split=r_split, r_merge=r_merge),
        PincerGuidance(target, base_factory(), -1.0, beta=beta, r_split=r_split, r_merge=r_merge),
    )
