"""Impact-Time-Control Guidance (ITCG) — biased PN for a commanded time of arrival.

Plain PN's time-to-go is fixed by the geometry; a **salvo** instead needs every interceptor to
arrive *at the same chosen time* ``t_impact``, saturating the defense simultaneously. ITCG (after
Jeon, Lee & Tahk, *IEEE TAES* 2006) adds a time-to-go-error feedback to PN that **lengthens the
path** — by raising the lead angle — when the interceptor would otherwise arrive early:

    a = N · Vc · λ̇ · ⟂̂_LOS  +  k · Vc · max(0, (t_impact − t) − R/Vc) · ⟂̂_v,away

``⟂̂_v,away`` is the unit perpendicular to the interceptor's velocity pointing *away* from the LOS,
so the bias increases the velocity–LOS angle and stretches the trajectory; as the time-to-go
error shrinks the bias vanishes and PN homes cleanly. A small gain (``k ≈ 0.2``) gives accurate,
non-chattering convergence: a battery launched from different ranges arrives within a fraction of a
second of the commanded time (see `experiments/p24_salvo.py`). The interceptor cannot *speed up*, so
``t_impact`` must be at least the slowest member's natural time-to-go.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from intercept.guidance.base import EPS, GuidanceLaw

Array = NDArray[np.float64]


class ImpactTimeGuidance(GuidanceLaw):
    """Biased-PN guidance steering toward a commanded absolute impact time ``t_impact`` (s).

    Parameters
    ----------
    target:
        Name of the target entity to home on.
    t_impact:
        Commanded absolute time of arrival (s). Feasible only if ``≥`` the natural time-to-go.
    N:
        Navigation constant for the PN term.
    k:
        Time-error feedback gain (1/s). ``≈0.2`` synchronizes tightly without overshoot; larger
        values over-delay.
    """

    def __init__(self, target: str, t_impact: float, N: float = 4.0, k: float = 0.2) -> None:
        super().__init__(target)
        if N <= 1 or t_impact <= 0 or k < 0:
            raise ValueError("require N > 1, t_impact > 0, k >= 0")
        self.t_impact = float(t_impact)
        self.N = float(N)
        self.k = float(k)

    def time_to_go(self, own_state: Array, target_state: Array) -> float:
        """Time-to-go estimate ``R / Vc`` (``inf`` if not closing)."""
        r, v, rng = self._rel(own_state, target_state)
        vc = self._closing_speed(r, v, rng)
        return rng / vc if vc > EPS else float("inf")

    def command(self, t: float, own_state: Array, target_state: Array) -> Array:
        r, v, rng = self._rel(own_state, target_state)
        if rng < EPS:
            return np.zeros(2)
        vc = self._closing_speed(r, v, rng)
        a_pn = self.N * vc * self._los_rate(r, v, rng) * self._los_perp(r, rng)
        if vc <= EPS:
            return a_pn
        e_t = (self.t_impact - t) - rng / vc  # time-to-go error (>0 ⇒ would arrive early)
        if e_t <= 0.0:  # late / on-time: PN homes (cannot speed up)
            return a_pn
        v_own = own_state[2:4]
        speed = float(np.linalg.norm(v_own))
        if speed < EPS:
            return a_pn
        v_hat = v_own / speed
        perp_v = np.array([-v_hat[1], v_hat[0]])  # ⟂ velocity, pointed away from the LOS
        if perp_v @ (r / rng) > 0.0:
            perp_v = -perp_v
        return a_pn + self.k * vc * e_t * perp_v  # lengthen the path to delay arrival


def impact_time_guidance(target: str, t_impact: float, N: float = 4.0, k: float = 0.2):
    """Convenience factory for impact-time-control (salvo) guidance."""
    return ImpactTimeGuidance(target, t_impact, N=N, k=k)
