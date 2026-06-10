"""Guidance laws (controllers that command interceptor acceleration).

Every guidance law conforms to the :data:`intercept.core.entities.Controller` signature
``(t, own_state, world) -> control`` so it drops directly into an :class:`~intercept.core.Entity`
and is benchmarked against all others on identical dynamics.

The library spans six paradigms: the Proportional Navigation family (``pn``: True/Pure/ZEM-PN,
``apn``: Augmented PN), optimal and robust control (``ogl``: optimal/LQ, ``smg``: sliding-mode,
``mpc``: NMPC via CasADi), learned guidance (``rl_policy``: a trained policy wrapped as a
controller), geometric/game-theoretic pursuit (``game``: Apollonius-circle), cooperative laws
(``impact_time``: salvo, ``pincer``: coverage), and estimation-aware arbitration (``adaptive``).
"""

from intercept.guidance.adaptive import ModeAdaptiveGuidance
from intercept.guidance.apn import AugmentedPN
from intercept.guidance.base import GuidanceLaw
from intercept.guidance.estimating import EstimatingGuidance
from intercept.guidance.game import ApolloniusGuidance, apollonius_circle, intercept_point
from intercept.guidance.impact_time import ImpactTimeGuidance, impact_time_guidance
from intercept.guidance.ogl import (
    OptimalGuidance,
    OptimalGuidance3D,
    optimal_guidance,
    optimal_guidance_3d,
)
from intercept.guidance.pincer import PincerGuidance, pincer_pair
from intercept.guidance.pn import ProportionalNavigation, pure_pn, true_pn, zem_pn
from intercept.guidance.pn3d import (
    AugmentedPN3D,
    ProportionalNavigation3D,
    augmented_pn_3d,
    true_pn_3d,
)
from intercept.guidance.smg import (
    SlidingModeGuidance,
    SlidingModeGuidance3D,
    sliding_mode,
    sliding_mode_3d,
)

__all__ = [
    "GuidanceLaw",
    "ProportionalNavigation",
    "true_pn",
    "pure_pn",
    "zem_pn",
    "AugmentedPN",
    "OptimalGuidance",
    "optimal_guidance",
    "SlidingModeGuidance",
    "sliding_mode",
    "ApolloniusGuidance",
    "apollonius_circle",
    "intercept_point",
    "ImpactTimeGuidance",
    "PincerGuidance",
    "pincer_pair",
    "impact_time_guidance",
    "ProportionalNavigation3D",
    "AugmentedPN3D",
    "true_pn_3d",
    "augmented_pn_3d",
    "OptimalGuidance3D",
    "optimal_guidance_3d",
    "SlidingModeGuidance3D",
    "sliding_mode_3d",
    "EstimatingGuidance",
    "ModeAdaptiveGuidance",
]
