"""Multi-agent interception: assignment, cooperative guidance, and swarm defense.

``assignment`` provides Hungarian weapon-target allocation (time- and kill-probability objectives),
``swarm`` runs the N-vs-M ``MultiEngagement`` with live re-assignment, and ``defense`` adds the
asset-value, decoy-aware allocator that counters coordinated saturation raids.
"""

from intercept.multiagent.assignment import (
    cost_matrix,
    expected_leakers,
    intercept_time_cost,
    kill_probability,
    kill_probability_matrix,
    weapon_target_assignment,
)
from intercept.multiagent.defense import (
    make_value_allocator,
    predict_closest_approach,
    threat_value,
    value_prioritized_assignment,
)
from intercept.multiagent.swarm import MultiEngagement, MultiEngagementResult

__all__ = [
    "weapon_target_assignment",
    "cost_matrix",
    "intercept_time_cost",
    "kill_probability",
    "kill_probability_matrix",
    "expected_leakers",
    "MultiEngagement",
    "MultiEngagementResult",
    "predict_closest_approach",
    "threat_value",
    "value_prioritized_assignment",
    "make_value_allocator",
]
