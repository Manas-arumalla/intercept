"""Visualization: publication-quality static plots and cinematic animations.

2-D and 3-D engagement plots, Matplotlib animations (a dark/neon theme with glow trails and
intercept flashes), swarm replays, Apollonius-circle overlays, capture-region heatmaps, and
interactive Plotly 3-D replays exported to standalone HTML.
"""

from intercept.viz.animation import (
    animate_comparison,
    animate_engagement,
    filmstrip_engagement,
)
from intercept.viz.benchmark_plots import plot_capture_region, plot_pintercept_bars
from intercept.viz.dashboard import has_plotly, interactive_engagement_3d
from intercept.viz.engagement2d import compare_engagements_2d, plot_engagement_2d
from intercept.viz.game_plots import plot_apollonius
from intercept.viz.swarm_plots import animate_swarm, plot_swarm
from intercept.viz.threed import (
    animate_engagement_3d,
    animate_engagement_3d_modern,
    animate_swarm_3d_modern,
    plot_engagement_3d,
    plot_engagement_3d_modern,
)

__all__ = [
    "plot_engagement_2d",
    "compare_engagements_2d",
    "plot_capture_region",
    "plot_pintercept_bars",
    "plot_apollonius",
    "animate_engagement",
    "animate_comparison",
    "filmstrip_engagement",
    "plot_engagement_3d",
    "animate_engagement_3d",
    "plot_engagement_3d_modern",
    "animate_engagement_3d_modern",
    "animate_swarm_3d_modern",
    "plot_swarm",
    "animate_swarm",
    "interactive_engagement_3d",
    "has_plotly",
]
