"""2-D engagement plotting.

Publication-leaning defaults (clean axes, equal aspect, annotated outcome) so figures are
portfolio-ready straight out of the simulator. Rendering is decoupled from any GUI: pass a
path to save, and/or ``show=False`` for headless/CI use.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from intercept.core.engagement import EngagementResult, TerminationReason

_ROLE_STYLE = {
    "interceptor": {"color": "#1f77b4", "label": "Interceptor"},
    "target": {"color": "#d62728", "label": "Target"},
    "decoy": {"color": "#7f7f7f", "label": "Decoy"},
}


def plot_engagement_2d(
    result: EngagementResult,
    *,
    roles: dict[str, str] | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Plot interceptor/target ground tracks for an :class:`EngagementResult`.

    Parameters
    ----------
    result:
        The engagement to render.
    roles:
        Optional ``{entity_name: role}`` map controlling color/legend; defaults to the
        interceptor/target named in ``result`` with any others drawn neutrally.
    title:
        Figure title; a sensible outcome-based default is used if omitted.
    save_path:
        If given, the figure is written here (parent dirs created).
    show:
        If ``True``, call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    (fig, ax):
        The Matplotlib figure and axes, for further customization.
    """
    roles = roles or {result.interceptor: "interceptor", result.target: "target"}
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, states in result.states.items():
        role = roles.get(name, "entity")
        style = _ROLE_STYLE.get(role, {"color": "#2ca02c", "label": name})
        ax.plot(states[:, 0], states[:, 1], "-", color=style["color"], lw=1.8, label=style["label"])
        ax.plot(states[0, 0], states[0, 1], "o", color=style["color"], ms=7)  # start
        ax.plot(states[-1, 0], states[-1, 1], "s", color=style["color"], ms=6)  # end

    # Mark closest approach between interceptor and target.
    ti = int(np.argmin(np.abs(result.times - result.closest_approach_time)))
    pi = result.states[result.interceptor][ti, :2]
    pt = result.states[result.target][ti, :2]
    ax.plot([pi[0], pt[0]], [pi[1], pt[1]], "k--", lw=1.0, alpha=0.7)
    mid = 0.5 * (pi + pt)
    marker = "*" if result.intercepted else "x"
    mcolor = "#2ca02c" if result.intercepted else "#ff7f0e"
    ax.plot(
        mid[0],
        mid[1],
        marker,
        color=mcolor,
        ms=16,
        mew=2,
        label=f"{'Intercept' if result.intercepted else 'Closest approach'}",
    )

    if title is None:
        outcome = "INTERCEPT" if result.intercepted else result.reason.name
        t_str = (
            f"t={result.intercept_time:.2f}s"
            if result.intercept_time is not None
            else f"closest @ t={result.closest_approach_time:.2f}s"
        )
        title = f"Engagement: {outcome} | miss={result.miss_distance:.2f} m | {t_str}"
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax


_COMPARE_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#e377c2"]


def compare_engagements_2d(
    results: dict[str, EngagementResult],
    *,
    title: str = "Guidance comparison",
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Overlay several engagements that share a target, one interceptor track per label.

    Each interceptor trajectory is drawn in its own color with an outcome marker (★ intercept,
    × miss) and a legend entry annotated with miss distance. The (assumed common) target track is
    drawn once in red. Useful for "PN vs Pure-Pursuit vs APN on the same target" figures.

    Parameters
    ----------
    results:
        ``{label: EngagementResult}``. The first result's target track is drawn as the reference.
    """
    if not results:
        raise ValueError("results must be non-empty")
    fig, ax = plt.subplots(figsize=(9, 6.5))

    first = next(iter(results.values()))
    tgt = first.states[first.target]
    ax.plot(tgt[:, 0], tgt[:, 1], "-", color="#d62728", lw=2.0, label="Target", zorder=3)
    ax.plot(tgt[0, 0], tgt[0, 1], "o", color="#d62728", ms=7)

    for i, (label, res) in enumerate(results.items()):
        color = _COMPARE_COLORS[i % len(_COMPARE_COLORS)]
        traj = res.states[res.interceptor]
        ax.plot(traj[:, 0], traj[:, 1], "-", color=color, lw=1.8)
        ax.plot(traj[0, 0], traj[0, 1], "o", color=color, ms=6)
        ti = int(np.argmin(np.abs(res.times - res.closest_approach_time)))
        end = res.states[res.interceptor][ti, :2]
        marker = "*" if res.intercepted else "x"
        ax.plot(end[0], end[1], marker, color=color, ms=15, mew=2)
        outcome = "hit" if res.intercepted else "miss"
        legend = f"{label}: {outcome} {res.miss_distance:.1f} m @ {res.closest_approach_time:.2f} s"
        ax.plot([], [], "-", color=color, lw=2.0, label=legend)

    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax


# Re-exported for callers that branch on outcome when titling/coloring.
__all__ = ["plot_engagement_2d", "compare_engagements_2d", "TerminationReason"]
