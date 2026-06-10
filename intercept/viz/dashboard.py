"""Interactive 3-D engagement replay (Plotly) — a browsable HTML dashboard.

Renders an :class:`~intercept.core.engagement.EngagementResult` as an interactive 3-D scene with a
**play button + time slider**, draggable camera, and hover read-outs — the interactive complement to
the static/animated Matplotlib views (`viz.threed`). Exports a self-contained HTML file (no server),
ideal for a portfolio: a recruiter can spin and scrub the engagement in a browser. Requires the
optional ``plotly`` dependency (``pip install -e ".[viz]"``); calling without it raises.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from intercept.core.engagement import EngagementResult

Array = NDArray[np.float64]

try:
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:  # pragma: no cover - exercised only when plotly missing
    _HAS_PLOTLY = False

_COLOR = {"interceptor": "#19e6ff", "target": "#ff4d6d", "decoy": "#9aa0a6"}


def has_plotly() -> bool:
    """Whether Plotly is available (so the interactive dashboard can be built)."""
    return _HAS_PLOTLY


def interactive_engagement_3d(
    result: EngagementResult,
    *,
    roles=None,
    title: str | None = None,
    save_path=None,
    max_frames: int = 120,
    tail: int = 60,
    show: bool = False,
):
    """Build an interactive 3-D Plotly replay of an engagement; optionally write a standalone HTML.

    Parameters
    ----------
    result:
        The engagement to replay (entity states must carry 3-D positions in ``state[:3]``).
    roles:
        Optional ``{entity_name: role}`` for coloring (defaults to interceptor/target).
    save_path:
        If given, write a self-contained HTML file there.
    max_frames, tail:
        Animation frame cap and fading-trail length (in samples).
    """
    if not _HAS_PLOTLY:
        raise ImportError(
            'interactive_engagement_3d requires Plotly. Install with: pip install -e ".[viz]"'
        )
    roles = roles or {result.interceptor: "interceptor", result.target: "target"}
    names = list(result.states)
    n = len(result.times)
    idx = np.arange(n) if n <= max_frames else np.linspace(0, n - 1, max_frames).astype(int)
    allpts = np.vstack([s[:, :3] for s in result.states.values()])
    lo, hi = allpts.min(0), allpts.max(0)

    def _frame_data(k: int):
        data = []
        for name in names:
            p = result.states[name][:, :3]
            kk = min(k, len(p) - 1)
            start = max(0, kk - tail)
            seg = p[start : kk + 1]
            c = _COLOR.get(roles.get(name, "entity"), "#7CFC00")
            data.append(
                go.Scatter3d(
                    x=seg[:, 0],
                    y=seg[:, 1],
                    z=seg[:, 2],
                    mode="lines",
                    line=dict(color=c, width=5),
                    name=name,
                    showlegend=bool(k == idx[0]),
                )
            )
            data.append(
                go.Scatter3d(
                    x=[p[kk, 0]],
                    y=[p[kk, 1]],
                    z=[p[kk, 2]],
                    mode="markers",
                    marker=dict(color=c, size=5),
                    showlegend=False,
                )
            )
        return data

    frames = [go.Frame(data=_frame_data(int(k)), name=str(i)) for i, k in enumerate(idx)]
    outcome = "INTERCEPT" if result.intercepted else result.reason.name
    fig = go.Figure(
        data=_frame_data(int(idx[0])),
        frames=frames,
        layout=go.Layout(
            title=title or f"{outcome} — miss {result.miss_distance:.1f} m",
            template="plotly_dark",
            scene=dict(
                xaxis=dict(title="x (m)", range=[lo[0], hi[0]]),
                yaxis=dict(title="y (m)", range=[lo[1], hi[1]]),
                zaxis=dict(title="z (m)", range=[lo[2], hi[2]]),
                aspectmode="data",
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.05,
                    y=0.05,
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                dict(frame=dict(duration=40, redraw=True), fromcurrent=True),
                            ],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    y=0,
                    x=0.15,
                    len=0.8,
                    steps=[
                        dict(
                            method="animate",
                            label=f"{result.times[k]:.1f}s",
                            args=[
                                [str(i)],
                                dict(mode="immediate", frame=dict(duration=0, redraw=True)),
                            ],
                        )
                        for i, k in enumerate(idx)
                    ],
                )
            ],
        ),
    )
    # Intercept / closest-approach marker.
    ti = int(np.argmin(np.abs(result.times - result.closest_approach_time)))
    mid = 0.5 * (result.states[result.interceptor][ti, :3] + result.states[result.target][ti, :3])
    fig.add_trace(
        go.Scatter3d(
            x=[mid[0]],
            y=[mid[1]],
            z=[mid[2]],
            mode="markers",
            marker=dict(color="#c6ff00", size=9, symbol="diamond"),
            name=("intercept" if result.intercepted else "closest approach"),
        )
    )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path), include_plotlyjs="cdn", auto_play=False)
    if show:
        fig.show()
    return fig
