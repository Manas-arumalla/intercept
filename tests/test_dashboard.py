"""Test the interactive Plotly dashboard (skipped if plotly is not installed)."""

from __future__ import annotations

import numpy as np
import pytest

from intercept.viz.dashboard import has_plotly, interactive_engagement_3d

pytestmark = pytest.mark.skipif(not has_plotly(), reason="plotly not installed")


def _engagement():
    from intercept.core import G0, Engagement, Entity, PointMass3D
    from intercept.guidance import true_pn_3d

    interceptor = Entity(
        "interceptor",
        PointMass3D(a_max=40 * G0),
        np.array([0, 0, 0, 700.0, 0.0, 80.0]),
        controller=true_pn_3d("target", N=4.0),
        role="interceptor",
    )
    target = Entity(
        "target", PointMass3D(), np.array([5000.0, 1200.0, 2000.0, -250.0, 0.0, 0.0]), role="target"
    )
    return Engagement(
        [interceptor, target],
        interceptor="interceptor",
        target="target",
        dt=0.02,
        t_max=20.0,
        kill_radius=25.0,
    ).run()


def test_interactive_dashboard_builds_figure_and_html(tmp_path):
    res = _engagement()
    out = tmp_path / "replay.html"
    fig = interactive_engagement_3d(res, max_frames=40, save_path=out)
    assert len(fig.frames) == 40  # animation frames built
    assert out.exists() and out.stat().st_size > 0  # standalone HTML written
    # The intercept / closest-approach marker trace is present (last trace).
    assert any(tr.name in ("intercept", "closest approach") for tr in fig.data)
