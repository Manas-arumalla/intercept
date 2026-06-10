"""Visualization for pursuit-evasion game theory: the Apollonius circle and dominance regions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from intercept.guidance.game import apollonius_circle, intercept_point

Array = NDArray[np.float64]


def plot_apollonius(
    pursuer_pos: Array,
    evader_pos: Array,
    evader_vel: Array,
    pursuer_speed: float,
    *,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Render the Apollonius circle, dominance regions, and predicted capture point.

    The evader speed is taken as ``|evader_vel|``; the speed ratio ``α = v_E / v_P`` must be < 1.
    Inside the circle (shaded) is the evader's dominance region; the star marks the soonest
    constant-velocity intercept point.
    """
    p = np.asarray(pursuer_pos, dtype=float)[:2]
    e = np.asarray(evader_pos, dtype=float)[:2]
    v = np.asarray(evader_vel, dtype=float)[:2]
    v_e = float(np.linalg.norm(v))
    alpha = v_e / pursuer_speed
    center, radius = apollonius_circle(p, e, alpha)

    fig, ax = plt.subplots(figsize=(8, 7))
    theta = np.linspace(0, 2 * np.pi, 200)
    circ = center[:, None] + radius * np.vstack([np.cos(theta), np.sin(theta)])
    ax.fill(circ[0], circ[1], color="#d62728", alpha=0.12, label="evader dominance region")
    ax.plot(circ[0], circ[1], color="#d62728", lw=1.5)

    ax.plot(*p, "o", color="#1f77b4", ms=12, label="pursuer P")
    ax.plot(*e, "o", color="#d62728", ms=12, label="evader E")
    ax.annotate(
        "",
        xy=e + v / v_e * radius * 0.6,
        xytext=e,
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=2),
    )

    cap = intercept_point(p, e, v, pursuer_speed)
    if cap is not None:
        ax.plot(*cap, "*", color="#2ca02c", ms=20, label="predicted capture point")
        ax.plot([p[0], cap[0]], [p[1], cap[1]], "--", color="#1f77b4", lw=1.2)
        ax.plot([e[0], cap[0]], [e[1], cap[1]], "--", color="#d62728", lw=1.2)

    ax.set_title(f"Apollonius circle — pursuit-evasion (speed ratio α = v_E/v_P = {alpha:.2f})")
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
