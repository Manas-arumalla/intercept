"""Visualization for many-vs-many (swarm) engagements: static map + animated replay (GIF)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from intercept.multiagent.swarm import MultiEngagementResult

_COLOR = {"interceptor": "#1f77b4", "target": "#d62728"}


def _bounds(result: MultiEngagementResult):
    allxy = np.vstack([t[:, :2] for t in result.tracks.values()])
    lo, hi = allxy.min(0), allxy.max(0)
    pad = 0.08 * (hi - lo + 1e-9)
    return lo - pad, hi + pad


def plot_swarm(
    result: MultiEngagementResult, *, title: str | None = None, save_path=None, show: bool = True
):
    """Static map of a many-vs-many engagement: all tracks, with intercept points marked."""
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, tr in result.tracks.items():
        c = _COLOR[result.roles[name]]
        ax.plot(tr[:, 0], tr[:, 1], "-", color=c, lw=1.3, alpha=0.7)
        ax.plot(tr[0, 0], tr[0, 1], "o", color=c, ms=5)
    for p in result.kill_points:
        ax.plot(p[0], p[1], "*", color="#2ca02c", ms=16, mec="k", mew=0.5)
    ax.plot([], [], "-", color=_COLOR["interceptor"], label=f"{result.n_interceptors} interceptors")
    ax.plot([], [], "-", color=_COLOR["target"], label=f"{result.n_targets} threats")
    ax.plot([], [], "*", color="#2ca02c", mec="k", label=f"intercepts ({result.n_killed})")
    title = title or f"Swarm defense — {result.n_killed}/{result.n_targets} intercepted"
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def animate_swarm(
    result: MultiEngagementResult,
    *,
    title: str | None = None,
    save_path=None,
    fps: int = 30,
    max_frames: int = 160,
    tail: int = 25,
    show: bool = False,
):
    """Animated GIF of the swarm engagement with fading trails and pop-in intercept bursts."""
    from intercept.viz.animation import style_dark_2d

    n = len(result.times)
    idx = np.arange(n) if n <= max_frames else np.linspace(0, n - 1, max_frames).astype(int)
    lo, hi = _bounds(result)
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor="#05060a")
    style_dark_2d(ax)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    neon = {"interceptor": "#19e6ff", "target": "#ff4d6d"}
    lines, glows, heads, names = [], [], [], list(result.tracks)
    for name in names:
        c = neon.get(result.roles[name], "#7CFC00")
        (gl,) = ax.plot([], [], "-", color=c, lw=4.0, alpha=0.16, solid_capstyle="round")
        (ln,) = ax.plot([], [], "-", color=c, lw=1.4, alpha=0.95)
        (hd,) = ax.plot([], [], "o", color="white", ms=3.5, markeredgecolor=c, markeredgewidth=1.2)
        glows.append(gl)
        lines.append(ln)
        heads.append(hd)
    burst = ax.scatter([], [], s=[], c="#c6ff00", marker="*", edgecolors="k", zorder=5)
    # Kill frame index for each intercept point (when it should pop in).
    kill_frames = [int(np.argmin(np.abs(result.times - k[2]))) for k in result.kills]

    def update(fi):
        k = idx[fi]
        for name, gl, ln, hd in zip(names, glows, lines, heads, strict=True):
            tr = result.tracks[name]
            kk = min(k, len(tr) - 1)
            lo_i = max(0, kk - tail)
            gl.set_data(tr[lo_i : kk + 1, 0], tr[lo_i : kk + 1, 1])
            ln.set_data(tr[lo_i : kk + 1, 0], tr[lo_i : kk + 1, 1])
            hd.set_data([tr[kk, 0]], [tr[kk, 1]])
        shown = [result.kill_points[c] for c, kf in enumerate(kill_frames) if kf <= k]
        if shown:
            burst.set_offsets(np.array(shown))
            burst.set_sizes([220] * len(shown))
        ax.set_title(
            title
            or f"Swarm defense  t={result.times[k]:.1f}s  "
            f"intercepts {sum(kf <= k for kf in kill_frames)}/{result.n_targets}",
            color="#e6edf3",
        )
        return [*glows, *lines, *heads, burst]

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=False)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(save_path), writer=PillowWriter(fps=fps))
    if show:
        plt.show()
    else:
        plt.close(fig)
    return anim
