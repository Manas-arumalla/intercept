"""Animated engagement replays — watch the intercept happen.

Renders an :class:`~intercept.core.engagement.EngagementResult` (or several, overlaid) as a moving
animation with fading trails, saved to GIF (PillowWriter) or MP4 (ffmpeg). Also a static "filmstrip"
montage of evenly-spaced snapshots for quick inline viewing. Trajectories are subsampled to keep
frame counts (and GIF size) reasonable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from numpy.typing import NDArray

from intercept.core.engagement import EngagementResult

Array = NDArray[np.float64]

_ROLE_COLOR = {"interceptor": "#19e6ff", "target": "#ff4d6d", "decoy": "#9aa0a6"}


@dataclass
class _Track:
    label: str
    xy: Array  # (T, 2)
    color: str
    is_target: bool = False


def _subsample(n: int, max_frames: int) -> np.ndarray:
    if n <= max_frames:
        return np.arange(n)
    return np.linspace(0, n - 1, max_frames).astype(int)


def _tracks_from_result(result: EngagementResult, roles: dict[str, str] | None) -> list[_Track]:
    roles = roles or {result.interceptor: "interceptor", result.target: "target"}
    tracks = []
    for name, states in result.states.items():
        role = roles.get(name, "entity")
        tracks.append(
            _Track(
                name, states[:, :2], _ROLE_COLOR.get(role, "#2ca02c"), is_target=(role == "target")
            )
        )
    return tracks


_BG2D = "#05060a"


def style_dark_2d(ax) -> None:
    """Dark cinematic theme for 2-D engagement axes (matches the 3-D modern style)."""
    ax.set_facecolor("#0a0c12")
    for spine in ax.spines.values():
        spine.set_color("#2a2e38")
    ax.grid(True, alpha=0.18, color="#3a3f4d")
    ax.tick_params(colors="#55585f", labelsize=8)
    ax.xaxis.label.set_color("#7d8590")
    ax.yaxis.label.set_color("#7d8590")
    ax.title.set_color("#e6edf3")


def _render(
    tracks: list[_Track],
    frames_idx: np.ndarray,
    title: str,
    save_path,
    fps: int,
    tail: int,
    show: bool,
    outcome_xy: Array | None,
    hit: bool,
):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=_BG2D)
    style_dark_2d(ax)
    allxy = np.vstack([t.xy for t in tracks])
    pad = 0.08 * (allxy.max(0) - allxy.min(0) + 1e-9)
    ax.set_xlim(allxy[:, 0].min() - pad[0], allxy[:, 0].max() + pad[0])
    ax.set_ylim(allxy[:, 1].min() - pad[1], allxy[:, 1].max() + pad[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)

    lines, glows, heads, halos = [], [], [], []
    for tr in tracks:
        (gl,) = ax.plot(
            [], [], "-", color=tr.color, lw=5.0, alpha=0.18, solid_capstyle="round"
        )  # soft neon under-glow
        (ln,) = ax.plot([], [], "-", color=tr.color, lw=1.8, alpha=0.95, label=tr.label)
        (halo,) = ax.plot([], [], "o", color=tr.color, ms=14, alpha=0.25)
        (hd,) = ax.plot(
            [], [], "o", color="white", ms=4.5, markeredgecolor=tr.color, markeredgewidth=1.4
        )
        glows.append(gl)
        lines.append(ln)
        halos.append(halo)
        heads.append(hd)
    burst = ax.scatter([], [], s=0, c="#c6ff00", marker="*", zorder=5)
    leg = ax.legend(
        loc="upper left",
        fontsize=9,
        facecolor="#0b0d12",
        edgecolor="#2a2e38",
        labelcolor="#cdd5df",
        framealpha=0.7,
    )
    leg.set_zorder(6)

    def update(fi):
        k = frames_idx[fi]
        for tr, gl, ln, halo, hd in zip(tracks, glows, lines, halos, heads, strict=True):
            kk = min(k, len(tr.xy) - 1)
            lo = max(0, kk - tail)
            gl.set_data(tr.xy[lo : kk + 1, 0], tr.xy[lo : kk + 1, 1])
            ln.set_data(tr.xy[lo : kk + 1, 0], tr.xy[lo : kk + 1, 1])
            for h in (halo, hd):
                h.set_data([tr.xy[kk, 0]], [tr.xy[kk, 1]])
        # Intercept flash that grows over the final frames.
        if hit and outcome_xy is not None and fi >= len(frames_idx) - 8:
            grow = (fi - (len(frames_idx) - 8)) / 8.0
            burst.set_offsets([outcome_xy])
            burst.set_sizes([120 + 700 * grow])
        return [*glows, *lines, *halos, *heads, burst]

    anim = FuncAnimation(fig, update, frames=len(frames_idx), interval=1000 / fps, blit=True)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(save_path), writer=PillowWriter(fps=fps))
    if show:
        plt.show()
    else:
        plt.close(fig)
    return anim


def animate_engagement(
    result: EngagementResult,
    *,
    roles=None,
    title: str | None = None,
    save_path=None,
    fps: int = 30,
    max_frames: int = 160,
    tail: int = 30,
    show: bool = False,
):
    """Animate one engagement (interceptor + target) with fading trails and an intercept burst."""
    tracks = _tracks_from_result(result, roles)
    frames_idx = _subsample(len(result.times), max_frames)
    if title is None:
        outcome = "INTERCEPT" if result.intercepted else result.reason.name
        title = f"{outcome} — miss {result.miss_distance:.1f} m"
    ti = int(np.argmin(np.abs(result.times - result.closest_approach_time)))
    pit = result.states[result.interceptor][ti, :2]
    ptt = result.states[result.target][ti, :2]
    return _render(
        tracks, frames_idx, title, save_path, fps, tail, show, 0.5 * (pit + ptt), result.intercepted
    )


def animate_comparison(
    results: dict[str, EngagementResult],
    *,
    target_label: str = "target",
    title: str = "Guidance comparison",
    save_path=None,
    fps: int = 30,
    max_frames: int = 160,
    tail: int = 30,
    show: bool = False,
):
    """Animate several engagements that share a target — one interceptor track each, overlaid."""
    first = next(iter(results.values()))
    colors = ["#19e6ff", "#7CFC00", "#b46bff", "#ffb347", "#2dd4bf"]
    tracks = [_Track(target_label, first.states[first.target][:, :2], "#ff4d6d", is_target=True)]
    longest = len(first.times)
    for i, (label, res) in enumerate(results.items()):
        tracks.append(_Track(label, res.states[res.interceptor][:, :2], colors[i % len(colors)]))
        longest = max(longest, len(res.times))
    frames_idx = _subsample(longest, max_frames)
    return _render(tracks, frames_idx, title, save_path, fps, tail, show, None, False)


def filmstrip_engagement(
    result: EngagementResult,
    *,
    roles=None,
    n_panels: int = 5,
    title: str | None = None,
    save_path=None,
    show: bool = False,
):
    """Static montage of evenly-spaced snapshots (trails grow left→right) for quick viewing."""
    tracks = _tracks_from_result(result, roles)
    n = len(result.times)
    idx = np.linspace(int(0.15 * n), n - 1, n_panels).astype(int)
    allxy = np.vstack([t.xy for t in tracks])
    pad = 0.08 * (allxy.max(0) - allxy.min(0) + 1e-9)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.4), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), idx, strict=True):
        for tr in tracks:
            ax.plot(tr.xy[: k + 1, 0], tr.xy[: k + 1, 1], "-", color=tr.color, lw=1.5, alpha=0.8)
            ax.plot(tr.xy[k, 0], tr.xy[k, 1], "o", color=tr.color, ms=7)
        ax.set_xlim(allxy[:, 0].min() - pad[0], allxy[:, 0].max() + pad[0])
        ax.set_ylim(allxy[:, 1].min() - pad[1], allxy[:, 1].max() + pad[1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"t = {result.times[k]:.1f} s", fontsize=10)
    outcome = "INTERCEPT" if result.intercepted else result.reason.name
    fig.suptitle(title or f"{outcome} — miss {result.miss_distance:.1f} m", fontsize=12)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
