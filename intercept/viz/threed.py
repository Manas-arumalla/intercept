"""3-D engagement visualization: static trajectory plot and animated replay (GIF).

Operates on an :class:`~intercept.core.engagement.EngagementResult` whose entity states carry 3-D
positions (``state[:3]``). Uses Matplotlib's mplot3d so it has no heavy extra dependencies.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from numpy.typing import NDArray

from intercept.core.engagement import EngagementResult

Array = NDArray[np.float64]

_ROLE_COLOR = {"interceptor": "#1f77b4", "target": "#d62728", "decoy": "#7f7f7f"}


def _roles(result: EngagementResult, roles):
    return roles or {result.interceptor: "interceptor", result.target: "target"}


def _set_equal_aspect(ax, pts: Array) -> None:
    lo, hi = pts.min(0), pts.max(0)
    span = float((hi - lo).max()) or 1.0
    mid = 0.5 * (hi + lo)
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
    ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)


def plot_engagement_3d(
    result: EngagementResult,
    *,
    roles=None,
    title: str | None = None,
    save_path=None,
    show: bool = True,
):
    """Static 3-D plot of an engagement's trajectories with start/end and intercept markers."""
    roles = _roles(result, roles)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    allpts = np.vstack([s[:, :3] for s in result.states.values()])
    for name, states in result.states.items():
        c = _ROLE_COLOR.get(roles.get(name, "entity"), "#2ca02c")
        p = states[:, :3]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], "-", color=c, lw=1.8, label=name)
        ax.scatter(*p[0], color=c, s=40)
        ax.scatter(*p[-1], color=c, s=40, marker="s")
    ti = int(np.argmin(np.abs(result.times - result.closest_approach_time)))
    mid = 0.5 * (result.states[result.interceptor][ti, :3] + result.states[result.target][ti, :3])
    ax.scatter(
        *mid,
        color=("#2ca02c" if result.intercepted else "#ff7f0e"),
        s=200,
        marker="*",
        label="intercept" if result.intercepted else "closest approach",
    )
    _set_equal_aspect(ax, allpts)
    outcome = "INTERCEPT" if result.intercepted else result.reason.name
    ax.set_title(title or f"3-D engagement: {outcome} — miss {result.miss_distance:.1f} m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def animate_engagement_3d(
    result: EngagementResult,
    *,
    roles=None,
    title: str | None = None,
    save_path=None,
    fps: int = 30,
    max_frames: int = 150,
    tail: int = 40,
    spin: bool = True,
    show: bool = False,
):
    """Animated 3-D replay (GIF) with fading trails and a slowly rotating view."""
    roles = _roles(result, roles)
    n = len(result.times)
    idx = np.arange(n) if n <= max_frames else np.linspace(0, n - 1, max_frames).astype(int)
    allpts = np.vstack([s[:, :3] for s in result.states.values()])

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(projection="3d")
    _set_equal_aspect(ax, allpts)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    outcome = "INTERCEPT" if result.intercepted else result.reason.name
    ax.set_title(title or f"3-D: {outcome} — miss {result.miss_distance:.1f} m")

    lines, heads, names = [], [], list(result.states)
    for name in names:
        c = _ROLE_COLOR.get(roles.get(name, "entity"), "#2ca02c")
        (ln,) = ax.plot([], [], [], "-", color=c, lw=1.8, label=name)
        (hd,) = ax.plot([], [], [], "o", color=c, ms=7)
        lines.append(ln)
        heads.append(hd)
    ax.legend(loc="upper left", fontsize=9)

    def update(fi):
        k = idx[fi]
        for name, ln, hd in zip(names, lines, heads, strict=True):
            p = result.states[name][:, :3]
            lo = max(0, k - tail)
            ln.set_data(p[lo : k + 1, 0], p[lo : k + 1, 1])
            ln.set_3d_properties(p[lo : k + 1, 2])
            hd.set_data([p[k, 0]], [p[k, 1]])
            hd.set_3d_properties([p[k, 2]])
        if spin:
            ax.view_init(elev=22, azim=(fi * 1.2) % 360)
        return [*lines, *heads]

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=False)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(save_path), writer=PillowWriter(fps=fps))
    if show:
        plt.show()
    else:
        plt.close(fig)
    return anim


# --- modern / cinematic 3-D styling ---------------------------------------

_MODERN = {"interceptor": "#19e6ff", "target": "#ff4d6d", "decoy": "#9aa0a6"}
_BG = "#05060a"
_BURST = "#c6ff00"


def _style_dark(ax) -> None:
    """Apply a dark, low-chrome 'space' theme to a 3-D axes."""
    ax.set_facecolor(_BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.03, 0.04, 0.07, 1.0))
        axis.pane.set_edgecolor((0.2, 0.22, 0.28, 1.0))
        axis._axinfo["grid"]["color"] = (0.18, 0.2, 0.26, 1.0)
        axis.label.set_color("#7d8590")
        axis.set_tick_params(colors="#55585f", labelsize=7)


def _glow_plot(ax, xs, ys, zs, color):
    """Draw a line as stacked translucent layers (a fake neon glow); return the layer artists."""
    layers = []
    for lw, alpha in ((7.0, 0.05), (4.0, 0.12), (2.2, 0.30), (1.2, 1.0)):
        (ln,) = ax.plot(xs, ys, zs, "-", color=color, lw=lw, alpha=alpha, solid_capstyle="round")
        layers.append(ln)
    return layers


def animate_engagement_3d_modern(
    result: EngagementResult,
    *,
    roles=None,
    title: str | None = None,
    save_path=None,
    fps: int = 30,
    max_frames: int = 170,
    tail: int = 55,
    show: bool = False,
):
    """Cinematic 3-D replay: dark theme, neon glow trails, a growing intercept flash, slow orbit."""
    roles = _roles(result, roles)
    n = len(result.times)
    idx = np.arange(n) if n <= max_frames else np.linspace(0, n - 1, max_frames).astype(int)
    allpts = np.vstack([s[:, :3] for s in result.states.values()])

    fig = plt.figure(figsize=(8.5, 7), facecolor=_BG)
    ax = fig.add_subplot(projection="3d")
    _style_dark(ax)
    _set_equal_aspect(ax, allpts)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    outcome = "INTERCEPT" if result.intercepted else result.reason.name
    ax.set_title(
        title or f"{outcome}   miss {result.miss_distance:.1f} m",
        color="#e6edf3",
        fontsize=13,
        pad=12,
    )

    names = list(result.states)
    glow = {
        name: _glow_plot(ax, [], [], [], _MODERN.get(roles.get(name, "entity"), "#7CFC00"))
        for name in names
    }
    heads = {}
    for name in names:
        c = _MODERN.get(roles.get(name, "entity"), "#7CFC00")
        (soft,) = ax.plot([], [], [], "o", color=c, ms=15, alpha=0.25)
        (core,) = ax.plot(
            [], [], [], "o", color="white", ms=5, alpha=0.95, markeredgecolor=c, markeredgewidth=1.5
        )
        heads[name] = (soft, core)
    burst = ax.scatter([], [], [], s=[], c=_BURST, marker="*", depthshade=False)

    ti = int(np.argmin(np.abs(result.times - result.closest_approach_time)))
    kill_xyz = 0.5 * (
        result.states[result.interceptor][ti, :3] + result.states[result.target][ti, :3]
    )

    def update(fi):
        k = idx[fi]
        artists = []
        for name in names:
            p = result.states[name][:, :3]
            kk = min(k, len(p) - 1)
            lo = max(0, kk - tail)
            seg = p[lo : kk + 1]
            for ln in glow[name]:
                ln.set_data(seg[:, 0], seg[:, 1])
                ln.set_3d_properties(seg[:, 2])
            soft, core = heads[name]
            for h in (soft, core):
                h.set_data([p[kk, 0]], [p[kk, 1]])
                h.set_3d_properties([p[kk, 2]])
            artists += [*glow[name], soft, core]
        # Intercept flash that grows over the final frames.
        if result.intercepted and fi > len(idx) - 12:
            grow = (fi - (len(idx) - 12)) / 12.0
            burst._offsets3d = ([kill_xyz[0]], [kill_xyz[1]], [kill_xyz[2]])
            burst.set_sizes([100 + 900 * grow])
        ax.view_init(elev=20 + 6 * np.sin(fi * 0.05), azim=(-50 + fi * 1.1) % 360)
        return artists

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=False)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(save_path), writer=PillowWriter(fps=fps))
    if show:
        plt.show()
    else:
        plt.close(fig)
    return anim


def animate_swarm_3d_modern(
    result,
    *,
    title: str | None = None,
    defended=(0.0, 0.0, 0.0),
    save_path=None,
    fps: int = 30,
    max_frames: int = 200,
    tail: int = 70,
    show: bool = False,
):
    """Cinematic 3-D **swarm** replay: interceptors (cyan) vs. many diverse threats (warm hues),
    neon glow trails, an intercept flash at each kill, a defended-point marker, and a slow orbit.

    Takes a :class:`~intercept.multiagent.swarm.MultiEngagementResult`.
    """
    tracks, roles = result.tracks, result.roles
    times = np.asarray(result.times)
    n = len(times)
    idx = np.arange(n) if n <= max_frames else np.linspace(0, n - 1, max_frames).astype(int)
    allpts = np.vstack([s[:, :3] for s in tracks.values()] + [np.asarray(defended)[None, :]])

    fig = plt.figure(figsize=(9, 7.5), facecolor=_BG)
    ax = fig.add_subplot(projection="3d")
    _style_dark(ax)
    _set_equal_aspect(ax, allpts)
    for lbl, axis in (("x (m)", ax.xaxis), ("y (m)", ax.yaxis), ("z (m)", ax.zaxis)):
        axis.label.set_text(lbl)
    kept = result.n_targets - result.leakers
    ax.set_title(
        title or f"Swarm defense   {kept}/{result.n_targets} intercepted",
        color="#e6edf3",
        fontsize=13,
        pad=12,
    )

    targets = [nm for nm, r in roles.items() if r == "target"]
    warm = plt.cm.autumn(np.linspace(0.0, 0.75, max(1, len(targets))))
    color = {}
    for nm, r in roles.items():
        color[nm] = "#19e6ff" if r == "interceptor" else warm[targets.index(nm)]

    glow = {nm: _glow_plot(ax, [], [], [], color[nm]) for nm in tracks}
    heads = {}
    for nm in tracks:
        (soft,) = ax.plot([], [], [], "o", color=color[nm], ms=12, alpha=0.25)
        (core,) = ax.plot(
            [],
            [],
            [],
            "o",
            color="white",
            ms=4,
            alpha=0.95,
            markeredgecolor=color[nm],
            markeredgewidth=1.2,
        )
        heads[nm] = (soft, core)
    d = np.asarray(defended, float)
    ax.scatter([d[0]], [d[1]], [d[2]], c="#19e6ff", marker="^", s=80, alpha=0.5, depthshade=False)
    bursts = ax.scatter([], [], [], s=[], c=_BURST, marker="*", depthshade=False)
    kill_t = np.array([k[2] for k in result.kills]) if result.kills else np.array([])
    kill_p = np.array(result.kill_points) if result.kill_points else np.zeros((0, 3))

    def _alive_len(nm, k):
        """Frozen-after-death index: tracks are padded, so clamp to the last moving sample."""
        return min(k, len(tracks[nm]) - 1)

    def update(fi):
        k = idx[fi]
        t = times[min(k, n - 1)]
        artists = []
        for nm in tracks:
            p = tracks[nm][:, :3]
            kk = _alive_len(nm, k)
            seg = p[max(0, kk - tail) : kk + 1]
            for ln in glow[nm]:
                ln.set_data(seg[:, 0], seg[:, 1])
                ln.set_3d_properties(seg[:, 2])
            soft, core = heads[nm]
            for h in (soft, core):
                h.set_data([p[kk, 0]], [p[kk, 1]])
                h.set_3d_properties([p[kk, 2]])
            artists += [*glow[nm], soft, core]
        if len(kill_t):
            fired = kill_t <= t
            if fired.any():
                pts = kill_p[fired]
                ages = np.clip((t - kill_t[fired]) / 0.6, 0.0, 1.0)
                bursts._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
                bursts.set_sizes(120 + 700 * (1.0 - ages))
        ax.view_init(elev=24 + 6 * np.sin(fi * 0.04), azim=(-55 + fi * 0.9) % 360)
        return artists

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=False)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(save_path), writer=PillowWriter(fps=fps))
    if show:
        plt.show()
    else:
        plt.close(fig)
    return anim


def plot_engagement_3d_modern(
    result: EngagementResult,
    *,
    roles=None,
    title: str | None = None,
    elev: float = 22,
    azim: float = -55,
    save_path=None,
    show: bool = False,
):
    """Static cinematic 3-D frame (dark theme + neon glow) — for inline viewing / thumbnails."""
    roles = _roles(result, roles)
    allpts = np.vstack([s[:, :3] for s in result.states.values()])
    fig = plt.figure(figsize=(8.5, 7), facecolor=_BG)
    ax = fig.add_subplot(projection="3d")
    _style_dark(ax)
    _set_equal_aspect(ax, allpts)
    ax.view_init(elev=elev, azim=azim)
    for name, states in result.states.items():
        c = _MODERN.get(roles.get(name, "entity"), "#7CFC00")
        p = states[:, :3]
        _glow_plot(ax, p[:, 0], p[:, 1], p[:, 2], c)
        ax.plot(
            [p[-1, 0]],
            [p[-1, 1]],
            [p[-1, 2]],
            "o",
            color="white",
            ms=6,
            markeredgecolor=c,
            markeredgewidth=1.5,
        )
    ti = int(np.argmin(np.abs(result.times - result.closest_approach_time)))
    mid = 0.5 * (result.states[result.interceptor][ti, :3] + result.states[result.target][ti, :3])
    if result.intercepted:
        ax.scatter([mid[0]], [mid[1]], [mid[2]], s=600, c=_BURST, marker="*", depthshade=False)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    outcome = "INTERCEPT" if result.intercepted else result.reason.name
    ax.set_title(
        title or f"{outcome}   miss {result.miss_distance:.1f} m",
        color="#e6edf3",
        fontsize=13,
        pad=12,
    )
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax
