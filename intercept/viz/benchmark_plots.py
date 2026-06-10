"""Benchmark visualization: capture-region heatmaps and probability-of-intercept bar charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from intercept.benchmark.capture_region import CaptureRegion


def _save_show(fig, save_path, show):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()


def plot_capture_region(
    region: CaptureRegion,
    *,
    metric: str = "miss",
    miss_clip: float = 200.0,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Heatmap of a capture-region sweep over target start positions.

    ``metric="miss"`` shows miss distance (clipped at ``miss_clip``); ``metric="hit"`` shows the
    binary intercept region. The interceptor launch point (origin) is marked.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))
    if metric == "hit":
        mesh = ax.pcolormesh(
            region.downrange,
            region.crossrange,
            region.intercepted.astype(float),
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            shading="auto",
        )
        cbar_label = "intercept (1) / miss (0)"
    else:
        data = np.minimum(region.miss_distance, miss_clip)
        mesh = ax.pcolormesh(
            region.downrange,
            region.crossrange,
            data,
            cmap="viridis_r",
            vmin=0,
            vmax=miss_clip,
            shading="auto",
        )
        cbar_label = f"miss distance (m, clipped at {miss_clip:.0f})"
        # Outline the capture boundary.
        ax.contour(
            region.downrange,
            region.crossrange,
            region.intercepted.astype(float),
            levels=[0.5],
            colors="white",
            linewidths=1.5,
        )

    ax.plot(0, 0, "w*", ms=16, mew=1.5, label="interceptor launch")
    fig.colorbar(mesh, ax=ax, label=cbar_label)
    ax.set_title(
        f"Capture region — {region.algorithm} on '{region.scenario_name}' "
        f"(capture {region.capture_fraction * 100:.0f}%)"
    )
    ax.set_xlabel("target initial down-range x (m)")
    ax.set_ylabel("target initial cross-range y (m)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _save_show(fig, save_path, show)
    return fig, ax


def plot_pintercept_bars(
    rows,
    *,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Grouped bar chart of probability-of-intercept (with Wilson CI error bars) per scenario.

    ``rows`` is a list of :class:`~intercept.benchmark.runner.BenchmarkRow`.
    """
    scenarios = sorted({r.scenario for r in rows})
    algorithms = sorted({r.algorithm for r in rows})
    by_cell = {(r.scenario, r.algorithm): r.summary for r in rows}

    x = np.arange(len(scenarios))
    width = 0.8 / max(1, len(algorithms))
    fig, ax = plt.subplots(figsize=(1.6 * len(scenarios) + 3, 6))

    for k, algo in enumerate(algorithms):
        p, lo_err, hi_err = [], [], []
        for sc in scenarios:
            s = by_cell.get((sc, algo))
            if s is None:
                p.append(0.0)
                lo_err.append(0.0)
                hi_err.append(0.0)
                continue
            p.append(s.p_intercept)
            lo_err.append(s.p_intercept - s.p_intercept_lo)
            hi_err.append(s.p_intercept_hi - s.p_intercept)
        # Clamp tiny negative float errors (Wilson hi == p at 100%); matplotlib rejects yerr<0.
        yerr = np.clip(np.array([lo_err, hi_err]), 0.0, None)
        ax.bar(
            x + k * width, p, width, label=algo, yerr=yerr, capsize=3, error_kw={"elinewidth": 1}
        )

    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(intercept)  (95% Wilson CI)")
    ax.set_title("Probability of intercept by scenario and guidance law")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save_show(fig, save_path, show)
    return fig, ax
