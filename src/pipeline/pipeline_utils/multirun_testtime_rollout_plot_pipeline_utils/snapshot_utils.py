"""Snapshot utilities for per-trajectory test-time rollout plots.

This module provides a tiny, dependency-light surface to embed a 1/8-scaled
3D ground-truth snapshot in the upper-left corner of a test-time rollout
plot (``show.breakdown: 'trajectory'``).

Design decisions (plan §4 R2 mitigation):
- :func:`render_trajectory_ground_truth_snapshot` is the *extension point* that
  will eventually call ``math_gymnasium.tools.plot_3d_utils.three_dimension_environment_space_plot``
  (per operator answer to Q2). It is intentionally isolated so tests can
  monkeypatch it without touching matplotlib.
- If the rendering call raises for any reason (missing backend, missing
  dependency, degenerate trajectory data, …), the helper returns ``None``
  and :func:`add_snapshot_inset` silently skips the inset — the main plot
  still renders. This keeps the new ``breakdown='trajectory'`` path robust
  in CI/headless environments.
- A process-local cache keyed by ``trajectory_name`` avoids re-rendering
  the same ground-truth snapshot across the N per-trajectory plots of a
  single pipeline run.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

__all__ = [
    "render_trajectory_ground_truth_snapshot",
    "add_snapshot_inset",
    "clear_snapshot_cache",
    "DEFAULT_SNAPSHOT_SCALE",
    "DEFAULT_SNAPSHOT_CONTOUR_COLOR",
    "DEFAULT_SNAPSHOT_CONTOUR_LW",
]

logger = logging.getLogger(__name__)

#: Default scale of the snapshot inset relative to the figure canvas.
#: 3x the original 1/8 sizing so the 3D ground-truth is actually readable.
DEFAULT_SNAPSHOT_SCALE: float = 3.0 / 8.0
#: Default contour color (thin gray) around the snapshot inset.
DEFAULT_SNAPSHOT_CONTOUR_COLOR: str = "#B0B0B0"
#: Default contour linewidth.
DEFAULT_SNAPSHOT_CONTOUR_LW: float = 0.5

# Process-local snapshot cache: {trajectory_name: np.ndarray | None}
_SNAPSHOT_CACHE: dict[str, Optional[np.ndarray]] = {}


def clear_snapshot_cache() -> None:
    """Reset the per-process snapshot cache. Intended for tests."""
    _SNAPSHOT_CACHE.clear()


def render_trajectory_ground_truth_snapshot(
    trajectory_name: str,
    entry: Any = None,
    figsize: tuple[float, float] = (4.0, 4.0),
    dpi: int = 150,
) -> Optional[np.ndarray]:
    """Render a small 3D ground-truth snapshot for ``trajectory_name``.

    The first successful render is cached per ``trajectory_name``.

    This function wraps
    ``math_gymnasium.tools.plot_3d_utils.three_dimension_environment_space_plot``
    (per plan §4.1 / operator Q2). On any failure it logs a warning and
    returns ``None`` so callers can gracefully skip the inset (plan R2).

    :param trajectory_name: ground-truth trajectory identifier (used as cache key).
    :param entry: optional :class:`TestTrajectoryEntry` holding the 3D poses
        for the trajectory. If ``None`` or missing poses, rendering is skipped.
    :param figsize: figsize of the offscreen figure used to rasterize the axe.
    :param dpi: dpi of the offscreen figure.
    :return: RGBA image as ``np.ndarray`` of shape ``(H, W, 4)``, or ``None``
        if rendering was not possible.
    """
    if trajectory_name in _SNAPSHOT_CACHE:
        return _SNAPSHOT_CACHE[trajectory_name]

    if entry is None:
        _SNAPSHOT_CACHE[trajectory_name] = None
        return None

    try:
        # Render the offscreen snapshot WITHOUT touching pyplot's global
        # figure manager. Using ``plt.figure(...)`` here would register the
        # offscreen fig with the pyplot manager; PyCharm's
        # ``backend_interagg`` watches that manager and dispatches figures
        # to SciView based on it — interleaving N offscreen create/close
        # cycles with the N real figures caused only the first real figure
        # to surface in SciView.
        #
        # Using ``matplotlib.figure.Figure`` + a local ``FigureCanvasAgg``
        # keeps the offscreen rendering completely invisible to pyplot.
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        # ``mpl_toolkits.mplot3d`` must be imported so the ``"3d"`` projection
        # is registered on the Axes factory.
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        # Imported for future use when we migrate to the full
        # ``three_dimension_environment_space_plot`` rendering (plan Q2).
        from math_gymnasium.tools.plot_3d_utils import (  # noqa: F401
            three_dimension_environment_space_plot,
        )

        # ``TestTrajectoryEntry`` wraps a ``TestMotionTrajectoryDataclass``
        # exposing ``pose_gt`` (ground-truth 3D poses). Fall back to a few
        # alternative attribute paths for robustness across fixtures/tests.
        poses = None
        env_obj = getattr(entry, "env", None)
        if env_obj is not None:
            poses = getattr(env_obj, "pose_gt", None)
            if poses is None:
                poses = getattr(env_obj, "pose", None)
        if poses is None:
            poses = getattr(entry, "pose_gt", None)
        if poses is None:
            poses = getattr(entry, "poses", None)
        if poses is None:
            poses = getattr(entry, "ground_truth_poses", None)
        if poses is None:
            raise ValueError(
                f"TestTrajectoryEntry for {trajectory_name!r} has no poses attribute"
            )

        fig = Figure(figsize=figsize, dpi=dpi)
        FigureCanvasAgg(fig)  # attach Agg canvas (side-effect: fig.canvas set)
        ax3d = fig.add_subplot(111, projection="3d")
        arr = np.asarray(poses)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            ax3d.plot(arr[:, 0], arr[:, 1], arr[:, 2], linewidth=1.2)
        ax3d.grid(True)
        for _axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            _axis.set_ticklabels([])
        ax3d.tick_params(axis="both", which="both", length=2, pad=0)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
        # No ``plt.close`` needed: the figure was never registered with pyplot.
        _SNAPSHOT_CACHE[trajectory_name] = rgba
        return rgba
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning(
            "Snapshot rendering failed for trajectory %r: %s — skipping inset.",
            trajectory_name,
            exc,
        )
        _SNAPSHOT_CACHE[trajectory_name] = None
        return None


def add_snapshot_inset(
    ax_main: Any,
    snapshot_img: Optional[np.ndarray],
    scale: float = DEFAULT_SNAPSHOT_SCALE,
    contour_color: str = DEFAULT_SNAPSHOT_CONTOUR_COLOR,
    contour_lw: float = DEFAULT_SNAPSHOT_CONTOUR_LW,
) -> Optional[Any]:
    """Embed ``snapshot_img`` in the upper-left corner of ``ax_main``.

    :param ax_main: main matplotlib ``Axes`` to decorate.
    :param snapshot_img: RGBA array as returned by
        :func:`render_trajectory_ground_truth_snapshot`. If ``None`` the inset
        is silently skipped.
    :param scale: scale of the inset relative to the main axes (default 1/8).
    :param contour_color: outline color of the inset (default thin gray).
    :param contour_lw: outline linewidth.
    :return: the inset ``Axes`` on success, ``None`` when skipped.
    """
    if snapshot_img is None:
        return None
    try:
        # Upper-left inset anchored to the **main axes** (not the figure
        # canvas), so the snapshot sits inside the plot area and cannot
        # occlude the figure's y-axis title. ``scale`` is expressed in
        # axes-fraction coordinates; we keep the inset visually square by
        # compensating for the (wide) main axes aspect ratio via the axes
        # bbox width/height in display units.
        fig = ax_main.figure
        bbox = ax_main.get_position()
        ax_w_in = bbox.width * fig.get_size_inches()[0]
        ax_h_in = bbox.height * fig.get_size_inches()[1]
        h_frac = scale
        # Compensate so the inset is square in display inches.
        w_frac = scale * (ax_h_in / ax_w_in) if ax_w_in > 0 else scale
        margin = 0.01  # axes-fraction
        left = margin
        bottom = 1.0 - h_frac - margin
        inset_ax = ax_main.inset_axes(
            [left, bottom, w_frac, h_frac], zorder=5
        )
        inset_ax.imshow(snapshot_img)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        for spine in inset_ax.spines.values():
            spine.set_edgecolor(contour_color)
            spine.set_linewidth(contour_lw)
        # Make sure the legend is repositioned so it never overlaps the inset.
        legend = ax_main.get_legend()
        if legend is not None:
            # Move legend to upper-right of the main axes.
            try:
                ax_main.legend(loc="upper right")
            except Exception:
                pass
        return inset_ax
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("add_snapshot_inset failed: %s — skipping inset.", exc)
        return None
