# coding=utf-8
import glob as _glob_mod
import math
import os
import re
import warnings
from typing import Any, Optional, Sequence

import numpy as np
import omegaconf
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.transforms import ScaledTranslation
from omegaconf import DictConfig

from algorithm.experience_replay_learning_loop.core.data_classes import (
    TestTimeRolloutPredictionMetric,
    get_metric_sub_dir,
)
from pipeline.pipeline_utils.multirun_testtime_rollout_plot_pipeline_utils.category_utils import (
    _get_category_max_length,
)
from pipeline.pipeline_utils.multirun_testtime_rollout_plot_pipeline_utils.main_utils import (
    _aggregate_group_over_all_trajectories,
    _aggregate_group_over_selected_trajectories,
    _build_entries,
    _build_trajectory_length_map,
    _format_grp_count_label,
    display_experiment_name,
)
from pipeline.pipeline_utils.multirun_testtime_rollout_plot_pipeline_utils.show_options_utils import (
    resolve_general_length,
    select_trajectories_by_length,
)
from pipeline.pipeline_utils.multirun_testtime_rollout_plot_pipeline_utils.snapshot_utils import (
    add_snapshot_inset,
    render_trajectory_ground_truth_snapshot,
)
from tools.benchmark_tools.benchmark_metric import (
    BenchmarkLevel,
    BenchmarkMetricSet,
    FeedbackRegime,
    TimingPass,
)
from pipeline.pipeline_utils.multirun_testtime_rollout_plot_pipeline_utils.trajectory_processing_utils import (
    _resolve_current_device,
    _resolve_simulator_config_path,
    _resolve_test_trajectories_split,
    apply_simulator_config_override,
    persist_rollout_relevant_cfg_overrides,
)
from tools.console_tools.message import consol_msg_universal_one_liner
from tools.experiment_data_consolidation.consolidation_guard import consolidation_guard
from tools.experiment_data_consolidation.consolidation_paths import (
    is_consolidation_enabled,
)
from tools.experiment_data_consolidation.drift_rate_consolidation import (
    DRIFT_TYPE_UNIT,
    build_drift_rate_rows,
    build_drift_rate_seed_tables,
    build_trajectory_name_mapping,
    consolidate_drift_rate,
    consolidate_drift_rate_samples,
    resolve_horizon_ratio_timesteps,
)
from tools.experiment_data_consolidation.wall_clock_consolidation import (
    build_training_wall_clock_time_row,
    consolidate_training_wall_clock_time,
)
from tools.hydra_apps_tools.hydra_utils import (
    fetch_project_root_path_via_hydra,
    get_hydra_original_cwd,
)
from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist
from tools.hydra_apps_tools.r2s2r_apps_utils import fetch_r2s2r_project_root_path
from tools.plot_tools.plot_general_utils import arbitrary_dimension_array_plot
from tools.plot_tools.plot_management import (
    manage_matplotlib_backend,
    manage_matplotlib_warnings,
    show_and_save_plot_helper,
)

# (NICE TO HAVE) ToDo: RLRP-801 refactor: consolidate matplotlib related cfg pipeline file
from tools.plot_tools.style import AXIS_LABEL_STYLE

# (NICE TO HAVE) ToDo: RLRP-801 refactor: consolidate matplotlib related cfg pipeline file
# InProgress: RLRP-802 style(multirun plot): improve color matrix
COLORS = [
    "mediumorchid",  # Replace tab:purple
    "tab:orange",
    # "tab:cyan",
    "mediumseagreen",
    "black",
    # "tab:red",
    "red",  # Replace tab:red
    "royalblue",
    "tab:green",
    "dimgrey",
    "magenta",
    "dodgerblue",
    "gray",  # Replace tab:gray
    # "olivedrab",
    "darkred",  # Replace tab:brown
]


def get_plot_render_type_cfg(cfg: DictConfig) -> DictConfig:
    if cfg.render_type == "standalone":
        return cfg.pipeline.plot
    elif cfg.render_type == "latex_includegraphics":
        return cfg.pipeline.plot_latex_includegraphics
    else:
        raise NotImplementedError(
            f"Figure render type {cfg.render_type} not supported!"
        )


def is_latex_includegraphics_render_type(cfg: DictConfig) -> bool:
    assert (
        cfg.render_type == "standalone" or cfg.render_type == "latex_includegraphics"
    ), f"Figure render type {cfg.render_type} not supported!"

    return cfg.render_type == "latex_includegraphics"


# Separator used in a figure-size spec string to apply a scale factor to a
# preset, e.g. ``"double_column_width * 0.5"`` -> half the double-column width.
_FIGSIZE_SCALE_SEP = "*"


def _fetch_figsize_preset_value(cfg: DictConfig, preset_name: str) -> float:
    """Resolve a named figure-size preset declared in the plot pipeline config.

    Presets are scalar keys declared under the ``figsize_preset`` group of a
    render-type config group, e.g.
    ``plot_latex_includegraphics.figsize_preset.single_column_width`` and
    ``plot_latex_includegraphics.figsize_preset.double_column_width``. The active
    render-type group is searched first, then the other groups so a preset
    declared only once (e.g. the LaTeX column widths) stays usable from every
    render type.

    :param cfg: Top-level multirun plot configuration.
    :param preset_name: The preset key name (e.g. ``"single_column_width"``).
    :return: The preset value in inches.
    :raise KeyError: When no config group declares ``preset_name``.
    """
    candidates: list[Any] = [get_plot_render_type_cfg(cfg)]
    for group_name in ("plot_latex_includegraphics", "plot"):
        try:
            candidates.append(cfg.pipeline[group_name])
        except (omegaconf.errors.ConfigAttributeError, AttributeError, KeyError):
            pass

    for candidate in candidates:
        if candidate is None:
            continue
        preset_group = candidate.get("figsize_preset", None)
        if preset_group is None:
            continue
        value = preset_group.get(preset_name, None)
        if value is not None:
            return float(value)

    raise KeyError(
        f"Unknown figure-size preset '{preset_name}'. Declare it in the plot "
        f"pipeline config (e.g. "
        f"'plot_latex_includegraphics.figsize_preset.{preset_name}') or use an "
        f"arbitrary number instead."
    )


def _resolve_figsize_dimension(cfg: DictConfig, spec: Any, fallback: Any) -> float:
    """Resolve a single figure-size dimension (width or height) spec, in inches.

    Accepted ``spec`` forms:
      - ``None``                      -> ``fallback`` (the render-type ``figsize`` value)
      - a number (e.g. ``5.5``)       -> that arbitrary value, in inches
      - a preset name (e.g. ``"single_column_width"``) -> the preset value
      - a scaled preset (e.g. ``"double_column_width * 0.5"``) -> preset x factor

    :param cfg: Top-level multirun plot configuration.
    :param spec: The user-provided dimension spec (see above).
    :param fallback: Value used when ``spec`` is ``None``.
    :return: The resolved dimension, in inches.
    """
    if spec is None:
        return float(fallback)
    if isinstance(spec, bool):
        raise TypeError(f"Invalid figure-size dimension spec: {spec!r}")
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, str):
        raw = spec.strip()
        scale = 1.0
        if _FIGSIZE_SCALE_SEP in raw:
            preset_part, _, scale_part = raw.partition(_FIGSIZE_SCALE_SEP)
            raw = preset_part.strip()
            scale = float(scale_part.strip())
        try:
            return float(raw) * scale
        except ValueError:
            pass
        return _fetch_figsize_preset_value(cfg, raw) * scale

    raise TypeError(
        f"Unsupported figure-size dimension spec type: {type(spec)} ({spec!r})"
    )


def _fetch_show_cfg_chain(cfg: DictConfig, show_key: str | None) -> list[Any]:
    """Walk ``cfg.show`` along a (possibly dotted) graphic-type key.

    :param cfg: Top-level multirun plot configuration.
    :param show_key: A ``cfg.show`` sub-group name, optionally dotted for a
        nested group (e.g. ``"generate_drift_rate_plot.per_train_epoch"``).
    :return: The visited config nodes, outermost first (missing tail nodes are
        simply omitted so an absent sub-group is a no-op).
    """
    if not show_key:
        return []
    try:
        node: Any = cfg.show
    except (omegaconf.errors.ConfigAttributeError, AttributeError, KeyError):
        return []

    chain: list[Any] = []
    for part in show_key.split("."):
        if not isinstance(node, DictConfig):
            break
        node = node.get(part, None)
        if node is None:
            break
        chain.append(node)
    return chain


def get_plot_figsize(
    cfg: DictConfig,
    show_key: str | None = None,
    *,
    default_width_scale: float = 1.0,
    default_height_scale: float = 1.0,
) -> list[float]:
    """Resolve the ``figsize`` of a plot, honoring per-graphic-type overrides.

    The render-type ``figsize`` (``plot.figsize`` /
    ``plot_latex_includegraphics.figsize``) remains the default. Any graphic
    type may however personalize its own width and/or height from its
    ``cfg.show`` sub-group, per render type::

        show:
          generate_drift_rate_plot:
            width_screen: 35            # arbitrary number (inches)
            height_screen: 14
            width_latex: double_column_width   # preset name
            height_latex: 1.4

    ``width_latex``/``height_latex`` apply to the LaTeX ``includegraphics``
    render type, ``width_screen``/``height_screen`` to the standalone one. Each
    value is either an arbitrary number, the name of a preset declared in the
    plot pipeline config (e.g. ``single_column_width``,
    ``double_column_width``), or a scaled preset (``"double_column_width * 0.5"``).
    Omitted keys fall back to the render-type ``figsize``; for a dotted
    ``show_key`` the nested sub-group inherits from its parent sub-group.

    :param cfg: Top-level multirun plot configuration.
    :param show_key: The ``cfg.show`` sub-group holding the per-graphic-type
        overrides (dotted for a nested group); ``None`` -> no override lookup.
    :param default_width_scale: Scale applied to the fallback width ONLY (i.e.
        when the graphic type declares no explicit width override).
    :param default_height_scale: Same as ``default_width_scale``, for the height.
    :return: The ``[width, height]`` figure size, in inches.
    """
    base = list(get_plot_render_type_cfg(cfg).figsize)
    suffix = "latex" if is_latex_includegraphics_render_type(cfg) else "screen"

    width_spec: Any = None
    height_spec: Any = None
    # Deepest sub-group wins; shallower ones act as inherited defaults.
    for node in reversed(_fetch_show_cfg_chain(cfg, show_key)):
        if not isinstance(node, DictConfig):
            continue
        if width_spec is None:
            width_spec = node.get(f"width_{suffix}", None)
        if height_spec is None:
            height_spec = node.get(f"height_{suffix}", None)
        if width_spec is not None and height_spec is not None:
            break

    width = _resolve_figsize_dimension(cfg, width_spec, base[0])
    height = _resolve_figsize_dimension(cfg, height_spec, base[1])
    if width_spec is None:
        width *= float(default_width_scale)
    if height_spec is None:
        height *= float(default_height_scale)
    return [width, height]


def _write_latex_includegraphics_meta_file(
    cfg: DictConfig,
    save_dir: str,
    meta_file_name: str,
    *,
    title_text: str | None,
    legend_labels: list[str] | None = None,
) -> str | None:
    """Consolidate a LaTeX-``includegraphics`` figure's textual meta info to a file.

    In the LaTeX ``includegraphics`` render type the on-figure title
    (:meth:`~matplotlib.axes.Axes.set_title`), the experiment-name overlay
    (:func:`display_experiment_name`) and the per-model-group legend are
    intentionally suppressed (``print_title``/``print_meta_info``/``print_legend``
    are ``False`` or the legend uses a compact short name) so the exported figure
    stays clean for embedding in the paper. To avoid losing that provenance,
    this helper writes the same information — the experiment name (``cfg.experiment``),
    the plot title and the standalone legend entries (each model group name with
    its number of model seeds and trajectories) — to a sidecar ``*.text`` file
    placed next to the saved figure (``save_dir``).

    :param cfg: Top-level multirun plot configuration.
    :param save_dir: Directory where the LaTeX figure is saved (the meta file is
        written alongside it); created if missing.
    :param meta_file_name: Name of the meta text file to write.
    :param title_text: The plot title text (as would have been passed to
        :meth:`~matplotlib.axes.Axes.set_title`); ``None``/empty is skipped.
    :param legend_labels: The per-model-group legend entries as rendered in the
        ``standalone`` render type (each carries the group name and its number of
        model seeds / trajectories). ``None``/empty is skipped.
    :return: The absolute path of the written meta file, or ``None`` when there
        was nothing to write.
    """
    lines: list[str] = []
    try:
        experiment_name = cfg.experiment
    except (omegaconf.errors.ConfigAttributeError, AttributeError, KeyError):
        experiment_name = None
    if experiment_name is not None:
        lines.append(f"Experiment name: {experiment_name}")


    if title_text:
        lines.append(str(title_text))

    if legend_labels:
        lines.append("Model groups (name, model seeds and trajectories):")
        for _legend_label in legend_labels:
            if _legend_label:
                lines.append(f"  - {_legend_label}")

    if not lines:
        return None

    os.makedirs(save_dir, exist_ok=True)
    meta_path = os.path.join(save_dir, meta_file_name)
    lines.append(f"Path to original: {save_dir}")

    with open(meta_path, "w", encoding="utf-8") as _meta_fh:
        _meta_fh.write("\n".join(lines))
        _meta_fh.write("\n")
    return meta_path


# Conversion factor from 1 second to the target time unit.
# The preprocessing pipeline resamples trajectories onto a target frame rate
# grid expressed in SECONDS (see ``online_pre_processing``), so timesteps map
# to seconds via ``1 / target_frame_rate``. The secondary x-axis then
# rescales seconds to the original timestamp unit declared in the simulator
# config (``data.original_timestamps_unit``).
_SECONDS_TO_UNIT_FACTOR = {
    "s": 1.0,
    "ms": 1e3,
    "us": 1e6,
    "ns": 1e9,
}

_RATE_UNIT_LABEL = {"hz": "Hz", "fps": "fps"}


def _resolve_timestamp_units(cfg: DictConfig) -> tuple[Any, Optional[str]]:
    """Resolve the timestep -> timestamp conversion inputs from ``cfg``.

    Shared, matplotlib-free core of :func:`_attach_timestamp_secondary_xaxis`
    (extracted by RLRP-818 so the experiment data consolidation emits the very
    same ``timestamp`` values the figure displays -- one single rule, one
    implementation).

    The multirun_testtime_rollout_plot pipeline does NOT load
    ``/simulator@environment`` into the top-level cfg -- only ``simulator_config``
    (a string filename) is provided. Resolve the simulator YAML on demand
    (mirrors the pattern in ``trajectory_processing_utils.apply_simulator_config_override``).
    Fall back to ``cfg.environment.data`` when it does happen to be present
    (e.g. in pipelines where the simulator is composed via Hydra defaults).

    :param cfg: Top-level multirun plot configuration.
    :return: ``(target_frame_rate, original_timestamps_unit)``; either element
        is ``None`` when it cannot be resolved.
    """
    target_frame_rate = None
    unit = None
    try:
        env_data = cfg.environment.data  # may raise if absent
        target_frame_rate = env_data.get("target_frame_rate", None)

        # (CRITICAL) ToDo: force timestamps axis to display in m/s
        unit = env_data.get("original_timestamps_unit", None)

    except (omegaconf.errors.ConfigAttributeError, AttributeError):
        pass

    if target_frame_rate is None or unit is None:
        sim_name = cfg.get("simulator_config", None) if hasattr(cfg, "get") else None
        if sim_name:
            try:
                # Walk Hydra ``defaults:`` chain manually: ``OmegaConf.load`` does
                # NOT resolve Hydra composition, so ``data.target_frame_rate`` and
                # ``data.original_timestamps_unit`` (which live in a parent like
                # ``quadcopter_general.yaml`` / ``ugv_general.yaml``) are missing
                # unless we merge the parent configs ourselves.
                visited: set[str] = set()
                merged = omegaconf.OmegaConf.create({})
                _stack: list[str] = [sim_name]
                _ordered: list[str] = []
                while _stack:
                    _name = _stack.pop()
                    if _name in visited:
                        continue
                    visited.add(_name)
                    _ordered.append(_name)
                    try:
                        _loaded = omegaconf.OmegaConf.load(
                            _resolve_simulator_config_path(_name)
                        )
                    except (
                        FileNotFoundError,
                        omegaconf.errors.OmegaConfBaseException,
                    ):
                        continue
                    _defaults = _loaded.get("defaults", []) or []
                    for _d in _defaults:
                        # Entries can be strings (``quadcopter_general``) or dicts.
                        # Skip Hydra's ``_self_`` sentinel (not a file).
                        if isinstance(_d, str) and _d != "_self_":
                            _stack.append(_d)
                # Merge from base parents (end of ordered list) toward the leaf
                # (sim_name) so leaf values override parents -- matches Hydra.
                for _name in reversed(_ordered):
                    try:
                        _loaded = omegaconf.OmegaConf.load(
                            _resolve_simulator_config_path(_name)
                        )
                    except (
                        FileNotFoundError,
                        omegaconf.errors.OmegaConfBaseException,
                    ):
                        continue
                    merged = omegaconf.OmegaConf.merge(merged, _loaded)
                sim_data = merged.get("data", None)
                if sim_data is not None:
                    if target_frame_rate is None:
                        target_frame_rate = sim_data.get("target_frame_rate", None)
                    if unit is None:
                        unit = sim_data.get("original_timestamps_unit", None)
            except (FileNotFoundError, omegaconf.errors.OmegaConfBaseException):
                pass

    return target_frame_rate, unit


def _resolve_timestamp_scale(
    target_frame_rate: Any, unit: Optional[str]
) -> tuple[Optional[float], Optional[str]]:
    """Turn the resolved conversion inputs into a timestep -> timestamp factor.

    ``timestamp = timestep * scale``. The ``ns`` unit is force-promoted to
    seconds for legibility (unchanged legacy behaviour of the figure's time
    annotation).

    :param target_frame_rate: The simulator target frame rate (Hz).
    :param unit: The declared ``original_timestamps_unit``.
    :return: ``(scale, resolved_unit)``, or ``(None, None)`` when the
        conversion is not resolvable / not supported.
    """
    if not target_frame_rate or unit is None or unit not in _SECONDS_TO_UNIT_FACTOR:
        return None, None

    seconds_per_step = 1.0 / float(target_frame_rate)
    if unit == "ns":
        # Switch for lisibility
        unit = "s"
    return seconds_per_step * _SECONDS_TO_UNIT_FACTOR[unit], unit


def _resolve_timestamp_conversion(
    cfg: DictConfig,
) -> tuple[Optional[float], Optional[str]]:
    """Resolve the timestep -> timestamp conversion of the current experiment.

    Single source of truth shared by the figure's time annotation
    (:func:`_attach_timestamp_secondary_xaxis`) and the RLRP-818 experiment
    data consolidation. Deliberately NOT gated by
    ``cfg.show.show_timestamp_xaxis``: that toggle only controls the decorative
    axis, whereas the consolidated CSV must carry its time column regardless.

    :param cfg: Top-level multirun plot configuration.
    :return: ``(scale, unit)`` such that ``timestamp = timestep * scale``, or
        ``(None, None)`` when unresolvable.
    """
    return _resolve_timestamp_scale(*_resolve_timestamp_units(cfg))


def _attach_timestamp_secondary_xaxis(
    cfg: DictConfig, ax: Axes, *, enforce_right_endpoint: bool = False
) -> None:
    """Annotate ``ax`` with a horizontal time-direction arrow below the timestep axis.

    Gated by ``cfg.show.show_timestamp_xaxis`` (default ``False``). The
    primary x-axis remains in timesteps. Instead of a full secondary axis
    with numeric ticks, this draws a horizontal line ending in an arrow
    head on the right side, below the timestep x-axis label, with the time
    unit (e.g. ``"[s]"``) written just below the arrow tip. This conveys
    the direction of time without consuming as much vertical real estate
    as a tick-bearing secondary axis would.

    The active simulator config is consulted only to retrieve
    ``data.original_timestamps_unit``; ``target_frame_rate`` is no longer
    strictly required (kept for symmetry / future use, but the arrow
    annotation does not depend on it).

    Only when ``enforce_right_endpoint`` is ``True`` is the right (arrow-tip)
    endpoint given a forced numeric timestamp label; the left endpoint and the
    interior ticks are always drawn. When it is ``False`` the right endpoint
    timestamp tick is dropped. This mirrors the ``enforce_right_endpoint`` flag
    of :func:`_ensure_primary_xaxis_endpoint_ticks`; the call site passes
    ``enforce_right_endpoint=not is_latex_includegraphics_render_type(cfg)`` so
    the right endpoint is labeled for the standalone render type but NOT for the
    LaTeX ``includegraphics`` render type, where a forced right-endpoint label
    spills past the (very narrow) axes frame. The default ``False`` therefore
    draws the left endpoint + interior ticks only.

    No-op when the toggle is off or when the unit cannot be resolved /
    is outside the supported set.

    :param cfg: Top-level multirun plot configuration.
    :param ax: The plot axis to annotate with the timestamps arrow axis.
    :param enforce_right_endpoint: When ``True`` force a numeric timestamp
        label at the right (arrow-tip) endpoint as well; when ``False`` only the
        left endpoint and interior ticks are labeled. Defaults to ``False``.
    """
    if not cfg.show.get("show_timestamp_xaxis", False):
        return

    # Conversion inputs resolved by the shared, matplotlib-free helper
    # (RLRP-818): the experiment data consolidation emits its ``timestamp``
    # column with the exact same rule.
    target_frame_rate, unit = _resolve_timestamp_units(cfg)

    if unit is None:
        return
    if unit not in _SECONDS_TO_UNIT_FACTOR:
        return

    # Place the arrow below the timestep x-axis label, in axes-fraction
    # coordinates (x: 0..1 across the data span; y: negative => below the
    # axes frame). The arrow head sits on the right side; the time-unit
    # label sits right next to the arrow tip, and numeric timestamp values
    # are written along the arrow (at start and end) so the reader can
    # quickly read the time span. A "timestamps" caption is centered below
    # the arrow.
    # Layout (top -> bottom below the primary x-axis):
    #   "Time-steps" xlabel (matplotlib default, ~y=-0.08)
    #   "timestamps" caption     (just above the arrow line, close to it)
    #   arrow line ----|>
    #   numeric timestamp ticks  (just below the arrow line)
    # Placement of the arrow line, the numeric timestamp ticks and the
    # "timestamps" caption below the primary x-axis.
    #
    # For the standalone ``plot`` render type the offsets stay expressed in
    # *axes fraction* (``transform=ax.transAxes``) — the historically
    # calibrated look on the tall ``[35, 14]`` figure.
    #
    # For the LaTeX ``includegraphics`` render type the figure is very short
    # (``figsize`` height ~1.35 in) and ``figure.constrained_layout`` is
    # enabled. The arrow / caption / ticks are drawn OUTSIDE the axes and are
    # baked into the exported canvas by the ``bbox_inches="tight"`` save
    # (see ``save_plot_special``). A purely FIXED physical band below the axes
    # therefore stayed the same absolute size while the data-axes shrank with
    # a shorter ``figsize`` height, so the plot looked disproportionately
    # squashed relative to that constant band. To make the whole below-axis
    # band (arrow + caption + ticks) shrink/grow EVENLY with the data-plot, the
    # physical offsets (in points, via
    # :class:`~matplotlib.transforms.ScaledTranslation`) are scaled
    # proportionally with the figure height relative to a reference height.
    # A per-offset physical FLOOR keeps the mandatory minimum clearance from
    # the "Time-steps" xlabel so the "timestamps" caption never overlaps it,
    # even on very short figures (hard requirement).
    # Optional user finetuning of the padding distance between the primary
    # "Time-steps" x-axis label and the "timestamps" caption/arrow band.
    # ``style.timestamp_arrow_pad`` is expressed in typographic points and is
    # ADDED (as a physical downward shift) to the whole below-axis band (arrow
    # line, centered "timestamps" caption, numeric timestamp ticks and unit
    # label), so a positive value pushes the band further from the xlabel and a
    # negative value brings it closer. Defaults to ``0.0`` -> unchanged legacy
    # placement.
    _timestamp_arrow_pad_pt = float(
        get_plot_render_type_cfg(cfg).style.get("timestamp_arrow_pad", 0.0)
    )

    # Color used for the whole below-axis timestamps band: the arrow line and
    # tip, the numeric timestamp ticks, the unit label and the "timestamps"
    # caption. ``style.timestamps_arrow_color`` accepts any matplotlib color
    # spec and defaults to ``"dimgray"``.
    _timestamps_arrow_color = get_plot_render_type_cfg(cfg).style.get(
        "timestamps_arrow_color", "dimgray"
    )

    if not is_latex_includegraphics_render_type(cfg):
        _ARROW_Y = -0.120  # axes-fraction (arrow line)
        _TICK_Y = -0.133  # axes-fraction (numeric timestamps, below the arrow)
        # Axes-fraction placement, plus an optional physical pad (points) shift.
        _pad_trans = ScaledTranslation(
            0.0, -_timestamp_arrow_pad_pt / 72.0, ax.figure.dpi_scale_trans
        )
        _arrow_trans = ax.transAxes + _pad_trans
        _tick_trans = ax.transAxes + _pad_trans
        _arrow_y = _ARROW_Y
        _tick_y = _TICK_Y
    else:
        # Reference calibration: at ``_REF_FIG_HEIGHT_IN`` the arrow line sits
        # ``_REF_ARROW_PT`` below the axes bottom and the numeric timestamp
        # ticks ``_REF_TICK_PT`` below it (clearing the LaTeX xlabel band: xtick
        # marks + tick labels @5 pt + labelpad + bold xlabel @6 pt).
        _REF_FIG_HEIGHT_IN = 1.35
        _REF_ARROW_PT = 26.0  # arrow line + centered "timestamps" caption
        _REF_TICK_PT = 31.0  # numeric timestamps, below the arrow
        # Minimum physical clearance (points) below the axes bottom so the
        # "timestamps" caption never overlaps the "Time-steps" xlabel.
        _MIN_ARROW_PT = 20.0
        _MIN_TICK_PT = 25.0

        _fig = ax.figure
        try:
            _fig_height_in = float(_fig.get_figheight())
        except Exception:
            _fig_height_in = _REF_FIG_HEIGHT_IN
        if _fig_height_in <= 0:
            _fig_height_in = _REF_FIG_HEIGHT_IN
        # Scale proportionally with the figure height (even shrink/grow), then
        # floor at the minimum clearance (no overlap on very short figures).
        _height_ratio = _fig_height_in / _REF_FIG_HEIGHT_IN
        # ``style.timestamp_arrow_pad`` (points) is applied on top of the
        # height-scaled/floored offsets so the whole band can be nudged
        # further from (positive) or closer to (negative) the "Time-steps"
        # xlabel without breaking the even shrink/grow behavior.
        _ARROW_PT = (
            max(_MIN_ARROW_PT, _REF_ARROW_PT * _height_ratio)
            + _timestamp_arrow_pad_pt
        )
        _TICK_PT = (
            max(_MIN_TICK_PT, _REF_TICK_PT * _height_ratio)
            + _timestamp_arrow_pad_pt
        )
        _arrow_trans = ax.transAxes + ScaledTranslation(
            0.0, -_ARROW_PT / 72.0, _fig.dpi_scale_trans
        )
        _tick_trans = ax.transAxes + ScaledTranslation(
            0.0, -_TICK_PT / 72.0, _fig.dpi_scale_trans
        )
        _arrow_y = 0.0
        _tick_y = 0.0

    ax.annotate(
        "",
        xy=(1.0, _arrow_y),
        xytext=(0.0, _arrow_y),
        xycoords=_arrow_trans,
        textcoords=_arrow_trans,
        arrowprops=dict(
            arrowstyle="-|>",
            color=_timestamps_arrow_color,
            # lw=1.2, # Disabled for latex
            # mutation_scale=14 # Disabled for latex
        ),
        annotation_clip=False,
    )

    # Compute start/end timestamps from the current x-axis (timesteps) and
    # the seconds-per-step conversion. When ``target_frame_rate`` is not
    # available we still display the unit label + caption but skip the
    # numeric ticks (no conversion possible).
    if target_frame_rate:
        # ``timestamp = timestep * _timestamp_scale`` (the ``ns`` unit is
        # force-promoted to seconds for legibility inside the helper).
        _timestamp_scale, unit = _resolve_timestamp_scale(target_frame_rate, unit)
        x_left, x_right = ax.get_xlim()

        t_left = x_left * _timestamp_scale
        t_right = x_right * _timestamp_scale

        def _fmt_ts(v: float) -> str:
            if abs(v) >= 100 or float(v).is_integer():
                return f"{v:.0f}"
            if abs(v) >= 1:
                return f"{v:.2f}"
            return f"{v:.3g}"

        # When the primary x-axis is in log scale (``cfg.show.x_axis_in_logscale``
        # or the matplotlib axis itself is log-scaled), the timesteps -> arrow
        # x-fraction mapping is logarithmic. To keep the secondary timestamp
        # ticks visually aligned with the primary timestep axis, distribute
        # both the tick fractional positions AND the corresponding timestamp
        # values along a geometric (log-spaced) progression instead of an
        # arithmetic one.
        _x_log = False
        try:
            _x_log = ax.get_xscale() == "log"
        except Exception:
            _x_log = False
        if not _x_log:
            try:
                _x_log = bool(cfg.show.get("x_axis_in_logscale", False))
            except Exception:
                _x_log = False
        # Log spacing requires strictly positive endpoints on both the
        # timestep axis and the resulting timestamp values.
        _log_ok = _x_log and x_left > 0 and x_right > 0 and t_left > 0 and t_right > 0

        # Build the list of (fractional_position, x_value) pairs at which
        # to draw a timestamp tick. In log mode, mirror the *primary* axis
        # major tick locations so the secondary timestamp ticks line up
        # vertically with ``10^0``, ``10^1``, ... on the primary axis (the
        # whole point of having a paired secondary axis). In linear mode,
        # fall back to 8 evenly-spaced ticks across the data span.
        _ticks: list[tuple[float, float]] = []
        if _log_ok:
            _log_xl = np.log10(x_left)
            _log_xr = np.log10(x_right)
            # Use the primary axis's major xticks; keep only those inside
            # the visible x-range (matplotlib may report ticks slightly
            # outside the current xlim).
            try:
                _primary_ticks = [
                    float(t)
                    for t in ax.get_xticks()
                    if x_left <= float(t) <= x_right and float(t) > 0
                ]
            except Exception:
                _primary_ticks = []
            # Always include the left endpoint so the arrow tail gets a
            # numeric timestamp label; the right endpoint (arrow tip) is only
            # force-labeled when ``enforce_right_endpoint`` is True.
            _endpoints = [x_left, x_right] if enforce_right_endpoint else [x_left]
            _xs = sorted(set(_endpoints + _primary_ticks))
            for _x_val in _xs:
                _frac = (np.log10(_x_val) - _log_xl) / (_log_xr - _log_xl)
                _ticks.append((_frac, _x_val * _timestamp_scale))
        else:
            _N_TICKS = 8
            for _i in range(_N_TICKS):
                _frac = _i / (_N_TICKS - 1)
                _t_val = t_left + _frac * (t_right - t_left)
                _ticks.append((_frac, _t_val))
            # Drop the right (arrow-tip) endpoint tick unless it is enforced,
            # mirroring the log branch and ``_ensure_primary_xaxis_endpoint_ticks``.
            if not enforce_right_endpoint:
                _ticks = [t for t in _ticks if t[0] < 1.0 - 1e-9]

        # Render each tick. The left endpoint uses ha=left; the right endpoint
        # uses ha=right (only when it is force-labeled, i.e.
        # ``enforce_right_endpoint`` is True) so it aligns with the arrow tip;
        # otherwise the last (interior) tick stays horizontally centered like
        # the other intermediate ticks.
        _n = len(_ticks)
        for _i, (_frac, _t_val) in enumerate(_ticks):
            if _i == 0:
                _ha = "left"
            elif _i == _n - 1 and enforce_right_endpoint:
                _ha = "right"
            else:
                _ha = "center"
            ax.text(
                _frac,
                _tick_y,
                _fmt_ts(_t_val),
                transform=_tick_trans,
                ha=_ha,
                va="top",
                clip_on=False,
                color=_timestamps_arrow_color,
            )

    # Time unit label right next to the arrow tip.
    ax.text(
        1.005,
        _arrow_y,
        f"[{unit}]",
        transform=_arrow_trans,
        ha="left",
        va="center",
        clip_on=False,
        color=_timestamps_arrow_color,
    )

    # "timestamps" caption centered on top of the arrow line. A white
    # background box — sized just to the caption's extent (tight bbox with
    # near-zero padding) — is painted between the arrow line and the text so
    # the arrow does not strike through the word and it stays legible.
    ax.text(
        0.5,
        _arrow_y,
        "Time-stamps",
        transform=_arrow_trans,
        ha="center",
        va="center",
        clip_on=False,
        color=_timestamps_arrow_color,
        zorder=6,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            boxstyle="square,pad=0.1",
        ),
        # fontstyle="italic",
    )

    # Make room at the bottom of the figure for the arrow + ticks +
    # caption. Tighter than before since the new axis is brought closer
    # to the "Time-steps" xlabel.
    fig = ax.figure
    # Skip ``subplots_adjust`` when a layout engine (constrained/tight) is
    # active — matplotlib emits a ``UserWarning`` and ignores the call in
    # that case. The layout engine already manages bottom padding.
    try:
        _layout_engine = fig.get_layout_engine()
    except Exception:
        _layout_engine = None

    if _layout_engine is None:
        try:
            fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.16))
        except Exception:
            pass


def _fmt_timestep_tick(value: float, *, is_log: bool) -> str:
    """Format a single primary timestep-axis tick label.

    On a log-scale axis every tick — the interior decades and the two forced
    edge ticks alike — is rendered in the power-of-ten ("scientific") mathtext
    style so the edge labels match the interior decade labels: exact powers of
    ten become ``$\\mathdefault{10^{n}}$`` and any other value (e.g. a warm-up
    boundary or a trajectory length that does not fall on a decade) becomes
    ``$\\mathdefault{m\\times10^{n}}$`` with a trimmed mantissa. On a linear axis
    the tick is an integer timestep count (``timesteps`` are whole numbers),
    falling back to a compact ``%g`` for the rare non-integer.

    The mathtext is wrapped in ``\\mathdefault{...}`` so it renders in the axis's
    regular (default) font rather than the math font — this mirrors matplotlib's
    own :class:`~matplotlib.ticker.LogFormatterMathtext` and keeps the forced
    edge labels in the exact same font style as the interior decade labels
    (under the ``classic`` style the bare-math font would otherwise switch to a
    Computer Modern serif and break the visual consistency).

    :param value: The tick position in data (timestep) coordinates.
    :param is_log: Whether the primary x-axis is log-scaled.
    :returns: The mathtext/plain string to use as the tick label.
    """
    if is_log and value > 0:
        exponent = int(math.floor(math.log10(value)))
        mantissa = value / (10.0**exponent)
        if abs(mantissa - 1.0) < 1e-9:
            return rf"$\mathdefault{{10^{{{exponent}}}}}$"
        mantissa_str = f"{mantissa:.2f}".rstrip("0").rstrip(".")
        return rf"$\mathdefault{{{mantissa_str}\times10^{{{exponent}}}}}$"
    if float(value).is_integer():
        return f"{int(round(value))}"
    return f"{value:g}"


def _ensure_primary_xaxis_endpoint_ticks(
    ax: Axes, *, enforce_right_endpoint: bool = False
) -> None:
    """Force the primary timestep x-axis to label both its left and right edges.

    matplotlib's default tick locators (in particular the log-scale
    ``LogLocator``) place major ticks only at "nice" positions — powers of ten
    on a log axis — so the current view's left/right x-limits are frequently
    left unlabeled. For example a log timestep axis spanning ``[26, 152]`` only
    labels ``10**2``, leaving both edges bare. This helper adds explicit major
    ticks at the current left/right x-limits (keeping the existing interior
    ticks) so the timestep axis always shows a value at both edges, mirroring
    the paired timestamps secondary axis (YouTrack RLRP-733 follow-up).

    All ticks keep the axis's native styling: on a log axis every tick (edges
    and interior) is rendered in the power-of-ten mathtext style via
    :func:`_fmt_timestep_tick`; on a linear axis the edges are plain integer
    timesteps, matching the linear interior ticks.

    Interior major ticks that sit almost on top of an edge (within 4% of the
    axis span, measured in the axis's own linear/log metric) are dropped so the
    forced edge labels do not collide with a pre-existing near-edge tick.

    The two forced edge labels are anchored inward — the left edge with
    ``ha="left"`` and the right edge with ``ha="right"`` — so a centered label
    does not spill left over the y-axis margin or right past the axes frame
    (the interior labels stay ``ha="center"``). This mirrors how the paired
    timestamps secondary axis anchors its endpoint values. All the relabeled
    ticks inherit the font size/family of the axis's original tick labels so
    the native tick styling is preserved.

    Must be called AFTER the axis x-limits are finalized, and BEFORE the
    timestamps secondary axis is attached, so the latter mirrors the same edge
    tick positions.

    Only when ``enforce_right_endpoint`` is ``True`` is the right (upper) edge
    force-labeled; the left edge and the interior ticks are always relabeled.
    When it is ``False`` the right edge is left to matplotlib's native tick
    behavior. The call site passes
    ``enforce_right_endpoint=not is_latex_includegraphics_render_type(cfg)`` so
    the right edge is forced for the standalone render type but NOT for the
    LaTeX ``includegraphics`` render type, where a forced right-edge label
    spills past the (very narrow) axes frame. The default ``False`` therefore
    labels the left edge + interior ticks only.

    :param ax: The plot axis whose primary (timestep) x-axis is relabeled.
    :param enforce_right_endpoint: When ``True`` force a label at the right
        (upper) x-axis edge as well; when ``False`` only the left edge and
        interior ticks are relabeled and the right edge keeps matplotlib's
        native behavior. Defaults to ``False``.
    """
    x_left, x_right = ax.get_xlim()
    if not (np.isfinite(x_left) and np.isfinite(x_right)):
        return
    if x_left == x_right:
        return

    lo, hi = (x_left, x_right) if x_left <= x_right else (x_right, x_left)
    is_log = ax.get_xscale() == "log"

    def _frac(v: float) -> float | None:
        """Fractional position of ``v`` across the view, in the axis metric."""
        if is_log:
            if v <= 0 or lo <= 0 or hi <= 0:
                return None
            return (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return (v - lo) / (hi - lo)

    interior: list[float] = []
    for _tick in ax.get_xticks():
        _tick = float(_tick)
        if not (lo < _tick < hi):
            continue
        _f = _frac(_tick)
        if _f is None or _f < 0.04 or _f > 0.96:
            continue
        interior.append(_tick)

    # Preserve the font size/family of the axis's original tick labels so the
    # relabeled ticks keep the native tick styling (``set_xticklabels`` would
    # otherwise create fresh ``Text`` objects at the rcParams defaults).
    _orig_labels = ax.get_xticklabels()
    _orig_fontsize = _orig_labels[0].get_fontsize() if _orig_labels else None
    _orig_fontfamily = _orig_labels[0].get_fontfamily() if _orig_labels else None

    ticks = [lo] + sorted(interior)
    if enforce_right_endpoint:
        ticks = ticks + [hi]
    ax.set_xticks(ticks)
    new_labels = ax.set_xticklabels(
        [_fmt_timestep_tick(_tick, is_log=is_log) for _tick in ticks]
    )
    # Anchor the two edge labels inward so they do not spill over the y-axis
    # margin (left) or past the axes frame (right); interior labels stay
    # centered. Keep every label un-clipped and in the original tick font.
    # The right edge is only force-labeled (and thus right-anchored) when
    # ``enforce_right_endpoint`` is False; otherwise the last tick is an
    # interior tick and keeps the centered alignment.
    _n_new = len(new_labels)
    for _idx, _label in enumerate(new_labels):
        if _idx == 0:
            _label.set_horizontalalignment("left")
        elif _idx == _n_new - 1 and enforce_right_endpoint:
            _label.set_horizontalalignment("right")
        _label.set_clip_on(False)
        if _orig_fontsize is not None:
            _label.set_fontsize(_orig_fontsize)
        if _orig_fontfamily is not None:
            _label.set_fontfamily(_orig_fontfamily)
    # ``set_xticks`` can nudge the view limits; restore the finalized extent so
    # the edge ticks stay pinned to the true left/right of the data span.
    ax.set_xlim(x_left, x_right)


def _fmt_decimal_tick(value: float) -> str:
    """Format a metric tick as a plain (non-scientific) decimal string.

    Used when scientific/power-of-ten notation is disabled on the metric
    y-axis (``plot_latex_includegraphics.style.yaxis_tick_scientific_notation:
    False``). ``numpy.format_float_positional`` renders the value in positional
    (fixed-point) notation without an exponent — e.g. ``0.001`` instead of
    ``1e-3`` or ``$\\mathdefault{10^{-3}}$`` — and ``trim="-"`` drops any
    trailing zeros and the trailing decimal point.

    :param value: The tick position in data (metric) coordinates.
    :returns: The plain decimal string to use as the tick label.
    """
    if value == 0:
        return "0"
    return np.format_float_positional(value, trim="-")


def _fmt_metric_tick(
    value: float, *, is_log: bool, scientific_notation: bool = True
) -> str:
    """Format a single primary metric y-axis tick label.

    Vertical-axis counterpart of :func:`_fmt_timestep_tick` (YouTrack
    RLRP-733 follow-up). On a log-scale axis every tick — the interior decades
    and the two forced edge ticks alike — is rendered in the power-of-ten
    ("scientific") mathtext style so the edge labels match the interior decade
    labels: exact powers of ten become ``$\\mathdefault{10^{n}}$`` and any
    other value (an edge y-limit that does not fall on a decade) becomes
    ``$\\mathdefault{m\\times10^{n}}$`` with a trimmed mantissa. The
    power-of-ten formatting is shared with :func:`_fmt_timestep_tick` so the
    two axes stay visually consistent (including the ``\\mathdefault{...}``
    wrapper that keeps the forced edge labels in the axis's regular font
    instead of the math/serif font).

    On a linear axis the metric tick is rendered with a compact ``%g`` — metric
    values are arbitrary reals (unlike the whole-number timesteps of the
    x-axis), so no integer rounding is applied.

    When ``scientific_notation`` is ``False`` (config key
    ``plot_latex_includegraphics.style.yaxis_tick_scientific_notation``) every
    tick is instead rendered as a plain decimal via :func:`_fmt_decimal_tick`,
    regardless of scale — so a log-scale axis keeps its logarithmic tick
    *positions* but shows decimal labels (``0.001``, ``0.01``, ``0.1``, …)
    rather than ``$\\mathdefault{10^{n}}$``.

    :param value: The tick position in data (metric) coordinates.
    :param is_log: Whether the primary y-axis is log-scaled.
    :param scientific_notation: When ``False`` render the tick as a plain
        decimal instead of power-of-ten/``%g`` scientific notation.
    :returns: The mathtext/plain string to use as the tick label.
    """
    if not scientific_notation:
        return _fmt_decimal_tick(value)
    if is_log and value > 0:
        return _fmt_timestep_tick(value, is_log=True)
    return f"{value:g}"


def _ensure_primary_yaxis_endpoint_ticks(
    ax: Axes, *, scientific_notation: bool = True
) -> None:
    """Force the primary metric y-axis to label both its bottom and top edges.

    Vertical-axis counterpart of :func:`_ensure_primary_xaxis_endpoint_ticks`
    (YouTrack RLRP-733 follow-up). matplotlib's default tick locators (in
    particular the log-scale ``LogLocator``) place major ticks only at "nice"
    positions — powers of ten on a log axis — so the current view's
    bottom/top y-limits are frequently left unlabeled. This helper adds
    explicit major ticks at the current bottom/top y-limits (keeping the
    existing interior ticks) so the metric axis always shows a value at both
    edges, mirroring the timestep x-axis behavior.

    All ticks keep the axis's native styling: on a log axis every tick (edges
    and interior) is rendered in the power-of-ten mathtext style via
    :func:`_fmt_metric_tick`; on a linear axis the edges use the compact ``%g``
    metric format, matching the linear interior ticks.

    Interior major ticks that sit almost on top of an edge (within 4% of the
    axis span, measured in the axis's own linear/log metric) are dropped so the
    forced edge labels do not collide with a pre-existing near-edge tick.

    The two forced edge labels are anchored inward — the bottom edge with
    ``va="bottom"`` and the top edge with ``va="top"`` — so a centered label
    does not spill below the x-axis margin or above the axes frame (the
    interior labels stay ``va="center"``). All the relabeled ticks inherit the
    font size/family of the axis's original tick labels so the native tick
    styling is preserved.

    Must be called AFTER the axis y-limits are finalized.

    :param ax: The plot axis whose primary (metric) y-axis is relabeled.
    :param scientific_notation: Forwarded to :func:`_fmt_metric_tick`; when
        ``False`` the tick labels use plain decimal notation instead of
        power-of-ten/``%g`` scientific notation (config key
        ``plot_latex_includegraphics.style.yaxis_tick_scientific_notation``).
    """
    y_bottom, y_top = ax.get_ylim()
    if not (np.isfinite(y_bottom) and np.isfinite(y_top)):
        return
    if y_bottom == y_top:
        return

    lo, hi = (y_bottom, y_top) if y_bottom <= y_top else (y_top, y_bottom)
    is_log = ax.get_yscale() == "log"

    def _frac(v: float) -> float | None:
        """Fractional position of ``v`` across the view, in the axis metric."""
        if is_log:
            if v <= 0 or lo <= 0 or hi <= 0:
                return None
            return (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return (v - lo) / (hi - lo)

    interior: list[float] = []
    for _tick in ax.get_yticks():
        _tick = float(_tick)
        if not (lo < _tick < hi):
            continue
        _f = _frac(_tick)
        if _f is None or _f < 0.04 or _f > 0.96:
            continue
        interior.append(_tick)

    # Preserve the font size/family of the axis's original tick labels so the
    # relabeled ticks keep the native tick styling (``set_yticklabels`` would
    # otherwise create fresh ``Text`` objects at the rcParams defaults).
    _orig_labels = ax.get_yticklabels()
    _orig_fontsize = _orig_labels[0].get_fontsize() if _orig_labels else None
    _orig_fontfamily = _orig_labels[0].get_fontfamily() if _orig_labels else None

    ticks = [lo] + sorted(interior) + [hi]
    ax.set_yticks(ticks)
    new_labels = ax.set_yticklabels(
        [
            _fmt_metric_tick(
                _tick, is_log=is_log, scientific_notation=scientific_notation
            )
            for _tick in ticks
        ]
    )
    # Anchor the two edge labels inward so they do not spill below the x-axis
    # margin (bottom) or above the axes frame (top); interior labels stay
    # centered. Keep every label un-clipped and in the original tick font.
    _n_new = len(new_labels)
    for _idx, _label in enumerate(new_labels):
        if _idx == 0:
            _label.set_verticalalignment("bottom")
        elif _idx == _n_new - 1:
            _label.set_verticalalignment("top")
        _label.set_clip_on(False)
        if _orig_fontsize is not None:
            _label.set_fontsize(_orig_fontsize)
        if _orig_fontfamily is not None:
            _label.set_fontfamily(_orig_fontfamily)
    # ``set_yticks`` can nudge the view limits; restore the finalized extent so
    # the edge ticks stay pinned to the true bottom/top of the data span.
    ax.set_ylim(y_bottom, y_top)


#: Label used for the ground-truth-feed warm-up reference-line marker.
_GROUND_TRUTH_WARMUP_LABEL = "Ground truth feed warm-up"


def _trim_warmup_interval(
    x: np.ndarray,
    y: np.ndarray,
    *,
    warmup_steps: int | None,
    x_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop the ground-truth-feed warm-up prefix from a plotted ``(x, y)`` curve.

    Pure helper for the ``'discard'`` treatment (YouTrack RLRP-733). Returns
    ``(x, y)`` with every sample whose x-coordinate is strictly below the
    warm-up boundary ``warmup_steps + x_offset`` removed, so the plotted curve
    begins exactly at the boundary rather than at ``t=0``. Because the arrays
    themselves start at the boundary, the primary timestep axis and the
    ``get_xlim``-driven timestamp secondary axis both begin at — and label —
    the warm-up boundary faithfully under both linear and log scale, and no
    ``x <= 0`` sample survives to poison a log-scale axis.

    Guards use ``is None`` (never Python truthiness): a resolved
    ``warmup_steps == 0`` is honored and trims nothing (nothing lies before the
    origin), while ``warmup_steps is None`` (unresolvable) returns the arrays
    unchanged.

    :param x: The curve's x-coordinates (typically ``np.arange(size)``).
    :param y: The curve's y-coordinates (same length as ``x``).
    :param warmup_steps: Auto-derived warm-up step count, or ``None``.
    :param x_offset: Added to ``warmup_steps`` to align the boundary with the
        plot's x-axis convention (``0`` for the general/final/per-category/
        per-trajectory plots; ``1`` for the drift-rate plot).
    :returns: ``(x_trimmed, y_trimmed)`` as numpy arrays.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if warmup_steps is None:
        return x, y
    boundary = warmup_steps + x_offset
    mask = x >= boundary
    return x[mask], y[mask]


def _trim_warmup_band(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    warmup_steps: int | None,
    x_offset: int = 0,
    where: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Fill-band variant of :func:`_trim_warmup_interval`.

    Trims the ``x`` / ``lower`` / ``upper`` arrays (and an optional ``where``
    mask) of an ``ax.fill_between(...)`` variation band with the same warm-up
    boundary mask, so the shaded ±variation band never extends left of the
    boundary in ``'discard'`` mode. Same ``is None`` guard semantics as
    :func:`_trim_warmup_interval` (a resolved ``0`` trims nothing; ``None``
    returns the arrays unchanged).

    :param x: The band's x-coordinates.
    :param lower: The band's lower envelope (same length as ``x``).
    :param upper: The band's upper envelope (same length as ``x``).
    :param warmup_steps: Auto-derived warm-up step count, or ``None``.
    :param x_offset: Added to ``warmup_steps`` (see :func:`_trim_warmup_interval`).
    :param where: Optional boolean mask passed to ``fill_between(where=...)``;
        trimmed with the same boundary mask when provided.
    :returns: ``(x, lower, upper, where)`` trimmed to the boundary.
    """
    x = np.asarray(x)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    if warmup_steps is None:
        return x, lower, upper, (None if where is None else np.asarray(where))
    boundary = warmup_steps + x_offset
    mask = x >= boundary
    trimmed_where = None if where is None else np.asarray(where)[mask]
    return x[mask], lower[mask], upper[mask], trimmed_where


def _trim_warmup_columns(
    x: np.ndarray,
    samples: np.ndarray,
    *,
    warmup_steps: int | None,
    x_offset: int = 0,
) -> np.ndarray:
    """Column-wise (``axis=1``) variant of :func:`_trim_warmup_interval`.

    Action ``R3`` of the RLRC ICRA2026 fig-2 / fig-3 standalone plots
    ``.junie`` plan
    (``feat_RLRP-819_two_new_standalone_plots_fig2_fig3_plan_20260910.md``):
    the RAW per-rollout population handed to the ``drift_rate_samples`` export
    is UNTRIMMED, so the EXACT same trim the plotted curve/band got must be
    re-applied on the timestep axis or the exported ``timestep`` column would
    no longer align with the reduced CSV. It DELEGATES to
    :func:`_trim_warmup_interval` so the trim rule keeps ONE implementation.

    :param x: The curve's x-coordinates (same convention as
        :func:`_trim_warmup_interval`).
    :param samples: ``(n_rollouts, len(x))`` per-rollout curves.
    :param warmup_steps: Auto-derived warm-up step count, or ``None``.
    :param x_offset: Added to ``warmup_steps`` (see
        :func:`_trim_warmup_interval`).
    :returns: ``samples`` with the warm-up COLUMNS dropped.
    """
    samples = np.asarray(samples)
    _, _kept = _trim_warmup_interval(
        x, np.arange(samples.shape[1]), warmup_steps=warmup_steps, x_offset=x_offset
    )
    return samples[:, _kept]


def _warmup_discard_left(
    warmup_mode: Any,
    warmup_steps: int | None,
    x_offset: int = 0,
) -> int | None:
    """Return the left x-limit a time-axis plot should use in ``'discard'`` mode.

    Companion to the ``'discard'`` array trimming (YouTrack RLRP-733). When
    ``'discard'`` is active with a resolved warm-up value, the plotted arrays
    are trimmed to begin at the warm-up boundary; the axis left-limit must be
    set to that same boundary so the primary timestep axis and the
    ``get_xlim``-driven timestamp secondary axis both start at — and label —
    the boundary faithfully (instead of the default ``left=0``).

    Guards use ``is None`` (never Python truthiness): a resolved
    ``warmup_steps == 0`` returns ``0`` (a valid boundary that trims nothing),
    while any non-discard mode or an unresolvable value returns ``None`` so the
    caller keeps its default ``left=0`` framing.

    :param warmup_mode: The resolved warm-up mode (``None`` / ``'discard'`` /
        ``'reference_line'``).
    :param warmup_steps: Auto-derived warm-up step count, or ``None``.
    :param x_offset: Added to ``warmup_steps`` to align with the plot's x-axis
        convention (``0`` for general/final/per-category/per-trajectory; ``1``
        for the drift-rate plot).
    :returns: The boundary left-limit (an ``int``) in ``'discard'`` mode with a
        resolved value, else ``None``.
    """
    if warmup_mode == "discard" and warmup_steps is not None:
        return warmup_steps + x_offset
    return None


def _apply_ground_truth_warmup_treatment(
    cfg: DictConfig,
    ax: Axes,
    warmup_steps: int | None,
    *,
    mode: Any,
    x_axis_in_logscale: bool = False,
    x_offset: int = 0,
) -> None:
    """Apply the configured ground-truth-feed warm-up treatment to ``ax``.

    Shared helper called by every time-axis plot (YouTrack RLRP-733). The
    treatment concerns the warm-up interval ``0 .. ground_truth_feed_warmup_steps``
    during which the model is fed ground truth (so its error is artificially
    near-zero). Depending on ``mode`` it either hides that interval or marks
    its boundary.

    Behavior by ``mode``:

    - ``None`` — feature disabled; returns immediately (no-op, no warning).
    - ``'discard'`` — the ``0 .. warm-up`` prefix is trimmed off the plotted
      curve/band arrays at each plotting site (see :func:`_trim_warmup_interval`
      / :func:`_trim_warmup_band`), and the plotting function sets the axis
      left-limit to the warm-up boundary (see :func:`_warmup_discard_left`) so
      the ``get_xlim``-driven timestamp secondary axis recomputes its start
      faithfully under both linear and log scale. This helper no longer clamps
      ``set_xlim`` in discard mode; it only draws a floating disclosure notice
      (see below).
    - ``'reference_line'`` — draws a labeled shaded marker over the warm-up
      interval via ``ax.axvspan(0, boundary, ...)`` and refreshes the legend.
      The span bounds are **data coordinates** (matplotlib's default
      transform), never axes fractions, so on a log-scale axis matplotlib
      repositions the span through the active scale transform automatically —
      its right edge always lands on the same underlying timestep while its
      pixel/axes-fraction position differs between linear and log scale
      (log-scale faithful).

    Discard disclosure notice: for ``'discard'`` with a resolved
    ``warmup_steps > 0`` (a non-zero interval actually trimmed), a floating
    axes-fraction annotation ``"Ground-truth feed warm-up interval [0..N] not
    shown"`` (``N == warmup_steps``) is drawn so a reader does not misread the
    trimmed curve as starting at ``t=0``. It is omitted when ``warmup_steps ==
    0`` (nothing trimmed) and is never drawn for ``'reference_line'``.

    Guards use ``is None`` checks (never Python truthiness), so a resolved
    ``warmup_steps == 0`` is a valid, applied value — ``'discard'`` trims
    nothing and ``'reference_line'`` draws at the origin. Only ``mode is None``
    (feature off) or ``warmup_steps is None`` (mode active but the value is
    genuinely unresolvable for every experiment) short-circuit; the latter also
    emits a console warning and leaves the plot unchanged (graceful skip).

    :param cfg: Top-level multirun plot configuration (kept for API parity /
        future styling hooks; the warm-up value is passed explicitly).
    :param ax: The matplotlib ``Axes`` to modify in place.
    :param warmup_steps: Auto-derived warm-up step count (an ``int``, possibly
        ``0``), or ``None`` when unresolvable.
    :param mode: One of ``None``, ``'discard'``, ``'reference_line'`` (already
        validated upstream by ``resolve_warmup_mode``).
    :param x_axis_in_logscale: Whether the x-axis is in log scale. Retained for
        API parity across all time-axis plots; the discard left-limit framing
        now lives in the plotting function (see :func:`_warmup_discard_left`),
        so this flag is no longer consulted here.
    :param x_offset: Added to ``warmup_steps`` to align the boundary with the
        plot's x-axis convention. ``0`` for plots that map step ``k`` to
        ``x = k`` (general/final/per-category/per-trajectory); ``1`` for the
        drift-rate plot, where step ``k`` is plotted at ``x = k + 1``.
    """
    # Feature disabled — no-op, no warning.
    if mode is None:
        return None
    if mode not in ("discard", "reference_line"):
        # Defensive: the mode is validated upstream by ``resolve_warmup_mode``.
        raise ValueError(
            f"Invalid ground-truth-feed warm-up mode: {mode!r}. "
            "Must be one of None, 'discard', 'reference_line'."
        )
    # Mode active but the warm-up value is genuinely unresolvable — graceful
    # skip with a console warning (a resolved ``0`` does NOT reach here).
    if warmup_steps is None:
        warnings.warn(
            "`show.ground_truth_feed_warmup.mode` is active "
            f"({mode!r}) but the ground-truth-feed warm-up step count could "
            "not be resolved from any recorded experiment's saved config. "
            "Leaving the plot unchanged.",
            stacklevel=2,
        )
        return None

    boundary = warmup_steps + x_offset
    if mode == "discard":
        # The plotted curve/band arrays are trimmed at each plotting site so
        # they begin at the warm-up boundary (see ``_trim_warmup_interval`` /
        # ``_trim_warmup_band``). The left x-limit framing is handled by the
        # plotting function itself (it sets the axis ``left`` to the warm-up
        # boundary in discard mode via :func:`_warmup_discard_left`), so the
        # ``get_xlim``-driven timestamp secondary axis recomputes its start
        # faithfully under both linear and log scale. This helper therefore no
        # longer clamps ``set_xlim`` in discard mode; it only draws the
        # disclosure notice below.
        #
        # Discard disclosure notice (Step 6): only when a non-zero warm-up
        # interval was actually trimmed. Drawn in axes-fraction coordinates so
        # it is scale-agnostic (unaffected by linear vs log x) and needs no
        # per-plot title knowledge.
        if warmup_steps > 0 and get_plot_render_type_cfg(cfg).print_meta_info:
            ax.text(
                0.01,
                # 0.98,
                0.01,
                f"Ground-truth feed warm-up interval [0..{warmup_steps}[ " "not shown",
                transform=ax.transAxes,
                # va="top",
                va="bottom",
                ha="left",
                fontsize="small",
                color="gray",
            )
    elif mode == "reference_line":
        # ax.axvline(
        #     boundary,
        #     # linestyle="--",
        #     linestyle="-",
        #     linewidth=2,
        #     # color="lightgray",
        #     color="red",
        #     zorder=0,  # Put in back
        #     label=_GROUND_TRUTH_WARMUP_LABEL,
        # )
        ax.axvspan(
            0,
            boundary,
            linestyle="-",
            linewidth=3,
            color="gray",
            # color="red",
            alpha=0.16,
            zorder=0,  # Put in back
            label=_GROUND_TRUTH_WARMUP_LABEL,
        )
        # Refresh the legend so the new marker entry is shown.
        ax.legend(loc="best")
    return None


def generate_arbitrary_len_general_plot(
    cfg: DictConfig,
    groups: dict[Any, Any],
    category_breakdown_data: dict[str, dict[str, dict]],
    general_length_spec: Any,
    do_average_across_trajectories: bool,
    do_category_breakdown: bool,
    title: str,
    exp_dir_relative_path: Any,
    fill_between_alpha: float,
    plot_linewidth: float,
    metric_acro_label: str,
    metric_label: str,
    headless: bool,
    *,
    trials_label: str = "trials",
    y_axis_in_logscale: bool = False,
    warmup_mode: Any = None,
    warmup_steps: int | None = None,
) -> None:
    """Render the General plot (Phase 5 — ``show.general_length``).

    Aggregates each group's metric over the trajectory subset selected by
    ``show.general_length``, augments the title with the resolved
    ``N=<k>, length ≥ <L>`` annotation, then renders the combined figure
    saved as ``test_time_models_comparaison.<ext>``.

    Mirrors :func:`generate_per_category_plots` and
    :func:`generate_per_trajectory_plots`: pure plotting step that consumes
    the dicts produced upstream (``groups``, ``category_breakdown_data``)
    plus the resolved styling/label strings from :func:`execute`.

    :param cfg: Top-level multirun plot configuration.
    :param groups: Per-group metric dict from :func:`compute_per_group_metric`.
        Mutated in place: ``grp_avg_mae`` / ``grp_std_mae`` are replaced
        with the subset-restricted aggregation when applicable, and the
        ``n_trials_subset`` / ``n_trajectories_subset`` legend counters are
        attached.
    :param category_breakdown_data: Per-category aggregates used as a
        back-compat fallback when no per-trajectory data is available.
    :param general_length_spec: Raw value of ``cfg.show.general_length``
        (``None``, ``"min"``, ``"max"``, or an ``int``). ``None``
        short-circuits — the General plot is disabled.
    :param do_average_across_trajectories: Selects pooled-rollout vs
        inter-trajectory-mean semantics for the subset aggregation.
    :param do_category_breakdown: Whether category breakdown is active
        (controls the legacy xlim fallback).
    :param title: Base title string (already enriched with metric/variation
        type/compounded/target tokens by :func:`execute`).
    :param exp_dir_relative_path: Where to save the figure.
    :param fill_between_alpha: Alpha for the variation band.
    :param plot_linewidth: Line width for group curves.
    :param metric_acro_label: Short metric acronym (e.g. ``"MAE"``).
    :param metric_label: Full metric name (e.g. ``"Mean Absolute Error"``).
    :param headless: Headless backend flag.
    :param trials_label: User-facing word substituted for ``"trials"`` in
        title/legend (``label_override.trials``); defaults to ``"trials"``.
    :param y_axis_in_logscale:
    :param warmup_mode: Ground-truth-feed warm-up treatment mode
        (``None`` / ``'discard'`` / ``'reference_line'``); see
        :func:`_apply_ground_truth_warmup_treatment`. Defaults to ``None``
        (no treatment).
    :param warmup_steps: Auto-derived warm-up step count applied by the
        treatment; ``None`` when unresolvable.
    """

    # Short-circuit: ``general_length=None`` disables the general plot entirely
    # (plan §0 / §4.9).
    if general_length_spec is None:
        plt.close("all")  # (Priority) ToDo: validate
        return None

    # Build per-trajectory length map from the raw metrics. Each trajectory's
    # authoritative length is the max ``mae`` timestep count observed across
    # all groups/trials that reported it (plan Q2).
    _trajectory_length_map = _build_trajectory_length_map(groups)

    # Resolve the spec. ``'min'``/``'max'`` consult the length pool; ``int``
    # specs skip it. Use the length-map values when available; otherwise fall
    # back to per-group ``grp_avg_mae`` shapes so degenerate test fixtures
    # (no ``_original_metrics``) still behave. When both sources are empty
    # (truly no groups, e.g. edge1 empty-groups case), ``general_length_int``
    # is left ``None`` — the plotting loop exits via the ``not fig_init``
    # guard without crashing.
    _length_pool: list[int] = (
        list(_trajectory_length_map.values())
        if _trajectory_length_map
        else [
            int(g["grp_avg_mae"].shape[0])
            for g in groups.values()
            if "grp_avg_mae" in g and hasattr(g["grp_avg_mae"], "shape")
        ]
    )
    if _length_pool or isinstance(general_length_spec, int):
        general_length_int = resolve_general_length(general_length_spec, _length_pool)
    else:
        general_length_int = None

    # Select the trajectory-name subset and restrict per-group aggregation to
    # it. ``show.general_length`` is honored regardless of whether
    # ``show.breakdown`` is ``'category'`` — the two axes are orthogonal.
    if _trajectory_length_map and general_length_int is not None:
        selected_tnames_list = select_trajectories_by_length(
            _trajectory_length_map, general_length_int
        )
    else:
        selected_tnames_list = []
    selected_tnames = set(selected_tnames_list)

    # Note: empty subset (integer cutoff exceeds every trajectory length) is
    # allowed here — we fall through to the pre-aggregated ``grp_avg_mae``
    # from ``compute_per_group_metric`` so the general plot still renders
    # with the requested xlim. This preserves D6 back-compat semantics for
    # callers that pass an explicit integer cutoff without necessarily
    # intending subset restriction.

    # If we have per-trajectory data (and are NOT in category-breakdown mode),
    # replace each group's ``grp_avg_mae`` / ``grp_std_mae`` with the
    # subset-restricted aggregation truncated to the cutoff. Groups with no
    # rollout matching the subset are marked so they are skipped in the
    # plotting loop.
    if selected_tnames and general_length_int is not None:
        use_cumulative = bool(cfg.show.cumulative_metric)
        for each_group_key, each_group in groups.items():
            result = _aggregate_group_over_selected_trajectories(
                each_group,
                selected_tnames,
                general_length_int,
                use_cumulative,
                do_average_across_trajectories,
            )
            if result is None:
                # No matching rollouts → drop the key so the plotting loop's
                # ``"grp_avg_mae" not in each_group`` guard skips the group.
                each_group.pop("grp_avg_mae", None)
                each_group.pop("grp_std_mae", None)
                continue
            avg, std, _ = result
            each_group["grp_avg_mae"] = avg
            each_group["grp_std_mae"] = std
            # Restrict legend counters to the General-plot subset so the
            # legend matches the title's ``N=<k> trajectories`` annotation.
            _subset_trial_keys = set()
            _subset_traj_names = set()
            for ogm in each_group.get("_original_metrics", []) or []:
                _tn = getattr(ogm, "_trajectory_name", None) or "unknown"
                if _tn not in selected_tnames:
                    continue
                _subset_traj_names.add(_tn)
                _tk = getattr(ogm, "_trial_key", None)
                if _tk is not None:
                    _subset_trial_keys.add(_tk)
            each_group["n_trajectories_subset"] = len(_subset_traj_names)
            each_group["n_trials_subset"] = len(_subset_trial_keys)
    elif do_category_breakdown and category_breakdown_data:
        # Back-compat path when per-trajectory data is absent: use the longest
        # available category curve for the general plot.
        for each_group_key in groups:
            cat_dict = category_breakdown_data.get(each_group_key, {})
            if not cat_dict:
                continue
            best_cat = None
            best_len = 0
            for cat in ["L", "M", "S"]:
                if cat in cat_dict:
                    clen = cat_dict[cat]["grp_avg_mae"].shape[0]
                    if clen > best_len:
                        best_len = clen
                        best_cat = cat
            if best_cat is not None:
                groups[each_group_key]["grp_avg_mae"] = cat_dict[best_cat][
                    "grp_avg_mae"
                ]
                groups[each_group_key]["grp_std_mae"] = cat_dict[best_cat][
                    "grp_std_mae"
                ]

    # Augment the title with the ``N=<k>, length ≥ <L>`` annotation when a
    # real subset is in play. ``N`` reports the actual aggregation unit count:
    # number of trajectories when ``average_across_trajectories=True`` (one
    # per-trajectory mean curve per unit), number of pooled rollouts otherwise.
    if selected_tnames and general_length_int is not None:
        # Singularize "trajectory" when only one trajectory passes the cutoff so
        # the title is grammatically consistent with the legend (which already
        # collapses to ``(<n> trials)`` in the single-trajectory case).
        _n_traj = len(selected_tnames)
        _traj_word = "trajectory" if _n_traj == 1 else "trajectories"
        if do_average_across_trajectories:
            if _n_traj == 1:
                _n_label = f"N=1 trajectory (mean across {trials_label})"
            else:
                _n_label = f"N={_n_traj} trajectories (mean across {trials_label})"
        else:
            _n_rollouts = sum(
                1
                for g in groups.values()
                for ogm in (g.get("_original_metrics", []) or [])
                if (getattr(ogm, "_trajectory_name", None) or "unknown")
                in selected_tnames
            )
            _n_label = f"N={_n_rollouts} rollouts over {_n_traj} {_traj_word}"
        title = (
            f"{title}\n"
            f"General plot: {_n_label} "
            f"(length ≥ {general_length_int})."
        )

    with warnings.catch_warnings():
        manage_matplotlib_warnings()
        manage_matplotlib_backend(get_plot_render_type_cfg(cfg).show_plot, headless)

        fig_init = False
        color_idx = 0
        for each_group_key, each_group in groups.items():
            # Skip groups that yielded no curve (e.g. all trajectories filtered
            # out by InD/OoD subdir filter — see BUG-P3-01 guard in
            # compute_per_group_metric).
            if "grp_avg_mae" not in each_group:
                continue
            grp_avg_mae = each_group["grp_avg_mae"]

            grp_std_mae = each_group["grp_std_mae"]
            grp_avg_uncertainty_std = each_group["gpr_avg_uncertainty_std"]
            grp_avg_uncertainty_std_epi = each_group["gpr_avg_uncertainty_std_epi"]

            grp_avg_training_wall_clock_time = each_group[
                "grp_avg_training_wall_clock_time"
            ]

            if cfg.show.variation_type == "group_mae_std":
                gpr_variation_type = grp_std_mae
            elif cfg.show.variation_type == "group_avg_uncertainty":
                gpr_variation_type = grp_avg_uncertainty_std
            elif cfg.show.variation_type == "group_avg_epi":
                gpr_variation_type = grp_avg_uncertainty_std_epi
            elif cfg.show.variation_type == "group_avg_ale":
                gpr_variation_type = np.absolute(
                    grp_avg_uncertainty_std - grp_avg_uncertainty_std_epi
                )
            else:
                raise ValueError(f"Unsupported {cfg.show.variation_type=}")

            grp_size = each_group["grp_size"]
            line_style = each_group["line_style"]
            if line_style is None:
                line_style = "-"

            if cfg.show.grp_size:
                # ``grp_size = len(metrics)`` post-aggregation counts pooled
                # ``(trial × trajectory)`` rollouts when averaging is OFF and
                # collapses to 1 when averaging is ON — neither is "trials".
                # Report ``n_trials`` and ``n_trajectories`` explicitly so the
                # legend matches the aggregation actually performed.
                # Prefer the subset counters when the General plot restricted
                # aggregation to ``selected_tnames`` so the legend matches the
                # title's ``N=<k> trajectories`` annotation.
                _n_tr = each_group.get("n_trials_subset", each_group.get("n_trials", 0))
                _n_tj = each_group.get(
                    "n_trajectories_subset", each_group.get("n_trajectories", 0)
                )
                label = _format_grp_count_label(
                    each_group["grp_name"],
                    n_trials=_n_tr,
                    n_trajectories=_n_tj,
                    n_total_rollouts=None,
                    trials_label=trials_label,
                    do_average_across_trajectories=do_average_across_trajectories,
                    show_acronym=cfg.show.get("acronym", True),
                )
            else:
                label = each_group["grp_name"]

            # Ground-truth-feed warm-up treatment (YouTrack RLRP-733):
            # in ``'discard'`` mode drop the ``0 .. warm-up`` prefix from the
            # plotted curve and its variation band so they begin at the
            # boundary (step ``k`` maps to ``x = k`` here, ``x_offset=0``).
            _x_full = np.arange(grp_avg_mae.size)
            _lower = grp_avg_mae - cfg.show.variation_scale * gpr_variation_type
            _upper = grp_avg_mae + cfg.show.variation_scale * gpr_variation_type
            if warmup_mode == "discard":
                _x_curve, _y_curve = _trim_warmup_interval(
                    _x_full, grp_avg_mae, warmup_steps=warmup_steps
                )
                _x_band, _lower, _upper, _ = _trim_warmup_band(
                    _x_full, _lower, _upper, warmup_steps=warmup_steps
                )
            else:
                _x_curve, _y_curve = _x_full, grp_avg_mae
                _x_band = _x_full

            if not fig_init:
                fig, ax = arbitrary_dimension_array_plot(
                    grp_avg_mae,
                    plot_label=label,
                    y_label=(f"{metric_label} ({metric_acro_label})",),
                    title=title,
                    figsize=get_plot_figsize(
                        cfg, "generate_arbitrary_len_general_plot"
                    ),
                    figdpi=get_plot_render_type_cfg(cfg).figdpi,
                    plot_style={
                        "color": COLORS[0],
                        "linestyle": line_style,
                        "linewidth": plot_linewidth,
                        "alpha": 1.0,
                    },
                )
                fig_init = True
                if warmup_mode == "discard" and warmup_steps is not None:
                    # ``arbitrary_dimension_array_plot`` draws the first curve
                    # at an implicit ``x = arange(size)``; re-seat it on the
                    # trimmed, boundary-based x so the discarded prefix is
                    # dropped from the first curve too.
                    ax.get_lines()[-1].set_data(_x_curve, _y_curve)
            else:
                plot_style = {
                    "color": COLORS[color_idx],
                    "linestyle": line_style,
                    "linewidth": plot_linewidth,
                    "alpha": 1.0,
                }
                ax.plot(
                    _x_curve,
                    _y_curve,
                    **plot_style,
                    label=label,
                )

            ax.fill_between(
                _x_band,
                _lower,
                _upper,
                color=COLORS[color_idx],
                alpha=fill_between_alpha,
            )

            if color_idx < len(COLORS) - 1:
                color_idx += 1
            else:
                color_idx = 0

        # Guard: if no group produced a curve (e.g. all groups filtered out by
        # cfg.show.grp_names), skip axis/legend/save to avoid UnboundLocalError
        # on `ax` (closes BUG-P1-01).
        if not fig_init:
            plt.close("all")  # (CRITICAL) ToDo: validate
            return None

        if y_axis_in_logscale:
            ax.set_yscale("log")
            if cfg.show.get("top_ylim", None) is not None:
                ax.set_ylim(top=cfg.show.top_ylim)
        elif cfg.show.get("top_ylim", None) is not None:
            ax.set_ylim(bottom=0, top=cfg.show.top_ylim)
        else:
            ax.set_ylim(bottom=0)

        # ``show.general_length`` always wins when it resolved to an integer,
        # regardless of ``show.breakdown`` — the two axes are orthogonal.
        # Fall back to the legacy category-max xlim only when no explicit
        # ``general_length`` was provided but category breakdown is active.
        # In ``'discard'`` mode the left bound is the warm-up boundary (the
        # arrays are already trimmed there) so the timestep/timestamp axes
        # start faithfully at the boundary (YouTrack RLRP-733).
        _discard_left = _warmup_discard_left(warmup_mode, warmup_steps)
        _left = _discard_left if _discard_left is not None else 0
        if general_length_int is not None:
            ax.set_xlim(left=_left, right=general_length_int)
        elif do_category_breakdown and category_breakdown_data:
            _cat_max_lengths = _get_category_max_length(cfg)
            _general_xlim = max(_cat_max_lengths.values(), default=None)
            if _general_xlim is not None:
                ax.set_xlim(left=_left, right=_general_xlim)
            else:
                ax.set_xlim(left=_left)
        else:
            ax.set_xlim(left=_left)

        # Ground-truth-feed warm-up treatment (YouTrack RLRP-733). No-op when
        # ``warmup_mode`` is ``None``. Step ``k`` maps to ``x = k`` here
        # (``x_offset=0``).
        if cfg.show.get("compounded_predictions_score", False):
            _apply_ground_truth_warmup_treatment(
                cfg, ax, warmup_steps, mode=warmup_mode
            )

        ax.legend(
            loc="best",
        )
        ax.grid(True)
        ax.set_xlabel("Time-steps", **AXIS_LABEL_STYLE)
        _ensure_primary_xaxis_endpoint_ticks(ax)
        _ensure_primary_yaxis_endpoint_ticks(ax)
        _attach_timestamp_secondary_xaxis(cfg, ax)
        fig.tight_layout(pad=0.99)

        display_experiment_name(cfg, fig)

        show_and_save_plot_helper(
            fig,
            exp_dir_relative_path,
            "test_time_models_comparaison",
            headless,
            get_plot_render_type_cfg(cfg).show_plot,
            get_plot_render_type_cfg(cfg).save_dpi,
        )

    # Close the local figure on the way out so repeated invocations do not
    # leak figures across runs (D12 oracle). ``execute`` also runs a final
    # ``plt.close("all")`` after this call.
    plt.close("all")
    return None


def generate_final_general_plot(
    cfg: DictConfig,
    groups: dict[Any, Any],
    do_average_across_trajectories: bool,
    title: str,
    exp_dir_relative_path: Any,
    fill_between_alpha: float,
    plot_linewidth: float,
    metric_acro_label: str,
    metric_label: str,
    headless: bool,
    *,
    trials_label: str = "trials",
    y_axis_in_logscale: bool = False,
    sem_alpha: float = 0.35,
    skip_nan_contaminated_mae: bool = True,
    skip_nan_contaminated_std_uncertainty: bool = True,
    group_stack_cumsum="post",
    final_length_spec: Any = None,
    warmup_mode: Any = None,
    warmup_steps: int | None = None,
) -> None:
    """
    Generates a final general plot aggregating and visualizing metrics over trajectories
    and trials with customization options for the plot's presentation.

    :param cfg: Configuration object containing the plotting parameters and settings.
    :param groups: A dictionary containing groups of trajectory and trial metrics.
    :param do_average_across_trajectories: If True, averages the metrics across trajectories
        before plotting.
    :param title: The title of the plot.
    :param exp_dir_relative_path: Relative path for export directory, used for saving the plot outputs.
    :param fill_between_alpha: Alpha transparency value for the fill between standard deviation
        or SEM boundaries.
    :param plot_linewidth: Line width of the plot.
    :param metric_acro_label: Abbreviated metric label for display.
    :param metric_label: Full metric label to be displayed on the plot.
    :param headless: If True, runs without displaying the graphical output. Generally used
        in headless environments.
    :param trials_label: Label used to represent trials in the plot, default is "trials".
    :param y_axis_in_logscale: If True, sets the y-axis in logarithmic scale.
    :param sem_alpha: Alpha transparency value for the fill between SEM boundaries.
    :param skip_nan_contaminated_mae: If True, skips processing metrics contaminated by NaN
        values for Mean Absolute Error computations.
    :param skip_nan_contaminated_std_uncertainty: If True, skips processing metrics
        contaminated by NaN values for standard uncertainty computations.
    :param group_stack_cumsum: group stack cumulative sum reduction ordering, either 'pre' or 'post'.
    :param final_length_spec: Optional cutoff for the x-axis trajectory length (option: ``None``, ``'min'``, ``'max'`` or an int).
        Analogue of ``show.general_length`` used by
        :func:`generate_arbitrary_len_general_plot`. Accepts ``None`` (no
        cutoff — full ``max_len`` from the trajectory length pool is used),
        ``'min'`` / ``'max'`` (resolved against the per-trajectory length
        pool), or a positive ``int`` (used as-is). When the resolved value
        is smaller than the natural ``max_len``, the plot's x-axis is
        truncated to that length (right xlim and per-group aggregation
        horizon).
    :param warmup_mode: Ground-truth-feed warm-up treatment mode
        (``None`` / ``'discard'`` / ``'reference_line'``); see
        :func:`_apply_ground_truth_warmup_treatment`. Defaults to ``None``
        (no treatment).
    :param warmup_steps: Auto-derived warm-up step count applied by the
        treatment; ``None`` when unresolvable.
    :return: None. The function saves the plot to the specified output directory or displays
        it based on the configuration.
    """
    length_map = _build_trajectory_length_map(groups)
    if not length_map:
        plt.close("all")
        return None

    # Resolve ``final_length_spec`` (analogue of ``show.general_length``).
    # ``None`` short-circuits: the full ``max_len`` from the trajectory
    # length pool is preserved (back-compat default). ``'min'`` / ``'max'``
    # consult the length pool; a positive ``int`` is used as-is. The
    # resolved value is then clamped against the natural ``max_len`` so a
    # spec larger than the longest trajectory never extends the x-axis
    # beyond available data.
    _natural_max_len = max(length_map.values())
    if final_length_spec is not None:
        _resolved_final_length = resolve_general_length(
            final_length_spec, list(length_map.values())
        )
        max_len = min(int(_resolved_final_length), _natural_max_len)
    else:
        max_len = _natural_max_len
    n_trajectories_total = len(length_map)

    # Title scheme mirrors ``generate_arbitrary_len_general_plot`` (see L680-708): report
    # the actual ``trials × trajectories = rollouts`` aggregation counts so
    # the Final plot annotation is grammatically and semantically consistent
    # with the General/per-category/per-trajectory titles. ``n_rollouts_total``
    # pools every ``(trial, trajectory)`` rollout across all selected groups;
    # ``n_trajectories_total`` is the number of distinct trajectories that
    # contributed at least one rollout.
    _selected_grp_names = set(cfg.show.grp_names) if cfg.show.grp_names else None
    _selected_groups = [
        g
        for g in groups.values()
        if _selected_grp_names is None or g.get("grp_name") in _selected_grp_names
    ]
    n_rollouts_total = sum(
        len(g.get("_original_metrics", []) or []) for g in _selected_groups
    )
    _traj_word = "trajectory" if n_trajectories_total == 1 else "trajectories"

    with warnings.catch_warnings():
        manage_matplotlib_warnings()
        manage_matplotlib_backend(get_plot_render_type_cfg(cfg).show_plot, headless)

        plt.rcParams.update(get_plot_render_type_cfg(cfg).rcParams_update)

        fig: Figure = plt.figure(
            figsize=get_plot_figsize(cfg, "generate_final_general_plot"),
            dpi=get_plot_render_type_cfg(cfg).figdpi,
        )
        ax: Axes = fig.add_subplot(111)

        fig_init = False
        color_idx = 0
        all_n_units = []

        # We collect all averages and SEMs to pick a tighter top_ylim if needed.
        all_upper_bounds = []

        for grp_key, group in groups.items():
            # ``groups`` is keyed by internal IDs ("group_0", "group_1", ...);
            # the human-readable name lives in ``group["grp_name"]`` and matches
            # the entries listed in ``cfg.show.grp_names`` (same convention as
            # ``regenerate_rollouts``). Comparing against ``grp_key`` would
            # always reject every group when ``grp_names`` is set, suppressing
            # the entire plot — see RLRP final-general-plot bugfix.
            _grp_display_name = group.get("grp_name", grp_key)
            if cfg.show.grp_names and _grp_display_name not in cfg.show.grp_names:
                continue

            agg = _aggregate_group_over_all_trajectories(
                group,
                max_len,
                use_cumulative=cfg.show.cumulative_metric,
                average_across_trajectories=do_average_across_trajectories,
                skip_nan_contaminated_mae=skip_nan_contaminated_mae,
                skip_nan_contaminated_std_uncertainty=skip_nan_contaminated_std_uncertainty,
                error_type=cfg.show.get("error_type", "elementwise_mae"),
                group_stack_cumsum=group_stack_cumsum,
            )
            if agg is None:
                continue

            avg, std, n_valid_per_t, n_units = agg
            all_n_units.append(n_units)
            fig_init = True

            # (CRITICAL) ToDo: assess if it should be raised as a warning >> next bloc ↓↓ (RLRP-654)
            with np.errstate(divide="ignore", invalid="ignore"):
                sem = std / np.sqrt(np.maximum(n_valid_per_t, 1))

            # Std / SEM band masking — *final general plot* policy:
            # compute the variance over *every* rollout still alive at each
            # timestep, so the bands extend up to the longest trajectory in
            # the pool. ``n_valid_per_t[t]`` naturally decreases past the
            # shortest trajectory's length, which (correctly) widens the SEM
            # band in the tail. The only lower bound is N(t) >= 2 (SEM is
            # undefined for a single sample).
            max_n_valid = int(np.max(n_valid_per_t))
            valid_mask = n_valid_per_t >= 2

            x = np.arange(max_len)
            color = COLORS[color_idx % len(COLORS)]

            # Honor the per-group ``line_style`` setting like the other
            # ``generate_*`` functions (see L746-748, L1136-1138, L2136). When
            # the YAML entry omits it or sets it to null, fall back to a solid
            # line.
            line_style = group.get("line_style") or "-"

            # Formatted legend label — same scheme as ``generate_arbitrary_len_general_plot``
            # (see L759-770): report explicit ``n_trials × n_trajectories``
            # counts pulled from the group dict so the legend collapses to the
            # canonical ``(<n_tr> trials × <n_tj> trajectories = <r> rollouts)``
            # form instead of the ``n_trials=None`` fallback.
            _n_trial = group.get("n_trials_subset", group.get("n_trials", 0))
            # _n_trj = group.get(
            #     "n_trajectories_subset",
            #     group.get(
            #         "n_trajectories",
            #         len(
            #             {
            #                 getattr(ogm, "_trajectory_name", "unknown")
            #                 for ogm in group.get("_original_metrics", [])
            #             }
            #         ),
            #     ),
            # )
            _n_trj = max_n_valid / _n_trial

            _gpr_count_label = _format_grp_count_label(
                str(_grp_display_name),
                n_trials=_n_trial,
                n_trajectories=_n_trj,
                n_total_rollouts=(
                    n_units if not do_average_across_trajectories else None
                ),
                trials_label=trials_label,
                do_average_across_trajectories=do_average_across_trajectories,
                show_acronym=cfg.show.get("acronym", True),
            )

            if not is_latex_includegraphics_render_type(cfg):
                label = _gpr_count_label
            else:
                # (Priority) ToDo: save _gpr_count_label to file for reference
                label = group.get("gpr_short_name", grp_key)

            # Ground-truth-feed warm-up treatment (YouTrack RLRP-733): in
            # ``'discard'`` mode drop the ``0 .. warm-up`` prefix from the
            # plotted curve and both variation bands so they begin at the
            # boundary (step ``k`` maps to ``x = k`` here, ``x_offset=0``).
            if warmup_mode == "discard":
                x_p, avg_p = _trim_warmup_interval(x, avg, warmup_steps=warmup_steps)
                _, std_p = _trim_warmup_interval(x, std, warmup_steps=warmup_steps)
                _, sem_p = _trim_warmup_interval(x, sem, warmup_steps=warmup_steps)
                _, mask_p = _trim_warmup_interval(
                    x, valid_mask, warmup_steps=warmup_steps
                )
            else:
                x_p, avg_p, std_p, sem_p, mask_p = x, avg, std, sem, valid_mask

            ax.plot(
                x_p,
                avg_p,
                color=color,
                linewidth=plot_linewidth,
                linestyle=line_style,
                label=label,
            )

            # Light band: ±1·std
            ax.fill_between(
                x_p,
                avg_p - std_p,
                avg_p + std_p,
                where=mask_p,
                color=color,
                alpha=fill_between_alpha,
                linewidth=0,
            )

            # Dark band: ±2·SEM
            ax.fill_between(
                x_p,
                avg_p - 2 * sem_p,
                avg_p + 2 * sem_p,
                where=mask_p,
                color=color,
                alpha=sem_alpha,
                linewidth=0,
            )

            # Collect for auto-ylim
            if valid_mask.any():
                all_upper_bounds.append(np.nanmax((avg + 2 * sem)[valid_mask]))
                all_upper_bounds.append(np.nanmax((avg + std)[valid_mask]))

            color_idx += 1

        if not fig_init:
            plt.close("all")
            return None

        if y_axis_in_logscale:
            ax.set_yscale("log")
            if cfg.show.get("top_ylim", None) is not None:
                ax.set_ylim(top=cfg.show.top_ylim)
        elif cfg.show.get("top_ylim", None) is not None:
            ax.set_ylim(bottom=0, top=cfg.show.top_ylim)
        else:
            # Operator decision Q1: pick an automatically tighter top_ylim when null
            if all_upper_bounds:
                tight_top = float(np.nanmax(all_upper_bounds)) * 1.1
                ax.set_ylim(bottom=0, top=np.nan_to_num(tight_top, posinf=1e35))
            else:
                ax.set_ylim(bottom=0)

        # In ``'discard'`` mode the left bound is the warm-up boundary (arrays
        # already trimmed there) so the timestep/timestamp axes start faithfully
        # at the boundary (YouTrack RLRP-733).
        _discard_left = _warmup_discard_left(warmup_mode, warmup_steps)
        ax.set_xlim(
            left=_discard_left if _discard_left is not None else 0,
            right=max_len,
        )
        # Ground-truth-feed warm-up treatment (YouTrack RLRP-733). No-op when
        # ``warmup_mode`` is ``None``. Step ``k`` maps to ``x = k`` here
        # (``x_offset=0``).
        if cfg.show.get("compounded_predictions_score", False):
            _apply_ground_truth_warmup_treatment(
                cfg, ax, warmup_steps, mode=warmup_mode
            )

        ax.grid(True)

        if get_plot_render_type_cfg(cfg).print_title:
            # Annotate the trajectory-length scope: "(all lengths)" by default,
            # or "(cut to <max_len> timesteps)" when a ``final_length_spec``
            # truncates the x-axis (analogue of the
            # ``generate_arbitrary_len_general_plot`` ``N=<k>, length ≥ <L>``
            # annotation).
            if final_length_spec is not None and max_len < _natural_max_len:
                _length_scope = f"cut to {max_len} timesteps"
            else:
                _length_scope = "all lengths"
            ax.set_title(
                f"{title}\n"
                f"Final plot: N={n_rollouts_total} rollouts over "
                f"{n_trajectories_total} {_traj_word} ({_length_scope}).\n"
                f"Light: \u00b11\u00b7std, Dark: \u00b12\u00b7SEM",
            )

        if get_plot_render_type_cfg(cfg).print_legend:
            ax.legend(loc="best")

        if get_plot_render_type_cfg(cfg).print_axis_label:
            ax.set_xlabel("Time-steps")

        _ensure_primary_xaxis_endpoint_ticks(ax)
        _ensure_primary_yaxis_endpoint_ticks(ax)
        _attach_timestamp_secondary_xaxis(cfg, ax)

        if get_plot_render_type_cfg(cfg).print_axis_label:
            ax.set_ylabel(f"{metric_acro_label} ({metric_label})")

        if get_plot_render_type_cfg(cfg).print_meta_info:
            display_experiment_name(cfg, fig)

        show_and_save_plot_helper(
            fig,
            exp_dir_relative_path,
            "test_time_models_comparaison_final_general",
            headless,
            get_plot_render_type_cfg(cfg).show_plot,
            get_plot_render_type_cfg(cfg).save_dpi,
            save=not is_latex_includegraphics_render_type(cfg),
        )

        if is_latex_includegraphics_render_type(cfg):
            # project_root = get_hydra_original_cwd(cfg)
            project_root = fetch_r2s2r_project_root_path(cfg, lvl_up=1)
            if cfg.show.compounded_predictions_score:
                rollout_type_name = "CP"
            else:
                rollout_type_name = "GT"
            if cfg.show.target_is_ood:
                rollout_type_name = f"OOD_{rollout_type_name}"
            else:
                rollout_type_name = f"IND_{rollout_type_name}"

            show_and_save_plot_helper(
                fig,
                os.path.join(project_root, cfg.latex_includegraphics.save_path),
                f"{cfg.latex_includegraphics.file_name}_{rollout_type_name}",
                headless,
                False,
                get_plot_render_type_cfg(cfg).latex_save_dpi,
                save=True,
            )

    plt.close("all")
    return None


def generate_per_category_plots(
    cfg: DictConfig,
    category_breakdown_data: dict[str, dict[str, dict]],
    compounded_pred_label: str,
    do_average_across_trajectories: bool,
    do_category_breakdown: bool,
    do_summary_table: bool,
    exp_dir_relative_path: Any,
    fill_between_alpha: float,
    plot_linewidth: float,
    metric_acro_label: str,
    metric_label: str,
    target_type_label: str,
    headless: bool,
    trials_label: str = "trials",
    y_axis_in_logscale: bool = False,
    warmup_mode: Any = None,
    warmup_steps: int | None = None,
):
    """
    Generates category-specific plots and summary tables for given metrics.

    This function creates visualizations and summary tables based on the provided
    category breakdown data. It supports creating separate plots for specific
    categories (e.g., "S", "M", "L"), computes variations between trajectories or
    trials, and optionally saves or displays the generated plots. The category
    summary table provides a tabular representation of terminal metrics across
    categories for different groups.

    :param cfg: Configuration object specifying pipeline and plotting settings.
    :param category_breakdown_data: Nested dictionary with breakdown data for categories.
        The hierarchy is group -> category -> metric-related details.
    :param compounded_pred_label: Descriptor for the compounded prediction model used (e.g., type of model aggregation).
    :param do_average_across_trajectories: Flag to compute averages across trajectories.
    :param do_category_breakdown: Flag to determine whether category-specific plots should be generated.
    :param do_summary_table: Flag to generate the category breakdown summary table.
    :param exp_dir_relative_path: Relative path within the experiment directory where plots or tables are saved.
    :param fill_between_alpha: Alpha (transparency) value for the fill-shaded region in the plots.
    :param headless: Flag indicating whether to run in headless mode without GUI display.
    :param metric_acro_label: Short acronym or symbol representing the metric (e.g., "MAE").
    :param metric_label: Full descriptive name of the metric (e.g., "Mean Absolute Error").
    :param plot_linewidth: Line width for plotting the graph lines.
    :param target_type_label: Type of the target being analyzed (e.g., prediction type or source of data).
    :param warmup_mode: Ground-truth-feed warm-up treatment mode
        (``None`` / ``'discard'`` / ``'reference_line'``) applied per-category
        axis; see :func:`_apply_ground_truth_warmup_treatment`. Defaults to
        ``None`` (no treatment).
    :param warmup_steps: Auto-derived warm-up step count applied by the
        treatment; ``None`` when unresolvable.
    :return: None. Outputs are visualizations and optionally saved files in the experiment directory.
    """
    if do_category_breakdown and category_breakdown_data:
        category_max_lengths = _get_category_max_length(cfg)

        with warnings.catch_warnings():
            manage_matplotlib_warnings()
            manage_matplotlib_backend(get_plot_render_type_cfg(cfg).show_plot, headless)

            for cat in ["S", "M", "L"]:
                # Collect groups that have data for this category
                cat_groups = {}
                for grp_key, cat_dict in category_breakdown_data.items():
                    if cat in cat_dict:
                        cat_groups[grp_key] = cat_dict[cat]
                if not cat_groups:
                    continue

                cat_max_len = category_max_lengths.get(cat, "?")
                if do_average_across_trajectories:
                    cat_variation_desc = f"inter-trajectory {metric_acro_label} std (averaged across {trials_label})"
                else:
                    cat_variation_desc = (
                        f"inter-trial {metric_acro_label} std (all rollouts)"
                    )
                cat_title = (
                    f"{cfg.env_name} test-time rollout {metric_label} ({metric_acro_label}) — "
                    f"Category {cat} (max {cat_max_len} timesteps).\n"
                    f"Variation type: {cat_variation_desc}. "
                    f"{compounded_pred_label}. {target_type_label}."
                )

                cat_fig_init = False
                cat_color_idx = 0
                cat_fig = None
                cat_ax = None
                for grp_key, cat_data in cat_groups.items():
                    grp_avg_mae = cat_data["grp_avg_mae"]
                    grp_std_mae = cat_data["grp_std_mae"]
                    line_style = cat_data["line_style"]
                    if line_style is None:
                        line_style = "-"
                    n_traj = cat_data["n_trajectories"]
                    n_trials = cat_data.get("n_trials")
                    n_total = cat_data.get("n_total_rollouts")
                    label = _format_grp_count_label(
                        cat_data["grp_name"],
                        n_trials=n_trials,
                        n_trajectories=n_traj,
                        n_total_rollouts=n_total,
                        trials_label=trials_label,
                        do_average_across_trajectories=do_average_across_trajectories,
                        show_acronym=cfg.show.get("acronym", True),
                    )

                    # Ground-truth-feed warm-up treatment (YouTrack RLRP-733):
                    # in ``'discard'`` mode drop the ``0 .. warm-up`` prefix
                    # from the curve and its band (step ``k`` -> ``x = k``,
                    # ``x_offset=0``).
                    _x_full = np.arange(grp_avg_mae.size)
                    _lower = grp_avg_mae - cfg.show.variation_scale * grp_std_mae
                    _upper = grp_avg_mae + cfg.show.variation_scale * grp_std_mae
                    if warmup_mode == "discard":
                        _x_curve, _y_curve = _trim_warmup_interval(
                            _x_full, grp_avg_mae, warmup_steps=warmup_steps
                        )
                        _x_band, _lower, _upper, _ = _trim_warmup_band(
                            _x_full, _lower, _upper, warmup_steps=warmup_steps
                        )
                    else:
                        _x_curve, _y_curve = _x_full, grp_avg_mae
                        _x_band = _x_full

                    if not cat_fig_init:
                        cat_fig, cat_ax = arbitrary_dimension_array_plot(
                            grp_avg_mae,
                            plot_label=label,
                            y_label=(f"{metric_label} ({metric_acro_label})",),
                            title=cat_title,
                            figsize=get_plot_figsize(
                                cfg, "generate_per_category_plots"
                            ),
                            figdpi=get_plot_render_type_cfg(cfg).figdpi,
                            plot_style={
                                "color": COLORS[0],
                                "linestyle": line_style,
                                "linewidth": plot_linewidth,
                                "alpha": 1.0,
                            },
                        )
                        cat_fig_init = True
                        if warmup_mode == "discard" and warmup_steps is not None:
                            # Re-seat the implicit-x first curve onto the
                            # trimmed, boundary-based x.
                            cat_ax.get_lines()[-1].set_data(_x_curve, _y_curve)
                    else:
                        cat_ax.plot(
                            _x_curve,
                            _y_curve,
                            color=COLORS[cat_color_idx],
                            linestyle=line_style,
                            linewidth=plot_linewidth,
                            alpha=1.0,
                            label=label,
                        )

                    cat_ax.fill_between(
                        _x_band,
                        _lower,
                        _upper,
                        color=COLORS[cat_color_idx],
                        alpha=fill_between_alpha,
                    )

                    if cat_color_idx < len(COLORS) - 1:
                        cat_color_idx += 1
                    else:
                        cat_color_idx = 0

                if cat_fig is not None:
                    if y_axis_in_logscale:
                        cat_ax.set_yscale("log")
                    else:
                        cat_ax.set_ylim(bottom=0)
                    # x-axis spans the category max length (S=2613, M=3577, L=10566)
                    # In ``'discard'`` mode start at the warm-up boundary
                    # (arrays already trimmed there) so the axes are faithful
                    # (YouTrack RLRP-733).
                    _discard_left = _warmup_discard_left(warmup_mode, warmup_steps)
                    _left = _discard_left if _discard_left is not None else 0
                    if isinstance(cat_max_len, int):
                        cat_ax.set_xlim(left=_left, right=cat_max_len)
                    else:
                        cat_ax.set_xlim(left=_left)

                    # Ground-truth-feed warm-up treatment (YouTrack RLRP-733).
                    # Applied per-category axis. Step ``k`` maps to ``x = k``
                    # here (``x_offset=0``); no-op when ``warmup_mode`` is None.
                    if cfg.show.get("compounded_predictions_score", False):
                        _apply_ground_truth_warmup_treatment(
                            cfg, cat_ax, warmup_steps, mode=warmup_mode
                        )

                    cat_ax.legend(loc="best")
                    cat_ax.grid(True)
                    cat_ax.set_xlabel("Time-steps", **AXIS_LABEL_STYLE)
                    _ensure_primary_xaxis_endpoint_ticks(cat_ax)
                    _ensure_primary_yaxis_endpoint_ticks(cat_ax)
                    _attach_timestamp_secondary_xaxis(cfg, cat_ax)
                    cat_fig.tight_layout(pad=0.99)

                    display_experiment_name(cfg, cat_fig)

                    show_and_save_plot_helper(
                        cat_fig,
                        exp_dir_relative_path,
                        f"test_time_models_comparaison_category_{cat}",
                        headless,
                        get_plot_render_type_cfg(cfg).show_plot,
                        get_plot_render_type_cfg(cfg).save_dpi,
                    )

        # .... Category summary table ..............................................................
        if do_summary_table:
            with warnings.catch_warnings():
                manage_matplotlib_warnings()
                manage_matplotlib_backend(
                    get_plot_render_type_cfg(cfg).show_plot, headless
                )

                # Build table data: rows = groups, columns = categories
                table_rows = []
                row_labels = []
                for grp_key, cat_dict in category_breakdown_data.items():
                    if not cat_dict:
                        continue
                    grp_name = next(iter(cat_dict.values()))["grp_name"]
                    row_labels.append(grp_name)
                    row = []
                    for cat in ["S", "M", "L"]:
                        if cat in cat_dict:
                            d = cat_dict[cat]
                            n_trials = d.get("n_trials")
                            n_traj = d.get("n_trajectories")
                            n_total = d.get("n_total_rollouts")
                            if n_trials and n_traj and n_traj > 1 and n_total:
                                count_str = (
                                    f"({n_trials} {trials_label} \u00d7 {n_traj} trajectories"
                                    f" = {n_total} rollouts)"
                                )
                            elif n_trials and (not n_traj or n_traj <= 1):
                                count_str = f"(n={n_trials} {trials_label})"
                            elif n_total is not None:
                                count_str = (
                                    f"(n={n_traj} trajectories, {n_total} rollouts)"
                                )
                            else:
                                count_str = f"(n={n_traj} rollouts)"
                            row.append(
                                f"{d['mean_terminal_mae']:.4f} ± {d['std_terminal_mae']:.4f}\n"
                                f"{count_str}"
                            )
                        else:
                            row.append("—")
                    table_rows.append(row)

                if table_rows:
                    col_labels = [
                        f"Cat {cat} (max {category_max_lengths.get(cat, '?')}t)"
                        for cat in ["S", "M", "L"]
                    ]
                    n_cols = len(col_labels)
                    n_rows = len(table_rows)

                    # Compute dynamic sizing robust to long model names
                    max_label_len = max((len(lbl) for lbl in row_labels), default=20)
                    label_width_inches = max(6.0, max_label_len * 0.14)
                    col_width_inches = 4.0
                    fig_w = label_width_inches + col_width_inches * n_cols + 2.0
                    row_h = 1.2  # inches per data row (2-line cells: value + count)
                    fig_h = max(4.0, 2.5 + row_h * (n_rows + 1))

                    table_fig, table_ax = plt.subplots(
                        figsize=(fig_w, fig_h),
                        dpi=150,
                    )
                    table_ax.axis("off")
                    table_ax.set_title(
                        f"{cfg.env_name} — Category Breakdown Summary ({metric_acro_label})\n"
                        f"{compounded_pred_label}. {target_type_label}.",
                        fontsize=22,
                        fontweight="bold",
                        pad=30,
                    )
                    tbl = table_ax.table(
                        cellText=table_rows,
                        rowLabels=row_labels,
                        colLabels=col_labels,
                        loc="center",
                        cellLoc="center",
                    )
                    tbl.auto_set_font_size(False)
                    tbl.set_fontsize(16)
                    tbl.auto_set_column_width(col=list(range(-1, n_cols)))
                    tbl.scale(1.0, 4.0)
                    table_fig.tight_layout(pad=3.0)

                    show_and_save_plot_helper(
                        table_fig,
                        exp_dir_relative_path,
                        "test_time_models_comparaison_category_summary_table",
                        headless,
                        get_plot_render_type_cfg(cfg).show_plot,
                        get_plot_render_type_cfg(cfg).save_dpi,
                    )


def generate_per_trajectory_plots(
    cfg: DictConfig,
    trajectory_breakdown_data: dict[str, dict[str, dict]],
    compounded_pred_label: str,
    do_trajectory_breakdown: bool,
    exp_dir_relative_path: Any,
    fill_between_alpha: float,
    plot_linewidth: float,
    metric_acro_label: str,
    metric_label: str,
    target_type_label: str,
    headless: bool,
    ground_truth_entries: Any = None,
    *,
    trials_label: str = "trials",
    y_axis_in_logscale: bool = False,
    warmup_mode: Any = None,
    warmup_steps: int | None = None,
) -> None:
    """Render one plot per ``trajectory_name``.

    :param cfg:
    :param trajectory_breakdown_data:
    :param compounded_pred_label:
    :param do_trajectory_breakdown:
    :param exp_dir_relative_path:
    :param fill_between_alpha:
    :param plot_linewidth:
    :param metric_acro_label:
    :param metric_label:
    :param target_type_label:
    :param headless:
    :param ground_truth_entries:
    :param trials_label: User-facing word substituted for ``"trials"`` in the
        legend's count label (e.g. ``({n} <trials_label>)``). Defaults to
        ``"trials"``; override via ``label_override.trials`` in the top-level
        plot config to keep this surface consistent with the General plot and
        category-breakdown plots.
    :param y_axis_in_logscale:
    :param warmup_mode: Ground-truth-feed warm-up treatment mode
        (``None`` / ``'discard'`` / ``'reference_line'``) applied per-trajectory
        axis; see :func:`_apply_ground_truth_warmup_treatment`. Defaults to
        ``None`` (no treatment).
    :param warmup_steps: Auto-derived warm-up step count applied by the
        treatment; ``None`` when unresolvable.

    Iterates over the union of trajectory names across all groups. For each
    trajectory, overlays every group's curve with its variation band, embeds a
    1/8-scaled 3D ground-truth snapshot inset in the upper-left corner (when a
    matching :class:`TestTrajectoryEntry` is provided in ``ground_truth_entries``),
    and saves via :func:`show_and_save_plot_helper`.

    Snapshot failures are non-fatal: the inset is silently skipped and the
    plot is still produced (plan R2 mitigation).
    """
    if not (do_trajectory_breakdown and trajectory_breakdown_data):
        return

    # Build an entry lookup keyed by trajectory_name so the snapshot helper
    # can be called by name regardless of ordering.
    entry_lookup: dict[str, Any] = {}
    if ground_truth_entries is not None:
        for entry in ground_truth_entries:
            tname = getattr(entry, "trajectory_name", None)
            if tname is not None:
                entry_lookup[tname] = entry
                # Also index by basename so mismatched key conventions
                # (e.g. cfg uses ``"test/ellipse"`` while the TTRPM directory
                # scan produces ``"ellipse"``) still resolve to the right entry.
                entry_lookup.setdefault(os.path.basename(tname), entry)
            short = getattr(entry, "short_name", None)
            if short:
                entry_lookup.setdefault(short, entry)

    # Union of trajectory names across groups (sorted for deterministic output).
    all_tnames: set[str] = set()
    for traj_dict in trajectory_breakdown_data.values():
        all_tnames.update(traj_dict.keys())

    with warnings.catch_warnings():
        manage_matplotlib_warnings()
        manage_matplotlib_backend(get_plot_render_type_cfg(cfg).show_plot, headless)

        for tname in sorted(all_tnames):
            # Collect groups that have data for this trajectory
            tname_groups = {}
            for grp_key, tdict in trajectory_breakdown_data.items():
                if tname in tdict:
                    tname_groups[grp_key] = tdict[tname]
            if not tname_groups:
                continue

            short_name = next(iter(tname_groups.values())).get(
                "short_name", os.path.basename(tname)
            )
            tname_len = next(iter(tname_groups.values())).get("length")
            variation_desc = f"inter-trial {metric_acro_label} std (all rollouts)"
            t_title = (
                f"{cfg.env_name} test-time rollout {metric_label} ({metric_acro_label}) — "
                f"Trajectory {short_name}"
                f"{f' ({tname_len} timesteps)' if tname_len is not None else ''}.\n"
                f"Variation type: {variation_desc}. "
                f"{compounded_pred_label}. {target_type_label}."
            )

            t_fig = None
            t_ax = None
            t_fig_init = False
            t_color_idx = 0
            for grp_key, t_data in tname_groups.items():
                grp_avg_mae = t_data["grp_avg_mae"]
                grp_std_mae = t_data["grp_std_mae"]
                line_style = t_data["line_style"] or "-"
                n_rollouts = t_data["n_rollouts"]
                label = f"{t_data['grp_name']} ({n_rollouts} {trials_label})"

                # Ground-truth-feed warm-up treatment (YouTrack RLRP-733): in
                # ``'discard'`` mode drop the ``0 .. warm-up`` prefix from the
                # curve and its band (step ``k`` -> ``x = k``, ``x_offset=0``).
                _x_full = np.arange(grp_avg_mae.size)
                _lower = grp_avg_mae - cfg.show.variation_scale * grp_std_mae
                _upper = grp_avg_mae + cfg.show.variation_scale * grp_std_mae
                if warmup_mode == "discard":
                    _x_curve, _y_curve = _trim_warmup_interval(
                        _x_full, grp_avg_mae, warmup_steps=warmup_steps
                    )
                    _x_band, _lower, _upper, _ = _trim_warmup_band(
                        _x_full, _lower, _upper, warmup_steps=warmup_steps
                    )
                else:
                    _x_curve, _y_curve = _x_full, grp_avg_mae
                    _x_band = _x_full

                if not t_fig_init:
                    t_fig, t_ax = arbitrary_dimension_array_plot(
                        grp_avg_mae,
                        plot_label=label,
                        y_label=(f"{metric_label} ({metric_acro_label})",),
                        title=t_title,
                        figsize=get_plot_figsize(
                            cfg, "generate_per_trajectory_plots"
                        ),
                        figdpi=get_plot_render_type_cfg(cfg).figdpi,
                        plot_style={
                            "color": COLORS[0],
                            "linestyle": line_style,
                            "linewidth": plot_linewidth,
                            "alpha": 1.0,
                        },
                    )
                    t_fig_init = True
                    if warmup_mode == "discard" and warmup_steps is not None:
                        # Re-seat the implicit-x first curve onto the trimmed,
                        # boundary-based x.
                        t_ax.get_lines()[-1].set_data(_x_curve, _y_curve)
                else:
                    t_ax.plot(
                        _x_curve,
                        _y_curve,
                        color=COLORS[t_color_idx],
                        linestyle=line_style,
                        linewidth=plot_linewidth,
                        alpha=1.0,
                        label=label,
                    )

                t_ax.fill_between(
                    _x_band,
                    _lower,
                    _upper,
                    color=COLORS[t_color_idx],
                    alpha=fill_between_alpha,
                )

                if t_color_idx < len(COLORS) - 1:
                    t_color_idx += 1
                else:
                    t_color_idx = 0

            if t_fig is not None:
                if y_axis_in_logscale:
                    t_ax.set_yscale("log")
                else:
                    t_ax.set_ylim(bottom=0)
                # In ``'discard'`` mode start at the warm-up boundary (arrays
                # already trimmed there) so the axes are faithful (RLRP-733).
                _discard_left = _warmup_discard_left(warmup_mode, warmup_steps)
                _left = _discard_left if _discard_left is not None else 0
                if isinstance(tname_len, int):
                    t_ax.set_xlim(left=_left, right=tname_len)
                else:
                    t_ax.set_xlim(left=_left)
                # Ground-truth-feed warm-up treatment (YouTrack RLRP-733).
                # Applied per-trajectory axis. Step ``k`` maps to ``x = k``
                # here (``x_offset=0``); no-op when ``warmup_mode`` is None.
                if cfg.show.get("compounded_predictions_score", False):
                    _apply_ground_truth_warmup_treatment(
                        cfg, t_ax, warmup_steps, mode=warmup_mode
                    )
                t_ax.legend(loc="best")
                t_ax.grid(True)
                t_ax.set_xlabel("Time-steps", **AXIS_LABEL_STYLE)
                _ensure_primary_xaxis_endpoint_ticks(t_ax)
                _ensure_primary_yaxis_endpoint_ticks(t_ax)
                _attach_timestamp_secondary_xaxis(cfg, t_ax)
                t_fig.tight_layout(pad=0.99)

                display_experiment_name(cfg, t_fig)

                # Snapshot inset (graceful-skip on any failure).
                _entry_for_snapshot = entry_lookup.get(tname) or entry_lookup.get(
                    os.path.basename(tname)
                )
                snapshot_img = render_trajectory_ground_truth_snapshot(
                    tname, entry=_entry_for_snapshot
                )
                add_snapshot_inset(t_ax, snapshot_img)

                # Sanitise short_name for filesystem use.
                safe_short = "".join(
                    c if c.isalnum() or c in ("-", "_", ".") else "_"
                    for c in short_name
                )
                # Save + show per iteration, mirroring the category-breakdown
                # pattern (``show_and_save_plot_helper``) which successfully
                # renders all 3 per-category figures in PyCharm SciView.
                # ``show_and_save_plot_helper`` dispatches the current figure
                # to the interagg backend on each call — this is the pattern
                # that actually works for multi-figure display in PyCharm.
                show_and_save_plot_helper(
                    t_fig,
                    exp_dir_relative_path,
                    f"test_time_models_comparaison_trajectory_{safe_short}",
                    headless,
                    get_plot_render_type_cfg(cfg).show_plot,
                    get_plot_render_type_cfg(cfg).save_dpi,
                )

    return None


def _collect_wall_clock_samples(
    cfg: DictConfig,
    groups: dict[Any, Any],
    key: str,
    unit_label: str = "[s]",
    *,
    return_records: bool = False,
) -> tuple:
    """Collect per-group, per-trial wall-clock-time samples for box plots.

    ``key`` is either ``"training_wall_clock_time"`` or
    ``"rollout_wall_clock_time"``. After :func:`compute_per_group_metric` each
    entry of ``groups[gk][key]`` is a per-trial scalar float (already reduced
    via ``feature_reduction``). Non-finite and sentinel ``0.0`` values
    (assigned when the original ``training_wall_clock_time`` list contained
    ``None``) are filtered out; groups that end up empty are dropped and a
    warning is emitted.

    Sample granularity differs per ``key`` because the two metrics live at
    different levels of the experiment tree:

    - ``training_wall_clock_time`` is a property of the TRAINED MODEL, so
      exactly ONE sample per trial (group seed) is emitted -- the metrics of
      ``_original_metrics`` are bucketed by their canonical ``_trial_key``
      first, hence ``len(samples[i]) == n_trials`` (RLRP-818 CR10). Before this
      de-duplication the same training time was counted once per trajectory,
      i.e. ``n_trials x n_trajectories`` times.
    - ``rollout_wall_clock_time`` is a property of each ROLLOUT, so every
      ``(trial, trajectory)`` pair legitimately contributes its own sample.

    Groups are filtered by ``cfg.show.grp_names`` when set, mirroring the
    selection logic used by :func:`generate_final_general_plot`.

    :param cfg: Top-level multirun plot configuration.
    :param groups: Per-group metric dict from :func:`compute_per_group_metric`.
    :param key: Either ``"training_wall_clock_time"`` or
        ``"rollout_wall_clock_time"``.
    :param return_records: RLRP-818 opt-in enrichment. When ``True`` a THIRD
        element is returned carrying, per retained group, the canonical
        identity that ``labels`` loses (``grp_key`` / ``grp_name`` /
        ``gpr_short_name``) -- needed by the experiment data consolidation.
        Defaults to ``False`` so the existing 2-tuple callers (in particular
        :func:`generate_rollout_wall_clock_time_plot`) are unaffected.
    :return: ``(labels, samples)``, or ``(labels, samples, records)`` when
        ``return_records`` is ``True``, where ``labels`` is the list of
        human-readable group names and ``samples[i]`` is the corresponding
        list of finite per-trial wall-clock-time samples.
    """
    _selected_grp_names = (
        set(cfg.show.grp_names) if cfg.show.get("grp_names", None) else None
    )

    labels: list[str] = []
    samples: list[list[float]] = []
    records: list[dict[str, Any]] = []
    for grp_key, group in groups.items():
        _grp_name = group.get("grp_name", grp_key)
        _grp_display_name = group.get("gpr_short_name", None) or _grp_name
        if _selected_grp_names is not None and _grp_name not in _selected_grp_names:
            continue

        finite: list[float] = []
        n_recorded = 0
        n_recovered = 0

        # --- Per-trial assembly for ``training_wall_clock_time`` ---
        # Iterate ``_original_metrics`` (bucketed per trial, see below) so
        # each trial can independently contribute either its recorded value
        # or — when stale (``0.0`` sentinel / non-finite) — a recovered value
        # (:func:`_recover_training_wall_clock_times_per_trial`). This
        # avoids the previous "all-or-nothing" group-level fallback
        # (which silently dropped stale sub-multiruns inside a group
        # that also contained fresh ones — the V0B vs V0B2 discrepancy)
        # and the "one synthetic sample per ``experiment_base``"
        # collapsing (which under-sampled recovered groups vs recorded
        # ones).
        _orig_metrics = group.get("_original_metrics", None) or []
        if key == "training_wall_clock_time" and _orig_metrics:
            _per_trial_cache: dict[str, dict[str, float]] = {}

            # RLRP-818 CR10 — ``_original_metrics`` holds ONE entry per
            # ``(trial x trajectory)`` rollout, whereas
            # ``training_wall_clock_time`` is a property of the TRAINED MODEL,
            # i.e. of the trial (group seed) alone. Bucket the metrics by their
            # canonical trial identity — ``_trial_key ==
            # (experiment_path, multirun_path)``, the very key
            # :func:`compute_per_group_metric` uses to derive ``n_trials`` — so
            # each training run contributes EXACTLY ONE sample instead of one
            # per trajectory (``n_trials`` samples instead of
            # ``n_trials x n_trajectories``).
            _metrics_per_trial: dict[Any, list[Any]] = {}
            for _idx, _metric in enumerate(_orig_metrics):
                _tk = getattr(_metric, "_trial_key", None)
                # No trial provenance (legacy fixtures / hand-built groups):
                # keep the metric standalone so the historical
                # one-sample-per-metric behaviour is preserved instead of
                # silently collapsing every such metric into one sample.
                _bucket_key = _tk if _tk is not None else ("__no_trial_key__", _idx)
                _metrics_per_trial.setdefault(_bucket_key, []).append(_metric)

            for _trial_metrics in _metrics_per_trial.values():
                # --- Recorded sample for this trial ---
                # Every rollout of a trial carries the same training time, so
                # the first resolvable value is authoritative.
                _recorded_val: float | None = None
                for _metric in _trial_metrics:
                    _rv = getattr(_metric, key, None)
                    if _rv is None:
                        continue
                    try:
                        _arr = np.asarray(_rv, dtype=float).ravel()
                        _arr = _arr[np.isfinite(_arr)]
                        _arr = _arr[_arr != 0.0]
                        if _arr.size:
                            _recorded_val = float(np.mean(_arr))
                            break
                    except Exception:
                        continue
                if _recorded_val is not None:
                    if unit_label == "[h]":
                        _recorded_val = _recorded_val / 60 / 60

                    finite.append(_recorded_val)
                    n_recorded += 1
                    continue

                # --- Per-trial recovery fallback ---
                _recovered = None
                for _metric in _trial_metrics:
                    _exp_base = getattr(_metric, "_experiment_base", None)
                    _trial_key = getattr(_metric, "_trial_key", None)
                    if not _exp_base or not _trial_key:
                        continue
                    if _exp_base not in _per_trial_cache:
                        try:
                            _per_trial_cache[_exp_base] = (
                                _recover_training_wall_clock_times_per_trial(_exp_base)
                            )
                        except Exception:
                            _per_trial_cache[_exp_base] = {}
                    _trial_dirname = (
                        _trial_key[1]
                        if isinstance(_trial_key, (tuple, list)) and len(_trial_key) > 1
                        else None
                    )
                    # Try the trial-key dirname first (multirun-root layout),
                    # then ``basename(_exp_base)`` (per-trial layout where
                    # ``experiment_base`` already is the trial dir — V0B2's
                    # TCN/GRU case: ``_trial_key[1]`` is a relative sub-path
                    # like ``dynamics_model@…,trial_nb=1`` and the recovered
                    # dict is keyed by ``basename(_exp_base)`` matching the
                    # same trial dirname).
                    if _trial_dirname:
                        _recovered = _per_trial_cache[_exp_base].get(_trial_dirname)
                    if _recovered is None:
                        _recovered = _per_trial_cache[_exp_base].get(
                            os.path.basename(os.path.normpath(_exp_base))
                        )
                    if _recovered is not None:
                        break
                if (
                    _recovered is not None
                    and np.isfinite(_recovered)
                    and _recovered != 0.0
                ):

                    if unit_label == "[h]":
                        _recovered = _recovered / 60 / 60

                    finite.append(float(_recovered))
                    n_recovered += 1
        else:
            # Rollout key (or training key without ``_original_metrics``
            # in legacy fixtures): keep the original raw-list filter.
            raw = group.get(key, None)
            if raw is None:
                continue
            for v in raw:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(fv):
                    continue
                # 0.0 is the sentinel assigned upstream when the trial's
                # wall-clock time was missing (None); skip it so it does
                # not collapse the box on the axis.
                if fv == 0.0:
                    continue

                if unit_label == "[h]":
                    fv = fv / 60 / 60

                finite.append(fv)
                n_recorded += 1

        if n_recovered:
            warnings.warn(
                f"_collect_wall_clock_samples: group '{_grp_name}' "
                f"for key '{key}' assembled {n_recorded} recorded + "
                f"{n_recovered} recovered per-trial sample(s) "
                f"(per-trial fallback via "
                f"_recover_training_wall_clock_times_per_trial)."
            )

        if not finite:
            warnings.warn(
                f"_collect_wall_clock_samples: group '{_grp_name}' has no "
                f"finite samples for key '{key}'; skipping in box plot."
            )
            continue
        labels.append(_grp_display_name)
        samples.append(finite)
        records.append(
            {
                "grp_key": grp_key,
                "grp_name": _grp_name,
                "gpr_short_name": group.get("gpr_short_name", None),
                "label": _grp_display_name,
                "n_recorded": n_recorded,
                "n_recovered": n_recovered,
                "n_trials": group.get("n_trials", None),
                "n_trajectories": group.get("n_trajectories", None),
            }
        )
    if return_records:
        return labels, samples, records
    return labels, samples


def _measured_inference_rate(
    bset: Any,
    level: str,
    regime: str | FeedbackRegime | None,
) -> float | None:
    """Read the measured rate (Hz) of one trial's ``BenchmarkMetricSet``.

    The timing pass follows ``BenchmarkMetricSet``'s own headline/model helpers:
    ``model_call`` is reported on the device-time pass, every other level on the
    deployable-latency pass (R1).

    :param bset: Per-trial :class:`BenchmarkMetricSet` (or ``None``).
    :param level: Benchmark level string (e.g. "control_loop_step").
    :param regime: Feedback regime, or ``None`` for the regime-agnostic headline.
    :return: The measured rate in Hz, or ``None`` when not instrumented.
    """
    if bset is None:
        return None
    _level = BenchmarkLevel(level)
    if regime is not None:
        t_pass = (
            TimingPass.DEVICE_TIME
            if _level is BenchmarkLevel.MODEL_CALL
            else TimingPass.DEPLOYABLE_LATENCY
        )
        metric = bset.get(_level, FeedbackRegime(regime), t_pass)
        return None if metric is None else metric.rate_hz
    # Regime-agnostic: use the public accessors (both prefer ESTIMATOR) rather
    # than the private ``_first_by``.
    if _level is BenchmarkLevel.MODEL_CALL:
        return bset.model_rate_hz()
    if _level is BenchmarkLevel.CONTROL_LOOP_STEP:
        return bset.headline_rate_hz()
    return None


def _collect_inference_rate_samples(
    group_dict: dict[str, Any],
    level: str,
    regime: str | FeedbackRegime | None = None,
    source: str = "measured",
) -> list[float]:
    """Collect per-trial rate samples (Hz) for one level (and optionally regime).

    HONESTY RULE: ``source="measured"`` NEVER falls back to the harness rate --
    a missing ``StepBenchmarkMetric`` yields no sample so the caller can fail
    fast instead of captioning a harness figure as measured. The
    ``steps / seconds`` fallback is reserved for ``"harness"`` and ``"auto"``.

    :param group_dict: Per-group metric dict.
    :param level: Benchmark level string (e.g. "control_loop_step").
    :param regime: Feedback regime (optional).
    :param source: "measured" (StepBenchmarkMetric only) | "harness"
        (steps/seconds only) | "auto" (measured, harness fallback).
    :return: List of finite per-trial rate samples (Hz).
    """
    samples: list[float] = []
    benchmarks = group_dict.get("benchmark", [])
    steps_list = group_dict.get("rollout_steps", [])
    times_list = group_dict.get("rollout_wall_clock_time", [])

    # Aligned walk over trials
    n_trials = len(times_list)
    for i in range(n_trials):
        rate = None
        if source in ("measured", "auto"):
            rate = _measured_inference_rate(
                benchmarks[i] if i < len(benchmarks) else None, level, regime
            )

        if rate is None and source != "measured":
            # Harness rate: steps / seconds (explicitly labelled as such upstream)
            s = steps_list[i] if i < len(steps_list) else None
            t = times_list[i]
            if s is not None and t is not None and t > 0:
                rate = float(s) / float(t)

        if rate is not None and np.isfinite(rate):
            samples.append(rate)

    return samples


def _has_measured_inference_metric(
    group_dict: dict[str, Any],
    level: str,
    regime: str | FeedbackRegime | None = None,
) -> bool:
    """``True`` when at least one trial of the group carries a measured rate."""
    for bset in group_dict.get("benchmark", []):
        if _measured_inference_rate(bset, level, regime) is not None:
            return True
    return False


def generate_rollout_inference_speed_plot(
    cfg: DictConfig,
    groups: dict[Any, Any],
    exp_dir_relative_path: Any,
    headless: bool,
    *,
    unit: str = "hz",
    levels: Sequence[str] = ("control_loop_step", "model_call"),
    source: str = "measured",
    y_axis_in_logscale: bool = False,
) -> None:
    """Render a box plot of per-trial inference speed (Hz or fps) per group.

    Levels are rendered as adjacent boxes within each experiment group.
    Feedback regimes (estimator vs autoregressive) are likewise separate populations.

    :param cfg: Top-level multirun plot configuration.
    :param groups: Per-group metric dict from :func:`compute_per_group_metric`.
    :param exp_dir_relative_path: Where to save the figure.
    :param headless: Headless backend flag.
    :param unit: "hz" | "fps" (label only, numerically identical).
    :param levels: Sequence of benchmark levels to plot.
    :param source: "measured" | "harness" | "auto".
    :param y_axis_in_logscale: When ``True``, render the y-axis in log scale.
    :raises RuntimeError: When ``source="measured"`` but no trial carries a
        ``StepBenchmarkMetric`` (HONESTY RULE: never caption a harness figure
        as measured).
    """
    _selected_grp_names = (
        set(cfg.show.grp_names) if cfg.show.get("grp_names", None) else None
    )

    # We reuse the _generate_wall_clock_time_boxplot renderer by passing it
    # a pseudo-groups dict where each "group" is a (group, level, regime) combination.
    pseudo_groups = {}
    # Track what the collected populations really are, so the caption can never
    # claim "measured" over a harness fallback.
    _any_measured = False
    _any_harness = False

    # We want to preserve the order of groups and levels.
    for grp_key, group in groups.items():
        _grp_display_name = group.get("grp_name", grp_key)
        if (
            _selected_grp_names is not None
            and _grp_display_name not in _selected_grp_names
        ):
            continue

        for level in levels:
            # Regimes: we only distinguish them if the samples are measured.
            # "the two feedback regimes are likewise separate populations"
            regimes = (
                (FeedbackRegime.ESTIMATOR, FeedbackRegime.AUTOREGRESSIVE)
                if source in ("measured", "auto")
                else (None,)
            )

            for regime in regimes:
                samples = _collect_inference_rate_samples(group, level, regime, source)
                if not samples:
                    continue

                # HONESTY RULE: an "auto" population that has no measured metric
                # IS a harness population and must be captioned as such.
                _is_harness = source == "harness" or (
                    source == "auto"
                    and not _has_measured_inference_metric(group, level, regime)
                )
                if _is_harness:
                    _any_harness = True
                else:
                    _any_measured = True

                # Build a descriptive label
                # HONESTY RULE: caption states which level it reports, and says "harness" explicitly.
                label = _grp_display_name
                if _is_harness:
                    label += "\n(harness)"
                else:
                    label += f"\n({level})"
                    if regime:
                        label += f"\n({regime.value})"

                pseudo_key = f"{grp_key}_{level}_{regime}"
                pseudo_groups[pseudo_key] = {
                    "grp_name": label,
                    "inference_rate": samples,
                }

    if not pseudo_groups:
        if source == "measured":
            # Fail fast: silently degrading to the harness rate would caption a
            # ``steps / seconds`` figure as a measured per-level one.
            raise RuntimeError(
                "generate_rollout_inference_speed_plot: 'source: measured' was "
                f"requested for level(s) {list(levels)} but no trial carries a "
                "StepBenchmarkMetric. Re-run the deployment with the "
                "'deploy.benchmark' instrumentation enabled (it is what writes "
                "the per-level BenchmarkMetricSet), or set "
                "'show.generate_rollout_inference_speed_plot.source: harness' "
                "to plot the harness rate (steps / seconds) explicitly."
            )
        warnings.warn(
            "generate_rollout_inference_speed_plot: no finite samples collected; skipping."
        )
        return None

    unit_label = _RATE_UNIT_LABEL.get(unit.lower(), "Hz")

    # HONESTY RULE: every figure/caption states which level it reports, and says
    # "harness" explicitly. This must be visible in the y-label or title.
    info_parts = []
    if _any_measured:
        info_parts.extend(levels)
    if _any_harness:
        info_parts.append("harness")
    display_info = f" ({', '.join(info_parts)})"

    _generate_wall_clock_time_boxplot(
        cfg,
        pseudo_groups,
        exp_dir_relative_path,
        headless,
        key="inference_rate",
        metric_human_name=f"Inference speed{display_info}",
        file_name="test_time_models_comparaison_rollout_inference_speed_boxplot",
        y_axis_in_logscale=y_axis_in_logscale,
        unit_label=f"[{unit_label}]",
    )
    return None


# Cost-attribution components returned by ``BenchmarkMetricSet.cost_breakdown_ms``,
# in control-loop nesting order (innermost first) so the stacked bar reads
# bottom-up like the instrumentation nests.
_COST_BREAKDOWN_COMPONENTS: tuple[str, ...] = (
    "model_call",
    "buffers_and_model_adapter",
    "env_adapters_and_feedback",
)

_COST_BREAKDOWN_COMPONENT_LABEL: dict[str, str] = {
    "model_call": "Model call",
    "buffers_and_model_adapter": "Buffers & model adapter",
    "env_adapters_and_feedback": "Env adapters & feedback",
}


def _collect_cost_breakdown_regimes(
    groups: dict[Any, Any],
) -> list[FeedbackRegime]:
    """The feedback regimes the cost-breakdown figure can be split on.

    Introduced by action ``RLRP-803-7.2`` of the RLRC test-time rollout deployer
    benchmarking `.junie` plan
    (``perf_RLRP-785_testtime_rollout_benchmarking_plan_20260903.md``).
    Permanent.

    An EMPTY list means no trial labels its level deltas -- a schema v2
    artifact -- and the figure must then say so instead of implying a regime.

    :param groups: Per-group metric dict from :func:`compute_per_group_metric`.
    :return: the labelled regimes, in :class:`FeedbackRegime` declaration order.
    """
    labelled: set = set()
    for group in groups.values():
        for bset in group.get("benchmark", []) or []:
            if isinstance(bset, BenchmarkMetricSet):
                labelled.update(bset.cost_breakdown_regimes())
    return [regime for regime in FeedbackRegime if regime in labelled]


def _collect_cost_breakdown_ms(
    group_dict: dict[str, Any],
    feedback_regime: Optional[FeedbackRegime] = None,
) -> dict[str, float]:
    """Aggregate the per-trial cost attribution (ms) of one group.

    Each trial's :meth:`BenchmarkMetricSet.cost_breakdown_ms` is a median-based
    decomposition; trials are combined with the median so one slow trial does
    not dominate the bar.

    :param group_dict: Per-group metric dict.
    :param feedback_regime: restrict the attribution to that feedback regime
        (action ``RLRP-803-7.2``). ``None`` keeps the pooled breakdown, whose
        pooling rule is documented on
        :meth:`BenchmarkMetricSet.cost_breakdown_ms`.
    :return: ``{component: median ms}`` restricted to the components measured.
    """
    per_component: dict[str, list[float]] = {}
    for bset in group_dict.get("benchmark", []):
        if not isinstance(bset, BenchmarkMetricSet):
            continue
        for component, value in bset.cost_breakdown_ms(
            feedback_regime=feedback_regime
        ).items():
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(fv):
                continue
            per_component.setdefault(component, []).append(fv)
    return {k: float(np.median(v)) for k, v in per_component.items() if v}


def generate_rollout_inference_cost_breakdown_plot(
    cfg: DictConfig,
    groups: dict[Any, Any],
    exp_dir_relative_path: Any,
    headless: bool,
    *,
    min_components: int = 2,
) -> None:
    """Render a stacked bar of the per-level inference cost breakdown (ms) per group.

    One stacked bar per group (model type); each stack segment is one component
    of :meth:`BenchmarkMetricSet.cost_breakdown_ms`, i.e. the honest
    per-step-paired decomposition of the control-loop budget (rev. 4, R5).

    Made PER REGIME by action ``RLRP-803-7.2`` of the RLRC test-time rollout
    deployer benchmarking `.junie` plan
    (``perf_RLRP-785_testtime_rollout_benchmarking_plan_20260903.md``).
    Permanent. One panel per feedback regime present in the artifact, so the
    figure lines up with the per-regime latency numbers reported elsewhere in
    the same artifact instead of implying a single regime.

    HONESTY RULE: the decomposition is built from ``level_deltas`` ONLY, and the
    caption always names the regime it reports:

      * one labelled regime  -> the current single-panel look, captioned with it;
      * two labelled regimes -> one panel per regime, side by side;
      * NO labelled regime   -> a schema v2 artifact, rendered as ONE panel
        captioned as explicitly regime-agnostic (its level deltas carry no
        ``feedback_regime``, so no regime can be claimed).

    Groups exposing fewer than ``min_components`` components are skipped in a
    panel: a single-segment bar is not a breakdown.

    :param cfg: Top-level multirun plot configuration.
    :param groups: Per-group metric dict from :func:`compute_per_group_metric`.
    :param exp_dir_relative_path: Where to save the figure.
    :param headless: Headless backend flag.
    :param min_components: Minimum number of measured components (levels)
        required to render a group's bar.
    """
    _selected_grp_names = (
        set(cfg.show.grp_names) if cfg.show.get("grp_names", None) else None
    )

    _regimes = _collect_cost_breakdown_regimes(groups)
    # An unlabelled (schema v2) artifact yields ONE regime-agnostic panel.
    _panel_regimes: list = list(_regimes) if _regimes else [None]

    panels: list[tuple] = []
    for _regime in _panel_regimes:
        labels: list[str] = []
        breakdowns: list[dict[str, float]] = []
        for grp_key, group in groups.items():
            _grp_display_name = group.get("grp_name", grp_key)
            if (
                _selected_grp_names is not None
                and _grp_display_name not in _selected_grp_names
            ):
                continue

            breakdown = _collect_cost_breakdown_ms(group, feedback_regime=_regime)
            if len(breakdown) < min_components:
                warnings.warn(
                    f"generate_rollout_inference_cost_breakdown_plot: group "
                    f"'{_grp_display_name}' exposes {len(breakdown)} cost component(s) "
                    f"(< {min_components}); skipping it in the breakdown plot."
                )
                continue
            labels.append(_grp_display_name)
            breakdowns.append(breakdown)
        if labels:
            panels.append((_regime, labels, breakdowns))

    if not panels:
        warnings.warn(
            "generate_rollout_inference_cost_breakdown_plot: no group has a "
            "cost breakdown (the 'deploy.benchmark' instrumentation writes the "
            "per-level BenchmarkMetricSet); skipping plot."
        )
        plt.close("all")
        return None

    with warnings.catch_warnings():
        manage_matplotlib_warnings()
        manage_matplotlib_backend(get_plot_render_type_cfg(cfg).show_plot, headless)

        plt.rcParams.update(get_plot_render_type_cfg(cfg).rcParams_update)

        _figsize = get_plot_figsize(
            cfg, "generate_rollout_inference_cost_breakdown_plot"
        )
        if len(panels) > 1:
            _figsize[0] = _figsize[0] * len(panels)
        fig: Figure = plt.figure(
            figsize=_figsize,
            dpi=get_plot_render_type_cfg(cfg).figdpi,
        )

        for _idx, (_regime, labels, breakdowns) in enumerate(panels):
            ax: Axes = fig.add_subplot(1, len(panels), _idx + 1)

            x = np.arange(len(labels))
            bottom = np.zeros(len(labels))
            for component in _COST_BREAKDOWN_COMPONENTS:
                values = np.array([b.get(component, 0.0) for b in breakdowns])
                if not np.any(values):
                    continue
                ax.bar(
                    x,
                    values,
                    bottom=bottom,
                    label=_COST_BREAKDOWN_COMPONENT_LABEL[component],
                )
                bottom = bottom + values

            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.grid(True, axis="y")
            if _idx == 0:
                ax.legend()

            if get_plot_render_type_cfg(cfg).print_title:
                if _regime is None:
                    _regime_caption = (
                        "regime-agnostic — this artifact's level deltas carry no "
                        "feedback regime (schema v2)"
                    )
                else:
                    _regime_caption = f"{_regime.value} regime"
                if len(panels) > 1:
                    ax.set_title(_regime_caption)
                else:
                    ax.set_title(
                        f"{cfg.env_name} — Control-loop cost breakdown per model type "
                        f"(median of per-trial level deltas; {_regime_caption}).",
                    )

            if get_plot_render_type_cfg(cfg).print_axis_label:
                ax.set_xlabel("Model type")
                if _idx == 0:
                    ax.set_ylabel("Cost breakdown [ms]")

        if len(panels) > 1 and get_plot_render_type_cfg(cfg).print_title:
            fig.suptitle(
                f"{cfg.env_name} — Control-loop cost breakdown per model type "
                f"(median of per-trial level deltas; one panel per feedback regime).",
            )

        if get_plot_render_type_cfg(cfg).print_meta_info:
            display_experiment_name(cfg, fig)

        show_and_save_plot_helper(
            fig,
            exp_dir_relative_path,
            "test_time_models_comparaison_rollout_inference_cost_breakdown_barplot",
            headless,
            get_plot_render_type_cfg(cfg).show_plot,
            get_plot_render_type_cfg(cfg).save_dpi,
        )

    plt.close("all")
    return None


def _generate_wall_clock_time_boxplot(
    cfg: DictConfig,
    groups: dict[Any, Any],
    exp_dir_relative_path: Any,
    headless: bool,
    *,
    key: str,
    metric_human_name: str,
    file_name: str,
    y_axis_in_logscale: bool = False,
    unit_label: str = "[s]",
    latex_plot_name: str | None = None,
    show_key: str | None = None,
) -> None:
    """Internal box-plot renderer shared by training & rollout wall-clock plots.

    Renders one box per group/model-type showing the per-trial wall-clock-time
    distribution; the mean (the "average" requested by the issue) is
    highlighted via ``showmeans=True``.

    When ``latex_plot_name`` is set and the render type is LaTeX
    ``includegraphics``, the (suppressed) on-figure title and experiment-name
    overlay are consolidated into a sidecar ``*.text`` meta file written under
    ``os.path.join(project_root, cfg.latex_includegraphics.save_path)`` with the
    name ``f"{cfg.latex_includegraphics.file_name}_{latex_plot_name}_meta.text"``
    (mirroring the drift-rate plot meta file).

    ``show_key`` is the ``cfg.show`` sub-group consulted by
    :func:`get_plot_figsize` for the per-graphic-type ``figsize`` overrides.
    """
    labels, samples = _collect_wall_clock_samples(
        cfg, groups, key, unit_label=unit_label
    )
    if not labels:
        warnings.warn(
            f"_generate_wall_clock_time_boxplot: no group has finite '{key}' "
            f"samples; skipping '{file_name}' plot."
        )
        plt.close("all")
        return None

    with warnings.catch_warnings():
        manage_matplotlib_warnings()
        manage_matplotlib_backend(get_plot_render_type_cfg(cfg).show_plot, headless)

        plt.rcParams.update(get_plot_render_type_cfg(cfg).rcParams_update)

        fig: Figure = plt.figure(
            figsize=get_plot_figsize(cfg, show_key),
            dpi=get_plot_render_type_cfg(cfg).figdpi,
        )
        ax: Axes = fig.add_subplot(111)

        ax.boxplot(
            samples,
            labels=labels,
            showmeans=True,
            meanline=False,
        )

        if y_axis_in_logscale:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)

        ax.grid(True, axis="y")

        # Build the title text unconditionally so it can be consolidated into
        # the LaTeX-``includegraphics`` sidecar meta file even when the
        # on-figure title is suppressed (``print_title`` False for that type).
        _boxplot_title_text = (
            f"{cfg.env_name} — {metric_human_name} per model type "
            f"(box: median/quartiles; ▲: mean)."
        )
        if get_plot_render_type_cfg(cfg).print_title:
            ax.set_title(_boxplot_title_text)

        if get_plot_render_type_cfg(cfg).print_axis_label:
            ax.set_xlabel("Model type")
            ax.set_ylabel(f"{metric_human_name} {unit_label}")

        if get_plot_render_type_cfg(cfg).print_meta_info:
            display_experiment_name(cfg, fig)

        show_and_save_plot_helper(
            fig,
            exp_dir_relative_path,
            file_name,
            headless,
            get_plot_render_type_cfg(cfg).show_plot,
            get_plot_render_type_cfg(cfg).save_dpi,
        )

        # Consolidate the (suppressed) on-figure title + experiment-name overlay
        # into a sidecar meta text file next to the LaTeX-includegraphics assets.
        if is_latex_includegraphics_render_type(cfg) and latex_plot_name is not None:
            project_root = fetch_r2s2r_project_root_path(cfg, lvl_up=1)
            _latex_save_dir = os.path.join(
                project_root, cfg.latex_includegraphics.save_path
            )
            _write_latex_includegraphics_meta_file(
                cfg,
                _latex_save_dir,
                f"{cfg.latex_includegraphics.file_name}_{latex_plot_name}_meta.text",
                title_text=_boxplot_title_text,
            )

    plt.close("all")
    return None


def _center_rotated_xticklabels(fig: Figure, labels: Sequence[Any]) -> None:
    """Shift each rotated x-tick ``label`` so its horizontal CENTRE sits on its tick (RLRP).

    The model ticks are placed at the exact centre of a bar, but the labels are drawn
    ``rotation=20, ha="right"`` -- which anchors the label's RIGHT edge on the tick, leaving the
    rotated text hanging to the LEFT of that centre. We render once to measure each label's on-screen
    width, then add a per-label ``ScaledTranslation`` that nudges it rightwards by HALF that width, so
    the label's own centre lands on the tick. Width is measured after a draw (in display pixels,
    converted to inches via the DPI) so the offset is correct for any label length and font size.
    Fails soft: if the canvas cannot be drawn (no renderer) the labels keep their right-anchored
    placement. Mirrors ``_center_rotated_xticklabels`` in ``plot_inference_benchmark``.
    """
    try:
        fig.canvas.draw()
    except Exception:
        return
    dpi = float(fig.dpi) or 1.0
    for label in labels:
        try:
            width_px = label.get_window_extent().width
        except Exception:
            continue
        # ``ScaledTranslation`` takes inches (scaled by ``dpi_scale_trans``); half the label width in
        # display pixels -> inches gives a rightwards shift that centres the bbox on the tick.
        offset = ScaledTranslation(
            (width_px / 2.0) / dpi, 0.0, fig.dpi_scale_trans
        )
        label.set_transform(label.get_transform() + offset)


def _generate_wall_clock_time_barplot(
    cfg: DictConfig,
    groups: dict[Any, Any],
    exp_dir_relative_path: Any,
    headless: bool,
    *,
    key: str,
    metric_human_name: str,
    file_name: str,
    y_axis_in_logscale: bool = False,
    unit_label: str = "[s]",
    latex_plot_name: str | None = None,
    show_key: str | None = None,
) -> None:
    """Internal bar-plot renderer for wall-clock plots.

    Renders one bar per group/model-type whose height is the per-trial
    wall-clock-time mean (the "average"); a symmetric error bar reports the
    per-trial standard deviation when a group has more than one finite sample.

    When ``latex_plot_name`` is set and the render type is LaTeX
    ``includegraphics``, the (suppressed) on-figure title and experiment-name
    overlay are consolidated into a sidecar ``*.text`` meta file written under
    ``os.path.join(project_root, cfg.latex_includegraphics.save_path)`` with the
    name ``f"{cfg.latex_includegraphics.file_name}_{latex_plot_name}_meta.text"``
    (mirroring the drift-rate plot meta file).

    ``show_key`` is the ``cfg.show`` sub-group consulted by
    :func:`get_plot_figsize` for the per-graphic-type ``figsize`` overrides.
    """
    labels, samples, _grp_records = _collect_wall_clock_samples(
        cfg, groups, key, unit_label=unit_label, return_records=True
    )
    if not labels:
        warnings.warn(
            f"_generate_wall_clock_time_barplot: no group has finite '{key}' "
            f"samples; skipping '{file_name}' plot."
        )
        plt.close("all")
        return None

    means = [float(np.mean(s)) for s in samples]
    # Per-group std (0.0 when a single sample) used as a symmetric error bar.
    errors = [float(np.std(s)) if len(s) > 1 else 0.0 for s in samples]

    # .... RLRP-818 experiment data consolidation ............................
    # Hook point: ``means`` / ``errors`` are the values rendered below and are
    # final here, outside the matplotlib context. Summary format: one row per
    # group (plan decision Q5).
    if key == "training_wall_clock_time" and is_consolidation_enabled(cfg):
        with consolidation_guard("training_wall_clock_time"):
            _unit = "h" if unit_label == "[h]" else "s"
            _group_rows = {}
            for _idx, _record in enumerate(_grp_records):
                _group_rows[str(_record["grp_name"])] = (
                    build_training_wall_clock_time_row(
                        grp_name=str(_record["grp_name"]),
                        gpr_short_name=_record.get("gpr_short_name", None),
                        samples=samples[_idx],
                        mean=means[_idx],
                        std=errors[_idx],
                        unit=_unit,
                    )
                )
            consolidate_training_wall_clock_time(
                cfg,
                _group_rows,
                file_stem=file_name,
                params={
                    "key": key,
                    "unit": _unit,
                    "y_axis_in_logscale": y_axis_in_logscale,
                    "n_samples_per_group": {
                        str(each["grp_name"]): len(samples[_i])
                        for _i, each in enumerate(_grp_records)
                    },
                    "n_recorded_per_group": {
                        str(each["grp_name"]): each.get("n_recorded", 0)
                        for each in _grp_records
                    },
                    "n_recovered_per_group": {
                        str(each["grp_name"]): each.get("n_recovered", 0)
                        for each in _grp_records
                    },
                    # RLRP-818 CR10 — one sample per TRAINED MODEL (trial /
                    # group seed), NOT one per (trial x trajectory) rollout.
                    "sample_scope": "per_trial",
                    "n_trials_per_group": {
                        str(each["grp_name"]): each.get("n_trials", None)
                        for each in _grp_records
                    },
                },
                source_experiments=_resolve_cfg_group_source_experiments(
                    cfg, [str(each["grp_name"]) for each in _grp_records]
                ),
            )

    with warnings.catch_warnings():
        manage_matplotlib_warnings()
        manage_matplotlib_backend(get_plot_render_type_cfg(cfg).show_plot, headless)

        plt.rcParams.update(get_plot_render_type_cfg(cfg).rcParams_update)

        # For the LaTeX ``includegraphics`` render type the wall-clock-time bar
        # plot DEFAULTS to HALF the configured ``figsize`` width (the height is
        # kept unchanged); other render types default to the configured width.
        # Both are overridable per graphic type via
        # ``show.<show_key>.width_latex`` / ``height_latex`` (resp. ``*_screen``).
        _figsize = get_plot_figsize(
            cfg,
            show_key,
            default_width_scale=(
                0.5 if is_latex_includegraphics_render_type(cfg) else 1.0
            ),
        )

        fig: Figure = plt.figure(
            figsize=_figsize,
            dpi=get_plot_render_type_cfg(cfg).figdpi,
        )
        ax: Axes = fig.add_subplot(111)

        x = np.arange(len(labels))
        _bars = ax.bar(
            x,
            means,
            # yerr=errors if any(e > 0.0 for e in errors) else None,
            capsize=4,
        )
        # Annotate the value on top of each bar (mirroring ``plot_inference_benchmark``).
        ax.bar_label(_bars, fmt="%.1f", padding=2)
        ax.set_xticks(x)
        # Rotate the model-type x-tick labels sideways (mirroring
        # ``plot_inference_benchmark``) so long labels stay readable and don't
        # overlap each other.
        _xticklabels = ax.set_xticklabels(labels, rotation=20, ha="right")
        # The x-tick sits at ``x`` -- the centre of each model's bar. The ``ha="right"`` rotation,
        # however, anchors each label's RIGHT edge there, so the rotated text visually hangs to the
        # LEFT of that centre. Offset every label rightwards by half its rendered width so the
        # label's own centre lands on the tick (mirroring ``plot_inference_benchmark``).
        _center_rotated_xticklabels(fig, _xticklabels)

        if y_axis_in_logscale:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)

        ax.grid(True, axis="y")

        # Build the title text unconditionally so it can be consolidated into
        # the LaTeX-``includegraphics`` sidecar meta file even when the
        # on-figure title is suppressed (``print_title`` False for that type).
        _barplot_title_text = (
            f"{cfg.env_name} — {metric_human_name} per model type "
            f"(bar: mean; error bar: std)."
        )
        if get_plot_render_type_cfg(cfg).print_title:
            ax.set_title(_barplot_title_text)

        if get_plot_render_type_cfg(cfg).print_axis_label:
            ax.set_xlabel("Model type")
            ax.set_ylabel(f"{metric_human_name} {unit_label}")

        if get_plot_render_type_cfg(cfg).print_meta_info:
            display_experiment_name(cfg, fig)

        show_and_save_plot_helper(
            fig,
            exp_dir_relative_path,
            file_name,
            headless,
            get_plot_render_type_cfg(cfg).show_plot,
            get_plot_render_type_cfg(cfg).save_dpi,
            save=not is_latex_includegraphics_render_type(cfg),
        )

        # Consolidate the (suppressed) on-figure title + experiment-name overlay
        # into a sidecar meta text file next to the LaTeX-includegraphics assets.
        if is_latex_includegraphics_render_type(cfg) and latex_plot_name is not None:
            project_root = fetch_r2s2r_project_root_path(cfg, lvl_up=1)
            _latex_save_dir = os.path.join(
                project_root, cfg.latex_includegraphics.save_path
            )

            # Save the figure into the LaTeX-includegraphics assets directory
            # (mirroring ``_generate_drift_rate_plot_single``).
            show_and_save_plot_helper(
                fig,
                _latex_save_dir,
                f"{cfg.latex_includegraphics.file_name}_{latex_plot_name}",
                headless,
                False,
                get_plot_render_type_cfg(cfg).latex_save_dpi,
                save=True,
            )

            _write_latex_includegraphics_meta_file(
                cfg,
                _latex_save_dir,
                f"{cfg.latex_includegraphics.file_name}_{latex_plot_name}_meta.text",
                title_text=_barplot_title_text,
            )

    plt.close("all")
    return None


def generate_training_wall_clock_time_plot(
    cfg: DictConfig,
    groups: dict[Any, Any],
    exp_dir_relative_path: Any,
    headless: bool,
    *,
    y_axis_in_logscale: bool = False,
) -> None:
    """Render a bar plot of per-trial *training* wall-clock time per group.

    One bar per group (model type) whose height is the mean per-trial training
    wall-clock time recorded in ``groups[gk]["training_wall_clock_time"]``, with
    a symmetric error bar reporting the per-trial standard deviation.

    :param cfg: Top-level multirun plot configuration.
    :param groups: Per-group metric dict from :func:`compute_per_group_metric`.
    :param exp_dir_relative_path: Where to save the figure.
    :param headless: Headless backend flag.
    :param y_axis_in_logscale: When ``True``, render the y-axis in log scale.
    """
    _generate_wall_clock_time_barplot(
        cfg,
        groups,
        exp_dir_relative_path,
        headless,
        key="training_wall_clock_time",
        metric_human_name="Training time",
        file_name="test_time_models_comparaison_training_wall_clock_time_barplot",
        y_axis_in_logscale=y_axis_in_logscale,
        unit_label="[h]",
        latex_plot_name="training_wall_clock_time",
        show_key="generate_training_wall_clock_time_plot",
    )


def generate_rollout_wall_clock_time_plot(
    cfg: DictConfig,
    groups: dict[Any, Any],
    exp_dir_relative_path: Any,
    headless: bool,
    *,
    y_axis_in_logscale: bool = False,
) -> None:
    """Render a box plot of per-trial *rollout* wall-clock time per group.

    One box per group (model type) summarising the per-trial rollout
    wall-clock time recorded in ``groups[gk]["rollout_wall_clock_time"]``.

    :param cfg: Top-level multirun plot configuration.
    :param groups: Per-group metric dict from :func:`compute_per_group_metric`.
    :param exp_dir_relative_path: Where to save the figure.
    :param headless: Headless backend flag.
    :param y_axis_in_logscale: When ``True``, render the y-axis in log scale.
    """
    _generate_wall_clock_time_boxplot(
        cfg,
        groups,
        exp_dir_relative_path,
        headless,
        key="rollout_wall_clock_time",
        metric_human_name="Rollout wall-clock time",
        file_name="test_time_models_comparaison_rollout_wall_clock_time_boxplot",
        y_axis_in_logscale=y_axis_in_logscale,
        show_key="generate_rollout_wall_clock_time_plot",
    )


# Pattern matching the training-time line written by the training pipeline
# into per-trial ``console.log`` files, e.g.::
#
#   "... model training done in 5627.86 seconds"
#
# Used as a fallback by :func:`_recover_training_wall_clock_time` when no
# previously saved TTRPM carries a non-zero ``training_wall_clock_time``
# (the case for TCN / GRU and any future baseline trained outside the
# rollout-regeneration path).
_TRAINING_DONE_LINE_RE = re.compile(
    r"model training done in\s+([0-9]+(?:\.[0-9]+)?)\s+seconds"
)


_INTRA_TRIAL_TOP_DIRS = ("testtime_rollouts", ".hydra", "console.log")


def _trial_dirname_from_path(experiment_base: str, inner_path: str) -> str | None:
    """Return the trial-directory name of ``inner_path`` relative to ``experiment_base``.

    Two on-disk layouts are supported:

    1. **Multirun-root layout** — ``experiment_base`` is the multirun
       directory containing one trial subdirectory per trial
       (e.g. ``.../165309/dynamics_model@…,trial_nb=1``). The trial
       dirname is the *first* path component below ``experiment_base``.
    2. **Per-trial layout** — ``experiment_base`` is itself a trial
       directory (V0B2 case: ``multirun_paths`` lists per-trial
       sub-paths, so each metric's resolved ``experiment_base`` ends
       at the trial level). The trial dirname is then
       ``os.path.basename(experiment_base)``. Detected when the first
       path component below ``experiment_base`` is an intra-trial
       artifact (``testtime_rollouts/``, ``.hydra/``, ``console.log``,
       ``TTRPM_*/``).

    Returns ``None`` when ``inner_path`` is not located under
    ``experiment_base``.
    """
    try:
        _rel = os.path.relpath(inner_path, experiment_base)
    except ValueError:
        return None
    if _rel.startswith(".."):
        return None
    _parts = _rel.split(os.sep)
    if not _parts or _parts[0] in (".", ""):
        return None
    _first = _parts[0]
    if _first in _INTRA_TRIAL_TOP_DIRS or _first.startswith("TTRPM_"):
        # Per-trial layout: ``experiment_base`` IS the trial dir.
        return os.path.basename(os.path.normpath(experiment_base))
    return _first


def _recover_training_wall_clock_times_per_trial(
    experiment_base: str,
) -> dict[str, float]:
    """Recover ``training_wall_clock_time`` *per trial* for an experiment.

    Walks the experiment tree and returns a mapping
    ``{trial_dirname: training_wall_clock_time_seconds}`` so that
    consumers can preserve per-trial granularity when falling back from
    stale TTRPM records (e.g. the TCN / GRU baselines whose past TTRPMs
    were saved with the ``0.0`` sentinel by the rollout-regeneration
    path).

    Sources, in order of precedence per trial:

    1. The first non-zero, finite ``training_wall_clock_time`` found in
       any ``TestTimeRolloutPredictionMetric.pkl`` reachable
       *recursively* under ``experiment_base/<trial>/testtime_rollouts/``
       or ``experiment_base/<trial>/TTRPM_*/``.
    2. The first float parsed out of a
       :data:`_TRAINING_DONE_LINE_RE`-matching line in
       ``experiment_base/<trial>/**/console.log``.

    Trials with no recoverable value are simply absent from the returned
    dict.

    :param experiment_base: Absolute path to the experiment directory
        (the multirun root containing one sub-directory per trial).
    :return: ``{trial_dirname: seconds}`` mapping; empty when nothing
        could be recovered.
    """
    per_trial: dict[str, float] = {}

    # --- 1) Recursive TTRPM-pickle lookup, grouped per trial ---
    _ttrpm_pkl_candidates = sorted(
        _glob_mod.glob(
            os.path.join(
                experiment_base,
                "testtime_rollouts",
                "**",
                "TestTimeRolloutPredictionMetric.pkl",
            ),
            recursive=True,
        )
        + _glob_mod.glob(
            os.path.join(
                experiment_base,
                "**",
                "testtime_rollouts",
                "**",
                "TestTimeRolloutPredictionMetric.pkl",
            ),
            recursive=True,
        )
        + _glob_mod.glob(
            os.path.join(
                experiment_base,
                "TTRPM_*",
                "**",
                "TestTimeRolloutPredictionMetric.pkl",
            ),
            recursive=True,
        )
        + _glob_mod.glob(
            os.path.join(
                experiment_base,
                "**",
                "TTRPM_*",
                "**",
                "TestTimeRolloutPredictionMetric.pkl",
            ),
            recursive=True,
        )
    )
    for _ttrpm_pkl in _ttrpm_pkl_candidates:
        _trial = _trial_dirname_from_path(experiment_base, _ttrpm_pkl)
        if _trial is None or _trial in per_trial:
            continue
        try:
            _ttrpm_dir = os.path.dirname(os.path.dirname(_ttrpm_pkl))
            _ttrpm_subdir = os.path.basename(os.path.dirname(_ttrpm_pkl))
            _metric = TestTimeRolloutPredictionMetric.load(
                os.path.join(_ttrpm_dir, _ttrpm_subdir)
            )
            _twc = getattr(_metric, "training_wall_clock_time", None)
            if _twc is None:
                continue
            _twc_arr = np.asarray(_twc, dtype=object).ravel()
            for v in _twc_arr:
                if v is None:
                    continue
                try:
                    _fv = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(_fv) and _fv != 0.0:
                    per_trial[_trial] = _fv
                    break
        except Exception:
            continue

    # --- 2) Fallback: per-trial ``console.log`` parsing ---
    _log_candidates = sorted(
        _glob_mod.glob(
            os.path.join(experiment_base, "**", "console.log"),
            recursive=True,
        )
    )
    for _log_path in _log_candidates:
        _trial = _trial_dirname_from_path(experiment_base, _log_path)
        if _trial is None or _trial in per_trial:
            continue
        try:
            with open(_log_path, "r", errors="ignore") as _fh:
                _txt = _fh.read()
        except OSError:
            continue
        _m = _TRAINING_DONE_LINE_RE.search(_txt)
        if _m is None:
            continue
        try:
            _fv = float(_m.group(1))
        except (TypeError, ValueError):
            continue
        if np.isfinite(_fv) and _fv != 0.0:
            per_trial[_trial] = _fv

    return per_trial


def _recover_training_wall_clock_time(experiment_base: str) -> float:
    """Recover ``training_wall_clock_time`` for an experiment (scalar).

    Scalar fallback used by the rollout-regeneration path
    (:func:`regenerate_rollouts`). Preserves the original
    TTRPM-precedence semantics:

    1. Returns the first non-zero, finite ``training_wall_clock_time``
       found in any TTRPM pickle reachable recursively under
       ``experiment_base``.
    2. Otherwise, returns the first float parsed out of a
       :data:`_TRAINING_DONE_LINE_RE`-matching line in any
       ``console.log`` reachable recursively under ``experiment_base``.
    3. Otherwise, returns ``0.0`` (sentinel).

    For per-trial granularity (used by the plot path) see
    :func:`_recover_training_wall_clock_times_per_trial`.

    :param experiment_base: Absolute path to the experiment directory.
    :return: Recovered training wall-clock time in seconds, or ``0.0``.
    """
    # --- 1) Recursive TTRPM-pickle lookup (precedence) ---
    _ttrpm_pkl_candidates = sorted(
        _glob_mod.glob(
            os.path.join(
                experiment_base,
                "testtime_rollouts",
                "**",
                "TestTimeRolloutPredictionMetric.pkl",
            ),
            recursive=True,
        )
        + _glob_mod.glob(
            os.path.join(
                experiment_base,
                "TTRPM_*",
                "**",
                "TestTimeRolloutPredictionMetric.pkl",
            ),
            recursive=True,
        )
    )
    for _ttrpm_pkl in _ttrpm_pkl_candidates:
        try:
            _ttrpm_dir = os.path.dirname(os.path.dirname(_ttrpm_pkl))
            _ttrpm_subdir = os.path.basename(os.path.dirname(_ttrpm_pkl))
            _metric = TestTimeRolloutPredictionMetric.load(
                os.path.join(_ttrpm_dir, _ttrpm_subdir)
            )
            _twc = getattr(_metric, "training_wall_clock_time", None)
            if _twc is None:
                continue
            _twc_arr = np.asarray(_twc, dtype=object).ravel()
            for v in _twc_arr:
                if v is None:
                    continue
                try:
                    _fv = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(_fv) and _fv != 0.0:
                    return _fv
        except Exception:
            continue

    # --- 2) Fallback: parse ``console.log`` files ---
    _log_candidates = sorted(
        _glob_mod.glob(
            os.path.join(experiment_base, "**", "console.log"),
            recursive=True,
        )
    )
    for _log_path in _log_candidates:
        try:
            with open(_log_path, "r", errors="ignore") as _fh:
                _txt = _fh.read()
        except OSError:
            continue
        _m = _TRAINING_DONE_LINE_RE.search(_txt)
        if _m is None:
            continue
        try:
            _fv = float(_m.group(1))
        except (TypeError, ValueError):
            continue
        if np.isfinite(_fv) and _fv != 0.0:
            return _fv

    return 0.0


def regenerate_rollouts(
    cfg: DictConfig,
    root_project_path: str | bytes,
    do_cleanup_legacy_rollouts: bool,
    torch_rng,
) -> None:
    """
    Regenerates test-time rollouts for all experiments based on the given configuration.

    This function processes experimental configurations and regenerates test-time rollouts
    for machine learning models. It resolves test trajectories, loads experiment-specific
    configurations, initializes models, and executes rollouts using the configured settings.

    :param cfg: Configuration object containing details about experimental groups, paths,
        and related settings.
    :param root_project_path: The root path of the project directory where experiment-related
        directories and data are stored.
    :param do_cleanup_legacy_rollouts: Boolean flag to indicate whether legacy rollouts should
        be cleaned up before regenerating new ones.
    :param torch_rng: PyTorch random number generator, used for reproducibility during
        test-time rollouts.
    :return: None
    """
    import glob
    import shutil

    import torch

    consol_msg_universal_one_liner(
        "regenerate_rollouts=true: re-running test-time rollouts for all experiments\n"
    )

    # Deferred imports to avoid circular dependency at module level
    from pipeline.pipeline_utils.general.setup import (
        get_model_description,
        setup_multistep_step_model,
    )
    from pipeline.pipeline_utils.general.train_and_deploy_utils import (
        execute_ms_model_test_time_rollouts,
        execute_ms_model_test_time_rollouts_over_epoch_checkpoints,
    )
    from tools.mbrl_lib_tools import persistent_checkpoint_utils

    # Collect unique experiment_base paths across ALL groups so that each experiment
    # (and its "Build TestTrajectoryEntry lists" step) is processed exactly once, regardless
    # of how many groups reference it.
    #
    # Scoping via ``cfg.show.grp_names``: when the user provides a non-empty
    # ``cfg.show.grp_names`` list, only groups whose ``grp_name`` appears in
    # that list are regenerated. This matches the downstream plotting filter
    # (see line ~932) so that ``regenerate_rollouts`` and the plot stage
    # operate on the same group subset. An empty / missing ``grp_names``
    # preserves the previous behaviour (regenerate all groups).
    selected_grp_names = list(
        omegaconf.OmegaConf.select(cfg, "show.grp_names", default=None) or []
    )
    if selected_grp_names:
        consol_msg_universal_one_liner(
            f"  Filtering regeneration by cfg.show.grp_names = "
            f"{selected_grp_names}\n"
        )

    experiment_bases: list[str] = []
    _seen_experiment_bases: set[str] = set()
    for each_grp in omegaconf.OmegaConf.to_container(cfg.groups):
        if "experiments" not in each_grp or each_grp["experiments"] is None:
            continue
        if selected_grp_names and each_grp.get("grp_name") not in selected_grp_names:
            continue
        for each_experiments in each_grp["experiments"]:
            multirun_paths = each_experiments.get("multirun_paths") or ["."]
            for each_multirun_path in multirun_paths:
                experiment_base = os.path.realpath(
                    os.path.join(
                        root_project_path,
                        str(each_experiments["experiment_path"]),
                        str(each_multirun_path),
                    )
                )
                if experiment_base in _seen_experiment_bases:
                    continue
                _seen_experiment_bases.add(experiment_base)
                experiment_bases.append(experiment_base)

    # --- Build TestTrajectoryEntry lists ONCE, shared across all groups/experiments ---
    # The goal of the multirun test-time rollout plot pipeline is to compare test-time
    # rollouts from models trained on the SAME data; therefore the simulator-side
    # ground-truth trajectory entries are identical for every model group. We build
    # them lazily, on the first experiment whose saved Hydra config is available, and
    # then reuse the resulting entries for every subsequent experiment.
    target_InD_entries = None
    target_OOD_entries = None

    for experiment_base in experiment_bases:
        consol_msg_universal_one_liner(
            f"Regenerating rollouts for: {experiment_base}\n"
        )

        # --- Load the experiment's saved Hydra config (guard first — BUG-P4-01 fix) ---
        experiment_hydra_cfg_path = os.path.join(
            experiment_base, ".hydra", "config.yaml"
        )
        if not os.path.isfile(experiment_hydra_cfg_path):
            consol_msg_universal_one_liner(
                f"  WARNING: skipping — saved Hydra config not found at {experiment_hydra_cfg_path}\n"
            )
            continue
        experiment_cfg = omegaconf.OmegaConf.load(experiment_hydra_cfg_path)
        # Refresh stale data-source keys (e.g. renamed dataset folder) from the
        # top-level ``simulator_config`` so ``setup_trajectory_from_csv`` resolves
        # CSVs against the *current* on-disk layout, not the path captured at
        # training time.
        experiment_cfg = apply_simulator_config_override(cfg, experiment_cfg)

        # --- Override device for cross-platform portability (Task 8.5) ---
        current_device = _resolve_current_device()
        with omegaconf.read_write(experiment_cfg):
            omegaconf.OmegaConf.update(
                experiment_cfg, "device", current_device, merge=False
            )

        # --- Locate the model directory ---
        model_dirs = [
            d
            for d in os.listdir(experiment_base)
            if d.startswith("model_")
            and os.path.isdir(os.path.join(experiment_base, d))
        ]
        if not model_dirs:
            consol_msg_universal_one_liner(
                f"  WARNING: skipping — no model_* directory found in {experiment_base}\n"
            )
            continue
        load_pretrained_path = os.path.join(experiment_base, model_dirs[0])
        consol_msg_universal_one_liner(
            f"  Loading model from: {load_pretrained_path}\n"
        )

        # --- Load the trained model ---
        try:
            motion_model_container = setup_multistep_step_model(
                experiment_cfg, load_pretrained_path, disable_compile=True
            )
        except Exception as e:
            consol_msg_universal_one_liner(f"  ERROR: failed to load model — {e}\n")
            continue

        # --- Build TestTrajectoryEntry lists ONCE (shared across all experiments) ---
        if target_InD_entries is None or target_OOD_entries is None:
            ind_entries, ood_entries = _resolve_test_trajectories_split(
                cfg, experiment_base
            )
            total_traj = len(ind_entries) + len(ood_entries)
            consol_msg_universal_one_liner(
                f"  Resolved {total_traj} test trajectories "
                f"({len(ind_entries)} InD, {len(ood_entries)} OOD)\n"
            )
            try:
                target_InD_entries = _build_entries(
                    ind_entries, experiment_cfg, target_is_ood=False
                )
                target_OOD_entries = _build_entries(
                    ood_entries, experiment_cfg, target_is_ood=True
                )
            except Exception as e:
                # The most common cause when this fires on a previously-working
                # experiment dir is a dataset rename: ``experiment_cfg.environment.data_path``
                # in the saved ``.hydra/config.yaml`` points at the old location.
                # Point the user at the override mechanism that fixes it.
                _data_path = omegaconf.OmegaConf.select(
                    experiment_cfg, "environment.data_path"
                )
                _sim_cfg = cfg.get("simulator_config", None)
                consol_msg_universal_one_liner(
                    f"  ERROR: failed to build trajectory entries — {e}\n"
                    f"         experiment data_path was: {_data_path!r}\n"
                    f"         top-level simulator_config: {_sim_cfg!r}\n"
                    f"         Hint: set 'simulator_config: <name>' in the plot "
                    f"pipeline config (or update the existing one) so that "
                    f"data_path / dataset_name / data are refreshed from "
                    f"<repo>/src/launcher/configs/simulator/<name>.yaml.\n"
                )
                target_InD_entries = None
                target_OOD_entries = None
                continue

        # --- Determine model description and state space label ---
        for cfg_model_key in ["ms_model", "ss_model", "model"]:
            if omegaconf.OmegaConf.select(experiment_cfg, cfg_model_key) is not None:
                break
        ms_model_description = get_model_description(
            getattr(experiment_cfg, cfg_model_key),
            motion_model_container,
        )

        # (CRITICAL) ToDo: validate (ref task RLRP-591)
        state_space_label = experiment_cfg.environment.data.label
        # state_space_label = experiment_cfg.environment.get('data', 'math_fct').label

        # --- Recover training_wall_clock_time ---
        # Re-generating rollouts must preserve train-time data gathered in
        # ``TestTimeRolloutPredictionMetric.training_wall_clock_time``. The
        # recovery helper tries every saved TTRPM pickle (recursively) and,
        # as a fallback for experiments where past TTRPMs were saved with
        # ``0.0`` (e.g. TCN / GRU baselines), parses ``console.log`` for the
        # ``"… model training done in <seconds> seconds"`` line written by
        # the training pipeline.
        preserved_training_time = _recover_training_wall_clock_time(experiment_base)

        # --- Wipe stale ``testtime_rollouts/`` BEFORE regenerating ---
        # Otherwise, if the test-trajectory configuration changes between
        # runs (e.g. ``r=28.000009`` removed / renamed), stale per-trajectory
        # subdirectories from a previous configuration would remain on disk
        # alongside the freshly-written ones, leading to inflated /
        # inconsistent rollout counts at plotting time (see issue: 7 dirs
        # instead of the expected 4).
        # IMPORTANT: this runs AFTER ``preserved_training_time`` has already
        # been recovered from any pre-existing TTRPM pickle, so we don't
        # lose the train-time data carried by the previous rollout records.
        testtime_rollouts_dir = os.path.join(experiment_base, "testtime_rollouts")
        if os.path.isdir(testtime_rollouts_dir):
            shutil.rmtree(testtime_rollouts_dir)
            consol_msg_universal_one_liner(
                f"  Cleared stale testtime_rollouts/ at: {testtime_rollouts_dir}\n"
            )

        # --- Persist rollout-relevant cfg overrides to .hydra/config.yaml ---
        # The freshly-written ``testtime_rollouts/`` directories will reflect
        # the *current* simulator config (e.g. updated ``test_InD_trajectory``
        # / OoD noise). Keep the on-disk Hydra snapshot in sync for the
        # whitelisted rollout-relevant keys ONLY — training-time keys
        # (``trajectories``, ``global_measurement_noise``,
        # ``measurement_noise``) are deliberately not touched.
        try:
            persist_rollout_relevant_cfg_overrides(
                experiment_cfg, experiment_hydra_cfg_path
            )
        except Exception as _e:
            consol_msg_universal_one_liner(
                f"  WARNING: failed to persist rollout-relevant cfg overrides "
                f"to {experiment_hydra_cfg_path}: {_e}\n"
            )

        # --- Execute test-time rollouts ---
        consol_msg_universal_one_liner(
            f"  Running rollouts: {len(target_InD_entries)} InD + {len(target_OOD_entries)} OOD trajectories\n"
        )
        try:
            execute_ms_model_test_time_rollouts(
                cfg=experiment_cfg,
                motion_model_container=motion_model_container,
                ms_model_description=ms_model_description,
                state_space_label=state_space_label,
                target_InD_rollouts=target_InD_entries,
                target_OOD_rollouts=target_OOD_entries,
                training_time=preserved_training_time,
                ms_tensorboard_writer=None,
                exp_dir_relative_path=".",
                headless=True,
                torch_rng=torch_rng,
                save_base_dir=experiment_base,
            )
        except Exception as e:
            consol_msg_universal_one_liner(f"  ERROR: rollout execution failed — {e}\n")
            continue

        consol_msg_universal_one_liner(
            f"  Regeneration complete for {experiment_base}\n"
        )

        # --- RLRP-773 (R6/R14/G1.1): regenerate per-epoch-checkpoint rollouts (opt-in) ---
        # Gated by the plot-pipeline cfg `regenerate.epoch_checkpoint_rollouts` (NOT the
        # experiment's `deploy` block). Iterates the `epoch_checkpoints/` tree under
        # `experiment_base` and writes per-epoch rollout dirs, giving the per-epoch tree its OWN
        # scoped wipe first (symmetric with the `testtime_rollouts/` wipe above) so a narrower
        # `epoch_stride`/`epochs` from a previous run leaves no orphan epoch dirs for the plot.
        _epoch_ckpt_regen_cfg = omegaconf.OmegaConf.select(
            cfg, "regenerate.epoch_checkpoint_rollouts", default=None
        )
        if _epoch_ckpt_regen_cfg is not None and bool(
            omegaconf.OmegaConf.select(_epoch_ckpt_regen_cfg, "enable", default=False)
        ):
            epoch_rollouts_dir = (
                persistent_checkpoint_utils.epoch_checkpoint_rollouts_root(
                    experiment_base
                )
            )
            if os.path.isdir(epoch_rollouts_dir):
                shutil.rmtree(epoch_rollouts_dir)
                consol_msg_universal_one_liner(
                    f"  Cleared stale epoch_checkpoints_rollouts/ at: {epoch_rollouts_dir}\n"
                )
            try:
                execute_ms_model_test_time_rollouts_over_epoch_checkpoints(
                    cfg=experiment_cfg,
                    motion_model_container=motion_model_container,
                    ms_model_description=ms_model_description,
                    state_space_label=state_space_label,
                    target_InD_rollouts=target_InD_entries,
                    target_OOD_rollouts=target_OOD_entries,
                    training_time=preserved_training_time,
                    ms_tensorboard_writer=None,
                    exp_dir_relative_path=".",
                    headless=True,
                    torch_rng=torch_rng,
                    save_base_dir=experiment_base,
                    checkpoints_source_dir=experiment_base,
                    epoch_stride=omegaconf.OmegaConf.select(
                        _epoch_ckpt_regen_cfg, "epoch_stride", default=None
                    ),
                    epochs=omegaconf.OmegaConf.select(
                        _epoch_ckpt_regen_cfg, "epochs", default=None
                    ),
                )
            except Exception as e:
                consol_msg_universal_one_liner(
                    f"  ERROR: per-epoch-checkpoint rollout regeneration failed — {e}\n"
                )

        # Free model memory before loading next experiment
        del motion_model_container
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Cleanup old flat-layout artifacts AFTER successful regeneration
        if do_cleanup_legacy_rollouts:
            for old_ttrpm in glob.glob(os.path.join(experiment_base, "TTRPM_*")):
                if os.path.isdir(old_ttrpm):
                    shutil.rmtree(old_ttrpm)
                    consol_msg_universal_one_liner(
                        f"  Removed legacy dir: {old_ttrpm}\n"
                    )
            # Remove old rollout PNG files at experiment root
            for old_png in glob.glob(os.path.join(experiment_base, "*_rollout.png")):
                os.remove(old_png)
                consol_msg_universal_one_liner(f"  Removed legacy PNG: {old_png}\n")
            # NOTE: legacy math-env partition cleanup
            # (``testtime_rollouts/{InD,OOD}/``) is no longer needed here —
            # the whole ``testtime_rollouts/`` directory is now wiped before
            # regeneration above.

    return None


def _build_gt_cum_length_map(ground_truth_entries: Any) -> dict[str, np.ndarray]:
    """Build the ``{trajectory_name: GT cumulative trajectory length [m]}`` map.

    Shared drift-rate compute core (RLRP-773 W9): materializes one cumulative-
    length array per ground-truth entry so every per-rollout curve can be paired
    with the matching ground-truth. The GT trajectory length pseudocode is::

        GT-trj-length[t] = cumsum(L2Norm(GT.linear_velocity) * dt)[t]

    Extracted verbatim from :func:`generate_drift_rate_plot` so both that
    function and :func:`generate_drift_rate_plot_per_train_epoch` share a single
    implementation (no logic duplicated).

    :param ground_truth_entries: List of ``TestTrajectoryEntry`` produced by
        ``_load_ground_truth_entries``.
    :return: A dict keyed by trajectory name (full path, basename, and the
        math/Lorenz ``.`` -> ``p`` spelling) mapping to the cumulative
        path-length array.
    """
    gt_cum_length_map: dict[str, np.ndarray] = {}
    for entry in ground_truth_entries:
        env = getattr(entry, "env", None)
        tname = getattr(entry, "trajectory_name", None)
        if env is None or tname is None:
            continue
        try:
            # Linear velocity: first 3 observation features (cf.
            # ``QuadcopterRobotic3D.on_begin_post_init_callback`` /
            # ``simulator/quadcopter_general.yaml``).
            lin_vel = np.asarray(env.observations[:, :3], dtype=float)
            # ``env.timestamps`` is a TCT ``TimestampsDataclass`` on robotic
            # envs (exposing ``.delta_stamps``) but on the math/Lorenz envs
            # it is stored as a plain ``numpy.ndarray`` of timestamps. Be
            # tolerant to both: prefer ``.delta_stamps`` when present,
            # otherwise fall back to ``np.diff`` on the raw timestamps
            # array (which is equivalent by construction).
            _ts = env.timestamps
            _delta = getattr(_ts, "delta_stamps", None)
            if _delta is None:
                _ts_arr = np.asarray(_ts, dtype=float).reshape(-1)
                if _ts_arr.size < 2:
                    raise ValueError(
                        "timestamps array too short to derive delta_stamps "
                        f"(size={_ts_arr.size})"
                    )
                # Prepend 0.0 so the resulting ``dt`` has the same length as
                # ``timestamps`` (matching ``.delta_stamps`` convention).
                dt = np.concatenate(([0.0], np.diff(_ts_arr)))
            else:
                dt = np.asarray(_delta, dtype=float)

            # Skip timestep t=0 since delta timestamp is undefined and mae start at t=1
            lin_vel = lin_vel[1:,]
            dt = dt[1:,]

        except Exception as _exc:
            warnings.warn(
                f"generate_drift_rate_plot: cannot read linear_velocity / "
                f"timestamps.delta_stamps for trajectory {tname!r} ({_exc}); "
                f"skipping this trajectory.",
                stacklevel=2,
            )
            continue

        # Align dt and lin_vel lengths (defensive — they should already match).
        n = min(lin_vel.shape[0], dt.shape[0])
        norm_type = 1  # ==> L1
        speed = np.linalg.norm(lin_vel[:n], ord=norm_type, axis=-1)
        cumul_path_len = np.cumsum(speed * dt[:n])

        # Index the GT cumulative length by *both* the full
        # ``trajectory_name`` (which on robotic envs is a relative path like
        # ``"adverse_test_set/ellipse_01_..."``) and its basename (which is
        # what TTRPM directories — and therefore ``metric._trajectory_name``
        # on the rollout side — are named after). Without this, pairing
        # silently fails and the plot is skipped.
        gt_cum_length_map[tname] = cumul_path_len
        _basename = os.path.basename(str(tname))
        if _basename and _basename != tname:
            gt_cum_length_map.setdefault(_basename, cumul_path_len)
        # Math/Lorenz envs name rollout dirs by substituting ``.`` -> ``p``
        # in numeric coordinates (e.g. GT name ``coord_0.9_0_0__b2.777..``
        # becomes rollout dir ``coord_0p9_0_0__b2p777..``). Index that
        # spelling too so :func:`_match_trajectory_name` below resolves.
        _dot2p = _basename.replace(".", "p") if _basename else ""
        if _dot2p and _dot2p != _basename:
            gt_cum_length_map.setdefault(_dot2p, cumul_path_len)

    return gt_cum_length_map


def select_ground_truth_entries(
    ground_truth_entries: Any,
    selected_trajectories: Sequence[str] | None,
    *,
    target_is_ood: bool,
    simulator_config_name: str | None,
    plot_name: str,
) -> Any:
    """Restrict ground-truth entries to ``show.selected_trajectories`` (RLRP-841).

    The drift-rate plots pair every rollout with its ground truth through
    :func:`_build_gt_cum_length_map` and silently skip rollouts with no GT
    match. Restricting the entry list is therefore the single choke point
    that limits the pooled rollouts, the horizon, the aggregation, the band,
    the legend ``trj`` count and the consolidated CSVs to the selected
    trajectories on every internal path.

    Names are compared **verbatim** against ``entry.trajectory_name``, i.e.
    the ``trajectory_name`` spelling of the simulator config
    (``data.test_InD_trajectory`` / ``data.test_OOD_trajectory``), which is
    what ``_load_ground_truth_entries`` produces. The simulator-config order
    of the retained entries is preserved.

    :param ground_truth_entries: List of ``TestTrajectoryEntry`` produced by
        ``_load_ground_truth_entries`` (or ``None``).
    :param selected_trajectories: The resolved ``show.selected_trajectories``
        list, or ``None`` to disable the selection.
    :param target_is_ood: Which simulator-config test list the entries come
        from (``InD`` vs ``OoD``); used in the error message only.
    :param simulator_config_name: The ``simulator_config`` cfg key value,
        used in the error message only.
    :param plot_name: Caller name for the error message.
    :return: ``ground_truth_entries`` unchanged when either argument is
        ``None``, else the filtered list.
    :raises ValueError: (fail fast) if any selected name is not an available
        ``trajectory_name``. The message lists the offending names, the
        target, the simulator config and every available name.
    """
    if selected_trajectories is None or ground_truth_entries is None:
        return ground_truth_entries

    selected = list(selected_trajectories)
    available = [
        getattr(entry, "trajectory_name", None) for entry in ground_truth_entries
    ]
    available = [name for name in available if name is not None]
    unknown = [name for name in selected if name not in available]
    if unknown:
        target = "OoD" if target_is_ood else "InD"
        sim_key = "data.test_OOD_trajectory" if target_is_ood else "data.test_InD_trajectory"
        raise ValueError(
            f"{plot_name}: `show.selected_trajectories` contains {len(unknown)} "
            f"trajectory name(s) not found in the {target} test set of "
            f"simulator config {simulator_config_name!r} ({sim_key}): "
            f"{unknown!r}. Names must match the simulator-config "
            f"`trajectory_name` spelling verbatim. Available names: {available!r}."
        )

    selected_set = set(selected)
    return [
        entry
        for entry in ground_truth_entries
        if getattr(entry, "trajectory_name", None) in selected_set
    ]


def _unmatched_selected_trajectories(
    selected_trajectories: Sequence[str] | None,
    matched_rollout_names: Any,
) -> list[str]:
    """Return the selected trajectories that matched NO pooled rollout (RLRP-841).

    Rollout-side names are TTRPM directory basenames (with the math/Lorenz
    ``.`` -> ``p`` spelling), so a selected full name counts as matched when
    either its verbatim form, its basename or its ``.`` -> ``p`` basename
    appears in ``matched_rollout_names`` (mirrors the dual indexing of
    :func:`_build_gt_cum_length_map`).
    """
    if not selected_trajectories:
        return []
    matched = {str(name) for name in matched_rollout_names}
    unmatched: list[str] = []
    for each_name in selected_trajectories:
        _basename = os.path.basename(str(each_name))
        candidates = {str(each_name), _basename, _basename.replace(".", "p")}
        if not (candidates & matched):
            unmatched.append(str(each_name))
    return unmatched


def _selection_note(selected_trajectories: Sequence[str] | None) -> str:
    """Title fragment ``selected trajectories=N, `` for an active selection (RLRP-841)."""
    if not selected_trajectories:
        return ""
    return f"selected trajectories={len(list(selected_trajectories))}, "


def _compute_logscale_band(
    stacked_curves: np.ndarray,  # shape (n_curves, horizon), strictly-positive drift curves
    *,
    method: str,  # "percentile" | "log_normal"
    dispersion_type: str,  # "std" | "std-err" | "mad"
    n_norm: int,  # N for std-err (n_trials / seeds), mirrors the linear path
    log_normal_center: str = "mean",  # "mean" -> geometric mean | "median"
    lower_pct: float = 16.0,
    upper_pct: float = 84.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a statistically sound band for a LOG y-axis (RLRP-807).

    On a log y-axis the linear ``avg +/- dispersion`` band flares downward and
    can violate the strictly-positive support of a cumulative drift metric. This
    helper returns ``(center, lower, upper)`` -- all strictly positive -- using
    one of two approaches selectable by ``method``:

    - ``"percentile"``: median center with an empirical ``[lower_pct, upper_pct]``
      coverage band (default 16th/84th ~= 1-sigma). Robust to a single diverging
      seed and invariant under ``log``; ``dispersion_type`` is ignored by design
      (a percentile band is a coverage interval, not a dispersion multiple).
    - ``"log_normal"``: center from the log-space average (``log_normal_center``:
      ``"mean"`` -> geometric mean, or ``"median"`` -> ordinary median, since
      ``exp(median(log y)) == median(y)``) with a symmetric (on the log axis)
      band from the log-space dispersion. ``dispersion_type`` selects the spread:
      ``"std"`` (log-space standard deviation), ``"std-err"`` (``std`` /
      ``sqrt(n_norm)``) or ``"mad"`` (robust log-space median-absolute-deviation,
      scaled by ``1.4826`` to be sigma-consistent for log-normal data). ``"mad"``
      pairs naturally with the median center to fully tame a single diverging
      seed. The band is anchored on the SAME log-space center so the plotted line
      stays visually centered in its band (RLRP-807, option 1).

    :param stacked_curves: ``(n_curves, horizon)`` strictly-positive drift curves.
    :param method: ``"percentile"`` or ``"log_normal"``.
    :param dispersion_type: ``"std"``, ``"std-err"`` or ``"mad"`` (used by
        ``log_normal``). ``"mad"`` gives a robust, outlier-insensitive spread.
    :param n_norm: ``N`` for the ``std-err`` normalisation (nb of models / seeds).
    :param log_normal_center: ``"mean"`` (geometric mean) or ``"median"`` central
        tendency for the ``log_normal`` center AND band anchor. Robust to a
        single diverging seed when ``"median"``.
    :param lower_pct: lower percentile for the ``percentile`` band.
    :param upper_pct: upper percentile for the ``percentile`` band.
    :param eps: small floor applied before ``log`` to avoid ``log(0)``.
    :return: ``(center, lower, upper)`` arrays of shape ``(horizon,)``.
    """
    if method == "percentile":
        center = np.median(stacked_curves, axis=0)
        lower = np.percentile(stacked_curves, lower_pct, axis=0)
        upper = np.percentile(stacked_curves, upper_pct, axis=0)
    elif method == "log_normal":
        log_c = np.log(np.maximum(stacked_curves, eps))
        if log_normal_center == "median":
            log_center = np.median(log_c, axis=0)
        else:
            log_center = np.mean(log_c, axis=0)
        if dispersion_type == "mad":
            # Robust log-space spread: median absolute deviation about the
            # log-space median, scaled by 1.4826 so it estimates the same
            # sigma as ``std`` for log-normal data but is insensitive to a
            # single diverging seed (RLRP-807).
            _log_med = np.median(log_c, axis=0)
            log_disp = 1.4826 * np.median(np.abs(log_c - _log_med), axis=0)
        elif dispersion_type == "std-err":
            log_disp = np.std(log_c, axis=0) / np.sqrt(n_norm)
        else:
            log_disp = np.std(log_c, axis=0)
        center = np.exp(log_center)
        lower = np.exp(log_center - log_disp)
        upper = np.exp(log_center + log_disp)
    else:
        raise NotImplementedError(
            f"logscale_band_method '{method}' not supported. "
            "Options: 'linear', 'percentile', 'log_normal'."
        )
    return center, lower, upper


def _resolve_percentile_band_bounds(cfg: DictConfig) -> tuple[float, float]:
    """Resolve the (lower, upper) percentile bounds for the ``percentile`` band.

    Selected via ``show.generate_drift_rate_plot.percentile_band_coverage``
    (RLRP-807):

    - ``"sigma"`` (default) -> ``[16th, 84th]`` (~68% coverage, 1-sigma
      equivalent), the original behaviour.
    - ``"iqr"`` -> ``[25th, 75th]`` interquartile range: a **narrower** band that
      overlaps less between groups while still conveying relative seed spread
      (wider IQR = less stable), improving readability of overlapping bands.
    """
    _coverage = str(
        cfg.show.generate_drift_rate_plot.get("percentile_band_coverage", "sigma")
    ).lower()
    if _coverage == "iqr":
        return 25.0, 75.0
    if _coverage == "sigma":
        return 16.0, 84.0
    raise NotImplementedError(
        f"percentile_band_coverage '{_coverage}' not supported. "
        "Options: 'sigma' (16th/84th), 'iqr' (25th/75th)."
    )


def _resolve_logscale_band_desc(
    cfg: DictConfig,
    *,
    y_axis_in_logscale: bool,
    dispersion_type: str,
) -> str:
    """Build the title band-description line, reflecting the resolved method.

    Mirrors the compute-side switch so the reader knows *how* the band was
    computed (empirical coverage vs. parametric dispersion). Reverts to the
    legacy ``+/-1*<dispersion_type>`` wording for ``linear`` / a linear y-axis.
    """
    _method = cfg.show.generate_drift_rate_plot.get("logscale_band_method", "linear")
    if y_axis_in_logscale and _method == "percentile":
        _lower_pct, _upper_pct = _resolve_percentile_band_bounds(cfg)
        return (
            f"Band: [{_lower_pct:g}th, {_upper_pct:g}th] percentile "
            "(median center) across models and trajectories."
        )
    if y_axis_in_logscale and _method == "log_normal":
        # Center follows the existing ``mae_grp_reduction`` key: ``median`` gives
        # the robust median center (band recentered on it), otherwise the
        # geometric mean (RLRP-807, option 1).
        _mae_grp_reduction = cfg.show.generate_drift_rate_plot.get(
            "mae_grp_reduction", "mean"
        )
        _center_desc = (
            "geometric median" if _mae_grp_reduction == "median" else "geometric mean"
        )
        # ``mad`` -> robust log-space spread; render it upper-case for clarity.
        _disp_desc = "MAD" if dispersion_type == "mad" else dispersion_type
        return (
            f"Band: {_center_desc} \u00b11\u00b7{_disp_desc} "
            "(log-space) across models and trajectories."
        )

    _mae_grp_reduction = cfg.show.generate_drift_rate_plot.get(
        "mae_grp_reduction", "mean"
    )
    _center_desc = "median" if _mae_grp_reduction == "median" else "mean"
    return f"Band: {_center_desc} \u00b11\u00b7{dispersion_type} across models and trajectories."


def _compute_single_rollout_drift(
    reduced_mae: np.ndarray,
    gt_cum_len_arr: np.ndarray,
    timesteps: np.ndarray,
    max_len: int,
    *,
    drift_type: str,
    warmup_mode: Any,
    warmup_steps: int | None,
    cfg: DictConfig,
    y_axis_in_logscale: bool,
) -> np.ndarray:
    """Compute a single per-rollout C-MAE drift-rate curve of length ``max_len``.

    Shared drift-rate compute core (RLRP-773 W9): extracted verbatim from the
    inner loop of :func:`generate_drift_rate_plot` so both that function and
    :param y_axis_in_logscale:
    :func:`generate_drift_rate_plot_per_train_epoch` divide the cumulative MAE
    by the same (timesteps / path-length) reference — nothing is duplicated. The
    returned curve is UNSCALED (the caller applies the ``drift_type`` scaling
    and the synthetic ``t=0`` prepend).

    :param reduced_mae: Feature-reduced per-timestep MAE (1D), length ``>= max_len``.
    :param gt_cum_len_arr: The matching GT cumulative path-length array.
    :param timesteps: 1-based timestep index array ``[1 .. max_len]``.
    :param max_len: Aggregation horizon.
    :param drift_type: ``"timesteps"`` or ``"path_length"``.
    :param warmup_mode: Ground-truth-feed warm-up treatment mode.
    :param warmup_steps: Auto-derived warm-up step count (or ``None``).
    :param cfg: Top-level multirun plot configuration (read for
        ``show.y_axis_in_logscale_min_value``).
    :return: The unscaled drift-rate curve (1D, length ``max_len``).
    """
    reduced_mae_t = reduced_mae[:max_len]
    gt_cum_len_t = gt_cum_len_arr[:max_len]

    # ...V1 and 2.....
    # cmae = np.cumsum(reduced_mae_t)
    # with np.errstate(divide="ignore", invalid="ignore"):
    #     # drift = cmae / (timesteps * gt_cum_len_t)
    #     drift = cmae / (len(timesteps) * gt_cum_len_t)

    # ...V3...........
    # Note: `reduced_mae_t` are already zeroed up to timestep `warmup_steps` at this point.
    pred_error = np.cumsum(reduced_mae_t)
    # pred_error = reduced_mae_t

    # RLRP-736-ORep: in ``discard`` mode the cumulative error is
    # reset to zero at the warm-up boundary ``t = warmup_steps``
    # (see ``_reset_cumulative_error_at_warmup``). The drift is an
    # *average up to t* (``C-MAE / t``), so its divisor must be
    # rebased to the elapsed time **since the boundary**
    # (``t - warmup_steps``) — otherwise the freshly-reset
    # numerator is divided by the full timestep index (which still
    # counts the discarded warm-up interval), artificially
    # suppressing the first post-warm-up samples and producing a
    # non-smooth rise. The path-length reference is rebased the same
    # way (subtracting the cumulative length accrued during warm-up)
    # so the ``path_length`` drift also restarts cleanly at zero.
    # TS_MINUS_WS_ENABLE = True # Note: It break the discard mode of generate drift plot per epoch ⚠️
    TS_MINUS_WS_ENABLE = False  # This is the way
    if (
        warmup_mode == "discard"
        and warmup_steps is not None
        and int(warmup_steps) > 0
        and TS_MINUS_WS_ENABLE
    ):
        ws = int(warmup_steps) - 1
        effective_timesteps = timesteps - ws

        ref_idx = min(ws - 1, gt_cum_len_t.shape[0] - 1)
        effective_gt_cum_len = gt_cum_len_t - gt_cum_len_t[ref_idx]
    else:
        effective_timesteps = timesteps
        effective_gt_cum_len = gt_cum_len_t

    with np.errstate(divide="ignore", invalid="ignore"):
        if drift_type == "timesteps":
            drift = pred_error / effective_timesteps  # <-- V3
        elif drift_type == "path_length":
            # drift = pred_error / (len(timesteps) * gt_cum_len_t) # <-- working but not legit
            # drift = pred_error / gt_cum_len_t
            drift = pred_error / (
                effective_timesteps * effective_gt_cum_len
            )  # <-- LA BONNE FACON
            # drift = 100.0 * drift
        else:
            raise ValueError(f"Unknown drift_type: {drift_type}")

    if y_axis_in_logscale:
        # Within the reset warm-up interval (and at the boundary point itself) the
        # numerator is exactly zero while the rebased divisor is zero/negative, which would
        # yield ``nan``/``inf``. The average drift there is zero by construction, so clamp
        # it explicitly — this keeps the boundary point from near ``0`` so the curve look
        # like its risingf smoothly from zero (RLRP-736-ORep).
        lowest_two_values = np.unique(drift)[:2]
        zero_proxy = lowest_two_values.mean()
        if cfg.show.get("y_axis_in_logscale_min_value", None) is not None:
            zero_proxy = cfg.show.y_axis_in_logscale_min_value
        # zero_proxy = 1e-3 # Manual override
        drift = np.where(drift == 0, zero_proxy, drift)

    return drift


def _make_array_finite(array: np.ndarray) -> np.ndarray:
    """Replace non-finite (``nan``/``inf``) array values with finite proxies.

    ``np.nanmean``/``np.nanstd`` ignore ``nan`` but NOT ``inf`` (and ``inf``
    triggers the "invalid value encountered in subtract" RuntimeWarning). Rather
    than propagating those non-finite values into the aggregate, clamp them to a
    finite, in-range proxy so the curve/scalar stays meaningful:

    * ``nan`` -> ``0.0`` (no measurable contribution),
    * ``+inf`` -> the maximum finite value in the same curve (or ``0.0`` when the
      whole curve is non-finite),
    * ``-inf`` -> the minimum finite value in the same curve (or ``0.0``).

    :param array: An array possibly containing ``nan``/``inf``.
    :return: A copy of ``array`` with every element finite. When the input is
        already all-finite the same array is returned unchanged.
    """
    array = np.asarray(array, dtype=float)
    finite_mask = np.isfinite(array)
    if finite_mask.all():
        return array
    finite_vals = array[finite_mask]

    nan_fill = 0.0
    pos_fill = float(np.max(finite_vals)) if finite_vals.size else 0.0
    neg_fill = float(np.min(finite_vals)) if finite_vals.size else 0.0
    # pos_fill = None
    # neg_fill = None

    return np.nan_to_num(array, nan=nan_fill, posinf=pos_fill, neginf=neg_fill)


def _add_horizon_annotation(
    ax: Axes,
    max_len: int,
    warmup_steps: int | None,
    cfg: DictConfig,
    train_epoch: int | None = None,
    free_running_in_bold: bool = False,
) -> None:
    """Draw a single boxed annotation with horizon, warm-up and free-running.

    The box stacks three lines (RLRP): the aggregation horizon and the warm-up
    length are rendered as secondary labels (large, gray, non-bold), while the
    ``free-running = horizon - warm-up`` line is the dominant label (large,
    bold, black) so it stands out. The warm-up and free-running lines are
    omitted when ``warmup_steps is None``.

    When ``train_epoch`` is not ``None`` (only :func:`generate_drift_rate_plot`
    passes it) an additional secondary ``train epoch = <E>`` line is stacked at
    the top of the box so the reader can tell which training-epoch snapshot the
    curve was rendered from.

    :param cfg:
    :param ax: Target axes to annotate (upper-right corner).
    :param max_len: Resolved aggregation horizon (effective ``length_cutoff``).
    :param warmup_steps: Warm-up step count, or ``None`` when unresolved.
    :param train_epoch: Training-epoch number of the plotted snapshot, or
        ``None`` to omit the line (per-model / legacy behaviour).
    """
    _secondary = dict(fontsize="large", fontweight="normal", color="gray")
    _standout = dict(fontsize="large", fontweight="bold", color="black")
    children = []
    if train_epoch is not None:
        children.append(
            TextArea(
                f"train epoch = {int(train_epoch)}",
                textprops=dict(_standout),
            )
        )
    if warmup_steps is not None and cfg.show.get("compounded_predictions_score", False):
        children.append(
            TextArea(
                f"GT-feed = {int(warmup_steps)}",
                textprops=dict(_secondary),
            )
        )
    if cfg.show.get("compounded_predictions_score", False):
        children.append(
            TextArea(
                f"free-running = {int(max_len) - int(warmup_steps if warmup_steps else 0)}",
                textprops=dict(_standout if free_running_in_bold else _secondary),
            )
        )
    children.append(
        TextArea(
            f"horizon = {max_len}",
            textprops=dict(
                _standout
                if not cfg.show.get("compounded_predictions_score", True)
                else _secondary
            ),
        )
    )
    box = VPacker(children=children, align="right", pad=0.0, sep=2.0)
    anchored = AnchoredOffsetbox(
        loc="upper left",
        child=box,
        pad=0.4,
        borderpad=0.4,
        frameon=True,
        bbox_to_anchor=(0.0, 1.0),
        bbox_transform=ax.transAxes,
    )
    anchored.patch.set(
        # boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="black",
        alpha=0.85,
    )
    ax.add_artist(anchored)


def _build_single_epoch_groups(
    cfg: DictConfig,
    groups: dict[Any, Any],
    ground_truth_entries: Any,
    root_project_path: str | bytes,
    train_epoch: int,
) -> dict[Any, Any]:
    """Build a ``groups``-like dict backed by ONE training-epoch's rollouts.

    Reuses the same per-epoch discovery machinery as
    :func:`generate_drift_rate_plot_per_train_epoch`
    (:func:`_resolve_group_epoch_rollout_sources` +
    :func:`_load_epoch_rollout_reduced_maes`) but, instead of reducing every
    epoch to a single scalar, it injects that epoch's per-rollout MAE arrays
    into a deep copy of ``groups`` (replacing each group's
    ``_original_metrics``) so the per-TIMESTEP core
    :func:`_generate_drift_rate_plot_single` can render the SAME curve it draws
    for the deployed model, just sourced from the ``epoch_<E>`` snapshot.

    :param cfg: Top-level multirun plot configuration.
    :param groups: Per-group metric dict (used for metadata + group selection).
    :param ground_truth_entries: GT entries used to pair rollouts by name.
    :param root_project_path: Project root used to resolve experiment trees.
    :param train_epoch: The training epoch ``E`` whose ``epoch_<E>`` rollouts
        are loaded.
    :return: A deep copy of ``groups`` whose ``_original_metrics`` hold that
        epoch's rollouts; groups with no epoch-``E`` rollout are dropped. Empty
        dict when nothing resolves for the requested epoch.
    """
    from copy import deepcopy
    from types import SimpleNamespace

    from tools.mbrl_lib_tools import persistent_checkpoint_utils

    gt_cum_length_map = _build_gt_cum_length_map(ground_truth_entries)
    if not gt_cum_length_map:
        return {}

    group_sources = _resolve_group_epoch_rollout_sources(cfg, groups, root_project_path)
    if not group_sources:
        return {}

    epoch_groups: dict[Any, Any] = {}
    for grp_key, src in group_sources.items():
        ttrpm_dir = src["ttrpm_dir"]
        per_rollout: list[tuple[str, Any, np.ndarray]] = []
        for experiment_base in src["experiment_bases"]:
            # Epoch rollout dirs are zero-padded (``epoch_000100``); resolve the
            # path via the shared helper so it matches what discovery finds.
            epoch_dir = persistent_checkpoint_utils.epoch_checkpoint_rollouts_dir(
                experiment_base, int(train_epoch)
            )
            if not os.path.isdir(epoch_dir):
                continue
            # RLRP-819 `W3`: ONE experiment base IS one model seed (trial), so
            # it is the canonical ``_trial_key`` of every rollout it yields --
            # without it the per-seed `drift_rate_samples` export of an epoch
            # checkpoint would be skipped by its fail-soft guard.
            per_rollout.extend(
                (tname, (str(experiment_base), f"epoch_{int(train_epoch)}"), reduced_mae)
                for tname, reduced_mae in _load_epoch_rollout_reduced_maes(
                    cfg, epoch_dir, ttrpm_dir, gt_cum_length_map
                )
            )
        if not per_rollout:
            continue

        # Wrap each ``(tname, trial_key, reduced_mae)`` into a lightweight
        # object exposing the ``_trajectory_name`` / ``_trial_key`` / ``mae``
        # attributes the per-timestep core reads from ``_original_metrics``
        # (the reduced 1D MAE is left as-is, the core only re-reduces when
        # ``ndim > 1``).
        _orig_metrics = [
            SimpleNamespace(
                _trajectory_name=tname, _trial_key=trial_key, mae=reduced_mae
            )
            for tname, trial_key, reduced_mae in per_rollout
        ]
        _epoch_group = deepcopy(groups[grp_key])
        _epoch_group["_original_metrics"] = _orig_metrics
        epoch_groups[grp_key] = _epoch_group

    return epoch_groups


def _discover_all_train_epochs(
    cfg: DictConfig,
    groups: dict[Any, Any],
    root_project_path: str | bytes,
) -> list[int]:
    """Discover every training epoch with a rollout tree across all groups.

    Scans each plotted group's ``epoch_checkpoints_rollouts/`` root (resolved by
    :func:`_resolve_group_epoch_rollout_sources`) for ``epoch_<E>`` directories,
    exactly like :func:`generate_drift_rate_plot_per_train_epoch`, and returns
    the sorted union of the discovered epoch numbers ``E``.

    :param cfg: Top-level multirun plot configuration.
    :param groups: Per-group metric dict (used for group selection / metadata).
    :param root_project_path: Project root used to resolve experiment trees.
    :return: Sorted list of unique training epochs; empty when none found.
    """
    from tools.mbrl_lib_tools import persistent_checkpoint_utils

    group_sources = _resolve_group_epoch_rollout_sources(cfg, groups, root_project_path)
    if not group_sources:
        return []

    _epoch_dir_re = re.compile(r"^epoch_(\d+)$")
    _epochs: set[int] = set()
    for src in group_sources.values():
        for experiment_base in src["experiment_bases"]:
            rollouts_root = persistent_checkpoint_utils.epoch_checkpoint_rollouts_root(
                experiment_base
            )
            if not os.path.isdir(rollouts_root):
                continue
            for name in os.listdir(rollouts_root):
                match = _epoch_dir_re.match(name)
                if match is not None:
                    _epochs.add(int(match.group(1)))
    return sorted(_epochs)


def _assemble_png_diaporama(
    png_paths: list[str],
    output_path: str,
    fps: float = 1.0,
) -> str | None:
    """Stitch an ORDERED list of PNG files into a slideshow video via ``ffmpeg``.

    Each frame is shown for ``1 / fps`` seconds (``fps=1.0`` -> one image per
    second, as requested). The frames are fed to ``ffmpeg`` through the
    ``concat`` demuxer (an explicit, order-preserving file list) so the produced
    video honours the exact order of ``png_paths`` regardless of file-name
    alpha-sorting. The output is re-emitted at 30 fps (duplicating frames) so
    low-frame-rate players (VLC/QuickTime) play it reliably while keeping the
    visible ``1 image / second`` cadence.

    :param png_paths: Ordered list of ABSOLUTE (or cwd-relative) PNG file paths.
        Missing files are skipped with a console note.
    :param output_path: Destination video path (e.g. ``.../foo.mp4``).
    :param fps: Frames per second of the slideshow (each image shown for
        ``1 / fps`` seconds); defaults to ``1.0``.
    :return: The written ``output_path`` on success; ``None`` when skipped
        (``ffmpeg`` missing, no usable frame, or ``ffmpeg`` failure).
    """
    import shutil
    import subprocess
    import tempfile

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        consol_msg_universal_one_liner(
            "generate diaporama: `ffmpeg` not found on PATH; skipping video "
            "assembly.\n"
        )
        return None

    _existing = [p for p in png_paths if p and os.path.isfile(p)]
    if not _existing:
        consol_msg_universal_one_liner(
            "generate diaporama: no PNG frame found on disk; skipping video "
            "assembly.\n"
        )
        return None

    if fps is None or float(fps) <= 0.0:
        fps = 1.0
    frame_duration = 1.0 / float(fps)

    os.makedirs(os.path.dirname(os.path.realpath(output_path)) or ".", exist_ok=True)

    # Build an ffmpeg ``concat`` list. Every frame carries an explicit
    # ``duration`` so each image is shown for exactly ``1 / fps`` seconds
    # (verified: the last frame IS rendered when its duration is explicit, so no
    # trailing-frame repetition is needed).
    _list_fd, _list_path = tempfile.mkstemp(prefix="diaporama_", suffix=".txt")
    try:
        with os.fdopen(_list_fd, "w", encoding="utf-8") as _list_file:
            for _p in _existing:
                _abs = os.path.realpath(_p).replace("'", "'\\''")
                _list_file.write(f"file '{_abs}'\n")
                _list_file.write(f"duration {frame_duration}\n")

        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            _list_path,
            # Pad odd dimensions so the yuv420p pixel format is valid.
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-r",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            output_path,
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except Exception as exc:  # pragma: no cover - defensive
            consol_msg_universal_one_liner(
                f"generate diaporama: ffmpeg invocation failed ({exc}); " "skipping.\n"
            )
            return None

        if proc.returncode != 0:
            _err_tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
            _err_tail = "\n".join(_err_tail[-5:]) if _err_tail else "(no stderr)"
            consol_msg_universal_one_liner(
                "generate diaporama: ffmpeg exited with code "
                f"{proc.returncode}; skipping. Last stderr lines:\n{_err_tail}\n"
            )
            return None

        consol_msg_universal_one_liner(
            f"generate diaporama: wrote {len(_existing)}-frame video at "
            f"{float(fps)} fps -> {output_path}\n"
        )
        return output_path
    finally:
        try:
            os.remove(_list_path)
        except OSError:
            pass


def _consolidate_epoch_checkpoint_drift_rates(
    cfg: DictConfig,
    groups: dict[Any, Any],
    ground_truth_entries: Any,
    title: str,
    exp_dir_relative_path: Any,
    fill_between_alpha: float,
    plot_linewidth: float,
    headless: bool,
    *,
    root_project_path: str | bytes | None,
    rendered_epochs: Sequence[int] = (),
    length_cutoff: int | None = None,
    trials_label: str = "trials",
    y_axis_in_logscale: bool = False,
    x_axis_in_logscale: bool = False,
    drift_type: str = "timesteps",
    warmup_mode: Any = None,
    warmup_steps: int | None = None,
    dispersion_type: str = "std-err",
    selected_trajectories: Sequence[str] | None = None,
) -> None:
    """Consolidate the drift rate of every NON-rendered epoch checkpoint.

    RLRP-818 scope extension: the drift-rate consolidation covers the main
    (deployed-model) test-time rollout AND every ``epoch_<E>`` checkpoint
    rollout found on disk, UNCONDITIONALLY -- i.e. even when
    ``show.generate_drift_rate_plot.train_epochs`` renders a single figure (or
    none of the epochs). Only the RENDERING follows ``train_epochs``; the
    consolidation is exhaustive.

    ⚠️ Collection boundary: the epoch discovery
    (:func:`_discover_all_train_epochs`) and the per-epoch group construction
    (:func:`_build_single_epoch_groups`) both go through
    :func:`_resolve_group_epoch_rollout_sources`, which filters on
    ``cfg.show.grp_names``. A group declared in ``cfg.groups`` but NOT enabled
    in ``show.grp_names`` is therefore never collected -- not even its epoch
    checkpoints. Enabling a group is an explicit user intent.

    No-op (with a console notice) when the consolidation is disabled or when
    no epoch-checkpoint rollout tree exists.

    :param cfg: Top-level multirun plot configuration.
    :param groups: Per-group metric dict from :func:`compute_per_group_metric`.
    :param ground_truth_entries: GT entries used to pair rollouts by name.
    :param title: Base title string (forwarded, unused in export-only mode).
    :param exp_dir_relative_path: Figure directory (unused in export-only mode).
    :param fill_between_alpha: Forwarded to the worker.
    :param plot_linewidth: Forwarded to the worker.
    :param headless: Forwarded to the worker.
    :param root_project_path: Project root used to resolve experiment trees;
        REQUIRED, the sweep is skipped (with a notice) when ``None``.
    :param rendered_epochs: The epochs already exported by the rendering path;
        subtracted from the discovered set to avoid doing the work twice.
    :param length_cutoff: Forwarded to the worker.
    :param trials_label: Forwarded to the worker.
    :param y_axis_in_logscale: Forwarded to the worker.
    :param x_axis_in_logscale: Forwarded to the worker.
    :param drift_type: Forwarded to the worker.
    :param warmup_mode: Forwarded to the worker.
    :param warmup_steps: Forwarded to the worker.
    :param dispersion_type: Forwarded to the worker.
    :param selected_trajectories: Forwarded to the worker (RLRP-841
        annotation/metadata only; ``ground_truth_entries`` is expected to be
        already restricted by the caller).
    :return: ``None``.
    """
    if not is_consolidation_enabled(cfg):
        return None

    if root_project_path is None:
        consol_msg_universal_one_liner(
            "generate_drift_rate_plot: `consolidate_data.enable` is on but "
            "`root_project_path` is None; the epoch-checkpoint consolidation "
            "is skipped.\n"
        )
        return None

    _already_exported = {int(each) for each in rendered_epochs}
    _epochs = _discover_all_train_epochs(cfg, groups, root_project_path)
    if not _epochs:
        consol_msg_universal_one_liner(
            "generate_drift_rate_plot: data consolidation found no "
            "'epoch_<E>' rollout snapshot; only the deployed-model (main) "
            "test-time rollout was consolidated.\n"
        )
        return None

    _pending = [each for each in _epochs if int(each) not in _already_exported]
    if not _pending:
        return None

    consol_msg_universal_one_liner(
        "generate_drift_rate_plot: data consolidation of "
        f"{len(_pending)} non-rendered epoch checkpoint(s): {_pending}\n"
    )

    for each_epoch in _pending:
        with consolidation_guard(f"drift_rate: epoch {each_epoch}"):
            _epoch_groups = _build_single_epoch_groups(
                cfg, groups, ground_truth_entries, root_project_path, each_epoch
            )
            if not _epoch_groups:
                continue
            _generate_drift_rate_plot_single(
                cfg,
                _epoch_groups,
                ground_truth_entries,
                title,
                exp_dir_relative_path,
                fill_between_alpha,
                plot_linewidth,
                headless,
                length_cutoff=length_cutoff,
                trials_label=trials_label,
                y_axis_in_logscale=y_axis_in_logscale,
                x_axis_in_logscale=x_axis_in_logscale,
                drift_type=drift_type,
                warmup_mode=warmup_mode,
                warmup_steps=warmup_steps,
                dispersion_type=dispersion_type,
                filename_prefix=f"{int(each_epoch)}_epoch_",
                train_epoch=each_epoch,
                export_only=True,
                selected_trajectories=selected_trajectories,
            )
    return None


def generate_drift_rate_plot(
    cfg: DictConfig,
    groups: dict[Any, Any],
    ground_truth_entries: Any,
    title: str,
    exp_dir_relative_path: Any,
    fill_between_alpha: float,
    plot_linewidth: float,
    headless: bool,
    *,
    root_project_path: str | bytes | None = None,
    train_epochs: int | list[int] | str | None = None,
    length_cutoff: int | None = None,
    trials_label: str = "trials",
    y_axis_in_logscale: bool = False,
    x_axis_in_logscale: bool = False,
    drift_type: str = "timesteps",
    warmup_mode: Any = None,
    warmup_steps: int | None = None,
    dispersion_type: str = "std-err",  # Options: std, std-err
    generate_diaporama: bool = False,
    diaporama_fps: float = 1.0,
    selected_trajectories: Sequence[str] | None = None,
) -> None:
    """Render one or MANY C-MAE drift-rate plots (all groups overlaid).

    Thin dispatcher over :func:`_generate_drift_rate_plot_single` that honours a
    ``train_epochs`` given either as a single ``int`` / ``None`` or as a
    ``list[int]`` of TRAINING EPOCHS. For each requested training epoch a
    SEPARATE figure is rendered from that ``epoch_<E>`` snapshot's rollouts
    (loaded with the same machinery as
    :func:`generate_drift_rate_plot_per_train_epoch`); the saved file name is
    prefixed with ``<train_epoch>_epoch_`` (e.g.
    ``500_epoch_test_time_models_comparaison_drift_rate.png``) so multiple
    epochs do not overwrite one another, and the training-epoch number is added
    to the in-axes horizon annotation.

    When ``train_epochs is None`` a single figure is rendered from the deployed
    (final) model rollouts carried by ``groups`` (legacy behaviour, no prefix,
    no train-epoch annotation).

    :param root_project_path: Project root used to resolve each group's
        ``experiment_base`` directories; REQUIRED when ``train_epochs`` selects
        one or more epochs (ignored for the legacy ``None`` case).
    :param train_epochs: ``None`` (legacy final-model plot), a single
        strictly-positive ``int``, a ``list[int]`` of training epochs to
        iterate over, or the string ``"all"`` to auto-discover and iterate over
        EVERY ``epoch_<E>`` rollout snapshot found across the plotted groups.
    :param length_cutoff: ``None`` (auto shortest-trajectory horizon) or a
        single strictly-positive ``int`` horizon. All other parameters are
        forwarded verbatim to :func:`_generate_drift_rate_plot_single`.
    :param generate_diaporama: When ``True`` AND a training-epoch SERIES is
        rendered, stitch the produced per-epoch PNG frames into a slideshow
        video (``..._diaporama.mp4``) via ``ffmpeg``. Ignored for the legacy
        single (``train_epochs is None``) figure.
    :param diaporama_fps: Frames-per-second of the slideshow video (each frame
        shown for ``1 / fps`` seconds); defaults to ``1.0`` (1 image/second).
    :param selected_trajectories: Resolved ``show.selected_trajectories``
        (RLRP-841). ``None`` keeps every test trajectory; a list restricts
        EVERY path below (deployed figure, ``train_epochs`` series, RLRP-818
        consolidation) to those trajectories by filtering the ground-truth
        entries up front. Unknown names fail fast with ``ValueError``.
    """
    # RLRP-841: restrict the ground-truth entries ONCE so every internal path
    # (which pairs rollouts through ``gt_cum_length_map``) only pools the
    # selected trajectories. Fail fast on any unknown name.
    ground_truth_entries = select_ground_truth_entries(
        ground_truth_entries,
        selected_trajectories,
        target_is_ood=bool(cfg.show.get("target_is_ood", False)),
        simulator_config_name=cfg.get("simulator_config", None),
        plot_name="generate_drift_rate_plot",
    )
    if selected_trajectories is not None:
        consol_msg_universal_one_liner(
            "generate_drift_rate_plot: `show.selected_trajectories` restricts the "
            f"drift rate to {len(list(selected_trajectories))} trajectory(ies): "
            f"{list(selected_trajectories)}\n"
        )

    # ``train_epochs is None`` -> legacy single figure from the deployed model
    # rollouts (no epoch injection, no prefix, no train-epoch annotation).
    if train_epochs is None:
        _generate_drift_rate_plot_single(
            cfg,
            groups,
            ground_truth_entries,
            title,
            exp_dir_relative_path,
            fill_between_alpha,
            plot_linewidth,
            headless,
            length_cutoff=length_cutoff,
            trials_label=trials_label,
            y_axis_in_logscale=y_axis_in_logscale,
            x_axis_in_logscale=x_axis_in_logscale,
            drift_type=drift_type,
            warmup_mode=warmup_mode,
            warmup_steps=warmup_steps,
            dispersion_type=dispersion_type,
            filename_prefix="",
            train_epoch=None,
            selected_trajectories=selected_trajectories,
        )
        # RLRP-818: the consolidation is EXHAUSTIVE -- sweep every
        # epoch-checkpoint rollout even though a single figure was rendered.
        _consolidate_epoch_checkpoint_drift_rates(
            cfg,
            groups,
            ground_truth_entries,
            title,
            exp_dir_relative_path,
            fill_between_alpha,
            plot_linewidth,
            headless,
            root_project_path=root_project_path,
            rendered_epochs=(),
            length_cutoff=length_cutoff,
            trials_label=trials_label,
            y_axis_in_logscale=y_axis_in_logscale,
            x_axis_in_logscale=x_axis_in_logscale,
            drift_type=drift_type,
            warmup_mode=warmup_mode,
            warmup_steps=warmup_steps,
            dispersion_type=dispersion_type,
            selected_trajectories=selected_trajectories,
        )
        return None

    else:

        if root_project_path is None:
            raise ValueError(
                "generate_drift_rate_plot: `root_project_path` is required when "
                "`train_epochs` selects one or more training epochs."
            )

        # Normalise ``train_epochs`` into an ordered list of training epochs.
        # The ``"all"`` sentinel auto-discovers every ``epoch_<E>`` snapshot.
        if isinstance(train_epochs, str) and train_epochs.strip().lower() == "all":
            _epochs = _discover_all_train_epochs(cfg, groups, root_project_path)
            if not _epochs:
                consol_msg_universal_one_liner(
                    "generate_drift_rate_plot: `train_epochs: all` found no "
                    "'epoch_<E>' rollout snapshot; skipping.\n"
                )
                return None
        elif isinstance(train_epochs, (list, tuple)) or (
            # OmegaConf list proxies are not ``list`` instances.
            hasattr(train_epochs, "__iter__")
            and not isinstance(train_epochs, (str, bytes))
        ):
            _epochs = [int(e) for e in train_epochs]
        else:
            _epochs = [int(train_epochs)]

        if not _epochs:
            return None

        # Collect the produced (ordered) PNG frames so an optional 1-fps slideshow
        # video can be assembled once the whole training-epoch series is rendered.
        _diaporama_frames: list[str] = []

        for _epoch in _epochs:
            # Source that epoch's rollouts into a ``groups``-like structure.
            _epoch_groups = _build_single_epoch_groups(
                cfg, groups, ground_truth_entries, root_project_path, _epoch
            )
            if not _epoch_groups:
                consol_msg_universal_one_liner(
                    "generate_drift_rate_plot: no rollout found for training "
                    f"epoch={_epoch}; skipping.\n"
                )
                continue
            # Each training epoch gets a ``<train_epoch>_epoch_`` file-name prefix so
            # a series of epochs produces distinct files.
            _prefix = f"{int(_epoch)}_epoch_"
            _generate_drift_rate_plot_single(
                cfg,
                _epoch_groups,
                ground_truth_entries,
                title,
                exp_dir_relative_path,
                fill_between_alpha,
                plot_linewidth,
                headless,
                length_cutoff=length_cutoff,
                trials_label=trials_label,
                y_axis_in_logscale=y_axis_in_logscale,
                x_axis_in_logscale=x_axis_in_logscale,
                drift_type=drift_type,
                warmup_mode=warmup_mode,
                warmup_steps=warmup_steps,
                dispersion_type=dispersion_type,
                filename_prefix=_prefix,
                train_epoch=_epoch,
                diaporama_mode=generate_diaporama,
                selected_trajectories=selected_trajectories,
            )
            # Reconstruct the standard (non-LaTeX) PNG path saved by
            # :func:`show_and_save_plot_helper` (matplotlib appends ``.png``).
            _diaporama_frames.append(
                os.path.join(
                    str(exp_dir_relative_path),
                    f"{_prefix}test_time_models_comparaison_drift_rate.png",
                )
            )

        # RLRP-818: consolidate the epoch checkpoints that were NOT rendered
        # (the rendered ones already exported from within the worker).
        _consolidate_epoch_checkpoint_drift_rates(
            cfg,
            groups,
            ground_truth_entries,
            title,
            exp_dir_relative_path,
            fill_between_alpha,
            plot_linewidth,
            headless,
            root_project_path=root_project_path,
            rendered_epochs=_epochs,
            length_cutoff=length_cutoff,
            trials_label=trials_label,
            y_axis_in_logscale=y_axis_in_logscale,
            x_axis_in_logscale=x_axis_in_logscale,
            drift_type=drift_type,
            warmup_mode=warmup_mode,
            warmup_steps=warmup_steps,
            dispersion_type=dispersion_type,
            selected_trajectories=selected_trajectories,
        )

        # Optionally stitch the rendered per-epoch frames into a slideshow video.
        if generate_diaporama and _diaporama_frames:
            _assemble_png_diaporama(
                _diaporama_frames,
                os.path.join(
                    str(exp_dir_relative_path),
                    "test_time_models_comparaison_drift_rate_diaporama.mp4",
                ),
                fps=diaporama_fps,
            )
        return None


def _resolve_cfg_group_source_experiments(
    cfg: DictConfig, grp_names: Any
) -> dict[str, Any]:
    """Collect the configured source experiment paths of the given groups.

    Provenance for the RLRP-818 metric-level ``meta.txt``: per group, the
    ``ttrpm_dir`` plus every ``experiment_path`` / ``multirun_paths`` entry
    declared in ``cfg.groups``.

    :param cfg: Top-level multirun plot configuration.
    :param grp_names: The raw group names to describe.
    :return: ``{grp_name: {"ttrpm_dir": ..., "experiments": [...]}}``; a group
        absent from ``cfg.groups`` is silently skipped.
    """
    try:
        _cfg_groups = omegaconf.OmegaConf.to_container(cfg.groups, resolve=True)
    except (omegaconf.errors.OmegaConfBaseException, AttributeError):
        return {}

    _by_name = {
        each_grp.get("grp_name"): each_grp
        for each_grp in (_cfg_groups or [])
        if isinstance(each_grp, dict)
    }

    resolved: dict[str, Any] = {}
    for each_grp_name in grp_names:
        each_cfg_grp = _by_name.get(each_grp_name, None)
        if each_cfg_grp is None:
            continue
        resolved[each_grp_name] = {
            "ttrpm_dir": each_cfg_grp.get("ttrpm_dir", None),
            "experiments": each_cfg_grp.get("experiments", None),
        }
    return resolved


def _aggregate_group_drift_curves(
    cfg: DictConfig,
    stacked_drift_curves: np.ndarray,
    *,
    dispersion_type: str,
    y_axis_in_logscale: bool,
    n_trials: int,
    n_retained_trajectories: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Reduce a group's per-rollout drift curves to the plotted curve + band.

    Behaviour-preserving extraction (RLRP-818) of the aggregation block that
    used to live inline in :func:`_generate_drift_rate_plot_single`, so the
    experiment data consolidation and the figure share ONE reduction.

    :param cfg: Top-level multirun plot configuration.
    :param stacked_drift_curves: ``(n_rollouts, max_len + 1)`` drift curves,
        already scaled and prepended with the synthetic ``t=0`` column.
    :param dispersion_type: ``std`` or ``std-err`` (linear band).
    :param y_axis_in_logscale: Whether the y-axis is log-scaled.
    :param n_trials: The group's number of model seeds (``std-err`` norm).
    :param n_retained_trajectories: The number of distinct retained
        trajectories (``std-err`` norm).
    :return: ``(avg, lower_curve, upper_curve, mae_grp_reduction,
        logscale_band_method)``.
    :raise NotImplementedError: On an unsupported ``mae_grp_reduction``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        _mae_grp_reduction = cfg.show.generate_drift_rate_plot.get(
            "mae_grp_reduction", "mean"
        )
        if stacked_drift_curves.ndim == 1:
            avg = np.asarray(stacked_drift_curves, dtype=float)
        elif _mae_grp_reduction == "mean":
            avg = np.mean(stacked_drift_curves, axis=0)
        elif _mae_grp_reduction == "median":
            avg = np.median(stacked_drift_curves, axis=0)
        else:
            raise NotImplementedError(
                f"Key 'generate_drift_rate_plot.mae_grp_reduction' value {_mae_grp_reduction} not supported"
            )

        # Log-scale-only band recomputation (RLRP-807). On a log y-axis the
        # linear ``avg +/- dispersion`` band flares downward and can violate
        # the strictly-positive support; replace it with an empirical
        # percentile or log-normal/geometric band when requested. The linear
        # path is byte-for-byte untouched (default ``linear`` / linear axis).
        _logband_method = cfg.show.generate_drift_rate_plot.get(
            "logscale_band_method", "linear"
        )
        if y_axis_in_logscale and _logband_method != "linear":
            # Reuse the existing ``mae_grp_reduction`` key to pick the
            # ``log_normal`` center (median -> robust, recentered band;
            # otherwise geometric mean), so the plotted line stays
            # visually centered in its band (RLRP-807, option 1).
            # Percentile coverage (``percentile`` method only): ``sigma``
            # -> [16th, 84th], ``iqr`` -> narrower [25th, 75th] for
            # readability (RLRP-807).
            _lower_pct, _upper_pct = _resolve_percentile_band_bounds(cfg)
            avg, lower_curve, upper_curve = _compute_logscale_band(
                stacked_drift_curves,
                method=_logband_method,
                dispersion_type=dispersion_type,
                n_norm=n_trials,
                log_normal_center=(
                    "median" if _mae_grp_reduction == "median" else "mean"
                ),
                lower_pct=_lower_pct,
                upper_pct=_upper_pct,
            )
        else:
            # avg = np.nanmean(stacked_drift_curves, axis=0)
            # avg = np.nanmedian(
            #     stacked_drift_curves, axis=0
            # )  # (CRITICAL) ToDo: on task end >> delete this line ←
            std = np.nanstd(stacked_drift_curves, axis=0)
            if dispersion_type == "std":
                dispersion = std
            elif dispersion_type == "std-err":
                # dispersion = std / np.sqrt(
                #     n_trials
                # )  # N <- nb of models (aka seed)
                dispersion = std / np.sqrt(
                    n_trials * n_retained_trajectories
                )  # N <- nb of models (aka seed) X nb of rollout
                # dispersion = std/np.sqrt(n_retained_trajectories) # N <- nb of rollout

            lower_curve = avg - dispersion
            upper_curve = avg + dispersion

    return avg, lower_curve, upper_curve, _mae_grp_reduction, _logband_method


def _generate_drift_rate_plot_single(
    cfg: DictConfig,
    groups: dict[Any, Any],
    ground_truth_entries: Any,
    title: str,
    exp_dir_relative_path: Any,
    fill_between_alpha: float,
    plot_linewidth: float,
    headless: bool,
    *,
    length_cutoff: int | None = None,
    trials_label: str = "trials",
    y_axis_in_logscale: bool = False,
    x_axis_in_logscale: bool = False,
    drift_type: str = "timesteps",
    warmup_mode: Any = None,
    warmup_steps: int | None = None,
    dispersion_type: str = "std-err",  # Options: std, std-err
    filename_prefix: str = "",
    train_epoch: int | None = None,
    diaporama_mode: bool = False,
    export_only: bool = False,
    selected_trajectories: Sequence[str] | None = None,
) -> None:
    """Render the C-MAE drift-rate plot (all groups overlaid).

    For every per-rollout ``mae`` array in each group, the curve

        C-MAE[t]            = cumsum(mean over feature dim of mae)[t]
        GT-trj-length[t]    = cumsum(||GT.linear_velocity|| * dt)[t]
        C-MAE-drift-rate[t] = C-MAE[t] / ((t + 1) * GT-trj-length[t])

    is computed against the *matching* ground-truth trajectory (paired by
    ``_trajectory_name``). The plot overlays one mean+std band per group,
    mirroring :func:`generate_final_general_plot`.

    The aggregation horizon (``max_len``) is the shortest available
    trajectory length when ``length_cutoff is None``; when ``length_cutoff``
    is set, only trajectories with ``length >= length_cutoff`` are pooled and
    the horizon is fixed to exactly ``length_cutoff`` (RLRP-734).

    :param cfg: Top-level multirun plot configuration.
    :param groups: Per-group metric dict from :func:`compute_per_group_metric`.
    :param ground_truth_entries: List of ``TestTrajectoryEntry`` produced by
        :func:`_load_ground_truth_entries`; used to fetch linear velocity
        (``env.observations[:, :3]``) and timestamps delta
        (``env.timestamps.delta_stamps``) per trajectory.
    :param title: Base title string (already enriched by :func:`execute`).
    :param exp_dir_relative_path: Where to save the figure.
    :param fill_between_alpha: Alpha for the ±std band.
    :param plot_linewidth: Line width for group curves.
    :param headless: Headless backend flag.
    :param length_cutoff: Minimum-trajectory-length selector (RLRP-734).
        ``None`` selects every trajectory and keeps the shortest-trajectory
        horizon. A strictly-positive int selects only trajectories with
        ``length >= length_cutoff`` and fixes the horizon to exactly
        ``length_cutoff``; raises ``ValueError`` otherwise.
    :param trials_label: User-facing word substituted for ``"trials"`` in
        legend (``label_override.trials``).
    :param y_axis_in_logscale: If True, sets the y-axis in log scale.
    :param x_axis_in_logscale:
    :param drift_type: how the drfit numerator is setup Options: "timesteps" (default), "path_length"
    :param warmup_mode: Ground-truth-feed warm-up treatment mode
        (``None`` / ``'discard'`` / ``'reference_line'``); see
        :func:`_apply_ground_truth_warmup_treatment`. Applied with
        ``x_offset=1`` because this plot's step ``k`` is drawn at ``x = k + 1``.
        Defaults to ``None`` (no treatment).
    :param warmup_steps: Auto-derived warm-up step count applied by the
        treatment; ``None`` when unresolvable.
    :param dispersion_type: Type measure of dispersion use for the band around the plot mean
    :param filename_prefix: Optional prefix prepended to the saved file name
        (e.g. ``<train_epoch>_epoch_``) so a series of figures does not
        overwrite one another.
    :param train_epoch: Training-epoch number of the plotted snapshot, added as
        a secondary line in the in-axes horizon annotation; ``None`` (default /
        legacy deployed-model plot) omits the line.
    :param export_only: RLRP-818 CONSOLIDATION-ONLY mode. When ``True`` the
        whole data pipeline runs unchanged (group selection, horizon
        resolution, per-rollout drift, aggregation, warm-up trimming) and the
        consolidated CSV files are written, but every TERMINAL side effect is
        skipped: no PNG is saved, no LaTeX asset is exported and no ``*.text``
        sidecar is written. Used by :func:`generate_drift_rate_plot` to
        consolidate the epoch-checkpoint rollouts that are NOT rendered.
        Defaults to ``False``.
    :param selected_trajectories: The active ``show.selected_trajectories``
        (RLRP-841), used for ANNOTATION/METADATA only (title note, unmatched
        warning, consolidated CSV ``params``). The filtering itself happens
        upstream via :func:`select_ground_truth_entries` on
        ``ground_truth_entries``.
    :return: ``None``. Saves the figure to ``exp_dir_relative_path``.
    """
    X_OFFSET = 0

    # .... Build {trajectory_name: GT cumulative trajectory length [m]} ......
    # The GT trajectory length pseudocode is:
    #   GT-trj-length[t] = cumsum(L2Norm(GT.linear_velocity) * dt)[t]
    # We materialize one cumulative-length array per GT entry so every
    # per-rollout curve can be paired with the matching ground-truth.
    if not ground_truth_entries:
        consol_msg_universal_one_liner(
            "generate_drift_rate_plot: no ground-truth entries available; skipping.\n"
        )
        plt.close("all")
        return None
    if drift_type not in ["timesteps", "path_length"]:
        raise ValueError(
            f"Invalid drift_type: {drift_type}. Options: 'timesteps', 'path_length'"
        )
    # ``length_cutoff`` is a minimum-trajectory-length *selector* (RLRP-734):
    # ``None`` selects all trajectories (horizon = shortest); a strictly-
    # positive int selects trajectories with ``length >= length_cutoff`` and
    # fixes the horizon to exactly ``length_cutoff``.
    if length_cutoff is not None:
        if isinstance(length_cutoff, bool) or not isinstance(length_cutoff, int):
            raise ValueError(
                f"Invalid `length_cutoff` value: {length_cutoff!r}. "
                "Must be null or a strictly-positive integer."
            )
        if length_cutoff <= 0:
            raise ValueError(
                f"Invalid `length_cutoff` integer: {length_cutoff}. "
                "Must be strictly positive."
            )

    gt_cum_length_map = _build_gt_cum_length_map(ground_truth_entries)

    if not gt_cum_length_map:
        consol_msg_universal_one_liner(
            "generate_drift_rate_plot: no usable GT entries; skipping.\n"
        )
        plt.close("all")
        return None

    # .... Resolve aggregation horizon .........................................
    # Pool every (group, rollout) length that has a matching GT entry.
    _selected_grp_names = set(cfg.show.grp_names) if cfg.show.grp_names else None
    _per_group_rollouts: dict[str, list[tuple[str, Any, np.ndarray]]] = {}
    pool_lengths: list[int] = []
    # Pre-filter lengths of every matched trajectory (RLRP-734) so an
    # over-large ``length_cutoff`` can report the longest available length.
    _prefilter_lengths: list[int] = []
    for grp_key, group in groups.items():
        if (
            _selected_grp_names is not None
            and group.get("grp_name") not in _selected_grp_names
        ):
            continue
        per_rollout_mae: list[tuple[str, Any, np.ndarray]] = []
        for ogm in group.get("_original_metrics", []) or []:
            tname = getattr(ogm, "_trajectory_name", None)
            # RLRP-819 `W3`: the MODEL SEED identity of the rollout, read right
            # beside the trajectory name so row `i` of `stacked_drift_curves`
            # always carries BOTH labels (never reconstructed from order).
            trial_key = getattr(ogm, "_trial_key", None)
            if tname is None:
                continue
            # Be tolerant to either spelling: rollout-side ``_trajectory_name``
            # is a TTRPM directory basename, GT-side ``trajectory_name`` may
            # be prefixed by a test-set subdir (cf. ``gt_cum_length_map``
            # construction above, which indexes both forms).
            if tname not in gt_cum_length_map:
                _alt = os.path.basename(str(tname))
                if _alt in gt_cum_length_map:
                    tname = _alt
                else:
                    continue

            three_dim_mae = getattr(ogm, "mae", None)
            if (
                three_dim_mae is None
                or not hasattr(three_dim_mae, "shape")
                or three_dim_mae.size == 0
            ):
                continue

            # (Priority) ToDo: assess droping the conditional
            # three_dim_mae = (
            #     _make_array_finite(three_dim_mae)
            #     if three_dim_mae.ndim > 1
            #     else three_dim_mae
            # )
            three_dim_mae = _make_array_finite(three_dim_mae)

            # Reduce feature dimension (3D coord MAE → 1D MAE).
            _mae_feature_reduction = cfg.show.generate_drift_rate_plot.get(
                "mae_feature_reduction", "mean"
            )
            if three_dim_mae.ndim == 1:
                one_dim_mae = np.asarray(three_dim_mae, dtype=float)
            elif _mae_feature_reduction == "mean":
                one_dim_mae = np.mean(three_dim_mae, axis=-1)
            elif _mae_feature_reduction == "median":
                one_dim_mae = np.median(three_dim_mae, axis=-1)
            else:
                raise NotImplementedError(
                    f"Key 'generate_drift_rate_plot.mae_feature_reduction' value {_mae_feature_reduction} not supported"
                )

            # Cap to GT trajectory length so the per-timestep division is
            # well-defined for every retained sample.
            gt_len_arr = gt_cum_length_map[tname]
            effective_len = min(one_dim_mae.shape[0], gt_len_arr.shape[0])
            if effective_len <= 0:
                continue

            _prefilter_lengths.append(effective_len)
            # RLRP-734: drop trajectories shorter than the requested minimum
            # length so the plotted horizon reflects the longer rollouts.
            if length_cutoff is not None and effective_len < length_cutoff:
                continue

            per_rollout_mae.append((tname, trial_key, one_dim_mae[:effective_len]))
            pool_lengths.append(effective_len)
        if per_rollout_mae:
            _per_group_rollouts[grp_key] = per_rollout_mae

    # RLRP-841: warn about selected trajectories that matched no pooled
    # rollout (e.g. rollouts not regenerated for them) so a thinner plot is
    # never silent.
    _unmatched_selection = _unmatched_selected_trajectories(
        selected_trajectories,
        [
            each_tname
            for each_rollouts in _per_group_rollouts.values()
            for each_tname, _, _ in each_rollouts
        ],
    )
    if _unmatched_selection:
        consol_msg_universal_one_liner(
            "generate_drift_rate_plot: WARNING `show.selected_trajectories` "
            f"entry(ies) matched no rollout in any plotted group: "
            f"{_unmatched_selection}\n"
        )

    if not _per_group_rollouts:
        if length_cutoff is not None and _prefilter_lengths:
            consol_msg_universal_one_liner(
                "generate_drift_rate_plot: no trajectory satisfies "
                f"length >= length_cutoff={length_cutoff} "
                f"(longest available={max(_prefilter_lengths)}); skipping.\n"
            )
        else:
            consol_msg_universal_one_liner(
                "generate_drift_rate_plot: no rollout matched any GT trajectory; skipping.\n"
            )
        plt.close("all")
        return None

    # .... Resolve horizon (RLRP-734) .........................................
    # ``length_cutoff is None`` -> horizon is the shortest available (retained)
    # trajectory. When set, every retained trajectory is >= length_cutoff, so
    # the horizon is fixed to exactly length_cutoff.
    shortest_traj_len = int(min(pool_lengths))
    if length_cutoff is not None:
        max_len = int(length_cutoff)
        consol_msg_universal_one_liner(
            f"generate_drift_rate_plot: min length>={length_cutoff}, "
            f"horizon={max_len}, kept {len(pool_lengths)}/{len(_prefilter_lengths)} "
            "trajectories.\n"
        )
    else:
        max_len = shortest_traj_len
    if max_len <= 0:
        plt.close("all")
        return None

    # Per-timestep divisor — ``each_timesteps`` in the issue pseudocode is
    # the 1-based timestep index ``t`` (so the t=0 sample is normalized by
    # 1, not 0).
    timesteps = np.arange(1, max_len + 1, dtype=float)

    # .... Render .............................................................
    with warnings.catch_warnings():
        manage_matplotlib_warnings()
        manage_matplotlib_backend(get_plot_render_type_cfg(cfg).show_plot, headless)

        plt.rcParams.update(get_plot_render_type_cfg(cfg).rcParams_update)

        fig: Figure = plt.figure(
            figsize=get_plot_figsize(cfg, "generate_drift_rate_plot"),
            dpi=get_plot_render_type_cfg(cfg).figdpi,
        )
        ax: Axes = fig.add_subplot(111)

        fig_init = False
        color_idx = 0
        all_upper_bounds: list[float] = []
        # Collect the standalone-render legend entries (model group name plus
        # its number of model seeds / trajectories) so they can be consolidated
        # into the LaTeX-``includegraphics`` sidecar meta file even though that
        # render type displays only a compact short name in the legend.
        _legend_group_labels: list[str] = []

        # .... RLRP-818 experiment data consolidation ........................
        # Opt-in export of the exact series rendered below, so the ICRA 2026
        # standalone plot functions (RLRP-817) can rebuild this figure from
        # the CSVs alone. The timestep -> timestamp conversion is resolved
        # once, with the very rule the figure's time annotation uses.
        _do_consolidate = is_consolidation_enabled(cfg)
        _consolidation_group_rows: dict[str, list] = {}
        _consolidation_group_metadata: dict[str, dict] = {}
        _timestamp_scale: float | None = None
        _timestamp_unit: str | None = None
        _logband_method = "linear"
        if _do_consolidate:
            with consolidation_guard("drift_rate: timestamp conversion"):
                _timestamp_scale, _timestamp_unit = _resolve_timestamp_conversion(cfg)

        # .... RLRP-819 raw per-rollout drift population ......................
        # Action `R3` of the RLRC ICRA2026 fig-2 / fig-3 standalone plots
        # `.junie` plan
        # (`feat_RLRP-819_two_new_standalone_plots_fig2_fig3_plan_20260910.md`).
        # Additive second metric family, opt-in through
        # `consolidate_data.emit_samples`: the reduced `drift_rate` family
        # above is left byte-identical. Gated to the DEPLOYED CSV by default
        # (`drift_rate_sample_epochs: deployed`) since that is all fig 2 reads;
        # `all` also dumps every `<E>_epoch_` checkpoint (8-16x the files).
        _consolidate_data_cfg = cfg.get("consolidate_data", None) or {}
        _emit_samples = bool(_consolidate_data_cfg.get("emit_samples", False))
        _samples_epoch_selected = (
            str(_consolidate_data_cfg.get("drift_rate_sample_epochs", "deployed"))
            == "all"
            or filename_prefix == ""
        )
        _sample_horizon_ratios = _consolidate_data_cfg.get(
            "drift_rate_sample_horizon_ratios", None
        )
        # 🆕 v6 (`W4`): the v5 abscissa selectors are RETIRED -- the full grid is
        # always dumped. What is left is a precision lever and a size lever.
        _sample_float_format = _consolidate_data_cfg.get(
            "drift_rate_sample_float_format", None
        )
        _sample_timestep_stride = _consolidate_data_cfg.get(
            "drift_rate_sample_timestep_stride", None
        )
        _consolidation_sample_inputs: dict[str, dict] = {}
        _consolidation_horizon_ratio_map: dict = {}

        EPS = 1e-9
        for grp_key, per_rollout_mae in _per_group_rollouts.items():
            group = groups[grp_key]
            grp_display_name = group.get("grp_name", grp_key)

            # Build per-rollout drift-rate curves of uniform length ``max_len``.
            # Track the distinct trajectories actually retained so the legend
            # ``× N trj`` count reflects the ``length_cutoff`` selection
            # (RLRP: legend trj count must follow the filtered pool).
            per_rollout_drift_curves: list[np.ndarray] = []
            # RLRP-819 `R3`: the ONLY rollout identity available at this hook
            # (ruling C1), captured in the ROW order of the stacked population
            # so `sample_index` and `trajectory_name` always agree.
            _rollout_trajectory_names: list[str] = []
            # RLRP-819 `W3`: appended in LOCKSTEP with the trajectory names, so
            # row `i` of the stacked population always has both labels.
            _rollout_seed_keys: list = []
            _retained_tnames: set[str] = set()
            _nonfinite_tnames: list[str] = []
            for tname, trial_key, one_dim_mae in per_rollout_mae:
                if one_dim_mae.shape[0] < max_len:
                    continue
                _retained_tnames.add(tname)
                # Shared drift-rate compute core (RLRP-773 W9): identical
                # per-rollout division as :func:`generate_drift_rate_plot_per_train_epoch`.
                drift = _compute_single_rollout_drift(
                    one_dim_mae,
                    gt_cum_length_map[tname],
                    timesteps,
                    max_len,
                    drift_type=drift_type,
                    warmup_mode=warmup_mode,
                    warmup_steps=warmup_steps,
                    cfg=cfg,
                    y_axis_in_logscale=y_axis_in_logscale,
                )

                # Non-finite drift values (``inf`` from a division by a
                # zero/near-zero GT path-length reference, or ``nan`` from
                # ``inf - inf``) would otherwise make the downstream
                # ``np.nanstd`` aggregation raise "invalid value encountered in
                # subtract" (``np.nanmean``/``np.nanstd`` only ignore ``nan``,
                # NOT ``inf``). Name the offending (group, trajectory) pair,
                # then clamp the curve to finite proxies so the value neither
                # crashes the aggregation nor propagates as ``inf``.
                if not np.all(np.isfinite(drift)):
                    _nonfinite_tnames.append(tname)
                    drift = _make_array_finite(drift)

                per_rollout_drift_curves.append(drift)
                _rollout_trajectory_names.append(str(tname))
                _rollout_seed_keys.append(trial_key)

            if not per_rollout_drift_curves:
                continue

            if _nonfinite_tnames:
                consol_msg_universal_one_liner(
                    "generate_drift_rate_plot: non-finite (nan/inf) drift "
                    f"values in group '{grp_display_name}' for "
                    f"{len(_nonfinite_tnames)} trajectory(ies): "
                    f"{sorted(_nonfinite_tnames)}. These were made finite (nan->0, "
                    "+inf->curve max, -inf->curve min) before aggregation.\n"
                )

            stacked_drift_curves = np.stack(per_rollout_drift_curves, axis=0)

            if drift_type == "timesteps":
                scaling = 1.0
            elif drift_type == "path_length":
                scaling = 100.0
            else:
                raise ValueError(f"Unknown drift_type: {drift_type}")

            stacked_drift_curves = scaling * stacked_drift_curves

            # Add the missing timestep 0 with an error=0.0 for plot smooth entry
            stacked_drift_curves = np.hstack(
                [np.zeros((len(stacked_drift_curves[:, 0]), 1)), stacked_drift_curves]
            )
            # stack = np.nan_to_num(stack)

            (
                avg,
                lower_curve,
                upper_curve,
                _mae_grp_reduction,
                _logband_method,
            ) = _aggregate_group_drift_curves(
                cfg,
                stacked_drift_curves,
                dispersion_type=dispersion_type,
                y_axis_in_logscale=y_axis_in_logscale,
                n_trials=group.get("n_trials", 1),
                n_retained_trajectories=len(_retained_tnames),
            )

            fig_init = True
            x = np.arange(len(avg))
            color = COLORS[color_idx % len(COLORS)]
            line_style = group.get("line_style") or "-"

            _n_trial = group.get("n_trials", 0)
            # Report the distinct trajectories retained after the
            # ``length_cutoff`` selection rather than the unfiltered group
            # count, so the legend ``× N trj = R roll`` figures stay coherent.
            _n_traj = len(_retained_tnames)
            _gpr_count_label = _format_grp_count_label(
                str(grp_display_name),
                n_trials=_n_trial,
                n_trajectories=_n_traj,
                n_total_rollouts=len(per_rollout_drift_curves),
                trials_label=trials_label,
                do_average_across_trajectories=False,
                show_acronym=cfg.show.get("acronym", True),
            )

            # Keep the full standalone-render legend entry (group name + model
            # seed / trajectory counts) for every plotted group so it can be
            # consolidated into the LaTeX-``includegraphics`` sidecar meta file.
            _legend_group_labels.append(_gpr_count_label)

            if not is_latex_includegraphics_render_type(cfg):
                label = _gpr_count_label
            else:
                label = group.get("gpr_short_name", grp_key)

            # Ground-truth-feed warm-up treatment (YouTrack RLRP-733): in
            # ``'discard'`` mode drop the ``0 .. warm-up`` prefix from the
            # curve and its band. After the synthetic ``t=0`` prepend the
            # drift-rate x-array maps exactly ``x = i`` -> timestep ``t = i``,
            # so the warm-up boundary lands at ``x = warmup_steps``
            # (``x_offset=0``). Trimming keeps ``x >= warmup_steps`` so the
            # curve starts *at* the boundary ``t = gt_warmup_length`` — where
            # the RLRP-736 cumulative reset makes the drift ~0 — instead of one
            # step later (RLRP-736-ORep).
            # x_p, avg_p, dispersion_p = x, avg, std # (CRITICAL) ToDo: on task end >> delete this line ←
            if warmup_mode == "discard":
                x_p, avg_p = _trim_warmup_interval(
                    x, avg, warmup_steps=warmup_steps, x_offset=X_OFFSET
                )
                _, lower_p = _trim_warmup_interval(
                    x, lower_curve, warmup_steps=warmup_steps, x_offset=X_OFFSET
                )
                _, upper_p = _trim_warmup_interval(
                    x, upper_curve, warmup_steps=warmup_steps, x_offset=X_OFFSET
                )
            else:
                x_p, avg_p = x, avg
                lower_p, upper_p = lower_curve, upper_curve

            # .... RLRP-818 export hook ......................................
            # The plotted series are FINAL here (post warm-up trimming) and
            # are exactly what ``ax.plot`` / ``ax.fill_between`` render below.
            if _do_consolidate:
                with consolidation_guard(f"drift_rate: group {grp_display_name!r}"):
                    _consolidation_group_rows[str(grp_display_name)] = (
                        build_drift_rate_rows(
                            timesteps=x_p,
                            drift_rate=avg_p,
                            drift_rate_lower=lower_p,
                            drift_rate_upper=upper_p,
                            timestamp_scale=_timestamp_scale,
                            timestamp_unit=_timestamp_unit,
                        )
                    )
                    _consolidation_group_metadata[str(grp_display_name)] = {
                        "gpr_short_name": group.get("gpr_short_name", None),
                        "n_trials": _n_trial,
                        "n_trajectories": _n_traj,
                        "n_rollouts": len(per_rollout_drift_curves),
                    }

            # .... RLRP-819 raw population export hook (`W3`) ................
            # Sits NEXT TO the reduced export above so the two families can
            # never drift apart, and inside a `consolidation_guard` so a
            # failure still cannot break the figure. The WIDE per-seed tables
            # themselves are built after the loop: their columns are the
            # EXPERIMENT-wide trajectory id mapping, i.e. the union over every
            # consolidated group (`W0b`).
            if _do_consolidate and _emit_samples and _samples_epoch_selected:
                with consolidation_guard(
                    f"drift_rate_samples: group {grp_display_name!r}"
                ):
                    # ⚠️ `stacked_drift_curves` is UNTRIMMED -- re-apply the
                    #    EXACT same trim used for avg/lower/upper, column-wise,
                    #    so `timestep` stays aligned with the reduced CSV.
                    if warmup_mode == "discard":
                        _samples_p = _trim_warmup_columns(
                            x,
                            stacked_drift_curves,
                            warmup_steps=warmup_steps,
                            x_offset=X_OFFSET,
                        )
                    else:
                        _samples_p = stacked_drift_curves
                    # Fail-soft (`W3`): without the model seed identity the
                    # per-seed split is undefined, so the EXPORT is skipped
                    # (the figure is never affected).
                    if any(each is None for each in _rollout_seed_keys):
                        raise ValueError(
                            "drift_rate_samples: group "
                            f"{grp_display_name!r} has rollout(s) without a "
                            "`_trial_key` (model seed identity); the per-seed "
                            "sample export is skipped for that group."
                        )
                    _consolidation_sample_inputs[str(grp_display_name)] = {
                        "timesteps": x_p,
                        "samples": _samples_p,
                        "trajectory_names": list(_rollout_trajectory_names),
                        "seed_keys": list(_rollout_seed_keys),
                    }
                    # Provenance only since v6: the resolved abscissae are
                    # recorded in the meta file for the `N10` producer/consumer
                    # cross-check; they no longer gate WHAT is dumped.
                    if _sample_horizon_ratios is not None:
                        _consolidation_horizon_ratio_map = (
                            resolve_horizon_ratio_timesteps(
                                _sample_horizon_ratios,
                                timesteps=x_p,
                                length_cutoff=length_cutoff,
                            )
                        )

            ax.plot(
                x_p,
                avg_p,
                color=color,
                linewidth=plot_linewidth,
                linestyle=line_style,
                label=label,
            )
            ax.fill_between(
                x_p,
                lower_p,
                upper_p,
                color=color,
                alpha=fill_between_alpha,
                linewidth=0,
            )
            # Readability improvement (RLRP-807): draw crisp, opaque band
            # boundaries on top of the semi-transparent fill so overlapping
            # bands stay distinguishable. Gated on
            # ``plot.style.fill_between_linewidth`` (default ``0.0`` -> strict
            # no-op, unchanged legacy appearance).
            _fill_between_linewidth = float(
                get_plot_render_type_cfg(cfg).style.get("fill_between_linewidth", 0.0)
            )
            if _fill_between_linewidth > 0.0:
                ax.plot(
                    x_p,
                    lower_p,
                    color=color,
                    linewidth=_fill_between_linewidth,
                    # linestyle=line_style,
                    linestyle="solid",
                )
                ax.plot(
                    x_p,
                    upper_p,
                    color=color,
                    linewidth=_fill_between_linewidth,
                    # linestyle=line_style,
                    linestyle="solid",
                )
            # finite_upper = (avg + std)[np.isfinite(avg + std)]
            finite_upper = upper_p[np.isfinite(upper_p)]
            if finite_upper.size > 0:
                all_upper_bounds.append(float(np.nanmax(finite_upper)))

            color_idx += 1

        # .... RLRP-818 experiment data consolidation write ...................
        # Written before the (optionally skipped) rendering tail so the
        # ``export_only`` mode shares the exact same export path.
        if _do_consolidate and _consolidation_group_rows:
            with consolidation_guard("drift_rate: write"):
                consolidate_drift_rate(
                    cfg,
                    _consolidation_group_rows,
                    file_stem=f"{filename_prefix}test_time_models_comparaison_drift_rate",
                    params={
                        "drift_type": drift_type,
                        "unit": DRIFT_TYPE_UNIT.get(str(drift_type), "m"),
                        "dispersion_type": dispersion_type,
                        "logscale_band_method": _logband_method,
                        "mae_grp_reduction": cfg.show.generate_drift_rate_plot.get(
                            "mae_grp_reduction", "mean"
                        ),
                        "mae_feature_reduction": cfg.show.generate_drift_rate_plot.get(
                            "mae_feature_reduction", "mean"
                        ),
                        "length_cutoff": length_cutoff,
                        "selected_trajectories": (
                            list(selected_trajectories)
                            if selected_trajectories is not None
                            else None
                        ),
                        "max_len": max_len,
                        # `train_epoch` is deliberately NOT recorded here: this
                        # metric-level meta file describes EVERY CSV of the
                        # directory (the deployed-model rollout plus one per
                        # epoch checkpoint), so a single value would be
                        # misleading. It is carried per row by the
                        # `train_epoch` CSV column instead.
                        "warmup_mode": warmup_mode,
                        "warmup_steps": warmup_steps,
                        "y_axis_in_logscale": y_axis_in_logscale,
                        "x_axis_in_logscale": x_axis_in_logscale,
                        "timestamp_scale": _timestamp_scale,
                        "timestamp_unit": _timestamp_unit,
                        "trials_label": trials_label,
                    },
                    source_experiments=_resolve_cfg_group_source_experiments(
                        cfg, _consolidation_group_rows.keys()
                    ),
                    group_metadata=_consolidation_group_metadata,
                )

        # .... RLRP-819 raw population consolidation write (`W3`) .............
        if _do_consolidate and _consolidation_sample_inputs:
            with consolidation_guard("drift_rate_samples: write"):
                # The trajectory id mapping is per EXPERIMENT and shared by
                # every group: the sorted union of the trajectory names of the
                # consolidated groups (`W0b`).
                _trajectory_name_mapping = build_trajectory_name_mapping(
                    {
                        each_grp: each_inputs["trajectory_names"]
                        for each_grp, each_inputs in _consolidation_sample_inputs.items()
                    }
                )
                _consolidation_seed_tables: dict[str, dict] = {}
                _consolidation_sample_provenance: dict[str, dict] = {}
                for each_grp, each_inputs in _consolidation_sample_inputs.items():
                    (
                        _consolidation_seed_tables[each_grp],
                        _consolidation_sample_provenance[each_grp],
                    ) = build_drift_rate_seed_tables(
                        timesteps=each_inputs["timesteps"],
                        samples=each_inputs["samples"],
                        trajectory_names=each_inputs["trajectory_names"],
                        seed_keys=each_inputs["seed_keys"],
                        trajectory_name_mapping=_trajectory_name_mapping,
                        timestamp_scale=_timestamp_scale,
                        timestamp_unit=_timestamp_unit,
                        stride=_sample_timestep_stride,
                        float_format=_sample_float_format,
                    )
                consolidate_drift_rate_samples(
                    cfg,
                    _consolidation_seed_tables,
                    file_stem=f"{filename_prefix}test_time_models_comparaison_drift_rate_samples",
                    trajectory_name_mapping=_trajectory_name_mapping,
                    group_provenance=_consolidation_sample_provenance,
                    params={
                        "drift_type": drift_type,
                        "unit": DRIFT_TYPE_UNIT.get(str(drift_type), "m"),
                        "mae_feature_reduction": cfg.show.generate_drift_rate_plot.get(
                            "mae_feature_reduction", "mean"
                        ),
                        # The population is RAW: it carries no reduction and no
                        # dispersion band, hence no `dispersion_type` /
                        # `logscale_band_method` here. `mae_grp_reduction` IS
                        # recorded: it is the reduction the consumer must apply
                        # to recover the plotted `drift_rate` curve.
                        "mae_grp_reduction": cfg.show.generate_drift_rate_plot.get(
                            "mae_grp_reduction", "mean"
                        ),
                        # UNCHANGED by v6 (ruling `Q14`): fig 2's box is still
                        # over ROLLOUTS, the storage layout alone changed.
                        # `n_seeds` / `n_rollouts` / `missing_cells` /
                        # `trajectory_*` are injected by
                        # `consolidate_drift_rate_samples` from the builder's
                        # provenance, so they can never disagree with the
                        # written columns.
                        "sample_reduction": "none",
                        "sample_identity": "rollout",
                        "float_format": (
                            "repr"
                            if _sample_float_format is None
                            else _sample_float_format
                        ),
                        "timestep_stride": _sample_timestep_stride,
                        "horizon_ratio_map": dict(_consolidation_horizon_ratio_map),
                        "warmup_end": int(x_p[0]) if len(x_p) else None,
                        "horizon_length": (
                            (length_cutoff if length_cutoff is not None else int(x_p[-1]))
                            - int(x_p[0])
                            if len(x_p)
                            else None
                        ),
                        "length_cutoff": length_cutoff,
                        "selected_trajectories": (
                            list(selected_trajectories)
                            if selected_trajectories is not None
                            else None
                        ),
                        "max_len": max_len,
                        "warmup_mode": warmup_mode,
                        "warmup_steps": warmup_steps,
                        "timestamp_scale": _timestamp_scale,
                        "timestamp_unit": _timestamp_unit,
                        "trials_label": trials_label,
                    },
                    source_experiments=_resolve_cfg_group_source_experiments(
                        cfg, _consolidation_seed_tables.keys()
                    ),
                    group_metadata=_consolidation_group_metadata,
                )

        if not fig_init:
            plt.close("all")
            return None

        # RLRP-818 ``export_only``: the consolidated data is written, every
        # terminal side effect (save / LaTeX export / sidecar meta) is skipped.
        if export_only:
            plt.close("all")
            return None

        # if is_cfg_key_exist(cfg, "show.generate_drift_rate_plot.top_ylim"):
        if cfg.show.generate_drift_rate_plot.get("top_ylim", None) is not None:
            _top_y_lim = cfg.show.generate_drift_rate_plot.top_ylim
        elif cfg.show.get("top_ylim", None) is not None:
            _top_y_lim = cfg.show.top_ylim
        else:
            _top_y_lim = None

        if y_axis_in_logscale:
            ax.set_yscale("log")
            ax.set_ylim(bottom=float(np.nanmin(avg_p[avg_p > 0.0])))
            if _top_y_lim is not None:
                ax.set_ylim(top=_top_y_lim)
            if cfg.show.generate_drift_rate_plot.get("bottom_ylim", None) is not None:
                ax.set_ylim(bottom=cfg.show.generate_drift_rate_plot.bottom_ylim)
        elif _top_y_lim is not None:
            ax.set_ylim(bottom=0, top=_top_y_lim)
        else:
            if all_upper_bounds:
                ax.set_ylim(bottom=0, top=float(np.nanmax(all_upper_bounds)) * 1.1)
            else:
                ax.set_ylim(bottom=0)

        # In ``'discard'`` mode the left bound is the warm-up boundary
        # (``warmup_steps`` here; arrays already trimmed there) so the timestep/
        # timestamp axes start faithfully at the boundary under both linear and
        # log scale (YouTrack RLRP-733 / RLRP-736-ORep).
        _discard_left = _warmup_discard_left(
            warmup_mode, warmup_steps, x_offset=X_OFFSET
        )
        if x_axis_in_logscale:
            ax.set_xscale("log")
            # ax.set_xscale('symlog', linthresh=1e-9)
            # ax.set_xscale('log', nonpositive='mask')
            if _discard_left is not None:
                ax.set_xlim(left=_discard_left, right=max_len)
            else:
                ax.set_xlim(right=max_len)
        else:
            ax.set_xlim(
                left=_discard_left if _discard_left is not None else 0,
                right=max_len,
            )

        # Ground-truth-feed warm-up treatment (YouTrack RLRP-733). No-op when
        # ``warmup_mode`` is ``None``. After the synthetic ``t=0`` prepend the
        # drift-rate x-array maps ``x = i`` -> timestep ``t = i``, so the
        # warm-up boundary lands at ``x = warmup_steps`` (``x_offset=0``).
        # ``x_axis_in_logscale`` is passed so the ``discard`` clamp is log-safe.
        if cfg.show.get("compounded_predictions_score", False):
            _apply_ground_truth_warmup_treatment(
                cfg,
                ax,
                warmup_steps,
                mode=warmup_mode,
                x_axis_in_logscale=x_axis_in_logscale,
                x_offset=X_OFFSET,
            )

        if y_axis_in_logscale or x_axis_in_logscale:
            ax.minorticks_on()
            ax.grid(True, which="major", linestyle="-", linewidth=0.3)
            ax.grid(True, which="minor",
                    linestyle="--",
                    linewidth=0.19)
        else:
            ax.grid(True)

        # Build the plot title text unconditionally so it can be consolidated
        # into the LaTeX-``includegraphics`` sidecar meta file even when the
        # on-figure title is suppressed (``print_title`` is False for that
        # render type). ``ax.set_title`` is still gated on ``print_title``.
        _cutoff_note = _selection_note(selected_trajectories) + (
            f"horizon={length_cutoff} (min length>={length_cutoff}), "
            if length_cutoff is not None
            else ""
        )
        _band_desc = _resolve_logscale_band_desc(
            cfg,
            y_axis_in_logscale=y_axis_in_logscale,
            dispersion_type=dispersion_type,
        )
        if drift_type == "timesteps":
            _drift_title_text = (
                f"{title}\n"
                f"Average drift up to t ({_cutoff_note}"
                f"shortest available trajectory={shortest_traj_len}).\n"
                f"{_band_desc}"
            )
        elif drift_type == "path_length":
            _drift_title_text = (
                f"{title}\n"
                f"Average drift rate ({_cutoff_note}"
                f"shortest available trajectory={shortest_traj_len}).\n"
                f"{_band_desc}"
            )
        else:
            _drift_title_text = title

        if get_plot_render_type_cfg(cfg).print_title:
            ax.set_title(_drift_title_text)

        # Make the aggregation horizon stand out with a bold, boxed annotation
        # inside the axes (RLRP: the horizon drives the whole metric, so it must
        # be visually prominent rather than buried in the title text). The
        # warm-up length is stacked UNDER the horizon in the SAME box, but less
        # prominent (small/gray) so the horizon stays dominant. The training
        # epoch (when this plot is rendered from an ``epoch_<E>`` snapshot) is
        # added as a secondary line — only this per-timestep plot passes it.
        if get_plot_render_type_cfg(cfg).print_meta_info:
            _add_horizon_annotation(
                ax, max_len, warmup_steps, cfg, train_epoch=train_epoch
            )

        if get_plot_render_type_cfg(cfg).print_legend:
            if diaporama_mode or train_epoch is not None:
                ax.legend(loc="lower right", framealpha=0.65)
            else:
                # ax.legend(loc="upper right")
                ax.legend(loc="lower right")
                # ax.legend(loc="best", framealpha=0.85)

        if get_plot_render_type_cfg(cfg).print_axis_label:
            ax.set_xlabel("Time-steps")

        # if not is_latex_includegraphics_render_type(cfg):
        _ensure_primary_yaxis_endpoint_ticks(
            ax,
            scientific_notation=get_plot_render_type_cfg(cfg).style.get(
                "yaxis_tick_scientific_notation", True
            ),
        )
        _ensure_primary_xaxis_endpoint_ticks(
            ax, enforce_right_endpoint=not is_latex_includegraphics_render_type(cfg)
        )
        _attach_timestamp_secondary_xaxis(
            cfg, ax, enforce_right_endpoint=not is_latex_includegraphics_render_type(cfg)
        )

        if get_plot_render_type_cfg(cfg).print_axis_label:
            if drift_type == "timesteps":
                if is_latex_includegraphics_render_type(cfg):
                    ax.set_ylabel("Avg drift up to t (m)")
                else:
                    ax.set_ylabel("Avg drift up to t (m) [C-MAE / t]")
            elif drift_type == "path_length":
                ax.set_ylabel("Avg drift rate (%) [C-MAE / (t \u00b7 GT-travel-len)]")
            else:
                raise ValueError(f"Unknown drift_type: {drift_type}")

        if get_plot_render_type_cfg(cfg).print_meta_info:
            display_experiment_name(cfg, fig)

        show_and_save_plot_helper(
            fig,
            exp_dir_relative_path,
            f"{filename_prefix}test_time_models_comparaison_drift_rate",
            headless,
            get_plot_render_type_cfg(cfg).show_plot,
            get_plot_render_type_cfg(cfg).save_dpi,
            save=not is_latex_includegraphics_render_type(cfg),
        )

        if is_latex_includegraphics_render_type(cfg):
            project_root = fetch_r2s2r_project_root_path(cfg, lvl_up=1)
            if cfg.show.compounded_predictions_score:
                rollout_type_name = "CP"
            else:
                rollout_type_name = "GT"
            if cfg.show.target_is_ood:
                rollout_type_name = f"OOD_{rollout_type_name}"
            else:
                rollout_type_name = f"IND_{rollout_type_name}"

            _latex_save_dir = os.path.join(
                project_root, cfg.latex_includegraphics.save_path
            )
            show_and_save_plot_helper(
                fig,
                _latex_save_dir,
                f"{filename_prefix}{cfg.latex_includegraphics.file_name}_drift_rate_{rollout_type_name}",
                headless,
                False,
                get_plot_render_type_cfg(cfg).latex_save_dpi,
                save=True,
            )

            # Consolidate the (suppressed) on-figure title + experiment-name
            # overlay into a sidecar meta text file next to the exported figure.
            _write_latex_includegraphics_meta_file(
                cfg,
                _latex_save_dir,
                f"{filename_prefix}{cfg.latex_includegraphics.file_name}"
                f"_drift_rate_{rollout_type_name}_meta.text",
                title_text=_drift_title_text,
                legend_labels=_legend_group_labels,
            )

    plt.close("all")
    return None


def _resolve_group_epoch_rollout_sources(
    cfg: DictConfig,
    groups: dict[Any, Any],
    root_project_path: str | bytes,
) -> dict[Any, dict[str, Any]]:
    """Resolve, per plotted group, the ``ttrpm_dir`` + experiment-base list.

    Mirrors the experiment-base resolution used by
    :func:`compute_per_group_metric` / :func:`regenerate_rollouts`
    (``root_project_path`` + ``experiment_path`` + ``multirun_paths``) so the
    per-train-epoch drift plot reads the SAME experiment trees, just one level
    deeper under ``epoch_checkpoints_rollouts/epoch_<E>/`` (RLRP-773 W9).

    :param cfg: Top-level multirun plot configuration.
    :param groups: Per-group metric dict from :func:`compute_per_group_metric`
        (used only to know which ``group_<n>`` keys / display names are live).
    :param root_project_path: Project root used to resolve relative experiment paths.
    :return: ``{grp_key: {"grp_name", "ttrpm_dir", "experiment_bases"}}`` for
        every plotted group. Empty when nothing resolves.
    """
    _selected_grp_names = set(cfg.show.grp_names) if cfg.show.grp_names else None

    # ``cfg.groups`` is the authoritative source of ``ttrpm_dir`` + experiment
    # layout; index it by ``grp_name`` so it can be paired to the live
    # ``group_<n>`` keys carried by ``groups``.
    _cfg_groups_by_name: dict[str, dict] = {}
    for each_grp in omegaconf.OmegaConf.to_container(cfg.groups):
        _cfg_groups_by_name[each_grp.get("grp_name")] = each_grp

    resolved: dict[Any, dict[str, Any]] = {}
    for grp_key, group in groups.items():
        grp_name = group.get("grp_name")
        if _selected_grp_names is not None and grp_name not in _selected_grp_names:
            continue
        each_grp = _cfg_groups_by_name.get(grp_name)
        if each_grp is None:
            continue

        experiment_bases: list[str] = []
        _seen: set[str] = set()
        for each_experiments in each_grp.get("experiments", []) or []:
            multirun_paths = each_experiments.get("multirun_paths") or ["."]
            for each_multirun_path in multirun_paths:
                experiment_base = os.path.realpath(
                    os.path.join(
                        str(root_project_path),
                        str(each_experiments["experiment_path"]),
                        str(each_multirun_path),
                    )
                )
                if experiment_base in _seen:
                    continue
                _seen.add(experiment_base)
                experiment_bases.append(experiment_base)

        resolved[grp_key] = {
            "grp_name": grp_name,
            "ttrpm_dir": str(each_grp.get("ttrpm_dir")),
            "experiment_bases": experiment_bases,
        }
    return resolved


def _load_epoch_rollout_reduced_maes(
    cfg: DictConfig,
    epoch_dir: str,
    ttrpm_dir: str,
    gt_cum_length_map: dict[str, np.ndarray],
) -> list[tuple[str, np.ndarray]]:
    """Read one epoch's per-trajectory rollouts into ``(tname, reduced_mae)`` pairs.

    Consumes the SAME ``testtime_rollouts/<traj>/<ttrpm_dir>/`` structure the
    per-model plot already reads (cf. :func:`compute_per_group_metric`), just
    nested under ``epoch_checkpoints_rollouts/epoch_<E>/``. Legacy-safe: any
    missing/malformed rollout is skipped, never raised (RLRP-773 W9).

    :param cfg: Top-level multirun plot configuration (selects the metric subdir).
    :param epoch_dir: An ``epoch_checkpoints_rollouts/epoch_<E>/`` directory.
    :param ttrpm_dir: The TTRPM sub-directory name for the owning group.
    :param gt_cum_length_map: GT cumulative-length map used to pair rollouts.
    :return: A list of ``(matched_trajectory_name, feature-reduced MAE)`` pairs.
    """
    testtime_rollouts_dir = os.path.join(epoch_dir, "testtime_rollouts")
    if not os.path.isdir(testtime_rollouts_dir):
        return []

    expected_metric_subdir = get_metric_sub_dir(
        cfg.show.compounded_predictions_score, cfg.show.target_is_ood
    )

    per_rollout: list[tuple[str, np.ndarray]] = []
    for traj_name in sorted(os.listdir(testtime_rollouts_dir)):
        ttrpm_path = os.path.join(testtime_rollouts_dir, traj_name, ttrpm_dir)
        if not os.path.isdir(ttrpm_path):
            continue
        # Skip trajectories that don't match the requested target_is_ood /
        # compounded selection (InD-only trees lack the OOD subdir, etc.).
        if not os.path.isdir(os.path.join(ttrpm_path, expected_metric_subdir)):
            continue

        # Pair to a GT trajectory by directory basename (or its ``.``->``p``
        # spelling handled by ``_build_gt_cum_length_map``).
        matched = None
        if traj_name in gt_cum_length_map:
            matched = traj_name
        else:
            _alt = os.path.basename(str(traj_name))
            if _alt in gt_cum_length_map:
                matched = _alt
        if matched is None:
            continue

        try:
            metric = TestTimeRolloutPredictionMetric.load(
                ttrpm_path,
                compounded_predictions_score=cfg.show.compounded_predictions_score,
                target_is_ood=cfg.show.target_is_ood,
            )
            mae_arr = getattr(metric, "mae", None)
        except Exception:
            # Legacy-safe: a malformed / unreadable rollout is skipped.
            continue

        if mae_arr is None or not hasattr(mae_arr, "shape") or mae_arr.size == 0:
            continue
        reduced_mae = np.nanmean(mae_arr, axis=-1) if mae_arr.ndim > 1 else mae_arr
        per_rollout.append((matched, np.asarray(reduced_mae, dtype=float)))

    return per_rollout


def _normalize_length_cutoffs(length_cutoff: Any) -> list[int | None]:
    """Normalise a ``length_cutoff`` config value into an ordered horizon list.

    Accepted forms (any :mod:`omegaconf` proxy is handled transparently):

    * ``None`` -> ``[None]`` (auto shortest-trajectory horizon, no file prefix).
    * a single ``int`` -> ``[int]`` (one horizon, legacy behaviour).
    * a ``list``/``tuple`` of ``int`` -> one horizon per element (a SERIES).
    * an INTERVAL mapping ``{start: <int>, stop: <int>, step: <int>}`` ->
      one horizon per ``step`` over ``[start, stop]`` (``stop`` INCLUSIVE when it
      lands on a step boundary), e.g. ``{start: 200, stop: 400, step: 100}`` ->
      ``[200, 300, 400]``. ``step`` defaults to ``1`` and must be non-zero.
    * an interval given as a ``list`` of single-key maps
      (``[{start: 200}, {stop: 400}, {step: 100}]``, YAML ``[start: .., ...]``
      flow syntax) is merged into the same interval mapping before expansion.

    :param length_cutoff: The raw config value.
    :return: An ordered ``list`` of ``int`` horizons (or ``[None]``).
    """

    def _as_interval_mapping(value: Any) -> dict[str, int] | None:
        """Return a ``{start, stop, step}`` dict if ``value`` is an interval."""
        _interval_keys = {"start", "stop", "step"}
        # Direct mapping form (dict / OmegaConf DictConfig).
        if isinstance(value, dict) or hasattr(value, "keys"):
            try:
                _keys = set(value.keys())
            except Exception:
                return None
            if _keys and _keys.issubset(_interval_keys) and "start" in _keys:
                return {str(k): int(value[k]) for k in _keys}
            return None
        # List of single-key maps: ``[{start: ..}, {stop: ..}, {step: ..}]``.
        if isinstance(value, (list, tuple)) or (
            hasattr(value, "__iter__") and not isinstance(value, (str, bytes))
        ):
            _merged: dict[str, int] = {}
            _all_maps = True
            for _item in value:
                if isinstance(_item, dict) or hasattr(_item, "keys"):
                    try:
                        for _k in _item.keys():
                            _merged[str(_k)] = int(_item[_k])
                    except Exception:
                        return None
                else:
                    _all_maps = False
                    break
            if (
                _all_maps
                and _merged
                and set(_merged).issubset(_interval_keys)
                and ("start" in _merged)
            ):
                return _merged
        return None

    if length_cutoff is None:
        return [None]

    _interval = _as_interval_mapping(length_cutoff)
    if _interval is not None:
        _start = int(_interval["start"])
        _stop = int(_interval.get("stop", _start))
        _step = int(_interval.get("step", 1))
        if _step == 0:
            raise ValueError("length_cutoff interval `step` must be non-zero.")
        # Make ``stop`` INCLUSIVE when it lands on a step boundary.
        _stop_inclusive = _stop + (1 if _step > 0 else -1)
        _cutoffs = [int(c) for c in range(_start, _stop_inclusive, _step)]
        return _cutoffs if _cutoffs else [None]

    if isinstance(length_cutoff, (list, tuple)) or (
        # OmegaConf list proxies are not ``list`` instances.
        hasattr(length_cutoff, "__iter__")
        and not isinstance(length_cutoff, (str, bytes))
    ):
        _cutoffs = [int(c) for c in length_cutoff]
        return _cutoffs if _cutoffs else [None]

    return [int(length_cutoff)]


def generate_drift_rate_plot_per_train_epoch(
    cfg: DictConfig,
    groups: dict[Any, Any],
    ground_truth_entries: Any,
    title: str,
    exp_dir_relative_path: Any,
    fill_between_alpha: float,
    plot_linewidth: float,
    headless: bool,
    root_project_path: str | bytes,
    *,
    length_cutoff: int | list[int] | dict[str, int] | None = None,
    trials_label: str = "trials",
    y_axis_in_logscale: bool = False,
    x_axis_in_logscale: bool = False,
    drift_type: str = "timesteps",
    warmup_mode: Any = None,
    warmup_steps: int | None = None,
    dispersion_type: str = "std-err",  # Options: std, std-err
    generate_diaporama: bool = False,
    diaporama_fps: float = 1.0,
    selected_trajectories: Sequence[str] | None = None,
) -> None:
    """Render one or MANY per-TRAIN-EPOCH C-MAE drift-rate plots (RLRP-773 W9).

    Thin dispatcher over :func:`_generate_drift_rate_plot_per_train_epoch_single`
    that honours a ``length_cutoff`` given either as a single ``int`` / ``None``
    (legacy behaviour) or as a ``list[int]`` of horizons. For each requested
    horizon a SEPARATE figure is rendered; when a concrete horizon is used the
    saved file name is prefixed with ``<length_cutoff>_horizon_`` (e.g.
    ``4000_horizon_test_time_models_comparaison_drift_rate_per_train_epoch.png``)
    so multiple horizons do not overwrite one another.

    :param length_cutoff: ``None`` (auto shortest-trajectory horizon), a single
        strictly-positive ``int``, a ``list[int]`` of strictly-positive
        horizons, or an INTERVAL mapping ``{start, stop, step}`` (expanded to one
        horizon per ``step`` over ``[start, stop]``, ``stop`` inclusive) to
        iterate over (see :func:`_normalize_length_cutoffs`). All other
        parameters are forwarded verbatim to
        :func:`_generate_drift_rate_plot_per_train_epoch_single`.
    :param generate_diaporama: When ``True`` AND a horizon SERIES (a
        ``list[int]`` ``length_cutoff``) is rendered, stitch the produced
        per-horizon PNG frames into a slideshow video (``..._diaporama.mp4``)
        via ``ffmpeg``.
    :param diaporama_fps: Frames-per-second of the slideshow video (each frame
        shown for ``1 / fps`` seconds); defaults to ``1.0`` (1 image/second).
    :param selected_trajectories: Resolved ``show.selected_trajectories``
        (RLRP-841). ``None`` keeps every test trajectory; a list restricts
        every ``epoch_<E>`` snapshot and every horizon of the series to those
        trajectories by filtering the ground-truth entries up front. Unknown
        names fail fast with ``ValueError``.
    """
    # RLRP-841: restrict the ground-truth entries ONCE so every horizon /
    # epoch snapshot only pairs the selected trajectories (fail fast on any
    # unknown name).
    ground_truth_entries = select_ground_truth_entries(
        ground_truth_entries,
        selected_trajectories,
        target_is_ood=bool(cfg.show.get("target_is_ood", False)),
        simulator_config_name=cfg.get("simulator_config", None),
        plot_name="generate_drift_rate_plot_per_train_epoch",
    )
    if selected_trajectories is not None:
        consol_msg_universal_one_liner(
            "generate_drift_rate_plot_per_train_epoch: `show.selected_trajectories` "
            f"restricts the drift rate to {len(list(selected_trajectories))} "
            f"trajectory(ies): {list(selected_trajectories)}\n"
        )

    # Normalise ``length_cutoff`` into an ordered list of horizons to iterate
    # over. ``None`` stays a single-element ``[None]`` (auto horizon, no prefix).
    _cutoffs: list[int | None] = _normalize_length_cutoffs(length_cutoff)

    # Collect the produced (ordered) PNG frames so an optional 1-fps slideshow
    # video can be assembled once the whole horizon series is rendered.
    _diaporama_frames: list[str] = []

    for _cutoff in _cutoffs:
        # Concrete horizons get a ``<length_cutoff>_horizon_`` file-name prefix so
        # a series of horizons produces distinct files; the auto (``None``) case
        # keeps the legacy un-prefixed name.
        _prefix = f"{int(_cutoff)}_horizon_" if _cutoff is not None else ""
        _generate_drift_rate_plot_per_train_epoch_single(
            cfg,
            groups,
            ground_truth_entries,
            title,
            exp_dir_relative_path,
            fill_between_alpha,
            plot_linewidth,
            headless,
            root_project_path,
            length_cutoff=_cutoff,
            trials_label=trials_label,
            y_axis_in_logscale=y_axis_in_logscale,
            x_axis_in_logscale=x_axis_in_logscale,
            drift_type=drift_type,
            warmup_mode=warmup_mode,
            warmup_steps=warmup_steps,
            dispersion_type=dispersion_type,
            filename_prefix=_prefix,
            diaporama_mode=True if generate_diaporama else False,
            selected_trajectories=selected_trajectories,
        )
        # Reconstruct the standard (non-LaTeX) PNG path saved by
        # :func:`show_and_save_plot_helper` (matplotlib appends ``.png``).
        _diaporama_frames.append(
            os.path.join(
                str(exp_dir_relative_path),
                f"{_prefix}test_time_models_comparaison_drift_rate"
                "_per_train_epoch.png",
            )
        )

    # Optionally stitch the rendered per-horizon frames into a slideshow video.
    if generate_diaporama and len(_diaporama_frames) > 1:
        _assemble_png_diaporama(
            _diaporama_frames,
            os.path.join(
                str(exp_dir_relative_path),
                "test_time_models_comparaison_drift_rate"
                "_per_train_epoch_diaporama.mp4",
            ),
            fps=diaporama_fps,
        )
    return None


def _generate_drift_rate_plot_per_train_epoch_single(
    cfg: DictConfig,
    groups: dict[Any, Any],
    ground_truth_entries: Any,
    title: str,
    exp_dir_relative_path: Any,
    fill_between_alpha: float,
    plot_linewidth: float,
    headless: bool,
    root_project_path: str | bytes,
    *,
    length_cutoff: int | None = None,
    trials_label: str = "trials",
    y_axis_in_logscale: bool = False,
    x_axis_in_logscale: bool = False,
    drift_type: str = "timesteps",
    warmup_mode: Any = None,
    warmup_steps: int | None = None,
    dispersion_type: str = "std-err",  # Options: std, std-err
    filename_prefix: str = "",
    diaporama_mode: bool = False,
    selected_trajectories: Sequence[str] | None = None,
) -> None:
    """Render the per-TRAIN-EPOCH C-MAE drift-rate plot (RLRP-773 W9).

    A NEW, opt-in companion to :func:`generate_drift_rate_plot`. Instead of a
    per-timestep curve, this plots ONE point per discovered training-epoch
    snapshot: the X axis is the TRAINING EPOCH ``E`` (one point/series per
    ``epoch_<E>`` directory found under ``<experiment_base>/epoch_checkpoints_rollouts/``,
    ordered ascending), and the Y axis is the SAME C-MAE drift-rate metric used
    by :func:`generate_drift_rate_plot`, reduced to a per-epoch scalar.

    **Per-epoch scalar reduction** (documented assumption): for each epoch the
    per-rollout drift-rate curve is computed against the matching GT trajectory
    with the shared compute core (:func:`_compute_single_rollout_drift`), and the
    scalar is the drift-rate value AT THE AGGREGATION HORIZON END
    (``drift[max_len - 1]``, i.e. the terminal / cumulative drift up to the
    horizon), then averaged across that epoch's rollouts. The horizon ``max_len``
    is resolved exactly as in :func:`generate_drift_rate_plot` (shortest pooled
    trajectory when ``length_cutoff is None``; fixed to ``length_cutoff`` when set).

    **ERLL pass boundaries** (vertical markers): the epoch-``E`` snapshot holds
    LAST-EPOCH weights (``weights_provenance == "last_epoch"``), so it does NOT
    coincide with the deployed BEST-metric model; the series is therefore
    DISCONTINUOUS at ERLL pass boundaries (best-weight rewind, RLRP-773 R11).
    Pass boundaries are inferred best-effort from the SAVE-side
    ``epoch_checkpoints/epoch_<E>/checkpoint_meta.json`` ``erll_pass`` field (read
    via :func:`persistent_checkpoint_utils.read_epoch_checkpoint_meta`) and drawn
    as dashed vertical markers; when ``erll_pass`` is unresolvable the markers are
    skipped gracefully — the plot never crashes. The best-weight-rewind /
    discontinuity caption line is only added when the plotted run(s) actually
    perform MULTIPLE ERLL passes with training on each — i.e. at least one plotted
    experiment's saved ``.hydra/config.yaml`` has ``UDER.uder_num_epochs > 1`` and
    ``UDER.num_epochs_train_model != 0`` (a single-pass or no-train run has no pass
    boundaries, so the caveat is dropped); when no experiment config is readable the
    caption is kept (prior behaviour).

    Legacy-safe: when NO ``epoch_checkpoints_rollouts/`` tree exists under any
    resolved ``experiment_base`` the function is a NO-OP (logs one line, returns).

    :param cfg: Top-level multirun plot configuration.
    :param groups: Per-group metric dict from :func:`compute_per_group_metric`
        (used to know which groups are plotted / their display names).
    :param ground_truth_entries: List of ``TestTrajectoryEntry`` produced by
        ``_load_ground_truth_entries``; used to pair each rollout with its GT
        cumulative path-length (see :func:`_build_gt_cum_length_map`).
    :param title: Base title string (already enriched by :func:`execute`).
    :param exp_dir_relative_path: Where to save the figure.
    :param fill_between_alpha: Alpha for the ±std band across an epoch's rollouts.
    :param plot_linewidth: Line width for group curves.
    :param headless: Headless backend flag.
    :param root_project_path: Project root used to resolve each group's
        ``experiment_base`` directories (same resolution as the pipeline).
    :param length_cutoff: Minimum-trajectory-length selector / horizon (RLRP-734
        semantics, mirroring :func:`generate_drift_rate_plot`).
    :param trials_label: User-facing word substituted for ``"trials"``.
    :param y_axis_in_logscale: If True, sets the y-axis in log scale.
    :param x_axis_in_logscale: If True, sets the (training-epoch) x-axis in log
        scale, mirroring :func:`generate_drift_rate_plot`.
    :param drift_type: ``"timesteps"`` (default) or ``"path_length"`` — same
        metric definition as :func:`generate_drift_rate_plot`.
    :param dispersion_type: Type of dispersion measure used for the band around
        the per-epoch mean; ``"std"`` (standard deviation) or ``"std-err"``
        (standard error, ``std / sqrt(n_trials)``), mirroring
        :func:`generate_drift_rate_plot`.
    :param warmup_mode: Ground-truth-feed warm-up treatment mode.
    :param warmup_steps: Auto-derived warm-up step count (or ``None``).
    :param selected_trajectories: The active ``show.selected_trajectories``
        (RLRP-841), used for ANNOTATION only (title note, unmatched warning);
        the filtering itself happens upstream on ``ground_truth_entries``.
    :return: ``None``. Saves the figure to ``exp_dir_relative_path`` (no-op when
        no per-epoch rollout tree exists).
    """
    # Deferred import (path/IO helpers own the on-disk layout — RLRP-773).
    from tools.mbrl_lib_tools import persistent_checkpoint_utils

    if not ground_truth_entries:
        consol_msg_universal_one_liner(
            "generate_drift_rate_plot_per_train_epoch: no ground-truth entries; skipping.\n"
        )
        plt.close("all")
        return None
    if drift_type not in ["timesteps", "path_length"]:
        raise ValueError(
            f"Invalid drift_type: {drift_type}. Options: 'timesteps', 'path_length'"
        )
    if length_cutoff is not None:
        if isinstance(length_cutoff, bool) or not isinstance(length_cutoff, int):
            raise ValueError(
                f"Invalid `length_cutoff` value: {length_cutoff!r}. "
                "Must be null or a strictly-positive integer."
            )
        if length_cutoff <= 0:
            raise ValueError(
                f"Invalid `length_cutoff` integer: {length_cutoff}. "
                "Must be strictly positive."
            )

    gt_cum_length_map = _build_gt_cum_length_map(ground_truth_entries)
    if not gt_cum_length_map:
        consol_msg_universal_one_liner(
            "generate_drift_rate_plot_per_train_epoch: no usable GT entries; skipping.\n"
        )
        plt.close("all")
        return None

    # .... Resolve per-group experiment trees .................................
    group_sources = _resolve_group_epoch_rollout_sources(cfg, groups, root_project_path)
    if not group_sources:
        consol_msg_universal_one_liner(
            "generate_drift_rate_plot_per_train_epoch: no groups resolved; skipping.\n"
        )
        plt.close("all")
        return None

    _epoch_dir_re = re.compile(r"^epoch_(\d+)$")

    # .... Discover per-epoch rollouts (first pass) ...........................
    # Collect, per group, ``{epoch: [(tname, reduced_mae), ...]}`` across every
    # experiment_base, and track pooled lengths so the horizon can be resolved
    # exactly like the sibling per-model drift plot.
    per_group_epoch_rollouts: dict[Any, dict[int, list[tuple[str, np.ndarray]]]] = {}
    # ``{experiment_base: True}`` — used only to decide the legacy no-op.
    _any_epoch_tree = False
    pool_lengths: list[int] = []
    _prefilter_lengths: list[int] = []
    # ``{experiment_base: {epoch: erll_pass}}`` for pass-boundary markers.
    erll_pass_by_epoch: dict[int, int] = {}

    for grp_key, src in group_sources.items():
        ttrpm_dir = src["ttrpm_dir"]
        epoch_rollouts: dict[int, list[tuple[str, np.ndarray]]] = {}
        for experiment_base in src["experiment_bases"]:
            rollouts_root = persistent_checkpoint_utils.epoch_checkpoint_rollouts_root(
                experiment_base
            )
            if not os.path.isdir(rollouts_root):
                continue
            _any_epoch_tree = True

            for name in sorted(os.listdir(rollouts_root)):
                match = _epoch_dir_re.match(name)
                if match is None:
                    continue
                epoch = int(match.group(1))
                epoch_dir = os.path.join(rollouts_root, name)

                per_rollout = _load_epoch_rollout_reduced_maes(
                    cfg, epoch_dir, ttrpm_dir, gt_cum_length_map
                )
                if not per_rollout:
                    continue

                _kept: list[tuple[str, np.ndarray]] = []
                for tname, reduced_mae in per_rollout:
                    gt_len_arr = gt_cum_length_map[tname]
                    effective_len = min(reduced_mae.shape[0], gt_len_arr.shape[0])
                    if effective_len <= 0:
                        continue
                    _prefilter_lengths.append(effective_len)
                    if length_cutoff is not None and effective_len < length_cutoff:
                        continue
                    _kept.append((tname, reduced_mae[:effective_len]))
                    pool_lengths.append(effective_len)
                if _kept:
                    epoch_rollouts.setdefault(epoch, []).extend(_kept)

                # Best-effort ERLL pass provenance (SAVE-side manifest).
                try:
                    _save_ckpt_dir = persistent_checkpoint_utils.epoch_checkpoint_dir(
                        experiment_base, epoch
                    )
                    _meta = persistent_checkpoint_utils.read_epoch_checkpoint_meta(
                        _save_ckpt_dir
                    )
                    _pass = _meta.get("erll_pass", None)
                    if _pass is not None:
                        erll_pass_by_epoch.setdefault(epoch, int(_pass))
                except Exception:
                    pass

        if epoch_rollouts:
            per_group_epoch_rollouts[grp_key] = epoch_rollouts

    # RLRP-841: warn about selected trajectories that matched no per-epoch
    # rollout in any plotted group.
    _unmatched_selection = _unmatched_selected_trajectories(
        selected_trajectories,
        [
            each_tname
            for each_epoch_rollouts in per_group_epoch_rollouts.values()
            for each_rollouts in each_epoch_rollouts.values()
            for each_tname, _ in each_rollouts
        ],
    )
    if _unmatched_selection:
        consol_msg_universal_one_liner(
            "generate_drift_rate_plot_per_train_epoch: WARNING "
            "`show.selected_trajectories` entry(ies) matched no per-epoch rollout "
            f"in any plotted group: {_unmatched_selection}\n"
        )

    # Legacy no-op: no ``epoch_checkpoints_rollouts/`` anywhere.
    if not _any_epoch_tree:
        consol_msg_universal_one_liner(
            "generate_drift_rate_plot_per_train_epoch: no "
            "'epoch_checkpoints_rollouts/' tree found; no-op.\n"
        )
        plt.close("all")
        return None
    if not per_group_epoch_rollouts or not pool_lengths:
        if length_cutoff is not None and _prefilter_lengths:
            consol_msg_universal_one_liner(
                "generate_drift_rate_plot_per_train_epoch: no trajectory satisfies "
                f"length >= length_cutoff={length_cutoff} "
                f"(longest available={max(_prefilter_lengths)}); skipping.\n"
            )
        else:
            consol_msg_universal_one_liner(
                "generate_drift_rate_plot_per_train_epoch: no per-epoch rollout "
                "matched any GT trajectory; skipping.\n"
            )
        plt.close("all")
        return None

    # .... Resolve horizon (RLRP-734, same rule as generate_drift_rate_plot) ...
    shortest_traj_len = int(min(pool_lengths))
    max_len = int(length_cutoff) if length_cutoff is not None else shortest_traj_len
    if max_len <= 0:
        plt.close("all")
        return None
    timesteps = np.arange(1, max_len + 1, dtype=float)

    if drift_type == "timesteps":
        scaling = 1.0
    else:  # path_length
        scaling = 100.0

    # .... Per-epoch scalar reduction (second pass) ...........................
    # For each (group, epoch): compute per-rollout drift curves with the shared
    # core, take the drift value at the horizon end, and aggregate across the
    # epoch's rollouts (mean + std for the band).
    per_group_series: dict[Any, dict[str, Any]] = {}
    for grp_key, epoch_rollouts in per_group_epoch_rollouts.items():
        epochs_sorted = sorted(epoch_rollouts.keys())
        means: list[float] = []
        stds: list[float] = []
        kept_epochs: list[int] = []
        # Raw per-epoch terminal drifts across models/seeds (RLRP-807). Kept
        # ragged (one array per kept epoch) so the log-scale percentile /
        # log-normal band can be recomputed per epoch without assuming a
        # rectangular (n_models, n_epochs) stack.
        terminal_drifts_per_epoch: list[np.ndarray] = []
        n_rollouts_total = 0
        _retained_tnames: set[str] = set()
        for epoch in epochs_sorted:
            terminal_drifts: list[float] = []
            _nonfinite_tnames: list[str] = []
            for tname, reduced_mae in epoch_rollouts[epoch]:
                if reduced_mae.shape[0] < max_len:
                    continue
                _reduced = reduced_mae
                # Mirror ``_reset_cumulative_error_at_warmup`` (RLRP-736) so the
                # per-epoch scalar matches the sibling plot's discard treatment.
                if (
                    warmup_mode == "discard"
                    and warmup_steps is not None
                    and int(warmup_steps) > 0
                ):
                    _n = min(int(warmup_steps), int(_reduced.shape[0]))
                    if _n > 0:
                        _reduced = np.array(_reduced, dtype=float, copy=True)
                        _reduced[:_n] = 0.0
                drift = _compute_single_rollout_drift(
                    _reduced,
                    gt_cum_length_map[tname],
                    timesteps,
                    max_len,
                    drift_type=drift_type,
                    warmup_mode=warmup_mode,
                    warmup_steps=warmup_steps,
                    cfg=cfg,
                    y_axis_in_logscale=y_axis_in_logscale,
                )
                # Non-finite terminal drift (``inf`` from a zero/near-zero GT
                # path-length reference, or ``nan`` from ``inf - inf``) would
                # otherwise make the downstream ``np.nanstd`` raise "invalid
                # value encountered in subtract" (``np.nanmean``/``np.nanstd``
                # only ignore ``nan``, NOT ``inf``). Name the offending
                # (group, epoch, trajectory), then clamp the curve to finite
                # proxies so the terminal scalar stays finite.
                if not np.all(np.isfinite(drift)):
                    _nonfinite_tnames.append(tname)
                    drift = _make_array_finite(drift)
                _terminal = float(scaling * drift[max_len - 1])
                terminal_drifts.append(_terminal)
                _retained_tnames.add(tname)
            if not terminal_drifts:
                continue
            kept_epochs.append(epoch)
            if _nonfinite_tnames:
                consol_msg_universal_one_liner(
                    "generate_drift_rate_plot_per_train_epoch: non-finite "
                    f"(nan/inf) terminal drift in group '{grp_key}' at epoch "
                    f"{epoch} for {len(_nonfinite_tnames)} trajectory(ies): "
                    f"{sorted(_nonfinite_tnames)}. These were made finite (nan->0, "
                    "+inf->curve max, -inf->curve min) before aggregation.\n"
                )
            # Terminal drifts are now finite (clamped above), so the
            # ``np.nanstd`` RuntimeWarning can no longer fire; the guard is
            # kept as a defensive no-op.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                means.append(float(np.nanmean(terminal_drifts)))
                stds.append(float(np.nanstd(terminal_drifts)))
            terminal_drifts_per_epoch.append(np.asarray(terminal_drifts, dtype=float))
            n_rollouts_total += len(terminal_drifts)
        if kept_epochs:
            per_group_series[grp_key] = {
                "epochs": np.asarray(kept_epochs, dtype=float),
                "means": np.asarray(means, dtype=float),
                "stds": np.asarray(stds, dtype=float),
                "terminal_drifts_per_epoch": terminal_drifts_per_epoch,
                "n_trajectories": len(_retained_tnames),
                "n_total_rollouts": n_rollouts_total,
            }

    if not per_group_series:
        consol_msg_universal_one_liner(
            "generate_drift_rate_plot_per_train_epoch: no per-epoch series "
            "reached the horizon; skipping.\n"
        )
        plt.close("all")
        return None

    # .... Pass-boundary markers (RLRP-773 R11) ...............................
    # A boundary is an epoch whose ``erll_pass`` differs from the previous
    # discovered epoch's. Best-effort: empty when provenance is unresolvable.
    pass_boundary_epochs: list[int] = []
    if erll_pass_by_epoch:
        _sorted_pass_epochs = sorted(erll_pass_by_epoch.keys())
        _prev_pass = None
        for _e in _sorted_pass_epochs:
            _p = erll_pass_by_epoch[_e]
            if _prev_pass is not None and _p != _prev_pass:
                pass_boundary_epochs.append(_e)
            _prev_pass = _p

    # .... Best-weight-rewind / discontinuity caption gate ....................
    # The caption line "epoch-E snapshot = LAST-epoch weights (!= deployed best-metric
    # model); series discontinuous at ERLL pass boundaries" only makes sense for a run
    # that actually performs MULTIPLE ERLL passes with training on each: with a single
    # ERLL pass (``UDER.uder_num_epochs <= 1``) there are no intermediate pass boundaries,
    # and with no inner training (``UDER.num_epochs_train_model == 0``) there is no
    # per-epoch weight evolution to be discontinuous. Read those training knobs from each
    # plotted experiment's saved ``.hydra/config.yaml`` (the plot ``cfg`` is the multirun
    # plot config and does NOT carry ``UDER``). Policy: show the caption when it applies to
    # AT LEAST ONE plotted experiment; if NO experiment config is readable, keep showing it
    # (preserve the prior behaviour rather than silently dropping the caveat).
    _show_pass_boundary_note = True
    _read_any_train_cfg = False
    _any_multi_pass_trained = False
    for _src in group_sources.values():
        for _experiment_base in _src["experiment_bases"]:
            _hydra_cfg_path = os.path.join(_experiment_base, ".hydra", "config.yaml")
            if not os.path.isfile(_hydra_cfg_path):
                continue
            try:
                _exp_cfg = omegaconf.OmegaConf.load(_hydra_cfg_path)
            except Exception:
                continue
            _uder_num_epochs = omegaconf.OmegaConf.select(
                _exp_cfg, "UDER.uder_num_epochs", default=None
            )
            if _uder_num_epochs is None:
                continue
            _num_epochs_train_model = omegaconf.OmegaConf.select(
                _exp_cfg, "UDER.num_epochs_train_model", default=None
            )
            _read_any_train_cfg = True
            # ``num_epochs_train_model: null`` means patience-driven training (still trains),
            # so treat null as "!= 0".
            _trains = (_num_epochs_train_model is None) or (
                int(_num_epochs_train_model) != 0
            )
            if int(_uder_num_epochs) > 1 and _trains:
                _any_multi_pass_trained = True
                break
        if _any_multi_pass_trained:
            break
    if _read_any_train_cfg:
        _show_pass_boundary_note = _any_multi_pass_trained

    # .... Render .............................................................
    with warnings.catch_warnings():
        manage_matplotlib_warnings()
        manage_matplotlib_backend(get_plot_render_type_cfg(cfg).show_plot, headless)

        plt.rcParams.update(get_plot_render_type_cfg(cfg).rcParams_update)

        fig: Figure = plt.figure(
            figsize=get_plot_figsize(
                cfg, "generate_drift_rate_plot.per_train_epoch"
            ),
            dpi=get_plot_render_type_cfg(cfg).figdpi,
        )
        ax: Axes = fig.add_subplot(111)

        color_idx = 0
        all_upper_bounds: list[float] = []
        for grp_key, series in per_group_series.items():
            group = groups[grp_key]
            grp_display_name = group.get("grp_name", grp_key)
            color = COLORS[color_idx % len(COLORS)]
            line_style = group.get("line_style") or "-"

            _gpr_count_label = _format_grp_count_label(
                str(grp_display_name),
                n_trials=group.get("n_trials", 0),
                n_trajectories=series["n_trajectories"],
                n_total_rollouts=series["n_total_rollouts"],
                trials_label=trials_label,
                do_average_across_trajectories=False,
                show_acronym=cfg.show.get("acronym", True),
            )
            if not is_latex_includegraphics_render_type(cfg):
                label = _gpr_count_label
            else:
                label = group.get("gpr_short_name", grp_key)

            x = series["epochs"]
            avg = series["means"]
            std = series["stds"]
            # Mirror :func:`generate_drift_rate_plot`: the band is either the
            # raw standard deviation or the standard error
            # (``std / sqrt(n_trials)`` with N <- nb of models / seeds).
            if dispersion_type == "std":
                disp = std
            elif dispersion_type == "std-err":
                # disp = std / np.sqrt(group.get("n_trials", 1))
                disp = std / np.sqrt(
                    group.get("n_trials", 1) * len(_retained_tnames)
                )  # N <- nb of models (aka seed) X nb of rollout
            else:
                raise ValueError(
                    f"Unknown dispersion_type: {dispersion_type}. "
                    "Options: 'std', 'std-err'"
                )

            # Log-scale-only band recomputation (RLRP-807). On a log y-axis the
            # linear ``avg +/- dispersion`` band flares downward and can violate
            # the strictly-positive support; replace it with an empirical
            # percentile or log-normal/geometric band computed per epoch from the
            # raw per-model terminal drifts. The linear path is untouched.
            _logband_method = cfg.show.generate_drift_rate_plot.get(
                "logscale_band_method", "linear"
            )
            if y_axis_in_logscale and _logband_method != "linear":
                # Reuse the existing ``mae_grp_reduction`` key to pick the
                # ``log_normal`` center (median -> robust, recentered band;
                # otherwise geometric mean), mirroring the horizon plot
                # (RLRP-807, option 1).
                _mae_grp_reduction = cfg.show.generate_drift_rate_plot.get(
                    "mae_grp_reduction", "mean"
                )
                _log_normal_center = (
                    "median" if _mae_grp_reduction == "median" else "mean"
                )
                # Percentile coverage (``percentile`` method only): ``sigma`` ->
                # [16th, 84th], ``iqr`` -> narrower [25th, 75th] (RLRP-807).
                _lower_pct, _upper_pct = _resolve_percentile_band_bounds(cfg)
                _centers: list[float] = []
                _lowers: list[float] = []
                _uppers: list[float] = []
                for _epoch_drifts in series["terminal_drifts_per_epoch"]:
                    _c, _l, _u = _compute_logscale_band(
                        _epoch_drifts.reshape(-1, 1),
                        method=_logband_method,
                        dispersion_type=dispersion_type,
                        n_norm=group.get("n_trials", 1),
                        log_normal_center=_log_normal_center,
                        lower_pct=_lower_pct,
                        upper_pct=_upper_pct,
                    )
                    _centers.append(float(_c[0]))
                    _lowers.append(float(_l[0]))
                    _uppers.append(float(_u[0]))
                avg = np.asarray(_centers, dtype=float)
                lower_curve = np.asarray(_lowers, dtype=float)
                upper_curve = np.asarray(_uppers, dtype=float)
            else:
                lower_curve = avg - disp
                upper_curve = avg + disp

            ax.plot(
                x,
                avg,
                color=color,
                linewidth=plot_linewidth,
                linestyle=line_style,
                marker="o",
                markersize=4,
                label=label,
            )
            ax.fill_between(
                x,
                lower_curve,
                upper_curve,
                color=color,
                alpha=fill_between_alpha,
                linewidth=0,
            )
            # Readability improvement (RLRP-807): opaque band boundaries on top
            # of the fill, gated on ``plot.style.fill_between_linewidth``
            # (default ``0.0`` -> strict no-op), mirroring the horizon plot.
            _fill_between_linewidth = float(
                get_plot_render_type_cfg(cfg).style.get("fill_between_linewidth", 0.0)
            )
            if _fill_between_linewidth > 0.0:
                ax.plot(
                    x,
                    lower_curve,
                    color=color,
                    linewidth=_fill_between_linewidth,
                    # linestyle=line_style,
                    linestyle="solid",
                )
                ax.plot(
                    x,
                    upper_curve,
                    color=color,
                    linewidth=_fill_between_linewidth,
                    # linestyle=line_style,
                    linestyle="solid",
                )
            finite_upper = upper_curve[np.isfinite(upper_curve)]
            if finite_upper.size > 0:
                all_upper_bounds.append(float(np.nanmax(finite_upper)))
            color_idx += 1

        # ERLL pass-boundary vertical markers (discontinuity at best-weight
        # rewind — RLRP-773 R11). Labelled once for the legend.
        SHOW_ERLL_EPOCH_BOUNDARY = False
        if SHOW_ERLL_EPOCH_BOUNDARY:
            _pass_label_done = False
            for _bx in pass_boundary_epochs:
                ax.axvline(
                    float(_bx),
                    color="0.4",
                    # linestyle=":",
                    # linewidth=1.0,
                    linewidth=2.0,
                    label=None if _pass_label_done else "ERLL pass boundary",
                )
                _pass_label_done = True

        _pte_top_ylim = omegaconf.OmegaConf.select(
            cfg, "show.generate_drift_rate_plot.per_train_epoch.top_ylim", default=None
        )
        if _pte_top_ylim is None:
            _pte_top_ylim = omegaconf.OmegaConf.select(
                cfg,
                "show.generate_drift_rate_plot.top_ylim",
                default=None,
            )

        _pte_bottom_ylim = omegaconf.OmegaConf.select(
            cfg,
            "show.generate_drift_rate_plot.per_train_epoch.bottom_ylim",
            default=None,
        )
        if _pte_bottom_ylim is None:
            _pte_bottom_ylim = omegaconf.OmegaConf.select(
                cfg,
                "show.generate_drift_rate_plot.bottom_ylim",
                default=None,
            )

        if y_axis_in_logscale:
            ax.set_yscale("log")
            if _pte_top_ylim is not None:
                ax.set_ylim(top=_pte_top_ylim)
            if _pte_bottom_ylim is not None:
                ax.set_ylim(bottom=_pte_bottom_ylim)
        else:
            if _pte_top_ylim is not None:
                ax.set_ylim(bottom=0, top=_pte_top_ylim)
            elif all_upper_bounds:
                ax.set_ylim(bottom=0, top=float(np.nanmax(all_upper_bounds)) * 1.1)
            else:
                ax.set_ylim(bottom=0)
            if _pte_bottom_ylim is not None:
                ax.set_ylim(bottom=_pte_bottom_ylim)

        # The X axis here is the TRAINING EPOCH (not the rollout timestep), so a
        # log scale simply compresses the epoch axis; mirror the sibling
        # :func:`generate_drift_rate_plot` x-axis handling.
        if x_axis_in_logscale:
            ax.set_xscale("log")

        if y_axis_in_logscale or x_axis_in_logscale:
            ax.minorticks_on()
            ax.grid(True, which="major", linestyle="-", linewidth=0.5)
            ax.grid(True, which="minor", linestyle="--", linewidth=0.25)
        else:
            ax.grid(True)

        if get_plot_render_type_cfg(cfg).print_title:
            _cutoff_note = _selection_note(selected_trajectories) + (
                f"horizon={length_cutoff} (min length>={length_cutoff}), "
                if length_cutoff is not None
                else ""
            )
            # RLRP-773: the best-weight-rewind / discontinuity caveat is only shown for a
            # multi-pass, actually-trained run (see ``_show_pass_boundary_note`` above);
            # single-pass or no-train runs have no pass boundaries so the line is dropped.
            _pass_boundary_note = (
                " epoch-E snapshot = LAST-epoch weights (\u2260 deployed best-metric model); "
                # "series discontinuous at ERLL pass boundaries."
                if _show_pass_boundary_note
                else ""
            )
            _band_desc = _resolve_logscale_band_desc(
                cfg,
                y_axis_in_logscale=y_axis_in_logscale,
                dispersion_type=dispersion_type,
            )
            ax.set_title(
                f"{title}\n"
                f"Per-train-epoch drift rate at horizon ({_cutoff_note}"
                f"shortest available trajectory={shortest_traj_len}).\n"
                f"{_band_desc}"
                f"{_pass_boundary_note}",
            )

        # Make the aggregation horizon stand out with a bold, boxed annotation
        # inside the axes (RLRP: the horizon drives the whole metric, so it must
        # be visually prominent rather than buried in the title text). The
        # warm-up length is stacked UNDER the horizon in the SAME box, but less
        # prominent (small/gray) so the horizon stays dominant.
        if get_plot_render_type_cfg(cfg).print_meta_info:
            _add_horizon_annotation(
                ax, max_len, warmup_steps, cfg, free_running_in_bold=True
            )

        if get_plot_render_type_cfg(cfg).print_legend:
            if diaporama_mode or length_cutoff is not None:
                ax.legend(loc="upper right", framealpha=0.65)
            else:
                ax.legend(loc="upper right")
                # ax.legend(loc="best", framealpha=0.85)

        if get_plot_render_type_cfg(cfg).print_axis_label:
            ax.set_xlabel("Training epoch")
            if drift_type == "timesteps":
                ax.set_ylabel("Drift at horizon (m) [C-MAE / t]")
            else:
                ax.set_ylabel(
                    "Drift rate at horizon (%) [C-MAE / (t \u00b7 GT-travel-len)]"
                )

        if get_plot_render_type_cfg(cfg).print_meta_info:
            display_experiment_name(cfg, fig)

        show_and_save_plot_helper(
            fig,
            exp_dir_relative_path,
            f"{filename_prefix}test_time_models_comparaison_drift_rate_per_train_epoch",
            headless,
            get_plot_render_type_cfg(cfg).show_plot,
            get_plot_render_type_cfg(cfg).save_dpi,
            save=not is_latex_includegraphics_render_type(cfg),
        )

        if is_latex_includegraphics_render_type(cfg):
            project_root = fetch_r2s2r_project_root_path(cfg, lvl_up=1)
            if cfg.show.compounded_predictions_score:
                rollout_type_name = "CP"
            else:
                rollout_type_name = "GT"
            if cfg.show.target_is_ood:
                rollout_type_name = f"OOD_{rollout_type_name}"
            else:
                rollout_type_name = f"IND_{rollout_type_name}"

            show_and_save_plot_helper(
                fig,
                os.path.join(project_root, cfg.latex_includegraphics.save_path),
                f"{filename_prefix}{cfg.latex_includegraphics.file_name}_drift_rate_per_train_epoch_{rollout_type_name}",
                headless,
                False,
                get_plot_render_type_cfg(cfg).latex_save_dpi,
                save=True,
            )

    plt.close("all")
    return None
