# coding=utf-8
"""Training wall-clock time consolidation (RLRP-818).

Exports the SUMMARY series computed by ``_generate_wall_clock_time_barplot``
as one row per group: the per-trial mean (the rendered bar height) and the
per-trial std (computed by the plotter as a symmetric error bar, currently
not drawn -- the CSV therefore carries slightly MORE than the figure shows,
never less).

Sample granularity (RLRP-818 CR10): ``training_wall_clock_time`` is a property
of the TRAINED MODEL, so the sample pool holds exactly ONE value per trial
(group seed) -- ``n_samples == n_trials``, NOT ``n_trials x n_trajectories``.
The de-duplication happens upstream, in ``_collect_wall_clock_samples``, so the
figure and the CSV always share the very same pool.

Matplotlib-free by design: the call site passes plain sequences.
"""
from typing import Any, Mapping, Optional, Sequence

from omegaconf import DictConfig

from tools.experiment_data_consolidation.consolidation_paths import (
    build_grp_name_slug_map,
    resolve_consolidated_data_dir,
)
from tools.experiment_data_consolidation.consolidation_schema import (
    METRIC_TRAINING_WALL_CLOCK_TIME,
    SCHEMA_VERSION,
    TRAINING_WALL_CLOCK_TIME_COLUMNS,
)
from tools.experiment_data_consolidation.csv_writer import (
    list_metric_csv_files,
    write_group_csv,
)
from tools.experiment_data_consolidation.meta_writer import (
    write_experiment_meta_file,
    write_metric_meta_file,
)


def build_training_wall_clock_time_row(
    *,
    grp_name: Any,
    gpr_short_name: Any,
    samples: Sequence[float],
    mean: float,
    std: float,
    unit: str = "h",
) -> dict:
    """Build the consolidated summary row of ONE group.

    ``mean`` / ``std`` are the values the bar plot actually renders and are
    therefore passed in verbatim rather than recomputed here; ``min_h`` /
    ``max_h`` / ``n_samples`` are derived from the per-trial ``samples``.

    :param grp_name: Raw group name (preserved verbatim).
    :param gpr_short_name: The group's short/acronym name.
    :param samples: The per-trial wall-clock times, in ``unit``. ONE value per
        trial (trained model / group seed), hence ``n_samples == n_trials``.
    :param mean: The rendered bar height.
    :param std: The plotter's symmetric error-bar half-width.
    :param unit: The time unit of every value. Defaults to ``"h"`` (hours).
    :return: The row mapping.
    :raise ValueError: When ``samples`` is empty.
    """
    _samples = [float(each) for each in samples]
    if not _samples:
        raise ValueError(
            "build_training_wall_clock_time_row: `samples` must not be empty "
            f"(group {grp_name!r})."
        )

    return {
        "grp_name": grp_name,
        "gpr_short_name": gpr_short_name,
        "mean_h": float(mean),
        "std_h": float(std),
        "n_samples": len(_samples),
        "min_h": min(_samples),
        "max_h": max(_samples),
        "unit": unit,
        "schema_version": SCHEMA_VERSION,
    }


def consolidate_training_wall_clock_time(
    cfg: DictConfig,
    group_rows: Mapping[Any, Mapping[str, Any]],
    *,
    file_stem: str,
    params: Mapping[str, Any],
    source_experiments: Optional[Mapping[Any, Any]] = None,
) -> list:
    """Write one training wall-clock CSV per group plus both meta files.

    :param cfg: Top-level multirun plot configuration.
    :param group_rows: ``{grp_name: row}`` as built by
        :func:`build_training_wall_clock_time_row`.
    :param file_stem: The producer's PNG file stem (without extension).
    :param params: Metric parameters recorded in the metric-level meta file.
    :param source_experiments: ``{grp_name: <source paths description>}``.
    :return: The absolute paths of the written CSV files.
    """
    metric_dir = resolve_consolidated_data_dir(cfg, METRIC_TRAINING_WALL_CLOCK_TIME)
    slug_map = build_grp_name_slug_map(list(group_rows.keys()))

    written = []
    for each_grp_name, each_row in group_rows.items():
        written.append(
            write_group_csv(
                metric_dir,
                slug_map[each_grp_name],
                file_stem,
                TRAINING_WALL_CLOCK_TIME_COLUMNS,
                [each_row],
            )
        )

    write_metric_meta_file(
        metric_dir,
        cfg,
        metric=METRIC_TRAINING_WALL_CLOCK_TIME,
        producer="generate_fct._generate_wall_clock_time_barplot",
        hook="src/pipeline/pipeline_utils/multirun_testtime_rollout_plot_pipeline_utils/"
        "generate_fct.py: after the per-group mean/std reduction, before the "
        "matplotlib context",
        params=params,
        source_experiments=source_experiments or {},
        csv_files=list_metric_csv_files(metric_dir),
        grp_name_slug_map=slug_map,
        columns=TRAINING_WALL_CLOCK_TIME_COLUMNS,
    )
    write_experiment_meta_file(
        resolve_consolidated_data_dir(cfg),
        cfg,
        metric_subdir=METRIC_TRAINING_WALL_CLOCK_TIME,
    )
    return written
