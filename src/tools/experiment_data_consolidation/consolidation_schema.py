# coding=utf-8
"""Consolidated experiment data CSV schema (RLRP-818).

Single source of truth for the CSV column layout of every consolidated
metric. The standalone ICRA 2026 plot functions (RLRP-817) consume this
contract, hence any column addition MUST bump :data:`SCHEMA_VERSION`.

Every emitted row carries the ``schema_version`` column so a consumer can
detect the layout it is reading without parsing the meta files -- with the
single, documented exception of the ``drift_rate_samples`` family, whose wide
header is data-dependent and which records the version in its ``meta.txt``
only (RLRP-819 action ``W1``).
"""

# Bump on ANY column layout change (added/removed/renamed/reordered column).
# 2.0 -> 2.1: additive metric family (RLRP-819)
# 2.1 -> 2.2: `drift_rate_samples` re-laid out WIDE, one CSV per model seed
#             (RLRP-819 action `W1`, ruling `Q12`)
SCHEMA_VERSION = "2.2"

# Metric sub-directory names (also used as the metric identifier).
METRIC_DRIFT_RATE = "drift_rate"
METRIC_DRIFT_RATE_SAMPLES = "drift_rate_samples"
METRIC_TRAINING_WALL_CLOCK_TIME = "training_wall_clock_time"
METRIC_INFERENCE_BENCHMARK = "inference_benchmark"

# .... Drift rate .............................................................
# One row per plotted timestep, per group. Carries only the core numeric
# coordinates: integer timestep, float timestamp (when resolvable), the
# plotted drift rate curve (ax.plot) and the rendered lower/upper dispersion
# bounds (ax.fill_between). All static experiment/group metadata is consolidated
# in the metric-level `meta.txt`.
DRIFT_RATE_COLUMNS = (
    "timestep",
    "timestamp",
    "drift_rate",
    "drift_rate_lower",
    "drift_rate_upper",
)

# .... Drift rate samples .....................................................
# Introduced by action `R1` of the RLRC ICRA2026 fig-2 / fig-3 standalone plots
# `.junie` plan
# (`feat_RLRP-819_two_new_standalone_plots_fig2_fig3_plan_20260910.md`).
# The RAW per-rollout drift population behind the reduced `drift_rate` curve:
# NO reduction and NO dispersion band. A SEPARATE metric family -- never extra
# columns of `drift_rate` -- because `write_metric_meta_file` renders exactly
# one `csv_columns:` section per metric directory, and because the sample
# table must never be confused with the reduced table by a reader. Purely
# additive: the `drift_rate` family stays byte-identical for anyone who does
# not opt in.
# Action `W1` (ruling `Q12`) REPLACED the v5 long format by a WIDE one, one
# CSV per model seed: `timestep`, `timestamp`, then ONE COLUMN PER TRAJECTORY
# named by its small integer trajectory id. The header is therefore
# DATA-DEPENDENT and is built by :func:`build_drift_rate_samples_columns`;
# only the fixed prefix below is a constant. `schema_version` LEFT the CSV
# (a wide, data-dependent header has no room for it) and lives in the metric
# `meta.txt` only -- precedent: the `drift_rate` family already carries no
# `schema_version` column.
DRIFT_RATE_SAMPLES_FIXED_COLUMNS = (
    "timestep",
    "timestamp",  # EMPTY when the conversion is unresolvable (e.g. Lorenz)
)

# .... Training wall-clock time ...............................................
# Summary format (one row per group) -- see plan Q5.
TRAINING_WALL_CLOCK_TIME_COLUMNS = (
    "grp_name",
    "gpr_short_name",
    "mean_h",
    "std_h",
    "n_samples",
    "min_h",
    "max_h",
    "unit",
    "schema_version",
)

# .... Inference benchmark ....................................................
# One row per (model, series) pair of the plotted figure.
INFERENCE_BENCHMARK_COLUMNS = (
    "model",
    "model_config",
    "device_label",
    "level",
    "regime",
    "timing_pass",
    "metric",
    "value",
    "value_unit",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_std_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "box_median",
    "box_q1",
    "box_q3",
    "box_whislo",
    "box_whishi",
    "box_mean",
    "box_from_raw_samples",
    "schema_version",
)


def build_drift_rate_samples_columns(trajectory_ids) -> tuple:
    """Build the RESOLVED wide header of one ``drift_rate_samples`` CSV.

    Action ``W1`` of the RLRC ICRA2026 fig-2 / fig-3 standalone plots
    ``.junie`` plan
    (``feat_RLRP-819_two_new_standalone_plots_fig2_fig3_plan_20260910.md``).

    The header is :data:`DRIFT_RATE_SAMPLES_FIXED_COLUMNS` followed by one
    column per trajectory id, in ASCENDING id order, so the file layout is
    deterministic whatever the insertion order of the mapping. The id -> name
    mapping itself lives in the metric ``meta.txt``
    (``param.trajectory_name_mapping``).

    :param trajectory_ids: The integer trajectory ids of the EXPERIMENT.
    :return: The ordered column-name tuple of that CSV.
    """
    return DRIFT_RATE_SAMPLES_FIXED_COLUMNS + tuple(
        str(each_id) for each_id in sorted(int(each) for each in trajectory_ids)
    )


METRIC_COLUMNS = {
    METRIC_DRIFT_RATE: DRIFT_RATE_COLUMNS,
    # ⚠️ The FIXED PREFIX only -- see `fetch_metric_columns`.
    METRIC_DRIFT_RATE_SAMPLES: DRIFT_RATE_SAMPLES_FIXED_COLUMNS,
    METRIC_TRAINING_WALL_CLOCK_TIME: TRAINING_WALL_CLOCK_TIME_COLUMNS,
    METRIC_INFERENCE_BENCHMARK: INFERENCE_BENCHMARK_COLUMNS,
}


def fetch_metric_columns(metric: str) -> tuple:
    """Return the CSV column tuple of ``metric``.

    ⚠️ :data:`METRIC_DRIFT_RATE_SAMPLES` is the ONE metric whose header is
    DYNAMIC (action ``W1``, risk ``W-S4``): what is returned here is only the
    FIXED PREFIX ``("timestep", "timestamp")``. The complete, resolved header
    of a given CSV is built by :func:`build_drift_rate_samples_columns` from
    the trajectory ids of the experiment, and is rendered verbatim in the
    metric ``meta.txt`` ``csv_columns`` section.

    :param metric: One of the ``METRIC_*`` identifiers of this module.
    :return: The ordered column-name tuple of that metric.
    :raise KeyError: When ``metric`` is not a known consolidated metric.
    """
    try:
        return METRIC_COLUMNS[metric]
    except KeyError as e:
        raise KeyError(
            f"Unknown consolidated metric '{metric}'. "
            f"Supported: {sorted(METRIC_COLUMNS.keys())}"
        ) from e
