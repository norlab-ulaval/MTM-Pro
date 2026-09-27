# coding=utf-8
"""Experiment data consolidation (RLRP-818).

Exports the numeric series a figure renders into version-controlled CSV
files so the ICRA 2026 standalone plot functions (RLRP-817) can rebuild the
identical figure without re-running the pipeline.

Opt-in per run via the ``consolidate_data.enable`` config flag; the
destination is
``data/data_consolidated/<overrides.experiment>/<metric>/<grp_name>/<png_stem>.csv``
(see ``README.md`` of this package for the complete layout and schema).

Public surface:
  * :mod:`~tools.experiment_data_consolidation.consolidation_schema` -- the
    CSV column contract consumed by RLRP-817.
  * :mod:`~tools.experiment_data_consolidation.consolidation_paths` -- output
    directory resolution + ``grp_name`` slugification.
  * :mod:`~tools.experiment_data_consolidation.csv_writer` /
    :mod:`~tools.experiment_data_consolidation.meta_writer` -- the writers.
  * ``*_consolidation`` modules -- one pure ``build_*_rows`` row builder plus
    one ``consolidate_*`` orchestrator per producer. The
    ``drift_rate_samples`` family (RLRP-819 ``W1``/``W2``) is the one metric
    with a DYNAMIC, data-dependent header: it builds WIDE per-seed tables
    through ``build_drift_rate_seed_tables`` and resolves its columns with
    :func:`~tools.experiment_data_consolidation.consolidation_schema.build_drift_rate_samples_columns`.
  * :func:`~tools.experiment_data_consolidation.consolidation_guard.consolidation_guard`
    -- failure isolation: an export error never breaks plot generation.
"""
from tools.experiment_data_consolidation.consolidation_guard import consolidation_guard
from tools.experiment_data_consolidation.consolidation_paths import (
    build_grp_name_slug_map,
    is_consolidation_enabled,
    resolve_consolidated_data_dir,
    slugify_grp_name,
)
from tools.experiment_data_consolidation.consolidation_schema import (
    DRIFT_RATE_SAMPLES_FIXED_COLUMNS,
    METRIC_DRIFT_RATE,
    METRIC_DRIFT_RATE_SAMPLES,
    METRIC_INFERENCE_BENCHMARK,
    METRIC_TRAINING_WALL_CLOCK_TIME,
    SCHEMA_VERSION,
    build_drift_rate_samples_columns,
    fetch_metric_columns,
)
from tools.experiment_data_consolidation.csv_writer import write_group_csv
from tools.experiment_data_consolidation.meta_writer import (
    write_experiment_meta_file,
    write_metric_meta_file,
)

__all__ = [
    "SCHEMA_VERSION",
    "METRIC_DRIFT_RATE",
    "METRIC_DRIFT_RATE_SAMPLES",
    "METRIC_TRAINING_WALL_CLOCK_TIME",
    "METRIC_INFERENCE_BENCHMARK",
    "DRIFT_RATE_SAMPLES_FIXED_COLUMNS",
    "build_drift_rate_samples_columns",
    "fetch_metric_columns",
    "is_consolidation_enabled",
    "resolve_consolidated_data_dir",
    "slugify_grp_name",
    "build_grp_name_slug_map",
    "write_group_csv",
    "write_experiment_meta_file",
    "write_metric_meta_file",
    "consolidation_guard",
]
