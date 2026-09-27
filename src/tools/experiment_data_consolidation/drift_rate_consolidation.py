# coding=utf-8
"""Drift-rate consolidation (RLRP-818).

Turns the numeric series that ``_generate_drift_rate_plot_single`` hands to
``ax.plot`` / ``ax.fill_between`` into consolidated CSV rows, then writes one
CSV per group under ``drift_rate/<slug(grp_name)>/<png_file_stem>.csv``.

Matplotlib-free by design: the call site passes plain sequences.

RLRP-819 (actions ``W1``-``W3``, ruling ``Q12``, of the RLRC ICRA2026 fig-2 /
fig-3 standalone plots ``.junie`` plan
``feat_RLRP-819_two_new_standalone_plots_fig2_fig3_plan_20260910.md``) adds a
PURELY ADDITIVE second family, ``drift_rate_samples``: the RAW per-rollout
drift population behind the reduced curve, which is what makes a genuine
boxplot possible. It is stored WIDE -- one CSV per model seed, one row per
timestep of the WHOLE grid, one column per trajectory id -- so ANY downstream
freeze point is reachable without re-running this pipeline. The
``drift_rate`` family above is left byte-identical.
"""
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from omegaconf import DictConfig

from tools.experiment_data_consolidation.consolidation_paths import (
    build_grp_name_slug_map,
    resolve_consolidated_data_dir,
)
from tools.experiment_data_consolidation.consolidation_schema import (
    DRIFT_RATE_COLUMNS,
    METRIC_DRIFT_RATE,
    METRIC_DRIFT_RATE_SAMPLES,
    build_drift_rate_samples_columns,
)
from tools.experiment_data_consolidation.csv_writer import (
    list_metric_csv_files,
    write_group_csv,
)
from tools.experiment_data_consolidation.meta_writer import (
    write_experiment_meta_file,
    write_metric_meta_file,
)

# Physical unit of the plotted drift value, per ``drift_type``.
DRIFT_TYPE_UNIT = {
    "timesteps": "m",  # Avg drift up to t (m) [C-MAE / t]
    "path_length": "%",  # Avg drift rate (%) [C-MAE / (t . GT-travel-len)]
}


def build_drift_rate_rows(
    *,
    timesteps: Sequence[float],
    drift_rate: Sequence[float],
    drift_rate_lower: Sequence[float],
    drift_rate_upper: Sequence[float],
    timestamp_scale: Optional[float] = None,
    timestamp_unit: Optional[str] = None,
    grp_name: Any = None,
    gpr_short_name: Any = None,
    dispersion_type: Any = None,
    logscale_band_method: Any = None,
    drift_type: Any = None,
    n_trials: Any = None,
    n_trajectories: Any = None,
    n_rollouts: Any = None,
    train_epoch: Optional[int] = None,
    length_cutoff: Optional[int] = None,
    warmup_mode: Any = None,
    warmup_steps: Any = None,
) -> list:
    """Build the consolidated rows of ONE group's drift-rate curve.

    One row per plotted timestep. Emits only the core numeric coordinates:
    integer ``timestep``, float ``timestamp``, ``drift_rate``,
    ``drift_rate_lower``, and ``drift_rate_upper``. Static parameters and group
    metadata are consolidated in ``meta.txt``.

    ``timestamp`` is ``timestep * timestamp_scale`` where ``timestamp_scale`` is
    ``seconds_per_step * _SECONDS_TO_UNIT_FACTOR[unit]`` -- i.e. the exact
    conversion the figure's own time annotation uses. ``timestamp`` stays
    EMPTY (``None``) when the conversion is unresolvable.

    :param timesteps: The plotted x values (converted to integer timesteps).
    :param drift_rate: The plotted curve (``ax.plot``).
    :param drift_rate_lower: The rendered band lower bound.
    :param drift_rate_upper: The rendered band upper bound.
    :param timestamp_scale: Timestep -> timestamp multiplicative factor.
    :param timestamp_unit: The resolved timestamp unit (``s``/``ms``/``us``).
    :param grp_name: Unused legacy arg, preserved for compatibility.
    :param gpr_short_name: Unused legacy arg, preserved for compatibility.
    :param dispersion_type: Unused legacy arg, preserved for compatibility.
    :param logscale_band_method: Unused legacy arg, preserved for compatibility.
    :param drift_type: Unused legacy arg, preserved for compatibility.
    :param n_trials: Unused legacy arg, preserved for compatibility.
    :param n_trajectories: Unused legacy arg, preserved for compatibility.
    :param n_rollouts: Unused legacy arg, preserved for compatibility.
    :param train_epoch: Unused legacy arg, preserved for compatibility.
    :param length_cutoff: Unused legacy arg, preserved for compatibility.
    :param warmup_mode: Unused legacy arg, preserved for compatibility.
    :param warmup_steps: Unused legacy arg, preserved for compatibility.
    :return: The list of row mappings, ordered by timestep.
    :raise ValueError: When the series do not share the same length.
    """
    _n = len(timesteps)
    if not (len(drift_rate) == len(drift_rate_lower) == len(drift_rate_upper) == _n):
        raise ValueError(
            "build_drift_rate_rows: `timesteps`, `drift_rate`, `drift_rate_lower` "
            f"and `drift_rate_upper` must share the same length; got "
            f"{_n}, {len(drift_rate)}, {len(drift_rate_lower)}, "
            f"{len(drift_rate_upper)}."
        )

    rows = []
    for each_idx in range(_n):
        _timestep = int(round(float(timesteps[each_idx])))
        if timestamp_scale is None or timestamp_unit is None:
            _timestamp = None
        else:
            _timestamp = float(_timestep * float(timestamp_scale))

        rows.append(
            {
                "timestep": _timestep,
                "timestamp": _timestamp,
                "drift_rate": float(drift_rate[each_idx]),
                "drift_rate_lower": float(drift_rate_lower[each_idx]),
                "drift_rate_upper": float(drift_rate_upper[each_idx]),
            }
        )
    return rows


def resolve_horizon_ratio_timesteps(
    ratios: Sequence[float],
    *,
    timesteps: Sequence[float],
    length_cutoff: Optional[int] = None,
) -> dict:
    """Resolve horizon RATIOS into the timesteps of THIS experiment.

    THE single implementation of the horizon-ratio formula of action ``R2``
    (ruling ``Q11``, §2.4.2) of the RLRC ICRA2026 fig-2 / fig-3 standalone
    plots ``.junie`` plan
    (``feat_RLRP-819_two_new_standalone_plots_fig2_fig3_plan_20260910.md``)::

        warmup_end     = timesteps[0]              # read FROM THE DATA
        horizon_length = length_cutoff - warmup_end
        timestep(r)    = warmup_end + int(math.floor(r * horizon_length + 0.5))

    The three quantities that make the experiments incommensurable -- warm-up
    length, effective rollout length and sample rate -- are exactly the three
    a ratio normalises away, so ONE ratio list is correct for every dataset
    (no per-run editing). ``warmup_end`` comes from the data rather than from
    ``param.warmup_steps`` so it is right whether the producer trimmed
    (``warmup_mode: discard``) or not.

    Ties (an exact ``.5``) round HALF-AWAY-FROM-ZERO, NOT with python's
    banker's :func:`round`: this is the tie rule the standalone consumer
    (``drift_freeze.resolve_horizon_ratio``, action ``E0c``) implements, and
    measure ``N10`` requires the two to be bit-identical.

    The standalone side imports the SAME rule, and the resolved map is
    recorded in ``param.horizon_ratio_map`` so the consumer can VERIFY rather
    than re-derive it.

    :param ratios: The horizon fractions to resolve, each in ``[0.0, 1.0]``.
    :param timesteps: The POST-trim abscissae, i.e. the ``x_p`` the reduced
        CSV uses; ``timesteps[0]`` IS the warm-up boundary.
    :param length_cutoff: The effective rollout cut-off; defaults to
        ``timesteps[-1]`` (the last plotted sample).
    :return: ``{ratio: timestep}``, insertion-ordered as ``ratios``.
    :raise ValueError: When ``timesteps`` is empty or a ratio lies outside
        ``[0.0, 1.0]``.
    """
    if len(timesteps) == 0:
        raise ValueError(
            "resolve_horizon_ratio_timesteps: `timesteps` must not be empty."
        )

    warmup_end = int(round(float(timesteps[0])))
    if length_cutoff is None:
        cutoff = int(round(float(timesteps[-1])))
    else:
        cutoff = int(length_cutoff)
    horizon_length = cutoff - warmup_end

    resolved: dict = {}
    for each_ratio in ratios:
        _ratio = float(each_ratio)
        if not (0.0 <= _ratio <= 1.0):
            raise ValueError(
                "resolve_horizon_ratio_timesteps: a horizon ratio must lie in "
                f"[0.0, 1.0]; got {_ratio}."
            )
        # Half-away-from-zero, bit-identical to `drift_freeze.resolve_horizon_ratio` (`E0c`).
        resolved[_ratio] = warmup_end + int(math.floor(_ratio * horizon_length + 0.5))
    return resolved


def build_trajectory_name_mapping(
    group_trajectory_names: Mapping[Any, Sequence[str]],
) -> dict:
    """Build the EXPERIMENT-wide ``{trajectory_id: trajectory_name}`` mapping.

    Action ``W2`` of the RLRC ICRA2026 fig-2 / fig-3 standalone plots
    ``.junie`` plan
    (``feat_RLRP-819_two_new_standalone_plots_fig2_fig3_plan_20260910.md``).

    The mapping is the SORTED UNION of the trajectory names of every
    consolidated group, numbered from ``1``: it is shared by every group of
    the experiment, so a reader can compare column ``3`` across groups (never
    across experiments -- risk ``W-S5``). A group that never saw a given
    trajectory still HAS its column, filled with EMPTY cells.

    :param group_trajectory_names: ``{grp_name: per-rollout trajectory names}``.
    :return: ``{trajectory_id: trajectory_name}``, ids 1-based and ascending.
    """
    _names = sorted(
        {
            str(each_name)
            for each_names in group_trajectory_names.values()
            for each_name in each_names
        }
    )
    return {_idx + 1: each_name for _idx, each_name in enumerate(_names)}


def build_drift_rate_seed_tables(
    *,
    timesteps: Sequence[float],
    samples: "np.ndarray",
    trajectory_names: Sequence[str],
    seed_keys: Sequence[tuple],
    trajectory_name_mapping: Mapping[int, str],
    timestamp_scale: Optional[float] = None,
    timestamp_unit: Optional[str] = None,
    stride: Optional[int] = None,
    float_format: Optional[str] = None,
) -> tuple:
    """Build the WIDE per-seed sample tables of ONE group.

    Action ``W2`` (ruling ``Q12``) of the RLRC ICRA2026 fig-2 / fig-3
    standalone plots ``.junie`` plan
    (``feat_RLRP-819_two_new_standalone_plots_fig2_fig3_plan_20260910.md``).

    The v5 long format is RETIRED: the population is now split by MODEL SEED
    (one table, hence one CSV, per distinct ``_trial_key``), one row per
    timestep of the WHOLE trimmed abscissa grid, and one column per
    trajectory id. Dumping the full grid is what makes ANY downstream freeze
    point reachable without re-running this pipeline, and makes measure
    ``N12`` checkable at EVERY timestep.

    A ``(seed, trajectory)`` pair with no rollout yields an EMPTY cell --
    never ``0.0``, which would silently inject a fake perfect rollout into
    fig 2's box (risk ``W-S2``) -- and is counted into ``missing_cells``.

    Seed ordinals are 1-based and deterministic: the distinct ``seed_keys``
    sorted LEXICOGRAPHICALLY. ⚠️ The ordinal is NOT the ``trial_nb`` number;
    ``seed_key_mapping`` of the returned provenance is the source of truth
    (risk ``W-S3``).

    :param timesteps: THE SAME ``x_p`` the reduced CSV uses (post warm-up
        trim).
    :param samples: ``(n_rollouts, len(timesteps))`` already trimmed and
        aligned per-rollout drift curves.
    :param trajectory_names: Per-rollout trajectory name, in the ROW order of
        ``samples``.
    :param seed_keys: Per-rollout ``_trial_key``, in the ROW order of
        ``samples``.
    :param trajectory_name_mapping: The EXPERIMENT-wide
        ``{trajectory_id: trajectory_name}`` (see
        :func:`build_trajectory_name_mapping`); it defines the columns.
    :param timestamp_scale: Timestep -> timestamp multiplicative factor.
    :param timestamp_unit: The resolved timestamp unit (``s``/``ms``/``us``).
    :param stride: Size escape hatch -- keep every n-th timestep; ``None`` ->
        the FULL grid (the whole point of the v6 layout).
    :param float_format: ``None`` -> python :func:`repr` (bit-exact
        round-trip); otherwise the given format is applied (ruling ``Q13``),
        e.g. ``'%.9g'``.
    :return: ``({seed_ordinal: rows}, provenance)``.
    :raise ValueError: When ``samples`` is not 2-D, when
        ``samples.shape[1] != len(timesteps)`` (the alignment guard), when an
        identity sequence does not cover every rollout, when a trajectory
        name is absent from ``trajectory_name_mapping``, when ``stride`` is
        not strictly positive, or when the same ``(seed, trajectory)`` pair
        occurs TWICE (the identity would not be a key and keeping the last
        one would silently drop data).
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError(
            "build_drift_rate_seed_tables: `samples` must be a 2-D "
            f"(n_rollouts, n_timesteps) array; got {samples.ndim} dimension(s)."
        )
    _n_timesteps = len(timesteps)
    if samples.shape[1] != _n_timesteps:
        raise ValueError(
            "build_drift_rate_seed_tables: `samples` and `timesteps` are not "
            f"aligned; got samples.shape[1]={samples.shape[1]} for "
            f"{_n_timesteps} timestep(s)."
        )
    _n_rollouts = samples.shape[0]
    if len(trajectory_names) != _n_rollouts:
        raise ValueError(
            "build_drift_rate_seed_tables: `trajectory_names` must cover every "
            f"rollout; got {len(trajectory_names)} name(s) for {_n_rollouts} "
            "rollout(s)."
        )
    if len(seed_keys) != _n_rollouts:
        raise ValueError(
            "build_drift_rate_seed_tables: `seed_keys` must cover every "
            f"rollout; got {len(seed_keys)} key(s) for {_n_rollouts} "
            "rollout(s)."
        )
    if stride is not None and int(stride) <= 0:
        raise ValueError(
            "build_drift_rate_seed_tables: `stride` must be strictly positive; "
            f"got {stride}."
        )

    _name_to_id = {
        str(each_name): int(each_id)
        for each_id, each_name in trajectory_name_mapping.items()
    }
    # 1-based, deterministic seed ordinals: the distinct `_trial_key` tuples,
    # sorted lexicographically (`W0b`). `str` normalises the sort key so a
    # heterogeneous key type can never raise here.
    _distinct_seed_keys = sorted(set(seed_keys), key=lambda each: str(each))
    _seed_ordinal_of = {
        each_key: _idx + 1 for _idx, each_key in enumerate(_distinct_seed_keys)
    }

    # `(seed_ordinal, trajectory_id) -> row index of `samples``, the ONLY
    # place the rollout identity is resolved.
    _cell_source: dict = {}
    for each_rollout_idx in range(_n_rollouts):
        _name = str(trajectory_names[each_rollout_idx])
        if _name not in _name_to_id:
            raise ValueError(
                "build_drift_rate_seed_tables: trajectory "
                f"{_name!r} is absent from `trajectory_name_mapping` "
                f"{sorted(_name_to_id.keys())}."
            )
        _cell_key = (
            _seed_ordinal_of[seed_keys[each_rollout_idx]],
            _name_to_id[_name],
        )
        if _cell_key in _cell_source:
            raise ValueError(
                "build_drift_rate_seed_tables: duplicate (seed, trajectory) "
                f"pair {_cell_key} -- seed key "
                f"{seed_keys[each_rollout_idx]!r}, trajectory {_name!r}. The "
                "rollout identity is not a key; keeping only one of the two "
                "rollouts would silently drop data."
            )
        _cell_source[_cell_key] = each_rollout_idx

    _grid = [int(round(float(each))) for each in timesteps]
    _indices = (
        list(range(_n_timesteps))
        if stride is None
        else list(range(0, _n_timesteps, int(stride)))
    )
    _trajectory_ids = sorted(int(each_id) for each_id in trajectory_name_mapping)

    seed_tables: dict = {}
    missing_cells = 0
    for each_seed_key in _distinct_seed_keys:
        _seed_ordinal = _seed_ordinal_of[each_seed_key]
        rows = []
        for each_idx in _indices:
            _timestep = _grid[each_idx]
            if timestamp_scale is None or timestamp_unit is None:
                _timestamp = None
            else:
                _timestamp = float(_timestep * float(timestamp_scale))

            row = {"timestep": _timestep, "timestamp": _timestamp}
            for each_trajectory_id in _trajectory_ids:
                _rollout_idx = _cell_source.get((_seed_ordinal, each_trajectory_id))
                # ⚠️ A missing `(seed, trajectory)` is an EMPTY cell, NEVER
                #    `0.0` (risk `W-S2`).
                row[str(each_trajectory_id)] = (
                    None
                    if _rollout_idx is None
                    else _format_sample(
                        float(samples[_rollout_idx, each_idx]), float_format
                    )
                )
            rows.append(row)
        seed_tables[_seed_ordinal] = rows
        missing_cells += len(_indices) * (
            len(_trajectory_ids)
            - sum(
                1
                for each_trajectory_id in _trajectory_ids
                if (_seed_ordinal, each_trajectory_id) in _cell_source
            )
        )

    provenance = {
        "trajectory_ids": sorted(
            {each_key[1] for each_key in _cell_source.keys()}
        ),
        "seed_key_mapping": {
            _seed_ordinal_of[each_key]: list(each_key)
            if isinstance(each_key, (tuple, list))
            else each_key
            for each_key in _distinct_seed_keys
        },
        "n_seeds": len(_distinct_seed_keys),
        "n_rollouts": _n_rollouts,
        "missing_cells": missing_cells,
        "n_timesteps": len(_indices),
    }
    return seed_tables, provenance


def _format_sample(value: float, float_format: Optional[str]) -> str:
    """Serialize ONE drift sample with the configured precision (ruling ``Q13``).

    Helper of action ``W2`` (see :func:`build_drift_rate_seed_tables`) of the
    RLRC ICRA2026 fig-2 / fig-3 standalone plots ``.junie`` plan
    (``feat_RLRP-819_two_new_standalone_plots_fig2_fig3_plan_20260910.md``).

    ``None`` -> python :func:`repr`, i.e. the shortest string that round-trips
    BIT-EXACTLY through :class:`float`, which is what keeps measure ``N12``
    an exact equality rather than an approximation.

    :param value: The raw drift sample.
    :param float_format: ``None``, a ``%``-style format (e.g. ``'%.9g'``) or
        a :func:`format` spec (e.g. ``'.9g'``).
    :return: The cell string.
    """
    if float_format is None:
        return repr(float(value))
    if "%" in str(float_format):
        return str(float_format) % float(value)
    return format(float(value), str(float_format))


def consolidate_drift_rate(
    cfg: DictConfig,
    group_rows: Mapping[Any, Sequence[Mapping[str, Any]]],
    *,
    file_stem: str,
    params: Mapping[str, Any],
    source_experiments: Optional[Mapping[Any, Any]] = None,
    group_metadata: Optional[Mapping[str, Any]] = None,
) -> list:
    """Write one drift-rate CSV per group plus both meta files.

    :param cfg: Top-level multirun plot configuration.
    :param group_rows: ``{grp_name: rows}`` as built by
        :func:`build_drift_rate_rows`.
    :param file_stem: The producer's PNG file stem (without extension), e.g.
        ``"500_epoch_test_time_models_comparaison_drift_rate"``.
    :param params: Metric parameters recorded in the metric-level meta file.
    :param source_experiments: ``{grp_name: <source paths description>}``.
    :param group_metadata: ``{grp_name: <metadata dictionary>}``.
    :return: The absolute paths of the written CSV files.
    """
    metric_dir = resolve_consolidated_data_dir(cfg, METRIC_DRIFT_RATE)
    slug_map = build_grp_name_slug_map(list(group_rows.keys()))

    written = []
    for each_grp_name, each_rows in group_rows.items():
        written.append(
            write_group_csv(
                metric_dir,
                slug_map[each_grp_name],
                file_stem,
                DRIFT_RATE_COLUMNS,
                each_rows,
            )
        )

    write_metric_meta_file(
        metric_dir,
        cfg,
        metric=METRIC_DRIFT_RATE,
        producer="generate_fct._generate_drift_rate_plot_single",
        hook="src/pipeline/pipeline_utils/multirun_testtime_rollout_plot_pipeline_utils/"
        "generate_fct.py: after the warm-up trimming, before `ax.plot`",
        params=params,
        source_experiments=source_experiments or {},
        group_metadata=group_metadata,
        csv_files=list_metric_csv_files(metric_dir),
        grp_name_slug_map=slug_map,
        columns=DRIFT_RATE_COLUMNS,
    )
    write_experiment_meta_file(
        resolve_consolidated_data_dir(cfg),
        cfg,
        metric_subdir=METRIC_DRIFT_RATE,
    )
    return written


def consolidate_drift_rate_samples(
    cfg: DictConfig,
    group_seed_tables: Mapping[Any, Mapping[int, Sequence[Mapping[str, Any]]]],
    *,
    file_stem: str,
    params: Mapping[str, Any],
    trajectory_name_mapping: Mapping[int, str],
    group_provenance: Mapping[Any, Mapping[str, Any]],
    source_experiments: Optional[Mapping[Any, Any]] = None,
    group_metadata: Optional[Mapping[str, Any]] = None,
) -> list:
    """Write one drift-rate SAMPLE CSV PER SEED per group, plus both meta files.

    Action ``W2`` (ruling ``Q12``) of the RLRC ICRA2026 fig-2 / fig-3
    standalone plots ``.junie`` plan
    (``feat_RLRP-819_two_new_standalone_plots_fig2_fig3_plan_20260910.md``).

    Writes ``<grp-slug>/seed_{k}_{file_stem}.csv`` through the existing
    :func:`write_group_csv`, under the SEPARATE ``drift_rate_samples/`` metric
    directory so ``drift_rate/`` stays byte-identical, then ONE ``meta.txt``
    carrying the whole ``W0b`` ID-mapping block. The mapping-derived
    parameters are re-derived HERE from ``group_provenance`` -- never taken
    from ``params`` -- so the recorded ids can never disagree with the written
    columns.

    :param cfg: Top-level multirun plot configuration.
    :param group_seed_tables: ``{grp_name: {seed_ordinal: rows}}`` as built by
        :func:`build_drift_rate_seed_tables`.
    :param file_stem: The producer's PNG file stem (without extension), e.g.
        ``"test_time_models_comparaison_drift_rate_samples"``.
    :param params: Metric parameters recorded in the metric-level meta file.
    :param trajectory_name_mapping: The EXPERIMENT-wide
        ``{trajectory_id: trajectory_name}``; it defines the wide header.
    :param group_provenance: ``{grp_name: provenance}``, the second element
        returned by :func:`build_drift_rate_seed_tables`.
    :param source_experiments: ``{grp_name: <source paths description>}``.
    :param group_metadata: ``{grp_name: <metadata dictionary>}``.
    :return: The absolute paths of the written CSV files.
    """
    metric_dir = resolve_consolidated_data_dir(cfg, METRIC_DRIFT_RATE_SAMPLES)
    slug_map = build_grp_name_slug_map(list(group_seed_tables.keys()))
    columns = build_drift_rate_samples_columns(trajectory_name_mapping.keys())

    written = []
    for each_grp_name, each_seed_tables in group_seed_tables.items():
        for each_seed_ordinal in sorted(each_seed_tables.keys()):
            written.append(
                write_group_csv(
                    metric_dir,
                    slug_map[each_grp_name],
                    f"seed_{each_seed_ordinal}_{file_stem}",
                    columns,
                    each_seed_tables[each_seed_ordinal],
                )
            )

    # .... The `W0b` ID-mapping block .........................................
    _meta_params = dict(params)
    _meta_params["trajectory_name_mapping"] = {
        int(each_id): str(each_name)
        for each_id, each_name in sorted(trajectory_name_mapping.items())
    }
    _meta_params["trajectory_ids_per_group"] = {
        str(each_grp): list(each_provenance.get("trajectory_ids", []))
        for each_grp, each_provenance in group_provenance.items()
    }
    _meta_params["seed_key_mapping"] = {
        str(each_grp): dict(each_provenance.get("seed_key_mapping", {}))
        for each_grp, each_provenance in group_provenance.items()
    }
    _meta_params["n_seeds"] = {
        str(each_grp): each_provenance.get("n_seeds", None)
        for each_grp, each_provenance in group_provenance.items()
    }
    _meta_params["n_rollouts"] = {
        str(each_grp): each_provenance.get("n_rollouts", None)
        for each_grp, each_provenance in group_provenance.items()
    }
    _meta_params["missing_cells"] = {
        str(each_grp): each_provenance.get("missing_cells", None)
        for each_grp, each_provenance in group_provenance.items()
    }

    write_metric_meta_file(
        metric_dir,
        cfg,
        metric=METRIC_DRIFT_RATE_SAMPLES,
        producer="generate_fct._generate_drift_rate_plot_single",
        hook="src/pipeline/pipeline_utils/multirun_testtime_rollout_plot_pipeline_utils/"
        "generate_fct.py: after the warm-up trimming, before `ax.plot`",
        params=_meta_params,
        source_experiments=source_experiments or {},
        group_metadata=group_metadata,
        csv_files=list_metric_csv_files(metric_dir),
        grp_name_slug_map=slug_map,
        columns=columns,
    )
    write_experiment_meta_file(
        resolve_consolidated_data_dir(cfg),
        cfg,
        metric_subdir=METRIC_DRIFT_RATE_SAMPLES,
    )
    return written
