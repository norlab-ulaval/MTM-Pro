# coding=utf-8
import math
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Union

import hydra
import numpy as np
import omegaconf
import pandas as pd
from mbrl.util import ReplayBuffer
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

import trajectory_container_tools as tct
from algorithm.motion_model.env2sim_components.robotic_r2s.trj_container_sampling import (
    collect_full_time_space_rollout,
)
from math_gymnasium.tools.plot_3d_utils import three_dimension_environment_space_plot
from pipeline.pipeline_utils.robotic_env_pipeline_utils.dataset_sanitize_pre_processing import (
    sanitize_csv_data,
)
from pipeline.pipeline_utils.robotic_env_pipeline_utils.quadcopter_trajectory_dataclass import (
    QuadcopterRobotic3D,
    QuadcopterRobotic3DFlat,
    QuadcopterFlatContainerToNested,
)
from pipeline.pipeline_utils.general.testtime_trajectory_dataclass_util import (
    TestTrajectoryEntry,
)
from pipeline.pipeline_utils.robotic_env_pipeline_utils.ugv_trajectory_dataclass import (
    UGVFlatContainerToNested,
    UGVRobotic3D,
    UGVRobotic3DFlat,
)
from tools.console_tools.message import consol_msg_universal_one_liner
from tools.hydra_apps_tools.hydra_utils import get_hydra_experiment_id
from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist
from tools.plot_tools.plot_management import (
    manage_matplotlib_backend,
    manage_matplotlib_warnings,
    show_and_save_plot_helper,
)
from trajectory_container_tools.dataclasses import TestMotionTrajectoryDataclass
from trajectory_container_tools.dataclasses.panda_dataframe_feature_dataclass import (
    BaseDataframeStampedFeatureDataclass,
)


def robotic_data_to_test_motion_trajectory_dataclass(
    test_trajectory: Union[
        BaseDataframeStampedFeatureDataclass, QuadcopterRobotic3D, UGVRobotic3D
    ],
) -> TestMotionTrajectoryDataclass:

    return TestMotionTrajectoryDataclass(
        feature_name=f"{test_trajectory.feature_name} test trajectory",
        # timestamps=test_trajectory.src_timestamps,
        timestamps=test_trajectory.timestamps,  # (CRITICAL) ToDo: validate (ref task RLRP-503)
        observations=test_trajectory.obs,
        actions=test_trajectory.act,
        pose=test_trajectory.poses.stack,
        pose_gt=test_trajectory.poses.stack,
        orientation_gt=test_trajectory.attitude.stack,
        obs_are_velocity=True,
        # RLRP-758: the TCT `TestMotionTrajectoryDataclass` keeps a single
        # `velocity_frame`; post-ingestion both RLRC channels equal the training
        # frame, so the linear channel is a faithful representative here.
        velocity_frame=test_trajectory.linear_velocity_frame,
    )


# Scale factors from supported source timestamp units to SECONDS.
# Used by `timestamps_column_to_seconds` and the unit-aware preflight check.
# Membership of `cfg.environment.data.original_timestamps_unit` in this mapping
# is hard-validated upstream by `pipeline.pipeline_utils.general.setup.uder_cfg_validation`.
_TIMESTAMP_UNIT_TO_SECONDS: dict[str, float] = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}


def timestamps_column_to_seconds(
    dataframe: pd.DataFrame, cfg: omegaconf.DictConfig
) -> np.ndarray:
    """Return the configured timestamp column as a float64 numpy array in SECONDS.

    The source unit is read from
    `cfg.environment.data.original_timestamps_unit` and must be one of
    ``"s" | "ms" | "us" | "ns"`` (hard-validated by
    `pipeline.pipeline_utils.general.setup.uder_cfg_validation`).

    Centralizing the unit-to-seconds scaling in this helper guarantees that:
      - `online_pre_processing` (100 Hz resampling) and any other caller
        operate on a seconds-scale timestamp axis;
      - a wrong unit cannot trigger a `np.arange` allocation in the
        TB range and an external SIGKILL (exit 137) at runtime.

    :param dataframe: source dataframe loaded from a per-trajectory CSV.
    :param cfg: Hydra config (uses `cfg.environment.data.original_timestamps_label`
                and `cfg.environment.data.original_timestamps_unit`).
    :return: 1-D ``np.ndarray`` (float64) with timestamps expressed in seconds.
    :raises ValueError: if the configured unit is not in
                       :data:`_TIMESTAMP_UNIT_TO_SECONDS`.
    """
    label = cfg.environment.data.original_timestamps_label
    unit = cfg.environment.data.original_timestamps_unit
    try:
        scale = _TIMESTAMP_UNIT_TO_SECONDS[unit]
    except KeyError as e:
        raise ValueError(
            f"`environment.data.original_timestamps_unit={unit!r}` not in "
            f"{sorted(_TIMESTAMP_UNIT_TO_SECONDS)}"
        ) from e
    return dataframe[label].values.astype(np.float64) * scale


def _is_already_at_target_frame_rate(
    t_seconds: np.ndarray, target_dt: float, rtol: float = 0.1
) -> bool:
    """Return ``True`` when ``t_seconds`` is already in the right time scale, i.e.
    its *nominal* sampling step matches the target ``target_dt``
    (= ``1 / target_frame_rate``).

    The dataset is considered "already in the right time scale" when:
      - it is strictly increasing (no zero/negative steps), and
      - its **nominal** (median) inter-sample spacing matches ``target_dt`` within
        the relative tolerance ``rtol``.

    Importantly, the grid does **not** need to be perfectly uniform. A real
    robotic log is rarely evenly spaced: message publishing rates routinely lag
    or jitter, which is genuine signal — not noise to be sanitized. As long as
    the dataset's nominal rate already matches the target, resampling onto an
    evenly-spaced grid would not change the time scale; it would only inject
    interpolation error *and* erase that natural non-uniformity. So we skip it
    and preserve the original (possibly lagging) timestamps as-is.

    :param t_seconds: 1-D timestamp axis expressed in SECONDS (float64).
    :param target_dt: target sampling step in seconds (``1 / target_frame_rate``).
    :param rtol: relative tolerance on the nominal spacing (default 10%).
    :return: ``True`` if the nominal rate already matches the target, else
             ``False``.
    """
    if t_seconds.size < 2:
        return False
    diffs = np.diff(t_seconds)
    if np.any(diffs <= 0.0):
        return False
    # Use the *median* step as the nominal rate so occasional publishing lag /
    # jitter does not force a (sanitizing) resample when the dataset is already
    # at the configured rate overall.
    nominal_dt = float(np.median(diffs))
    return bool(np.isclose(nominal_dt, target_dt, rtol=rtol, atol=0.0))


def _subscale_by_interpolation(
    dataframe: pd.DataFrame,
    t_seconds: np.ndarray,
    target_dt: float,
    src_timestamps_label: str,
) -> pd.DataFrame:
    """Sub-scale by linear interpolation onto an evenly-spaced grid.

    Builds a uniform ``target_frame_rate`` Hz time axis (in seconds) spanning
    the source trajectory and linearly interpolates every non-timestamp column
    onto it. This is the legacy behavior; note that interpolating attitude
    (quaternion) and angular-velocity columns independently does NOT preserve
    the kinematic relationship between them — prefer ``decimate`` for robotic
    state data (see RLRP-742).

    :param dataframe: source dataframe (columns include ``src_timestamps_label``).
    :param t_seconds: source timestamp axis in SECONDS (float64).
    :param target_dt: target sampling step in seconds (``1 / target_frame_rate``).
    :param src_timestamps_label: name of the timestamp column.
    :return: a new dataframe resampled on the uniform seconds-scale grid.
    """
    t_new = np.arange(t_seconds[0], t_seconds[-1], target_dt)  # target Hz grid (seconds)

    df_resampled = pd.DataFrame({src_timestamps_label: t_new})
    for col in dataframe.columns:
        if col != src_timestamps_label:
            df_resampled[col] = np.interp(
                t_new, t_seconds, dataframe[col].values.astype(np.float64)
            )
    return df_resampled


def _subscale_by_decimation(
    dataframe: pd.DataFrame,
    t_seconds: np.ndarray,
    target_dt: float,
    src_timestamps_label: str,
) -> pd.DataFrame:
    """Sub-scale by dropping timestamp rows (integer-factor decimation).

    Keeps one row every ``stride`` rows so the surviving nominal rate matches
    the target ``1 / target_dt``. Unlike interpolation, this preserves each
    surviving row's *measured* attitude/angular-velocity pair exactly — it never
    fabricates new samples — so the kinematic relationship between attitude and
    angular velocity is left intact (RLRP-742).

    The surviving timestamps are the measured instants, expressed in SECONDS
    (``t_seconds``), to keep the output consistent with the interpolate / skip
    paths and preserve the downstream ``dt`` / ``delta_stamps`` contract
    (``compute_position_from_velocity_and_attitude`` consumes
    ``timestamps.delta_stamps`` directly as a seconds-scale ``dt``).

    Decimation is **down-sampling only**: requesting a target rate higher than
    the source nominal rate raises ``ValueError`` (decimation cannot create
    samples — use ``interpolate`` instead). Non-integer rate ratios are rounded
    to the nearest integer stride and a warning is emitted reporting the
    effective realized rate.

    :param dataframe: source dataframe (columns include ``src_timestamps_label``).
    :param t_seconds: source timestamp axis in SECONDS (float64).
    :param target_dt: target sampling step in seconds (``1 / target_frame_rate``).
    :param src_timestamps_label: name of the timestamp column.
    :return: a new decimated dataframe on a seconds-scale time axis.
    :raises ValueError: if the request would require up-sampling.
    """
    if t_seconds.size < 2:
        return dataframe.copy()

    diffs = np.diff(t_seconds)
    # Nominal source step/rate from the median (robust to publishing lag/jitter).
    src_dt = float(np.median(diffs))
    src_rate = 1.0 / src_dt
    target_rate = 1.0 / target_dt

    raw_stride = target_dt / src_dt  # == src_rate / target_rate
    if raw_stride < 1.0 and not np.isclose(raw_stride, 1.0, rtol=0.1):
        raise ValueError(
            f"[RLRP-742] `subscaling_kind='decimate'` supports down-sampling only: "
            f"requested target rate ({target_rate:.4g} Hz) is higher than the source "
            f"nominal rate ({src_rate:.4g} Hz). Decimation cannot create samples — "
            f"use `subscaling_kind='interpolate'` for up-sampling."
        )

    stride = max(1, int(round(raw_stride)))
    if not np.isclose(raw_stride, stride, rtol=1e-3):
        effective_rate = src_rate / stride
        warnings.warn(
            f"[RLRP-742] decimation stride rounded to nearest integer: requested "
            f"{target_rate:.4g} Hz from a {src_rate:.4g} Hz source implies a "
            f"non-integer stride ({raw_stride:.4g}); using stride={stride}, i.e. an "
            f"effective realized rate of {effective_rate:.4g} Hz.",
            stacklevel=2,
        )

    df_decimated = dataframe.iloc[::stride].copy().reset_index(drop=True)
    # Keep the *measured* surviving instants, expressed in seconds.
    df_decimated[src_timestamps_label] = t_seconds[::stride]
    return df_decimated


def online_pre_processing(
    dataframe: pd.DataFrame, cfg: omegaconf.DictConfig
) -> pd.DataFrame:
    """Resample the per-trajectory dataframe to a configurable frame rate.

    The target frame rate (in Hz) is read from
    ``cfg.environment.data.target_frame_rate``:

    * if the value is a positive number (e.g. ``100`` for quadcopter,
      ``10`` for UGV) the dataframe is sub-scaled to that rate using the
      method selected by ``cfg.environment.data.subscaling_kind``
      (default ``"decimate"`` when the key is absent, RLRP-742):

        - ``"decimate"``: keep one measured row every ``n`` rows
          (integer down-sampling), preserving each surviving row's
          attitude<->angular-velocity pair exactly. Down-sampling only;
          see :func:`_subscale_by_decimation`.
        - ``"interpolate"``: legacy per-column linear interpolation onto
          an evenly-spaced grid; see :func:`_subscale_by_interpolation`.

      In both cases the output timestamp column is expressed in
      **seconds** (regardless of the source unit declared by
      ``cfg.environment.data.original_timestamps_unit``) so the downstream
      ``dt`` / ``delta_stamps`` contract is preserved;
    * if the value is ``None`` (i.e. YAML ``null``) no upscaling /
      downscaling is performed. The source dataframe is returned as-is
      (the timestamp column keeps its original unit and values).

    When ``target_frame_rate`` is set but the source dataframe is already at that
    nominal rate, the resampling/interpolation step is skipped (it would be a
    costly near-identity that only injects interpolation error). The grid does
    **not** have to be perfectly uniform: real robotic logs routinely lag /
    jitter their publishing rate, which is genuine signal and must not be
    sanitized away by re-gridding. In the skip case only the timestamp axis is
    rescaled to seconds (the original, possibly non-uniform, spacing is
    preserved) so the output stays consistent with the resampled path.

    :param dataframe: source dataframe loaded from a per-trajectory CSV.
    :param cfg: Hydra config (uses
                ``cfg.environment.data.original_timestamps_label`` /
                ``cfg.environment.data.original_timestamps_unit`` /
                ``cfg.environment.data.target_frame_rate``).
    :return: a new dataframe sampled at ``target_frame_rate`` Hz on a
             seconds-scale time axis, or the input dataframe unchanged
             when ``target_frame_rate`` is ``None``.
    """
    # NOTE (RLRP-742): dataset frame-rate sub-scaling currently supports two
    # kinds (see `cfg.environment.data.subscaling_kind`): `interpolate` (per-column
    # linear interpolation onto an even grid) and `decimate` (integer row drop that
    # preserves the measured attitude<->angular-velocity relationship exactly).
    # A third, more complex option -- SO(3)/quaternion-aware resampling (SLERP the
    # attitude and integrate angular velocity consistently) -- is intentionally
    # NOT implemented here. It would be the option to reach for if we ever need
    # non-integer up/down rate changes while preserving the rotational physics.
    src_timestamps_label = cfg.environment.data.original_timestamps_label
    target_frame_rate = cfg.environment.data.target_frame_rate
    force_dataset_fps_rescaling = cfg.environment.data.force_dataset_fps_rescaling

    # `null` / `None` → no up/down-scaling. Return the dataframe unchanged.
    if target_frame_rate is None:
        return dataframe

    # Convert the raw timestamp column to SECONDS (float64) before building the
    # resampling grid. Without this scaling, a `ns`-scale column (e.g. ROS2
    # default `timestamp_ns ~ 1.78e18`) would make `np.arange(t0, tN, dt)`
    # request a ~10^13-element array and trigger an external SIGKILL (exit 137).
    t_orig = timestamps_column_to_seconds(dataframe, cfg)
    dt = 1.0 / float(target_frame_rate)

    # Skip the resampling step when the dataset is already in the right time
    # scale (its nominal rate already matches `target_frame_rate`). Re-gridding
    # would only add interpolation error and would sanitize away the natural
    # publishing lag/jitter, which is genuine robotic signal. We keep the
    # original (possibly non-uniform) spacing and only rescale the timestamp
    # axis to seconds to keep the output consistent with the resampled path.
    if not force_dataset_fps_rescaling and _is_already_at_target_frame_rate(t_orig, dt):
        consol_msg_universal_one_liner(f"Skip dataset frame-rate re-scaling. Already at {target_frame_rate} fps.")
        df_rescaled = dataframe.copy()
        df_rescaled[src_timestamps_label] = t_orig
        return df_rescaled
    else:
        # Dispatch on the configured sub-scaling method. Defaults to `decimate`
        # (RLRP-742) when the key is absent, since row decimation preserves the
        # attitude<->angular-velocity relationship that naive per-column
        # interpolation would corrupt.
        subscaling_kind = cfg.environment.data.get("subscaling_kind", "decimate")
        consol_msg_universal_one_liner(
            f"Re-scale dataset to {target_frame_rate} fps (subscaling_kind='{subscaling_kind}')."
        )

        if subscaling_kind == "interpolate":
            return _subscale_by_interpolation(
                dataframe, t_orig, dt, src_timestamps_label
            )
        elif subscaling_kind == "decimate":
            return _subscale_by_decimation(
                dataframe, t_orig, dt, src_timestamps_label
            )
        else:
            raise ValueError(
                f"[RLRP-742] unknown `environment.data.subscaling_kind="
                f"{subscaling_kind!r}`; expected one of ['interpolate', 'decimate']."
            )


def setup_trajectory_from_csv(
    cfg: DictConfig, env_init_cfg: DictConfig, verbose: bool = True
) -> Union[QuadcopterRobotic3D, UGVRobotic3D]:

    trajectory_name = env_init_cfg.trajectory_name

    csv_path = os.path.join(
        cfg.environment.data_path,
        f"{trajectory_name}.csv",
    )
    csv_path = tct.utils.dn_sanitize_path(csv_path)

    flat_container_to_nested = hydra.utils.instantiate(
        cfg.environment.flat_container_to_nested, _recursive_=False
    )

    if isinstance(flat_container_to_nested, QuadcopterFlatContainerToNested):
        trajectory = tct.extractor.from_stamped_csv(
            csv_path,
            dataset_info=f"{cfg.environment.data.label}\ntrajectory name: {trajectory_name}\ncsv path: {csv_path}",
            features_config={
                "sample": QuadcopterRobotic3DFlat,
            },
            # timestamp_column=f"{src_timestamps_label}_int64",
            timestamp_column=cfg.environment.data.original_timestamps_label,
            pre_extraction_callback=partial(online_pre_processing, **{"cfg": cfg}),
            fail_causal_ordering_violation=True, # (CRITICAL) ToDo: validate switching to True
        )
    elif isinstance(flat_container_to_nested, UGVFlatContainerToNested):
        trajectory = tct.extractor.from_stamped_csv(
            csv_path,
            dataset_info=f"{cfg.environment.data.label}\ntrajectory name: {trajectory_name}\ncsv path: {csv_path}",
            features_config={
                "sample": UGVRobotic3DFlat,
            },
            # timestamp_column=f"{src_timestamps_label}_int64",
            timestamp_column=cfg.environment.data.original_timestamps_label,
            pre_extraction_callback=partial(online_pre_processing, **{"cfg": cfg}),
            fail_causal_ordering_violation=True, # (CRITICAL) ToDo: validate switching to True
        )

    trajectory = flat_container_to_nested.execute(cfg, trajectory.sample)

    if verbose:
        consol_msg_universal_one_liner(
            f"Extrated {trajectory.trajectory_len} timesteps from {os.path.basename(csv_path)}"
        )
    return trajectory


def setup_test_target_rollouts(
    cfg: DictConfig, exp_dir_relative_path: str | Any, headless: bool
) -> tuple[
    list[TestTrajectoryEntry],
    list[TestTrajectoryEntry],
]:

    target_InD_envs, target_OOD_envs = setup_test_source_data(cfg)

    target_InD_entries = [
        TestTrajectoryEntry(
            env=robotic_data_to_test_motion_trajectory_dataclass(env),
            trajectory_name=traj_name,
            category=category,
            short_name=os.path.basename(traj_name),
        )
        for env, traj_name, category in target_InD_envs
    ]
    target_OOD_entries = [
        TestTrajectoryEntry(
            env=robotic_data_to_test_motion_trajectory_dataclass(env),
            trajectory_name=traj_name,
            category=category,
            short_name=os.path.basename(traj_name),
        )
        for env, traj_name, category in target_OOD_envs
    ]

    return (
        target_InD_entries,
        target_OOD_entries,
    )


def _validate_test_trajectory_config(
    cfg_key_name: str, cfg_value, allow_empty: bool = False
) -> list:
    """Validate that a test trajectory config value is in the new list-of-dicts format.

    Raises a descriptive error if the legacy single-dict format is detected,
    guiding the user to update their simulator config file.

    Empty/``null`` support (``feature_empty_ood_test_trajectory_support_plan_
    20260821_RLRP-778.md``): a ``None`` (YAML ``null``) value is normalised to
    an empty list. An empty list is accepted **only** when ``allow_empty=True``
    (used for ``test_OOD_trajectory`` so a simulator config can legitimately
    declare zero OOD test trajectories). ``test_InD_trajectory`` keeps
    ``allow_empty=False`` and fails loud on empty/``null`` (Q3 parity guard
    with the math validator).

    :param cfg_key_name: Name of the cfg key being validated (used in the
        error message), e.g. ``"test_InD_trajectory"``.
    :param cfg_value: Value read from ``cfg.environment.data.<key>``.
    :param allow_empty: When ``True``, a ``None``/empty list is accepted and
        normalised to ``[]``. When ``False`` (default), an empty/``null`` value
        raises :class:`ValueError`.
    :raises ValueError: When ``cfg_value`` is a single trajectory dict (legacy
        format), or when it is empty/``null`` and ``allow_empty`` is ``False``.
    """
    if cfg_value is None:
        cfg_value = []
    if isinstance(cfg_value, (dict, DictConfig)) and "trajectory_name" in cfg_value:
        raise ValueError(
            f"Legacy single-dict format detected for '{cfg_key_name}'.\n"
            f"  Current (unsupported): {cfg_key_name}:\n"
            f"    trajectory_name: {cfg_value['trajectory_name']}\n"
            f"  Please update your simulator config to the new list format:\n"
            f"    {cfg_key_name}:\n"
            f"      - trajectory_name: {cfg_value['trajectory_name']}\n"
            f"        category: S  # or M, L\n"
        )
    if not isinstance(cfg_value, (list, ListConfig)):
        raise ValueError(
            f"Expected '{cfg_key_name}' to be a list of trajectory dicts, "
            f"got {type(cfg_value).__name__}."
        )
    if len(cfg_value) == 0 and not allow_empty:
        raise ValueError(
            f"'{cfg_key_name}' is empty; expected at least one entry with "
            f"'trajectory_name'."
        )
    return list(cfg_value)


def setup_test_source_data(cfg: DictConfig, print_container: bool = False) -> tuple[
    list[tuple[QuadcopterRobotic3D, str, str]],
    list[tuple[QuadcopterRobotic3D, str, str]],
]:
    consol_msg_universal_one_liner(f"Begin test trajectory rollout extraction\n")

    # RLRP-761 S5.8: WARN (or fail fast, see `require_orientation_feature`) when
    # the attitude is propagated along the predicted trajectory while NEITHER
    # `attitude.*` nor `gravity.*` is declared in `environment.obs_dims`.
    from tools.feature_handling_tools.env_handlers import (
        validate_orientation_propagation_source,
    )

    validate_orientation_propagation_source(cfg)

    ind_cfg = _validate_test_trajectory_config(
        "test_InD_trajectory", cfg.environment.data.test_InD_trajectory
    )
    ood_cfg = _validate_test_trajectory_config(
        "test_OOD_trajectory",
        cfg.environment.data.test_OOD_trajectory,
        allow_empty=True,
    )

    target_InD_envs = []
    for entry in ind_cfg:
        env = setup_trajectory_from_csv(cfg, entry)
        if print_container:
            print(env)
        target_InD_envs.append((env, entry.trajectory_name, entry.category))

    target_OOD_envs = []
    for entry in ood_cfg:
        env = setup_trajectory_from_csv(cfg, entry)
        if print_container:
            print(env)
        target_OOD_envs.append((env, entry.trajectory_name, entry.category))

    print("")
    consol_msg_universal_one_liner(f"Test trajectory rollout extraction DONE\n")
    return target_InD_envs, target_OOD_envs


def preflight_check_csv_files(cfg: DictConfig, env_trj_cfg) -> list[str]:
    """Pre-flight check: crawl all CSV paths in the trajectory list and warn on any issue.

    Checks performed per file:
      - File exists on disk
      - File size > 0 (not empty)
      - Pandas can parse at least the header row

    Path resolution mirrors `tct.utils.dn_sanitize_path`: if the configured data_path is
    relative and does not exist from the current working directory (which Hydra may have
    changed to the experiment output directory), the function falls back to resolving it
    relative to the `DN_PROJECT_PATH` environment variable (the project root inside the
    Apptainer container).

    Returns a list of problematic file paths (empty list means all OK).
    A consol message is emitted for each bad file and a summary is printed at the end.
    This function never raises — it only warns, allowing the caller to decide.
    """
    bad_files: list[str] = []
    total = len(env_trj_cfg)

    # .... Resolve data_path using DN_PROJECT_PATH fallback (mirrors dn_sanitize_path logic) ......
    raw_data_path = cfg.environment.data_path
    resolved_data_path = Path(raw_data_path)
    if not resolved_data_path.exists():
        dn_project_path = os.getenv("DN_PROJECT_PATH")
        if dn_project_path is not None and os.path.exists(dn_project_path):
            resolved_data_path = Path(dn_project_path) / raw_data_path

    # .... Resolve configured timestamp column + unit (for the Δt sanity check below) .............
    ts_label = cfg.environment.data.original_timestamps_label
    ts_unit = cfg.environment.data.get("original_timestamps_unit", None)
    ts_unit_scale = _TIMESTAMP_UNIT_TO_SECONDS.get(ts_unit, None) if ts_unit else None
    # `uder_cfg_validation` already hard-fails on a missing/unknown unit before
    # the pipeline reaches here; the conservative `get(...)` keeps preflight
    # robust if it is called in isolation (e.g. from unit tests).

    consol_msg_universal_one_liner(
        f"Pre-flight CSV check:"
        f"\n   cwd='{os.getcwd()}'"
        f"\n   data_path (config)='{raw_data_path}'"
        f"\n   data_path (resolved)='{resolved_data_path.resolve()}'"
        f"\n   timestamps_label='{ts_label}' unit='{ts_unit}'"
        f"\n   Scanning {total} trajectory file(s)..."
    )

    # Acceptable per-sample Δt range in SECONDS. Real-world robotic datasets
    # sample at ~ 10 Hz .. 100 kHz, i.e. Δt_s ∈ [1e-5, 5e-1]. We widen the
    # upper bound to 5 s to tolerate sparsely-sampled CSVs without false alarms.
    # An out-of-range Δt is the unmistakable signature of a wrong
    # `original_timestamps_unit` and would later trigger a TB-scale
    # `np.arange` allocation → SIGKILL in `online_pre_processing`.
    _TS_DT_S_MIN, _TS_DT_S_MAX = 1e-5, 5.0

    for env_init_cfg in env_trj_cfg:
        csv_path = resolved_data_path / f"{env_init_cfg.trajectory_name}.csv"

        issue: str | None = None
        if not csv_path.exists():
            issue = (
                f"File not found: '{csv_path}'\n"
                f"   Hint: verify that the data files were correctly transferred to the server "
                f"and that the path above is a valid, non-empty CSV file."
            )
        elif csv_path.stat().st_size == 0:
            issue = (
                f"File is empty (0 bytes): '{csv_path}'\n"
                f"   Hint: the file exists but has zero content — re-transfer the data to the server."
            )
        else:
            try:
                head = pd.read_csv(csv_path, nrows=3)
            except Exception as e:
                issue = (
                    f"File cannot be parsed (pandas error: {e}): '{csv_path}'\n"
                    f"   Hint: the file may be corrupted — re-transfer the data to the server."
                )
            else:
                # .... Timestamp column + unit sanity check ...................................
                if ts_label not in head.columns:
                    issue = (
                        f"Timestamp column '{ts_label}' not found in CSV header.\n"
                        f"   File: '{csv_path}'\n"
                        f"   Available columns: {list(head.columns)}\n"
                        f"   Hint: verify `environment.data.original_timestamps_label` "
                        f"in the active simulator config."
                    )
                elif ts_unit_scale is not None and len(head) >= 2:
                    try:
                        dt_raw = float(
                            head[ts_label].astype(np.float64).iloc[1]
                            - head[ts_label].astype(np.float64).iloc[0]
                        )
                    except (ValueError, TypeError):
                        dt_raw = None
                    if dt_raw is not None:
                        dt_s = dt_raw * ts_unit_scale
                        if not (_TS_DT_S_MIN < dt_s < _TS_DT_S_MAX):
                            issue = (
                                f"Suspicious timestamp scale: Δt(first two rows)"
                                f"={dt_raw:.6e} in unit={ts_unit!r} "
                                f"→ {dt_s:.6e} s (expected ∈ "
                                f"[{_TS_DT_S_MIN:.0e}, {_TS_DT_S_MAX:.0e}] s).\n"
                                f"   File: '{csv_path}'\n"
                                f"   Hint: the value of "
                                f"`environment.data.original_timestamps_unit={ts_unit!r}` "
                                f"is almost certainly wrong for this CSV. Letting it "
                                f"through would request a huge `np.arange` allocation "
                                f"in `online_pre_processing` and SIGKILL the process "
                                f"(exit 137) with no Python traceback."
                            )

        if issue is not None:
            bad_files.append(str(csv_path))
            consol_msg_universal_one_liner(f"[WARNING] {issue}")

    n_ok = total - len(bad_files)
    if bad_files:
        consol_msg_universal_one_liner(
            f"Pre-flight CSV check DONE: {n_ok}/{total} OK — {len(bad_files)} problematic file(s) logged above"
        )
    else:
        consol_msg_universal_one_liner(
            f"Pre-flight CSV check DONE: all {total} file(s) OK"
        )

    return bad_files


def create_ss_trajectory_replaybuffer_from_csv(
    cfg: DictConfig, exp_dir_relative_path: str | Any, headless: bool
) -> list[ReplayBuffer]:
    ss_full_time_space_replay_buffers = []
    state_space_label = cfg.environment.data.label
    env_trj_cfg = cfg.environment.data.trajectories
    total_env_rollout = len(env_trj_cfg)
    consol_msg_universal_one_liner(f"Begin train/val dataset extraction\n")
    num_samples = 0
    if cfg.debug_mode:
        consol_msg_universal_one_liner(f"pwd: {os.getcwd()}")

    preflight_check_csv_files(cfg, env_trj_cfg)

    for idx, env_init_cfg in enumerate(env_trj_cfg):

        if cfg.debug_mode:
            consol_msg_universal_one_liner(f"Processing csv {env_init_cfg}")

        try:
            train_trj = setup_trajectory_from_csv(cfg, env_init_cfg)
            num_samples += train_trj.trajectory_len

            # .... Setup source replay buffer .........................................................
            progressbar = tqdm(
                desc="Dataset generation", leave=False, total=total_env_rollout
            )
            cfg_model_key = None
            for each_key in ["ms_model", "ss_model", "model"]:
                if is_cfg_key_exist(cfg, each_key):
                    cfg_model_key = each_key
                    break
            # RLRP-824 (dtype hygiene): on the ``pipeline.data_manager: dataloader`` path the
            # single-step buffers are collected directly in the SOURCE dtype (the normalizer
            # dtype, ``one_dim_transition_model.normalize_double_precision`` -> model dtype when
            # null); the legacy ``replay-buffer`` path keeps its historical
            # ``<model>.model_use_double_precision`` (bit-exact).
            from tools.multistep_tools.window_dataset.pipeline_utils import (
                resolve_source_buffer_double_precision,
            )

            source_buffer_double_precision = resolve_source_buffer_double_precision(
                cfg,
                legacy_double_precision=cfg.get(cfg_model_key, {}).get(
                    "model_use_double_precision", False
                ),
            )
            for each_env_rnd in np.arange(cfg.environment.get("env_rnd_instances", 1)):
                # Note: New noise is generated on environment reset
                ss_full_time_space_replay_buffers.append(
                    collect_full_time_space_rollout(
                        train_trj,
                        replace_rewards_with_timestep_index=True,
                        # max_trajectory_length=cfg.UDER
                        # .buffer_max_trajectory_len
                        max_trajectory_length=None,
                        double_precision=source_buffer_double_precision,
                    )
                )
                progressbar.update()
            progressbar.close()

            consol_msg_universal_one_liner(
                f"Replay buffer: obs dtype={ss_full_time_space_replay_buffers[0].obs_type}, act dtype={ss_full_time_space_replay_buffers[0].action_type}"
            )

            # ToDo: implement for skiping env plot saving. Can be execute once via a dedicated script.
            if (
                cfg.pipeline.plot.show_environment_plot
                or cfg.pipeline.plot.save_environment_plot
            ):
                with warnings.catch_warnings():
                    manage_matplotlib_warnings()
                    manage_matplotlib_backend(
                        cfg.pipeline.plot.show_environment_plot, headless
                    )

                    env_fig, ax_3d, ax_z, ax_x, ax_y = (
                        three_dimension_environment_space_plot(
                            cfg,
                            time_space=train_trj.timestamps.stamps,
                            state_space_3d=train_trj.poses.stack,
                            state_space_3d_with_noise=train_trj.poses.stack,
                            title=(
                                f"State space and explored space. Environment configuration {idx + 1}/"
                                f"{len(env_trj_cfg)}"
                            ),
                            state_space_label=state_space_label,
                            subplot_1d_interval=cfg.pipeline.plot.show_environment_1d_subplot_interval,
                            show_samples=True,
                            show_3d_grid=cfg.pipeline.plot.get("show_3d_grid", True),
                            show_3d_axes=cfg.pipeline.plot.get("show_3d_axes", True),
                            figsize=cfg.pipeline.plot.figsize,
                            extra_info_str=(
                                f"  Initiale coordinates: {train_trj[0].poses.stack}\n"
                            ),
                            experiment_id=get_hydra_experiment_id(),
                        )
                    )

                    show_and_save_plot_helper(
                        env_fig,
                        exp_dir_relative_path,
                        "state_space_and_explorable_space_samples"
                        f"_env_cfg_{idx + 1}_of_{len(env_trj_cfg)}",
                        headless,
                        cfg.pipeline.plot.show_environment_plot,
                        cfg.pipeline.plot.save_dpi,
                        save=cfg.pipeline.plot.save_environment_plot,
                    )

        except ValueError as e:
            if "[TCT error] No data found" in str(e):
                pass
            else:
                raise e

    print("")
    consol_msg_universal_one_liner(f"Collected {num_samples} timesteps")
    consol_msg_universal_one_liner(f"Single step source dataset extraction DONE\n")

    if not ss_full_time_space_replay_buffers:
        raise RuntimeError(
            "[create_ss_trajectory_replaybuffer_from_csv] No replay buffer was created: "
            "all trajectory CSV files failed to load.\n"
            f"  data_path: {cfg.environment.data_path}\n"
            f"  trajectories configured: {total_env_rollout}\n"
            "Hint: verify that the CSV data files exist and are non-empty at the configured "
            "data_path on the target machine (e.g. HPC server). "
            "Check for empty files with: find <data_path> -name '*.csv' -empty"
        )

    return ss_full_time_space_replay_buffers


def create_validation_trajectory_rollouts(
    cfg: DictConfig,
) -> list[TestMotionTrajectoryDataclass]:
    env_trj_cfg = cfg.environment.data.trajectories
    total_env_rollout = len(env_trj_cfg)
    consol_msg_universal_one_liner(f"Begin validation trajectory rollout extraction")
    rng = np.random.default_rng(cfg.seed)

    val_env_nb = int(
        math.ceil(total_env_rollout * cfg.deploy.val_rollout.replay_buffer_ratio)
    )
    val_env_trjs = []
    val_rollout_trj_length = []
    target_val_rollout_trj_length = []
    env_init_names = []

    # Bounded-retry safety break: without this guard, if every candidate
    # trajectory fails the accept-gate below (e.g. ``trajectory_len`` is
    # shorter than ``3 * ground_truth_feed_warmup_steps /
    # trajectory_length_ratio``), the ``while`` loop spins forever and the
    # process hangs silently. Observed on CI with the ``quadcopter_mock``
    # fixture (~599 timesteps) against the default ``warmup=200`` +
    # ``length_ratio=0.25`` which requires >= 2400 timesteps. We now cap the
    # number of full passes over ``env_trj_cfg`` and raise a descriptive
    # ``RuntimeError`` instead, so the misconfiguration fails loudly.
    _MAX_VAL_ROLLOUT_ATTEMPTS = 10
    _attempts = 0
    while len(val_rollout_trj_length) < val_env_nb:
        if _attempts >= _MAX_VAL_ROLLOUT_ATTEMPTS:
            _warmup_steps = cfg.deploy.target_experiment.ground_truth_feed_warmup_steps
            _length_ratio = cfg.deploy.val_rollout.trajectory_length_ratio
            _min_required_trj_len = int(
                3 * _warmup_steps / max(float(_length_ratio), 1e-12)
            )
            raise RuntimeError(
                "[create_validation_trajectory_rollouts] Unable to collect "
                f"{val_env_nb} validation rollout(s) after {_attempts} full "
                f"pass(es) over the {total_env_rollout} configured trajectory "
                "config(s): every candidate was rejected by the accept-gate.\n"
                f"  accept-gate: trajectory_len * "
                f"deploy.val_rollout.trajectory_length_ratio "
                f"(={_length_ratio}) "
                f">= 3 * deploy.target_experiment.ground_truth_feed_warmup_steps "
                f"(={_warmup_steps}) "
                f"i.e. trajectory_len >= {_min_required_trj_len}\n"
                "Hint: either lower 'deploy.target_experiment."
                "ground_truth_feed_warmup_steps', raise "
                "'deploy.val_rollout.trajectory_length_ratio', or provide "
                "longer trajectory CSVs."
            )
        _attempts += 1
        env_init_cfg_indexes = rng.choice(len(env_trj_cfg), size=val_env_nb).tolist()
        for each_init_idx in env_init_cfg_indexes:
            env_init_cfg = env_trj_cfg[each_init_idx]

            val_trj = setup_trajectory_from_csv(cfg, env_init_cfg, verbose=False)

            val_rollout = robotic_data_to_test_motion_trajectory_dataclass(val_trj)

            if (
                val_rollout.trajectory_len
                * cfg.deploy.val_rollout.trajectory_length_ratio
                < 3 * cfg.deploy.target_experiment.ground_truth_feed_warmup_steps
            ):
                continue

            env_init_names.append(os.path.basename(env_init_cfg.trajectory_name))
            val_env_trjs.append(val_rollout)
            target_val_rollout_trj_length.append(
                int(
                    val_rollout.trajectory_len
                    * cfg.deploy.val_rollout.trajectory_length_ratio
                )
            )
            val_rollout_trj_length.append(val_rollout.trajectory_len)

    target_val_rollout_trj_length_min = min(
        min(val_rollout_trj_length), min(target_val_rollout_trj_length)
    )

    equal_length_val_env_trjs = []
    for each_val_rollout in val_env_trjs:
        equal_length_val_env_trjs.append(
            each_val_rollout[:target_val_rollout_trj_length_min]
        )

    consol_msg_universal_one_liner(
        f"Collected {val_env_nb} X {target_val_rollout_trj_length_min} (size X trj len) validation trajectories from {', '.join(env_init_names)}"
    )
    consol_msg_universal_one_liner(f"Validation trajectory rollout extraction DONE\n")
    return equal_length_val_env_trjs
