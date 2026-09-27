# coding=utf-8
import os
import os.path
from functools import partial
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np
import trajectory_container_tools as tct

from decimal import Decimal, getcontext, ROUND_HALF_UP

from tools.console_tools.message import consol_msg_universal_one_liner
from tools.dataset_tools.pre_sanitized_marker import (
    MarkerState,
    PRE_SANITIZED_MARKER_SUFFIX as _PRE_SANITIZED_MARKER_SUFFIX,
    read_state as _read_marker_state,
    require_sanitized as _require_sanitized,
)


#: Environment variable opting into the strict marker policy: when set to
#: a truthy value (``1``/``true``/``yes``), :func:`sanitize_csv_data` fails
#: fast unless the consolidated directory-level
#: ``<dirname>.pre_sanitized`` marker has an entry for this CSV at
#: :class:`MarkerState.TRUE`. Intended to be exported by HPC SLURM job
#: scripts running parallel HPO under joblib (``n_jobs > 1``) where the
#: read-modify-write code path is unsafe. Local development / single-process
#: runs leave it unset and retain the historical behavior.
_STRICT_MARKER_ENV_VAR: str = "RLRC_REQUIRE_PRE_SANITIZED_MARKER"


def _strict_marker_policy_enabled() -> bool:
    """Return ``True`` iff the strict pre-sanitized marker policy is
    enabled via the :data:`_STRICT_MARKER_ENV_VAR` environment variable.
    """
    raw = os.environ.get(_STRICT_MARKER_ENV_VAR, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def sanitize_csv_data(
    csv_path: str,
    timestamps_pre_processing=False,
    timestamp_column: str = "t",
    timestamp_precision: str = "picoseconds",
    verbose: bool = True,
) -> Path:
    csv_path = tct.utils.dn_sanitize_path(csv_path)

    # Fast path: if the consolidated directory-level
    # ``<dirname>.pre_sanitized`` marker records this CSV at TRUE and no
    # timestamp pre-processing is requested, the CSV is known to already have
    # sanitized column labels and must be treated as read-only. This avoids
    # any ``pd.read_csv`` / ``df.to_csv`` on the hot path, which previously
    # caused pytest-xdist workers to race and truncate VCS-tracked committed
    # fixtures under ``data/repository_data/quadcopter_mock/**`` (observed on
    # TeamCity as ``CSV file is empty or has no parseable columns`` and,
    # after adding a FileLock, as a full test hang on a stale ``.csv.lock``).
    # The marker is produced at generation time by
    # ``_generate_unittest_subset.py`` and committed alongside the CSV.
    if not timestamps_pre_processing:
        marker_state = _read_marker_state(csv_path)
        if marker_state == MarkerState.TRUE:
            consol_msg_universal_one_liner(
                f"Loading CSV file (read-only, pre-sanitized marker is TRUE): {csv_path}",
                print_it=verbose,
            )
            return csv_path
        if _strict_marker_policy_enabled():
            # HPC strict policy: any state != TRUE is a hard failure to
            # prevent the joblib n_jobs > 1 read-modify-write race in
            # `sanitize_csv_data` (Phase E.6 of RLRP-624).
            _require_sanitized(csv_path)

    consol_msg_universal_one_liner(f"Loading CSV file: {csv_path}", print_it=verbose)
    try:
        df = pd.read_csv(csv_path)
    except EmptyDataError as e:
        raise ValueError(
            f"[sanitize_csv_data] CSV file is empty (no columns/rows): {csv_path}\n"
            f"Hint: verify that the data files were correctly transferred to the HPC server "
            f"and that the path '{csv_path}' points to a valid, non-empty CSV file."
        ) from e

    if df.empty:
        raise ValueError(
            f"[sanitize_csv_data] CSV file is empty or has no parseable columns: {csv_path}\n"
            f"Hint: verify that the data files were correctly transferred to the HPC server "
            f"and that the path '{csv_path}' points to a valid, non-empty CSV file."
        )

    # ... existing column mapping code ...
    if "neurobem" in csv_path.parts or "neurobem_adverse" in csv_path.parts:
        column_mapping = {
            "pos x": "pos_x",
            "pos y": "pos_y",
            "pos z": "pos_z",
            "vel x": "l_vel_x",
            "vel y": "l_vel_y",
            "vel z": "l_vel_z",
            "quat w": "quat_w",
            "quat x": "quat_x",
            "quat y": "quat_y",
            "quat z": "quat_z",
            "ang vel x": "a_vel_x",
            "ang vel y": "a_vel_y",
            "ang vel z": "a_vel_z",
            "mot 1": "mot_1",
            "mot 2": "mot_2",
            "mot 3": "mot_3",
            "mot 4": "mot_4",
        }
    elif "pi_tcn" in csv_path.parts:
        column_mapping = {
            "p_x": "pos_x",
            "p_y": "pos_y",
            "p_z": "pos_z",
            "v_x": "l_vel_x",
            "v_y": "l_vel_y",
            "v_z": "l_vel_z",
            "q_w": "quat_w",
            "q_x": "quat_x",
            "q_y": "quat_y",
            "q_z": "quat_z",
            "w_x": "a_vel_x",
            "w_y": "a_vel_y",
            "w_z": "a_vel_z",
            "u_0": "mot_1",
            "u_1": "mot_2",
            "u_2": "mot_3",
            "u_3": "mot_4",
        }
    elif "husky_adverse" in csv_path.parts:
        # timestamp_ns,
        # pos_x,pos_y,pos_z,
        # quat_x,quat_y,quat_z,quat_w,
        # l_vel_x,l_vel_y,l_vel_z,
        # a_vel_x,a_vel_y,a_vel_z,
        # steering,speed
        column_mapping = {
            "x": "pos_x",
            "y": "pos_y",
            "z": "pos_z",
            "linear_vel_x": "l_vel_x",
            "linear_vel_y": "l_vel_y",
            "linear_vel_z": "l_vel_z",
            "qw": "quat_w",
            "qx": "quat_x",
            "qy": "quat_y",
            "qz": "quat_z",
            "angular_vel_x": "a_vel_x",
            "angular_vel_y": "a_vel_y",
            "angular_vel_z": "a_vel_z",
            "cmd_linear_vel": "steering",
            "cmd_angular_vel": "speed",
        }
    # elif "husky_adverse_extended" in csv_path.parts:
    #     # (Priority) ToDo: implement (ref task RLRP-776 feat: pull and setup husky dataset with raw feature)
    #     # timestamp_ns,
    #     # online_x, online_y, online_z,
    #     # online_qx, online_qy, online_qz, online_qw,
    #     # offline_x, offline_y, offline_z,
    #     # offline_qx, offline_qy, offline_qz, offline_qw,
    #     # smoothed_x, smoothed_y, smoothed_z,
    #     # smoothed_qx, smoothed_qy, smoothed_qz, smoothed_qw,
    #     # smoothed_linear_vel_x, smoothed_linear_vel_y, smoothed_linear_vel_z,
    #     # smoothed_angular_vel_x, smoothed_angular_vel_y, smoothed_angular_vel_z,
    #     # cmd_linear_vel, cmd_angular_vel
    #     column_mapping = {
    #         "x": "pos_x",
    #         "y": "pos_y",
    #         "z": "pos_z",
    #         "linear_vel_x": "l_vel_x",
    #         "linear_vel_y": "l_vel_y",
    #         "linear_vel_z": "l_vel_z",
    #         "qw": "quat_w",
    #         "qx": "quat_x",
    #         "qy": "quat_y",
    #         "qz": "quat_z",
    #         "angular_vel_x": "a_vel_x",
    #         "angular_vel_y": "a_vel_y",
    #         "angular_vel_z": "a_vel_z",
    #         "cmd_linear_vel": "steering",
    #         "cmd_angular_vel": "speed",
    #     }
    else:
        column_mapping = {}
    original_columns = tuple(df.columns)
    df = df.rename(columns=column_mapping)

    # Clean up column label names
    df.columns = df.columns.str.replace(" ", "_")

    # Save the modified CSV only if column sanitization actually changed
    # something. Skipping the rewrite on already-sanitized files (e.g. the
    # committed ``data/repository_data/quadcopter_mock`` fixtures) avoids a
    # pytest-xdist race where one worker truncates the file via
    # ``df.to_csv(csv_path)`` while another worker reads it, which manifested
    # on TeamCity as ``CSV file is empty or has no parseable columns``.
    if tuple(df.columns) != original_columns:
        df.to_csv(csv_path, index=False)

    # .... Timestamps pre-processing ..............................................................
    if timestamps_pre_processing:
        df = pd.read_csv(
            csv_path,
            dtype={timestamp_column: str},  # Preserve precision for conversion
        )

        # Universal column name (no precision suffix)
        timestamp_int64_column = f"{timestamp_column}_int64"

        if timestamp_int64_column not in df.columns:
            # Choose a timestamp conversion based on the precision requirement

            if timestamp_precision == "picoseconds":
                conversion_func = convert_to_picoseconds_int64
                consol_msg_universal_one_liner(
                    f"Using picosecond precision (1e-12s, ±292 year range)",
                    print_it=verbose,
                )
            elif timestamp_precision == "nanoseconds":
                conversion_func = convert_to_nanoseconds_int64
                consol_msg_universal_one_liner(
                    f"Using nanosecond precision (1e-9s)", print_it=verbose
                )
            elif timestamp_precision == "femtoseconds":
                conversion_func = partial(convert_to_scaled_time_units, scale=1e15)
                consol_msg_universal_one_liner(
                    f"Using femtosecond precision (1e-15s, ±2.6 hour range)",
                    print_it=verbose,
                )
            else:
                raise ValueError(f"Unknown precision: {timestamp_precision}")

            consol_msg_universal_one_liner(
                f"Creating high-precision timestamp column: {timestamp_int64_column}",
                print_it=verbose,
            )
            df[timestamp_int64_column] = df[timestamp_column].apply(conversion_func)
            df[timestamp_int64_column] = df[timestamp_int64_column].astype("int64")

        # Convert the original timestamp column back to float64 for CSV output
        df[timestamp_column] = df[timestamp_column].astype("float64")

        # Save the modified CSV
        df.to_csv(csv_path, index=False)

    return csv_path


def convert_to_picoseconds_int64(timestamp_str):
    """Convert to picoseconds with proper rounding to minimize integration bias.

    Why use picosecond instead of nanosecond storage format:
    - Precision: 1e-12 seconds (1000x better than nanoseconds)
    - Storage: Standard int64 (no pandas object issues)
    - Computational efficiency (standard integer arithmetic)
    - Future-proof for high-frequency sensors
    """
    try:
        from decimal import Decimal, getcontext, ROUND_HALF_UP

        getcontext().prec = 50
        decimal_time = Decimal(timestamp_str.strip())
        picoseconds_decimal = decimal_time * Decimal("1000000000000")  # 10^12

        # Use proper rounding to minimize cumulative bias
        picoseconds = int(
            picoseconds_decimal.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        )
        return picoseconds
    except (ValueError, OverflowError):
        # Fallback with rounding
        timestamp_float = float(timestamp_str.strip())
        return round(timestamp_float * 1_000_000_000_000)


def convert_to_nanoseconds_int64(timestamp_str):
    """Convert to nanoseconds with proper rounding."""
    try:
        timestamp_float = float(timestamp_str.strip())
        nanoseconds = round(timestamp_float * 1_000_000_000)
        return nanoseconds
    except (ValueError, OverflowError):
        return round(float(timestamp_str.strip()) * 1_000_000_000)


def convert_to_scaled_time_units(timestamp_str, time_scale_factor=1e15):
    """Convert using custom scaling factor with proper rounding."""
    try:
        timestamp_float = float(timestamp_str.strip())
        scaled_units = round(timestamp_float * time_scale_factor)
        return scaled_units
    except (ValueError, OverflowError):
        from decimal import Decimal, getcontext, ROUND_HALF_UP

        getcontext().prec = 50
        decimal_time = Decimal(timestamp_str.strip())
        scaled_decimal = decimal_time * Decimal(str(time_scale_factor))
        scaled_units = int(
            scaled_decimal.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        )
        return scaled_units


def convert_to_nanoseconds(timestamp_str):
    """Convert to nanoseconds with Decimal precision and proper rounding."""
    from decimal import Decimal, getcontext, ROUND_HALF_UP

    getcontext().prec = 50
    decimal_time = Decimal(timestamp_str.strip())
    nanoseconds_decimal = decimal_time * Decimal("1000000000")
    nanoseconds = int(
        nanoseconds_decimal.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    return nanoseconds


def convert_to_attoseconds(timestamp_str):
    """Convert to attoseconds with proper rounding.

    (!) Be advised that the returned value may exceed the range of standard integer types.
    Pandas dataframe will load this as object dtype which might break TCT logic."""
    from decimal import Decimal, getcontext, ROUND_HALF_UP

    getcontext().prec = 50
    decimal_time = Decimal(timestamp_str.strip())
    attoseconds_decimal = decimal_time * Decimal("1000000000000000000")
    attoseconds = int(
        attoseconds_decimal.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    return attoseconds
