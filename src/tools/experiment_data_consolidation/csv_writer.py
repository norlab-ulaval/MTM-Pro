# coding=utf-8
"""Consolidated experiment data CSV writer (RLRP-818).

Stdlib ``csv`` only -- deliberately no ``pandas`` so the writer can be
imported from the matplotlib plotting path without adding a heavy
dependency to it (see plan section 2.4).
"""
import csv
import os
from typing import Any, Mapping, Sequence

from tools.experiment_data_consolidation.consolidation_paths import resolve_group_dir


def write_group_csv(
    metric_dir: Any,
    grp_name_slug: str,
    file_stem: str,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Write one group's consolidated rows to ``<metric_dir>/<slug>/<stem>.csv``.

    ``file_stem`` is the producer's OWN figure (PNG) file stem, passed in by
    the call site and never rebuilt here, so the CSV <-> figure pairing stays
    unambiguous (see plan section 3.2).

    :param metric_dir: The metric-level directory.
    :param grp_name_slug: Filesystem-safe group token (the sub-directory name).
    :param file_stem: File name without the ``.csv`` extension.
    :param columns: The ordered column names (from
        :mod:`tools.experiment_data_consolidation.consolidation_schema`).
    :param rows: The row mappings; missing keys are written as an empty cell
        and unknown keys are ignored.
    :return: The absolute path of the written CSV file.
    """
    group_dir = resolve_group_dir(metric_dir, grp_name_slug)
    csv_path = os.path.join(group_dir, f"{file_stem}.csv")

    # `lineterminator="\n"`: `csv.writer` defaults to CRLF, which every git clone
    # of the `icra_2026_plot` submodule normalizes back to LF (`core.autocrlf=input`),
    # so each freshly consolidated CSV was reported as "CRLF will be replaced by LF"
    # on `git add`. Writing LF directly makes the worktree bytes match the index.
    with open(csv_path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle, lineterminator="\n")
        writer.writerow(list(columns))
        for each_row in rows:
            writer.writerow(
                [_format_cell(each_row.get(each_column, None)) for each_column in columns]
            )

    return csv_path


def list_metric_csv_files(metric_dir: Any) -> list:
    """List every CSV file currently present under a metric directory.

    Used by the metric-level ``meta.txt`` so its ``csv_files`` section always
    describes the WHOLE metric directory, whichever producer call writes it
    last (the drift rate, for instance, writes the deployed-model CSVs and
    then one batch per epoch checkpoint).

    :param metric_dir: The metric-level directory.
    :return: Sorted ``<group_slug>/<file>.csv`` paths, relative to ``metric_dir``.
    """
    metric_dir = str(metric_dir)
    csv_files = []
    for each_root, _, each_file_names in os.walk(metric_dir):
        for each_file_name in each_file_names:
            if not each_file_name.endswith(".csv"):
                continue
            csv_files.append(
                os.path.relpath(
                    os.path.join(each_root, each_file_name), start=metric_dir
                )
            )
    return sorted(csv_files)


def _format_cell(value: Any) -> Any:
    """Normalize a single cell value for CSV serialization.

    ``None`` becomes an empty cell (the convention used for a not-applicable
    column, e.g. ``train_epoch`` on the deployed-model CSV). Every other type
    is handed to :class:`csv.writer` as-is, which serializes it with ``str``
    (booleans therefore render as ``True``/``False``).

    :param value: The raw cell value.
    :return: The value to hand to :class:`csv.writer`.
    """
    if value is None:
        return ""
    return value
