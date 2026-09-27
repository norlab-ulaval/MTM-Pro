# coding=utf-8
"""Single-writer dataset pre-sanitization stage for robotic 3D pipelines.

Iterates every CSV referenced by an environment configuration (train,
val, test-InD, test-OOD), runs :func:`sanitize_csv_data` on it once,
and transitions its entry in the consolidated directory-level
``<dirname>.pre_sanitized`` marker through
``FALSE → INPROGRESS → TRUE``. On exception the marker is left at
``INPROGRESS`` so the operator sees exactly which file failed and can
re-run after fixing the cause (failure breadcrumb is preserved by
design).

This stage is intended to be run **locally** (single-process), then the
resulting ``*.pre_sanitized`` markers are rsynced to HPC alongside the
CSVs. HPC HPO workers then take the read-only fast path in
:func:`sanitize_csv_data`, eliminating the read-modify-write race that
breaks parallel HPO under joblib (``n_jobs > 1``).

See ``.junie/active_plans/improve_hpo_capabilities_RLRP-624.md`` Phase
E.6 for the full design.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Set

from omegaconf import DictConfig

from pipeline.pipeline_utils.robotic_env_pipeline_utils.dataset_sanitize_pre_processing import (
    sanitize_csv_data,
)
from tools.console_tools.message import consol_msg_universal_one_liner
from tools.dataset_tools.pre_sanitized_marker import (
    MarkerState,
    read_state,
    write_state,
)


def _resolve_path(raw: str) -> Path:
    """Resolve ``raw`` as: absolute → as-is; CWD-relative existing → as-is;
    otherwise relative to ``DN_PROJECT_PATH``. Mirrors the resolution
    contract documented in ``simulator/quadcopter_general.yaml``.
    """
    resolved = Path(raw)
    if resolved.is_absolute():
        return resolved
    if resolved.exists():
        return resolved
    dn_project_path = os.environ.get("DN_PROJECT_PATH")
    if dn_project_path is not None and os.path.exists(dn_project_path):
        return Path(dn_project_path) / raw
    return resolved


def _resolve_data_path(cfg: DictConfig) -> Path:
    """Resolve ``cfg.environment.data_path`` against ``DN_PROJECT_PATH`` if
    relative and missing from CWD (mirrors :func:`tct.utils.dn_sanitize_path`).
    """
    return _resolve_path(cfg.environment.data_path)


def _collect_trajectory_csv_paths(cfg: DictConfig) -> List[Path]:
    """Collect every CSV path that needs pre-sanitization for the active
    environment configuration.

    Two complementary sources are merged (de-duplicated, iteration order
    preserved):

    1. **Configured trajectories** — every entry under
       ``cfg.environment.data.{trajectories,test_InD_trajectory,test_OOD_trajectory}``
       (resolved against ``cfg.environment.data_path``).
    2. **Sanitization dataset roots** — every directory listed under
       ``cfg.environment.data.sanitization_dataset_roots`` is crawled
       recursively for ``*.csv`` files. This catches every CSV in the
       dataset (``test/`` ``test_neurobem/`` ``train/`` ``valid/`` for
       neurobem, plus ``data/repository_data/quadcopter_mock``), not
       only the subset referenced by the active simulator config (e.g.
       ``neurobem_quarter_dataset.yaml``). See Phase E.6 of
       ``.junie/active_plans/improve_hpo_capabilities_RLRP-624.md``.
    """
    data_path = _resolve_data_path(cfg)
    data_cfg = cfg.environment.data

    seen: Set[str] = set()
    csv_paths: List[Path] = []

    def _add(path: Path) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        csv_paths.append(path)

    # (1) Configured trajectories ------------------------------------------------
    trajectory_groups: List[Iterable] = []
    for key in ("trajectories", "test_InD_trajectory", "test_OOD_trajectory"):
        group = data_cfg.get(key) if hasattr(data_cfg, "get") else None
        if group is None:
            continue
        # Normalize legacy single-dict format to a list.
        if isinstance(group, DictConfig) and "trajectory_name" in group:
            group = [group]
        trajectory_groups.append(group)
    for group in trajectory_groups:
        for entry in group:
            name = entry.trajectory_name
            _add(data_path / f"{name}.csv")

    # (2) Extra dataset roots (recursive crawl) ---------------------------------
    extra_roots = (
        data_cfg.get("sanitization_dataset_roots")
        if hasattr(data_cfg, "get")
        else None
    )
    if extra_roots:
        for raw_root in extra_roots:
            root = _resolve_path(str(raw_root))
            if not root.exists():
                consol_msg_universal_one_liner(
                    f"[dataset_sanitization] sanitization_dataset_roots: "
                    f"'{root}' does not exist — skipped"
                )
                continue
            if not root.is_dir():
                consol_msg_universal_one_liner(
                    f"[dataset_sanitization] sanitization_dataset_roots: "
                    f"'{root}' is not a directory — skipped"
                )
                continue
            for csv_file in sorted(root.rglob("*.csv")):
                _add(csv_file)

    return csv_paths


def sanitize_dataset(
    cfg: DictConfig, force: bool = False, verbose: bool = True
) -> dict:
    """Sanitize every CSV referenced by ``cfg.environment.data`` and
    emit explicit tri-state markers.

    :param cfg: Hydra configuration carrying ``environment.data*`` keys.
    :param force: When ``True``, re-sanitize even files whose marker is
        already at :class:`MarkerState.TRUE`. Default ``False`` (idempotent).
    :param verbose: Forward verbosity to :func:`sanitize_csv_data`.
    :returns: Dict with counters: ``processed``, ``skipped`` (already TRUE),
        ``missing`` (CSV file not found), ``failed`` (sanitization error,
        marker left at INPROGRESS).
    """
    csv_paths = _collect_trajectory_csv_paths(cfg)
    consol_msg_universal_one_liner(
        f"[dataset_sanitization] Begin pre-sanitization of "
        f"{len(csv_paths)} CSV file(s)"
    )

    counters = {"processed": 0, "skipped": 0, "missing": 0, "failed": 0}

    for csv_path in csv_paths:
        if not csv_path.exists():
            consol_msg_universal_one_liner(
                f"[dataset_sanitization] MISSING: '{csv_path}' — skipped"
            )
            counters["missing"] += 1
            continue

        current_state = read_state(csv_path)
        if current_state == MarkerState.TRUE and not force:
            consol_msg_universal_one_liner(
                f"[dataset_sanitization] SKIP (already TRUE): '{csv_path.name}'"
            )
            counters["skipped"] += 1
            continue

        # Atomic CAS: FALSE/None/INPROGRESS → INPROGRESS, then sanitize.
        write_state(csv_path, MarkerState.INPROGRESS)
        try:
            sanitize_csv_data(
                str(csv_path),
                timestamps_pre_processing=False,
                timestamp_column=cfg.environment.data.original_timestamps_label,
                verbose=verbose,
            )
        except (OSError, ValueError, RuntimeError) as e:
            # Leave marker at INPROGRESS as a failure breadcrumb.
            consol_msg_universal_one_liner(
                f"[dataset_sanitization] FAILED on '{csv_path}': {e}\n"
                f"   Marker left at INPROGRESS for operator inspection."
            )
            counters["failed"] += 1
            continue

        write_state(csv_path, MarkerState.TRUE)
        counters["processed"] += 1
        consol_msg_universal_one_liner(
            f"[dataset_sanitization] OK: '{csv_path.name}' → TRUE"
        )

    consol_msg_universal_one_liner(
        f"[dataset_sanitization] DONE — "
        f"processed={counters['processed']}, skipped={counters['skipped']}, "
        f"missing={counters['missing']}, failed={counters['failed']}"
    )
    return counters
