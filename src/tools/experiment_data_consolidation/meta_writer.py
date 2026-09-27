# coding=utf-8
"""Consolidated experiment data meta-file writers (RLRP-818).

Two meta levels are written (plan decision N2):

* ``<experiment>/meta.txt`` -- EXPERIMENT level, shared by every producer:
  generation date, experiment name, originating config, git sha, schema
  version, the run-wide figure-variant tokens and the accumulated
  ``produced_by`` list of metric sub-directories.
* ``<experiment>/<metric>/meta.txt`` -- METRIC level, single owner: the
  producing function and its hook, the metric parameters, the per-group
  source experiment paths, the raw ``grp_name`` -> slug mapping and the CSV
  column list.

The format is a plain, human-readable ``key: value`` text file (no parser
dependency); only the ``produced_by:`` list of the experiment-level file is
ever read back, so it can accumulate across the producers of a single run.
"""
import os
import subprocess
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

import omegaconf
from omegaconf import DictConfig

from tools.experiment_data_consolidation.consolidation_schema import SCHEMA_VERSION

META_FILE_NAME = "meta.txt"

_PRODUCED_BY_KEY = "produced_by"
_LIST_ITEM_PREFIX = "  - "
_EMPTY_LIST_PLACEHOLDER = "<none>"

# ``key``/``value`` separators of the accumulated metric-level sections.
_SLUG_MAP_SEPARATOR = " -> "
_SOURCE_EXPERIMENT_SEPARATOR = ": "


def _fetch_git_sha(cwd: Optional[str] = None) -> str:
    """Best-effort RLRC git commit sha of the producing run.

    Never fatal: any git failure (not a repository, git missing, timeout)
    yields the ``"unknown"`` sentinel.

    :param cwd: Directory the git commands run in; defaults to the cwd.
    :return: ``"<sha>"``, ``"<sha> (dirty)"`` or ``"unknown"``.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        if not sha:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        return f"{sha} (dirty)" if dirty else sha
    except Exception:  # noqa: BLE001 -- provenance is best-effort
        return "unknown"


def _fetch_hydra_config_name(cfg: DictConfig) -> str:
    """Resolve the originating Hydra config name.

    Read from the Hydra runtime choices when a Hydra job is active, with a
    ``cfg``-derived fallback so the meta file stays informative when the
    producer is called outside a Hydra job (e.g. from the tests).

    :param cfg: Top-level Hydra configuration of the producing app.
    :return: The config name, or an informative fallback token.
    """
    try:
        from hydra.core.hydra_config import HydraConfig

        _hydra_cfg = HydraConfig.get()
        _job_config_name = getattr(_hydra_cfg.job, "config_name", None)
        _choices = dict(_hydra_cfg.runtime.choices)
        _experiment_choice = _choices.get("multirun_testtime_rollout_plot@_global_")
        if _job_config_name and _experiment_choice:
            return f"{_job_config_name} (experiment: {_experiment_choice})"
        if _job_config_name:
            return str(_job_config_name)
    except Exception:  # noqa: BLE001 -- provenance is best-effort
        pass

    _fallback = omegaconf.OmegaConf.select(cfg, "overrides.experiment", default=None)
    return f"<no hydra runtime> (overrides.experiment: {_fallback})"


def _fetch_hydra_overrides(cfg: DictConfig) -> list:
    """Best-effort list of the Hydra command-line overrides of this run.

    :param cfg: Top-level Hydra configuration of the producing app.
    :return: The override strings; empty when unavailable.
    """
    try:
        from hydra.core.hydra_config import HydraConfig

        return [str(each) for each in HydraConfig.get().overrides.task]
    except Exception:  # noqa: BLE001 -- provenance is best-effort
        return []


def _read_list_section(meta_path: str, section_key: str) -> list:
    """Read back one rendered list section of an existing meta file.

    A section is the ``<section_key>:`` line followed by the consecutive
    ``  - <item>`` lines rendered by :func:`_render_lines`; the scan stops at
    the first line that is not an item (i.e. the section's trailing blank
    line), so the section may sit anywhere in the file.

    :param meta_path: Path of the ``meta.txt`` to read.
    :param section_key: The section name, WITHOUT the trailing colon.
    :return: The recorded items (the ``<none>`` placeholder is dropped);
        empty when the file or the section is absent/unreadable.
    """
    if not os.path.isfile(meta_path):
        return []

    items: list = []
    in_section = False
    try:
        with open(meta_path, "r", encoding="utf-8") as file_handle:
            for each_line in file_handle:
                if each_line.rstrip("\n") == f"{section_key}:":
                    in_section = True
                    continue
                if in_section:
                    if each_line.startswith(_LIST_ITEM_PREFIX):
                        items.append(each_line[len(_LIST_ITEM_PREFIX) :].strip())
                    else:
                        break
    except OSError:
        return []
    return [each for each in items if each != _EMPTY_LIST_PLACEHOLDER]


def _read_produced_by(meta_path: str) -> list:
    """Read back the ``produced_by`` list of an existing experiment meta file.

    :param meta_path: Path of the experiment-level ``meta.txt``.
    :return: The recorded metric identifiers; empty when the file is absent.
    """
    return _read_list_section(meta_path, _PRODUCED_BY_KEY)


def _merge_keyed_list_section(
    previous_items: Sequence[str], new_items: Sequence[str], separator: str
) -> list:
    """Union two rendered ``key<separator>value`` list sections.

    Used to ACCUMULATE the provenance sections of the metric-level meta file:
    a producer may be invoked several times per run with a different group
    subset each time (e.g. the drift-rate epoch-checkpoint sweep), and a plain
    rewrite would drop the entries contributed by the earlier calls.

    :param previous_items: The items read back from the existing file.
    :param new_items: The items rendered by the current call; they WIN on key
        collision (they are the freshest description of that key).
    :param separator: The ``key``/``value`` separator of the rendered item.
    :return: The merged items, sorted for a deterministic file content.
    """
    merged: dict = {}
    for each_item in list(previous_items) + list(new_items):
        each_key = each_item.split(separator, 1)[0]
        merged[each_key] = each_item
    return [merged[each_key] for each_key in sorted(merged.keys())]


def _render_lines(
    header: str, entries: Sequence, list_sections: Sequence = ()
) -> str:
    """Render a meta file body from ``(key, value)`` entries and list sections.

    :param header: The first (comment) line of the file.
    :param entries: Ordered ``(key, value)`` pairs rendered as ``key: value``.
    :param list_sections: Ordered ``(key, items)`` pairs rendered as a
        ``key:`` line followed by one indented ``- item`` line per element.
    :return: The complete file content.
    """
    lines = [f"# {header}", ""]
    for each_key, each_value in entries:
        lines.append(f"{each_key}: {each_value}")
    for each_key, each_items in list_sections:
        lines.append("")
        lines.append(f"{each_key}:")
        if not each_items:
            lines.append(f"{_LIST_ITEM_PREFIX}{_EMPTY_LIST_PLACEHOLDER}")
        for each_item in each_items:
            lines.append(f"{_LIST_ITEM_PREFIX}{each_item}")
    lines.append("")
    return "\n".join(lines)


def write_experiment_meta_file(
    experiment_dir: Any, cfg: DictConfig, *, metric_subdir: str
) -> str:
    """Write (idempotently) the EXPERIMENT-level ``meta.txt``.

    The whole file is re-derived from ``cfg`` on every call; only the
    ``produced_by`` list is read back and unioned so the three producers of a
    single run converge to one consistent file (no per-section merge logic).

    :param experiment_dir: The experiment-level output directory.
    :param cfg: Top-level Hydra configuration of the producing app.
    :param metric_subdir: The metric identifier this call contributes.
    :return: The absolute path of the written meta file.
    """
    meta_path = os.path.join(str(experiment_dir), META_FILE_NAME)

    produced_by = _read_produced_by(meta_path)
    if metric_subdir not in produced_by:
        produced_by.append(metric_subdir)

    entries = [
        ("generated_at", datetime.now().astimezone().isoformat(timespec="seconds")),
        (
            "experiment",
            omegaconf.OmegaConf.select(cfg, "overrides.experiment", default=""),
        ),
        ("schema_version", SCHEMA_VERSION),
        ("config_name", _fetch_hydra_config_name(cfg)),
        ("git_sha", _fetch_git_sha(cwd=os.path.dirname(os.path.abspath(__file__)))),
        ("render_type", omegaconf.OmegaConf.select(cfg, "render_type", default="")),
    ]
    # Producer-specific keys: recorded only when the producing app defines them
    # (e.g. the inference-benchmark plot has no `show.*` rollout semantic), so
    # the meta file never carries a misleading empty value.
    for each_key in ("show.target_is_ood", "show.compounded_predictions_score"):
        each_value = omegaconf.OmegaConf.select(cfg, each_key, default=None)
        if each_value is not None:
            entries.append((each_key, each_value))

    content = _render_lines(
        "RLRP-818 experiment data consolidation -- EXPERIMENT level meta",
        entries,
        list_sections=[
            ("hydra_overrides", _fetch_hydra_overrides(cfg)),
            ("produced_by", sorted(produced_by)),
        ],
    )
    with open(meta_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(content)
    return meta_path


def write_metric_meta_file(
    metric_dir: Any,
    cfg: DictConfig,
    *,
    metric: str,
    producer: str,
    hook: str,
    params: Mapping[str, Any],
    source_experiments: Mapping[str, Any],
    csv_files: Sequence[str],
    grp_name_slug_map: Optional[Mapping[Any, str]] = None,
    group_metadata: Optional[Mapping[str, Any]] = None,
    columns: Sequence[str] = (),
) -> str:
    """Write the METRIC-level ``meta.txt`` (single owner, accumulating).

    The scalar entries are fully re-derived on every call (the freshest call
    wins), but the ``grp_name_to_slug``, ``group_metadata``, and
    ``source_experiments`` provenance sections are UNIONED with what the file
    already holds: a producer can be invoked several times per run with a
    different group subset each time -- e.g. the drift-rate epoch-checkpoint
    sweep, where an epoch may expose fewer groups than the deployed-model
    rollout -- and a plain rewrite would silently drop the raw ``grp_name`` of
    the groups written by the earlier calls, breaking the guarantee that the
    slug is always reversible.

    :param metric_dir: The metric-level output directory.
    :param cfg: Top-level Hydra configuration of the producing app.
    :param metric: The metric identifier.
    :param producer: Fully qualified name of the producing function.
    :param hook: ``file:line`` of the export hook inside ``producer``.
    :param params: The metric-specific parameters to record.
    :param source_experiments: ``{grp_name: <source paths description>}``.
    :param csv_files: The CSV files written by this producer, relative to
        ``metric_dir``.
    :param grp_name_slug_map: The raw ``grp_name`` -> slug mapping.
    :param group_metadata: Per-group metadata summary (e.g. short name,
        n_trials, n_trajectories, n_rollouts).
    :param columns: The CSV column list of this metric.
    :return: The absolute path of the written meta file.
    """
    meta_path = os.path.join(str(metric_dir), META_FILE_NAME)

    entries = [
        ("generated_at", datetime.now().astimezone().isoformat(timespec="seconds")),
        ("metric", metric),
        ("producer", producer),
        ("hook", hook),
        ("schema_version", SCHEMA_VERSION),
    ]
    entries += [(f"param.{each_key}", each_value) for each_key, each_value in params.items()]

    _slug_items = _merge_keyed_list_section(
        _read_list_section(meta_path, "grp_name_to_slug"),
        [
            f"{each_raw!r}{_SLUG_MAP_SEPARATOR}{each_slug}"
            for each_raw, each_slug in (grp_name_slug_map or {}).items()
        ],
        _SLUG_MAP_SEPARATOR,
    )
    _group_meta_items = _merge_keyed_list_section(
        _read_list_section(meta_path, "group_metadata"),
        [
            f"{each_grp!r}{_SOURCE_EXPERIMENT_SEPARATOR}short_name={each_meta.get('gpr_short_name')}, "
            f"n_trials={each_meta.get('n_trials')}, n_trajectories={each_meta.get('n_trajectories')}, "
            f"n_rollouts={each_meta.get('n_rollouts')}"
            for each_grp, each_meta in (group_metadata or {}).items()
        ],
        _SOURCE_EXPERIMENT_SEPARATOR,
    )
    _source_items = _merge_keyed_list_section(
        _read_list_section(meta_path, "source_experiments"),
        [
            f"{each_grp!r}{_SOURCE_EXPERIMENT_SEPARATOR}{each_src}"
            for each_grp, each_src in (source_experiments or {}).items()
        ],
        _SOURCE_EXPERIMENT_SEPARATOR,
    )

    list_sections = [
        ("csv_columns", list(columns)),
        ("grp_name_to_slug", _slug_items),
    ]
    if _group_meta_items:
        list_sections.append(("group_metadata", _group_meta_items))
    list_sections.extend([
        ("source_experiments", _source_items),
        ("csv_files", list(csv_files)),
    ])

    content = _render_lines(
        f"RLRP-818 experiment data consolidation -- METRIC level meta ({metric})",
        entries,
        list_sections=list_sections,
    )
    with open(meta_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(content)
    return meta_path
