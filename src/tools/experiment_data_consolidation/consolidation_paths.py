# coding=utf-8
"""Consolidated experiment data path resolution (RLRP-818).

Resolves the ``data_consolidated`` destination tree and turns a raw group
name (``grp_name``) into a deterministic, filesystem-safe directory name.

Output layout::

    data/data_consolidated/
    └── <cfg.overrides.experiment>/
        ├── meta.txt                     # experiment-level
        └── <metric>/
            ├── meta.txt                 # metric-level
            └── <slug(grp_name)>/
                └── <png_file_stem>.csv

The raw ``grp_name`` is ALWAYS preserved verbatim inside the CSV rows and in
the metric-level ``meta.txt``; the slug is presentational only.
"""
import os
import re
from typing import Any, AnyStr, Optional, Sequence

import omegaconf
from omegaconf import DictConfig

from tools.hydra_apps_tools.r2s2r_apps_utils import fetch_r2s2r_project_root_path

# Destination directory, relative to the RLRC project root. It lives in the
# ``icra_2026_plot`` git SUBMODULE and is version-controlled there (the
# submodule ``.gitignore`` only ignores ``**/artifact/``).
CONSOLIDATED_DATA_RELATIVE_PATH = os.path.join(
    "data", "data_consolidated"
)

# Characters that are either not filesystem-safe or hostile to shell/glob use.
_SLUG_UNSAFE_CHAR_RE = re.compile(r"[^A-Za-z0-9._+-]+")


def is_consolidation_enabled(cfg: DictConfig) -> bool:
    """Whether the experiment data consolidation is opted in for this run.

    Read defensively so a config that predates RLRP-818 (no
    ``consolidate_data`` block at all) simply stays disabled.

    :param cfg: Top-level Hydra configuration of the producing app.
    :return: ``True`` when ``consolidate_data.enable`` is truthy.
    """
    return bool(
        omegaconf.OmegaConf.select(cfg, "consolidate_data.enable", default=False)
    )


def resolve_experiment_name(cfg: DictConfig) -> str:
    """Resolve the per-experiment output directory name.

    Uses ``cfg.overrides.experiment`` -- the same token that names the
    ``artifact/ICRA2026/latex_includegraphics/<experiment>`` directory.

    :param cfg: Top-level Hydra configuration of the producing app.
    :return: The experiment name.
    :raise ValueError: When ``overrides.experiment`` is absent/empty, since
        without it the consolidated data could not be attributed to an
        experiment.
    """
    experiment = omegaconf.OmegaConf.select(cfg, "overrides.experiment", default=None)
    if experiment is None or not str(experiment).strip():
        raise ValueError(
            "experiment_data_consolidation: `overrides.experiment` is required to "
            "name the consolidated data directory but is missing/empty from the "
            "configuration. Set it in the experiment config (see "
            "`src/launcher/configs/global_config.yaml`) or disable "
            "`consolidate_data.enable`."
        )
    return str(experiment).strip()


def resolve_consolidated_data_root(cfg: DictConfig) -> AnyStr:
    """Resolve the ``data_consolidated`` root directory.

    Honours the ``consolidate_data.output_dir`` override, which exists so the
    TESTS can redirect every write to a ``tmp_path`` and never pollute the
    version-controlled ``icra_2026_plot`` submodule work-tree.

    :param cfg: Top-level Hydra configuration of the producing app.
    :return: Absolute path to the consolidated data root (not created).
    """
    output_dir = omegaconf.OmegaConf.select(
        cfg, "consolidate_data.output_dir", default=None
    )
    if output_dir is not None and str(output_dir).strip():
        return os.path.abspath(str(output_dir))

    return os.path.join(
        fetch_r2s2r_project_root_path(cfg, lvl_up=1), CONSOLIDATED_DATA_RELATIVE_PATH
    )


def resolve_consolidated_data_dir(
    cfg: DictConfig,
    metric_subdir: Optional[str] = None,
    *,
    create: bool = True,
) -> str:
    """Resolve the experiment (and optionally the metric) output directory.

    :param cfg: Top-level Hydra configuration of the producing app.
    :param metric_subdir: Metric identifier (see
        :mod:`tools.experiment_data_consolidation.consolidation_schema`); when
        ``None`` the experiment-level directory is returned.
    :param create: Create the directory tree when missing. Defaults to ``True``.
    :return: The absolute directory path.
    """
    out_dir = os.path.join(
        resolve_consolidated_data_root(cfg), resolve_experiment_name(cfg)
    )
    if metric_subdir is not None:
        out_dir = os.path.join(out_dir, str(metric_subdir))

    if create:
        os.makedirs(out_dir, exist_ok=True)
    return out_dir


def slugify_grp_name(grp_name: Any) -> str:
    """Turn a raw group name into a deterministic filesystem-safe token.

    Example: ``"(ours) Dist-MTM-Pro-MS+CP"`` -> ``"ours_Dist-MTM-Pro-MS+CP"``.

    Unsafe runs of characters (spaces, parentheses, ``/``, ``\\``, ...) collapse
    to a single ``_``; leading dots and leading/trailing separators are
    stripped so the result is never a hidden or empty directory name.

    :param grp_name: The raw ``grp_name`` config value.
    :return: The slug (never empty -- falls back to ``"unnamed_group"``).
    """
    slug = _SLUG_UNSAFE_CHAR_RE.sub("_", str(grp_name))
    slug = slug.strip("._-")
    slug = re.sub(r"_{2,}", "_", slug)
    return slug or "unnamed_group"


def build_grp_name_slug_map(grp_names: Sequence[Any]) -> dict:
    """Map every raw group name to a UNIQUE slug.

    Slugification is lossy, so two distinct ``grp_name`` values can collapse to
    the same token. Collisions are resolved deterministically by appending
    ``__<n>`` (``n`` starting at 2) following the input order; the mapping is
    recorded in the metric-level ``meta.txt`` so the raw name is always
    recoverable.

    :param grp_names: The raw group names, in a stable (deterministic) order.
    :return: ``{raw_grp_name: unique_slug}``.
    """
    slug_map: dict = {}
    taken: dict = {}
    for each_grp_name in grp_names:
        if each_grp_name in slug_map:
            continue
        base_slug = slugify_grp_name(each_grp_name)
        count = taken.get(base_slug, 0) + 1
        taken[base_slug] = count
        slug_map[each_grp_name] = base_slug if count == 1 else f"{base_slug}__{count}"
    return slug_map


def resolve_group_dir(
    metric_dir: AnyStr, grp_name_slug: str, *, create: bool = True
) -> str:
    """Resolve the per-group sub-directory of a metric directory.

    :param metric_dir: The metric-level directory (see
        :func:`resolve_consolidated_data_dir`).
    :param grp_name_slug: The group slug (see :func:`build_grp_name_slug_map`).
    :param create: Create the directory when missing. Defaults to ``True``.
    :return: The absolute per-group directory path.
    """
    group_dir = os.path.join(str(metric_dir), str(grp_name_slug))
    if create:
        os.makedirs(group_dir, exist_ok=True)
    return group_dir
