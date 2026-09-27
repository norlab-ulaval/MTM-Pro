# coding=utf-8
import os

import numpy as np
import omegaconf

from pipeline.pipeline_utils.multirun_testtime_rollout_plot_pipeline_utils.trajectory_processing_utils import \
    _normalize_trajectory_entries


def _get_category_max_length(cfg) -> dict[str, int]:
    """Retrieve the category → max rollout length mapping from the config.

    Priority:
      1. Top-level ``cfg.simulator_config`` (resolves
         ``<src>/launcher/configs/simulator/<sim_name>.yaml``, falling back to
         the parent ``quadcopter_general.yaml`` when the simulator yaml inherits
         ``test_time_rollout_categories`` via Hydra ``defaults:``).
      2. Legacy ``cfg.environment.dataset_name`` + ``cfg.environment.test_time_rollout_categories``.
      3. Hardcoded neurobem fallback.
    """
    # --- Priority 1: top-level simulator_config override ---
    sim_name = cfg.get("simulator_config", None) if hasattr(cfg, "get") else None
    if sim_name:
        # Resolve <src_root>/launcher/configs/simulator/<sim_name>.yaml.
        # This file lives at:
        #   src/pipeline/pipeline_utils/multirun_testtime_rollout_plot_pipeline_utils/category_utils.py
        # → climb 4 levels to reach <src_root>.
        _here = os.path.abspath(__file__)
        for _ in range(4):
            _here = os.path.dirname(_here)
        sim_cfg_dir = os.path.join(_here, "launcher", "configs", "simulator")
        sim_cfg_path = os.path.join(sim_cfg_dir, f"{sim_name}.yaml")
        if os.path.isfile(sim_cfg_path):
            try:
                sim_cfg = omegaconf.OmegaConf.load(sim_cfg_path)
                dataset_name = sim_cfg.get("dataset_name", None)
                cats_root = sim_cfg.get("test_time_rollout_categories", None)
                # Hydra `defaults:` chain isn't resolved by bare OmegaConf.load,
                # so also probe the parent quadcopter_general.yaml when the
                # simulator yaml inherits these keys.
                if cats_root is None or dataset_name is None:
                    parent_cfg_path = os.path.join(sim_cfg_dir, "quadcopter_general.yaml")
                    if os.path.isfile(parent_cfg_path):
                        parent_cfg = omegaconf.OmegaConf.load(parent_cfg_path)
                        if cats_root is None:
                            cats_root = parent_cfg.get("test_time_rollout_categories", None)
                        if dataset_name is None:
                            dataset_name = parent_cfg.get("dataset_name", None)
                if dataset_name and cats_root is not None:
                    block = omegaconf.OmegaConf.to_container(cats_root).get(dataset_name)
                    if block:
                        return {str(k): int(v) for k, v in block.items()}
            except Exception:
                pass

    # --- Priority 2: legacy `cfg.environment.*` path ---
    try:
        dataset_name = cfg.environment.dataset_name
        categories = omegaconf.OmegaConf.to_container(
            cfg.environment.test_time_rollout_categories.get(dataset_name)
        )
        return {str(k): int(v) for k, v in categories.items()}
    except (omegaconf.errors.ConfigAttributeError, AttributeError, KeyError):
        # --- Priority 3: hardcoded neurobem fallback ---
        return {"S": 2613, "M": 3577, "L": 10566}


def _truncate_mae_to_category_length(
    mae_array: np.ndarray, max_len: int
) -> np.ndarray:
    """Truncate a MAE array to the category max length.

    If the array is longer than max_len, truncate it.
    If the array is shorter or equal, return as-is (no padding).
    """
    if mae_array.shape[0] <= max_len:
        return mae_array
    return mae_array[:max_len]


def _resolve_category_lookup(
    experiment_base: str,
    simulator_config_name: str | None = None,
    project_config_root: str | None = None,
) -> dict[str, str]:
    """Build a trajectory short_name → category lookup.

    Priority:
      1. If *simulator_config_name* is provided, load from
         ``<project_config_root>/simulator/<simulator_config_name>.yaml``.
      2. Otherwise fall back to the experiment's saved Hydra config.

    Returns an empty dict if the config cannot be loaded.
    """
    # --- Priority 1: explicit simulator config ---
    if simulator_config_name and project_config_root:
        sim_cfg_path = os.path.join(
            project_config_root, "simulator", f"{simulator_config_name}.yaml"
        )
        if os.path.isfile(sim_cfg_path):
            try:
                sim_cfg = omegaconf.OmegaConf.load(sim_cfg_path)
                env_data = sim_cfg.data if hasattr(sim_cfg, "data") else sim_cfg.get("data")
                if env_data is not None:
                    lookup = _extract_category_lookup_from_env_data(env_data)
                    if lookup:
                        return lookup
            except Exception:
                pass

    # --- Priority 2: saved Hydra config ---
    hydra_cfg_path = os.path.join(experiment_base, ".hydra", "config.yaml")
    if not os.path.isfile(hydra_cfg_path):
        return {}
    try:
        saved_cfg = omegaconf.OmegaConf.load(hydra_cfg_path)
        env_data = saved_cfg.environment.data
        return _extract_category_lookup_from_env_data(env_data)
    except Exception:
        return {}


def _extract_category_lookup_from_env_data(env_data) -> dict[str, str]:
    """Extract trajectory short_name → category from an env data config node."""
    lookup = {}
    for key in ("test_InD_trajectory", "test_OOD_trajectory"):
        raw = getattr(env_data, key, None) if hasattr(env_data, key) else env_data.get(key)
        if raw is None:
            continue
        entries = _normalize_trajectory_entries(
            omegaconf.OmegaConf.to_container(raw) if hasattr(raw, "_metadata") else raw
        )
        for entry in entries:
            short = os.path.basename(entry["trajectory_name"])
            lookup[short] = entry.get("category", "unknown")
    return lookup
