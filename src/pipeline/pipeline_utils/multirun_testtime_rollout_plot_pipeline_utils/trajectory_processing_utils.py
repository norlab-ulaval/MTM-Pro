# coding=utf-8
import os

import omegaconf


def _normalize_trajectory_entries(raw) -> list[dict]:
    """Normalize trajectory config to a list of dicts.

    Handles both the new list-of-dicts format and the legacy single-dict
    formats from saved Hydra configs. Two legacy formats are supported:

    - Robotic env: ``{trajectory_name: "train/..."}``.
    - Math env:    ``{initiale_coordinates: [...], param: {...}}``.

    The old format is wrapped in a list with ``category`` defaulting to ``"unknown"``.

    A ``None`` (YAML ``null``) value is normalised to an empty list so a
    simulator/saved-Hydra config with zero OOD test trajectories regenerates
    cleanly (feature_empty_ood_test_trajectory_support_plan_20260821_RLRP-778.md).
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        entry = dict(raw)
        entry.setdefault("category", "unknown")
        return [entry]
    if isinstance(raw, list):
        normalized = []
        for entry in raw:
            entry = dict(entry)
            entry.setdefault("category", "unknown")
            normalized.append(entry)
        return normalized
    raise TypeError(
        f"Expected list or dict for trajectory entries, got {type(raw).__name__}. "
        f"Check the simulator config format."
    )


def _resolve_test_trajectories(cfg, experiment_base: str) -> list[dict]:
    """Resolve the test trajectory list for regeneration.

    Priority:
      1. If ``cfg.simulator_config`` is set, load from
         ``src/launcher/configs/simulator/<simulator_config>.yaml``.
      2. Otherwise load from the experiment's saved Hydra config at
         ``<experiment_base>/.hydra/config.yaml``.

    Returns a combined list of dicts with keys ``trajectory_name`` and ``category``.
    """
    ind_entries, ood_entries = _resolve_test_trajectories_split(cfg, experiment_base)
    return ind_entries + ood_entries


def _resolve_test_trajectories_split(
    cfg, experiment_base: str
) -> tuple[list[dict], list[dict]]:
    """Resolve the test trajectory lists for regeneration, split by InD/OOD.

    Priority:
      1. If ``cfg.simulator_config`` is set, load from
         ``src/launcher/configs/simulator/<simulator_config>.yaml``.
      2. Otherwise load from the experiment's saved Hydra config at
         ``<experiment_base>/.hydra/config.yaml``.

    Returns (ind_entries, ood_entries) — each a list of dicts with
    ``trajectory_name`` and ``category``.
    """
    if cfg.get("simulator_config", None) is not None:
        # ``OmegaConf.load`` does NOT resolve Hydra's ``defaults:`` list, so a
        # simulator variant that only overrides a few keys (e.g.
        # ``husky_gravel_and_grass_full_dataset_vel_only_online`` overriding
        # only ``obs_dims``/``data.label``) would otherwise be missing
        # ``data.test_InD_trajectory`` inherited from its parent yaml. Use the
        # shared defaults-aware loader (RLRP-819) instead of a raw file load.
        sim_cfg = _load_simulator_cfg_with_defaults(cfg.simulator_config)
        ind_entries = _normalize_trajectory_entries(
            omegaconf.OmegaConf.to_container(sim_cfg["data"]["test_InD_trajectory"])
        )
        ood_entries = _normalize_trajectory_entries(
            omegaconf.OmegaConf.to_container(sim_cfg["data"]["test_OOD_trajectory"]) if sim_cfg["data"]["test_OOD_trajectory"] else []
        )
    else:
        hydra_cfg_path = os.path.join(experiment_base, ".hydra", "config.yaml")
        if not os.path.isfile(hydra_cfg_path):
            raise FileNotFoundError(
                f"Cannot resolve test trajectories: saved Hydra config not found at '{hydra_cfg_path}'. "
                f"Set 'simulator_config' in the plot pipeline config to provide an explicit source."
            )
        saved_cfg = omegaconf.OmegaConf.load(hydra_cfg_path)
        env_data = saved_cfg.environment.data
        ind_entries = _normalize_trajectory_entries(
            omegaconf.OmegaConf.to_container(env_data.test_InD_trajectory)
        )
        ood_entries = _normalize_trajectory_entries(
            omegaconf.OmegaConf.to_container(env_data.test_OOD_trajectory)
        )

    return ind_entries, ood_entries


def _resolve_ground_truth_feed_warmup_steps(experiment_base: str) -> int | None:
    """Resolve the ground-truth-feed warm-up step count for an experiment.

    Reads ``deploy.target_experiment.ground_truth_feed_warmup_steps`` from the
    experiment's persisted Hydra config at
    ``<experiment_base>/.hydra/config.yaml``.

    A resolved value of ``0`` is a fully valid warm-up length ("no
    ground-truth feed warm-up") and is preserved as ``0`` — it is **not**
    coerced to ``None``. ``None`` is returned only when the value is genuinely
    unresolvable: the saved config file is absent, or the
    ``deploy.target_experiment.ground_truth_feed_warmup_steps`` key is missing.

    :param experiment_base: absolute path to the experiment run directory
        (the parent of its ``.hydra`` directory).
    :returns: the warm-up step count (an ``int``, possibly ``0``), or ``None``
        when the key/config is absent.
    """
    hydra_cfg_path = os.path.join(experiment_base, ".hydra", "config.yaml")
    if not os.path.isfile(hydra_cfg_path):
        return None
    saved_cfg = omegaconf.OmegaConf.load(hydra_cfg_path)
    value = omegaconf.OmegaConf.select(
        saved_cfg,
        "deploy.target_experiment.ground_truth_feed_warmup_steps",
        default=None,
    )
    if value is None:
        return None
    return int(value)


def _resolve_simulator_config_path(simulator_config_name: str) -> str:
    """Resolve ``<repo>/src/launcher/configs/simulator/<name>.yaml`` from this file."""
    _src_root = os.path.abspath(__file__)
    # __file__ lives at:
    #   <repo>/src/pipeline/pipeline_utils/multirun_testtime_rollout_plot_pipeline_utils/trajectory_processing_utils.py
    # so we must climb 4 directory levels to reach <repo>/src.
    for _ in range(4):
        _src_root = os.path.dirname(_src_root)
    return os.path.join(
        _src_root,
        "launcher",
        "configs",
        "simulator",
        f"{simulator_config_name}.yaml",
    )


def _load_simulator_cfg_with_defaults(sim_name: str):
    """Load a simulator yaml AND merge its Hydra ``defaults`` parents.

    ``OmegaConf.load`` does NOT resolve Hydra's ``defaults:`` list — so keys
    defined only in a parent yaml (e.g. ``flat_container_to_nested`` lives
    in ``quadcopter_general.yaml`` and is inherited by
    ``neurobem_adverse_full_dataset.yaml`` via ``defaults: [- quadcopter_general]``)
    would otherwise be invisible to :func:`apply_simulator_config_override`.

    This helper walks the ``defaults`` list (best-effort, sibling yamls in the
    same ``configs/simulator/`` directory only) and merges them bottom-up so
    the child wins, exactly like Hydra's compose semantics for this slice.
    Unknown / non-sibling defaults (e.g. ``/simulator/softwinsorization@...``)
    are silently skipped — they don't carry the structural keys we need.
    """
    sim_path = _resolve_simulator_config_path(sim_name)
    sim_cfg = omegaconf.OmegaConf.load(sim_path)
    defaults_list = sim_cfg.pop("defaults", None) if "defaults" in sim_cfg else None
    if not defaults_list:
        return sim_cfg

    sim_dir = os.path.dirname(sim_path)
    merged = omegaconf.OmegaConf.create({})
    for entry in defaults_list:
        # Entries are either a string (e.g. ``quadcopter_general``) or a
        # single-key dict (e.g. ``{/simulator/softwinsorization@normalizer: ...}``
        # or ``{_self_: ...}``). We only handle plain sibling-name strings.
        if entry == "_self_":
            continue
        if not isinstance(entry, str):
            continue
        parent_path = os.path.join(sim_dir, f"{entry}.yaml")
        if not os.path.isfile(parent_path):
            continue
        try:
            parent_cfg = _load_simulator_cfg_with_defaults(entry)
            merged = omegaconf.OmegaConf.merge(merged, parent_cfg)
        except Exception:
            continue
    return omegaconf.OmegaConf.merge(merged, sim_cfg)


def apply_simulator_config_override(cfg, experiment_cfg):
    """Override stale data-source keys in ``experiment_cfg`` from ``cfg.simulator_config``.

    Model checkpoints carry the simulator config captured at training time, which can
    drift over time (renamed dataset folders, new test split, etc.). The top-level
    plot-pipeline ``simulator_config`` already overrides the test trajectory list via
    :func:`_resolve_test_trajectories_split` — this helper extends the override to the
    file-system data source so that ``setup_trajectory_from_csv`` and the snapshot-inset
    ground-truth builder resolve CSVs against the *current* on-disk layout.

    Only a whitelisted subset of ``environment.*`` keys is refreshed (``data_path``,
    ``dataset_name``, ``data``) to avoid clobbering training-time fields the model
    genuinely depends on at load time (e.g. measurement noise, ``time_space``).

    :param cfg: Top-level plot-pipeline config (may carry ``simulator_config: <name>``).
    :param experiment_cfg: Per-experiment Hydra config loaded from
        ``<experiment_base>/.hydra/config.yaml`` (mutated in place and returned).
    :return: The same ``experiment_cfg`` instance, possibly mutated.
    """
    sim_name = cfg.get("simulator_config", None) if hasattr(cfg, "get") else None
    if sim_name is None:
        return experiment_cfg

    # Explicit opt-out gate. Default is ``True`` so existing plot-pipeline
    # configs keep their current behavior (refresh stale data-source keys
    # from ``simulator_config``). Set to ``False`` in a plot-pipeline yaml to
    # honor the saved experiment Hydra config verbatim (no refresh).
    override_flag = (
        cfg.get("override_record_config_using_simulator_config", True)
        if hasattr(cfg, "get")
        else True
    )
    if not override_flag:
        return experiment_cfg

    sim_cfg = _load_simulator_cfg_with_defaults(sim_name)

    # Whitelisted refresh: only the keys the rename/dataset-relayout problem
    # affects. Scalar keys are overwritten outright; ``data`` is *deep-merged*
    # because simulator yamls typically only declare a partial ``data`` block
    # (label / trajectories / test_* lists) and we must preserve training-time
    # keys (e.g. ``original_timestamps_label``) that the model load path needs.
    scalar_keys = ("data_path", "dataset_name")
    # Rollout-only env-level keys (OoD test measurement noise). These drive
    # the OoD test rollouts but NOT training, so they must follow the current
    # simulator config when regenerating. Training-time noise keys
    # (``global_measurement_noise``, ``measurement_noise``) are intentionally
    # NOT refreshed.
    ood_noise_keys = (
        "ood_test_global_measurement_noise",
        "ood_test_measurement_noise",
    )
    # Structural rollout-helper keys that were introduced after some saved
    # Hydra configs were written (legacy experiments dating from before the
    # simulator-yaml split). They are required by ``setup_trajectory_from_csv``
    # (``flat_container_to_nested``) and by the deploy-rollout postprocessing
    # path. Refreshing them from the *current* simulator config is safe
    # because they describe how to interpret the CSV layout, not what the
    # model was trained on. Without this, ``_load_ground_truth_entries``
    # fails on every legacy experiment with::
    #     Missing key flat_container_to_nested
    #     full_key: environment.flat_container_to_nested
    #
    # NOTE: The legacy spelling ``flat_contrainer_to_nested`` (with the typo)
    # is intentionally listed alongside the corrected ``flat_container_to_nested``
    # so this override keeps working during the codebase-wide typo fix
    # transition (some simulator yamls and consumers still reference the old
    # spelling at the time of writing).
    structural_keys = (
        "flat_container_to_nested",
        "flat_contrainer_to_nested",
        "deploy_rollout_post_processing",
    )
    with omegaconf.read_write(experiment_cfg), omegaconf.open_dict(experiment_cfg):
        for key in scalar_keys:
            if key in sim_cfg:
                experiment_cfg.environment[key] = sim_cfg[key]
        for key in ood_noise_keys:
            if key in sim_cfg:
                experiment_cfg.environment[key] = sim_cfg[key]
        for key in structural_keys:
            if key in sim_cfg and key not in experiment_cfg.environment:
                experiment_cfg.environment[key] = sim_cfg[key]
        if "data" in sim_cfg:
            current_data = experiment_cfg.environment.get("data", None)
            if current_data is None:
                experiment_cfg.environment["data"] = sim_cfg["data"]
            else:
                # Per-key shallow override (instead of ``OmegaConf.merge``): the
                # legacy saved Hydra configs store ``test_InD_trajectory`` /
                # ``test_OOD_trajectory`` as a single dict, while the new
                # simulator yamls store them as a list — ``merge`` rejects that
                # shape mismatch. Per-key assignment lets the simulator yaml
                # win on every key it declares while preserving the rest.
                with omegaconf.open_dict(current_data):
                    for sub_key in sim_cfg["data"]:
                        current_data[sub_key] = sim_cfg["data"][sub_key]
    return experiment_cfg


def persist_rollout_relevant_cfg_overrides(
    experiment_cfg, experiment_hydra_cfg_path: str
) -> None:
    """Persist rollout-relevant overrides back to ``<experiment_base>/.hydra/config.yaml``.

    When regenerating test-time rollouts with a *changed* simulator
    configuration (e.g. updated ``test_InD_trajectory`` parameters, new OoD
    measurement noise), the per-experiment saved Hydra config on disk would
    otherwise still advertise the *previous* test-trajectory layout — even
    though the freshly-written ``testtime_rollouts/`` directories were
    produced against the new layout. That mismatch is confusing at plot /
    analysis time.

    This helper rewrites ONLY the rollout-relevant keys to the on-disk
    ``.hydra/config.yaml``:
    
    - ``environment.data.test_InD_trajectory``
    - ``environment.data.test_OOD_trajectory``
    - ``environment.ood_test_global_measurement_noise``
    - ``environment.ood_test_measurement_noise``
    
    Training-time keys (``environment.data.trajectories``,
    ``environment.global_measurement_noise``, ``environment.measurement_noise``)
    are intentionally left untouched — they describe the data the model was
    trained on and must not be rewritten post-hoc.
    
    :param experiment_cfg: In-memory experiment config, already refreshed by
        :func:`apply_simulator_config_override`.
    :param experiment_hydra_cfg_path: Absolute path to the on-disk
        ``.hydra/config.yaml`` of the experiment being regenerated.
    """
    if not os.path.isfile(experiment_hydra_cfg_path):
        return

    on_disk_cfg = omegaconf.OmegaConf.load(experiment_hydra_cfg_path)

    rollout_data_keys = ("test_InD_trajectory", "test_OOD_trajectory")
    rollout_env_keys = (
        "ood_test_global_measurement_noise",
        "ood_test_measurement_noise",
    )

    changed = False
    with omegaconf.read_write(on_disk_cfg), omegaconf.open_dict(on_disk_cfg):
        env_in_mem = experiment_cfg.environment
        env_on_disk = on_disk_cfg.environment

        # environment.data.<test_*_trajectory>
        data_in_mem = env_in_mem.get("data", None) if hasattr(env_in_mem, "get") else None
        if data_in_mem is not None:
            data_on_disk = env_on_disk.get("data", None)
            if data_on_disk is None:
                env_on_disk["data"] = omegaconf.OmegaConf.create({})
                data_on_disk = env_on_disk["data"]
            with omegaconf.open_dict(data_on_disk):
                for k in rollout_data_keys:
                    if k in data_in_mem:
                        data_on_disk[k] = data_in_mem[k]
                        changed = True

        # environment.ood_test_*
        for k in rollout_env_keys:
            if k in env_in_mem:
                env_on_disk[k] = env_in_mem[k]
                changed = True

    if changed:
        omegaconf.OmegaConf.save(on_disk_cfg, experiment_hydra_cfg_path)


def _resolve_current_device() -> str:
    """Resolve the best available device string for the current machine."""
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"
