# coding=utf-8
"""Generate (or load) the single-step trajectory HDF5 store of a robotic dataset (RLRP-824 Step 6a).

Permanent pipeline. Introduced by Step 6a of the Ultra-long-horizon MS->MS training via a lazy
window DataLoader ``.junie`` plan (``ultra_long_horizon_dataloader_plan_RLRP-824_20260915.md``).
Sibling of ``robotic_3d_env_generate_ms_replaybuffer_pipeline`` for ``pipeline.data_manager:
dataloader``: CSV -> single-step ``ReplayBuffer``s (``create_ss_trajectory_replaybuffer_from_csv``,
the SAME ingestion as the legacy path: frame conversion, feature-dim selection, sanitization) ->
:class:`SingleStepTrajectoryStore` -> spec-guarded HDF5 cache
(``tools.multistep_tools.window_dataset.ms_hdf5_save_load_utils``). The store is
``(history_len, horizon_len)``-independent: one cache per dataset serves every window shape.

``load_or_generate`` is what the full pipeline calls: cache lookup (staged read-only root first,
then the data root) unless ``pipeline.force_regenerating_saved_ms_hdf5``; on a miss it generates
when ``pipeline.ms_hdf5_allow_generate`` (default ``true``) and fails fast otherwise (staged SLURM
GPU jobs run with ``false`` so they never race on regeneration -- the CPU GEN-HDF5 job owns it).

``execute`` is also the standalone GEN-HDF5 entry point (``launcher/robotic_3d_env_generate_ms_hdf5.py``,
``slurm_jobs/RLRP-757-GEN-HDF5/``): with ``pipeline.ms_hdf5_write_to_stage_dir: true`` the store is
written into ``<pipeline.ms_hdf5_stage_dir>/<ms_hdf5_save_dir>`` (the ``RLRC_STAGE_DATA_DIR`` bound
``:rw`` by ``hpo_cpu_apptainer_exec``) instead of the data root, so every later GPU job finds it
through the read-only stage bind.
"""
from __future__ import annotations

import omegaconf

from algorithm.utils import seed_me
from pipeline.pipeline_utils.general.setup import uder_cfg_validation
from pipeline.pipeline_utils.robotic_env_pipeline_utils.setup_utils import (
    create_ss_trajectory_replaybuffer_from_csv,
)
from tools.console_tools.message import consol_msg_universal_one_liner
from tools.hydra_apps_tools.hydra_utils import get_hydra_experiment_cwd
from tools.multistep_tools.window_dataset.ms_hdf5_save_load_utils import (
    load_single_step_store_with_spec,
    resolve_ms_hdf5_stage_root,
    save_single_step_store_with_spec,
)
from tools.multistep_tools.window_dataset.pipeline_utils import (
    build_store_from_ss_replay_buffers,
    resolve_source_data_torch_dtype,
    stamp_environment_shapes_from_data,
)
from tools.multistep_tools.window_dataset.single_step_trajectory_store import (
    SingleStepTrajectoryStore,
)


def execute(cfg: omegaconf.DictConfig, headless: bool = False) -> SingleStepTrajectoryStore:
    """Generate the single-step store from the CSV dataset and (optionally) persist it to HDF5.

    :param cfg: Hydra configuration file
    :param headless: turn environment rendering off (param for commandline flag)
    :return: the in-memory :class:`SingleStepTrajectoryStore` (model dtype).
    """
    exp_dir_relative_path = get_hydra_experiment_cwd(cfg)
    seed_me(cfg, output_torch_rdn_generator=True)

    # .... Configuration setting validation .......................................................
    uder_cfg_validation(cfg)

    # .... Create source single step replay buffer ................................................
    ss_full_time_space_replay_buffers = create_ss_trajectory_replaybuffer_from_csv(
        cfg, exp_dir_relative_path, headless
    )
    stamp_environment_shapes_from_data(
        cfg,
        ss_full_time_space_replay_buffers[0].obs_shape[-1],
        ss_full_time_space_replay_buffers[0].action_shape[-1],
    )

    # .... Single-step trajectory store (O(T), model dtype) .......................................
    store = build_store_from_ss_replay_buffers(cfg, ss_full_time_space_replay_buffers)

    if cfg.pipeline.get("enable_ms_hdf5_saving_to_data_dir", True):
        store_dir = save_single_step_store_with_spec(cfg, store, data_root=resolve_ms_hdf5_write_root(cfg))
        consol_msg_universal_one_liner(f"Single-step trajectory HDF5 store saved to '{store_dir}'")

    return store


def resolve_ms_hdf5_write_root(cfg: omegaconf.DictConfig):
    """``None`` (= the writable data root) unless ``pipeline.ms_hdf5_write_to_stage_dir`` is true,
    in which case the staged root (``<ms_hdf5_stage_dir>/<ms_hdf5_save_dir>``) is returned; a
    ``true`` flag without a configured stage dir fails fast (the GEN-HDF5 job would otherwise
    silently write where no GPU job looks)."""
    if not cfg.pipeline.get("ms_hdf5_write_to_stage_dir", False):
        return None
    stage_root = resolve_ms_hdf5_stage_root(cfg)
    if stage_root is None:
        raise ValueError(
            "pipeline.ms_hdf5_write_to_stage_dir=true requires pipeline.ms_hdf5_stage_dir (set the "
            "RLRC_STAGE_DATA_DIR environment variable or override the key) -- RLRP-824 GEN-HDF5 job."
        )
    return stage_root


def load_or_generate(cfg: omegaconf.DictConfig, headless: bool = False) -> SingleStepTrajectoryStore:
    """Cache lookup then (gated) generation; the entry point of the full pipeline."""
    store = None
    if not cfg.pipeline.get("force_regenerating_saved_ms_hdf5", False):
        store = load_single_step_store_with_spec(cfg, dtype=resolve_source_data_torch_dtype(cfg))
        if store is not None:
            stamp_environment_shapes_from_data(cfg, store.obs_dim, store.act_dim)
    if store is None:
        if not cfg.pipeline.get("ms_hdf5_allow_generate", True):
            raise FileNotFoundError(
                "No cached single-step trajectory HDF5 store for this dataset and "
                "pipeline.ms_hdf5_allow_generate=false: generate it once with the CPU GEN-HDF5 job "
                "(robotic_3d_env_generate_ms_hdf5_pipeline) into RLRC_STAGE_DATA_DIR / the data "
                "root before launching GPU jobs (RLRP-824 FR13d, plan risk R10)."
            )
        store = execute(cfg, headless)
    return store
