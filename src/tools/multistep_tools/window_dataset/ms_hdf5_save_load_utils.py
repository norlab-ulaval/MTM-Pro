# coding=utf-8
"""Spec-guarded HDF5 cache of the single-step trajectory store (RLRP-824 Step 6a, KD2 / FR13c-d).

Permanent module. Introduced by Step 6a of the Ultra-long-horizon MS->MS training via a lazy
window DataLoader ``.junie`` plan (``ultra_long_horizon_dataloader_plan_RLRP-824_20260915.md``).
Mirror of ``tools.multistep_tools.ms_replaybuffer_save_load_utils`` for the ``dataloader`` path.

Layout::

    <data_root>/<pipeline.ms_hdf5_save_dir>/<environment.name>/source_size=<value>_obsD=<N>_actD=<M>/
        single_step_store.h5
        single_step_store_spec.yaml

The store holds SINGLE-STEP trajectories, so -- unlike the materialized ``HI<h>_HO<ho>`` replay
buffers -- ONE cache serves every ``(history_len, horizon_len)`` window shape; the directory key
is the dataset identity only (simulator cfg name, source sub-sampling size, RLRP-798
feature-dimension tag). The spec records ``obs_dims`` / ``act_dims`` by NAME and is validated on
load exactly like the replay-buffer spec (an equal-width dim rename is invisible in the dir name).

Lookup precedence on load (FR13c): ``cfg.pipeline.ms_hdf5_stage_dir`` (a read-only staged copy,
``${oc.env:RLRC_STAGE_DATA_DIR,null}`` -- the Apptainer ``--bind ...:ro`` of the SLURM jobs) is
tried FIRST, then ``<project_root>/<environment.data_path>/<ms_hdf5_save_dir>``. Generation is
gated by ``cfg.pipeline.ms_hdf5_allow_generate`` (default ``true``; staged SLURM GPU jobs pass
``false`` so a cache miss fails fast instead of racing on regeneration, plan risk R10 -- the CPU
GEN-HDF5 job ``slurm_jobs/RLRP-757-GEN-HDF5/`` writes the store into the staged root with
``pipeline.ms_hdf5_write_to_stage_dir: true``) and writes atomically (``save_hdf5`` -> ``.tmp`` +
``os.replace``).
"""
from __future__ import annotations

import os
from typing import Optional, Union

import omegaconf
import torch

from tools.console_tools.message import consol_msg_universal_one_liner
from tools.hydra_apps_tools.hydra_utils import get_hydra_original_cwd
from tools.multistep_tools.ms_replaybuffer_save_load_utils import (
    _resolve_ms_replaybuffer_feature_dim_tag,
    _validate_ms_replaybuffer_spec_feature_dims,
)
from tools.multistep_tools.window_dataset.single_step_trajectory_store import (
    SingleStepTrajectoryStore,
)

STORE_FILE_NAME = "single_step_store.h5"
SPEC_FILE_NAME = "single_step_store_spec.yaml"
_SPEC_FORMAT_VERSION = 1


def resolve_ms_hdf5_save_dir_name(cfg: omegaconf.DictConfig) -> str:
    return str(cfg.pipeline.get("ms_hdf5_save_dir", "ms_hdf5"))


def resolve_ms_hdf5_data_root(cfg: omegaconf.DictConfig) -> str:
    """``<project_root>/<environment.data_path>/<ms_hdf5_save_dir>`` (the writable cache root)."""
    data_path = cfg.environment.data_path
    base = data_path if os.path.isabs(data_path) else os.path.join(cfg.project_root_path, data_path)
    return os.path.realpath(os.path.join(base, resolve_ms_hdf5_save_dir_name(cfg)))


def resolve_ms_hdf5_stage_root(cfg: omegaconf.DictConfig) -> Optional[str]:
    """``<pipeline.ms_hdf5_stage_dir>/<ms_hdf5_save_dir>`` when a staged read-only copy is
    configured (``null`` / unset -> ``None``); relative paths resolve against the hydra original
    cwd."""
    stage_dir = cfg.pipeline.get("ms_hdf5_stage_dir", None)
    if stage_dir in (None, "", "null", "None"):
        return None
    stage_dir = os.fspath(stage_dir)
    if not os.path.isabs(stage_dir):
        stage_dir = os.path.join(get_hydra_original_cwd(cfg), stage_dir)
    return os.path.realpath(os.path.join(stage_dir, resolve_ms_hdf5_save_dir_name(cfg)))


def resolve_ms_hdf5_store_dir(cfg: omegaconf.DictConfig, data_root: Union[str, os.PathLike]) -> str:
    """``<data_root>/<environment.name>/source_size=<value>_obsD=<N>_actD=<M>/``."""
    return os.path.realpath(
        os.path.join(
            os.fspath(data_root),
            str(cfg.environment.name),
            f"source_size={cfg.source_replay_buffer.source_size}_{_resolve_ms_replaybuffer_feature_dim_tag(cfg)}",
        )
    )


def _dtype_spec(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def save_single_step_store_with_spec(
    cfg: omegaconf.DictConfig,
    store: SingleStepTrajectoryStore,
    data_root: Optional[Union[str, os.PathLike]] = None,
) -> str:
    """Write ``single_step_store.h5`` + its spec under the canonical directory; returns the dir."""
    from tools.feature_handling_tools.env_handlers import _read_dims

    data_root = resolve_ms_hdf5_data_root(cfg) if data_root is None else os.fspath(data_root)
    store_dir = resolve_ms_hdf5_store_dir(cfg, data_root)
    os.makedirs(store_dir, exist_ok=True)
    obs_dims, act_dims = _read_dims(cfg)
    spec = omegaconf.OmegaConf.create(
        {
            "format_version": _SPEC_FORMAT_VERSION,
            "environment_name": str(cfg.environment.name),
            "source_size": str(cfg.source_replay_buffer.source_size),
            "num_trajectories": int(store.num_trajectories),
            "num_samples": int(store.num_samples),
            "obs_dim": int(store.obs_dim),
            "act_dim": int(store.act_dim),
            "dtype": _dtype_spec(store.dtype),
            "obs_dims": list(obs_dims),
            "act_dims": list(act_dims),
            "obs_dim_count": len(obs_dims),
            "act_dim_count": len(act_dims),
            "store_file": STORE_FILE_NAME,
        }
    )
    store_path = store.save_hdf5(os.path.join(store_dir, STORE_FILE_NAME))
    spec_path = os.path.join(store_dir, SPEC_FILE_NAME)
    tmp_spec = spec_path + ".tmp"
    with open(tmp_spec, "w") as fh:
        omegaconf.OmegaConf.save(spec, fh)
    os.replace(tmp_spec, spec_path)
    consol_msg_universal_one_liner(
        f"Single-step store saved to {store_path!r} ({store.nbytes / 2**20:.1f} MB in RAM, "
        f"{os.path.getsize(store_path) / 2**20:.1f} MB on disk); spec {spec_path!r}"
    )
    return store_dir


def _try_load_from_dir(
    cfg: omegaconf.DictConfig, store_dir: str, dtype: Optional[torch.dtype]
) -> Optional[SingleStepTrajectoryStore]:
    store_path = os.path.join(store_dir, STORE_FILE_NAME)
    spec_path = os.path.join(store_dir, SPEC_FILE_NAME)
    if not os.path.isdir(store_dir):
        return None
    if not os.path.isfile(store_path) or not os.path.isfile(spec_path):
        raise FileNotFoundError(
            f"Single-step store directory {store_dir!r} exists but {STORE_FILE_NAME!r} / "
            f"{SPEC_FILE_NAME!r} is missing -- incomplete or corrupted cache; delete it and let the "
            "pipeline regenerate (or run the GEN-HDF5 job)."
        )
    with open(spec_path) as fh:
        spec = omegaconf.OmegaConf.load(fh)
    if int(spec.get("format_version", -1)) != _SPEC_FORMAT_VERSION:
        raise ValueError(
            f"{spec_path!r}: unsupported single-step store spec format_version "
            f"{spec.get('format_version')} (expected {_SPEC_FORMAT_VERSION})"
        )
    _validate_ms_replaybuffer_spec_feature_dims(cfg, spec, spec_path)
    store = SingleStepTrajectoryStore.load_hdf5(store_path, dtype=dtype)
    if store.num_samples != int(spec.num_samples) or store.num_trajectories != int(spec.num_trajectories):
        raise ValueError(
            f"{store_path!r} disagrees with its spec: {store.num_samples} samples / "
            f"{store.num_trajectories} trajectories vs spec {spec.num_samples} / {spec.num_trajectories}"
        )
    consol_msg_universal_one_liner(f"Single-step store loaded from {store_path!r}: {store!r}")
    return store


def load_single_step_store_with_spec(
    cfg: omegaconf.DictConfig,
    dtype: Optional[torch.dtype] = None,
    data_root: Optional[Union[str, os.PathLike]] = None,
) -> Optional[SingleStepTrajectoryStore]:
    """Load the cached store (staged read-only root first, then the data root); ``None`` on a
    plain cache miss; raises on a corrupted / mismatched cache."""
    roots = []
    stage_root = resolve_ms_hdf5_stage_root(cfg)
    if stage_root is not None:
        roots.append(("stage", stage_root))
    roots.append(("data", resolve_ms_hdf5_data_root(cfg) if data_root is None else os.fspath(data_root)))
    for label, root in roots:
        store_dir = resolve_ms_hdf5_store_dir(cfg, root)
        store = _try_load_from_dir(cfg, store_dir, dtype)
        if store is not None:
            consol_msg_universal_one_liner(f"Single-step store cache hit ({label} root): {store_dir!r}")
            return store
        consol_msg_universal_one_liner(f"No single-step store at {store_dir!r} ({label} root)")
    return None
