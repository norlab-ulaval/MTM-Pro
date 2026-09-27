# coding=utf-8
"""Hydra-cfg → lazy window data path builders (RLRP-824 Step 6a, FR1 / KD2 / KD3 / KD5 / KD12).

Permanent module. Introduced by Step 6a of the Ultra-long-horizon MS->MS training via a lazy
window DataLoader ``.junie`` plan (``ultra_long_horizon_dataloader_plan_RLRP-824_20260915.md``).

The two full pipelines (``robotic_3d_env_full_pipeline.py`` / ``math_env_full_pipeline.py``) branch
on ``cfg.pipeline.data_manager``:

- ``replay-buffer`` (default, legacy, bit-exact): the materialized multistep mbrl ``ReplayBuffer``.
- ``dataloader``: single-step trajectories kept ONCE in a
  :class:`~tools.multistep_tools.window_dataset.single_step_trajectory_store.SingleStepTrajectoryStore`
  (HDF5-cached on the robotic path, in-memory on the math path), composed lazily per batch by a
  :class:`~tools.multistep_tools.window_dataset.multistep_window_dataset.MultistepWindowDataset` and
  served to the ERLL through a
  :class:`~tools.multistep_tools.window_dataset.erll_data_source.WindowDataLoaderDataSource`.

Everything cfg-driven for the ``dataloader`` branch lives here so both pipelines share one
implementation: data-manager resolution, model-dtype resolution, store construction from the
single-step buffers, the legacy ``explorable_space`` anchor restriction (protocol parity with
``restrict_replay_buffer_to_explorable_region`` -- windows are composed on the FULL trajectory,
rows whose anchor ``t`` falls outside the intervals are dropped), dataset / source construction
from ``cfg.mbrl_lib.dataloader`` + ``cfg.UDER.batch_size`` + ``cfg.seed`` +
``cfg.pipeline.dataloader``, the store residency (``cfg.pipeline.dataloader.store_device``, see
:func:`move_store_to_training_device`), and the ``cfg.environment.obs_shape`` /
``source_replay_buffer.original_dataset_size`` stamping the legacy path performs in
``setup_source_multi_step_replay_buffer``.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

import omegaconf
import torch
from mbrl.util.replay_buffer import ReplayBuffer

from tools.console_tools.message import consol_msg_universal_one_liner
from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist
from tools.multistep_tools.window_dataset.erll_data_source import (
    DATA_MANAGER_DATALOADER,
    DATA_MANAGER_REPLAY_BUFFER,
    DATA_MANAGERS,
    DEFAULT_NORMALIZER_FIT_MAX_WINDOWS,
    WindowDataLoaderDataSource,
)
from tools.multistep_tools.window_dataset.multistep_window_dataset import (
    MultistepWindowDataset,
)
from tools.multistep_tools.window_dataset.single_step_trajectory_store import (
    BOUNDARY_MODES,
    AnchorIndex,
    SingleStepTrajectoryStore,
)

DATA_MANAGER_OPTIONS: Tuple[str, ...] = tuple(DATA_MANAGERS)

# ``cfg.pipeline.dataloader.store_device`` keywords (an explicit torch device string such as
# ``cuda:0`` is accepted too).
STORE_DEVICE_AUTO = "auto"
STORE_DEVICE_CPU = "cpu"
STORE_DEVICE_MODEL = "model"
STORE_DEVICE_KEYWORDS: Tuple[str, ...] = (STORE_DEVICE_AUTO, STORE_DEVICE_CPU, STORE_DEVICE_MODEL)


# ==== cfg resolution =============================================================================
def resolve_data_manager(cfg: omegaconf.DictConfig) -> str:
    """``cfg.pipeline.data_manager`` with the legacy default (``replay-buffer``) and a fail-fast
    on unknown values."""
    value = str(cfg.pipeline.get("data_manager", DATA_MANAGER_REPLAY_BUFFER)).strip().lower()
    aliases = {"replay_buffer": DATA_MANAGER_REPLAY_BUFFER, "replaybuffer": DATA_MANAGER_REPLAY_BUFFER}
    value = aliases.get(value, value)
    if value not in DATA_MANAGER_OPTIONS:
        raise ValueError(
            f"pipeline.data_manager={cfg.pipeline.get('data_manager')!r} is not one of "
            f"{DATA_MANAGER_OPTIONS} (RLRP-824 FR1)"
        )
    return value


def is_dataloader_data_manager(cfg: omegaconf.DictConfig) -> bool:
    return resolve_data_manager(cfg) == DATA_MANAGER_DATALOADER


def resolve_model_cfg_key(cfg: omegaconf.DictConfig) -> Optional[str]:
    """The model config node key in use (``ms_model`` for the multistep pipelines)."""
    for each_key in ("ms_model", "ss_model", "model"):
        if is_cfg_key_exist(cfg, each_key):
            return each_key
    return None


def resolve_model_torch_dtype(cfg: omegaconf.DictConfig) -> torch.dtype:
    """The model dtype from ``<model>.model_use_double_precision`` (default ``float32``)."""
    key = resolve_model_cfg_key(cfg)
    use_double = bool(cfg[key].get("model_use_double_precision", False)) if key else False
    return torch.float64 if use_double else torch.float32


def resolve_source_data_torch_dtype(cfg: omegaconf.DictConfig) -> torch.dtype:
    """The dtype of the SOURCE data (single-step buffers, store, dataset batches) on the
    ``dataloader`` path: the NORMALIZER dtype.

    Resolution mirrors ``OneDTransitionRewardModelV2.__init__``:
    ``one_dim_transition_model.normalize_double_precision`` when it is a bool, otherwise (``null`` /
    absent) the model dtype (``resolve_model_torch_dtype``).

    Rationale: the normalizers compute ``(x - mu) / sigma`` at ``max(input dtype, statistics
    dtype)`` and PRESERVE the input dtype (``ZScoreNormalizer.normalize``), and the cast to the
    model compute dtype happens afterwards in ``OneDTransitionRewardModelV2._get_model_input`` /
    ``_process_batch``. The buffer dtype therefore fixes the precision of the RAW data entering
    the normalization -- a ``float64`` normalizer fed ``float32`` buffers works on already
    truncated data, and a ``float32`` normalizer fed ``float64`` buffers (the ``RLRP-757L-E2E1000``
    logs) doubles the host footprint for nothing. Aligning the source dtype on the normalizer
    dtype is the only choice that keeps the aligned-precision design (normalizer precision
    controlled independently of the model precision) meaningful end-to-end.
    """
    use_double = omegaconf.OmegaConf.select(
        cfg, "one_dim_transition_model.normalize_double_precision", default=None
    )
    if use_double is None:
        return resolve_model_torch_dtype(cfg)
    return torch.float64 if bool(use_double) else torch.float32


def resolve_source_buffer_double_precision(
    cfg: omegaconf.DictConfig, legacy_double_precision: Optional[bool] = None
) -> Optional[bool]:
    """``double_precision`` argument of the single-step rollout collection
    (``collect_full_time_space_rollout``), shared by the math-env and the robotic-3D-env paths.

    On the ``dataloader`` path the single-step buffers are collected directly in the SOURCE dtype
    (:func:`resolve_source_data_torch_dtype`, i.e. the normalizer dtype -- the store converts to
    it anyway, so any other intermediate dtype is pure waste). On the legacy ``replay-buffer`` path
    *legacy_double_precision* is returned unchanged (bit-exact, FR1): the math env passes ``None``
    (environment dtype), the robotic env passes its historical
    ``<model>.model_use_double_precision``.
    """
    if not is_dataloader_data_manager(cfg):
        return legacy_double_precision
    return resolve_source_data_torch_dtype(cfg) == torch.float64


def resolve_dataloader_pipeline_cfg(cfg: omegaconf.DictConfig) -> Tuple[int, str]:
    """``(normalizer_fit_max_windows, boundary_mode)`` from ``cfg.pipeline.dataloader`` (defaults
    ``65536`` / ``pad`` = legacy anchor set)."""
    node = cfg.pipeline.get("dataloader", None) or {}
    max_windows = int(node.get("normalizer_fit_max_windows", DEFAULT_NORMALIZER_FIT_MAX_WINDOWS))
    boundary_mode = str(node.get("boundary_mode", "pad"))
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(
            f"pipeline.dataloader.boundary_mode={boundary_mode!r} must be one of {BOUNDARY_MODES}"
        )
    return max_windows, boundary_mode


def resolve_store_device(
    cfg: omegaconf.DictConfig, model_device: Union[str, torch.device, None]
) -> torch.device:
    """The residency of the single-step store from ``cfg.pipeline.dataloader.store_device``.

    - ``auto`` (default): the model device when it is a CUDA device, otherwise CPU. A CPU-resident
      store with a CPU model gains nothing from a move, and the gather is cheap relative to a CPU
      forward pass anyway; on CUDA the device-resident gather removes the host bottleneck measured
      on Valeria (2026-09-17: ~3.5 s host gather + H2D per ~1 s of GPU work at ``F=500``).
    - ``cpu``: legacy behaviour (host gather + per-batch H2D copy).
    - ``model``: always the model device (whatever its type).
    - any other value: parsed as a ``torch.device`` (``cuda:1`` ...).
    """
    node = cfg.pipeline.get("dataloader", None) or {}
    requested = str(node.get("store_device", STORE_DEVICE_AUTO)).strip().lower()
    model_device = torch.device(model_device) if model_device is not None else torch.device("cpu")
    if requested == STORE_DEVICE_AUTO:
        return model_device if model_device.type == "cuda" else torch.device("cpu")
    if requested == STORE_DEVICE_CPU:
        return torch.device("cpu")
    if requested == STORE_DEVICE_MODEL:
        return model_device
    try:
        return torch.device(requested)
    except (RuntimeError, TypeError) as e:
        raise ValueError(
            f"pipeline.dataloader.store_device={requested!r} must be one of {STORE_DEVICE_KEYWORDS} "
            f"or a valid torch device string"
        ) from e


# ==== store / anchors ============================================================================
def build_store_from_ss_replay_buffers(
    cfg: omegaconf.DictConfig, ss_replay_buffers: Iterable[ReplayBuffer]
) -> SingleStepTrajectoryStore:
    """``SingleStepTrajectoryStore.from_replay_buffers`` in the source (normalizer) dtype, with a
    console line."""
    store = SingleStepTrajectoryStore.from_replay_buffers(
        ss_replay_buffers, dtype=resolve_source_data_torch_dtype(cfg)
    )
    consol_msg_universal_one_liner(f"Single-step trajectory store built: {store!r}")
    return store


def move_store_to_training_device(
    cfg: omegaconf.DictConfig,
    store: SingleStepTrajectoryStore,
    model_device: Union[str, torch.device, None],
) -> SingleStepTrajectoryStore:
    """Place the store on :func:`resolve_store_device` (identity when already there), with a
    console line. Call it BEFORE :func:`build_window_dataset` so the anchors, the split subsets and
    every composed batch inherit the residency (the ``WindowDataLoaderDataSource`` then resolves
    ``num_workers=0`` / ``pin_memory=False`` by itself)."""
    target = resolve_store_device(cfg, model_device)
    if store.device == target:
        return store
    moved = store.to(target)
    consol_msg_universal_one_liner(
        f"Single-step trajectory store moved {store.device} -> {moved.device} "
        f"({moved.nbytes / 2**20:.1f} MB; windows are now composed on-device, no per-batch H2D copy)"
    )
    return moved


def explorable_anchor_mask(cfg: omegaconf.DictConfig, anchors: AnchorIndex) -> Optional[torch.Tensor]:
    """Boolean mask of the anchors whose ``t`` lies in ``cfg.environment.explorable_space``.

    Protocol parity with the legacy path: ``restrict_replay_buffer_to_explorable_region`` keeps the
    composed rows ``t`` with ``a <= t < b - 1`` for every interval ``[a, b]`` (the ``b`` end is the
    next-obs index, see ``remove_next_obs_idx_from_inteval_end``), AFTER the windows were composed
    on the full trajectory -- so the kept windows still see the data outside the intervals exactly
    as the legacy buffer rows do. Every trajectory (one per single-step rollout) shares the same
    intervals. ``None`` when the cfg declares no ``explorable_space``.
    """
    if not is_cfg_key_exist(cfg, "environment.explorable_space"):
        return None
    intervals = omegaconf.OmegaConf.to_object(cfg.environment.explorable_space)
    if not intervals:
        return None
    mask = torch.zeros(len(anchors), dtype=torch.bool, device=anchors.device)
    for start, stop in intervals:
        lo, hi = int(start), int(stop) - 1  # rows [lo, hi)
        mask |= (anchors.ts >= lo) & (anchors.ts < hi)
    return mask


def restrict_anchors_to_explorable_region(cfg: omegaconf.DictConfig, anchors: AnchorIndex) -> AnchorIndex:
    """Return ``anchors`` restricted to ``cfg.environment.explorable_space`` (identity when unset)."""
    mask = explorable_anchor_mask(cfg, anchors)
    if mask is None or bool(mask.all()):
        return anchors
    if not bool(mask.any()):
        raise ValueError(
            "environment.explorable_space leaves zero anchors "
            f"(intervals={omegaconf.OmegaConf.to_object(cfg.environment.explorable_space)})"
        )
    restricted = AnchorIndex(
        traj_ids=anchors.traj_ids[mask],
        ts=anchors.ts[mask],
        history_len=anchors.history_len,
        horizon_len=anchors.horizon_len,
        boundary_mode=anchors.boundary_mode,
    )
    consol_msg_universal_one_liner(
        f"Explorable-region restriction: {len(anchors)} -> {len(restricted)} anchors "
        f"(environment.explorable_space={omegaconf.OmegaConf.to_object(cfg.environment.explorable_space)})"
    )
    return restricted


# ==== dataset / source ===========================================================================
def build_window_dataset(
    cfg: omegaconf.DictConfig,
    store: SingleStepTrajectoryStore,
    output_window_len: Optional[int] = None,
) -> MultistepWindowDataset:
    """The FULL-anchor ``MultistepWindowDataset`` of ``cfg.ms_model.{history_len,horizon_len}``
    (explorable-region restricted, ``boundary_mode`` from ``cfg.pipeline.dataloader``)."""
    history_len = int(cfg.ms_model.history_len)
    horizon_len = int(cfg.ms_model.horizon_len)
    _, boundary_mode = resolve_dataloader_pipeline_cfg(cfg)
    anchors = restrict_anchors_to_explorable_region(
        cfg, store.anchor_index(history_len, horizon_len, boundary_mode=boundary_mode)
    )
    dataset = MultistepWindowDataset(
        store, history_len, horizon_len, anchors=anchors, output_window_len=output_window_len
    )
    consol_msg_universal_one_liner(f"Lazy window dataset built: {dataset!r}")
    return dataset


def build_window_data_source(
    cfg: omegaconf.DictConfig,
    dataset: MultistepWindowDataset,
    device: Union[str, torch.device, None],
) -> WindowDataLoaderDataSource:
    """The run's ``WindowDataLoaderDataSource`` from ``cfg.mbrl_lib.dataloader`` (workers /
    pinning / persistence / prefetch), ``cfg.UDER.batch_size.init_value`` (the FR12 constant batch
    size) and ``cfg.seed`` (split + shuffle seed). ``split()`` is called by the ERLL."""
    dl_cfg = cfg.mbrl_lib.get("dataloader", None) or {}
    max_windows, _ = resolve_dataloader_pipeline_cfg(cfg)
    source = WindowDataLoaderDataSource(
        dataset,
        batch_size=int(cfg.UDER.batch_size.init_value),
        seed=cfg.get("seed", None),
        num_workers=int(dl_cfg.get("num_workers", 0) or 0),
        pin_memory=bool(dl_cfg.get("pin_memory", False)),
        persistent_workers=bool(dl_cfg.get("persistent_workers", True)),
        prefetch_factor=dl_cfg.get("prefetch_factor", None),
        device=device,
        normalizer_fit_max_windows=max_windows,
    )
    consol_msg_universal_one_liner(f"ERLL data source (dataloader path): {source!r}")
    return source


# ==== cfg stamping (legacy side effects of ``setup_source_multi_step_replay_buffer``) ============
def stamp_environment_shapes_from_data(cfg: omegaconf.DictConfig, obs_dim: int, act_dim: int) -> None:
    """Stamp ``environment.obs_shape`` / ``act_shape`` from the DATA (robotic: fail-loud-checked
    against ``obs_dims`` / ``act_dims``; math: data-derived) exactly as the legacy
    ``setup_source_multi_step_replay_buffer`` does, so every downstream interpolation
    (``ms_model.singlestep_obs_len: ${environment.obs_shape[0]}`` ...) resolves."""
    from tools.feature_handling_tools.env_handlers import resolve_act_shape, resolve_obs_shape

    resolved_obs_shape = resolve_obs_shape(cfg, data_shape=[int(obs_dim)])
    resolved_act_shape = resolve_act_shape(cfg, data_shape=[int(act_dim)])
    with omegaconf.read_write(cfg), omegaconf.open_dict(cfg):
        omegaconf.OmegaConf.update(cfg, "environment.obs_shape", resolved_obs_shape, merge=False)
        omegaconf.OmegaConf.update(cfg, "environment.act_shape", resolved_act_shape, merge=False)


def stamp_original_dataset_size(cfg: omegaconf.DictConfig, num_anchors: int) -> None:
    """``source_replay_buffer.original_dataset_size`` = the anchor count (the legacy value is the
    materialized buffer's ``num_stored``, i.e. the same quantity)."""
    with omegaconf.read_write(cfg):
        omegaconf.OmegaConf.update(
            cfg, "source_replay_buffer.original_dataset_size", int(num_anchors), merge=False
        )


def describe_window_path(store: SingleStepTrajectoryStore, dataset: MultistepWindowDataset) -> str:
    """One console line comparing the O(T) store footprint with the materialized-buffer cost."""
    row_floats = dataset.obs_width + dataset.act_dim + dataset.out_size + 3
    materialized = len(dataset) * row_floats * store.obs_all.element_size()
    return (
        f"data_manager=dataloader: store {store.nbytes / 2**20:.1f} MB ({store.num_samples:,} single-step "
        f"samples) serves {len(dataset):,} lazy (H={dataset.history_len}, F={dataset.horizon_len}, "
        f"W={dataset.output_window_len}) windows; a materialized replay buffer of the same rows would be "
        f"~{materialized / 2**30:.2f} GB"
    )
