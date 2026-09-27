# coding=utf-8
import logging
import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import mbrl.models
import numpy as np
import torch
from mbrl.models.one_dim_tr_model import OneDTransitionRewardModel
from mbrl.util import ReplayBuffer
import omegaconf

from tools.mbrl_lib_tools.models.one_dim_tr_model_v2 import OneDTransitionRewardModelV2
from tools.mbrl_lib_tools.replay_buffer_load_with_spec_utils import (
    load_replay_buffer_with_spec,
)
from custom_types.custom_types_all_gym_wrappers import ALL_CUSTOM_GYM_WRAPPER_TYPES
from custom_types.custom_types_learnable_f110_gym_env import LEARNABLE_F110ENV_TYPES
from tools.mbrl_lib_tools.common_tools import (
    create_one_dim_tr_model_v2_explicit_size,
    create_sampling_replay_buffer_from_env,
)


def setup_mbrllib_learning_components(
    cfg: omegaconf.DictConfig,
    sampling_env: Union[LEARNABLE_F110ENV_TYPES, ALL_CUSTOM_GYM_WRAPPER_TYPES],
    in_size: int,
    out_size: int,
    seed: Union[int, None] = None,
) -> Tuple[OneDTransitionRewardModelV2, ReplayBuffer]:
    """Setup model-based reinforcement learning (MBRL) components based on sampling environments
    and target model spec.
    Note: replay buffer space shape and model corresponding in/out size can be different

        1. Set up the replay buffers components
        2. Set up the target dynamic models components

    :param cfg: An hydra config file.
    :param sampling_env: An gym/gymnasium environment.
    :param in_size: Input size of the dynamic model.
    :param out_size: Output size of the dynamic model.
    :param seed: (optional) Random number generator seed for the replay buffer.
    :return: dynamics_model, singlestep_replay_buffer
    """
    dynamics_model = create_one_dim_tr_model_v2_explicit_size(cfg, in_size, out_size)

    replay_buffer_experiment_path = omegaconf.OmegaConf.select(
        cfg, "pipeline.replay_buffer_experiment_path", default=None, throw_on_missing=False
    )
    if replay_buffer_experiment_path:
        singlestep_replay_buffer = load_replay_buffer_with_spec(
            cfg, replay_buffer_experiment_path, seed
        )
        return dynamics_model, singlestep_replay_buffer
    else:
        empty_replay_buffer = create_sampling_replay_buffer_from_env(cfg, sampling_env, seed)
        return dynamics_model, empty_replay_buffer


def setup_saved_model_dir(model: mbrl.models.OneDTransitionRewardModel) -> str:
    return os.path.join(
        f"model_{model.model.__class__.__name__}",
        "saved_dynamic_model",
    )


# --------------------------------------------------------------------
# Training Speed & Efficiency stage-1 follow-up plan helpers.
#
# Centralised cfg → kwargs resolvers for the vendored
# ``utilities/mbrl-lib/mbrl/models/model_trainer.py`` fork.
# Keeping this logic in one place avoids drift between the three
# RLRC call sites that instantiate ``mbrl.models.ModelTrainer`` and
# lets the unit tests assert the legacy-bit-exact defaults once.
# --------------------------------------------------------------------

def resolve_mbrl_dataloader_kwargs_from_cfg(
    cfg: omegaconf.DictConfig,
) -> Dict[str, Any]:
    """Build the F-C2 ``DataLoader`` kwargs dict from a Hydra cfg.

    Reads the opt-in ``cfg.mbrl_lib.dataloader.*`` block introduced by
    the stage-1 follow-up plan (action F-C2). All four keys default to
    the legacy single-process-no-prefetch behaviour — calling this
    helper on a cfg that has no ``mbrl_lib.dataloader`` block returns
    exactly the pre-patch ``ModelTrainer`` kwargs.

    Returned mapping is keyed by the ``ModelTrainer.__init__`` kwarg
    names (``dataloader_num_workers``, ``dataloader_pin_memory``,
    ``dataloader_persistent_workers``, ``dataloader_prefetch_factor``)
    so call sites can splat it with ``**`` directly.

    ``pin_memory`` is **not** gated on the device here — the gate lives
    inside ``ModelTrainer.train(...)`` where ``self.model.device`` is
    known. The helper only resolves the cfg intent.
    """
    num_workers = int(
        omegaconf.OmegaConf.select(
            cfg, "mbrl_lib.dataloader.num_workers", default=0
        )
    )
    pin_memory = bool(
        omegaconf.OmegaConf.select(
            cfg, "mbrl_lib.dataloader.pin_memory", default=False
        )
    )
    persistent_workers = bool(
        omegaconf.OmegaConf.select(
            cfg,
            "mbrl_lib.dataloader.persistent_workers",
            default=False,
        )
    )
    prefetch_factor_raw = omegaconf.OmegaConf.select(
        cfg, "mbrl_lib.dataloader.prefetch_factor", default=None
    )
    prefetch_factor: Optional[int] = (
        int(prefetch_factor_raw) if prefetch_factor_raw is not None else None
    )
    return {
        "dataloader_num_workers": num_workers,
        "dataloader_pin_memory": pin_memory,
        "dataloader_persistent_workers": persistent_workers,
        "dataloader_prefetch_factor": prefetch_factor,
    }


def resolve_mbrl_trainer_opt_in_kwargs_from_cfg(
    cfg: omegaconf.DictConfig,
) -> Dict[str, Any]:
    """Collect every opt-in ``ModelTrainer`` kwarg surfaced in cfg.

    Union of:
      * ``use_preallocated_best_weights_buffer`` (B0-bis, stage-1 main).
      * F-C2 ``dataloader_*`` kwargs (stage-1 follow-up).

    Call-site usage::

        ModelTrainer(
            model,
            optim_lr=...,
            weight_decay=...,
            **resolve_mbrl_trainer_opt_in_kwargs_from_cfg(cfg),
        )
    """
    merged: Dict[str, Any] = {
        "use_preallocated_best_weights_buffer": bool(
            omegaconf.OmegaConf.select(
                cfg,
                "mbrl_lib.use_preallocated_best_weights_buffer",
                default=False,
            )
        ),
    }
    merged.update(resolve_mbrl_dataloader_kwargs_from_cfg(cfg))
    return merged


# --------------------------------------------------------------------
# Training Speed & Efficiency plan — action ``C1`` (parent plan, stage
# 2) merged with stage-1 follow-up action ``F-C1b`` into a single
# submodule ctor patch (``ReplayBuffer(device=...)``). This helper
# centralises the RLRC-side cfg→device resolution + VRAM auto-fallback
# guard so every call-site that instantiates ``ReplayBuffer`` from
# persisted RLRC data gets the same behaviour.
# --------------------------------------------------------------------


_LOGGER = logging.getLogger(__name__)


def _estimate_replay_buffer_bytes(
    capacity: int,
    obs_shape: Sequence[int],
    action_shape: Sequence[int],
    obs_type: Union[torch.dtype, np.dtype, str],
    action_type: Union[torch.dtype, np.dtype, str],
    reward_type: Union[torch.dtype, np.dtype, str],
    max_trajectory_length: Optional[int] = None,
) -> int:
    """Byte-size of the ``TensorDict`` storage ``ReplayBuffer`` will allocate.

    Mirrors the allocation in
    ``utilities/mbrl-lib/mbrl/util/replay_buffer.py:ReplayBuffer.__init__``
    (L537–554 after revision): six tensors of shape
    ``[capacity + max_trajectory_length, ...]`` plus two boolean masks.
    Used by the C1/F-C1b VRAM auto-fallback guard before we decide to
    place the buffer on-device.
    """
    def _itemsize(dtype: Union[torch.dtype, np.dtype, str]) -> int:
        # Resolve both torch.dtype and numpy/string dtypes through torch's
        # dtype table without importing the private helper from mbrl-lib.
        if isinstance(dtype, torch.dtype):
            return torch.empty((), dtype=dtype).element_size()
        return torch.empty((), dtype=torch.from_numpy(np.empty(1, dtype=dtype)).dtype).element_size()

    total_rows = int(capacity) + int(max_trajectory_length or 0)
    obs_numel = int(np.prod(obs_shape))
    act_numel = int(np.prod(action_shape))

    obs_bytes = total_rows * obs_numel * _itemsize(obs_type)
    act_bytes = total_rows * act_numel * _itemsize(action_type)
    rew_bytes = total_rows * 1 * _itemsize(reward_type)
    bool_bytes = total_rows * 1 * torch.empty((), dtype=torch.bool).element_size()

    # observation + next.observation + action + next.reward + 2 bool masks
    return 2 * obs_bytes + act_bytes + rew_bytes + 2 * bool_bytes


def resolve_replay_buffer_device(
    cfg: omegaconf.DictConfig,
    *,
    capacity: int,
    obs_shape: Sequence[int],
    action_shape: Sequence[int],
    obs_type: Union[torch.dtype, np.dtype, str] = torch.float32,
    action_type: Union[torch.dtype, np.dtype, str] = torch.float32,
    reward_type: Union[torch.dtype, np.dtype, str] = torch.float32,
    max_trajectory_length: Optional[int] = None,
    cuda_available: Optional[bool] = None,
    free_vram_bytes: Optional[int] = None,
) -> Optional[torch.device]:
    """Resolve the ``device`` kwarg for ``mbrl.util.ReplayBuffer``.

    Reads ``cfg.mbrl_lib.keep_replay_buffer_on_device`` (default
    ``False``) and ``cfg.mbrl_lib.replay_buffer_max_frac_of_free_vram``
    (default ``0.5``). Returns ``None`` whenever the legacy CPU path is
    selected (flag OFF, no CUDA, or the auto-fallback guard fires).
    Returns ``torch.device("cuda")`` only when the flag is ON, CUDA is
    available, and the estimated buffer byte-size fits within the
    configured fraction of free VRAM.

    ``cuda_available`` / ``free_vram_bytes`` are injection points used by
    the focused tests; in production both are read from
    ``torch.cuda.{is_available, mem_get_info}``.
    """
    enabled = bool(
        omegaconf.OmegaConf.select(
            cfg, "mbrl_lib.keep_replay_buffer_on_device", default=False
        )
    )
    if not enabled:
        return None

    if cuda_available is None:
        cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        _LOGGER.warning(
            "[C1/F-C1b] cfg.mbrl_lib.keep_replay_buffer_on_device=true but "
            "CUDA is not available; staying on CPU (legacy path)."
        )
        return None

    max_frac = float(
        omegaconf.OmegaConf.select(
            cfg,
            "mbrl_lib.replay_buffer_max_frac_of_free_vram",
            default=0.5,
        )
    )
    buffer_bytes = _estimate_replay_buffer_bytes(
        capacity=capacity,
        obs_shape=obs_shape,
        action_shape=action_shape,
        obs_type=obs_type,
        action_type=action_type,
        reward_type=reward_type,
        max_trajectory_length=max_trajectory_length,
    )
    if free_vram_bytes is None:
        try:
            free_vram_bytes, _total = torch.cuda.mem_get_info()
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.warning(
                "[C1/F-C1b] torch.cuda.mem_get_info() failed (%s); "
                "staying on CPU (legacy path).",
                exc,
            )
            return None

    if buffer_bytes > max_frac * float(free_vram_bytes):
        _LOGGER.warning(
            "[C1/F-C1b] cfg.mbrl_lib.keep_replay_buffer_on_device=true but "
            "estimated buffer size %.1f MB > %.0f %% of free VRAM %.1f MB "
            "(cfg.mbrl_lib.replay_buffer_max_frac_of_free_vram=%.2f); "
            "auto-fallback to CPU path.",
            buffer_bytes / 1e6,
            max_frac * 100.0,
            free_vram_bytes / 1e6,
            max_frac,
        )
        return None

    _LOGGER.info(
        "[C1/F-C1b] Placing ReplayBuffer storage on CUDA "
        "(estimated %.1f MB, free VRAM %.1f MB, max_frac=%.2f).",
        buffer_bytes / 1e6,
        free_vram_bytes / 1e6,
        max_frac,
    )
    return torch.device("cuda")
