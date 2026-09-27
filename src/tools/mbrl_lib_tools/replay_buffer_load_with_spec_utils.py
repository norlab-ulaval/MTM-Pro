# coding=utf-8
import os
from typing import Union

import numpy as np
import omegaconf
from mbrl.util import ReplayBuffer, common as common_util

from tools.console_tools.message import consol_msg_universal_one_liner


def load_replay_buffer_with_spec(
    cfg: omegaconf.DictConfig,
    replay_buffer_experiment_path: str,
    seed: Union[int, None] = None,
) -> ReplayBuffer:
    """Load a pre-recorded replay buffer for learning a dynamic model.

    :param cfg: An hydra config file.
    :param replay_buffer_experiment_path: Path to a saved replay buffer.
    :param seed: (optional) Random number generator seed for the replay buffer
    :return: A replay buffer populated with pre-recorded samples.
    """
    consol_msg_universal_one_liner(
        f"loading saved replay buffer from path {replay_buffer_experiment_path}"
    )
    root = cfg.project_root_path
    data_dir_root = "mbrl_data"
    replay_dir = "saved_replay_buffer"
    replay_buffer_experiment_path = os.path.join(
        root, replay_buffer_experiment_path, data_dir_root, replay_dir
    )

    with open(f"{replay_buffer_experiment_path}/replaybuffer_spec.yaml") as file:
        replay_buffer_spec = omegaconf.OmegaConf.load(file)

    # .... Update cfg and var base on saved replay buffer spec ................................

    # Note: `dataset_size` is used by mbrl-lib ReplayBuffer init
    omegaconf.OmegaConf.update(
        cfg,
        "algorithm.dataset_size",
        replay_buffer_spec.capacity,
        merge=False,
    )

    collect_trajectories_ = replay_buffer_spec.stores_trajectories
    omegaconf.OmegaConf.update(
        cfg,
        "overrides.postprocess_replay_buffer.collect_trajectories",
        collect_trajectories_,
        merge=False,
    )
    omegaconf.OmegaConf.update(
        cfg,
        "overrides.sampler_rollout.trajectory_max_length",
        replay_buffer_spec.sampler_rollout.max_trajectory_length,
        merge=False,
    )

    buffer_spec_history_len = replay_buffer_spec.motion_model.history_len
    cfg_history_len = cfg.overrides.motion_model.history_len
    assert buffer_spec_history_len == cfg_history_len, (
        f"multistep configuration mismatch. The "
        f"loaded replay buffer is `history_len={buffer_spec_history_len}` but current "
        f"cfg is `history_len={cfg_history_len}`"
    )

    buffer_spec_horizon_len = horizon_len_factor_to_int(
        replay_buffer_spec.motion_model.horizon_len, buffer_spec_history_len
    )
    cfg_horizon_len = horizon_len_factor_to_int(
        cfg.overrides.motion_model.horizon_len, cfg_history_len
    )
    assert buffer_spec_horizon_len == cfg_horizon_len, (
        f"multistep configuration mismatch. The "
        f"loaded replay buffer is `horizon_len={buffer_spec_horizon_len}` but current "
        f"cfg is `horizon_len={cfg_horizon_len}`"
    )
    omegaconf.OmegaConf.update(
        cfg, "dynamics_model.horizon_len", cfg_horizon_len, merge=False, force_add=True
    )

    omegaconf.OmegaConf.update(
        cfg,
        "overrides.exploration_policy",
        replay_buffer_spec.exploration_policy,
        merge=False,
        force_add=True,
    )
    omegaconf.OmegaConf.update(
        cfg,
        "overrides.domain_randomisation",
        replay_buffer_spec.domain_randomisation,
        merge=False,
        force_add=True,
    )

    use_double_dtype = replay_buffer_spec.normalize_double_precision
    omegaconf.OmegaConf.update(
        cfg,
        "algorithm.normalize_double_precision",
        use_double_dtype,
        merge=False,
    )
    dtype = np.double if use_double_dtype else np.float32

    raw_replay_buffer = common_util.create_replay_buffer(
        cfg,
        (replay_buffer_spec.obs_shape,),
        (replay_buffer_spec.action_shape,),
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
        rng=np.random.default_rng(seed=seed),
        collect_trajectories=collect_trajectories_,
        load_dir=replay_buffer_experiment_path,
    )

    consol_msg_universal_one_liner(
        "Loaded replay buffer: "
        f"num_stored={raw_replay_buffer.num_stored}, "
        f"capacity={raw_replay_buffer.capacity}, "
        f"collect_trajectories={raw_replay_buffer.stores_trajectories}, "
        f"dtype={dtype}"
    )
    return raw_replay_buffer


def horizon_len_factor_to_int(horizon_len: Union[int, float], history_len: int) -> int:
    if isinstance(horizon_len, float):
        horizon_len = round(history_len * horizon_len)
        if horizon_len < 1:
            horizon_len = 1
    return horizon_len
