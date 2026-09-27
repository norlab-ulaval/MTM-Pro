# coding=utf-8
from typing import Optional, Union

import numpy as np
import torch
from mbrl.util.replay_buffer import ReplayBuffer

from pipeline.pipeline_utils.robotic_env_pipeline_utils.quadcopter_trajectory_dataclass import (
    QuadcopterRobotic3D,
)
from pipeline.pipeline_utils.robotic_env_pipeline_utils.ugv_trajectory_dataclass import (
    UGVRobotic3D,
)
from tools.console_tools.message import consol_msg_universal_one_liner
from tools.mbrl_lib_tools.replaybuffer_tools import (
    check_replay_buffer_is_all_finite,
    resolve_target_replay_buffer_device,
)
from trajectory_container_tools.dataclasses.panda_dataframe_feature_dataclass import (
    BaseDataframeStampedFeatureDataclass,
)


def collect_full_time_space_rollout(
    trj_container: Union[
        QuadcopterRobotic3D, UGVRobotic3D, BaseDataframeStampedFeatureDataclass
    ],
    replace_rewards_with_timestep_index: bool = False,
    max_trajectory_length=None,
    double_precision: Optional[bool] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> ReplayBuffer:
    """
    Collects a rollout spanning the full time-space of the environment and stores it in
    the provided replay buffer.

    :param double_precision:
    :param trj_container: A gym/gumnasium environment with a 1 dimension single discrete action (forward).
    :param replace_rewards_with_timestep_index:
    :param max_trajectory_length: (optional) Store trajectory information, terminated, truncated
     and close trajectory at max len (required for using SequenceTransitionIterator).
    :param device: (optional) storage device of the produced replay buffer. RLRP-775
     action ``A20``: this is a COLLECTION-time buffer (filled row by row and usually
     serialized to disk right after), so it defaults to the host — the training-time
     placement decision belongs to the load path
     (``cfg.mbrl_lib.keep_replay_buffer_on_device``, action ``A17``). Set it only when
     the collected buffer is consumed on-device in the same process; the buffer is
     still built on the host and relocated ONCE at the end.
    :return: A `ReplayBuffer` object filled with the full time-space rollout data from `env`.
    """
    target_device = resolve_target_replay_buffer_device(device)

    if double_precision is None:
        obs_type = trj_container.obs.dtype
        action_type = trj_container.act.dtype
    else:
        obs_type = np.double if double_precision else np.float32
        action_type = np.double if double_precision else np.float32

    trajectory_len_wo_terminal_state = trj_container.trajectory_len - 1
    ss_obs_shape = (trj_container.obs.shape[-1],)
    ss_act_shape = (trj_container.act.shape[-1],)
    full_time_space_replay_buffer = ReplayBuffer(
        capacity=trajectory_len_wo_terminal_state,
        obs_shape=ss_obs_shape,
        action_shape=ss_act_shape,
        obs_type=obs_type,
        action_type=action_type,
        max_trajectory_length=max_trajectory_length,
    )

    # .... Crawl the environment full time space ..................................................
    # Note: New noise is generated on environment reset

    t_count = 0
    terminated = False
    truncated = False
    while (
        full_time_space_replay_buffer.num_stored
        < full_time_space_replay_buffer.capacity
    ):

        if (
            full_time_space_replay_buffer.num_stored + 1
            == full_time_space_replay_buffer.capacity
        ) and not terminated:
            truncated = True

        full_time_space_replay_buffer.add(
            obs=trj_container.obs[t_count],
            action=trj_container.act[t_count],
            next_obs=trj_container.obs[t_count + 1],
            reward=1.0,
            terminated=terminated,
            truncated=truncated,
        )

        done = terminated or truncated
        if done:
            # # (CRITICAL) ToDo: validate explicit closing step  (ref task RLRP-220)
            # if full_time_space_replay_buffer.stores_trajectories:
            #     full_time_space_replay_buffer.close_trajectory()
            break

        if (
            full_time_space_replay_buffer.stores_trajectories
            and t_count % max_trajectory_length == 0
        ):
            full_time_space_replay_buffer.close_trajectory()

        t_count += 1

    if (
        full_time_space_replay_buffer.stores_trajectories
        and len(full_time_space_replay_buffer.trajectory_indices) == 0
        and max_trajectory_length >= full_time_space_replay_buffer.num_stored
    ):
        full_time_space_replay_buffer.close_trajectory()

    if replace_rewards_with_timestep_index:
        full_time_space_replay_buffer.reward = trj_container.timesteps_indices[
            :trajectory_len_wo_terminal_state
        ]

    # .... Sanity check ...........................................................................
    check_replay_buffer_is_all_finite(full_time_space_replay_buffer)

    assert (
        full_time_space_replay_buffer.num_stored
        == full_time_space_replay_buffer.capacity
    ), f"{full_time_space_replay_buffer.num_stored=} == {full_time_space_replay_buffer.capacity=}"

    assert (
        full_time_space_replay_buffer.num_stored
        == full_time_space_replay_buffer.capacity
    )

    # RLRP-775 action ``A20``: single bulk relocation of the finished buffer.
    if target_device is not None:
        full_time_space_replay_buffer.to(target_device)

    return full_time_space_replay_buffer
