# coding=utf-8
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch

from mbrl.util.replay_buffer import ReplayBuffer

from math_gymnasium.envs.arbitrary_dim_math_continuous import MathContinuousGymnasium
from tools.console_tools.message import consol_msg_universal_one_liner
from tools.mbrl_lib_tools.replaybuffer_tools import (
    check_replay_buffer_is_all_finite,
    resolve_target_replay_buffer_device,
)


def collect_full_time_space_rollout(
    env: Union[MathContinuousGymnasium, gym.Env],
    replace_rewards_with_timestep_index: bool = False,
    max_trajectory_length=None,
    double_precision: Optional[bool] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> ReplayBuffer:
    """
    Collects a rollout spanning the full time-space of the environment and stores it in
    the provided replay buffer.

    :param env: A gym/gumnasium environment with a 1 dimension single discrete action (forward).
    :param replace_rewards_with_timestep_index:
    :param max_trajectory_length: (optional) Store trajectory information, terminated, truncated
     and close trajectory at max len (required for using SequenceTransitionIterator).
    :param double_precision: (optional) If True, use np.double (float64) for observation and action
     data types. If False, use np.float32. If None, use the data types from the environment's
     observation and action spaces.
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

    action_shape = env.action_space.shape
    obs_shape = env.observation_space.shape

    if double_precision is None:
        obs_type = env.observation_space.dtype
        action_type = env.action_space.dtype
    else:
        obs_type = np.double if double_precision else np.float32
        action_type = np.double if double_precision else np.float32

    trajectory_len_wo_terminal_state = env.trj.trajectory_len - 1
    full_time_space_replay_buffer = ReplayBuffer(
        capacity=trajectory_len_wo_terminal_state,
        obs_shape=obs_shape,
        action_shape=action_shape,
        obs_type=obs_type,
        action_type=action_type,
        max_trajectory_length=max_trajectory_length,
    )

    # .... Crawl the environment full time space ..................................................
    assert action_shape == (
        1,
    ), f"Action space shape greater than 1 are not supported yet"

    # Note: New noise is generated on environment reset
    obs, info = env.reset(time_space_init_idx=0, explorable_space_only=False)

    timestep_count = 0
    while (
        full_time_space_replay_buffer.num_stored
        < full_time_space_replay_buffer.capacity
    ):

        next_obs, reward, terminated, truncated, next_info = env.step()

        if replace_rewards_with_timestep_index:
            reward = env.trj.timesteps_indices[timestep_count]

        if (
            full_time_space_replay_buffer.num_stored + 1
            == full_time_space_replay_buffer.capacity
        ) and not terminated:
            truncated = True

        full_time_space_replay_buffer.add(
            obs=obs["state_axes_obs_with_noise"],
            action=obs["time_axis_obs_with_noise"],
            next_obs=next_obs["state_axes_obs_with_noise"],
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

        done = terminated or truncated
        if done:
            # # (CRITICAL) ToDo: validate explicit closing step  (ref task RLRP-220)
            # if full_time_space_replay_buffer.stores_trajectories:
            #     full_time_space_replay_buffer.close_trajectory()
            break
        else:
            obs = next_obs

        timestep_count += 1
        if (
            full_time_space_replay_buffer.stores_trajectories
            and timestep_count % max_trajectory_length == 0
        ):
            full_time_space_replay_buffer.close_trajectory()

    if (
        full_time_space_replay_buffer.stores_trajectories
        and len(full_time_space_replay_buffer.trajectory_indices) == 0
        and max_trajectory_length >= full_time_space_replay_buffer.num_stored
    ):
        full_time_space_replay_buffer.close_trajectory()

    # (Priority) ToDo: on task end >> delete next bloc ↓↓
    # if replace_rewards_with_timestep_index:
    #     full_time_space_replay_buffer.reward = env.trj.timesteps_indices[
    #         :trajectory_len_wo_terminal_state
    #     ]

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
