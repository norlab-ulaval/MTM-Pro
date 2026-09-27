# coding=utf-8
from typing import Optional, Tuple, Union

import numpy as np
import omegaconf
import torch
from mbrl.types import TransitionBatch
from mbrl.util import ReplayBuffer

from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist
from tools.mbrl_lib_tools.replaybuffer_tools import (
    INHERIT_SOURCE_DEVICE,
    resolve_target_replay_buffer_device,
    transition_batch_to_host,
    was_replay_buffer_filled_over_capacity,
    )


def restrict_replay_buffer_to_explorable_region(
    cfg: omegaconf.DictConfig,
    replay_buffer: ReplayBuffer,
    device: Optional[Union[torch.device, str]] = INHERIT_SOURCE_DEVICE,
) -> ReplayBuffer:
    """Strip replay buffer with a single time space rollout from part of the trajectory that are
    not in of the environment explorable regions.

    Note: Assume that the replay buffer is composed of a full time space trajectory rollout and
    that data ar stored contigously.

    :param cfg: hydra configuration
    :param replay_buffer: a mbrl-lib replay buffer with trajectories stored contigously
    :param device: storage device of the produced replay buffer. Defaults to the
     source ``replay_buffer`` device (RLRP-775 action ``A20``) so a device-resident
     source (``cfg.mbrl_lib.keep_replay_buffer_on_device``, action ``A17``) is not
     silently degraded back to the legacy host path. Pass ``None`` to force the host
     storage.
    :return: a new replay_buffer limited to the explorable region
    """

    # .... Pre-condition ..........................................................................
    was_replay_buffer_filled_over_capacity(replay_buffer)

    # RLRP-775 action ``A20``: the buffer below is filled ROW BY ROW, so it is
    # always built on the host and relocated ONCE at the end (a per-``add`` H→D
    # copy would cost far more than the single bulk move).
    target_device = resolve_target_replay_buffer_device(device, replay_buffer)

    # .... Construct replay buffer restricted to explorable region ................................
    target_capacity = 0
    mock_trajectory_index = np.arange(replay_buffer.num_stored)
    if is_cfg_key_exist(cfg, "environment.explorable_space"):
        explorable_space_list = omegaconf.OmegaConf.to_object(cfg.environment.explorable_space)
    else:
        explorable_space_list = [[0, replay_buffer.num_stored]]

    for each_explorable_interval in explorable_space_list:
        each_explorable_interval = remove_next_obs_idx_from_inteval_end(each_explorable_interval)
        target_capacity += mock_trajectory_index[slice(*each_explorable_interval)].size
    if target_capacity > replay_buffer.num_stored:
        target_capacity = replay_buffer.num_stored

    restricted_replay_buffer = ReplayBuffer(
        capacity=target_capacity,
        obs_shape=replay_buffer.obs_shape,
        action_shape=replay_buffer.action_shape,
        obs_type=replay_buffer.obs_type,
        action_type=replay_buffer.action_type,
        reward_type=replay_buffer.reward_type,
    )

    all_samples: TransitionBatch
    # RLRP-775 action ``A20``: host copy so the per-sample ``terminateds`` /
    # ``truncateds`` python branching below does not synchronize the device on
    # every iteration when the source buffer is device-resident.
    all_samples = transition_batch_to_host(replay_buffer.get_all(shuffle=False))

    for each_explorable_interval in explorable_space_list:
        explorable_interval_slice = slice(*each_explorable_interval)
        explorable_interval_sanity_check(all_samples, explorable_interval_slice)

        for idx in range(explorable_interval_slice.start, explorable_interval_slice.stop):
            each_sample = all_samples[idx]

            truncated = each_sample.truncateds
            terminated = each_sample.terminateds
            if (
                restricted_replay_buffer.num_stored + 1 == restricted_replay_buffer.capacity
                or idx + 1 == explorable_interval_slice.stop
            ) and not terminated:
                truncated = True

            restricted_replay_buffer.add(
                obs=each_sample.obs,
                action=each_sample.act,
                next_obs=each_sample.next_obs,
                reward=each_sample.rewards,
                terminated=terminated,
                truncated=truncated,
            )

    was_replay_buffer_filled_over_capacity(restricted_replay_buffer)

    # RLRP-775 action ``A20``: single bulk relocation of the finished buffer.
    if target_device is not None:
        restricted_replay_buffer.to(target_device)

    return restricted_replay_buffer


def explorable_interval_sanity_check(
    all_samples: TransitionBatch, explorable_interval_slice: slice
) -> None:
    assert (
        explorable_interval_slice.start < explorable_interval_slice.stop
    ), f"{explorable_interval_slice.start=} !< {explorable_interval_slice.stop=}"
    assert explorable_interval_slice.stop <= len(
        all_samples
    ), f"{explorable_interval_slice.stop=} !<= { len(all_samples)=}"
    return None


def remove_next_obs_idx_from_inteval_end(explorable_interval: Tuple[int, int]) -> Tuple[int, int]:
    explorable_interval[1] -= 1
    return explorable_interval


def enforce_explorable_bounds(
    trajectory_start_idx: int,
    trajectory_max_length: Optional[int],
    explorable_interval_slice: slice,
) -> int:
    """Enforces the boundaries of an explorable interval for a given trajectory.
    Remark: Handling interval end index after sampling on purpose so that the explorable
    interval get evenly visited at reset

    :param trajectory_start_idx: The starting index of the trajectory.
    :param trajectory_max_length: The maximum allowable length of the trajectory.
    :param explorable_interval_slice: Define the limits of the explorable interval.
    :return: The adjusted ending index of the trajectory respecting the explorable interval's
    bounds.
    """
    if trajectory_max_length is None:
        trajectory_max_length = np.inf
    trajectory_end_idx = trajectory_start_idx + trajectory_max_length
    if trajectory_end_idx > explorable_interval_slice.stop:
        trajectory_end_idx = explorable_interval_slice.stop
    return trajectory_end_idx
