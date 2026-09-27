# coding=utf-8
from typing import Optional

import numpy as np
from mbrl.util.replay_buffer import ReplayBuffer

from tools.mbrl_lib_tools.replaybuffer_tools import (
    check_replay_buffer_is_all_finite, resolve_replay_buffer_source_device,
    transition_batch_to_host, was_replay_buffer_filled_over_capacity,
    )
from tools.multistep_tools.data_buffer_processor import MultistepDataBufferProcessorAbstract


def convert_singlestep_replaybuffer_to_multistep(
    singlestep_replay_buffer: ReplayBuffer,
    multistep_data_buffer_processor: MultistepDataBufferProcessorAbstract,
    max_trajectory_length: Optional[int] = None,
) -> ReplayBuffer:
    """
    Convert a single-step replay buffer to a multistep replay buffer using a specified
    multistep_data_buffer_processor.

    Note: its the responsability of the user to make sure that trajectories stored in
    singlestep_replay_buffer are contiguous.

    :param singlestep_replay_buffer: The source replay buffer containing single-step transitions
     (with stores_trajectories=False).
    :param multistep_data_buffer_processor: Processor responsible for handling multistep data
     conversion.
    :param max_trajectory_length: (optional) Store trajectory information, terminated, truncated
     and close trajectory at max len (required for using SequenceTransitionIterator).
    :return: A new replay buffer containing multistep transitions.
    """
    ss_rb = singlestep_replay_buffer
    ms_db_processor = multistep_data_buffer_processor
    horizon_len = ms_db_processor.horizon_len
    history_len = ms_db_processor.history_len

    # .... Pre-condition ..........................................................................
    was_replay_buffer_filled_over_capacity(ss_rb)
    check_replay_buffer_is_all_finite(ss_rb)

    assert ss_rb.stores_trajectories is False, (
        f"singlestep_replay_buffer.stores_trajectories is required to be False in order to be "
        f"processed. Trajectory information will be added at convertion if "
        f"max_trajectory_length is set."
    )

    assert (
        max_trajectory_length or np.inf
    ) >= history_len, (
        f"max_trajectory_length {max_trajectory_length} not >= history_len {history_len}"
    )

    # RLRP-775 action ``A19``: the multistep buffer is filled ROW BY ROW
    # below, so it is always built on the host and relocated ONCE at the end
    # to the source-buffer device (a per-``add`` H→D copy would cost far more
    # than the single bulk move).
    target_device = resolve_replay_buffer_source_device(ss_rb)

    # .... Setup multistep buffer .................................................................
    ms_rb = ReplayBuffer(
        capacity=ss_rb.capacity,
        obs_shape=ms_db_processor.composed_observation_space.shape,
        action_shape=ms_db_processor.composed_action_space.shape,
        obs_type=ms_db_processor.composed_observation_space.dtype,
        action_type=ms_db_processor.composed_action_space.dtype,
        reward_type=ss_rb.reward.dtype,
        rng=ss_rb.rng,
        max_trajectory_length=max_trajectory_length,
    )

    # .... Convert single-step buffer .............................................................
    # RLRP-775 action ``A19``: host copy so the per-sample python branching
    # below does not synchronize the device on every iteration.
    all_samples = transition_batch_to_host(ss_rb.get_all(shuffle=False))

    done = True
    ms_reset_history_idx = 0
    recorded_step_count = 0

    for each_sample_idx in range(ss_rb.num_stored):
        each_sample = all_samples[each_sample_idx]

        if done:
            ms_reset_history_idx = 1
            if ms_db_processor.act_buffer_reset_callback:
                ms_db_processor.reset_buffers(each_sample.obs)
            else:
                ms_db_processor.reset_buffers(obs=each_sample.obs, act=each_sample.act)

        ms_db_processor.add_step(
            action=each_sample.act,
            next_obs=each_sample.next_obs,
            reward=each_sample.rewards,
            terminated=each_sample.terminateds,
            truncated=each_sample.truncateds,
        )
        done = each_sample.terminateds or each_sample.truncateds

        if ms_reset_history_idx < horizon_len:
            # Skip multistep data buffer processor offset
            ms_reset_history_idx += 1
        else:
            _add_composed_data_to_multistep_replay_buffer(ms_db_processor, ms_rb)
            recorded_step_count += 1

            if not done:
                if _as_reach_buffer_max_trj_len(max_trajectory_length, recorded_step_count):
                    if ms_rb.stores_trajectories:
                        # Note: Handle the case where trajectories need to be split at
                        # `max_trajectory_length` without using `truncated=True` so that the rest
                        # of the trajectory be preserved in the second split.
                        ms_rb.close_trajectory()

            elif done:
                while not ms_db_processor.horizon_offset_done_is_aligned_with_timestep_t():
                    ms_db_processor.pad_horizon_by_one()
                    _add_composed_data_to_multistep_replay_buffer(ms_db_processor, ms_rb)
                    recorded_step_count += 1

    if (ms_rb.num_stored < ms_rb.capacity) and (ss_rb.num_stored != ms_rb.num_stored):
        while (not ms_db_processor.horizon_offset_done_is_aligned_with_timestep_t()) or (
            ms_rb.num_stored != ms_rb.capacity and ss_rb.num_stored != ms_rb.num_stored
        ):
            ms_db_processor.pad_horizon_by_one()
            _add_composed_data_to_multistep_replay_buffer(ms_db_processor, ms_rb)
            recorded_step_count += 1

    if (
        ms_rb.stores_trajectories
        and len(ms_rb.trajectory_indices) == 0
        and max_trajectory_length >= ms_rb.num_stored
    ):
        ms_rb.close_trajectory()

    _buffers_sanity_check(recorded_step_count, ss_rb, ms_rb)

    # RLRP-775 action ``A19``: single bulk relocation of the finished buffer.
    if target_device is not None:
        ms_rb.to(target_device)

    return ms_rb


def _as_reach_buffer_max_trj_len(
    max_trajectory_length: Optional[int], recorded_step_counter: int
) -> bool:
    return recorded_step_counter % (max_trajectory_length or np.inf) == 0


def _buffers_sanity_check(
    recorded_step_count: int,
    singlestep_replay_buffer: ReplayBuffer,
    multistep_replay_buffer: ReplayBuffer,
) -> None:
    ss = singlestep_replay_buffer
    ms = multistep_replay_buffer
    try:
        assert ss.capacity == ms.capacity, f"\n{ss.capacity=}\n!=\n{ms.capacity=}"
        assert (
            recorded_step_count == ms.num_stored
        ), f"\n{recorded_step_count=}\n!=\n{ms.num_stored=}"
    except AssertionError as e:
        raise AssertionError(f"Buffers sanity check failled with {e}")
    try:
        assert recorded_step_count == ss.num_stored, (
            f"\n{ss.num_stored=}\n!=\n{recorded_step_count=}\n==\n{ms.num_stored=}\n\n"
            f"\n{ss.capacity=}\n{ms.capacity=}\n"
        )
    except AssertionError as e:
        raise RuntimeWarning(f"Some sample from the original replay buffer where not copied. {e}")

    check_replay_buffer_is_all_finite(ms)

    return None


def _add_composed_data_to_multistep_replay_buffer(
    multistep_data_buffer_processor: MultistepDataBufferProcessorAbstract,
    multistep_replay_buffer: ReplayBuffer,
) -> None:
    multistep_replay_buffer.add(
        obs=multistep_data_buffer_processor.get_compose_obs(),
        action=multistep_data_buffer_processor.get_compose_act(),
        next_obs=multistep_data_buffer_processor.get_compose_next_obs(),
        reward=multistep_data_buffer_processor.get_reward_at_timestep_t(),
        terminated=multistep_data_buffer_processor.get_terminated_at_timestep_t(),
        truncated=multistep_data_buffer_processor.get_truncated_at_timestep_t(),
    )
    return None
