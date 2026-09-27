# coding=utf-8
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import omegaconf
import torch
from mbrl.types import TransitionBatch
from mbrl.util import ReplayBuffer

from algorithm.policy.replay_buffer_exploration_policy import (
    BaseReplayBufferExplorationPolicy,
    RandomReplayBufferExplorationPolicy,
    )
from tools.console_tools.format import ConsoleFormat
from tools.console_tools.progressbar_tools import init_progressbar
from tools.mbrl_lib_tools.replaybuffer_tools import (
    INHERIT_SOURCE_DEVICE,
    check_replay_buffer_is_all_finite,
    resolve_target_replay_buffer_device,
    transition_batch_to_host,
    was_replay_buffer_filled_over_capacity,
    )


def explore_and_partition_replay_buffer(
    replay_buffer_exploration_policy: BaseReplayBufferExplorationPolicy,
    replay_buffer_size: int,
    replay_buffer: ReplayBuffer,
    buffer_max_trajectory_len: Optional[int] = None,
    show_progressbar: bool = True,
    device: Optional[Union[torch.device, str]] = INHERIT_SOURCE_DEVICE,
) -> Tuple[ReplayBuffer, List[int], BaseReplayBufferExplorationPolicy]:
    """Explore replay buffer and score sample using provided policy, then partition replay buffer
    base on score.
    Note: Assume that the replay buffer store trajectories contigously.

    :param replay_buffer_exploration_policy: an instanciated policy for exploring the replay buffer
    :param replay_buffer_size: the size of the output replay buffer
    :param replay_buffer: a source replay buffer with trajectories stored contigously
    :param buffer_max_trajectory_len: (optional) Store trajectory information, terminated,
    truncated
     and close trajectory at max len (required for using SequenceTransitionIterator).
    :param show_progressbar:
    :param device: storage device of the produced replay buffer. Defaults to the
     source ``replay_buffer`` device (RLRP-775 action ``A19``), so a
     device-resident source (``cfg.mbrl_lib.keep_replay_buffer_on_device``,
     action ``A17``) is not silently degraded back to the CPU path. Pass
     ``None`` to force the legacy host storage.
    :return: a new replay_buffer limited to the explorable region, sampled trj start idx log and
     the replay buffer exploration policy
    """
    sampled_trj_start_idx_log = []

    # .... Pre-condition ..........................................................................
    was_replay_buffer_filled_over_capacity(replay_buffer)
    assert isinstance(replay_buffer_size, int)
    assert replay_buffer_size <= replay_buffer.num_stored, (
        f"{replay_buffer_size=} !<= " f"{replay_buffer.num_stored=}"
    )
    assert replay_buffer_size > 0

    # RLRP-775 action ``A19``: the buffer below is filled ROW BY ROW in a
    # python loop, so it is always built on the host (one tiny synchronous
    # H→D copy per ``add`` would be far more expensive than the single bulk
    # relocation performed at the end of this function).
    target_device = resolve_target_replay_buffer_device(device, replay_buffer)

    # .... Construct replay buffer restricted to explorable region ................................
    sub_replay_buffer = ReplayBuffer(
        capacity=replay_buffer_size,
        obs_shape=replay_buffer.obs_shape,
        action_shape=replay_buffer.action_shape,
        obs_type=replay_buffer.obs_type,
        action_type=replay_buffer.action_type,
        reward_type=replay_buffer.reward_type,
        max_trajectory_length=buffer_max_trajectory_len,
    )

    if show_progressbar:
        progressbar = init_progressbar(
            sub_replay_buffer.capacity,
            "Explore full replay buffer and construct optimized replay buffer subset",
        )

    # (CRITICAL) ToDo: implement batching logic (ref task RLRP-538)
    all_samples: TransitionBatch
    # RLRP-775 action ``A19``: host copy so the per-sample ``terminateds`` /
    # ``truncateds`` python branching below does not synchronize the device
    # on every iteration when the source buffer is device-resident.
    all_samples = transition_batch_to_host(replay_buffer.get_all(shuffle=False))
    while (
        sub_replay_buffer.num_stored < sub_replay_buffer.capacity
        and replay_buffer_exploration_policy.available_trj_start_idx_len > 0
    ):
        trajectory_start_idx, trajectory_end_idx = replay_buffer_exploration_policy.act(
            obs_target_replay_buffer=sub_replay_buffer
        )

        for idx in range(trajectory_start_idx, trajectory_end_idx):
            if sub_replay_buffer.num_stored == sub_replay_buffer.capacity:
                if sub_replay_buffer.stores_trajectories:
                    sub_replay_buffer.close_trajectory()
                break

            each_sample = all_samples[idx]

            truncated = each_sample.truncateds
            terminated = each_sample.terminateds
            if (
                sub_replay_buffer.num_stored + 1 == sub_replay_buffer.capacity
                or idx + 1 == trajectory_end_idx
            ) and not terminated:
                truncated = True

            sub_replay_buffer.add(
                obs=each_sample.obs,
                action=each_sample.act,
                next_obs=each_sample.next_obs,
                reward=each_sample.rewards,
                terminated=terminated,
                truncated=truncated,
            )
            if show_progressbar:
                progressbar.update(1)

            done = terminated or truncated
            if done:
                sampled_trj_start_idx_log.append(trajectory_start_idx)
                break

    if (
        sub_replay_buffer.stores_trajectories
        and len(sub_replay_buffer.trajectory_indices) == 0
        and buffer_max_trajectory_len is not None
        and buffer_max_trajectory_len >= sub_replay_buffer.num_stored
    ):
        sub_replay_buffer.close_trajectory()

    if show_progressbar:
        progressbar.close()

    # .... Sanity check ...........................................................................
    check_replay_buffer_is_all_finite(sub_replay_buffer)

    # RLRP-775 action ``A19``: single bulk relocation of the finished buffer.
    if target_device is not None:
        sub_replay_buffer.to(target_device)

    return sub_replay_buffer, sampled_trj_start_idx_log, replay_buffer_exploration_policy


def _take_rows(field, row_idx: np.ndarray):
    """Gather ``row_idx`` along the first axis of a ``TransitionBatch`` field (torch or numpy)."""
    if isinstance(field, torch.Tensor):
        return field[torch.as_tensor(row_idx, dtype=torch.long, device=field.device)]
    return np.asarray(field)[row_idx]


def partition_replay_buffer_by_rows(
    replay_buffer: ReplayBuffer,
    row_idx: Sequence[int],
    device: Optional[Union[torch.device, str]] = INHERIT_SOURCE_DEVICE,
) -> ReplayBuffer:
    """Vectorised (O(N)) sample-level partition: a new ``ReplayBuffer`` holding the rows ``row_idx``
    of ``replay_buffer``, in that order, materialised with ONE ``get_all`` + fancy index +
    ``add_batch`` instead of one python-level ``add`` per row.

    Reproduces the flags the legacy per-row loop of :func:`explore_and_partition_replay_buffer`
    writes when its scan window is a single sample (``buffer_max_trajectory_len is None``): every
    row is its own trajectory, so ``truncated`` is forced ``True`` on every row that is not
    ``terminated``. The produced buffer does not store trajectory information
    (``max_trajectory_length=None``), exactly like the legacy one in that regime.

    :param replay_buffer: the source replay buffer.
    :param row_idx: the source row indexes to keep (no duplicate).
    :param device: storage device of the produced buffer, see :func:`split_replay_buffer`.
    :return: the partition replay buffer (``num_stored == len(row_idx)``).
    """
    row_idx = np.asarray(row_idx, dtype=np.int64).reshape(-1)

    # .... Pre-condition ..........................................................................
    was_replay_buffer_filled_over_capacity(replay_buffer)
    assert row_idx.size > 0, "Empty partition requested"
    assert row_idx.min() >= 0 and row_idx.max() < replay_buffer.num_stored, (
        f"Row index out of range: [{row_idx.min()}, {row_idx.max()}] !< "
        f"{replay_buffer.num_stored=}"
    )

    target_device = resolve_target_replay_buffer_device(device, replay_buffer)

    sub_replay_buffer = ReplayBuffer(
        capacity=int(row_idx.size),
        obs_shape=replay_buffer.obs_shape,
        action_shape=replay_buffer.action_shape,
        obs_type=replay_buffer.obs_type,
        action_type=replay_buffer.action_type,
        reward_type=replay_buffer.reward_type,
        max_trajectory_length=None,
    )

    # Host copy (like the legacy loop) so the gather + the bool logic below never synchronize a
    # device-resident source; one bulk relocation at the end.
    all_samples = transition_batch_to_host(replay_buffer.get_all(shuffle=False))
    obs, act, next_obs, rewards, terminateds, truncateds = all_samples.astuple()

    terminateds = _take_rows(terminateds, row_idx)
    truncateds = _take_rows(truncateds, row_idx)
    if isinstance(terminateds, torch.Tensor):
        terminateds = terminateds.bool()
        truncateds = truncateds.bool() | ~terminateds
    else:
        terminateds = np.asarray(terminateds, dtype=bool)
        truncateds = np.asarray(truncateds, dtype=bool) | ~terminateds

    sub_replay_buffer.add_batch(
        obs=_take_rows(obs, row_idx),
        action=_take_rows(act, row_idx),
        next_obs=_take_rows(next_obs, row_idx),
        reward=_take_rows(rewards, row_idx),
        terminated=terminateds,
        truncated=truncateds,
    )
    assert sub_replay_buffer.num_stored == row_idx.size

    # .... Sanity check ...........................................................................
    check_replay_buffer_is_all_finite(sub_replay_buffer)

    if target_device is not None:
        sub_replay_buffer.to(target_device)

    return sub_replay_buffer


def split_replay_buffer(
    replay_buffer: ReplayBuffer,
    datasets_size: Optional[int] = None,
    val_ratio: float = 0.0,
    buffer_max_trajectory_len: Optional[int] = None,
    device: Optional[Union[torch.device, str]] = INHERIT_SOURCE_DEVICE,
    seed: Optional[int] = None,
) -> Tuple[ReplayBuffer, ReplayBuffer]:
    """
    Splits a replay buffer into training and validation buffers based on the specified
    configuration settings, replay buffer size, and validation ratio.

    This function determines the sizes of the training and validation buffers based on the overall
    replay buffer size and the validation ratio provided in the configuration.

    Two implementations, same protocol (sample-level uniform split without replacement,
    ``val_size = int(datasets_size * val_ratio)``, disjoint train / val, ``datasets_size``
    sub-sampling of the source rows):

    - ``buffer_max_trajectory_len is None`` (every ICRA2026 multistep cfg): ONE
      ``rng.permutation(num_stored)`` and two vectorised gathers
      (:func:`partition_replay_buffer_by_rows`). O(N), seconds on the 451 k-row NeuroBEM buffer.
      The legacy per-row loop was O(N^2) (a full-buffer mask + fancy index per row) and took about
      an hour on that buffer, once per trial (ref task RLRP-538).
    - ``buffer_max_trajectory_len`` set: the legacy trajectory-preserving loop over
      `RandomReplayBufferExplorationPolicy` + :func:`explore_and_partition_replay_buffer` (scan
      window of ``buffer_max_trajectory_len`` contiguous rows, trajectory bookkeeping for
      ``SequenceTransitionIterator``).

    The realisation of the split differs between the two implementations for a given seed (same
    distribution); the legacy loop was never seeded by its callers anyway.

    :param replay_buffer: The initial replay buffer containing the original dataset to be split.
    :param datasets_size:
    :param val_ratio:
    :param buffer_max_trajectory_len: Store trajectory information for the train set.
    :param device: storage device of the produced train/val buffers. Defaults to
     the source ``replay_buffer`` device (RLRP-775 action ``A19``) so that a
     device-resident source buffer stays device-resident for training. Pass
     ``None`` to force the legacy host storage.
    :param seed: seed of the split permutation (``numpy.random.default_rng(seed)``); ``None``
     (default) draws a fresh split each call, like the legacy implementation.
    :return: A tuple containing the training and validation replay buffers.
    """
    # .... Pre-condition ..........................................................................
    if datasets_size:
        assert (
            datasets_size <= replay_buffer.num_stored
        ), f"Not enough data in the {replay_buffer.num_stored=} for the requested {datasets_size=}"
    else:
        datasets_size = replay_buffer.num_stored

    assert val_ratio >= 0.0

    progressbar = init_progressbar(datasets_size, "Split replay buffer")

    # .... Setup ..................................................................................
    val_size = int(datasets_size * val_ratio)
    train_size = datasets_size - val_size

    if buffer_max_trajectory_len is None:
        # .... Vectorised sample-level split (no trajectory bookkeeping) ..........................
        assert train_size > 0, f"{train_size=} !> 0 ({datasets_size=}, {val_ratio=})"
        assert val_size > 0, (
            f"{val_size=} !> 0 ({datasets_size=}, {val_ratio=}): the ERLL requires a non-empty "
            f"validation set"
        )
        perm = np.random.default_rng(seed).permutation(replay_buffer.num_stored)
        train_replay_buffer = partition_replay_buffer_by_rows(
            replay_buffer, perm[:train_size], device=device
        )
        progressbar.update(train_size)
        val_replay_buffer = partition_replay_buffer_by_rows(
            replay_buffer, perm[train_size : train_size + val_size], device=device
        )
        progressbar.update(val_size)

        assert train_replay_buffer.num_stored == train_size
        assert val_replay_buffer.num_stored == val_size
        progressbar.close()
        return train_replay_buffer, val_replay_buffer

    if buffer_max_trajectory_len is not None:
        warning_str = ""
        if train_size % buffer_max_trajectory_len != 0:
            warning_str += (
                f"\n  - parameter {ConsoleFormat.MSG_DIMMED_FORMAT}{buffer_max_trajectory_len=}"
                f"{ConsoleFormat.MSG_END_FORMAT} is not an even number of the target "
                f"{ConsoleFormat.MSG_DIMMED_FORMAT}{train_size=}{ConsoleFormat.MSG_END_FORMAT} "
                # f"i.e {train_size=} % {buffer_max_trajectory_len=} != 0. "
            )

        if val_size % buffer_max_trajectory_len != 0:
            warning_str += (
                f"\n  - parameter {ConsoleFormat.MSG_DIMMED_FORMAT}{buffer_max_trajectory_len=}"
                f"{ConsoleFormat.MSG_END_FORMAT} is not an even number of the target "
                f"{ConsoleFormat.MSG_DIMMED_FORMAT}{val_size=}{ConsoleFormat.MSG_END_FORMAT} "
                # f"i.e {val_size=} % {buffer_max_trajectory_len=} != 0. "
            )

        if len(warning_str) > 0:
            progressbar.write(
                f"[split_replay_buffer] "
                f"{ConsoleFormat.MSG_WARNING_FORMAT}buffer_max_trajectory_len "
                f"cfg warning:{ConsoleFormat.MSG_END_FORMAT}"
                f"{warning_str}\nUnexpected beaviour can occur."
            )

    cfg_random_policy = omegaconf.OmegaConf.create(
        f"""
        UDER:
            uder_exploration_policy:
                scan_window_len: {buffer_max_trajectory_len if isinstance(buffer_max_trajectory_len, int) else 1}
                scan_window_registered_len: {buffer_max_trajectory_len if isinstance(buffer_max_trajectory_len, int) else 1}
        environment: null
        seed: {seed if seed is not None else 'null'}
        """
    )
    replay_buffer_exploration_policy = RandomReplayBufferExplorationPolicy(
        cfg_random_policy, replay_buffer
    )

    # .... Build the training replay buffer .......................................................
    train_replay_buffer, _, replay_buffer_exploration_policy = explore_and_partition_replay_buffer(
        replay_buffer_exploration_policy,
        replay_buffer_size=train_size,
        replay_buffer=replay_buffer,
        show_progressbar=True,
        buffer_max_trajectory_len=buffer_max_trajectory_len,
        device=device,
    )
    progressbar.update(train_size)

    # .... Build the validation replay buffer .....................................................
    val_replay_buffer, _, replay_buffer_exploration_policy = explore_and_partition_replay_buffer(
        replay_buffer_exploration_policy,
        replay_buffer_size=val_size,
        replay_buffer=replay_buffer,
        show_progressbar=True,
        buffer_max_trajectory_len=buffer_max_trajectory_len,
        device=device,
    )
    progressbar.update(val_size)

    # .... Sanity check ...........................................................................
    assert not was_replay_buffer_filled_over_capacity(train_replay_buffer)
    assert not was_replay_buffer_filled_over_capacity(val_replay_buffer)

    warning_str = ""
    if train_replay_buffer.num_stored != train_size:
        warning_str += (
            f"\n  - the {ConsoleFormat.MSG_DIMMED_FORMAT}training"
            f"{ConsoleFormat.MSG_END_FORMAT} "
            f"replay buffer store fewer samples than expected i.e. "
            f"{train_replay_buffer.num_stored=} != {train_size=}."
        )
    if val_replay_buffer.num_stored != val_size:
        warning_str += (
            f"\n  - the {ConsoleFormat.MSG_DIMMED_FORMAT}validation"
            f"{ConsoleFormat.MSG_END_FORMAT} "
            f"replay buffer store fewer samples than expected i.e. "
            f"{val_replay_buffer.num_stored=} != {val_size=}."
        )

    if len(warning_str) > 0:
        progressbar.write(
            f"[split_replay_buffer] {ConsoleFormat.MSG_WARNING_FORMAT}replay buffer size "
            f"warning."
            f"{ConsoleFormat.MSG_END_FORMAT} Be advise, "
            f"{warning_str}\nUnexpected beaviour can occur."
        )

    progressbar.close()
    return train_replay_buffer, val_replay_buffer
