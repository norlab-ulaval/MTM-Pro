# coding=utf-8
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from mbrl.types import TransitionBatch
from mbrl.util import ReplayBuffer

# RLRP-775 action ``A20``: shared sentinel distinguishing "inherit the source
# buffer device" (the default of every buffer-DERIVATION helper) from an
# explicit ``device=None`` request (force the legacy host storage).
INHERIT_SOURCE_DEVICE: Any = "__inherit_source_device__"


def resolve_replay_buffer_source_device(
    replay_buffer: ReplayBuffer,
) -> Optional[torch.device]:
    """Return the storage device of ``replay_buffer`` (``None`` for host).

    RLRP-775 action ``A19``: every RLRC helper that derives a NEW buffer
    from an existing one must propagate this, otherwise a device-resident
    source (``cfg.mbrl_lib.keep_replay_buffer_on_device: true``, action
    ``A17``) silently degrades back to the legacy CPU path and the
    per-training-batch H→D copy comes back.
    """
    return getattr(replay_buffer, "device", None)


def transition_batch_to_host(batch: TransitionBatch) -> TransitionBatch:
    """Return ``batch`` with every tensor field moved to the host.

    RLRP-775 action ``A19``: the buffer-derivation helpers walk
    ``get_all()`` sample by sample in a python loop and branch on
    ``terminateds`` / ``truncateds``. With a device-resident source that
    is one device synchronization PER SAMPLE, so the loop input is
    copied to the host once up front instead.
    """
    return TransitionBatch(
        *(
            field.detach().cpu() if isinstance(field, torch.Tensor) else field
            for field in batch.astuple()
        )
    )


def resolve_target_replay_buffer_device(
    device: Optional[Union[torch.device, str]] = INHERIT_SOURCE_DEVICE,
    source_replay_buffer: Optional[ReplayBuffer] = None,
) -> Optional[torch.device]:
    """Resolve the storage device a DERIVED replay buffer must end up on.

    RLRP-775 action ``A20``: single entry point used by every helper that
    builds a new buffer, so the placement policy (action ``A17``
    ``cfg.mbrl_lib.keep_replay_buffer_on_device``) is applied consistently:

    - ``INHERIT_SOURCE_DEVICE`` (default) → the ``source_replay_buffer``
      device, or the host when no source buffer is provided;
    - ``None`` → force the legacy host storage;
    - anything else → that explicit device.
    """
    if device is INHERIT_SOURCE_DEVICE:
        if source_replay_buffer is None:
            return None
        return resolve_replay_buffer_source_device(source_replay_buffer)
    return torch.device(device) if device is not None else None


def stretch_replay_buffer_capacity(replay_buffer: ReplayBuffer, free_space: int) -> ReplayBuffer:
    """Spawn a new mbrl-lib ReplayBuffer object with same setting and content but with increassed
    capacity.

    :param replay_buffer: original mbrl-lib replay buffer
    :param free_space: free space to add to the replay buffer
    :return: a new increassed capacity replaybuffer object with the same content
    """
    new_replay_buffer = aggregate_replay_buffer((replay_buffer,), free_space=free_space)

    return new_replay_buffer


def aggregate_replay_buffer(
    replay_buffers: Sequence[ReplayBuffer], free_space: int = 0
) -> ReplayBuffer:
    """Aggregate trajectories from multiple mbrl-lib replay buffer in a single replay buffer.
    Perform check to validate that all replay buffers have the same obs/action shape,
    obs/action/reward dtype and collect the same maximum trajectory length or don't collect
    trajectory information at all.

    :param replay_buffers: a sequence of populated replay buffer
    :param free_space: (optional) add free space to the new replay buffer
    :return: a replay buffer with all trajectories from the replay buffers list
    """
    obs_shape: Tuple[int] | None = None
    action_shape: Tuple[int] | None = None
    obs_type: Type | None = None
    action_type: Type | None = None
    reward_type: Type | None = None
    stores_trajectories: bool | None = None
    max_trj_len: int | None = None
    next_max_trj_len: int | None = None

    # .... Validate that all replay buffer are compatible with each other .........................
    for each_buffer in replay_buffers:
        # Note:
        # - case stores_trajectories=False
        #       replay_buffer.obs.shape = (capacity, *obs.shape)
        # - case stores_trajectories=True
        #       replay_buffer.obs.shape = (capacity+max_trajectory_length, *obs.shape)
        next_obs_shape = each_buffer.obs_shape
        next_action_shape = each_buffer.action_shape
        next_obs_type = each_buffer.obs_type
        next_action_type = each_buffer.action_type
        next_reward_type = each_buffer.reward_type
        next_stores_trajectories = each_buffer.stores_trajectories
        if next_stores_trajectories:
            next_max_trj_len = each_buffer.max_trajectory_length

        if obs_shape:
            assert obs_shape == next_obs_shape, "All replay buffers must have the same obs.shape."
            assert (
                action_shape == next_action_shape
            ), "All replay buffers must have the same act.shape."
            assert obs_type == next_obs_type, "All replay buffers must have the same obs.dtype."
            assert (
                action_type == next_action_type
            ), "All replay buffers must have the same act.dtype."
            assert (
                reward_type == next_reward_type
            ), "All replay buffers must have the same reward.dtype."
            assert (
                stores_trajectories == next_stores_trajectories
            ), "Either all replay buffers store trajectory information or none of them do."
            if next_stores_trajectories:
                assert (
                    max_trj_len == next_max_trj_len
                ), "All replay buffers must store trajectory of same max len."

        obs_shape = next_obs_shape
        action_shape = next_action_shape
        obs_type = next_obs_type
        action_type = next_action_type
        reward_type = next_reward_type
        stores_trajectories = next_stores_trajectories
        if next_stores_trajectories:
            max_trj_len = next_max_trj_len

    # .... Aggregate samples from all buffers .....................................................
    total_capacity = 0
    for each_buffer in replay_buffers:
        total_capacity += each_buffer.capacity

    total_capacity += free_space

    aggregated_replay_buffer = ReplayBuffer(
        capacity=total_capacity,
        obs_shape=obs_shape,
        action_shape=action_shape,
        obs_type=obs_type,
        action_type=action_type,
        reward_type=reward_type,
        rng=replay_buffers[0].rng,
        max_trajectory_length=max_trj_len,
        # RLRP-775 action ``A19``: keep the aggregate on the same device as
        # its source. ``add_batch`` writes in one vectorized copy, so
        # allocating on-device up front is already optimal here.
        device=resolve_replay_buffer_source_device(replay_buffers[0]),
    )

    trj_indices_start_idx = 0
    for each_buffer in replay_buffers:
        (
            obss,
            actions,
            next_obss,
            rewards,
            terminateds,
            truncateds,
        ) = each_buffer.get_all().astuple()

        if stores_trajectories:
            # Translate source trajectory indices to the aggregated buffer offset and
            # append them manually.  We must NOT let ``add_batch`` perform its own
            # trajectory bookkeeping because it would create incorrect boundaries.
            saved_trj_indices = list(aggregated_replay_buffer.trajectory_indices)
            aggregated_replay_buffer.trajectory_indices = None  # disable bookkeeping

        aggregated_replay_buffer.add_batch(
            obss, actions, next_obss, rewards, terminateds, truncateds
        )

        if stores_trajectories:
            aggregated_replay_buffer.trajectory_indices = saved_trj_indices
            translated_trj_idx = np.array(each_buffer.trajectory_indices) + trj_indices_start_idx
            aggregated_replay_buffer.trajectory_indices += list(map(tuple, translated_trj_idx))
            trj_indices_start_idx = aggregated_replay_buffer.cur_idx

    return aggregated_replay_buffer


def was_replay_buffer_filled_over_capacity(replay_buffer: ReplayBuffer) -> None:
    """
    Checks if the trajectories samples in the replay buffer are stored contiguously
    from 0 to num_stored.

    :param replay_buffer: Instance of ReplayBuffer to be checked.
    :exception AssertionError: Thrown if the check fail
    :return: None
    """
    assert (replay_buffer.num_stored == replay_buffer.capacity and replay_buffer.cur_idx == 0) or (
        replay_buffer.num_stored < replay_buffer.capacity
        and replay_buffer.cur_idx < replay_buffer.capacity
    ), (
        f"Trajectories were added to replay_buffer after reaching max capacity, "
        f"meaning that trajectory samples are not stored contiguously "
        f"(ss.num_stored: {replay_buffer.num_stored}, ss.capacity: {replay_buffer.capacity}, "
        f"ss.cur_idx: {replay_buffer.cur_idx})."
    )
    return None


def get_replay_buffer_valide_indice(replay_buffer: ReplayBuffer) -> slice:
    if replay_buffer.stores_trajectories:
        assert len(replay_buffer.trajectory_indices) > 0

        valide_indices = slice(
                replay_buffer.trajectory_indices[0][0], replay_buffer.trajectory_indices[-1][1]
                )
    else:
        valide_indices = slice(0, replay_buffer.num_stored)
    return valide_indices


def check_replay_buffer_is_all_finite(replay_buffer: ReplayBuffer) -> None:
    # Use public API to check data finiteness
    try:
        all_data = replay_buffer.get_all(shuffle=False)
        obs_tensor = all_data.obs if isinstance(all_data.obs, torch.Tensor) else torch.as_tensor(all_data.obs)
        act_tensor = all_data.act if isinstance(all_data.act, torch.Tensor) else torch.as_tensor(all_data.act)
        next_obs_tensor = all_data.next_obs if isinstance(all_data.next_obs, torch.Tensor) else torch.as_tensor(all_data.next_obs)
        assert torch.all(
                torch.isfinite(obs_tensor)
                ), f"{torch.argwhere(~torch.isfinite(obs_tensor))= }"
        assert torch.all(
                torch.isfinite(act_tensor)
                ), f"{torch.argwhere(~torch.isfinite(act_tensor))=}"
        assert torch.all(
                torch.isfinite(next_obs_tensor)
                ), f"{torch.argwhere(~torch.isfinite(next_obs_tensor))=}"
    except AssertionError as e:
        raise AssertionError(f"Buffers has non finite value(s) {e}")
    return None
