# coding=utf-8
"""Lazy multistep window dataset (RLRP-824, KD3 / FR3 / FR10).

Permanent module. Introduced by Step 3 of the Ultra-long-horizon MS->MS training via a lazy
window DataLoader ``.junie`` plan (``ultra_long_horizon_dataloader_plan_RLRP-824_20260915.md``).

:class:`MultistepWindowDataset` is a map-style ``torch.utils.data.Dataset`` over the anchors of a
:class:`~tools.multistep_tools.window_dataset.single_step_trajectory_store.SingleStepTrajectoryStore`
that yields ``mbrl.types.TransitionBatch`` rows in the EXACT legacy multistep replay-buffer layout
(``MultistepDataBufferProcessorArbitraryDimension.get_compose_obs / get_compose_act /
get_compose_next_obs``), generalized to the asymmetric ``W = max(H, F)`` output window::

    obs      = [o_{t-H+1..t} (H*Do) | a_{t-H+1..t-1} ((H-1)*Da)]       # get_compose_obs
    act      = a_t                                                   # get_compose_act
    next_obs = [o_{t+F-W+1..t+F} (W*Do) | a_{t+F-W+1..t+F-1} ((W-1)*Da)] # get_compose_next_obs

so ``OneDTransitionRewardModelV2._process_batch`` consumes it unchanged. Composed-observation
compliance (FR10): every flat row is produced by the canonical
:func:`~tools.multistep_tools.multistep_model_util.revert_timestep_first_multistep_dim_unflaten_array`
on gathered ``(B, MS, Do + Da)`` per-step blocks -- never a hand-written ``reshape``.

The vectorized ``__getitems__`` (one gather per batch) is what a ``DataLoader`` built with
``batch_sampler=`` + the identity :func:`identity_collate` calls, so a batch costs one advanced
index into the flat store regardless of ``F``.

The dataset follows the residency of its store / anchors (``dataset.device``): with a
device-resident :class:`SingleStepTrajectoryStore` the sampler's CPU ids are moved once per batch
and the composed ``TransitionBatch`` is produced directly on that device (no host gather, no H2D
copy of the composed rows). Rows are bit-identical to the CPU path.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from mbrl.types import TransitionBatch
from torch.utils.data import Dataset

from tools.multistep_tools.multistep_model_util import (
    revert_timestep_first_multistep_dim_unflaten_array,
)
from tools.multistep_tools.utils import (
    compute_multistep_model_in_size,
    compute_multistep_model_out_size,
)
from tools.multistep_tools.window_dataset.single_step_trajectory_store import (
    AnchorIndex,
    SingleStepTrajectoryStore,
)


def identity_collate(batch):
    """``collate_fn`` for a ``DataLoader`` whose dataset already returns a ready
    ``TransitionBatch`` per index list (``__getitems__``)."""
    return batch


class MultistepWindowDataset(Dataset):
    """Map-style view over the anchors of a single-step store composing ``(H, F)`` windows lazily.

    :param store: the single-step trajectory store.
    :param history_len: ``H``, the composed input window length.
    :param horizon_len: ``F``, the forecast depth.
    :param anchors: the anchor index (``store.anchor_index(H, F, boundary_mode)``); when
        ``anchor_ids`` is given the dataset is the SUBSET of those anchors (train / val split).
    :param anchor_ids: optional flat anchor ids selecting a subset (``split_anchors`` output).
    :param output_window_len: ``W`` (default ``max(H, F)``, the MS->MS forecast family rule;
        pass the model's ``output_window_len`` when it differs).
    """

    def __init__(
        self,
        store: SingleStepTrajectoryStore,
        history_len: int,
        horizon_len: int,
        anchors: Optional[AnchorIndex] = None,
        anchor_ids: Optional[Union[torch.Tensor, Sequence[int]]] = None,
        output_window_len: Optional[int] = None,
        boundary_mode: str = "pad",
    ):
        self.store = store
        self.history_len = int(history_len)
        self.horizon_len = int(horizon_len)
        self.output_window_len = (
            max(self.history_len, self.horizon_len) if output_window_len is None else int(output_window_len)
        )
        if self.output_window_len < self.horizon_len:
            raise ValueError(
                f"output_window_len {self.output_window_len} must be >= horizon_len {self.horizon_len}"
            )
        self.anchors = (
            store.anchor_index(self.history_len, self.horizon_len, boundary_mode) if anchors is None else anchors
        )
        if self.anchors.history_len != self.history_len or self.anchors.horizon_len != self.horizon_len:
            raise ValueError(
                f"anchor index built for (H, F) = ({self.anchors.history_len}, {self.anchors.horizon_len}) "
                f"but the dataset asks for ({self.history_len}, {self.horizon_len})"
            )
        if self.anchors.device != store.device:
            # Anchors always follow the store (one residency per dataset): a CPU anchor index handed
            # to a device-resident store (or the reverse) is moved rather than rejected.
            self.anchors = self.anchors.to(store.device)
        if anchor_ids is None:
            self._ids = self.anchors.ids
        else:
            ids = torch.as_tensor(anchor_ids, dtype=torch.int64).reshape(-1)
            if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= len(self.anchors)):
                raise IndexError(f"anchor_ids out of range [0, {len(self.anchors)})")
            self._ids = ids.to(self.device)
        if self._ids.numel() == 0:
            raise ValueError("MultistepWindowDataset: empty anchor selection")

    # ==== Sizes ==================================================================================
    @property
    def obs_dim(self) -> int:
        return self.store.obs_dim

    @property
    def act_dim(self) -> int:
        return self.store.act_dim

    @property
    def in_size(self) -> int:
        """Model input width ``Do*H + Da*H`` (``obs`` + ``act`` columns)."""
        return compute_multistep_model_in_size(self.obs_dim, self.act_dim, self.history_len)

    @property
    def out_size(self) -> int:
        """Model output / ``next_obs`` width ``Do*W + Da*(W-1)``."""
        return compute_multistep_model_out_size(self.obs_dim, self.act_dim, self.output_window_len)

    @property
    def obs_width(self) -> int:
        """Width of the ``obs`` column of a row: ``Do*H + Da*(H-1)`` (legacy ``get_compose_obs``)."""
        return self.in_size - self.act_dim

    @property
    def anchor_ids(self) -> torch.Tensor:
        """The flat anchor ids this dataset iterates over (subset of ``anchors.ids``)."""
        return self._ids

    @property
    def device(self) -> torch.device:
        """Residency of the store / anchors, i.e. the device the composed batches are produced on."""
        return self.store.device

    def __len__(self) -> int:
        return int(self._ids.numel())

    # ==== Gather =================================================================================
    def gather(self, anchor_ids: torch.Tensor) -> TransitionBatch:
        """Compose the rows of FLAT anchor ids (indices into ``self.anchors``), vectorized.
        ``anchor_ids`` may live on any device; they are moved to :attr:`device` once."""
        anchor_ids = torch.as_tensor(anchor_ids, dtype=torch.int64, device=self.device).reshape(-1)
        traj_ids = self.anchors.traj_ids[anchor_ids]
        ts = self.anchors.ts[anchor_ids]
        hist_obs, hist_act, out_obs, out_act, reward, terminated, truncated = self.store.gather_window_steps(
            traj_ids, ts, self.history_len, self.horizon_len, self.output_window_len
        )
        Do, Da = self.obs_dim, self.act_dim
        # obs column: H obs blocks + (H-1) act blocks -> drop the trailing a_t (carried by ``act``).
        obs = revert_timestep_first_multistep_dim_unflaten_array(
            torch.cat([hist_obs, hist_act], dim=-1), Do, Da, remove_last_action_padding=True
        )
        act = hist_act[:, -1, :]
        # next_obs column: W obs blocks + (W-1) act blocks -> drop the a_{t+F} slot.
        next_obs = revert_timestep_first_multistep_dim_unflaten_array(
            torch.cat([out_obs, out_act], dim=-1), Do, Da, remove_last_action_padding=True
        )
        return TransitionBatch(
            obs=obs,
            act=act,
            next_obs=next_obs,
            rewards=reward,
            terminateds=terminated,
            truncateds=truncated,
        )

    def __getitem__(self, index: int) -> TransitionBatch:
        return self.gather(self._ids[index : index + 1])

    def __getitems__(self, indices: Sequence[int]) -> TransitionBatch:
        """Vectorized batch fetch used by ``DataLoader`` (auto-collation + ``batch_sampler``)."""
        idx = torch.as_tensor(indices, dtype=torch.int64, device=self.device).reshape(-1)
        return self.gather(self._ids[idx])

    def all_rows(self) -> TransitionBatch:
        """Every row of the dataset in anchor order (materializes ``len(self)`` windows)."""
        return self.gather(self._ids)

    def subset(self, anchor_ids: Union[torch.Tensor, Sequence[int]]) -> "MultistepWindowDataset":
        """A dataset over a SUBSET of this dataset's anchor index (train / val split)."""
        return MultistepWindowDataset(
            self.store,
            self.history_len,
            self.horizon_len,
            anchors=self.anchors,
            anchor_ids=anchor_ids,
            output_window_len=self.output_window_len,
        )

    def __repr__(self) -> str:
        return (
            f"MultistepWindowDataset(n={len(self)}/{len(self.anchors)} anchors, H={self.history_len}, "
            f"F={self.horizon_len}, W={self.output_window_len}, Do={self.obs_dim}, Da={self.act_dim}, "
            f"boundary_mode={self.anchors.boundary_mode!r}, in_size={self.in_size}, out_size={self.out_size}, "
            f"device={self.device})"
        )
