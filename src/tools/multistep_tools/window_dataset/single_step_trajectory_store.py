# coding=utf-8
"""Single-step trajectory store for the lazy multistep window DataLoader path (RLRP-824, KD2).

Permanent module. Introduced by Step 3 of the Ultra-long-horizon MS->MS training via a lazy
window DataLoader ``.junie`` plan (``ultra_long_horizon_dataloader_plan_RLRP-824_20260915.md``).

A :class:`SingleStepTrajectoryStore` keeps every trajectory ONCE as single-step tensors
(``obs`` ``(T_i + 1, Do)``, ``act`` ``(T_i, Da)``, ``reward`` / ``terminated`` / ``truncated``
``(T_i,)``), so host memory is ``O(sum_i T_i * (Do + Da))`` -- independent of the forecast depth
``F`` -- and the composed ``(H, F)`` windows are gathered lazily per batch by
:class:`~tools.multistep_tools.window_dataset.multistep_window_dataset.MultistepWindowDataset`.

Anchor protocol (legacy parity with ``convert_singlestep_replaybuffer_to_multistep`` +
``MultistepDataBufferProcessorArbitraryDimension``): one anchor per single-step sample
``t in [0, T_i)`` of every trajectory, in sample order, so that **anchor id == legacy multistep
replay-buffer row id**. Windows that cross a trajectory boundary are PADDED exactly as the legacy
composer pads its ``MultistepBuffer``s (``boundary_mode="pad"``, default): observations before the
start repeat ``o_0`` and after the end repeat ``o_T``; actions before the start are ZERO (the
legacy ``reset_buffers(obs)`` path of the arbitrary-dimension processor) and after the end repeat
``a_{T-1}``. ``boundary_mode="exclude"`` keeps only the anchors whose ``[t-H+1, t+F]`` window is
fully inside the trajectory (leak-free windows; smaller dataset -- documented protocol change).

Device residency (RLRP-824 follow-up, Valeria A100 host-bound profile of 2026-09-17): the flat
storage and the anchors can live on the TRAINING device (``device=`` / :meth:`~SingleStepTrajectoryStore.to`).
The single-step store is tiny (tens of MB) while the composed batch is large
(``B x (H + W) x (Do + Da)``), so a device-resident store turns the per-batch host gather +
unflatten + synchronous H2D copy into a few CUDA index kernels producing the batch in place. The
gather is pure integer indexing, so the composed rows are bit-identical on every device.
"""
from __future__ import annotations

import copy
import os
import warnings
from dataclasses import dataclass, replace as dataclass_replace
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from mbrl.util.replay_buffer import ReplayBuffer

BOUNDARY_MODES: Tuple[str, ...] = ("pad", "exclude")

_HDF5_FORMAT_VERSION = 1


@dataclass(frozen=True)
class AnchorIndex:
    """The ``(traj_id, t)`` anchors of a store for a given ``(H, F)`` window.

    ``traj_ids`` / ``ts`` are aligned ``(N,)`` int64 tensors; ``ids`` are the flat anchor ids
    (``0 .. N-1``) used by the dataset / split record. For ``boundary_mode="pad"`` the flat id is
    the legacy single-step sample id (``sum_{j<traj} T_j + t``).
    """

    traj_ids: torch.Tensor
    ts: torch.Tensor
    history_len: int
    horizon_len: int
    boundary_mode: str

    def __len__(self) -> int:
        return int(self.traj_ids.numel())

    @property
    def device(self) -> torch.device:
        return self.traj_ids.device

    @property
    def ids(self) -> torch.Tensor:
        return torch.arange(len(self), dtype=torch.int64, device=self.device)

    def pairs(self) -> List[Tuple[int, int]]:
        return list(zip(self.traj_ids.tolist(), self.ts.tolist()))

    def to(self, device: Union[str, torch.device]) -> "AnchorIndex":
        """The same anchors with ``traj_ids`` / ``ts`` on ``device`` (``self`` when already there)."""
        device = torch.device(device)
        if self.traj_ids.device == device and self.ts.device == device:
            return self
        return dataclass_replace(self, traj_ids=self.traj_ids.to(device), ts=self.ts.to(device))


class SingleStepTrajectoryStore:
    """Per-trajectory single-step tensors kept once (``O(T)`` memory) with a vectorized window
    gather over concatenated flat storage.

    :param obs: per-trajectory observation sequences ``(T_i + 1, Do)`` (``o_0 .. o_{T_i}``).
    :param act: per-trajectory action sequences ``(T_i, Da)`` (``a_0 .. a_{T_i - 1}``).
    :param reward: per-trajectory rewards ``(T_i,)``; ``None`` -> zeros.
    :param terminated: per-trajectory terminated flags ``(T_i,)``; ``None`` -> all ``False`` except
        the last step of every trajectory, which is marked terminated so the legacy ``done`` scan
        recovers the boundaries.
    :param truncated: per-trajectory truncated flags ``(T_i,)``; ``None`` -> all ``False``.
    :param dtype: storage dtype of ``obs`` / ``act`` / ``reward`` (the MODEL dtype, plan FR
        "dtype hygiene").
    :param device: residency of the flat storage (``None`` -> CPU). Inputs are always assembled
        on the host first (validation, HDF5 / replay-buffer sources are host data) and moved once.
    """

    _TENSOR_ATTRS: Tuple[str, ...] = (
        "traj_lengths",
        "obs_offsets",
        "act_offsets",
        "obs_all",
        "act_all",
        "reward_all",
        "terminated_all",
        "truncated_all",
    )

    def __init__(
        self,
        obs: Sequence[Union[np.ndarray, torch.Tensor]],
        act: Sequence[Union[np.ndarray, torch.Tensor]],
        reward: Optional[Sequence[Union[np.ndarray, torch.Tensor]]] = None,
        terminated: Optional[Sequence[Union[np.ndarray, torch.Tensor]]] = None,
        truncated: Optional[Sequence[Union[np.ndarray, torch.Tensor]]] = None,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device, None] = None,
    ):
        if len(obs) == 0:
            raise ValueError("SingleStepTrajectoryStore needs at least one trajectory")
        if len(obs) != len(act):
            raise ValueError(f"obs has {len(obs)} trajectories but act has {len(act)}")
        self.dtype = dtype

        obs_t = [torch.as_tensor(np.asarray(o) if not isinstance(o, torch.Tensor) else o).detach().cpu().to(dtype) for o in obs]
        act_t = [torch.as_tensor(np.asarray(a) if not isinstance(a, torch.Tensor) else a).detach().cpu().to(dtype) for a in act]
        for k, (o, a) in enumerate(zip(obs_t, act_t)):
            if o.ndim != 2 or a.ndim != 2:
                raise ValueError(f"trajectory {k}: obs/act must be 2-D, got {tuple(o.shape)} / {tuple(a.shape)}")
            if o.shape[0] != a.shape[0] + 1:
                raise ValueError(
                    f"trajectory {k}: expected obs of length T+1 = {a.shape[0] + 1} for {a.shape[0]} "
                    f"actions, got {o.shape[0]}"
                )
            if a.shape[0] < 1:
                raise ValueError(f"trajectory {k}: needs at least one transition")
        self.obs_dim = int(obs_t[0].shape[1])
        self.act_dim = int(act_t[0].shape[1])
        if any(o.shape[1] != self.obs_dim for o in obs_t) or any(a.shape[1] != self.act_dim for a in act_t):
            raise ValueError("every trajectory must share the same obs_dim / act_dim")

        lengths = [int(a.shape[0]) for a in act_t]

        def _flags(seq, default_last: bool) -> List[torch.Tensor]:
            if seq is None:
                out = []
                for T in lengths:
                    f = torch.zeros(T, dtype=torch.bool)
                    if default_last:
                        f[-1] = True
                    out.append(f)
                return out
            out = [torch.as_tensor(np.asarray(s) if not isinstance(s, torch.Tensor) else s).detach().cpu().reshape(-1).to(torch.bool) for s in seq]
            for k, (f, T) in enumerate(zip(out, lengths)):
                if f.numel() != T:
                    raise ValueError(f"trajectory {k}: flag length {f.numel()} != T {T}")
            return out

        if reward is None:
            rew_t = [torch.zeros(T, dtype=dtype) for T in lengths]
        else:
            rew_t = [torch.as_tensor(np.asarray(r) if not isinstance(r, torch.Tensor) else r).detach().cpu().reshape(-1).to(dtype) for r in reward]
            for k, (r, T) in enumerate(zip(rew_t, lengths)):
                if r.numel() != T:
                    raise ValueError(f"trajectory {k}: reward length {r.numel()} != T {T}")
        term_t = _flags(terminated, default_last=True)
        trunc_t = _flags(truncated, default_last=False)

        # .... Flat storage + offsets (vectorized gather) .........................................
        self.traj_lengths = torch.tensor(lengths, dtype=torch.int64)  # T_i
        self.num_trajectories = len(lengths)
        # obs flat: trajectory k occupies [obs_offsets[k], obs_offsets[k] + T_k + 1)
        self.obs_offsets = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(self.traj_lengths + 1, 0)[:-1]])
        # act / reward / flags flat: trajectory k occupies [act_offsets[k], act_offsets[k] + T_k)
        self.act_offsets = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(self.traj_lengths, 0)[:-1]])
        self.obs_all = torch.cat(obs_t, dim=0).contiguous()
        self.act_all = torch.cat(act_t, dim=0).contiguous()
        self.reward_all = torch.cat(rew_t, dim=0).contiguous()
        self.terminated_all = torch.cat(term_t, dim=0).contiguous()
        self.truncated_all = torch.cat(trunc_t, dim=0).contiguous()
        self.num_samples = int(self.traj_lengths.sum())
        # Host copies of the lengths / offsets: the anchor enumeration and the HDF5 writer read them
        # per trajectory and must not pay one device sync per ``int(...)`` on a device-resident store.
        self._traj_lengths_list: List[int] = list(lengths)
        self._obs_offsets_list: List[int] = self.obs_offsets.tolist()
        self._act_offsets_list: List[int] = self.act_offsets.tolist()
        if device is not None:
            self._move_tensors_(torch.device(device))

    # ==== Device residency =======================================================================
    @property
    def device(self) -> torch.device:
        return self.obs_all.device

    def _move_tensors_(self, device: torch.device) -> None:
        for name in self._TENSOR_ATTRS:
            setattr(self, name, getattr(self, name).to(device))

    def to(self, device: Union[str, torch.device]) -> "SingleStepTrajectoryStore":
        """A store whose flat storage lives on ``device`` (``self`` when already there; otherwise a
        shallow copy sharing nothing but the Python metadata -- the source store is untouched)."""
        device = torch.device(device)
        if self.device == device:
            return self
        other = copy.copy(self)
        other._move_tensors_(device)
        return other

    def cpu(self) -> "SingleStepTrajectoryStore":
        return self.to("cpu")

    # ==== Introspection ==========================================================================
    def __len__(self) -> int:
        return self.num_samples

    def __repr__(self) -> str:
        return (
            f"SingleStepTrajectoryStore(num_trajectories={self.num_trajectories}, "
            f"num_samples={self.num_samples}, obs_dim={self.obs_dim}, act_dim={self.act_dim}, "
            f"dtype={self.dtype}, device={self.device}, nbytes={self.nbytes / 2**20:.2f} MB)"
        )

    @property
    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.obs_all, self.act_all, self.reward_all, self.terminated_all, self.truncated_all)
        )

    def trajectory(self, traj_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(obs (T+1, Do), act (T, Da))`` of one trajectory (views on the flat storage)."""
        T = self._traj_lengths_list[traj_id]
        o0, a0 = self._offsets_of(traj_id)
        return self.obs_all[o0 : o0 + T + 1], self.act_all[a0 : a0 + T]

    def _offsets_of(self, traj_id: int) -> Tuple[int, int]:
        """``(obs_offset, act_offset)`` of one trajectory from the host copies (no device sync)."""
        traj_id = int(traj_id)
        if not 0 <= traj_id < self.num_trajectories:
            raise IndexError(f"traj_id {traj_id} out of range [0, {self.num_trajectories})")
        return self._obs_offsets_list[traj_id], self._act_offsets_list[traj_id]

    # ==== Constructors ===========================================================================
    @classmethod
    def from_replay_buffers(
        cls,
        replay_buffers: Iterable[ReplayBuffer],
        dtype: torch.dtype = torch.float32,
    ) -> "SingleStepTrajectoryStore":
        """Build the store from SINGLE-STEP mbrl ``ReplayBuffer``s.

        Each buffer is scanned in storage order and split into trajectories on
        ``terminated | truncated`` (the legacy ``done`` scan of
        ``convert_singlestep_replaybuffer_to_multistep``); a buffer with no ``done`` flag is one
        trajectory. ``o_T`` is taken from the last sample's ``next_obs``. Multistep (composed)
        buffers are rejected: their ``obs`` is wider than ``obs_shape`` of a single step.
        """
        obs_l, act_l, rew_l, term_l, trunc_l = [], [], [], [], []
        for rb in replay_buffers:
            n = int(rb.num_stored)
            if n == 0:
                continue
            batch = rb.get_all(shuffle=False)
            obs = _to_cpu_numpy(batch.obs)[:n]
            act = _to_cpu_numpy(batch.act)[:n]
            next_obs = _to_cpu_numpy(batch.next_obs)[:n]
            rew = _to_cpu_numpy(batch.rewards)[:n].reshape(n)
            term = _to_cpu_numpy(batch.terminateds)[:n].reshape(n).astype(bool)
            trunc = _to_cpu_numpy(batch.truncateds)[:n].reshape(n).astype(bool)
            done = term | trunc
            start = 0
            for k in range(n):
                if done[k] or k == n - 1:
                    sl = slice(start, k + 1)
                    obs_l.append(np.concatenate([obs[sl], next_obs[k : k + 1]], axis=0))
                    act_l.append(act[sl])
                    rew_l.append(rew[sl])
                    term_l.append(term[sl])
                    trunc_l.append(trunc[sl])
                    start = k + 1
        if not obs_l:
            raise ValueError("from_replay_buffers: no stored sample in any replay buffer")
        return cls(obs_l, act_l, rew_l, term_l, trunc_l, dtype=dtype)

    # ==== Anchors ================================================================================
    def anchor_index(self, history_len: int, horizon_len: int, boundary_mode: str = "pad") -> AnchorIndex:
        """Enumerate the ``(traj_id, t)`` anchors of the ``(H, F)`` window (see module docstring).

        :param boundary_mode: ``"pad"`` -> one anchor per single-step sample (legacy parity);
            ``"exclude"`` -> only ``t in [H-1, T-F]`` (window fully inside the trajectory).
        """
        if boundary_mode not in BOUNDARY_MODES:
            raise ValueError(f"boundary_mode must be one of {BOUNDARY_MODES}, got {boundary_mode!r}")
        H, F = int(history_len), int(horizon_len)
        if H < 1 or F < 1:
            raise ValueError(f"history_len and horizon_len must be >= 1, got {H}, {F}")
        traj_ids, ts = [], []
        skipped = []
        for k in range(self.num_trajectories):
            T = self._traj_lengths_list[k]
            if boundary_mode == "pad":
                lo, hi = 0, T - 1
            else:
                lo, hi = H - 1, T - F
            if hi < lo:
                skipped.append((k, T))
                continue
            t = torch.arange(lo, hi + 1, dtype=torch.int64)
            traj_ids.append(torch.full_like(t, k))
            ts.append(t)
        if skipped:
            warnings.warn(
                f"SingleStepTrajectoryStore.anchor_index(H={H}, F={F}, boundary_mode={boundary_mode!r}): "
                f"{len(skipped)} trajectory(ies) shorter than H + F - 1 yield no anchor and are skipped "
                f"(traj_id, T): {skipped[:8]}{' ...' if len(skipped) > 8 else ''}",
                RuntimeWarning,
            )
        if not traj_ids:
            raise ValueError(
                f"SingleStepTrajectoryStore.anchor_index(H={H}, F={F}, boundary_mode={boundary_mode!r}): "
                "zero anchors in the whole store"
            )
        # Anchors are enumerated on the host (cheap, O(N) once) and follow the store's residency.
        return AnchorIndex(torch.cat(traj_ids), torch.cat(ts), H, F, boundary_mode).to(self.device)

    # ==== Vectorized window gather ===============================================================
    def gather_window_steps(
        self,
        traj_ids: torch.Tensor,
        ts: torch.Tensor,
        history_len: int,
        horizon_len: int,
        output_window_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather the per-step blocks of a batch of anchors (boundary padding by index clamping).

        Every index tensor is built on the store's device, so ``traj_ids`` / ``ts`` given on
        another device (e.g. the CPU ids of a ``DataLoader`` sampler) are moved once and the
        outputs live on ``self.device``.

        :return: ``hist_obs (B, H, Do)`` = ``o_{t-H+1..t}``, ``hist_act (B, H, Da)`` =
            ``a_{t-H+1..t}`` (zero before the trajectory start), ``out_obs (B, W, Do)`` =
            ``o_{t+F-W+1..t+F}``, ``out_act (B, W, Da)`` = ``a_{t+F-W+1..t+F}`` (the last step is
            the ``remove_last_action_padding`` slot), ``reward (B,)``, ``terminated (B,)``,
            ``truncated (B,)`` at ``t``.
        """
        H, F = int(history_len), int(horizon_len)
        W = max(H, F) if output_window_len is None else int(output_window_len)
        device = self.device
        traj_ids = torch.as_tensor(traj_ids, dtype=torch.int64, device=device)
        ts = torch.as_tensor(ts, dtype=torch.int64, device=device)
        T = self.traj_lengths[traj_ids]  # (B,)
        obs_off = self.obs_offsets[traj_ids]
        act_off = self.act_offsets[traj_ids]

        # History window: relative obs idx t-H+1 .. t (clamp to [0, T]); act idx clamp to [0, T-1]
        # with the pre-start slots zeroed (legacy ``reset_buffers(obs)`` -> zero act reset).
        hist_rel = ts[:, None] + torch.arange(-H + 1, 1, dtype=torch.int64, device=device)[None, :]  # (B, H)
        hist_obs_idx = obs_off[:, None] + hist_rel.clamp(min=0)
        hist_act_valid = hist_rel >= 0
        hist_act_idx = act_off[:, None] + hist_rel.clamp(min=0).minimum((T - 1)[:, None])
        hist_obs = self.obs_all[hist_obs_idx]
        hist_act = self.act_all[hist_act_idx] * hist_act_valid.unsqueeze(-1).to(self.dtype)

        # Output window: relative obs idx t+F-W+1 .. t+F (clamp to [0, T]); act idx clamp to
        # [0, T-1] (post-end slots repeat a_{T-1}, the legacy ``pad_extension`` behaviour).
        out_rel = ts[:, None] + torch.arange(F - W + 1, F + 1, dtype=torch.int64, device=device)[None, :]  # (B, W)
        out_obs_idx = obs_off[:, None] + out_rel.clamp(min=0).minimum(T[:, None])
        out_act_rel = out_rel.clamp(min=0).minimum((T - 1)[:, None])
        out_act_valid = out_rel >= 0
        out_act_idx = act_off[:, None] + out_act_rel
        out_obs = self.obs_all[out_obs_idx]
        out_act = self.act_all[out_act_idx] * out_act_valid.unsqueeze(-1).to(self.dtype)

        at_t = act_off + ts
        return (
            hist_obs,
            hist_act,
            out_obs,
            out_act,
            self.reward_all[at_t],
            self.terminated_all[at_t],
            self.truncated_all[at_t],
        )

    # ==== HDF5 persistence =======================================================================
    def save_hdf5(self, path: Union[str, os.PathLike], compression: Optional[str] = "lzf") -> str:
        """Persist the store (one group per trajectory, chunked + compressed) atomically:
        written to ``<path>.tmp`` then renamed (plan risk R10: no half-written cache is ever read).
        """
        path = os.fspath(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with h5py.File(tmp, "w") as f:
            f.attrs["format_version"] = _HDF5_FORMAT_VERSION
            f.attrs["num_trajectories"] = self.num_trajectories
            f.attrs["obs_dim"] = self.obs_dim
            f.attrs["act_dim"] = self.act_dim
            f.attrs["dtype"] = str(self.dtype).replace("torch.", "")
            for k in range(self.num_trajectories):
                g = f.create_group(f"traj_{k:06d}")
                obs, act = self.trajectory(k)
                T = self._traj_lengths_list[k]
                _, a0 = self._offsets_of(k)
                for name, arr in (
                    ("obs", _to_cpu_numpy(obs)),
                    ("act", _to_cpu_numpy(act)),
                    ("reward", _to_cpu_numpy(self.reward_all[a0 : a0 + T])),
                    ("terminated", _to_cpu_numpy(self.terminated_all[a0 : a0 + T])),
                    ("truncated", _to_cpu_numpy(self.truncated_all[a0 : a0 + T])),
                ):
                    g.create_dataset(name, data=arr, chunks=True, compression=compression)
        os.replace(tmp, path)
        return path

    @classmethod
    def load_hdf5(
        cls,
        path: Union[str, os.PathLike],
        dtype: Optional[torch.dtype] = None,
        device: Union[str, torch.device, None] = None,
    ) -> "SingleStepTrajectoryStore":
        """Load a store written by :meth:`save_hdf5` fully into RAM (the single-step data is
        small; the windows are what explode). ``dtype`` overrides the stored dtype (model dtype);
        ``device`` places the flat storage (``None`` -> CPU).
        """
        with h5py.File(os.fspath(path), "r") as f:
            version = int(f.attrs.get("format_version", -1))
            if version != _HDF5_FORMAT_VERSION:
                raise ValueError(f"{path}: unsupported store format_version {version} (expected {_HDF5_FORMAT_VERSION})")
            stored_dtype = getattr(torch, str(f.attrs["dtype"]))
            n = int(f.attrs["num_trajectories"])
            obs_l, act_l, rew_l, term_l, trunc_l = [], [], [], [], []
            for k in range(n):
                g = f[f"traj_{k:06d}"]
                obs_l.append(np.asarray(g["obs"]))
                act_l.append(np.asarray(g["act"]))
                rew_l.append(np.asarray(g["reward"]))
                term_l.append(np.asarray(g["terminated"]))
                trunc_l.append(np.asarray(g["truncated"]))
        return cls(
            obs_l, act_l, rew_l, term_l, trunc_l, dtype=stored_dtype if dtype is None else dtype, device=device
        )


def _to_cpu_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
