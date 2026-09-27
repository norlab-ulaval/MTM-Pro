# coding=utf-8
"""Sample-level train / val anchor split with a reproducible, verifiable record (RLRP-824, FR5 / KD6).

Permanent module. Introduced by Step 3 of the Ultra-long-horizon MS->MS training via a lazy
window DataLoader ``.junie`` plan (``ultra_long_horizon_dataloader_plan_RLRP-824_20260915.md``).

Protocol parity with the legacy ``split_replay_buffer`` (operator decision 11): the split is
SAMPLE-level -- a uniform random permutation of the anchor ids, the first ``train_size`` go to the
training set and the next ``val_size = int(datasets_size * val_ratio)`` to the validation set
(``datasets_size`` defaults to every anchor; a smaller value sub-samples the dataset exactly like
the legacy ``datasets_size`` knob). Two deliberate improvements over the legacy function, which
seeds its ``RandomReplayBufferExplorationPolicy`` with ``np.random.default_rng(None)`` (OS entropy,
hence a DIFFERENT split at every run): the permutation here is drawn from
``numpy.random.default_rng(seed)`` so the split is reproducible under ``cfg.seed``, and it is
RECORDED (:class:`DataSplitRecord`, JSON) so a resumed run reuses the very same anchors and any
reader can re-derive / verify it from the source store alone.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from tools.multistep_tools.window_dataset.single_step_trajectory_store import (
    AnchorIndex,
    SingleStepTrajectoryStore,
)

_RECORD_FORMAT_VERSION = 1


@dataclass
class DataSplitRecord:
    """Everything needed to reconstruct and verify a sample-level anchor split.

    ``val_anchors`` is the SORTED list of validation ``(traj_id, t)`` pairs; the training anchors
    are the complement within the first ``num_anchors_used`` permuted anchors (``train_anchor_ids``
    keeps them explicitly so a ``datasets_size < num_anchors`` sub-sample is reconstructible too).
    """

    history_len: int
    horizon_len: int
    boundary_mode: str
    seed: Optional[int]
    val_ratio: float
    datasets_size: Optional[int]
    num_anchors: int
    num_train: int
    num_val: int
    val_anchor_hash: str
    val_anchors: List[Tuple[int, int]]
    train_anchor_ids: List[int] = field(default_factory=list)
    format_version: int = _RECORD_FORMAT_VERSION

    # ==== (De)serialization ======================================================================
    def to_json(self, path: Union[str, os.PathLike]) -> str:
        path = os.fspath(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = asdict(self)
        payload["val_anchors"] = [list(p) for p in self.val_anchors]
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        os.replace(tmp, path)
        return path

    @classmethod
    def from_json(cls, path: Union[str, os.PathLike]) -> "DataSplitRecord":
        with open(os.fspath(path), "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        version = int(payload.get("format_version", -1))
        if version != _RECORD_FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported split record format_version {version}")
        payload["val_anchors"] = [tuple(int(v) for v in p) for p in payload["val_anchors"]]
        payload["train_anchor_ids"] = [int(v) for v in payload.get("train_anchor_ids", [])]
        return cls(**payload)

    # ==== Reconstruction / verification ==========================================================
    def anchor_ids(self, anchors: AnchorIndex) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(train_ids, val_ids)`` flat anchor ids of ``anchors`` matching this record."""
        _check_anchor_compat(self, anchors)
        lookup = {pair: i for i, pair in enumerate(anchors.pairs())}
        try:
            val_ids = torch.tensor([lookup[p] for p in self.val_anchors], dtype=torch.int64)
        except KeyError as e:
            raise ValueError(f"split record references an anchor absent from the store: {e}") from e
        train_ids = torch.tensor(self.train_anchor_ids, dtype=torch.int64)
        return train_ids, val_ids

    def verify(self, store: SingleStepTrajectoryStore) -> bool:
        """Re-derive the split from ``store`` + this record's parameters and compare the hashes.

        :raises ValueError: when the record is inconsistent with the store (tampered anchors,
            different seed / ratio / window / boundary mode, wrong anchor count).
        """
        anchors = store.anchor_index(self.history_len, self.horizon_len, self.boundary_mode)
        _check_anchor_compat(self, anchors)
        train_ids, val_ids, rederived = split_anchors(
            anchors, val_ratio=self.val_ratio, datasets_size=self.datasets_size, seed=self.seed
        )
        if rederived.val_anchor_hash != self.val_anchor_hash:
            raise ValueError(
                "split record verification failed: re-derived validation anchor hash "
                f"{rederived.val_anchor_hash[:12]}... != recorded {self.val_anchor_hash[:12]}..."
            )
        if rederived.val_anchors != self.val_anchors or rederived.train_anchor_ids != self.train_anchor_ids:
            raise ValueError("split record verification failed: anchor lists differ from the re-derived split")
        if _hash_pairs(self.val_anchors) != self.val_anchor_hash:
            raise ValueError("split record verification failed: recorded val anchors do not match their hash")
        return True


def split_anchors(
    anchors: AnchorIndex,
    val_ratio: float,
    datasets_size: Optional[Union[int, str]] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, DataSplitRecord]:
    """Sample-level seeded split of an anchor index (legacy ``split_replay_buffer`` protocol).

    :param anchors: the full anchor index of the store for ``(H, F)``.
    :param val_ratio: validation fraction in ``[0, 1)``; ``val_size = int(datasets_size * val_ratio)``.
    :param datasets_size: number of anchors to use (``None`` / ``"all"`` -> every anchor);
        must be ``<= len(anchors)``.
    :param seed: permutation seed (``numpy.random.default_rng(seed)``).
    :return: ``(train_ids, val_ids, record)`` -- flat anchor ids (int64 tensors) + the record.
    """
    n = len(anchors)
    if isinstance(datasets_size, str):
        if datasets_size.lower() not in ("all", "full", "none"):
            raise ValueError(f"datasets_size string must be 'all', got {datasets_size!r}")
        datasets_size = None
    if datasets_size is None:
        datasets_size = n
    datasets_size = int(datasets_size)
    if datasets_size <= 0 or datasets_size > n:
        raise ValueError(f"datasets_size {datasets_size} must be in [1, {n}] (number of anchors)")
    if not (0.0 <= float(val_ratio) < 1.0):
        raise ValueError(f"val_ratio {val_ratio} must be in [0, 1)")
    val_size = int(datasets_size * float(val_ratio))
    train_size = datasets_size - val_size
    if train_size <= 0:
        raise ValueError(f"empty training set: datasets_size={datasets_size}, val_ratio={val_ratio}")
    if float(val_ratio) > 0.0 and val_size == 0:
        raise ValueError(
            f"val_ratio={val_ratio} yields an EMPTY validation set for datasets_size={datasets_size}"
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)[:datasets_size]
    train_ids = np.sort(perm[:train_size])
    val_ids = np.sort(perm[train_size : train_size + val_size])

    pairs = anchors.pairs()
    val_pairs = sorted(pairs[i] for i in val_ids.tolist())
    record = DataSplitRecord(
        history_len=int(anchors.history_len),
        horizon_len=int(anchors.horizon_len),
        boundary_mode=str(anchors.boundary_mode),
        seed=None if seed is None else int(seed),
        val_ratio=float(val_ratio),
        datasets_size=datasets_size,
        num_anchors=n,
        num_train=int(train_size),
        num_val=int(val_size),
        val_anchor_hash=_hash_pairs(val_pairs),
        val_anchors=val_pairs,
        train_anchor_ids=[int(i) for i in train_ids.tolist()],
    )
    return (
        torch.as_tensor(train_ids, dtype=torch.int64),
        torch.as_tensor(val_ids, dtype=torch.int64),
        record,
    )


def _hash_pairs(pairs: List[Tuple[int, int]]) -> str:
    h = hashlib.sha256()
    for traj_id, t in sorted(pairs):
        h.update(f"{int(traj_id)}:{int(t)};".encode("ascii"))
    return h.hexdigest()


def _check_anchor_compat(record: DataSplitRecord, anchors: AnchorIndex) -> None:
    if (
        anchors.history_len != record.history_len
        or anchors.horizon_len != record.horizon_len
        or anchors.boundary_mode != record.boundary_mode
    ):
        raise ValueError(
            f"split record (H={record.history_len}, F={record.horizon_len}, "
            f"boundary_mode={record.boundary_mode!r}) does not match the anchor index "
            f"(H={anchors.history_len}, F={anchors.horizon_len}, boundary_mode={anchors.boundary_mode!r})"
        )
    if len(anchors) != record.num_anchors:
        raise ValueError(
            f"split record was built on {record.num_anchors} anchors but the store yields {len(anchors)}"
        )
