# coding=utf-8
"""ERLL data-source seam: the legacy replay buffer and the lazy window DataLoader behind one
contract (RLRP-824, KD4 / KD5 / KD9 / KD12 / FR8 / FR12).

Permanent module. Introduced by Step 4 of the Ultra-long-horizon MS->MS training via a lazy
window DataLoader ``.junie`` plan (``ultra_long_horizon_dataloader_plan_RLRP-824_20260915.md``).

``AbstractExperienceReplayLearningLoop`` touches its source data at six points only (normalizer
fit, feature-weight / normalization diagnostics, dataset sizing, device resolution, train / val
split, per-pass iterator construction). :class:`ERLLDataSource` names them; two realisations exist:

- :class:`ReplayBufferDataSource` wraps a materialized mbrl ``ReplayBuffer`` and reproduces the
  legacy behaviour byte-for-byte (the ERLL keeps calling ``split_replay_buffer`` and building its
  ``TransitionIterator``s exactly as before -- this wrapper only exposes the buffer);
- :class:`WindowDataLoaderDataSource` wraps a
  :class:`~tools.multistep_tools.window_dataset.multistep_window_dataset.MultistepWindowDataset`:
  sample-level seeded split recorded as a :class:`DataSplitRecord`, ONE ``DataLoader`` pair built
  for the whole run (constant batch size, FR12: ``persistent_workers`` when ``num_workers > 0``,
  ``pin_memory`` on CUDA only, ``prefetch_factor`` from ``cfg.mbrl_lib.dataloader``), a bounded
  normalizer-fit window sub-sample (KD5) and the daemonic-process worker guard (KD9). When the
  dataset is DEVICE-resident (store on the training device, see
  ``SingleStepTrajectoryStore.to``) the batches are composed on that device directly, so
  ``num_workers`` is forced to ``0`` (workers cannot share CUDA storage) and ``pin_memory`` is
  meaningless -- both are resolved (and reported) at construction.
"""
from __future__ import annotations

import multiprocessing
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
from mbrl.types import TransitionBatch
from mbrl.util.replay_buffer import ReplayBuffer
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from tools.multistep_tools.window_dataset.multistep_window_dataset import (
    MultistepWindowDataset,
    identity_collate,
)
from tools.multistep_tools.window_dataset.split_record import (
    DataSplitRecord,
    split_anchors,
)

DATA_MANAGER_REPLAY_BUFFER = "replay-buffer"
DATA_MANAGER_DATALOADER = "dataloader"
DATA_MANAGERS: Tuple[str, ...] = (DATA_MANAGER_REPLAY_BUFFER, DATA_MANAGER_DATALOADER)

DEFAULT_NORMALIZER_FIT_MAX_WINDOWS = 65_536


class ERLLDataSource:
    """Read-only provider of the ERLL's initial training data + metadata (the seam contract).

    Sub-classes implement the six touch-points of ``AbstractExperienceReplayLearningLoop``;
    ``is_replay_buffer`` tells the loop whether the materialized-buffer machinery (sequence
    iterators, partition surgery, device relocation) is available.
    """

    is_replay_buffer: bool = False

    # ---- metadata ---------------------------------------------------------------------------
    @property
    def num_stored(self) -> int:
        raise NotImplementedError

    @property
    def obs_shape(self) -> tuple:
        raise NotImplementedError

    @property
    def action_shape(self) -> tuple:
        raise NotImplementedError

    @property
    def obs_type(self):
        raise NotImplementedError

    @property
    def action_type(self):
        raise NotImplementedError

    @property
    def reward_type(self):
        raise NotImplementedError

    # ---- normalizer fit / diagnostics -------------------------------------------------------
    def normalizer_fit_batch(self) -> TransitionBatch:
        """The batch handed to ``model.update_normalizer`` (and to the feature-weight /
        normalization diagnostics through :meth:`get_all`)."""
        raise NotImplementedError

    def get_all(self) -> TransitionBatch:
        """Duck-typed ``ReplayBuffer.get_all()`` for the diagnostics that only read a batch."""
        return self.normalizer_fit_batch()

    # ---- split / iterables ------------------------------------------------------------------
    def split(self, datasets_size: Optional[int], val_ratio: float) -> None:
        raise NotImplementedError

    @property
    def train_num_stored(self) -> int:
        raise NotImplementedError

    @property
    def val_num_stored(self) -> int:
        raise NotImplementedError

    def train_iterable(self, batch_size: int):
        raise NotImplementedError

    def val_iterable(self, batch_size: int):
        raise NotImplementedError

    def split_record(self) -> Optional[DataSplitRecord]:
        return None


class ReplayBufferDataSource(ERLLDataSource):
    """The legacy source: a materialized multistep mbrl ``ReplayBuffer``.

    Deliberately THIN: the ERLL keeps its historical ``split_replay_buffer`` /
    ``get_basic_buffer_iterators`` / ``get_sequence_buffer_iterator`` code for this source so the
    replay-buffer path stays byte-identical; this class only exposes the buffer and its metadata.
    """

    is_replay_buffer = True

    def __init__(self, replay_buffer: ReplayBuffer):
        self.replay_buffer = replay_buffer

    @property
    def num_stored(self) -> int:
        return int(self.replay_buffer.num_stored)

    @property
    def obs_shape(self) -> tuple:
        return tuple(self.replay_buffer.obs_shape)

    @property
    def action_shape(self) -> tuple:
        return tuple(self.replay_buffer.action_shape)

    @property
    def obs_type(self):
        return self.replay_buffer.obs_type

    @property
    def action_type(self):
        return self.replay_buffer.action_type

    @property
    def reward_type(self):
        return self.replay_buffer.reward_type

    def normalizer_fit_batch(self) -> TransitionBatch:
        return self.replay_buffer.get_all()

    def get_all(self) -> TransitionBatch:
        return self.replay_buffer.get_all()

    def split(self, datasets_size: Optional[int], val_ratio: float) -> None:  # pragma: no cover
        raise NotImplementedError(
            "ReplayBufferDataSource: the ERLL splits a replay buffer itself (split_replay_buffer)"
        )

    def train_iterable(self, batch_size: int):  # pragma: no cover
        raise NotImplementedError("ReplayBufferDataSource: the ERLL builds TransitionIterators itself")

    def val_iterable(self, batch_size: int):  # pragma: no cover
        raise NotImplementedError("ReplayBufferDataSource: the ERLL builds TransitionIterators itself")


class _WindowSplitView:
    """Buffer-like view of one side of a split (``num_stored`` / ``stores_trajectories`` are the
    two attributes the ERLL reads on its train / val buffers)."""

    stores_trajectories = False

    def __init__(self, dataset: MultistepWindowDataset, name: str):
        self.dataset = dataset
        self.name = name

    @property
    def num_stored(self) -> int:
        return len(self.dataset)

    def get_all(self) -> TransitionBatch:
        return self.dataset.all_rows()

    def __repr__(self) -> str:
        return f"_WindowSplitView({self.name}, {self.dataset!r})"


def resolve_dataloader_num_workers(requested: int, *, warn: bool = True) -> int:
    """KD9 daemon guard: ``DataLoader`` workers cannot be spawned from a daemonic process (the
    Joblib / loky launcher workers of ``multirun_named_subdir.yaml``); fall back to ``0``."""
    requested = int(requested)
    if requested > 0 and multiprocessing.current_process().daemon:
        if warn:
            warnings.warn(
                f"mbrl_lib.dataloader.num_workers={requested} requested inside a DAEMONIC process "
                f"({multiprocessing.current_process().name}; e.g. a Joblib/loky Hydra launcher "
                "worker): daemonic processes cannot spawn DataLoader workers -> falling back to "
                "num_workers=0. Use the sequential launcher (`hydra/launcher: basic`, e.g. "
                "`multirun_base_sequential.yaml` / the SLURM job-array config) to keep workers "
                "(RLRP-824 KD9).",
                RuntimeWarning,
            )
        return 0
    return requested


class WindowDataLoaderDataSource(ERLLDataSource):
    """The lazy window path: a ``MultistepWindowDataset`` split sample-level and served by ONE
    ``DataLoader`` pair for the whole run.

    :param dataset: the full-anchor ``MultistepWindowDataset`` (all anchors of the store).
    :param batch_size: the run's constant training batch size (FR12).
    :param seed: split permutation seed (``cfg.seed``).
    :param num_workers: ``DataLoader`` workers (``mbrl_lib.dataloader.num_workers``; daemon-guarded).
    :param pin_memory: pin host batches (honoured on a CUDA ``device`` only).
    :param persistent_workers: keep workers alive across epochs (forced ``True`` when
        ``num_workers > 0`` -- one loader pair serves every pass).
    :param prefetch_factor: batches prefetched per worker (``None`` -> torch default).
    :param device: the model device (decides ``pin_memory``). Independent of the DATASET device:
        a device-resident dataset forces ``num_workers=0`` / ``pin_memory=False`` whatever the
        requested values.
    :param normalizer_fit_max_windows: bounded window sub-sample size for the normalizer fit (KD5).
    :param val_batch_size: validation batch size (default: ``batch_size``).
    :param shuffle_seed: seed of the per-epoch training shuffle generator (default: ``seed``).
    """

    is_replay_buffer = False

    def __init__(
        self,
        dataset: MultistepWindowDataset,
        batch_size: int,
        seed: Optional[int],
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        prefetch_factor: Optional[int] = None,
        device: Union[str, torch.device, None] = None,
        normalizer_fit_max_windows: int = DEFAULT_NORMALIZER_FIT_MAX_WINDOWS,
        val_batch_size: Optional[int] = None,
        shuffle_seed: Optional[int] = None,
    ):
        if len(dataset) != len(dataset.anchors):
            raise ValueError(
                "WindowDataLoaderDataSource expects the FULL anchor dataset (it performs the "
                f"train / val split itself); got a {len(dataset)}/{len(dataset.anchors)} subset"
            )
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.val_batch_size = self.batch_size if val_batch_size is None else int(val_batch_size)
        if self.batch_size < 1 or self.val_batch_size < 1:
            raise ValueError("batch sizes must be >= 1")
        self.seed = None if seed is None else int(seed)
        self.shuffle_seed = self.seed if shuffle_seed is None else int(shuffle_seed)
        self.dataset_device: torch.device = dataset.device
        if self.dataset_device.type != "cpu":
            if int(num_workers) > 0:
                warnings.warn(
                    f"mbrl_lib.dataloader.num_workers={int(num_workers)} requested but the window dataset is "
                    f"resident on {self.dataset_device}: batches are composed on that device in the main "
                    "process (DataLoader workers cannot share device storage) -> num_workers=0, "
                    "pin_memory=False.",
                    RuntimeWarning,
                )
            self.num_workers = 0
            self.pin_memory = False
        else:
            self.num_workers = resolve_dataloader_num_workers(num_workers)
            dev_type = torch.device(device).type if device is not None else "cpu"
            self.pin_memory = bool(pin_memory) and dev_type == "cuda"
        self.persistent_workers = bool(persistent_workers) or self.num_workers > 0
        self.prefetch_factor = prefetch_factor
        self.normalizer_fit_max_windows = int(normalizer_fit_max_windows)

        self._train_ds: Optional[MultistepWindowDataset] = None
        self._val_ds: Optional[MultistepWindowDataset] = None
        self._record: Optional[DataSplitRecord] = None
        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None
        self._fit_batch: Optional[TransitionBatch] = None

    # ---- metadata ---------------------------------------------------------------------------
    @property
    def num_stored(self) -> int:
        return len(self.dataset)

    @property
    def obs_shape(self) -> tuple:
        return (self.dataset.obs_width,)

    @property
    def action_shape(self) -> tuple:
        return (self.dataset.act_dim,)

    @property
    def obs_type(self):
        return _np_dtype(self.dataset.store.dtype)

    @property
    def action_type(self):
        return _np_dtype(self.dataset.store.dtype)

    @property
    def reward_type(self):
        return _np_dtype(self.dataset.store.dtype)

    # ---- normalizer fit -----------------------------------------------------------------------
    def normalizer_fit_batch(self) -> TransitionBatch:
        """A deterministic (seeded) sub-sample of at most ``normalizer_fit_max_windows`` composed
        windows of the FULL anchor set (KD5). Statistics are per single-step feature (the
        normalizer reshapes the composed blocks to ``(-1, Do)``), so a bounded sample converges
        with far fewer windows than the full set; the whole set is used when it is smaller."""
        if self._fit_batch is None:
            n = len(self.dataset)
            if n <= self.normalizer_fit_max_windows:
                ids = torch.arange(n, dtype=torch.int64, device=self.dataset_device)
            else:
                rng = np.random.default_rng(self.seed)
                ids = torch.as_tensor(
                    np.sort(rng.choice(n, size=self.normalizer_fit_max_windows, replace=False)),
                    dtype=torch.int64,
                )
            self._fit_batch = self.dataset.gather(ids)
        return self._fit_batch

    # ---- split --------------------------------------------------------------------------------
    def split(self, datasets_size: Optional[Union[int, str]], val_ratio: float) -> None:
        train_ids, val_ids, record = split_anchors(
            self.dataset.anchors, val_ratio=val_ratio, datasets_size=datasets_size, seed=self.seed
        )
        self._apply_split(train_ids, val_ids, record)

    def restore_split(self, record: DataSplitRecord) -> None:
        """Reuse a recorded split verbatim (resume-from-checkpoint, FR7)."""
        train_ids, val_ids = record.anchor_ids(self.dataset.anchors)
        self._apply_split(train_ids, val_ids, record)

    def _apply_split(self, train_ids: torch.Tensor, val_ids: torch.Tensor, record: DataSplitRecord) -> None:
        if train_ids.numel() == 0:
            raise ValueError("WindowDataLoaderDataSource: empty training split")
        self._train_ds = self.dataset.subset(train_ids)
        self._val_ds = self.dataset.subset(val_ids) if val_ids.numel() > 0 else None
        self._record = record
        self._train_loader = None
        self._val_loader = None

    def _require_split(self) -> None:
        if self._train_ds is None:
            raise RuntimeError("WindowDataLoaderDataSource.split() must be called before use")

    @property
    def train_dataset(self) -> MultistepWindowDataset:
        self._require_split()
        return self._train_ds

    @property
    def val_dataset(self) -> Optional[MultistepWindowDataset]:
        self._require_split()
        return self._val_ds

    @property
    def train_num_stored(self) -> int:
        return len(self.train_dataset)

    @property
    def val_num_stored(self) -> int:
        return 0 if self.val_dataset is None else len(self.val_dataset)

    def train_view(self) -> _WindowSplitView:
        return _WindowSplitView(self.train_dataset, "train")

    def val_view(self) -> Optional[_WindowSplitView]:
        return None if self.val_dataset is None else _WindowSplitView(self.val_dataset, "val")

    def split_record(self) -> Optional[DataSplitRecord]:
        return self._record

    # ---- loaders (built ONCE, FR12) ---------------------------------------------------------
    def _loader_kwargs(self) -> dict:
        kwargs = dict(num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=identity_collate)
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return kwargs

    def train_iterable(self, batch_size: int) -> DataLoader:
        """The training ``DataLoader`` (one object for the whole run; ``batch_size`` must equal
        the constructor's constant batch size -- FR12)."""
        self._check_batch_size(batch_size, self.batch_size, "train")
        if self._train_loader is None:
            gen = torch.Generator()
            if self.shuffle_seed is not None:
                gen.manual_seed(self.shuffle_seed)
            sampler = BatchSampler(
                RandomSampler(self.train_dataset, generator=gen), batch_size=self.batch_size, drop_last=False
            )
            self._train_loader = DataLoader(self.train_dataset, batch_sampler=sampler, **self._loader_kwargs())
        return self._train_loader

    def val_iterable(self, batch_size: Optional[int] = None) -> Optional[DataLoader]:
        """The validation ``DataLoader`` (sequential, one object for the whole run) or ``None``
        when ``val_ratio == 0``."""
        if batch_size is not None:
            self._check_batch_size(batch_size, self.val_batch_size, "val")
        if self.val_dataset is None:
            return None
        if self._val_loader is None:
            sampler = BatchSampler(
                range(len(self.val_dataset)), batch_size=self.val_batch_size, drop_last=False
            )
            self._val_loader = DataLoader(self.val_dataset, batch_sampler=sampler, **self._loader_kwargs())
        return self._val_loader

    @staticmethod
    def _check_batch_size(requested: int, fixed: int, name: str) -> None:
        if int(requested) != int(fixed):
            raise ValueError(
                f"WindowDataLoaderDataSource: the {name} loader was built for a CONSTANT batch size "
                f"{fixed} but {requested} was requested. Progressive batch-size scheduling "
                "(`UDER.batch_size.init_value != limit` / `gamma != 1`) is not supported on the "
                "dataloader path (RLRP-824 FR12, deferred follow-up)."
            )

    def __repr__(self) -> str:
        return (
            f"WindowDataLoaderDataSource(batch_size={self.batch_size}, num_workers={self.num_workers}, "
            f"pin_memory={self.pin_memory}, persistent_workers={self.persistent_workers}, "
            f"dataset_device={self.dataset_device}, seed={self.seed}, dataset={self.dataset!r})"
        )


def _np_dtype(torch_dtype: torch.dtype):
    return torch.empty(0, dtype=torch_dtype).numpy().dtype


def validate_constant_batch_size_for_dataloader(cfg, loop_kind: str) -> None:
    """FR12 gate: ``pipeline.data_manager: dataloader`` requires a constant batch size.

    ``ValueError`` when ``UDER.batch_size.init_value != UDER.batch_size.limit`` or, for the PBER
    loop, ``gamma != 1.0`` with a non-trivial schedule; ``single_global_loop`` is constant by
    construction. ``uder`` (partition surgery) is rejected outright.
    """
    bs = cfg.UDER.batch_size
    loop_kind = str(loop_kind)
    if loop_kind == "uder":
        raise NotImplementedError(
            "pipeline.data_manager=dataloader does not support UDER.loop_kind=uder "
            "(PartitionBasedUncertaintyDrivenERLL needs replay-buffer partition surgery); use "
            "single_global_loop or pber (RLRP-824 deferred follow-up)."
        )
    init_value = int(bs.init_value)
    limit = bs.get("limit", None)
    if loop_kind == "pber" and limit is not None and int(limit) != init_value:
        # With ``init_value == limit`` the geometric ``gamma`` schedule is clamped at the very
        # first pass, so the batch size is constant whatever ``gamma`` is; the ``limit`` check
        # is therefore the complete condition.
        raise ValueError(
            f"pipeline.data_manager=dataloader requires a CONSTANT batch size but "
            f"UDER.batch_size.init_value={init_value} != UDER.batch_size.limit={limit} "
            f"(gamma={bs.get('gamma', None)}). Progressive batch-size scheduling on the dataloader "
            "path is a deferred follow-up (RLRP-824 FR12 / operator decision 7)."
        )
    return None
