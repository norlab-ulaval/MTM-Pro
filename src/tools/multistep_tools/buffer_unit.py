# coding=utf-8
import collections
from copy import copy, deepcopy
from typing import Optional, Tuple, Deque, Any, Union
from warnings import warn

import numpy as np
import torch

from tools.console_tools.message import (
    consol_msg_universal_one_liner,
)


def _use_gpu_optimized_storage(device: Optional[torch.device]) -> bool:
    """Determine whether to use GPU-optimized pre-allocated tensor storage.

    On GPU (cuda, mps), pre-allocated circular tensor storage avoids repeated GPU memory
    allocations and is faster. On CPU, deque + torch.cat is faster due to PyTorch's dispatch
    overhead on small tensors.

    :param device: The torch device or None.
    :return: True if GPU-optimized storage should be used.
    """
    if device is None:
        return False
    if isinstance(device, str):
        device = torch.device(device)
    return device.type in ("cuda", "mps")


class MultistepBuffer:
    multistep_length: int
    single_step_len: int
    multistep_extension_len: int
    _buffer: Deque[Any]
    info: Optional[str]
    extension_padding_counter: int
    stored_type: Any = None
    stored_shape: tuple

    def __init__(
        self,
        multistep_len: int,
        single_step_len: int,
        multistep_extension_len: int = 0,
        info: Optional[str] = None,
        consol_log=True,
        device: Optional[torch.device] = None,
    ):
        """Tools to collect and agregate multiple step observation/action samples

        :param multistep_len: The numbers of past steps conserved in the buffer
        :param single_step_len: The numbers of components per step, eg: action=(steer, speed).
        :param multistep_extension_len: Extend the buffer lenght
        :param info: Information about what the buffer collect, eg: "Observation"
        :param consol_log:
        :param device: Torch device for GPU-optimized storage. When device is GPU (cuda/mps),
            uses pre-allocated circular tensor storage for better performance. When None or CPU,
            uses deque-based storage which is faster for small CPU tensors.
        """

        # .... Pre-condition ......................................................................
        assert single_step_len >= 1
        assert multistep_len >= 1
        assert multistep_extension_len >= 0

        # .... Setup buffer .......................................................................
        self.single_step_len = single_step_len
        self.multistep_length = multistep_len
        self.multistep_extension_len = multistep_extension_len
        self.extension_padding_counter = 0
        self._capacity_val = self.multistep_length + self.multistep_extension_len

        self._device = device
        self._use_gpu_storage = _use_gpu_optimized_storage(device)

        # GPU-optimized: pre-allocated circular tensor storage
        self._gpu_storage: Optional[torch.Tensor] = None
        self._gpu_write_idx: int = 0
        self._gpu_count: int = 0

        # CPU-optimized: deque-based storage
        self._buffer = collections.deque(maxlen=self._capacity_val)

        # .... Setup buffer user note and console feedback ........................................
        self.info = info
        self._consol_log = consol_log

        if consol_log:
            info_str = ""
            if self.info:
                info_str = f"{info} ›"

            storage_mode = "GPU pre-allocated" if self._use_gpu_storage else "deque"
            consol_msg_universal_one_liner(
                f"{info_str} Initialize with multistep_length="
                f"{self.multistep_length}, single_step_len={self.single_step_len} and "
                f"multistep_extension_len={self.multistep_extension_len}"
                f" (storage: {storage_mode})."
            )

    def append(self, sample: Any) -> None:
        """Append sampled observation/action to multistep buffer

        :param sample: a single-step data sample
        """
        if isinstance(sample, (np.ndarray, torch.Tensor)):
            assert self.single_step_len == sample.shape[-1], (
                f"{self.single_step_len} != " f"{sample.shape[-1]}"
            )
        else:
            assert self.single_step_len == 1, f"{self.single_step_len} != 1"

        if self._use_gpu_storage and isinstance(sample, torch.Tensor):
            self._gpu_append(sample)
        elif isinstance(sample, torch.Tensor):
            if self.capacity == len(self):
                removed_sample = self._buffer.popleft()
                del removed_sample
            self._buffer.append(sample.detach().clone())
        elif isinstance(sample, np.ndarray):
            if self.capacity == len(self):
                removed_sample = self._buffer.popleft()
                del removed_sample
            self._buffer.append(sample.copy())
        else:
            self._buffer.append(deepcopy(sample))
        return None

    def _gpu_append(self, sample: torch.Tensor) -> None:
        """GPU-optimized append: write into pre-allocated circular tensor storage."""
        if self._gpu_storage is None:
            self._gpu_storage = torch.zeros(
                self._capacity_val, self.single_step_len,
                device=sample.device, dtype=sample.dtype,
            )
            self._gpu_write_idx = 0
            self._gpu_count = 0

        self._gpu_storage[self._gpu_write_idx] = sample.detach()
        self._gpu_write_idx = (self._gpu_write_idx + 1) % self._capacity_val
        self._gpu_count = min(self._gpu_count + 1, self._capacity_val)

    def reset(self, sample: Any) -> None:
        """Bootstrap the multistep buffer with an initial observation/action.

        :param sample: a single-step data sample
        """
        self.stored_type = type(sample)
        if isinstance(sample, (np.ndarray, torch.Tensor)):
            assert self.single_step_len == sample.shape[-1], (
                f"{self.single_step_len=} != " f"{sample.shape[-1]=}"
            )
        else:
            assert self.single_step_len == 1, f"{self.single_step_len=} != 1"

        if self._use_gpu_storage and isinstance(sample, torch.Tensor):
            self._gpu_reset(sample)
        else:
            self._buffer.clear()
            if isinstance(sample, np.ndarray):
                sample = sample.copy()
            while len(self._buffer) != self._capacity_val:
                self.append(sample)

        self.extension_padding_counter = 0
        return None

    def _gpu_reset(self, sample: torch.Tensor) -> None:
        """GPU-optimized reset: fill pre-allocated storage in one operation."""
        if self._gpu_storage is None:
            self._gpu_storage = torch.zeros(
                self._capacity_val, self.single_step_len,
                device=sample.device, dtype=sample.dtype,
            )
        # Fill all slots with the same sample value
        self._gpu_storage[:] = sample.detach()
        self._gpu_write_idx = 0
        self._gpu_count = self._capacity_val

    def get_buffer(self) -> Union[np.ndarray, torch.Tensor]:
        # (CRITICAL) todo: validate usage considering the new support for ensemble of RLRP-350
        # (CRITICAL) todo: update usage considering the new support for bootstrap sequence of RLRP-349
        """Return a numpy array or torch tensor copy of the mutistep buffer following shape

            (..., F[1:D]_1 + ... +  F[1:D]_MS)

        with feature F of D dimension len and MS multistep len i.e., features dim are chunked by
        timesteps from 1 to MS.

        :return: a multistep buffer array
        """
        if self._use_gpu_storage and self._gpu_storage is not None:
            return self._gpu_get_buffer()

        buffer = list(self._buffer)
        if issubclass(self.stored_type, torch.Tensor):
            # Device-safety: the deque-storage path keeps whatever tensors
            # are appended verbatim. Callers can mix CPU- and CUDA-resident
            # samples across a rollout (e.g. CPU-side ``reset`` with the
            # env-frame initial obs followed by CUDA-side ``append`` with
            # model-predicted obs migrated to the rollout device), which
            # would otherwise raise ``Expected all tensors to be on the
            # same device`` here. Align every entry onto the most recent
            # tensor's device/dtype before concatenation.
            target = buffer[-1]
            buffer = [
                b.to(dtype=target.dtype, device=target.device) for b in buffer
            ]
            x = torch.cat(buffer, dim=-1)
        elif issubclass(self.stored_type, np.ndarray):
            x = np.concatenate(buffer, axis=-1)
        elif issubclass(self.stored_type, (int, float)):
            x = np.array(self._buffer.copy())
        else:
            raise NotImplementedError(f"Type {self.stored_type=} not supported yet.")

        return x

    def _gpu_get_buffer(self) -> torch.Tensor:
        """GPU-optimized get_buffer: reorder circular storage and flatten to 1D.

        Returns a contiguous 1D tensor with features concatenated across timesteps.
        """
        if self._gpu_count < self._capacity_val:
            # Buffer not yet full — return available slots in order
            ordered = self._gpu_storage[:self._gpu_count]
        elif self._gpu_write_idx == 0:
            # Write pointer wrapped exactly — storage is already in order
            ordered = self._gpu_storage
        else:
            # Circular reorder: oldest is at _gpu_write_idx
            ordered = torch.cat([
                self._gpu_storage[self._gpu_write_idx:],
                self._gpu_storage[:self._gpu_write_idx],
            ], dim=0)
        return ordered.reshape(-1)

    @property
    def shape(self) -> Tuple[int]:
        """Get the shape of the returned array last dimension when calling the get_buffer() method:

        :return: the shape of the multistep buffer
        """
        # noinspection PyRedundantParentheses
        return (self.single_step_len * (self.multistep_length + self.multistep_extension_len),)

    @property
    def get_buffer_extension_start_idx(self) -> int:
        """
        The starting index of the extension part of the ndarray returned by `get_buffer()`.

        :return: The starting index of the buffer extension.
        """
        return self.single_step_len * self.multistep_length

    def __len__(self) -> int:
        if self._use_gpu_storage and self._gpu_storage is not None:
            return self._gpu_count
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        return self._capacity_val

    def __getitem__(self, idx):
        if self._use_gpu_storage and self._gpu_storage is not None:
            # Map logical index to physical index in circular buffer
            if self._gpu_count < self._capacity_val:
                return self._gpu_storage[idx]
            physical_idx = (self._gpu_write_idx + idx) % self._capacity_val
            return self._gpu_storage[physical_idx]
        return self._buffer[idx]

    def pad_extension(self, sample: Optional[Any] = None):
        """Pad the buffer extension by one using the provided sample or the last appended sample
        otherwise.

        :param sample: (optional) a single-step data sample
        """
        if self._use_gpu_storage and self._gpu_storage is not None:
            if sample is not None:
                self._gpu_append(sample if isinstance(sample, torch.Tensor) else torch.tensor(sample, device=self._device))
            else:
                # Append the last written sample
                last_idx = (self._gpu_write_idx - 1) % self._capacity_val
                self._gpu_append(self._gpu_storage[last_idx])
        else:
            if sample:
                self._buffer.append(sample)
            else:
                self._buffer.append(self._buffer[-1])
        self.extension_padding_counter += 1
        return None

    def __repr__(self):
        """User representation. Dynamically handle property added at run time"""
        repr_str = f"{self.__class__.__name__}("
        repr_str += f"info: {self.info}"
        for k, v in self.__dict__.items():
            if k in ("_buffer", "info", "_consol_log", "_gpu_storage"):
                pass
            else:
                repr_str += f", {k}: {v.__class__.__name__} {v}"
        repr_str += f")"
        return repr_str

    def __del__(self):
        # (CRITICAL) ToDo: validate (ref task RLRP-364)
        if self._use_gpu_storage and self._gpu_storage is not None:
            del self._gpu_storage
            self._gpu_storage = None
        elif self.stored_type and self.stored_type == torch.Tensor:
            # Explicitly release tensor memory allocation
            while len(self._buffer) > 0:
                removed_sample = self._buffer.popleft()
                del removed_sample
