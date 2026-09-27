# coding=utf-8
"""RLRP-786 — manual ``torch.cuda.CUDAGraph`` capture of a whole training step.

Introduced by the RLRC CUDA-graph captured training step `.junie` plan
(``perf_RLRP-786_cuda_graph_train_step_plan_20260919.md``, Step 3, Key Decisions 1–3).

Why
    The RLRP-830 A100 profile of the Distributional-MTM-Pro MS+CP training step is
    **launch-bound**: ≈ 6 800 kernels/step, GPU kernel-busy 12 %, the host dispatcher (Python +
    ATen + ``cudaLaunchKernel``) idles the device 88 % of the wall-clock. Removing kernels one by
    one (RLRP-830) gave ×1.04. The lever that scales with *dispatch* cost is to stop paying it per
    op: record the static-shape step ``loss -> backward -> optimizer.step()`` ONCE in a CUDA graph
    and ``replay()`` it every step (one launch for the whole step).

What is captured / what stays on the host
    Captured region (Key Decision 3): ``loss_fn(static_batch)`` -> ``loss.backward()`` ->
    ``optimizer.step()``. Outside: the H2D ``copy_`` of the incoming batch into the static device
    buffers, the ``before_replay`` hook (e.g. refilling the teacher-forcing coin mask from the CPU
    scheduler, stepping that scheduler), the NaN stop check, the meta flush and the logging — all
    executed by the caller around :meth:`replay`.

Numerics
    Graph replay executes the very same kernels on the very same memory the eager step used at
    capture time, so the replayed step is expected ``torch.equal`` to eager (RLRP-786 FR3 — the
    operator's *bit-identical or better* gate); this is verified by
    ``tests/tests_tools/tests_torch_tools/test_cuda_graph_train_step.py`` (tiny MLP) and
    ``tests/tests_tools/tests_multistep_tools/tests_models/test_mtm_pro_cuda_graph_step_parity_rlrp786.py``
    (the sampling-free DMtm-Pro paper path). Warm-up steps are REAL eager training steps on the
    first ``warmup_iters`` batches (identical maths to an eager run); the capture pass records
    without executing, then the capture batch is replayed once, so every batch trains exactly once.

Preconditions (asserted at construction / capture)
    * the device is CUDA;
    * every ``optimizer.param_groups`` entry has ``capturable=True`` (the Adam ``step`` counter must
      live on the device — see ``optimizer_instantiation._resolve_adam_fused_kwargs``);
    * the learning rate is a Python number baked into the recorded kernels: when a scheduler
      changes it, :meth:`replay` re-records the graph once (``recapture_on_lr_change``, default) --
      a device-tensor lr is also accepted (read live by the fused/capturable Adam) but is NOT
      bit-identical to the eager float-lr update, so the production path keeps the float;
    * the loss function has no host<->device synchronisation and no CPU-side data-dependent control
      flow inside the captured region (a ``.item()`` / ``.cpu()`` / Python ``if`` on a tensor raises
      ``RuntimeError`` at capture time — that is the wanted, explicit failure mode; for the MTM-Pro
      family those live behind ``performance_mode: fast`` + ``cuda_graph_capture_ready``).

Usage
    >>> step = CudaGraphTrainStep(loss_fn=wrapper.loss, optimizer=opt, device=torch.device("cuda"))
    >>> for batch in loader:
    ...     if step.is_capturable(batch):
    ...         loss, meta = step.step(batch)      # eager warm-up -> capture -> replay
    ...     else:
    ...         loss, meta = eager_step(batch)     # ragged tail batch: eager fallback
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from mbrl.types import TransitionBatch

__all__ = [
    "CudaGraphTrainStep",
    "BatchSpec",
    "batch_spec_of",
    "Batch",
    "is_recording_cuda_graph",
]


def is_recording_cuda_graph() -> bool:
    """``True`` while the current CUDA stream is being CAPTURED into a ``torch.cuda.CUDAGraph``.

    The recording pass of :meth:`CudaGraphTrainStep.capture` runs the Python body of the loss
    once WITHOUT executing it; the host-side per-step bookkeeping of that body (scheduler
    ``step`` calls, training-step counters) must therefore be SKIPPED during that pass -- it is
    replayed by the caller's ``before_replay`` hook (e.g. ``advance_host_step_state``) exactly
    once per trained batch. Shared by the MTM-Pro and the AR MS2SS families; not a dev/debug gate
    (those live behind ``performance_mode``): this is the capture protocol itself.
    """
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


_FIELDS = ("obs", "act", "next_obs", "rewards", "terminateds", "truncateds")

#: What the step consumes: a :class:`TransitionBatch` (the loader contract) OR a tuple / list of
#: tensors (an already pre-processed ``(model_in, target, ...)`` -- the RLRP-786 production /
#: benchmark path keeps the wrapper's ``_process_batch`` normalisation OUTSIDE the graph because
#: the mbrl-lib normalizer's non-finite guard is a host sync).
Batch = Union[TransitionBatch, Tuple[Any, ...], List[Any]]


@dataclass(frozen=True)
class BatchSpec:
    """Shapes / dtypes of the non-``None`` leaves of a batch (the static contract)."""

    fields: Tuple[Tuple[str, Tuple[int, ...], torch.dtype], ...]

    @property
    def batch_size(self) -> int:
        return self.fields[0][1][0] if self.fields else 0


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _leaves(batch: Batch) -> List[Tuple[str, Any]]:
    """``[(key, value)]`` of a batch -- ``TransitionBatch`` fields or positional tuple entries."""
    if isinstance(batch, TransitionBatch):
        return [(name, getattr(batch, name)) for name in _FIELDS]
    if isinstance(batch, (tuple, list)):
        return [(str(i), v) for i, v in enumerate(batch)]
    raise TypeError(f"CudaGraphTrainStep: unsupported batch type {type(batch).__name__}")


def _rebuild(batch: Batch, values: List[Any]) -> Batch:
    if isinstance(batch, TransitionBatch):
        return TransitionBatch(**dict(zip(_FIELDS, values)))
    return tuple(values)


def batch_spec_of(batch: Batch) -> BatchSpec:
    """Build the :class:`BatchSpec` of a batch (``None`` leaves are skipped, numpy is accepted)."""
    fields = []
    for name, value in _leaves(batch):
        if value is None:
            continue
        t = _as_tensor(value)
        fields.append((name, tuple(t.shape), t.dtype))
    return BatchSpec(tuple(fields))


class CudaGraphTrainStep:
    """Static-shape captured ``loss -> backward -> optimizer.step()`` (see the module docstring).

    :param loss_fn: ``loss_fn(batch) -> (loss, meta)``; ``batch`` is a :class:`TransitionBatch` or a
        tuple of tensors (see :data:`Batch`) whose tensors live on ``device`` (the static buffers).
        ``loss`` must be a 0-dim tensor requiring grad.
    :param optimizer: the optimizer whose ``step()`` is recorded; every param group must be ``capturable``.
    :param device: the CUDA device the step runs on.
    :param before_replay: host-side hook called right before each ``graph.replay()`` ONLY (neither
        during the eager warm-up steps, whose Python body performs its own per-step host bookkeeping,
        nor during the recording pass) — the place for the CPU-side per-step state the captured body
        can no longer advance itself (training-step counter, scheduler ``step``s, teacher-forcing
        coin mask refill). Defaults to a no-op.
    :param warmup_iters: number of REAL eager steps run on a side stream before the capture (the
        official ``torch.cuda.graph`` recipe: lets the allocator / cuDNN autotuner settle).
    """

    def __init__(
        self,
        loss_fn: Callable[[Batch], Tuple[torch.Tensor, Dict[str, Any]]],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        before_replay: Optional[Callable[[], None]] = None,
        warmup_iters: int = 3,
        recapture_on_lr_change: bool = True,
    ) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"CudaGraphTrainStep needs a CUDA device, got {device!r}")
        if not torch.cuda.is_available():
            raise RuntimeError("CudaGraphTrainStep: torch.cuda.is_available() is False")
        for i, group in enumerate(optimizer.param_groups):
            if not group.get("capturable", False):
                raise ValueError(
                    f"CudaGraphTrainStep: optimizer param group {i} is not capturable=True "
                    f"({type(optimizer).__name__}); the step counter must live on the device "
                    "(request `optimizer.adam_capturable: true`, see optimizer_instantiation.py)"
                )
            lr = group.get("lr")
            if isinstance(lr, torch.Tensor) and lr.device.type != device.type:
                # A DEVICE tensor lr is the supported way to schedule the LR under replay: the
                # recorded fused/capturable Adam kernels read it from device memory, and
                # ``torch.optim.lr_scheduler`` updates it in place (``fill_``). A Python float is
                # baked into the recorded launch args (drift is refused in :meth:`replay`).
                raise ValueError(
                    f"CudaGraphTrainStep: tensor `lr` of param group {i} must live on {device} "
                    f"(got {lr.device}) for the captured optimizer step to read it"
                )
        if warmup_iters < 1:
            # Lazy first-use allocations (Adam moments / ``step``, model-side static buffers, cuBLAS
            # handles) MUST happen in an eager step BEFORE the recording: a ``torch.zeros`` captured
            # inside the graph is re-executed at every replay and silently resets that state
            # (observed on the Orin: Adam moments re-zeroed each replay when no warm-up ran).
            raise ValueError(f"warmup_iters must be >= 1 (official capture recipe), got {warmup_iters}")
        # ``recapture_on_lr_change``: a Python-float ``lr`` is baked into the recorded fused Adam
        # launch (host-side ``lr / bias_correction`` in double); when a scheduler / the ERLL Law-U
        # pass restart assigns a new float, :meth:`replay` RE-RECORDS the graph (one recording pass,
        # no training step lost) instead of refusing. NOT a device-tensor lr: the fused kernel then
        # computes the step size on device in fp32 and the parameter update differs from the eager
        # float-lr path at the ulp level (measured on the Orin: params differ after the FIRST step,
        # grads and moments equal), which fails the ``torch.equal`` gate. LR schedules step once per
        # epoch, so a re-capture per epoch is negligible.
        self._recapture_on_lr_change = bool(recapture_on_lr_change)
        self._recaptures = 0
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._device = device
        self._before_replay = before_replay if before_replay is not None else (lambda: None)
        self._warmup_iters = int(warmup_iters)
        self._warmups_done = 0
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._spec: Optional[BatchSpec] = None
        self._static_batch: Optional[Batch] = None
        self._static_loss: Optional[torch.Tensor] = None
        self._static_meta: Dict[str, Any] = {}
        self._captured_lrs: Tuple[Optional[float], ...] = ()
        self._static_grads: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        self._replays = 0

    # ---- introspection -------------------------------------------------------------------------
    @property
    def captured(self) -> bool:
        return self._graph is not None

    @property
    def spec(self) -> Optional[BatchSpec]:
        """The static batch contract (``None`` until the first :meth:`step` / :meth:`warmup_and_capture`)."""
        return self._spec

    @property
    def warmups_done(self) -> int:
        return self._warmups_done

    @property
    def replays(self) -> int:
        return self._replays

    @property
    def recaptures(self) -> int:
        """Number of re-recordings triggered by a Python-float ``lr`` change after capture."""
        return self._recaptures

    @property
    def device(self) -> torch.device:
        return self._device

    def is_capturable(self, batch: Batch) -> bool:
        """``True`` when ``batch`` matches the static contract (shapes / dtypes / ``None``-ness).

        Before the first step every batch is capturable (it DEFINES the contract). A ragged tail
        batch (``drop_last=False``) or a validation batch of another size returns ``False``: the
        caller must run it eagerly (no re-capture).
        """
        if self._spec is None:
            return True
        return batch_spec_of(batch) == self._spec

    # ---- static buffers ------------------------------------------------------------------------
    def _allocate_static_batch(self, batch: Batch) -> None:
        self._spec = batch_spec_of(batch)
        values = []
        for _name, value in _leaves(batch):
            if value is None:
                values.append(None)
                continue
            values.append(_as_tensor(value).detach().to(self._device, copy=True).contiguous())
        self._static_batch = _rebuild(batch, values)

    def _copy_into_static(self, batch: Batch) -> None:
        assert self._static_batch is not None
        for (_name, dst), (_n2, src) in zip(_leaves(self._static_batch), _leaves(batch)):
            if dst is None:
                continue
            src_t = _as_tensor(src)
            if src_t.data_ptr() == dst.data_ptr() and src_t.device == dst.device:
                continue  # the caller handed us our own static buffer
            dst.copy_(src_t, non_blocking=True)

    # ---- the three phases ----------------------------------------------------------------------
    @staticmethod
    def _detached(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Shallow copy of ``meta`` with every tensor value detached (see :meth:`_eager_step_on_static`)."""
        if not meta:
            return {}
        return {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in meta.items()}

    def _eager_step_on_static(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """One REAL eager training step on the static buffers (warm-up).

        Returns the loss / meta DETACHED: the autograd graph of this step lives on the warm-up side
        stream and is dropped here. A caller keeping the attached ``loss`` alive across the next
        phase (recording on the capture stream) would keep the parameters' ``AccumulateGrad`` nodes
        of THIS stream alive -> ``UserWarning: The AccumulateGrad node's stream does not match ...``
        from the next backward (observed in the Narval parity run) and, on the default stream, a
        potential capture break.
        """
        self._optimizer.zero_grad(set_to_none=True)
        loss, meta = self._loss_fn(self._static_batch)
        loss.backward()
        self._optimizer.step()
        out = loss.detach()
        del loss  # release the autograd graph before leaving the side stream
        return out, self._detached(meta)

    def warmup(self, batch: Batch) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """One real eager step on a side stream (counts towards ``warmup_iters``)."""
        if self.captured:
            raise RuntimeError("warmup() after capture: use replay()")
        if self._static_batch is None:
            self._allocate_static_batch(batch)
        elif not self.is_capturable(batch):
            raise ValueError("warmup(): batch does not match the static contract")
        else:
            self._copy_into_static(batch)
        # NO ``before_replay`` here: the eager body runs and advances its own host-side state.
        side = torch.cuda.Stream(device=self._device)
        side.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(side):
            loss, meta = self._eager_step_on_static()
        torch.cuda.current_stream(self._device).wait_stream(side)
        self._warmups_done += 1
        return loss, meta

    def capture(self, batch: Batch) -> None:
        """Record the step on ``batch`` WITHOUT executing it (call :meth:`replay` to train on it).

        ``before_replay`` is NOT invoked here (recording does not consume the host-side per-step
        state); the :meth:`replay` that follows invokes it exactly once for the capture batch. The
        recorded Python body must itself skip its host bookkeeping while the stream is capturing
        (``torch.cuda.is_current_stream_capturing()``), otherwise that batch is counted twice.
        """
        if self.captured:
            raise RuntimeError("capture() called twice")
        if self._warmups_done < 1:
            raise RuntimeError(
                "capture() before any warm-up step: run warmup() at least once so every lazily "
                "allocated state (optimizer moments, model static buffers) exists OUTSIDE the graph"
            )
        if not self.is_capturable(batch):
            raise ValueError("capture(): batch does not match the static contract")
        self._copy_into_static(batch)
        # Python-float lrs are frozen into the recording (drift refused at replay); device-tensor
        # lrs are read by the kernels at every replay -> not tracked (``None``).
        self._captured_lrs = tuple(
            None if isinstance(g["lr"], torch.Tensor) else float(g["lr"])
            for g in self._optimizer.param_groups
        )
        # The official whole-network recipe: grads set to None BEFORE capture so the captured
        # backward allocates + writes them from the graph's private pool; replays refill them in
        # place (no zero_grad between replays).
        self._optimizer.zero_grad(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_loss, static_meta = self._loss_fn(self._static_batch)
            static_loss.backward()
            self._optimizer.step()
        self._graph = graph
        # Keep the static OUTPUT buffers only (detached): the recorded autograd graph must not stay
        # alive after the recording -- a later re-recording (lr drift) or an eager fallback step runs
        # on another stream and would otherwise hit the ``AccumulateGrad`` stream-mismatch warning.
        self._static_loss = static_loss.detach()
        self._static_meta = self._detached(static_meta)
        del static_loss, static_meta
        # The recorded backward writes the grads into graph-owned buffers; keep (param, buffer) so a
        # caller's ``zero_grad(set_to_none=True)`` between replays (Lightning epoch-end hooks) can be
        # undone by :meth:`replay` -- gradient monitoring keeps seeing the live values.
        self._static_grads = [
            (p, p.grad) for group in self._optimizer.param_groups for p in group["params"]
            if p.grad is not None
        ]

    def warmup_and_capture(self, batch: Batch) -> None:
        """Convenience: ``warmup_iters`` real eager steps on ``batch`` then :meth:`capture` (benchmark use;
        production callers feed distinct batches through :meth:`step` instead)."""
        for _ in range(self._warmup_iters):
            self.warmup(batch)
        self.capture(batch)

    def replay(self, batch: Batch) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """H2D copy of ``batch`` into the static buffers -> ``before_replay()`` -> ``graph.replay()``.

        Returns the STATIC loss tensor (overwritten by the next replay: read / clone it before) and a
        shallow copy of the meta dict recorded at capture (device tensors in it are refreshed by the
        replay; Python numbers are those of the capture step).
        """
        if not self.captured:
            raise RuntimeError("replay() before capture()")
        if not self.is_capturable(batch):
            raise ValueError("replay(): batch does not match the static contract (run it eagerly)")
        drifted = self._lr_drift()
        if drifted:
            if not self._recapture_on_lr_change:
                i, lr0, lr = drifted
                raise RuntimeError(
                    f"CudaGraphTrainStep: param group {i} lr changed ({lr0} -> {lr}) after capture; "
                    "a Python-float lr is baked into the recorded step (recapture_on_lr_change=True "
                    "re-records it, or use a device tensor lr)"
                )
            self._graph = None
            self._static_grads = []
            self._static_loss = None
            self._static_meta = {}
            self._recaptures += 1
            self.capture(batch)
        self._copy_into_static(batch)
        self._before_replay()
        self._graph.replay()
        self._replays += 1
        for p, g in self._static_grads:
            if p.grad is not g:  # set_to_none / an eager fallback step re-pointed it
                p.grad = g
        return self._static_loss, dict(self._static_meta)

    def _lr_drift(self) -> Optional[Tuple[int, float, Any]]:
        """``(group index, captured lr, current lr)`` of the first param group whose Python-float lr
        moved since the recording (device-tensor lrs are read live and never drift); else ``None``."""
        for i, group in enumerate(self._optimizer.param_groups):
            lr0 = self._captured_lrs[i] if i < len(self._captured_lrs) else None
            lr = group["lr"]
            if lr0 is None:
                continue
            if isinstance(lr, torch.Tensor) or float(lr) != lr0:
                return i, lr0, lr
        return None

    def step(self, batch: Batch) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Train on ``batch`` exactly once: eager warm-up for the first ``warmup_iters`` batches, then
        capture + replay, then replay. ``batch`` must be capturable (check first)."""
        if self.captured:
            return self.replay(batch)
        if self._warmups_done < self._warmup_iters:
            return self.warmup(batch)
        self.capture(batch)
        return self.replay(batch)

    def reset(self) -> None:
        """Drop the graph and the static buffers (a new contract can be captured)."""
        self._graph = None
        self._static_grads = []
        self._spec = None
        self._static_batch = None
        self._static_loss = None
        self._static_meta = {}
        self._captured_lrs = ()
        self._warmups_done = 0
