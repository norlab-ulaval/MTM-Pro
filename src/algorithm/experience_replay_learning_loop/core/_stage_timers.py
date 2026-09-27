# coding=utf-8
"""
Per-stage timer helper for the ERLL outer loop.

Permanent ERLL diagnostic utility. Introduced by action ``F-E1-instr`` of
the Training Speed & Efficiency stage-1 follow-up plan
(``performance_training_speed_efficiency_stage1_followup_plan_20260421.md``).
Gated by ``cfg.diagnostics.profile_breakdown`` (legacy default OFF) and kept
as a standing profiling knob for future perf investigations.

Design goals
------------
- **Zero-cost when OFF**. When ``cfg.diagnostics.profile_breakdown`` is
  ``false`` (legacy default), every timer call must short-circuit at
  the top of the helper — no context manager body, no device query,
  no list allocation. Guarantees byte-identical logs vs. the legacy
  path.
- **One ``cuda.synchronize`` per ERLL epoch max**. On CUDA, we enqueue
  ``torch.cuda.Event(enable_timing=True)`` pairs around each stage
  without synchronising mid-epoch. The single synchronisation point
  is the call to ``event.elapsed_time(...)`` inside :meth:`report`,
  triggered once per ERLL epoch after the last stage boundary.
- **CPU / MPS fallback**. When the selected device is not CUDA, use
  :func:`time.perf_counter` (monotonic, sub-microsecond resolution on
  macOS/Linux). No CUDA import is forced.
- **Fused stages**. This first-cut wiring measures five boundaries:
  ``data``, ``train_step`` (forward + backward combined, since they
  are fused inside ``mbrl.models.ModelTrainer.train``), ``rollout``,
  ``save``, ``tb``. Splitting ``train_step`` into ``forward`` and
  ``backward`` requires a ``batch_callback`` hook inside the vendored
  ``utilities/mbrl-lib`` fork and is deferred to a follow-up refinement
  (flagged in the follow-up plan's execution summary for F-E1-instr).

The helper is **not** part of the public API of the ERLL loop — it is
an internal diagnostic tool gated entirely by the cfg flag.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

import torch


# Stage keys wired at the outer ERLL loop.
# Kept in a tuple so the ordering in the per-epoch report is stable.
ERLL_STAGE_KEYS: Tuple[str, ...] = (
    "data",
    "train_step",
    "rollout",
    "save",
    "tb",
)


class ERLLStageTimer:
    """Per-stage timer for the ERLL outer loop.

    When ``enabled`` is ``False``, every call is a zero-cost no-op and
    the helper never touches CUDA. When ``enabled`` is ``True``, each
    :meth:`stage` invocation wraps the context body with either a
    ``torch.cuda.Event`` pair (CUDA devices) or a
    :func:`time.perf_counter` pair (CPU / MPS / any non-CUDA device).

    Times are accumulated per-stage across an ERLL epoch and flushed
    by :meth:`report`, which returns an ``OrderedDict``-like mapping
    ``{stage_key: elapsed_ms}`` with zero for stages that never ran.
    Calling :meth:`report` also resets the internal accumulators so
    the timer is reusable across ERLL epochs.
    """

    def __init__(
        self,
        enabled: bool,
        device: Optional[torch.device] = None,
    ) -> None:
        self._enabled: bool = bool(enabled)
        # Record the backend choice at construction time so the cost
        # of ``device.type`` is paid once, not per-stage.
        self._use_cuda_events: bool = (
            self._enabled
            and device is not None
            and getattr(device, "type", None) == "cuda"
            and torch.cuda.is_available()
        )
        # Raw accumulators.
        #   * CUDA path: list of (start_event, end_event) pairs per stage.
        #     ``elapsed_time`` is not called until :meth:`report` to avoid
        #     mid-epoch synchronisations.
        #   * perf_counter path: accumulated seconds per stage.
        self._cuda_events: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self._perf_accum_s: Dict[str, float] = {}
        if self._enabled:
            for key in ERLL_STAGE_KEYS:
                self._cuda_events[key] = []
                self._perf_accum_s[key] = 0.0

    # ------------------------------------------------------------------
    # Public properties (useful for tests / introspection)
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def backend(self) -> str:
        """Return ``"cuda_event"``, ``"perf_counter"``, or ``"off"``."""
        if not self._enabled:
            return "off"
        return "cuda_event" if self._use_cuda_events else "perf_counter"

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    @contextmanager
    def stage(self, key: str) -> Iterator[None]:
        """Time the body of the ``with`` block under ``key``.

        Unknown keys are tolerated (the helper simply ignores them at
        :meth:`report` time) to avoid coupling the instrumentation too
        tightly to the current list of stages.
        """
        if not self._enabled:
            # Zero-cost OFF path: no timing, no allocation.
            yield
            return

        if self._use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                # Events are queued; ``elapsed_time`` is deferred to
                # :meth:`report` to keep the hot path sync-free.
                self._cuda_events.setdefault(key, []).append((start, end))
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self._perf_accum_s[key] = (
                    self._perf_accum_s.get(key, 0.0) + (time.perf_counter() - t0)
                )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def report(self) -> Dict[str, float]:
        """Return ``{stage_key: elapsed_ms}`` and reset accumulators.

        When ``enabled`` is ``False``, returns an empty dict — callers
        are expected to no-op on empty reports (see the ERLL loop
        integration site).
        """
        if not self._enabled:
            return {}

        out: Dict[str, float] = {}
        if self._use_cuda_events:
            # Single synchronise for the whole report: ensure all
            # enqueued events on the current stream have completed
            # before reading their timings.
            torch.cuda.synchronize()
            for key in ERLL_STAGE_KEYS:
                pairs = self._cuda_events.get(key, [])
                total_ms = 0.0
                for start, end in pairs:
                    total_ms += float(start.elapsed_time(end))
                out[key] = total_ms
            # Reset.
            for key in ERLL_STAGE_KEYS:
                self._cuda_events[key] = []
        else:
            for key in ERLL_STAGE_KEYS:
                out[key] = self._perf_accum_s.get(key, 0.0) * 1000.0
            for key in ERLL_STAGE_KEYS:
                self._perf_accum_s[key] = 0.0
        return out

    # ------------------------------------------------------------------
    # Formatting helper (kept out of :meth:`report` so callers can
    # choose their logging channel without paying the string cost when
    # they don't need it).
    # ------------------------------------------------------------------
    @staticmethod
    def format_report(report: Dict[str, float]) -> str:
        """Render a :meth:`report` dict as a compact one-line string."""
        if not report:
            return ""
        parts = [f"{k}={report.get(k, 0.0):.2f}ms" for k in ERLL_STAGE_KEYS]
        return "[profile_breakdown] " + " ".join(parts)
