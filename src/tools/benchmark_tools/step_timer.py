# coding=utf-8
"""Per-step inference timer for the test-time rollout deploy path.

Permanent benchmarking utility. Introduced by action ``A1`` of the RLRC
test-time rollout deployer benchmarking `.junie` plan
(``perf_RLRP-785_testtime_rollout_benchmarking_plan_20260903.md``, rev. 5).
Gated by ``cfg.deploy.benchmark.enable_step_timing`` (default OFF), so the
legacy deploy path stays byte-identical.

The timer is **multi-level** (one independent pre-allocated pool per enabled
:class:`BenchmarkLevel`, nesting freely, plan section 3.5), **dual-pass**
(``device_time`` = CUDA events + one deferred sync -> throughput; ``deployable_latency``
= ``perf_counter`` + one per-step ``torch.cuda.synchronize()`` -> the headline latency,
plan section 3.7 / R1), **taint-aware** (``step(level, record=False)`` executes the region
but stores no sample, plan R4) and computes **per-step paired deltas** for the cost
breakdown (plan R5). Its :meth:`report` returns a :class:`BenchmarkMetricSet` (never a bare
dict, never a ``MIXED`` population).

Design contract (plan section 3 gates G2/G4/G11/G12):
  * ``enabled is False`` -> every call short-circuits; no allocation, no device query.
  * CUDA device-time -> ``capacity`` pre-allocated ``torch.cuda.Event(enable_timing=True)``
    pairs per level; ``elapsed_time`` is never called before :meth:`report`, which performs the
    single ``torch.cuda.synchronize()``.
  * non-CUDA / deployable-latency -> ``time.perf_counter`` into pre-allocated ``float64`` arrays.
  * No file/console I/O of any kind inside :meth:`step`.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch

from tools.benchmark_tools.benchmark_metric import (
    BenchmarkLevel,
    BenchmarkMetricSet,
    FeedbackRegime,
    LevelDeltaMetric,
    StepBenchmarkMetric,
    TimingPass,
)

# Outer-to-inner nesting rank: control_loop_step contains deployer_step contains model_call.
_LEVEL_RANK: Dict[BenchmarkLevel, int] = {
    BenchmarkLevel.MODEL_CALL: 0,
    BenchmarkLevel.DEPLOYER_STEP: 1,
    BenchmarkLevel.CONTROL_LOOP_STEP: 2,
}

# The two nesting pairs whose per-step deltas form the cost breakdown (plan section 3.5 / R5).
_DELTA_PAIRS = (
    (BenchmarkLevel.DEPLOYER_STEP, BenchmarkLevel.MODEL_CALL),
    (BenchmarkLevel.CONTROL_LOOP_STEP, BenchmarkLevel.DEPLOYER_STEP),
)


class _LevelPool:
    """Pre-allocated per-level sample storage (events on CUDA device-time, floats otherwise)."""

    def __init__(self, capacity: int, use_cuda_events: bool) -> None:
        self.capacity = int(capacity)
        self.use_cuda_events = use_cuda_events
        self.n = 0
        self.n_tainted = 0
        self.loop_ids = np.full(self.capacity, -1, dtype=np.int64)
        if use_cuda_events:
            self.starts: List[torch.cuda.Event] = [
                torch.cuda.Event(enable_timing=True) for _ in range(self.capacity)
            ]
            self.ends: List[torch.cuda.Event] = [
                torch.cuda.Event(enable_timing=True) for _ in range(self.capacity)
            ]
            self.samples_s = None
        else:
            self.starts = None
            self.ends = None
            self.samples_s = np.zeros(self.capacity, dtype=np.float64)

    def release(self) -> None:
        self.starts = None
        self.ends = None
        self.samples_s = None


class DeployStepTimer:
    """Time N individual control-loop steps without any mid-loop synchronisation.

    :param enabled: master switch; ``False`` -> a fully zero-cost OFF path.
    :param device: the torch device the work runs on (used to pick the CUDA path).
    :param capacity: number of steps to pre-allocate per level (``rollout_len``).
    :param enabled_levels: which :class:`BenchmarkLevel`s to time; a level absent from this
        set short-circuits exactly like the global OFF path.
    :param timing_pass: :class:`TimingPass` -- ``DEVICE_TIME`` (CUDA events, deferred sync) or
        ``DEPLOYABLE_LATENCY`` (``perf_counter`` + one per-step ``synchronize()``).
    :param keep_raw_samples: default for :meth:`report` -- carry the per-step population into the
        emitted metrics (action ``RLRP-803-2``). ``report`` can override it per call.
    :param max_raw_samples: default cap for that population (uniform decimation above it).
    """

    def __init__(
        self,
        enabled: bool,
        device: Optional[torch.device],
        capacity: int,
        enabled_levels: Sequence[BenchmarkLevel] = (BenchmarkLevel.CONTROL_LOOP_STEP,),
        timing_pass: TimingPass = TimingPass.DEVICE_TIME,
        keep_raw_samples: bool = False,
        max_raw_samples: int = 10_000,
    ) -> None:
        self._enabled = bool(enabled)
        self._timing_pass = TimingPass(timing_pass)
        self._keep_raw_samples = bool(keep_raw_samples)
        self._max_raw_samples = int(max_raw_samples)
        self._loop_id = -1
        self._pools: Dict[BenchmarkLevel, _LevelPool] = {}
        self._stream = None

        if not self._enabled:
            self._is_cuda = False
            self._use_cuda_events = False
            self._enabled_levels: tuple = ()
            self._outermost = None
            return

        self._is_cuda = bool(
            device is not None
            and getattr(device, "type", None) == "cuda"
            and torch.cuda.is_available()
        )
        # CUDA events are the device-time instrument; the deployable-latency pass always uses
        # ``perf_counter`` (+ a per-step sync on CUDA), even on a CUDA device.
        self._use_cuda_events = self._is_cuda and self._timing_pass is TimingPass.DEVICE_TIME
        self._deployable_sync = (
            self._is_cuda and self._timing_pass is TimingPass.DEPLOYABLE_LATENCY
        )

        self._enabled_levels = tuple(BenchmarkLevel(lvl) for lvl in enabled_levels)
        if self._enabled_levels:
            self._outermost = max(self._enabled_levels, key=lambda lvl: _LEVEL_RANK[lvl])
        else:
            self._outermost = None

        capacity = int(capacity)
        for lvl in self._enabled_levels:
            self._pools[lvl] = _LevelPool(capacity, self._use_cuda_events)

        if self._is_cuda:
            self._stream = torch.cuda.current_stream()

    # ---- introspection ---------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def backend(self) -> str:
        if not self._enabled:
            return "off"
        if self._use_cuda_events:
            return "cuda_event"
        if self._deployable_sync:
            return "perf_counter_sync"
        return "perf_counter"

    @property
    def enabled_levels(self) -> tuple:
        return self._enabled_levels

    # ---- the hot path ----------------------------------------------------------------------
    @contextmanager
    def step(self, level: BenchmarkLevel = BenchmarkLevel.CONTROL_LOOP_STEP, *, record: bool = True) -> Iterator[None]:
        """Time one nested region of one control-loop step.

        With ``record=False`` (plan R4) the region still executes -- bit-exactness is never
        affected -- but no sample is stored and ``n_tainted`` is incremented. Levels nest freely;
        pass the outermost enabled level once per loop step to advance the internal step id used
        for per-step delta pairing.
        """
        pool = self._pools.get(level) if self._enabled else None
        if pool is None:
            # Global OFF path or a level that is not enabled: zero cost, no device query.
            yield
            return

        is_outer = level is self._outermost
        if is_outer:
            self._loop_id += 1
        loop_id = self._loop_id

        if (not record) or pool.n >= pool.capacity:
            try:
                yield
            finally:
                if self._deployable_sync and is_outer:
                    # A deployed node pays this stall even on a tainted/overflow step; keep the
                    # pipeline state identical between recorded and non-recorded steps.
                    torch.cuda.synchronize()
                if not record:
                    pool.n_tainted += 1
            return

        i = pool.n
        if self._use_cuda_events:
            pool.starts[i].record(self._stream)
            try:
                yield
            finally:
                pool.ends[i].record(self._stream)
                pool.loop_ids[i] = loop_id
                pool.n = i + 1
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                if self._deployable_sync and is_outer:
                    torch.cuda.synchronize()
                pool.samples_s[i] = time.perf_counter() - t0
                pool.loop_ids[i] = loop_id
                pool.n = i + 1

    # ---- reduction -------------------------------------------------------------------------
    def _level_samples_s(self, pool: _LevelPool):
        """Return ``(values_s, loop_ids)`` for one level. Single CUDA sync on the event path."""
        if pool.n == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
        loop_ids = pool.loop_ids[: pool.n].copy()
        if pool.use_cuda_events:
            torch.cuda.synchronize()
            values = np.fromiter(
                (pool.starts[i].elapsed_time(pool.ends[i]) / 1000.0 for i in range(pool.n)),
                dtype=np.float64,
                count=pool.n,
            )
            return values, loop_ids
        return pool.samples_s[: pool.n].copy(), loop_ids

    def _assert_stream(self) -> None:
        if self._is_cuda and self._stream is not None:
            current = torch.cuda.current_stream()
            if current != self._stream:
                raise RuntimeError(
                    "DeployStepTimer refuses to report: events were recorded on stream "
                    f"{self._stream} but the current stream is {current}. Timing across streams "
                    "produces meaningless deltas (plan R11, deployer.py @use_cuda_stream guard)."
                )

    def report(
        self,
        *,
        feedback_regime: FeedbackRegime,
        regime_split_index: Optional[int] = None,
        discard_warmup_steps: int = 0,
        provenance: Optional[Dict[str, Any]] = None,
        keep_raw_samples: Optional[bool] = None,
        max_raw_samples: Optional[int] = None,
    ) -> BenchmarkMetricSet:
        """Reduce every enabled level into a :class:`BenchmarkMetricSet` (called ONCE).

        :param feedback_regime: regime of the pre-boundary population (and of the whole
            population when ``regime_split_index`` is ``None``).
        :param regime_split_index: the rollout's ``ground_truth_feed_warmup_steps`` boundary;
            samples with ``loop_id < k`` are labelled ``ESTIMATOR`` and ``loop_id >= k``
            ``AUTOREGRESSIVE`` (plan R3). Never produces a ``MIXED`` population.
        :param discard_warmup_steps: TIMER warm-up (cache/autotune) -- a different thing from the
            regime boundary; drops the first N recorded samples by position.
        :param keep_raw_samples: action ``RLRP-803-2`` -- carry each population's per-step samples
            into its :class:`StepBenchmarkMetric`, so the plotter draws TRUE quartiles instead of
            the normal approximation. ``None`` inherits the construction-time default; an explicit
            value wins (that is how the overhead-CALIBRATION pass opts out, see
            ``model_inference_benchmark._calibrate_timer_overhead``).
        :param max_raw_samples: ``None`` inherits the construction-time cap.
        """
        result = BenchmarkMetricSet()
        if not self._enabled or not self._pools:
            return result

        keep_raw = (
            self._keep_raw_samples if keep_raw_samples is None else bool(keep_raw_samples)
        )
        max_raw = (
            self._max_raw_samples if max_raw_samples is None else int(max_raw_samples)
        )

        self._assert_stream()
        provenance = dict(provenance or {})
        device_type = "cuda" if self._is_cuda else "cpu"
        backend = self.backend

        # Cache each level's post-warm-up samples keyed by loop_id, for delta pairing.
        per_level: Dict[BenchmarkLevel, Dict[int, float]] = {}

        for lvl, pool in self._pools.items():
            values, loop_ids = self._level_samples_s(pool)
            if discard_warmup_steps > 0:
                values = values[discard_warmup_steps:]
                loop_ids = loop_ids[discard_warmup_steps:]
            per_level[lvl] = {int(lid): float(v) for lid, v in zip(loop_ids, values)}

            if values.size == 0:
                continue

            populations = self._split_populations(
                values, loop_ids, feedback_regime, regime_split_index
            )
            for regime, pop_values in populations:
                if pop_values.size == 0:
                    continue
                metric = StepBenchmarkMetric.from_samples(
                    pop_values,
                    level=lvl,
                    feedback_regime=regime,
                    timing_pass=self._timing_pass,
                    device_type=device_type,
                    timing_backend=backend,
                    n_warmup_discarded=int(discard_warmup_steps),
                    n_tainted_excluded=int(pool.n_tainted),
                    regime_split_index=regime_split_index,
                    platform_info=dict(provenance),
                    keep_raw_samples=keep_raw,
                    max_raw_samples=max_raw,
                )
                result.add(metric)

        # Action ``RLRP-803-7.2``: the regime is THREADED into the deltas here (it is already
        # known by this method), never re-derived downstream -- a regime-less delta is what
        # made the A4 cost-breakdown figure regime-agnostic.
        result.level_deltas = self._compute_level_deltas(
            per_level,
            feedback_regime=feedback_regime,
            regime_split_index=regime_split_index,
        )
        return result

    @staticmethod
    def _split_populations(
        values: np.ndarray,
        loop_ids: np.ndarray,
        feedback_regime: FeedbackRegime,
        regime_split_index: Optional[int],
    ):
        """Split a level's samples into labelled populations at the regime boundary (R3)."""
        if regime_split_index is None:
            return [(feedback_regime, values)]
        pre_mask = loop_ids < int(regime_split_index)
        post_mask = ~pre_mask
        return [
            (FeedbackRegime.ESTIMATOR, values[pre_mask]),
            (FeedbackRegime.AUTOREGRESSIVE, values[post_mask]),
        ]

    def _compute_level_deltas(
        self,
        per_level: Dict[BenchmarkLevel, Dict[int, float]],
        *,
        feedback_regime: FeedbackRegime = FeedbackRegime.ESTIMATOR,
        regime_split_index: Optional[int] = None,
    ) -> List[LevelDeltaMetric]:
        """Per-step paired deltas (R5): match outer/inner by loop_id, then reduce.

        Made regime-aware by action ``RLRP-803-7.2`` of the RLRC test-time rollout deployer
        benchmarking `.junie` plan
        (``perf_RLRP-785_testtime_rollout_benchmarking_plan_20260903.md``). Permanent. The
        pairing is split at ``regime_split_index`` exactly like the level populations are
        (:meth:`_split_populations`), so each emitted :class:`LevelDeltaMetric` describes ONE
        feedback regime and lines up with that regime's latency numbers.

        :param per_level: ``{level: {loop_id: seconds}}`` post-warm-up samples.
        :param feedback_regime: label of the whole pairing when there is no split boundary
            (and of the pre-boundary population when there is one).
        :param regime_split_index: the ``ground_truth_feed_warmup_steps`` boundary, or ``None``.
        :return: one record per ``(level pair, regime)`` actually paired.
        """
        deltas: List[LevelDeltaMetric] = []
        for outer, inner in _DELTA_PAIRS:
            outer_map = per_level.get(outer)
            inner_map = per_level.get(inner)
            if not outer_map or not inner_map:
                continue
            shared = sorted(set(outer_map) & set(inner_map))
            if not shared:
                continue
            for regime, loop_ids in self._split_loop_ids(
                shared, feedback_regime, regime_split_index
            ):
                if not loop_ids:
                    continue
                paired = np.fromiter(
                    (outer_map[lid] - inner_map[lid] for lid in loop_ids),
                    dtype=np.float64,
                    count=len(loop_ids),
                )
                n_negative = int((paired < 0.0).sum())
                deltas.append(
                    LevelDeltaMetric(
                        outer_level=outer,
                        inner_level=inner,
                        n_pairs=int(paired.size),
                        delta_median_ms=float(np.median(paired) * 1e3),
                        delta_p95_ms=float(np.percentile(paired, 95) * 1e3),
                        overhead_correction_ms=0.0,
                        n_negative_pairs=n_negative,
                        feedback_regime=regime,
                    )
                )
        return deltas

    @staticmethod
    def _split_loop_ids(
        loop_ids: List[int],
        feedback_regime: FeedbackRegime,
        regime_split_index: Optional[int],
    ) -> List[tuple]:
        """Label paired ``loop_id``s by regime, mirroring :meth:`_split_populations` (R3)."""
        if regime_split_index is None:
            return [(feedback_regime, list(loop_ids))]
        boundary = int(regime_split_index)
        return [
            (FeedbackRegime.ESTIMATOR, [lid for lid in loop_ids if lid < boundary]),
            (FeedbackRegime.AUTOREGRESSIVE, [lid for lid in loop_ids if lid >= boundary]),
        ]

    # ---- lifecycle -------------------------------------------------------------------------
    def reset(self) -> None:
        """Rewind counters so the pools can be reused for another rollout."""
        self._loop_id = -1
        for pool in self._pools.values():
            pool.n = 0
            pool.n_tainted = 0
            pool.loop_ids[:] = -1

    def close(self) -> None:
        """Release the pre-allocated pools deterministically (plan R11 -- no per-rollout leak)."""
        for pool in self._pools.values():
            pool.release()
        self._pools = {}
        self._stream = None
