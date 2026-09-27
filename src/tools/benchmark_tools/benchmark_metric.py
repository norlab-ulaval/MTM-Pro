# coding=utf-8
"""First-class benchmark metric for the test-time rollout deploy path.

Permanent benchmarking utility. Introduced by action ``A7`` of the RLRC test-time
rollout deployer benchmarking `.junie` plan
(``perf_RLRP-785_testtime_rollout_benchmarking_plan_20260903.md``, rev. 5).

One :class:`StepBenchmarkMetric` == one ``(level, feedback regime, timing pass)`` population
of per-step latency samples, reduced ONCE after the timed window closed. A
:class:`BenchmarkMetricSet` collects every population measured on one run together with the
per-step-paired cost attribution (:class:`LevelDeltaMetric`).

Design invariants (plan section 3, gate ``R7``):
  * **Pure data + pure reductions.** No ``torch`` import is needed to *read* this module, so
    old and new artifacts can be inspected and aggregated on a host without CUDA.
  * **JSON-safe.** Every dict key that reaches JSON is a ``str`` (:meth:`BenchmarkMetricSet.key`
    is the single place the composite key is formed and parsed); no field is ever an
    ``np.ndarray`` (raw samples are ``List[float]``); enums are ``str``-valued so they
    serialise as their own names; ``to_dict() -> json.dumps -> from_dict()`` is lossless.
"""
from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any, Dict, List, Optional, Sequence

# NOTE: ``numpy`` is imported lazily inside :meth:`StepBenchmarkMetric.from_samples` only, so
# that merely *reading* an artifact (``from_dict``) never requires numpy/torch on the host.


class BenchmarkLevel(str, enum.Enum):
    """The three strictly nested timed regions of the deploy path (plan section 3.5)."""

    MODEL_CALL = "model_call"  # sample_1d_direct_model_call
    DEPLOYER_STEP = "deployer_step"  # predict_next_state
    CONTROL_LOOP_STEP = "control_loop_step"  # obs(+estimator|own pred) + act -> next env obs


class FeedbackRegime(str, enum.Enum):
    """Exactly TWO members -- rev. 5 (R17/Q5) dropped the legacy ``MIXED`` marker outright.

    A rollout is split at ``ground_truth_feed_warmup_steps`` into two labelled populations
    (see plan section 3.5 / R3), so a ``MIXED`` value would never be correct to emit. No
    existing artifact ever contained it, so there is no back-compat read path to keep.
    """

    ESTIMATOR = "estimator"  # obs from state estimator / ground truth
    AUTOREGRESSIVE = "autoregressive"  # obs from our own previous prediction (model space, :422)


class TimingPass(str, enum.Enum):  # rev. 4 (R1) -- see plan section 3.7
    DEVICE_TIME = "device_time"  # cuda events, deferred sync -> throughput
    DEPLOYABLE_LATENCY = "deployable_latency"  # perf_counter + per-step sync -> HEADLINE


def decimate_raw_samples(
    samples_ms: Any, max_raw_samples: int = 10_000
) -> "tuple[List[float], bool]":
    """Cap a raw latency population by UNIFORM DECIMATION, preserving temporal coverage.

    Permanent size policy. Introduced by action ``RLRP-803-2`` of the RLRC test-time rollout
    deployer benchmarking `.junie` plan
    (``perf_RLRP-785_testtime_rollout_benchmarking_plan_20260903.md``).

    Keeping every step of a ``n_iter x n_repeat`` run would put megabytes of ``float`` into
    ``inference_benchmark.json``; 10k samples already give quartile/whisker estimates far
    tighter than any plotting resolution while bounding one population at ~O(100 kB). The
    retained indices are ``linspace(0, n-1, cap)`` on the **time-ordered** samples -- the
    population is never sorted and never randomly sampled, so a drift/thermal-throttling
    pattern remains visible in the subsample.

    :param samples_ms: the FULL population, in milliseconds (array-like).
    :param max_raw_samples: the cap; a non-positive value disables decimation entirely.
    :return: ``(samples, decimated)`` where ``samples`` is a JSON-safe ``List[float]``.
    """
    import numpy as np  # local import: reading an artifact must not require numpy

    arr = np.asarray(samples_ms, dtype=np.float64)
    cap = int(max_raw_samples)
    if cap <= 0 or arr.size <= cap:
        return [float(x) for x in arr.tolist()], False
    keep_idx = np.linspace(0, arr.size - 1, cap).round().astype(np.int64)
    return [float(x) for x in arr[keep_idx].tolist()], True


@dataclasses.dataclass
class StepBenchmarkMetric:
    """A single reduced ``(level, regime, pass)`` population of per-step latencies."""

    level: BenchmarkLevel
    feedback_regime: FeedbackRegime
    timing_pass: TimingPass
    # ---- reduction (always populated) ----
    n_steps: int
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_std_ms: float
    rate_hz: float  # 1 / median  -- the control-loop frequency convention
    throughput_hz: float  # n / sum     -- reported alongside, never instead
    # ---- provenance (mandatory: a rate without it is not reproducible) ----
    device_type: str = "cpu"  # "cuda" | "cpu"
    timing_backend: str = "perf_counter"  # "cuda_event" | "perf_counter" | "perf_counter_sync"
    n_warmup_discarded: int = 0  # TIMER warm-up (cache/autotune)
    # ---- rev. 4 (R3/R6/R12): population hygiene + dispersion, both first-class ----
    n_tainted_excluded: int = 0  # gate G12 (re-anchor / tutoring steps)
    regime_split_index: Optional[int] = None  # ground_truth_feed_warmup_steps boundary
    n_repeat: int = 1  # gate G5/G13
    per_repeat_rate_hz: List[float] = dataclasses.field(default_factory=list)
    cv_hz: Optional[float] = None  # gate G5, CARRIED not just checked
    headline_repeat_index: Optional[int] = None  # which repeat the headline came from
    timer_overhead_ms: Optional[float] = None  # gate G4, measured not assumed
    platform_info: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # ---- optional raw samples: JSON-safe list, NEVER a bare ndarray (rev. 4, R7) ----
    latency_samples_ms: Optional[List[float]] = None  # or a side-car .npz when large
    # Size-policy bookkeeping. Introduced by action ``RLRP-803-2`` of the RLRC test-time
    # rollout deployer benchmarking `.junie` plan
    # (``perf_RLRP-785_testtime_rollout_benchmarking_plan_20260903.md``). Permanent.
    # ``latency_samples_ms`` is capped at ``max_raw_samples`` by UNIFORM DECIMATION, so a
    # consumer must be able to tell the CARRIED population apart from the TRUE one; without
    # these two fields an ``n_steps``/``len(latency_samples_ms)`` mismatch is unreadable.
    latency_samples_n_total: Optional[int] = None  # true population size before decimation
    latency_samples_decimated: bool = False  # True <=> the carried list is a subsample
    schema_version: int = 3  # rev. 4 bumped v1->v2; action ``RLRP-803-7.2`` bumped v2->v3

    @classmethod
    def from_samples(
        cls,
        samples_s: Sequence[float],
        *,
        level: BenchmarkLevel,
        feedback_regime: FeedbackRegime,
        timing_pass: TimingPass,
        keep_raw_samples: bool = False,
        max_raw_samples: int = 10_000,
        **provenance: Any,
    ) -> "StepBenchmarkMetric":
        """Reduce per-step latency samples (in **seconds**) into one metric population.

        ``rate_hz`` is the outlier-robust ``1 / median`` control-loop frequency; ``throughput_hz``
        is ``n / sum`` and is reported alongside, never instead (plan section 6). All timing fields
        are expressed in **milliseconds**.

        :param keep_raw_samples: carry the population itself into the artifact, so the plotter can
            draw TRUE quartiles instead of the ``median -/+ 0.6745 * std`` normal approximation.
        :param max_raw_samples: size policy for that population (action ``RLRP-803-2``). Above the
            cap the CARRIED list is a uniformly decimated subsample -- see
            :func:`decimate_raw_samples`. The reduction below is ALWAYS computed from the FULL
            population, never from the subsample.
        """
        import numpy as np  # local import: reading an artifact must not require numpy

        arr = np.asarray(list(samples_s), dtype=np.float64)
        if arr.size == 0:
            raise ValueError(
                "StepBenchmarkMetric.from_samples got an empty sample population -- an "
                "empty population must be handled by the caller (see DeployStepTimer.report)."
            )
        median_s = float(np.median(arr))
        total_s = float(arr.sum())
        metric = cls(
            level=level,
            feedback_regime=feedback_regime,
            timing_pass=timing_pass,
            n_steps=int(arr.size),
            latency_mean_ms=float(arr.mean() * 1e3),
            latency_median_ms=float(median_s * 1e3),
            latency_p95_ms=float(np.percentile(arr, 95) * 1e3),
            latency_p99_ms=float(np.percentile(arr, 99) * 1e3),
            latency_std_ms=float(arr.std(ddof=1) * 1e3) if arr.size > 1 else 0.0,
            rate_hz=float(1.0 / median_s) if median_s > 0.0 else float("inf"),
            throughput_hz=float(arr.size / total_s) if total_s > 0.0 else float("inf"),
        )
        for key, value in provenance.items():
            if not hasattr(metric, key):
                raise TypeError(
                    f"StepBenchmarkMetric.from_samples got an unknown provenance field {key!r}"
                )
            setattr(metric, key, value)
        if keep_raw_samples:
            samples_ms, decimated = decimate_raw_samples(arr * 1e3, max_raw_samples)
            metric.latency_samples_ms = samples_ms
            metric.latency_samples_n_total = int(arr.size)
            metric.latency_samples_decimated = decimated
        return metric

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe mapping. Enums serialise to their ``str`` values."""
        payload: Dict[str, Any] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, enum.Enum):
                value = value.value
            payload[field.name] = value
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StepBenchmarkMetric":
        """Reconstruct from a JSON-decoded mapping. Raises on an unknown regime (rev. 5)."""
        data = dict(payload)
        data["level"] = BenchmarkLevel(data["level"])
        data["timing_pass"] = TimingPass(data["timing_pass"])
        # rev. 5 (R17/Q5): a payload carrying a dropped regime (e.g. "mixed") must NOT be
        # silently resurrected -- ``FeedbackRegime`` has exactly two members and this raises.
        data["feedback_regime"] = FeedbackRegime(data["feedback_regime"])
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise TypeError(f"StepBenchmarkMetric.from_dict got unknown field(s): {sorted(unknown)}")
        return cls(**data)


@dataclasses.dataclass
class LevelDeltaMetric:
    """rev. 4 (R5): a reduction of PER-STEP PAIRED deltas, computed while the pairing existed.

    ``median(a) - median(b) != median(a - b)``, so the cost breakdown can never be derived
    from two already-reduced :class:`StepBenchmarkMetric`s. The timer computes
    ``outer[i] - inner[i]`` per step, subtracts the calibrated inner-level overhead, and only
    then reduces.
    """

    outer_level: BenchmarkLevel
    inner_level: BenchmarkLevel
    n_pairs: int
    delta_median_ms: float
    delta_p95_ms: float
    overhead_correction_ms: float = 0.0  # calibrated t_ovh removed from every pair
    n_negative_pairs: int = 0  # MUST be 0 -- a negative delta means broken nesting
    # Regime label. Introduced by action ``RLRP-803-7.2`` of the RLRC test-time rollout
    # deployer benchmarking `.junie` plan
    # (``perf_RLRP-785_testtime_rollout_benchmarking_plan_20260903.md``). Permanent.
    # A delta pooled over BOTH feedback regimes cannot be lined up with the per-regime
    # latency numbers reported elsewhere in the same artifact, which made the A4
    # cost-breakdown figure misleading (RLRP-803 item 7.2). ``None`` means "unknown" and
    # occurs ONLY when reading a schema v2 artifact, whose deltas carry no regime.
    feedback_regime: Optional[FeedbackRegime] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, enum.Enum):
                value = value.value
            payload[field.name] = value
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LevelDeltaMetric":
        """Reconstruct from a JSON-decoded mapping.

        Back-compat (action ``RLRP-803-7.2``): a schema v2 record has NO ``feedback_regime``
        key, so it defaults to ``None`` ("unknown") rather than being guessed.
        """
        data = dict(payload)
        data["outer_level"] = BenchmarkLevel(data["outer_level"])
        data["inner_level"] = BenchmarkLevel(data["inner_level"])
        regime = data.get("feedback_regime", None)
        data["feedback_regime"] = None if regime is None else FeedbackRegime(regime)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise TypeError(f"LevelDeltaMetric.from_dict got unknown field(s): {sorted(unknown)}")
        return cls(**data)


@dataclasses.dataclass
class BenchmarkMetricSet:
    """All levels/regimes/passes measured on one run + the derived cost attribution.

    rev. 4 (R7): the key is a STRING (``"<level>|<regime>|<pass>"``), not a tuple -- tuple
    keys are not JSON-serialisable and silently break ``to_dict()`` round-trips.
    """

    metrics: Dict[str, StepBenchmarkMetric] = dataclasses.field(default_factory=dict)
    level_deltas: List[LevelDeltaMetric] = dataclasses.field(default_factory=list)
    schema_version: int = 3

    # ---- the ONE place the composite key is formed and parsed ------------------------------
    @staticmethod
    def key(
        level: BenchmarkLevel, regime: FeedbackRegime, timing_pass: TimingPass
    ) -> str:
        return f"{BenchmarkLevel(level).value}|{FeedbackRegime(regime).value}|{TimingPass(timing_pass).value}"

    @staticmethod
    def parse_key(key: str):
        level_s, regime_s, pass_s = key.split("|")
        return BenchmarkLevel(level_s), FeedbackRegime(regime_s), TimingPass(pass_s)

    def add(self, metric: StepBenchmarkMetric) -> None:
        self.metrics[self.key(metric.level, metric.feedback_regime, metric.timing_pass)] = metric

    def get(
        self,
        level: BenchmarkLevel,
        regime: FeedbackRegime,
        timing_pass: TimingPass,
    ) -> Optional[StepBenchmarkMetric]:
        return self.metrics.get(self.key(level, regime, timing_pass))

    def _first_by(
        self, level: BenchmarkLevel, timing_pass: TimingPass
    ) -> Optional[StepBenchmarkMetric]:
        """First metric matching ``level``/``pass`` regardless of regime.

        ``ESTIMATOR`` is preferred when present, so the headline is the deployable regime a
        state-estimator-fed controller actually pays.
        """
        for regime in (FeedbackRegime.ESTIMATOR, FeedbackRegime.AUTOREGRESSIVE):
            found = self.get(level, regime, timing_pass)
            if found is not None:
                return found
        return None

    def headline_rate_hz(self) -> Optional[float]:
        """control_loop_step + DEPLOYABLE_LATENCY -- never the device-time figure (R1)."""
        metric = self._first_by(
            BenchmarkLevel.CONTROL_LOOP_STEP, TimingPass.DEPLOYABLE_LATENCY
        )
        return None if metric is None else metric.rate_hz

    def model_rate_hz(self) -> Optional[float]:
        """model_call, device-time pass -- the figure RLRP-786/787 will move."""
        metric = self._first_by(BenchmarkLevel.MODEL_CALL, TimingPass.DEVICE_TIME)
        return None if metric is None else metric.rate_hz

    def async_gap_ms(self) -> Optional[float]:
        """rev. 4: deployable-latency minus device-time median for the control-loop step.

        The amount of asynchrony the deployment relies on -- a reported quantity (gate G11).
        """
        deployable = self._first_by(
            BenchmarkLevel.CONTROL_LOOP_STEP, TimingPass.DEPLOYABLE_LATENCY
        )
        device = self._first_by(BenchmarkLevel.CONTROL_LOOP_STEP, TimingPass.DEVICE_TIME)
        if deployable is None or device is None:
            return None
        return deployable.latency_median_ms - device.latency_median_ms

    def cost_breakdown_regimes(self) -> List[FeedbackRegime]:
        """The feedback regimes the cost breakdown can be split on.

        Introduced by action ``RLRP-803-7.2`` of the RLRC test-time rollout deployer
        benchmarking `.junie` plan
        (``perf_RLRP-785_testtime_rollout_benchmarking_plan_20260903.md``). Permanent.

        An EMPTY list means the deltas carry no regime label -- i.e. a schema v2 artifact --
        and the only honest rendering is a single, explicitly regime-agnostic breakdown.

        :return: the labelled regimes, in :class:`FeedbackRegime` declaration order.
        """
        labelled = {d.feedback_regime for d in self.level_deltas if d.feedback_regime is not None}
        return [regime for regime in FeedbackRegime if regime in labelled]

    @staticmethod
    def _weighted_mean(terms: "List[tuple]") -> float:
        """``sum(w_i * x_i) / sum(w_i)``, degrading to the plain mean when every weight is 0."""
        total_w = float(sum(w for _, w in terms))
        if total_w <= 0.0:
            return float(sum(x for x, _ in terms) / len(terms))
        return float(sum(x * w for x, w in terms) / total_w)

    def _model_call_metrics(
        self, regime: Optional[FeedbackRegime]
    ) -> List[StepBenchmarkMetric]:
        """``model_call`` populations, device-time pass preferred, then deployable latency.

        With ``regime`` given, exactly that regime's population; otherwise EVERY regime's
        (the pooling is then done by the caller, explicitly).
        """
        regimes = list(FeedbackRegime) if regime is None else [regime]
        for timing_pass in (TimingPass.DEVICE_TIME, TimingPass.DEPLOYABLE_LATENCY):
            found = [
                metric
                for metric in (
                    self.get(BenchmarkLevel.MODEL_CALL, reg, timing_pass) for reg in regimes
                )
                if metric is not None
            ]
            if found:
                return found
        return []

    def cost_breakdown_ms(
        self, feedback_regime: Optional[FeedbackRegime] = None
    ) -> Dict[str, float]:
        """Cost attribution built from ``level_deltas`` ONLY (rev. 4, R5).

        Made REGIME-AWARE by action ``RLRP-803-7.2`` of the RLRC test-time rollout deployer
        benchmarking `.junie` plan
        (``perf_RLRP-785_testtime_rollout_benchmarking_plan_20260903.md``). Permanent.

        Returns ``{"model_call", "buffers_and_model_adapter", "env_adapters_and_feedback"}`` --
        the honest per-step-paired decomposition of the control-loop budget.

        **Pooling rule (explicit, documented, and asserted by the unit tests).** With
        ``feedback_regime=None`` every regime is pooled with the SAMPLE-COUNT-WEIGHTED MEAN
        (``n_pairs`` for the deltas, ``n_steps`` for ``model_call``). Before ``RLRP-803-7.2``
        the ``model_call`` term instead silently returned the estimator regime whenever it
        existed, which did not line up with the pooled delta terms next to it; that accidental
        preference is GONE. With a single regime measured the pooling rule is the identity, so
        a one-regime artifact keeps its previous numbers exactly.

        :param feedback_regime: restrict every term to that regime. Ignored -- with the pooled
            behaviour returned instead -- when the deltas carry no regime label at all (a
            schema v2 artifact, see :meth:`cost_breakdown_regimes`), because such an artifact
            cannot honestly answer a per-regime question.
        :return: ``{component: milliseconds}``, restricted to the components measured.
        """
        regime = None if feedback_regime is None else FeedbackRegime(feedback_regime)
        if not self.cost_breakdown_regimes():
            regime = None  # schema v2 artifact -> regime-agnostic terms, never a guess

        breakdown: Dict[str, float] = {}
        model_calls = self._model_call_metrics(regime)
        if model_calls:
            breakdown["model_call"] = self._weighted_mean(
                [(m.latency_median_ms, m.n_steps) for m in model_calls]
            )

        buckets: Dict[str, List[tuple]] = {}
        for delta in self.level_deltas:
            if regime is not None and delta.feedback_regime is not regime:
                continue
            if (
                delta.outer_level is BenchmarkLevel.DEPLOYER_STEP
                and delta.inner_level is BenchmarkLevel.MODEL_CALL
            ):
                component = "buffers_and_model_adapter"
            elif (
                delta.outer_level is BenchmarkLevel.CONTROL_LOOP_STEP
                and delta.inner_level is BenchmarkLevel.DEPLOYER_STEP
            ):
                component = "env_adapters_and_feedback"
            else:
                continue
            buckets.setdefault(component, []).append((delta.delta_median_ms, delta.n_pairs))
        for component, terms in buckets.items():
            breakdown[component] = self._weighted_mean(terms)
        return breakdown

    def _control_loop_metric(
        self, regime: Optional[FeedbackRegime]
    ) -> Optional[StepBenchmarkMetric]:
        """The control-loop population the cost breakdown must reconstruct."""
        if regime is None:
            return self._first_by(
                BenchmarkLevel.CONTROL_LOOP_STEP, TimingPass.DEVICE_TIME
            ) or self._first_by(
                BenchmarkLevel.CONTROL_LOOP_STEP, TimingPass.DEPLOYABLE_LATENCY
            )
        return self.get(
            BenchmarkLevel.CONTROL_LOOP_STEP, regime, TimingPass.DEVICE_TIME
        ) or self.get(
            BenchmarkLevel.CONTROL_LOOP_STEP, regime, TimingPass.DEPLOYABLE_LATENCY
        )

    def assert_level_consistency(self, tol_ms: float = 0.5) -> None:
        """Gate G9, asserted on the per-step paired deltas (never on differences of medians).

        Raises when any ``LevelDeltaMetric`` recorded a negative pair (an inner level outlasting
        its outer level -- broken nesting), or when the reconstructed control-loop median is not
        within ``tol_ms`` of the measured control-loop median.

        Since action ``RLRP-803-7.2`` the reconstruction is checked PER REGIME whenever the
        deltas are labelled: pooling two regimes before comparing would hide a per-regime
        inconsistency behind an average. A schema v2 artifact keeps the single pooled check.
        """
        for delta in self.level_deltas:
            if delta.n_negative_pairs > 0:
                raise AssertionError(
                    f"Level consistency violated: {delta.n_negative_pairs} negative pair(s) "
                    f"for {delta.outer_level.value} - {delta.inner_level.value} (an inner level "
                    "outlasted its outer level -- broken nesting)."
                )
        for regime in self.cost_breakdown_regimes() or [None]:
            control_loop = self._control_loop_metric(regime)
            breakdown = self.cost_breakdown_ms(feedback_regime=regime)
            if control_loop is None or len(breakdown) != 3:
                continue
            reconstructed = sum(breakdown.values())
            if not math.isclose(
                reconstructed, control_loop.latency_median_ms, abs_tol=tol_ms
            ):
                _scope = "pooled" if regime is None else regime.value
                raise AssertionError(
                    f"Level consistency violated ({_scope}): reconstructed control-loop median "
                    f"{reconstructed:.4f}ms != measured {control_loop.latency_median_ms:.4f}ms "
                    f"(tol {tol_ms}ms)."
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "level_deltas": [d.to_dict() for d in self.level_deltas],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BenchmarkMetricSet":
        return cls(
            metrics={
                k: StepBenchmarkMetric.from_dict(v)
                for k, v in payload.get("metrics", {}).items()
            },
            level_deltas=[
                LevelDeltaMetric.from_dict(d) for d in payload.get("level_deltas", [])
            ],
            schema_version=int(payload.get("schema_version", 2)),
        )
