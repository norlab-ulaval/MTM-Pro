# coding=utf-8
"""Test-time rollout deployer benchmarking tools (RLRP-785).

Permanent, env-agnostic benchmarking utilities for the test-time rollout deploy path.
Introduced by the RLRC test-time rollout deployer benchmarking `.junie` plan
``perf_RLRP-785_testtime_rollout_benchmarking_plan_20260903.md`` (rev. 5).

Public surface:
  * :mod:`tools.benchmark_tools.benchmark_metric` -- action ``A7``: the first-class,
    JSON-safe benchmark metric carrier (``StepBenchmarkMetric`` / ``LevelDeltaMetric`` /
    ``BenchmarkMetricSet``) plus the ``BenchmarkLevel`` / ``FeedbackRegime`` / ``TimingPass``
    taxonomy (action ``A8``).
  * :mod:`tools.benchmark_tools.step_timer` -- action ``A1``: :class:`DeployStepTimer`, the
    multi-level, dual-pass, taint-aware per-step timer with a zero-cost OFF path.
"""
from tools.benchmark_tools.benchmark_metric import (
    BenchmarkLevel,
    BenchmarkMetricSet,
    FeedbackRegime,
    LevelDeltaMetric,
    StepBenchmarkMetric,
    TimingPass,
)
from tools.benchmark_tools.step_timer import DeployStepTimer

__all__ = [
    "BenchmarkLevel",
    "FeedbackRegime",
    "TimingPass",
    "StepBenchmarkMetric",
    "LevelDeltaMetric",
    "BenchmarkMetricSet",
    "DeployStepTimer",
]
