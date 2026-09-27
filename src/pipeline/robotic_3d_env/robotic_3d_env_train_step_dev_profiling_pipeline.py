# coding=utf-8
"""DEVELOPMENT training-step profiling of ONE robotic 3D env experiment (RLRP-824 / RLRP-830 / RLRP-786).

Development tool, NOT the publication benchmark. It is the single-experiment sweep entry point:
the experiment multirun cfg with the ``/pipeline`` group swapped to
``robotic_3d_env_train_step_dev_profiling_pipeline`` (``launcher/configs/multirun_overrides/
robotic_3d_env/*/benchmark-train-step-*.yaml``), driven by ``robotic_3d_env_main_multirun.py`` /
``launcher/robotic_3d_env_train_step_dev_profiling.py``. Its ``pipeline.benchmark.*`` block carries the
DEV-ONLY levers that rewrite the experiment's training regime for A/B profiling (``compile_mode`` incl.
the RLRP-786 ``cuda-graph`` step, ``cuda_graph_allow_autocast``, ``precision`` presets, ``source_dtype``,
``append_report_to`` aggregate rows). None of those exist in the publication tool.

The CORE timing/report logic lives in ``tools.benchmark_tools.model_training_benchmark`` (RLRP-838):
this module keeps the public ``execute(cfg, headless)`` entry point and re-exports the result types /
helpers the dev-profiling tests import.

Publication counterpart (RLRP-838, experiment- and environment-agnostic, composed source experiment
cfg authoritative, measurement knobs only): ``launcher/model_training_benchmark.py`` ->
``tools.benchmark_tools.model_training_benchmark.run_training_benchmark``.
"""
from __future__ import annotations

from tools.benchmark_tools.model_training_benchmark import (  # noqa: F401
    COMPILE_MODES,
    CUDA_GRAPH_MODE,
    PHASES,
    PRECISION_PRESETS,
    PROFILER_TRACE_BASENAME,
    REPORT_BASENAME,
    VAL_WARMUP_BATCHES,
    ProfilerOpRow,
    ProfilerSummary,
    TrainStepBenchmarkResult,
    TrainStepBenchmarkSettings,
    _Clock,
    _DataPath,
    _GpuUtilizationSampler,
    _TrainStep,
    _apply_precision_preset,
    _batch_dtype,
    _batch_to_device,
    _build_dataloader_path,
    _build_replay_buffer_path,
    _cast_replay_buffer,
    _count_trainable_params,
    _cuda_graph_blockers,
    _cycle,
    _describe_loss_composition,
    _load_training_source,
    _make_train_step,
    _resolve_epochs,
    _resolve_label,
    _resolve_source_dtype,
    _run_loss,
    _self_device_us,
    _torch_profiler_pass,
    _write_reports,
    execute_one,
)


def execute(cfg, headless: bool = False):
    """Time the training step of the configured experiment under the DEV ``pipeline.benchmark`` levers.

    Thin wrapper around :func:`tools.benchmark_tools.model_training_benchmark.execute_one`: the
    measurement knobs AND the dev-only regime levers are read from ``cfg.pipeline.benchmark``
    (``TrainStepBenchmarkSettings.from_pipeline_cfg``), which is exactly what the publication tool
    forbids on its composed source experiment cfgs.

    :param cfg: Hydra configuration file (a main-experiment + multirun override cfg with
        ``pipeline: robotic_3d_env_train_step_dev_profiling_pipeline``)
    :param headless: turn environment rendering off (param for commandline flag)
    :return: the benchmark result (also written to the run directory)
    """
    return execute_one(cfg, headless=headless)
