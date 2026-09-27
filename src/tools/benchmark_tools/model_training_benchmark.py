# coding=utf-8
"""Experiment- and environment-agnostic training-speed benchmark (RLRP-838).

PUBLICATION tool: times the training step of N models, each composed from an EXISTING multirun
experiment cfg (robotic 3D env OR math env), back-to-back on one node/process, and projects the
per-epoch / per-trial wall-clock so architectures can be compared on the same allocation.

The composed source experiment cfg is AUTHORITATIVE. The only way to change what is timed is the
``source_experiment_cfg.overrides`` / ``model_overrides`` lists (verbatim Hydra compose overrides,
meant for the heterogeneous sweep keys such as ``A_trj_len``). The launcher-level ``benchmark`` block
only says HOW LONG to time and WHAT to record (:class:`TrainStepBenchmarkSettings`); nothing is
overlaid onto the composed cfg, and any ``benchmark`` / ``pipeline.benchmark`` node -- on the composed
cfg or in the overrides -- is rejected (:func:`_assert_no_benchmark_override`,
:func:`_assert_composed_cfg_has_no_benchmark_node`).

Public API
----------
* :class:`TrainStepBenchmarkSettings` -- measurement knobs; ``from_launcher_cfg`` (publication) /
  ``from_pipeline_cfg`` (the dev profiler, which additionally carries regime levers).
* :func:`execute_one` -- time ONE already-composed experiment cfg (the single-model core).
* :func:`run_training_benchmark` -- compose every
  ``benchmark.source_experiment_cfg.benchmark_selected_models`` entry in-process
  (``GlobalHydra`` clear/re-init, shared + per-model overrides via
  :func:`source_experiment_cfg._merge_overrides`), call :func:`execute_one` per model,
  and write the aggregate ``training_benchmark.json`` / ``.md``.
* :class:`TrainStepBenchmarkResult` -- per-model result carrying two wall-clock projections over
  the full fetched dataset: ``projected_epoch_hours``/``projected_trial_hours`` (the measured step
  incl. its data path: loader + h2d + forward + backward + optimizer) and the COMPUTE-ONLY bound
  ``approx_epoch_hours``/``approx_trial_hours`` (forward + backward + optimizer only, i.e. what the
  trial would cost with a free data path -- the ``GPU duty %`` upper bound turned into hours).

DEVELOPMENT counterpart: ``pipeline.robotic_3d_env.robotic_3d_env_train_step_dev_profiling_pipeline``
(RLRP-824/830/786 single-experiment A/B sweeps) wraps :func:`execute_one` with
``TrainStepBenchmarkSettings.from_pipeline_cfg`` -- the ONLY path where ``pipeline.benchmark.*`` regime
levers (``compile_mode``, ``precision``, ``source_dtype``, ...) are honoured.
"""
from __future__ import annotations
import contextlib
import gc
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import omegaconf
import torch
from mbrl.types import TransitionBatch
from mbrl.util import common as common_utils

from algorithm.experience_replay_learning_loop.core.replay_buffer_utils import split_replay_buffer
from algorithm.utils import seed_me
from pipeline.pipeline_utils.general.setup import (
    setup_multistep_step_model_and_trainer,
    uder_cfg_validation,
)
from tools.console_tools.message import consol_msg_universal, consol_msg_universal_one_liner
from tools.hydra_apps_tools.hydra_utils import get_hydra_experiment_cwd
from tools.hydra_apps_tools.r2s2r_apps_utils import (
    apply_torch_backend_cfg,
    configure_cuda_float32_high_precision,
)
from tools.multistep_tools.ms_replaybuffer_save_load_utils import load_multistep_replaybuffer_with_spec
from tools.multistep_tools.window_dataset.pipeline_utils import (
    build_window_data_source,
    build_window_dataset,
    describe_window_path,
    is_dataloader_data_manager,
    move_store_to_training_device,
    stamp_original_dataset_size,
)
from tools.torch_tools.cuda_graph_train_step import CudaGraphTrainStep

PHASES = ("loader", "h2d", "forward", "backward", "optimizer")
REPORT_BASENAME = "train_step_benchmark"
AGGREGATE_REPORT_BASENAME = "training_benchmark"
# v1: measured `epoch_hours`/`trial_hours` + `approx_*` alias of the same value.
# v2 (2026-09-22): `projected_*` (measured step, data path included) + `approx_*` = COMPUTE-ONLY bound
#     (forward+backward+optimizer, val eval_score only), new `val_compute_ms` / `data_path_ms` /
#     `data_path_share_pct`; the `epoch_hours` / `trial_hours` keys are gone.
AGGREGATE_SCHEMA_VERSION = 2
PROFILER_TRACE_BASENAME = "train_step_benchmark_trace.json"
VAL_WARMUP_BATCHES = 3  # untimed eval-mode batches before the timed validation steps (capped by warmup_batches)
VAL_OUTLIER_MEAN_TO_MEDIAN_RATIO = 1.25  # mean/median above this -> a one-off val batch cost is reported as a note


def summarize_val_batch_times(samples_ms: Sequence[float]) -> Tuple[float, Optional[str]]:
    """Steady-state validation ms/batch = median of the timed batches (+ a note when the mean diverges).

    The projection multiplies ``val_ms`` by the val batches of EVERY epoch, so a one-off cost hit by
    a single timed batch (cudnn autotune of the ragged last val batch, first-call allocations) must
    not be folded into it. Returns ``(nan, None)`` when nothing was timed.
    """
    if not samples_ms:
        return float("nan"), None
    ordered = sorted(float(s) for s in samples_ms)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    mean = sum(ordered) / n
    note = None
    if median > 0.0 and mean / median > VAL_OUTLIER_MEAN_TO_MEDIAN_RATIO:
        note = (
            f"validation one-off cost: mean {mean:.1f} ms vs median {median:.1f} ms over {n} timed val batches "
            f"(max {ordered[-1]:.1f} ms) -> val ms/batch reports the median (steady state)"
        )
    return median, note


COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
# RLRP-786: ``pipeline.benchmark.compile_mode=cuda-graph`` -> NOT ``torch.compile``: the whole production
# step (``loss -> backward -> fused+capturable Adam``) is recorded ONCE in a ``torch.cuda.CUDAGraph``
# (``tools.torch_tools.cuda_graph_train_step.CudaGraphTrainStep``) and replayed for the timed / profiled
# steps. Same kernels (``torch.equal`` to eager, RLRP-786 FR3), one launch. Requires
# ``pipeline.performance_mode=fast`` + a capture-ready MTM-Pro model; else eager + a note.
CUDA_GRAPH_MODE = "cuda-graph"
# ``pipeline.benchmark.precision`` presets -> (``ms_model.mixed_precision`` autocast value,
# ``torch_backend.cuda_high_precision_float32``). ``fp32`` is the path of the ICRA baselines today
# (no autocast, TF32 off, "highest" matmul precision); ``tf32`` keeps fp32 storage / math but lets the
# A100 tensor cores round the matmul / conv inputs to 10 mantissa bits; ``bf16`` is the RLRP-824 ULH
# opt-in (autocast on the forward + loss, fp32 masters, TF32 off); ``bf16+tf32`` adds TF32 for the
# fp32 ops autocast leaves alone.
PRECISION_PRESETS: Dict[str, Tuple[Optional[str], bool]] = {
    "fp32": (None, True),
    "tf32": (None, False),
    "bf16": ("bf16", True),
    "bf16+tf32": ("bf16", False),
}
# Environment families the core can build a training source for, keyed by the ``pipeline.name``
# prefix of the composed experiment cfg (``robotic_3d_env_full_pipeline_multirun`` /
# ``math_env_full_pipeline_multirun`` ...).
ROBOTIC_3D_ENV_FAMILY = "robotic_3d_env"
MATH_ENV_FAMILY = "math_env"
ENV_FAMILIES = (ROBOTIC_3D_ENV_FAMILY, MATH_ENV_FAMILY)


# ==== Measurement settings =======================================================================
_NULLS = (None, "", "null", "None", "none", False)


@dataclass
class TrainStepBenchmarkSettings:
    """HOW LONG to time and WHAT to record -- everything the core needs besides the experiment cfg.

    Two constructors, two tools:

    * :meth:`from_launcher_cfg` -- the PUBLICATION tool (``model_training_benchmark.yaml``, top-level
      ``benchmark`` block): MEASUREMENT knobs only. The regime levers below are forced to their
      "as the experiment cfg says" value and a launcher block that carries one raises, so the composed
      source experiment cfg stays authoritative.
    * :meth:`from_pipeline_cfg` -- the DEV profiler (``robotic_3d_env_train_step_dev_profiling_pipeline``,
      ``cfg.pipeline.benchmark``): measurement knobs PLUS the dev-only regime levers used by the
      RLRP-830 / RLRP-786 A/B sweeps.
    """

    warmup_batches: int = 5
    num_batches: int = 30
    val_batches: int = 10
    slurm_wall_time_hours: float = 96.0
    torch_profiler: bool = False
    torch_profiler_batches: int = 5
    torch_profiler_top_n: int = 25
    torch_profiler_export_trace: bool = False
    # ---- DEV-ONLY regime levers (``from_pipeline_cfg`` only; publication = "as configured") ----
    compile_mode: Optional[str] = None  # null | torch.compile mode | ``cuda-graph`` (forces the step)
    cuda_graph_allow_autocast: bool = False
    precision: Optional[str] = None  # null | fp32 | tf32 | bf16 | bf16+tf32 (rewrites mixed_precision/TF32)
    source_dtype: Optional[str] = None  # null | float32 | float64 (re-materialises the replay buffer)
    append_report_to: Optional[str] = None  # aggregate markdown the dev sweeps append one row to

    MEASUREMENT_KEYS = (
        "warmup_batches",
        "num_batches",
        "val_batches",
        "slurm_wall_time_hours",
        "torch_profiler",
        "torch_profiler_batches",
        "torch_profiler_top_n",
        "torch_profiler_export_trace",
    )
    DEV_ONLY_KEYS = (
        "compile_mode",
        "cuda_graph_allow_autocast",
        "precision",
        "source_dtype",
        "append_report_to",
    )

    @staticmethod
    def _opt_str(raw: Any) -> Optional[str]:
        return None if raw in _NULLS else str(raw)

    @classmethod
    def _measurement_kwargs(cls, node: Any) -> Dict[str, Any]:
        node = node or {}
        kwargs: Dict[str, Any] = {}
        for key in cls.MEASUREMENT_KEYS:
            value = node.get(key, None)
            if value is not None:
                kwargs[key] = value
        for key in ("warmup_batches", "num_batches", "val_batches", "torch_profiler_batches", "torch_profiler_top_n"):
            if key in kwargs:
                kwargs[key] = int(kwargs[key])
        for key in ("torch_profiler", "torch_profiler_export_trace"):
            if key in kwargs:
                kwargs[key] = bool(kwargs[key])
        if "slurm_wall_time_hours" in kwargs:
            kwargs["slurm_wall_time_hours"] = float(kwargs["slurm_wall_time_hours"])
        return kwargs

    @classmethod
    def from_launcher_cfg(cls, bench_cfg: Any) -> "TrainStepBenchmarkSettings":
        """PUBLICATION constructor: the top-level ``benchmark`` block of ``model_training_benchmark.yaml``.

        Raises when the block carries a dev-only regime lever: the training regime of a benched model
        is defined by its composed source experiment cfg (+ ``source_experiment_cfg.overrides``), never
        by the launcher.
        """
        bench_cfg = bench_cfg or {}
        offending = [key for key in cls.DEV_ONLY_KEYS if key in bench_cfg]
        if offending:
            raise ValueError(
                f"benchmark.{{{', '.join(offending)}}} are DEV-ONLY profiling levers "
                "(robotic_3d_env_train_step_dev_profiling_pipeline) and are not allowed in the publication "
                "training-speed benchmark: the composed source experiment cfg is authoritative. Select the "
                "experiment cfg that IS the wanted regime (e.g. `multirun-X+cudagraph`) or state the change in "
                "`benchmark.source_experiment_cfg.overrides` / `model_overrides` on production keys."
            )
        return cls(**cls._measurement_kwargs(bench_cfg))

    @classmethod
    def from_pipeline_cfg(cls, cfg: omegaconf.DictConfig) -> "TrainStepBenchmarkSettings":
        """DEV constructor: ``cfg.pipeline.benchmark`` of the dev profiling pipeline (all keys honoured)."""
        node = omegaconf.OmegaConf.select(cfg, "pipeline.benchmark", default=None)
        if node is None:
            raise ValueError(
                "TrainStepBenchmarkSettings.from_pipeline_cfg: `pipeline.benchmark` is missing -- use the "
                "`robotic_3d_env_train_step_dev_profiling_pipeline` /pipeline group (dev profiler), or pass "
                "`settings=` explicitly (publication tool)."
            )
        kwargs = cls._measurement_kwargs(node)
        kwargs["compile_mode"] = cls._opt_str(node.get("compile_mode", None))
        kwargs["cuda_graph_allow_autocast"] = bool(node.get("cuda_graph_allow_autocast", False))
        kwargs["precision"] = cls._opt_str(node.get("precision", None))
        kwargs["source_dtype"] = cls._opt_str(node.get("source_dtype", None))
        kwargs["append_report_to"] = cls._opt_str(node.get("append_report_to", None))
        return cls(**kwargs)


# ==== torch.profiler summary =====================================================================
@dataclass
class ProfilerOpRow:
    """One ``torch.profiler.key_averages()`` entry, per profiled training step."""

    name: str
    calls_per_step: float
    self_cpu_ms_per_step: float
    self_cuda_ms_per_step: float


@dataclass
class ProfilerSummary:
    """Where a training step spends its time, from ``torch.profiler`` over ``batches`` steps.

    ``cuda_kernels_per_step`` and ``cuda_busy_pct`` are the launch-bound tell-tales: a step made of
    thousands of kernels whose summed device time is a small share of the wall-clock is waiting on
    the Python thread issuing them (the GPU duty-cycle of the phase timing is then an upper bound
    the SM never reaches)."""

    batches: int
    wall_ms_per_step: float
    cuda_kernel_ms_per_step: float
    cuda_kernels_per_step: float
    cpu_op_calls_per_step: float
    top_cpu: List[ProfilerOpRow]
    top_cuda: List[ProfilerOpRow]
    trace_path: Optional[str] = None

    @property
    def cuda_busy_pct(self) -> float:
        """Share of the profiled step the GPU actually executed kernels (device-side time / wall)."""
        if not self.wall_ms_per_step:
            return float("nan")
        return 100.0 * self.cuda_kernel_ms_per_step / self.wall_ms_per_step

    @property
    def mean_kernel_us(self) -> float:
        if not self.cuda_kernels_per_step:
            return float("nan")
        return 1e3 * self.cuda_kernel_ms_per_step / self.cuda_kernels_per_step

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update(cuda_busy_pct=self.cuda_busy_pct, mean_kernel_us=self.mean_kernel_us)
        return d

    @staticmethod
    def _table(rows: List[ProfilerOpRow]) -> List[str]:
        lines = ["| op | calls/step | self CPU ms/step | self CUDA ms/step |", "|---|---|---|---|"]
        for r in rows:
            lines.append(
                f"| `{r.name}` | {r.calls_per_step:.0f} | {r.self_cpu_ms_per_step:.2f} | {r.self_cuda_ms_per_step:.2f} |"
            )
        return lines

    def markdown_section(self) -> List[str]:
        has_cuda = self.cuda_kernels_per_step > 0
        lines = [
            f"## torch.profiler ({self.batches} training steps)",
            "",
            f"- wall: {self.wall_ms_per_step:.0f} ms/step; CPU-side op calls: {self.cpu_op_calls_per_step:.0f}/step",
        ]
        if has_cuda:
            lines.append(
                f"- CUDA: {self.cuda_kernels_per_step:.0f} kernels/step, mean {self.mean_kernel_us:.0f} us/kernel,"
                f" summed device time {self.cuda_kernel_ms_per_step:.0f} ms/step -> **GPU kernel-busy {self.cuda_busy_pct:.0f} %**"
                " of the step wall-clock"
            )
            if self.cuda_busy_pct < 50.0:
                lines.append(
                    f"- **LAUNCH-BOUND**: the GPU executes kernels {self.cuda_busy_pct:.0f} % of the time; the rest is"
                    " the single Python thread issuing them. Fewer / fused kernels (vectorize the horizon loop,"
                    " `torch.compile(mode='reduce-overhead')` = CUDA graphs) or co-located trials (`HPO_N_JOBS`)"
                    " are the levers, not the data path."
                )
        if self.trace_path:
            lines.append(f"- chrome trace: `{self.trace_path}` (open in `chrome://tracing` / Perfetto)")
        lines += ["", f"### Top {len(self.top_cpu)} ops by self CPU time", ""] + self._table(self.top_cpu)
        if has_cuda:
            lines += ["", f"### Top {len(self.top_cuda)} ops by self CUDA time", ""] + self._table(self.top_cuda)
        return lines


# ==== Result =====================================================================================
@dataclass
class TrainStepBenchmarkResult:
    """One benchmark run (one trial of the multirun): mean ms per phase over the timed batches,
    validation step time, run description and the wall-clock projection."""

    label: str
    host: str
    device: str
    gpu: Optional[str]
    torch_version: str
    store_device: str
    dataset_device: str
    num_workers: int
    pin_memory: bool
    mixed_precision: Optional[str]
    batch_size: int
    history_len: int
    horizon_len: int
    output_window_len: int
    params: int
    obs_dim: int
    act_dim: int
    num_train_windows: int
    num_val_windows: int
    train_batches_per_epoch: int
    val_batches_per_epoch: int
    epochs: int
    warmup_batches: int
    timed_batches: int
    timed_val_batches: int
    ms: Dict[str, float]
    total_ms: float
    val_ms: float  # steady-state (median) validation step incl. its loader gather + h2d
    peak_cuda_mem_mb: Optional[float]
    last_loss: float
    gpu_util_pct: Optional[float]  # mean SM utilization (pynvml / nvidia-smi) over the timed batches
    wall_time_hours: float
    notes: List[str] = field(default_factory=list)
    compile_mode: Optional[str] = None  # ``torch.compile`` mode applied to ``loss`` for the timed steps
    profiler: Optional[ProfilerSummary] = None
    # Cross-dataset diagnostics (Husky vs UAV): the per-environment feature handler and whether the
    # composed loss carries the per-feature geometry (orientation) term over the horizon.
    feature_handler: Optional[str] = None
    obs_groups: Optional[str] = None  # ``kind[indices]`` of every observation group, ``*`` = has a geometry loss_term
    geometry_loss: Optional[str] = None  # ``off`` | ``<objective> x<weight> on <n> group(s)``
    # Numerical regime of the row (precision-parity question, 2026-09-17): the training data path,
    # the dtype of the batches the loss receives, the process-global float32 matmul precision
    # (``highest`` = TF32 off, ``high`` / ``medium`` = TF32 on) and the ``pipeline.benchmark.precision``
    # preset that produced ``mixed_precision`` / ``float32_matmul_precision`` (``None`` = as configured).
    data_manager: str = "dataloader"
    data_dtype: Optional[str] = None
    float32_matmul_precision: Optional[str] = None
    precision_preset: Optional[str] = None
    # Steady-state (median) ``eval_score`` cost of a validation batch WITHOUT its loader gather / h2d
    # (``val_ms`` minus the data path): the validation term of the compute-only bound. ``None`` when
    # validation was not timed (legacy rows: the bound then counts the training steps only).
    val_compute_ms: Optional[float] = None

    # ---- derived -------------------------------------------------------------------------------
    # Two projections over the FULL fetched dataset (``train_batches_per_epoch`` /
    # ``val_batches_per_epoch`` x ``epochs``), independent of how many batches were timed:
    #   projected_* = measured step incl. its data path (loader + h2d + forward + backward + optimizer)
    #   approx_*    = compute-only bound (forward + backward + optimizer; val: eval_score only), i.e.
    #                 what the trial would cost with a FREE data path -- the ``GPU duty %`` upper
    #                 bound expressed in hours. projected / approx ~= 1 -> data path negligible.
    @property
    def compute_ms(self) -> float:
        """Model phases only (forward + backward + optimizer): the step without its data path."""
        return self.ms["forward"] + self.ms["backward"] + self.ms["optimizer"]

    @property
    def data_path_ms(self) -> float:
        """Data-management share of the step (loader gather + h2d)."""
        return self.ms["loader"] + self.ms["h2d"]

    @property
    def gpu_duty_cycle_pct(self) -> float:
        """Share of the step spent in the model phases (forward / backward / optimizer): the
        upper bound of the GPU utilization a single trial can reach on this node."""
        return 100.0 * self.compute_ms / self.total_ms if self.total_ms else float("nan")

    @property
    def projected_epoch_hours(self) -> float:
        """Measured step (data path INCLUDED) x the real epoch size: the wall-clock h/epoch
        projection of the trial (training + validation steps only)."""
        train_s = self.train_batches_per_epoch * self.total_ms / 1e3
        val_s = self.val_batches_per_epoch * self.val_ms / 1e3
        return (train_s + val_s) / 3600.0

    @property
    def projected_trial_hours(self) -> float:
        """``projected_epoch_hours * epochs``: the SLURM wall-time to budget (lower bound: rollouts /
        checkpoints / logging excluded)."""
        return self.projected_epoch_hours * self.epochs

    @property
    def approx_epoch_hours(self) -> float:
        """Compute-only h/epoch bound: the model phases (forward + backward + optimizer) of every
        training batch plus the ``eval_score`` cost of every validation batch, NO loader gather / h2d.

        What the trial would cost on this node with a free data path (device-resident data, zero-cost
        iterator): ``projected_epoch_hours * GPU duty %`` when validation is negligible. The gap
        ``projected - approx`` is the data-management cost per epoch; when ``val_compute_ms`` is
        unknown (legacy rows) the validation term is omitted.
        """
        train_s = self.train_batches_per_epoch * self.compute_ms / 1e3
        val_compute = self.val_compute_ms if self.val_compute_ms is not None else 0.0
        val_s = self.val_batches_per_epoch * val_compute / 1e3
        return (train_s + val_s) / 3600.0

    @property
    def approx_trial_hours(self) -> float:
        """Compute-only h/trial bound (``approx_epoch_hours * epochs``)."""
        return self.approx_epoch_hours * self.epochs

    @property
    def data_path_share_pct(self) -> float:
        """Share of the projected trial spent in data management: ``1 - approx / projected``."""
        if not self.projected_trial_hours:
            return float("nan")
        return 100.0 * (1.0 - self.approx_trial_hours / self.projected_trial_hours)

    @property
    def resubmissions_needed(self) -> int:
        """``resume_from_checkpoint`` chain length beyond the first submission (on the PROJECTED
        trial, data path included -- that is what the allocation pays)."""
        if self.wall_time_hours <= 0:
            return 0
        return max(int(math.ceil(self.projected_trial_hours / self.wall_time_hours)) - 1, 0)

    # ---- rendering -----------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update(
            compute_ms=self.compute_ms,
            data_path_ms=self.data_path_ms,
            gpu_duty_cycle_pct=self.gpu_duty_cycle_pct,
            projected_epoch_hours=self.projected_epoch_hours,
            projected_trial_hours=self.projected_trial_hours,
            approx_epoch_hours=self.approx_epoch_hours,
            approx_trial_hours=self.approx_trial_hours,
            data_path_share_pct=self.data_path_share_pct,
            resubmissions_needed=self.resubmissions_needed,
            profiler=None if self.profiler is None else self.profiler.to_dict(),
        )
        return d

    @staticmethod
    def markdown_header() -> str:
        cols = [
            "run", "device", "data path", "store", "data dtype", "AMP", "fp32 matmul", "compile", "B", "H/F", "Do/Da",
            "geom loss", "params",
        ] + [f"{p} (ms)" for p in PHASES] + [
            "total ms/batch", "val ms/batch", "GPU duty %", "GPU SM %", "kernel-busy %", "kernels/step",
            "peak CUDA", "h/epoch", "h/trial", "compute-only h/trial", "resubmits",
        ]
        return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols)

    def markdown_row(self) -> str:
        prof = self.profiler
        cells = [
            self.label, self.device, self.data_manager, self.store_device, str(self.data_dtype),
            str(self.mixed_precision), str(self.float32_matmul_precision), str(self.compile_mode),
            str(self.batch_size), f"{self.history_len}/{self.horizon_len}", f"{self.obs_dim}/{self.act_dim}",
            str(self.geometry_loss), f"{self.params / 1e6:.2f} M",
        ] + [f"{self.ms[p]:.1f}" for p in PHASES] + [
            f"{self.total_ms:.1f}", f"{self.val_ms:.1f}", f"{self.gpu_duty_cycle_pct:.0f}",
            "n/a" if self.gpu_util_pct is None else f"{self.gpu_util_pct:.0f}",
            "n/a" if (prof is None or not prof.cuda_kernels_per_step) else f"{prof.cuda_busy_pct:.0f}",
            "n/a" if (prof is None or not prof.cuda_kernels_per_step) else f"{prof.cuda_kernels_per_step:.0f}",
            "n/a" if self.peak_cuda_mem_mb is None else f"{self.peak_cuda_mem_mb:.0f} MB",
            f"{self.projected_epoch_hours:.3f}", f"{self.projected_trial_hours:.2f}",
            f"{self.approx_trial_hours:.2f}", str(self.resubmissions_needed),
        ]
        return "| " + " | ".join(cells) + " |"

    def markdown_report(self) -> str:
        lines = [
            f"# Training-step benchmark — {self.label}",
            "",
            f"- **date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **host**: {self.host}",
            f"- **device**: {self.device}" + (f" — {self.gpu}" if self.gpu else ""),
            f"- **torch**: {self.torch_version}",
            f"- **data path**: `{self.data_manager}`, store / dataset device: {self.store_device} / {self.dataset_device}"
            f" (num_workers={self.num_workers}, pin_memory={self.pin_memory}), batch dtype: {self.data_dtype}",
            f"- **model**: {self.params / 1e6:.3f} M trainable params, H={self.history_len} F={self.horizon_len}"
            f" W={self.output_window_len}, mixed_precision={self.mixed_precision}, torch.compile={self.compile_mode}",
            f"- **precision regime**: preset={self.precision_preset}, autocast={self.mixed_precision},"
            f" float32 matmul precision={self.float32_matmul_precision}"
            " (`highest` = TF32 off, the ICRA baselines' path; `high` / `medium` = TF32 on)",
            f"- **features**: Do={self.obs_dim} Da={self.act_dim}, handler={self.feature_handler}, obs groups: {self.obs_groups}",
            f"- **composed loss**: feature-geometry term {self.geometry_loss}"
            " (the horizon-looped orientation penalty; `off` on velocity-only datasets)",
            f"- **data**: {self.num_train_windows:,} train / {self.num_val_windows:,} val windows, B={self.batch_size}"
            f" -> {self.train_batches_per_epoch} train + {self.val_batches_per_epoch} val batches/epoch,"
            f" {self.epochs} epochs/trial",
            f"- **timing**: {self.timed_batches} timed training batches (+{self.warmup_batches} warm-up),"
            f" {self.timed_val_batches} validation batches, `torch.cuda.synchronize()` at every phase boundary",
            "",
            "## Per-batch phases (mean ms)",
            "",
            "| phase | ms | % of step |",
            "|---|---|---|",
        ]
        for p in PHASES:
            share = 100.0 * self.ms[p] / self.total_ms if self.total_ms else float("nan")
            lines.append(f"| {p} | {self.ms[p]:.1f} | {share:.1f} |")
        lines += [
            f"| **total (train)** | **{self.total_ms:.1f}** | 100 |",
            f"| validation step (median) | {self.val_ms:.1f} | — |",
            "| validation compute only (median, no loader / h2d) | "
            + ("n/a" if self.val_compute_ms is None else f"{self.val_compute_ms:.1f}") + " | — |",
            "",
            "## Assessment",
            "",
            f"- host-side share (loader + h2d): {100.0 - self.gpu_duty_cycle_pct:.0f} % of the step;"
            f" GPU duty-cycle upper bound: {self.gpu_duty_cycle_pct:.0f} %"
            + ("" if self.gpu_util_pct is None else f"; measured SM utilization: {self.gpu_util_pct:.0f} %")
            + ("" if (self.profiler is None or not self.profiler.cuda_kernels_per_step)
               else f"; profiler kernel-busy: {self.profiler.cuda_busy_pct:.0f} %"),
            f"- last training loss: {self.last_loss:.6g}"
            + ("" if self.peak_cuda_mem_mb is None else f"; peak CUDA memory: {self.peak_cuda_mem_mb:.0f} MB"),
            "",
            "## Projection (training + validation steps only; rollouts / checkpoints / logging excluded)",
            "",
            f"- **{self.projected_epoch_hours * 3600:.0f} s/epoch** ({self.projected_epoch_hours * 60:.1f} min,"
            f" {self.projected_epoch_hours:.3f} h) -- measured step, data path included",
            f"- **{self.projected_trial_hours:.1f} h/trial** (projected) for {self.epochs} epochs",
            f"- **{self.approx_trial_hours:.1f} h/trial compute-only bound** (forward + backward + optimizer"
            f" + val eval_score, no loader / h2d): the trial with a free data path;"
            f" data management = {self.data_path_share_pct:.0f} % of the projected trial",
            f"- SLURM wall-time {self.wall_time_hours:g} h -> "
            + ("fits in ONE submission" if self.resubmissions_needed == 0
               else f"needs {self.resubmissions_needed} `resume_from_checkpoint` resubmission(s)"),
        ]
        if self.profiler is not None:
            lines += [""] + self.profiler.markdown_section()
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        lines += ["", "## Aggregate row", "", self.markdown_header(), self.markdown_row(), ""]
        return "\n".join(lines)


# ==== Helpers ====================================================================================
class _Clock:
    """Wall-clock timer that drains the CUDA stream at every read (same contract as the FR15 gate)."""

    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda" and torch.cuda.is_available()

    def now(self) -> float:
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()


def _batch_to_device(batch: TransitionBatch, device: torch.device) -> TransitionBatch:
    """What Lightning's ``transfer_batch_to_device`` does per step: a no-op for a device-resident
    dataset, a synchronous H2D copy of the composed rows for a CPU one."""

    def mv(x):
        return x if (x is None or x.device == device) else x.to(device)

    return TransitionBatch(
        obs=mv(batch.obs), act=mv(batch.act), next_obs=mv(batch.next_obs),
        rewards=mv(batch.rewards), terminateds=mv(batch.terminateds), truncateds=mv(batch.truncateds),
    )


def _cycle(loader, n: int) -> Iterator[TransitionBatch]:
    """``n`` batches out of a (possibly shorter) ``DataLoader``, re-iterating like the next epoch would."""
    served = 0
    while served < n:
        for batch in loader:
            yield batch
            served += 1
            if served >= n:
                return


class _GpuUtilizationSampler:
    """SM utilization of ``device`` (``None`` when unavailable): pynvml through
    ``torch.cuda.utilization`` first, else the ``nvidia-smi`` binary the ``--nv`` bind exposes in the
    SIF (pynvml is NOT in the image, which left the column at ``n/a`` on the 2026-09-17 runs). A
    backend that fails once is not tried again, so the sampling stays off the timed path."""

    def __init__(self, device: torch.device):
        self.enabled = device.type == "cuda" and torch.cuda.is_available()
        if not self.enabled:
            self.index = -1  # CPU run: never touch `torch.cuda` (no driver on the dev laptop)
        else:
            self.index = device.index if device.index is not None else torch.cuda.current_device()
        self._pynvml_ok = self.enabled
        self._smi = shutil.which("nvidia-smi") if self.enabled else None

    def sample(self) -> Optional[float]:
        if not self.enabled:
            return None
        if self._pynvml_ok:
            try:
                return float(torch.cuda.utilization(self.index))
            except Exception:  # noqa: BLE001 -- pynvml missing or unsupported: fall back to nvidia-smi
                self._pynvml_ok = False
        if self._smi is not None:
            try:
                out = subprocess.run(
                    [self._smi, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                     f"--id={self._smi_index()}"],
                    capture_output=True, text=True, timeout=5, check=True,
                ).stdout.strip().splitlines()
                return float(out[0]) if out else None
            except Exception:  # noqa: BLE001 -- binary present but unusable: the metric is optional
                self._smi = None
        return None

    def _smi_index(self) -> str:
        """``nvidia-smi`` numbers the GPUs of ``CUDA_VISIBLE_DEVICES`` as the driver does, not as torch
        does: map the torch index back to the visible list when SLURM set one."""
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        ids = [v.strip() for v in visible.split(",") if v.strip()]
        if ids and self.index < len(ids):
            return ids[self.index]
        return str(self.index)

    @property
    def backend(self) -> Optional[str]:
        if not self.enabled:
            return None
        if self._pynvml_ok:
            return "pynvml"
        return "nvidia-smi" if self._smi is not None else None


class _TrainStep:
    """One production training step ``batch -> (loss scalar, phase clock stamps)``.

    The timed loop and the profiler pass share this object so the eager numerics stay exactly the
    historical ``zero_grad -> loss (autocast) -> backward -> optimizer.step`` sequence, while the
    RLRP-786 ``cuda-graph`` mode swaps the body for a ``CudaGraphTrainStep`` replay (the same
    kernels, one launch). ``stamps`` = ``(t_fwd_end, t_bwd_end, t_opt_end)`` from ``clock``; the
    graph mode reports the whole replay under ``forward`` (a graph has no phase boundaries).
    """

    def __init__(self, loss_fn: Callable, autocast: Callable, optimizer: torch.optim.Optimizer, clock: "_Clock") -> None:
        self._loss_fn = loss_fn
        self._autocast = autocast
        self._optimizer = optimizer
        self._clock = clock
        self.graph_step: Optional["CudaGraphTrainStep"] = None
        # cuda-graph mode: the EAGER host half of ``wrapper.loss`` (``_process_batch``: normalisation
        # + target composition, a host-synchronising mbrl-lib guard inside) -> the tensors the
        # captured half consumes. ``None`` in eager mode.
        self.prepare: Optional[Callable[[TransitionBatch], Tuple[torch.Tensor, ...]]] = None

    @property
    def mode(self) -> str:
        return "cuda-graph" if self.graph_step is not None else "eager"

    def __call__(self, batch: TransitionBatch) -> Tuple[torch.Tensor, Tuple[float, float, float]]:
        if self.graph_step is not None:
            with torch.no_grad():
                inputs = self.prepare(batch)
            loss, _meta = self.graph_step.step(inputs)
            t = self._clock.now()
            return loss, (t, t, t)
        self._optimizer.zero_grad(set_to_none=True)
        loss = _run_loss(self._loss_fn, self._autocast, batch)
        t2 = self._clock.now()
        loss.backward()
        t3 = self._clock.now()
        self._optimizer.step()
        t4 = self._clock.now()
        return loss, (t2, t3, t4)


def _make_train_step(
    wrapper: torch.nn.Module, optimizer: torch.optim.Optimizer, autocast: Callable, compile_mode: Optional[str],
    notes: List[str], clock: Optional["_Clock"] = None, device: Optional[torch.device] = None,
    warmup_batches: int = 3, allow_autocast: bool = False,
) -> Tuple[_TrainStep, Optional[str]]:
    """The production Lightning ``training_step`` body (``zero_grad`` -> ``loss`` under autocast ->
    ``backward`` -> ``optimizer.step``) as a :class:`_TrainStep` shared by the phase timing and the
    profiler. ``compile_mode`` wraps ``wrapper.loss`` in ``torch.compile`` (``COMPILE_MODES``) or,
    for ``cuda-graph`` (RLRP-786), records the whole step in a ``torch.cuda.CUDAGraph`` through
    ``CudaGraphTrainStep`` once the model reports no capture blocker (else eager + a note).

    :param allow_autocast: ``pipeline.benchmark.cuda_graph_allow_autocast`` (AR-baseline extension,
        FR4): capture a ``mixed_precision`` step too -- EXPLORATORY row. The captured region runs
        under ``torch.autocast(..., cache_enabled=False)`` (the cast cache is not replayable):
        one weight re-cast per unroll step instead of one per step, i.e. NOT the eager kernel
        sequence; parity is ``allclose`` only. Default ``False`` -> autocast rows stay eager + note.
    """
    loss_fn = wrapper.loss
    applied: Optional[str] = None
    clock = clock if clock is not None else _Clock(device if device is not None else torch.device("cpu"))
    train_step = _TrainStep(loss_fn, autocast, optimizer, clock)
    if compile_mode == CUDA_GRAPH_MODE:
        blockers = _cuda_graph_blockers(wrapper, optimizer, device, allow_autocast=allow_autocast)
        if blockers:
            notes.append(f"{CUDA_GRAPH_MODE} requested but NOT capturable, eager step timed instead: " + "; ".join(blockers))
            consol_msg_universal_one_liner(f"[warn] {notes[-1]}")
            return train_step, None
        from tools.torch_tools.cuda_graph_train_step import CudaGraphTrainStep

        model = wrapper.model
        autocast_dtype = getattr(wrapper, "resolve_autocast_dtype", lambda: None)() if allow_autocast else None

        def graph_loss_fn(inputs: Tuple[torch.Tensor, ...]):
            if autocast_dtype is not None:
                ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype, cache_enabled=False)
            else:
                ctx = contextlib.nullcontext()
            with ctx:
                loss, _meta = wrapper.loss_from_processed(*inputs)
            if isinstance(loss, tuple):
                loss = loss[0]
            return (loss.mean() if loss.ndim > 0 else loss), {}

        train_step.prepare = wrapper.process_batch_for_loss
        train_step.graph_step = CudaGraphTrainStep(
            loss_fn=graph_loss_fn, optimizer=optimizer, device=device,
            before_replay=model.advance_host_step_state, warmup_iters=max(1, min(int(warmup_batches), 3)),
        )
        if autocast_dtype is not None:
            notes.append(
                f"{CUDA_GRAPH_MODE}: autocast {str(autocast_dtype).replace('torch.', '')} under capture "
                "(cache_enabled=False, exploratory: per-unroll-step weight re-casts, allclose parity only)"
            )
        consol_msg_universal_one_liner(
            f"RLRP-786 {CUDA_GRAPH_MODE}: the training step is recorded in a torch.cuda.CUDAGraph after "
            f"{train_step.graph_step._warmup_iters} eager warm-up step(s) and REPLAYED for the timed steps "
            + ("(same kernels, one launch; phases fwd/bwd/opt collapse into `forward`)" if autocast_dtype is None
               else f"under torch.autocast({autocast_dtype}, cache_enabled=False) -- EXPLORATORY row")
        )
        return train_step, CUDA_GRAPH_MODE
    if compile_mode not in (None, "", "null", "None", "none", False):
        compile_mode = str(compile_mode)
        if compile_mode not in COMPILE_MODES:
            raise ValueError(f"pipeline.benchmark.compile_mode={compile_mode!r} not in {COMPILE_MODES + (CUDA_GRAPH_MODE,)}")
        try:
            # `meta[...] = ....item()` in the loss is a data-dependent op: let dynamo capture it
            # (same workaround as `algorithm.motion_model.utils.inference_optimization_utils`).
            torch._dynamo.config.capture_scalar_outputs = True
            loss_fn = torch.compile(wrapper.loss, mode=compile_mode)
            applied = compile_mode
            consol_msg_universal_one_liner(
                f"torch.compile(mode={compile_mode!r}) applied to the model loss for the timed steps "
                "(compile time lands in the warm-up batches; raise benchmark.warmup_batches if it recompiles)"
            )
        except Exception as exc:  # noqa: BLE001 -- the A/B knob must not kill the benchmark
            notes.append(f"torch.compile(mode={compile_mode!r}) unavailable, eager loss timed instead: {exc!r}")
    train_step._loss_fn = loss_fn
    return train_step, applied


def _cuda_graph_blockers(
    wrapper: torch.nn.Module, optimizer: torch.optim.Optimizer, device: Optional[torch.device],
    allow_autocast: bool = False,
) -> List[str]:
    """Why the RLRP-786 captured step cannot be used here (empty = go). Mirrors the production FR5 gate
    (``allow_autocast``: the exploratory ``pipeline.benchmark.cuda_graph_allow_autocast`` opt-in)."""
    blockers: List[str] = []
    if device is None or device.type != "cuda":
        blockers.append(f"device {device} is not CUDA")
    model = getattr(wrapper, "model", None)
    report = getattr(model, "cuda_graph_capture_blockers", None)
    if report is None:
        blockers.append(f"{type(model).__name__} does not implement cuda_graph_capture_blockers (MTM-Pro / AR MS2SS families)")
    else:
        blockers.extend(report())
    if not all(g.get("capturable", False) for g in optimizer.param_groups):
        blockers.append("optimizer is not capturable (pipeline.optimizer.adam_capturable: true + adam_fused on CUDA)")
    if getattr(wrapper, "resolve_autocast_dtype", lambda: None)() is not None and not allow_autocast:
        blockers.append(
            "mixed_precision autocast is on (the captured step is validated in fp32 only; "
            "pipeline.benchmark.cuda_graph_allow_autocast=true opts into the exploratory cache_enabled=False capture)"
        )
    if not (hasattr(wrapper, "process_batch_for_loss") and hasattr(wrapper, "loss_from_processed")):
        blockers.append(f"{type(wrapper).__name__} lacks the process_batch_for_loss / loss_from_processed seam")
    return blockers


def _run_loss(loss_fn: Callable, autocast: Callable, batch: TransitionBatch) -> torch.Tensor:
    with autocast():
        loss, _meta = loss_fn(batch)
    if isinstance(loss, tuple):
        loss = loss[0]
    return loss.mean() if loss.ndim > 0 else loss


def _self_device_us(evt: Any) -> float:
    """``FunctionEventAvg`` self device time in us across the torch profiler API renames."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(evt, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def _torch_profiler_pass(
    loader, train_step: _TrainStep, device: torch.device,
    num_batches: int, top_n: int, trace_path: Optional[str], clock: _Clock,
) -> ProfilerSummary:
    """``num_batches`` production training steps under ``torch.profiler`` -> :class:`ProfilerSummary`.

    The steps are extra (not the timed ones: the profiler adds per-op overhead) and run with the
    same batches, autocast and optimizer. ``record_shapes`` / ``with_stack`` are OFF to keep that
    overhead small; export the chrome trace (``trace_path``) when the op names are not enough."""
    from torch.profiler import DeviceType, ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if clock.cuda:
        activities.append(ProfilerActivity.CUDA)
    t0 = clock.now()
    with profile(activities=activities, record_shapes=False, with_stack=False, profile_memory=False) as prof:
        for host_batch in _cycle(loader, num_batches):
            batch = _batch_to_device(host_batch, device)
            train_step(batch)
        if clock.cuda:
            torch.cuda.synchronize()
    wall_ms = (clock.now() - t0) * 1e3 / max(num_batches, 1)

    averages = list(prof.key_averages())
    n = float(max(num_batches, 1))
    cuda_evts = [e for e in averages if getattr(e, "device_type", None) == DeviceType.CUDA]
    cpu_evts = [e for e in averages if getattr(e, "device_type", None) != DeviceType.CUDA]
    kernel_ms = sum(_self_device_us(e) for e in cuda_evts) / 1e3
    kernels = float(sum(int(e.count) for e in cuda_evts))
    cpu_calls = float(sum(int(e.count) for e in cpu_evts))

    def row(e) -> ProfilerOpRow:
        return ProfilerOpRow(
            name=str(e.key), calls_per_step=e.count / n,
            self_cpu_ms_per_step=float(e.self_cpu_time_total) / 1e3 / n,
            self_cuda_ms_per_step=_self_device_us(e) / 1e3 / n,
        )

    top_cpu = [row(e) for e in sorted(cpu_evts, key=lambda e: -float(e.self_cpu_time_total))[:top_n]]
    top_cuda = [row(e) for e in sorted(averages, key=lambda e: -_self_device_us(e))[:top_n] if _self_device_us(e) > 0]

    written: Optional[str] = None
    if trace_path:
        try:
            os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
            prof.export_chrome_trace(trace_path)
            written = trace_path
        except Exception as exc:  # noqa: BLE001 -- the trace is a convenience artifact
            consol_msg_universal_one_liner(f"[warn] chrome trace export failed: {exc!r}")

    return ProfilerSummary(
        batches=num_batches, wall_ms_per_step=wall_ms, cuda_kernel_ms_per_step=kernel_ms / n,
        cuda_kernels_per_step=kernels / n, cpu_op_calls_per_step=cpu_calls / n,
        top_cpu=top_cpu, top_cuda=top_cuda, trace_path=written,
    )


def _count_trainable_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _describe_loss_composition(wrapper) -> Tuple[Optional[str], Optional[str], str]:
    """``(feature_handler, obs_groups, geometry_loss)`` of the model under test, read from the
    ``FeatureGeometryLossMixin`` state of the wrapped ensemble (``wrapper.model``).

    Cross-dataset lever of the launch-bound diagnosis: the UAV datasets (NeuroBEM, PI-TCN) carry a
    quaternion group with a geodesic / chordal ``loss_term`` that the MS loss evaluates ONCE PER
    HORIZON STEP (``_feature_geometry_penalty_sequence``: F ``extra_loss`` calls per batch), the
    velocity-only Husky rows have no orientation group so the term is a no-op there. Comparing the
    two profiles tells the geometry term's share of ``forward + loss`` apart from the flat-forecast
    composition / normalizer chain both share.
    """
    ensemble = getattr(wrapper, "model", wrapper)
    handler = getattr(ensemble, "_feature_handler", None)
    handler_name = None if handler is None else type(handler).__name__
    groups_desc: Optional[str] = None
    geom_groups = 0
    groups = getattr(handler, "obs_groups", None)
    if groups is not None:
        parts = []
        for g in groups:
            kind = getattr(getattr(g, "kind", None), "value", getattr(g, "kind", "?"))
            idx = list(getattr(g, "indices", ()))
            span = f"{idx[0]}:{idx[-1] + 1}" if idx else ""
            has_term = getattr(g, "loss_term", None) is not None
            geom_groups += int(has_term)
            parts.append(f"{kind}[{span}]" + ("*" if has_term else ""))
        groups_desc = ", ".join(parts) if parts else "none"
    weight = float(getattr(ensemble, "_feature_geometry_loss_weight", 0.0) or 0.0)
    objective = getattr(ensemble, "_feature_geometry_loss_objective", None)
    active_fn = getattr(ensemble, "_feature_loss_active", None)
    active = bool(active_fn()) if callable(active_fn) else (handler is not None and weight != 0.0 and objective is not None)
    if not active or geom_groups == 0:
        reason = (
            "no handler" if handler is None else
            "weight 0" if weight == 0.0 else
            "objective null" if objective is None else
            "no orientation group"
        )
        return handler_name, groups_desc, f"off ({reason})"
    return handler_name, groups_desc, f"{objective} x{weight:g} on {geom_groups} group(s)"


def _resolve_epochs(cfg: omegaconf.DictConfig) -> int:
    """Optimizer epochs per trial as the FR15 gate counts them: the single global loop budget
    (``final_max_num_epochs_train_model``), else the UDER pass budget (``num_epochs_train_model``)."""
    for key in ("final_max_num_epochs_train_model", "num_epochs_train_model"):
        value = cfg.UDER.get(key, None)
        if value is not None:
            return int(value)
    raise ValueError("cannot resolve the epoch budget: UDER.final_max_num_epochs_train_model / num_epochs_train_model unset")


def _resolve_label(cfg: omegaconf.DictConfig) -> str:
    exp = str(omegaconf.OmegaConf.select(cfg, "overrides.experiment", default="benchmark"))
    trial = omegaconf.OmegaConf.select(cfg, "trial_nb", default=None)
    label = os.path.basename(exp.rstrip("/"))
    if trial is not None:
        label += f" trial {trial}"
    return label


def _write_reports(
    cfg: omegaconf.DictConfig,
    result: TrainStepBenchmarkResult,
    exp_dir: str,
    append_report_to: Optional[str] = None,
) -> None:
    """Per-run ``train_step_benchmark.{md,json}`` in *exp_dir* (+ the dev-only aggregate row when
    *append_report_to* is set -- relative paths resolve from ``cfg.project_root_path``)."""
    os.makedirs(exp_dir, exist_ok=True)
    md_path = os.path.join(exp_dir, f"{REPORT_BASENAME}.md")
    json_path = os.path.join(exp_dir, f"{REPORT_BASENAME}.json")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(result.markdown_report())
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, default=str)
    consol_msg_universal_one_liner(f"Benchmark report written to '{md_path}' (+ .json)")

    if append_report_to in _NULLS:
        return
    append_to = os.fspath(append_report_to)
    if not os.path.isabs(append_to):
        append_to = os.path.join(cfg.project_root_path, append_to)
    os.makedirs(os.path.dirname(append_to), exist_ok=True)
    write_header = not os.path.isfile(append_to) or os.path.getsize(append_to) == 0
    with open(append_to, "a", encoding="utf-8") as fh:
        if write_header:
            fh.write(f"# Training-step benchmark aggregate — {result.host}\n\n")
            fh.write(result.markdown_header() + "\n")
        fh.write(result.markdown_row() + "\n")
    consol_msg_universal_one_liner(f"Benchmark aggregate row appended to '{append_to}'")


# ==== Precision preset ===========================================================================
def _apply_precision_preset(settings: TrainStepBenchmarkSettings, wrapper, notes: List[str]) -> Optional[str]:
    """DEV-ONLY ``settings.precision`` (``pipeline.benchmark.precision``) -> the two production knobs,
    applied to THIS process / model. ``None`` on the publication path (regime as composed).

    ``null`` (default) leaves the run as configured (``ms_model.mixed_precision`` of the cfg and the
    ``torch_backend.cuda_high_precision_float32`` the app setup applied). A preset name (see
    :data:`PRECISION_PRESETS`) sets ``mixed_precision`` on the constructed ensemble (``wrapper.model``,
    where the mbrl ``Model.resolve_autocast_dtype`` reads it -- so it works for EVERY model family, not
    only the MS->MS one whose constructor exposes the kwarg) and re-applies the TF32 switch through the
    single writer ``configure_cuda_float32_high_precision``. Meant to be SWEPT
    (``pipeline.benchmark.precision: fp32,tf32,bf16``): one key, one row per numerical regime.
    """
    raw = settings.precision
    if raw in _NULLS:
        return None
    preset = str(raw).strip().lower()
    if preset not in PRECISION_PRESETS:
        raise ValueError(f"pipeline.benchmark.precision={raw!r} not in {sorted(PRECISION_PRESETS)} (or null)")
    mixed_precision, high_precision_fp32 = PRECISION_PRESETS[preset]
    ensemble = getattr(wrapper, "model", wrapper)
    previous = getattr(ensemble, "mixed_precision", None)
    ensemble.mixed_precision = mixed_precision
    if previous is not None and previous != mixed_precision:
        notes.append(f"precision preset {preset!r} overrides ms_model.mixed_precision={previous!r} -> {mixed_precision!r}")
    configure_cuda_float32_high_precision(enable=high_precision_fp32)
    consol_msg_universal_one_liner(
        f"Precision preset {preset!r}: autocast={mixed_precision}, TF32 {'OFF' if high_precision_fp32 else 'ON'} "
        f"(float32 matmul precision = {torch.get_float32_matmul_precision()!r})"
    )
    return preset


# ==== Data paths =================================================================================
@dataclass
class _DataPath:
    """What the timed loop needs from EITHER training data path, resolved the way the ERLL does it
    (``abstract_experience_replay_learning_loop``: seeded split, iterators, device placement)."""

    data_manager: str
    train_loader: Any  # ``DataLoader`` (dataloader path) | ``BootstrapIterator`` (replay-buffer path)
    val_loader: Any  # ``DataLoader`` | ``TransitionIterator`` | ``None``
    num_train: int
    num_val: int
    history_len: int
    horizon_len: int
    output_window_len: int
    obs_dim: int
    act_dim: int
    store_device: str
    dataset_device: torch.device
    num_workers: int
    pin_memory: bool
    normalizer_fit_batch: TransitionBatch
    feature_weights_source: Any  # what ``resolve_and_apply_feature_loss_weights`` reads the statistics from


SOURCE_DTYPES: Dict[str, Any] = {"float32": np.float32, "float64": np.float64}


def _resolve_train_step_mode(
    cfg: omegaconf.DictConfig, settings: TrainStepBenchmarkSettings, notes: List[str]
) -> Tuple[Optional[str], bool]:
    """Which training step the timed pass runs: ``(compile_mode, cuda_graph_allow_autocast)``.

    Two writers, the EXPERIMENT cfg being authoritative when the dev lever is unset:

    - ``settings.compile_mode`` (DEV-ONLY ``pipeline.benchmark.compile_mode`` of the dev profiler:
      ``null`` / torch.compile mode / ``cuda-graph``) -- when set it wins. Always ``None`` on the
      publication path.
    - the production RLRP-786 opt-in ``pipeline.cuda_graph_training_step: true`` of the composed
      experiment cfg (e.g. ``multirun-DMtm-Pro-MS+CP+cudagraph.yaml``) -> ``cuda-graph``. Without
      this mapping a ``+cudagraph`` experiment would be timed EAGER: the bench times
      ``wrapper.loss -> backward -> optimizer.step`` itself, not the Lightning ``training_step``
      the production key routes through ``CudaGraphTrainStep``.

    ``cuda_graph_allow_autocast`` is the OR of the dev lever and the production
    ``pipeline.cuda_graph_allow_autocast`` (both are the same FR4 exploratory opt-in).
    """
    raw = settings.compile_mode
    compile_mode: Optional[str] = None if raw in _NULLS else str(raw)
    production_graph = bool(
        omegaconf.OmegaConf.select(cfg, "pipeline.cuda_graph_training_step", default=False)
    )
    if compile_mode is None and production_graph:
        compile_mode = CUDA_GRAPH_MODE
        notes.append(
            f"{CUDA_GRAPH_MODE} step selected by the experiment cfg (pipeline.cuda_graph_training_step=true)"
        )
    elif compile_mode is not None and compile_mode != CUDA_GRAPH_MODE and production_graph:
        notes.append(
            f"pipeline.benchmark.compile_mode={compile_mode!r} overrides the experiment cfg's "
            "pipeline.cuda_graph_training_step=true (captured step NOT timed)"
        )
    allow_autocast = bool(settings.cuda_graph_allow_autocast) or bool(
        omegaconf.OmegaConf.select(cfg, "pipeline.cuda_graph_allow_autocast", default=False)
    )
    return compile_mode, allow_autocast


def _resolve_source_dtype(settings: TrainStepBenchmarkSettings) -> Optional[Any]:
    """DEV-ONLY ``settings.source_dtype`` -> numpy dtype (``None`` = the cached buffer's own dtype)."""
    raw = settings.source_dtype
    if raw in _NULLS:
        return None
    key = str(raw).strip().lower().replace("torch.", "").replace("numpy.", "").replace("np.", "")
    key = {"fp32": "float32", "fp64": "float64", "double": "float64", "single": "float32"}.get(key, key)
    if key not in SOURCE_DTYPES:
        raise ValueError(f"pipeline.benchmark.source_dtype={raw!r} not in {sorted(SOURCE_DTYPES)} (or null)")
    return SOURCE_DTYPES[key]


def _cast_replay_buffer(replay_buffer, np_dtype) -> Any:
    """Re-materialise a multistep ``ReplayBuffer`` with ``obs`` / ``action`` in *np_dtype* (same
    capacity, shapes, reward dtype, rng, device; one vectorised ``add_batch``). No-op (same object) when
    the buffer already has that dtype.

    Why it exists: the cached ICRA buffers (``.../ms_replaybuffer/<dataset>/.../replaybuffer_spec.yaml``,
    ``obs_type: numpy.float64`` on NeuroBEM, PI-TCN and Husky) were generated BEFORE the RLRP-824 dtype
    rule and ``load_multistep_replaybuffer_with_spec`` honours the spec dtype, so the replay-buffer
    path reproduces the float64-data regime of the ICRA runs by default. That float64 row is a REFERENCE
    ANCHOR to the ``artifact/ICRA2026`` results only (no model / normalizer is going to be trained in
    double precision); the float32 row is the regime every future run gets and is what the cast measures."""
    from mbrl.util.replay_buffer import ReplayBuffer

    if np.dtype(replay_buffer.obs_type) == np.dtype(np_dtype) and np.dtype(replay_buffer.action_type) == np.dtype(np_dtype):
        return replay_buffer
    if replay_buffer.stores_trajectories:
        raise NotImplementedError(
            "pipeline.benchmark.source_dtype: casting a trajectory-indexed replay buffer is not wired "
            "(the cached multistep buffers do not store trajectories)."
        )
    casted = ReplayBuffer(
        capacity=int(replay_buffer.capacity),
        obs_shape=tuple(replay_buffer.obs_shape),
        action_shape=tuple(replay_buffer.action_shape),
        obs_type=np_dtype,
        action_type=np_dtype,
        reward_type=replay_buffer.reward_type,
        rng=replay_buffer.rng,
        max_trajectory_length=None,
        device=replay_buffer.device,
    )
    obss, actions, next_obss, rewards, terminateds, truncateds = replay_buffer.get_all().astuple()
    casted.add_batch(obss, actions, next_obss, rewards, terminateds, truncateds)
    return casted


def resolve_env_family(cfg: omegaconf.DictConfig) -> str:
    """``robotic_3d_env`` | ``math_env`` from the composed cfg's ``pipeline.name`` prefix (the group the
    experiment cfg selects: ``robotic_3d_env_full_pipeline_multirun``, ``math_env_full_pipeline_multirun``,
    ...). Fail loud on anything else: the core only knows how to build THESE training sources."""
    name = str(omegaconf.OmegaConf.select(cfg, "pipeline.name", default="") or "")
    for family in ENV_FAMILIES:
        if name.startswith(family):
            return family
    raise ValueError(
        f"training-speed benchmark: cannot resolve the environment family from pipeline.name={name!r} "
        f"(expected a prefix in {ENV_FAMILIES}; is `source_experiment_cfg.path` an experiment multirun cfg dir?)"
    )


def _load_robotic_3d_env_training_source(cfg: omegaconf.DictConfig, headless: bool) -> Any:
    """Robotic 3D env: the CACHED source, exactly as ``robotic_3d_env_full_pipeline.execute`` resolves it
    (``dataloader`` -> HDF5 ``SingleStepTrajectoryStore``; ``replay-buffer`` -> cached multistep buffer)."""
    from pipeline.robotic_3d_env import (
        robotic_3d_env_generate_ms_hdf5_pipeline,
        robotic_3d_env_generate_ms_replaybuffer_pipeline,
    )

    if is_dataloader_data_manager(cfg):
        return robotic_3d_env_generate_ms_hdf5_pipeline.load_or_generate(cfg, headless)
    replay_buffer = None
    if not cfg.pipeline.get("force_regenerating_saved_ms_replaybuffer", False):
        replay_buffer = load_multistep_replaybuffer_with_spec(
            cfg,
            override_load_dir=os.path.realpath(os.path.join(cfg.project_root_path, cfg.environment.data_path)),
        )
    if replay_buffer is None:
        replay_buffer = robotic_3d_env_generate_ms_replaybuffer_pipeline.execute(cfg, headless)
    return replay_buffer


def _load_math_env_training_source(cfg: omegaconf.DictConfig, headless: bool, exp_dir: str) -> Any:
    """Math env: GENERATE the single-step rollouts the way ``math_env_full_pipeline`` does
    (``setup_train_and_val_source_data``: gym math env, noise cfg, seeded), then the path the experiment
    trains on (``math_env_ms_model_train_and_deploy``): ``dataloader`` -> in-memory
    ``SingleStepTrajectoryStore``; ``replay-buffer`` -> the aggregated multistep ``ReplayBuffer``
    restricted to the explorable region. The InD/OOD test-target rollouts are NOT generated (no deploy)."""
    from pipeline.pipeline_utils.general.setup import setup_source_multi_step_replay_buffer
    from pipeline.pipeline_utils.math_env_pipeline_utils.setup_utils import (
        setup_train_and_val_source_data,
    )
    from tools.multistep_tools.window_dataset.pipeline_utils import (
        build_store_from_ss_replay_buffers,
        stamp_environment_shapes_from_data,
    )

    _label, ss_full_time_space_replay_buffers, _val_env_trjs = setup_train_and_val_source_data(
        cfg, exp_dir, headless
    )
    if is_dataloader_data_manager(cfg):
        stamp_environment_shapes_from_data(
            cfg,
            ss_full_time_space_replay_buffers[0].obs_shape[-1],
            ss_full_time_space_replay_buffers[0].action_shape[-1],
        )
        return build_store_from_ss_replay_buffers(cfg, ss_full_time_space_replay_buffers)
    _processor, ms_replay_buffer = setup_source_multi_step_replay_buffer(cfg, ss_full_time_space_replay_buffers)
    return ms_replay_buffer


def _load_training_source(
    cfg: omegaconf.DictConfig,
    settings: TrainStepBenchmarkSettings,
    headless: bool,
    notes: List[str],
    exp_dir: str = ".",
) -> Tuple[str, Any]:
    """The training source BEFORE the model is built (both paths stamp the environment shapes from
    the data): ``("dataloader", SingleStepTrajectoryStore)`` or ``("replay-buffer", ReplayBuffer)``,
    resolved per environment family (:func:`resolve_env_family`) exactly as the experiment's own full
    pipeline resolves them.

    DEV-ONLY ``settings.source_dtype`` (``null | float32 | float64``) re-materialises the replay buffer in
    that dtype (:func:`_cast_replay_buffer`) so the two data regimes -- the cached float64 buffers the
    ICRA runs trained on (reference anchor only) vs. the float32 buffers the current dtype rule
    regenerates (every future run) -- can be swept next to ``precision`` by the dev profiler. On the
    ``dataloader`` path the store dtype is the normalizer dtype by construction
    (``resolve_source_data_torch_dtype``); the knob is ignored there with a note."""
    source_dtype = _resolve_source_dtype(settings)
    family = resolve_env_family(cfg)
    use_dataloader = is_dataloader_data_manager(cfg)
    if use_dataloader and source_dtype is not None:
        notes.append(
            f"pipeline.benchmark.source_dtype={np.dtype(source_dtype).name} ignored on the dataloader path "
            "(the store dtype follows the normalizer dtype: one_dim_transition_model.normalize_double_precision)"
        )
    if family == MATH_ENV_FAMILY:
        source = _load_math_env_training_source(cfg, headless, exp_dir)
    else:
        source = _load_robotic_3d_env_training_source(cfg, headless)
    if use_dataloader:
        return "dataloader", source
    replay_buffer = source
    if source_dtype is not None:
        loaded_dtype = np.dtype(replay_buffer.obs_type).name
        replay_buffer = _cast_replay_buffer(replay_buffer, source_dtype)
        consol_msg_universal_one_liner(
            f"Source replay buffer dtype: {loaded_dtype} (cached) -> {np.dtype(replay_buffer.obs_type).name} "
            f"(pipeline.benchmark.source_dtype)"
        )
        if loaded_dtype != np.dtype(source_dtype).name:
            notes.append(f"source replay buffer re-materialised {loaded_dtype} -> {np.dtype(source_dtype).name} (pipeline.benchmark.source_dtype)")
    return "replay-buffer", replay_buffer


def _build_dataloader_path(cfg: omegaconf.DictConfig, store, motion_model_container, device: torch.device) -> _DataPath:
    """RLRP-824 lazy window path: device-resident store -> ``MultistepWindowDataset`` ->
    ``WindowDataLoaderDataSource`` with the run's seeded sample-level split (FR5) and ONE DataLoader pair."""
    store = move_store_to_training_device(cfg, store, device)
    window_dataset = build_window_dataset(cfg, store, output_window_len=motion_model_container.output_window_len)
    motion_model_container.validate_with_window_dataset(window_dataset)
    stamp_original_dataset_size(cfg, len(window_dataset))
    consol_msg_universal_one_liner(describe_window_path(store, window_dataset))
    erll_source = build_window_data_source(cfg, window_dataset, device=device)
    source_size = cfg.source_replay_buffer.source_size
    erll_source.split(
        datasets_size=None if source_size == "all" else int(source_size),
        val_ratio=float(cfg.source_replay_buffer.val_ratio),
    )
    batch_size = int(cfg.UDER.batch_size.init_value)
    train_loader = erll_source.train_iterable(batch_size)
    val_loader = erll_source.val_iterable(batch_size)
    consol_msg_universal_one_liner(f"ERLL data source: {erll_source!r}")
    return _DataPath(
        data_manager="dataloader",
        train_loader=train_loader,
        val_loader=val_loader,
        num_train=len(train_loader.dataset),
        num_val=0 if val_loader is None else len(val_loader.dataset),
        history_len=int(window_dataset.history_len),
        horizon_len=int(window_dataset.horizon_len),
        output_window_len=int(window_dataset.output_window_len),
        obs_dim=int(window_dataset.obs_dim),
        act_dim=int(window_dataset.act_dim),
        store_device=str(store.device),
        dataset_device=erll_source.dataset_device,
        num_workers=int(getattr(erll_source, "num_workers", 0)),
        pin_memory=bool(getattr(erll_source, "pin_memory", False)),
        normalizer_fit_batch=erll_source.normalizer_fit_batch(),
        feature_weights_source=erll_source,
    )


def _build_replay_buffer_path(
    cfg: omegaconf.DictConfig, replay_buffer, motion_model_container, wrapper, device: torch.device
) -> _DataPath:
    """Legacy materialized multistep buffer path (every non-MS2MS baseline of the ICRA experiment):
    the ERLL's ``split_replay_buffer`` under the run seed on the device ``resolve_replay_buffer_device``
    picks (``mbrl_lib.keep_replay_buffer_on_device`` + VRAM guard, RLRP-775 A17 / A19), then the
    ``get_basic_buffer_iterators`` pair ``ModelTrainer.train`` consumes (a ``BootstrapIterator`` yields
    an ``(E, B, ...)`` batch when ``num_members > 1``, a flat batch otherwise). Sequence-batch AR
    models (``receive_sequence_batch: true``, none in the ICRA cfgs) are not covered."""
    from tools.feature_handling_tools.env_handlers import resolve_act_shape, resolve_obs_shape
    from tools.mbrl_lib_tools.setup_utils import resolve_replay_buffer_device
    from tools.multistep_tools.models.autoregressive_sequence_iterator import AutoRegressiveSequenceIterator

    motion_model_container.validate_with_replay_buffer(replay_buffer)
    ensemble = wrapper.model
    if isinstance(ensemble, AutoRegressiveSequenceIterator) and getattr(ensemble, "receive_sequence_batch", False):
        raise NotImplementedError(
            "robotic_3d_env_train_step_dev_profiling_pipeline: `receive_sequence_batch: true` models need the ERLL "
            "`get_sequence_buffer_iterator` path, not wired in the benchmark (no ICRA cfg uses it)."
        )
    source_size = cfg.source_replay_buffer.source_size
    datasets_size = int(replay_buffer.num_stored) if source_size == "all" else int(source_size)
    val_ratio = float(cfg.source_replay_buffer.val_ratio)
    buffer_max_trajectory_len = cfg.UDER.get("buffer_max_trajectory_len", None)
    train_val_device = resolve_replay_buffer_device(
        cfg,
        capacity=datasets_size,
        obs_shape=tuple(replay_buffer.obs_shape),
        action_shape=tuple(replay_buffer.action_shape),
        obs_type=replay_buffer.obs_type,
        action_type=replay_buffer.action_type,
        reward_type=replay_buffer.reward_type,
        max_trajectory_length=buffer_max_trajectory_len,
    )
    train_rb, val_rb = split_replay_buffer(
        replay_buffer,
        datasets_size=datasets_size,
        val_ratio=val_ratio,
        buffer_max_trajectory_len=buffer_max_trajectory_len,
        device=train_val_device,
        seed=cfg.get("seed", None),
    )
    consol_msg_universal_one_liner(
        f"Train replay buffer size: {train_rb.num_stored}, validation replay buffer size: {val_rb.num_stored}"
        f" (device: {train_rb.device if train_rb.device is not None else 'cpu'})"
    )
    batch_size = int(cfg.UDER.batch_size.init_value)
    num_members = int(getattr(ensemble, "num_members", 1))
    train_iter, _ = common_utils.get_basic_buffer_iterators(
        train_rb, batch_size, val_ratio=0, ensemble_size=num_members, shuffle_each_epoch=True
    )
    val_iter = None
    if val_rb.num_stored > 0:
        val_iter, _ = common_utils.get_basic_buffer_iterators(
            val_rb, batch_size, val_ratio=0, ensemble_size=num_members, shuffle_each_epoch=True
        )
    rb_device = torch.device(train_rb.device) if train_rb.device is not None else torch.device("cpu")
    return _DataPath(
        data_manager="replay-buffer",
        train_loader=train_iter,
        val_loader=val_iter,
        num_train=int(train_rb.num_stored),
        num_val=int(val_rb.num_stored),
        history_len=int(cfg.ms_model.history_len),
        horizon_len=int(cfg.ms_model.horizon_len),
        output_window_len=int(motion_model_container.output_window_len),
        # Single-step feature widths (the buffer rows are the flattened ``H`` window).
        obs_dim=int(resolve_obs_shape(cfg)[0]),
        act_dim=int(resolve_act_shape(cfg)[0]),
        store_device=str(replay_buffer.device if replay_buffer.device is not None else "cpu"),
        dataset_device=rb_device,
        num_workers=0,
        pin_memory=False,
        normalizer_fit_batch=replay_buffer.get_all(),
        feature_weights_source=replay_buffer,
    )


def _batch_dtype(batch: TransitionBatch) -> str:
    obs = getattr(batch, "obs", None)
    return str(getattr(obs, "dtype", type(obs).__name__)).replace("torch.", "")


# ==== Pipeline ===================================================================================
def execute_one(
    cfg: omegaconf.DictConfig,
    headless: bool = False,
    report_dir: Optional[str] = None,
    settings: Optional[TrainStepBenchmarkSettings] = None,
    label: Optional[str] = None,
) -> TrainStepBenchmarkResult:
    """Time the training step of ONE already-composed experiment cfg (robotic 3D env or math env).

    :param cfg: Hydra configuration of the EXPERIMENT (main-experiment + model + its ``/pipeline``
        group). Never mutated for benchmark purposes: what is timed is what the cfg trains.
    :param headless: turn environment rendering off (param for commandline flag)
    :param report_dir: where ``train_step_benchmark.*`` is written; ``None`` -> the Hydra run dir
        (dev profiler behaviour). The multi-model driver passes one sub-directory per model so
        successive models do not overwrite each other's per-model report.
    :param settings: measurement knobs (:class:`TrainStepBenchmarkSettings`). ``None`` -> the DEV
        profiler contract ``TrainStepBenchmarkSettings.from_pipeline_cfg(cfg)`` (reads
        ``cfg.pipeline.benchmark`` incl. the dev-only regime levers). The publication driver ALWAYS
        passes its own launcher-level settings, so ``cfg.pipeline.benchmark`` is never consulted.
    :param label: report label; ``None`` -> derived from ``overrides.experiment`` / ``trial_nb``.
    :return: the benchmark result (also written to ``report_dir`` as train_step_benchmark.*)
    """
    exp_dir_relative_path = report_dir if report_dir is not None else get_hydra_experiment_cwd(cfg)
    seed_me(cfg, output_torch_rdn_generator=True)
    if settings is None:
        settings = TrainStepBenchmarkSettings.from_pipeline_cfg(cfg)
    notes: List[str] = []
    compile_mode_requested, cuda_graph_allow_autocast = _resolve_train_step_mode(cfg, settings, notes)

    # .... Configuration setting validation .......................................................
    uder_cfg_validation(cfg)
    if compile_mode_requested == CUDA_GRAPH_MODE:
        # RLRP-786: the captured ``optimizer.step()`` needs the Adam ``step`` counter on the device
        # (``capturable=True``, combined with the fused kernel by ``_resolve_adam_fused_kwargs``).
        # ``change_optimizer(cfg.ms_training, ...)`` reads ``ms_training.optimizer.adam_capturable``
        # (``ms_training: ${training_common}`` may already be materialised -> set both).
        for _group_key in ("training_common", "ms_training"):
            _group = cfg.get(_group_key, None)
            if _group is not None and _group.get("optimizer", None) is not None:
                with omegaconf.open_dict(_group.optimizer):
                    _group.optimizer.adam_capturable = True

    # :::: Same setup as the full pipeline (source -> model/trainer -> split / iterators) :::::::::
    # ``dataloader``: HDF5 store -> lazy window dataset -> ERLL ``WindowDataLoaderDataSource``;
    # ``replay-buffer``: cached multistep buffer -> ERLL ``split_replay_buffer`` -> basic iterators.
    data_manager, training_source = _load_training_source(
        cfg, settings, headless, notes, exp_dir=exp_dir_relative_path
    )

    motion_model_container, ms_trainer = setup_multistep_step_model_and_trainer(cfg)
    device = torch.device(motion_model_container.dynamics_model.device)
    wrapper = ms_trainer.model
    precision_preset = _apply_precision_preset(settings, wrapper, notes)

    if data_manager == "dataloader":
        data_path = _build_dataloader_path(cfg, training_source, motion_model_container, device)
    else:
        data_path = _build_replay_buffer_path(cfg, training_source, motion_model_container, wrapper, device)

    # The ERLL's normalizer fit on the source (abstract_experience_replay_learning_loop).
    if hasattr(wrapper, "update_normalizer"):
        consol_msg_universal_one_liner("Updates normalizer statistics using source dataset")
        wrapper.update_normalizer(data_path.normalizer_fit_batch)
        from tools.feature_handling_tools.feature_loss_weights import (
            resolve_and_apply_feature_loss_weights,
        )

        resolve_and_apply_feature_loss_weights(cfg, wrapper, data_path.feature_weights_source)

    batch_size = int(cfg.UDER.batch_size.init_value)
    train_loader = data_path.train_loader
    val_loader = data_path.val_loader

    # :::: Timed training steps :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    warmup = int(settings.warmup_batches)
    num_batches = int(settings.num_batches)
    num_val_batches = int(settings.val_batches)
    optimizer = ms_trainer.optimizer
    autocast = wrapper._autocast_context  # the Lightning ``training_step`` autocast (mixed_precision)
    clock = _Clock(device)
    train_step, compile_mode = _make_train_step(
        wrapper, optimizer, autocast, compile_mode_requested, notes, clock=clock, device=device,
        warmup_batches=warmup, allow_autocast=cuda_graph_allow_autocast,
    )
    util_sampler = _GpuUtilizationSampler(device)

    wrapper.train()
    if clock.cuda:
        torch.cuda.reset_peak_memory_stats(device)
    sums = {p: 0.0 for p in PHASES}
    timed = 0
    last_loss = float("nan")
    util_samples: List[float] = []

    consol_msg_universal_one_liner(
        f"Timing {num_batches} training batches (+{warmup} warm-up) of B={batch_size} on {device} ..."
    )
    data_dtype: Optional[str] = None
    t_prev = clock.now()
    for i, host_batch in enumerate(_cycle(train_loader, warmup + num_batches)):
        t0 = clock.now()  # the loader gather finished when the iterator yielded
        if data_dtype is None:
            data_dtype = _batch_dtype(host_batch)
        batch = _batch_to_device(host_batch, device)
        t1 = clock.now()
        loss_scalar, (t2, t3, t4) = train_step(batch)
        if i >= warmup:
            sums["loader"] += (t0 - t_prev) * 1e3
            sums["h2d"] += (t1 - t0) * 1e3
            sums["forward"] += (t2 - t1) * 1e3
            sums["backward"] += (t3 - t2) * 1e3
            sums["optimizer"] += (t4 - t3) * 1e3
            timed += 1
            util = util_sampler.sample()
            if util is not None:
                util_samples.append(util)
        if i == warmup + num_batches - 1 or (i + 1) % max(1, (warmup + num_batches) // 5) == 0:
            last_loss = float(loss_scalar.detach().float().cpu())
            consol_msg_universal_one_liner(
                f"  batch {i + 1}/{warmup + num_batches}: loss={last_loss:.6g}"
                + ("" if i < warmup else f", step={(t4 - t_prev) * 1e3:.0f} ms")
            )
        t_prev = clock.now()
    if not math.isfinite(last_loss):
        notes.append(f"non-finite training loss observed ({last_loss})")

    # :::: Timed validation steps (eval_score under the same autocast, no backward) ::::::::::::::::
    # The first eval-mode batches carry one-off costs (cudnn.benchmark autotune of the no_grad
    # forward, first-call allocations): RLRP-824 round 3 measured 228 ms/val batch vs 88 ms/train
    # batch on the Husky F500 control with only 20 timed batches and no warm-up. Discard a few.
    # The steady-state figure is the MEDIAN of the timed batches: when the timed window reaches the
    # ragged last batch of the val loader (new shape -> one more cudnn autotune / autocast re-cast,
    # paid once per trial, not per epoch), a single 1-2 s batch would otherwise inflate a 10-batch
    # mean 5-10x (RLRP-835 Husky job 3629916: E2E-TCN-F50 158 ms vs 16 ms train, AR-TCN-bf16-cg
    # 144 ms vs 35 ms train). The mean is kept as a note when it diverges.
    # The eval_score-only time (from the batch being on the device to the score being ready) is
    # kept apart as ``val_compute_ms``: the validation term of the compute-only bound (approx_*).
    val_ms = float("nan")
    val_compute_ms: Optional[float] = None
    timed_val = 0
    if val_loader is not None and num_val_batches > 0:
        val_warmup = min(VAL_WARMUP_BATCHES, warmup)
        wrapper.eval()
        val_samples_ms: List[float] = []
        val_compute_samples_ms: List[float] = []
        t_prev = clock.now()
        with torch.no_grad():
            for j, host_batch in enumerate(_cycle(val_loader, val_warmup + num_val_batches)):
                batch = _batch_to_device(host_batch, device)
                t_ready = clock.now()  # loader gather + h2d done
                with autocast():
                    wrapper.eval_score(batch)
                t_now = clock.now()
                if j >= val_warmup:
                    val_samples_ms.append((t_now - t_prev) * 1e3)
                    val_compute_samples_ms.append((t_now - t_ready) * 1e3)
                    timed_val += 1
                t_prev = t_now
        val_ms, val_note = summarize_val_batch_times(val_samples_ms)
        val_compute_ms, _ = summarize_val_batch_times(val_compute_samples_ms)
        if val_note is not None:
            notes.append(val_note)
        wrapper.train()
    elif val_loader is None:
        notes.append("no validation loader (val_ratio == 0): validation time not projected")

    # :::: Optional torch.profiler pass (extra steps, top-N ops, kernel count / kernel-busy share) ::
    profiler_summary: Optional[ProfilerSummary] = None
    if bool(settings.torch_profiler):
        prof_batches = int(settings.torch_profiler_batches)
        top_n = int(settings.torch_profiler_top_n)
        trace_path = (
            os.path.join(exp_dir_relative_path, PROFILER_TRACE_BASENAME)
            if bool(settings.torch_profiler_export_trace) else None
        )
        consol_msg_universal_one_liner(f"torch.profiler pass: {prof_batches} training batches, top {top_n} ops ...")
        try:
            profiler_summary = _torch_profiler_pass(
                train_loader, train_step, device, prof_batches, top_n, trace_path, clock
            )
            if profiler_summary.cuda_kernels_per_step and profiler_summary.cuda_busy_pct < 50.0:
                notes.append(
                    f"LAUNCH-BOUND: profiler kernel-busy {profiler_summary.cuda_busy_pct:.0f} % over "
                    f"{profiler_summary.cuda_kernels_per_step:.0f} kernels/step -> the Python thread issuing kernels "
                    "paces the step; see the top self-CPU ops"
                )
        except Exception as exc:  # noqa: BLE001 -- the profiler pass is diagnostic, never fatal
            notes.append(f"torch.profiler pass failed: {exc!r}")
            consol_msg_universal_one_liner(f"[warn] torch.profiler pass failed: {exc!r}")

    # :::: Result + projection ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    ms = {p: sums[p] / max(timed, 1) for p in PHASES}
    total_ms = sum(ms.values())
    epochs = _resolve_epochs(cfg)
    handler_name, obs_groups_desc, geometry_loss_desc = _describe_loss_composition(wrapper)
    gpu_name = torch.cuda.get_device_name(device) if clock.cuda else None
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if clock.cuda else None
    # The autocast the timed steps actually ran under (the preset may have overridden the cfg knob).
    autocast_dtype = wrapper.resolve_autocast_dtype() if hasattr(wrapper, "resolve_autocast_dtype") else None
    if autocast_dtype is None:
        mixed_precision = None
    else:
        mixed_precision = {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(autocast_dtype, str(autocast_dtype))
    float32_matmul_precision = torch.get_float32_matmul_precision() if clock.cuda else None
    if data_path.dataset_device.type == "cpu" and device.type == "cuda":
        notes.append(
            "training data on CPU (host gather + per-batch H2D copy): compare with "
            + ("pipeline.dataloader.store_device=auto" if data_manager == "dataloader"
               else "mbrl_lib.keep_replay_buffer_on_device=true")
        )
    if gpu_name is not None and total_ms and (ms["loader"] + ms["h2d"]) / total_ms > 0.5:
        notes.append("HOST-BOUND: more than half of the step is loader gather + H2D -> the GPU idles")
    if clock.cuda and not util_samples:
        notes.append(
            "GPU SM utilization not sampled (neither pynvml nor `nvidia-smi` usable in the container): "
            "watch `nvtop` on the node, or read the profiler kernel-busy column"
        )
    elif util_sampler.backend:
        notes.append(f"GPU SM utilization sampled through {util_sampler.backend} after every timed batch")

    result = TrainStepBenchmarkResult(
        label=label if label is not None else _resolve_label(cfg),
        host=f"{platform.node()} ({platform.machine()})",
        device=str(device),
        gpu=gpu_name,
        torch_version=f"{torch.__version__} (CUDA {torch.version.cuda or 'n/a'})",
        store_device=data_path.store_device,
        dataset_device=str(data_path.dataset_device),
        num_workers=data_path.num_workers,
        pin_memory=data_path.pin_memory,
        mixed_precision=mixed_precision,
        batch_size=batch_size,
        history_len=data_path.history_len,
        horizon_len=data_path.horizon_len,
        output_window_len=data_path.output_window_len,
        params=_count_trainable_params(wrapper),
        obs_dim=data_path.obs_dim,
        act_dim=data_path.act_dim,
        num_train_windows=data_path.num_train,
        num_val_windows=data_path.num_val,
        train_batches_per_epoch=len(train_loader),
        val_batches_per_epoch=0 if val_loader is None else len(val_loader),
        epochs=epochs,
        warmup_batches=warmup,
        timed_batches=timed,
        timed_val_batches=timed_val,
        ms=ms,
        total_ms=total_ms,
        val_ms=val_ms if math.isfinite(val_ms) else 0.0,
        peak_cuda_mem_mb=peak,
        last_loss=last_loss,
        gpu_util_pct=(sum(util_samples) / len(util_samples)) if util_samples else None,
        wall_time_hours=float(settings.slurm_wall_time_hours),
        notes=notes,
        compile_mode=compile_mode,
        profiler=profiler_summary,
        feature_handler=handler_name,
        obs_groups=obs_groups_desc,
        geometry_loss=geometry_loss_desc,
        data_manager=data_manager,
        data_dtype=data_dtype,
        float32_matmul_precision=float32_matmul_precision,
        precision_preset=precision_preset,
        val_compute_ms=(
            val_compute_ms if (val_compute_ms is not None and math.isfinite(val_compute_ms)) else None
        ),
    )

    report = result.markdown_report()
    consol_msg_universal(f"Training-step benchmark report:\n\n{report}")
    consol_msg_universal_one_liner(
        f"BENCHMARK {result.label}: {result.total_ms:.0f} ms/batch (loader {ms['loader']:.0f} + h2d {ms['h2d']:.0f} "
        f"+ fwd {ms['forward']:.0f} + bwd {ms['backward']:.0f} + opt {ms['optimizer']:.0f}), GPU duty "
        f"{result.gpu_duty_cycle_pct:.0f} %, {result.projected_epoch_hours * 3600:.0f} s/epoch, "
        f"{result.projected_trial_hours:.1f} h/trial projected / {result.approx_trial_hours:.1f} h compute-only "
        f"({epochs} epochs), {result.resubmissions_needed} resubmission(s) at {result.wall_time_hours:g} h wall-time"
        + f", Do/Da {result.obs_dim}/{result.act_dim}, geometry loss {geometry_loss_desc}"
        + f", data {data_manager}/{data_dtype}, AMP {mixed_precision}, fp32 matmul {float32_matmul_precision}"
        + ("" if compile_mode is None else f", torch.compile={compile_mode}")
        + ("" if (profiler_summary is None or not profiler_summary.cuda_kernels_per_step)
           else f", profiler kernel-busy {profiler_summary.cuda_busy_pct:.0f} % / {profiler_summary.cuda_kernels_per_step:.0f} kernels/step")
    )
    _write_reports(cfg, result, exp_dir_relative_path, append_report_to=settings.append_report_to)
    return result



# ==== Multi-model driver (RLRP-838) ==============================================================
# THE COMPOSED SOURCE EXPERIMENT CFG IS AUTHORITATIVE. Everything that defines the model's training
# REGIME (captured step `pipeline.cuda_graph_training_step` / `cuda_graph_allow_autocast`,
# `ms_model.mixed_precision`, `torch_backend.cuda_high_precision_float32`, data path / dtype, batch
# size, window, ...) comes from the composed cfg and is changed ONLY through
# `source_experiment_cfg.overrides` / `model_overrides` on PRODUCTION keys. The launcher-level
# `benchmark` block is turned into `TrainStepBenchmarkSettings` (how long / what to record) and
# handed to `execute_one` as an argument: nothing is written onto the composed cfg, so
# `multirun-X.yaml` and `multirun-X+cudagraph.yaml` are timed as the experiments they are.
# Config nodes that would let a bench knob leak into the experiment (`benchmark`, `pipeline.benchmark`)
# are REJECTED both in the override lists and on the composed cfg (a `benchmark-train-step-*.yaml`
# dev-profiling cfg is not a valid source).
FORBIDDEN_OVERRIDE_ROOTS = ("benchmark", "pipeline.benchmark")


def _override_target(override: str) -> str:
    """``'+pipeline.benchmark.num_batches=5'`` -> ``'pipeline.benchmark.num_batches'`` (Hydra grammar:
    optional ``+`` / ``++`` / ``~`` prefix, ``key=value`` or bare ``~key``; ``@pkg`` suffix stripped)."""
    key = str(override).strip().lstrip("+~")
    key = key.split("=", 1)[0].strip()
    return key.split("@", 1)[0].strip()


def _assert_no_benchmark_override(overrides: Sequence[str], *, model_name: str) -> None:
    """Reject compose overrides that target a ``benchmark`` / ``pipeline.benchmark`` node."""
    offending = []
    for override in overrides:
        target = _override_target(override)
        if any(target == root or target.startswith(root + ".") for root in FORBIDDEN_OVERRIDE_ROOTS):
            offending.append(str(override))
    if offending:
        raise ValueError(
            f"[training-benchmark] model {model_name!r}: overrides {offending} target a `benchmark` / "
            "`pipeline.benchmark` node. The source experiment cfg is authoritative and the publication "
            "benchmark carries no such node: state the measurement knobs in the launcher `benchmark` block "
            "(warmup/num/val batches, profiler, wall-time) and regime changes on PRODUCTION keys "
            "(e.g. `pipeline.cuda_graph_training_step=true`, `ms_model.mixed_precision=bf16`). Dev-only "
            "profiling levers belong to `robotic_3d_env_train_step_dev_profiling_pipeline`."
        )


def _assert_composed_cfg_has_no_benchmark_node(composed: omegaconf.DictConfig, *, model_name: str) -> None:
    """Reject a composed source cfg carrying a ``benchmark`` / ``pipeline.benchmark`` node (a dev
    profiling cfg such as ``benchmark-train-step-*.yaml``, or a stray override that slipped through)."""
    present = [
        root for root in FORBIDDEN_OVERRIDE_ROOTS
        if omegaconf.OmegaConf.select(composed, root, default=None) is not None
    ]
    if present:
        raise ValueError(
            f"[training-benchmark] model {model_name!r}: the composed source experiment cfg carries "
            f"{present} -- not an experiment cfg (a `benchmark-train-step-*.yaml` dev-profiling cfg swaps "
            "the /pipeline group to `robotic_3d_env_train_step_dev_profiling_pipeline`). Point "
            "`benchmark.source_experiment_cfg.path` / `benchmark_selected_models` at the experiment's own "
            "`multirun-*.yaml` / `*_main_multirun_*.yaml` cfgs."
        )


def _default_aggregate_save_path() -> str:
    """``<cwd>/training_benchmark.json`` under Hydra ``chdir: true`` (the run dir)."""
    return os.path.join(os.getcwd(), f"{AGGREGATE_REPORT_BASENAME}.json")


def _resolve_model_label(model_name: str, model_labels: Optional[Dict[str, str]]) -> str:
    if model_labels and model_name in model_labels:
        return str(model_labels[model_name])
    return str(model_name)


def render_detailed_summary_table(
    results: List[Tuple[str, TrainStepBenchmarkResult]],
) -> str:
    """Wide per-model x per-phase markdown table (opt-in ``detailed_summary.enable``).

    Columns mirror the existing report style (``markdown_header`` / ``markdown_row``): loader/h2d/
    forward/backward/optimizer ms, total ms/batch, val ms/batch (+ its compute-only part), GPU duty %,
    the epoch size (train windows / batches per epoch / epochs) the projection is built on, projected
    s/epoch, h/epoch & h/trial (measured step, data path included), the compute-only bound h/epoch &
    h/trial (free data path) and the data-path share of the trial. Does NOT alter the JSON schema.
    """
    cols = (
        ["model"]
        + [f"{p} (ms)" for p in PHASES]
        + [
            "total ms/batch",
            "val ms/batch",
            "val compute ms",
            "GPU duty %",
            "train windows",
            "batches/epoch",
            "epochs",
            "s/epoch",
            "projected h/epoch",
            "projected h/trial",
            "compute-only h/epoch",
            "compute-only h/trial",
            "data path %",
        ]
    )
    lines = [
        "## Detailed summary (per model x per phase)",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for model_name, result in results:
        cells = (
            [model_name]
            + [f"{result.ms[p]:.1f}" for p in PHASES]
            + [
                f"{result.total_ms:.1f}",
                f"{result.val_ms:.1f}",
                "n/a" if result.val_compute_ms is None else f"{result.val_compute_ms:.1f}",
                f"{result.gpu_duty_cycle_pct:.0f}",
                str(result.num_train_windows),
                str(result.train_batches_per_epoch),
                str(result.epochs),
                f"{result.projected_epoch_hours * 3600:.1f}",
                f"{result.projected_epoch_hours:.3f}",
                f"{result.projected_trial_hours:.2f}",
                f"{result.approx_epoch_hours:.3f}",
                f"{result.approx_trial_hours:.2f}",
                f"{result.data_path_share_pct:.0f}",
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _compose_source_experiment_cfg(
    *,
    path: str,
    model_name: str,
    overrides: Sequence[str],
    root_project_path: Optional[str],
) -> omegaconf.DictConfig:
    """Hydra-compose ONE multirun launcher config (``GlobalHydra`` clear/re-init per model).

    Reuses the search-root / override-merge contract of
    :mod:`tools.benchmark_tools.source_experiment_cfg` so nested compose inside an already-running
    ``@hydra.main`` app leaves the outer app clean.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from tools.benchmark_tools.model_group_config import _default_root_project_path
    from tools.benchmark_tools.source_experiment_cfg import (
        _resolve_source_dir,
        _split_launcher_configs_root,
    )

    root = root_project_path or _default_root_project_path()
    source_dir = _resolve_source_dir(path, root)
    search_root, rel_dir = _split_launcher_configs_root(source_dir)
    config_name = f"{rel_dir}/{model_name}" if rel_dir else str(model_name)

    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=search_root, version_base=None)
    try:
        composed = compose(config_name=config_name, overrides=list(overrides))
        # Materialise a free-standing DictConfig so the outer Hydra app can keep running after
        # GlobalHydra is cleared (the compose result is otherwise bound to the nested instance).
        return omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(composed, resolve=False)
        )
    finally:
        GlobalHydra.instance().clear()


# Run-CONTEXT keys (where / on what this process runs), NOT experiment semantics: the composed cfg
# inherits them from the outer launcher run so every model shares the node, the device and the
# checkout. Nothing else of the outer cfg reaches the composed cfg.
RUN_CONTEXT_KEYS = ("project_root_path", "orig_cwd", "device")


def _prepare_composed_cfg(
    composed: omegaconf.DictConfig,
    outer_cfg: omegaconf.DictConfig,
    *,
    model_name: str,
) -> omegaconf.DictConfig:
    """The per-model working cfg = the composed source experiment cfg, untouched, + run context.

    The composed cfg is AUTHORITATIVE (regime, data path, batch size, window, seed, ...): this helper
    copies it and only fills the run-context keys the outer ``R2S2RPipelineHydraApp`` run resolved
    (:data:`RUN_CONTEXT_KEYS`; a nested ``compose`` leaves ``orig_cwd`` missing and ``device`` at
    its ``???`` default) plus the ``set_core_hydra_missing_keys`` auto-fills the app applies to any cfg.
    It REJECTS a composed cfg carrying a ``benchmark`` / ``pipeline.benchmark`` node
    (:func:`_assert_composed_cfg_has_no_benchmark_node`). Measurement knobs travel separately as
    :class:`TrainStepBenchmarkSettings`; the report label as the ``execute_one(label=...)`` argument.
    """
    from tools.hydra_apps_tools.r2s2r_apps_utils import set_core_hydra_missing_keys

    _assert_composed_cfg_has_no_benchmark_node(composed, model_name=model_name)
    working = omegaconf.OmegaConf.create(
        omegaconf.OmegaConf.to_container(composed, resolve=False)
    )
    with omegaconf.open_dict(working):
        for key in RUN_CONTEXT_KEYS:
            if key in outer_cfg and not omegaconf.OmegaConf.is_missing(outer_cfg, key):
                working[key] = outer_cfg.get(key)
        try:
            set_core_hydra_missing_keys(working)
        except (AttributeError, omegaconf.errors.OmegaConfBaseException):
            # Minimal cfgs (unit tests) without the `host_user` / `IDE` nodes of global_config.
            if not omegaconf.OmegaConf.select(working, "device", throw_on_missing=False):
                working.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return working


def _write_aggregate_reports(
    *,
    results: List[Tuple[str, TrainStepBenchmarkResult]],
    save_path: str,
    detailed_summary: bool = False,
    failures: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, str]:
    """Write ``training_benchmark.json`` + ``.md`` next to each other; return both paths.

    ``failures`` (``continue_on_error`` mode) are recorded in a SEPARATE ``failures`` list of the
    JSON payload and as a markdown section, so ``results`` stays a homogeneous list of
    :meth:`TrainStepBenchmarkResult.to_dict` rows for the plot tool.
    """
    failures = list(failures or [])
    json_path = save_path
    if not json_path.endswith(".json"):
        json_path = f"{json_path}.json"
    md_path = os.path.splitext(json_path)[0] + ".md"
    os.makedirs(os.path.dirname(os.path.abspath(json_path)) or ".", exist_ok=True)

    payload = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "action": "RLRP-838",
        "n_models": len(results),
        "results": [
            {
                "model": model_name,
                "label": result.label,
                **result.to_dict(),
            }
            for model_name, result in results
        ],
        "n_failures": len(failures),
        "failures": failures,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    lines = [
        "# Training-speed benchmark (multi-model)",
        "",
        f"- **date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **models**: {len(results)}",
        "",
        "## Aggregate row",
        "",
    ]
    if results:
        lines.append(TrainStepBenchmarkResult.markdown_header())
        for _name, result in results:
            lines.append(result.markdown_row())
        lines.append("")
    if detailed_summary and results:
        lines.append(render_detailed_summary_table(results))
    if failures:
        lines.extend(["## Failed models (continue_on_error)", ""])
        for failure in failures:
            lines.append(f"- **{failure['model']}**: `{failure['error_type']}` -- {failure['error']}")
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    consol_msg_universal_one_liner(
        f"Aggregate training-benchmark report written to '{md_path}' (+ .json)"
    )
    return json_path, md_path


def _release_device_memory(device_hint: Any) -> None:
    """Free what the previous model left behind so successive models share a clean GPU.

    Between two models of one multi-model run the previous model/optimizer/dataset go out of
    scope, but the CUDA caching allocator keeps its blocks reserved and ``torch.compile`` /
    CUDA-graph pools stay alive. Left as-is, model *k+1* would see an inflated
    ``peak_cuda_mem_mb`` baseline and can OOM on a shared-memory device (Jetson AGX Orin) even
    though it fits alone. ``gc.collect`` + ``empty_cache`` (+ ``reset_peak_memory_stats``) is the
    documented remedy; ``torch._dynamo.reset`` drops compiled graphs when compile was used.
    """
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        importlib.import_module("torch._dynamo").reset()
    except Exception:  # pragma: no cover - dynamo optional / version dependent
        pass
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    try:
        torch.cuda.reset_peak_memory_stats(torch.device(str(device_hint)))
    except Exception:  # pragma: no cover - cpu device hint
        pass


def run_training_benchmark(cfg: omegaconf.DictConfig) -> List[TrainStepBenchmarkResult]:
    """Compose + bench every ``benchmark.source_experiment_cfg.benchmark_selected_models`` entry.

    All models run on THIS process/node (sequential). Shared ``overrides`` + per-model
    ``model_overrides`` are merged by key (per-model wins) via the
    :mod:`tools.benchmark_tools.source_experiment_cfg` contract; they are the ONLY way to alter what
    is timed, and may not target ``benchmark`` / ``pipeline.benchmark`` nodes
    (:func:`_assert_no_benchmark_override`). The composed cfg is benched AS IS -- robotic 3D env or
    math env alike (:func:`resolve_env_family`) -- with the launcher's measurement knobs passed
    separately (:meth:`TrainStepBenchmarkSettings.from_launcher_cfg`). Results are aggregated into
    ``training_benchmark.json`` / ``.md``; when ``detailed_summary.enable`` is true a wide
    per-model x per-phase table is appended to the markdown only (JSON schema unchanged).

    Per-model ``train_step_benchmark.*`` reports land in ``<save dir>/per_model/<model>/`` so
    they do not overwrite each other. Device memory is released between models. With
    ``benchmark.continue_on_error: true`` a failing model (OOM, capture blocker, ...) is recorded
    in the report's ``failures`` list and the run proceeds to the next model; the default
    (``false``) fails fast.

    :param cfg: Outer launcher cfg (``model_training_benchmark.yaml``).
    :return: One :class:`TrainStepBenchmarkResult` per SUCCESSFUL model, selection order preserved.
    """
    from tools.benchmark_tools.source_experiment_cfg import (
        _coerce_selected_models,
        _merge_overrides,
    )

    bench_cfg = cfg.get("benchmark", None)
    if bench_cfg is None:
        raise ValueError("run_training_benchmark requires a top-level `benchmark` config block.")
    source_cfg = bench_cfg.get("source_experiment_cfg", None)
    if source_cfg is None:
        raise ValueError(
            "benchmark.source_experiment_cfg is required (path + benchmark_selected_models of "
            "the multirun launcher configs to compose and time)."
        )
    path = source_cfg.get("path", None)
    if path is None:
        raise ValueError("benchmark.source_experiment_cfg.path is required.")
    selected = _coerce_selected_models(source_cfg.get("benchmark_selected_models", None))

    base_overrides = source_cfg.get("overrides", None)
    base_overrides = [str(o) for o in base_overrides] if base_overrides is not None else []
    model_overrides = source_cfg.get("model_overrides", None)
    if model_overrides is not None:
        model_overrides = {
            str(name): [str(o) for o in ov] for name, ov in dict(model_overrides).items()
        }
    model_labels_raw = source_cfg.get("model_labels", None)
    model_labels: Dict[str, str] = {}
    if model_labels_raw is not None:
        model_labels = {str(n): str(lbl) for n, lbl in dict(model_labels_raw).items()}

    root_project_path: Optional[str] = None
    try:
        from tools.hydra_apps_tools.hydra_utils import fetch_project_root_path_via_hydra

        root_project_path = fetch_project_root_path_via_hydra(
            cfg, 1, "MTM-Pro"
        )
    except Exception:
        root_project_path = cfg.get("project_root_path", None)

    detailed_summary_cfg = cfg.get("detailed_summary", None)
    if detailed_summary_cfg is None:
        detailed_summary_cfg = bench_cfg.get("detailed_summary", None)
    detailed_summary = bool(
        (detailed_summary_cfg or {}).get("enable", False)
        if detailed_summary_cfg is not None
        else False
    )

    save_path = str(bench_cfg.get("save_path", None) or _default_aggregate_save_path())
    per_model_root = os.path.join(os.path.dirname(os.path.abspath(save_path)), "per_model")
    continue_on_error = bool(bench_cfg.get("continue_on_error", False))
    # Measurement knobs only; raises on a dev-only regime lever in the launcher block.
    settings = TrainStepBenchmarkSettings.from_launcher_cfg(bench_cfg)

    paired: List[Tuple[str, TrainStepBenchmarkResult]] = []
    failures: List[Dict[str, str]] = []
    total = len(selected)
    for idx, model_name in enumerate(selected, start=1):
        consol_msg_universal_one_liner(
            f"[training-benchmark] model {idx}/{total}: {model_name}"
        )
        per_model = model_overrides.get(model_name) if model_overrides else None
        overrides_list = _merge_overrides(base_overrides, per_model)
        label = _resolve_model_label(str(model_name), model_labels)
        # Operator misconfiguration, not a benchmark result: always fail fast (even with
        # continue_on_error) so a forbidden override can never be silently skipped.
        _assert_no_benchmark_override(overrides_list, model_name=str(model_name))
        try:
            composed = _compose_source_experiment_cfg(
                path=str(path),
                model_name=str(model_name),
                overrides=overrides_list,
                root_project_path=str(root_project_path) if root_project_path else None,
            )
            working = _prepare_composed_cfg(composed, cfg, model_name=str(model_name))
            # Process-global torch.backends state (TF32 / cudnn.benchmark) is normally applied
            # ONCE by the launcher app from the OUTER cfg; here every composed model is its own
            # experiment, so re-apply ITS `torch_backend` block before timing it.
            apply_torch_backend_cfg(working)
            result = execute_one(
                working,
                headless=True,
                report_dir=os.path.join(per_model_root, str(model_name)),
                settings=settings,
                label=label,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            consol_msg_universal_one_liner(
                f"[training-benchmark] model {idx}/{total}: {model_name} FAILED "
                f"({type(exc).__name__}: {exc}) -- continue_on_error=true, moving on"
            )
            failures.append(
                {"model": str(model_name), "label": label,
                 "error_type": type(exc).__name__, "error": str(exc)}
            )
            continue
        finally:
            _release_device_memory(cfg.get("device", "cpu"))
        paired.append((str(model_name), result))

    _write_aggregate_reports(
        results=paired, save_path=save_path, detailed_summary=detailed_summary,
        failures=failures,
    )
    return [result for _name, result in paired]
