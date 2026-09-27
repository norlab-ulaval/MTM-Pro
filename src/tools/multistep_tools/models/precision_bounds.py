# coding=utf-8
"""Dtype-aware numerical bounds for the (multi-step) dynamics-model stack.

Single source of truth for the variance / std / weight floors **and** the
log-variance ceiling that guard the ``variance -> log -> exp`` and sampling paths
across the MTM-Pro inheritance chain (``ExponentialFamilyMLP`` ->
``Abstract*WeightedMultiStepMLP`` -> ``MS2MS2SSArTemporalMixturePME`` and the
transformer-mixer / sampling-free variants).

Both the lower numerical floors (variance / std / weight) and the upper
log-variance ceiling (``LOGVAR_SAFE_MAX``) live here so the numerical-safety
bounds are organized in one place and easier to reason about / debug, rather than
being duplicated as per-class magic numbers (the former ``_VAR_SAFE_MIN`` /
``_MIX_LOGVAR_SAFE_MAX`` class attributes).

Motivation (RLRC, multi-step precision hardening, 2026-06 — Gate C):
    Historically these floors were fixed scalars chosen during early, *float32-era*
    training (e.g. ``_VAR_SAFE_MIN=1e-10``, sampling ``EPS=1e-6``, ``_WEIGHTS_EPS=1e-10``).
    They were duplicated and inconsistent across classes (``1e-6`` vs ``1e-10`` vs the
    semantic ``log(1e-30)`` limit). Under genuine ``float64`` (the supported ``cpu`` /
    ``cuda`` backends) these float32 floors needlessly discard precision for low-variance
    dimensions, which matters for multi-step AR compounding.

Policy:
    - ``float32`` values are kept **identical to the historical constants** (no behavior
      change for existing float32 runs / checkpoints / tests).
    - ``float64`` values are lowered to the semantic near-zero limit that is safely
      representable in double precision, giving multi-step models the full dynamic range.
      ``1e-30`` matches the already-documented semantic variance limit
      (``ExponentialFamilyMLP._LOGVAR_MIN_LIMIT = log(1e-30)``).

These are *numerical guards*, not modeling bounds; see the constants audit
(`.junie/ai_artifact/reports/mtm_pro_constants_audit_20260613.md`).
"""

from __future__ import annotations

from typing import Optional

import torch

# Historical float32-era anchors (unchanged for float32).
# (CRITICAL) ToDo: assess lowering to 1e-12 on Part-MTM-Pro (ref task RLRP-761)
# (CRITICAL) ToDo: assess lowering to 1e-12 on RePart-MTM-Pro (ref task RLRP-761)
# (CRITICAL) ToDo: assess lowering to 1e-12 on Distrib-MTM-Pro (ref task RLRP-761)
_VARIANCE_FLOOR_F32: float = 1e-10

_STD_FLOOR_F32: float = 1e-6 # ~sqrt(1e-12)
_WEIGHTS_EPS_F32: float = 1e-10

# Float64 near-zero limits (semantic floor `1e-30`, representable in double precision).
_VARIANCE_FLOOR_F64: float = 1e-30
_STD_FLOOR_F64: float = 1e-15  # ~ sqrt(1e-30): below this a std is numerically deterministic
_WEIGHTS_EPS_F64: float = 1e-15

# Mixture log-variance ceiling (semantic, dtype-independent). Keeps the
# distribution scale numerically tractable: ``exp(LOGVAR_SAFE_MAX / 2)`` =
# ``exp(10) ~ 22026`` as a std. Unlike the lower floors this bound is a modeling
# guard on component-mean disagreement, not a dtype-precision artifact, so it is
# the same for float32 and float64. Consolidated here from the former per-class
# ``_MIX_LOGVAR_SAFE_MAX`` attributes.
# (CRITICAL) ToDo: assess raising to ??? on Part-MTM-Pro (ref task RLRP-761)
# (CRITICAL) ToDo: assess raising to ??? on RePart-MTM-Pro (ref task RLRP-761)
# (CRITICAL) ToDo: assess raising to ??? on Distrib-MTM-Pro (ref task RLRP-761)
LOGVAR_SAFE_MAX: float = 20.0

# AR-sample clamp ceiling (normalized space). The clamp guards the AR
# denormalize->shift->renormalize round-trip against float32 overflow (RLRP-530).
# float32 keeps the historical tight `1e2`; float64 has a vastly higher overflow
# threshold, so the clamp is relaxed to avoid biasing long-horizon rollouts while
# still catching true divergence / non-finite samples.
# (CRITICAL) ToDo: assess raising to ??? on Part-MTM-Pro (ref task RLRP-761)
# (CRITICAL) ToDo: assess raising to ??? on RePart-MTM-Pro (ref task RLRP-761)
# (CRITICAL) ToDo: assess raising to ??? on Distrib-MTM-Pro (ref task RLRP-761)
_AR_SAMPLE_CLAMP_F32: float = 1e2
_AR_SAMPLE_CLAMP_F64: float = 1e6


def _is_double(dtype: torch.dtype) -> bool:
    return dtype == torch.float64


def variance_floor(dtype: torch.dtype) -> float:
    """Minimum variance before ``log()`` to avoid ``-inf`` (dtype-aware).

    float32: ``1e-10`` (historical). float64: ``1e-30`` (semantic near-zero limit).
    """
    return _VARIANCE_FLOOR_F64 if _is_double(dtype) else _VARIANCE_FLOOR_F32


def std_floor(dtype: torch.dtype) -> float:
    """Minimum standard deviation used at sampling time (dtype-aware).

    float32: ``1e-6`` (historical). float64: ``1e-15`` (~ ``sqrt(variance_floor)``).
    """
    return _STD_FLOOR_F64 if _is_double(dtype) else _STD_FLOOR_F32


def weights_eps(dtype: torch.dtype) -> float:
    """Floor for discount / feature weight clamping (dtype-aware).

    float32: ``1e-10`` (historical, preserves existing test expectations).
    float64: ``1e-15``.
    """
    return _WEIGHTS_EPS_F64 if _is_double(dtype) else _WEIGHTS_EPS_F32


def variance_safe_min(dtype: torch.dtype) -> float:
    """Minimum mixture variance before ``log()`` (dtype-aware).

    Semantic alias of :func:`variance_floor`, kept as the canonical replacement
    for the former per-class ``_VAR_SAFE_MIN`` constant so the mixture log-variance
    call sites read self-documentingly.

    float32: ``1e-10`` (historical). float64: ``1e-30`` (semantic near-zero limit).
    """
    return variance_floor(dtype)


def logvar_safe_max(dtype: Optional[torch.dtype] = None) -> float:
    """Ceiling for mixture log-variance (semantic, dtype-independent).

    Returns :data:`LOGVAR_SAFE_MAX` (``20.0``). The ``dtype`` argument is accepted
    for call-site symmetry with the dtype-aware floors but is ignored: this is a
    modeling guard on component-mean disagreement, not a precision artifact, so it
    is identical for float32 and float64.
    """
    return LOGVAR_SAFE_MAX


def ar_sample_clamp_ceil(dtype: torch.dtype) -> float:
    """Ceiling (normalized space) for AR mixture samples (dtype-aware).

    float32: ``1e2`` (historical tight guard). float64: ``1e6`` (relaxed; the
    overflow threshold is far higher in double precision, so the tight float32
    clamp would needlessly bias long-horizon AR rollouts).
    """
    return _AR_SAMPLE_CLAMP_F64 if _is_double(dtype) else _AR_SAMPLE_CLAMP_F32
