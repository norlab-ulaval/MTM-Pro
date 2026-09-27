# coding=utf-8
"""Numerical-precision diagnostics for the dynamics-model stack.

This module provides lightweight, process-wide *warn-once* helpers used to surface
silent ``float64 -> float32`` downcasts and to emit a one-shot device/dtype summary
at the first forward pass.

Motivation (RLRC, multi-step precision hardening, 2026-06):
    Multi-step models of the *MTM-Pro* family (``ms2ms2ss_ar_temporal_mixture_pme*``)
    are run with ``model_use_double_precision=true`` together with a double-precision
    normalizer.  On the supported backends (``cpu`` local DNA container, ``cuda`` on
    Valeria / Jetson-AGX-Orin) ``float64`` is fully supported, so any downcast to
    ``float32`` is *unexpected* and silently degrades precision.  These helpers make
    such downcasts (and any unexpected ``float32`` fallback) visible exactly once per
    call-site, without flooding the logs inside the training / rollout hot loops.

    The Apple-Silicon ``mps`` backend is NOT reachable from inside the DNA container,
    so the ``mps`` force-downcast guard is expected to be dead code; its warn-once is
    kept only as a defensive safety net.

Design notes:
    - Dedup is keyed by a caller-provided ``site`` string, so each instrumented
      location warns at most once for the lifetime of the process.
    - Emission goes through :func:`warnings.warn` (consistent with the existing
      ``_warn_numpy_input`` guard) so it integrates with the standard warnings filter.
    - :func:`reset_precision_warning_state` is provided for unit tests.
"""

from __future__ import annotations

import threading
import warnings
from typing import Optional, Set

import torch

from tools.console_tools.message import consol_msg_universal_one_liner
from tools.dna_dev_tools.dn_pytest_tools import is_pytest_run

# Process-wide registry of already-emitted warn-once sites.
_SEEN_SITES: Set[str] = set()
_SEEN_LOCK = threading.Lock()


class PrecisionDowncastWarning(UserWarning):
    """Raised once per call-site when a ``float64`` tensor is silently downcast to ``float32``."""


class PrecisionDiagnosticInfo(UserWarning):
    """Informational one-shot device/dtype summary emitted at the first forward pass."""


def reset_precision_warning_state() -> None:
    """Clear the warn-once registry. Intended for tests only."""
    with _SEEN_LOCK:
        _SEEN_SITES.clear()


def _mark_seen(site: str) -> bool:
    """Atomically record ``site`` as seen. Return ``True`` if this is the first sighting."""
    with _SEEN_LOCK:
        if site in _SEEN_SITES:
            return False
        _SEEN_SITES.add(site)
        return True


def warn_once_float32_downcast(
    site: str,
    *,
    tensor_name: str,
    from_dtype: torch.dtype,
    to_dtype: torch.dtype = torch.float32,
    reason: Optional[str] = None,
) -> None:
    """Emit a warn-once when a tensor is downcast to a lower-precision dtype.

    No-op unless ``from_dtype`` is strictly higher precision than ``to_dtype``
    (i.e. an actual precision loss), and unless this ``site`` has not warned yet.

    Args:
        site: Stable identifier of the call-site (used for dedup), e.g.
            ``"OneDTransitionRewardModelV2._process_batch:_as_float"``.
        tensor_name: Human-readable name of the affected tensor (e.g. ``"target"``).
        from_dtype: The original tensor dtype.
        to_dtype: The dtype it is being cast to (default ``torch.float32``).
        reason: Optional short explanation of why the downcast happens.
    """
    if from_dtype != torch.float64 or to_dtype == torch.float64:
        # Only a float64 -> non-float64 transition is a precision loss we care about.
        return

    if not _mark_seen(site):
        return

    reason_str = f" Reason: {reason}" if reason else ""
    warnings.warn(
        f"[RLRC precision] '{tensor_name}' downcast {from_dtype} -> {to_dtype} at {site}. "
        f"This silently reduces numerical precision of the (multi-step) model.{reason_str} "
        f"This warning is emitted once per call-site.",
        PrecisionDowncastWarning,
        stacklevel=3,
    )


def log_precision_summary_once(
    site: str,
    *,
    device: torch.device,
    param_dtype: Optional[torch.dtype] = None,
    normalizer_dtype: Optional[torch.dtype] = None,
    model_in_dtype: Optional[torch.dtype] = None,
    target_dtype: Optional[torch.dtype] = None,
) -> None:
    """Emit a one-shot device/dtype summary (e.g. at the first forward pass).

    Useful to confirm that ``float64`` survives end-to-end across model parameters,
    normalizer buffers and the assembled model input/target on ``cpu`` / ``cuda``.

    Args:
        site: Stable call-site identifier (used for dedup).
        device: The compute device.
        param_dtype: Dtype of the wrapped model's parameters (if known).
        normalizer_dtype: Dtype of the normalizer statistics (if known).
        model_in_dtype: Dtype of the assembled model input (if known).
        target_dtype: Dtype of the training target (if known).
    """
    if not _mark_seen(f"summary:{site}"):
        return

    parts = [f"device={device}"]
    if param_dtype is not None:
        parts.append(f"params={param_dtype}")
    if normalizer_dtype is not None:
        parts.append(f"normalizer={normalizer_dtype}")
    if model_in_dtype is not None:
        parts.append(f"model_in={model_in_dtype}")
    if target_dtype is not None:
        parts.append(f"target={target_dtype}")

    _warning_msg = (
        f"[RLRC precision] one-shot dtype/device summary at {site}: "
        + ", ".join(parts)
        + ". (Emitted once per process.)"
    )
    if is_pytest_run():
        warnings.warn(
            _warning_msg,
            PrecisionDiagnosticInfo,
            stacklevel=3,
        )
    else:
        consol_msg_universal_one_liner(
            _warning_msg, caller_name=PrecisionDiagnosticInfo.__name__
        )


def warn_once_precision_mismatch(
    site: str,
    *,
    requested_double_precision: bool,
    resolved_dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> None:
    """Warn-once when double precision was requested but the resolved dtype is ``float32``.

    Catches any unexpected ``float32`` fallback on ``cpu`` / ``cuda``.

    Args:
        site: Stable call-site identifier (used for dedup).
        requested_double_precision: The model's ``model_use_double_precision`` flag.
        resolved_dtype: The dtype actually used for compute.
        device: Optional device the compute runs on (for context in the message).
    """
    if not requested_double_precision or resolved_dtype == torch.float64:
        return

    if not _mark_seen(site):
        return

    device_str = f" on device '{device}'" if device is not None else ""
    warnings.warn(
        f"[RLRC precision] model_use_double_precision=True but resolved compute dtype is "
        f"{resolved_dtype}{device_str} at {site}. Double precision is NOT in effect; "
        f"expected float64 on cpu/cuda. This warning is emitted once per call-site.",
        PrecisionDowncastWarning,
        stacklevel=3,
    )
