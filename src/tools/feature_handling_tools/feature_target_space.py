# coding=utf-8
"""Target-space / normalizer scale primitives (RLRP-761 stage ``S10.2``).

Permanent framework module.

This module is the **single home** of "how the composed target / model input is
scaled per feature". It was extracted verbatim (pure relocation, no behaviour
change) out of :mod:`tools.feature_handling_tools.normalization_diagnostic`,
which had grown to own collection, rendering, flags, DR-staleness *and* these
scale primitives at once. The tell-tale was a function-local import of a
*diagnostic* module from the loss-weight *resolver*
(:mod:`tools.feature_handling_tools.feature_loss_weights`): a resolver has no
business depending on a rendering module.

By making this a **lower layer** with no dependency back on the diagnostic, both
the diagnostic and the resolver become plain consumers, and the ``std`` used by
the budget cap and by the S2 ``tgt%`` column has one source of truth — exactly
the drift class the ``S9.5`` / ``S10.2`` work exists to remove.

The three public entry points are:

- :func:`resolve_normalizer_output_std` — the ``norm std`` column of the S2
  table and the quantity ``per_feature_scale_mode: relative_to_normalized_std``
  multiplies by (measures the **input** facade);
- :func:`resolve_target_space_scale` — the per-feature scale of the **target**
  space and its label (``S8.4a``);
- :func:`resolve_target_space_delta_std` — a raw one-step innovation std
  expressed in the target space (``S8.4a``).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

#: Maximum number of replay-buffer rows sampled to compute the raw statistics.
_MAX_ROWS = 20_000

#: RLRP-777 -- degrade channel. The functions in this module feed *columns* of the
#: S2 table (the target-space scale / normalized std), so a silent `return None` here
#: does not blank the report -- it blanks columns INSIDE a table that still looks
#: valid, which is strictly worse than a missing file. Each non-emitting exit records
#: WHY here; the diagnostic emitter reads the accumulated reasons via
#: :func:`reset_target_space_degrade` / :func:`consume_target_space_degrade` and prints
#: exactly ONE aggregated `[feature-normalization] degraded: ...` line so the
#: degradation is visible in `console.log`.
_DEGRADE_REASONS: List[str] = []


def _note_degrade(reason: str) -> None:
    """Record *reason* for a degraded (columns-unavailable) exit (RLRP-777)."""
    _DEGRADE_REASONS.append(reason)


def reset_target_space_degrade() -> None:
    """Clear the degrade reason accumulator (call before a collection pass)."""
    _DEGRADE_REASONS.clear()


def consume_target_space_degrade() -> List[str]:
    """Return and clear the accumulated degrade reasons (RLRP-777)."""
    reasons = list(_DEGRADE_REASONS)
    _DEGRADE_REASONS.clear()
    return reasons


def _to_numpy(tensor) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().to("cpu").numpy()


def _build_slot_index(
    obs_len: int, act_len: int, history_len: int, in_size: int
) -> Optional[List[List[int]]]:
    """Return, per single-step feature, its column indices in the flat input.

    Layout (see ``OneDTransitionRewardModel._get_model_input`` /
    ``_normalize_composed_obs``): ``[obs_len x history_len][act_len x
    history_len]``, each block stored slot-major.
    """
    expected = (obs_len + act_len) * history_len
    if in_size != expected:
        return None
    index: List[List[int]] = []
    for d in range(obs_len):
        index.append([s * obs_len + d for s in range(history_len)])
    offset = obs_len * history_len
    for j in range(act_len):
        index.append([offset + s * act_len + j for s in range(history_len)])
    return index


def resolve_normalizer_output_std(dynamics_model, batch) -> Optional[np.ndarray]:
    """Return the per-single-step-feature std of the **normalizer output**.

    This is the single definition of "normalized sigma" in the codebase
    (RLRP-761 ``S1.7``): it is the ``norm std`` column of the S2 table *and* the
    quantity ``train_time_domain_randomization.per_feature_scale_mode:
    relative_to_normalized_std`` multiplies the configured dimensionless scale
    by. It resolves to ``~1.0`` for a plainly standardized dim, to the raw std
    for a pass-through (``IDENTITY``) dim, and to the robust-moment-implied
    value under ``winsorized`` / ``quantile`` — i.e. exactly what the model
    sees, measured rather than assumed.

    :returns: an array of length ``singlestep_obs_len + singlestep_act_len``, or
        ``None`` when the layout cannot be resolved (the caller must then stay a
        no-op).
    """
    model = getattr(dynamics_model, "model", None)
    if model is None:
        _note_degrade("norm-std: dynamics_model exposes no wrapped model")
        return None
    obs_len = getattr(model, "singlestep_obs_len", None)
    act_len = getattr(model, "singlestep_act_len", None)
    history_len = getattr(model, "history_len", None)
    if not obs_len or act_len is None or not history_len:
        _note_degrade(
            "norm-std: model exposes no singlestep_obs_len / singlestep_act_len / "
            "history_len"
        )
        return None
    try:
        raw_obs = _to_numpy(batch.obs)
        raw_act = _to_numpy(batch.act)
        if raw_obs.shape[0] > _MAX_ROWS:
            raw_obs = raw_obs[:_MAX_ROWS]
            raw_act = raw_act[:_MAX_ROWS]
        with torch.no_grad():
            # Torch-first: feed tensors so the mbrl `to_tensor` numpy-input
            # UserWarning is not raised from this diagnostic path (the raw numpy
            # arrays above are still used for the raw-stat slicing).
            norm_in = _to_numpy(
                dynamics_model._get_model_input(
                    torch.as_tensor(raw_obs), torch.as_tensor(raw_act)
                )
            )
        slot_index = _build_slot_index(
            obs_len, act_len, history_len, norm_in.shape[-1]
        )
        if slot_index is None:
            _note_degrade("norm-std: slot index could not be built")
            return None
        return np.array(
            [float(np.std(norm_in[:, cols])) for cols in slot_index], dtype=float
        )
    except Exception as exc:
        _note_degrade(f"norm-std: _get_model_input raised: {exc!r}")
        return None


#: Labels for the resolved TARGET space, as stamped into the report and the
#: resolver log line (RLRP-761 ``S8.4d``). ``raw`` and the two affine labels are
#: EXACT; ``local_linear`` is a first-order approximation at the normalized mean.
TARGET_SPACE_RAW = "raw"
TARGET_SPACE_STATE_SIGMA = "state_sigma"
TARGET_SPACE_INNOVATION = "innovation"
TARGET_SPACE_LOCAL_LINEAR = "local_linear"

_JACOBIAN_EPS = 1e-12


def _unwrap_base_normalizer(normalizer):
    """Return the innermost base of a transparent wrapper.

    ``StrategyAwareNormalizer`` delegates to ``self.base``; the *type* that
    determines the target space is the base's, not the wrapper's.
    """
    seen = 0
    while normalizer is not None and hasattr(normalizer, "base") and seen < 8:
        normalizer = getattr(normalizer, "base")
        seen += 1
    return normalizer


def _target_space_label(normalizer) -> str:
    """Classify a sub-normalizer into one of the ``TARGET_SPACE_*`` labels."""
    base = _unwrap_base_normalizer(normalizer)
    if base is None:
        return TARGET_SPACE_RAW
    try:
        # ``is_affine`` is a PROPERTY on the normalizer API, not a method.
        affine = getattr(base, "is_affine")
        if callable(affine):  # defensive: tolerate either spelling
            affine = affine()
        if not bool(affine):
            return TARGET_SPACE_LOCAL_LINEAR
    except Exception:
        return TARGET_SPACE_LOCAL_LINEAR
    # `InnovationScaledNormalizer` subclasses `ZScoreNormalizer`, so the name is
    # the discriminator; both are affine, but they scale by DIFFERENT quantities
    # and the distinction is exactly what `S8` exists to expose.
    return (
        TARGET_SPACE_INNOVATION
        if "Innovation" in type(base).__name__
        else TARGET_SPACE_STATE_SIGMA
    )


def _sub_jacobian_diag(normalizer, width: int) -> Optional[np.ndarray]:
    """Per-dim ``d denormalize / d z`` of a single-step sub-normalizer.

    Evaluated at ``z = 0``, i.e. at the **normalized mean**: for the affine types
    the slope is constant so the point is irrelevant, and for the non-affine ones
    it is the canonical linearization point (the centre of the soft-clip /
    quantile map, where the transform is least distorted). This keeps the helper
    DATA-FREE, which is what lets the weight resolver call it before any batch is
    in scope.
    """
    if normalizer is None or not width:
        return None
    try:
        with torch.no_grad():
            probe = torch.zeros(1, int(width))
            jac = normalizer.denormalize_jacobian_diag(probe)
        jac = _to_numpy(jac).reshape(-1)[: int(width)]
    except Exception:
        return None
    if jac.size != int(width) or not np.all(np.isfinite(jac)):
        return None
    return np.maximum(np.abs(jac), _JACOBIAN_EPS)


def resolve_target_space_scale(
    dynamics_model, obs_len: int, act_len: int, facade: str = "output"
) -> Optional[Tuple[np.ndarray, str]]:
    """Per-single-step-feature scale of the **target** space, and its label.

    RLRP-761 ``S8.4a`` — the single definition of "how the composed target is
    scaled", and the twin of :func:`resolve_normalizer_output_std` (which
    measures the *input* facade). It exists because two independent code paths
    used to ASSUME this quantity, each covering a different subset of the
    normalizer types, and the running ``S7`` A/B was on the one type where both
    were wrong:

    - the loss-weight resolver's budget cap assumed ``sigma_Delta / sigma_state``
      (correct for ``standard_symmetric`` only);
    - the ``tgt%`` column measured the INPUT facade (correct for every type whose
      input and output facades share their sub-normalizers, i.e. all but
      ``standard_symmetric_innovation``, which decouples them by design).

    The returned scale is ``d denormalize / d z``, so the target-space innovation
    is ``sigma_Delta_raw / scale`` — one formula covering all five types:

    ==================================  ==========================  ============
    ``normalizer_type``                 scale                       label
    ==================================  ==========================  ============
    ``standard``                        ``1`` (no output facade)    ``raw``
    ``standard_symmetric``              ``sigma_state``             ``state_sigma``
    ``standard_symmetric_innovation``   ``s`` (innovation)          ``innovation``
    ``winsorized`` / ``quantile``       local slope at the mean     ``local_linear``
    ==================================  ==========================  ============

    **Layout note.** The output facade is a :class:`_InputOutputNormalizerFacade`
    in *block* configuration, so ``obs_sub`` / ``act_sub`` are already
    per-single-step (widths ``Do`` / ``Da``). The concatenated
    ``[obs_dims + act_dims]`` vector is therefore a direct concatenation of the
    two Jacobian diagonals -- no slot index and no history-window de-tiling,
    unlike :func:`resolve_normalizer_output_std`.

    :param dynamics_model: the transition-model wrapper owning the facades.
    :param obs_len: single-step observation width.
    :param act_len: single-step action width.
    :param facade: ``"output"`` (the target space) or ``"input"``. The latter is
        used to form the input/output CORRECTION of ``S8.4c``, which is exactly
        ``1`` whenever the two facades share their sub-normalizers.
    :returns: ``(scale, label)`` with ``len(scale) == obs_len + act_len``, or
        ``None`` when the facade cannot be introspected (the caller must then
        fall back explicitly and say so).
    """
    obs_len = int(obs_len or 0)
    act_len = int(act_len or 0)
    total = obs_len + act_len
    if total <= 0:
        _note_degrade("target-scale: obs_len + act_len <= 0")
        return None

    attr = "input_normalizer" if facade == "input" else "output_normalizer"
    output_normalizer = getattr(dynamics_model, attr, None)
    if output_normalizer is None:
        # `standard`: no output transform at all -- the target IS physical.
        return np.ones(total, dtype=float), TARGET_SPACE_RAW

    if not bool(getattr(output_normalizer, "is_block", False)):
        # A single flat facade (the `standard` INPUT path) is per-history-slot,
        # not per-single-step; it is not a configuration the block-facade target
        # accounting can describe, so refuse rather than guess.
        _note_degrade(f"target-scale ({facade}): facade is not a block facade")
        return None

    obs_sub = getattr(output_normalizer, "obs_sub", None)
    act_sub = getattr(output_normalizer, "act_sub", None)
    obs_jac = _sub_jacobian_diag(obs_sub, obs_len) if obs_len else np.empty(0)
    act_jac = _sub_jacobian_diag(act_sub, act_len) if act_len else np.empty(0)
    if obs_jac is None or act_jac is None:
        _note_degrade(
            f"target-scale ({facade}): sub-normalizer Jacobian diagonal unavailable"
        )
        return None

    scale = np.concatenate([obs_jac, act_jac])
    if scale.size != total:
        _note_degrade(f"target-scale ({facade}): scale width mismatch")
        return None

    labels = []
    if obs_len:
        labels.append(_target_space_label(obs_sub))
    if act_len:
        labels.append(_target_space_label(act_sub))
    # An approximation anywhere makes the whole vector approximate; otherwise the
    # blocks agree in practice (one `normalizer_type` builds both).
    if TARGET_SPACE_LOCAL_LINEAR in labels:
        label = TARGET_SPACE_LOCAL_LINEAR
    else:
        label = labels[0] if len(set(labels)) == 1 else "mixed"
    return scale, label


def _resolve_target_correction(
    dynamics_model, obs_len: int, act_len: int
) -> Tuple[np.ndarray, str]:
    """Factor mapping an INPUT-space innovation to TARGET space, per feature.

    RLRP-761 ``S8.4c``. The ``tgt%`` column is measured on the model input
    (``_get_model_input``), which is the cheapest and most faithful source --
    except that under a *decoupled* facade the input is not the target. Since

    .. math::
       \\sigma^{\\text{in}}_{\\Delta,d} = \\sigma_{\\Delta,d}/J^{\\text{in}}_d,
       \\qquad
       \\sigma^{\\text{tgt}}_{\\Delta,d} = \\sigma_{\\Delta,d}/J^{\\text{out}}_d,

    the correction is the ratio :math:`J^{\\text{in}}_d / J^{\\text{out}}_d`. It is
    **exactly** ``1.0`` whenever the two facades share their sub-normalizers,
    which makes ``S8.4c`` bit-exact for `standard_symmetric` / `winsorized` /
    `quantile` and a genuine correction only for
    `standard_symmetric_innovation`.

    :returns: ``(correction, label)``; the neutral vector with the label
        ``input-facade`` when either facade cannot be introspected.
    """
    total = int(obs_len or 0) + int(act_len or 0)
    neutral = np.ones(max(total, 0), dtype=float)
    out = resolve_target_space_scale(dynamics_model, obs_len, act_len, "output")
    inp = resolve_target_space_scale(dynamics_model, obs_len, act_len, "input")
    if out is None or inp is None or out[0].size != total or inp[0].size != total:
        return neutral, "input-facade (target scale unresolved)"
    correction = inp[0] / np.maximum(out[0], _JACOBIAN_EPS)
    if not np.all(np.isfinite(correction)):
        return neutral, "input-facade (target scale unresolved)"
    return correction, out[1]


def resolve_target_space_delta_std(
    dynamics_model,
    raw_delta_std: Sequence[float],
    obs_len: int,
    act_len: int,
) -> Tuple[np.ndarray, str]:
    """Express a per-dim raw one-step innovation std in the **target** space.

    RLRP-761 ``S8.4a``. This is the ``sigma_Delta`` of the budget-share formula

    .. math:: \\text{share}_d \\propto w_d \\, \\sigma_{\\Delta,d}^{\\text{tgt}\\,2}

    :param dynamics_model: the transition-model wrapper.
    :param raw_delta_std: per-single-step-feature ``sigma_Delta``, PHYSICAL units.
    :param obs_len: single-step observation width.
    :param act_len: single-step action width.
    :returns: ``(delta_std_in_target_space, label)``. When the facade cannot be
        introspected the raw values are returned unchanged with the label
        ``unresolved`` -- an explicit, loggable fallback rather than a silent
        assumption.
    """
    raw = np.asarray(list(raw_delta_std), dtype=np.float64)
    resolved = resolve_target_space_scale(dynamics_model, obs_len, act_len)
    if resolved is None or resolved[0].size != raw.size:
        return raw, "unresolved"
    scale, label = resolved
    return raw / scale, label
