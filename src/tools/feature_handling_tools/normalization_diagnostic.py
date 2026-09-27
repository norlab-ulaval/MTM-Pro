# coding=utf-8
"""Training-start feature-normalization diagnostic (RLRP-761 stage S2).

Permanent framework module.

Nothing in the run log used to state *what the network actually receives per
feature*. Both root causes identified by the RLRP-761 investigation were
therefore invisible:

- **H1** — under a symmetric facade the target is z-scored by the **state**
  std, which hands most of the one-step innovation energy to the
  terrain-vibration channels while every reported metric lives in raw space.
- **H2** — ``NormStrategy.IDENTITY`` conflated "exempt from robust warping" with
  "exempt from normalization", so the whole action block (``dt`` included) was
  passed through raw and contributed ~0 % of the input energy.

This module renders, once per training start (right after the normalizer
statistics are fitted and before the first gradient step), a per-feature table
of the resolved normalization contract.

Energy / gradient share definitions
-----------------------------------

``input_energy_share[d] = Var(x_d) / sum_j Var(x_j)`` over the resolved
**single-step** input vector, post-normalization. Under i.i.d. first-layer
initialization this is *simultaneously*

- the share of first-layer pre-activation variance contributed by channel
  ``d``, and
- the expected share of squared gradient received by the first-layer weights
  fanning in from ``d`` (since ``dL/dW[:, d] = delta * x_d``).

One number, both readings.

``target_energy_share[d] = w_d * Var(dy_d) / sum_j w_j * Var(dy_j)`` — the share
of the per-feature loss budget, including the per-feature loss weight ``w_d``.

Both are computed for the **resolved** configuration, so switching
``normalizer_type`` visibly changes the table.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# RLRP-761 S10.2 -- the target-space / normalizer scale primitives were extracted
# into a dedicated lower-layer module so this diagnostic (and the loss-weight
# resolver) are plain CONSUMERS of one source of truth. Re-exported here so the
# historical `from ...normalization_diagnostic import <name>` call sites (tests,
# collectors) keep working unchanged.
from tools.feature_handling_tools.feature_target_space import (  # noqa: F401
    _JACOBIAN_EPS,
    _MAX_ROWS,
    TARGET_SPACE_INNOVATION,
    TARGET_SPACE_LOCAL_LINEAR,
    TARGET_SPACE_RAW,
    TARGET_SPACE_STATE_SIGMA,
    _build_slot_index,
    _resolve_target_correction,
    _sub_jacobian_diag,
    _target_space_label,
    _to_numpy,
    _unwrap_base_normalizer,
    consume_target_space_degrade,
    reset_target_space_degrade,
    resolve_normalizer_output_std,
    resolve_target_space_delta_std,
    resolve_target_space_scale,
)

#: Flag thresholds (see the module docstring / the RLRP-761 plan, S2).
_UNSCALED_STD_BAND = (0.2, 5.0)
_NEAR_CONSTANT_FACTOR = 0.005
_NOISE_DOMINATED_RATIO = 1.5
_TARGET_HOG_FACTOR = 3.0
_DEGENERATE_STD = 1e-8
#: RLRP-761 S7.4 -- absolute share of the composed TARGET loss budget above
#: which a single feature is considered to have captured the objective. The
#: relative ``TARGET-HOG`` test (``> 3/D``) scales with the feature count and was
#: therefore satisfied, and ignored, by the run in which ``dt`` took 52.4 % of
#: the budget. This one does not scale: one feature owning more than 40 % of a
#: multi-feature objective is a misconfiguration regardless of ``D``.
_TARGET_DOMINATED_SHARE = 0.40

#: Tolerance on the ``TARGET-DOMINATED`` test.
#:
#: ``feature_loss_weights.max_target_share`` caps a dim AT its configured share;
#: capping one dim then renormalizing the others leaves the capped dim a few ULP
#: ABOVE the cap. With a strict ``>`` test, a run whose budget cap is working
#: EXACTLY as intended (``max_target_share: 0.40``) still raised the flag -- a
#: false alarm on the remedy itself, observed on the S7 ``T4`` cell. The
#: tolerance is absolute and far below any share difference that matters.
_TARGET_DOMINATED_TOL = 1e-4

#: Relative slack granted to an ARMED ``max_target_share`` cap.
#:
#: The absolute tolerance above is NOT sufficient, as the `S7` revision-2 debug
#: pass demonstrated: ``T4`` (cap ``0.40``) raised the flag on a share the table
#: renders as exactly ``40.0 %`` while ``T5`` (same cap) did not. The two numbers
#: are not computed from the same estimate --
#:
#:   * the resolver water-fills the cap using the ``sigma_Delta`` measured on the
#:     REPLAY-BUFFER slot sequences;
#:   * this diagnostic recomputes the share from the ``sigma_Delta`` measured on a
#:     SAMPLED batch.
#:
#: The disagreement is therefore an estimator-level ~1e-3, not a rounding ULP, and
#: it is one-sided in neither direction. Widening the absolute tolerance to cover
#: it would blunt the guard for the un-capped case, so the slack is instead granted
#: ONLY when a cap is actually armed, and only relative to that cap.
#:
#: RLRP-761 ``S9.6`` decision (2026-08-03) -- **the band is KEPT, deliberately**.
#: ``S9.5`` pinned the two estimators' disagreement to two STRUCTURAL, BOUNDED
#: causes rather than drift:
#:   * ``one_step_delta`` -- the Bessel correction (``torch.std`` unbiased
#:     ``ddof=1`` vs ``numpy.std`` population ``ddof=0``), agreement to
#:     ``rtol=1e-6`` after the ``sqrt(n/(n-1))`` factor;
#:   * ``noise_floor`` -- the median convention (``torch.median`` returns the
#:     lower middle element on an even count, ``numpy.median`` averages the two),
#:     bounded below 2 % and guard-asserted under 5 %.
#: The resolver (replay-buffer slot sequences) and this diagnostic (a sampled
#: batch) legitimately observe DIFFERENT populations, so the band is a correct
#: mitigation of an inherent difference -- and the ``S9.5`` regression test
#: converts any future growth into a hard test failure. Sharing a single
#: estimator would couple the two populations and remove that independent
#: cross-check for no correctness gain; the band is therefore preferred over the
#: shared-estimate alternative.
_ARMED_CAP_RELATIVE_SLACK = 0.05

#: RLRP-777 -- skip channel. `collect_feature_normalization_rows` keeps its public
#: `Optional[FeatureNormalizationReport]` return type (no caller changes), but records
#: WHY it returned `None` here so `log_feature_normalization_report` can print exactly
#: one `[feature-normalization]` line on STDOUT (which `ConsoleRecorder` tees into
#: `console.log`) instead of vanishing silently. A CUDA device split silenced this
#: diagnostic for weeks on RLRP-757-E4; a named skip reason makes the next occurrence
#: self-explaining.
_LAST_SKIP_REASON: Optional[str] = None


def _skip(reason: str) -> None:
    """Record *reason* for the last no-report exit and return ``None``.

    Used at every non-emitting exit of :func:`collect_feature_normalization_rows`
    so the emitter can surface the cause on stdout (RLRP-777).
    """
    global _LAST_SKIP_REASON
    _LAST_SKIP_REASON = reason
    return None


@dataclass(frozen=True)
class FeatureNormalizationRow:
    """One resolved feature dimension of the normalization contract."""

    name: str
    block: str  # 'obs' | 'act'
    kind: str  # FeatureKind value ('' when no handler is available)
    strategy: str  # resolved NormStrategy value
    base_enabled: bool  # normalize_dims[d]
    raw_mean: float
    raw_std: float
    raw_min: float
    raw_max: float
    raw_delta_std: float  # sigma_Delta (one-step innovation)
    noise_floor: float  # robust high-frequency floor
    norm_mean: float  # AFTER the resolved transform
    norm_std: float
    norm_absmax: float
    norm_delta_std: float
    slot_std_spread: float  # max-min of the per-history-slot normalized std
    input_energy_share: float
    target_energy_share: float  # NaN when the block is not a target
    is_target: bool
    loss_weight: float
    dr_sigma: float  # NaN when DR is disabled
    dr_over_norm_std: float  # dr_sigma / norm_std -- injected noise-to-signal
    dr_over_delta_std: float  # dr_sigma / norm_delta_std -- noise vs innovation
    dr_scale_is_stale: bool  # the absolute DR vector predates the contract
    flags: Tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable mapping (for the ``verbose`` dump)."""
        payload = asdict(self)
        payload["flags"] = list(self.flags)
        return payload


@dataclass(frozen=True)
class FeatureNormalizationReport:
    """The collected rows plus the context needed to render/interpret them."""

    rows: Tuple[FeatureNormalizationRow, ...]
    normalizer_type: str
    handler_name: str
    target_space: str  # 'NORMALIZED' | 'RAW (output_normalizer=None)'
    singlestep_obs_len: int
    singlestep_act_len: int
    history_len: int
    horizon_len: int
    in_size: int
    n_rows_sampled: int
    #: RLRP-761 S5.3 -- the resolved per-feature contract entries the
    #: ``standard`` single-facade path DROPPED (``{dim_name: strategy}``), or
    #: ``None`` when nothing was dropped. Rendered as a ``CONTRACT-DROPPED``
    #: banner so a ``standard`` run is VISIBLY not honouring its handler.
    dropped_feature_contract: Optional[Dict[str, str]] = None

    def contract_dropped(self) -> bool:
        """Whether the normalizer silently discarded the resolved contract."""
        return bool(self.dropped_feature_contract)

    def warning_flags(self) -> Tuple[str, ...]:
        """Return the subset of flags that must be surfaced at WARNING level."""
        actionable = {
            "UNSCALED",
            "TARGET-HOG",
            "TARGET-DOMINATED",
            "DEGENERATE",
            "DR-SWAMPED",
            "DR-STALE",
        }
        flags = {f for row in self.rows for f in row.flags if f in actionable}
        if self.contract_dropped():
            flags.add("CONTRACT-DROPPED")
        return tuple(sorted(flags))

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            k: v for k, v in asdict(self).items() if k != "rows"
        }
        payload["rows"] = [row.as_dict() for row in self.rows]
        return payload


# =================================================================================================
# Statistics helpers
# =================================================================================================


def _std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else float("nan")


def _noise_floor(slots: np.ndarray) -> float:
    """Robust high-frequency noise floor of a ``(N, H)`` per-slot signal.

    Estimated from the **second** difference along the history axis: for a
    smooth signal plus white noise ``e`` of std ``s``, ``Var(x[t+1] - 2 x[t] +
    x[t-1]) -> 6 s^2``. The MAD is used instead of the sample std so that rare
    adverse excursions (precisely the events this research targets) inflate the
    *innovation* ``sigma_Delta`` without inflating the noise floor they are
    compared against.
    """
    if slots.ndim != 2 or slots.shape[1] < 3:
        return float("nan")
    d2 = slots[:, 2:] - 2.0 * slots[:, 1:-1] + slots[:, :-2]
    mad = float(np.median(np.abs(d2 - np.median(d2))))
    return mad / (0.6745 * np.sqrt(6.0))


def _delta_std(slots: np.ndarray) -> float:
    """One-step innovation std of a ``(N, H)`` per-slot signal."""
    if slots.ndim != 2 or slots.shape[1] < 2:
        return float("nan")
    return float(np.std(slots[:, 1:] - slots[:, :-1]))


def _ratio(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator``, or ``NaN`` when it is meaningless."""
    if not (np.isfinite(numerator) and np.isfinite(denominator)):
        return float("nan")
    if denominator <= 0.0:
        return float("nan")
    return float(numerator) / float(denominator)


# =================================================================================================
# Collector
# =================================================================================================


def _resolve_handler(cfg):
    """Return the feature handler, or ``None`` when the config declares none."""
    try:
        from tools.feature_handling_tools.env_handlers import (
            instantiate_feature_handler,
        )

        return instantiate_feature_handler(cfg)
    except Exception:  # defensive: math env / exotic cfg shapes (R-F)
        return None


def _resolve_feature_weights(cfg, key: str, length: int) -> List[float]:
    """Read the resolved per-dim loss-weight vector, defaulting to all-ones.

    RLRP-761 ``S10``: :func:`resolve_and_apply_feature_loss_weights` materializes
    the resolved (post-cap) per-dim weights back into the node that DECLARED the
    criterion -- the canonical ``feature_loss_weights`` node for the reference
    config, or the deprecated ``ms_model`` node otherwise. The diagnostic runs at
    the training-start seam AFTER that resolution, so it must read the SAME node;
    reading only ``ms_model`` (as it once did) left every ``w`` at ``1.0`` and
    computed ``tgt%`` on a uniform-weight budget -- a misleading
    ``TARGET-DOMINATED`` that does not reflect the objective the model was given.

    A still-unresolved criterion (e.g. a bare ``'predictability'`` string on a
    path that never ran the resolver) is not interpretable dimension-wise here and
    falls through to all-ones.
    """
    try:
        from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist

        value = None
        # Canonical node first, then the deprecated `ms_model` alias.
        for node in ("feature_loss_weights", "ms_model"):
            if is_cfg_key_exist(cfg, f"{node}.{key}"):
                candidate = cfg[node][key]
                if candidate is not None:
                    value = candidate
                    break
        if value is None:
            return [1.0] * length
        if isinstance(value, (int, float)):
            return [float(value)] * length
        if isinstance(value, str):
            # An unresolved named criterion -- cannot be expanded per-dim here.
            return [1.0] * length
        weights = [float(v) for v in value]
        if len(weights) != length:
            return [1.0] * length
        return weights
    except Exception:
        return [1.0] * length


def _resolve_dr_sigma(dynamics_model, n_features: int) -> List[float]:
    """Return the effective per-feature DR noise std, or ``NaN`` when off."""
    nan = [float("nan")] * n_features
    model = getattr(dynamics_model, "model", None)
    getter = getattr(
        model, "get_train_time_domain_randomization_noise_scale", None
    )
    if getter is None:
        return nan
    try:
        scale = getter()
    except Exception:
        return nan
    if scale is None:
        return nan
    try:
        flat = np.asarray(_to_numpy(scale)).reshape(-1)
    except Exception:
        return nan
    if flat.size == 0:
        return nan
    if flat.size == 1:
        return [float(flat[0])] * n_features
    if flat.size % n_features != 0:
        return nan
    # The scale is tiled over the history/horizon axis; report the single-step
    # block (the tiling is by construction identical across slots).
    return [float(v) for v in flat[:n_features]]


def _resolve_randomizer(dynamics_model):
    """Return the train-time domain randomizer, or ``None``."""
    model = getattr(dynamics_model, "model", None)
    randomizer = getattr(model, "train_time_domain_randomizer", None)
    if randomizer is None or not bool(getattr(randomizer, "enable", False)):
        return None
    return randomizer


def _resolve_dr_staleness(
    dynamics_model, strategies: Sequence[str]
) -> List[bool]:
    """Per-dim ``DR-STALE`` verdict (RLRP-761 ``S2.8``).

    A calibrated ``per_feature_scale`` is expressed in normalizer-OUTPUT units
    (the randomizer perturbs already-normalized tensors), so it is silently
    invalidated by any change to the normalization contract — risk ``R-I``: the
    ``S1.5`` handler flip multiplies the *relative* noise injected on ``dt`` by
    ~90 without changing a single configured number.

    Verdict, only for ``per_feature_scale_mode: 'absolute'`` (the dimensionless
    ``relative_to_normalized_std`` vector cannot go stale):

    - when the config RECORDS the calibration contract
      (``per_feature_scale_calibrated_strategy``): stale exactly where the
      resolved strategy differs from the recorded one;
    - otherwise (nothing recorded, the current state of every config): stale on
      every ``zscore`` dim, since those are precisely the dims whose output
      scale the ``S1.5`` flip changed.
    """
    n = len(strategies)
    randomizer = _resolve_randomizer(dynamics_model)
    if randomizer is None:
        return [False] * n
    if getattr(randomizer, "per_feature_scale_mode", "absolute") != "absolute":
        return [False] * n
    recorded = getattr(randomizer, "per_feature_scale_calibrated_strategy", None)
    if recorded is not None and len(recorded) == n:
        return [str(recorded[i]) != str(strategies[i]) for i in range(n)]
    return [str(s) == "zscore" for s in strategies]


def collect_feature_normalization_rows(
    cfg,
    dynamics_model,
    replay_buffer,
    max_rows: int = _MAX_ROWS,
) -> Optional[FeatureNormalizationReport]:
    """Collect the resolved per-feature normalization contract.

    Read-only: no statistics are fitted, no model state is touched. Returns
    ``None`` when the configuration cannot be introspected (no multistep model,
    no replay buffer, no fitted normalizer) so the caller can stay a silent
    no-op (risk ``R-F``).

    :param cfg: the resolved Hydra config.
    :param dynamics_model: the ``OneDTransitionRewardModel`` (V2) wrapper, with
        its normalizer statistics already fitted.
    :param replay_buffer: the mbrl-lib replay buffer.
    :param max_rows: cap on the number of transitions sampled.
    """
    # RLRP-777 -- start each pass with a clean degrade accumulator so the emitter
    # reports only the reasons produced by THIS collection.
    reset_target_space_degrade()
    model = getattr(dynamics_model, "model", None)
    if model is None:
        return _skip("dynamics_model exposes no wrapped model")
    obs_len = getattr(model, "singlestep_obs_len", None)
    act_len = getattr(model, "singlestep_act_len", None)
    history_len = getattr(model, "history_len", None)
    if not obs_len or act_len is None or not history_len:
        return _skip(
            "model exposes no singlestep_obs_len / singlestep_act_len / history_len "
            f"(obs_len={obs_len!r}, act_len={act_len!r}, history_len={history_len!r})"
        )
    if getattr(dynamics_model, "input_normalizer", None) is None:
        return _skip("dynamics_model exposes no fitted input_normalizer")

    try:
        batch = replay_buffer.get_all()
        raw_obs = _to_numpy(batch.obs)
        raw_act = _to_numpy(batch.act)
    except Exception as exc:
        return _skip(f"replay_buffer.get_all() failed: {exc!r}")
    if raw_obs.ndim != 2 or raw_obs.shape[0] == 0:
        return _skip(f"raw obs is not a non-empty 2-D array (shape={raw_obs.shape})")
    if raw_obs.shape[0] > max_rows:
        raw_obs = raw_obs[:max_rows]
        raw_act = raw_act[:max_rows]

    try:
        raw_in = np.concatenate([raw_obs, raw_act], axis=-1)
        with torch.no_grad():
            # Torch-first: feed tensors so the mbrl `to_tensor` numpy-input
            # UserWarning is not raised from this diagnostic path (the raw numpy
            # arrays above are still used for the per-slot raw statistics below).
            norm_in = _to_numpy(
                dynamics_model._get_model_input(
                    torch.as_tensor(raw_obs), torch.as_tensor(raw_act)
                )
            )
    except Exception as exc:
        return _skip(f"_get_model_input raised: {exc!r}")
    if norm_in.shape != raw_in.shape:
        return _skip(
            "normalized/raw input width mismatch "
            f"({norm_in.shape} vs {raw_in.shape})"
        )

    handler = _resolve_handler(cfg)
    names, kinds, strategies, base_enabled = _resolve_feature_metadata(
        handler, obs_len, act_len
    )
    n_features = obs_len + act_len

    # Flattened input layout: [obs: history_len x obs_len][act: history_len x act_len].
    slot_index = _build_slot_index(obs_len, act_len, history_len, raw_in.shape[-1])
    if slot_index is None:
        return _skip(
            "slot index could not be built "
            f"(obs_len={obs_len}, act_len={act_len}, history_len={history_len}, "
            f"width={raw_in.shape[-1]})"
        )

    horizon_len = int(getattr(model, "horizon_len", 1) or 1)
    uses_block_facade = bool(
        getattr(dynamics_model, "output_normalizer", None) is not None
    )
    # RLRP-761 S8.4c/S8.4d -- the `tgt%` column is measured on the INPUT facade,
    # which is the target transform for every type that SHARES its sub-normalizers
    # between the two facades (`standard_symmetric`, `winsorized`, `quantile`) but
    # NOT for `standard_symmetric_innovation`, which decouples them by design.
    # `target_correction` maps the measured input-space innovation to target space
    # and is exactly 1.0 (bit-exact) in the shared case.
    target_correction, target_label = _resolve_target_correction(
        dynamics_model, obs_len, act_len
    )
    target_space = (
        f"NORMALIZED ({target_label})"
        if uses_block_facade
        else "RAW (output_normalizer=None)"
    )
    obs_weights = _resolve_feature_weights(cfg, "obs_feature_weights", obs_len)
    act_weights = _resolve_feature_weights(cfg, "act_feature_weights", act_len)
    armed_cap = _resolve_armed_target_share(cfg)
    dr_sigma = _resolve_dr_sigma(dynamics_model, n_features)
    dr_stale = _resolve_dr_staleness(dynamics_model, strategies)

    raw_stats, norm_stats = [], []
    for idx in range(n_features):
        cols = slot_index[idx]
        raw_slots = raw_in[:, cols]
        norm_slots = norm_in[:, cols]
        # Under `standard` the target is RAW, so the target-side statistics are
        # simply the raw ones (there is no output transform to apply).
        tgt_slots = norm_slots if uses_block_facade else raw_slots
        raw_stats.append(
            {
                "mean": float(np.mean(raw_slots)),
                "std": _std(raw_slots),
                "min": float(np.min(raw_slots)),
                "max": float(np.max(raw_slots)),
                "delta_std": _delta_std(raw_slots),
                "noise_floor": _noise_floor(raw_slots),
            }
        )
        per_slot_std = np.std(norm_slots, axis=0)
        norm_stats.append(
            {
                "mean": float(np.mean(norm_slots)),
                "std": _std(norm_slots),
                "absmax": float(np.max(np.abs(norm_slots))),
                "delta_std": _delta_std(tgt_slots) * float(target_correction[idx]),
                "spread": float(np.max(per_slot_std) - np.min(per_slot_std)),
            }
        )

    input_energy = np.array([s["std"] ** 2 for s in norm_stats], dtype=float)
    input_share = _share(input_energy)

    weights = list(obs_weights) + list(act_weights)
    # The action block is part of the composed target only for a multi-step
    # horizon (`_compose_next_obs_multistep_act_horizon_slice`).
    is_target = [True] * obs_len + [horizon_len > 1] * act_len
    target_energy = np.array(
        [
            (weights[i] * norm_stats[i]["delta_std"] ** 2) if is_target[i] else 0.0
            for i in range(n_features)
        ],
        dtype=float,
    )
    target_share = _share(target_energy)

    rows = []
    for idx in range(n_features):
        share_in = float(input_share[idx])
        share_tgt = float(target_share[idx]) if is_target[idx] else float("nan")
        row = FeatureNormalizationRow(
            name=names[idx],
            block="obs" if idx < obs_len else "act",
            kind=kinds[idx],
            strategy=strategies[idx],
            base_enabled=base_enabled[idx],
            raw_mean=raw_stats[idx]["mean"],
            raw_std=raw_stats[idx]["std"],
            raw_min=raw_stats[idx]["min"],
            raw_max=raw_stats[idx]["max"],
            raw_delta_std=raw_stats[idx]["delta_std"],
            noise_floor=raw_stats[idx]["noise_floor"],
            norm_mean=norm_stats[idx]["mean"],
            norm_std=norm_stats[idx]["std"],
            norm_absmax=norm_stats[idx]["absmax"],
            norm_delta_std=norm_stats[idx]["delta_std"],
            slot_std_spread=norm_stats[idx]["spread"],
            input_energy_share=share_in,
            target_energy_share=share_tgt,
            is_target=is_target[idx],
            loss_weight=weights[idx],
            dr_sigma=dr_sigma[idx],
            dr_over_norm_std=_ratio(dr_sigma[idx], norm_stats[idx]["std"]),
            dr_over_delta_std=_ratio(dr_sigma[idx], norm_stats[idx]["delta_std"]),
            dr_scale_is_stale=bool(dr_stale[idx]),
            flags=(),
        )
        rows.append(_with_flags(row, n_features, armed_cap=armed_cap))

    return FeatureNormalizationReport(
        rows=tuple(rows),
        normalizer_type=str(getattr(dynamics_model, "normalizer_type", "?")),
        handler_name=type(handler).__name__ if handler is not None else "none",
        target_space=target_space,
        singlestep_obs_len=int(obs_len),
        singlestep_act_len=int(act_len),
        history_len=int(history_len),
        horizon_len=horizon_len,
        in_size=int(raw_in.shape[-1]),
        n_rows_sampled=int(raw_in.shape[0]),
        # RLRP-761 S5.3 -- recorded by the normalizer facade when the `standard`
        # single-facade path discarded the resolved contract.
        dropped_feature_contract=(
            {
                str(k): str(v)
                for k, v in (
                    getattr(dynamics_model, "dropped_feature_contract", None) or {}
                ).items()
            }
            or None
        ),
    )


def collect_sequence_normalization_rows(
    feature_names: Sequence[str],
    raw_sequences: Sequence[np.ndarray],
    normalized_sequences: Sequence[np.ndarray],
    normalizer_type: str,
    strategies: Optional[Sequence[str]] = None,
    base_enabled: Optional[Sequence[bool]] = None,
    n_obs: Optional[int] = None,
    target_space: str = "NORMALIZED",
    armed_target_share: Optional[float] = None,
) -> Optional[FeatureNormalizationReport]:
    """Build the S2 report from **raw trajectory sequences** instead of a model.

    RLRP-761 ``S2.6``. The model-free DR calibration
    (:func:`deploy_drift_eval.compute_per_feature_one_step_delta_std`) has no
    ``dynamics_model`` and no replay buffer, but it *does* fit the very same
    normalizer on the very same features — so it must be able to print the very
    same table. This entry point takes the two aligned tensor lists it already
    has in hand (one array per trajectory, shape ``(T, D)``) and produces a
    :class:`FeatureNormalizationReport` renderable by
    :func:`render_feature_normalization_table`.

    Time-ordering matters: ``sigma_Delta`` and the noise floor are computed
    **within** each trajectory, never across a trajectory boundary.

    :param feature_names: The resolved per-feature names (length ``D``).
    :param raw_sequences: Per-trajectory raw arrays ``(T, D)``.
    :param normalized_sequences: The same trajectories after the fitted
        transform (pass ``raw_sequences`` for the raw/identity regime).
    :param normalizer_type: The type the statistics were fitted with.
    :param strategies: Optional per-dim resolved strategy values.
    :param base_enabled: Optional per-dim ``normalize_dims`` mask.
    :param n_obs: Number of leading dims belonging to the ``obs`` block
        (default: all of them).
    :param target_space: ``'NORMALIZED'`` or ``'RAW (output_normalizer=None)'``.
    :returns: the report, or ``None`` when the inputs are unusable.
    """
    if not raw_sequences or not normalized_sequences:
        return None
    n_features = len(feature_names)
    obs_len = n_features if n_obs is None else int(n_obs)
    strategies = list(strategies or ["inherit"] * n_features)
    base_enabled = list(base_enabled or [True] * n_features)
    if len(strategies) != n_features or len(base_enabled) != n_features:
        return None

    def _slots(sequences: Sequence[np.ndarray], dim: int) -> np.ndarray:
        """Return a ``(N, 3)`` sliding window over each trajectory, per feature."""
        windows = [
            np.stack([s[:-2, dim], s[1:-1, dim], s[2:, dim]], axis=1)
            for s in sequences
            if np.asarray(s).shape[0] >= 3
        ]
        if not windows:
            return np.empty((0, 3))
        return np.concatenate(windows, axis=0)

    rows: List[FeatureNormalizationRow] = []
    norm_stats: List[Dict[str, float]] = []
    raw_stats: List[Dict[str, float]] = []
    for d in range(n_features):
        raw_slots = _slots(raw_sequences, d)
        norm_slots = _slots(normalized_sequences, d)
        if raw_slots.size == 0 or norm_slots.size == 0:
            return None
        raw_stats.append(
            {
                "mean": float(np.mean(raw_slots)),
                "std": _std(raw_slots),
                "min": float(np.min(raw_slots)),
                "max": float(np.max(raw_slots)),
                "delta_std": _delta_std(raw_slots),
                "noise_floor": _noise_floor(raw_slots),
            }
        )
        norm_stats.append(
            {
                "mean": float(np.mean(norm_slots)),
                "std": _std(norm_slots),
                "absmax": float(np.max(np.abs(norm_slots))),
                "delta_std": _delta_std(norm_slots),
            }
        )

    input_share = _share(np.array([s["std"] ** 2 for s in norm_stats], dtype=float))
    target_share = _share(
        np.array([s["delta_std"] ** 2 for s in norm_stats], dtype=float)
    )
    for d in range(n_features):
        row = FeatureNormalizationRow(
            name=str(feature_names[d]),
            block="obs" if d < obs_len else "act",
            kind="",
            strategy=str(strategies[d]),
            base_enabled=bool(base_enabled[d]),
            raw_mean=raw_stats[d]["mean"],
            raw_std=raw_stats[d]["std"],
            raw_min=raw_stats[d]["min"],
            raw_max=raw_stats[d]["max"],
            raw_delta_std=raw_stats[d]["delta_std"],
            noise_floor=raw_stats[d]["noise_floor"],
            norm_mean=norm_stats[d]["mean"],
            norm_std=norm_stats[d]["std"],
            norm_absmax=norm_stats[d]["absmax"],
            norm_delta_std=norm_stats[d]["delta_std"],
            slot_std_spread=float("nan"),
            input_energy_share=float(input_share[d]),
            target_energy_share=float(target_share[d]),
            is_target=True,
            loss_weight=1.0,
            dr_sigma=float("nan"),
            dr_over_norm_std=float("nan"),
            dr_over_delta_std=float("nan"),
            dr_scale_is_stale=False,
            flags=(),
        )
        rows.append(_with_flags(row, n_features, armed_cap=armed_target_share))

    return FeatureNormalizationReport(
        rows=tuple(rows),
        normalizer_type=str(normalizer_type),
        handler_name="none (model-free calibration)",
        target_space=target_space,
        singlestep_obs_len=obs_len,
        singlestep_act_len=n_features - obs_len,
        history_len=1,
        horizon_len=1,
        in_size=n_features,
        n_rows_sampled=int(sum(np.asarray(s).shape[0] for s in raw_sequences)),
    )


def _share(energy: np.ndarray) -> np.ndarray:
    total = float(np.nansum(energy))
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(energy, np.nan)
    return energy / total


def _resolve_feature_metadata(handler, obs_len: int, act_len: int):
    """Resolve per-dim names / kinds / strategies / base-enabled from *handler*."""
    n = obs_len + act_len
    names = [f"obs[{i}]" for i in range(obs_len)] + [
        f"act[{j}]" for j in range(act_len)
    ]
    kinds = [""] * n
    strategies = ["inherit"] * n
    base_enabled = [True] * n
    if handler is None:
        return names, kinds, strategies, base_enabled
    try:
        handler_names = list(handler.feature_dim_names())
        if len(handler_names) == n:
            names = handler_names
        kwargs = handler.build_normalizer_kwargs()
        strat = list(kwargs["per_dim_strategy"])
        if len(strat) == n:
            strategies = [str(s) for s in strat]
        mask = kwargs["normalize_dims"]
        base_enabled = [bool(mask.get(name, True)) for name in names]
        kind_by_name = {
            dim: group.kind.value
            for group in list(handler.obs_groups) + list(handler.act_groups)
            for dim in group.dim_names
        }
        kinds = [kind_by_name.get(name, "") for name in names]
    except Exception:
        pass
    return names, kinds, strategies, base_enabled


def _resolve_armed_target_share(cfg) -> Optional[float]:
    """Read ``feature_loss_weights.max_target_share`` tolerantly.

    Returns ``None`` when the node, the key or a numeric value is absent, so a
    config that never opted into the budget cap keeps the plain absolute guard.
    """
    try:
        policy = cfg.get("feature_loss_weights", None)
        if policy is None:
            return None
        value = policy.get("max_target_share", None)
        if value is None:
            return None
        return float(value)
    except (AttributeError, TypeError, ValueError):
        return None


def _is_target_dominated(share: float, armed_cap: Optional[float]) -> bool:
    """Decide the ``TARGET-DOMINATED`` guard, honouring an armed cap.

    The guard is the plain absolute :data:`_TARGET_DOMINATED_SHARE`, with ONE
    exception: a share sitting in the narrow band ``[c, c*(1 + slack)]`` of an
    armed ``max_target_share = c`` is the cap **binding exactly**, i.e. the remedy
    working, and is not reported as the defect.

    The exception is deliberately a *band* and not a raised threshold. Raising the
    threshold to ``c`` would mean that arming a LOOSE cap (say ``0.80``) silently
    suppresses a genuine 52 % dominance -- the cap is a *policy* knob, the flag is
    a *diagnostic*, and a policy that permits a pathology must not also hide it.
    With the band, a loose cap that is not binding changes nothing.
    """
    if not np.isfinite(share) or share <= _TARGET_DOMINATED_SHARE + _TARGET_DOMINATED_TOL:
        return False
    if armed_cap is None or not np.isfinite(armed_cap) or armed_cap >= 1.0:
        return True
    cap = float(armed_cap)
    cap_is_binding_exactly = cap <= share <= cap * (1.0 + _ARMED_CAP_RELATIVE_SLACK)
    return not cap_is_binding_exactly


def _with_flags(
    row: FeatureNormalizationRow,
    n_features: int,
    armed_cap: Optional[float] = None,
):
    """Return *row* with its resolved flag tuple.

    :param armed_cap: the configured ``feature_loss_weights.max_target_share``,
        when a budget cap is armed. Passed so the ``TARGET-DOMINATED`` guard does
        not fire on its own remedy -- see
        :data:`_ARMED_CAP_RELATIVE_SLACK`.
    """
    flags: List[str] = []
    if row.raw_std < _DEGENERATE_STD:
        flags.append("DEGENERATE")
    if not row.base_enabled and row.strategy in ("identity", "inherit"):
        if not (_UNSCALED_STD_BAND[0] <= row.norm_std <= _UNSCALED_STD_BAND[1]):
            flags.append("UNSCALED")
    if (
        np.isfinite(row.input_energy_share)
        and row.input_energy_share < _NEAR_CONSTANT_FACTOR / max(n_features, 1)
    ):
        flags.append("NEAR-CONSTANT")
    if (
        np.isfinite(row.raw_delta_std)
        and np.isfinite(row.noise_floor)
        and row.noise_floor > 0.0
        and row.raw_delta_std / row.noise_floor < _NOISE_DOMINATED_RATIO
    ):
        flags.append("NOISE-DOMINATED")
    if (
        np.isfinite(row.target_energy_share)
        and row.target_energy_share > _TARGET_HOG_FACTOR / max(n_features, 1)
    ):
        flags.append("TARGET-HOG")
    # RLRP-761 S7.4: an ABSOLUTE budget-share guard. `TARGET-HOG` is relative to
    # the feature count, so it fires often enough to be read as background noise;
    # this one fires only when a single feature has effectively become the
    # objective, and it is escalated to a banner by the renderer.
    if _is_target_dominated(row.target_energy_share, armed_cap):
        flags.append("TARGET-DOMINATED")
    if np.isfinite(row.dr_over_delta_std) and row.dr_over_delta_std > 1.0:
        flags.append("DR-SWAMPED")
    if row.dr_scale_is_stale:
        flags.append("DR-STALE")
    return FeatureNormalizationRow(
        **{
            **{k: v for k, v in asdict(row).items() if k != "flags"},
            "flags": tuple(flags),
        }
    )


# =================================================================================================
# Renderer
# =================================================================================================

#: Column specification -- the SINGLE source of truth for the layout.
#:
#: ``(group title, ((column label, width, right_aligned), ...))``. The header,
#: the group band, the separators and the data cells are all generated from
#: this, so a width can never drift between the header and the rows (the
#: previous renderer hand-aligned a literal header string against a hand-built
#: f-string, and the two had diverged).
_COLUMNS: Tuple[Tuple[str, Tuple[Tuple[str, int, bool], ...]], ...] = (
    (
        "resolved feature contract",
        (
            ("#", 2, True),
            ("feature", 24, False),
            ("blk", 3, False),
            ("kind", 16, False),
            ("contract", 10, False),
            ("effective", 10, False),
            ("base", 4, False),
        ),
    ),
    (
        "raw (physical units)",
        (("mean", 9, True), ("std", 8, True), ("σΔ", 9, True)),
    ),
    (
        "normalized (model)",
        (("mean", 8, True), ("std", 7, True), ("σΔ", 6, True)),
    ),
    ("energy %", (("in%", 5, True), ("tgt%", 5, True))),
    (
        "loss w & DR noise",
        (("w", 5, True), ("dr σ", 8, True), ("dr/σn", 5, True), ("dr/σΔ", 5, True)),
    ),
    ("", (("flags", 26, False),)),
)

#: Symbol glossary rendered under every table.
_LEGEND: Tuple[Tuple[str, str], ...] = (
    ("blk", "feature block — `obs`ervation or `act`ion"),
    ("kind", "the FeatureKind of the feature contract"),
    (
        "contract / effective",
        "`contract` = the AUTHORED, type-agnostic NormStrategy declared by the "
        "handler; `effective` = the map ACTUALLY applied for the active "
        "normalizer_type. They differ only where the contract would behave "
        "differently under another type: e.g. under `standard_symmetric_innovation` "
        "(no robust stage) an `inherit` obs dim and a `zscore` control dim both "
        "read `effective=zscore` (identical affine map), whereas under `winsorized` "
        "the `inherit` dim reads `winsorized` and the `zscore` dim reads `zscore`",
    ),
    (
        "base",
        "`on` = the base normalizer's OWN transform is applied to this dim; "
        "`OFF` = a `zscore`/`unit_norm`/`identity` strategy replaced it. `OFF` "
        "does NOT mean unscaled: `zscore` still divides by the base's fitted "
        "scale, so on a `standard_symmetric_innovation` TARGET facade a `zscore` "
        "dim (e.g. the commands / `dt`) is still innovation-scaled",
    ),
    (
        "norm std / norm σΔ",
        "the two 'normalized (model)' scale columns can live in DIFFERENT spaces "
        "under `standard_symmetric_innovation`: `std` is the model INPUT (state-std, "
        "≈1), `σΔ` is the model TARGET (innovation). A near-constant command "
        "therefore shows `std`≈1 yet a very large `σΔ` — both are correct",
    ),
    (
        "σΔ",
        "one-step innovation std — the scale of `x[t+1] - x[t]`, i.e. what the "
        "model must actually predict (as opposed to `std`, the state spread)",
    ),
    (
        "in%",
        "share of the first-layer pre-activation energy contributed by this dim; "
        "under i.i.d. init this is ALSO its expected share of the gradient",
    ),
    (
        "tgt%",
        "share of the composed multi-step target loss budget, `w_d · σΔ,norm²` "
        "renormalized; `n/a` when the dim is not part of the target. This is the "
        "column to read for supervision: a dim can have `w`≈0 yet a large `tgt%` "
        "when its normalized σΔ is large (it saturates the budget cap)",
    ),
    (
        "w",
        "resolved per-feature loss weight (mean-normalized). A printed `0.00` on a "
        "large-σΔ dim means 'weight so small it already saturates `max_target_share`' "
        "— read `tgt%`, not `w`, to judge whether a dim is supervised",
    ),
    (
        "dr σ / dr/σn / dr/σΔ",
        "train-time domain-randomization noise injected into this dim: absolute, "
        "relative to the normalized std, and relative to the innovation",
    ),
)

#: One-line definition of every flag the collector can raise. Only the flags
#: actually present in a report are rendered, so the table stays terse when the
#: contract is healthy but is self-explanatory the moment it is not.
_FLAG_DEFINITIONS: Dict[str, str] = {
    "DEGENERATE": (
        "the raw std is numerically zero — the dim carries no information and "
        "its normalization is ill-conditioned."
    ),
    "UNSCALED": (
        "the base normalizer is OFF and the dim reaches the network far from "
        "unit scale — it is either invisible to, or dominating, the first layer."
    ),
    "NEAR-CONSTANT": (
        "the dim contributes essentially no input energy, hence essentially no "
        "gradient: the model is blind to it."
    ),
    "NOISE-DOMINATED": (
        "the one-step innovation is at the robust high-frequency noise floor — "
        "this dim is mostly unpredictable at the model's timestep."
    ),
    "TARGET-HOG": (
        "the dim *hogs* (monopolizes) the loss budget: it holds more than 3x its "
        "even share of the composed target budget (threshold `3/D`, i.e. RELATIVE "
        "to the feature count)."
    ),
    "TARGET-DOMINATED": (
        "this dim ALONE holds more than "
        f"{100.0 * _TARGET_DOMINATED_SHARE:.0f} % of the target loss budget — it "
        "has effectively become the objective. Absolute threshold, does not "
        "scale with the feature count."
    ),
    "DR-SWAMPED": (
        "the injected domain-randomization noise exceeds the innovation the "
        "model must predict on this dim — the supervision signal is buried."
    ),
    "DR-STALE": (
        "the absolute `per_feature_scale` vector was authored for a different "
        "normalization contract than the resolved one; re-calibrate it or use "
        "`per_feature_scale_mode: relative_to_normalized_std`."
    ),
}



def _fmt(value: float, width: int, precision: int = 4) -> str:
    """Right-justified float guaranteed to fit in exactly *width* characters.

    A Python format ``width`` is only a *minimum*, so a large magnitude — e.g.
    an innovation-normalized command/`dt` σΔ in the thousands under ``S12`` — would
    overflow the column and shift every cell to its right, breaking the table
    alignment. This degrades the fixed-point precision first, then falls back to a
    compact scientific form, so ``len(result) <= width`` always holds.

    A very small *non-zero* value is shown in scientific notation (e.g.
    ``1e-04``) rather than being rounded to ``0.0000``, so a genuinely tiny
    quantity is never confused with a true zero; only an exact ``0.0`` renders as
    ``0``.
    """
    if value is None or not np.isfinite(value):
        return "---".rjust(width)
    if value == 0.0:
        # A true zero: prefer the plain fixed-point ``0.0`` form.
        for p in range(precision, -1, -1):
            text = f"{0.0:.{p}f}"
            if len(text) <= width:
                return text.rjust(width)
        return "0".rjust(width)
    # Non-zero: try fixed-point, but REJECT any precision that rounds a genuine
    # value down to zero (all-zero digits) -- that is what the scientific
    # fallback below exists to avoid.
    for p in range(precision, -1, -1):
        text = f"{value:.{p}f}"
        if len(text) <= width and float(text) != 0.0:
            return text.rjust(width)
    # Either too large for fixed-point, or too small (would round to zero):
    # fall back to a compact scientific form.
    for p in range(precision, -1, -1):
        text = f"{value:.{p}e}"
        if len(text) <= width:
            return text.rjust(width)
    return f"{value:.0e}"[-width:].rjust(width)


def _fmt_pct(value: float, width: int = 5) -> str:
    """Right-justified percentage guaranteed to fit in exactly *width* chars.

    Mirrors ``_fmt``'s small-value contract: a genuinely tiny *non-zero* share
    (e.g. an action target whose normalized σΔ is ~1e-4, so its
    ``w_d · σΔ,norm²`` budget is orders of magnitude below the obs dims) would
    round to ``0.0`` under a plain ``%.1f`` and be indistinguishable from a true
    zero. Such a value is instead shown in compact scientific notation, while an
    exact ``0.0`` still renders as ``0.0``.
    """
    if value is None or not np.isfinite(value):
        return "---".rjust(width)
    pct = 100.0 * value
    if pct == 0.0:
        return f"{0.0:.1f}".rjust(width)
    # Fixed-point first, but REJECT any precision that rounds a genuine non-zero
    # share down to all-zero digits -- that is what the scientific fallback below
    # exists to avoid.
    text = f"{pct:.1f}"
    if len(text) <= width and float(text) != 0.0:
        return text.rjust(width)
    # Too small (would round to zero) or too wide: fall back to scientific.
    for p in range(2, -1, -1):
        text = f"{pct:.{p}e}"
        if len(text) <= width:
            return text.rjust(width)
    return f"{pct:.0e}"[-width:].rjust(width)


def _cell(text: str, width: int, right: bool) -> str:
    """Fit *text* to exactly *width* characters."""
    text = text if len(text) <= width else text[:width]
    return text.rjust(width) if right else text.ljust(width)


def _group_widths(columns) -> List[int]:
    """Inner width of each group (columns joined by a single space)."""
    return [
        sum(width for _, width, _ in cols) + max(len(cols) - 1, 0)
        for _, cols in columns
    ]


def _row_line(cells_per_group: Sequence[Sequence[str]]) -> str:
    return "│" + "│".join(" " + " ".join(cells) + " " for cells in cells_per_group) + "│"


def _rule(widths: Sequence[int], left: str, join: str, right: str) -> str:
    return left + join.join("─" * (width + 2) for width in widths) + right


def _banner_lines(text: str, total_width: int, prefix: str = "") -> List[str]:
    """Word-wrap *text* into full-width ``│ ... │`` lines."""
    inner = total_width - 4
    words, lines, current = text.split(), [], prefix
    for word in words:
        candidate = f"{current}{word} "
        if len(candidate) > inner + 1 and current.strip():
            lines.append(current.rstrip())
            current = " " * len(prefix) + word + " "
        else:
            current = candidate
    if current.strip():
        lines.append(current.rstrip())
    return [f"│ {line.ljust(inner)} │" for line in lines]


#: Normalizer types whose transform has NO robust warping stage, so a
#: base-applied (`inherit`) dim reduces to a plain affine z-score by the base's
#: fitted scale. For these, `inherit` and `zscore` apply the IDENTICAL map --
#: the `effective` column collapses both to `zscore` to make that explicit,
#: which is exactly the `standard_symmetric_innovation` confusion this resolves.
_AFFINE_NORMALIZER_TYPES = frozenset(
    {"standard", "standard_symmetric", "standard_symmetric_innovation"}
)
_ROBUST_NORMALIZER_TYPES = frozenset({"winsorized", "quantile"})


def _effective_strategy(
    strategy: str, base_enabled: bool, normalizer_type: str
) -> str:
    """The transform actually APPLIED to a dim for the active ``normalizer_type``.

    The ``contract`` column reports the AUTHORED, type-agnostic ``NormStrategy``;
    this reports the map realized here. The point (RLRP-761): under
    ``standard_symmetric_innovation`` an ``inherit`` observation dim and a
    ``zscore`` control dim BOTH read ``zscore`` -- they apply the identical
    affine map, since the type has no robust stage to bypass -- whereas under
    ``winsorized`` the ``inherit`` dim reads ``winsorized`` and the ``zscore``
    dim still reads ``zscore``. So the two labels stop looking gratuitously
    different exactly when they are, in fact, the same.
    """
    if not base_enabled:
        # A strategy-aware wrapper replaced the base's own transform; the
        # authored strategy IS the applied map (identity / unit_norm / zscore).
        return strategy
    # Base applied. A forced robust strategy keeps its own map.
    if strategy in _ROBUST_NORMALIZER_TYPES:
        return strategy
    # `inherit` (or any base-applied affine): collapse to the realized map.
    if normalizer_type in _AFFINE_NORMALIZER_TYPES:
        return "zscore"
    if normalizer_type in _ROBUST_NORMALIZER_TYPES:
        return normalizer_type
    return strategy


def render_feature_normalization_table(report: FeatureNormalizationReport) -> str:
    """Render *report* as a fixed-width, fully closed console table.

    Layout contract:

    - a **title block** (top of the box) carrying the resolved
      ``normalizer_type``, the handler, the **target space** — the point of
      ``S2.3``/``S2.7``, so the ``norm``/``tgt%`` columns are never read as
      implying a transform that does not exist (under ``normalizer_type:
      standard`` the target is raw) — and the layout summary;
    - a **two-tier header**: a group band (which statistic space each block of
      columns lives in) above the column labels;
    - the data rows, every column right-bordered, the box closed on both sides;
    - a **legend** naming every abbreviation, and a **definition of each flag
      actually raised**, so the table is self-contained in a run log.

    Every width comes from :data:`_COLUMNS`; nothing is hand-aligned.
    """
    flag_width = max(
        len("flags"),
        max((len(" ".join(row.flags)) for row in report.rows), default=0),
    )
    columns = tuple(
        (
            title,
            tuple(cols)
            if cols[0][0] != "flags"
            else (("flags", flag_width, False),),
        )
        for title, cols in _COLUMNS
    )
    widths = _group_widths(columns)
    total_width = sum(width + 2 for width in widths) + len(widths) + 1
    # Widths of the identity block, taken from the spec so a data cell can never
    # drift from its header.
    id_w = [width for _, width, _ in columns[0][1]]

    lines: List[str] = []

    # -- Title block ------------------------------------------------------------------------
    box_title = " Feature normalization contract (RLRP-761 S2) "
    lines.append(
        "╭─" + box_title + "─" * max(total_width - len(box_title) - 3, 0) + "╮"
    )
    lines.extend(
        _banner_lines(
            f"normalizer_type='{report.normalizer_type}' · "
            f"handler='{report.handler_name}' · "
            f"target space: {report.target_space}",
            total_width,
        )
    )
    lines.extend(
        _banner_lines(
            f"obs {report.singlestep_obs_len} · act {report.singlestep_act_len} · "
            f"history_len {report.history_len} · horizon_len {report.horizon_len} · "
            f"in_size {report.in_size} · rows sampled {report.n_rows_sampled}",
            total_width,
        )
    )

    # -- Header (group band + column labels) ------------------------------------------------
    lines.append(_rule(widths, "├", "┬", "┤"))
    lines.append(
        _row_line(
            [
                [_cell(group_title.center(width), width, False)]
                for (group_title, _), width in zip(columns, widths)
            ]
        )
    )
    lines.append(
        _row_line(
            [
                [_cell(label, width, right) for label, width, right in cols]
                for _, cols in columns
            ]
        )
    )
    lines.append(_rule(widths, "├", "┼", "┤"))

    # -- Data rows --------------------------------------------------------------------------
    for idx, row in enumerate(report.rows):
        tgt = "  n/a" if not row.is_target else _fmt_pct(row.target_energy_share)
        lines.append(
            _row_line(
                [
                    [
                        _cell(str(idx), id_w[0], True),
                        _cell(row.name, id_w[1], False),
                        _cell(row.block, id_w[2], False),
                        _cell(row.kind, id_w[3], False),
                        _cell(row.strategy, id_w[4], False),
                        _cell(
                            _effective_strategy(
                                row.strategy,
                                row.base_enabled,
                                report.normalizer_type,
                            ),
                            id_w[5],
                            False,
                        ),
                        _cell(
                            "on" if row.base_enabled else "OFF", id_w[6], False
                        ),
                    ],
                    [
                        _fmt(row.raw_mean, 9),
                        _fmt(row.raw_std, 8),
                        _fmt(row.raw_delta_std, 9, 5),
                    ],
                    [
                        _fmt(row.norm_mean, 8, 3),
                        _fmt(row.norm_std, 7, 3),
                        _fmt(row.norm_delta_std, 6, 3),
                    ],
                    [_fmt_pct(row.input_energy_share), tgt],
                    [
                        _fmt(row.loss_weight, 5, 2),
                        _fmt(row.dr_sigma, 8, 4),
                        _fmt(row.dr_over_norm_std, 5, 2),
                        _fmt(row.dr_over_delta_std, 5, 2),
                    ],
                    [_cell(" ".join(row.flags), flag_width, False)],
                ]
            )
        )
    lines.append(_rule(widths, "╰", "┴", "╯"))

    # -- Legend -----------------------------------------------------------------------------
    label_width = max(len(label) for label, _ in _LEGEND)
    lines.append("  Legend")
    for label, definition in _LEGEND:
        wrapped = _wrap_note(definition, total_width - label_width - 8)
        lines.append(f"    {label.ljust(label_width)}  {wrapped[0]}")
        for extra in wrapped[1:]:
            lines.append(f"    {' ' * label_width}  {extra}")

    # -- Flags actually raised ----------------------------------------------------------------
    counts: Dict[str, int] = {}
    for row in report.rows:
        for flag in row.flags:
            counts[flag] = counts.get(flag, 0) + 1
    if counts:
        lines.append(
            "  Flags raised — "
            + ", ".join(f"{flag} x{count}" for flag, count in sorted(counts.items()))
        )
        for flag in sorted(counts):
            definition = _FLAG_DEFINITIONS.get(flag, "(no definition registered)")
            wrapped = _wrap_note(definition, total_width - 24)
            lines.append(f"    {flag.ljust(17)}  {wrapped[0]}")
            for extra in wrapped[1:]:
                lines.append(f"    {' ' * 17}  {extra}")
    else:
        lines.append("  Flags raised — none")

    # -- Actionable banners --------------------------------------------------------------------
    if report.contract_dropped():
        dropped = report.dropped_feature_contract or {}
        lines.extend(
            _wrap_note(
                f"⚠ CONTRACT-DROPPED — normalizer_type='{report.normalizer_type}' "
                f"discards the resolved per-feature contract; the `contract` column "
                f"above has NO effect on {len(dropped)} dim(s): "
                + ", ".join(f"{k}->{v}" for k, v in dropped.items()),
                total_width - 4,
                indent="  ",
            )
        )
    for row in report.rows:
        if "TARGET-DOMINATED" in row.flags:
            lines.extend(
                _wrap_note(
                    f"⚠ TARGET-DOMINATED — '{row.name}' alone carries "
                    f"{100.0 * row.target_energy_share:.1f} % of the composed target "
                    f"loss budget (w={row.loss_weight:.2f}, normalized σ_Δ="
                    f"{row.norm_delta_std:.3f}). The budget share is "
                    f"`w_d · σ_Δ,norm²` renormalized — down-weight this feature, or "
                    f"exclude it from the target (see `dt_target_weight` for an "
                    f"exogenous dim such as `timestamps.delta_stamps`).",
                    total_width - 4,
                    indent="  ",
                )
            )
    return "\n".join(lines)


def _wrap_note(text: str, width: int, indent: str = "") -> List[str]:
    """Word-wrap *text* to *width*, prefixing every line with *indent*."""
    width = max(width, 20)
    lines: List[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(indent + current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(indent + current)
    return lines or [indent]


# =================================================================================================
# Hook
# =================================================================================================


def _read_report_mode(cfg) -> str:
    """Read ``diagnostic.feature_normalization_report`` (default ``true``)."""
    try:
        from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist

        if not is_cfg_key_exist(cfg, "diagnostic.feature_normalization_report"):
            return "on"
        value = cfg.diagnostic.feature_normalization_report
    except Exception:
        return "on"
    if value is False or str(value).lower() in ("false", "off", "none", "null"):
        return "off"
    if str(value).lower() == "verbose":
        return "verbose"
    return "on"


def feature_normalization_report_enabled(cfg) -> bool:
    """Whether the diagnostic is enabled (``diagnostic.feature_normalization_report``).

    Public so every emitter — the training-start hook and the model-free
    calibration of ``deploy_drift_eval`` (RLRP-761 ``S2.6``) — obeys the same
    single switch.
    """
    return _read_report_mode(cfg) != "off"


def log_feature_normalization_report(
    cfg,
    dynamics_model,
    replay_buffer,
    logger=None,
) -> Optional[FeatureNormalizationReport]:
    """Collect, render and log the diagnostic. Never raises (risk ``R-F``).

    Call site: right after ``dynamics_model.update_normalizer(...)`` and before
    the first training epoch. Cost is ``O(rows x in_size)`` **once** per training
    start; it must never be called inside the epoch loop.

    :returns: the collected report, or ``None`` when the diagnostic is disabled
        or the configuration cannot be introspected.
    """
    from tools.console_tools.message import consol_msg_motion_model_learner

    mode = _read_report_mode(cfg)
    if mode == "off":
        # RLRP-777 -- an explicit "disabled by config" line beats silence: a reader
        # of `console.log` can tell "turned off" apart from "crashed / not reached".
        consol_msg_motion_model_learner(
            "[feature-normalization] diagnostic DISABLED by config "
            "(diagnostic.feature_normalization_report=off)"
        )
        return None
    try:
        global _LAST_SKIP_REASON
        _LAST_SKIP_REASON = None
        report = collect_feature_normalization_rows(
            cfg, dynamics_model, replay_buffer
        )
        if report is None:
            # RLRP-777 -- never vanish silently: name the reason on STDOUT so it lands
            # in `console.log` (ConsoleRecorder tees stdout only).
            consol_msg_motion_model_learner(
                "[feature-normalization] diagnostic SKIPPED: "
                f"{_LAST_SKIP_REASON or 'unknown'}"
            )
            return None
        table = render_feature_normalization_table(report)
        _emit(table, report, logger, mode, cfg)
        return report
    except Exception as exc:  # pragma: no cover - defensive by contract
        import warnings

        # Keep the stderr warning for the SLURM stderr stream, AND print an
        # equivalent stdout line so `console.log` always tells the story (RLRP-777).
        warnings.warn(
            f"Feature-normalization diagnostic skipped: {exc!r}", RuntimeWarning
        )
        consol_msg_motion_model_learner(
            f"[feature-normalization] diagnostic FAILED: {exc!r}"
        )
        return None


def _emit(table: str, report: FeatureNormalizationReport, logger, mode: str, cfg):
    """Print the table, escalate the actionable flags, archive the artifact.

    Note ``mode`` no longer gates the archive: since ``S6.4`` the payload is
    written on every enabled run (see
    :func:`archive_feature_normalization_report`).
    """
    from tools.console_tools.message import consol_msg_motion_model_learner

    consol_msg_motion_model_learner("\n" + table)
    # RLRP-777 -- if any target-space column silently degraded (e.g. a CUDA device
    # split in the compute core), the table above looks complete but is not. Surface
    # exactly ONE aggregated line naming the reason(s).
    degrade_reasons = consume_target_space_degrade()
    if degrade_reasons:
        consol_msg_motion_model_learner(
            "[feature-normalization] degraded: target-space column(s) unavailable "
            "-- " + "; ".join(dict.fromkeys(degrade_reasons))
        )
    warning_flags = report.warning_flags()
    if warning_flags:
        message = (
            "Feature-normalization contract raised actionable flag(s): "
            + ", ".join(warning_flags)
            + " — see the table above (RLRP-761)."
        )
        if logger is not None and hasattr(logger, "warning"):
            logger.warning(message)
        else:
            import warnings

            warnings.warn(message, RuntimeWarning)
    archive_feature_normalization_report(report, table)


def archive_feature_normalization_report(
    report: FeatureNormalizationReport,
    table: Optional[str] = None,
    out_dir=None,
) -> Optional[Any]:
    """Persist the S2 table into the run directory (RLRP-761 ``S6.4``).

    Writes BOTH representations next to each other:

    - ``feature_normalization.json`` — the machine-readable payload consumed by
      :mod:`tools.feature_handling_tools.normalization_report_diff` to diff two
      A/B cells (e.g. ``B0`` vs ``B3``);
    - ``feature_normalization.txt`` — the rendered table, so the artifact stays
      readable without a tool.

    Archiving is UNCONDITIONAL whenever the diagnostic is enabled (it used to
    be gated behind ``verbose``): the A/B protocol compares the per-feature
    tables ACROSS cells, and a table that only exists in the console log of a
    multi-hour run is not an artifact. Never raises — a failure to archive must
    not take a training run down.

    :param report: the collected report.
    :param table: the pre-rendered table; re-rendered when omitted.
    :param out_dir: destination directory; defaults to the active Hydra run dir.
    :returns: the directory written to, or ``None`` when unavailable.
    """
    try:
        import json
        import pathlib

        if out_dir is None:
            from hydra.core.hydra_config import HydraConfig

            out_dir = HydraConfig.get().runtime.output_dir
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "feature_normalization.json").write_text(
            json.dumps(report.as_dict(), indent=2)
        )
        (out_dir / "feature_normalization.txt").write_text(
            table if table is not None else render_feature_normalization_table(report)
        )
        # RLRP-777 -- log the destination so a reader knows WHERE the artifact went.
        from tools.console_tools.message import consol_msg_motion_model_learner

        consol_msg_motion_model_learner(
            f"[feature-normalization] archived to {out_dir}"
        )
        return out_dir
    except Exception as exc:
        # Never raise (a failed archive must not take a run down), but do not swallow
        # the reason either -- surface it on stdout so `console.log` records it.
        try:
            from tools.console_tools.message import consol_msg_motion_model_learner

            consol_msg_motion_model_learner(
                f"[feature-normalization] archive FAILED: {exc!r}"
            )
        except Exception:
            pass
        return None
