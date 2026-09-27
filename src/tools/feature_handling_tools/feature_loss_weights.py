# coding=utf-8
"""Statistics-derived per-feature loss weights (RLRP-761 stage S3).

Permanent framework module.

.. note:: **One config node** (RLRP-761 ``S10``).
   The whole per-feature loss-weight policy lives under ``feature_loss_weights``::

       feature_loss_weights:
         obs_feature_weights: predictability   # criterion | float | [per-dim]
         act_feature_weights: null             # unset -> inherits the obs criterion
         feature_weight_mode: multiplicative   # semantic: multiplicative | tempered
         max_weight_ratio: 20.0                # bound on the WEIGHT ratio
         feature_weight_min: 0.15              # bound on the WEIGHT
         max_target_share: 0.15                # bound on the realized BUDGET
         dt_target_weight: 1.0                 # inert opt-out (S12.3); 0.0 = pre-S12

   The three bounds act on three *different* quantities, which is exactly why they
   must be read side by side. Each key is still accepted under ``ms_model`` (its
   historical home) with a deprecation warning -- see :func:`resolve_policy_key`.

``obs_feature_weights`` / ``act_feature_weights`` historically accepted only a
hand-authored ``float | list``. This module resolves, in addition, the **named
criteria** of ``S3.2`` from the dataset statistics measured by
:mod:`tools.feature_handling_tools.feature_statistics`:

- ``'predictability'`` — ``w_d ∝ (sigma_Delta,d / sigma_floor,d)**2``: how much
  of a channel's one-step movement is REDUCIBLE. The recommended criterion for
  robotic-3D: it down-weights *nominal vibration* without down-weighting the
  rare adverse excursions that are the research target (they are large in floor
  units, so the loss still reacts strongly to them).
- ``'raw_equivalent'`` — ``w_d ∝ sigma_Delta,d**2``: weight by ONE-STEP
  INNOVATION ENERGY.

  .. warning::
     RLRP-761 ``S8.4f``. This criterion was documented, and used in the ``S6``
     A/B, as "the raw-space control". **It is not.** Under ``standard`` the
     per-dim penalty is ``Delta_d**2``; under a symmetric facade with weight
     ``w_d`` it is ``w_d * (Delta_d / sigma_d)**2``. Equality for all ``d``
     therefore requires ``w_d ∝ sigma_d**2`` -- the **state** std, not the
     innovation std. The two coincide only if ``sigma_Delta,d / sigma_d`` is
     constant across dims, and the measured spread on the reference UGV data is
     **12x**. Use ``'state_equivalent'`` for the control; ``'raw_equivalent'`` is
     kept unchanged so an already-scored cell does not change meaning
     retroactively.

- ``'state_equivalent'`` — ``w_d ∝ sigma_d**2``: the ACTUAL raw-space
  equivalence, i.e. the scientific CONTROL of the ``S6`` A/B (RLRP-761 ``S8.4e``).
- ``'uniform'`` — the **scalar** ``1.0``.

Two contracts that are easy to get wrong and are therefore enforced here:

1. ``'uniform'`` must resolve to the scalar float ``1.0``, never to ``[1.0]*D``.
   ``AbstractFeatureWeightedMultiStepMLP`` decides whether to activate the whole
   weighting path from the resolved value, and a *list* of ones is not the same
   object as the scalar ``1.0`` (RLRP-761 blocker ``B2``).
2. The model normalizes the composed weight vector by its own sum, so only the
   **ratios** survive; an absolute clip is meaningless. The dynamic range is
   therefore bounded as a ratio, ``max_d w_d / min_d w_d <= max_ratio``, by a
   symmetric log-space squash applied BEFORE the model's normalization
   (blocker ``B3``).

Stage ``S7`` (the ``TARGET-HOG`` regression) adds three corrections, applied in
this order after the criterion is evaluated:

``S7.1`` :func:`apply_weight_floor`
    The criteria are *quadratic* in a measured scale, so the spread they produce
    is far wider than intended: on the reference UGV run ``angular_vels.x``
    resolved to ``0.04`` against ``linear_vels.y``'s ``3.76`` (x94), leaving the
    adverse-event channels effectively unsupervised. A floor keeps a
    de-emphasised channel supervised.

``S7.3`` act-block inheritance
    The obs and act weights are CONCATENATED and normalized jointly by the model,
    so an act block left at the scalar ``1.0`` against a mean-normalized obs block
    is not neutral -- it is an above-average weight on every action dim.

``S7.2`` :func:`apply_exogenous_target_policy`
    ``dt`` is read from the clock and the deploy path emits observations only, so
    it is an INPUT, never a prediction target. ``S1.5`` standardized it -- which
    correctly fixed the model's input blindness -- and thereby promoted pure
    clock jitter into the composed multi-step target, where it took **52.4 %** of
    the loss budget. This policy removes it from the target budget without
    touching its (now correct) input scaling.
"""
from __future__ import annotations

import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from tools.feature_handling_tools.feature_statistics import FeatureScales
from tools.feature_handling_tools.normalization_diagnostic import (
    TARGET_SPACE_RAW,
)

#: The named criteria accepted by ``obs_feature_weights`` / ``act_feature_weights``.
CRITERIA = (
    "predictability",
    "raw_equivalent",
    "state_equivalent",
    "uniform",
)

#: Default dynamic-range bound (``feature_loss_weights.max_weight_ratio``; the
#: legacy ``ms_model.feature_weight_max_ratio`` is a deprecated alias).
DEFAULT_MAX_RATIO = 50.0

#: RLRP-761 S7.1 — default floor on a resolved weight, expressed in units of the
#: mean weight. The criteria are *quadratic* in a measured scale, so a channel
#: whose innovation is small in floor units collapses by two orders of magnitude
#: (measured on the reference UGV run: ``angular_vels.x -> 0.04``, i.e. x94 below
#: ``linear_vels.y``). Those are exactly the adverse-event channels the research
#: targets, so a criterion is allowed to DE-EMPHASISE them, never to ANNIHILATE
#: them. ``0.0`` disables the floor and restores the raw criterion.
DEFAULT_WEIGHT_FLOOR = 0.25

#: RLRP-761 S7.2 / S12.3 — default target weight of the ``dt`` (delta-timestamp)
#: feature.
#:
#: **Demoted to an inert default of ``1.0`` by S12.3 (2026-08-03).** The original
#: S7.2 rationale ("``dt`` is exogenous, forecasting it buys nothing") is WRONG for
#: the FORECAST head: the MS forecast/mixture self-feed (``state_history_update``,
#: ``obs_space_only=False``) predicts the commands AND ``dt`` and re-injects them
#: to advance the clock / replay the control over the horizon, so ``dt`` is a
#: genuine forecast target at TRAINING time (even though the deploy head emits obs
#: only). Under ``S12`` the act block is innovation-scaled like the obs block, so
#: ``dt`` is an ordinary forecast target governed by the inherited ``predictability``
#: criterion and bounded by the joint budget cap — it needs no dedicated override.
#:
#: The knob is therefore RETAINED only as a DEPRECATED, optional, by-kind opt-out
#: (resolve every ``FeatureKind.DT`` dim to a low/zero target weight without
#: hand-authoring a positional act vector), kept for the deploy-only-obs baseline;
#: the default ``1.0`` makes ``apply_exogenous_target_policy`` a strict no-op. Set
#: to ``0.0`` to reproduce the pre-``S12`` (S7.2) behaviour for a controlled A/B.
DEFAULT_DT_TARGET_WEIGHT = 1.0

#: Feature kinds whose target contribution is governed by
#: ``feature_loss_weights.dt_target_weight``.
_EXOGENOUS_TARGET_KINDS = ("dt",)

#: RLRP-761 S7.5 -- maximum share of the composed TARGET loss budget a single
#: feature may hold. This is the bound that actually MATTERS, and the reason a
#: weight floor alone is not enough: the share is ``w_d * sigma_Delta_norm_d**2``,
#: so it is quadratic in a quantity the weight bound never sees. On the reference
#: run, flooring the weights still left ``linear_vels.y`` at 59 % of the budget.
#: ``1.0`` (or a non-finite value) disables the cap.
DEFAULT_MAX_TARGET_SHARE = 0.40

_POSITIVE_EPS = 1e-30


def is_criterion(value) -> bool:
    """Whether *value* is one of the named criteria of :data:`CRITERIA`."""
    return isinstance(value, str) and value in CRITERIA


def needs_dataset_statistics(value) -> bool:
    """Whether *value* is a DATA-DRIVEN criterion (needs ``estimate_feature_scales``).

    ``uniform`` is the explicit opt-out ("weighting path OFF"): it resolves to the scalar
    ``1.0`` without looking at the dataset, so it must not trigger the statistics
    estimation -- which needs a history window of length ``>= 3`` and therefore cannot run
    on the ``H = 1`` MS->MS shapes (M3 ULH, RLRP-824).
    """
    return is_criterion(value) and value != "uniform"


def squash_dynamic_range(
    weights: Sequence[float], max_ratio: float = DEFAULT_MAX_RATIO
) -> Tuple[List[float], bool]:
    """Bound ``max/min`` of *weights* by a symmetric log-space squash.

    Applied in log space so the ORDERING and the relative spacing of the weights
    are preserved; only the spread is compressed. Returns the mean-normalized
    vector (mean ``1.0``), which makes the printed weights directly readable as
    "x times the average feature".

    :param weights: The strictly positive raw criterion values.
    :param max_ratio: The maximum tolerated ``max/min`` ratio (``<= 1`` or a
        non-finite value disables the squash).
    :returns: ``(weights, squash_did_bind)``.
    """
    values = np.asarray(list(weights), dtype=np.float64)
    values = np.maximum(values, _POSITIVE_EPS)
    log_values = np.log(values)
    log_values = log_values - log_values.mean()
    bound = False
    spread = float(log_values.max() - log_values.min())
    if (
        np.isfinite(max_ratio)
        and max_ratio > 1.0
        and spread > np.log(max_ratio)
    ):
        log_values = log_values * (float(np.log(max_ratio)) / spread)
        bound = True
    values = np.exp(log_values)
    values = values / values.mean()
    return [float(v) for v in values], bound


def apply_weight_floor(
    weights: Sequence[float], floor: float = DEFAULT_WEIGHT_FLOOR
) -> Tuple[List[float], bool]:
    """Clamp *weights* from below, then re-normalize the mean back to ``1.0``.

    RLRP-761 ``S7.1``. Applied AFTER :func:`squash_dynamic_range` (whose output
    already has mean ``1.0``), so *floor* reads directly as "a feature may never
    weigh less than ``floor`` times the average feature".

    :param weights: the mean-normalized criterion weights.
    :param floor: the lower bound, in units of the mean weight. ``<= 0``
        disables the floor and returns *weights* unchanged.
    :returns: ``(weights, floor_did_bind)``.
    """
    values = np.asarray(list(weights), dtype=np.float64)
    if not np.isfinite(floor) or floor <= 0.0 or values.size == 0:
        return [float(v) for v in values], False
    bound = bool(np.any(values < floor))
    if not bound:
        return [float(v) for v in values], False
    values = np.maximum(values, float(floor))
    values = values / values.mean()
    return [float(v) for v in values], True


def apply_target_budget_cap(
    weights: Sequence[float],
    delta_std_norm: Sequence[float],
    max_share: float = DEFAULT_MAX_TARGET_SHARE,
    max_iterations: int = 100,
) -> Tuple[List[float], bool]:
    """Bound each feature's share of the composed target loss budget.

    RLRP-761 ``S7.5``. The model allocates its objective as

    .. math:: \\text{share}_d \\propto w_d \\, \\sigma_{\\Delta,d}^2

    in NORMALIZED units, so bounding the *weights* (``feature_weight_max_ratio``)
    bounds the wrong quantity: the realized spread is quadratic in a scale the
    weight bound never observes. Measured on the reference run, a weight ratio of
    94 produced a budget ratio of ~200, and even after the ``S7.1`` floor a
    single feature still held 59 % of the objective.

    Implemented as a **fixed-point** water-filling (RLRP-761 ``S9.2``): the worst
    dim is pinned to EXACTLY the cap by giving every capped dim the closed-form
    energy

    .. math:: e_d = \\frac{s}{1 - k\\,s}\\,\\sum_{j \\notin K} e_j,

    where ``K`` is the set of capped dims (``k = |K|``). This makes each capped
    dim's share exactly ``s`` *accounting for the shrinking total*, so — unlike
    the previous ``factor = max_share / share`` rescale, which assumed a fixed
    total and oscillated when a single dim dominated (e.g. ``linear_vels.y`` at
    70 %) — it converges in at most one pass per newly-saturated dim. The set
    ``K`` only grows, so the process is monotone and terminates in ``<= D``
    steps. Uncapped and zero-energy dims keep their original weight; the returned
    vector is mean-normalized (mean weight ``1``), matching the criterion output.

    Capping ``D`` dims each at ``max_share`` is only feasible when the caps can
    jointly cover the whole budget (roughly ``n_eff * max_share >= 1``, with
    ``n_eff`` the number of innovation-carrying dims). When it is not, no
    allocation can hold every dim at or below the cap; the cap is then reported
    as **not enforced** (``S9.2`` warning) and the weights are returned as-is.

    :param weights: the per-dim weights, covering the whole composed target.
    :param delta_std_norm: the per-dim one-step innovation std, in the model's
        normalized units (``sigma_Delta / sigma_state``).
    :param max_share: the cap. ``>= 1`` or non-finite disables it.
    :param max_iterations: retained for back-compatibility; the fixed point needs
        no iteration budget (it converges in ``<= D`` deterministic steps).
    :returns: ``(weights, cap_did_bind)``.
    """
    values = np.asarray(list(weights), dtype=np.float64)
    energy_scale = np.asarray(list(delta_std_norm), dtype=np.float64) ** 2
    if values.size == 0 or values.size != energy_scale.size:
        return [float(v) for v in values], False
    if not np.isfinite(max_share) or max_share >= 1.0 or max_share <= 0.0:
        return [float(v) for v in values], False
    non_finite_idx = [int(i) for i in np.flatnonzero(~np.isfinite(energy_scale))]
    if non_finite_idx:
        # RLRP-761 S9.7 -- assigning zero energy makes these dims report a 0 %
        # budget share and renders them PERMANENTLY un-cappable. Legitimate
        # (a constant channel has no innovation to allocate budget to) but never
        # silent: the accounting the user reads no longer covers the full vector.
        warnings.warn(
            "RLRP-761 S9.7: non-finite normalized innovation std at composed "
            f"indices {non_finite_idx}; these dims are assigned ZERO target-budget "
            "energy, so they report a 0 % share and can never trip `max_share`.",
            RuntimeWarning,
            stacklevel=2,
        )
    energy_scale = np.where(np.isfinite(energy_scale), energy_scale, 0.0)

    base_energy = values * energy_scale
    if float(base_energy.sum()) <= 0.0:
        # No dim carries any target energy -> nothing to cap.
        return [float(v) for v in values], False

    # Greedy fixed point: repeatedly pin the current worst dim to EXACTLY the cap.
    # Each capped dim gets the closed-form energy `s/(1-k*s) * sum(uncapped e)`,
    # so its share is exactly `s` even as the total shrinks; the capped set only
    # grows, hence the loop is monotone and cannot oscillate (max `D` steps).
    n = values.size
    capped = np.zeros(n, dtype=bool)
    for _ in range(n):
        uncapped = ~capped
        uncapped_sum = float(base_energy[uncapped].sum())
        k = int(capped.sum())
        denom = 1.0 - k * max_share
        if uncapped_sum <= 0.0 or denom <= 0.0:
            break
        cap_energy = max_share / denom * uncapped_sum
        current_total = k * cap_energy + uncapped_sum
        unc_idx = np.flatnonzero(uncapped)
        worst = int(unc_idx[int(np.argmax(base_energy[unc_idx]))])
        if base_energy[worst] / current_total > max_share + 1e-12:
            capped[worst] = True
            continue
        break

    out = values.astype(np.float64).copy()
    if capped.any():
        uncapped_sum = float(base_energy[~capped].sum())
        denom = 1.0 - int(capped.sum()) * max_share
        if denom > 0.0 and uncapped_sum > 0.0:
            cap_energy = max_share / denom * uncapped_sum
            for i in np.flatnonzero(capped):
                scale_i = float(energy_scale[i])
                # A zero-energy-scale dim keeps its original weight (it carries no
                # share anyway); only innovation-carrying dims are re-scaled.
                if scale_i > 0.0:
                    out[i] = cap_energy / scale_i
        out = np.maximum(out, 0.0)

    # Verify the cap actually holds. It will NOT when the request is infeasible
    # (too few innovation-carrying dims to cover the budget, `n_eff*max_share<1`):
    # no allocation can hold every dim at/below the cap. That is the one
    # informative failure -- warn and return the weights unchanged.
    final_energy = out * energy_scale
    final_total = float(final_energy.sum())
    residual = float((final_energy / final_total).max()) if final_total > 0.0 else 0.0
    if residual > max_share + 1e-9:
        warnings.warn(
            "RLRP-761 S9.2: the target-budget cap is INFEASIBLE -- the active "
            f"target dims cannot all be held at or below max_share={max_share} "
            "(too few dims carry innovation energy); the cap is NOT enforced, "
            f"residual max share is {residual:.4f}. The resolved loss weights are "
            "returned as-is.",
            RuntimeWarning,
            stacklevel=2,
        )
        return [float(v) for v in values], False

    if not capped.any():
        return [float(v) for v in values], False

    mean = float(out.mean())
    if mean > 0.0:
        out = out / mean
    return [float(v) for v in out], True


def resolve_feature_weights(
    spec: Union[None, float, int, str, Sequence[float]],
    length: int,
    scales: Optional[FeatureScales] = None,
    dim_names: Optional[Sequence[str]] = None,
    max_ratio: float = DEFAULT_MAX_RATIO,
    weight_floor: float = DEFAULT_WEIGHT_FLOOR,
) -> Union[float, List[float]]:
    """Resolve a feature-weight *spec* to the value the model consumes.

    :param spec: ``None`` / a scalar / a per-dim sequence / one of
        :data:`CRITERIA`.
    :param length: The expected per-single-step block length.
    :param scales: The measured dataset scales, required by the named criteria.
    :param dim_names: The block's dim names, used to select the block's columns
        out of *scales* (which covers ``obs + act``). Defaults to the leading
        ``length`` entries.
    :returns: the scalar ``1.0`` (neutral) or a list of ``length`` weights whose
        mean is ``1.0``.
    :raise ValueError: on an unknown string, a length mismatch, or a named
        criterion requested without the statistics it needs.
    """
    if spec is None:
        return 1.0
    if isinstance(spec, str):
        if spec not in CRITERIA:
            raise ValueError(
                f"feature weights {spec!r} is not a known criterion; expected one "
                f"of {CRITERIA}, a scalar, or a per-dim sequence."
            )
        if spec == "uniform":
            # Scalar, NEVER [1.0] * length -- see the module docstring (B2).
            return 1.0
        if scales is None:
            raise ValueError(
                f"feature weights {spec!r} requires the dataset statistics "
                f"(estimate_feature_scales) to be resolved; none were provided."
            )
        # `predictability` is stored PRE-squared; the two energy criteria square
        # the scale they are named after.
        key = {
            "predictability": "predictability",
            "raw_equivalent": "delta_std",
            "state_equivalent": "state_std",
        }[spec]
        values = _select_block(scales, key, length, dim_names)
        if spec in ("raw_equivalent", "state_equivalent"):
            values = np.asarray(values, dtype=np.float64) ** 2
        weights, _ = squash_dynamic_range(values, max_ratio=max_ratio)
        weights, _ = apply_weight_floor(weights, floor=weight_floor)
        return weights
    if isinstance(spec, (int, float)):
        return float(spec)
    values = [float(v) for v in spec]
    if len(values) != length:
        raise ValueError(
            f"feature weights has length {len(values)} != expected {length}."
        )
    return values


def resolve_and_apply_feature_loss_weights(
    cfg, dynamics_model, replay_buffer, logger=None
) -> Optional[dict]:
    """Resolve the named ``S3.2`` criteria on the dataset and apply them.

    RLRP-761 ``S3.3``. Called ONCE at training start, right after the normalizer
    statistics are fitted: that is the earliest point where the dataset is in
    scope (the model is constructed before any data is loaded, so the criterion
    cannot be resolved by the Hydra instantiation itself).

    A strict no-op — and a silent one — unless ``ms_model.obs_feature_weights``
    or ``ms_model.act_feature_weights`` is one of :data:`CRITERIA`. The resolved
    numeric vectors are written back into *cfg* and logged, so a data-dependent
    weight is recoverable from the run record.

    :param cfg: the resolved Hydra config (mutated in place on resolution).
    :param dynamics_model: the transition-model wrapper.
    :param replay_buffer: the mbrl-lib replay buffer (source of the statistics).
    :param logger: optional logger for the resolved vectors.
    :returns: ``{'obs': [...], 'act': [...], 'scales': FeatureScales}`` when a
        criterion was resolved, else ``None``.
    """
    import omegaconf

    from tools.feature_handling_tools.feature_statistics import (
        estimate_feature_scales,
    )

    # RLRP-761 S10 -- the whole policy is read from ONE node, with the historical
    # `ms_model` keys honoured as a deprecated fallback (`resolve_policy_key`).
    try:
        ms_cfg = cfg.ms_model
    except Exception:
        return None
    policy = _target_budget_policy(cfg)
    obs_spec = resolve_policy_key(cfg, policy, ms_cfg, "obs_feature_weights", 1.0)
    act_spec = resolve_policy_key(cfg, policy, ms_cfg, "act_feature_weights", 1.0)
    if not (is_criterion(obs_spec) or is_criterion(act_spec)):
        return None

    model = getattr(dynamics_model, "model", None)
    obs_len = int(getattr(model, "singlestep_obs_len", 0) or 0)
    act_len = int(getattr(model, "singlestep_act_len", 0) or 0)
    history_len = int(getattr(model, "history_len", 0) or 0)
    if not obs_len or not history_len:
        raise ValueError(
            "A named feature-weight criterion requires a multistep model exposing "
            "singlestep_obs_len / history_len; none was found."
        )

    obs_names, act_names = _resolve_block_names(cfg, obs_len, act_len)
    if needs_dataset_statistics(obs_spec) or needs_dataset_statistics(act_spec):
        scales = estimate_feature_scales(
            _slot_sequences(replay_buffer, obs_len, act_len, history_len),
            dim_names=list(obs_names) + list(act_names),
        )
        if scales is None:
            raise ValueError(
                "A named feature-weight criterion was requested but the dataset "
                "statistics could not be estimated (no history window of length >= 3)."
            )
    else:
        # `uniform` opt-out (RLRP-824 fix): no data-driven criterion => no statistics, no
        # budget cap; the weights below resolve to the neutral scalar 1.0 (or a literal).
        scales = None

    max_ratio = _resolve_max_weight_ratio(cfg, policy, ms_cfg)
    weight_floor = policy.get("feature_weight_min", None)
    weight_floor = (
        DEFAULT_WEIGHT_FLOOR if weight_floor is None else float(weight_floor)
    )
    # The semantic is resolved from the SAME node as the criterion, then pushed onto
    # the model: it is read at loss time, so a mode declared only in the canonical
    # node must reach the model object or it would silently stay `tempered` --
    # i.e. gradient-inert (blocker `B1`).
    weight_mode = _resolve_and_apply_weight_mode(cfg, policy, ms_cfg, model)
    _guard_weight_mode_coherence(weight_mode, obs_spec, act_spec)

    # RLRP-761 S7.3 -- the two blocks are CONCATENATED and normalized jointly by
    # the model, so leaving the act block at the scalar 1.0 while the obs block
    # is mean-normalized silently grants every action dim an above-average share.
    # An unspecified act block therefore inherits the obs criterion; set
    # `act_feature_weights: uniform` to opt out explicitly.
    if act_len and is_criterion(obs_spec) and not is_criterion(act_spec):
        if act_spec is None or (
            isinstance(act_spec, (int, float)) and float(act_spec) == 1.0
        ):
            act_spec = obs_spec

    obs_weights = resolve_feature_weights(
        obs_spec,
        obs_len,
        scales=scales,
        dim_names=obs_names,
        max_ratio=max_ratio,
        weight_floor=weight_floor,
    )
    act_weights = resolve_feature_weights(
        act_spec,
        act_len,
        scales=scales,
        dim_names=act_names,
        max_ratio=max_ratio,
        weight_floor=weight_floor,
    )
    act_weights, exogenous = apply_exogenous_target_policy(
        cfg, act_names, act_weights, policy.get("dt_target_weight", None)
    )

    # RLRP-761 S7.5 -- the cap operates on the WHOLE composed target at once,
    # because the budget is shared between the two blocks.
    max_share = policy.get("max_target_share", None)
    max_share = (
        DEFAULT_MAX_TARGET_SHARE if max_share is None else float(max_share)
    )
    if scales is not None:
        obs_weights, act_weights, capped, target_space = _apply_composed_budget_cap(
            obs_weights,
            act_weights,
            obs_names,
            act_names,
            scales,
            max_share,
            dynamics_model,
        )
    else:
        capped, target_space = False, "n/a (uniform opt-out, no dataset statistics)"

    # RLRP-761 S10 -- materialize the resolved vectors back into the node that
    # DECLARED the criterion, so the run record is self-describing and a second
    # resolution pass does not see the same key set in both nodes.
    with omegaconf.open_dict(cfg):
        if "obs_feature_weights" in policy or "act_feature_weights" in policy:
            cfg.feature_loss_weights.obs_feature_weights = obs_weights
            cfg.feature_loss_weights.act_feature_weights = act_weights
        else:
            cfg.ms_model.obs_feature_weights = obs_weights
            cfg.ms_model.act_feature_weights = act_weights

    n_rows = scales.n_samples if scales is not None else 0
    message = (
        f"RLRP-761 S3: feature loss weights resolved from {n_rows} rows "
        f"(obs_spec={obs_spec!r}, act_spec={act_spec!r}, max_ratio={max_ratio}): "
        f"obs={_pretty(obs_names, obs_weights)}, act={_pretty(act_names, act_weights)}"
    )
    if exogenous:
        message += (
            "; EXOGENOUS target dims down-weighted "
            + "{"
            + ", ".join(
                f"{str(k)!r}: {_fmt_weight(v)}" for k, v in exogenous.items()
            )
            + "}"
        )
    # RLRP-761 S8.4d -- the share is only interpretable together with the space it
    # was computed in; stamp it whether or not the cap bound.
    message += f"; target_space={target_space}"
    if capped:
        message += (
            f"; TARGET-BUDGET CAP bound at max_share={max_share} -> "
            f"obs={_pretty(obs_names, obs_weights)}, "
            f"act={_pretty(act_names, act_weights)}"
        )
    if logger is not None:
        logger.info(message)
    else:
        from tools.console_tools.message import consol_msg_universal_one_liner

        consol_msg_universal_one_liner(message)

    _apply_to_model(model, obs_weights, act_weights)
    return {
        "obs": obs_weights,
        "act": act_weights,
        "scales": scales,
        "target_space": target_space,
    }


def _apply_composed_budget_cap(
    obs_weights,
    act_weights,
    obs_names,
    act_names,
    scales,
    max_share,
    dynamics_model=None,
):
    """Apply :func:`apply_target_budget_cap` across both blocks jointly.

    The obs and act weight vectors are concatenated by the model into a SINGLE
    composed target, so the cap is meaningless if applied per block. Scalar
    (neutral) blocks are expanded only when the other block is already a vector;
    when BOTH are scalars the weighting path is genuinely neutral (blocker
    ``B2``) and the cap is skipped so the run stays bit-exact.

    RLRP-761 ``S8.4b``: the normalized ``sigma_Delta`` is now resolved from the
    **output** facade (:func:`resolve_target_space_delta_std`) instead of being
    assumed to be ``sigma_Delta / sigma_state``. That assumption was correct for
    ``standard_symmetric`` ONLY -- under ``standard`` the target is raw (there is
    no output transform to divide by) and under
    ``standard_symmetric_innovation`` the target is already innovation-scaled, so
    the cap was bounding a quantity the objective does not use. The analytic ratio
    is kept as an explicit, LOGGED fallback.

    :returns: ``(obs_weights, act_weights, cap_did_bind, target_space_label)``.
    """
    obs_len, act_len = len(obs_names), len(act_names)
    if isinstance(obs_weights, (int, float)) and isinstance(
        act_weights, (int, float)
    ):
        return obs_weights, act_weights, False, "neutral (cap skipped)"

    obs_values = (
        [float(obs_weights)] * obs_len
        if isinstance(obs_weights, (int, float))
        else [float(v) for v in obs_weights]
    )
    act_values = (
        [float(act_weights)] * act_len
        if isinstance(act_weights, (int, float))
        else [float(v) for v in act_weights]
    )

    names = list(obs_names) + list(act_names)
    delta = scales.as_mapping("delta_std")
    raw_delta = [float(delta.get(name, 0.0) or 0.0) for name in names]

    delta_std_norm, target_space = _resolve_target_space_delta(
        dynamics_model, raw_delta, obs_len, act_len, scales, names
    )

    # RLRP-761 S8.4h -- `standard` uses the single concatenated facade and DROPS
    # the resolved per-feature contract, so a per-feature budget cap is bounding a
    # budget that path does not honour dimension-wise. The cap still acts on a
    # well-defined quantity (the RAW innovation energy), so this is a warning and
    # not a refusal -- but silently accepting it would let an operator believe the
    # feature contract is live when it is not.
    if target_space == TARGET_SPACE_RAW and float(max_share) < 1.0:
        warnings.warn(
            f"max_target_share={max_share} is armed while the resolved target "
            "space is RAW (`normalizer_type: standard` has no output "
            "normalizer, and its single flat facade has already dropped the "
            "per-feature contract). The cap is applied on the PHYSICAL "
            "innovation energy; use `standard_symmetric` (or set "
            "`max_target_share: 1.0`) if that is not what you intend.",
            RuntimeWarning,
            stacklevel=3,
        )

    capped, did_bind = apply_target_budget_cap(
        obs_values + act_values, delta_std_norm, max_share=max_share
    )
    if not did_bind:
        return obs_weights, act_weights, False, target_space
    return capped[:obs_len], capped[obs_len:], True, target_space


def _resolve_target_space_delta(
    dynamics_model, raw_delta, obs_len, act_len, scales, names
):
    """``sigma_Delta`` in the space the composed TARGET actually lives in.

    RLRP-761 ``S8.4b``. Preference order:

    1. the **measured** output facade (:func:`resolve_target_space_delta_std`) --
       correct for all five normalizer types;
    2. the historical analytic ratio ``sigma_Delta / sigma_state`` -- correct for
       ``standard_symmetric`` only, kept so a config the facade cannot be
       introspected from still behaves as before, and LABELLED so an archived
       share is never ambiguous.

    :returns: ``(delta_std_in_target_space, target_space_label)``.
    """
    route = "no dynamics_model in scope"
    if dynamics_model is not None:
        try:
            # RLRP-761 S10.2: sourced from the dedicated lower-layer module, not
            # the diagnostic -- a resolver has no business importing a rendering
            # module (the function-local import was the very smell S10.2 removes).
            from tools.feature_handling_tools.feature_target_space import (
                resolve_target_space_delta_std,
            )

            values, label = resolve_target_space_delta_std(
                dynamics_model, raw_delta, obs_len, act_len
            )
            if label != "unresolved":
                return [float(v) for v in values], label
            # RLRP-761 S9.1: this is a NORMAL RETURN (a `None` output-facade scale
            # or a width mismatch), and it is the DOMINANT route to the analytic
            # fallback -- narrowing the `except` below would leave it silent.
            route = "output facade unresolved (None scale or width mismatch)"
        except (AttributeError, TypeError) as error:
            # Deliberately narrow: a broad `except` around what is really a
            # CLASSIFICATION already cost this task a debugging cycle (`is_affine`
            # is a property; `is_affine()` raised and every affine type was
            # mislabelled `local_linear`).
            route = f"output-facade introspection raised {type(error).__name__}: {error}"

    # RLRP-761 S9.1 -- the analytic ratio is the TARGET transform of
    # `standard_symmetric` ONLY, i.e. the wrong quantity for the other two live
    # target spaces (`standard` keeps a raw target, `standard_symmetric_innovation`
    # scales it by sigma_Delta). The returned label records it, but a label is not a
    # warning: without this nothing in a run makes the substitution visible.
    warnings.warn(
        "RLRP-761 S9.1: falling back to the ANALYTIC target-space scale "
        f"(sigma_Delta / sigma_state) because {route}. This ratio is the correct "
        "target transform for `standard_symmetric` only -- under `standard` or "
        "`standard_symmetric_innovation` the resulting loss-budget shares (and any "
        "`max_target_share` cap computed from them) are measured in the wrong space.",
        RuntimeWarning,
        stacklevel=2,
    )
    state = scales.as_mapping("state_std")
    fallback = []
    non_finite = []
    for name, numerator in zip(names, raw_delta):
        denominator = float(state.get(name, 0.0) or 0.0)
        if denominator > 0.0:
            fallback.append(numerator / denominator)
        else:
            # RLRP-761 S9.7: a constant / heavily quantized channel. `NaN` is
            # propagated (the caller maps it to zero energy) but MUST NOT be
            # silent: such a dim reports a 0 % budget share, can never trip the
            # cap, and disappears from the accounting.
            fallback.append(float("nan"))
            non_finite.append(str(name))
    if non_finite:
        warnings.warn(
            "RLRP-761 S9.7: target-space sigma_Delta is NOT FINITE for "
            f"{non_finite} (state sigma is zero -- a constant or heavily quantized "
            "channel). These dims are DROPPED from the loss-budget accounting: "
            "they report a 0 % share and can never trip `max_target_share`.",
            RuntimeWarning,
            stacklevel=2,
        )
    return fallback, "state_sigma (analytic fallback)"


def _target_budget_policy(cfg) -> dict:
    """Read the ``S7`` target-budget policy node.

    RLRP-761 ``S10``. This node is the **single canonical home** of the whole
    per-feature loss-weighting policy: the criteria (``obs_feature_weights`` /
    ``act_feature_weights``), the semantic (``feature_weight_mode``) and the three
    bounds (``feature_weight_min`` on the weight, ``max_weight_ratio`` on the
    weight RATIO, ``max_target_share`` on the realized BUDGET).

    The design reason is **separation of concerns**: this is a *training-objective
    policy*, resolved from the dataset at the training-start seam, long after the
    model object exists. The model never reads it — it is handed the
    already-resolved numeric vectors through :func:`_apply_to_model` — so it does
    not belong on the model's construction surface. Keeping one node also means
    the three bounds, routinely confused because they bound three different
    quantities, are read together.

    Every key is also accepted under ``ms_model`` (its historical home) with a
    deprecation warning, so no existing config breaks.

    :param cfg: the resolved Hydra config.
    :returns: a plain mapping, empty when the node is absent.
    """
    try:
        node = cfg.get("feature_loss_weights", None)
    except Exception:
        return {}
    if node is None:
        return {}
    try:
        import omegaconf

        plain = omegaconf.OmegaConf.to_container(node, resolve=True)
    except Exception:
        plain = node
    return dict(plain) if isinstance(plain, dict) else {}


#: ``feature_loss_weights`` key -> its DEPRECATED ``ms_model`` counterpart.
#:
#: RLRP-761 ``S10``. The historical layout split one policy across two nodes: the
#: criteria and the semantic on ``ms_model``, the bounds on
#: ``feature_loss_weights``. Both halves are read by the SAME function, so the
#: split bought nothing and cost readability -- ``feature_weight_min`` and
#: ``max_weight_ratio`` read as a matched pair and were not even in the same node.
_LEGACY_MS_MODEL_KEYS = {
    "obs_feature_weights": "obs_feature_weights",
    "act_feature_weights": "act_feature_weights",
    "feature_weight_mode": "feature_weight_mode",
    "max_weight_ratio": "feature_weight_max_ratio",
}


def resolve_policy_key(cfg, policy: dict, ms_cfg, key: str, default=None):
    """Read one policy key, canonical node first, ``ms_model`` as a legacy fallback.

    RLRP-761 ``S10``. :data:`_LEGACY_MS_MODEL_KEYS` gives the historical
    ``ms_model`` name of *key*; when only that one is set the value is honoured and
    a :class:`DeprecationWarning` names the replacement, so no existing config
    breaks and none stays silently on the old surface.

    :param cfg: the resolved Hydra config (unused, kept for call-site symmetry).
    :param policy: the ``feature_loss_weights`` node as a plain mapping.
    :param ms_cfg: the ``ms_model`` node.
    :param key: the canonical key name.
    :param default: returned when neither node carries the key.
    :returns: the effective value.
    """
    canonical = policy.get(key, None)
    legacy_name = _LEGACY_MS_MODEL_KEYS.get(key, key)
    try:
        legacy = ms_cfg.get(legacy_name, None)
    except (AttributeError, TypeError):
        legacy = None

    if canonical is not None and legacy is not None:
        warnings.warn(
            f"Both `feature_loss_weights.{key}` and the DEPRECATED "
            f"`ms_model.{legacy_name}` are set; the former wins. Remove the "
            f"`ms_model` key.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif canonical is None and legacy is not None:
        warnings.warn(
            f"`ms_model.{legacy_name}` is DEPRECATED; the per-feature loss-weight "
            f"policy now lives in ONE place. Move it to "
            f"`feature_loss_weights.{key}`.",
            DeprecationWarning,
            stacklevel=3,
        )

    value = canonical if canonical is not None else legacy
    return default if value is None else value


def _resolve_max_weight_ratio(cfg, policy: dict, ms_cfg) -> float:
    """Resolve the bound on the resolved-criterion weight RATIO.

    Note the model keeps its own ``feature_weight_max_ratio`` attribute: its clamp
    helper is a defensive fallback for a HAND-AUTHORED weight vector, which never
    reaches this resolver.

    :returns: the effective bound, :data:`DEFAULT_MAX_RATIO` when unset.
    """
    value = resolve_policy_key(cfg, policy, ms_cfg, "max_weight_ratio")
    if value is None:
        return DEFAULT_MAX_RATIO
    value = float(value)
    # A 0.0 / non-finite bound is meaningless and would collapse every weight.
    return value if value and math.isfinite(value) else DEFAULT_MAX_RATIO


def _resolve_and_apply_weight_mode(cfg, policy: dict, ms_cfg, model) -> str:
    """Resolve ``feature_weight_mode`` and make sure the MODEL uses it.

    RLRP-761 ``S10``. The semantic belongs with the criterion it applies -- both
    are training-objective policy -- so its canonical home is
    ``feature_loss_weights.feature_weight_mode``. The model reads its own
    ``feature_weight_mode`` attribute at loss time, so the resolved value is
    assigned onto the model object; without that step a mode set ONLY in the
    canonical node would leave the model on its ``tempered`` default, which is
    gradient-inert (blocker ``B1``) -- a null result that looks like a
    measurement.

    :returns: the effective mode.
    """
    # RLRP-761 S10.1g: a FRAMEWORK DEFAULT must never be pushed onto the model.
    # Resolving with a "tempered" default and assigning it would overwrite a model
    # constructed with `multiplicative` whenever NEITHER node declares the key,
    # moving it onto the gradient-inert semantic (blocker `B1`) -- i.e. causing the
    # exact failure this push exists to prevent. Only a value DECLARED in a config
    # node is authoritative over the model's own construction-time value.
    declared = resolve_policy_key(cfg, policy, ms_cfg, "feature_weight_mode", None)
    current = getattr(model, "feature_weight_mode", None)
    if declared is None:
        return str(current) if current is not None else "tempered"
    mode = str(declared)
    if current is not None and current != mode:
        # Only assign when the model exposes the attribute: a family that ignores
        # the knob must not silently acquire one.
        setattr(model, "feature_weight_mode", mode)
    return mode


def _guard_weight_mode_coherence(mode, obs_spec, act_spec) -> None:
    """Refuse a named criterion paired with the gradient-inert weighting semantic.

    RLRP-761 ``S3`` blocker ``B1``. ``feature_weight_mode: tempered`` applies the
    weight as ``values + log(w)``; on a per-feature NLL that is an ADDITIVE
    CONSTANT, so under the (linear) downstream reduction ``dL/dtheta`` is
    UNCHANGED. Resolving an expensive data-derived criterion and then applying it
    in that semantic produces a **null result that looks like a measurement** --
    the failure mode that would have been misread as falsifying hypothesis ``H1``.

    This is the one incoherence only the resolver can see: it is the single place
    where the resolved semantic and the resolved criterion are both in scope.

    :param mode: the effective ``feature_weight_mode``.
    :param obs_spec: the obs-block weight spec.
    :param act_spec: the act-block weight spec.
    :raises ValueError: on the incoherent pairing.
    """
    if str(mode) != "tempered":
        return
    # `uniform` is a named criterion but resolves to the SCALAR 1.0, i.e. no
    # re-allocation is requested in the first place -- pairing it with the
    # gradient-inert semantic is the legitimate control cell of the S6/S7 A/B.
    named = [
        s
        for s in (obs_spec, act_spec)
        if is_criterion(s) and str(s) != "uniform"
    ]
    if not named:
        return
    raise ValueError(
        f"Feature-weight criterion {named!r} was requested with "
        "`feature_weight_mode: tempered`, which applies the weight as "
        "`values + log(w)` -- an ADDITIVE CONSTANT on an NLL, hence GRADIENT-INERT "
        "(RLRP-761 S3 blocker B1). The criterion would be resolved, logged and "
        "then have no effect on training, producing a null result that looks like "
        "a measurement. Set `feature_loss_weights.feature_weight_mode: "
        "multiplicative`, or use "
        "an explicit numeric / `uniform` weight spec if the tempered semantic is "
        "intended."
    )


def resolve_exogenous_target_dims(cfg, act_names: Sequence[str]) -> List[int]:
    """Return the indices of *act_names* that are EXOGENOUS to the forecast.

    RLRP-761 ``S7.2``. Resolved from the feature contract
    (:data:`_EXOGENOUS_TARGET_KINDS`), never by name matching, so it stays
    correct for every environment family and for any action-block ordering.

    :param cfg: the resolved Hydra config.
    :param act_names: the ordered single-step action dim names.
    :returns: the positions in *act_names* whose feature kind is exogenous.
    """
    try:
        from tools.feature_handling_tools.env_handlers import (
            instantiate_feature_handler,
        )

        handler = instantiate_feature_handler(cfg)
        exogenous_names = {
            dim
            for group in list(handler.act_groups)
            for dim in group.dim_names
            if str(getattr(group.kind, "value", group.kind)) in _EXOGENOUS_TARGET_KINDS
        }
    except Exception:
        return []
    return [i for i, name in enumerate(act_names) if str(name) in exogenous_names]


def apply_exogenous_target_policy(
    cfg,
    act_names: Sequence[str],
    act_weights: Union[float, Sequence[float]],
    dt_target_weight,
):
    """Down-weight the exogenous action-target dims (``dt``).

    RLRP-761 ``S7.2``. The composed action block exists ONLY to let the
    multi-step forecast roll forward without being handed a plan -- the model has
    to infer the command process implicitly -- and the deploy path emits the next
    OBSERVATION only. Forecasting the *clock* is therefore pure waste, and on the
    reference run it consumed **52.4 %** of the target loss budget.

    The weight is applied AFTER the floor and the squash, deliberately: it is a
    structural statement about what the model is asked to predict, not a
    data-derived preference, so it must not be compressed by the dynamic-range
    bound.

    :param cfg: the resolved Hydra config (for the feature contract).
    :param act_names: the ordered single-step action dim names.
    :param act_weights: the resolved action weights (scalar or per-dim).
    :param dt_target_weight: the configured weight, or ``None`` for
        :data:`DEFAULT_DT_TARGET_WEIGHT`.
    :returns: ``(act_weights, {name: weight})`` -- the mapping is empty when the
        policy is a no-op.
    """
    weight = (
        DEFAULT_DT_TARGET_WEIGHT if dt_target_weight is None else float(dt_target_weight)
    )
    if weight == 1.0:
        return act_weights, {}
    indices = resolve_exogenous_target_dims(cfg, act_names)
    if not indices:
        return act_weights, {}
    if isinstance(act_weights, (int, float)):
        values = [float(act_weights)] * len(act_names)
    else:
        values = [float(v) for v in act_weights]
    applied = {}
    for index in indices:
        values[index] = weight
        applied[str(act_names[index])] = weight
    return values, applied


def _fmt_weight(value: float, precision: int = 4) -> str:
    """Format a single loss weight for the console log.

    A genuinely tiny *non-zero* value is shown in scientific notation (e.g.
    ``1e-08``) rather than being rounded to ``0.0``, so a heavily down-weighted
    (but still supervised) dim is never confused with a true zero; only an exact
    ``0.0`` renders as ``0.0`` (RLRP-761 S2, mirroring the diagnostic table).
    """
    v = float(value)
    if v == 0.0:
        return "0.0"
    rounded = round(v, precision)
    if rounded != 0.0:
        return str(rounded)
    # Non-zero but rounds to zero at `precision`: keep it visible via scientific.
    return f"{v:.1e}"


def _pretty(names: Sequence[str], weights: Union[float, Sequence[float]]) -> str:
    if isinstance(weights, float):
        return f"uniform({_fmt_weight(weights)})"
    return (
        "{"
        + ", ".join(
            f"{str(n)!r}: {_fmt_weight(w)}" for n, w in zip(names, weights)
        )
        + "}"
    )


def _apply_to_model(model, obs_weights, act_weights) -> None:
    """Push the resolved weights onto an already-constructed model."""
    if not isinstance(obs_weights, float):
        setter = getattr(model, "set_observation_feature_weights", None)
        if setter is None:
            raise ValueError(
                f"{type(model).__name__} does not expose "
                f"`set_observation_feature_weights`, so a resolved per-feature "
                f"criterion cannot be applied."
            )
        setter(tuple(float(w) for w in obs_weights))
    if not isinstance(act_weights, float):
        setter = getattr(model, "set_action_feature_weights", None)
        if setter is not None:
            setter(tuple(float(w) for w in act_weights))


def _resolve_block_names(cfg, obs_len: int, act_len: int):
    """Return the ``(obs_names, act_names)`` of the resolved feature contract."""
    try:
        from tools.feature_handling_tools.env_handlers import (
            instantiate_feature_handler,
        )

        handler = instantiate_feature_handler(cfg)
        obs_names = [str(n) for n in (handler.obs_dims or [])]
        act_names = [str(n) for n in (handler.act_dims or [])]
        if len(obs_names) == obs_len and len(act_names) == act_len:
            return obs_names, act_names
    except Exception:
        pass
    return (
        [f"obs[{i}]" for i in range(obs_len)],
        [f"act[{j}]" for j in range(act_len)],
    )


def _as_numpy(array):
    return np.asarray(
        array.detach().cpu().numpy() if hasattr(array, "detach") else array
    )


def _split_slot_blocks(
    row_width: int, obs_len: int, act_len: int, history_len: int
) -> Tuple[int, int]:
    """Return ``(n_obs_slots, n_act_slots)`` for a composed row of *row_width*.

    The multistep replay buffer does NOT store the obs history alone: depending
    on the buffer flavour, ``batch.obs`` is either

    * ``obs_len * history_len``                              -- obs history only;
    * ``obs_len * history_len + act_len * history_len``      -- full composed input;
    * ``obs_len * history_len + act_len * (history_len - 1)`` -- the composed
      ``next_obs`` layout, whose action block carries ONE SLOT LESS (the terminal
      action is not part of the composed target).

    The historical implementation assumed the first form unconditionally and
    crashed with ``cannot reshape array of size N into shape (rows, HL, obs_len)``
    on the reference UGV run (row width 177 = 6*20 + 3*19). Deriving the split
    from the measured width keeps the helper correct for every flavour, and for
    any environment family.

    :param row_width: the width of one buffer row.
    :param obs_len: single-step observation width.
    :param act_len: single-step action width.
    :param history_len: number of history slots.
    :return: the number of obs slots and of act slots present in the row.
    :raises ValueError: when *row_width* matches none of the known layouts.
    """
    obs_width = obs_len * history_len
    if row_width == obs_width:
        return history_len, 0
    if act_len:
        for n_act_slots in (history_len, history_len - 1):
            if n_act_slots > 0 and row_width == obs_width + act_len * n_act_slots:
                return history_len, n_act_slots
    raise ValueError(
        f"Cannot interpret a replay-buffer row of width {row_width} with "
        f"singlestep_obs_len={obs_len}, singlestep_act_len={act_len}, "
        f"history_len={history_len}: none of the known composed layouts "
        f"(obs-only / obs+act*HL / obs+act*(HL-1)) matches."
    )


def _slot_sequences(replay_buffer, obs_len: int, act_len: int, history_len: int):
    """Return one ``(n_slots, obs_len + act_len)`` sequence per transition.

    The multistep model input is a HISTORY WINDOW, so each row of the replay
    buffer already is a short, correctly-ordered trajectory: the slot axis IS
    time. Using the windows (rather than the row axis) is what keeps the
    estimator trajectory-aware without needing episode ids.

    When the obs and act blocks do not carry the same number of slots (the
    composed ``next_obs`` layout drops the terminal action), both are truncated
    to the common slot count so the per-slot pairing stays aligned in time.
    """
    batch = replay_buffer.get_all()
    obs = _as_numpy(batch.obs)
    act = _as_numpy(batch.act) if getattr(batch, "act", None) is not None else None

    n_rows, row_width = obs.shape[0], obs.shape[-1]
    n_obs_slots, n_act_slots = _split_slot_blocks(
        row_width, obs_len, act_len, history_len
    )

    obs_width = obs_len * n_obs_slots
    obs_block = obs[:, :obs_width].reshape(n_rows, n_obs_slots, obs_len)

    if n_act_slots:
        # The action history lives in the SAME row, right after the obs block.
        act_block = obs[:, obs_width:].reshape(n_rows, n_act_slots, act_len)
    elif act_len and act is not None and act.shape[-1] == act_len * history_len:
        # Split flavour: the action history is carried by `batch.act`.
        act_block = act.reshape(n_rows, history_len, act_len)
        n_act_slots = history_len
    else:
        act_block = None

    if act_block is None:
        return [obs_block[i] for i in range(n_rows)]

    n_slots = min(n_obs_slots, n_act_slots)
    return [
        np.concatenate([obs_block[i, :n_slots], act_block[i, :n_slots]], axis=-1)
        for i in range(n_rows)
    ]


def _select_block(
    scales: FeatureScales,
    key: str,
    length: int,
    dim_names: Optional[Sequence[str]],
) -> List[float]:
    """Return *key*'s values for the requested block, in the block's order."""
    mapping = scales.as_mapping(key)
    if dim_names is not None:
        names = [str(n) for n in dim_names]
        missing = [n for n in names if n not in mapping]
        if missing:
            raise ValueError(
                f"The measured feature statistics have no entry for {missing}; "
                f"available: {list(mapping)}."
            )
        return [mapping[n] for n in names]
    values = getattr(scales, key)
    if len(values) < length:
        raise ValueError(
            f"The measured feature statistics cover {len(values)} dims < the "
            f"requested block length {length}."
        )
    return [float(v) for v in values[:length]]
