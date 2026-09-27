# coding=utf-8
"""RLRP-736 — per-environment feature geometry loss, isolated as a mixin.

Plan provenance:
  `.junie/ai_artifact/plans/rlrp-736-per-environment-feature-handling-plan-20260711.md`
  (Variant A, YouTrack RLRP-736).

This module extracts the opt-in per-feature geometry-loss concern (e.g. the
quaternion geodesic penalty) out of :class:`ExponentialFamilyMLP` into a focused
``FeatureGeometryLossMixin`` (the "Option A / mixin" refactor). The mixin is
inserted into the MRO of :class:`ExponentialFamilyMLP` so every downstream
multi-step model (``MultiStepMLP`` and its whole family) keeps the exact same
public API (``set_feature_handler`` / ``get_feature_handler`` / ...), while the
RLRP-736 concern lives in one orthogonal place instead of being interleaved with
the exponential-family network/loss code.

Neutral by default: with no handler registered and a zero weight the penalty
is a zero scalar and none of the ``meta`` keys are created, so the term is a
no-op on the legacy loss path.

RLRP-751 (rev.4 architecture pivot): the geometry term is now a first-class
composite term composed INLINE by each host model via
:meth:`_compose_feature_geometry` (the old stash/drain seam is removed), and it
carries a STATIC-vs-AUTO weighting toggle (``_feature_geometry_auto_weighting``,
plan `.junie/ai_artifact/plans/rlrp-751-feature-geometry-composite-loss-auto-weighting-plan-20260723.md`).
Bit-exactness with the pre-RLRP-751 path is explicitly NOT a goal.
"""
import inspect
from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn


def _handler_loss_terms_are_horizon_batchable(feature_handler: Any) -> bool:
    """True iff every group ``loss_term`` of ``feature_handler`` accepts ``keep_batch``.

    RLRP-824 (Valeria A100 profile of 2026-09-17): the per-step ``extra_loss`` loop of
    :meth:`FeatureGeometryLossMixin._feature_geometry_penalty_sequence` issued ``F`` groups
    of tiny kernels + syncs per training step (``F=500`` -> ~2/3 of a 1.8 s step, GPU
    kernel-busy 13 %). The loop can be collapsed into ONE ``extra_loss`` call on the
    ``(F, ..., Do)`` stack only when the leaf callables reduce the FEATURE axis alone
    (``keep_batch=True`` -> ``(..., 1)``), so the horizon mean stays exactly the mean of the
    per-step scalars whatever the leaf's own scalar reduction is. The built-in rotation /
    gravity losses (``rotation_geodesic_loss`` / ``rotation_chordal_loss`` /
    ``gravity_direction_loss``) do; a custom 2-arg ``loss_term`` (e.g. a ``sum``-reduced test
    probe) does not and keeps the legacy per-step loop.
    """
    groups = getattr(feature_handler, "obs_groups", None)
    if not groups:
        return False

    saw_term = False
    for group in groups:
        term = getattr(group, "loss_term", None)
        if term is None:
            continue

        saw_term = True
        try:
            params = inspect.signature(term).parameters
        except (TypeError, ValueError):
            return False

        if "keep_batch" not in params:
            return False

    return saw_term


class FeatureGeometryLossMixin(nn.Module):
    """Opt-in per-feature geometry loss term (RLRP-736).

    Intended to be mixed into an :class:`mbrl.models.Ensemble`-derived model that
    also exposes ``get_one_d_trj_model_obs_denorm_handle`` (as
    :class:`ExponentialFamilyMLP` does). The mixin owns only the feature-loss
    state and hooks:

    * ``_feature_handler`` — the per-environment
      :class:`tools.feature_handling_tools.feature_spec.EnvFeatureHandler`
      (or ``None`` → OFF).
    * ``_feature_geometry_loss_weight`` — scalar weight (``0.0`` → OFF).

    The weight can be supplied either at construction time (as a model
    parameter, via :meth:`_init_feature_geometry_loss`) or later via
    :meth:`set_feature_handler`.
    """

    _feature_handler: Any
    _feature_geometry_loss_weight: float
    _feature_geometry_loss_objective: Any
    _feature_geometry_auto_weighting: bool

    def _init_feature_geometry_loss(
        self,
        feature_geometry_loss_weight: float = 0.0,
        feature_geometry_loss_objective: Any = "geodesic",
        feature_geometry_auto_weighting: bool = False,
    ) -> None:
        """Initialise the feature-loss state.

        Call once from the host model ``__init__`` (after ``nn.Module`` init).
        ``feature_geometry_loss_weight`` is surfaced **solely** as a model constructor
        parameter, so a run enables the term purely from the *model* Hydra config.
        The handler itself is registered post-construction via
        :meth:`set_feature_handler` (it is derived from the environment config at
        ``setup_multistep_step_model``).

        ``feature_geometry_loss_objective`` (RLRP-736 follow-up) mirrors the
        ``ms_model.feature_geometry.loss_objective`` config key and provides a
        SECOND, independent way to disable the geometry term: ``None`` (Hydra
        ``null``) turns the term OFF regardless of the weight, so a multi-run /
        HPO sweep can toggle the term either through ``loss_weight: 0.0`` or
        through ``loss_objective: null``. A truthy objective
        (``"geodesic"`` / ``"chordal"``) leaves the term enabled (subject to the
        weight + handler gates).
        """
        self._feature_handler = None
        self._feature_geometry_loss_weight = float(feature_geometry_loss_weight)
        self._feature_geometry_loss_objective = feature_geometry_loss_objective
        # RLRP-751 (task T2): STATIC vs AUTO weighting toggle for the feature-
        # geometry term. ``False`` (default) -> the term is scaled by the static
        # ``_feature_geometry_loss_weight``; ``True`` -> the composed SS/MS/CP
        # channels are routed through ``composite_loss_automatic_weighting``
        # (Kendall 2018), with ``_feature_geometry_loss_weight`` acting as the
        # pre-multiplier. AUTO gracefully degrades to STATIC on families that do
        # not expose an (enabled) ``composite_loss_automatic_weighting`` module
        # (see :meth:`_feature_geom_auto_weighting_active`).
        self._feature_geometry_auto_weighting = bool(feature_geometry_auto_weighting)
        return None

    def get_feature_handler(self):
        """Return the registered per-environment feature handler (or ``None``)."""
        return self._feature_handler

    def set_feature_handler(self, feature_handler) -> None:
        """Register the per-environment feature handler.

        Wired at ``setup_multistep_step_model`` from the Hydra ``feature_handler``
        selection. The geometry-loss weight is **not** set here — it is a model
        constructor parameter (``feature_geometry_loss_weight``); this method only attaches
        the handler and leaves the constructed weight untouched.

        ``feature_handler=None`` or a constructed weight of ``0.0`` keeps the loss
        byte-identical to the legacy path.
        """
        self._feature_handler = feature_handler
        # RLRP-824: the horizon-batched ``extra_loss`` fast path is decided per handler
        # (see ``_handler_loss_terms_are_horizon_batchable``); reset the memo on re-attach.
        self._feature_geometry_horizon_batchable = None
        return None

    _feature_geometry_horizon_batchable: Optional[bool] = None

    def _feature_geometry_horizon_batched_ok(self) -> bool:
        """Memoised ``_handler_loss_terms_are_horizon_batchable(self._feature_handler)``."""
        memo = getattr(self, "_feature_geometry_horizon_batchable", None)
        if memo is None:
            memo = _handler_loss_terms_are_horizon_batchable(self._feature_handler)
            self._feature_geometry_horizon_batchable = memo
        return bool(memo)

    def _feature_loss_active(self) -> bool:
        """True iff the geometry term is ENABLED.

        Three independent gates must all pass: a handler is registered, the
        weight is non-zero, AND the objective is not ``None``. The last gate
        (RLRP-736 follow-up) lets a run disable the term via either
        ``feature_geometry.loss_weight: 0.0`` OR ``feature_geometry.loss_objective:
        null`` — two disabling alternatives for multi-run / HPO sweeps.
        """
        return (
            self._feature_handler is not None
            and self._feature_geometry_loss_weight != 0.0
            and getattr(self, "_feature_geometry_loss_objective", "geodesic")
            is not None
        )

    def _feature_geom_auto_weighting_active(self) -> bool:
        """True iff the feature-geometry term should be AUTO-weighted this call.

        RLRP-751 (task T3). AUTO engages only when the config toggle is on AND the
        host model exposes an ENABLED ``composite_loss_automatic_weighting`` module
        (a multi-step / MTM-Pro composite concept). On families without an enabled
        module the term gracefully DEGRADES to STATIC ``loss_weight`` scaling.
        """
        if not getattr(self, "_feature_geometry_auto_weighting", False):
            return False

        caw = getattr(self, "composite_loss_automatic_weighting", None)
        return caw is not None and getattr(caw, "enable", False)

    def _feature_geometry_penalty(
        self,
        pred_point: torch.Tensor,
        target: torch.Tensor,
        reduce_batch: bool = False,
    ) -> torch.Tensor:
        """Opt-in per-feature geometry penalty (e.g. quaternion geodesic).

        ``pred_point`` / ``target`` are the model's point estimate and target in
        (denormalized) obs space. Returns a zero scalar when no handler/weight is
        configured (bit-exact legacy default). The handler's ``extra_loss``
        operates group-wise: ``SCALAR`` groups contribute 0, the ``QUATERNION``
        group contributes the geodesic term.

        ``reduce_batch`` (RLRP-751, task T3): when ``True`` the batch axis is
        PRESERVED (only the feature axis is reduced) so the returned tensor keeps
        the ``(E, B, 1)`` shape the composite auto-weighting module requires; the
        flag is threaded down to ``FeatureSpec.extra_loss`` -> the leaf
        ``loss_term`` callables. ``False`` (default) keeps the reduced scalar form
        used by the STATIC path.
        """
        if not self._feature_loss_active():
            if reduce_batch:
                return pred_point.new_zeros(pred_point.shape[:-1] + (1,))
            return pred_point.new_zeros(())

        denorm = self.get_one_d_trj_model_obs_denorm_handle()
        if denorm is not None:
            pred_point = denorm(pred_point)
            target = denorm(target)

        return self._feature_geometry_loss_weight * self._feature_handler.extra_loss(
            pred_point, target, keep_batch=reduce_batch
        )

    def _feature_geometry_penalty_sequence(
        self, point_sequence, reduce_batch: bool = False
    ) -> torch.Tensor:
        """Aggregate the per-step geometry penalties (§14B S3.5).

        Returns the MEAN over the unrolled horizon steps of the per-step
        ``weight * extra_loss`` penalty (so the horizon term does not scale with
        the number of unrolled steps and cannot dominate the single-step
        deploy-head term). Returns a zero scalar for an empty sequence or an
        inactive term (bit-exact legacy default).

        Efficiency (RLRP-736 issue follow-up): the previous implementation looped
        the FULL :meth:`_feature_geometry_penalty` per step, which re-ran the
        (potentially expensive) observation denormalization once per horizon step.
        Here the ordered ``(point, target)`` sequence is instead STACKED along a
        new leading horizon axis and denormalized in a SINGLE call; the handler's
        ``extra_loss`` is then applied per step and averaged. This is a strict
        numerical no-op vs. the per-step loop (the mean over steps is preserved
        EXACTLY, independent of the handler ``loss_term``'s own internal reduction
        — mean, sum, ...) while collapsing ``horizon_len`` denorm calls into one.

        Note: the mean is a deliberately simple, model-agnostic reduction. A model
        whose horizon steps must follow its own temporal-discount schedule can
        instead pre-weight each per-step point or override this method.

        Horizon-batched fast path (RLRP-824, Valeria A100 training-step benchmark of
        2026-09-17): on the ULH windows (``F=500``) the per-step ``extra_loss`` loop
        below was the dominant cost of the whole training step (E2E-TCN H20/F500:
        1838 ms/batch with the term vs 605 ms with ``loss_weight=0``, 82 k vs 20 k CUDA
        kernels/step, GPU kernel-busy 6 %): ``F`` PyPose ``SO3 -> matrix`` conversions +
        Frobenius reductions, each a
        handful of tiny kernels launched from the single Python thread. When every
        group ``loss_term`` is ``keep_batch``-aware (the built-in rotation / gravity
        losses), the handler is called ONCE on the whole ``(F, ..., Do)`` stack with
        ``keep_batch=True`` (feature-axis reduction only -> ``(F, ..., 1)``) and the
        horizon / batch means are taken here: identical to the per-step loop up to
        float summation order (the per-step scalar is itself the mean over the same
        ``(...)`` elements). Custom ``loss_term`` callables without ``keep_batch`` keep
        the exact legacy loop (their own scalar reduction may be a ``sum``).
        """
        if not self._feature_loss_active() or not point_sequence:
            # Mirror :meth:`_feature_geometry_penalty`'s zero-return: honour
            # ``reduce_batch`` (keep the ``(..., 1)`` batch layout in keep-batch
            # mode) and allocate on the input's device/dtype when a sequence is
            # available (falls back to a plain scalar for a truly empty call).
            if point_sequence:
                ref = point_sequence[0][0]
                if reduce_batch:
                    return ref.new_zeros(ref.shape[:-1] + (1,))
                return ref.new_zeros(())
            return torch.zeros(())

        # Stack the ordered horizon steps into a single batched tensor
        # ``(horizon_len, ..., Do)`` so the (potentially expensive) denorm runs
        # ONCE for the whole horizon instead of once per step.
        pred_stack = torch.stack([point for (point, _) in point_sequence], dim=0)
        target_stack = torch.stack([target for (_, target) in point_sequence], dim=0)
        denorm = self.get_one_d_trj_model_obs_denorm_handle()
        if denorm is not None:
            pred_stack = denorm(pred_stack)
            target_stack = denorm(target_stack)

        # Apply the handler ``extra_loss`` per horizon step and MEAN over the
        # horizon. We keep the per-step ``extra_loss`` + explicit mean (rather than
        # folding the horizon into ``extra_loss``'s own reduction) so the result is
        # invariant to whether ``loss_term`` reduces by mean or sum.
        #
        # ``reduce_batch`` (RLRP-751, task T3): in keep-batch mode each per-step
        # ``extra_loss`` keeps the ``(E, B, 1)`` batch layout, so stacking yields
        # ``(H, E, B, 1)`` and the horizon MEAN over dim 0 collapses to the
        # ``(E, B, 1)`` tensor the composite auto-weighting module requires.
        if self._feature_geometry_horizon_batched_ok():
            # ONE handler call over the stacked horizon: ``(F, ..., 1)`` per-element
            # contributions, then the horizon mean (keep-batch) or the horizon+batch
            # mean (scalar mode == mean of the per-step ``extra_loss`` means).
            per_elem = self._feature_handler.extra_loss(
                pred_stack, target_stack, keep_batch=True
            )
            if reduce_batch:
                penalties = per_elem.mean(dim=0)
            else:
                penalties = per_elem.mean()
            return self._feature_geometry_loss_weight * penalties

        penalties = torch.stack(
            [
                self._feature_handler.extra_loss(
                    pred_stack[step], target_stack[step], keep_batch=reduce_batch
                )
                for step in range(pred_stack.shape[0])
            ]
        )
        penalties = penalties.mean(dim=0)
        return self._feature_geometry_loss_weight * penalties

    def _feature_geom_auto_slots_ctor_on(self) -> bool:
        """True iff FEAT_GEOM AUTO slots should be registered at construction (RLRP-751 T6).

        AUTO toggle on AND the ctor-known geometry gates hold (non-zero
        ``loss_weight`` + non-``None`` objective). The handler is attached
        post-construction, so the runtime ``_feature_loss_active`` gate cannot be
        used at build time; this is the build-time analogue used both to register
        the slots and to keep the ``composite_loss_automatic_weighting`` module
        alive when geometry is the ONLY AUTO-routed term.
        """
        if not getattr(self, "_feature_geometry_auto_weighting", False):
            return False

        return (
            self._feature_geometry_loss_weight != 0.0
            and getattr(self, "_feature_geometry_loss_objective", "geodesic")
            is not None
        )

    def _register_feature_geometry_auto_slots(
        self, channels: Sequence[str] = ("SS", "MS")
    ) -> None:
        """Register the requested ``FEAT_GEOM_*`` auto-weighting slots (RLRP-751 T6).

        Idempotent, next-free-index registration on the host model's
        ``composite_loss_automatic_weighting`` module. No-op unless AUTO is
        configured (``_feature_geometry_auto_weighting``) AND the ctor-known
        geometry gates are on (non-zero weight + non-``None`` objective); the
        handler is attached post-construction, so the runtime ``_feature_loss_active``
        gate cannot be used here. Must be called AFTER every other slot
        registration of the host model so the ``FEAT_GEOM_*`` indices are the last
        (free) ones and never collide with the pre-existing slots.
        """
        if not getattr(self, "_feature_geometry_auto_weighting", False):
            return None

        ctor_on = (
            self._feature_geometry_loss_weight != 0.0
            and getattr(self, "_feature_geometry_loss_objective", "geodesic")
            is not None
        )
        caw = getattr(self, "composite_loss_automatic_weighting", None)
        if not ctor_on or caw is None:
            return None

        for channel in channels:
            key = f"FEAT_GEOM_{channel}"
            if key in caw.auto_loss_weights_map:
                continue
            caw.extend({key: len(caw.auto_loss_weights_map)})
        return None

    def _compose_feature_geometry(
        self,
        losses: torch.Tensor,
        meta: Dict[str, Any],
        *,
        ss: Any = None,
        ms_head: Any = None,
        ms_head_legacy_composed_shape: bool = True,
        cp_sequence: Any = None,
    ) -> torch.Tensor:
        """Compose the feature-geometry term(s) INLINE into the caller's accumulator.

        RLRP-751 (tasks T3/T5). Replaces the removed stash/drain seam: each model
        calls this explicitly inside its own ``_probabilistic_loss`` /
        ``_deterministic_loss`` (or CP scorer), at the SAME reduction stage as
        every other composite term, and uses the returned (updated) ``losses``.

        Channels (all optional; only the ones the caller has in hand are composed):

        * ``ss=(point, target)`` — the single-step / deploy-head (``t=1``) point
          estimate + target, in the single-step obs layout ``(..., Do)`` -> the
          ``FEAT_GEOM_SS`` channel / ``feature_geom_loss`` metric.
        * ``ms_head=(composed_pred, composed_target)`` — the multi-step forecast
          head in the composed multi-step next-obs layout; decomposed per horizon
          step via :meth:`decompose_composed_next_obs_to_singlestep_horizon`
          (``ms_head_legacy_composed_shape`` selects the layout). ``FEAT_GEOM_MS``
          channel / ``feature_geom_loss_ms`` metric. No-op when ``horizon_len<=1``
          (that single step is the ``t=1`` step the SS channel already covers).
        * ``cp_sequence`` — an ordered list of per-step ``(point, target)`` pairs
          for the compounded-prediction / free-running unroll. ``FEAT_GEOM_CP``
          channel / ``feature_geom_loss_cp`` metric.

        STATIC mode scales each term by ``loss_weight`` (feature/batch-reduced
        scalar-like); AUTO mode (when :meth:`_feature_geom_auto_weighting_active`)
        routes the batch-preserved ``(E, B, 1)`` penalty through
        ``composite_loss_automatic_weighting`` (``are_log_prob_losses=False``,
        matching DH/IPROJ/RC). Returns ``losses`` unchanged when the term is
        inactive (bit-neutral OFF).
        """
        if not self._feature_loss_active():
            return losses
        auto = self._feature_geom_auto_weighting_active()

        def _channel_auto(channel: str) -> bool:
            # AUTO for this channel only when its dedicated slot is registered;
            # otherwise gracefully degrade to STATIC (avoids a slot-map KeyError on
            # families that did not register the ``FEAT_GEOM_*`` slots).
            if not auto:
                return False
            return (
                f"FEAT_GEOM_{channel}"
                in self.composite_loss_automatic_weighting.auto_loss_weights_map
            )

        if ss is not None:
            ss_auto = _channel_auto("SS")
            point, feat_target = ss
            term = self._feature_geometry_penalty(
                point, feat_target, reduce_batch=ss_auto
            )
            # A7 (RLRP-788): gate the diagnostic write (``term`` still feeds ``losses``)
            # (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
            if self._enable_meta_collection:
                meta["feature_geom_loss"] = term.detach().mean().item()
            if ss_auto:
                term, _ = self.composite_loss_automatic_weighting(
                    term, "FEAT_GEOM_SS", meta, are_log_prob_losses=False
                )
            losses = losses + term

        if ms_head is not None and getattr(self, "horizon_len", 1) > 1:
            ms_auto = _channel_auto("MS")
            ms_pred, ms_target = ms_head
            pred_steps = self.decompose_composed_next_obs_to_singlestep_horizon(
                ms_pred, legacy_composed_shape=ms_head_legacy_composed_shape
            )
            target_steps = self.decompose_composed_next_obs_to_singlestep_horizon(
                ms_target, legacy_composed_shape=ms_head_legacy_composed_shape
            )
            seq = list(zip(pred_steps, target_steps))
            if seq:
                term = self._feature_geometry_penalty_sequence(
                    seq, reduce_batch=ms_auto
                )
                # A7 (RLRP-788): gate the diagnostic write (``term`` still feeds ``losses``)
                # (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
                if self._enable_meta_collection:
                    meta["feature_geom_loss_ms"] = term.detach().mean().item()
                if ms_auto:
                    term, _ = self.composite_loss_automatic_weighting(
                        term, "FEAT_GEOM_MS", meta, are_log_prob_losses=False
                    )
                losses = losses + term

        if cp_sequence:
            cp_auto = _channel_auto("CP")
            term = self._feature_geometry_penalty_sequence(
                list(cp_sequence), reduce_batch=cp_auto
            )
            # A7 (RLRP-788): gate the diagnostic write (``term`` still feeds ``losses``)
            # (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
            if self._enable_meta_collection:
                meta["feature_geom_loss_cp"] = term.detach().mean().item()
            if cp_auto:
                term, _ = self.composite_loss_automatic_weighting(
                    term, "FEAT_GEOM_CP", meta, are_log_prob_losses=False
                )
            losses = losses + term

        return losses
