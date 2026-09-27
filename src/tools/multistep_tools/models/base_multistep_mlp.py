# coding=utf-8
import pathlib
import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F

from tools.multistep_tools.models.base_multistep_mlp_utils import (
    CompositeLossAutomaticWeighting,
)
from tools.multistep_tools.multistep_model_util import (
    EnsembleDeployHead,
    extract_multistep_features_from_array,
    timestep_first_multistep_dim_unflaten_array,
)
from tools.multistep_tools.utils import compute_multistep_model_in_size
from tools.feature_handling_tools.env_handlers import orientation_slot_layout
from tools.multistep_tools.models.exponential_family_mlp import ExponentialFamilyMLP
from tools.multistep_tools.models.orientation_config import (
    resolve_feature_geometry_cfg,
    resolve_internal_orientation_cfg,
)
from tools.model_adapter_tools.ms_to_ss_observation_adapter import (
    MultistepObservationToSinglestepObservationAdapter,
)


class MultiStepMLP(ExponentialFamilyMLP):
    singlestep_obs_len: int
    singlestep_act_len: int
    history_len: int
    horizon_len: int
    _compose_next_obs_multistep_obs_horizon_slice: slice
    _compose_next_obs_multistep_act_horizon_slice: slice
    _multistep_to_singlestep_adapter: Optional[
        Union[Callable, MultistepObservationToSinglestepObservationAdapter]
    ]
    deploy_head_mean_adapter: Optional[nn.Module]
    deploy_head_logvar_adapter: Optional[nn.Module]

    @staticmethod
    def _expand_singlestep_orientation_slots(
        base_slots: Optional[Sequence[int]],
        singlestep_obs_len: int,
        history_len: int,
    ) -> Optional[Sequence[int]]:
        """Expand single-step attitude base slot(s) to the composed multi-step window.

        RLRP-736 §S2.2-B composed-layout orientation. ``base_slots`` are the
        attitude base index/indices WITHIN a single-step obs block (e.g. ``3``
        for a width-10 robotic-3D obs). The composed input window and the base
        ``MultiStepMLP`` composed output both lay out the observation part as
        ``history_len`` contiguous obs blocks (each ``singlestep_obs_len`` wide),
        so each base slot ``b`` maps to ``k * singlestep_obs_len + b`` for every
        ``k in range(history_len)``. Returns a sorted, non-overlapping index list
        (or ``None`` / the input unchanged when there is nothing to expand).

        The expansion is pure index arithmetic and therefore WIDTH-AGNOSTIC. RLRP-796
        Stage 1: when ``base_slots`` is an
        :class:`~tools.feature_handling_tools.env_handlers.OrientationSlotLayout`,
        the resolved ``external_width`` / ``kind`` are CARRIED THROUGH to the
        expanded value, so the leaf ``_setup_orientation_rep`` can size the trunk
        off the block that is actually present (4-D attitude vs. 3-D gravity)
        instead of assuming the quaternion block.

        :param base_slots: Single-step attitude base slot indices, or ``None``.
        :param singlestep_obs_len: Width of a single-step observation block.
        :param history_len: Number of obs blocks in the composed window.
        :return: The expanded composed-window slot indices (an
            ``OrientationSlotLayout`` when the input carried one).
        """
        from tools.feature_handling_tools.env_handlers import OrientationSlotLayout

        if not base_slots:
            return base_slots
        expanded = sorted(
            k * singlestep_obs_len + int(b)
            for k in range(history_len)
            for b in base_slots
        )
        if isinstance(base_slots, OrientationSlotLayout):
            return OrientationSlotLayout(
                expanded,
                external_width=base_slots.external_width,
                kind=base_slots.kind,
            )
        return expanded

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        singlestep_obs_len: int,
        singlestep_act_len: int,
        history_len: int,
        horizon_len: int,
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        deterministic: bool = False,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        logvar_bound_grad_clip: Optional[float] = None,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        distribution_name: str = "gaussian",
        mae_loss: bool = True,
        model_use_double_precision: bool = False,
        dropout: float = 0.0,
        ms_head_dropout: float = 0.0,
        ms_head_num_layers: int = 1,
        residual_form: str = "identity_v2",
        layer_bloc=None,  # RLRP-768 per-region layer-bloc selector (forwarded)
        enable_auto_loss_weighting: bool = False,
        auto_weighting_noise_model: str = "gaussian",
        auto_weighting_scheme: str = "tempered_likelihood",
        # RLRP-736 nested config groups (replace the former flat orientation /
        # feature-geometry kwargs).
        feature_geometry: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        internal_orientation: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        # RLRP-736 orientation-slots auto-inference: the attitude base slot(s)
        # WITHIN a single-step obs block, derived automatically at the setup seam
        # from ``environment.obs_dims`` (NOT a user-facing config key). ``None`` ->
        # neutral / bit-exact OFF. This replaces the former
        # ``internal_orientation.{input_slots, output_slots}`` config coupling.
        orientation_singlestep_slots: Optional[Sequence[int]] = None,
        # RLRP-761 S3: per-feature loss-weighting knobs. This base is NOT
        # feature-weighted (see ``AbstractFeatureWeightedMultiStepMLP``, which
        # CONSUMES them instead of forwarding); they are accepted here purely for
        # Hydra config-surface compatibility -- see
        # :meth:`_validate_unsupported_feature_weighting`.
        feature_weight_mode: str = "tempered",
        feature_weight_max_ratio: Optional[float] = None,
        description: Optional[str] = None,
    ):
        self._validate_unsupported_feature_weighting(
            feature_weight_mode, feature_weight_max_ratio
        )
        # RLRP-736: unpack the nested config groups into the flat locals used by the
        # composed-window slot expansion below; the (possibly expanded) values are
        # re-packed when forwarding to ``ExponentialFamilyMLP``.
        _io = resolve_internal_orientation_cfg(internal_orientation)
        internal_orientation_rep = _io["representation"]
        orientation_tangent_nll_right_jacobian = _io["tangent_nll_right_jacobian"]
        # Tri-state manifold-aware NLL lever. Introduced by item A (§A.2) of the
        # Quaternion manifold upgrade + memoryless ``w >= 0`` removal `.junie` plan
        # (`rlrp-744-quaternion-manifold-upgrade-and-memoryless-wge0-removal-plan-20260720.md`).
        orientation_manifold_aware_nll = _io["manifold_aware_nll"]
        # RLRP-744 item E: forward the resolved ``enforce_continuity`` flag to the
        # base so it reaches the model and can gate the model-side reference-relative
        # attitude alignment (`_quaternion_ar_continuity_active`), not just ingestion.
        orientation_enforce_continuity = _io["enforce_continuity"]
        # RLRP-736 orientation-slots auto-inference: the attitude slot is no longer a
        # config key. The SAME single-step attitude base slot drives BOTH the obs
        # input and the obs output (robotic attitude appears in both), so the input
        # and output composed-window expansions start from ``orientation_singlestep_slots``.
        orientation_input_slots = orientation_singlestep_slots
        orientation_output_slots = orientation_singlestep_slots
        # RLRP-736 §S2.2-B — composed-layout orientation.
        # A ``MultiStepMLP`` consumes/emits a COMPOSED multi-step window
        # (``[obs_0..obs_{H-1}][act_0..act_{H-1}]``, each obs block
        # ``singlestep_obs_len`` wide, ``history_len`` of them). The operator
        # supplies ``orientation_{input,output}_slots`` as the attitude base
        # index WITHIN a single-step obs block (e.g. ``3`` for a width-10
        # robotic-3D obs); here we expand it to every obs block of the composed
        # window so the single-step attitude slot encode/decode
        # (``ExponentialFamilyMLP._splice_slots``) lines up. Input and the base
        # ``MultiStepMLP`` output both use the ``out_size`` composed layout with
        # ``history_len`` obs blocks, so the SAME expansion applies to both.
        # RLRP-824: the composed OUTPUT window is ``W = output_window_len`` obs blocks
        # (``history_len`` for every family except the MS->MS forecast family, where
        # ``W = max(H, F)``), so the output-slot expansion must span ``W`` blocks. Resolved
        # here (before ``super().__init__``) from the raw ctor args through the class-level
        # :meth:`resolve_output_window_len`; ``W == H`` for every legacy config => bit-exact.
        horizon_len = self._coerce_horizon_len(history_len, horizon_len)
        _output_window_len = type(self).resolve_output_window_len(history_len, horizon_len)
        _rep_active = (
            str(internal_orientation_rep) not in ("quaternion", "quaternion_legacy")
            and bool(orientation_input_slots or orientation_output_slots)
        )
        # RLRP-744 item A (§A.3) + item C (rev.3b): the plain-``quaternion``
        # by-construction path (ALWAYS-ON in-graph unit decode; tangent-NLL when
        # ``manifold_aware_nll: true``) needs the SAME composed-window slot
        # expansion (one attitude slot per obs block) so the decode / tangent-NLL
        # scoring line up — with ``width == 4`` the expansion is purely positional
        # (no trunk-width change). ``quaternion_legacy`` stays excluded (raw,
        # un-normalized bit-exact baseline).
        _quaternion_by_construction_active = (
            str(internal_orientation_rep) == "quaternion"
            and bool(orientation_input_slots or orientation_output_slots)
        )
        if _rep_active or _quaternion_by_construction_active:
            orientation_input_slots = self._expand_singlestep_orientation_slots(
                orientation_input_slots, singlestep_obs_len, history_len
            )
            orientation_output_slots = self._expand_singlestep_orientation_slots(
                orientation_output_slots, singlestep_obs_len, _output_window_len
            )
        super().__init__(
            in_size,
            out_size,
            device,
            mae_loss=mae_loss,
            num_layers=num_layers,
            ensemble_size=ensemble_size,
            hid_size=hid_size,
            deterministic=deterministic,
            propagation_method=propagation_method,
            learn_logvar_bounds=learn_logvar_bounds,
            logvar_bound_grad_clip=logvar_bound_grad_clip,
            activation_fn_cfg=activation_fn_cfg,
            distribution_name=distribution_name,
            model_use_double_precision=model_use_double_precision,
            dropout=dropout,
            ms_head_dropout=ms_head_dropout,
            ms_head_num_layers=ms_head_num_layers,
            residual_form=residual_form,
            layer_bloc=layer_bloc,
            # RLRP-736: ``feature_geometry`` passes through unchanged; the
            # ``internal_orientation`` group carries ONLY ``representation`` +
            # ``tangent_nll_right_jacobian`` (no slots). The composed-window EXPANDED
            # attitude slots computed above travel on the DEDICATED explicit channel
            # ``orientation_{input,output}_slots`` of ``ExponentialFamilyMLP``.
            feature_geometry=feature_geometry,
            internal_orientation={
                "representation": internal_orientation_rep,
                "tangent_nll_right_jacobian": orientation_tangent_nll_right_jacobian,
                "manifold_aware_nll": orientation_manifold_aware_nll,
                "enforce_continuity": orientation_enforce_continuity,
            },
            orientation_input_slots=orientation_input_slots,
            orientation_output_slots=orientation_output_slots,
            # RLRP-736 bespoke-forward plan §2.1/§5.1: hand the single-step obs
            # width to the base so the DEDICATED SS (deploy) head of architectures
            # B/C can be sized (``_ss_trunk_out_len``) and its attitude slot(s)
            # tracked (``_ss_ori_out_slots``) BEFORE ``_build_network`` runs. The
            # ``orientation_output_slots`` above are already the composed-window
            # expansion; the base derives the SS base slots from those that fall in
            # the first block (``s < singlestep_obs_len``). Neutral/bit-exact when
            # the wiring is off.
            singlestep_obs_len=singlestep_obs_len,
            description=description,
        )

        self.singlestep_obs_len = singlestep_obs_len
        self.singlestep_act_len = singlestep_act_len

        # RLRP-736 attitude double-cover consistency (Item 1): a REP-AGNOSTIC store
        # of the attitude base slot(s) WITHIN a single-step obs block. Unlike
        # ``_ss_ori_*_slots`` / ``_ori_*_slots`` (populated ONLY when
        # ``_internal_orientation_rep_active`` is True, i.e. EMPTY for BOTH the ``quaternion``
        # and ``quaternion_legacy`` reps), this store keeps the auto-inferred slot
        # for EVERY rep. It is the single source of truth for the history-relative
        # AR-splice sign-continuity pass, whose ablation-lever gate is
        # ``bool(self._orientation_singlestep_slots) and rep is not QUATERNIONLEGACY``
        # (so it fires for the sign-SENSITIVE plain ``quaternion`` rep, which
        # ``_internal_orientation_rep_active`` would wrongly exclude). This lives in
        # ``MultiStepMLP.__init__`` ONLY: the whole MRO of both the MTM-Pro family
        # AND the ``ms2ss`` AR chain flows through here, so a single store covers
        # every hosting model (do NOT duplicate it in ``AbstractMS2SSAutoRegressive``).
        # RLRP-796 Stage 1: kept as an ``OrientationSlotLayout`` (a ``tuple``
        # subclass, so every existing tuple consumer is unaffected) so the block's
        # external width / kind stay retrievable post-construction.
        self._orientation_singlestep_slots: Tuple[int, ...] = orientation_slot_layout(
            orientation_singlestep_slots
        )

        self._multistep_to_singlestep_adapter = None
        self.deploy_head_mean_adapter = None
        self.deploy_head_logvar_adapter = None

        # .... Multi-step values ..................................................................
        self.history_len = history_len
        self.horizon_len = horizon_len
        self._validate_multistep_window_lens()

        # .... composed observation indexes + deploy-head adapter .................................
        self._build_multistep_derived_state()

        # `auto_weighting_noise_model` / `auto_weighting_scheme` are the weighting scheme's own
        # knobs (INDEPENDENT from `distribution_name`, the predictive head). Threaded by stage 4.8
        # of the "improve CompositeLossAutomaticWeighting" .junie plan
        # (refactor_composite_loss_auto_weighting_and_shifted_softplus_plan_20260623.md, RLRP-720).
        self.composite_loss_automatic_weighting = CompositeLossAutomaticWeighting(
            dtype=self.model_dtype,
            enable=enable_auto_loss_weighting,
            noise_model=auto_weighting_noise_model,
            weighting_scheme=auto_weighting_scheme,
            ensemble_size=self.num_members,
        )

        # .... Final build step ...................................................................
        if model_use_double_precision:
            self.to(dtype=torch.double)
        else:
            self.to(dtype=self.model_dtype)

    # ==== Multistep window contract (RLRP-824) ===================================================
    @staticmethod
    def _coerce_horizon_len(history_len: int, horizon_len: Union[int, float]) -> int:
        """Resolve a fractional ``horizon_len`` (a ratio of ``history_len``) to an integer.

        Pure arithmetic hoisted out of ``__init__`` (RLRP-824) so the output window can be
        resolved BEFORE ``super().__init__``; ``int`` inputs are returned unchanged.
        """
        if isinstance(horizon_len, float):
            horizon_len = round(history_len * horizon_len)
            if horizon_len < 1:
                horizon_len = 1
        return horizon_len

    @classmethod
    def resolve_output_window_len(cls, history_len: int, horizon_len: int) -> int:
        """Number of composed obs blocks ``W`` in this family's OUTPUT window.

        Permanent contract hook. Introduced by the Ultra-long-horizon MS->MS training via a
        lazy window DataLoader ``.junie`` plan (RLRP-824,
        ``ultra_long_horizon_dataloader_plan_RLRP-824_20260915.md``, KD1).

        **Default: ``history_len``** -- the legacy composed output layout
        ``obs*H + act*(H-1)`` shared by the MTM-Pro / dual-head / AR families, for which
        ``horizon_len <= history_len`` is a hard invariant. The MS->MS forecast family
        (``AbstractMS2MSForecast``: E2E-TCN / M3 / TBM) overrides it to ``max(H, F)`` so that
        an asymmetric ``F > H`` forecast can be emitted as a ``W = F`` obs-block window.

        A ``classmethod`` (not a property) on purpose: the setup seam
        (``create_one_dim_tr_model_v2`` / ``R2SMotionModelFactorySpec``) must size ``out_size``
        from the Hydra ``_target_`` BEFORE any instance exists.

        :param history_len: ``H``, the composed input window length.
        :param horizon_len: ``F``, the forecast depth (already integer-coerced).
        :return: ``W``, the composed output window length in obs blocks.
        """
        return int(history_len)

    @property
    def output_window_len(self) -> int:
        """``W``: the composed OUTPUT window length in obs blocks (see
        :meth:`resolve_output_window_len`). Equals ``history_len`` for every legacy family."""
        return type(self).resolve_output_window_len(self.history_len, self.horizon_len)

    def composed_window_len(self, is_model_output: bool) -> int:
        """Multistep axis length of a composed array: ``H`` for a model INPUT, ``W`` for an OUTPUT.

        Composed-observation compliance hook (RLRP-824 FR10): every helper of this class that
        unflattens / slices a composed array dispatches its multistep axis on this method
        instead of hard-coding ``history_len``.
        """
        return self.output_window_len if is_model_output else int(self.history_len)

    def _validate_multistep_window_lens(self) -> None:
        """Enforce ``1 <= horizon_len <= output_window_len`` (RLRP-824 relaxation of the legacy
        ``horizon_len <= history_len`` invariant; identical for every family whose output
        window IS ``history_len``)."""
        horizon_len = int(self.horizon_len)
        output_window_len = int(self.output_window_len)
        assert 1 <= horizon_len <= output_window_len, (
            f"1 <= horizon_len {horizon_len} <= output_window_len {output_window_len} "
            f"(history_len={self.history_len}; the output window is history_len for every "
            "family except the MS->MS forecast family, where it is max(history_len, horizon_len))"
        )
        return None

    # ==== Config-surface compatibility ===========================================================
    def _validate_unsupported_feature_weighting(
        self,
        feature_weight_mode: str,
        feature_weight_max_ratio: Optional[float],
    ) -> None:
        """Warn when a per-feature loss-weighting knob reaches a family that ignores it.

        RLRP-761 ``S3``. ``ms_model`` is a Hydra INSTANTIATION TARGET, so every key
        it carries is forwarded to the leaf model ``__init__``. Accepting the two
        feature-weight knobs at this base — the root of the multistep tree — means
        the *whole* family shares one uniform instantiation contract instead of a
        per-class one, which is the property that makes a knob safely promotable to
        a shared default or reusable across an experiment's model families.

        Accepting is not the same as honouring, hence the warning below.

        Overridden (consumed) by :class:`AbstractFeatureWeightedMultiStepMLP`, which
        is the only family that HONOURS the knobs; reaching this implementation with
        a non-default value therefore means the operator expects a loss-budget
        re-allocation that will NOT happen -- a silent null result, so warn.

        :param feature_weight_mode: the requested weighting semantic.
        :param feature_weight_max_ratio: the requested weight-ratio bound.
        """
        requested = []
        if feature_weight_mode is not None and str(feature_weight_mode) != "tempered":
            requested.append(f"feature_weight_mode={feature_weight_mode!r}")
        if feature_weight_max_ratio is not None:
            requested.append(f"feature_weight_max_ratio={feature_weight_max_ratio!r}")
        if requested:
            warnings.warn(
                f"{type(self).__name__} does NOT support per-feature loss weighting "
                f"(RLRP-761 S3); {', '.join(requested)} is accepted for config-surface "
                "compatibility but IGNORED. Remove it from the `ms_model` node, or use "
                "a model of the feature-weighted family, if the re-allocation is "
                "intended.",
                RuntimeWarning,
                stacklevel=3,
            )

    def set_deploy_head_multistep_to_singlestep_adapter(
        self,
        adapter: Union[Callable, MultistepObservationToSinglestepObservationAdapter],
    ) -> None:
        """set the multistep composed next observation to single-step next obserbvation adapter
        for the 'deploy_head' module and 'multistep_to_singlestep_next_obs_adapter'.

        :param adapter: The multistep composed input tensor representing the model's next obs.
        :return: None
        """
        # (Priority) ToDo: implement adapter serialization (add to save/load method)
        self._multistep_to_singlestep_adapter = adapter
        if self.deterministic:
            self.deploy_head_mean_adapter = EnsembleDeployHead(adapter)
        else:
            self.deploy_head_mean_adapter = EnsembleDeployHead(adapter)
            self.deploy_head_logvar_adapter = EnsembleDeployHead(adapter)
        return None

    def multistep_to_singlestep_next_obs_adapter(
        self, obs: torch.Tensor
    ) -> torch.Tensor:
        """Adapts multistep composed next observation to single-step next obserbvation.
        This method is for manual/explicit use on tensor. Model prediction output from
         the 'deploy_head' module are already adapted to single-step next observation.

        :param obs: The multistep composed input tensor representing the model's next observation.
        :return: The single-step next observation tensor.
        """
        return self._multistep_to_singlestep_adapter(obs)

    def decompose_composed_next_obs_to_singlestep_horizon(
        self,
        composed_next_obs: torch.Tensor,
        legacy_composed_shape: bool = True,
    ) -> list[torch.Tensor]:
        """Split a composed multi-step next-obs tensor into its per-horizon-step
        single-step observations (RLRP-736 §14B S3.5 — composed-observation compliant).

        The MS (multi-step forecast) head emits its prediction / target in a
        *composed multi-step next-observation* layout: a flattened window whose
        predicted horizon observations are ``horizon_len`` consecutive single-step
        observation blocks of ``singlestep_obs_len`` (the SS / deploy head extracts
        only the FIRST block, ``t=1``, via
        :meth:`multistep_to_singlestep_next_obs_adapter`). This helper returns
        those blocks IN HORIZON ORDER as a list of ``(..., singlestep_obs_len)``
        single-step observations.

        Rationale: a per-feature geometry term (e.g. the quaternion geodesic
        penalty) is defined on the single-step observation layout — it locates
        the attitude dims WITHIN a single-step obs and denormalizes it with the
        single-step obs denorm handle. Applying it directly to a flattened
        composed window (or to a raw strided index into it) would be
        ill-defined; decomposing first makes the MS-head supervision
        composed-observation compliant and identical in kind to the SS-head
        supervision, just repeated per forecast step.

        Two composed layouts are supported, selected by ``legacy_composed_shape``:

        * ``legacy_composed_shape=True`` (default) — the **legacy** composed
          next-obs layout, a flattened window of length ``out_size`` spanning the
          full ``history_len`` (``obs*history_len + act*(history_len-1)``). The
          predicted horizon observations are the LAST ``horizon_len`` timesteps.
          Decomposed with the already tested + vectorized
          :meth:`unflaten_multistep_composed_array` (which reshapes to
          ``(..., F, history_len)``, obs features first) and sliced to the last
          ``horizon_len`` timesteps.
        * ``legacy_composed_shape=False`` — the **future** composed next-obs
          layout used by the MTM-Pro family, a flattened window of length
          ``ho_out_size`` spanning only the ``horizon_len`` forecast steps
          (``obs*horizon_len + act*(horizon_len-1)``, i.e. the last action
          timestep padding already removed). Decomposed with the tested +
          vectorized :func:`timestep_first_multistep_dim_unflaten_array` with
          ``sequence_len=self.horizon_len`` (and last-action padding re-added), so
          NO history-length slicing is needed — every one of the ``horizon_len``
          timesteps is a forecast step.

        In both layouts the returned list is in horizon order, index 0 == ``t=1``
        == the SS / deploy-head step.

        :param composed_next_obs: tensor in the composed multi-step next-obs
            layout, shape ``(..., out_size)`` (legacy) or ``(..., ho_out_size)``
            (future).
        :param legacy_composed_shape: select the composed layout — ``True`` for
            the history-length legacy layout (default), ``False`` for the
            horizon-length future layout (MTM-Pro).
        :return: list of ``horizon_len`` single-step obs tensors, ``(..., Do)``,
            in horizon order (index 0 == ``t=1`` == the SS/deploy-head step).
        """
        obs_len = self.singlestep_obs_len
        if legacy_composed_shape:
            # Legacy history-length composed window. Reshape to a timestep-indexed
            # tensor ``(..., F, history_len)`` (obs features first) in one
            # vectorized op, then keep the LAST ``horizon_len`` timesteps (the
            # predicted horizon) and the first ``singlestep_obs_len`` feature rows.
            unflat = self.unflaten_multistep_composed_array(
                composed_next_obs, is_model_output=True
            )
            # RLRP-824: the window is ``W = output_window_len`` steps (``== history_len`` for
            # every legacy family); the forecast is its trailing ``horizon_len`` steps.
            horizon_obs = unflat[
                ..., :obs_len, self.output_window_len - self.horizon_len :
            ]  # (..., Do, horizon_len), timestep-last
            # ``unbind`` on the timestep axis yields the per-step single-step obs
            # views (the heavy reshape above is already vectorized).
            return list(torch.unbind(horizon_obs, dim=-1))

        # Future horizon-length composed window (MTM-Pro). Reshape to a
        # timestep-first tensor ``(..., horizon_len, F)`` via the tested +
        # vectorized ``timestep_first_multistep_dim_unflaten_array`` with the last
        # action timestep padding re-added (the future layout drops it, matching
        # ``ho_out_size``). Every one of the ``horizon_len`` timesteps is a
        # forecast step, so NO history-length slicing is required.
        unflat = timestep_first_multistep_dim_unflaten_array(
            composed_next_obs,
            self.singlestep_obs_len,
            self.singlestep_act_len,
            sequence_len=self.horizon_len,
            enable_last_action_padding=self.singlestep_act_len > 0,
        )  # (..., horizon_len, F), timestep-first
        horizon_obs = unflat[..., :obs_len]  # (..., horizon_len, Do)
        # ``unbind`` on the timestep axis (dim=-2) yields the per-step single-step
        # obs views in horizon order.
        return list(torch.unbind(horizon_obs, dim=-2))

    def _default_deploy_head(
        self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert (
            self.deploy_head_mean_adapter is not None
        ), f"Deploy head where not setup, use 'set_deploy_head_multistep_to_singlestep_adapter' method first."

        self._maybe_toggle_layers_use_only_elite(only_elite)

        x = self._maybe_cast_to_model_dtype(x)

        if self.deterministic:
            ms_mean, _ = self._default_forward(x, only_elite=True)
            ss_mean = self.deploy_head_mean_adapter(ms_mean)
            ss_logvar = None
        else:
            ms_mean, ms_logvar = self._default_forward(x, only_elite=True)
            ss_mean = self.deploy_head_mean_adapter(ms_mean)
            ss_logvar = self.deploy_head_logvar_adapter(ms_logvar)

        self._maybe_toggle_layers_use_only_elite(only_elite)
        return ss_mean, ss_logvar

    def deploy(
        self, x: torch.Tensor, only_elite: bool = True, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ss_mean, ss_logvar = self._default_deploy_head(x, only_elite)
        return ss_mean, ss_logvar

    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Note: Assume sample returned by BootstrapIterator/SequenceTransitionIterator
        #       are with property _bootstrap_iter=False => sample are received whitout
        #       the model ensemble dimension.
        meta = {}
        with torch.inference_mode():
            self.eval()

            assert (
                model_in.ndim == 2 and target.ndim == 2
            ), f"{model_in.ndim=} != 2 and/or {target.ndim=} != 2"

            ss_pred_mean, _ = self.deploy(model_in)

            target = target.repeat((self.num_members, 1, 1))
            ss_target = self.multistep_to_singlestep_next_obs_adapter(target)

            if self.mae_loss:
                ss_losses = F.l1_loss(ss_pred_mean, ss_target, reduction="none")
            else:
                ss_losses = F.mse_loss(ss_pred_mean, ss_target, reduction="none")

            # .... memory management ..............................................................
            del model_in, target

            return ss_losses, meta

    def _forward_propagation(
        self,
        model_input: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.propagation_method in ["expectation", "post_sampling_expectation"]:
            # Reproduce the expectation propagation logic but without the applying the mean
            # at the last step i.e., output E x B x Od instead of B x Od

            assert model_input.ndim == 2
            model_len = (
                len(self.elite_models) if self.elite_models is not None else len(self)
            )
            if model_input.shape[0] % model_len != 0:
                raise ValueError(
                    "Requires batch size to be a multiple of "
                    f"the number of models. Current batch size is {model_input.shape[0]} for "
                    f"{model_len} models."
                )

            model_input = model_input.unsqueeze(0)
            ensemble_means, ensemble_logvars = self.deploy(model_input, only_elite=True)

        elif self.propagation_method is None:

            ensemble_means, ensemble_logvars = self.deploy(model_input, only_elite=True)

        else:

            ensemble_means, ensemble_logvars = self.forward(
                model_input,
                use_propagation=True,
                propagation_indices=model_state["propagation_indices"],
            )

        # # .... Weight ...........................................................................
        # ensemble_means = self.apply_next_obs_temporal_discount_factor_weights(ensemble_means)
        # ensemble_means = self.apply_next_obs_feature_weights(ensemble_means)
        # ensemble_logvars = self.apply_next_obs_temporal_discount_factor_weights(
        #     ensemble_logvars, log_space=True
        # )
        # ensemble_logvars = self.apply_next_obs_feature_weights(ensemble_logvars, log_space=True)
        #
        # # # .... Normalize ......................................................................
        # # ensemble_means /= ensemble_means.sum(2, keepdim=True)
        # # ensemble_logvars -= torch.logsumexp(
        # #     torch.logsumexp(
        # #         self.unflaten_multistep_composed_array(-ensemble_logvars,
        # #                                                is_model_output=True),
        # #         dim=3,
        # #         keepdim=False,
        # #     ),
        # #     dim=2,
        # #     keepdim=True,
        # # )

        return ensemble_means, ensemble_logvars

    def _cached_act_pad(
        self,
        framework,
        leading_shape: Tuple[int, ...],
        act_pad_value: float,
        reference: Union[np.ndarray, "torch.Tensor"],
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """Return a (cached) action-padding block for :meth:`unflaten_multistep_composed_array`.

        Permanent hot-path helper. Introduced by action ``A5`` of the RLRC MTM-Pro models
        code optimization `.junie` plan
        (``perf_RLRP-783_mtm_pro_models_code_optimization_plan_20260827.md``).
        The pad is a CONSTANT for a given (framework, leading shape, reference dtype, device,
        fill value) tuple, so it is built once and reused instead of being rebuilt on every
        call of this central hot-path helper; the returned tensor/array is read-only by
        contract (it is only ever an operand of ``concatenate``). Bit-exact: the pad is built
        the SAME way as the legacy inline code -- ``framework.full`` default dtype, and for
        torch a device move ONLY (no dtype cast, matching the legacy behaviour, R-A5b).
        """
        cache = getattr(self, "_act_pad_cache", None)
        if cache is None:
            cache = {}
            self._act_pad_cache = cache
        key = (
            "torch" if framework is torch else "numpy",
            tuple(int(s) for s in leading_shape),
            str(reference.dtype),
            str(getattr(reference, "device", "cpu")),
            float(act_pad_value),
        )
        pad = cache.get(key)
        if pad is None:
            pad = framework.full(
                (*leading_shape, self.singlestep_act_len, 1), fill_value=act_pad_value
            )
            if framework is torch:
                # R-A5b: reproduce the legacy behaviour EXACTLY -- device move only, NO dtype cast.
                pad = pad.to(device=reference.device)
            if len(cache) >= 8:
                # R-A5a: cap the cache (FIFO eviction) so a ragged batch-shape stream cannot
                # grow it unboundedly; the shape set is effectively {train, eval, 1} in practice.
                cache.pop(next(iter(cache)))
            cache[key] = pad
        return pad

    def unflaten_multistep_composed_array(
        self,
        multistep_composed_array: Union[np.ndarray, torch.Tensor],
        is_model_output: bool = True,
        act_pad_value: float = 0.0,
    ) -> Union[np.ndarray, torch.Tensor]:
        # (PRIORITY) ToDo: refactor casse horizon_len=1, it should output the action dimension with
        #   pad. It curretly output the observation dimension only which is counter intuitive,
        #   error prone and reduce the flexibility of the method.
        # (NICE TO HAVE) ToDo: implement test case (its indirectly tested for now) (ref task RLRP-220)
        """Reshape the composed multistep flatten dimension array such that multistep has its own
        dimension and the last actions timesteps is padded with arbitrary values i.e.

            (..., O[1:Do]_1  + ... + O[1:Do]_MS + A[1:Da]_1 + ... + A[1:Da]_MS-1) --> (..., O+A, MS)

        with
            - O the observation dimensions, Do the number of observation dimensions,
            - A the action dimensions, Da the number of action dimensions,
            - and MS the multistep len (History or horizon).

        Note
            - that the action dimensions are expected to have one timestep less than the
              observation dimensions.
            - casse horizon_len=1 only output the observation dimension (TODO refactor)

        :param multistep_composed_array: The flatted composed multistep array of shape
            (..., O[1:Do]_1  + ... + O[1:Do]_MS + A[1:Da]_1 + ... + A[1:Da]_MS-1) with action
            dimensions missing the last timesteps.
        :param is_model_output: Set to False if composed_multistep_array is a model input.
        :param act_pad_value: Arbitrary value to fill the action dimensions last timesteps.
        :return: The unflatted composed multistep array of shape be (..., F, MS) with all action
            dimensions padded to the same timestemps length as observation dimensions.
        """
        # RLRP-824 (FR10): the composed OUTPUT window is ``W = output_window_len`` obs blocks and
        # the composed INPUT window is ``H = history_len`` obs blocks. Both widths are derived
        # from the canonical layout formulas (``obs*MS + act*(MS-1)`` for an output, ``in_size``
        # for an input) -- the legacy ``in_size == out_size + Da`` identity holds iff ``W == H``.
        expected_width = (
            self.out_size
            if is_model_output
            else compute_multistep_model_in_size(
                self.singlestep_obs_len, self.singlestep_act_len, self.history_len
            )
        )
        assert multistep_composed_array.shape[-1] == expected_width, (
            f"composed-observation array width {multistep_composed_array.shape[-1]} != "
            f"{expected_width} expected for a model {'output' if is_model_output else 'input'} "
            f"(history_len={self.history_len}, output_window_len={self.output_window_len}, "
            f"singlestep_obs_len={self.singlestep_obs_len}, "
            f"singlestep_act_len={self.singlestep_act_len})"
        )

        framework = np
        if isinstance(multistep_composed_array, torch.Tensor):
            framework = torch

        original_shape = multistep_composed_array.shape

        obs_feature = self.extract_obs_features_from_multistep_composed_array(
            multistep_composed_array, is_model_output
        )

        if self.horizon_len > 1:
            act_feature = self.extract_act_features_from_multistep_composed_array(
                multistep_composed_array, is_model_output
            )
            if is_model_output:
                # A5 (RLRP-783): the pad is a constant for a given (framework, leading shape,
                # dtype, device, fill value) tuple; build it once and reuse it instead of
                # rebuilding on every call. Bit-exact -- see :meth:`_cached_act_pad`.
                act_pad = self._cached_act_pad(
                    framework,
                    original_shape[:-1],
                    act_pad_value,
                    multistep_composed_array,
                )

                padded_act_feature = framework.concatenate(
                    [
                        act_feature,
                        act_pad,
                    ],
                    axis=-1,
                )
            else:
                # RLRP-824 (FR10): a composed model INPUT carries ``H`` action steps (one per obs
                # block, ``a_t`` included), so there is no missing trailing step to pad.
                padded_act_feature = act_feature

            multistep_composed_array = framework.concatenate(
                [obs_feature, padded_act_feature], axis=-2
            )
        else:
            multistep_composed_array = obs_feature

        # .... Sanity check .......................................................................
        if self.horizon_len > 1:
            composed_obs_len = self.singlestep_obs_len + self.singlestep_act_len
        else:
            composed_obs_len = self.singlestep_obs_len

        expected_shape = (
            *original_shape[:-1],
            composed_obs_len,
            self.composed_window_len(is_model_output),
        )

        assert (
            multistep_composed_array.shape == expected_shape
        ), f"{multistep_composed_array.shape=} != {expected_shape=}"
        return multistep_composed_array

    def _extract_obs_features_from_multistep_composed_model_output(
        self, multistep_model_preds: Union[np.ndarray, torch.Tensor], vectorized: bool
    ) -> Union[np.ndarray, torch.Tensor]:
        return self._extract_obs_features_from_multistep_composed(
            multistep_model_preds, is_model_output=True, vectorized=vectorized
        )

    def _extract_obs_features_from_multistep_composed(
        self,
        multistep_composed_array: Union[np.ndarray, torch.Tensor],
        is_model_output: bool,
        vectorized: bool,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Obs-block extraction dispatching the multistep axis on
        :meth:`composed_window_len` (``W`` for an output, ``H`` for an input; RLRP-824 FR10).
        Identical to the historical output-only implementation when ``W == H``."""
        stop_idx = self.singlestep_obs_len * self.composed_window_len(is_model_output)
        feature = extract_multistep_features_from_array(
            x=multistep_composed_array,
            start_idx=0,
            stop_idx=stop_idx,
            single_step_features_size=self.singlestep_obs_len,
            vectorized=vectorized,
        )

        original_shape = multistep_composed_array.shape
        assert (
            original_shape[:-1] == feature.shape[:-2]
        ), f"{original_shape[:-1]=} != {feature.shape[:-2]=}"
        assert (
            feature.shape[-2] == self.singlestep_obs_len
        ), f"{feature.shape[-2]=} != {self.singlestep_obs_len=}"

        return feature

    def extract_obs_features_from_multistep_composed_array(
        self,
        multistep_model_input: Union[np.ndarray, torch.Tensor],
        is_model_output: bool = True,
        vectorized: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Utility for extracting observation features from an array of multistep input

            (..., O[1:D]_1  + ... + O[1:D]_MS) --> (..., O, MS)

        with O[1:D]=observation dimension len and MS=multistep len.

        :param multistep_model_input: The multistep input array.
        :param is_model_output: Set to False if composed_multistep_array is a model input. The
         multistep axis is ``output_window_len`` (``W``) for an output and ``history_len``
         (``H``) for an input (RLRP-824 FR10); both coincide for every legacy family.
        :param vectorized: use vectorize implementation
        :return: An observation feature array of shape
         (trajectory_len, feature_size, multistep_lens)
        """
        return self._extract_obs_features_from_multistep_composed(
            multistep_model_input, is_model_output=is_model_output, vectorized=vectorized
        )

    def _extract_act_features_from_multistep_composed_model_input(
        self, multistep_model_input: Union[np.ndarray, torch.Tensor], vectorized: bool
    ) -> Union[np.ndarray, torch.Tensor]:
        # RLRP-824: the INPUT act block starts after the ``H`` obs blocks (NOT after the ``W``
        # output obs blocks); identical when ``W == H``.
        feature = extract_multistep_features_from_array(
            x=multistep_model_input,
            start_idx=self.singlestep_obs_len * self.history_len,
            stop_idx=self.in_size,
            single_step_features_size=self.singlestep_act_len,
            vectorized=vectorized,
        )
        original_shape = multistep_model_input.shape
        assert (
            original_shape[:-1] == feature.shape[:-2]
        ), f"{original_shape[:-1]=} != {feature.shape[:-2]=}"
        assert (
            feature.shape[-2] == self.singlestep_act_len
        ), f"{feature.shape[-2]=} != {self.singlestep_act_len=}"

        return feature

    def extract_act_features_from_multistep_composed_array(
        self,
        multistep_model_input: Union[np.ndarray, torch.Tensor],
        is_model_output: int = True,
        vectorized: bool = True,
    ) -> Optional[Union[np.ndarray, torch.Tensor]]:
        # ToDo: implement test case
        """Utility for extracting action features from an array of multistep model prediction

            (..., A[1:D]_1 + ... + A[1:Da]_MS-1) --> (..., A, MS)

        with A[1:D]=action dimension len and MS=multistep len.

        :param multistep_model_input: The multistep model input or model prediction array
        :param is_model_output: Set to False if composed_multistep_array is a model input
        :param vectorized: use vectorize implementation
        :return: An action feature array of shape (trajectory_len, features_size, multistep_len).
         Return None if is_model_output=True and prediction horizon is set to 1.
        """
        if is_model_output:
            return self._extract_act_features_from_multistep_composed_model_output(
                multistep_model_input, vectorized
            )
        else:
            return self._extract_act_features_from_multistep_composed_model_input(
                multistep_model_input, vectorized
            )

    def _extract_act_features_from_multistep_composed_model_output(
        self, multistep_model_preds: Union[np.ndarray, torch.Tensor], vectorized: bool
    ) -> Optional[Union[np.ndarray, torch.Tensor]]:
        if self.horizon_len > 1:
            feature = extract_multistep_features_from_array(
                x=multistep_model_preds,
                start_idx=self._compose_next_obs_multistep_obs_horizon_slice.stop,
                stop_idx=self.out_size,
                single_step_features_size=self._compose_next_obs_multistep_act_horizon_slice.step,
                vectorized=vectorized,
            )
            original_shape = multistep_model_preds.shape
            assert (
                original_shape[:-1] == feature.shape[:-2]
            ), f"{original_shape[:-1]=} != {feature.shape[:-2]=}"
            assert (
                feature.shape[-2] == self.singlestep_act_len
            ), f"{feature.shape[-2]=} != {self.singlestep_act_len=}"

            return feature
        else:
            return None

    def _save_state_dict(self) -> dict[str, list[int] | dict[str, Any] | Any]:
        model_dict = super()._save_state_dict()

        model_dict["model_config"]["singlestep_obs_len"] = self.singlestep_obs_len
        model_dict["model_config"]["singlestep_act_len"] = self.singlestep_act_len
        model_dict["model_config"]["history_len"] = self.history_len
        model_dict["model_config"]["horizon_len"] = self.horizon_len

        return model_dict

    _MULTISTEP_REQUIRED_CONFIG_KEYS: Tuple[str, ...] = (
        "singlestep_obs_len",
        "singlestep_act_len",
        "history_len",
        "horizon_len",
    )

    def _load_state_dict(self, model_dict):
        # .... Validate required keys before mutating any state ...................................
        if "model_config" not in model_dict:
            raise KeyError(
                f"{type(self).__name__}._load_state_dict: missing 'model_config' "
                f"entry in model_dict."
            )
        model_config = model_dict["model_config"]
        missing = [
            k for k in self._MULTISTEP_REQUIRED_CONFIG_KEYS if k not in model_config
        ]
        if missing:
            raise KeyError(
                f"{type(self).__name__}._load_state_dict: missing keys in "
                f"model_dict['model_config']: {missing}"
            )

        super()._load_state_dict(model_dict)

        singlestep_obs_len = model_config["singlestep_obs_len"]
        singlestep_act_len = model_config["singlestep_act_len"]
        history_len = model_config["history_len"]
        horizon_len = model_config["horizon_len"]

        self.singlestep_obs_len = singlestep_obs_len
        self.singlestep_act_len = singlestep_act_len
        self.history_len = history_len
        self.horizon_len = horizon_len

        # .... Re-validate invariants enforced by __init__ ........................................
        self._validate_multistep_window_lens()

        # .... Recompute derived state (slices + deploy-head adapter) .............................
        self._build_multistep_derived_state()
        return None

    def _build_multistep_derived_state(self) -> None:
        """Recompute the multistep slice attributes and rebuild the deploy-head adapter
        from the current values of ``singlestep_obs_len``, ``singlestep_act_len``,
        ``history_len`` and ``horizon_len``.

        Mirrors the derived-state setup performed in ``__init__`` so that
        ``_load_state_dict`` does not leave the model with stale slices/adapters
        when loaded values differ from those used at construction time.

        RLRP-824: every OUTPUT slice is derived from the composed output window
        ``W = output_window_len`` (``== history_len`` for every legacy family, hence bit-exact):
        the forecast obs blocks are ``[Do*(W-F), Do*W)`` and the forecast act blocks start at
        ``Do*W + Da*(W-F)``.
        """
        singlestep_obs_len = self.singlestep_obs_len
        singlestep_act_len = self.singlestep_act_len
        output_window_len = self.output_window_len
        horizon_len = self.horizon_len

        compose_next_obs_multistep_obs_horizon_start_idx = singlestep_obs_len * (
            output_window_len - horizon_len
        )
        compose_next_obs_multistep_obs_horizon_end_idx = (
            singlestep_obs_len * output_window_len
        )
        compose_next_obs_multistep_act_horizon_start_idx = (
            singlestep_obs_len * output_window_len
            + (singlestep_act_len * (output_window_len - horizon_len))
        )

        assert (
            compose_next_obs_multistep_act_horizon_start_idx <= self.out_size
        ), f"{compose_next_obs_multistep_act_horizon_start_idx} not <= {self.out_size}"

        self._compose_next_obs_multistep_obs_horizon_slice = slice(
            compose_next_obs_multistep_obs_horizon_start_idx,
            compose_next_obs_multistep_obs_horizon_end_idx,
            singlestep_obs_len,
        )
        self._compose_next_obs_multistep_act_horizon_slice = slice(
            compose_next_obs_multistep_act_horizon_start_idx,
            self.out_size,
            singlestep_act_len,
        )
        self._compose_next_multistep_obs_to_next_singlestep_obs_slice = slice(
            compose_next_obs_multistep_obs_horizon_start_idx,
            compose_next_obs_multistep_obs_horizon_start_idx + singlestep_obs_len,
        )

        adapter = MultistepObservationToSinglestepObservationAdapter(
            array_in_len=self.out_size,
            array_out_len=singlestep_obs_len,
            multistep_obs_to_singlestep_obs_slice=self._compose_next_multistep_obs_to_next_singlestep_obs_slice,
        )
        self.set_deploy_head_multistep_to_singlestep_adapter(adapter)
        return None
