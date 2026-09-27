# coding=utf-8
import math
import pathlib
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import omegaconf
from mbrl.models import Ensemble, truncated_normal_init
from mbrl.types import ModelInput
from mbrl.util.normalization import (
    ZScoreNormalizer,
    SoftWinsorizedNormalizer,
    QuantileNormalizer,
)
from torch import Tensor, nn as nn
from torch.nn import functional as F
from torch import distributions as dist

from tools.console_tools.message import consol_msg_universal_one_liner
from tools.mbrl_lib_tools.models.prediction_statistics import SENTINEL_LOGVAR
from tools.multistep_tools.models.exponential_family_mlp_utils import (
    LogvarBoundLayer,
    create_activation_,
    create_linear_layer_,
    create_logvar_bound_layer,
    create_layer_bloc_factory,
    fast_distribution_mode_approximation,
    layer_bloc_signature_from_resolved,
    logistic_distribution,
    make_layer_bloc_seq,
    negative_loglikelihood_loss__explicit,
    resolve_layer_bloc_settings,
    zero_init_residual_blocks_,
    RESIDUAL_FORMS,
)
from tools.multistep_tools.models.utils import (
    reduce_deterministic_compose_loss,
    reduce_probabilistic_compose_loss,
)
from tools.multistep_tools.models.precision_bounds import (
    std_floor,
    variance_floor,
    logvar_safe_max,
)
from tools.multistep_tools.models.feature_geometry_loss_mixin import (
    FeatureGeometryLossMixin,
)
from tools.multistep_tools.models.orientation_config import (
    resolve_feature_geometry_cfg,
    resolve_internal_orientation_cfg,
)
from tools.multistep_tools.models.training_frame_mixin import (
    TrainingFrameMixin,
    _validate_training_frame,
)
from tools.feature_handling_tools.feature_spec import InternalOrientationRep
from tools.feature_handling_tools.orientation_heads import (
    orientation_slot_bases,
    attitude_slot_index_tensor,
    DEFAULT_EXTERNAL_ORIENTATION_WIDTH,
    decode_internal_rep_to_orientation_slot,
    encode_orientation_slot_to_internal_rep,
    gather_attitude_slots,
    internal_rep_in_width,
    internal_rep_out_width,
    internal_rep_tangent_dim,
    rotation_tangent_nll,
    s2_tangent_nll,
    scatter_attitude_slots,
    split_attitude_and_rest_index_tensors,
)


class ExponentialFamilyMLP(FeatureGeometryLossMixin, TrainingFrameMixin, Ensemble):
    # RLRP-736 §4A-B (probabilistic orientation wiring, Phase 1). Per-family
    # opt-in flag: when ``False`` (default) an ACTIVE non-quaternion rep on a
    # PROBABILISTIC model still fails loud in ``_setup_orientation_rep`` (the
    # by-construction tangent-NLL head is only supported once a family wires the
    # asymmetric mean/logvar head + split loss). MTM-Pro sets this ``True``.
    _supports_probabilistic_by_construction_orientation: bool = False
    # A7 (RLRP-788): class-level root of the diagnostic ``meta`` collection master switch
    # so ``self._enable_meta_collection`` resolves for EVERY MTM-Pro writer family
    # (all descend from this class) WITHOUT depending on the owned ``mbrl.models.Model``
    # submodule being the direct base at import time. Default ``True`` -> today's
    # behaviour; driven by ``pipeline.tensorboard.enable_meta_collection`` via the
    # ``set_meta_collection_enabled`` setup seam. Permanent diagnostic default. Introduced
    # by action ``A7`` of the RLRC meta-collection kill-switch `.junie` plan
    # (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
    _enable_meta_collection: bool = True
    mae_loss: bool = True
    distribution_name: str
    hidden_layers: nn.Sequential
    mean_and_logvar: nn.Sequential
    mean_layer: nn.Sequential
    logvar_layer: nn.Sequential
    _one_d_trj_model_input_normalizer: Optional[
        Union[ZScoreNormalizer, SoftWinsorizedNormalizer, QuantileNormalizer]
    ]

    # (Priority) done: assessment hypothesis-121 -> smaller log var min initialisation
    # _LOGVAR_MIN_BOUND_INIT = math.log(1e-3) # original version
    # Learnable log-variance bound initialization:
    #   MIN_BOUND_INIT = log(1e-10) ≈ -23.03  →  variance floor 1e-10, std ≈ 1e-5
    #   MAX_BOUND_INIT = log(1.7)   ≈   0.53  →  variance ceil  1.7,  std ≈ 1.3
    #   Both are safe in float32 (representable range and exp() does not underflow).
    # dtype note: These are semantic bounds (modeling choices), NOT numerical precision
    #   guards. They do NOT need dtype-dependent variants: the same variance floor/ceiling
    #   applies regardless of float32 vs float64. Relative representation error < 3e-8.
    _LOGVAR_MIN_BOUND_INIT = math.log(1e-10)
    _LOGVAR_MAX_BOUND_INIT = math.log(1.7)
    # Near-deterministic logvar fill value for models without learned variance.
    # Using 1e-30 (not 1e-290) so that exp(logvar) remains representable in float32
    # (float32 tiny ≈ 1.18e-38; exp(log(1e-30)) = 1e-30 > tiny).
    # dtype note: same value for float32 and float64 — 1e-30 variance is already far
    #   below any physically meaningful variance for RL/robotics applications.
    # RLRP-761 S10.4: single canonical source (see prediction_statistics.SENTINEL_LOGVAR).
    _LOGVAR_MIN_LIMIT = SENTINEL_LOGVAR

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        mae_loss: bool = True,
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        deterministic: bool = False,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        logvar_bound_grad_clip: Optional[float] = None,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        distribution_name: str = "gaussian",
        model_use_double_precision: bool = False,
        dropout: float = 0.0,
        ms_head_dropout: float = 0.0,
        ms_head_num_layers: int = 1,
        residual_form: str = "identity_v2",
        # RLRP-768: per-region layer-bloc selector (superset of `residual_form`).
        # ``None`` -> use the per-region defaults (:data:`DEFAULT_LAYER_BLOC_TYPE_BY_REGION`);
        # a mapping ``{region: {type, layer_norm, dropout}}`` overrides them.
        layer_bloc: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        # RLRP-736 nested config groups (replace the former flat
        # ``feature_geometry_loss_weight`` / ``internal_orientation_rep`` /
        # ``orientation_{input,output}_slots`` /
        # ``orientation_tangent_nll_right_jacobian`` kwargs).
        feature_geometry: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        internal_orientation: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        # RLRP-736 orientation-slots auto-inference: the attitude slot(s) no longer
        # travel inside the (user-facing) ``internal_orientation`` group. They are
        # derived internally (setup seam -> ``MultiStepMLP`` composed-window
        # expansion) and handed to this base on a DEDICATED, EXPLICITLY-NAMED
        # channel. Already-expanded (composed-window) slot indices; ``None`` -> the
        # head stays neutral / bit-exact OFF.
        orientation_input_slots: Optional[Sequence[int]] = None,
        orientation_output_slots: Optional[Sequence[int]] = None,
        singlestep_obs_len: Optional[int] = None,
        description: Optional[str] = None,
    ):
        super().__init__(
            ensemble_size, device, propagation_method, deterministic=deterministic
        )

        # RLRP-736: unpack the nested ``feature_geometry`` / ``internal_orientation``
        # config groups into the flat locals used by the rest of this constructor.
        _fg = resolve_feature_geometry_cfg(feature_geometry)
        _io = resolve_internal_orientation_cfg(internal_orientation)
        feature_geometry_loss_weight = _fg["loss_weight"]
        feature_geometry_loss_objective = _fg["loss_objective"]
        # RLRP-751 (task T2): STATIC vs AUTO weighting toggle for the geometry term.
        feature_geometry_auto_weighting = _fg["auto_weighting"]
        internal_orientation_rep = _io["representation"]
        orientation_tangent_nll_right_jacobian = _io["tangent_nll_right_jacobian"]
        # Tri-state manifold-aware NLL lever (§A.0). Introduced by item A of the
        # Quaternion manifold upgrade + memoryless ``w >= 0`` removal `.junie` plan
        # (`rlrp-744-quaternion-manifold-upgrade-and-memoryless-wge0-removal-plan-20260720.md`).
        orientation_manifold_aware_nll = _io["manifold_aware_nll"]
        # RLRP-744 item E: the resolved ``enforce_continuity`` flag (default True).
        # Threaded into the model so it can ALSO gate the model-side
        # reference-relative attitude alignment (not just the ingestion pass) via
        # ``_quaternion_ar_continuity_active`` (Quaternion manifold upgrade +
        # memoryless ``w >= 0`` removal `.junie` plan
        # (`rlrp-744-quaternion-manifold-upgrade-and-memoryless-wge0-removal-plan-20260720.md`)).
        orientation_enforce_continuity = _io["enforce_continuity"]
        # RLRP-736: ``orientation_{input,output}_slots`` are NO LONGER read from the
        # ``internal_orientation`` group (the group validates only ``representation``
        # + ``tangent_nll_right_jacobian``). They arrive on the explicit ctor channel
        # above (already expanded to the composed window by ``MultiStepMLP``).

        # RLRP-718 (follow-up): optional per-experiment gradient clip applied element-wise to the
        # learnable logvar-bound parameters (``logvar_min``/``logvar_max``). ``None`` -> disabled
        # (default, no behaviour change). Stored on ``self`` BEFORE ``_build_network`` so every
        # ``create_logvar_bound_layer`` call site (this base plus subclass ``build_network_post``
        # hooks) can read it via ``self.logvar_bound_grad_clip``.
        self.logvar_bound_grad_clip = logvar_bound_grad_clip

        # Residual block variant (back-compat switch). Default `identity_v2` is the
        # canonical identity-preserving block; `legacy_preact_wrapped` reproduces
        # the historical `act(x + Wx + b)` behavior for ablation / old checkpoints.
        assert residual_form in RESIDUAL_FORMS, (
            f"residual_form must be one of {RESIDUAL_FORMS}, got {residual_form!r}"
        )
        self.residual_form = residual_form

        # RLRP-768: resolve the per-region layer-bloc settings (type / layer_norm /
        # dropout) with the `residual_form` alias precedence (§3.5.b). The raw
        # config is kept for reference; the resolved mapping drives every
        # `_build_network` call site and the checkpoint-topology signature.
        self.layer_bloc = layer_bloc
        self._layer_bloc_resolved = resolve_layer_bloc_settings(
            layer_bloc, residual_form
        )

        # Initialise normalizer handles as instance attributes so that
        # nn.Module.__setattr__/``__getattr__`` can later store and retrieve
        # Normalizer objects through ``_modules`` without being shadowed by a
        # class-level ``None`` default (see Python MRO + nn.Module interaction).
        # RLRP-684 (Amendment A1): only the asymmetric ``"standard"`` Z-score path
        # propagates a normalizer handle into the wrapped model (it is needed for the
        # AR denormalize->shift->renormalize round-trip).  The block facades
        # (``"standard_symmetric"`` / ``"winsorized"`` / ``"quantile"``) keep the AR
        # loop entirely in normalized space, so the former obs/act handles were
        # removed.
        self._one_d_trj_model_input_normalizer: Optional[ZScoreNormalizer] = (
            None  # ABC type
        )

        # RLRP-736 S1.3 / mixin refactor: optional, opt-in per-feature geometry
        # loss term (e.g. a quaternion geodesic penalty). The whole concern now
        # lives in :class:`FeatureGeometryLossMixin`; here we only initialise its
        # state. ``feature_geometry_loss_weight`` is surfaced as a model constructor
        # parameter so a run can enable the term purely from the model Hydra
        # config. Neutral / OFF by default (``None`` handler, weight 0.0), so
        # ``loss`` is BIT-EXACT with the legacy path unless a handler is
        # registered AND the effective weight is non-zero.
        self._init_feature_geometry_loss(
            feature_geometry_loss_weight,
            feature_geometry_loss_objective,
            feature_geometry_auto_weighting,
        )

        # RLRP-758 (task T8b): persist the model-internal velocity training frame
        # with the checkpoint. Neutral default (``body``); the run-selected frame
        # is attached post-construction at the setup seam via
        # ``set_training_frame`` (mirrors ``set_feature_handler``), so no new ctor
        # kwarg has to thread through the whole model MRO.
        self._init_training_frame()

        # RLRP-736 §S2.2-B (Variant B / PyPose) — model-internal, by-construction
        # rotation representation. NEUTRAL / OFF by default: ``quaternion`` +
        # no slots => the encoder/head are inactive, the trunk is built at the
        # external ``in_size``/``out_size`` and ``forward`` applies no transform,
        # so the model is BIT-EXACT with the legacy path. When a non-quaternion
        # ``internal_orientation_rep`` is selected AND attitude slot(s) are
        # supplied, each 4-D attitude slot is encoded to a continuous internal
        # representation before the trunk and the trunk mean output is decoded
        # back to a unit quaternion by construction (Geist et al. 2024; Chen et
        # al. 2023). The external I/O stay 4-D quaternions in the same slot (the
        # symmetric external contract; report §4.4). ``self.in_size`` /
        # ``self.out_size`` remain the EXTERNAL widths; the trunk is built at the
        # rep-expanded widths.
        # RLRP-736 §4A-B: sub-mode selector for the probabilistic attitude NLL.
        # ``False`` (default) => plain tangent-Gaussian NLL (plan §4A-B default);
        # ``True`` => add the Forster et al. 2016 right-Jacobian volume correction
        # so the tangent density is a proper concentrated SO(3) density. Read by
        # :meth:`_orientation_split_nll`.
        self._tangent_nll_right_jac = bool(orientation_tangent_nll_right_jacobian)
        self._setup_orientation_rep(
            internal_orientation_rep,
            orientation_input_slots,
            orientation_output_slots,
            in_size=in_size,
            out_size=out_size,
            deterministic=deterministic,
            singlestep_obs_len=singlestep_obs_len,
            manifold_aware_nll=orientation_manifold_aware_nll,
            enforce_continuity=orientation_enforce_continuity,
        )

        self.in_size = in_size
        self.out_size = out_size
        self.hid_size = hid_size
        self.description = description
        self.dropout = dropout

        assert ms_head_num_layers >= 0
        self.ms_head_num_layers = ms_head_num_layers
        self.ms_head_dropout = ms_head_dropout

        self._build_network(
            num_layers=num_layers,
            in_size=self._trunk_in_size,
            hid_size=hid_size,
            out_size=self._trunk_out_size,
            ensemble_size=ensemble_size,
            activation_fn_cfg=activation_fn_cfg,
            deterministic=deterministic,
            learn_logvar_bounds=learn_logvar_bounds,
            dropout=dropout,
        )

        self.elite_models: List[int] = None

        # .... Set Loss type ......................................................................
        self.mae_loss = mae_loss
        if mae_loss:
            consol_msg_universal_one_liner("Using MAE loss")
        else:
            consol_msg_universal_one_liner("Using MSE loss")

        # .... Distribution .......................................................................
        self.distribution_name = distribution_name
        if not self.deterministic:
            consol_msg_universal_one_liner(f"Using {distribution_name} distribution")

        self.model_use_double_precision = model_use_double_precision
        if model_use_double_precision:
            consol_msg_universal_one_liner("Model double precision enabled")
            self.to(dtype=torch.double)
            self.model_dtype = torch.double
        else:
            try:
                self.model_dtype = self.get_submodule("hidden_layers")[0][
                    0
                ].weight.dtype
            except AttributeError:
                self.model_dtype = torch.float32

    def _setup_orientation_rep(
        self,
        internal_orientation_rep: str,
        orientation_input_slots: Optional[Sequence[int]],
        orientation_output_slots: Optional[Sequence[int]],
        in_size: int,
        out_size: int,
        deterministic: bool,
        singlestep_obs_len: Optional[int] = None,
        manifold_aware_nll: Optional[bool] = None,
        enforce_continuity: bool = True,
    ) -> None:
        """Resolve the RLRP-736 §S2.2-B by-construction rotation-rep wiring.

        Computes the (rep-expanded) *trunk* input/output widths and stores the
        attitude-slot bookkeeping used by :meth:`_apply_orientation_input_encoding`
        / :meth:`_apply_orientation_output_decoding`. NEUTRAL by default
        (``quaternion`` or no slots): the trunk widths equal the external widths
        and no transform is applied, so the model is bit-exact.

        :param internal_orientation_rep: One of :class:`InternalOrientationRep`
            values (``quaternion`` | ``sixd`` | ``nine_d_svd`` | ``so3_relative``).
        :param orientation_input_slots: Start indices (external input layout) of
            each 4-D attitude quaternion slot to encode, or ``None``.
        :param orientation_output_slots: Start indices (external output layout) of
            each 4-D attitude quaternion slot to decode, or ``None``.
        :param in_size: External input width.
        :param out_size: External output width.
        :param deterministic: Whether the model is deterministic (the by-construction
            head is currently supported only on the deterministic mean path; the
            probabilistic tangent-NLL head is the documented compute-gated follow-up).
        :param singlestep_obs_len: Width of a single-step observation block, used to
            size the DEDICATED single-step (SS / deploy) head of architectures B/C
            (:class:`WeightedMultiStepDualHeadMLP` and the encoder->decoder AR
            family). When provided AND the wiring is active, the SS-head attitude
            base slot(s) (those output slots that fall within the FIRST single-step
            obs block, i.e. ``s < singlestep_obs_len``) are tracked in
            ``_ss_ori_out_slots`` and the SS head's final layer width
            ``_ss_trunk_out_len`` is rep-expanded so
            :meth:`_apply_ss_orientation_output_decoding` can map it back to a 4-D
            unit quaternion. ``None`` (or architecture-A families that mask the
            already-decoded MS mean) leaves the SS bookkeeping neutral. RLRP-736
            bespoke-forward orientation-wiring plan (2026-07-13).
        :param manifold_aware_nll: Tri-state lever (§A.0 of the RLRP-744 `.junie`
            plan,
            ``rlrp-744-quaternion-manifold-upgrade-and-memoryless-wge0-removal-plan-20260720.md``).
            Resolved into ``self._manifold_aware_nll``, the narrow gate for the
            so3-tangent NLL + by-construction unit decode + 3-D logvar-head
            narrowing — orthogonal to ``_internal_orientation_rep_active`` (which keeps owning
            the rep transform + trunk-width expansion): ``None`` (auto) ->
            ``self._internal_orientation_rep_active`` (bit-exact today); ``True`` -> force ON
            (plain-``quaternion`` upgrade); ``False`` -> force OFF (matrix-rep
            ablation). ``quaternion_legacy`` always resolves ``False``.
        :param enforce_continuity: RLRP-744 item E. The resolved
            ``ms_model.internal_orientation.enforce_continuity`` flag (default
            ``True``). Stored as ``self._orientation_enforce_continuity`` and
            ANDed into ``_quaternion_ar_continuity_active`` so the SAME config key
            that gates the ingestion continuity pass ALSO gates the model-side
            reference-relative attitude alignment (AR splice / deploy /
            sampling-free / resampled / trsf + the DH residual). Orthogonal to the
            in-graph unit decode (``_unit_decode``) and the conversion-internal
            ``w >= 0`` in ``quaternion_to_axis_angle`` (both excluded by design).
        """
        # Local import: ``env_handlers`` owns the resolver-side layout value object
        # and pulls in the feature-handling stack; importing it at module scope from
        # a model would couple the two packages at import time.
        from tools.feature_handling_tools.env_handlers import orientation_slot_layout

        # RLRP-744 item E — store the enforce-continuity gate for the model-side
        # reference-relative alignment predicate (``_quaternion_ar_continuity_active``).
        self._orientation_enforce_continuity = bool(enforce_continuity)
        self._orientation_rep = InternalOrientationRep(internal_orientation_rep)
        self._ori_in_slots = (
            tuple(int(s) for s in orientation_input_slots)
            if orientation_input_slots
            else ()
        )
        self._ori_out_slots = (
            tuple(int(s) for s in orientation_output_slots)
            if orientation_output_slots
            else ()
        )
        self._internal_orientation_rep_active = self._orientation_rep not in (
            InternalOrientationRep.QUATERNION,
            InternalOrientationRep.QUATERNIONLEGACY,
        ) and bool(self._ori_in_slots or self._ori_out_slots)

        # RLRP-744 item A (§A.0) — tri-state resolution of the manifold-aware NLL
        # lever (single source of truth, computed BEFORE the early-return below).
        if self._orientation_rep is InternalOrientationRep.QUATERNIONLEGACY:
            self._manifold_aware_nll = False  # always OFF, byte-exact ablation
        elif manifold_aware_nll is None:
            self._manifold_aware_nll = self._internal_orientation_rep_active  # auto == today
        elif manifold_aware_nll:
            self._manifold_aware_nll = bool(self._ori_in_slots or self._ori_out_slots)
        else:
            self._manifold_aware_nll = False

        self._deterministic_rep = bool(deterministic)
        # Neutral / OFF: trunk widths == external widths, bit-exact.
        self._trunk_in_size = in_size
        self._trunk_out_size = out_size
        # RLRP-736 §4A-B: probabilistic LOGVAR-head widths. Neutral OFF => equal to
        # the external widths (the head is symmetric ⇒ bit-exact). When active on a
        # probabilistic model, each attitude slot's logvar NARROWS to the block's
        # tangent (``TANGENT_DIM - external_width`` per slot: -1 for the so3
        # tangent of the quaternion block, Forster et al. 2016).
        self._logvar_trunk_out_size = out_size
        self._ss_logvar_trunk_out_len: Optional[int] = singlestep_obs_len
        # Dedicated single-step (SS / deploy) head bookkeeping (architectures B/C).
        # Neutral OFF: no SS attitude slots, SS head width == singlestep_obs_len,
        # no per-block input expansion.
        self._ss_ori_out_slots: Tuple[int, ...] = ()
        self._ss_trunk_out_len: Optional[int] = singlestep_obs_len
        self._ss_ori_in_slots: Tuple[int, ...] = ()
        self._ss_block_in_extra: int = 0
        # RLRP-796 ``D-793-3-A`` / ``D-796-S1-6-A`` — chart base point of the
        # ``S2_TANGENT`` decode. The tangent is expressed at the PREVIOUS step's
        # direction, so the decode needs a reference the trunk output alone cannot
        # supply. It is captured on the INPUT side (the last history orientation
        # slot of the composed window) by
        # :meth:`_apply_orientation_input_encoding`, which runs on every forward
        # BEFORE the trunk, and consumed by both decode paths. Deliberately a
        # per-forward scratch value, not a buffer: it is derived from the current
        # input, never carried across calls, and stays ``None`` for every other rep
        # so the quaternion route is untouched.
        self._ori_decode_reference: Optional[torch.Tensor] = None
        # RLRP-796 Stage 1 (``D-796-S1-4-A``) — the orientation width arithmetic
        # below is DERIVED, never literal. ``_ori_external_width`` is the EXTERNAL
        # slot width of the orientation block the slots address (4 = attitude
        # quaternion on S^3, 3 = gravity direction on S^2). It rides along on the
        # resolved ``OrientationSlotLayout`` threaded through the existing
        # ``orientation_singlestep_slots`` kwarg; a LEGACY bare tuple (the
        # pre-Stage-1 contract, and what the existing string-key test call sites
        # inject) normalizes to the quaternion block's 4, so every pre-Stage-1
        # result is unchanged.
        self._ori_external_width = int(
            orientation_slot_layout(
                orientation_output_slots or orientation_input_slots
            ).external_width
        )
        external_width = self._ori_external_width
        # Trunk in/out widths per slot may differ (an output-only tangent rep
        # narrows what the trunk EMITS while its encode stays absolute).
        ori_in_width = internal_rep_in_width(self._orientation_rep, external_width)
        ori_out_width = internal_rep_out_width(self._orientation_rep, external_width)
        # Tangent dimension of ONE slot's (log-)variance: 3 for so3 (quaternion
        # block, Forster et al. 2016), 2 for the S² gravity block.
        TANGENT_DIM = internal_rep_tangent_dim(self._orientation_rep, external_width)
        # RLRP-744 item C (rev.3b) — ALWAYS-ON in-graph unit-normalize decode on the
        # plain-``quaternion`` path. Introduced by item C of the Quaternion manifold
        # upgrade + memoryless ``w >= 0`` removal `.junie` plan
        # (``rlrp-744-quaternion-manifold-upgrade-and-memoryless-wge0-removal-plan-20260720.md``).
        # PRIVATE derived flag (NOT a config knob): ``quaternion`` with wired
        # attitude slots -> ON; ``quaternion_legacy`` -> OFF (raw, un-normalized
        # bit-exact baseline); matrix reps -> decode already mandatory via
        # ``_internal_orientation_rep_active`` (flag inert / False).
        #
        # RLRP-796 Stage 1 flag audit: this flag exists ONLY to force the decode on
        # for the otherwise-NEUTRAL ``quaternion`` rep, whose
        # ``_internal_orientation_rep_active`` is ``False``. Every S^2 rep is a
        # non-neutral rep, so it takes the ACTIVE branch below and the decode is
        # already mandatory there — the flag stays inert (``False``) for them, as it
        # does for the matrix reps. Deliberately NOT widened to ``S2_IDENTITY``:
        # doing so would make it reach the neutral early-return branch, whose
        # ``assert rep is QUATERNION`` correctly forbids that.
        self._unit_decode = (
            self._orientation_rep is InternalOrientationRep.QUATERNION
            and bool(self._ori_in_slots or self._ori_out_slots)
        )
        if not self._internal_orientation_rep_active:
            # RLRP-744 item A (§A.3 step 1) + item C — plain-``quaternion`` path.
            # Populate the attitude slot bookkeeping WITHOUT any trunk expansion
            # (the in/out widths equal ``external_width`` for a passthrough rep, so
            # every trunk width stays equal to the external width — asserted
            # below); narrow the probabilistic logvar head to the block's tangent
            # per slot ONLY under the manifold-aware NLL lever (item C decode-only
            # keeps the ambient external-width ``_base_nll`` / logvar head).
            if self._manifold_aware_nll or self._unit_decode:
                assert self._orientation_rep is InternalOrientationRep.QUATERNION
                assert ori_in_width == external_width
                assert ori_out_width == external_width
                if (
                    not deterministic
                    and not self._supports_probabilistic_by_construction_orientation
                ):
                    # Same per-family opt-in guard as the active matrix-rep path
                    # below: the probabilistic by-construction seam (decode +
                    # tangent-NLL) is wired per family (RLRP-736 §4A-B Phase 1:
                    # MTM-Pro only; RLRP-737 / RLRP-738 extend it).
                    raise NotImplementedError(
                        "The by-construction quaternion path (in-graph unit "
                        "decode / manifold_aware_nll) on a PROBABILISTIC model "
                        "requires the per-family orientation wiring (RLRP-736 "
                        "§4A-B Phase 1 covers MTM-Pro only; see RLRP-737 / "
                        "RLRP-738). Use representation: quaternion_legacy for "
                        "this family."
                    )
                assert self._trunk_in_size == in_size
                assert self._trunk_out_size == out_size
                if not deterministic and self._manifold_aware_nll:
                    self._logvar_trunk_out_size = out_size + len(
                        self._ori_out_slots
                    ) * (TANGENT_DIM - external_width)
                if singlestep_obs_len is not None and self._ori_out_slots:
                    self._ss_ori_out_slots = tuple(
                        s for s in self._ori_out_slots if s < singlestep_obs_len
                    )
                    # ``_ss_trunk_out_len`` unchanged (out delta == 0).
                    if not deterministic and self._manifold_aware_nll:
                        self._ss_logvar_trunk_out_len = singlestep_obs_len + len(
                            self._ss_ori_out_slots
                        ) * (TANGENT_DIM - external_width)
                if singlestep_obs_len is not None and self._ori_in_slots:
                    self._ss_ori_in_slots = tuple(
                        s for s in self._ori_in_slots if s < singlestep_obs_len
                    )
                    # ``_ss_block_in_extra`` stays 0 (no per-block input expansion).
            return

        if self._orientation_rep is InternalOrientationRep.SO3_RELATIVE:
            raise NotImplementedError(
                "internal_orientation_rep='so3_relative' is DEFERRED (RLRP-736): it "
                "requires per-step q_ref plumbing and is a poor fit for the "
                "adverse-condition large-rotation target. Use 'nine_d_svd' or 'sixd'."
            )
        if not deterministic and not self._supports_probabilistic_by_construction_orientation:
            # RLRP-736 §4A-B: the by-construction PROBABILISTIC head (quaternion
            # mean + 3-D so3 tangent (co)variance scored by ``rotation_tangent_nll``)
            # is wired per-family (the asymmetric mean/logvar head + split loss).
            # A family opts in by setting the class flag
            # ``_supports_probabilistic_by_construction_orientation = True`` once it
            # has wired the split-NLL seam (Phase 1: MTM-Pro only).
            #
            # DEFERRED (2026-07-13): the remaining deterministic-track families
            # (VAE ``vaev*`` + mixture ``weighted_ms2ss_mixture{,_siw_mp,_vi}``)
            # are probabilistic-only, so they also hit this guard on an active
            # rep. Their by-construction wiring is DEFERRED to RLRP-737 (not a
            # priority for the adverse-condition target).
            raise NotImplementedError(
                "By-construction orientation output for the PROBABILISTIC path "
                "(tangent-NLL head) is not wired for this family yet (RLRP-736 "
                "§4A-B Phase 1 covers MTM-Pro only). The VI family is Phase 2 "
                "(RLRP-738); the VAE / mixture families are DEFERRED to RLRP-737. "
                "Use 'quaternion' for these models."
            )

        # Each encoded input slot changes the width by ``ori_in_width -
        # external_width``; each decoded output slot by ``ori_out_width -
        # external_width``. Both deltas are 0 for a passthrough rep.
        self._trunk_in_size = in_size + len(self._ori_in_slots) * (
            ori_in_width - external_width
        )
        self._trunk_out_size = out_size + len(self._ori_out_slots) * (
            ori_out_width - external_width
        )
        # LOGVAR head (probabilistic only): the attitude slot occupies the 3-D so3
        # tangent instead of 4 quaternion components, i.e. NARROWS by 1 per slot.
        # RLRP-744 item A (§A.3 step 4): the narrowing tracks the manifold lever —
        # the matrix-rep OFF ablation (``manifold_aware_nll: false``) keeps the
        # 4-D logvar slot so ``_base_nll`` scores the decoded quaternion
        # consistently.
        if not deterministic and self._manifold_aware_nll:
            self._logvar_trunk_out_size = out_size + len(self._ori_out_slots) * (
                TANGENT_DIM - external_width
            )

        # Dedicated single-step (SS / deploy) head (architectures B/C). The SS head
        # emits ONE single-step obs block, so it has its own slot set + rep-expanded
        # width, distinct from the composed MS-window bookkeeping above. The
        # ``_ori_out_slots`` handed in are the composed-window expansion
        # (``k * singlestep_obs_len + b``); the SS base slots are exactly those that
        # fall in the FIRST block (``k == 0`` => ``s < singlestep_obs_len``). This
        # mirrors the composed/MS bookkeeping one-to-one at single-step granularity
        # so the dedicated SS final layer gets a correctly-sized output and a
        # correctly-scoped decode (RLRP-736 bespoke-forward plan §2.1/§5.1).
        if singlestep_obs_len is not None and self._ori_out_slots:
            self._ss_ori_out_slots = tuple(
                s for s in self._ori_out_slots if s < singlestep_obs_len
            )
            self._ss_trunk_out_len = singlestep_obs_len + len(
                self._ss_ori_out_slots
            ) * (ori_out_width - external_width)
            # Probabilistic dedicated SS logvar head: narrowed 3-D tangent slot
            # (tracks the manifold lever — RLRP-744 §A.3 step 4).
            if not deterministic and self._manifold_aware_nll:
                self._ss_logvar_trunk_out_len = singlestep_obs_len + len(
                    self._ss_ori_out_slots
                ) * (TANGENT_DIM - external_width)

        # Single-step INPUT bookkeeping for the encoder->decoder AR family
        # (architecture C.2). The AR encoder consumes the composed window as
        # per-single-step-block ``(..., MS, F)`` slices (F = obs+act of ONE step),
        # so the attitude input slot must be encoded PER BLOCK, and each child's
        # encoder input width must grow by ``_ss_block_in_extra`` accordingly. The
        # SS input base slots are those input slots that fall in the FIRST block
        # (``s < singlestep_obs_len``). Neutral/0 when the wiring is off (RLRP-736
        # bespoke-forward plan §3.4).
        if singlestep_obs_len is not None and self._ori_in_slots:
            self._ss_ori_in_slots = tuple(
                s for s in self._ori_in_slots if s < singlestep_obs_len
            )
            self._ss_block_in_extra = len(self._ss_ori_in_slots) * (
                ori_in_width - external_width
            )

    @staticmethod
    def _splice_slots(
        x: torch.Tensor,
        slots: Sequence[int],
        external_width: int,
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Rebuild ``x`` replacing each ``external_width``-wide slot by ``transform(slot)``.

        Walks the trailing feature dimension in slot order, copying the
        non-attitude segments verbatim and substituting the transformed attitude
        segments. Slots must be sorted, non-overlapping and ``external_width``
        wide. Parameter-free and ensemble-safe (operates only on the last axis).

        **RLRP-747: deliberately NOT gather-ised.** This generic route serves the
        WIDTH-CHANGING reps (``sixd`` -> 6, ``nine_d_svd`` -> 9,
        ``so3_relative`` -> 3): ``transform`` maps a ``external_width``-wide slot
        to an ``internal_rep_out_width(rep, external_width)``-wide one, so the
        mapping is NOT a permutation of the feature axis and cannot be expressed as an
        ``index_select`` / ``index_copy`` pair. The WIDTH-PRESERVING quaternion
        route uses :meth:`_splice_slots_width_preserving` instead; this method
        remains the semantic reference (including its fail-loud non-overlap
        check, mirrored by ``orientation_heads.attitude_slot_index``).
        """
        if not slots:
            return x
        ordered = sorted(slots)
        segments: List[torch.Tensor] = []
        cursor = 0
        for start in ordered:
            if start < cursor:
                raise ValueError(
                    f"Orientation slots must be non-overlapping and sorted; got {slots}."
                )
            if start > cursor:
                segments.append(x[..., cursor:start])
            segments.append(transform(x[..., start : start + external_width]))
            cursor = start + external_width
        if cursor < x.shape[-1]:
            segments.append(x[..., cursor:])
        return torch.cat(segments, dim=-1)

    @staticmethod
    def _splice_slots_width_preserving(
        x: torch.Tensor,
        slots: Sequence[int],
        slot_width: int,
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Vectorized :meth:`_splice_slots` for a WIDTH-PRESERVING ``transform`` (RLRP-747).

        When ``transform`` keeps the slot width (the quaternion reps: ``QUATERNION``
        is ``F.normalize``, ``QUATERNIONLEGACY`` is the identity — both
        ``4 -> 4``), the splice IS a feature-axis permutation. The ``S`` separate
        ``transform`` invocations then collapse into ONE call over a stacked
        ``(..., S, slot_width)`` slot axis (a single ``index_select`` gather),
        while the rebuild keeps the SINGLE full-width ``torch.cat`` of
        :meth:`_splice_slots`. ``transform`` must act only on the trailing axis
        (true of :func:`decode_internal_rep_to_wxyz`).

        This matters on the COMPOSED multi-step window: ``self._ori_out_slots`` is
        the composed expansion ``k * singlestep_obs_len + b``, i.e. one slot per
        horizon step, so the replaced ``S`` transform invocations happened ``H``
        times on every ``forward``.

        **Measured (RLRP-747, 2026-08-27,** ``bench_rlrp747_orientation_vectorization.py``
        **).** Speed-up factors, generic-``_splice_slots``-relative (``> 1`` = this
        route wins):

        ====================  ==========  ==========  ==========  ====  ====
        shape (E x B x H)     MacBook     Orin CPU    Orin CUDA   gen   this
                              CPU         (arm64)     (sm_87)     ops   ops
        ====================  ==========  ==========  ==========  ====  ====
        5 x 256 x 4  (S=4)         0.77x       1.04x       1.57x    26    18
        5 x 256 x 8  (S=8)         0.92x       0.96x       2.16x    50    26
        ====================  ==========  ==========  ==========  ====  ====

        **Kept deliberately, despite the CPU numbers.** The launch-count reduction
        converts into a clear **1.6-2.2x GPU win**, which is where every training
        (HPC amd64) and deployment (Jetson AGX Orin, arm64) ``forward`` actually
        runs; the CPU path is exercised only by tests and is at worst ~20% slower on
        a small tensor. This is the RLRP-783 mechanism (launch-bound, not
        FLOP-bound) behaving exactly as predicted.

        An earlier ``index_copy``-based rebuild was rejected outright: out-of-place
        ``index_copy`` is ``clone() + scatter``, i.e. TWO full-width writes, which
        was slower than the generic route on GPU as well as CPU.

        Bit-exact with :meth:`_splice_slots`. Falls back to the generic route when
        the slot set is not fully in range (so the existing fail-loud behaviour is
        preserved rather than silently skipped).

        :param x: Tensor ``(..., F)`` to rebuild (not mutated).
        :param slots: Attitude slot start indices in ``x``'s own layout.
        :param slot_width: Slot width, IDENTICAL on input and output.
        :param transform: Trailing-axis-only, width-preserving slot transform.
        :return: ``x`` with ``transform`` applied to every attitude slot.
        """
        if not slots:
            return x
        bases = orientation_slot_bases(slots, x.shape[-1], slot_width)
        if len(bases) != len({int(s) for s in slots}):
            # Out-of-range slot: defer to the generic (fail-loud) implementation.
            return ExponentialFamilyMLP._splice_slots(
                x, slots, slot_width, transform
            )
        index = attitude_slot_index_tensor(bases, x.shape[-1], slot_width, x.device)
        stacked = gather_attitude_slots(x, index, slot_width)
        return scatter_attitude_slots(x, bases, transform(stacked))

    def _capture_s2_tangent_reference(
        self, x: torch.Tensor, in_slots: Sequence[int]
    ) -> None:
        """Capture the ``S2_TANGENT`` chart base point from the input window.

        RLRP-796 ``D-793-3-A``. The tangent decode is expressed at the PREVIOUS
        step's direction, which the trunk output cannot supply — it is an INPUT
        fact. The reference is therefore read here, on the input encode that runs
        on every forward before the trunk, as the LAST (highest-index, i.e. most
        recent) orientation slot of the window; the slots are still in their
        EXTERNAL layout at this point, so the value is the absolute 3-D gravity
        direction as observed. This is exactly the ``u = 0`` seed the
        ``D-796-S1-6-A`` decode scan chains from.

        No-op for every non-tangent rep, so the quaternion route never pays for it.

        :param x: The input tensor, orientation slots still EXTERNAL-width.
        :param in_slots: The input orientation slot base indices in ``x``'s layout.
        """
        if self._orientation_rep is not InternalOrientationRep.S2_TANGENT:
            return
        last = max(int(s) for s in in_slots)
        self._ori_decode_reference = x[..., last : last + self._ori_external_width]

    def _s2_tangent_decode_transform(
        self,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build the reference-chaining ``S2_TANGENT`` decode (RLRP-796 ``D-796-S1-6-A``).

        The returned transform is STATEFUL BY DESIGN: it decodes slot ``u`` at the
        base point produced by slot ``u-1`` and then advances the reference, so a
        sequence of calls realizes the per-step chart of ``D-793-3-A``. That is
        sound because :meth:`_splice_slots` — the route every width-changing rep
        takes, and ``S2_TANGENT`` is one (2 raw values -> a 3-D direction) — invokes
        the transform exactly once per slot in ASCENDING slot order, and the
        composed-window slot order IS the horizon order
        (``k * singlestep_obs_len + b``). A fresh transform is built per decode
        call, so no state leaks between forwards.

        The single-base-point fallback documented in ``D-796-S1-6-A`` (all steps
        referenced to the seed) would simply drop the reference advance below.

        :return: A per-slot decode closure seeded at the captured reference.
        :raises RuntimeError: When no reference was captured this forward.
        """
        reference = self._ori_decode_reference
        if reference is None:
            raise RuntimeError(
                f"{type(self).__name__}: the S2_TANGENT decode needs the previous "
                f"step's gravity direction as its chart base point (RLRP-796 "
                f"'D-793-3-A'), but none was captured on this forward. The "
                f"reference is read by _capture_s2_tangent_reference during the "
                f"orientation INPUT encode, so this means the output slots are "
                f"wired without matching input slots — a wiring bug, not a "
                f"recoverable state."
            )
        state = {"g_ref": reference}

        def transform(raw: torch.Tensor) -> torch.Tensor:
            decoded = decode_internal_rep_to_orientation_slot(
                raw, self._orientation_rep, g_ref=state["g_ref"]
            )
            state["g_ref"] = decoded
            return decoded

        return transform

    def _apply_orientation_input_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Encode each 4-D attitude input slot to the internal rep (RLRP-736 §S2.2-B).

        No-op (returns ``x`` unchanged) unless the by-construction wiring is
        active and input slots were provided.

        **RLRP-747: no work needed here.** This method is UNREACHABLE on the
        quaternion path — ``_internal_orientation_rep_active`` is False for ``QUATERNION`` /
        ``QUATERNIONLEGACY`` (see ``_setup_orientation_rep``), so the guard below
        early-returns. The width-changing reps keep the generic
        :meth:`_splice_slots` route.
        """
        if not getattr(self, "_internal_orientation_rep_active", False) or not self._ori_in_slots:
            return x
        self._capture_s2_tangent_reference(x, self._ori_in_slots)
        return self._splice_slots(
            x,
            self._ori_in_slots,
            external_width=self._ori_external_width,
            transform=lambda slot: encode_orientation_slot_to_internal_rep(
                slot, self._orientation_rep
            ),
        )

    def _apply_ss_orientation_input_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the attitude input slot(s) of ONE single-step block (RLRP-736 §3.4).

        For the encoder->decoder AR family (architecture C.2) the raw composed
        input is reshaped to per-single-step-block slices ``(..., MS, F)`` before
        the specialized encoder (GRU/LSTM/TCN cell). This helper encodes the
        attitude quaternion slot(s) WITHIN a single block (``_ss_ori_in_slots``,
        the unexpanded base indices) so the encoder sees the continuous, sign-
        invariant internal rep (``R(q)=R(-q)``; Geist et al. 2024) rather than the
        raw discontinuous quaternion. No-op / bit-exact unless active and SS input
        slot(s) were tracked. The child encoder's input width is widened by
        ``_ss_block_in_extra`` to match.
        """
        if not getattr(self, "_internal_orientation_rep_active", False) or not getattr(
            self, "_ss_ori_in_slots", ()
        ):
            return x
        self._capture_s2_tangent_reference(x, self._ss_ori_in_slots)
        return self._splice_slots(
            x,
            self._ss_ori_in_slots,
            external_width=self._ori_external_width,
            transform=lambda slot: encode_orientation_slot_to_internal_rep(
                slot, self._orientation_rep
            ),
        )

    def _apply_orientation_output_decoding(self, mean: torch.Tensor) -> torch.Tensor:
        """Decode each trunk-output attitude slot to a unit quaternion (RLRP-736 §S2.2-B).

        The trunk emits ``internal_rep_out_width(rep, external_width)`` raw values
        per attitude slot (in the *trunk* output layout); each is mapped onto
        ``SO(3)`` and returned as a 4-D unit quaternion in the external output
        layout. No-op unless the wiring is active and output slots were provided.
        """
        if (
            not (
                getattr(self, "_internal_orientation_rep_active", False)
                or getattr(self, "_manifold_aware_nll", False)
                or getattr(self, "_unit_decode", False)
            )
            or not self._ori_out_slots
        ):
            return mean
        external_width = self._ori_external_width
        raw_width = internal_rep_out_width(self._orientation_rep, external_width)
        # ``_ori_out_slots`` are EXTERNAL output positions (same convention as the
        # input slots), but the trunk emits ``raw_width`` values per attitude slot,
        # so in the (rep-expanded) TRUNK layout every successive slot start shifts
        # by ``raw_width - external_width`` relative to its external position. Map
        # external → trunk positions before splicing. (For a single attitude slot
        # the offset is zero, so the single-step robotic-3D path is unchanged; this
        # also hardens the multi-slot composed-window case — RLRP-736 §S2.2-B.)
        ordered = sorted(self._ori_out_slots)
        trunk_slots = [
            start + i * (raw_width - external_width) for i, start in enumerate(ordered)
        ]

        if self._orientation_rep is InternalOrientationRep.S2_TANGENT:
            # RLRP-796 ``D-796-S1-6-A``: the per-step chart makes the decode a
            # SEQUENTIAL scan over the horizon (block ``u`` at block ``u-1``'s
            # decoded direction, seeded from the last history gravity). Confined to
            # this route; the quaternion and ``S2_IDENTITY`` routes keep the
            # RLRP-747 vectorized gather/scatter below.
            transform = self._s2_tangent_decode_transform()
        else:

            def transform(raw: torch.Tensor) -> torch.Tensor:
                return decode_internal_rep_to_orientation_slot(
                    raw, self._orientation_rep
                )

        if raw_width == external_width:
            # RLRP-747: the quaternion reps keep the slot width (``QUATERNION`` is
            # ``F.normalize``, ``QUATERNIONLEGACY`` is the identity), so this is a
            # feature-axis permutation -> ONE gather + ONE scatter with a single
            # stacked transform, instead of one ``cat`` of ``2S+1`` segments with
            # ``S == H`` (``_ori_out_slots`` is the COMPOSED-window expansion, so
            # the replaced loop ran once per horizon step on every ``forward``).
            return self._splice_slots_width_preserving(
                mean, trunk_slots, external_width, transform
            )
        return self._splice_slots(
            mean,
            trunk_slots,
            external_width=raw_width,
            transform=transform,
        )

    def _apply_ss_orientation_output_decoding(
        self, ss_mean: torch.Tensor
    ) -> torch.Tensor:
        """Decode the DEDICATED single-step (SS / deploy) head raw attitude slot(s).

        Mirror of :meth:`_apply_orientation_output_decoding` for the SS head of
        architectures B/C (:class:`WeightedMultiStepDualHeadMLP` and the
        encoder->decoder AR family): the SS head emits ONE single-step obs block
        widened to ``_ss_trunk_out_len`` (``singlestep_obs_len`` with each attitude
        slot expanded to ``internal_rep_out_width(rep, external_width)`` raw
        numbers). Each such slot is mapped onto ``SO(3)`` and returned as a 4-D
        unit quaternion in the external single-step layout, so the SS/deploy
        output (what the real robot consumes)
        is a unit quaternion by construction (Geist et al. 2024; Chen et al. 2023;
        Zhou et al. 2019). No-op / bit-exact unless the wiring is active and SS
        attitude slot(s) were tracked (RLRP-736 bespoke-forward plan §2.2/§5.2).
        """
        if (
            not (
                getattr(self, "_internal_orientation_rep_active", False)
                or getattr(self, "_manifold_aware_nll", False)
                or getattr(self, "_unit_decode", False)
            )
            or not getattr(self, "_ss_ori_out_slots", ())
        ):
            return ss_mean
        external_width = self._ori_external_width
        raw_width = internal_rep_out_width(self._orientation_rep, external_width)
        # ``_ss_ori_out_slots`` are EXTERNAL single-step positions; in the
        # (rep-expanded) SS-head layout every successive slot start shifts by
        # ``raw_width - external_width`` (identical external->trunk mapping as the
        # MS decode).
        ordered = sorted(self._ss_ori_out_slots)
        trunk_slots = [
            start + i * (raw_width - external_width) for i, start in enumerate(ordered)
        ]

        if self._orientation_rep is InternalOrientationRep.S2_TANGENT:
            # RLRP-796 ``D-796-S1-6-A``: the dedicated SS / deploy head emits ONE
            # block, so the scan degenerates to a single decode at the seed — the
            # last history gravity direction, which is exactly the previous step
            # relative to the single predicted one.
            transform = self._s2_tangent_decode_transform()
        else:

            def transform(raw: torch.Tensor) -> torch.Tensor:
                return decode_internal_rep_to_orientation_slot(
                    raw, self._orientation_rep
                )

        if raw_width == external_width:
            # RLRP-747: width-preserving route (the quaternion reps and the
            # width-preserving S^2 identity rep; see the MS decode).
            return self._splice_slots_width_preserving(
                ss_mean, trunk_slots, external_width, transform
            )
        return self._splice_slots(
            ss_mean,
            trunk_slots,
            external_width=raw_width,
            transform=transform,
        )

    def _layer_bloc_factory(
        self,
        region: str,
        ensemble_size: int,
        create_activation: Callable[[], nn.Module],
        region_dropout: float,
    ):
        """Resolve the ``(kind, make_bloc)`` builder for ``region`` (RLRP-768).

        Reads the per-region settings resolved in ``__init__``
        (:func:`resolve_layer_bloc_settings`) and dispatches through
        :func:`create_layer_bloc_factory`. A per-region ``dropout`` override
        (``layer_bloc[region].dropout``) takes precedence over the region
        default rate ``region_dropout``.
        """
        settings = self._layer_bloc_resolved[region]
        dropout = region_dropout if settings["dropout"] is None else settings["dropout"]
        return create_layer_bloc_factory(
            settings["type"],
            ensemble_size,
            create_activation,
            dropout=dropout,
            layer_norm=settings["layer_norm"],
            region=region,
        )

    def layer_bloc_signature(self) -> str:
        """Stable per-region bloc-topology signature for the checkpoint guard."""
        return layer_bloc_signature_from_resolved(self._layer_bloc_resolved)

    def _build_network(
        self,
        num_layers: int,
        in_size: int,
        hid_size: int,
        out_size: int,
        ensemble_size: int,
        activation_fn_cfg: omegaconf.DictConfig,
        deterministic: bool,
        learn_logvar_bounds: bool,
        instanciate_logvar_bound_module: bool = True,
        dropout: float = 0.0,
    ) -> None:
        # (Priority) done: test hypothesis-105 mean and logvar head hiden size alt
        MS_HEAD_HIDDEN_SIZE = out_size
        # (Priority) inprogress: test hypothesis-105b mean and logvar head hiden size
        # MS_HEAD_HIDDEN_SIZE = hid_size

        create_activation = partial(create_activation_, activation_fn_cfg)
        create_linear_layer = partial(create_linear_layer_, ensemble_size)

        # .... Layer-bloc resolvers (RLRP-768: per-region type / layer_norm / dropout) ...........
        # Two independent resolvers so the encoder/hidden stack (region "encoder", uses `dropout`)
        # and the MS-head stack (region "ms_head", uses `ms_head_dropout`) are toggled separately.
        # The shared `make_layer_bloc_seq` helper keeps the elite toggler's `layer[0]` convention.
        enc_res_kind, make_enc_res = self._layer_bloc_factory(
            "encoder", ensemble_size, create_activation, dropout
        )
        ms_res_kind, make_ms_res = self._layer_bloc_factory(
            "ms_head", ensemble_size, create_activation, self.ms_head_dropout
        )

        if num_layers >= 2:
            hidden_layers = [
                nn.Sequential(
                    nn.Dropout(p=dropout),
                    create_linear_layer(in_size, hid_size),
                    create_activation(),
                )
            ]

            # Stack residual layer
            for i in range(num_layers - 2):
                hidden_layers.append(
                    make_layer_bloc_seq(
                        enc_res_kind, make_enc_res, hid_size, create_activation, dropout
                    )
                )

            # Dual head split
            hidden_layers.append(
                nn.Sequential(
                    nn.Dropout(p=dropout),
                    create_linear_layer(hid_size, out_size),
                    create_activation(),
                )
            )
        else:
            hidden_layers = [
                nn.Sequential(
                    nn.Dropout(p=dropout),
                    create_linear_layer(in_size, out_size),
                    create_activation(),
                )
            ]

        self.hidden_layers = nn.Sequential(*hidden_layers)

        # .... Multi-Step head ....................................................................
        ms_mean_layers = []
        ms_logvar_layers = []

        # Note: Dropout is only on the first ms head layer on purposes and use a dedicated
        # parammeter 'ms_head_dropout' instead of the 'dropout' one which is used for
        # the encoder hiden layers.
        if deterministic:
            ms_mean_layers.append(
                nn.Sequential(
                    nn.Dropout(p=self.ms_head_dropout),
                    create_linear_layer(out_size, MS_HEAD_HIDDEN_SIZE),
                    create_activation(),
                )
            )
        else:
            self.mean_and_logvar = nn.Sequential(
                nn.Dropout(p=self.ms_head_dropout),
                create_linear_layer(out_size, 2 * MS_HEAD_HIDDEN_SIZE),
                create_activation(),
            )

        # Stack multi-step mean and logvar residual layers
        for i in range(self.ms_head_num_layers - 1):
            ms_mean_layers.append(
                make_layer_bloc_seq(
                    ms_res_kind, make_ms_res, MS_HEAD_HIDDEN_SIZE,
                    create_activation, self.ms_head_dropout,
                )
            )

            if not self.deterministic:
                ms_logvar_layers.append(
                    make_layer_bloc_seq(
                        ms_res_kind, make_ms_res, MS_HEAD_HIDDEN_SIZE,
                        create_activation, self.ms_head_dropout,
                    )
                )

        # Finale multi-step mean layer
        ms_mean_layers.append(
            nn.Sequential(
                create_linear_layer(MS_HEAD_HIDDEN_SIZE, out_size),
            )
        )
        self.mean_layer = nn.Sequential(*ms_mean_layers)

        if not self.deterministic:
            # RLRP-736 §4A-B: the probabilistic LOGVAR head is narrowed at each
            # attitude slot to the 3-D so3 tangent (``_logvar_trunk_out_size``),
            # ASYMMETRIC with the mean head's rep-expanded ``out_size``. Only
            # override when the by-construction rep is ACTIVE (and the caller-passed
            # ``out_size`` matches the main-trunk output width the bookkeeping was
            # computed for); otherwise use the caller-passed ``out_size`` verbatim so
            # every neutral / OFF family — including per-horizon heads that call
            # ``_build_network`` with a DIFFERENT ``out_size`` — stays BIT-EXACT.
            logvar_out_size = out_size
            if (
                getattr(self, "_internal_orientation_rep_active", False)
                or getattr(self, "_manifold_aware_nll", False)
            ) and out_size == getattr(self, "_trunk_out_size", out_size):
                logvar_out_size = getattr(self, "_logvar_trunk_out_size", out_size)
            # Finale multi-step logvar layer
            ms_logvar_layers.append(
                nn.Sequential(
                    create_linear_layer(MS_HEAD_HIDDEN_SIZE, logvar_out_size),
                )
            )
            self.logvar_layer = nn.Sequential(*ms_logvar_layers)

            if instanciate_logvar_bound_module:
                # (NICE TO HAVE) ToDo: RLRP-377 feat(ExponentialFamilyMLP): consider bounding
                #   logvar wihtout adding a layer
                self.logvar_layer.add_module(  # Make the module fetchable by name
                    "logvar_bound",
                    create_logvar_bound_layer(
                        logvar_out_size,
                        learn_logvar_bounds,
                        bound_min_init=torch.tensor(
                            self._LOGVAR_MIN_BOUND_INIT, device=self.device
                        ),
                        bound_max_init=torch.tensor(
                            self._LOGVAR_MAX_BOUND_INIT, device=self.device
                        ),
                        grad_clip=self.logvar_bound_grad_clip,
                    ),
                )

        self.apply(truncated_normal_init)
        # Re-zero the identity residual blocks' last linear AFTER the global
        # truncated-normal init so they start as an exact identity (y == x).
        zero_init_residual_blocks_(self)
        self.to(self.device)

        return None

    def get_one_d_trj_model_input_normalizer(self) -> Optional[ZScoreNormalizer]:
        return self._one_d_trj_model_input_normalizer

    def set_one_d_trj_model_input_normalizer(
        self, normalizer: Optional[ZScoreNormalizer]
    ) -> None:  # ABC type
        self._one_d_trj_model_input_normalizer = normalizer
        return None

    def get_one_d_trj_model_obs_denorm_handle(
        self,
    ) -> Optional[Callable[[Tensor], Tensor]]:
        return getattr(self, "_one_d_trj_model_obs_denorm_handle", None)

    def set_one_d_trj_model_obs_denorm_handle(
        self, denorm_handle: Optional[Callable[[Tensor], Tensor]]
    ) -> None:
        """Register the obs-space-only DENORM handle (RLRP-731, action ``B3-handle``).

        A wrapper-provided ``Tensor -> Tensor`` callable mapping a NORMALIZED single-step
        obs tensor ``(..., Do)`` to RAW (denormalized) obs space, used by the deploy-history
        drift residual (DH) loss. ``None`` -> identity (asymmetric ``standard`` normalizer,
        whose targets/predictions are already raw, or normalization disabled). Kept SEPARATE
        from ``set_one_d_trj_model_input_normalizer`` so the RLRP-684 ``None``-handle
        invariant of the AR round-trip is never disturbed.
        """
        self._one_d_trj_model_obs_denorm_handle = denorm_handle
        return None

    def get_one_d_trj_model_ar_bridge_gain(self) -> Optional[Tensor]:
        return getattr(self, "_one_d_trj_model_ar_bridge_gain", None)

    def set_one_d_trj_model_ar_bridge_gain(self, gain: Optional[Tensor]) -> None:
        """Register the AR splice TARGET-space -> INPUT-space obs gain (RLRP-761 ``S4.4``).

        A ``(Do,)`` diagonal affine gain, or ``None`` meaning "identity, the two
        spaces coincide" — which is the case for **every** normalizer type
        except ``standard_symmetric_innovation``, whose obs input scale (state
        std) and obs target scale (one-step innovation) are decoupled. Kept as
        a THIRD handle, separate from the input-normalizer and the DH denorm
        ones, so the RLRP-684 ``None``-handle invariant of the round-trip guard
        is untouched: the block facades still propagate ``None`` there and the
        expensive denormalize/renormalize pair stays skipped.
        """
        self._one_d_trj_model_ar_bridge_gain = gain
        return None

    def get_one_d_trj_model_ar_bridge_gain_act(self) -> Optional[Tensor]:
        return getattr(self, "_one_d_trj_model_ar_bridge_gain_act", None)

    def set_one_d_trj_model_ar_bridge_gain_act(self, gain: Optional[Tensor]) -> None:
        """Register the AR splice TARGET-space -> INPUT-space **act** gain (RLRP-761 ``S12.4``).

        Act analogue of :meth:`set_one_d_trj_model_ar_bridge_gain`: a ``(Da,)``
        diagonal affine gain, or ``None`` meaning "identity, the two act spaces
        coincide" — the case for every normalizer type whose act facades are
        shared (all legacy types and pre-``S12`` innovation runs). Only
        ``standard_symmetric_innovation`` under ``S12`` decouples the act input
        (state std) and act target (one-step innovation) scales, so the commands
        / ``dt`` re-injected by the MS forecast self-feed cross this gain.
        """
        self._one_d_trj_model_ar_bridge_gain_act = gain
        return None

    def _get_logvar_bound_layer(self) -> Optional[LogvarBoundLayer]:
        return (
            self.logvar_layer.get_submodule("logvar_bound")
            if not self.deterministic
            else None
        )

    def _logvar_bound_penalty_specs(
        self,
    ) -> List[Tuple[Optional[LogvarBoundLayer], Optional[Any]]]:
        """Declare the ``LogvarBoundLayer`` bounds whose fixed PETS/Chua penalty must be
        added ONCE to the composite total, together with the adapter that selects the
        penalised parameter sub-slice (``None`` = identity, i.e. the whole vector).

        Subclasses owning more than one bound (e.g. an encoder/MS bound and a deploy/SS
        bound) override this to return one ``(bound_layer, adapter)`` spec per bound that
        is *active* given the current config (SS on/off, projection head built, ...).
        The shared :meth:`_logvar_bound_penalty` dedupes by ``id(bound_layer)`` so an
        aliased layer is penalised at most once.
        """
        layer = self._get_logvar_bound_layer()
        if layer is None:
            return []
        return [(layer, None)]

    def _logvar_bound_penalty(self) -> Union[Tensor, float]:
        """Standalone fixed-coefficient ``0.01`` penalty on the learnable logvar bounds.

        Returns ``Sum_b 0.01 * (adapter_b(b.logvar_max).sum() - adapter_b(b.logvar_min).sum())``
        over the active, *learnable* bounds declared by :meth:`_logvar_bound_penalty_specs`,
        deduped by ``id(bound_layer)``. Frozen bounds (``requires_grad=False``) are skipped
        (they only contribute a constant). Returns ``0.0`` if no active learnable bound.

        This replaces the historical in-path ``LogvarBoundLayer.bound_losses(nll)`` call:
        the penalty math is identical, but it is now added to the *final composite loss*
        (after temporal/feature weighting, the horizon reduction and the auto-weighting),
        so its effective coefficient is the intended fixed ``0.01`` rather than being
        scaled/reshaped by the data-dependent, learned machinery.
        """
        total: Union[Tensor, float] = 0.0
        seen = set()
        for layer, adapter in self._logvar_bound_penalty_specs():
            if layer is None:
                continue
            key = id(layer)
            if key in seen:
                continue
            seen.add(key)
            if (
                not layer.logvar_max.requires_grad
                and not layer.logvar_min.requires_grad
            ):
                # Frozen bound -> constant term, skip to avoid clutter.
                continue

            # Reuse the bound's own penalty logic (single area of responsibility):
            # ``bound_losses`` adds ``0.01 * (adapter(logvar_max).sum() -
            # adapter(logvar_min).sum())`` to the running accumulator (identity when
            # ``adapter is None``).
            total = layer.bound_losses(total, adapter)
        return total

    def _maybe_cast_to_model_dtype(self, x: torch.Tensor) -> torch.Tensor:
        """Cast input tensor to model dtype if needed. (ref task RLRP-369)"""
        if not isinstance(type(x.dtype), type(self.model_dtype)):
            x = x.to(self.model_dtype)
        return x

    def _default_forward(
        self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._maybe_toggle_layers_use_only_elite(only_elite)

        x = self._maybe_cast_to_model_dtype(x)

        # RLRP-736 §S2.2-B: encode the 4-D attitude input slot(s) to the internal
        # continuous rotation rep before the trunk. No-op (bit-exact) unless the
        # by-construction wiring is active.
        x = self._apply_orientation_input_encoding(x)

        x = self.hidden_layers(x)

        if self.deterministic:
            mean_head = self.mean_layer(x)
            # RLRP-736 §S2.2-B: decode the trunk's raw attitude slot(s) back to a
            # unit quaternion by construction. No-op (bit-exact) unless active.
            mean_head = self._apply_orientation_output_decoding(mean_head)
            logvar_head = None
        else:
            mean_and_logvar = self.mean_and_logvar(x)

            # Architecture 1
            mean_split, logvar_split = torch.chunk(mean_and_logvar, chunks=2, dim=-1)
            mean_head = self.mean_layer(mean_split)
            # RLRP-736 §4A-B: decode the trunk's raw attitude slot(s) of the MEAN
            # head back to a unit quaternion by construction (same as the det.
            # path). The LOGVAR head is NOT decoded — its narrowed attitude slot(s)
            # ARE the 3-D so3 tangent log-variance consumed by ``rotation_tangent_nll``
            # in ``_orientation_split_nll``. No-op (bit-exact) unless active.
            mean_head = self._apply_orientation_output_decoding(mean_head)
            logvar_head = self.logvar_layer(logvar_split)
            logvar_head = self._get_logvar_bound_layer()(logvar_head)

            # Architecture 2
            # mean_head = self.mean_layer(mean_and_logvar)
            # logvar_head = self.logvar_layer(mean_and_logvar)

        self._maybe_toggle_layers_use_only_elite(only_elite)
        return mean_head, logvar_head

    def _maybe_toggle_layers_use_only_elite(self, only_elite: bool):
        if self.elite_models is None:
            return
        if self.num_members > 1 and only_elite:
            for layer in self.hidden_layers:
                # each layer is (linear layer, activation_func)
                layer[0].set_elite(self.elite_models)
                layer[0].toggle_use_only_elite()
            self.mean_and_logvar.set_elite(self.elite_models)
            self.mean_and_logvar.toggle_use_only_elite()
            for layer in self.mean_layer:
                layer[0].set_elite(self.elite_models)
                layer[0].toggle_use_only_elite()
            for layer in self.logvar_layer:
                layer[0].set_elite(self.elite_models)
                layer[0].toggle_use_only_elite()

    def _forward_from_indices(
        self, x: torch.Tensor, model_shuffle_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, batch_size, _ = x.shape

        num_models = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        shuffled_x = x[:, model_shuffle_indices, ...].view(
            num_models, batch_size // num_models, -1
        )

        mean, logvar = self._default_forward(shuffled_x, only_elite=True)
        # note that mean and logvar are shuffled
        mean = mean.view(batch_size, -1)
        mean[model_shuffle_indices] = mean.clone()  # invert the shuffle

        if logvar is not None:
            logvar = logvar.view(batch_size, -1)
            logvar[model_shuffle_indices] = logvar.clone()  # invert the shuffle

        return mean, logvar

    def _forward_ensemble(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.propagation_method is None:
            mean, logvar = self._default_forward(x, only_elite=False)
            if self.num_members == 1:
                mean = mean[0]
                logvar = logvar[0] if logvar is not None else None
            return mean, logvar
        assert x.ndim == 2
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if x.shape[0] % model_len != 0:
            raise ValueError(
                "ExponentialFamilyMLP ensemble requires batch size to be a multiple of the "
                f"number of models. Current batch size is {x.shape[0]} for "
                f"{model_len} models."
            )
        x = x.unsqueeze(0)
        if self.propagation_method == "random_model":
            # passing generator causes segmentation fault
            # see https://github.com/pytorch/pytorch/issues/44714
            model_indices = torch.randperm(x.shape[1], device=self.device)
            return self._forward_from_indices(x, model_indices)
        if self.propagation_method == "fixed_model":
            if propagation_indices is None:
                raise ValueError(
                    "When using propagation='fixed_model', `propagation_indices` must be provided."
                )
            return self._forward_from_indices(x, propagation_indices)
        if self.propagation_method == "expectation":
            mean, logvar = self._default_forward(x, only_elite=True)
            return mean.mean(dim=0), logvar.mean(dim=0)
        raise ValueError(f"Invalid propagation method {self.propagation_method}.")

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes mean and logvar predictions for the given input.

        When ``self.num_members > 1``, the model supports uncertainty propagation options
        that can be used to aggregate the outputs of the different models in the ensemble.
        Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

        If a set of elite models has been indicated (via :meth:`set_elite()`), then all
        propagation methods will operate with only on the elite set. This has no effect when
        ``propagation is None``, in which case the forward pass will return one output for
        each model.

        Args:
            x (tensor): the input to the model. When ``self.propagation is None``,
                the shape must be ``E x B x Id`` or ``B x Id``, where ``E``, ``B``
                and ``Id`` represent ensemble size, batch size, and input dimension,
                respectively. In this case, each model in the ensemble will get one slice
                from the first dimension (e.g., the i-th ensemble member gets ``x[i]``).

                For other values of ``self.propagation`` (and ``use_propagation=True``),
                the shape must be ``B x Id``.
            rng (torch.Generator, optional): random number generator to use for "random_model"
                propagation.
            propagation_indices (tensor, optional): propagation indices to use,
                as generated by :meth:`sample_propagation_indices`. Ignore if
                `use_propagation == False` or `self.propagation_method != "fixed_model".
            use_propagation (bool): if ``False``, the propagation method will be ignored
                and the method will return outputs for all models. Defaults to ``True``.

        Returns:
            (tuple of two tensors): the predicted mean and log variance of the output. If
            ``propagation is not None``, the output will be 2-D (batch size, and output dimension).
            Otherwise, the outputs will have shape ``E x B x Od``, where ``Od`` represents
            output dimension.

        Note:
            For efficiency considerations, the propagation method used by this class is an
            approximate version of that described by Chua et al. In particular, instead of
            sampling models independently for each input in the batch, we ensure that each
            model gets exactly the same number of samples (which are assigned randomly
            with equal probability), resulting in a smaller batch size which we use for the forward
            pass. If this is a concern, consider using ``propagation=None``, and passing
            the output to :func:`mbrl.util.math.propagate`.

        """
        if use_propagation:
            return self._forward_ensemble(
                x, rng=rng, propagation_indices=propagation_indices
            )
        return self._default_forward(x)

    @torch.compiler.disable
    def _deterministic_loss(
        self, model_in: torch.Tensor, target: torch.Tensor, reduce=True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        model_in, target = self._setup_loss_input(model_in, target)

        pred_mean, _ = self.forward(model_in, use_propagation=False)

        if self.mae_loss:
            losses = F.l1_loss(pred_mean, target, reduction="none")
        else:
            losses = F.mse_loss(pred_mean, target, reduction="none")

        if reduce:
            losses = reduce_deterministic_compose_loss(losses)

        meta = {}
        # RLRP-751 (task T5): compose the per-feature geometry term(s) INLINE,
        # AFTER the reduction, into this loss accumulator — replacing the removed
        # stash/drain seam. The base reduce SUMS over the ensemble axis, so (like
        # the historical geometry seam) the STATIC scalar term is added ONCE here
        # rather than per-ensemble-member. SS = deploy point
        # (``pred_mean``/``target``); the MS (multi-step forecast head) is composed
        # by the hook (no-op on single-step / SS-only models, overridden by the
        # ``ms2ms`` forecast baselines). Bit-neutral OFF (no handler / weight 0).
        #
        # The SS pair goes through :meth:`_feature_geometry_ss_channel` so a family
        # whose ``pred_mean`` is a COMPOSED multi-step window can hand over its
        # genuine ``t=1`` single-step point instead of the flat window (the geometry
        # term is defined on the single-step obs layout).
        losses = self._compose_feature_geometry(
            losses, meta, ss=self._feature_geometry_ss_channel(pred_mean, target)
        )
        losses = self._maybe_compose_feature_geometry_ms_head(
            losses, pred_mean, target, meta
        )

        # .... memory management ..................................................................
        # del pred_mean, model_in, target  # (CRITICAL) ToDo: validate grad ok

        return losses, meta

    def _orientation_nll_active(self) -> bool:
        """Whether the by-construction PROBABILISTIC attitude NLL split is live.

        Requires the resolved manifold-aware NLL lever (``self._manifold_aware_nll``,
        RLRP-744 §A.3 step 3 — replaces the former ``_internal_orientation_rep_active`` gate so
        the plain ``quaternion`` rep can opt IN and the matrix reps can opt OUT),
        output slot(s), and a probabilistic model. Bit-exact OFF otherwise
        (RLRP-736 §4A-B; lever ``null`` reproduces the pre-lever behaviour since
        ``_manifold_aware_nll == _internal_orientation_rep_active`` then).
        """
        return bool(
            getattr(self, "_manifold_aware_nll", False)
            and getattr(self, "_ori_out_slots", ())
            and not getattr(self, "_deterministic_rep", True)
        )

    def _gather_orientation_segments(
        self, tensor: torch.Tensor, att_width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split ``tensor`` (last axis) into its STACKED attitude slot(s) + the rest.

        ``self._ori_out_slots`` are EXTERNAL (4-per-slot) output positions. For a
        tensor whose attitude slots occupy ``att_width`` each (4 for the decoded
        mean / target, 3 for the narrowed so3-tangent logvar), the i-th sorted
        slot starts at ``s_ext - i * (4 - att_width)`` and the intervening
        non-attitude segments are copied verbatim. Ensemble-axis-agnostic (last
        axis only) (RLRP-736 §4A-B, review 2026-07-13).

        **COMPOSED multi-step observation: YES, by construction (RLRP-747 ticket
        TODO 2).** ``self._ori_out_slots`` IS the composed-window expansion
        ``k * singlestep_obs_len + b`` (see ``_setup_orientation_slots``), i.e. it
        carries ONE entry per horizon step, not one per robot. This helper is
        therefore composed-aware by construction and needs no layout branch: the
        index arithmetic is purely positional.

        **RLRP-747 vectorization.** Returns the attitude slots STACKED on a new
        second-to-last axis, ``(..., S, att_width)``, instead of a Python ``list``
        of ``S`` slices. That is what lets :meth:`_orientation_split_nll` issue ONE
        ``rotation_tangent_nll`` call instead of ``S`` (== ``H``) separate PyPose
        ``Inv() @ … .Log()`` autograd subgraphs. The index pair is built once and
        cached per ``(slots, width, att_width, device)``.

        :param tensor: ``(..., F)`` mean / target (``att_width == 4``) or
            log-variance (``att_width == 3``) tensor.
        :param att_width: Scalars per attitude slot in ``tensor``.
        :return: ``(stacked_attitude, rest)`` — ``(..., S, att_width)`` and
            ``(..., F - S * att_width)``.
        """
        att_index, rest_index = split_attitude_and_rest_index_tensors(
            self._ori_out_slots, tensor.shape[-1], att_width, tensor.device
        )
        att = gather_attitude_slots(tensor, att_index, att_width)
        rest = tensor.index_select(-1, rest_index)
        return att, rest

    def _base_nll(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
        reduce: bool = True,
    ) -> torch.Tensor:
        """The current per-element Gaussian/Laplace/StudentT NLL, unchanged.

        Wraps BOTH base-NLL code paths (the manual
        ``negative_loglikelihood_loss__explicit`` for
        ``gaussian_manual``/``gaussian_wo_cons``/``logistic_manual`` and the
        ``_to_distribution(...).log_prob`` path otherwise), so the split-NLL
        helper can score the NON-attitude columns identically to the legacy loss
        (RLRP-736 §4A-B).
        """
        if self.distribution_name in (
            "gaussian_manual",
            "gaussian_wo_cons",
            "logistic_manual",
        ):
            nll = negative_loglikelihood_loss__explicit(
                mean, logvar, target, self.distribution_name
            )
        else:
            nll = -self._to_distribution(mean, logvar).log_prob(target)
        if reduce:
            nll = reduce_probabilistic_compose_loss(nll)
        return nll

    def _orientation_split_nll(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
        reduce: bool = True,
    ) -> torch.Tensor:
        """Score attitude slot(s) with ``rotation_tangent_nll``, the rest with the base NLL.

        Bit-exact passthrough to :meth:`_base_nll` when the by-construction
        probabilistic wiring is inactive. When active, each attitude slot's mean
        (a DECODED unit quaternion, 4-D) + target quaternion + 3-D so3-tangent
        log-variance are scored by :func:`rotation_tangent_nll` (Geist et al.
        2024; Forster et al. 2016 for the right-Jacobian sub-mode) while the
        remaining dims keep the exact base Gaussian/Laplace/StudentT NLL. The two
        contributions are concatenated on the feature axis and reduced downstream
        by ``reduce_probabilistic_compose_loss`` — operating on the UNREDUCED
        per-sample tensors so the mixture/PME + VI per-particle log-sum-exp
        (``reduce=False``) stays valid (RLRP-736 §4A-B, review 2026-07-13).

        **COMPOSED multi-step observation: YES (RLRP-747 ticket TODO 2).** This
        scores the COMPOSED multi-step output: ``self._ori_out_slots`` is the
        composed-window expansion ``k * singlestep_obs_len + b``, so there is one
        attitude slot — and therefore one rotation NLL column — PER HORIZON STEP.

        **RLRP-747 vectorization (the headline win).** The attitude slots are now
        scored by a SINGLE ``rotation_tangent_nll`` call over a stacked
        ``(..., S, ·)`` slot axis, replacing the former per-slot loop of ``S``
        (== ``H``) independent PyPose ``Inv() @ … .Log()`` subgraphs. That shrinks
        both the forward kernel-launch count and the autograd tape by a factor of
        ``H``. ``rotation_tangent_nll`` is fully batched over its leading dims
        (it reduces only the trailing axis of size 3), so the per-element
        arithmetic is IDENTICAL (bit-exact, verified for ``float32`` / ``float64``
        and both ``with_right_jacobian`` settings) and the returned ``(..., S)``
        tensor concatenates in exactly the previous sorted-slot column order.

        **Measured (RLRP-747, 2026-08-27,** ``bench_rlrp747_orientation_vectorization.py``
        **).** Speed-up factors, loop-relative:

        =====================  ==========  ==========  ==========  ====  ====
        shape (E x B x H)      MacBook     Orin CPU    Orin CUDA   loop  vec
                               CPU         (arm64)     (sm_87)     ops   ops
        =====================  ==========  ==========  ==========  ====  ====
        5 x 256 x 4   (S=4)         1.72x       2.19x       3.39x   362    97
        5 x 256 x 8   (S=8)         2.09x       2.91x       6.59x   710    97
        5 x 1024 x 8  (S=8)         3.07x       3.08x       6.46x   710    97
        =====================  ==========  ==========  ==========  ====  ====

        Note that ``vec ops`` is CONSTANT at 97 regardless of ``H`` — this is the
        one path in the orientation stack where the work per call genuinely stops
        scaling with the horizon.
        """
        if not self._orientation_nll_active():
            return self._base_nll(mean, logvar, target, reduce=reduce)
        # RLRP-796 Stage 1: both widths are DERIVED from the resolved orientation
        # block, never literal — 4 / 3 for the attitude quaternion block (so3
        # tangent) and 3 / 2 for the gravity direction block (S^2 tangent).
        # ``getattr`` fallback (mirroring the PME bookkeeping): the split-NLL seam
        # is exercised by harnesses that stub the orientation state without going
        # through ``_setup_orientation_rep``, and the pre-Stage-1 behaviour of
        # every such caller is the quaternion block's external width.
        external_width = getattr(
            self, "_ori_external_width", DEFAULT_EXTERNAL_ORIENTATION_WIDTH
        )
        rep = getattr(self, "_orientation_rep", InternalOrientationRep.QUATERNION)
        tangent_dim = internal_rep_tangent_dim(rep, external_width)
        mean_att, mean_rest = self._gather_orientation_segments(mean, external_width)
        target_att, target_rest = self._gather_orientation_segments(
            target, external_width
        )
        # logvar orientation slot narrows to the manifold's tangent dimension; its
        # non-orientation columns line up with the mean/target rest so the Gaussian
        # rest NLL is strictly the current per-element loss.
        logvar_att, logvar_rest = self._gather_orientation_segments(logvar, tangent_dim)

        # ONE stacked call (was one per attitude slot / horizon step): the slot
        # axis is just another leading batch dim for the tangent NLL, so the
        # result ``(..., S)`` already has one column per slot in sorted order.
        if rep is InternalOrientationRep.S2_TANGENT:
            # RLRP-796 ``D-793-4``: score the 2-D S^2 tangent instead of the
            # ambient 3-D slot, so the learned covariance is not spent on the
            # radial direction the manifold forbids. Selected by the REP (hence by
            # the block, via the D-796-S1-5-A table), not by a config knob.
            rot_cols = s2_tangent_nll(
                mean_att, target_att, logvar_att, reduce=False
            )
        else:
            rot_cols = rotation_tangent_nll(
                mean_att,
                target_att,
                logvar_att,
                with_right_jacobian=getattr(self, "_tangent_nll_right_jac", False),
                reduce=False,
            )

        rest_nll = self._base_nll(
            mean_rest, logvar_rest, target_rest, reduce=False
        )
        nll_losses = torch.cat([rest_nll, rot_cols], dim=-1)
        if reduce:
            nll_losses = reduce_probabilistic_compose_loss(nll_losses)
        return nll_losses

    @torch.compiler.disable
    def _probabilistic_loss(
        self, model_in: torch.Tensor, target: torch.Tensor, reduce: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        model_in, target = self._setup_loss_input(model_in, target)

        pred_mean, pred_logvar = self.forward(model_in, use_propagation=False)

        # RLRP-736 §4A-B: when the by-construction rep is active on a probabilistic
        # model, the attitude slot(s) are scored by the so3-tangent NLL and the
        # remaining dims by the usual Gaussian NLL; otherwise this is the exact
        # legacy per-element NLL (bit-exact OFF).
        nll_losses = self._orientation_split_nll(
            pred_mean, pred_logvar, target, reduce=False
        )

        if reduce:
            nll_losses = reduce_probabilistic_compose_loss(nll_losses)

        meta = {}
        # RLRP-751 (task T5): compose the per-feature geometry term(s) INLINE,
        # AFTER the reduction, into this NLL accumulator — replacing the removed
        # stash/drain seam. The base reduce SUMS over the ensemble axis, so (like
        # the historical geometry seam) the STATIC scalar term is added ONCE here.
        # SS = deploy point (distribution mean ``pred_mean`` / ``target``); the MS
        # (multi-step forecast head) is composed by the hook (no-op on single-step
        # / SS-only models, overridden by the ``ms2ms`` forecast baselines).
        # Bit-neutral OFF.
        nll_losses = self._compose_feature_geometry(
            nll_losses,
            meta,
            ss=self._feature_geometry_ss_channel(pred_mean, target),
        )
        nll_losses = self._maybe_compose_feature_geometry_ms_head(
            nll_losses, pred_mean, target, meta
        )

        # .... Variance bounds ....................................................................
        # Standalone fixed-coefficient bound penalty (added once on the composite, after
        # the reduction); replaces the historical in-path ``bound_losses(nll)`` call.
        nll_losses = nll_losses + self._logvar_bound_penalty()

        # .... memory management ..................................................................
        # del model_in, target  # (CRITICAL) ToDo: validate grad ok

        return nll_losses, meta

    def _maybe_compose_feature_geometry_ms_head(
        self,
        losses: torch.Tensor,
        pred_mean: torch.Tensor,
        target: torch.Tensor,
        meta: Dict[str, Any],
    ) -> torch.Tensor:
        """Hook: compose the MS (multi-step forecast) head geometry from the base loss.

        RLRP-751 (task T5), inline successor of the removed
        ``_maybe_stash_feature_loss_ms_head_from_base_loss`` stash hook. No-op by
        default (returns ``losses`` unchanged): single-step / SS-only models do
        NOT expose a composed multi-step forecast head through this base loss.
        Multi-step forecast families whose composed forecast prediction/target
        flow through this base ``_deterministic_loss`` / ``_probabilistic_loss``
        (the ``ms2ms`` forecast baselines: End-to-End-TCN / M3 / TBM) override
        this to compose the ``FEAT_GEOM_MS`` channel via
        :meth:`FeatureGeometryLossMixin._compose_feature_geometry` (``ms_head=``).

        :param losses: the running (pre-reduction) loss accumulator.
        :param pred_mean: model point estimate (distribution mean — NEVER a
            sample) in the composed multi-step next-obs layout.
        :param target: matching composed multi-step next-obs target.
        :param meta: loss metadata dict (mutated in place).
        """
        return losses

    def _feature_geometry_ss_channel(
        self, pred_mean: torch.Tensor, target: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Hook: the ``(point, target)`` pair fed to the SS feature-geometry channel.

        Identity by default (``(pred_mean, target)``), which is correct for every
        family whose base-loss prediction IS a single-step observation
        ``(..., singlestep_obs_len)`` — the layout the geometry term is defined on
        (:meth:`EnvFeatureHandler.extra_loss` indexes single-step obs feature
        indices on the last axis, and the registered obs denorm handle expects
        ``Do * k`` obs columns and nothing else).

        Families whose base-loss ``pred_mean`` is a **composed multi-step window**
        (``obs*history_len + act*(history_len-1)``) override this to return their
        genuine ``t=1`` single-step point — or ``None`` to skip the channel
        entirely. Handing the flat window over instead is a defect: the act tail
        makes the registered denorm handle raise (its obs-only width guard), and
        without a handle the term silently scores the leading window timestep
        (a PAST observation when ``horizon_len < history_len``) under the
        ``feature_geom_loss`` label.

        :param pred_mean: base-loss point estimate (distribution mean — NEVER a
            sample).
        :param target: matching base-loss target.
        :return: the SS ``(point, target)`` pair, or ``None`` to skip the channel.
        """
        return pred_mean, target

    def _setup_loss_input(
        self, model_in: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        if target.shape[0] != self.num_members:
            target = target.repeat(self.num_members, 1, 1)
        return model_in, target

    def _to_distribution(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor,
        overwrite_distribution_name: Optional[str] = None,
        scale_are_log_variance: bool = True,
        validate_args: Optional[bool] = None,
    ) -> dist.Laplace | dist.Normal | dist.StudentT | dist.ExponentialFamily:
        """
        Generates a probability distribution based on the specified mean, scale, and distribution name.

        This function creates and returns a PyTorch distribution object, such as Gaussian, Laplace,
        Student's T, or Logistic, based on the provided parameters. The function also supports
        interpretation of the `scale` parameter as log-variance for convenience, and allows for
        the optional overriding of the distribution type.

        :param mean: The mean or location parameter of the distribution. A tensor.
        :param scale: The scale parameter of the distribution. If `scale_are_log_variance` is True,
                      this will be interpreted as log-variance and converted to standard deviation.
        :param overwrite_distribution_name: Overrides the default distribution name defined in the
                                            class, if provided. Defaults to None.
        :param scale_are_log_variance: A flag indicating whether the `scale` parameter should be
                                       interpreted as log-variance. Defaults to True.
        :return: A PyTorch distribution object corresponding to the specified configuration.
        :rtype: Union[torch.distributions.Laplace, torch.distributions.Normal,
                      torch.distributions.StudentT, torch.distributions.ExponentialFamily]
        """
        # (Priority) ToDo: implement kl-divergence for Student-T and register_kl
        # (Priority) ToDo: implement kl-divergence for logistic and register_kl
        # see
        # - https://pytorch.org/docs/stable/distributions.html#torch.distributions.kl.kl_divergence
        # - https://pytorch.org/docs/stable/distributions.html#torch.distributions.kl.register_kl

        if scale_are_log_variance:
            # Convert log-variance to standard deviation: std = exp(logvar / 2)
            # Note: using exp(logvar * 0.5) instead of sqrt(exp(logvar)) to avoid
            # float32 overflow — exp(x) overflows at x ≈ 88.7, while exp(x * 0.5)
            # only overflows at x ≈ 177.
            scale = torch.exp(scale * 0.5)

        if overwrite_distribution_name:
            distribution_name = overwrite_distribution_name
        else:
            distribution_name = self.distribution_name

        # Note: variables dependance/independance logic is explicitly computed
        # in the loss/forward/deploy methods.
        if distribution_name == "gaussian":
            distribution = dist.Normal(mean, scale, validate_args)
        elif distribution_name == "laplace":
            distribution = dist.Laplace(mean, scale, validate_args)
        elif distribution_name == "student":
            distribution = dist.StudentT(
                df=2, loc=mean, scale=scale, validate_args=validate_args
            )
        elif distribution_name == "logistic":
            # (CRITICAL) ToDo: implement test case
            scale = scale * (np.sqrt(3) / torch.pi)
            distribution = logistic_distribution(a=mean, b=scale)
        else:
            raise NotImplementedError(
                f"Distribution {distribution_name} not implemented yet! Choose between: "
                "gaussian, student, laplace, gaussian_manual or logistic_manual"
            )
        return distribution

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes Gaussian NLL loss.

        It also includes terms for ``max_logvar`` and ``min_logvar`` with small weights,
        with positive and negative signs, respectively.

        This function returns no metadata, so the second output is set to an empty dict.

        Args:
            model_in (tensor): input tensor. The shape must be ``E x B x Id``, or ``B x Id``
                where ``E``, ``B`` and ``Id`` represent ensemble size, batch size, and input
                dimension, respectively.
            target (tensor): target tensor. The shape must be ``E x B x Od``, or ``B x Od``
                where ``E``, ``B`` and ``Od`` represent ensemble size, batch size, and output
                dimension, respectively.

        Returns:
            (tensor): a loss tensor representing the NLL or the MSE/MAE of the model over the given
             input/target. If the model is an ensemble, returns the average over all models.
        """
        # RLRP-751 (task T4): the outer per-feature geometry "drain seam" was
        # REMOVED. The geometry term is now composed INLINE by each model inside
        # its own ``_deterministic_loss`` / ``_probabilistic_loss`` (and the CP
        # scorer), at the same accumulator/reduction stage as every other
        # composite term, via ``FeatureGeometryLossMixin._compose_feature_geometry``.
        if self.deterministic:
            losses, meta = self._deterministic_loss(model_in, target)
        else:
            losses, meta = self._probabilistic_loss(model_in, target)

        return losses, meta

    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the squared error for the model over the given input/target.

        When model is not an ensemble, this is equivalent to
        `F.mse_loss(model(model_in, target), reduction="none")`. If the model is ensemble,
        then return is batched over the model dimension.

        This function returns no metadata, so the second output is set to an empty dict.

        Args:
            model_in (tensor): input tensor. The shape must be ``B x Id``, where `B`` and ``Id``
                batch size, and input dimension, respectively.
            target (tensor): target tensor. The shape must be ``B x Od``, where ``B`` and ``Od``
                represent batch size, and output dimension, respectively.

        Returns:
            (tensor): a tensor with the squared error per output dimension, batched over model.
        """
        assert (
            model_in.ndim == 2 and target.ndim == 2
        ), f"{model_in.ndim=} != 2 and/or {target.ndim=} != 2"

        with torch.inference_mode():
            self.eval()

            pred_mean, _ = self.forward(model_in, use_propagation=False)

            target = target.repeat((self.num_members, 1, 1))

            if self.mae_loss:
                losses = F.l1_loss(pred_mean, target, reduction="none")
            else:
                losses = F.mse_loss(pred_mean, target, reduction="none")

            # .... memory management ..............................................................
            del model_in, pred_mean, target

            return losses, {}

    def sample_propagation_indices(
        self, batch_size: int, _rng: torch.Generator
    ) -> torch.Tensor:
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if batch_size % model_len != 0:
            raise ValueError(
                "To use ExponentialFamilyMLP's ensemble propagation, the batch size must "
                "be a multiple of the number of models in the ensemble."
            )
        # rng causes segmentation fault, see https://github.com/pytorch/pytorch/issues/44714
        return torch.randperm(batch_size, device=self.device)

    def set_elite(self, elite_indices: Sequence[int]):
        if len(elite_indices) != self.num_members:
            self.elite_models = list(elite_indices)

    def save(self, save_dir: Union[str, pathlib.Path]) -> None:
        """Saves the model to the given directory."""
        torch.save(self._save_state_dict(), pathlib.Path(save_dir) / self._MODEL_FNAME)
        return None

    def load(self, load_dir: Union[str, pathlib.Path]) -> None:
        """Loads the model from the given path."""
        model_dict = torch.load(
            pathlib.Path(load_dir) / self._MODEL_FNAME,
            weights_only=False,
            map_location=None if torch.cuda.is_available() else torch.device("cpu"),
        )
        self._load_state_dict(model_dict)
        self.eval()
        return None

    def _save_state_dict(self) -> dict[str, list[int] | dict[str, Any] | Any]:
        model_dict = {
            "state_dict": self.state_dict(),
            "elite_models": self.elite_models,
            "model_config": {
                "in_size": self.in_size,
                "out_size": self.out_size,
                "hid_size": self.hid_size,
                "num_members": self.num_members,  # ensemble size
                "ms_head_num_layers": self.ms_head_num_layers,
                "deterministic": self.deterministic,
                "distribution_name": self.distribution_name,
                "model_use_double_precision": self.model_use_double_precision,
                "residual_form": self.residual_form,
                # RLRP-768: per-region layer-bloc topology signature. Widens the
                # historical `residual_form` guard so a dense / gated / LayerNorm
                # topology mismatch fails loud instead of raising a cryptic
                # state_dict key error at load time.
                "layer_bloc": self.layer_bloc_signature(),
                # RLRP-758 (T8b): the velocity frame the model was trained in.
                "training_frame": self.get_training_frame(),
            },
        }
        return model_dict

    _EXP_FAMILLY_REQUIRED_CONFIG_KEYS: Tuple[str, ...] = (
        "in_size",
        "out_size",
        "hid_size",
        "num_members",
        "ms_head_num_layers",
        "deterministic",
        "distribution_name",
        "model_use_double_precision",
        # RLRP-758 (T8b): required -> a frame-less legacy checkpoint fails loud
        # (per D1: every pre-existing saved model is wrong w.r.t. the new default).
        "training_frame",
    )

    def _load_state_dict(self, model_dict):
        if "model_config" not in model_dict:
            raise KeyError(
                f"{type(self).__name__}._load_state_dict: missing 'model_config' "
                f"entry in model_dict."
            )
        model_config = model_dict["model_config"]
        missing = [
            k for k in self._EXP_FAMILLY_REQUIRED_CONFIG_KEYS if k not in model_config
        ]
        if missing:
            raise KeyError(
                f"{type(self).__name__}._load_state_dict: missing keys in "
                f"model_dict['model_config']: {missing}"
            )

        # RLRP-768: validate the layer-bloc topology signature BEFORE load_state_dict
        # (the dense / gated / LayerNorm topologies differ in parameter names /
        # counts). Newer checkpoints store `layer_bloc`; when absent we fall back
        # to the legacy `residual_form` comparison below.
        loaded_layer_bloc = model_config.get("layer_bloc", None)
        if loaded_layer_bloc is not None:
            current_layer_bloc = self.layer_bloc_signature()
            if loaded_layer_bloc != current_layer_bloc:
                raise ValueError(
                    f"{type(self).__name__}._load_state_dict: layer_bloc topology "
                    f"mismatch. Checkpoint was trained with layer_bloc="
                    f"{loaded_layer_bloc!r} but the model was constructed with "
                    f"layer_bloc={current_layer_bloc!r}. Construct the model with a "
                    f"matching `layer_bloc` / `residual_form` to load this checkpoint "
                    f"(RLRP-768)."
                )

        # Validate residual_form BEFORE load_state_dict: the block topology differs
        # between forms (legacy=1 inner linear, identity_v2=2), so a mismatch would
        # otherwise surface as a cryptic state_dict key error. Checkpoints saved
        # before the switch have no stored key and used the historical legacy form.
        loaded_residual_form = model_config.get(
            "residual_form", "legacy_preact_wrapped"
        )
        if loaded_residual_form != self.residual_form:
            raise ValueError(
                f"{type(self).__name__}._load_state_dict: residual_form mismatch. "
                f"Checkpoint was trained with residual_form={loaded_residual_form!r} "
                f"but the model was constructed with residual_form={self.residual_form!r}. "
                f"Construct the model with the matching residual_form to load this "
                f"checkpoint (use 'legacy_preact_wrapped' for pre-switch checkpoints)."
            )
        self.residual_form = loaded_residual_form

        self.load_state_dict(model_dict["state_dict"])
        self.elite_models = model_dict["elite_models"]

        self.in_size = model_config["in_size"]
        self.out_size = model_config["out_size"]
        self.hid_size = model_config["hid_size"]
        self.num_members = model_config["num_members"]
        self.deterministic = model_config["deterministic"]
        self.ms_head_num_layers = model_config["ms_head_num_layers"]
        self.distribution_name = model_config["distribution_name"]
        self.model_use_double_precision = model_config["model_use_double_precision"]

        # RLRP-758 (T8b): recover + guard the velocity training frame. The stored
        # frame MUST match the frame the (freshly constructed / cfg-configured)
        # model is set to, else a body-trained model could be silently deployed
        # under a world integrator (or vice-versa) — RLRP-755 report §4 hazard.
        stored_training_frame = _validate_training_frame(
            str(model_config["training_frame"])
        )
        active_training_frame = self.get_training_frame()
        if stored_training_frame != active_training_frame:
            raise ValueError(
                f"{type(self).__name__}._load_state_dict: training_frame mismatch. "
                f"Checkpoint was trained with training_frame="
                f"{stored_training_frame!r} but the model is configured for "
                f"training_frame={active_training_frame!r}. Load the checkpoint "
                f"under the matching ms_model.training_frame (RLRP-758 T8b guard)."
            )
        self._training_frame = stored_training_frame
        return None

    # ==== Copied from GaussianMLPExtended ========================================================
    # Copied from GaussianMLPExtended
    def sample_1d(
        self,
        model_input: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        next_state_sampling_size: int = 1,
        rng: Optional[torch.Generator] = None,
        epi_knn=False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        # (CRITICAL) ToDo: implement test case (ref task RLRP-332)
        # (Priority) ToDo: RLRP-354 feat(GaussianMLPExtended): override extend `_forward_ensemble`
        # with 'post_sampling_expectation' logic
        """Samples an output from the model using .

        Note: re-implementation which output ensemble mean and logvariance via the `model_state`

        This method will be used by :class:`ModelEnv` to simulate a transition of the form.
            outputs_t+1, s_t+1 = sample(model_input_t, s_t), where

            - model_input_t: observation and action at time t, concatenated across axis=1.
            - s_t: model state at time t (as returned by :meth:`reset()` or :meth:`sample()`.
            - outputs_t+1: observation and reward at time t+1, concatenated across axis=1.

        The default implementation returns `s_t+1=s_t`.

        :param model_input: the observation and action at.
        :param model_state: the model state st. Must contain a key "propagation_indices" to use
         for uncertainty propagation.
        :param deterministic: if ``True``, the model returns a deterministic "sample"
         (e.g., the mean prediction). Defaults to ``False``.
        :param next_state_sampling_size: Nb of sample to draw if deterministic=False
        :param rng: an optional random number generator to use.
        :param epi_knn: epistemic k-nearest sampling

        :return: predicted observation and model state dictionary.
        """

        ensemble_means, ensemble_logvars = self._forward_propagation(
            model_input, model_state, rng
        )

        # RLRP-761 P1.5 — the sentinel below is a MARKER ("this model has no
        # variance"), NOT a statistic. It must be propagated as such: rescaling it
        # in the P1.1 variance transport would manufacture a per-dimension
        # pseudo-variance that looks meaningful and is not (risk ``Q-A``).
        logvars_are_absent = ensemble_logvars is None
        if ensemble_logvars is None:
            # Quick-hack for casse model has an ensemble of one or is deterministic
            ensemble_logvars = torch.full_like(ensemble_means, self._LOGVAR_MIN_LIMIT)

        model_state["ensemble_means"] = ensemble_means
        model_state["ensemble_logvars"] = ensemble_logvars
        model_state["ensemble_logvars_is_absent"] = logvars_are_absent

        next_obs = self._sample_1d_next_obs_ensemble_decision(
            ensemble_means,
            ensemble_logvars,
            deterministic,
            next_state_sampling_size,
            epi_knn,
        )

        # .... Memory management ..................................................................
        # del ensemble_means, ensemble_logvars, means
        # if not deterministic:
        #     del logvars

        return next_obs, model_state

    def _sample_1d_next_obs_ensemble_decision(
        self,
        ensemble_means: Tensor,
        ensemble_logvars: Tensor,
        deterministic: bool = False,
        next_state_sampling_size: int = 1,
        epi_knn: bool = False,
    ) -> Tensor:
        if ensemble_logvars is None and not deterministic:
            # Quick-hack for casse model has an ensemble of one or is deterministic
            # RLRP-761 P1.5 — second sentinel site. Unlike the ``sample_1d`` one
            # this fill is PURELY LOCAL: it never reaches ``model_state``, so the
            # statistics channel exposed to consumers is unaffected and no
            # ``ensemble_logvars_is_absent`` flag is emitted here. Keep it that way
            # — if this value is ever surfaced, flag it like ``sample_1d`` does.
            ensemble_logvars = torch.full_like(ensemble_means, self._LOGVAR_MIN_LIMIT)

        ensemble_size = self.num_members
        if ensemble_size == 1:
            # Case non-model-ensemble NN
            means = ensemble_means.squeeze(dim=0)
            # RLRP-761 P1.7 / P8 (closes the RLRP-456 marker) — ABSENT-vs-ZERO
            # convention, now settled and uniform across the statistics channel:
            #   * the TENSOR is always returned (never ``None``), because every
            #     upstream consumer is written against a tensor and a ``None``
            #     would move the branch to a dozen call sites;
            #   * "this quantity does not exist" is carried by the SEPARATE
            #     boolean ``model_state["ensemble_logvars_is_absent"]`` set in
            #     :meth:`sample_1d`, NOT by the value itself.
            # This is what lets the P1 variance transport skip the sentinel
            # instead of rescaling a constant into a per-dimension
            # pseudo-variance.
            if not deterministic:
                logvars = ensemble_logvars.squeeze(dim=0)
        elif self.propagation_method == "expectation":
            # Finalize expectation computation step from `_forward_propagation()`
            means = ensemble_means.mean(dim=0)
            logvars = ensemble_logvars.mean(dim=0)
        elif self.propagation_method == "post_sampling_expectation":
            means = ensemble_means
            logvars = ensemble_logvars
        elif epi_knn:
            # (Priority) ToDo: RLRP-389 feat: implement multistep sample method logic

            assert ensemble_logvars.ndim == 1
            logvars_T = ensemble_logvars.reshape(ensemble_logvars.size(0), -1)
            distance_matrix = torch.cdist(logvars_T, logvars_T)

            # Example:
            #   >>> aa = torch.arange(1,6, dtype=float).reshape(5, -1)
            #   tensor([[1.],
            #           [2.],
            #           [3.],
            #           [4.],
            #           [5.]], dtype=torch.float64)
            #   >>> dist_matrix = torch.cdist(aa,aa)
            #   >>> dist_matrix
            #   tensor([[0., 1., 2., 3., 4.],
            #           [1., 0., 1., 2., 3.],
            #           [2., 1., 0., 1., 2.],
            #           [3., 2., 1., 0., 1.],
            #           [4., 3., 2., 1., 0.]], dtype=torch.float64)
            #   # Remove matrix indice row/col (margin)
            #   >>> dist_matrix[1:,1:]
            #   tensor([[0., 1., 2., 3.],
            #           [1., 0., 1., 2.],
            #           [2., 1., 0., 1.],
            #           [3., 2., 1., 0.]], dtype=torch.float64)
            #   >>> tril_indices = torch.tril_indices(4,4,-1)
            #   >>> tril_indices
            #   tensor([[1, 2, 2, 3, 3, 3],
            #           [0, 0, 1, 0, 1, 2]])
            #   >>> dist_matrix[tril_indices[0], tril_indices[1]]
            #   tensor([1., 2., 1., 3., 2., 1.], dtype=torch.float64)

            raise NotImplementedError("ToDo: epi k-nearest ")
        else:
            means = ensemble_means
            logvars = ensemble_logvars

        if deterministic or self.deterministic:
            # (CRITICAL) ToDo: validate averaging over ensemble dim for deterministic sampling
            if means.ndim >= 2 and 1 < ensemble_size == means.shape[0]:
                next_obs = torch.mean(means, dim=0)
            else:
                next_obs = means.squeeze(dim=0)
        else:

            next_obs = self._sample_1d_next_obs_per_ensemble_probabilisitic_decision(
                means, logvars, next_state_sampling_size
            )

            if ensemble_size > 1:
                if self.propagation_method == "post_sampling_expectation":
                    next_obs = torch.mean(next_obs, dim=0)
            else:
                next_obs = next_obs.squeeze(dim=0)
        return next_obs

    def _sample_1d_next_obs_per_ensemble_probabilisitic_decision(
        self,
        means: Tensor,
        logvars: Tensor | None,
        next_state_sampling_size: int = 1,
    ) -> Tensor:
        ensemble_size = self.num_members

        # Sample each ensemble models independantly. Output next obs with shape E X OutDim.
        if (
            self.distribution_name == "gaussian_manual"
            or self.distribution_name == "logistic_manual"
        ):
            stds = torch.sqrt(logvars.exp())

            # Note: nan handling is usefull when using this method early in the training stage
            means = torch.nan_to_num(means)
            # Dtype-aware std floor, finfo-derived (RLRP-750): ``sqrt(16*finfo.tiny)``,
            # i.e. ~4.3e-19 (float32) / ~6.0e-154 (float64) -- orders of magnitude
            # below the former frozen float32-era ``1e-6`` (see precision_bounds), so a
            # numerically-deterministic (minimal-entropy) draw is representable in both
            # dtypes rather than being discarded for low-variance dims.
            stds = torch.nan_to_num(stds).clamp(min=std_floor(self.model_dtype))

            if next_state_sampling_size > 1:
                # Expand means and stds to (N_SAMPLES, ENSEMBLE, BATCH, DIM)
                # This allows torch.randn to broadcast across the new dimension efficiently
                expanded_means = means.unsqueeze(0).expand(
                    next_state_sampling_size, *means.shape
                )
                expanded_stds = stds.unsqueeze(0).expand(
                    next_state_sampling_size, *stds.shape
                )

                next_obs = expanded_means + expanded_stds * torch.randn_like(
                    expanded_means
                )
                # next_obs = torch.mean(next_obs, dim=0)
                # (CRITICAL) ToDo: validate
                next_obs = fast_distribution_mode_approximation(next_obs, ensemble_size)
            else:
                next_obs = means + stds * torch.randn_like(means)

            del stds
        else:
            # Note: nan handling is usefull when using this method early in the training stage
            means = torch.nan_to_num(means)
            # RLRP-750: migrate the logvar clamp onto the consolidated dtype-aware
            # precision bounds instead of the local magic numbers. Lower bound is
            # ``log(variance_floor(dtype))`` (finfo-derived, ~log(1.9e-37)=-84.5 for
            # float32) -- far below the former hard-coded logvar floor ``EPS=1e-6``
            # (which pinned variance >= ~1.0). Upper bound is the consolidated
            # ``logvar_safe_max()`` (== LOGVAR_SAFE_MAX = 20 -> scale = exp(10) ~ 22026,
            # safe for float32 arithmetic).
            _logvar_floor = math.log(variance_floor(self.model_dtype))
            logvars = torch.nan_to_num(logvars).clamp(
                min=_logvar_floor, max=logvar_safe_max(self.model_dtype)
            )
            distribution = self._to_distribution(
                means, logvars, scale_are_log_variance=True
            )
            if next_state_sampling_size > 1:
                next_obs = distribution.sample([next_state_sampling_size])
                # next_obs = torch.mean(next_obs, dim=0)
                # (CRITICAL) ToDo: validate
                next_obs = fast_distribution_mode_approximation(next_obs, ensemble_size)

            else:
                next_obs = distribution.sample()
        return next_obs

    def _forward_propagation(
        self,
        model_input: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Copied from GaussianMLPExtended

        if self.propagation_method in ["expectation", "post_sampling_expectation"]:
            # Reproduce the expectation propagation logic but without the applying the mean
            # at the last step i.e., output E x B x Od instead of B x Od

            assert model_input.ndim == 2
            model_len = (
                len(self.elite_models) if self.elite_models is not None else len(self)
            )
            if model_input.shape[0] % model_len != 0:
                raise ValueError(
                    "ExponentialFamilyMLP ensemble requires batch size to be a multiple of "
                    "the "
                    f"number of models. Current batch size is {model_input.shape[0]} for "
                    f"{model_len} models."
                )

            model_input = model_input.unsqueeze(0)
            ensemble_means, ensemble_logvars = self._default_forward(
                model_input, only_elite=True
            )

            # # Merge batch dim
            # ensemble_means = ensemble_means.mean(dim=1, keepdim=True)
            # ensemble_logvars = ensemble_logvars.mean(dim=1, keepdim=True)

            # # Compute expectation
            # means = ensemble_means.mean(dim=0)
            # logvars = ensemble_logvars.mean(dim=0)

        elif self.propagation_method is None:

            # Note: Equivalent to running '_default_forward()' method with 'only_elite=True'
            self._maybe_toggle_layers_use_only_elite(only_elite=True)
            ensemble_means, ensemble_logvars = self.forward(
                model_input, use_propagation=False
            )
            self._maybe_toggle_layers_use_only_elite(only_elite=True)

        else:

            ensemble_means, ensemble_logvars = self.forward(
                model_input,
                use_propagation=True,
                propagation_indices=model_state["propagation_indices"],
            )

            # # Merge batch dim
            # ensemble_means = ensemble_means.mean(dim=1, keepdim=True)
            # ensemble_logvars = ensemble_logvars.mean(dim=1, keepdim=True)

        return ensemble_means, ensemble_logvars
