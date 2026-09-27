# coding=utf-8
"""Centralized per-environment feature-handling contract.

Permanent framework module. Introduced by stage 1 (action S1.1) of the
Per-Environment Feature Handling ``.junie`` plan
(``rlrp-736-per-environment-feature-handling-plan-20260711.md``, YouTrack
RLRP-736).

The contract declares, for each named/contiguous block of the observation or
action vector, how it is handled at the three integration levels: the
normalizer level (A), the model level (B) and the deploy/rollout level (C).
It defaults to identity so that configurations without a ``feature_handler``
reproduce the pre-plan behaviour bit-for-bit.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch


class NormStrategy(str, enum.Enum):
    """Per-block normalization strategy resolved by the normalizer facade."""

    INHERIT = "inherit"  # use the vector-global normalizer_type (back-compat)
    IDENTITY = "identity"  # disable normalization for the block (pass-through)
    UNIT_NORM = "unit_norm"  # project the block onto the unit sphere (quaternion)
    #: Plain standardization ``(x - mean)/std``, BYPASSING any robust warping
    #: (soft-clipping / quantile binning) implied by the vector-global
    #: ``normalizer_type``. RLRP-761 S1.1: this is the strategy a block should
    #: use when the intent is "never winsorize this" — :attr:`IDENTITY` used to
    #: be the only way to say that, and it silently also meant "never scale
    #: this", which starved the whole action block (``dt`` included).
    ZSCORE = "zscore"
    WINSORIZED = "winsorized"
    QUANTILE = "quantile"

    def is_base_disabled(self) -> bool:
        """Whether the *base* statistical normalizer must be bypassed for the dim.

        RLRP-761 S1.4 — single source of truth replacing the hard-coded
        ``{IDENTITY, UNIT_NORM}`` set that used to be inlined at every call
        site. ``True`` means ``normalize_dims[d] = False``: the base passes the
        raw value through and either nothing further happens
        (:attr:`IDENTITY`) or the :class:`StrategyAwareNormalizer` wrapper takes
        over (:attr:`UNIT_NORM`, :attr:`ZSCORE`).
        """
        return self in (
            NormStrategy.IDENTITY,
            NormStrategy.UNIT_NORM,
            NormStrategy.ZSCORE,
        )


class FeatureKind(str, enum.Enum):
    """Semantic kind of a feature block."""

    SCALAR = "scalar"
    LINEAR_VELOCITY = "linear_velocity"
    ANGULAR_VELOCITY = "angular_velocity"
    QUATERNION = "quaternion"
    #: RLRP-761 S5.4 -- 3-D yaw-invariant orientation: the gravity direction
    #: expressed in the body frame (``gravity.{x,y,z}``, a unit vector).
    #: Alternative to :attr:`QUATERNION` for a ground/aerial vehicle whose
    #: dynamics are yaw-invariant: it carries roll/pitch only, is free of the
    #: quaternion double cover (hence needs NO temporal sign-continuity pass)
    #: and lives on ``S^2`` rather than ``S^3``. Normalized with
    #: :attr:`NormStrategy.UNIT_NORM`.
    GRAVITY_DIRECTION = "gravity_direction"
    BOUNDED_CONTROL = "bounded_control"
    DT = "dt"


class InternalOrientationRep(str, enum.Enum):
    """Model-internal orientation representation for the ACTIVE orientation block.

    The external input/output contract is never changed by this enum: it stays
    whatever the declared orientation block is — a 4-D quaternion
    (:attr:`FeatureKind.QUATERNION`) or a 3-D unit gravity direction
    (:attr:`FeatureKind.GRAVITY_DIRECTION`). This only selects the representation
    used *inside* the model.

    Every rep is **block-scoped**: a quaternion-block rep cannot consume a 3-D
    direction and an ``S²`` rep cannot consume a 4-D quaternion. The declared
    compatibility is :data:`_ORIENTATION_REP_SUPPORTED_KINDS`, validated at setup
    through :func:`orientation_rep_supported_kinds` (RLRP-796 ``D-796-S1-5-A``).

    Two neutral (no continuous-rep switch, 4-D passthrough) quaternion variants
    differ only in how the emitted 4-D slot is finalized:

    - ``QUATERNIONLEGACY`` is the **actual legacy contract** (pre-RLRP-736): the
      raw 4-D output slot is returned *un-normalized* (the network is trusted to
      emit a near-unit quaternion; no by-construction guarantee).
    - ``QUATERNION`` is the (default) **improvement** over the legacy contract:
      the raw 4-D slot is unit-normalized so the emitted quaternion is valid by
      construction, while staying a cheap, continuous-rep-free passthrough.

    ``QUATERNION`` is the neutral default; ``QUATERNIONLEGACY`` restores the
    exact pre-RLRP-736 un-normalized behavior for reproducibility / ablation.

    The over-parameterized continuous heads :attr:`NINE_D_SVD` (R^9+SVD) and
    :attr:`SIXD` (R^6+GSO) are **abandoned legacy prototypes**: their
    encoder/head branches are kept only for reproducibility of archived
    experiments and are **not** recommended live options. The live / default
    representation is :attr:`QUATERNION` (unit-normalized). The literature
    ranking that originally motivated them (Geist et al. 2024, "Learning with
    3D Rotations: a Hitchhiker's Guide to SO(3)", ICML 2024; Chen et al. 2023,
    arXiv:2312.00462) is retained here as historical context only and does not
    reflect the current recommendation.

    .. deferred:: :attr:`SO3_RELATIVE` is **DEFERRED** (RLRP-736).
        The small-angle ``q_ref ⊗ Exp(δ)`` tangent representation is only a
        good option when per-step rotations are demonstrably small (Geist et
        al. 2024). Our primary target is precisely the opposite regime — motion
        dynamics under **adverse conditions** (aggressive driving and adverse
        environmental conditions) where per-step rotations are large (high
        angular rates, spins, saturated ``dt``). In that regime the small-angle
        premise breaks (absolute ``Log`` / near-π discontinuity, risk R-B3).
        The over-parameterized continuous heads :attr:`NINE_D_SVD` /
        :attr:`SIXD` are abandoned legacy prototypes (see above) and are **not**
        a recommended alternative; the live / default representation is
        :attr:`QUATERNION` (normalized). The already-landed encoder/head
        branches are kept for reproducibility of archived experiments, but the
        representation is **not** on the RLRP-736 critical path and its remaining
        wiring (per-step ``q_ref`` plumbing) is deferred — do not select it for
        the adverse-condition robotic-3D models.
    """

    QUATERNION = "quaternion"
    #: Actual legacy contract (pre-RLRP-736): raw 4-D slot returned un-normalized
    #: (no by-construction unit guarantee). Prefer QUATERNION (normalized). See docstring.
    QUATERNIONLEGACY = "quaternion_legacy"
    #: DEFERRED (RLRP-736) — small-angle only; poor fit for the adverse-condition
    #: (large per-step rotation) target. The live / default representation is
    #: QUATERNION (normalized). See docstring.
    SO3_RELATIVE = "so3_relative"
    #: ABANDONED LEGACY PROTOTYPE — R^6 + Gram-Schmidt continuous head. Kept only
    #: for reproducibility of archived experiments; NOT a recommended live option.
    #: The live / default representation is QUATERNION (normalized). See docstring.
    SIXD = "sixd"
    #: ABANDONED LEGACY PROTOTYPE — R^9 + SVD-projection continuous head. Kept only
    #: for reproducibility of archived experiments; NOT a recommended live option.
    #: The live / default representation is QUATERNION (normalized). See docstring.
    NINE_D_SVD = "nine_d_svd"
    #: RLRP-796 ``D-793-1-C`` — GRAVITY_DIRECTION block only. `S^2`-native identity:
    #: external 3-D == internal 3-D, so the trunk delta is 0 and an existing gravity
    #: checkpoint still loads. The decode is the L2 projection onto `S^2`, making the
    #: emitted direction unit BY CONSTRUCTION (the `S^2` analogue of what QUATERNION
    #: does on `S^3`). Largely redundant with the shipped UNIT_NORM + AR projection
    #: path; it exists as the low-risk carrier of the width plumbing and as the
    #: control arm for S2_TANGENT. Opt-in; no shipped config selects it.
    S2_IDENTITY = "s2_identity"
    #: RLRP-796 ``D-793-1-D`` — GRAVITY_DIRECTION block only. `S^2`-native TANGENT:
    #: the trunk emits 2 raw numbers per OUTPUT slot (trunk delta -1), which are
    #: ``exp``-mapped onto `S^2` at the PREVIOUS step's direction (ruling
    #: ``D-793-3-A``), so the prediction lives on the true 2-DoF manifold instead of
    #: an ambient 3-vector plus a unit constraint. The INPUT encode deliberately
    #: stays the ABSOLUTE 3-D direction (ruling ``D-796-S1-2-A``): a relative input
    #: encoding would destroy the absolute tilt at the start of every window, so the
    #: network could no longer know it is inverted. Near the chart tear the tangent
    #: norm is clamped and warned once per run (``D-796-S1-1-A``). A NEW checkpoint
    #: family (the trunk widths differ). Opt-in; no shipped config selects it.
    S2_TANGENT = "s2_tangent"


#: Which orientation BLOCK each internal representation can consume (RLRP-796
#: ``D-796-S1-5-A``). Declared as a table rather than an ad-hoc branch so the
#: mismatch is caught symmetrically in BOTH directions at setup time: a
#: quaternion-family rep on a gravity observation space, and (once the ``S²``
#: reps land) an ``S²`` rep on an attitude observation space.
#:
#: The two NEUTRAL reps (``QUATERNION`` / ``QUATERNIONLEGACY``) are passthroughs
#: that never derive a slot, so they are compatible with every block by
#: construction — they are the documented fallback the mismatch message points at.
_ORIENTATION_REP_SUPPORTED_KINDS = {
    InternalOrientationRep.QUATERNION: frozenset(
        {FeatureKind.QUATERNION, FeatureKind.GRAVITY_DIRECTION}
    ),
    InternalOrientationRep.QUATERNIONLEGACY: frozenset(
        {FeatureKind.QUATERNION, FeatureKind.GRAVITY_DIRECTION}
    ),
    InternalOrientationRep.SO3_RELATIVE: frozenset({FeatureKind.QUATERNION}),
    InternalOrientationRep.SIXD: frozenset({FeatureKind.QUATERNION}),
    InternalOrientationRep.NINE_D_SVD: frozenset({FeatureKind.QUATERNION}),
    InternalOrientationRep.S2_IDENTITY: frozenset({FeatureKind.GRAVITY_DIRECTION}),
    InternalOrientationRep.S2_TANGENT: frozenset({FeatureKind.GRAVITY_DIRECTION}),
}

#: The `S^2`-native reps (RLRP-796 Stage 1). They consume/emit a 3-D body-frame
#: gravity DIRECTION, never a quaternion, so they take the ``*_gravity_*`` encode /
#: decode path in ``orientation_heads`` rather than the ``*_wxyz_*`` one.
_S2_ORIENTATION_REPS = frozenset(
    {InternalOrientationRep.S2_IDENTITY, InternalOrientationRep.S2_TANGENT}
)


def is_s2_orientation_rep(rep: InternalOrientationRep) -> bool:
    """Whether ``rep`` operates on the `S^2` gravity-direction block.

    :param rep: The model-internal orientation representation.
    :return: ``True`` for an `S^2`-native rep, ``False`` for a quaternion-block one.
    """
    return InternalOrientationRep(rep) in _S2_ORIENTATION_REPS


def orientation_rep_supported_kinds(
    rep: InternalOrientationRep,
) -> frozenset:
    """Orientation block kinds ``rep`` can consume (RLRP-796 ``D-796-S1-5-A``).

    :param rep: The requested model-internal orientation representation.
    :return: The :class:`FeatureKind` values whose block ``rep`` can encode/decode.
    :raises KeyError: For a representation absent from the declared table (a new
        enum member must declare its block — failing loud here is deliberate).
    """
    return _ORIENTATION_REP_SUPPORTED_KINDS[InternalOrientationRep(rep)]


def _identity(value: torch.Tensor) -> torch.Tensor:
    """Return the input tensor unchanged."""
    return value


@dataclass
class FeatureGroupSpec:
    """Describe one named feature block and how it is handled at all levels.

    :ivar name: Human-readable group name.
    :ivar kind: Semantic :class:`FeatureKind` of the block.
    :ivar dim_names: Ordered feature dimension names (e.g. ``('attitude.w',
        ..., 'attitude.z')``) as they appear in ``obs_dims``/``act_dims``.
    :ivar indices: Resolved positional indices in the obs/act vector.
    :ivar norm_strategy: Per-block :class:`NormStrategy`.
    :ivar encode: Transition-boundary encode callable (identity by default).
        Under the symmetric external contract this is identity for all
        Stage-2 groups; the orientation representation switch is model-internal
        (see :attr:`internal_orientation_rep`), never applied here.
    :ivar decode: Transition-boundary decode callable (identity by default).
    :ivar internal_orientation_rep: Model-internal orientation representation
        (``QUATERNION`` group only). The external I/O stays a 4-D quaternion.
    :ivar loss_term: Optional additive per-block train loss (raw block space).
    :ivar is_absolute_target: When ``True`` the block is excluded from the
        Euclidean delta target (goes into ``no_delta_list``).
    :ivar enforce_continuity: When ``True`` (``QUATERNION`` block) the ingestion
        seam enforces temporal sign continuity ``<q_t, q_{t-1}> >= 0`` along the
        causal trajectory (S3.1). Requires ordered trajectories.
    """

    name: str
    kind: FeatureKind
    dim_names: Sequence[str]
    indices: Tuple[int, ...] = field(default_factory=tuple)
    norm_strategy: NormStrategy = NormStrategy.INHERIT
    encode: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    decode: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    internal_orientation_rep: InternalOrientationRep = InternalOrientationRep.QUATERNION
    loss_term: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    is_absolute_target: bool = False
    enforce_continuity: bool = False

    def resolve(self, dim_order: Sequence[str]) -> "FeatureGroupSpec":
        """Resolve :attr:`indices` from the given obs/act dimension order."""
        order = list(dim_order)
        missing = [name for name in self.dim_names if name not in order]
        if missing:
            raise ValueError(
                f"FeatureGroupSpec '{self.name}': dim_names {missing} not found "
                f"in dimension order {order}"
            )
        self.indices = tuple(order.index(name) for name in self.dim_names)
        return self

    @property
    def effective_encode(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return the encode callable, defaulting to identity."""
        return self.encode if self.encode is not None else _identity

    @property
    def effective_decode(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return the decode callable, defaulting to identity."""
        return self.decode if self.decode is not None else _identity


@dataclass
class EnvFeatureHandler:
    """Compose the ordered feature specs for one environment.

    The handler exposes the operations consumed by the three integration
    levels. All operations default to identity/no-op so an all-``INHERIT``
    handler is behaviourally neutral (bit-exact back-compat).

    :ivar obs_groups: Ordered specs covering the observation vector.
    :ivar act_groups: Ordered specs covering the action vector.
    :ivar obs_dims: Resolved observation dimension names.
    :ivar act_dims: Resolved action dimension names.
    """

    obs_groups: List[FeatureGroupSpec]
    act_groups: List[FeatureGroupSpec]
    obs_dims: Optional[List[str]] = None
    act_dims: Optional[List[str]] = None

    @classmethod
    def from_groups(
        cls,
        obs_groups: Sequence[FeatureGroupSpec],
        act_groups: Sequence[FeatureGroupSpec],
        obs_dims: Sequence[str],
        act_dims: Sequence[str],
    ) -> "EnvFeatureHandler":
        """Build a handler and resolve every group's indices."""
        obs_dims = list(obs_dims)
        act_dims = list(act_dims)
        obs_groups = [group.resolve(obs_dims) for group in obs_groups]
        act_groups = [group.resolve(act_dims) for group in act_groups]
        return cls(
            obs_groups=list(obs_groups),
            act_groups=list(act_groups),
            obs_dims=obs_dims,
            act_dims=act_dims,
        )

    # --- (A) normalizer level ------------------------------------------------
    def per_dim_strategy(self) -> Dict[str, List[NormStrategy]]:
        """Return the per-dimension :class:`NormStrategy` for obs and act.

        The returned lists are aligned with :attr:`obs_dims` / :attr:`act_dims`
        and default to :attr:`NormStrategy.INHERIT` for any unassigned index.
        """
        return {
            "obs": self._per_dim_strategy(self.obs_groups, len(self.obs_dims or [])),
            "act": self._per_dim_strategy(self.act_groups, len(self.act_dims or [])),
        }

    def build_normalizer_kwargs(self) -> Dict[str, object]:
        """Produce ``feature_dim_names`` + ``normalize_dims`` + ``per_dim_strategy``.

        ``normalize_dims`` maps every dimension name to ``False`` when the base
        statistical normalizer must be **disabled** for it — see
        :meth:`NormStrategy.is_base_disabled`: :attr:`NormStrategy.IDENTITY`
        (pure pass-through), :attr:`NormStrategy.UNIT_NORM` (the base passes the
        raw block through so the :class:`StrategyAwareNormalizer` wrapper can
        project it onto the unit sphere) and :attr:`NormStrategy.ZSCORE` (idem,
        so the wrapper can apply the plain standardization) — and ``True``
        otherwise.

        ``per_dim_strategy`` is the concatenated ``obs + act`` list of strategy
        string values consumed by the facade to build the block-strategy wrapper
        (:attr:`NormStrategy.UNIT_NORM` and, since RLRP-761 S1.4,
        :attr:`NormStrategy.ZSCORE`).

        With every group :attr:`NormStrategy.INHERIT` the result is
        byte-identical to the legacy behaviour (all ``normalize_dims`` ``True``,
        no ``unit_norm`` → no wrapper).
        """
        obs_dims = list(self.obs_dims or [])
        act_dims = list(self.act_dims or [])
        strategy = self.per_dim_strategy()
        normalize_dims: Dict[str, bool] = {}
        for name, strat in zip(obs_dims, strategy["obs"]):
            normalize_dims[name] = not strat.is_base_disabled()
        for name, strat in zip(act_dims, strategy["act"]):
            normalize_dims[name] = not strat.is_base_disabled()
        per_dim_strategy = [strat.value for strat in strategy["obs"]] + [
            strat.value for strat in strategy["act"]
        ]
        return {
            "feature_dim_names": obs_dims + act_dims,
            "normalize_dims": normalize_dims,
            "per_dim_strategy": per_dim_strategy,
        }

    def feature_dim_names(self) -> List[str]:
        """Return the canonical ordered ``obs + act`` feature dimension names.

        RLRP-736 S1.6 (consolidate/deprecate). This is the single source of
        truth from which the hand-maintained ``feature_dim_names`` duplicates in
        ``src/launcher/configs/simulator/softwinsorization/*.yaml`` are derived
        (and validated against via :meth:`validate_feature_dim_names`), so the
        concatenation is generated + checked rather than kept by hand.
        """
        return list(self.obs_dims or []) + list(self.act_dims or [])

    def validate_feature_dim_names(
        self, feature_dim_names: Sequence[str]
    ) -> None:
        """Validate a config-provided ``feature_dim_names`` against the handler.

        Raises :class:`ValueError` when the provided ordered concatenation does
        not exactly match the handler-derived :meth:`feature_dim_names` (order
        included). This lets tooling (e.g.
        ``src/tools/dataset_tools/analyze_winsorizer_config.py``) treat the
        hand-written ``softwinsorization/*.yaml`` lists as generated artifacts
        checked against the centralized handler.

        :param feature_dim_names: The ordered ``obs + act`` names to validate.
        :raises ValueError: When the names do not match (with a precise diff).
        """
        expected = self.feature_dim_names()
        provided = list(feature_dim_names)
        if provided == expected:
            return None
        details = [f"length {len(provided)} != expected {len(expected)}"] if len(
            provided
        ) != len(expected) else []
        for idx, (got, exp) in enumerate(zip(provided, expected)):
            if got != exp:
                details.append(f"[{idx}] '{got}' != expected '{exp}'")
        raise ValueError(
            "feature_dim_names mismatch vs handler-derived names: "
            + "; ".join(details)
            + f" (provided={provided}, expected={expected})"
        )

    # --- (B) model level -----------------------------------------------------
    def obs_process_fn(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply each observation group's ``encode`` (identity by default).

        This is the single transition-boundary application point for group
        ``encode``; the model<->env adapters stay identity and do not re-apply
        it. Encodes are width-preserving under the symmetric external contract.
        """
        return self._apply_group_transforms(
            obs, self.obs_groups, lambda group: group.effective_encode
        )

    def decode(self, pred: torch.Tensor) -> torch.Tensor:
        """Apply each observation group's ``decode`` (identity by default)."""
        return self._apply_group_transforms(
            pred, self.obs_groups, lambda group: group.effective_decode
        )

    def internal_orientation_rep(self) -> InternalOrientationRep:
        """Return the ``QUATERNION`` group's internal orientation representation.

        Defaults to :attr:`InternalOrientationRep.QUATERNION` (neutral) when no
        quaternion group is present.
        """
        for group in self.obs_groups:
            if group.kind is FeatureKind.QUATERNION:
                return group.internal_orientation_rep
        return InternalOrientationRep.QUATERNION

    def no_delta_list(self) -> List[int]:
        """Return the observation indices flagged as absolute targets."""
        indices: List[int] = []
        for group in self.obs_groups:
            if group.is_absolute_target:
                indices.extend(group.indices)
        return sorted(indices)

    def extra_loss(
        self, pred: torch.Tensor, target: torch.Tensor, keep_batch: bool = False
    ) -> torch.Tensor:
        """Sum the per-group ``loss_term`` contributions.

        Returns a zero scalar (matching ``pred`` dtype/device) when no group
        defines a ``loss_term`` so the shared loss seam stays bit-exact.

        ``keep_batch`` (introduced by task T3 of the RLRP-751 feature-geometry
        composite-loss auto-weighting `.junie` plan
        (`rlrp-751-feature-geometry-composite-loss-auto-weighting-plan-20260723.md`)):
        when ``True`` the flag is threaded to each group ``loss_term`` so the
        leaf callable reduces the feature axis only (keeping the leading batch
        dims and a trailing size-1 axis), yielding an ``(..., 1)`` contribution
        the composite auto-weighting module can consume. ``False`` (default)
        keeps the historical scalar reduction.
        """
        total: Optional[torch.Tensor] = None
        for group in self.obs_groups:
            if group.loss_term is None:
                continue
            block_pred = self._select_group_block(pred, group)
            block_target = self._select_group_block(target, group)
            if keep_batch:
                # Batch-preserving reduction is only requested by the composite
                # auto-weighting path, whose active groups define a keep-batch-aware
                # ``loss_term`` (the quaternion geodesic/chordal callables).
                contribution = group.loss_term(
                    block_pred, block_target, keep_batch=True
                )
            else:
                # Legacy scalar reduction: call with the historical 2-arg signature
                # so custom / user-supplied ``loss_term`` callables that predate the
                # ``keep_batch`` flag keep working unchanged.
                contribution = group.loss_term(block_pred, block_target)
            total = contribution if total is None else total + contribution
        if total is None:
            return pred.new_zeros(pred.shape[:-1] + (1,)) if keep_batch else (
                pred.new_zeros(())
            )
        return total

    # --- (C) deploy/rollout level -------------------------------------------
    def deploy_reconstruct(self, pred_obs: torch.Tensor) -> torch.Tensor:
        """Per-group deploy fixups before the env-specific integrator.

        Identity by default; groups override via :attr:`FeatureGroupSpec.decode`.
        """
        return self.decode(pred_obs)

    # --- internals -----------------------------------------------------------
    def _select_group_block(
        self, vector: torch.Tensor, group: FeatureGroupSpec
    ) -> torch.Tensor:
        """``vector[..., group.indices]`` without a per-call host->device index upload.

        RLRP-786 (``perf_RLRP-786_cuda_graph_train_step_plan_20260919.md``, Step 3): the former
        ``vector[..., list(indices)]`` built a CPU index tensor and copied it to the device at EVERY
        call -- a pageable H2D copy that is illegal while a CUDA graph is being recorded (and a
        needless sync-prone upload in eager). Contiguous groups (the common quaternion block) are
        a plain slice (a view, no kernel); ragged groups use an index tensor cached per device.
        Values are identical to the advanced-indexing result.
        """
        indices = tuple(group.indices)
        if indices and all(b == a + 1 for a, b in zip(indices, indices[1:])):
            return vector[..., indices[0] : indices[-1] + 1]
        cache = self.__dict__.setdefault("_group_index_cache", {})
        key = (indices, vector.device)
        index = cache.get(key)
        if index is None:
            index = torch.tensor(indices, dtype=torch.long, device=vector.device)
            cache[key] = index
        return vector.index_select(-1, index)

    @staticmethod
    def _per_dim_strategy(
        groups: Sequence[FeatureGroupSpec], size: int
    ) -> List[NormStrategy]:
        """Expand per-group strategies onto a per-dimension list of length ``size``."""
        strategy = [NormStrategy.INHERIT] * size
        for group in groups:
            for index in group.indices:
                strategy[index] = group.norm_strategy
        return strategy

    @staticmethod
    def _apply_group_transforms(
        vector: torch.Tensor,
        groups: Sequence[FeatureGroupSpec],
        transform_getter: Callable[
            [FeatureGroupSpec], Callable[[torch.Tensor], torch.Tensor]
        ],
    ) -> torch.Tensor:
        """Apply width-preserving per-group transforms onto a copy of ``vector``."""
        needs_copy = any(group.encode is not None for group in groups) or any(
            group.decode is not None for group in groups
        )
        if not needs_copy:
            return vector
        out = vector.clone()
        for group in groups:
            transform = transform_getter(group)
            if transform is _identity:
                continue
            index = list(group.indices)
            transformed = transform(vector[..., index])
            if transformed.shape[-1] != len(index):
                raise ValueError(
                    f"FeatureGroupSpec '{group.name}': transform changed the "
                    f"block width ({len(index)} -> {transformed.shape[-1]}); "
                    f"transition-boundary transforms must be width-preserving "
                    f"(representation switches are model-internal)."
                )
            out[..., index] = transformed
        return out
