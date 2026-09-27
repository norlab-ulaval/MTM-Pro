# coding=utf-8
"""Per-environment :class:`EnvFeatureHandler` factory builders.

Permanent framework module. Introduced by stage 1 (action S1.1) of the
Per-Environment Feature Handling ``.junie`` plan
(``rlrp-736-per-environment-feature-handling-plan-20260711.md``, YouTrack
RLRP-736).

Builders are selected per simulator configuration through Hydra ``_target_``
(mirroring ``flat_container_to_nested`` / ``deploy_rollout_post_processing``).
Each builder reads ``cfg.environment.obs_dims`` / ``act_dims`` and returns a
resolved :class:`EnvFeatureHandler`. The default builder yields a behaviourally
neutral (all ``SCALAR`` / ``INHERIT``) handler.
"""
from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence

from numpy import dtype, generic, ndarray
from omegaconf import omegaconf
from torch import Tensor

from pipeline.pipeline_utils.robotic_env_pipeline_utils.utils import (
    quaternion_normalize,
)
from tools.feature_handling_tools.feature_spec import (
    EnvFeatureHandler,
    FeatureGroupSpec,
    FeatureKind,
    NormStrategy,
)
from tools.feature_handling_tools.orientation_heads import (
    gravity_direction_loss,
    rotation_chordal_loss,
    rotation_geodesic_loss,
)


def _read_dims(cfg) -> tuple:
    """Return ``(obs_dims, act_dims)`` as plain lists from a config object.

    Resilient to configs that do not declare ``environment.obs_dims`` /
    ``environment.act_dims`` (RLRP-736: many ERLL / launcher test configs run in
    OmegaConf *struct* mode without those keys). A missing block resolves to an
    empty list, which yields a behaviourally neutral (empty) handler — the same
    as the pre-plan "no handler" path — instead of raising
    ``ConfigAttributeError``.
    """
    from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist

    obs_dims = (
        list(cfg.environment.obs_dims)
        if is_cfg_key_exist(cfg, "environment.obs_dims")
        else []
    )
    act_dims = (
        list(cfg.environment.act_dims)
        if is_cfg_key_exist(cfg, "environment.act_dims")
        else []
    )
    return obs_dims, act_dims


def _read_explicit_shape(cfg, key: str) -> Optional[list]:
    """Return ``cfg.environment.<key>`` as a plain list, or ``None`` when absent."""
    from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist

    full_key = f"environment.{key}"
    if is_cfg_key_exist(cfg, full_key):
        value = getattr(cfg.environment, key)
        return None if value is None else list(value)
    return None


def _resolve_shape(
    cfg,
    key: str,
    dims: Sequence[str],
    data_shape: Optional[Sequence[int]],
) -> list:
    """Shared body for :func:`resolve_obs_shape` / :func:`resolve_act_shape`.

    RLRP-736 shape-key removal (2026-07-17). The single source of truth for the
    single-step feature WIDTH is:
      - ROBOTIC envs: the declared dimension names (``[len(dims)]``).
      - MATH envs (no dims): the loaded trajectory / replay-buffer DATA, passed by
        the caller as ``data_shape``.
    An explicit ``environment.<key>`` may still be present (legacy / not-yet-removed
    config); when both an authoritative source AND an explicit key are present they
    must AGREE (fail-loud on drift — kills the same class as the ``260 != 200`` bug).

    :param cfg: The resolved experiment configuration.
    :param key: ``"obs_shape"`` or ``"act_shape"``.
    :param dims: The declared dimension names (``obs_dims`` / ``act_dims``); empty
        for math envs.
    :param data_shape: The data-derived single-step shape (math source of truth), or
        ``None`` when not available (e.g. before any data is loaded).
    :return: The resolved single-step shape as a one-element ``[width]`` list.
    """
    explicit = _read_explicit_shape(cfg, key)
    resolved: Optional[list] = None
    if dims:
        resolved = [len(dims)]
    elif data_shape is not None:
        resolved = [int(data_shape[-1])]

    if resolved is None:
        if explicit is None:
            raise ValueError(
                f"Cannot resolve environment.{key}: no dimension names, no "
                f"data-derived shape, and no explicit key are available."
            )
        return explicit

    if explicit is not None and [int(s) for s in explicit] != [
        int(s) for s in resolved
    ]:
        raise ValueError(
            f"environment.{key} mismatch: explicit config value {explicit} "
            f"disagrees with the authoritative source {resolved} (from "
            f"{'dim names' if dims else 'loaded data'}). Remove the stale "
            f"explicit key or align it (RLRP-736 shape-key removal)."
        )
    return resolved


def resolve_obs_shape(cfg, *, data_shape: Optional[Sequence[int]] = None) -> list:
    """Resolve the single-step observation shape (RLRP-736 shape-key removal).

    ROBOTIC: ``[len(obs_dims)]``. MATH: the data-derived ``data_shape``. Falls back
    to a still-present explicit ``environment.obs_shape`` and fail-loud-checks
    agreement. See :func:`_resolve_shape`.
    """
    obs_dims, _ = _read_dims(cfg)
    return _resolve_shape(cfg, "obs_shape", obs_dims, data_shape)


def resolve_act_shape(cfg, *, data_shape: Optional[Sequence[int]] = None) -> list:
    """Resolve the single-step action shape (RLRP-736 shape-key removal).

    ROBOTIC: ``[len(act_dims)]``. MATH: the data-derived ``data_shape``. Falls back
    to a still-present explicit ``environment.act_shape`` and fail-loud-checks
    agreement. See :func:`_resolve_shape`.
    """
    _, act_dims = _read_dims(cfg)
    return _resolve_shape(cfg, "act_shape", act_dims, data_shape)


def _scalar_groups(dim_names: Sequence[str]) -> List[FeatureGroupSpec]:
    """Return one ``SCALAR`` / ``INHERIT`` group per dimension name."""
    return [
        FeatureGroupSpec(name=name, kind=FeatureKind.SCALAR, dim_names=(name,))
        for name in dim_names
    ]


def build_scalar_handler_from_dims(
    obs_dims: Sequence[str], act_dims: Sequence[str]
) -> EnvFeatureHandler:
    """Build a neutral (all ``SCALAR`` / ``INHERIT``) handler from plain dim lists.

    Config-object-free variant of :func:`build_default_scalar_handler` for
    tooling that already holds the resolved ``obs_dims`` / ``act_dims`` lists
    (e.g. ``src/tools/dataset_tools/analyze_winsorizer_config.py``, RLRP-736
    S1.6), where no Hydra config object is available.
    """
    obs_dims = list(obs_dims)
    act_dims = list(act_dims)
    return EnvFeatureHandler.from_groups(
        obs_groups=_scalar_groups(obs_dims),
        act_groups=_scalar_groups(act_dims),
        obs_dims=obs_dims,
        act_dims=act_dims,
    )


def build_default_scalar_handler(cfg) -> EnvFeatureHandler:
    """Build a neutral handler: every dimension is a ``SCALAR`` / ``INHERIT`` group.

    This is the fallback used when a configuration declares no explicit
    ``feature_handler``. It reproduces the pre-plan behaviour bit-for-bit.
    """
    obs_dims, act_dims = _read_dims(cfg)
    return build_scalar_handler_from_dims(obs_dims, act_dims)


def build_math_env_handler(cfg) -> EnvFeatureHandler:
    """Build the math-environment handler (all ``SCALAR`` / ``INHERIT``).

    The math environment (e.g. ``lorenz_*``) has no special feature semantics;
    this is the bit-exactness regression guard (plan S2.1).
    """
    return build_default_scalar_handler(cfg)


# ---------------------------------------------------------------------------
# Robotic-3D handlers (plan S2.2 / S2.3 / S2.4)
# ---------------------------------------------------------------------------
# Shared obs layout (obs_shape [10]):
#   linear_vels.{x,y,z}  -> LINEAR_VELOCITY (INHERIT)
#   attitude.{w,x,y,z}   -> QUATERNION      (UNIT_NORM, absolute target, geodesic)
#   angular_vels.{x,y,z} -> ANGULAR_VELOCITY (INHERIT; physical clamp handled at
#                                             deploy / obs_physical_bounds)
# ---------------------------------------------------------------------------

_LINEAR_VEL_DIMS = ("linear_vels.x", "linear_vels.y", "linear_vels.z")
_ATTITUDE_DIMS = ("attitude.w", "attitude.x", "attitude.y", "attitude.z")
_ANGULAR_VEL_DIMS = ("angular_vels.x", "angular_vels.y", "angular_vels.z")
#: RLRP-761 S5.4 — 3-D yaw-invariant orientation (body-frame gravity direction),
#: the intended replacement for the 4-D quaternion on the UGV. Mutually
#: exclusive with :data:`_ATTITUDE_DIMS` (see :func:`_robotic3d_obs_groups`).
_GRAVITY_DIMS = ("gravity.x", "gravity.y", "gravity.z")
_DT_DIM = "timestamps.delta_stamps"


def _orientation_geodesic_loss(block_pred, block_target, keep_batch: bool = False):
    """Quaternion geodesic (angular) error — PyPose-backed adapter.

    Thin ``env_handlers`` adapter (RLRP-746) over the canonical PyPose
    :func:`tools.feature_handling_tools.orientation_heads.rotation_geodesic_loss`.
    It only supplies the feature-handling glue: unit-normalizing the ``[w, x, y, z]``
    prediction/target blocks (as the removed in-house term did) before delegating
    the ``SO(3)`` geodesic computation to PyPose. The two implementations are
    numerically identical on unit quaternions (both = mean squared ``SO(3)`` angle).

    ``keep_batch`` (RLRP-751, task T3): when ``True`` the ``.mean()`` collapse is
    dropped and a ``(..., 1)`` tensor is returned so the composite auto-weighting
    module receives the batch-preserved ``(E, B, 1)`` shape. ``False`` (default)
    keeps the historical scalar reduction.

    Returned only when the shared feature-loss hook is active (the model's
    ``feature_geometry_loss_weight`` constructor parameter is ``> 0``); default
    weight ``0`` keeps training bit-exact.

    **Rationale / caveat (Geist et al. 2024, "Learning with 3D Rotations: a
    Hitchhiker's Guide to SO(3)", ICML 2024).** The squared geodesic angle has a
    vanishing gradient near 0 and curvature issues near ``pi``; it is the default
    diagnostic objective, with :func:`_orientation_chordal_loss` offered as a
    configurable alternative (report §5.3).
    """
    pred_q = quaternion_normalize(block_pred)
    gt_q = quaternion_normalize(block_target)
    return rotation_geodesic_loss(pred_q, gt_q, keep_batch=keep_batch)


def _orientation_chordal_loss(block_pred, block_target, keep_batch: bool = False):
    """Quaternion chordal (Frobenius) error — PyPose-backed adapter.

    Thin ``env_handlers`` adapter (RLRP-746) over the canonical PyPose
    :func:`tools.feature_handling_tools.orientation_heads.rotation_chordal_loss`,
    supplying only the unit-normalization glue before delegating to PyPose. It is
    a configurable alternative training objective to :func:`_orientation_geodesic_loss`
    (report §5.3), recommended by Geist et al. 2024 as competitive-to-better than
    the geodesic for over-parameterized rotation outputs and free of the
    geodesic-squared gradient pathologies.

    The PyPose Frobenius distance ``‖R_pred - R_target‖_F^2`` is numerically
    identical to the removed in-house closed form ``8 (1 - w_rel^2)`` on unit
    quaternions (since ``‖R_pred - R_target‖_F^2 = 4 (1 - cos θ) = 8 (1 - w^2)``).

    ``keep_batch`` (RLRP-751, task T3): when ``True`` the batch is preserved and a
    ``(..., 1)`` tensor is returned for the composite auto-weighting module.
    """
    pred_q = quaternion_normalize(block_pred)
    gt_q = quaternion_normalize(block_target)
    return rotation_chordal_loss(pred_q, gt_q, keep_batch=keep_batch)


# Configurable orientation geometry loss objective (report §5.3). ``geodesic`` is
# the neutral default (bit-exact vs the pre-config path); ``chordal`` selects the
# Frobenius objective. Selected per dataset via the optional Hydra key
# ``environment.feature_geometry_loss_objective``. Both objectives delegate their
# core math to the canonical PyPose losses (RLRP-746).
_ORIENTATION_LOSS_OBJECTIVES = {
    "geodesic": _orientation_geodesic_loss,
    "chordal": _orientation_chordal_loss,
}


def _resolve_orientation_loss_term(objective):
    """Return the orientation ``loss_term`` callable for ``objective``.

    :param objective: ``"geodesic"`` (default), ``"chordal"`` (report §5.3), or
        ``None`` (Hydra ``null``) to DISABLE the geometry term. ``None`` returns
        ``None`` (no ``loss_term`` on the attitude group), mirroring the
        ``feature_geometry.loss_objective: null`` disabling alternative surfaced
        in ``FeatureGeometryLossMixin._feature_loss_active``.
    :raises ValueError: when ``objective`` is not a known objective.
    """
    if objective is None:
        return None
    try:
        return _ORIENTATION_LOSS_OBJECTIVES[objective]
    except KeyError:
        raise ValueError(
            f"Unknown feature_geometry_loss_objective '{objective}'; expected one "
            f"of {sorted(_ORIENTATION_LOSS_OBJECTIVES)}."
        )


def _read_orientation_loss_objective(cfg) -> str:
    """Read ``ms_model.feature_geometry.loss_objective`` (default ``geodesic``).

    RLRP-736 config refactor: the orientation geometry training objective now
    lives in the nested ``ms_model.feature_geometry`` group (report §5.3),
    alongside ``loss_weight`` (formerly the flat
    ``environment.feature_geometry_loss_objective`` key). Absent key resolves to
    ``"geodesic"``, keeping every existing config bit-exact.
    """
    from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist

    if is_cfg_key_exist(cfg, "ms_model.feature_geometry.loss_objective"):
        objective = cfg.ms_model.feature_geometry.loss_objective
        # ``null`` disables the geometry term (see
        # ``FeatureGeometryLossMixin._feature_loss_active``); keep it as ``None``
        # so ``_resolve_orientation_loss_term`` builds no attitude ``loss_term``.
        return None if objective is None else str(objective)
    return "geodesic"


def _read_orientation_enforce_continuity(cfg) -> bool:
    """Read ``ms_model.internal_orientation.enforce_continuity`` (default ``True``).

    RLRP-736: the attitude temporal sign-continuity pass
    (``<q_t, q_{t-1}> >= 0`` along the causal trajectory) is enabled by default
    (bit-exact with the prior hardcoded behaviour). Set the key to ``false`` to
    disable it.
    """
    from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist

    if is_cfg_key_exist(cfg, "ms_model.internal_orientation.enforce_continuity"):
        return bool(cfg.ms_model.internal_orientation.enforce_continuity)
    return True


#: RLRP-761 S1.5 — accepted values for
#: ``environment.feature_handler.control_norm_strategy``.
_CONTROL_NORM_STRATEGIES = {
    "zscore": NormStrategy.ZSCORE,
    "identity": NormStrategy.IDENTITY,
}


def _read_control_norm_strategy(cfg, override=None) -> NormStrategy:
    """Read ``environment.feature_handler.control_norm_strategy``.

    :param override: value forwarded by ``instantiate_feature_handler`` as a
        builder keyword (Hydra passes every non-``_target_`` key of
        ``environment.feature_handler`` as a kwarg); takes precedence over the
        config lookup, which resolves to the same value anyway.

    RLRP-761 S1.5. Selects the :class:`NormStrategy` applied to the
    ``BOUNDED_CONTROL`` and ``DT`` action blocks.

    - ``"zscore"`` (**default**) — the blocks are plainly standardized and stay
      exempt from any robust warping. This is what ``IDENTITY`` was always
      *meant* to express.
    - ``"identity"`` — pure pass-through, restoring the pre-RLRP-761 behaviour
      bit-for-bit (the whole action block raw, ``dt`` contributing ~0.002 % of
      the per-single-step input energy — see the RLRP-761 root-cause report,
      hypothesis H2).

    :raises ValueError: on an unknown value.
    """
    from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist

    value = "zscore"
    if override is not None:
        value = str(override)
    elif is_cfg_key_exist(cfg, "environment.feature_handler.control_norm_strategy"):
        value = str(cfg.environment.feature_handler.control_norm_strategy)
    try:
        return _CONTROL_NORM_STRATEGIES[value]
    except KeyError:
        raise ValueError(
            f"Unknown environment.feature_handler.control_norm_strategy "
            f"'{value}'; expected one of {sorted(_CONTROL_NORM_STRATEGIES)}."
        )


def _gravity_direction_group(loss_term=None) -> FeatureGroupSpec:
    """Return the 3-D yaw-invariant gravity-direction observation group.

    RLRP-761 ``S5.4``. The body-frame gravity direction is a **unit vector on
    ``S^2``** carrying roll/pitch only; it is the intended replacement for the
    4-D quaternion on the UGV, where the motion dynamics are yaw-invariant.

    Contract, mirroring the ``attitude`` group:

    - :attr:`NormStrategy.UNIT_NORM` — the base normalizer is disabled for the
      block and :class:`StrategyAwareNormalizer` re-projects it onto the unit
      sphere on both directions. No normalizer change was required: the wrapper
      projects *any* contiguous ``unit_norm`` run, so a 3-D block works as-is.
    - ``is_absolute_target=True`` — a direction is never a delta target.
    - ``enforce_continuity=False`` — unlike a quaternion, a 3-D direction has NO
      double cover, so ``g`` and ``-g`` are physically distinct and there is
      nothing to canonicalize (``S5.6``).

    - ``loss_term`` — RLRP-753 task ``G-11`` / ruling ``D-loss`` CLOSES the
      deferral this docstring used to carry: the group now ships the ``S^2``
      chordal term
      :func:`tools.feature_handling_tools.orientation_heads.gravity_direction_loss`
      (plan ``feat_gravity_vector_orientation_representation_plan_RLRP-753_20260724.md``).
      It already unit-normalizes both blocks, so — unlike the quaternion
      objectives — it needs no ``env_handlers`` normalization adapter and is
      attached directly.

    **Gating (identical to the ``attitude`` group).** The term is attached
    unconditionally, exactly like the attitude ``loss_term`` in
    :func:`_robotic3d_obs_groups`, and is ENABLED only by the model-side hook
    :meth:`FeatureGeometryLossMixin._feature_loss_active` — i.e. it fires only
    when ``ms_model.feature_geometry.loss_weight > 0`` (default ``0.0``). A
    default-weight run therefore stays bit-exact.

    :param loss_term: Geometry ``loss_term`` callable; ``None`` (the only value
        any production call site passes) resolves to
        :func:`~tools.feature_handling_tools.orientation_heads.gravity_direction_loss`.
    :return: The resolved ``gravity_direction`` feature group.
    """
    if loss_term is None:
        loss_term = gravity_direction_loss
    return FeatureGroupSpec(
        name="gravity_direction",
        kind=FeatureKind.GRAVITY_DIRECTION,
        dim_names=_GRAVITY_DIMS,
        norm_strategy=NormStrategy.UNIT_NORM,
        is_absolute_target=True,
        enforce_continuity=False,
        loss_term=loss_term,
    )


def _assert_single_orientation_block(obs_dims: Sequence[str]) -> None:
    """Fail fast when BOTH orientation representations are declared.

    RLRP-761 ``S5.5``. ``attitude.*`` (4-D quaternion) and ``gravity.*`` (3-D
    yaw-invariant direction) are two representations of the *same* physical
    quantity and are mutually exclusive: declaring both would give the model two
    partially-redundant absolute targets, two ``unit_norm`` blocks and an
    ambiguous source for ``deploy.trajectory_computation.attitude_propagation``.

    :param obs_dims: The declared observation dimension names.
    :raises ValueError: when both blocks are fully present.
    """
    present = set(obs_dims)
    if all(name in present for name in _ATTITUDE_DIMS) and all(
        name in present for name in _GRAVITY_DIMS
    ):
        raise ValueError(
            "environment.obs_dims declares BOTH orientation representations "
            f"({list(_ATTITUDE_DIMS)} and {list(_GRAVITY_DIMS)}); they are "
            "mutually exclusive (same physical quantity, two absolute "
            "unit-norm targets). Keep exactly one."
        )


def _assert_gravity_requires_angular_velocity(obs_dims: Sequence[str]) -> None:
    """Fail fast when the gravity block is declared without ``angular_vels.*``.

    RLRP-753 ruling ``D-omega``. The body-frame gravity direction evolves as

    .. math:: \\dot{g}^{B} = -\\omega \\times g^{B}

    i.e. the angular velocity is *literally the only thing that moves* the
    gravity vector. Without ``angular_vels.{x,y,z}`` in the observation space the
    gravity channel is therefore **non-Markovian**: ``g^B_{t+1}`` is not
    predictable from ``g^B_t`` alone, so a multistep / compounded-prediction
    model (MTM-Pro) is being asked to regress an unobservable transition.

    ``omega`` is also the *only* yaw source at deploy time: ``g_hat^B`` pins
    roll/pitch and is structurally blind to yaw (it is the yaw-quotient of the
    attitude), so the gravity attitude-reconstruction regime
    (``_gravity_reckon_attitude_sequence``) integrates yaw from ``omega_z``.

    :param obs_dims: The declared observation dimension names.
    :raises ValueError: when the gravity block is fully present but the angular
        velocity block is not. This is a fail-loud configuration SAFETY guard, so
        it is a real ``raise`` — NOT an ``assert`` (``python -O`` strips those).
    """
    present = set(obs_dims)
    if all(name in present for name in _GRAVITY_DIMS) and not all(
        name in present for name in _ANGULAR_VEL_DIMS
    ):
        raise ValueError(
            f"environment.obs_dims declares the gravity block {list(_GRAVITY_DIMS)} "
            f"WITHOUT the angular velocity block {list(_ANGULAR_VEL_DIMS)}; the "
            "body-frame gravity direction obeys `g_dot^B = -omega x g^B`, so "
            "without omega the gravity channel is non-Markovian (g^B_{t+1} is "
            "not predictable from g^B_t) and yaw has no source at deploy time. "
            "Declare angular_vels.{x,y,z} as well."
        )


def validate_orientation_propagation_source(cfg) -> bool:
    """Validate ``deploy.trajectory_computation.attitude_propagation``.

    RLRP-761 ``S5.8``. Propagating an attitude along a predicted trajectory
    requires an orientation source in the observation vector — either the 4-D
    ``attitude.*`` quaternion or the 3-D ``gravity.*`` direction (``S5.4``).
    With neither declared, the propagation silently degenerates to the
    angular-velocity blend seeded by an initial ground-truth quaternion, which
    is a *valid but very different* computation — and it is the configuration
    the reference UGV experiment is currently running.

    Strictness is configurable because that degenerate mode is in active use:

    - ``deploy.trajectory_computation.require_orientation_feature: false``
      (**default**) — emit a ``RuntimeWarning`` naming the degeneracy, so it is
      observable instead of silent, and keep running;
    - ``true`` — :class:`ValueError`, i.e. the reject the plan specifies.

    :param cfg: The resolved experiment configuration.
    :returns: ``True`` when the combination is degenerate (no orientation
        source with propagation enabled), ``False`` otherwise.
    :raises ValueError: on the degenerate combination when
        ``require_orientation_feature`` is set.
    """
    from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist

    key = "deploy.trajectory_computation.attitude_propagation"
    if not is_cfg_key_exist(cfg, key):
        return False
    if not bool(cfg.deploy.trajectory_computation.attitude_propagation):
        return False
    obs_dims, _ = _read_dims(cfg)
    blocks = robotic3d_obs_block_indices(obs_dims)
    if "attitude" in blocks or "gravity" in blocks:
        return False

    message = (
        f"{key}=true but environment.obs_dims declares NEITHER an "
        f"'attitude.*' quaternion NOR a 'gravity.*' direction block "
        f"({list(obs_dims)}): the attitude cannot come from the model output "
        f"and the propagation DEGENERATES to the angular-velocity blend seeded "
        f"by the initial ground-truth orientation (RLRP-761 S5.8)."
    )
    strict = False
    if is_cfg_key_exist(cfg, "deploy.trajectory_computation.require_orientation_feature"):
        strict = bool(cfg.deploy.trajectory_computation.require_orientation_feature)
    if strict:
        raise ValueError(
            message
            + " Set deploy.trajectory_computation.require_orientation_feature=false "
            "to accept the degenerate mode."
        )
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return True


def _robotic3d_obs_groups(
    loss_term=None,
    obs_dims: Optional[Sequence[str]] = None,
    enforce_continuity: bool = True,
) -> List[FeatureGroupSpec]:
    """Return the shared robotic-3D observation groups (plan S2.2).

    Orientation stays a 4-D quaternion at every data-path boundary (symmetric
    external contract): the transition-boundary ``encode``/``decode`` are
    identity, ``norm_strategy`` is ``UNIT_NORM`` on the contiguous 4-D attitude
    block, the block is an absolute target (``no_delta_list``), and it carries
    the opt-in geometry ``loss_term``.

    :param loss_term: Orientation geometry ``loss_term`` callable; defaults to
        :func:`_orientation_geodesic_loss` (report §5.3 configurable objective).
    :param obs_dims: Optional resolved observation dimension order. When given,
        a group is emitted ONLY if every one of its ``dim_names`` is present in
        ``obs_dims``. This restores the pre-RLRP-736 ability to disable an
        arbitrary observation block (e.g. commenting ``angular_vels.*`` out of
        ``environment.obs_dims``) without :meth:`FeatureGroupSpec.resolve`
        raising for the absent dims. When ``None`` (or empty) all groups are
        returned (bit-exact back-compat).
    """
    if loss_term is None:
        loss_term = _orientation_geodesic_loss
    groups = [
        FeatureGroupSpec(
            name="linear_velocity",
            kind=FeatureKind.LINEAR_VELOCITY,
            dim_names=_LINEAR_VEL_DIMS,
        ),
        FeatureGroupSpec(
            name="attitude",
            kind=FeatureKind.QUATERNION,
            dim_names=_ATTITUDE_DIMS,
            norm_strategy=NormStrategy.UNIT_NORM,
            is_absolute_target=True,
            loss_term=loss_term,
            enforce_continuity=enforce_continuity,
        ),
        FeatureGroupSpec(
            name="angular_velocity",
            kind=FeatureKind.ANGULAR_VELOCITY,
            dim_names=_ANGULAR_VEL_DIMS,
        ),
    ]
    if obs_dims:
        present = set(obs_dims)
        _assert_single_orientation_block(obs_dims)
        # RLRP-753 D-omega: the gravity channel is non-Markovian without omega.
        _assert_gravity_requires_angular_velocity(obs_dims)
        # RLRP-761 S5.4/S5.5: the gravity group is declared ONLY when its dims
        # are actually present, so a handler built without `obs_dims` (legacy
        # call sites) keeps its historical group list bit-for-bit.
        if all(name in present for name in _GRAVITY_DIMS):
            groups.insert(2, _gravity_direction_group())
        groups = [
            group
            for group in groups
            if all(name in present for name in group.dim_names)
        ]
    return groups


def _dt_group(
    act_dims: Sequence[str],
    norm_strategy: NormStrategy = NormStrategy.ZSCORE,
) -> List[FeatureGroupSpec]:
    """Return the ``DT`` action group when the time-base dim is present.

    The ``timestamps.delta_stamps`` dimension is the integration time base and
    must never be winsorized/clipped (plan S2.3/S2.4).

    RLRP-761 S1.5: "never winsorized" is now expressed by
    :attr:`NormStrategy.ZSCORE` rather than :attr:`NormStrategy.IDENTITY`. The
    latter *also* removed the plain affine standardization, which left ``dt``
    at its physical scale (``sigma ~ 0.011 s`` around a mean of ``0.10 s``)
    while every observation input was unit-variance — i.e. effectively
    invisible to the network. Pass ``NormStrategy.IDENTITY`` (via
    ``environment.feature_handler.control_norm_strategy: identity``) to restore
    the pre-RLRP-761 behaviour.
    """
    if _DT_DIM in act_dims:
        return [
            FeatureGroupSpec(
                name="dt",
                kind=FeatureKind.DT,
                dim_names=(_DT_DIM,),
                norm_strategy=norm_strategy,
            )
        ]
    return []


def build_robotic3d_general_handler(
    cfg, control_norm_strategy=None
) -> EnvFeatureHandler:
    """Build the shared robotic-3D handler (plan S2.2).

    Note: Generic entry point for future robotic-3D environments. Use `build_ugv_handler` or
          `build_quadcopter_handler` for specialized verision

    Observation groups implement the quaternion specialization; action
    dimensions default to ``SCALAR``/``INHERIT`` except the ``DT`` time base.
    UGV/UAV builders (S2.3/S2.4) extend this with their specific control groups.
    """
    obs_dims, act_dims = _read_dims(cfg)
    loss_term = _resolve_orientation_loss_term(_read_orientation_loss_objective(cfg))
    enforce_continuity = _read_orientation_enforce_continuity(cfg)
    act_groups = _dt_group(
        act_dims, _read_control_norm_strategy(cfg, control_norm_strategy)
    )
    covered = {name for group in act_groups for name in group.dim_names}
    act_groups.extend(
        FeatureGroupSpec(name=name, kind=FeatureKind.SCALAR, dim_names=(name,))
        for name in act_dims
        if name not in covered
    )
    return EnvFeatureHandler.from_groups(
        obs_groups=_robotic3d_obs_groups(loss_term, obs_dims, enforce_continuity),
        act_groups=act_groups,
        obs_dims=obs_dims,
        act_dims=act_dims,
    )


def build_ugv_handler(cfg, control_norm_strategy=None) -> EnvFeatureHandler:
    """Build the robotic-3D UGV handler (plan S2.3).

    Note: Instantiated by Hydra through _target_ strings.

    Inherits the S2.2 quaternion observation handling; action groups:
    ``command.{steering,speed}`` → ``BOUNDED_CONTROL`` and
    ``timestamps.delta_stamps`` → ``DT``, both never winsorized.

    RLRP-761 S1.5: both blocks now default to :attr:`NormStrategy.ZSCORE`
    (standardized, warp-exempt) instead of :attr:`NormStrategy.IDENTITY` (raw
    pass-through); gated by
    ``environment.feature_handler.control_norm_strategy``.
    """
    obs_dims, act_dims = _read_dims(cfg)
    loss_term = _resolve_orientation_loss_term(_read_orientation_loss_objective(cfg))
    enforce_continuity = _read_orientation_enforce_continuity(cfg)
    control_strategy = _read_control_norm_strategy(cfg, control_norm_strategy)
    act_groups = _dt_group(act_dims, control_strategy)
    control_dims = [
        name for name in ("command.steering", "command.speed") if name in act_dims
    ]
    act_groups.extend(
        FeatureGroupSpec(
            name=name,
            kind=FeatureKind.BOUNDED_CONTROL,
            dim_names=(name,),
            norm_strategy=control_strategy,
        )
        for name in control_dims
    )
    covered = {name for group in act_groups for name in group.dim_names}
    act_groups.extend(
        FeatureGroupSpec(name=name, kind=FeatureKind.SCALAR, dim_names=(name,))
        for name in act_dims
        if name not in covered
    )
    return EnvFeatureHandler.from_groups(
        obs_groups=_robotic3d_obs_groups(loss_term, obs_dims, enforce_continuity),
        act_groups=act_groups,
        obs_dims=obs_dims,
        act_dims=act_dims,
    )


def build_quadcopter_handler(
    cfg, control_norm_strategy=None
) -> EnvFeatureHandler:
    """Build the robotic-3D UAV handler (plan S2.4).

    Note: Instantiated by Hydra through _target_ strings.

    Inherits the S2.2 quaternion observation handling; action groups:
    ``motor.m1..m4`` → ``BOUNDED_CONTROL`` and ``timestamps.delta_stamps`` →
    ``DT``, both never winsorized.

    RLRP-761 S1.5: both blocks now default to :attr:`NormStrategy.ZSCORE`
    (standardized, warp-exempt) instead of :attr:`NormStrategy.IDENTITY` (raw
    pass-through); gated by
    ``environment.feature_handler.control_norm_strategy``.
    """
    obs_dims, act_dims = _read_dims(cfg)
    loss_term = _resolve_orientation_loss_term(_read_orientation_loss_objective(cfg))
    enforce_continuity = _read_orientation_enforce_continuity(cfg)
    control_strategy = _read_control_norm_strategy(cfg, control_norm_strategy)
    act_groups = _dt_group(act_dims, control_strategy)
    control_dims = [name for name in act_dims if name.startswith("motor.")]
    act_groups.extend(
        FeatureGroupSpec(
            name=name,
            kind=FeatureKind.BOUNDED_CONTROL,
            dim_names=(name,),
            norm_strategy=control_strategy,
        )
        for name in control_dims
    )
    covered = {name for group in act_groups for name in group.dim_names}
    act_groups.extend(
        FeatureGroupSpec(name=name, kind=FeatureKind.SCALAR, dim_names=(name,))
        for name in act_dims
        if name not in covered
    )
    return EnvFeatureHandler.from_groups(
        obs_groups=_robotic3d_obs_groups(loss_term, obs_dims, enforce_continuity),
        act_groups=act_groups,
        obs_dims=obs_dims,
        act_dims=act_dims,
    )


def robotic3d_obs_block_indices(obs_dims: Sequence[str]) -> dict:
    """Resolve the robotic-3D observation block indices via the feature contract.

    RLRP-736 S1.6 (consolidate/deprecate). This is the single source of truth
    for the positional indices of the three robotic-3D observation blocks —
    ``linear_vels`` (:class:`FeatureKind.LINEAR_VELOCITY`), ``attitude``
    (the :class:`FeatureKind.QUATERNION` block) and ``angular_vels``
    (:class:`FeatureKind.ANGULAR_VELOCITY`) — resolved through
    :meth:`FeatureGroupSpec.resolve` rather than hand-written ``list.index(...)``
    calls scattered in ``robotic_trajectory_dataclass.obs_2_vector``.

    Only blocks whose every dimension name is present in ``obs_dims`` are
    returned, preserving the partial-presence semantics of the legacy
    ``obs_2_vector`` (a block absent from the config yields no entry, i.e.
    ``None`` downstream).

    :param obs_dims: The ordered observation dimension names.
    :return: A mapping ``{block_name: (idx, ...)}`` for each present block, with
        ``block_name`` in ``{"linear_vels", "attitude", "gravity",
        "angular_vels"}`` (``gravity`` added by RLRP-761 ``S5.6``).
    """
    obs_dims = list(obs_dims)
    _assert_single_orientation_block(obs_dims)
    # RLRP-753 D-omega: the gravity channel is non-Markovian without omega.
    _assert_gravity_requires_angular_velocity(obs_dims)
    blocks = (
        ("linear_vels", FeatureKind.LINEAR_VELOCITY, _LINEAR_VEL_DIMS),
        ("attitude", FeatureKind.QUATERNION, _ATTITUDE_DIMS),
        ("gravity", FeatureKind.GRAVITY_DIRECTION, _GRAVITY_DIMS),
        ("angular_vels", FeatureKind.ANGULAR_VELOCITY, _ANGULAR_VEL_DIMS),
    )
    resolved = {}
    for name, kind, dims in blocks:
        if all(dim in obs_dims for dim in dims):
            spec = FeatureGroupSpec(name=name, kind=kind, dim_names=dims).resolve(
                obs_dims
            )
            resolved[name] = spec.indices
    return resolved


#: The two mutually-exclusive orientation blocks, as ``(block_name, kind, dims)``.
#: Both are ``NormStrategy.UNIT_NORM`` absolute-target blocks; the EXTERNAL slot
#: width of each is simply ``len(dims)`` — 4 for the attitude quaternion (``S³``)
#: and 3 for the gravity direction (``S²``). Single source of truth for the
#: orientation-group-driven slot resolver (RLRP-796 Stage 1), replacing the
#: literal ``"attitude"`` key lookup it used to do.
_ORIENTATION_BLOCKS = (
    ("attitude", FeatureKind.QUATERNION, _ATTITUDE_DIMS),
    ("gravity", FeatureKind.GRAVITY_DIRECTION, _GRAVITY_DIMS),
)


class OrientationSlotLayout(tuple):
    """Where the ACTIVE orientation block sits in ONE single-step obs block.

    RLRP-796 Stage 1 (``D-796-S1-4-A``). The internal-orientation head needs three
    facts about the orientation block, not one: *where* its slot(s) start, *how
    wide* one slot is externally, and *which* block it is. Before Stage 1 only the
    first was resolved and the width was the literal ``4`` baked in at 17 sites.

    Deliberately a ``tuple`` SUBCLASS of the base indices rather than a fresh
    dataclass: the value is threaded to the leaf model through the existing
    ``orientation_singlestep_slots`` ctor kwarg, which ~18 model families forward
    opaquely and 10 test call sites inject as a dict string key. Subclassing keeps
    every tuple semantic those sites rely on — ``len()``, indexing, truthiness and
    equality with a plain ``(base,)`` tuple — so widening the value costs **zero**
    edits there, while ``external_width`` / ``kind`` ride along for the consumers
    that need them. A legacy bare tuple is accepted everywhere and normalized by
    :func:`orientation_slot_layout` with ``external_width = 4``.

    :ivar external_width: External scalars per slot (4 = attitude, 3 = gravity).
    :ivar kind: The orientation block's :class:`FeatureKind`, or ``None`` when no
        orientation block is present (empty layout).
    """

    external_width: int
    kind: Optional[FeatureKind]

    def __new__(
        cls,
        bases: Sequence[int] = (),
        external_width: int = 4,
        kind: Optional[FeatureKind] = None,
    ) -> "OrientationSlotLayout":
        self = super().__new__(cls, (int(b) for b in bases))
        self.external_width = int(external_width)
        self.kind = kind
        return self

    def __repr__(self) -> str:
        return (
            f"OrientationSlotLayout(bases={tuple(self)}, "
            f"external_width={self.external_width}, kind={self.kind})"
        )


def orientation_slot_layout(value) -> OrientationSlotLayout:
    """Normalize a slot value into an :class:`OrientationSlotLayout`.

    RLRP-796 ``D-796-S1-4-A`` — the compatibility seam. Accepts ``None``, a legacy
    bare ``tuple[int, ...]`` (the pre-Stage-1 contract, whose external width was
    implicitly the quaternion block's 4) or an already-resolved layout, and always
    returns a layout. Idempotent, so it is safe to call at every consumer.

    :param value: ``None``, an iterable of base slot indices, or a layout.
    :return: The equivalent :class:`OrientationSlotLayout` (empty for ``None``).
    """
    if isinstance(value, OrientationSlotLayout):
        return value
    if not value:
        return OrientationSlotLayout(())
    return OrientationSlotLayout(tuple(value))


def orientation_singlestep_slots_from_dims(
    obs_dims: Sequence[str],
) -> OrientationSlotLayout:
    """Return the ACTIVE orientation block's base slot(s) and width.

    RLRP-736 orientation-slots auto-inference (2026-07-17). The by-construction
    internal-orientation head must know where the orientation block sits inside a
    single-step observation vector. That location is an INTERNAL, layout-derived
    fact — it is resolved here from the declared ``obs_dims`` (the single source of
    truth) rather than hand-copied into config.

    RLRP-796 Stage 1: the lookup is now ORIENTATION-GROUP-DRIVEN rather than
    reading the literal ``"attitude"`` key, and the resolved external slot width
    travels with the bases. Both orientation blocks are ``NormStrategy.UNIT_NORM``
    absolute-target groups and ``_assert_single_orientation_block`` already
    guarantees they are mutually exclusive, so at most ONE matches and the layout's
    uniform ``external_width`` is well-defined:

    - ``attitude.w/x/y/z`` -> ``external_width=4``, ``kind=QUATERNION`` (``S³``);
    - ``gravity.{x,y,z}``  -> ``external_width=3``, ``kind=GRAVITY_DIRECTION`` (``S²``).

    Config-object-free variant of :func:`resolve_orientation_singlestep_slots`
    (mirrors the :func:`build_scalar_handler_from_dims` pattern) for tests /
    tooling that already hold the resolved ``obs_dims`` list.

    The returned layout is a ``tuple`` subclass, so it compares equal to the plain
    ``(base_index,)`` tuple this function returned before Stage 1 — every existing
    caller and assertion is unaffected. Honors user-disabled dims (e.g. dropping
    ``angular_vels.*`` keeps the base index correct). An empty layout means no
    orientation block is declared -> the head stays neutral / bit-exact OFF. The
    orientation block lives ONLY in the observation vector; ``act_dims`` are
    intentionally NOT consulted.

    :param obs_dims: The ordered observation dimension names of one single-step
        obs block.
    :return: The :class:`OrientationSlotLayout` of the active block (empty when
        none is present).
    """
    resolved = robotic3d_obs_block_indices(obs_dims)
    for name, kind, dims in _ORIENTATION_BLOCKS:
        indices = resolved.get(name)
        if indices:
            return OrientationSlotLayout(
                (int(indices[0]),), external_width=len(dims), kind=kind
            )
    return OrientationSlotLayout(())


def resolve_orientation_singlestep_slots(
    cfg: omegaconf.DictConfig,
) -> OrientationSlotLayout:
    """Resolve the orientation single-step layout from ``cfg.environment.obs_dims``.

    RLRP-736 orientation-slots auto-inference. Reads ``obs_dims`` via the
    resilient :func:`_read_dims` (empty when the key is absent → empty layout),
    then delegates to :func:`orientation_singlestep_slots_from_dims`. This is the
    value threaded to the model as the internal ``orientation_singlestep_slots``
    ctor argument at the setup seam; since RLRP-796 Stage 1 it carries the block's
    external slot width and kind alongside the bases (still tuple-compatible).

    :param cfg: The resolved experiment configuration.
    :return: The :class:`OrientationSlotLayout` of the active orientation block.
    """
    obs_dims, _ = _read_dims(cfg)
    return orientation_singlestep_slots_from_dims(obs_dims)


def _attitude_group(handler: EnvFeatureHandler):
    """Return the ``QUATERNION`` observation group of a handler, or ``None``."""
    for group in handler.obs_groups:
        if group.kind is FeatureKind.QUATERNION:
            return group
    return None


def canonicalize_attitude_components(
    cfg: omegaconf.DictConfig, w: ndarray, x: ndarray, y: ndarray, z: ndarray
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Temporally-continue an attitude quaternion trajectory.

    RLRP-736 S3.1 (gap (2) — double cover). Centralized entry point invoked by
    the robotic-3D dataset-ingestion seam
    (``UGVFlatContainerToNested`` / ``QuadcopterFlatContainerToNested``.execute),
    where the trajectory is assembled in causal order — the only place where the
    temporal-continuity pass (``<q_t, q_{t-1}> >= 0``) is well-defined.

    The behaviour is *declared centrally* on the ``QUATERNION`` feature group
    (:attr:`FeatureGroupSpec.enforce_continuity`): when the selected handler
    carries no quaternion group or the flag is off (the neutral / math
    handler), the inputs are returned unchanged (bit-exact).

    The memoryless ``w >= 0`` flip was removed by action B-1b, and the former
    ``canonicalize_sign`` config key / plumbing was FULLY REMOVED by item D of
    the Quaternion manifold upgrade + memoryless ``w >= 0`` removal `.junie`
    plan
    (``rlrp-744-quaternion-manifold-upgrade-and-memoryless-wge0-removal-plan-20260720.md``);
    a stale ``canonicalize_sign`` key now fails loudly at the config resolver.

    :param cfg: the resolved experiment configuration (selects the handler).
    :param w: quaternion ``w`` component, shape ``(T,)`` (numpy or torch).
    :param x: quaternion ``x`` component, shape ``(T,)``.
    :param y: quaternion ``y`` component, shape ``(T,)``.
    :param z: quaternion ``z`` component, shape ``(T,)``.
    :return: the (possibly canonicalized) ``(w, x, y, z)`` tuple, same types.
    """
    handler = instantiate_feature_handler(cfg)
    group = _attitude_group(handler)
    if group is None or not group.enforce_continuity:
        return w, x, y, z

    import numpy as np
    import torch

    from pipeline.pipeline_utils.robotic_env_pipeline_utils.utils import (
        quaternion_enforce_continuity,
    )

    is_torch = isinstance(w, torch.Tensor)
    if is_torch:
        q = torch.stack([w, x, y, z], dim=-1)
    else:
        q = torch.as_tensor(np.stack([w, x, y, z], axis=-1), dtype=torch.float64)

    # Trajectory axis is the leading (time) axis of the (T, 4) stack.
    if group.enforce_continuity:
        q = quaternion_enforce_continuity(q, dim=0)

    if is_torch:
        return q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    q_np = q.numpy()
    return (
        q_np[..., 0].astype(w.dtype),
        q_np[..., 1].astype(x.dtype),
        q_np[..., 2].astype(y.dtype),
        q_np[..., 3].astype(z.dtype),
    )


def instantiate_feature_handler(cfg: omegaconf.DictConfig) -> EnvFeatureHandler:
    """Instantiate the per-environment feature handler from a config.

    RLRP-736 S1.5 — the single, centralized selection point mirroring the
    existing Hydra ``_target_`` hooks (``flat_container_to_nested`` /
    ``deploy_rollout_post_processing``). When ``cfg.environment.feature_handler``
    is present it is instantiated via ``hydra.utils.instantiate``; otherwise we
    fall back to the neutral scalar-only handler (:func:`build_default_scalar_handler`),
    which reproduces the pre-plan behaviour bit-for-bit.

    Reused by both the model-level wiring (``setup_multistep_step_model``) and the
    deploy-level wiring (:func:`attach_feature_handler_to_deploy`) so a single
    consistent handler definition drives all three integration seams (A)/(B)/(C).
    """
    # Local import to avoid a heavy/circular import at module load time.
    import hydra

    from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist

    if is_cfg_key_exist(cfg, "environment.feature_handler"):
        # NOTE (RLRP-736): we resolve the ``_target_`` and call the builder
        # directly instead of ``hydra.utils.instantiate(..., cfg=cfg)``. Passing
        # the whole experiment ``cfg`` as an instantiate parameter makes Hydra
        # attempt ``OmegaConf.resolve`` on the entire config; when a pipeline
        # marks ``cfg`` read-only (e.g. the math-env pipeline) this raises
        # ``ReadonlyConfigError``. Calling the builder directly only *reads* the
        # config, so it is read-only-safe for every pipeline.
        handler_cfg = cfg.environment.feature_handler
        builder = hydra.utils.get_method(handler_cfg._target_)
        extra = {key: value for key, value in handler_cfg.items() if key != "_target_"}
        return builder(cfg=cfg, **extra)
    return build_default_scalar_handler(cfg)


def merge_feature_handler_normalizer_kwargs(cfg: omegaconf.DictConfig, normalizer_kwargs: omegaconf.DictConfig | dict) -> omegaconf.DictConfig | dict:
    """Merge the handler's normalizer kwargs into the config-provided ones.

    RLRP-736 S3.7 (§14C — production integration of the normalizer level (A)).
    The per-block normalizer strategies declared by the feature handler
    (quaternion ``UNIT_NORM``, ``dt -> IDENTITY``, the per-dimension
    ``normalize_dims`` disable mask) were previously validated only in unit
    tests and never reached the production model: the ``normalizer_kwargs``
    passed to ``OneDTransitionRewardModelV2`` came solely from the Hydra config
    (the hand-written ``softwinsorization/*.yaml`` lists). This helper injects
    the handler-derived keys at the single model-construction choke point
    (:func:`tools.mbrl_lib_tools.common_tools.create_one_dim_tr_model_v2`) so the
    fork's ``_build_normalizers`` / ``StrategyAwareNormalizer`` routing (which
    ``OneDTransitionRewardModelV2`` inherits) actually applies during training.

    Precedence / merge rules:

    - The handler is **authoritative** for the RLRP-736 keys it owns —
      ``feature_dim_names`` (S1.6 single source of truth), ``normalize_dims``
      (per-dim enable/disable mask) and ``per_dim_strategy`` (block strategy
      vector). These replace any config-provided values of the same key.
    - Every **other** config key (e.g. ``clip_range``, ``winsor_percentile``,
      ``soft_clip_iqr_mult``) is preserved verbatim.

    Neutrality / back-compat: when no handler is configured, the handler carries
    no dims, or every dimension is :attr:`NormStrategy.INHERIT` (the neutral
    scalar / math handler), the input ``normalizer_kwargs`` is returned
    **unchanged** — so the math env stays bit-exact (S2.1 guard). For robotic
    envs this is an *intended* behaviour change (attitude normalization switches
    to unit-sphere re-projection) → a one-time robotic-3D retrain (plan R10).

    :param cfg: the resolved experiment configuration (selects the handler).
    :param normalizer_kwargs: the config-derived normalizer kwargs (or ``None``).
    :return: the (possibly merged) normalizer kwargs dict, or the original value
        when the handler is neutral.
    """
    handler = instantiate_feature_handler(cfg)
    handler_kwargs = handler.build_normalizer_kwargs()
    per_dim_strategy = handler_kwargs.get("per_dim_strategy") or []
    # Neutral handler (no dims, or every dim INHERIT) -> no behavioural change.
    is_non_neutral = any(
        strat != NormStrategy.INHERIT.value for strat in per_dim_strategy
    )
    if not is_non_neutral:
        return normalizer_kwargs
    merged = dict(normalizer_kwargs or {})
    merged["feature_dim_names"] = handler_kwargs["feature_dim_names"]
    merged["normalize_dims"] = handler_kwargs["normalize_dims"]
    merged["per_dim_strategy"] = handler_kwargs["per_dim_strategy"]
    return merged


def resolve_feature_handler_no_delta_list(cfg, config_no_delta_list):
    """Resolve the ``no_delta_list`` for the production model construction.

    RLRP-736 S3.8 (§14C — production integration of the model-input level (B)).
    The handler declares which observation indices are *absolute targets* (not
    delta targets) via :meth:`EnvFeatureHandler.no_delta_list` — for robotic-3D
    this is the quaternion attitude block, which must never be regressed as a
    Euclidean delta. This was previously test-only: the production model sourced
    ``no_delta_list`` solely from ``cfg.overrides.no_delta_list``.

    Merge rule: when the handler declares a non-empty ``no_delta_list`` (i.e. a
    robotic-3D handler with a quaternion group), the **union** of the handler
    indices and any config-provided indices is returned (handler cannot silently
    drop a config request, and vice-versa). When the handler is neutral (scalar /
    math → empty list) the config value is returned **unchanged**, so non-robotic
    environments stay byte-identical.

    Note on ``obs_process_fn`` (B): under the symmetric external-quaternion
    contract the handler's ``obs_process_fn`` is identity for every current group
    (the representation switch is model-internal, deferred to S3.2), so it is not
    threaded here — doing so would be a no-op that only risks perturbing the
    neutral path. Only ``no_delta_list`` has an observable effect today.

    :param cfg: the resolved experiment configuration (selects the handler).
    :param config_no_delta_list: the ``cfg.overrides.no_delta_list`` value (or
        ``None``).
    :return: the resolved ``no_delta_list`` (union) or the original config value
        when the handler is neutral.
    """
    handler = instantiate_feature_handler(cfg)
    handler_no_delta = handler.no_delta_list()
    if not handler_no_delta:
        # Neutral handler (scalar / math) -> no behavioural change.
        return config_no_delta_list
    if config_no_delta_list is None:
        config_indices = []
    else:
        config_indices = list(config_no_delta_list)
    merged = sorted(set(config_indices) | set(handler_no_delta))
    return merged


def resolve_feature_handler_transition_model_overrides(cfg, transition_model_cfg):
    """Compute the ``instantiate`` overrides that make handler levels (A)+(B) live in training.

    RLRP-736 S3.11 (§14C — production integration into the *training* pipeline).
    The training pipeline builds the transition-model wrapper via
    ``hydra.utils.instantiate(cfg.one_dim_transition_model, model=..., ...)``
    (``setup_single_step_model`` / ``setup_multistep_step_model``), reading
    ``normalizer_kwargs`` / ``no_delta_list`` **verbatim from the config node** —
    it never calls :func:`tools.mbrl_lib_tools.common_tools.create_one_dim_tr_model_v2`,
    where the S3.7/S3.8 merge was originally wired. That merge is therefore dead
    on the training path (it only affects the model-load / deploy path). This
    helper closes that gap: it returns the subset of ``instantiate`` keyword
    overrides that carry the handler's normalizer-level (A) and model-input (B)
    handling, sourced from the *config node* as the merge base so the handler is
    authoritative only for the keys it owns.

    Neutrality / back-compat: overrides are emitted **only** when the handler is
    non-neutral (some dim is not :attr:`NormStrategy.INHERIT`, and/or a non-empty
    absolute-target ``no_delta_list``). For the neutral scalar / math handler the
    returned dict is **empty**, so the ``instantiate`` call is byte-identical to
    today (math S2.1 guard holds). For robotic envs this activates the intended
    behaviour change (attitude ``UNIT_NORM`` + absolute-target) → one-time retrain
    (plan R10).

    :param cfg: the resolved experiment configuration (selects the handler).
    :param transition_model_cfg: the ``cfg.one_dim_transition_model`` config node
        (the merge base for ``normalizer_kwargs`` / ``no_delta_list``).
    :return: a dict of ``instantiate`` keyword overrides (possibly empty).
    """
    import omegaconf

    handler = instantiate_feature_handler(cfg)
    overrides = {}

    # --- (A) normalizer level ------------------------------------------------
    handler_kwargs = handler.build_normalizer_kwargs()
    per_dim_strategy = handler_kwargs.get("per_dim_strategy") or []
    is_non_neutral = any(
        strat != NormStrategy.INHERIT.value for strat in per_dim_strategy
    )
    if is_non_neutral:
        base_norm = None
        if transition_model_cfg is not None:
            base_norm = transition_model_cfg.get("normalizer_kwargs", None)
            if base_norm is not None:
                base_norm = omegaconf.OmegaConf.to_container(base_norm, resolve=True)
        overrides["normalizer_kwargs"] = merge_feature_handler_normalizer_kwargs(
            cfg, base_norm
        )

    # --- (B) model input level ----------------------------------------------
    handler_no_delta = handler.no_delta_list()
    if handler_no_delta:
        base_ndl = None
        if transition_model_cfg is not None:
            base_ndl = transition_model_cfg.get("no_delta_list", None)
        overrides["no_delta_list"] = resolve_feature_handler_no_delta_list(
            cfg, base_ndl
        )

    return overrides


def attach_feature_handler_to_deploy(cfg, deploy_post_proc):
    """Attach the per-environment feature handler onto a deploy postprocessing.

    RLRP-736 S1.5 — threads the same handler contract into the deploy/rollout
    seam (level C). Fully defensive: if the deploy object does not expose
    ``set_feature_handler`` (e.g. ``None`` or a non-conforming double) this is a
    no-op and the deploy path stays byte-identical to the legacy behaviour.

    :param cfg: the resolved experiment configuration.
    :param deploy_post_proc: an instantiated ``Deploy_Rollout_PostProcessing``
        (or ``None``).
    :return: ``deploy_post_proc`` (unchanged reference), for call-site chaining.
    """
    setter = getattr(deploy_post_proc, "set_feature_handler", None)
    if setter is None:
        return deploy_post_proc
    setter(instantiate_feature_handler(cfg))
    return deploy_post_proc


# ---------------------------------------------------------------------------
# RLRP-736 S3.12 — single feature-handler applicator + completeness marker
# ---------------------------------------------------------------------------
#: Marker attribute stamped on every ``OneDTransitionRewardModelV2`` that has
#: gone through :func:`apply_feature_handler_to_transition_model`. It is a plain
#: instance attribute (bool) — deliberately NOT a ``Parameter``/buffer — so it is
#: excluded from ``state_dict`` / the wrapper ``save``/``load`` and never touches
#: ``torch.save``/``load`` (plan §14C S3.12 G4). Legacy checkpoints that predate
#: the marker simply lack the attribute; consumers must default to ``False``.
FEATURE_HANDLER_APPLIED_MARKER = "_rlrp736_feature_handler_applied"
#: Name of the handler builder that was applied (or ``None`` for the neutral
#: scalar/math handler), stored alongside the boolean marker for diagnostics.
FEATURE_HANDLER_NAME_MARKER = "_rlrp736_feature_handler_name"


def apply_feature_handler_to_transition_model(
    cfg, transition_model, model_ensemble=None
):
    """Single RLRP-736 seam applied at *every* transition-model construction site.

    RLRP-736 S3.12 (§14C) — the two prior audits showed the per-environment
    feature-handling concerns were wired at partially-overlapping subsets of the
    three ``OneDTransitionRewardModelV2`` construction sites
    (``create_one_dim_tr_model_v2`` and ``setup_single_step_model`` /
    ``setup_multistep_step_model``), so it was easy to miss a call site. This
    function consolidates the *post-construction* portion of the seam into one
    place that all sites call:

    1. **Loss-handler registration (level B / geometry loss).** Registers the
       per-environment feature handler on the model ensemble's opt-in geometry
       hook (:meth:`ExponentialFamilyMLP.set_feature_handler`). Fully defensive:
       if ``model_ensemble`` is ``None`` or does not expose ``set_feature_handler``
       (e.g. a ``GaussianMLP`` / ``BasicEnsemble`` single-step ensemble) this is a
       no-op — so wiring it at the single-step site too is a consistency fix, not
       a behavioural change for today's hookless single-step models (plan §14C
       S3.12 G2).
    2. **Completeness marker.** Stamps :data:`FEATURE_HANDLER_APPLIED_MARKER` /
       :data:`FEATURE_HANDLER_NAME_MARKER` on the wrapper so a coverage test can
       assert every construction path went through this seam (plan §14C S3.12.4).

    The *pre-construction* portion of the seam — the normalizer-level (A) and
    ``no_delta_list`` (B) merge — must be applied as constructor kwargs before the
    wrapper exists; it stays in :func:`merge_feature_handler_normalizer_kwargs` /
    :func:`resolve_feature_handler_no_delta_list` /
    :func:`resolve_feature_handler_transition_model_overrides`, which every site
    already routes through. Unifying the *config source* those read
    (``cfg.algorithm.*``/``cfg.overrides.*`` vs the ``cfg.one_dim_transition_model``
    node) is the separate, behaviour-changing S3.12.2 step, deferred pending the
    G3 config-agreement audit.

    Neutrality / back-compat: the marker is inert (a plain bool attribute) and the
    registration is a no-op unless the model exposes the hook, so this is
    byte-identical for every existing run (math S2.1 guard holds; the geometry
    term stays OFF at the default ``feature_geometry_loss_weight = 0.0``).

    :param cfg: the resolved experiment configuration (selects the handler).
    :param transition_model: the constructed ``OneDTransitionRewardModelV2``.
    :param model_ensemble: the wrapped model ensemble carrying the geometry hook
        (defaults to ``transition_model.model`` when ``None``).
    :return: ``transition_model`` (unchanged reference), for call-site chaining.
    """
    from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist

    handler = instantiate_feature_handler(cfg)

    ensemble = model_ensemble
    if ensemble is None:
        ensemble = getattr(transition_model, "model", None)
    setter = getattr(ensemble, "set_feature_handler", None)
    if setter is not None:
        setter(handler)

    handler_name = None
    if is_cfg_key_exist(cfg, "environment.feature_handler._target_"):
        handler_name = cfg.environment.feature_handler._target_
    try:
        setattr(transition_model, FEATURE_HANDLER_APPLIED_MARKER, True)
        setattr(transition_model, FEATURE_HANDLER_NAME_MARKER, handler_name)
    except (AttributeError, TypeError):
        # Extremely defensive: never let marker stamping break construction.
        pass
    return transition_model


def feature_handler_applied(transition_model) -> bool:
    """Return whether ``transition_model`` went through the RLRP-736 seam.

    RLRP-736 S3.12.4 (§14C). Reads the completeness marker stamped by
    :func:`apply_feature_handler_to_transition_model`. Legacy checkpoints /
    wrappers built before the marker existed simply lack the attribute and are
    reported as ``False`` (the marker is intentionally excluded from
    ``state_dict`` — G4).

    :param transition_model: a constructed ``OneDTransitionRewardModelV2``.
    :return: ``True`` iff the wrapper carries the applied marker.
    """
    return bool(getattr(transition_model, FEATURE_HANDLER_APPLIED_MARKER, False))


def assert_feature_handler_applied(transition_model, model_ensemble=None) -> None:
    """Fail-loud guard: a wrapper must have gone through the single seam.

    RLRP-736 S3.12.4 (§14C) — turns the audit-class "wired to the wrong choke
    point" bug (a construction path that silently bypasses
    :func:`apply_feature_handler_to_transition_model`) into an immediate,
    explicit error instead of a silent no-op that only surfaces as degraded
    training. Two checks:

    1. **Marker presence.** The wrapper must carry
       :data:`FEATURE_HANDLER_APPLIED_MARKER`; otherwise some construction path
       skipped the seam.
    2. **Active-term consistency.** If the model ensemble exposes the geometry
       hook AND has an active geometry-loss weight
       (``feature_geometry_loss_weight > 0``), a feature handler must actually be
       registered — otherwise the run would silently train without the requested
       geometry term (the exact failure mode the guard exists to catch).

    Neutral by default: for the scalar/math handler and hookless single-step
    ensembles the second check is skipped, so this raises only on a genuine
    mis-wiring.

    :param transition_model: the constructed ``OneDTransitionRewardModelV2``.
    :param model_ensemble: the wrapped ensemble (defaults to
        ``transition_model.model``).
    :raises RuntimeError: if the seam was bypassed or an active term has no
        registered handler.
    """
    if not feature_handler_applied(transition_model):
        raise RuntimeError(
            "RLRP-736 S3.12: the transition model was constructed without going "
            "through `apply_feature_handler_to_transition_model` (the completeness "
            f"marker `{FEATURE_HANDLER_APPLIED_MARKER}` is absent). A construction "
            "site is bypassing the single feature-handler seam — route it through "
            "`apply_feature_handler_to_transition_model`."
        )

    ensemble = model_ensemble
    if ensemble is None:
        ensemble = getattr(transition_model, "model", None)
    weight = getattr(ensemble, "_feature_geometry_loss_weight", 0.0)
    has_hook = hasattr(ensemble, "set_feature_handler")
    if has_hook and weight and float(weight) != 0.0:
        if getattr(ensemble, "_feature_handler", None) is None:
            raise RuntimeError(
                "RLRP-736 S3.12: model has an active geometry-loss weight "
                f"(feature_geometry_loss_weight={weight}) but no feature handler is "
                "registered on the ensemble. The geometry term would silently "
                "contribute nothing. Ensure the handler is selected via "
                "`cfg.environment.feature_handler` and applied at construction."
            )
