# coding=utf-8
"""Shared resolvers for the RLRP-736 nested orientation config groups.

The by-construction orientation feature and the per-feature geometry loss are
configured through two nested Hydra groups under ``ms_model`` (RLRP-736 config
refactor)::

    ms_model:
      feature_geometry:
        loss_weight: 0.0        # 0 -> disabled
        loss_objective: geodesic  # geodesic (default) | chordal | null (disabled)
      internal_orientation:
        # quaternion (default, normalized) | quaternion_legacy (raw, pre-RLRP-736
        # un-normalized contract) | nine_d_svd | sixd
        representation: quaternion
        tangent_nll_right_jacobian: false

These replace the former FLAT constructor kwargs
(``feature_geometry_loss_weight`` / ``internal_orientation_rep`` /
``orientation_tangent_nll_right_jacobian``) and the former
``environment.feature_geometry_loss_objective`` key.

RLRP-736 orientation-slots auto-inference (2026-07-17): the former
``input_slots`` / ``output_slots`` keys were REMOVED. The attitude slot is an
internal, layout-derived fact (where the 4-D ``attitude.*`` quaternion block sits
inside a single-step obs vector) that is now inferred automatically from
``environment.obs_dims`` at the setup seam and threaded to the model as the
internal ``orientation_singlestep_slots`` ctor argument. A stale ``input_slots`` /
``output_slots`` config key now raises the resolver's "Unknown key(s)" error
(intended; the only compat guarantee is the default ``quaternion`` path).

The two ``resolve_*`` helpers merge a (possibly ``None`` / ``DictConfig``) group
with its defaults and validate top-level keys so config typos surface
immediately (mirrors the ``_resolve_loss_cfg_group`` pattern used for the other
nested loss groups).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import omegaconf

FEATURE_GEOMETRY_CFG_DEFAULTS: Dict[str, Any] = {
    "loss_weight": 0.0,
    "loss_objective": "geodesic",
    # Introduced by task T1 of the RLRP-751 feature-geometry composite-loss
    # auto-weighting `.junie` plan
    # (`rlrp-751-feature-geometry-composite-loss-auto-weighting-plan-20260723.md`).
    # ``False`` (default) -> STATIC weighting: the feature-geometry term is scaled
    # by ``loss_weight`` exactly like before. ``True`` -> route the SS/MS/CP
    # feature-geometry channels through the composite uncertainty auto-weighting
    # (Kendall 2018), with ``loss_weight`` acting as the pre-multiplier. ONE key
    # controls all three channels; it gracefully degrades to STATIC on model
    # families that do not expose ``composite_loss_automatic_weighting``.
    "auto_weighting": False,
}

INTERNAL_ORIENTATION_CFG_DEFAULTS: Dict[str, Any] = {
    "representation": "quaternion",
    "tangent_nll_right_jacobian": False,
    # Tri-state manifold-aware NLL lever. Introduced by item A (§A.0/§A.2) of the
    # Quaternion manifold upgrade + memoryless ``w >= 0`` removal `.junie` plan
    # (`rlrp-744-quaternion-manifold-upgrade-and-memoryless-wge0-removal-plan-20260720.md`).
    # ``None`` (auto, default) -> manifold NLL ON for the active matrix reps
    # ``sixd``/``nine_d_svd``, OFF for ``quaternion``; ``True`` -> force ON
    # (plain-``quaternion`` manifold upgrade); ``False`` -> force OFF (matrix-rep
    # ablation: score the decoded quaternion with the ambient base NLL).
    # ``quaternion_legacy`` ignores the lever (always OFF / byte-exact).
    # NOTE (item C, rev.3b of the same plan): the by-construction IN-GRAPH unit
    # decode is ALWAYS ON for the plain ``quaternion`` rep (no lever; independent
    # of this key); ``quaternion_legacy`` keeps the raw un-normalized output.
    "manifold_aware_nll": None,
    # RLRP-736: attitude double-cover handling applied at the robotic-3D
    # dataset-ingestion seam (see ``env_handlers.canonicalize_attitude_components``
    # / ``FeatureGroupSpec.enforce_continuity``). ``enforce_continuity`` defaults
    # ``True`` (bit-exact with the prior hardcoded behaviour); set to ``false``
    # to disable the temporal-continuity pass. The key is consumed directly from
    # ``cfg`` by the feature-handler builders and is listed here so the resolver
    # accepts (rather than rejects) it.
    # RLRP-744 item E: the key ALSO gates the MODEL-SIDE reference-relative
    # attitude alignment. It is threaded into the model (via
    # ``base_multistep_mlp`` -> ``ExponentialFamilyMLP._setup_orientation_rep`` ->
    # ``self._orientation_enforce_continuity``) and ANDed into
    # ``_quaternion_ar_continuity_active`` so a single config value governs BOTH the
    # ingestion continuity pass AND every model-graph reference-relative alignment
    # (``align_quaternion_slots_to_reference`` at AR splice / deploy / sampling-free /
    # resampled / trsf + the DH residual). Orthogonal to the in-graph unit decode
    # (``_unit_decode``, item C) and the conversion-internal ``w >= 0`` in
    # ``quaternion_to_axis_angle`` (both intentionally NOT gated).
    # NOTE: the former ``canonicalize_sign`` key (memoryless ``w >= 0`` flip,
    # already no-effect since action B-1b) was FULLY REMOVED by item D of the
    # Quaternion manifold upgrade + memoryless ``w >= 0`` removal `.junie` plan
    # (`rlrp-744-quaternion-manifold-upgrade-and-memoryless-wge0-removal-plan-20260720.md`);
    # a stale ``canonicalize_sign`` key now fails loudly ("Unknown key(s)").
    "enforce_continuity": True,
}


def _resolve_group(
    name: str,
    provided: Optional[Union[Dict, omegaconf.DictConfig]],
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge ``provided`` over ``defaults``; raise on unknown top-level keys."""
    merged = dict(defaults)
    if provided is None:
        return merged
    provided = dict(provided)
    unknown = set(provided) - set(defaults)
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in '{name}' config group; "
            f"allowed keys: {sorted(defaults)}"
        )
    merged.update(provided)
    return merged


def resolve_feature_geometry_cfg(
    provided: Optional[Union[Dict, omegaconf.DictConfig]],
) -> Dict[str, Any]:
    """Resolve the ``ms_model.feature_geometry`` group (``loss_weight`` /
    ``loss_objective`` / ``auto_weighting``) against its defaults."""
    return _resolve_group(
        "feature_geometry", provided, FEATURE_GEOMETRY_CFG_DEFAULTS
    )


def resolve_internal_orientation_cfg(
    provided: Optional[Union[Dict, omegaconf.DictConfig]],
) -> Dict[str, Any]:
    """Resolve the ``ms_model.internal_orientation`` group (``representation`` /
    ``tangent_nll_right_jacobian``) against its defaults.

    RLRP-736 orientation-slots auto-inference: ``input_slots`` / ``output_slots``
    are no longer valid keys (the attitude slot is inferred automatically); a
    stale key raises the "Unknown key(s)" error."""
    return _resolve_group(
        "internal_orientation", provided, INTERNAL_ORIENTATION_CFG_DEFAULTS
    )
