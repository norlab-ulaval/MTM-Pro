# coding=utf-8
"""Model-internal orientation encoder/head (Variant B, PyPose) for RLRP-736.

Introduced by action **§14A S3.2 / S2.2-B** of the Per-Environment Feature
Handling ``.junie`` plan — Variant B (PyPose-based):
``rlrp-736-per-environment-feature-handling-pypose-based-plan-20260711.md``
(YouTrack RLRP-736).

**Objective / rationale.** The network's attitude *output* must be a valid
rotation **by construction** (not a raw 4-D vector fixed up with a unit-norm
projection after the fact), and its attitude *input* must be a **continuous**
parameterization. Geist et al. 2024 ("Learning with 3D Rotations: a
Hitchhiker's Guide to SO(3)", ICML 2024) show the discontinuous 3-/4-parameter
representations (Euler, axis-angle, raw quaternion) hurt learning when a
rotation is a network *output*, and recommend over-parameterized continuous
representations — ``R^9`` (+SVD) preferred, then ``R^6`` (+Gram-Schmidt). Zhou
et al. 2019 ("On the Continuity of Rotation Representations in Neural
Networks", CVPR) introduced the continuous 6D representation used here. For the
9D route, Chen et al. 2023 (arXiv:2312.00462) / its gradient follow-up
recommend **training on the un-orthogonalized matrix and orthogonalizing only at
inference (SVD)** to sidestep the Gram-Schmidt/SVD gradient pathologies.

**Symmetric external contract (report §4.4, operator constraint).** These are
purely *in-graph* submodules: the encoder consumes the model's existing 4-D
``[w, x, y, z]`` quaternion input slot and the head emits a 4-D ``[w, x, y, z]``
quaternion in the same output slot. The alternate representation exists only
*between* the encoder and the head — there is **no** dataset pre-processing and
the observation/action spec (linear-vel 3D, attitude quaternion 4D, angular-vel
3D) is preserved end-to-end so the model stays compatible with the real-robot
pipeline.

All rotation group ops are delegated to **PyPose** (``pp.SO3`` / ``pp.so3`` /
``pp.mat2SO3``) via the single tested convention boundary
:mod:`tools.feature_handling_tools.pypose_so3`.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import nn

from tools.feature_handling_tools.feature_spec import (
    InternalOrientationRep,
    is_s2_orientation_rep,
)
from tools.feature_handling_tools.pypose_so3 import (
    pp,
    pp_SO3_to_wxyz,
    wxyz_to_pp_SO3,
)

# Width (number of scalar features) of the model-internal representation each
# rep exposes between the encoder and the head. The external slot width is a
# property of the ORIENTATION BLOCK, not of the rep: 4 for the quaternion block
# (``attitude.w/x/y/z``) and 3 for the gravity block (``gravity.{x,y,z}``). The
# reps listed here are all quaternion-block reps, hence the 4s below; the
# block-driven external width is threaded separately (RLRP-796 Stage 1).
_REP_INTERNAL_WIDTH = {
    InternalOrientationRep.QUATERNION: 4,  # passthrough (neutral / no switch)
    InternalOrientationRep.QUATERNIONLEGACY: 4,  # passthrough (legacy, un-normalized)
    InternalOrientationRep.SO3_RELATIVE: 3,  # so3 tangent (Log)
    InternalOrientationRep.SIXD: 6,  # first two rotation-matrix columns
    InternalOrientationRep.NINE_D_SVD: 9,  # full (raw) rotation matrix
    InternalOrientationRep.S2_IDENTITY: 3,  # S^2 identity (unit 3-D direction)
    InternalOrientationRep.S2_TANGENT: 2,  # S^2 tangent at the previous direction
}

# Reps whose INPUT encode is a passthrough of the EXTERNAL slot even though their
# OUTPUT is narrower, i.e. the reps whose in/out widths DISAGREE. ``S2_TANGENT``
# is output-only by ruling ``D-796-S1-2-A``: its encode keeps the ABSOLUTE 3-D
# direction so the network never loses the absolute tilt, and only the trunk
# OUTPUT is narrowed to the 2-D tangent (RLRP-796 Stage 1).
_ABSOLUTE_INPUT_REPS = (InternalOrientationRep.S2_TANGENT,)

# Reps whose encode/decode is a pure passthrough of the EXTERNAL slot: their
# internal width is the external width by definition, whatever the block is.
_NEUTRAL_REPS = (
    InternalOrientationRep.QUATERNION,
    InternalOrientationRep.QUATERNIONLEGACY,
)

# Dimension of the tangent space the (log-)variance head occupies for one slot
# of the given rep's orientation block. ``SO(3)`` (quaternion block) is 3-D
# (Forster et al. 2016); ``S²`` (gravity block) is 2-D. Keyed by the EXTERNAL
# slot width, which uniquely identifies the block (RLRP-796 Stage 1).
_EXTERNAL_WIDTH_TANGENT_DIM = {
    4: 3,  # attitude quaternion block -> so3 tangent
    3: 2,  # gravity direction block -> S2 tangent
}

# Reps whose (log-)variance head stays AMBIENT, i.e. is NOT narrowed to the
# manifold's tangent dimension. ``S2_IDENTITY`` is trunk-delta-0 by design (it is
# the control arm for ``S2_TANGENT``), so its logvar keeps the external width; the
# 3 -> 2 narrowing is the distinguishing feature of the tangent rep (``D-793-4``).
_AMBIENT_LOGVAR_REPS = (InternalOrientationRep.S2_IDENTITY,)

# Default external orientation slot width. 4 = the quaternion block, the only
# orientation block the internal-rep path resolved before RLRP-796 Stage 1;
# retained as the default so every legacy call site keeps its exact arithmetic.
DEFAULT_EXTERNAL_ORIENTATION_WIDTH = 4


def internal_rep_width(rep: InternalOrientationRep) -> int:
    """Return the internal feature width produced/consumed for ``rep``.

    Legacy, quaternion-block-only accessor kept for the existing call sites and
    regression tests. New code should use the explicit in/out pair
    :func:`internal_rep_in_width` / :func:`internal_rep_out_width`, which take
    the block's external slot width and therefore also serve the gravity block.

    :param rep: The model-internal orientation representation.
    :return: Number of scalar features (4 for the neutral ``QUATERNION`` /
        ``QUATERNIONLEGACY`` cases).
    """
    return _REP_INTERNAL_WIDTH[rep]


def internal_rep_in_width(
    rep: InternalOrientationRep,
    external_width: int = DEFAULT_EXTERNAL_ORIENTATION_WIDTH,
) -> int:
    """Width the TRUNK CONSUMES per orientation slot under ``rep`` (RLRP-796 §D-796-S1-3-A).

    Split out of :func:`internal_rep_width` because the input and the output
    width of a rep need not agree: a tangent rep narrows only what the trunk
    EMITS, while its encode keeps the absolute (external-width) direction so the
    network never loses the absolute tilt.

    :param rep: The model-internal orientation representation.
    :param external_width: External slot width of the orientation block
        (4 = attitude quaternion, 3 = gravity direction).
    :return: Scalars per slot on the trunk INPUT side. Equal to
        ``external_width`` for every passthrough rep, so all pre-Stage-1
        arithmetic is unchanged.
    """
    if rep in _NEUTRAL_REPS or rep in _ABSOLUTE_INPUT_REPS:
        return int(external_width)
    return _REP_INTERNAL_WIDTH[rep]


def internal_rep_out_width(
    rep: InternalOrientationRep,
    external_width: int = DEFAULT_EXTERNAL_ORIENTATION_WIDTH,
) -> int:
    """Raw values the TRUNK EMITS per orientation slot under ``rep``.

    Sibling of :func:`internal_rep_in_width`; see that docstring for why the two
    are separate. Both return ``external_width`` for every rep that existed
    before RLRP-796 Stage 1.

    :param rep: The model-internal orientation representation.
    :param external_width: External slot width of the orientation block.
    :return: Scalars per slot on the trunk OUTPUT side.
    """
    if rep in _NEUTRAL_REPS:
        return int(external_width)
    return _REP_INTERNAL_WIDTH[rep]


def internal_rep_tangent_dim(
    rep: InternalOrientationRep,
    external_width: int = DEFAULT_EXTERNAL_ORIENTATION_WIDTH,
) -> int:
    """Tangent dimension of ONE slot's (log-)variance under ``rep``'s block.

    Generalizes the ``TANGENT_DIM = 3`` constant the model seam used to hard-code:
    the manifold-aware logvar head narrows each external slot to the tangent
    dimension of the manifold the slot lives on — 3 for ``SO(3)`` (quaternion
    block), 2 for ``S²`` (gravity block).

    :param rep: The model-internal orientation representation.
    :param external_width: External slot width of the orientation block.
    :return: Tangent dimension (``<= external_width``).
    :raises KeyError: For an unknown external orientation block width.
    """
    if rep in _AMBIENT_LOGVAR_REPS:
        return int(external_width)
    return _EXTERNAL_WIDTH_TANGENT_DIM[int(external_width)]


def sixd_to_rotation_matrix(sixd: torch.Tensor) -> torch.Tensor:
    """Map a 6D representation to a ``[..., 3, 3]`` rotation matrix via Gram-Schmidt.

    Continuous 6D parameterization of Zhou et al. 2019: the 6 numbers are the
    first two (un-normalized) rotation-matrix columns; Gram-Schmidt
    orthonormalizes them and the third column is their cross product.

    :param sixd: Tensor ``[..., 6]``.
    :return: Rotation matrix ``[..., 3, 3]`` (columns are the orthonormal basis).
    """
    a1 = sixd[..., 0:3]
    a2 = sixd[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    a2_proj = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2_proj, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    # Columns [b1 | b2 | b3].
    return torch.stack([b1, b2, b3], dim=-1)


def svd_project_to_so3(matrix: torch.Tensor) -> torch.Tensor:
    """Project a raw ``[..., 3, 3]`` matrix onto ``SO(3)`` via (special) SVD.

    ``R = U · diag(1, 1, det(U·Vᵀ)) · Vᵀ`` (Levinson et al. / Geist et al. 2024):
    the closest rotation to ``M`` in Frobenius norm, with the ``det`` correction
    guaranteeing ``det(R) = +1`` (a proper rotation, not a reflection). Used for
    the ``NINE_D_SVD`` **infer-SVD** step; the raw ``M`` is what the
    train-unorthogonalized loss scores directly (Chen et al. 2023,
    arXiv:2312.00462).

    :param matrix: Raw matrix ``[..., 3, 3]``.
    :return: Rotation matrix ``[..., 3, 3]`` on ``SO(3)``.
    """
    u, _, vh = torch.linalg.svd(matrix)
    det = torch.det(u @ vh)
    diag = torch.ones(matrix.shape[:-2] + (3,), dtype=matrix.dtype, device=matrix.device)
    diag[..., -1] = det
    return u @ torch.diag_embed(diag) @ vh


# ::::Vectorized attitude-slot primitives (RLRP-747)::::::::::::::::::::::::::::::::::::::::::::::::
# Single, documented home for the attitude-slot index arithmetic that used to be
# re-implemented as a per-slot Python loop in
#   - ``align_quaternion_slots_to_reference`` (below),
#   - ``MS2MS2SSArTemporalMixturePME._canonicalize_quaternion_slots``,
#   - ``ExponentialFamilyMLP._gather_orientation_segments``,
#   - the quaternion route of ``ExponentialFamilyMLP._apply_orientation_output_decoding``.
# Plan: ``perf_RLRP-747_orientation_vectorization_plan_20260827.md``.
#
# The primitives are deliberately *layout-only*: they never touch the per-element
# arithmetic, so every rewritten call site stays bit-exact. They operate purely on
# the trailing feature axis, hence ensemble-/horizon-axis agnostic.
#
# ``_splice_slots`` (``exponential_family_mlp.py``) stays generic and is NOT
# replaced: the ``sixd`` / ``nine_d_svd`` reps have a width-CHANGING transform
# (``4 -> internal_rep_width(rep)``), which is not a permutation and therefore
# cannot be expressed as a gather/scatter pair.

# Cache of flat index tensors, keyed by ``(bases, width, slot_width, device)``.
# The key includes the device so a cached CPU index is never reused after a
# ``.to("cuda")`` move (RLRP-747 risk "index-tensor device drift"). The number of
# distinct keys is bounded by the model configuration (one per slot set / width /
# device), so this cache is O(1) in practice.
_ATTITUDE_INDEX_CACHE: Dict[Tuple, Any] = {}


def orientation_slot_bases(
    slots: Optional[Iterable[int]],
    width: int,
    slot_width: int = DEFAULT_EXTERNAL_ORIENTATION_WIDTH,
) -> Tuple[int, ...]:
    """Sorted, de-duplicated, in-range attitude BASE slot indices.

    The python-level companion of :func:`attitude_slot_index`: same normalisation
    and the same fail-loud overlap guard, but one entry per SLOT rather than per
    feature. Kept as a plain ``tuple`` (never a tensor) so callers can build a
    ``torch.cat`` segment list without a device->host synchronisation — an RLRP-783
    hard constraint.

    :param slots: Iterable of attitude base slot indices (``None`` / empty allowed).
    :param width: Feature width of the tensor the slots address.
    :param slot_width: Scalars per attitude slot.
    :return: In-range base indices, ascending.
    :raises ValueError: When two in-range slots OVERLAP (see
        :func:`attitude_slot_index`).
    """
    bases = sorted({int(s) for s in (slots or ())})
    in_range = [s for s in bases if s + slot_width <= int(width)]
    for previous, following in zip(in_range, in_range[1:]):
        if following < previous + slot_width:
            raise ValueError(
                f"Attitude slots must be non-overlapping and {slot_width}-wide; "
                f"got {tuple(bases)} for width {width}."
            )
    return tuple(in_range)


def attitude_slot_index(
    slots: Optional[Iterable[int]],
    width: int,
    slot_width: int = DEFAULT_EXTERNAL_ORIENTATION_WIDTH,
) -> Tuple[int, ...]:
    """Flat feature indices of every in-range attitude slot, sorted, slot-major.

    Given attitude *base* slot indices (each ``slot_width`` scalars wide) and the
    feature width of the tensor they address, return the flattened index list
    ``(s0, s0+1, …, s0+slot_width-1, s1, …)``.

    Semantics deliberately reproduced from the loops this replaces:

    - **Sorted**: mirrors the ``sorted(...)`` normalisation of every call site.
    - **De-duplicated**: the loops rebuilt the tensor once per slot, so a repeated
      slot was idempotent. A single ``index_copy`` with repeated indices is
      documented-NONDETERMINISTIC in PyTorch, so duplicates are collapsed here.
    - **Out-of-range slots are silently skipped** (``s + slot_width > width``),
      exactly like the ``continue`` in :func:`align_quaternion_slots_to_reference`.

    :param slots: Iterable of attitude base slot indices (``None`` / empty allowed).
    :param width: Feature width of the tensor the indices address.
    :param slot_width: Scalars per attitude slot (4 for a ``[w, x, y, z]``
        quaternion, 3 for an so3-tangent log-variance).
    :return: Flat, slot-major feature indices (empty tuple when nothing is in range).
    :raises ValueError: When two in-range slots OVERLAP. The loops this replaces
        composed overlapping slots sequentially, which a single scatter cannot
        reproduce; failing loud mirrors the existing
        ``ExponentialFamilyMLP._splice_slots`` contract and turns a silent wrong
        answer into a crash. No production call site can reach this (all pass
        ``_orientation_singlestep_slots`` / ``_ori_out_slots``, non-overlapping by
        construction).
    """
    return tuple(
        s + k
        for s in orientation_slot_bases(slots, width, slot_width)
        for k in range(slot_width)
    )


def attitude_slot_index_tensor(
    slots: Optional[Iterable[int]],
    width: int,
    slot_width: int = DEFAULT_EXTERNAL_ORIENTATION_WIDTH,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Cached ``int64`` index tensor of :func:`attitude_slot_index` on ``device``.

    ``int64`` (not ``int32``) because that is what ``index_select`` /
    ``index_copy`` require on CUDA.

    :param slots: Iterable of attitude base slot indices.
    :param width: Feature width of the tensor the indices address.
    :param slot_width: Scalars per attitude slot.
    :param device: Target device (defaults to CPU).
    :return: ``[S * slot_width]`` index tensor (shared, do NOT mutate).
    """
    flat = attitude_slot_index(slots, width, slot_width)
    key = (flat, int(width), int(slot_width), str(device))
    cached = _ATTITUDE_INDEX_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(flat, dtype=torch.long, device=device)
        _ATTITUDE_INDEX_CACHE[key] = cached
    return cached


def gather_attitude_slots(
    x: torch.Tensor,
    index: torch.Tensor,
    slot_width: int = DEFAULT_EXTERNAL_ORIENTATION_WIDTH,
) -> torch.Tensor:
    """``(..., F) -> (..., S, slot_width)`` in ONE ``index_select``.

    Stacks every attitude slot on a NEW second-to-last axis so the downstream
    math (sign flip, unit projection, ``rotation_tangent_nll``) runs once over
    all slots instead of once per slot. Autograd-safe; correct for
    non-contiguous ``x`` (``index_select`` materialises a contiguous copy).

    :param x: Tensor ``(..., F)`` whose trailing axis carries the attitude slots.
    :param index: Flat slot-major index tensor from
        :func:`attitude_slot_index_tensor`.
    :param slot_width: Scalars per attitude slot.
    :return: Stacked slots ``(..., S, slot_width)`` where ``S = index.numel() // slot_width``.
    """
    gathered = x.index_select(-1, index)
    n_slots = index.numel() // slot_width
    return gathered.reshape(*gathered.shape[:-1], n_slots, slot_width)


def scatter_attitude_slots(
    x: torch.Tensor,
    bases: Sequence[int],
    slots_value: torch.Tensor,
) -> torch.Tensor:
    """Rebuild ``x`` with every attitude slot replaced, in ONE full-width write.

    Takes a STACKED ``(..., S, slot_width)`` replacement (as produced by
    :func:`gather_attitude_slots`) and splices it back in a single ``torch.cat``.
    Autograd-safe and out-of-place; ``x`` is not mutated.

    **Why ``cat`` and not ``index_copy`` (measured 2026-08-27, RLRP-747).** An
    out-of-place ``torch.index_copy`` is a ``clone()`` followed by an in-place
    scatter, i.e. **two** full-width writes, and the benchmark
    (``bench_rlrp747_orientation_vectorization.py``) showed that doubling the
    memory traffic outweighed the launch saving on a memory-bound splice. The
    ``cat`` form keeps the SINGLE full-width write of the loop this replaces while
    still applying the slot transform ONCE over the stacked slot axis — better on
    both metrics. The per-slot python work below allocates only *views*, so it
    costs no kernel launches.

    The leading shapes are explicitly broadcast first, since ``torch.cat``
    requires an exact match on every non-concatenated dimension (this is a no-op
    for every audited production call site; see the RLRP-747 plan Key Decision 8,
    whose premise is corrected there).

    :param x: Tensor ``(..., F)`` to rebuild (not mutated).
    :param bases: In-range attitude base indices, ascending and non-overlapping
        (from :func:`orientation_slot_bases`) — must match ``slots_value``'s slot axis.
    :param slots_value: Replacement slots ``(..., S, slot_width)``.
    :return: ``x`` with every attitude slot replaced by ``slots_value``.
    """
    slot_width = slots_value.shape[-1]
    leading = torch.broadcast_shapes(x.shape[:-1], slots_value.shape[:-2])
    x = x.expand(tuple(leading) + (x.shape[-1],))
    slots_value = slots_value.expand(
        tuple(leading) + tuple(slots_value.shape[-2:])
    )

    segments: list = []
    cursor = 0
    for i, start in enumerate(bases):
        if start > cursor:
            segments.append(x[..., cursor:start])
        segments.append(slots_value[..., i, :])
        cursor = start + slot_width
    if cursor < x.shape[-1]:
        segments.append(x[..., cursor:])
    return torch.cat(segments, dim=-1)


def split_attitude_and_rest_index(
    slots: Optional[Iterable[int]],
    width: int,
    att_width: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Attitude index + complementary 'rest' index for the split-NLL layout.

    Encodes, once, the cursor rule that
    ``ExponentialFamilyMLP._gather_orientation_segments`` implements with a
    Python loop: ``slots`` are EXTERNAL (4-per-slot) output positions, but the
    tensor being split carries ``att_width`` scalars per attitude slot (4 for the
    decoded mean / target, 3 for the narrowed so3-tangent log-variance), so the
    i-th sorted slot starts at ``s_ext - i * (4 - att_width)``.

    The 'rest' index lists the non-attitude columns in ASCENDING order, which is
    exactly the column order the replaced ``torch.cat(rest_segments, dim=-1)``
    produced — so the split-NLL feature layout is unchanged.

    **Composed multi-step contract.** ``slots`` may be the composed-window
    expansion ``k * singlestep_obs_len + b`` (one entry per horizon step); this
    function is layout-agnostic and handles it by construction.

    :param slots: EXTERNAL attitude base slot indices.
    :param width: Feature width of the tensor being split.
    :param att_width: Scalars per attitude slot in THIS tensor (4 or 3).
    :return: ``(attitude_index, rest_index)`` flat feature indices.
    :raises ValueError: When the narrowed slots overlap (mirrors
        ``_splice_slots``); unreachable from production callers.
    """
    ordered = sorted(int(s) for s in (slots or ()))
    attitude: list = []
    rest: list = []
    cursor = 0
    for i, s_ext in enumerate(ordered):
        start = s_ext - i * (4 - int(att_width))
        if start < cursor:
            raise ValueError(
                f"Orientation slots must be non-overlapping and sorted; got {tuple(ordered)}."
            )
        if start > cursor:
            rest.extend(range(cursor, start))
        attitude.extend(range(start, start + int(att_width)))
        cursor = start + int(att_width)
    if cursor < int(width):
        rest.extend(range(cursor, int(width)))
    return tuple(attitude), tuple(rest)


def split_attitude_and_rest_index_tensors(
    slots: Optional[Iterable[int]],
    width: int,
    att_width: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cached ``int64`` index tensor pair of :func:`split_attitude_and_rest_index`.

    :param slots: EXTERNAL attitude base slot indices.
    :param width: Feature width of the tensor being split.
    :param att_width: Scalars per attitude slot in this tensor (4 or 3).
    :param device: Target device (defaults to CPU).
    :return: ``(attitude_index, rest_index)`` tensors (shared, do NOT mutate).
    """
    attitude, rest = split_attitude_and_rest_index(slots, width, att_width)
    key = ("split", attitude, rest, int(width), int(att_width), str(device))
    cached = _ATTITUDE_INDEX_CACHE.get(key)
    if cached is None:
        cached = (
            torch.tensor(attitude, dtype=torch.long, device=device),
            torch.tensor(rest, dtype=torch.long, device=device),
        )
        _ATTITUDE_INDEX_CACHE[key] = cached
    return cached


def align_quaternion_slots_to_reference(
    x: torch.Tensor,
    reference: torch.Tensor,
    slots,
    eps: float = 1e-12,
    project_to_s3: bool = True,
) -> torch.Tensor:
    """History-relative sign-continuity for 4-D quaternion slot(s) (RLRP-736 Item 1).

    **Quaternion-specific by construction — never pass a gravity slot.** It
    hard-codes the 4-wide slice (``s + 4``), performs the double-cover hemisphere
    sign flip and projects onto ``S^3``. The gravity direction ``ĝ^B`` is a 3-vector
    on ``S^2`` with NO double cover, so a hemisphere flip is meaningless for it; use
    the sibling :func:`project_slots_to_unit_norm` (``slot_width=3``) instead.

    For every quaternion base slot ``s`` in ``slots`` (a 4-D ``[w, x, y, z]``
    quaternion sub-vector), flip the sign of ``x``'s slot so it lies in the SAME
    hemisphere as the corresponding slot of ``reference`` (the ``⟨q_t, q_{t-1}⟩ ≥ 0``
    continuity test used by ``quaternion_enforce_continuity``), then (when
    ``project_to_s3`` is True) L2-project the slot back onto the unit sphere ``S^3``.
    This is the reference-based replacement for the *memoryless* ``w >= 0``
    canonicalisation: unlike the ``w >= 0`` rule it keeps the trajectory
    hemisphere-continuous even when the reference frame lives in the ``w < 0``
    hemisphere (exactly the aggressive-rotation regime RLRP-736 targets;
    Geist et al. 2024).

    ``x`` and ``reference`` must share the trailing feature layout at every slot
    (same base indices) and be broadcast-compatible on the leading dims. The ``S^3``
    projection is sign-equivariant, so it never undoes the alignment. Bit-exact no-op
    when ``slots`` is empty. Idempotent on an already-continuous, already-unit input.

    **Single-step contract (RLRP-747, audit of 2026-08-27).** All seven production
    call sites pass the SINGLE-STEP slot set ``_orientation_singlestep_slots`` — the AR
    splice (``ms2ms2ss_ar_temporal_mixture_pme.py``), the DH residual (same module,
    which slices ``[..., :singlestep_obs_len]`` BEFORE calling), the two deploy /
    compounded-prediction sites
    (``compounded_prediction_multistep_iterator.py``), the sampling-free, the
    resampled-particle and the ``trsf`` variants. This function is therefore NOT
    used on a composed multi-step observation; to align a composed window the
    caller must pass the composed slot expansion ``k * singlestep_obs_len + b``
    (the function itself is layout-agnostic and would handle it, but no caller
    does). Out-of-range slots stay silently skipped.

    **RLRP-747: the per-slot loop below is KEPT, on measurement.** The ticket asked
    for this function to be vectorized (gather -> stacked math -> single scatter).
    It was implemented and then **reverted**: the benchmark
    (``.junie/ai_artifact/scripts/bench_rlrp747_orientation_vectorization.py``,
    which still carries both variants) measured the vectorized form SLOWER on
    **all three** platforms, with a dispatched-op (kernel-launch) count that goes
    UP rather than down (``15 -> 20``). Speed-up factors, loop-relative
    (``> 1`` = vectorized wins):

    =============================  ==========  ==========  ==========  ====  ====
    shape (E x B*C x F)            MacBook     Orin CPU    Orin CUDA   loop  vec
                                   CPU         (arm64)     (sm_87)     ops   ops
    =============================  ==========  ==========  ==========  ====  ====
    5 x 1280 x 10   (S=1, prod)         0.77x       0.69x       0.74x    15    20
    5 x 5120 x 10   (S=1, prod)         0.43x       0.93x       0.76x    15    20
    5 x 1280 x 80   (S=8, hypoth.)      2.03x       1.95x       2.99x   120    34
    =============================  ==========  ==========  ==========  ====  ====

    Root cause: every one of the SEVEN production call sites passes the
    single-step slot set, so ``S == 1`` (see the contract above) — there is no loop
    to collapse, and the gather only adds an ``index_select`` copy on top of
    identical arithmetic. The ``S = 8`` row shows the crossover: gather-isation is
    a ~2-3x WIN as soon as there really are several slots. Per the plan's ROI table
    this item was **benchmark-gated**, and the gate closed it. Re-vectorize (the
    primitives are ready and tested) if and when a caller passes ``S > 1`` — e.g.
    the future gravity-vector representation or a composed-layout caller.

    RLRP-744 item E: whether this pass fires at all is gated by the model predicate
    ``_quaternion_ar_continuity_active`` (all model-side call sites), which is in turn
    gated by the ``ms_model.internal_orientation.enforce_continuity`` config key — so
    the SAME key that gates the ingestion continuity pass also enables/disables this
    reference-relative alignment. This function itself is unconditional; the config
    gating lives at the call sites.

    :param x: Tensor ``(..., F)`` whose attitude slot(s) are to be aligned in place
        (returned as a new tensor; ``x`` is not mutated).
    :param reference: Tensor ``(..., F_ref)`` broadcast-compatible with ``x`` and
        carrying the SAME attitude slot base indices (the sign reference).
    :param slots: Iterable of attitude base slot indices (each 4-D wide).
    :param eps: Norm floor for the ``S^3`` projection.
    :param project_to_s3: When True (default) L2-project each aligned slot onto
        ``S^3``. Set False for the sign-invariance-only use (RLRP-736 Item 2, the DH
        residual): the slot is flipped into ``reference``'s hemisphere but its
        magnitude is left untouched, so a non-antipodal raw MSE is unchanged.
    :return: ``x`` with every attitude slot sign-aligned to ``reference`` (and, when
        ``project_to_s3`` is True, unit-normalised).
    """
    ordered = sorted(int(s) for s in (slots or ()))
    if not ordered:
        return x
    out = x
    width = out.shape[-1]
    for s in ordered:
        if s + 4 > width:
            continue
        q = out[..., s : s + 4]
        q_ref = reference[..., s : s + 4]
        dot = (q * q_ref).sum(dim=-1, keepdim=True)
        # ``new_full`` (NOT a python float): a python scalar in ``torch.where``
        # goes through type promotion and can silently change the result dtype.
        sign = torch.where(dot < 0, q.new_full((), -1.0), q.new_full((), 1.0))
        q = q * sign
        if project_to_s3:
            q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
        out = torch.cat([out[..., :s], q, out[..., s + 4 :]], dim=-1)
    return out


def project_slots_to_unit_norm(
    x: torch.Tensor,
    slots,
    slot_width: int = 3,
    eps: float = 1e-12,
) -> torch.Tensor:
    """L2-project contiguous slot(s) of ``x`` onto the unit sphere — **no sign flip**.

    Task ``G-10`` of the RLRP-753 gravity-vector plan
    (``feat_gravity_vector_orientation_representation_plan_RLRP-753_20260724.md``).
    The sign-free sibling of :func:`align_quaternion_slots_to_reference`: it applies
    ONLY the manifold projection, because a 3-D direction (the body-frame gravity
    ``g_hat^B``) has NO double cover — ``g`` and ``-g`` are physically distinct, so
    a hemisphere flip against a reference frame would silently CORRUPT the
    prediction (ruling ``S5.6``).

    Deliberately mirrors :func:`align_quaternion_slots_to_reference`'s conventions:
    out-of-place (``x`` is not mutated), autograd-safe, bit-exact no-op for an
    empty ``slots``, out-of-range slots silently skipped, and idempotent on an
    already-unit input.

    :param x: Tensor ``(..., F)`` whose trailing axis carries the slot(s).
    :param slots: Iterable of base slot indices (each ``slot_width`` scalars wide).
    :param slot_width: Scalars per slot (3 for a ``gravity.{x,y,z}`` block).
    :param eps: Norm floor of the projection (guards the degenerate zero block).
    :return: ``x`` with every in-range slot unit-normalised.
    """
    ordered = sorted(int(s) for s in (slots or ()))
    if not ordered:
        return x
    out = x
    width = out.shape[-1]
    for s in ordered:
        if s + slot_width > width:
            continue
        block = out[..., s : s + slot_width]
        block = block / block.norm(dim=-1, keepdim=True).clamp_min(eps)
        out = torch.cat([out[..., :s], block, out[..., s + slot_width :]], dim=-1)
    return out


def encode_wxyz_to_internal_rep(
    q_wxyz: torch.Tensor, rep: InternalOrientationRep
) -> torch.Tensor:
    """Encode a 4-D ``[w, x, y, z]`` quaternion slot into a continuous internal rep.

    Standalone, **parameter-free** counterpart of :class:`OrientationInputEncoder`
    (RLRP-736 §S2.2-B model-forward wiring). It is the function a model uses to
    map its (already normalized) 4-D attitude input slot to the network-internal
    orientation features before the trunk. ``QUATERNION`` / ``QUATERNIONLEGACY``
    are strict passthroughs (bit-exact when the switch is off).

    :param q_wxyz: ``[..., 4]`` attitude quaternion.
    :param rep: The model-internal orientation representation.
    :return: ``[..., internal_rep_width(rep)]`` internal features.
    """
    if rep in (
        InternalOrientationRep.QUATERNION,
        InternalOrientationRep.QUATERNIONLEGACY,
    ):
        return q_wxyz
    rotation = wxyz_to_pp_SO3(q_wxyz)
    if rep is InternalOrientationRep.SO3_RELATIVE:
        # DEFERRED (RLRP-736): small-angle only; poor fit for the adverse-condition
        # (large per-step rotation) target. Kept for small-per-step datasets.
        return rotation.Log().tensor()
    matrix = rotation.matrix()  # [..., 3, 3]
    if rep is InternalOrientationRep.SIXD:
        # First two columns (Zhou et al. 2019 continuous 6D encoding).
        return matrix[..., :, 0:2].reshape(*matrix.shape[:-2], 6)
    # NINE_D_SVD: full flattened matrix.
    return matrix.reshape(*matrix.shape[:-2], 9)


def decode_internal_rep_to_wxyz(
    raw: torch.Tensor, rep: InternalOrientationRep
) -> torch.Tensor:
    """Decode a raw internal-rep trunk output slot into a **unit quaternion by construction**.

    Standalone, **parameter-free** counterpart of the geometric map inside
    :class:`OrientationOutputHead` (RLRP-736 §S2.2-B model-forward wiring): the
    linear projection is owned by the model's own (ensemble-aware) mean head, so
    this function only performs the (parameter-free) map onto ``SO(3)``, which is
    ensemble-safe (it operates purely on the trailing feature dimension).

    - ``QUATERNIONLEGACY``: return the raw 4-D slot **un-normalized** (the actual
      pre-RLRP-736 legacy contract; no by-construction unit guarantee).
    - ``QUATERNION``: unit-normalize the raw 4-D slot (the improvement over the
      legacy contract: valid-by-construction unit quaternion, still a cheap
      continuous-rep-free passthrough).
    - ``SIXD``: Gram-Schmidt to a rotation matrix (Zhou et al. 2019).
    - ``NINE_D_SVD``: SVD-project the raw 3x3 matrix onto ``SO(3)`` (infer-SVD;
      Chen et al. 2023, arXiv:2312.00462 / Geist et al. 2024).
    - ``SO3_RELATIVE``: **DEFERRED** (needs a per-step ``q_ref``); raises.

    :param raw: ``[..., internal_rep_width(rep)]`` raw trunk output slot.
    :param rep: The model-internal orientation representation.
    :return: ``[..., 4]`` ``[w, x, y, z]`` quaternion (unit for every rep except
        ``QUATERNIONLEGACY``, which passes the raw slot through un-normalized).
    """
    if rep is InternalOrientationRep.QUATERNIONLEGACY:
        # Actual legacy contract (pre-RLRP-736): raw slot, no unit normalization.
        return raw
    if rep is InternalOrientationRep.QUATERNION:
        return torch.nn.functional.normalize(raw, dim=-1)
    if rep is InternalOrientationRep.SO3_RELATIVE:
        raise NotImplementedError(
            "decode_internal_rep_to_wxyz(SO3_RELATIVE) is DEFERRED (RLRP-736): it "
            "requires a per-step reference quaternion q_ref and is a poor fit for "
            "the adverse-condition large-rotation target. Prefer NINE_D_SVD / SIXD."
        )
    if rep is InternalOrientationRep.SIXD:
        matrix = sixd_to_rotation_matrix(raw)
        return pp_SO3_to_wxyz(pp.mat2SO3(matrix, check=False))
    # NINE_D_SVD: raw 3x3 matrix -> SVD projection onto SO(3) (infer-SVD).
    matrix = raw.reshape(*raw.shape[:-1], 3, 3)
    rotation = svd_project_to_so3(matrix)
    return pp_SO3_to_wxyz(pp.mat2SO3(rotation, check=False))


# ---------------------------------------------------------------------------
# S^2 chart maths (RLRP-796 `D-793-1-D`, chart base point ruled `D-793-3-A`)
# ---------------------------------------------------------------------------

#: Largest geodesic radius (radians) a tangent vector may carry before the chart
#: at its base point tears. The exponential map at ``g_ref`` is a diffeomorphism
#: only for ``theta < pi``; AT ``theta == pi`` (the antipode ``-g_ref``) every
#: direction maps to the SAME point, the ``log`` map is discontinuous and the
#: gradient blows up. ``D-796-S1-1-A`` (operator): clamp below the tear and warn
#: ONCE per run rather than raising, so a long PME training run cannot die on a
#: single outlier sample; the hit counter below makes the resulting bias auditable.
S2_MAX_TANGENT_NORM = 3.10  # ~= pi - 0.0416 rad (~2.4 deg of margin)

#: Number of times the chart clamp has fired this run (see :func:`s2_clamp_hits`).
_S2_CLAMP_HITS = 0
_S2_CLAMP_WARNED = False


def s2_clamp_hits() -> int:
    """Times the ``S²`` chart-tear clamp fired since process start.

    ``D-796-S1-1-A``: the clamp keeps a run alive at the cost of a biased sample,
    so the count must be retrievable for post-hoc inspection — a run with a large
    count is not comparable to one with zero.

    :return: The monotonically-increasing hit count.
    """
    return _S2_CLAMP_HITS


def reset_s2_clamp_hits() -> None:
    """Reset the clamp counter and the once-per-run warning latch (tests)."""
    global _S2_CLAMP_HITS, _S2_CLAMP_WARNED
    _S2_CLAMP_HITS = 0
    _S2_CLAMP_WARNED = False


def _register_s2_clamp(hits: int) -> None:
    """Record ``hits`` clamp events, warning once per run (``D-796-S1-1-A``).

    :param hits: Number of clamped elements in the current call.
    """
    global _S2_CLAMP_HITS, _S2_CLAMP_WARNED
    if hits <= 0:
        return
    _S2_CLAMP_HITS += int(hits)
    if not _S2_CLAMP_WARNED:
        _S2_CLAMP_WARNED = True
        warnings.warn(
            f"RLRP-796 D-796-S1-1-A: an S^2 tangent vector reached the chart tear "
            f"and was CLAMPED to |v| <= {S2_MAX_TANGENT_NORM} rad "
            f"(~pi). This happens when a direction is ~180 deg from its chart base "
            f"point g_ref (the previous step's direction, D-793-3-A), i.e. the "
            f"per-step rotation is near-antipodal. The forward stays finite and "
            f"differentiable but the clamped samples are BIASED; read "
            f"orientation_heads.s2_clamp_hits() to judge whether the run is "
            f"well-conditioned. Warned once per run.",
            RuntimeWarning,
            stacklevel=3,
        )


def s2_tangent_basis(g_ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Orthonormal basis ``[..., 3, 2]`` of the tangent plane ``T_{g_ref} S²``.

    Numerically stable everywhere on the sphere, including at the poles. A naive
    ``cross(g_ref, e_z)`` basis DEGENERATES at ``g_ref = ±e_z`` — which is exactly
    the level-attitude case (``ĝ^B ≈ (0, 0, ±1)``), i.e. the most common input.
    The seed axis is therefore chosen per element as the world axis LEAST aligned
    with ``g_ref`` (Hughes & Moller 1999), guaranteeing a well-conditioned cross
    product for every direction.

    :param g_ref: ``[..., 3]`` chart base point; normalized internally, so a raw
        (non-unit) AR draw is safe.
    :param eps: Norm floor guarding a degenerate / zero base point.
    :return: ``[..., 3, 2]`` columns ``(e1, e2)``, orthonormal and both orthogonal
        to ``g_ref``.
    """
    base = g_ref / g_ref.norm(dim=-1, keepdim=True).clamp_min(eps)
    # Seed = the canonical axis least aligned with ``base`` (smallest |component|).
    least = base.abs().argmin(dim=-1, keepdim=True)
    seed = torch.zeros_like(base).scatter_(-1, least, 1.0)
    e1 = torch.cross(base, seed, dim=-1)
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(eps)
    e2 = torch.cross(base, e1, dim=-1)
    e2 = e2 / e2.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.stack((e1, e2), dim=-1)


def log_map_s2(
    g: torch.Tensor, g_ref: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Inverse chart: a direction on ``S²`` -> its 2-D tangent at ``g_ref``.

    ``v = theta * (u_1, u_2)`` where ``theta`` is the geodesic (great-circle)
    angle between ``g`` and ``g_ref`` and ``u`` is the unit direction of travel
    expressed in the :func:`s2_tangent_basis` frame. Exactly inverts
    :func:`exp_map_s2` for ``theta < pi``. This is the map that turns a TRAINING
    TARGET into the 2 numbers the network is scored against.

    Near the tear the radius is clamped (``D-796-S1-1-A``); see
    :func:`s2_clamp_hits`.

    :param g: ``[..., 3]`` direction (normalized internally).
    :param g_ref: ``[..., 3]`` chart base point (normalized internally).
    :param eps: Numerical floor for the norms and the ``sin(theta)`` division.
    :return: ``[..., 2]`` tangent coordinates.
    """
    base = g_ref / g_ref.norm(dim=-1, keepdim=True).clamp_min(eps)
    target = g / g.norm(dim=-1, keepdim=True).clamp_min(eps)
    cosine = (target * base).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(cosine)
    # Component of ``target`` orthogonal to ``base``, i.e. the direction of travel.
    orthogonal = target - cosine * base
    orthogonal = orthogonal / orthogonal.norm(dim=-1, keepdim=True).clamp_min(eps)
    clamped = theta.clamp_max(S2_MAX_TANGENT_NORM)
    _register_s2_clamp(int((theta > S2_MAX_TANGENT_NORM).sum().item()))
    tangent = clamped * orthogonal  # [..., 3], lies in T_{base} S^2
    basis = s2_tangent_basis(base, eps=eps)  # [..., 3, 2]
    return torch.einsum("...i,...ij->...j", tangent, basis)


def exp_map_s2(
    v: torch.Tensor, g_ref: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Chart: a 2-D tangent at ``g_ref`` -> a **unit** direction on ``S²``.

    ``exp_{g_ref}(v) = cos(theta) * g_ref + sin(theta) * d``, with
    ``theta = |v|`` (the geodesic distance travelled) and ``d`` the unit tangent
    direction. Unit by construction for every input, and smooth at ``v = 0``
    (the ``sin(theta)/theta`` ratio is guarded), so the network may predict the
    base point exactly.

    Near the tear the radius is clamped (``D-796-S1-1-A``); see
    :func:`s2_clamp_hits`.

    :param v: ``[..., 2]`` tangent coordinates in the :func:`s2_tangent_basis` frame.
    :param g_ref: ``[..., 3]`` chart base point (normalized internally).
    :param eps: Numerical floor for the norms and the small-angle ratio.
    :return: ``[..., 3]`` unit direction.
    """
    base = g_ref / g_ref.norm(dim=-1, keepdim=True).clamp_min(eps)
    basis = s2_tangent_basis(base, eps=eps)  # [..., 3, 2]
    theta_raw = v.norm(dim=-1, keepdim=True)
    _register_s2_clamp(int((theta_raw > S2_MAX_TANGENT_NORM).sum().item()))
    theta = theta_raw.clamp_max(S2_MAX_TANGENT_NORM)
    # Ambient tangent vector, re-scaled to the (possibly clamped) radius. The
    # ``clamp_min`` keeps the ratio finite AND the gradient defined at v = 0.
    ambient = torch.einsum("...j,...ij->...i", v, basis)
    ambient = ambient * (theta / theta_raw.clamp_min(eps))
    direction = ambient / theta.clamp_min(eps)
    return torch.cos(theta) * base + torch.sin(theta) * direction


def _assert_quaternion_block_rep(rep: InternalOrientationRep, who: str) -> None:
    """Fail loud when an ``S²`` rep reaches a QUATERNION-block-only seam.

    RLRP-796 Stage 1. The two ``nn.Module`` wrappers below hard-code the 4-D
    ``[w, x, y, z]`` boundary and the ``SO(3)`` maps; the model seam does NOT use
    them (it calls the parameter-free functions), so rather than generalize dead
    code they refuse an ``S²`` rep explicitly — a silent wrong-branch fall-through
    is the failure mode this stage's flag audit is guarding against.

    :param rep: The model-internal orientation representation.
    :param who: Name of the refusing seam, for the message.
    :raises NotImplementedError: When ``rep`` is an ``S²`` rep.
    """
    if is_s2_orientation_rep(rep):
        raise NotImplementedError(
            f"{who} is QUATERNION-block only (it assumes a 4-D [w,x,y,z] boundary "
            f"and the SO(3) maps), but got the S^2 rep {rep}. The gravity block "
            f"path is the parameter-free encode_gravity_to_internal_rep / "
            f"decode_internal_rep_to_gravity pair, which is what the model seam "
            f"calls (RLRP-796 Stage 1)."
        )


def encode_gravity_to_internal_rep(
    g_xyz: torch.Tensor, rep: InternalOrientationRep
) -> torch.Tensor:
    """Encode a 3-D body-frame gravity DIRECTION slot into its internal rep.

    ``S²`` sibling of :func:`encode_wxyz_to_internal_rep`, deliberately a separate
    function rather than an overload: that function's name and its ``[w, x, y, z]``
    contract do not generalize to a 3-vector, and ``ĝ^B`` has no double cover, so
    none of the quaternion-side hemisphere handling applies (RLRP-796 Stage 1).

    - ``S2_IDENTITY``: strict passthrough (the absolute 3-D direction).
    - ``S2_TANGENT``: ALSO a strict passthrough — the tangent rep is output-only
      (ruling ``D-796-S1-2-A``). Encoding the input relative to a moving base point
      would destroy the absolute tilt at the start of every window, so the network
      could no longer tell upright from inverted; only the trunk OUTPUT narrows.

    :param g_xyz: ``[..., 3]`` body-frame gravity direction.
    :param rep: The model-internal orientation representation (must be ``S²``).
    :return: ``[..., internal_rep_in_width(rep, 3)]`` internal features.
    :raises NotImplementedError: For a rep with no ``S²`` encode.
    """
    if rep in (
        InternalOrientationRep.S2_IDENTITY,
        InternalOrientationRep.S2_TANGENT,
    ):
        return g_xyz
    raise NotImplementedError(
        f"encode_gravity_to_internal_rep({rep}) is not implemented; the S^2 "
        f"encode path serves the GRAVITY_DIRECTION block only (RLRP-796)."
    )


def decode_internal_rep_to_gravity(
    raw: torch.Tensor,
    rep: InternalOrientationRep,
    eps: float = 1e-8,
    g_ref: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decode a raw internal-rep slot into a **unit gravity direction by construction**.

    ``S²`` sibling of :func:`decode_internal_rep_to_wxyz`. Parameter-free and
    ensemble-safe (operates purely on the trailing feature axis).

    - ``S2_IDENTITY``: L2-project the raw 3-D slot onto ``S²``. Same semantics as
      :func:`project_slots_to_unit_norm` at ``slot_width=3`` — deliberately NOT a
      sign-flip / hemisphere canonicalisation, because ``ĝ^B`` has no double cover.
    - ``S2_TANGENT``: ``exp``-map the raw 2-D tangent at ``g_ref``, which by ruling
      ``D-793-3-A`` is the PREVIOUS step's direction. ``g_ref`` is therefore
      MANDATORY for this rep — a missing reference is a wiring bug, never a
      silently-defaulted identity, so it fails loud.

    :param raw: ``[..., internal_rep_out_width(rep, 3)]`` raw trunk output slot.
    :param rep: The model-internal orientation representation (must be ``S²``).
    :param eps: Norm floor of the projection (guards a degenerate zero slot).
    :param g_ref: ``[..., 3]`` chart base point, required by ``S2_TANGENT`` and
        ignored by ``S2_IDENTITY``.
    :return: ``[..., 3]`` unit body-frame gravity direction.
    :raises ValueError: When ``S2_TANGENT`` is decoded without a reference.
    :raises NotImplementedError: For a rep with no ``S²`` decode.
    """
    if rep is InternalOrientationRep.S2_IDENTITY:
        return raw / raw.norm(dim=-1, keepdim=True).clamp_min(eps)
    if rep is InternalOrientationRep.S2_TANGENT:
        if g_ref is None:
            raise ValueError(
                "decode_internal_rep_to_gravity(S2_TANGENT) requires g_ref: the "
                "chart base point is the PREVIOUS step's direction (RLRP-796 "
                "'D-793-3-A'), so decoding without it would silently score the "
                "tangent against the wrong chart."
            )
        return exp_map_s2(raw, g_ref)
    raise NotImplementedError(
        f"decode_internal_rep_to_gravity({rep}) is not implemented; the S^2 "
        f"decode path serves the GRAVITY_DIRECTION block only (RLRP-796)."
    )


def encode_orientation_slot_to_internal_rep(
    slot: torch.Tensor, rep: InternalOrientationRep
) -> torch.Tensor:
    """Block-dispatching encode: quaternion slot or gravity-direction slot.

    Single entry point for the model seam so a call site does not have to know
    which orientation block it is wired to — the REP determines the block
    (``is_s2_orientation_rep``), which is why the rep x block compatibility table
    is validated once at setup (``D-796-S1-5-A``).

    :param slot: ``[..., external_width]`` external orientation slot.
    :param rep: The model-internal orientation representation.
    :return: ``[..., internal_rep_in_width(rep, external_width)]`` internal feats.
    """
    if is_s2_orientation_rep(rep):
        return encode_gravity_to_internal_rep(slot, rep)
    return encode_wxyz_to_internal_rep(slot, rep)


def decode_internal_rep_to_orientation_slot(
    raw: torch.Tensor,
    rep: InternalOrientationRep,
    g_ref: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Block-dispatching decode; sibling of :func:`encode_orientation_slot_to_internal_rep`.

    :param raw: ``[..., internal_rep_out_width(rep, external_width)]`` raw slot.
    :param rep: The model-internal orientation representation.
    :param g_ref: ``[..., 3]`` chart base point, required by ``S2_TANGENT`` only.
    :return: ``[..., external_width]`` external orientation slot, on its manifold
        by construction (unit quaternion on ``S³`` / unit direction on ``S²``,
        except the deliberately-raw ``QUATERNIONLEGACY`` baseline).
    """
    if is_s2_orientation_rep(rep):
        return decode_internal_rep_to_gravity(raw, rep, g_ref=g_ref)
    return decode_internal_rep_to_wxyz(raw, rep)


class OrientationInputEncoder(nn.Module):
    """Encode the 4-D quaternion input slot into a continuous internal feature.

    Pure in-graph, parameter-free transform (NOT a dataset stage). For the
    neutral ``QUATERNION`` rep it is a strict passthrough so the model stays
    bit-exact when the switch is off.

    - ``QUATERNION`` / ``QUATERNIONLEGACY``: passthrough (4-D).
    - ``SO3_RELATIVE`` (**DEFERRED**, RLRP-736): ``Log`` tangent about identity
      (3-D). A running reference ``q_ref`` is handled downstream by the head, not
      here. Kept for completeness / small-per-step-rotation datasets only; it is
      a poor fit for our adverse-condition (large per-step rotation) target and
      is not on the critical path (see :class:`InternalOrientationRep`).
    - ``SIXD``: first two columns of the rotation matrix (6-D), continuous.
    - ``NINE_D_SVD``: flattened rotation matrix (9-D), continuous.
    """

    def __init__(self, rep: InternalOrientationRep):
        super().__init__()
        _assert_quaternion_block_rep(rep, "OrientationInputEncoder")
        self._rep = rep
        self.out_width = internal_rep_width(rep)

    def forward(self, q_wxyz: torch.Tensor) -> torch.Tensor:
        """:param q_wxyz: ``[..., 4]`` attitude quaternion; :return: internal feats."""
        if self._rep in (
            InternalOrientationRep.QUATERNION,
            InternalOrientationRep.QUATERNIONLEGACY,
        ):
            return q_wxyz
        rotation = wxyz_to_pp_SO3(q_wxyz)
        if self._rep is InternalOrientationRep.SO3_RELATIVE:
            # DEFERRED (RLRP-736): small-angle only; poor fit for the
            # adverse-condition (large per-step rotation) target. Kept landed.
            return rotation.Log().tensor()
        matrix = rotation.matrix()  # [..., 3, 3]
        if self._rep is InternalOrientationRep.SIXD:
            # First two columns (Zhou et al. 2019 continuous 6D encoding).
            return matrix[..., :, 0:2].reshape(*matrix.shape[:-2], 6)
        # NINE_D_SVD: full flattened matrix.
        return matrix.reshape(*matrix.shape[:-2], 9)


class OrientationOutputHead(nn.Module):
    """Map trunk activations to a **unit quaternion by construction** (4-D slot).

    The head owns the linear projection from the trunk feature width to the
    raw representation width, then maps it onto ``SO(3)`` so the emitted 4-D
    ``[w, x, y, z]`` quaternion is always valid:

    - ``QUATERNION`` / ``QUATERNIONLEGACY`` (neutral): the caller is expected to
      bypass the head; a passthrough is provided for completeness (raw 4-D,
      caller normalizes as today for ``QUATERNION`` / leaves it raw for
      ``QUATERNIONLEGACY``). Not used on the bit-exact default path.
    - ``SO3_RELATIVE`` (**DEFERRED**, RLRP-736): predict a 3-D ``so3`` tangent
      ``delta`` and right-compose with the per-step reference ``q_ref``:
      ``R_next = SO3(q_ref) @ Exp(delta)`` (unit by construction; the standard
      small-relative-rotation mitigation of the double cover, Geist et al. 2024).
      Only valid for demonstrably small per-step rotations; **deferred** because
      our target is the adverse-condition large-rotation regime where the
      small-angle premise breaks (near-π discontinuity, R-B3). The remaining
      per-step ``q_ref`` plumbing is deferred — prefer ``NINE_D_SVD`` / ``SIXD``.
    - ``SIXD``: predict 6-D, Gram-Schmidt to a rotation matrix (Zhou et al. 2019).
    - ``NINE_D_SVD``: predict a raw 9-D matrix ``M``. **Train-unorthogonalized /
      infer-SVD** (Chen et al. 2023, arXiv:2312.00462): the raw ``M`` is exposed
      via :attr:`last_raw_matrix` so the training loss can be applied directly on
      it, while the *returned* quaternion is the SVD projection of ``M`` onto
      ``SO(3)`` (``pp.mat2SO3``) so the external boundary is always a valid
      rotation.
    """

    def __init__(self, rep: InternalOrientationRep, in_features: int):
        super().__init__()
        _assert_quaternion_block_rep(rep, "OrientationOutputHead")
        self._rep = rep
        self.raw_width = internal_rep_width(rep)
        self.proj = nn.Linear(in_features, self.raw_width)
        # Last raw (un-orthogonalized) matrix from a NINE_D_SVD forward, kept so
        # the train-unorthogonalized loss can score M directly (Chen et al. 2023).
        self.last_raw_matrix: torch.Tensor | None = None

    def forward(
        self, trunk_feats: torch.Tensor, q_ref_wxyz: torch.Tensor | None = None
    ) -> torch.Tensor:
        """:param trunk_feats: ``[..., in_features]``; :return: ``[..., 4]`` unit quaternion."""
        raw = self.proj(trunk_feats)
        self.last_raw_matrix = None

        if self._rep in (
            InternalOrientationRep.QUATERNION,
            InternalOrientationRep.QUATERNIONLEGACY,
        ):
            return raw  # passthrough; caller handles unit-norm as today

        if self._rep is InternalOrientationRep.SO3_RELATIVE:
            # DEFERRED (RLRP-736): kept landed for small-per-step-rotation
            # datasets; not on the adverse-condition critical path.
            if q_ref_wxyz is None:
                raise ValueError(
                    "OrientationOutputHead(SO3_RELATIVE) requires a per-step "
                    "reference quaternion q_ref_wxyz (R_next = SO3(q_ref) @ Exp(delta))."
                )
            r_next = wxyz_to_pp_SO3(q_ref_wxyz) @ pp.so3(raw).Exp()
            return pp_SO3_to_wxyz(r_next)

        if self._rep is InternalOrientationRep.SIXD:
            # Gram-Schmidt already yields an orthonormal rotation matrix, so the
            # pypose orthogonality check is skipped (check=False) to avoid false
            # rejections from float tolerance.
            matrix = sixd_to_rotation_matrix(raw)
            return pp_SO3_to_wxyz(pp.mat2SO3(matrix, check=False))

        # NINE_D_SVD: reshape to a raw 3x3 matrix, keep the *un-orthogonalized*
        # M for the training loss (train-unorthogonalized), and SVD-project it
        # onto SO(3) for the emitted quaternion (infer-SVD).
        matrix = raw.reshape(*raw.shape[:-1], 3, 3)
        self.last_raw_matrix = matrix
        rotation = svd_project_to_so3(matrix)
        return pp_SO3_to_wxyz(pp.mat2SO3(rotation, check=False))


def rotation_geodesic_loss(
    q_pred_wxyz: torch.Tensor,
    q_target_wxyz: torch.Tensor,
    keep_batch: bool = False,
) -> torch.Tensor:
    """Squared geodesic (``SO(3)``) angle between two ``[..., 4]`` quaternions.


    ``geodesic = ‖Log(SO3(q_pred)^-1 · SO3(q_target))‖``; returns the mean of the
    squared angle (radians²). This is the diagnostic/auxiliary metric; note
    (Geist et al. 2024) the squared geodesic has a vanishing gradient near 0 and
    curvature issues near π, hence :func:`rotation_chordal_loss` is offered as a
    configurable alternative training objective.

    This is the canonical (PyPose) replacement for the deprecated in-house
    ``_quaternion_geodesic_loss`` (removed RLRP-746); the two are numerically
    identical on unit quaternions (both = mean squared ``SO(3)`` angle).

    :param keep_batch: When True, return the per-element squared angle with a
        trailing singleton ``(..., 1)`` instead of the scalar mean, preserving
        the batch dimension for the composite auto-weighting path
        (:meth:`FeatureGroupSpec.extra_loss`, RLRP-751).
    """
    r_pred = wxyz_to_pp_SO3(q_pred_wxyz)
    r_target = wxyz_to_pp_SO3(q_target_wxyz)
    angle = (r_pred.Inv() @ r_target).Log().tensor().norm(dim=-1)
    per_elem = angle**2
    if keep_batch:
        return per_elem.unsqueeze(-1)
    return per_elem.mean()


def rotation_chordal_loss(
    q_pred_wxyz: torch.Tensor,
    q_target_wxyz: torch.Tensor,
    keep_batch: bool = False,
) -> torch.Tensor:
    """Chordal / Frobenius rotation-matrix loss between two ``[..., 4]`` quaternions.


    ``‖R_pred - R_target‖_F²`` averaged over the batch. Recommended by Geist et
    al. 2024 as competitive-to-better than the geodesic for ``R^9`` outputs and
    free of the geodesic-squared gradient pathologies.

    This is the canonical (PyPose) replacement for the deprecated in-house
    ``_quaternion_chordal_loss`` (removed RLRP-746); the two are numerically
    identical on unit quaternions since
    ``‖R_pred - R_target‖_F² = 4(1 - cos θ) = 8(1 - w_rel²)``.

    :param keep_batch: When True, return the per-element Frobenius distance with
        a trailing singleton ``(..., 1)`` instead of the scalar mean, preserving
        the batch dimension for the composite auto-weighting path
        (:meth:`FeatureGroupSpec.extra_loss`, RLRP-751).
    """
    r_pred = wxyz_to_pp_SO3(q_pred_wxyz).matrix()
    r_target = wxyz_to_pp_SO3(q_target_wxyz).matrix()
    per_elem = ((r_pred - r_target) ** 2).sum(dim=(-2, -1))
    if keep_batch:
        return per_elem.unsqueeze(-1)
    return per_elem.mean()


#: Norm floor of the ``S^2`` unit projection in :func:`gravity_direction_loss`.
#: A plain ``clamp_min`` on the norm (never a division by a raw norm) keeps the
#: gradient finite for a degenerate all-zero block.
_S2_NORM_EPS = 1e-12


def gravity_direction_loss(
    g_pred: torch.Tensor,
    g_target: torch.Tensor,
    keep_batch: bool = False,
) -> torch.Tensor:
    """Chordal (``S^2``) error between two 3-D body-frame gravity directions.

    Task ``G-11`` / ruling ``D-loss`` of the RLRP-753 gravity-vector plan
    (``feat_gravity_vector_orientation_representation_plan_RLRP-753_20260724.md``).
    The ``S^2`` counterpart of :func:`rotation_chordal_loss`: both blocks are
    unit-normalized (the prediction is NOT guaranteed to be on ``S^2``) and
    scored with the squared chordal distance

    ``‖ĝ_pred - ĝ_target‖² = 2 (1 - cos θ)``,

    i.e. ``0`` at coincidence, ``4`` at the antipode, and strictly monotone in
    the angle ``θ ∈ [0, π]``. The chordal form (rather than ``acos`` of the
    cosine, i.e. the geodesic angle) is deliberate: ``acos`` has an INFINITE
    derivative at both ``θ = 0`` and ``θ = π``, which is exactly the
    coincidence / antipode regime this term is trained through, whereas the
    chordal form is a polynomial in the normalized blocks and therefore smooth
    everywhere — the same argument Geist et al. 2024 make for the ``SO(3)``
    chordal loss over the squared geodesic (see :func:`rotation_chordal_loss`).

    **No sign folding.** Unlike a quaternion, a 3-D direction has NO double
    cover: ``g`` and ``-g`` are physically distinct (upright vs. upside-down),
    so the antipode must be the MAXIMALLY penalised configuration and the
    hemisphere flip of :func:`align_quaternion_slots_to_reference` must NOT be
    applied here (RLRP-753 ``S5.6``, mirrored by the projection-only ``S^2``
    branch of ``CompoundedPredictionMultiStepIterator._apply_unit_norm_ar_projection``).

    :param g_pred: Predicted gravity-direction block ``[..., 3]`` (any norm).
    :param g_target: Target gravity-direction block ``[..., 3]`` (any norm).
    :param keep_batch: When True, return the per-element chordal distance with a
        trailing singleton ``(..., 1)`` instead of the scalar mean, preserving
        the batch dimension for the composite auto-weighting path
        (:meth:`FeatureGroupSpec.extra_loss`, RLRP-751). Mirrors the
        :func:`rotation_geodesic_loss` / :func:`rotation_chordal_loss` contract.
    :return: Scalar mean chordal distance when ``keep_batch`` is False, else the
        per-element ``(..., 1)`` tensor.
    """
    pred_unit = g_pred / g_pred.norm(dim=-1, keepdim=True).clamp_min(_S2_NORM_EPS)
    target_unit = g_target / g_target.norm(dim=-1, keepdim=True).clamp_min(_S2_NORM_EPS)
    per_elem = ((pred_unit - target_unit) ** 2).sum(dim=-1)
    if keep_batch:
        return per_elem.unsqueeze(-1)
    return per_elem.mean()


# Numerical floors: tangent variance is exponentiated from a log-variance, and
# the right-Jacobian determinant is log'd — both are clamped to avoid ``log(0)``.
_TANGENT_NLL_EPS = 1e-8


def _so3_right_jacobian_log_det(angle: torch.Tensor) -> torch.Tensor:
    """Return ``log|det J_r(φ)|`` for a rotation of magnitude ``angle`` (radians).

    The right Jacobian of ``SO(3)`` has closed-form determinant
    ``det J_r(φ) = 2 (1 - cos θ) / θ²`` with ``θ = ‖φ‖`` (Forster et al. 2016,
    "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry", eq. 8/9).
    It is the volume-change factor of the exponential map, so it is the correct
    normalization to turn a plain tangent-space Gaussian into a proper
    concentrated ``SO(3)`` density. The determinant → 1 (log → 0) as ``θ → 0``.

    :param angle: Rotation magnitude ``θ = ‖φ‖`` ``[...]`` (radians).
    :return: ``log|det J_r|`` ``[...]`` (0 in the small-angle limit).
    """
    theta_sq = angle.pow(2)
    # det = 2(1-cosθ)/θ² → 1 as θ→0; guard the removable singularity at θ=0.
    det = torch.where(
        angle > _TANGENT_NLL_EPS,
        2.0 * (1.0 - torch.cos(angle)) / theta_sq.clamp_min(_TANGENT_NLL_EPS),
        torch.ones_like(angle),
    )
    return torch.log(det.clamp_min(_TANGENT_NLL_EPS))


def s2_tangent_nll(
    g_pred: torch.Tensor,
    g_target: torch.Tensor,
    tangent_log_var: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    """Gaussian NLL in the 2-D tangent of ``S²`` at the predicted direction.

    RLRP-796 ``D-793-4`` — the `S^2` analogue of :func:`rotation_tangent_nll`, and
    the actual research contribution of the ``S2_TANGENT`` arm: the gravity
    direction has **2** degrees of freedom, so a diagonal Gaussian scored in the
    ambient 3-D slot (``_base_nll``) wastes one covariance dimension on the
    RADIAL direction the manifold forbids. Here the residual is the geodesic
    displacement from the prediction to the target expressed in the tangent,
    ``v = log_{ĝ_pred}(ĝ_target)`` ``[..., 2]``, and the model emits a diagonal
    (log-)variance in that same 2-D tangent:

    ``0.5 · [ Σᵢ (vᵢ² / σᵢ² + log σᵢ²) + 2·log(2π) ]``.

    **Signature parity over the plan sketch.** The plan sketched
    ``(v_pred, logvar, g_target, g_ref)``; this takes the EXTERNAL pred/target
    slots and derives the residual internally, exactly as its ``SO(3)`` sibling
    does with ``Log(R_pred⁻¹ R_target)``. That makes it a **drop-in swap** at the
    single PME split-NLL seam and needs no additional reference threaded into the
    loss — the chart is the prediction itself, which is always available there.
    It is also the right chart on principle: the residual is measured where the
    density is centred, so ``v = 0`` at a perfect prediction.

    **No volume correction.** :func:`rotation_tangent_nll` offers the Forster et
    al. 2016 right-Jacobian normalization; the ``S²`` exponential map's own
    volume factor is ``sin(θ)/θ``, which is a strictly smaller correction over
    the well-conditioned regime this rep targets (``D-793-3-A`` keeps ``θ``
    small by construction) and is deliberately NOT applied, so the term stays the
    plain tangent Gaussian. Near the tear the residual is clamped by
    :func:`log_map_s2` (``D-796-S1-1-A``), so the term stays finite there.

    **No sign folding** — ``ĝ^B`` has no double cover, so ``g`` and ``-g`` are
    physically distinct and the antipode must remain maximally penalised
    (see :func:`gravity_direction_loss`).

    :param g_pred: Predicted direction(s) ``[..., 3]`` (normalized internally).
    :param g_target: Target direction(s) ``[..., 3]`` (normalized internally).
    :param tangent_log_var: Predicted per-axis log-variance ``[..., 2]`` in the
        tangent at ``g_pred``.
    :param reduce: When ``True`` (default) return the scalar mean NLL; when
        ``False`` return the per-sample NLL tensor ``[...]``, which is what the
        mixture / PME per-particle log-sum-exp paths require.
    :return: Scalar mean NLL (nats) when ``reduce`` else the per-sample tensor.
    """
    residual = log_map_s2(g_target, g_pred)  # v, [..., 2]
    log_var = tangent_log_var
    inv_var = torch.exp(-log_var)
    two_pi = residual.new_tensor(2.0 * torch.pi)
    nll = 0.5 * (
        (residual.pow(2) * inv_var).sum(dim=-1)
        + log_var.sum(dim=-1)
        + 2.0 * torch.log(two_pi)
    )
    if reduce:
        return nll.mean()
    return nll


def rotation_tangent_nll(
    q_pred_wxyz: torch.Tensor,
    q_target_wxyz: torch.Tensor,
    tangent_log_var: torch.Tensor,
    with_right_jacobian: bool = False,
    reduce: bool = True,
) -> torch.Tensor:
    """Gaussian negative-log-likelihood in the ``so3`` tangent of the relative rotation.

    The probabilistic attitude objective for the ★ MTM-Pro path (plan §4A-B).
    The residual is the relative rotation expressed in the tangent,
    ``φ = Log(SO3(q_pred)⁻¹ · SO3(q_target))`` ``[..., 3]``, and the model emits a
    diagonal (log-)variance ``Σ = diag(exp(tangent_log_var))`` in that same
    tangent. The (diagonal) Gaussian NLL is

    ``0.5 · [ Σᵢ (φᵢ² / σᵢ² + log σᵢ²) + 3·log(2π) ]``.

    This is only *approximately* a proper ``SO(3)`` density — a correct density
    requires the **right-Jacobian normalization** (Forster et al. 2016, eq. 14).
    Two sub-modes (plan §4A-B):

    - ``with_right_jacobian=False`` (default): the plain tangent-Gaussian NLL —
      cheap, but biased for large angles.
    - ``with_right_jacobian=True``: add ``log|det J_r(φ)|`` so the tangent density
      is normalized to a proper concentrated ``SO(3)`` density (correct, slightly
      costlier).

    :param q_pred_wxyz: Predicted mean quaternion(s) ``[..., 4]`` (``[w,x,y,z]``).
    :param q_target_wxyz: Target quaternion(s) ``[..., 4]`` (``[w,x,y,z]``).
    :param tangent_log_var: Predicted per-axis log-variance ``[..., 3]`` in the
        ``so3`` tangent.
    :param with_right_jacobian: When ``True``, apply the Forster et al. 2016
        right-Jacobian volume correction.
    :param reduce: When ``True`` (default) return the scalar mean NLL; when
        ``False`` return the per-sample NLL tensor ``[...]`` (before ``.mean()``).
        The unreduced mode is mandatory for the mixture/PME + VI per-particle
        log-sum-exp paths and for splicing into the base loss' unreduced
        per-element NLL tensor (RLRP-736 §4A-B, review 2026-07-13).
    :return: Scalar mean NLL (nats) when ``reduce`` else the per-sample NLL
        tensor ``[...]``.
    """
    r_pred = wxyz_to_pp_SO3(q_pred_wxyz)
    r_target = wxyz_to_pp_SO3(q_target_wxyz)
    residual = (r_pred.Inv() @ r_target).Log().tensor()  # φ, [..., 3]

    log_var = tangent_log_var
    inv_var = torch.exp(-log_var)
    two_pi = residual.new_tensor(2.0 * torch.pi)
    nll = 0.5 * (
        (residual.pow(2) * inv_var).sum(dim=-1)
        + log_var.sum(dim=-1)
        + 3.0 * torch.log(two_pi)
    )
    if with_right_jacobian:
        nll = nll + _so3_right_jacobian_log_det(residual.norm(dim=-1))
    if reduce:
        return nll.mean()
    return nll
