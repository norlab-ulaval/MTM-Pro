# coding=utf-8
from typing import Optional, Union

import numpy as np
import omegaconf
import torch
from scipy.spatial.transform import Rotation
import trajectory_container_tools as tct
from tools.console_tools.message import consol_msg_universal_one_liner
from trajectory_container_tools.dataclasses import TestMotionTrajectoryDataclass


# ---------------------------------------------------------------------------
# Angular-velocity saturation cap (rad/s) — sourced from the observation spec.
# ---------------------------------------------------------------------------
# ``compute_position_from_velocity_and_attitude`` clamps the *magnitude* of the
# angular-velocity vector ``|omega|`` (in rad/s) before integrating attitude, as
# a numerical safety rail against bad / overflowing model predictions feeding
# the ``omega * dt`` rotation-vector step. The cap must therefore sit *above*
# the fastest body rate the platform actually exhibits (so real motion is never
# clipped) while still being tight enough to catch non-physical outliers.
#
# ``|omega|`` is an *observation* dimension (``angular_vels.{x,y,z}`` in
# ``obs_dims``), so its physical bound is a property of the observation spec, not
# of the action space. The per-platform value therefore lives entirely in the
# per-platform simulator (observation) config under
# ``obs_physical_bounds.angular_velocity_max_rad_s``:
#
#   UGV (ugv_general.yaml):        3.0  rad/s  (husky_adverse  max 1.44, ~2.1x headroom)
#   UAV (quadcopter_general.yaml): 30.0 rad/s  (neurobem_adverse max 18.27, ~1.6x headroom)
#
# (Empirical audit: see ``.junie/ai_artifact/reports/max_angular_velocity_default_audit_20260613.md``.)
#
# The constant below is *only* a conservative fallback used when the config does
# not declare the bound (e.g. malformed / partial cfg). It is intentionally the
# loosest physically-meaningful cap so it never clips real UGV or UAV motion.
DEFAULT_MAX_ANGULAR_VELOCITY_RAD_S: float = 30.0


def resolve_max_angular_velocity_from_cfg(cfg) -> float:
    """Resolve the angular-velocity saturation cap (rad/s) from the obs spec.

    The bound is sourced from the per-platform observation spec at
    ``cfg.environment.obs_physical_bounds.angular_velocity_max_rad_s`` (declared
    in ``ugv_general.yaml`` / ``quadcopter_general.yaml``), since ``|omega|`` is
    an observation dimension (``angular_vels.{x,y,z}``). Falls back to the
    conservative :data:`DEFAULT_MAX_ANGULAR_VELOCITY_RAD_S` (emitting a console
    notice) when the config is missing or omits the bound.

    :param cfg: the Hydra/omegaconf config (expects
        ``cfg.environment.obs_physical_bounds.angular_velocity_max_rad_s``).
    :return: the angular-velocity saturation cap in rad/s for the active platform.
    """
    try:
        cap = cfg.environment.obs_physical_bounds.angular_velocity_max_rad_s
    except Exception:
        cap = None

    if cap is None:
        consol_msg_universal_one_liner(
            "resolve_max_angular_velocity_from_cfg: "
            "'environment.obs_physical_bounds.angular_velocity_max_rad_s' not found in cfg; "
            f"falling back to default {DEFAULT_MAX_ANGULAR_VELOCITY_RAD_S} rad/s."
        )
        return DEFAULT_MAX_ANGULAR_VELOCITY_RAD_S

    return float(cap)


def report_angular_velocity_clamp(
    omega_magnitudes: Union[np.ndarray, torch.Tensor],
    max_angular_velocity: float,
    context: str = "compute_position_from_velocity_and_attitude",
) -> Optional[dict]:
    """Emit an aggregated, always-on summary when ``|omega|`` exceeds the cap.

    The attitude integrator rescales any angular-velocity sample whose magnitude
    ``|omega| = norm(angular_vels.{x,y,z})`` exceeds ``max_angular_velocity`` (a
    numerical-overflow rail). Such a clamp means the *input* (typically a model
    prediction at deploy time) is physically implausible — exactly the signal a
    resilient-controller / multi-step-precision study wants surfaced. This helper
    therefore reports the clamp statistics (count, fraction, max overshoot)
    **regardless of any ``debug`` flag**, instead of the previous per-index print
    that was silenced in production (``debug=False``).

    :param omega_magnitudes: per-sample angular-velocity magnitudes ``|omega|`` (rad/s),
        as a numpy array or torch tensor of any shape (flattened internally).
    :param max_angular_velocity: the saturation cap (rad/s).
    :param context: caller label used in the console message.
    :return: a summary dict when at least one sample is clamped, else ``None``.
    """
    if isinstance(omega_magnitudes, torch.Tensor):
        mags = omega_magnitudes.detach().to("cpu", torch.float64).reshape(-1).numpy()
    else:
        mags = np.asarray(omega_magnitudes, dtype=float).reshape(-1)

    if mags.size == 0:
        return None

    over = mags > max_angular_velocity
    n_violations = int(over.sum())
    if n_violations == 0:
        return None

    n_total = int(mags.size)
    fraction = n_violations / n_total
    max_observed = float(mags.max())
    summary = {
        "context": context,
        "n_violations": n_violations,
        "n_total": n_total,
        "fraction": fraction,
        "max_observed_rad_s": max_observed,
        "cap_rad_s": float(max_angular_velocity),
    }
    consol_msg_universal_one_liner(
        f"{context}: clamped angular velocity on {n_violations}/{n_total} "
        f"samples ({fraction:.1%}); max |omega|={max_observed:.2f} rad/s "
        f"exceeded cap {max_angular_velocity:.2f} rad/s."
    )
    return summary


# ---------------------------------------------------------------------------
# Torch-native quaternion utilities (Phase 1)
# ---------------------------------------------------------------------------
# Convention: quaternions are stored as [w, x, y, z] tensors unless noted.
# All functions support batched inputs (..., 4) or (..., 3).
# ---------------------------------------------------------------------------


def quaternion_normalize(q: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion tensor, returning identity for near-zero norms.

    :param q: quaternion tensor of shape (..., 4) in [w, x, y, z] order.
    :return: unit quaternion tensor of the same shape.
    """
    norm = torch.norm(q, dim=-1, keepdim=True)
    safe = norm > 1e-10
    identity = torch.zeros_like(q)
    identity[..., 0] = 1.0
    result = torch.where(safe, q / norm, identity)
    # Handle NaN after normalization
    nan_mask = ~torch.isfinite(result).all(dim=-1, keepdim=True)
    result = torch.where(nan_mask, identity, result)
    return result


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternion tensors.

    :param q1: first quaternion (..., 4) in [w, x, y, z].
    :param q2: second quaternion (..., 4) in [w, x, y, z].
    :return: product quaternion (..., 4) in [w, x, y, z].
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate (inverse for unit quaternions).

    :param q: quaternion (..., 4) in [w, x, y, z].
    :return: conjugate quaternion (..., 4).
    """
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def quaternion_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) by quaternion(s) via q ⊗ v ⊗ q*.

    :param q: quaternion (..., 4) in [w, x, y, z].
    :param v: vector (..., 3).
    :return: rotated vector (..., 3).
    """
    v_quat = torch.zeros(v.shape[:-1] + (4,), dtype=v.dtype, device=v.device)
    v_quat[..., 1:] = v
    q_conj = quaternion_conjugate(q)
    rotated = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
    return rotated[..., 1:]


def axis_angle_to_quaternion(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert rotation vector (axis-angle) to quaternion.

    :param rotvec: rotation vector (..., 3).
    :return: quaternion (..., 4) in [w, x, y, z].
    """
    angle = torch.norm(rotvec, dim=-1, keepdim=True)
    half_angle = angle * 0.5
    # For near-zero angles, use small-angle approximation
    small = angle.squeeze(-1) < 1e-12
    safe_angle = torch.where(angle > 1e-12, angle, torch.ones_like(angle))
    axis = rotvec / safe_angle
    w = torch.cos(half_angle)
    xyz = axis * torch.sin(half_angle)
    result = torch.cat([w, xyz], dim=-1)
    # For near-zero rotation vectors, return identity
    identity = torch.zeros_like(result)
    identity[..., 0] = 1.0
    result[small] = identity[small]
    return result


def quaternion_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation vector (axis-angle).

    :param q: quaternion (..., 4) in [w, x, y, z].
    :return: rotation vector (..., 3).
    """
    q = quaternion_normalize(q)
    # Ensure w >= 0 for unique representation
    q = torch.where(q[..., 0:1] < 0, -q, q)
    xyz = q[..., 1:]
    sin_half = torch.norm(xyz, dim=-1, keepdim=True)
    cos_half = q[..., 0:1]
    # For near-identity: angle ≈ 2 * sin(half_angle) ≈ 2 * ||xyz||
    two_atan = 2.0 * torch.atan2(sin_half, cos_half)
    safe_sin = torch.where(sin_half > 1e-12, sin_half, torch.ones_like(sin_half))
    axis = xyz / safe_sin
    rotvec = axis * two_atan
    # Near-zero case
    small = sin_half.squeeze(-1) < 1e-12
    rotvec[small] = torch.zeros(3, dtype=q.dtype, device=q.device)
    return rotvec


def quaternion_enforce_continuity(q: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Enforce temporal sign continuity of a quaternion trajectory.

    Walks along the trajectory ``dim`` and flips the sign of each quaternion
    whenever it points into the opposite hemisphere of its predecessor
    (``<q_t, q_{t-1}> < 0``), removing double-cover discontinuities while
    preserving the represented rotation exactly.

    IMPORTANT: this operates on an **ordered** trajectory; it must NOT be
    applied to a shuffled minibatch (the continuity would be meaningless).

    Introduced by S3.1 of the Per-Environment Feature Handling `.junie` plan
    (`rlrp-736-per-environment-feature-handling-plan-20260711.md`, YouTrack
    RLRP-736).

    :param q: quaternion tensor of shape (..., 4) in [w, x, y, z] order.
    :param dim: the time/trajectory axis along which to enforce continuity.
    :return: sign-continuous quaternion tensor of the same shape.
    """
    q = q.movedim(dim, 0)
    if q.shape[0] <= 1:
        return q.movedim(0, dim)
    out = q.clone()
    for t in range(1, out.shape[0]):
        dot = (out[t] * out[t - 1]).sum(dim=-1, keepdim=True)
        out[t] = torch.where(dot < 0, -out[t], out[t])
    return out.movedim(0, dim)


def safe_rotation_from_quat(quat):
    """
    Safely create a Rotation object from a quaternion.
    Returns identity rotation if quaternion is invalid.
    """
    quat = np.array(quat, dtype=float)
    norm = np.linalg.norm(quat)

    # Check for zero or near-zero norm
    if norm < 1e-10:
        # Return identity rotation (no rotation)
        return Rotation.identity()

    # Normalize the quaternion to ensure unit length
    normalized_quat = quat / norm

    # Additional safety check: ensure no NaN or inf values
    if not np.all(np.isfinite(normalized_quat)):
        return Rotation.identity()

    return Rotation.from_quat(normalized_quat)


def safe_rotation_from_rotvec(rotvec, max_angle_rad=2 * np.pi):
    """
    Safely create a Rotation object from a rotation vector.
    Returns identity rotation if rotation vector is invalid.

    Args:
        rotvec: Rotation vector (axis-angle representation)
        max_angle_rad: Maximum allowed rotation angle in radians (default: 2π)
    """
    rotvec = np.array(rotvec, dtype=float)

    # Check for NaN or inf
    if not np.all(np.isfinite(rotvec)):
        return Rotation.identity()

    # Check for zero or near-zero magnitude
    magnitude = np.linalg.norm(rotvec)
    if magnitude < 1e-12:
        return Rotation.identity()

    # Clamp to prevent overflow - cap the rotation magnitude
    if magnitude > max_angle_rad:
        # Scale down to maximum allowed rotation
        rotvec = rotvec * (max_angle_rad / magnitude)
        magnitude = max_angle_rad

    return Rotation.from_rotvec(rotvec)


# --- Velocity-frame re-expression (RLRP-758) ---------------------------------
# Single source of truth for the world<->body coordinate change, reused by the
# numpy integrator branch (below) AND the ingestion-time converter in the
# ``*FlatContainerToNested`` builders. The torch integrator keeps its own
# hand-rolled quaternion algebra and is kept in lockstep by the numpy<->torch
# parity test (see the RLRP-758 plan §5.1 / §8).
#
# Introduced by task T1 of the RLRP-758 extend-velocity-frame-logic `.junie`
# plan (`feat_extend_velocity_frame_logic_plan_RLRP-758_20260726.md`).
GRAVITY_ALIGNED_FRAME = "gravity_aligned"
_VALID_FRAMES = (
    "world",
    "body",
    GRAVITY_ALIGNED_FRAME,
)  # gravity_aligned implemented by RLRP-792


def _assert_frame(name: str, frame: str) -> None:
    """Validate a velocity/training frame label.

    ``gravity_aligned`` — the yaw/heading frame, see
    :func:`yaw_quaternion_about_axis` — was a reserved-but-unimplemented slot
    until RLRP-792 (``G-9`` of the RLRP-753 gravity-vector plan) implemented it.
    """
    # RLRP-758 (merit review): this is a fail-loud SAFETY guard against a silent
    # frame mismatch, so it must be a real ``raise`` — NOT an ``assert`` (which
    # ``python -O`` / ``PYTHONOPTIMIZE`` strips, re-opening the exact hazard the
    # guard exists to prevent).
    if frame not in _VALID_FRAMES:
        raise ValueError(f"bad {name}={frame!r} (expected one of {_VALID_FRAMES})")


def _safe_rotations_from_quat_wxyz(quat_wxyz) -> Rotation:
    """Vectorized, degenerate-safe ``Rotation`` batch from ``(N, 4)`` ``[w,x,y,z]``.

    Mirrors :func:`safe_rotation_from_quat` row-wise (non-finite or (near-)zero
    norm rows collapse to the identity) but builds the whole batch in ONE scipy
    call — the same boundary :func:`quaternion_to_body_gravity` uses, so a frame
    rotation can never drift from the gravity projection.

    Introduced by RLRP-792 (`training_frame: gravity_aligned`).
    """
    quat = np.asarray(quat_wxyz, dtype=float)
    if quat.ndim != 2 or quat.shape[-1] != 4:
        raise ValueError(
            f"quaternions must be (N, 4) scalar-first [w, x, y, z] (got {quat.shape})"
        )
    quat_scipy = quat[:, [1, 2, 3, 0]]
    norms = np.linalg.norm(quat_scipy, axis=-1)
    degenerate = ~np.isfinite(norms) | (norms < 1e-10)
    degenerate |= ~np.all(np.isfinite(quat_scipy), axis=-1)
    if np.any(degenerate):
        quat_scipy = quat_scipy.copy()
        quat_scipy[degenerate] = (0.0, 0.0, 0.0, 1.0)  # scipy identity
    return Rotation.from_quat(quat_scipy)


def yaw_quaternion_about_axis(quat_wxyz, axis=None) -> np.ndarray:
    """Twist (yaw/heading) component of a body->world attitude about the vertical.

    Swing-twist decomposition ``q = q_twist ⊗ q_swing`` with the twist taken
    about the world gravity axis ``n``: the twist is the rotation *about* the
    vertical (the heading), the swing is the residual tilt. Concretely, with
    ``q = [w, v]``::

        q_twist = normalize([w, (v · n̂) n̂])

    This is the exact **dual** of the RLRP-753 gravity observation: because a
    rotation about ``n̂`` fixes ``g^W``, the body-frame gravity direction
    ``ĝ^B = R(q)^T ĝ^W = R(q_swing)^T ĝ^W`` depends only on the *swing*. So
    ``gravity.{x,y,z}`` carries the swing and the ``gravity_aligned`` frame
    quotients out the twist — the two are complementary halves of the attitude,
    never redundant (which is also why ``ĝ`` expressed in the gravity-aligned
    frame would be the constant ``ĝ^W`` and carry no information).

    Degenerate rows — a 180° rotation about an axis orthogonal to ``n̂``, i.e.
    ``w == 0`` and ``v · n̂ == 0``, a measure-zero set — have no defined heading
    and fall back to the identity (documented tie-break).

    :param quat_wxyz: ``(N, 4)`` (or ``(4,)``) scalar-first body->world attitude.
    :param axis: World vertical (the gravity axis); only the *line* matters, the
        sign is irrelevant. Defaults to :data:`DEFAULT_GRAVITY_WORLD_AXIS`.
    :return: ``(N, 4)`` (or ``(4,)``) unit yaw quaternion, scalar-first.

    Introduced by RLRP-792 (`training_frame: gravity_aligned`), i.e. task ``G-9``
    of `feat_gravity_vector_orientation_representation_plan_RLRP-753_20260724.md`.
    """
    axis_arr = _normalize_gravity_world_axis(
        DEFAULT_GRAVITY_WORLD_AXIS if axis is None else axis
    )
    quat = np.asarray(quat_wxyz, dtype=float)
    single = quat.ndim == 1
    if single:
        quat = quat[None, :]
    if quat.ndim != 2 or quat.shape[-1] != 4:
        raise ValueError(
            f"quat_wxyz must be (N, 4) scalar-first [w, x, y, z] (got {quat.shape})"
        )
    scalar = quat[:, 0:1]
    vector = quat[:, 1:4]
    twist = np.concatenate(
        [scalar, (vector @ axis_arr)[:, None] * axis_arr[None, :]], axis=-1
    )
    norms = np.linalg.norm(twist, axis=-1, keepdims=True)
    safe = np.isfinite(norms) & (norms >= 1e-10)
    identity = np.zeros_like(twist)
    identity[:, 0] = 1.0
    twist = np.where(safe, twist / np.where(safe, norms, 1.0), identity)
    return twist[0] if single else twist


def _rotate_gravity_aligned_to_world(vectors, quaternions, gravity_world_axis=None):
    """Rotate ``(N, 3)`` heading-frame vectors into the world frame.

    ``v^W = R(q_yaw) v^GA`` with ``q_yaw`` the twist of ``quaternions`` about the
    world gravity axis (:func:`yaw_quaternion_about_axis`). Backend-dispatching
    (numpy via scipy, torch via the in-module quaternion algebra) so the
    integrator can demote the heading frame on BOTH paths from a single seam.

    Introduced by RLRP-792 (`training_frame: gravity_aligned`).
    """
    if isinstance(vectors, torch.Tensor):
        quat = quaternions
        if not isinstance(quat, torch.Tensor):
            quat = torch.as_tensor(
                np.asarray(quat, dtype=float),
                dtype=vectors.dtype,
                device=vectors.device,
            )
        quat = quat.to(dtype=vectors.dtype, device=vectors.device)
        axis = torch.as_tensor(
            _normalize_gravity_world_axis(
                DEFAULT_GRAVITY_WORLD_AXIS
                if gravity_world_axis is None
                else gravity_world_axis
            ),
            dtype=vectors.dtype,
            device=vectors.device,
        )
        projection = (quat[..., 1:4] * axis).sum(dim=-1, keepdim=True) * axis
        # ``quaternion_normalize`` already returns the identity for a (near-)zero
        # norm, which is exactly the documented degenerate tie-break.
        twist = quaternion_normalize(torch.cat([quat[..., 0:1], projection], dim=-1))
        return quaternion_apply(twist, vectors)
    twist = yaw_quaternion_about_axis(quaternions, gravity_world_axis)
    return _safe_rotations_from_quat_wxyz(twist).apply(
        np.asarray(vectors, dtype=float)
    )


def _frame_alignment_quaternion(frame: str, quaternions, gravity_world_axis):
    """Body->world quaternion of ``frame`` (``None`` for the world frame itself)."""
    if frame == "world":
        return None
    if frame == "body":
        return quaternions
    return yaw_quaternion_about_axis(quaternions, gravity_world_axis)


def rotate_velocity_between_frames(
    velocity, quaternions, src_frame, dst_frame, gravity_world_axis=None
):
    """Re-express a (N, 3) LINEAR velocity from ``src_frame`` to ``dst_frame``.

    Uses the body->world attitude ``R(q)`` (``quaternions`` in ``[w, x, y, z]``):
    ``v_world = R(q) @ v_body`` and ``v_body = R(q)^T @ v_world``. This is the
    exact rotation used by the integrator's ``linear_velocity_frame`` branch, so
    ingestion-time conversion can never drift from the integrator.

    RLRP-792 adds the third frame: ``gravity_aligned`` is the *heading* frame,
    whose body->world rotation is the yaw/twist quaternion
    :func:`yaw_quaternion_about_axis`. Every pair is then handled uniformly as
    ``v_dst = R(q_dst)^T R(q_src) v_src``. The pre-existing ``world``/``body``
    pairs keep their original code path **bit-exactly**.

    Introduced by task T1 of the RLRP-758 extend-velocity-frame-logic `.junie`
    plan (`feat_extend_velocity_frame_logic_plan_RLRP-758_20260726.md`);
    extended with ``gravity_aligned`` by RLRP-792.
    """
    _assert_frame("src_frame", src_frame)
    _assert_frame("dst_frame", dst_frame)
    velocity = np.asarray(velocity, dtype=float)
    if src_frame == dst_frame:
        return velocity
    # quaternions are [w, x, y, z]; scipy expects [x, y, z, w].
    quaternions = np.asarray(quaternions, dtype=float)
    if GRAVITY_ALIGNED_FRAME in (src_frame, dst_frame):
        rotated = velocity
        q_src = _frame_alignment_quaternion(
            src_frame, quaternions, gravity_world_axis
        )
        if q_src is not None:  # src -> world
            rotated = _safe_rotations_from_quat_wxyz(q_src).apply(rotated)
        q_dst = _frame_alignment_quaternion(
            dst_frame, quaternions, gravity_world_axis
        )
        if q_dst is not None:  # world -> dst
            rotated = _safe_rotations_from_quat_wxyz(q_dst).inv().apply(rotated)
        return np.atleast_2d(rotated)
    quat_scipy = quaternions[:, [1, 2, 3, 0]]
    rotations = [safe_rotation_from_quat(q) for q in quat_scipy]
    if src_frame == "world" and dst_frame == "body":
        return np.stack(
            [rotations[i].inv().apply(velocity[i]) for i in range(len(velocity))]
        )
    # body -> world
    return np.stack([rotations[i].apply(velocity[i]) for i in range(len(velocity))])


def reexpress_angular_velocity_between_frames(
    omega, quaternions, src_frame, dst_frame, gravity_world_axis=None
):
    """Re-express a (N, 3) ANGULAR velocity from ``src_frame`` to ``dst_frame``.

    For a rigid body's own rate the coordinate change is the SAME ``R(q)``/
    ``R(q)^T`` as the linear channel (``omega_world = R(q) @ omega_body``); this
    is a real rotation, NOT a relabel (RLRP-755 report §6.2 double-negative
    hazard). Kept as a separately-named entry point so RLRP-792's
    ``gravity_aligned`` angular handling has a dedicated seam — the heading frame
    is *not* body-fixed, so ``omega^GA = R(q_swing) omega^B`` still differs from
    the body rate whenever the platform is tilted.

    Introduced by task T1 of the RLRP-758 extend-velocity-frame-logic `.junie`
    plan (`feat_extend_velocity_frame_logic_plan_RLRP-758_20260726.md`).
    """
    return rotate_velocity_between_frames(
        omega, quaternions, src_frame, dst_frame, gravity_world_axis
    )


def resolve_training_frame_from_cfg(cfg) -> str:
    """Resolve the model-internal target training frame from the Hydra config.

    Reads ``cfg.ms_model.training_frame`` (``world`` legacy | ``body`` new
    default | ``gravity_aligned``, the heading frame implemented by RLRP-792).
    Defaults to ``body`` when the key is absent so datasets are converted into
    the new default frame.

    Introduced by task T7 of the RLRP-758 extend-velocity-frame-logic `.junie`
    plan (`feat_extend_velocity_frame_logic_plan_RLRP-758_20260726.md`).
    """
    training_frame = omegaconf.OmegaConf.select(
        cfg, "ms_model.training_frame", default="body"
    )
    if training_frame is None:
        training_frame = "body"
    _assert_frame("ms_model.training_frame", training_frame)
    # RLRP-792 review finding `A` added a guard here rejecting
    # `gravity_aligned` on a gravity-configured observation space, because the
    # deploy integrator could not demote the heading-frame velocity channels
    # without a measured attitude. RLRP-794 (observation 1) implements that
    # demotion as a sequential heading/rate co-integration
    # (`_gravity_aligned_reckon_attitude_sequence`), so the pairing is now
    # SUPPORTED and the guard is gone. There is no remaining unsupported
    # `(training_frame, obs space)` combination to reject.
    return training_frame


def convert_velocity_channels_to_training_frame(
    l_vel,
    a_vel,
    quat_wxyz,
    native_linear_frame,
    native_angular_frame,
    training_frame,
    gravity_world_axis=None,
):
    """Re-express the linear/angular velocity channels into ``training_frame``.

    ``l_vel``/``a_vel`` are ``(x, y, z)`` component tuples; ``quat_wxyz`` is the
    ``(w, x, y, z)`` attitude component tuple. Returns ``(linear (N, 3),
    angular (N, 3))`` numpy arrays already expressed in ``training_frame``.

    Introduced by task T7 of the RLRP-758 extend-velocity-frame-logic `.junie`
    plan (`feat_extend_velocity_frame_logic_plan_RLRP-758_20260726.md`).
    """
    lin = np.stack([np.asarray(c, dtype=float) for c in l_vel], axis=-1)
    ang = np.stack([np.asarray(c, dtype=float) for c in a_vel], axis=-1)
    quat = np.stack([np.asarray(c, dtype=float) for c in quat_wxyz], axis=-1)
    lin = rotate_velocity_between_frames(
        lin, quat, native_linear_frame, training_frame, gravity_world_axis
    )
    ang = reexpress_angular_velocity_between_frames(
        ang, quat, native_angular_frame, training_frame, gravity_world_axis
    )
    return lin, ang


# --- Body-frame gravity direction (RLRP-753) ---------------------------------
# The yaw-invariant orientation representation ``g_hat^B = R(q)^T g^W`` on ``S^2``.
# Producer half of the RLRP-761 ``gravity.{x,y,z}`` contract; see rev. 4 of
# `feat_gravity_vector_orientation_representation_plan_RLRP-753_20260724.md` §6.1.
#
#: Default world gravity axis (z-up inertial frame, gravity pointing DOWN).
#: All three reference datasets (``pi_tcn``, ``neurobem``, ``husky_adverse``) are
#: z-up, so the default is a no-op; the config key exists (ruling ``D-axis``) so a
#: future non-z-up dataset cannot be silently mis-projected (risk ``R-N6``).
DEFAULT_GRAVITY_WORLD_AXIS = (0.0, 0.0, -1.0)


def resolve_gravity_world_axis_from_cfg(cfg) -> np.ndarray:
    """Resolve the world-frame gravity axis from the Hydra config.

    Reads ``cfg.environment.data.gravity_world_axis`` and falls back to
    :data:`DEFAULT_GRAVITY_WORLD_AXIS` when the key is absent (every pre-RLRP-753
    config), so this resolver is a no-op for existing runs.

    The returned axis is **unit-normalized**: only the *direction* is meaningful
    for the ``S^2`` representation, so declaring ``[0, 0, -9.81]`` is equivalent
    to ``[0, 0, -1]``.

    :param cfg: The (partial) Hydra/OmegaConf config.
    :return: A unit-norm ``(3,)`` float array.
    :raises ValueError: when the declared axis is not a finite 3-vector, or is
        (near-)zero and therefore carries no direction. This is a fail-loud
        SAFETY guard against a silently degenerate projection, so it is a real
        ``raise`` — NOT an ``assert`` (``python -O`` strips those).

    Introduced by task ``G-1`` of the RLRP-753 gravity-vector `.junie` plan
    (`feat_gravity_vector_orientation_representation_plan_RLRP-753_20260724.md`).
    """
    axis = omegaconf.OmegaConf.select(
        cfg, "environment.data.gravity_world_axis", default=None
    )
    if axis is None:
        axis = DEFAULT_GRAVITY_WORLD_AXIS
    elif isinstance(axis, (omegaconf.ListConfig, omegaconf.DictConfig)):
        axis = omegaconf.OmegaConf.to_object(axis)
    return _normalize_gravity_world_axis(axis)


def _normalize_gravity_world_axis(axis) -> np.ndarray:
    """Validate + unit-normalize a world gravity axis (see resolver above)."""
    axis_arr = np.asarray(axis, dtype=float).reshape(-1)
    if axis_arr.shape != (3,) or not np.all(np.isfinite(axis_arr)):
        raise ValueError(
            "environment.data.gravity_world_axis must be a finite 3-vector "
            f"(got {axis!r})"
        )
    norm = float(np.linalg.norm(axis_arr))
    if norm < 1e-10:
        raise ValueError(
            "environment.data.gravity_world_axis is (near-)zero and carries no "
            f"direction (got {axis!r}); expected e.g. {list(DEFAULT_GRAVITY_WORLD_AXIS)}"
        )
    return axis_arr / norm


def quaternion_to_body_gravity(
    quat_wxyz,
    g_world=None,
    normalize: bool = True,
) -> np.ndarray:
    """Project the world gravity axis into the body frame: ``g^B = R(q)^T g^W``.

    This is the RLRP-753 orientation representation ``g_hat^B in S^2`` — the
    *yaw-quotient* of the attitude, i.e. a 2-DoF roll/pitch ("which-way-is-down")
    signal that is invariant to a world-frame yaw rotation and identical for
    ``q`` and ``-q`` (the quaternion double cover collapses).

    Fully **vectorized**: a single ``Rotation.from_quat(...).inv().apply(...)``
    call, no per-row Python loop (plan §5.2 non-functional requirement). Uses the
    exact same ``[w, x, y, z] -> [x, y, z, w]`` scipy boundary as
    :func:`rotate_velocity_between_frames`, so a gravity projection can never
    drift from the velocity frame conversion done at the same ingestion seam.

    Degenerate rows (non-finite or (near-)zero-norm quaternion) fall back to the
    identity rotation, mirroring :func:`safe_rotation_from_quat` /
    :func:`quaternion_normalize`; the resulting row is then exactly ``g_world``.

    :param quat_wxyz: ``(N, 4)`` scalar-first ``[w, x, y, z]`` body->world
        attitude (canonicalized upstream by ``canonicalize_attitude_components``).
        A single ``(4,)`` quaternion is accepted and returns a ``(3,)`` vector.
    :param g_world: World-frame gravity axis; defaults to
        :data:`DEFAULT_GRAVITY_WORLD_AXIS`. Only its direction matters (it is
        unit-normalized).
    :param normalize: When ``True`` (default) the output rows are re-projected
        onto the unit sphere, guaranteeing ``||g^B|| = 1`` to floating-point
        precision even after the rotation round-trip.
    :return: ``(N, 3)`` float array on ``S^2`` (or ``(3,)`` for a single input).

    Introduced by task ``G-1`` of the RLRP-753 gravity-vector `.junie` plan
    (`feat_gravity_vector_orientation_representation_plan_RLRP-753_20260724.md`).
    """
    axis = _normalize_gravity_world_axis(
        DEFAULT_GRAVITY_WORLD_AXIS if g_world is None else g_world
    )
    quat = np.asarray(quat_wxyz, dtype=float)
    single = quat.ndim == 1
    if single:
        quat = quat[None, :]
    if quat.ndim != 2 or quat.shape[-1] != 4:
        raise ValueError(
            f"quat_wxyz must be (N, 4) scalar-first [w, x, y, z] (got {quat.shape})"
        )
    # quaternions are [w, x, y, z]; scipy expects [x, y, z, w].
    quat_scipy = quat[:, [1, 2, 3, 0]]
    # Vectorized safety rail, mirroring ``safe_rotation_from_quat`` row-wise:
    # non-finite or (near-)zero-norm rows are replaced by the identity so scipy
    # never raises on a malformed trajectory.
    norms = np.linalg.norm(quat_scipy, axis=-1)
    degenerate = ~np.isfinite(norms) | (norms < 1e-10)
    degenerate |= ~np.all(np.isfinite(quat_scipy), axis=-1)
    if np.any(degenerate):
        quat_scipy = quat_scipy.copy()
        quat_scipy[degenerate] = (0.0, 0.0, 0.0, 1.0)  # scipy identity
    g_body = Rotation.from_quat(quat_scipy).inv().apply(axis)
    g_body = np.atleast_2d(g_body)
    if normalize:
        out_norms = np.linalg.norm(g_body, axis=-1, keepdims=True)
        safe = out_norms >= 1e-10
        g_body = np.where(safe, g_body / np.where(safe, out_norms, 1.0), axis)
    return g_body[0] if single else g_body


def compute_position_from_world_velocity(
    linear_velocity_world: Union[np.ndarray, torch.Tensor],
    timestamps: Union[np.ndarray, torch.Tensor],
    initial_position: Optional[Union[np.ndarray, torch.Tensor]] = None,
    integration_scheme="forward_euler",
    debug: bool = False,
) -> tuple:
    """Direct integration of world-frame velocities to positions.

    Supports both numpy and torch tensor inputs. When inputs are tensors the
    computation stays on-device using vectorised ``torch.cumsum``.
    """
    # --- Torch path --------------------------------------------------------
    if isinstance(linear_velocity_world, torch.Tensor):
        if initial_position is None:
            initial_position = torch.zeros(
                3,
                dtype=linear_velocity_world.dtype,
                device=linear_velocity_world.device,
            )
        elif isinstance(initial_position, np.ndarray):
            initial_position = torch.from_numpy(initial_position).to(
                dtype=linear_velocity_world.dtype, device=linear_velocity_world.device
            )
        if isinstance(timestamps, np.ndarray):
            timestamps = torch.from_numpy(timestamps).to(
                dtype=linear_velocity_world.dtype, device=linear_velocity_world.device
            )

        N = len(timestamps)
        dt_values = torch.zeros(
            N, dtype=linear_velocity_world.dtype, device=linear_velocity_world.device
        )
        if N > 1:
            dt_values[1:] = timestamps[1:] - timestamps[:-1]
            dt_values[0] = dt_values[1]

        if integration_scheme == "trapezoidal":
            # Vectorised trapezoidal integration (RLRP-707): cumsum of the
            # endpoint-averaged velocity * dt. Kept consistent with
            # ``compute_position_from_velocity_and_attitude`` (see the rationale
            # there) so this reference reconstruction does not itself drift.
            avg_velocities = 0.5 * (linear_velocity_world[1:] + linear_velocity_world[:-1])
            increments = torch.zeros_like(linear_velocity_world)
            increments[1:] = avg_velocities * dt_values[1:].unsqueeze(-1)
            positions = torch.cumsum(increments, dim=0) + initial_position

        elif integration_scheme == "forward_euler":
            # Right-endpoint rectangle (forward-Euler) rule ``pos[i] = pos[i-1] + vel[i] * dt[i]``

            # Alt 1
            # increments = torch.zeros_like(velocities_world)
            # increments[1:] = velocities_world[1:] * dt_values[1:].unsqueeze(-1)
            # positions = torch.cumsum(increments, dim=0) + initial_position

            # Alt 2
            increments = linear_velocity_world * dt_values.unsqueeze(-1)
            increments[0] = torch.zeros_like(linear_velocity_world[0])
            positions = torch.cumsum(increments, dim=0) + initial_position

        else:
            raise NotImplementedError(
                f"Unsupported integration scheme: {integration_scheme}"
            )

        return positions, linear_velocity_world

    # --- Numpy path (original) ---------------------------------------------
    if initial_position is None:
        initial_position = np.array([0.0, 0.0, 0.0])

    N = len(timestamps)
    positions = np.zeros((N, 3))
    positions[0] = initial_position

    if debug:
        print(
            "Using direct world-frame velocity integration (no attitude transformation)"
        )
        for i, axis in enumerate(["X", "Y", "Z"]):
            vel_component = linear_velocity_world[:, i]
            print(
                f"World vel {axis}: mean={np.mean(vel_component):.4f}, std={np.std(vel_component):.4f}"
            )

    # Compute dt from consecutive timestamps
    dt_values = np.zeros(N, dtype=float)
    dt_values[1:] = np.diff(timestamps)
    dt_values[0] = dt_values[1] if N > 1 else 0.0

    if debug and N > 1:
        positive_dt = dt_values[dt_values > 0]
        if positive_dt.size:
            print(
                f"Computed dt stats: mean={np.mean(positive_dt):.9f}s, std={np.std(positive_dt):.9f}s"
            )

    if integration_scheme == "trapezoidal":
        # Integration (trapezoidal rule, RLRP-707): kept consistent with
        # ``compute_position_from_velocity_and_attitude`` so this reference
        # reconstruction shares the same O(dt**2) accuracy and does not drift.
        for i in range(1, N):
            dt = dt_values[i]
            avg_velocity = 0.5 * (
                linear_velocity_world[i - 1] + linear_velocity_world[i]
            )
            positions[i] = positions[i - 1] + avg_velocity * dt

    elif integration_scheme == "forward_euler":
        # Right-endpoint rectangle (forward-Euler) rule ``pos[i] = pos[i-1] + vel[i] * dt[i]``
        for i in range(1, N):
            dt = dt_values[i]
            positions[i] = positions[i - 1] + linear_velocity_world[i] * dt
    else:
        raise NotImplementedError(
            f"Unsupported integration scheme: {integration_scheme}"
        )

    if debug:
        final_drift = positions[-1] - initial_position
        print(
            f"Final drift: {final_drift}, magnitude: {np.linalg.norm(final_drift):.4f}"
        )

    return positions, linear_velocity_world  # Return velocities as-is


def _resolve_timestep_deltas(
    timestamps: Union[np.ndarray, torch.Tensor, tct.temporal.Timestamps],
) -> np.ndarray:
    """Return the per-index integration steps ``dt[i] = t[i] - t[i-1]`` (seconds).

    Verbatim mirror of the ``dt`` convention used by both integration paths:
    ``dt[0]`` is seeded with ``dt[1]`` (the index-0 increment is zero anyway, so
    the value only matters for the attitude propagation of index 1).

    Introduced by action `A2` of the RLRC Explicit attitude source in deployer
    trajectory integration `.junie` plan
    (`feat_explicit_attitude_source_deploy_integration_plan_RLRP-791_20260829.md`).

    :param timestamps: ``(N,)`` time axis (TCT ``Timestamps``, numpy or torch).
    :return: ``(N,)`` float64 numpy array of time steps.
    """
    if isinstance(timestamps, tct.temporal.Timestamps):
        return np.array(timestamps.delta_stamps, dtype=float)

    if isinstance(timestamps, torch.Tensor):
        stamps = timestamps.detach().cpu().numpy().astype(float)
    else:
        stamps = np.asarray(timestamps, dtype=float)

    n = len(stamps)
    dt_values = np.zeros(n, dtype=float)
    if n > 1:
        dt_values[1:] = np.diff(stamps)
        dt_values[0] = dt_values[1]
    return dt_values


def _validate_initial_orientation(
    initial_orientation: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """Return the start attitude flattened to ``(4,)``, or fail loud.

    The start attitude anchors the whole reconstruction, so a malformed value
    must never be silently ignored (nor silently replaced by identity): it is
    reported with a ``raise`` — deliberately **not** an ``assert``, which
    ``python -O`` strips.

    Introduced by action `A2` of the RLRC Explicit attitude source in deployer
    trajectory integration `.junie` plan
    (`feat_explicit_attitude_source_deploy_integration_plan_RLRP-791_20260829.md`).

    :param initial_orientation: the start attitude quaternion ``[qw, qx, qy, qz]``
        (shape ``(4,)`` or any shape holding exactly 4 elements).
    :return: the same quaternion flattened to ``(4,)``, dtype/backend preserved.
    :raises ValueError: when it does not hold exactly 4 elements, is non-finite,
        or has a (near-)zero norm (i.e. defines no rotation).
    """
    if isinstance(initial_orientation, torch.Tensor):
        flat = initial_orientation.reshape(-1)
        size = flat.numel()
        is_finite = bool(torch.all(torch.isfinite(flat)))
        norm = float(torch.linalg.vector_norm(flat.double())) if is_finite else 0.0
    else:
        flat = np.asarray(initial_orientation).reshape(-1)
        size = flat.size
        is_finite = bool(np.all(np.isfinite(flat.astype(np.float64))))
        norm = float(np.linalg.norm(flat.astype(np.float64))) if is_finite else 0.0

    if size != 4:
        raise ValueError(
            f"`initial_orientation` must hold exactly 4 elements "
            f"([qw, qx, qy, qz]), got {size}."
        )
    if not is_finite:
        raise ValueError(
            f"`initial_orientation` must be finite, got {initial_orientation!r}."
        )
    if norm <= 1e-12:
        raise ValueError(
            f"`initial_orientation` must have a non-zero norm (it defines the "
            f"reconstruction start attitude), got norm={norm}."
        )
    return flat


def _seed_attitude_sequence(
    quaternions: Union[np.ndarray, torch.Tensor],
    initial_orientation: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """Return ``quaternions`` with row 0 replaced by the explicit start attitude.

    Mirrors ``initial_position``: the integration *anchor* attitude becomes an
    explicit caller decision instead of an implicit ``quaternions[0]`` read. The
    caller's array/tensor is never mutated (copy-on-seed) — the deploy path
    reuses the very same object for other purposes.

    Introduced by action `A2` of the RLRC Explicit attitude source in deployer
    trajectory integration `.junie` plan
    (`feat_explicit_attitude_source_deploy_integration_plan_RLRP-791_20260829.md`).

    :param quaternions: ``(N, 4)`` attitude sequence ``[qw, qx, qy, qz]``.
    :param initial_orientation: validated ``(4,)`` start attitude.
    :return: ``(N, 4)`` sequence whose row 0 is ``initial_orientation``.
    """
    if isinstance(quaternions, torch.Tensor):
        if isinstance(initial_orientation, torch.Tensor):
            row = initial_orientation.detach().to(
                dtype=quaternions.dtype, device=quaternions.device
            )
        else:
            row = torch.as_tensor(
                np.asarray(initial_orientation),
                dtype=quaternions.dtype,
                device=quaternions.device,
            )
        return torch.cat([row.reshape(1, -1), quaternions[1:]], dim=0)

    seeded = np.array(quaternions, copy=True)
    if isinstance(initial_orientation, torch.Tensor):
        seeded[0] = initial_orientation.detach().cpu().numpy()
    else:
        seeded[0] = np.asarray(initial_orientation)
    return seeded


def _dead_reckon_attitude_sequence(
    initial_orientation: Union[np.ndarray, torch.Tensor],
    angular_velocity: Union[np.ndarray, torch.Tensor],
    timestamps: Union[np.ndarray, torch.Tensor, tct.temporal.Timestamps],
    angular_velocity_frame: str,
    max_angular_velocity: float,
) -> Union[np.ndarray, torch.Tensor]:
    """Propagate a full attitude sequence from the start attitude and body rates.

    Dead-reckoning regime (RLRP-791 ruling `D4`): once the start attitude is an
    explicit argument, an absent *measured* attitude no longer prevents the
    reconstruction — the sequence is fully determined by
    ``q[i] = q[i-1] (x) delta(omega[i-1] * dt[i])``.

    The composition convention is the RLRP-758-corrected one shared with the
    measured-attitude propagation branches: a **body**-frame own-rate is a
    right-multiplied (intrinsic) increment ``prev * delta``, a **world**/spatial
    rate is left-multiplied ``delta * prev``. The ``max_angular_velocity``
    saturation rail (and its always-on report) is shared as well, so a
    physically-implausible rate cannot silently blow the attitude up.

    Introduced by action `A2` (folding former action `A6`) of the RLRC Explicit
    attitude source in deployer trajectory integration `.junie` plan
    (`feat_explicit_attitude_source_deploy_integration_plan_RLRP-791_20260829.md`).

    :param initial_orientation: validated ``(4,)`` start attitude ``[qw, qx, qy, qz]``.
    :param angular_velocity: ``(N, 3)`` angular velocity ``[wx, wy, wz]``.
    :param timestamps: ``(N,)`` time axis.
    :param angular_velocity_frame: "body" or "world" — frame of the rates.
    :param max_angular_velocity: saturation cap (rad/s) applied before propagation.
    :return: ``(N, 4)`` propagated attitude sequence in the backend of
        ``angular_velocity``.
    """
    dt_values = _resolve_timestep_deltas(timestamps)
    n = len(dt_values)
    is_body_frame = angular_velocity_frame == "body"

    if isinstance(angular_velocity, torch.Tensor):
        device = angular_velocity.device
        dtype = torch.float64
        omega = angular_velocity.to(dtype=dtype)
        omega_magnitudes = torch.norm(omega, dim=-1, keepdim=True)
        report_angular_velocity_clamp(
            omega_magnitudes.squeeze(-1),
            max_angular_velocity,
            context="compute_position_from_velocity_and_attitude (dead reckoning, torch)",
        )
        scale = torch.where(
            omega_magnitudes > max_angular_velocity,
            max_angular_velocity / omega_magnitudes,
            torch.ones_like(omega_magnitudes),
        )
        omega = omega * scale
        dt_tensor = torch.as_tensor(dt_values, dtype=dtype, device=device)

        if isinstance(initial_orientation, torch.Tensor):
            current_q = initial_orientation.detach().to(dtype=dtype, device=device)
        else:
            current_q = torch.as_tensor(
                np.asarray(initial_orientation, dtype=float),
                dtype=dtype,
                device=device,
            )
        current_q = quaternion_normalize(current_q)

        quat_list = [current_q]
        for i in range(1, n):
            delta_q = axis_angle_to_quaternion(
                (omega[i - 1] * dt_tensor[i]).unsqueeze(0)
            ).squeeze(0)
            if is_body_frame:
                current_q = quaternion_multiply(current_q, delta_q)
            else:
                current_q = quaternion_multiply(delta_q, current_q)
            current_q = quaternion_normalize(current_q)
            quat_list.append(current_q)

        return torch.stack(quat_list, dim=0)

    omega_magnitudes = np.linalg.norm(np.asarray(angular_velocity), axis=-1)
    report_angular_velocity_clamp(
        omega_magnitudes,
        max_angular_velocity,
        context="compute_position_from_velocity_and_attitude (dead reckoning, numpy)",
    )
    omega = np.array(angular_velocity, dtype=float, copy=True)
    over = omega_magnitudes > max_angular_velocity
    if np.any(over):
        omega[over] = (
            omega[over] * (max_angular_velocity / omega_magnitudes[over])[:, None]
        )

    # scipy convention is [qx, qy, qz, qw]
    q0_scipy = np.asarray(initial_orientation, dtype=float)[[1, 2, 3, 0]]
    rotation = safe_rotation_from_quat(q0_scipy)

    quaternions = np.zeros((n, 4), dtype=float)
    for i in range(n):
        if i > 0:
            delta_rotation = safe_rotation_from_rotvec(omega[i - 1] * dt_values[i])
            if is_body_frame:
                rotation = rotation * delta_rotation
            else:
                rotation = delta_rotation * rotation
        quat_scipy = rotation.as_quat()
        quaternions[i] = quat_scipy[[3, 0, 1, 2]]

    return quaternions


def _minimal_tilt_correction(u_world, axis_world) -> Rotation:
    """Minimal world-frame rotation carrying each ``u_world`` row onto ``axis_world``.

    The chart-free tilt correction shared by the gravity-reckoning regimes: the
    rotation whose axis is ``u x g^W`` (horizontal, hence injecting **no** rotation
    about ``g^W`` — it cannot forge yaw) and whose angle is the geodesic angle
    ``atan2(||u x g^W||, u . g^W)``. It is the identity exactly when ``u`` already
    agrees with ``g^W``.

    An explicit ``R_z(psi) . R_tilt`` factorization would need an Euler chart,
    singular at pitch ``+-90 deg`` — precisely the NeuroBEM / PI-TCN aerobatic
    regime that made RLRP-753 ruling ``D-euler`` reject Euler channels.

    Antipodal rows (``u ~= -g^W``, a ~180 deg tilt error) have no defined
    correction axis; an arbitrary axis orthogonal to ``g^W`` is used rather than
    silently leaving the tilt unconstrained (documented tie-break).

    Extracted verbatim from :func:`_gravity_reckon_attitude_sequence` by RLRP-794
    so the sequential heading/rate co-integration regime
    (:func:`_gravity_aligned_reckon_attitude_sequence`) reuses the *same*
    correction and the two can never drift.

    :param u_world: ``(N, 3)`` world-frame directions to be carried onto the axis.
    :param axis_world: ``(3,)`` unit world gravity axis.
    :return: the ``(N,)`` batched scipy :class:`Rotation` correction.
    """
    u = np.atleast_2d(np.asarray(u_world, dtype=float))
    cross = np.cross(u, axis_world)
    sin_angle = np.linalg.norm(cross, axis=-1)
    cos_angle = u @ axis_world
    angle = np.arctan2(sin_angle, cos_angle)

    rotvec = np.zeros_like(cross)
    aligned = sin_angle >= 1e-12
    rotvec[aligned] = (
        cross[aligned] / sin_angle[aligned][:, None] * angle[aligned][:, None]
    )
    antipodal = (~aligned) & (cos_angle < 0.0)
    if np.any(antipodal):
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(helper @ axis_world)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        perpendicular = np.cross(axis_world, helper)
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        rotvec[antipodal] = perpendicular * np.pi
    return Rotation.from_rotvec(rotvec)


def _unit_body_gravity(gravity, axis_world) -> np.ndarray:
    """Defensively unit-normalize a predicted ``(N, 3)`` body gravity direction.

    A model prediction is not guaranteed to be exactly on ``S^2`` even with the
    RLRP-753 autoregressive re-projection in place; (near-)zero rows fall back to
    ``axis_world``.
    """
    g_body = np.asarray(gravity, dtype=float)
    norms = np.linalg.norm(g_body, axis=-1, keepdims=True)
    safe = norms >= 1e-10
    return np.where(safe, g_body / np.where(safe, norms, 1.0), axis_world)


def _gravity_reckon_attitude_sequence(
    initial_orientation: Union[np.ndarray, torch.Tensor],
    gravity: Union[np.ndarray, torch.Tensor],
    angular_velocity: Union[np.ndarray, torch.Tensor],
    timestamps: Union[np.ndarray, torch.Tensor, tct.temporal.Timestamps],
    angular_velocity_frame: str,
    max_angular_velocity: float,
    gravity_world_axis=None,
) -> Union[np.ndarray, torch.Tensor]:
    """Reconstruct an attitude sequence from a predicted body gravity + body rates.

    RLRP-753 ruling ``D-deploy``, built as a **fourth regime** on the RLRP-791
    ``_resolve_attitude_sequence`` seam. It exists so that a gravity-configured
    run (no ``attitude.*`` block) integrates its world pose from the model's own
    *predicted* orientation instead of falling back to ground truth (risk
    ``R-N1``).

    ``g_hat^B`` is the yaw-quotient of the attitude: it pins **roll/pitch** and is
    structurally blind to **yaw**. The reconstruction therefore combines the two
    complementary sources the observation space provides:

    1. **yaw (and a baseline attitude)** — dead-reckoned from ``angular_velocity``
       via :func:`_dead_reckon_attitude_sequence`, i.e. the landed body/world
       composition convention and the ``max_angular_velocity`` saturation rail are
       reused verbatim, anchored on ``initial_orientation``;
    2. **roll/pitch** — imposed at every step from the predicted ``g_hat^B`` by the
       *minimal* world-frame correction ``C_i`` that maps where the dead-reckoned
       attitude thinks the measured gravity points, ``u_i = R_dr_i g_hat^B_i``, back
       onto the world axis ``g^W``:

       .. code-block:: text

           R_i = C_i . R_dr_i      with   C_i = minimal_rotation(u_i -> g^W)

       so ``R_i^T g^W = g_hat^B_i`` holds by construction.

    Why the minimal correction rather than an explicit ``R_z(psi) . R_tilt``
    factorization: extracting ``psi`` from a quaternion needs an Euler chart,
    which is singular at pitch ``+-90 deg`` — precisely the NeuroBEM / PI-TCN
    aerobatic regime that made ruling ``D-euler`` reject Euler channels. ``C_i``
    has a horizontal axis (``u_i x g^W``), injects no rotation about ``g^W``, is
    smooth everywhere except the antipodal case (a ~180 deg prediction error), and
    collapses to the identity exactly when the predicted gravity already agrees
    with the dead-reckoned attitude. Consequence: with a *perfect* prediction this
    regime reproduces the dead-reckoned (and hence the ground-truth) attitude, and
    every deviation of the predicted ``g_hat^B`` really does move the rollout
    metrics.

    Row 0 is the ``initial_orientation`` **anchor**, verbatim — same convention as
    ``initial_position`` (RLRP-791 ruling `D2`).

    Computed in numpy and converted back to the caller's backend, which makes the
    numpy and torch deploy paths bit-identical by construction (no second
    hand-rolled quaternion algebra to keep in lockstep).

    Introduced by task ``G-5`` of the RLRP-753 gravity-vector `.junie` plan
    (`feat_gravity_vector_orientation_representation_plan_RLRP-753_20260724.md`).

    :param initial_orientation: validated ``(4,)`` start attitude ``[qw, qx, qy, qz]``.
    :param gravity: ``(N, 3)`` predicted body-frame gravity direction on ``S^2``.
    :param angular_velocity: ``(N, 3)`` angular velocity ``[wx, wy, wz]``.
    :param timestamps: ``(N,)`` time axis.
    :param angular_velocity_frame: "body" or "world" — frame of the rates.
    :param max_angular_velocity: saturation cap (rad/s).
    :param gravity_world_axis: world gravity axis; defaults to
        :data:`DEFAULT_GRAVITY_WORLD_AXIS`.
    :return: ``(N, 4)`` attitude sequence, in the backend of ``angular_velocity``.
    :raises ValueError: when ``gravity`` is not ``(N, 3)`` or its length does not
        match ``angular_velocity``.
    """
    axis_world = _normalize_gravity_world_axis(
        DEFAULT_GRAVITY_WORLD_AXIS if gravity_world_axis is None else gravity_world_axis
    )

    return_torch = isinstance(angular_velocity, torch.Tensor) or isinstance(
        gravity, torch.Tensor
    )
    torch_reference = angular_velocity if isinstance(angular_velocity, torch.Tensor) else gravity

    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(float)
        return np.asarray(value, dtype=float)

    g_body = _to_numpy(gravity)
    omega_np = _to_numpy(angular_velocity)
    if g_body.ndim != 2 or g_body.shape[-1] != 3:
        raise ValueError(
            f"`gravity` must be (N, 3) body-frame directions, got {g_body.shape}."
        )
    if len(g_body) != len(omega_np):
        raise ValueError(
            f"`gravity` and `angular_velocity` must have the same length, got "
            f"{len(g_body)} vs {len(omega_np)}."
        )

    start_np = _to_numpy(initial_orientation).reshape(-1)

    # (1) yaw + baseline attitude from the body rates (landed RLRP-791 rails).
    q_dead_reckoned = _dead_reckon_attitude_sequence(
        start_np,
        omega_np,
        timestamps,
        angular_velocity_frame,
        max_angular_velocity,
    )

    # (2) impose roll/pitch from the predicted gravity direction.
    g_unit = _unit_body_gravity(g_body, axis_world)

    rotations = Rotation.from_quat(q_dead_reckoned[:, [1, 2, 3, 0]])
    # Where the dead-reckoned attitude maps the *measured* body gravity to, in
    # world coordinates. Equals `g^W` exactly when the two sources agree.
    u_world = rotations.apply(g_unit)

    corrected = _minimal_tilt_correction(u_world, axis_world) * rotations
    quat_scipy = corrected.as_quat()
    quaternions = np.atleast_2d(quat_scipy)[:, [3, 0, 1, 2]]

    # Row 0 is the anchor, verbatim (same contract as `initial_position`).
    quaternions[0] = start_np / max(float(np.linalg.norm(start_np)), 1e-12)

    if return_torch:
        return torch.as_tensor(
            quaternions,
            dtype=torch_reference.dtype
            if torch_reference.dtype.is_floating_point
            else torch.float64,
            device=torch_reference.device,
        )
    return quaternions


def _gravity_aligned_reckon_attitude_sequence(
    initial_orientation: Union[np.ndarray, torch.Tensor],
    angular_velocity: Union[np.ndarray, torch.Tensor],
    timestamps: Union[np.ndarray, torch.Tensor, tct.temporal.Timestamps],
    max_angular_velocity: float,
    gravity: Optional[Union[np.ndarray, torch.Tensor]] = None,
    gravity_world_axis=None,
) -> Union[np.ndarray, torch.Tensor]:
    """Sequential heading/rate co-integration: the **fifth** attitude regime.

    RLRP-794, observation 1. It is what makes ``ms_model.training_frame:
    gravity_aligned`` (RLRP-792) deployable *together with* a 9-D gravity
    observation space (RLRP-753) — the one combination that RLRP-792 had to
    reject at config resolution.

    **Why the other regimes cannot do it.** The heading frame's body->world
    rotation is the twist ``q_yaw(q)`` of the attitude
    (:func:`yaw_quaternion_about_axis`), so demoting a ``gravity_aligned``
    angular rate back to ``world`` needs the attitude. With no ``attitude.*``
    block the attitude is exactly what is being reconstructed *from that rate* —
    a circular dependency the one-shot vectorized demotion cannot resolve. It is
    not a re-ordering problem: the loop below is the only well-posed form::

        heading(q_i)  ->  omega^W_i = R(q_yaw(q_i)) omega^GA_i
                      ->  q_{i+1} = delta(omega^W_i dt) (x) q_i
                      ->  tilt-correct q_{i+1} from the predicted g_hat^B_{i+1}

    Note the *linear* channel has no such circularity: it is demoted in one shot
    **after** this function returns, using the very attitude sequence it produced
    (RLRP-792 review finding ``C``), so the two channels stay consistent.

    Everything else is the landed machinery, reused verbatim so the regimes cannot
    drift: the ``max_angular_velocity`` saturation rail (and its always-on report),
    the world/spatial left-composition convention (the demoted rate *is* a spatial
    rate), the ``initial_orientation`` row-0 anchor contract, and the chart-free
    :func:`_minimal_tilt_correction`.

    ``gravity=None`` is supported and degenerates to a *dead-reckoning* variant of
    the same co-integration (no tilt correction) — the observation space may carry
    neither an ``attitude`` nor a ``gravity`` block while still expressing its
    rates in the heading frame.

    Computed in numpy and converted back to the caller's backend, which makes the
    numpy and torch deploy paths bit-identical by construction.

    :param initial_orientation: validated ``(4,)`` start attitude ``[qw, qx, qy, qz]``.
    :param angular_velocity: ``(N, 3)`` angular velocity expressed in the
        ``gravity_aligned`` (heading) frame.
    :param timestamps: ``(N,)`` time axis.
    :param max_angular_velocity: saturation cap (rad/s).
    :param gravity: ``(N, 3)`` predicted body-frame gravity direction on ``S^2``,
        or ``None`` for the pure dead-reckoning variant.
    :param gravity_world_axis: world gravity axis; defaults to
        :data:`DEFAULT_GRAVITY_WORLD_AXIS`. It defines BOTH the heading twist and
        the tilt correction, so a single value keeps them consistent.
    :return: ``(N, 4)`` attitude sequence, in the backend of ``angular_velocity``.
    :raises ValueError: when ``gravity`` is not ``(N, 3)`` or its length does not
        match ``angular_velocity``.
    """
    axis_world = _normalize_gravity_world_axis(
        DEFAULT_GRAVITY_WORLD_AXIS if gravity_world_axis is None else gravity_world_axis
    )

    return_torch = isinstance(angular_velocity, torch.Tensor) or isinstance(
        gravity, torch.Tensor
    )
    torch_reference = (
        angular_velocity if isinstance(angular_velocity, torch.Tensor) else gravity
    )

    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(float)
        return np.asarray(value, dtype=float)

    omega_ga = _to_numpy(angular_velocity)
    start_np = _to_numpy(initial_orientation).reshape(-1)
    dt_values = _resolve_timestep_deltas(timestamps)
    n = len(dt_values)

    g_unit = None
    if gravity is not None:
        g_body = _to_numpy(gravity)
        if g_body.ndim != 2 or g_body.shape[-1] != 3:
            raise ValueError(
                f"`gravity` must be (N, 3) body-frame directions, got {g_body.shape}."
            )
        if len(g_body) != len(omega_ga):
            raise ValueError(
                f"`gravity` and `angular_velocity` must have the same length, got "
                f"{len(g_body)} vs {len(omega_ga)}."
            )
        g_unit = _unit_body_gravity(g_body, axis_world)

    # Same saturation rail (and always-on report) as every other regime.
    omega_magnitudes = np.linalg.norm(omega_ga, axis=-1)
    report_angular_velocity_clamp(
        omega_magnitudes,
        max_angular_velocity,
        context=(
            "compute_position_from_velocity_and_attitude "
            "(gravity_aligned co-integration)"
        ),
    )
    omega = np.array(omega_ga, dtype=float, copy=True)
    over = omega_magnitudes > max_angular_velocity
    if np.any(over):
        omega[over] = (
            omega[over] * (max_angular_velocity / omega_magnitudes[over])[:, None]
        )

    # Row 0 is the anchor, verbatim (same contract as `initial_position`).
    q_anchor = start_np / max(float(np.linalg.norm(start_np)), 1e-12)
    rotation = safe_rotation_from_quat(q_anchor[[1, 2, 3, 0]])

    quaternions = np.zeros((n, 4), dtype=float)
    quaternions[0] = q_anchor
    for i in range(1, n):
        # Heading of the attitude at step i-1 -- the rotation that DEFINES the
        # `gravity_aligned` frame there.
        twist = yaw_quaternion_about_axis(quaternions[i - 1], axis_world)
        omega_world = safe_rotation_from_quat(twist[[1, 2, 3, 0]]).apply(omega[i - 1])
        # A world/spatial rate composes on the LEFT (RLRP-758 convention).
        rotation = safe_rotation_from_rotvec(omega_world * dt_values[i]) * rotation
        if g_unit is not None:
            u_world = rotation.apply(g_unit[i])
            rotation = _minimal_tilt_correction(u_world, axis_world)[0] * rotation
        quaternions[i] = rotation.as_quat()[[3, 0, 1, 2]]

    if return_torch:
        return torch.as_tensor(
            quaternions,
            dtype=torch_reference.dtype
            if torch_reference.dtype.is_floating_point
            else torch.float64,
            device=torch_reference.device,
        )
    return quaternions


def _resolve_attitude_sequence(
    quaternions: Optional[Union[np.ndarray, torch.Tensor]],
    angular_velocity: Optional[Union[np.ndarray, torch.Tensor]],
    timestamps: Union[np.ndarray, torch.Tensor, tct.temporal.Timestamps],
    initial_orientation: Optional[Union[np.ndarray, torch.Tensor]],
    angular_velocity_frame: str,
    max_angular_velocity: float,
    gravity: Optional[Union[np.ndarray, torch.Tensor]] = None,
    gravity_world_axis=None,
) -> Union[np.ndarray, torch.Tensor]:
    """Resolve the attitude sequence the integrator must use.

    Single decision point for the three attitude regimes (RLRP-791), applied
    *before* the numpy/torch dispatch so the two backends cannot diverge:

    - measured attitude, implicit anchor (``initial_orientation is None``) —
      legacy behaviour, bit-exact;
    - measured attitude, explicit anchor — row 0 is replaced by
      ``initial_orientation`` (see :func:`_seed_attitude_sequence`);
    - **gravity reckoning** (RLRP-753 ruling ``D-deploy``) — no measured attitude
      but a predicted body-frame gravity direction: roll/pitch come from
      ``gravity``, yaw from ``angular_velocity``, anchored on
      ``initial_orientation`` (see :func:`_gravity_reckon_attitude_sequence`);
    - **``gravity_aligned`` co-integration** (RLRP-794) — the rates are expressed
      in the *heading* frame, which is itself defined by the attitude being
      reconstructed, so the propagation is sequential (see
      :func:`_gravity_aligned_reckon_attitude_sequence`). Subsumes both of the
      branches below whenever ``angular_velocity_frame`` is the heading frame,
      with or without a ``gravity`` block;
    - no measured attitude and no gravity (``quaternions is None``) — dead
      reckoning from ``initial_orientation`` with ``angular_velocity`` (ruling
      `D4`, see :func:`_dead_reckon_attitude_sequence`).

    The gravity branch is evaluated **before** dead reckoning: when the model
    predicts an orientation channel, that prediction must drive the
    reconstruction; pure dead reckoning is the *last* resort. The heading-frame
    branch is evaluated before **both**, because in that frame neither of them is
    even well-posed (:func:`_dead_reckon_attitude_sequence` only knows the
    ``body``/``world`` composition conventions).

    Introduced by action `A2` of the RLRC Explicit attitude source in deployer
    trajectory integration `.junie` plan
    (`feat_explicit_attitude_source_deploy_integration_plan_RLRP-791_20260829.md`);
    extended with the gravity regime by task ``G-5`` of the RLRP-753
    gravity-vector `.junie` plan
    (`feat_gravity_vector_orientation_representation_plan_RLRP-753_20260724.md`).

    :param quaternions: ``(N, 4)`` measured/predicted attitude sequence, or ``None``.
    :param angular_velocity: ``(N, 3)`` angular velocity, or ``None``.
    :param timestamps: ``(N,)`` time axis.
    :param initial_orientation: explicit ``(4,)`` start attitude, or ``None``.
    :param angular_velocity_frame: "body", "world" or "gravity_aligned" — frame of
        the rates.
    :param max_angular_velocity: saturation cap (rad/s).
    :param gravity: ``(N, 3)`` predicted body-frame gravity direction, or ``None``.
    :param gravity_world_axis: world gravity axis for the gravity regime.
    :return: the ``(N, 4)`` attitude sequence to integrate with.
    :raises ValueError: when no attitude information is available at all, when
        the dead-reckoning regime is selected without ``angular_velocity``, or
        when the gravity regime is selected without ``angular_velocity``.
    """
    start_attitude = (
        None
        if initial_orientation is None
        else _validate_initial_orientation(initial_orientation)
    )

    if quaternions is None:
        if start_attitude is None:
            raise ValueError(
                "no attitude information: `quaternions` is None and no "
                "`initial_orientation` was provided. Pass either a measured "
                "attitude sequence or an `initial_orientation` (+ "
                "`angular_velocity`) to dead-reckon the attitude."
            )
        if angular_velocity is None:
            raise ValueError(
                "the dead-reckoning regime (`quaternions=None`) requires "
                "`angular_velocity` to propagate the attitude from "
                "`initial_orientation`, got `angular_velocity=None`. "
                "(A gravity-configured observation space additionally requires "
                "`angular_vels.{x,y,z}` in `environment.obs_dims`; see "
                "`_assert_gravity_requires_angular_velocity`.)"
            )
        if angular_velocity_frame == GRAVITY_ALIGNED_FRAME:
            # RLRP-794: the heading frame is defined by the attitude, so the rate
            # cannot be demoted up front -- co-integrate heading and rate.
            return _gravity_aligned_reckon_attitude_sequence(
                start_attitude,
                angular_velocity,
                timestamps,
                max_angular_velocity,
                gravity=gravity,
                gravity_world_axis=gravity_world_axis,
            )
        if gravity is not None:
            return _gravity_reckon_attitude_sequence(
                start_attitude,
                gravity,
                angular_velocity,
                timestamps,
                angular_velocity_frame,
                max_angular_velocity,
                gravity_world_axis=gravity_world_axis,
            )
        return _dead_reckon_attitude_sequence(
            start_attitude,
            angular_velocity,
            timestamps,
            angular_velocity_frame,
            max_angular_velocity,
        )

    if start_attitude is None:
        return quaternions

    return _seed_attitude_sequence(quaternions, start_attitude)


def compute_position_from_velocity_and_attitude(
    linear_velocity: Union[np.ndarray, torch.Tensor],
    angular_velocity: Optional[Union[np.ndarray, torch.Tensor]],
    quaternions: Optional[Union[np.ndarray, torch.Tensor]],
    timestamps: Union[np.ndarray, torch.Tensor, tct.temporal.Timestamps],
    initial_position: Optional[Union[np.ndarray, torch.Tensor]] = None,
    initial_orientation: Optional[Union[np.ndarray, torch.Tensor]] = None,
    linear_velocity_frame: str = "body",
    angular_velocity_frame: str = "body",
    interpolate_quaternions: bool = False,
    quaternion_blend_weight: float = 0.8,
    integration_scheme: str = "forward_euler",
    max_angular_velocity: float = DEFAULT_MAX_ANGULAR_VELOCITY_RAD_S,
    gravity: Optional[Union[np.ndarray, torch.Tensor]] = None,
    gravity_world_axis=None,
    debug: bool = True,
) -> tuple:
    """Compute 3D position from linear velocity and attitude quaternions.

    Supports both numpy and torch tensor inputs. When inputs are tensors the
    computation uses torch-native quaternion math and vectorised integration,
    keeping all data on-device.

    Integration scheme (RLRP-707): positions are integrated either with the
    **trapezoidal rule** (i.e., ``pos[i] = pos[i-1] + 0.5 * (vel[i-1] + vel[i]) * dt[i]``)
    or a right-endpoint forward-Euler step. Forward-Euler has an O(dt)
    local error that accumulates to an O(T) position drift over a long horizon;
    the trapezoidal rule is O(dt**2) and exact for constant
    acceleration, so it strongly attenuates the accumulated drift.

    Alignment contract: ``vel[i]``, ``quaternions[i]``, ``timestamps[i]`` and the
    output ``positions[i]`` are all assumed to refer to the *same* time index
    ``i`` (with ``positions[0] == initial_position``). Callers reconstructing a
    *predicted* trajectory must therefore pass velocities/attitudes already
    aligned to that grid (a forward dynamics model predicts the state at ``t+1``,
    so its output must be shifted before being fed here).

    Attitude-anchor contract (RLRP-791): the start attitude is the explicit
    ``initial_orientation`` argument, symmetric to ``initial_position``. It was
    previously an implicit ``quaternions[0]`` read, which on the deploy path
    silently resolved to either a *model prediction* or the *ground truth*
    depending only on the observation-space configuration. The three supported
    regimes are resolved by :func:`_resolve_attitude_sequence`:

    - ``initial_orientation=None`` -> legacy: the anchor is ``quaternions[0]``;
    - ``initial_orientation`` given -> it replaces row 0 of ``quaternions``;
    - ``quaternions=None`` + ``gravity`` + ``initial_orientation`` +
      ``angular_velocity`` -> **gravity reckoning** (RLRP-753 ruling
      ``D-deploy``): roll/pitch from the predicted body gravity direction, yaw
      from the rates. This is the regime of a 9-D gravity observation space, and
      it is what keeps the rollout metrics free of a ground-truth orientation
      fallback (risk ``R-N1``);
    - ``quaternions=None`` + ``initial_orientation`` + ``angular_velocity`` ->
      **dead reckoning**: the whole attitude sequence is propagated from the
      anchor with the body/world rates;
    - ``quaternions=None`` + ``angular_velocity_frame='gravity_aligned'`` ->
      **heading/rate co-integration** (RLRP-794): the heading frame is defined by
      the attitude being reconstructed, so the rate demotion and the propagation
      are interleaved step by step. This is the regime of a 9-D gravity
      observation space trained with ``ms_model.training_frame: gravity_aligned``
      — the pairing RLRP-792 had to reject.

    :param linear_velocity: (N, 3) linear velocity [vx, vy, vz].
    :param angular_velocity: (N, 3) angular velocity [wx, wy, wz]. Required when
        ``quaternions`` is ``None`` (dead-reckoning regime).
    :param quaternions: (N, 4) attitude quaternions [qw, qx, qy, qz], or ``None``
        to dead-reckon the attitude from ``initial_orientation``.
    :param timestamps: (N,) timestamps in seconds.
    :param initial_position: (3,) initial position [x, y, z].
    :param initial_orientation: (4,) start attitude [qw, qx, qy, qz] anchoring the
        reconstruction. ``None`` keeps the legacy implicit ``quaternions[0]``
        anchor. Callers reconstructing a *predicted* trajectory should pass the
        trajectory ground truth, so the anchor never depends on the model output.
    :param linear_velocity_frame: "body" or "world" — frame of linear_velocity.
    :param angular_velocity_frame: "body" or "world" — frame of angular_velocity.
    :param interpolate_quaternions: whether to interpolate quaternions.
    :param quaternion_blend_weight: blending weight for interpolation.
    :param integration_scheme: trapezoidal, forward_euler (default)
    :param max_angular_velocity: maximum angular-velocity magnitude (rad/s) used
        as a saturation rail before attitude integration. Defaults to the
        conservative :data:`DEFAULT_MAX_ANGULAR_VELOCITY_RAD_S`; callers should
        pass the platform-specific cap sourced from the observation spec
        (``environment.obs_physical_bounds.angular_velocity_max_rad_s``), e.g.
        via :func:`resolve_max_angular_velocity_from_cfg`.
    :param gravity: (N, 3) predicted body-frame gravity direction on ``S^2``, or
        ``None``. Only consulted when ``quaternions`` is ``None`` (a gravity and
        an attitude block are mutually exclusive in the observation space).
    :param gravity_world_axis: world gravity axis used by the gravity regime;
        defaults to :data:`DEFAULT_GRAVITY_WORLD_AXIS`. Resolve it from the config
        with :func:`resolve_gravity_world_axis_from_cfg`.
    :param debug: whether to print debugging information.
    :return: (positions, velocities_world) each of shape (N, 3).
    :raises ValueError: on a malformed ``initial_orientation``, or when no
        attitude information is available at all (see
        :func:`_resolve_attitude_sequence`).
    """
    # RLRP-758: validate the per-channel frames (RLRP-792 opened `gravity_aligned`).
    _assert_frame("linear_velocity_frame", linear_velocity_frame)
    _assert_frame("angular_velocity_frame", angular_velocity_frame)

    # RLRP-792: the heading frame is demoted to `world` further down, once the
    # attitude sequence has been resolved (review finding `C`).
    #
    # RLRP-792 originally REJECTED `quaternions is None` here: the heading is
    # defined BY the attitude, so with no measured/predicted quaternion sequence
    # there was nothing to rotate with. RLRP-794 lifts that restriction — the
    # heading is now co-integrated with the rates by
    # `_gravity_aligned_reckon_attitude_sequence` (the fifth attitude regime), and
    # the *linear* channel never had the circularity in the first place because it
    # is demoted AFTER the resolution, with the attitude that regime produced. The
    # remaining preconditions (an `initial_orientation` anchor and an
    # `angular_velocity`) are enforced by `_resolve_attitude_sequence`.
    demote_gravity_aligned = GRAVITY_ALIGNED_FRAME in (
        linear_velocity_frame,
        angular_velocity_frame,
    )

    # RLRP-791: resolve the attitude sequence (and its explicit anchor) ONCE,
    # before the numpy/torch dispatch, so both backends integrate the very same
    # attitude and cannot diverge (the RLRP-758 class of bug).
    quaternions = _resolve_attitude_sequence(
        quaternions,
        angular_velocity,
        timestamps,
        initial_orientation,
        angular_velocity_frame,
        max_angular_velocity,
        gravity=gravity,
        gravity_world_axis=gravity_world_axis,
    )

    # RLRP-792: DEMOTE the heading frame to `world` ONCE, instead of threading a
    # third case through every branch of both backends. The heading frame's
    # body->world rotation is the yaw/twist quaternion, so a single rotation per
    # channel makes the remaining `world`/`body` branches (and their
    # numpy<->torch parity) bit-exactly the pre-RLRP-792 code.
    #
    # RLRP-792 review finding `C`: the demotion runs *after*
    # `_resolve_attitude_sequence`, so the heading used to rotate the velocities
    # is the SAME anchored attitude the integrator itself uses. Demoting before
    # the resolution used the raw, un-anchored `quaternions[0]`, which made row 0
    # of the velocity rotation inconsistent with row 0 of the reconstruction.
    if demote_gravity_aligned:
        if linear_velocity_frame == GRAVITY_ALIGNED_FRAME:
            linear_velocity = _rotate_gravity_aligned_to_world(
                linear_velocity, quaternions, gravity_world_axis
            )
            linear_velocity_frame = "world"
        if angular_velocity_frame == GRAVITY_ALIGNED_FRAME:
            angular_velocity = _rotate_gravity_aligned_to_world(
                angular_velocity, quaternions, gravity_world_axis
            )
            angular_velocity_frame = "world"

    # RLRP-744 action B-2: the memoryless ``w >= 0`` deploy/rollout flip was
    # removed (Quaternion manifold upgrade + memoryless ``w >= 0`` removal
    # `.junie` plan,
    # `rlrp-744-quaternion-manifold-upgrade-and-memoryless-wge0-removal-plan-20260720.md`).
    # Hemisphere handling relies on trajectory continuity (ingestion
    # `enforce_continuity` + RLRP-736 `align_quaternion_slots_to_reference`);
    # a sign flip is the same rotation, so reconstructed positions are
    # unaffected either way.

    # --- Torch path --------------------------------------------------------
    if isinstance(linear_velocity, torch.Tensor):
        return _compute_position_torch(
            linear_velocity,
            angular_velocity,
            quaternions,
            timestamps,
            initial_position,
            linear_velocity_frame,
            angular_velocity_frame,
            interpolate_quaternions,
            quaternion_blend_weight,
            max_angular_velocity,
            integration_scheme=integration_scheme,
        )

    # --- Numpy / scipy path (original) -------------------------------------
    if initial_position is None:
        initial_position = np.array([0.0, 0.0, 0.0])

    N = len(timestamps)
    positions = np.zeros((N, 3))
    velocities_world = np.zeros((N, 3))

    if debug:
        print(f"Linear velocity frame: {linear_velocity_frame}")
        print(f"Angular velocity frame: {angular_velocity_frame}")
        print(f"Interpolate quaternions: {interpolate_quaternions}")

    # Compute dt from consecutive timestamps
    if isinstance(timestamps, tct.temporal.Timestamps):
        dt_values = np.array(timestamps.delta_stamps, dtype=float)
    else:
        dt_values = np.zeros(N, dtype=float)
        dt_values[1:] = np.diff(timestamps)
        dt_values[0] = dt_values[1] if N > 1 else 0.0

    if debug and N > 1:
        positive_dt = dt_values[dt_values > 0]
        if positive_dt.size:
            print(
                f"Computed dt stats: mean={np.mean(positive_dt):.9f}s, std={np.std(positive_dt):.9f}s"
            )

    # Convert quaternions to scipy format [qx, qy, qz, qw]
    quaternions_scipy = quaternions[:, [1, 2, 3, 0]]

    if (
        interpolate_quaternions
        and angular_velocity is not None
        and quaternions is not None
    ):
        # Sanitize angular velocity to prevent overflow (numerical-overflow rail).
        # The clamp statistics are reported unconditionally (independent of
        # ``debug``) so physically-implausible inputs/predictions are surfaced.
        omega_magnitudes = np.linalg.norm(angular_velocity, axis=1)
        report_angular_velocity_clamp(
            omega_magnitudes,
            max_angular_velocity,
            context="compute_position_from_velocity_and_attitude (numpy)",
        )
        angular_velocity_sanitized = angular_velocity.copy()
        over = omega_magnitudes > max_angular_velocity
        if np.any(over):
            scale = max_angular_velocity / omega_magnitudes[over]
            angular_velocity_sanitized[over] = angular_velocity[over] * scale[:, None]

        rotations = []
        for i in range(N):
            if i == 0:
                rotations.append(safe_rotation_from_quat(quaternions_scipy[i]))
            else:
                dt = dt_values[i]
                # dt = 1.0 # (CRITICAL) ToDo: validate (ref task RLRP-503)

                prev_rotation = safe_rotation_from_quat(quaternions_scipy[i - 1])
                omega = angular_velocity_sanitized[i - 1]  # Use sanitized version
                omega_magnitude = np.linalg.norm(omega)

                if omega_magnitude > 1e-12:
                    # Use safe rotation creation with clamping
                    delta_rotation = safe_rotation_from_rotvec(omega * dt)

                    # RLRP-758: corrected angular-frame branch labels (were
                    # inverted; RLRP-755 report §6.2). A body-frame own-rate is a
                    # right-multiplied (intrinsic) increment `prev * delta`; a
                    # world/spatial rate is left-multiplied `delta * prev`.
                    if angular_velocity_frame == "body":
                        interpolated_rotation = prev_rotation * delta_rotation
                    else:  # world frame
                        interpolated_rotation = delta_rotation * prev_rotation

                    measured_rotation = safe_rotation_from_quat(quaternions_scipy[i])
                    blended_rotation = interpolated_rotation.inv() * measured_rotation
                    blend_rotvec = (
                        blended_rotation.as_rotvec() * quaternion_blend_weight
                    )
                    final_rotation = interpolated_rotation * safe_rotation_from_rotvec(
                        blend_rotvec
                    )

                    rotations.append(final_rotation)
                else:
                    rotations.append(safe_rotation_from_quat(quaternions_scipy[i]))

        if debug:
            print(
                f"Using angular velocity for quaternion interpolation (frame: {angular_velocity_frame})"
            )
    else:
        rotations = [safe_rotation_from_quat(q) for q in quaternions_scipy]
        if debug:
            print("Using measured quaternions directly (no interpolation)")

    # Initialize position
    positions[0] = initial_position

    # Handle linear velocity transformation based on reference frame
    if linear_velocity_frame == "body":
        for i in range(N):
            velocities_world[i] = rotations[i].apply(linear_velocity[i])
        if debug:
            print("Transformed body frame velocities to world frame")
    elif linear_velocity_frame == "world":
        velocities_world = linear_velocity.copy()
        if debug:
            print("Using world frame velocities directly")

    if integration_scheme == "trapezoidal":
        # Integrate position using world frame velocities and dt computed from timestamps.
        #
        # Integration scheme: trapezoidal rule (RLRP-707 deploy-drift fix).
        #   pos[i] = pos[i-1] + 0.5 * (vel[i-1] + vel[i]) * dt[i]
        #
        # Rationale: the right-endpoint rectangle (forward-Euler) rule
        # ``pos[i] = pos[i-1] + vel[i] * dt[i]`` has an O(dt) local truncation error
        # that *accumulates* to O(T) over the horizon, producing a systematic position
        # drift that grows with the rollout length. This is invisible during the short
        # ground-truth-feed warmup but dominates the long compounded-prediction phase of
        # the test-time rollout. The trapezoidal rule averages the interval endpoints,
        # reducing the per-step error to O(dt**2) (and being *exact* for constant
        # acceleration), which strongly attenuates the accumulated drift.
        for i in range(1, N):
            dt = dt_values[i]
            avg_velocity = 0.5 * (velocities_world[i - 1] + velocities_world[i])
            positions[i] = positions[i - 1] + avg_velocity * dt

    elif integration_scheme == "forward_euler":
        # Right-endpoint rectangle (forward-Euler) rule ``pos[i] = pos[i-1] + vel[i] * dt[i]``
        for i in range(1, N):
            dt = dt_values[i]
            positions[i] = positions[i - 1] + velocities_world[i] * dt
    else:
        raise NotImplementedError(
            f"Unsupported integration scheme: {integration_scheme}"
        )

    if debug:
        print(f"Final position drift: {positions[-1] - initial_position}")

    return positions, velocities_world


def _compute_position_torch(
    linear_velocity: torch.Tensor,
    angular_velocity: Optional[torch.Tensor],
    quaternions: Optional[torch.Tensor],
    timestamps: Union[np.ndarray, torch.Tensor, tct.temporal.Timestamps],
    initial_position: Optional[Union[np.ndarray, torch.Tensor]],
    linear_velocity_frame: str,
    angular_velocity_frame: str,
    interpolate_quaternions: bool,
    quaternion_blend_weight: float,
    max_angular_velocity: float,
    integration_scheme: str = "forward_euler",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch-native implementation of position computation from velocity and attitude.

    All internal computation is performed in float64 to match the numerical
    precision of the numpy/scipy path (scipy ``Rotation`` operates in float64
    internally).  The output tensors are cast back to the caller's dtype.
    """
    device = linear_velocity.device
    input_dtype = linear_velocity.dtype
    # Upcast to float64 for numerical stability during integration —
    # matches scipy.spatial.transform.Rotation which operates in float64.
    dtype = torch.float64

    linear_velocity = linear_velocity.to(dtype=dtype)

    if initial_position is None:
        initial_position = torch.zeros(3, dtype=dtype, device=device)
    elif isinstance(initial_position, np.ndarray):
        initial_position = torch.from_numpy(initial_position).to(
            dtype=dtype, device=device
        )
    else:
        initial_position = initial_position.to(dtype=dtype, device=device)

    # Convert timestamps
    if isinstance(timestamps, tct.temporal.Timestamps):
        dt_values = torch.tensor(timestamps.delta_stamps, dtype=dtype, device=device)
    elif isinstance(timestamps, np.ndarray):
        ts = torch.from_numpy(timestamps).to(dtype=dtype, device=device)
        N = len(ts)
        dt_values = torch.zeros(N, dtype=dtype, device=device)
        if N > 1:
            dt_values[1:] = ts[1:] - ts[:-1]
            dt_values[0] = dt_values[1]
    else:
        N = len(timestamps)
        dt_values = torch.zeros(N, dtype=dtype, device=device)
        if N > 1:
            dt_values[1:] = timestamps[1:] - timestamps[:-1]
            dt_values[0] = dt_values[1]

    N = len(dt_values)

    # Convert quaternions/angular_velocity to tensors if needed
    if quaternions is not None and isinstance(quaternions, np.ndarray):
        quaternions = torch.from_numpy(quaternions).to(dtype=dtype, device=device)
    elif quaternions is not None:
        quaternions = quaternions.to(dtype=dtype, device=device)
    if angular_velocity is not None and isinstance(angular_velocity, np.ndarray):
        angular_velocity = torch.from_numpy(angular_velocity).to(
            dtype=dtype, device=device
        )
    elif angular_velocity is not None:
        angular_velocity = angular_velocity.to(dtype=dtype, device=device)

    # --- Quaternion preparation (input is [w, x, y, z]) ---
    if (
        interpolate_quaternions
        and angular_velocity is not None
        and quaternions is not None
    ):
        # Sanitize angular velocity
        omega_mag = torch.norm(angular_velocity, dim=-1, keepdim=True)
        scale = torch.where(
            omega_mag > max_angular_velocity,
            max_angular_velocity / omega_mag,
            torch.ones_like(omega_mag),
        )

        ang_vel_sanitized = angular_velocity * scale

        # Report clamp statistics unconditionally (independent of ``debug``),
        # mirroring the numpy path — surfaces physically-implausible inputs.
        report_angular_velocity_clamp(
            omega_mag.squeeze(-1),
            max_angular_velocity,
            context="compute_position_from_velocity_and_attitude (torch)",
        )

        # Sequential loop — match numpy path: use measured quaternion at i-1
        # (not the previous blended result) as the integration starting point.
        quat_list = [quaternion_normalize(quaternions[0])]
        for i in range(1, N):
            dt = dt_values[i]
            prev_q = quaternion_normalize(quaternions[i - 1])
            omega = ang_vel_sanitized[i - 1]
            omega_magnitude = torch.norm(omega)

            if omega_magnitude > 1e-12:
                delta_q = axis_angle_to_quaternion((omega * dt).unsqueeze(0)).squeeze(0)

                # RLRP-758: corrected angular-frame branch labels — verbatim
                # mirror of the numpy path (RLRP-755 report §6.2 double
                # inversion). `quaternion_multiply` composes like scipy's
                # `Rotation.__mul__`, so body = `prev * delta` (right-multiply),
                # world = `delta * prev` (left-multiply). Landed atomically with
                # the numpy swap; guaranteed equivalent by the numpy↔torch
                # parity test (RLRP-758 plan §8).
                if angular_velocity_frame == "body":
                    interp_q = quaternion_multiply(prev_q, delta_q)
                else:
                    interp_q = quaternion_multiply(delta_q, prev_q)

                measured_q = quaternion_normalize(quaternions[i])
                blend_q = quaternion_multiply(
                    quaternion_conjugate(interp_q), measured_q
                )
                blend_rv = quaternion_to_axis_angle(blend_q.unsqueeze(0)).squeeze(0)
                blend_rv = blend_rv * quaternion_blend_weight
                correction_q = axis_angle_to_quaternion(blend_rv.unsqueeze(0)).squeeze(
                    0
                )
                final_q = quaternion_multiply(interp_q, correction_q)
                quat_list.append(quaternion_normalize(final_q))
            else:
                quat_list.append(quaternion_normalize(quaternions[i]))

        quats = torch.stack(quat_list, dim=0)
    else:
        quats = quaternion_normalize(quaternions)

    # --- Velocity transformation ---
    if linear_velocity_frame == "body":
        velocities_world = quaternion_apply(quats, linear_velocity)
    else:
        velocities_world = linear_velocity.clone()

    if integration_scheme == "trapezoidal":
        # --- Vectorised trapezoidal integration (RLRP-707 deploy-drift fix) ---
        # pos[i] = pos[i-1] + 0.5 * (vel[i-1] + vel[i]) * dt[i]
        #
        # Mirrors the numpy path: the trapezoidal rule replaces the
        # right-endpoint rectangle (forward-Euler) rule whose O(dt) local error
        # accumulated to a horizon-growing O(T) position drift in the long
        # compounded-prediction phase of the test-time rollout. Averaging the
        # interval endpoints gives an O(dt**2) per-step error (exact for constant
        # acceleration), strongly attenuating the accumulated drift.
        avg_velocities = 0.5 * (velocities_world[1:] + velocities_world[:-1])
        increments = torch.zeros_like(velocities_world)
        increments[1:] = avg_velocities * dt_values[1:].unsqueeze(-1)
        positions = torch.cumsum(increments, dim=0) + initial_position

    elif integration_scheme == "forward_euler":
        # Right-endpoint rectangle (forward-Euler) rule ``pos[i] = pos[i-1] + vel[i] * dt[i]``

        # Alt 1
        # increments = torch.zeros_like(velocities_world)
        # increments[1:] = velocities_world[1:] * dt_values[1:].unsqueeze(-1)
        # positions = torch.cumsum(increments, dim=0) + initial_position

        # Alt 2
        increments = velocities_world * dt_values.unsqueeze(-1)
        increments[0] = torch.zeros(3, dtype=dtype, device=device)
        positions = torch.cumsum(increments, dim=0) + initial_position

    else:
        raise NotImplementedError(
            f"Unsupported integration scheme: {integration_scheme}"
        )

    return positions.to(input_dtype), velocities_world.to(input_dtype)


def compute_position_from_imu_data(
    linear_vel_body, angular_vel_body, quaternions, timestamps, **kwargs
):
    """Convenience function for IMU data (both velocities in body frame)."""
    return compute_position_from_velocity_and_attitude(
        linear_vel_body,
        angular_vel_body,
        quaternions,
        timestamps,
        linear_velocity_frame="body",
        angular_velocity_frame="body",
        **kwargs,
    )


def compute_position_from_motion_capture_data(
    linear_vel_world, angular_vel_world, quaternions, timestamps, **kwargs
):
    """Convenience function for motion capture data (both velocities in world frame)."""
    return compute_position_from_velocity_and_attitude(
        linear_vel_world,
        angular_vel_world,
        quaternions,
        timestamps,
        linear_velocity_frame="world",
        angular_velocity_frame="world",
        **kwargs,
    )
