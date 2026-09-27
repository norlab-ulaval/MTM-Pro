# coding=utf-8
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import omegaconf
import torch

from trajectory_container_tools.dataclasses import TestMotionTrajectoryDataclass
from trajectory_container_tools.dataclasses.core.base_trajectory_dataclass import (
    BaseTrajectoryFeature,
)
from trajectory_container_tools.dataclasses.panda_dataframe_feature_dataclass import (
    BaseDataframeStampedFeatureDataclass,
)
import trajectory_container_tools as tct


@dataclass()
class Vector3D(BaseTrajectoryFeature):
    x: Union[np.ndarray, torch.Tensor]
    y: Union[np.ndarray, torch.Tensor]
    z: Union[np.ndarray, torch.Tensor]
    stack: Union[np.ndarray, torch.Tensor] = field(default=None, init=False)

    def on_begin_post_init_callback(self) -> None:
        super().on_begin_post_init_callback()
        if isinstance(self.x, torch.Tensor):
            self.stack = torch.stack([self.x, self.y, self.z])
            self.stack = self.stack.swapaxes(-2, -1)
        else:
            self.stack = np.vstack([self.x, self.y, self.z])
            self.stack = self.stack.swapaxes(-2, -1)
        assert (
            self.stack.shape[-1] == 3
        ), f"Expected stack shape (..., 3), got {self.stack.shape}"
        return None


@dataclass()
class Quaternion(BaseTrajectoryFeature):
    w: Union[np.ndarray, torch.Tensor]
    x: Union[np.ndarray, torch.Tensor]
    y: Union[np.ndarray, torch.Tensor]
    z: Union[np.ndarray, torch.Tensor]
    stack: Union[np.ndarray, torch.Tensor] = field(default=None, init=False)

    def on_begin_post_init_callback(self) -> None:
        super().on_begin_post_init_callback()
        if isinstance(self.w, torch.Tensor):
            self.stack = torch.stack([self.w, self.x, self.y, self.z])
            self.stack = self.stack.swapaxes(-2, -1)
        else:
            self.stack = np.vstack([self.w, self.x, self.y, self.z])
            self.stack = self.stack.swapaxes(-2, -1)
        assert (
            self.stack.shape[-1] == 4
        ), f"Expected stack shape (..., 4), got {self.stack.shape}"
        return None


@dataclass()
class RoboticObs3D(BaseDataframeStampedFeatureDataclass):
    """
    Represents a 3D observational feature of a quadcopter.

    This class encapsulates the state of a quadcopter in a 3D space, including its
    linear and angular velocities, as well as its orientation represented by a
    quaternion. It extends the `BaseDataframeStampedFeatureDataclass` for handling
    time-stamped feature data.

    :ivar feature_name: Name of the feature associated with the trajectory.
    :ivar timesteps_indices: Represent the indices of timesteps in the trajectory which can pertain
        to a subset of a larger trajectory (Automaticaly generated if set to None).
    :ivar batch: Boolean indicating if the data is batched (True) or pertaining to a
        single trajectory (False).
    :ivar timestamps: The timestamps associated with the dataframe features. If
        provided as a numpy array, it will be converted to a `Timestamps` object.
    :type timestamps: Union[Timestamps, np.ndarray]
    :ivar linear_vels: The linear velocities of the quadcopter in 3D space.
    :type linear_vels: Vector3D
    :ivar angular_vels: The angular velocities of the quadcopter in 3D space.
    :type angular_vels: Vector3D
    :ivar attitude: The orientation of the quadcopter represented as a quaternion.
    :type attitude: Quaternion
    """

    linear_vels: Vector3D
    attitude: Quaternion  # in quaternion
    angular_vels: Vector3D
    #: RLRP-761 S5.6 -- optional 3-D yaw-invariant orientation (body-frame
    #: gravity direction), mutually exclusive with :attr:`attitude`. ``None``
    #: when ``gravity.*`` is not declared in ``environment.obs_dims``, which
    #: keeps every pre-RLRP-761 construction site unchanged.
    gravity: Optional[Vector3D] = None
    # feature_name: ContainerInternalField[Optional[str]] = field(default="Robotic 3D obs", init=False)

    def on_begin_post_init_callback(self) -> None:
        super().on_begin_post_init_callback()


def obs_2_vector(
    obs: Union[np.ndarray, torch.Tensor],
    timestamps: Union[np.ndarray, torch.Tensor, tct.temporal.Timestamps],
    cfg: omegaconf.DictConfig,
) -> RoboticObs3D:
    obs_dims: list = omegaconf.OmegaConf.to_object(cfg.environment.obs_dims)
    act_dim: list = omegaconf.OmegaConf.to_object(cfg.environment.act_dims)

    # RLRP-736 S1.6 (consolidate): block index resolution — including the
    # quaternion attitude block — is delegated to the centralized feature
    # contract (single source of truth) instead of hand-written `.index(...)`
    # calls. Partial-presence semantics are preserved: a block absent from the
    # config simply has no entry and yields `None` here.
    from tools.feature_handling_tools.env_handlers import (
        robotic3d_obs_block_indices,
    )

    block_indices = robotic3d_obs_block_indices(obs_dims)

    if "linear_vels" in block_indices:
        lin_ix = block_indices["linear_vels"]
        linear_vels = Vector3D(
            x=obs[..., lin_ix[0]],
            y=obs[..., lin_ix[1]],
            z=obs[..., lin_ix[2]],
        )
    else:
        linear_vels = None

    if "attitude" in block_indices:
        att_ix = block_indices["attitude"]
        attitude = Quaternion(
            w=obs[..., att_ix[0]],
            x=obs[..., att_ix[1]],
            y=obs[..., att_ix[2]],
            z=obs[..., att_ix[3]],
        )
    else:
        attitude = None

    if "angular_vels" in block_indices:
        ang_ix = block_indices["angular_vels"]
        angular_vels = Vector3D(
            x=obs[..., ang_ix[0]],
            y=obs[..., ang_ix[1]],
            z=obs[..., ang_ix[2]],
        )
    else:
        angular_vels = None

    # RLRP-761 S5.6: the yaw-invariant orientation alternative. A 3-D direction
    # has no double cover, so — unlike the quaternion — it needs no temporal
    # sign-continuity handling anywhere on the data path.
    if "gravity" in block_indices:
        grav_ix = block_indices["gravity"]
        gravity = Vector3D(
            x=obs[..., grav_ix[0]],
            y=obs[..., grav_ix[1]],
            z=obs[..., grav_ix[2]],
        )
    else:
        gravity = None

    return RoboticObs3D(
        linear_vels=linear_vels,
        attitude=attitude,
        angular_vels=angular_vels,
        gravity=gravity,
        timestamps=timestamps,
    )


def obs_2_dict(
    obs: Union[np.ndarray, torch.Tensor],
    test_env: TestMotionTrajectoryDataclass,
    timestamps: Union[np.ndarray, torch.Tensor, tct.temporal.Timestamps],
    cfg: omegaconf.DictConfig,
) -> dict:
    """Unpack an observation sequence into the trajectory-integrator input keys.

    RLRP-791: the *attitude source* is now reported explicitly through
    ``quaternions_are_gt`` instead of being an unrecorded consequence of
    ``cfg.environment.obs_dims``. The attitude channel has two mutually
    exclusive sources:

    - the ``attitude`` block of ``obs`` — i.e. a **model prediction** at deploy
      time — whenever that block is part of the observation space;
    - ``test_env.orientation_gt`` — the trajectory **ground truth** — when the
      observation space carries no attitude block.

    Both are legitimate, but they are *not* interchangeable: the integrator uses
    the attitude to rotate the body-frame velocity into the world frame and (its
    first row) to anchor the attitude propagation, so a silent source swap
    changes the reconstruction without leaving a trace. Callers must therefore
    consume the flag (and, for the anchor, pass an explicit ``initial_orientation``
    to :func:`~pipeline.pipeline_utils.robotic_env_pipeline_utils.utils.compute_position_from_velocity_and_attitude`).

    Permanent contract. Introduced by action `A1` of the RLRC Explicit attitude
    source in deployer trajectory integration `.junie` plan
    (`feat_explicit_attitude_source_deploy_integration_plan_RLRP-791_20260829.md`).

    :param obs: ``(N, obs_dim)`` observation sequence (numpy or torch).
    :param test_env: the test trajectory; read only for its ``orientation_gt``
        fallback (may be ``None`` when the obs space carries the attitude block).
    :param timestamps: ``(N,)`` time axis carried through to the integrator.
    :param cfg: Hydra config (uses ``cfg.environment.obs_dims`` /
        ``cfg.environment.act_dims``).
    :return: the integrator input keys plus ``quaternions_are_gt``, a bool
        stating whether ``quaternions`` came from the ground truth (``True``) or
        from the predicted observation block (``False``), and ``orientation_source``
        (RLRP-753), a three-valued label naming the source explicitly. Note that
        ``quaternions_are_gt`` and ``orientation_source`` are **metadata**: they are
        not integrator arguments and must be popped before a ``**`` splat.
        ``gravity`` / ``gravity_world_axis`` *are* integrator arguments;
        ``gravity_world_axis`` is forwarded unconditionally (RLRP-792 review
        finding ``B``) because the ``gravity_aligned`` heading frame is defined
        about the very same axis while reading the ``attitude`` block.

    RLRP-753 ruling ``D-deploy`` adds a **third** attitude source: the 3-D
    ``gravity`` block. It is mutually exclusive with ``attitude``
    (``_assert_single_orientation_block``) and, being a *model prediction*, it
    takes precedence over the ground-truth fallback. In that case ``quaternions``
    is ``None`` and ``gravity`` is populated, which selects the gravity-reckoning
    regime of
    :func:`~pipeline.pipeline_utils.robotic_env_pipeline_utils.utils._resolve_attitude_sequence`
    (roll/pitch from ``g_hat^B``, yaw from ``angular_velocity``). ``quaternions_are_gt``
    stays ``False`` there, preserving the RLRP-791 contract.
    """
    from pipeline.pipeline_utils.robotic_env_pipeline_utils.utils import (
        resolve_gravity_world_axis_from_cfg,
    )

    obs3d = obs_2_vector(obs, timestamps, cfg)
    # RLRP-753 `D-deploy`: the gravity block is a *predicted* orientation source
    # and is mutually exclusive with `attitude` (`_assert_single_orientation_block`),
    # so it takes precedence over the ground-truth fallback. `quaternions=None`
    # + `gravity` selects the gravity-reckoning regime of
    # `_resolve_attitude_sequence`.
    gravity_is_source = obs3d.attitude is None and obs3d.gravity is not None
    quaternions_are_gt = obs3d.attitude is None and not gravity_is_source

    if gravity_is_source:
        quaternions = None
        orientation_source = "gravity_prediction"
    elif quaternions_are_gt:
        quaternions = test_env.orientation_gt
        orientation_source = "ground_truth"
    else:
        quaternions = obs3d.attitude.stack
        orientation_source = "attitude_prediction"

    return {
        "linear_velocity": obs3d.linear_vels.stack,
        "quaternions": quaternions,
        "gravity": obs3d.gravity.stack if gravity_is_source else None,
        # RLRP-792 review finding `B`: forward the configured axis
        # UNCONDITIONALLY. It is an integrator *argument* (not gravity-regime
        # metadata): the `gravity_aligned` heading frame is also defined about it,
        # and that regime reads the *attitude* block, so gating the axis on
        # `gravity_is_source` silently fell back to `DEFAULT_GRAVITY_WORLD_AXIS`
        # at deploy while ingestion used the configured one — a silent frame
        # mismatch on the first non-z-up dataset (defeats ruling `D-axis`).
        "gravity_world_axis": resolve_gravity_world_axis_from_cfg(cfg),
        "quaternions_are_gt": quaternions_are_gt,
        "orientation_source": orientation_source,
        "angular_velocity": (
            obs3d.angular_vels.stack if obs3d.angular_vels is not None else None
        ),
        "timestamps": obs3d.timestamps,
    }


def resolve_gravity_channel(
    cfg: omegaconf.DictConfig,
    obs_dims,
    quat_wxyz,
) -> Optional[Vector3D]:
    """Derive the body-frame gravity-direction channel at ingestion, or ``None``.

    RLRP-753 task ``G-2`` / ruling ``D-place-C``. Shared by
    ``QuadcopterFlatContainerToNested.execute`` and its UGV twin so the two
    builders can never drift.

    The channel is computed **only** when all three ``gravity.{x,y,z}`` dims are
    declared in ``environment.obs_dims`` — i.e. it is a strict no-op (``None``)
    for every pre-RLRP-753 configuration, which keeps the ingested observation
    vector bit-exact.

    ``quat_wxyz`` MUST be the *canonicalized* attitude components (the output of
    ``canonicalize_attitude_components``); the gravity direction is by
    construction invariant to the quaternion sign, but sharing the canonicalized
    source keeps a single ingestion source of truth.

    :param cfg: The Hydra config (read for ``environment.data.gravity_world_axis``).
    :param obs_dims: The declared observation dimension names.
    :param quat_wxyz: ``(w, x, y, z)`` component tuple of the attitude.
    :return: A :class:`Vector3D` on ``S^2``, or ``None`` when ``gravity.*`` is
        not declared.
    """
    from tools.feature_handling_tools.env_handlers import _GRAVITY_DIMS

    from pipeline.pipeline_utils.robotic_env_pipeline_utils.utils import (
        quaternion_to_body_gravity,
        resolve_gravity_world_axis_from_cfg,
    )

    present = set(obs_dims or ())
    if not all(name in present for name in _GRAVITY_DIMS):
        return None
    quat = np.stack([np.asarray(c, dtype=float) for c in quat_wxyz], axis=-1)
    g_body = quaternion_to_body_gravity(
        quat, g_world=resolve_gravity_world_axis_from_cfg(cfg)
    )
    return Vector3D(x=g_body[:, 0], y=g_body[:, 1], z=g_body[:, 2])


class AbstractFlatContainerToNested(ABC):
    """
    Provides an abstract base class for defining the transformation of a flat
    container into a nested or structured data format.

    This class enforces the implementation of the `execute` method, which should
    define how the input flat container is processed and converted into a
    nested structure. It is intended to be subclassed and customized for specific
    transformation scenarios.

    """

    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def execute(
        cfg: omegaconf.DictConfig, flat_container: BaseDataframeStampedFeatureDataclass
    ) -> BaseDataframeStampedFeatureDataclass:
        pass
