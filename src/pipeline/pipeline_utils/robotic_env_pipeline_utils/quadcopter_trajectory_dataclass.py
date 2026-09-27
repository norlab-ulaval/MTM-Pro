# coding=utf-8
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import omegaconf
import torch
from omegaconf import DictConfig

from pipeline.pipeline_utils.robotic_env_pipeline_utils.robotic_trajectory_dataclass import (
    AbstractFlatContainerToNested,
    Quaternion,
    Vector3D,
    resolve_gravity_channel,
)
from pipeline.pipeline_utils.robotic_env_pipeline_utils.utils import (
    convert_velocity_channels_to_training_frame,
    resolve_gravity_world_axis_from_cfg,
    resolve_training_frame_from_cfg,
)
from trajectory_container_tools import BaseTrajectoryFeature
from trajectory_container_tools.dataclasses.panda_dataframe_feature_dataclass import (
    BaseDataframeStampedFeatureDataclass,
)


@dataclass()
class QuadcopterMotor4D(BaseTrajectoryFeature):
    m1: Union[np.ndarray, torch.Tensor]
    m2: Union[np.ndarray, torch.Tensor]
    m3: Union[np.ndarray, torch.Tensor]
    m4: Union[np.ndarray, torch.Tensor]
    stack: Union[np.ndarray, torch.Tensor] = field(default=None, init=False)

    def on_begin_post_init_callback(self) -> None:
        super().on_begin_post_init_callback()
        if isinstance(self.m1, torch.Tensor):
            self.stack = torch.stack([self.m1, self.m2, self.m3, self.m4])
            self.stack = self.stack.swapaxes(-2, -1)
        else:
            self.stack = np.vstack([self.m1, self.m2, self.m3, self.m4])
            self.stack = self.stack.swapaxes(-2, -1)
        assert (
            self.stack.shape[-1] == 4
        ), f"Expected stack shape (..., 4), got {self.stack.shape}"
        return None


@dataclass()
class QuadcopterRobotic3D(BaseDataframeStampedFeatureDataclass):
    """
    Represents a 3D quadcopter system with associated dynamic and control-related
    features.

    This class models the state of a 3D quadcopter, including its poses, linear and
    angular velocities, orientation (as quaternion), motor inputs, and associated
    timestamps. Its purpose is to facilitate the representation and manipulation
    of quadcopter data for simulations, control systems, or analysis.

    :ivar feature_name: Name of the feature associated with the trajectory.
    :ivar timesteps_indices: Represent the indices of timesteps in the trajectory which can pertain
        to a subset of a larger trajectory (Automaticaly generated if set to None).
    :ivar batch: Boolean indicating if the data is batched (True) or pertaining to a
        single trajectory (False).
    :ivar poses: The 3D spatial positions of the quadcopter.
    :type poses: Vector3D
    :ivar linear_vels: The linear velocities of the quadcopter in 3D space.
    :type linear_vels: Vector3D
    :ivar angular_vels: The angular velocities of the quadcopter around its axes.
    :type angular_vels: Vector3D
    :ivar attitude: The orientation of the quadcopter represented as a quaternion.
    :type attitude: Quaternion
    :ivar gravity: RLRP-753 -- optional 3-D yaw-invariant orientation (the
        body-frame gravity direction ``g_hat^B = R(q)^T g^W`` on ``S^2``),
        mutually exclusive with :attr:`attitude` in ``environment.obs_dims``
        (``_assert_single_orientation_block``). ``None`` when ``gravity.*`` is
        not declared, which keeps every pre-RLRP-753 construction site
        unchanged. Mirrors :attr:`RoboticObs3D.gravity`.
    :type gravity: Optional[Vector3D]
    :ivar motor: The motor thrust values for the quadcopter's four motors.
    :type motor: QuadcopterMotor4D
    :ivar src_timestamps: The source timestamps associated with the quadcopter
                          data.
    :type src_timestamps: numpy.ndarray
    :ivar timestamps: The timestamps associated with the dataframe features. If
        provided as a numpy array, it will be converted to a `Timestamps` object.
    :type timestamps: Union[Timestamps, np.ndarray]
    :ivar obs: A computed array of state observations combining linear
               velocities, quaternion components, and angular velocities.
    :type obs: numpy.ndarray
    :ivar act: A computed array of motor actions/thrusts.
    :type act: numpy.ndarray
    """

    poses: Vector3D
    linear_vels: Vector3D
    angular_vels: Vector3D
    attitude: Quaternion  # in quaternion
    motor: QuadcopterMotor4D
    obs_shape: tuple[int]
    act_shape: tuple[int]
    obs_dims: tuple[str]
    act_dims: tuple[str]
    obs: Union[np.ndarray, torch.Tensor] = field(default=None, init=False)
    act: Union[np.ndarray, torch.Tensor] = field(default=None, init=False)
    # RLRP-758: `velocity_frame` split into per-channel frames. See the
    # RLRP-758 extend-velocity-frame-logic `.junie` plan.
    linear_velocity_frame: Optional[str] = None
    angular_velocity_frame: Optional[str] = None
    # RLRP-753 (task `G-3`): trailing defaulted field so every existing
    # construction site is unchanged. Populated by
    # `QuadcopterFlatContainerToNested.execute` ONLY when `gravity.*` is
    # declared in `environment.obs_dims` (ruling `D-place-C`).
    gravity: Optional[Vector3D] = None
    # feature_name: ContainerInternalField[Optional[str]] = field(default="Robotic 3D", init=False)

    def on_begin_post_init_callback(self) -> None:
        super().on_begin_post_init_callback()

        obs_dims = []
        for each_obs_dim in self.obs_dims:
            obs_dims.append(self.get_dynamic_attribute(each_obs_dim))

        if obs_dims and isinstance(obs_dims[0], torch.Tensor):
            self.obs = torch.stack(obs_dims)
        else:
            self.obs = np.stack(obs_dims)

        act_dims = []
        for each_act_dim in self.act_dims:
            act_dims.append(self.get_dynamic_attribute(each_act_dim))

        if act_dims and isinstance(act_dims[0], torch.Tensor):
            self.act = torch.stack(act_dims)
        else:
            self.act = np.stack(act_dims)

        self.obs = self.obs.swapaxes(-2, -1)
        self.act = self.act.swapaxes(-2, -1)
        assert (
            self.obs.shape[-1] == self.obs_shape[-1]
        ), f"Expected obs shape (..., {self.obs.shape[-1]}), got {self.obs_shape[-1]}"
        assert (
            self.act.shape[-1] == self.act_shape[-1]
        ), f"Expected act shape (..., {self.act.shape[-1]}), got {self.act_shape[-1]}"
        return None


@dataclass()
class QuadcopterRobotic3DFlat(BaseDataframeStampedFeatureDataclass):
    """
    Quadcopter robotic environment trajectory dataclass
    """

    pos_x: np.ndarray
    pos_y: np.ndarray
    pos_z: np.ndarray
    l_vel_x: np.ndarray
    l_vel_y: np.ndarray
    l_vel_z: np.ndarray
    quat_w: np.ndarray
    quat_x: np.ndarray
    quat_y: np.ndarray
    quat_z: np.ndarray
    a_vel_x: np.ndarray
    a_vel_y: np.ndarray
    a_vel_z: np.ndarray
    mot_1: np.ndarray
    mot_2: np.ndarray
    mot_3: np.ndarray
    mot_4: np.ndarray
    # t: np.ndarray
    # feature_name: ContainerInternalField[Optional[str]] = field(default="Robotic 3D flat", init=False)

    def on_begin_post_init_callback(self) -> None:
        super().on_begin_post_init_callback()
        return None


class QuadcopterFlatContainerToNested(AbstractFlatContainerToNested):

    @staticmethod
    def execute(
        cfg: omegaconf.DictConfig, flat_container: QuadcopterRobotic3DFlat
    ) -> QuadcopterRobotic3D:
        """
        Convenience function to converts a flat representation of a quadcopter's trajectory to a nested
         representatioon.

        :param flat_container: An instance of `Quadcopter3DFlat`.
        :param cfg: "body" or "world" - angular and linear velocity frame of reference.
        :return: An instance of `Quadcopter3D` using a nested structure.
        """
        # RLRP-758: read per-channel native frames + target training frame.
        native_linear_frame = cfg.environment.data.linear_velocity_frame
        native_angular_frame = cfg.environment.data.angular_velocity_frame
        training_frame = resolve_training_frame_from_cfg(cfg)
        obs_dims = omegaconf.OmegaConf.to_object(cfg.environment.obs_dims)
        act_dim = omegaconf.OmegaConf.to_object(cfg.environment.act_dims)
        # RLRP-736 S3.1 (gap (2) — double cover): sign-canonicalize + enforce
        # temporal continuity on the attitude quaternion at the causal-ordered
        # ingestion seam, so normalizer statistics and regression targets are
        # smooth. Centralized on the QUATERNION feature group; no-op for the
        # neutral/math handler (bit-exact).
        from tools.feature_handling_tools.env_handlers import (
            canonicalize_attitude_components,
            resolve_act_shape,
            resolve_obs_shape,
        )

        quat_w, quat_x, quat_y, quat_z = canonicalize_attitude_components(
            cfg,
            flat_container.quat_w,
            flat_container.quat_x,
            flat_container.quat_y,
            flat_container.quat_z,
        )
        attitude = Quaternion(
            w=quat_w,
            x=quat_x,
            y=quat_y,
            z=quat_z,
        )

        # RLRP-753 (task `G-2`, ruling `D-place-C`): derive the body-frame
        # gravity direction from the *canonicalized* quaternion, ONCE per
        # trajectory, gated on the declared observation space. The quaternion
        # itself stays the ingestion source of truth (it still drives the
        # velocity frame conversion below) even when it is not exposed in the
        # observation vector.
        gravity = resolve_gravity_channel(
            cfg, obs_dims, (quat_w, quat_x, quat_y, quat_z)
        )

        # RLRP-758: ingestion-time native->training-frame conversion (both
        # channels), then stamp stored frames = training frame.
        lin_arr, ang_arr = convert_velocity_channels_to_training_frame(
            l_vel=(flat_container.l_vel_x, flat_container.l_vel_y, flat_container.l_vel_z),
            a_vel=(flat_container.a_vel_x, flat_container.a_vel_y, flat_container.a_vel_z),
            quat_wxyz=(quat_w, quat_x, quat_y, quat_z),
            native_linear_frame=native_linear_frame,
            native_angular_frame=native_angular_frame,
            training_frame=training_frame,
            # RLRP-792: the heading frame is defined about this axis; a no-op for
            # every `world`/`body` run (the argument is only read when one of the
            # frames is `gravity_aligned`).
            gravity_world_axis=resolve_gravity_world_axis_from_cfg(cfg),
        )

        container = QuadcopterRobotic3D(
            poses=Vector3D(
                x=flat_container.pos_x,
                y=flat_container.pos_y,
                z=flat_container.pos_z,
            ),
            linear_vels=Vector3D(x=lin_arr[:, 0], y=lin_arr[:, 1], z=lin_arr[:, 2]),
            angular_vels=Vector3D(x=ang_arr[:, 0], y=ang_arr[:, 1], z=ang_arr[:, 2]),
            attitude=attitude,
            gravity=gravity,
            motor=QuadcopterMotor4D(
                m1=flat_container.mot_1,
                m2=flat_container.mot_2,
                m3=flat_container.mot_3,
                m4=flat_container.mot_4,
            ),
            # timestamps=flat_container.t,
            timestamps=flat_container.timestamps,
            linear_velocity_frame=training_frame,
            angular_velocity_frame=training_frame,
            obs_dims=obs_dims,
            act_dims=act_dim,
            # RLRP-736 shape-key removal: resolve from obs_dims/act_dims
            # (single source of truth) rather than the removed config key.
            obs_shape=resolve_obs_shape(cfg),
            act_shape=resolve_act_shape(cfg),
        )
        return container
