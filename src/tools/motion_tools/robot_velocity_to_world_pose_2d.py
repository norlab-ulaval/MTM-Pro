# coding=utf-8

from typing import Optional, Tuple
import math
import numpy as np


def compute_next_2d_world_pose_from_robot_velocity(
    world_pose: np.ndarray,
    robot_vel: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute next 2D robot pose (coordinate and orientation) in the world frame based on
    the current pose and velocity update (longitudinal, lateral and angular velocity) in the
    robot frame.

    :param world_pose: reference 2D pose in the world frame
    :param robot_vel: current longitudinal, lateral and angular velocity in the robot frame
    :param dt: time delta between poses (time delta measurement or timestep increment)
    :return: next_world_pose
    """
    pose_x = world_pose[0]
    pose_y = world_pose[1]
    pose_theta = world_pose[2]

    longitudinal_vel = robot_vel[0]
    lateral_vel = robot_vel[1]
    ang_vel_z = robot_vel[2]

    # ....Compute linear velocity in the world frame...............................................
    vel_x_world = np.cos(pose_theta) * longitudinal_vel - np.sin(pose_theta) * lateral_vel
    vel_y_world = np.sin(pose_theta) * longitudinal_vel + np.cos(pose_theta) * lateral_vel

    # ....Compute next pose in the world frame.....................................................
    next_world_pose_x = pose_x + vel_x_world * dt
    next_world_pose_y = pose_y + vel_y_world * dt
    next_world_pose_theta = normalize_world_orientation(pose_theta + ang_vel_z * dt)

    next_world_pose = np.array(
        [next_world_pose_x, next_world_pose_y, next_world_pose_theta], dtype=np.float64
    )
    return next_world_pose


def normalize_world_orientation(yaw: float) -> float:
    """Bound orientation (yaw) between 0 and 2pi

    Note: as in original f110-gym
    :param yaw: orentation in radian
    :return: normalized orientation
    """
    if yaw > 2 * np.pi:
        yaw -= 2 * np.pi
    elif yaw < 0:
        yaw += 2 * np.pi

    return yaw
