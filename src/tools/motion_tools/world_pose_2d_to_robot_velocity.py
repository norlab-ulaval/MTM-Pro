# coding=utf-8
import math
import warnings
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_robot_velocity_from_2d_world_poses(
    world_pose: np.ndarray, next_world_pose: np.ndarray, dt: float
) -> Tuple[float, float, float]:
    """Compute longitudinal, lateral and angular velocity in the robot frame based on two
    consecutive world frame poses with 2D coordinate and orientation.

    :param world_pose: reference pose in the world frame
    :param next_world_pose: following pose in the world frame
    :param dt: delta time between poses (time delta measurement or timestep increment)
    :return: longitudinal, lateral and angular velocity in the robot frame
    """
    pose_x = world_pose[0]
    pose_y = world_pose[1]
    pose_theta = world_pose[2]

    next_pose_x = next_world_pose[0]
    next_pose_y = next_world_pose[1]
    next_pose_theta = next_world_pose[2]

    try:
        vel_x_world = next_pose_x - pose_x
        vel_y_world = next_pose_y - pose_y

        longitudinal_vel = compute_robot_longitudinal_vel_2d(vel_x_world, vel_y_world, pose_theta)
        lateral_vel = compute_robot_lateral_vel_2d(vel_x_world, vel_y_world, pose_theta)

        # ang_vel_z = compute_robot_angular_velocity_euler_3d(pose_theta, next_pose_theta)
        # ang_vel_z = compute_robot_angular_velocity_2d_arctan2(pose_theta, next_pose_theta)
        ang_vel_z = compute_robot_angular_velocity_2d(pose_theta, next_pose_theta)

        longitudinal_vel /= dt
        lateral_vel /= dt
        ang_vel_z /= dt

    except RuntimeWarning as e:
        if str(e) == "divide by zero encountered in double_scalars":
            raise ZeroDivisionError(
                f"Parameter `dt` must be greater than 0, Currently {dt=}"
            ) from e
        else:
            raise
    except ZeroDivisionError as e:
        raise ZeroDivisionError(f"Parameter `dt` must be greater than 0, Currently {dt=}") from e
    else:
        return longitudinal_vel, lateral_vel, ang_vel_z


def compute_robot_longitudinal_vel_2d(
    vel_x_world: float, vel_y_world: float, pose_theta: float
) -> float:
    """
    Robot frame linear velocity on the x-axis (aka body-frame forward velocity)
    """
    return np.sin(pose_theta) * vel_y_world + np.cos(pose_theta) * vel_x_world


def compute_robot_lateral_vel_2d(
    vel_x_world: float, vel_y_world: float, pose_theta: float
) -> float:
    """
    Robot frame linear velocity on the y-axis  (aka body-frame sideway velocity)
    """
    return math.cos(pose_theta) * vel_y_world - math.sin(pose_theta) * vel_x_world


def compute_robot_angular_velocity_2d(pose_theta: float, next_pose_theta: float) -> float:
    """
    Angle wrap basic version (slow). Handle rotation crossing over the x-axis i.e. 0 vs 2pi
    """
    ang_vel_z = next_pose_theta - pose_theta

    ang_vel_z = normalize_angular_velocity_hack(ang_vel_z)

    return ang_vel_z


def normalize_angular_velocity_hack(ang_vel_z):
    """Assume the car did not flip orientation in one timestep"""
    if ang_vel_z > np.pi:
        ang_vel_z -= 2 * np.pi
    elif ang_vel_z < -np.pi:
        ang_vel_z += 2 * np.pi
    return ang_vel_z


def compute_robot_angular_velocity_2d_arctan2(pose_theta: float, next_pose_theta: float) -> float:
    """
    Angle wrap arctan version (faster). Handle rotation crossing over the x-axis i.e. 0 vs 2pi

    Credit https://gist.github.com/IamPhytan/923773aaf1785f2845d1503c09970faf
    """
    ang_vel_z = next_pose_theta - pose_theta

    ang_vel_z = np.arctan2(np.sin(ang_vel_z), np.cos(ang_vel_z))

    return ang_vel_z


def compute_robot_angular_velocity_euler_3d(pose_theta: float, next_pose_theta: float) -> float:
    """
    Note: It's the foundation implementation for the 3D case
    """

    initial_orientation = R.from_euler(seq="z", angles=pose_theta, degrees=False)
    next_orientation = R.from_euler(seq="z", angles=next_pose_theta, degrees=False)
    delta_orientation = R.from_matrix(
        np.matmul(next_orientation.as_matrix(), np.transpose(initial_orientation.as_matrix()))
    )
    ang_vel_z = delta_orientation.as_euler(seq="xyz", degrees=False)[2]

    raise warnings.warn(
        "todo: assess numerical stability of angular velocity alternate computation"
    )  # todo

    return ang_vel_z
