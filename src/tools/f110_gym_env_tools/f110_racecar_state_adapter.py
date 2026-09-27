# coding=utf-8
import numpy as np


# ::: Convert f110-gym RaceCar innner state representation ::::::::::::::::::::::::::::::::::::::::
# ....Motion only..................................................................................
def convert_f110_racecar_extended_state_to_motion_only_representation(
    extended_state: np.ndarray,
) -> np.ndarray:
    """Convert a f110-gym RaceCar innner state representation (extended with `linear_vels_y`)
    to a motion only RaceCar state representation for neural net ingestion.

    f110-gym RaceCar state representation:
    - state (extended):
          0  1      2           3             4            5          6            7
        [ x, y,      _     , lin_vels_x, poses_theta, ang_vels_z,     _     , lin_vels_y ]
    - state (motion):
          0  1      2           3             4            5          6            7
        [ x, y, poses_theta, lin_vels_x, lin_vels_y , ang_vels_z ]

    Note that the original RaceCar state representation uses different naming convention in the
    codebase

    - state (original):
          0  1      2           3             4            5          6            7
        [ x, y, steer_angle,   vel     ,  yaw_angle ,  yaw_rate , slip_angle]

    :param extended_state: flat observation extended representation
    :return: flat motion only state representation
    """
    motion_only_state = extended_state[[0, 1, 4, 3, 7, 5]].copy()
    return motion_only_state


def convert_f110_racecar_motion_only_state_to_extended_representation(
    motion_only_state: np.ndarray,
) -> np.ndarray:
    """Convert a f110-gym RaceCar inner state representation (motion only) to a RaceCar
    extended state representation.

    f110-gym RaceCar state representation:
    - state (extended):
          0  1      2           3             4            5          6            7
        [ x, y,      _     , lin_vels_x, poses_theta, ang_vels_z,     _     , lin_vels_y ]
    - state (motion):
          0  1      2           3             4            5          6            7
        [ x, y, poses_theta, lin_vels_x, lin_vels_y , ang_vels_z ]

    Note that the original RaceCar state representation uses different naming convention in the
    codebase

    - state (original):
          0  1      2           3             4            5          6            7
        [ x, y, steer_angle,   vel     ,  yaw_angle ,  yaw_rate , slip_angle]

    :param motion_only_state: flat motion only state representation
    :return: flat observation extended representation
    """
    extended_state = motion_only_state[[0, 1, 1, 3, 2, 5, 5, 4]].copy()
    extended_state[2] = 0.0
    extended_state[6] = 0.0
    return extended_state


# ....Pose only....................................................................................
def convert_f110_racecar_extended_state_to_pose_only_representation(
    extended_state: np.ndarray,
) -> np.ndarray:
    """Convert a f110-gym RaceCar innner state representation (extended with `linear_vels_y`)
    to a pose only RaceCar state representation for neural net ingestion.

    f110-gym RaceCar state representation:
    - state (extended):
          0  1      2           3             4            5          6            7
        [ x, y,      _     , lin_vels_x, poses_theta, ang_vels_z,     _     , lin_vels_y ]
    - state (motion):
          0  1      2           3             4            5          6            7
        [ x, y, poses_theta, lin_vels_x, lin_vels_y , ang_vels_z ]
    - state (pose):
          0  1      2           3             4            5          6            7
        [ x, y, poses_theta ]

    Note that the original RaceCar state representation uses different naming convention in the
    codebase

    - state (original):
          0  1      2           3             4            5          6            7
        [ x, y, steer_angle,   vel     ,  yaw_angle ,  yaw_rate , slip_angle]

    :param extended_state: flat observation extended representation
    :return: flat motion pose only state representation
    """
    pose_only_state = extended_state[[0, 1, 4]].copy()
    return pose_only_state


def convert_f110_racecar_pose_only_state_to_extended_representation(
    pose_only_state: np.ndarray,
) -> np.ndarray:
    """Convert a f110-gym RaceCar inner state representation (pose only) to a RaceCar
    extended state representation.

    f110-gym RaceCar state representation:
    - state (extended):
          0  1      2           3             4            5          6            7
        [ x, y,      _     , lin_vels_x, poses_theta, ang_vels_z,     _     , lin_vels_y ]
    - state (motion):
          0  1      2           3             4            5          6            7
        [ x, y, poses_theta, lin_vels_x, lin_vels_y , ang_vels_z ]
    - state (pose):
          0  1      2           3             4            5          6            7
        [ x, y, poses_theta ]

    Note that the original RaceCar state representation uses different naming convention in the
    codebase

    - state (original):
          0  1      2           3             4            5          6            7
        [ x, y, steer_angle,   vel     ,  yaw_angle ,  yaw_rate , slip_angle]

    :param pose_only_state: flat pose only state representation
    :return: flat observation extended representation
    """
    extended_state = pose_only_state[[0, 1, 1, 1, 2, 2, 2, 2]].copy()
    extended_state[2:4] = 0.0
    extended_state[5:7] = 0.0
    return extended_state


def append_pose_to_f110_racecar_extended_representation(
    extended_state: np.ndarray,
    pose_only_state: np.ndarray,
) -> np.ndarray:
    """Append velocity state information to a RaceCar extended state representation.

    f110-gym RaceCar state representation:
    - state (extended):
          0  1      2           3             4            5          6            7
        [ x, y,      _     , lin_vels_x, poses_theta, ang_vels_z,     _     , lin_vels_y ]
    - state (motion):
          0  1      2           3             4            5          6            7
        [ x, y, poses_theta, lin_vels_x, lin_vels_y , ang_vels_z ]
    - state (pose):
          0  1      2           3             4            5          6            7
        [ x, y, poses_theta ]
    - state (velocity):
                          [ lin_vels_x, lin_vels_y , ang_vels_z ]


    Note that the original RaceCar state representation uses different naming convention in the
    codebase

    - state (original):
          0  1      2           3             4            5          6            7
        [ x, y, steer_angle,   vel     ,  yaw_angle ,  yaw_rate , slip_angle]

    :param extended_state: flat observation extended representation
    :param pose_only_state: flat motion pose only state representation
    :return: flat observation extended representation
    """
    extended_state[0:2] = pose_only_state[0:2].copy()
    extended_state[4] = pose_only_state[2].copy()
    return extended_state


# ....Velocity2D only................................................................................
def append_velocity_to_f110_racecar_extended_representation(
    extended_state: np.ndarray,
    velocity_only_state: np.ndarray,
) -> np.ndarray:
    """Append velocity state information to a RaceCar extended state representation.

    f110-gym RaceCar state representation:
    - state (extended):
          0  1      2           3             4            5          6            7
        [ x, y,      _     , lin_vels_x, poses_theta, ang_vels_z,     _     , lin_vels_y ]
    - state (motion):
          0  1      2           3             4            5          6            7
        [ x, y, poses_theta, lin_vels_x, lin_vels_y , ang_vels_z ]
    - state (pose):
          0  1      2           3             4            5          6            7
        [ x, y, poses_theta ]
    - state (velocity):
                          [ lin_vels_x, lin_vels_y , ang_vels_z ]

    Note that the original RaceCar state representation uses different naming convention in the
    codebase

    - state (original):
          0  1      2           3             4            5          6            7
        [ x, y, steer_angle,   vel     ,  yaw_angle ,  yaw_rate , slip_angle]

    :param extended_state: flat observation extended representation
    :param velocity_only_state: flat velocity only state representation
    :return: flat observation extended representation
    """
    extended_state[3] = velocity_only_state[0].copy()
    extended_state[7] = velocity_only_state[1].copy()
    extended_state[5] = velocity_only_state[2].copy()
    return extended_state


def convert_f110_racecar_extended_state_to_velocity_only_representation(
    extended_state: np.ndarray,
) -> np.ndarray:
    """Convert a f110-gym RaceCar innner state representation (extended with `linear_vels_y`)
    to a velocity only RaceCar state representation for neural net ingestion.

    f110-gym RaceCar state representation:
    - state (extended):
          0  1      2           3             4            5          6            7
        [ x, y,      _     , lin_vels_x, poses_theta, ang_vels_z,     _     , lin_vels_y ]
    - state (motion):
          0  1      2           3             4            5          6            7
        [ x, y, poses_theta, lin_vels_x, lin_vels_y , ang_vels_z ]
    - state (pose):
          0  1      2           3             4            5          6            7
        [ x, y, poses_theta ]
    - state (velocity):
                          [ lin_vels_x, lin_vels_y , ang_vels_z ]

    Note that the original RaceCar state representation uses different naming convention in the
    codebase

    - state (original):
          0  1      2           3             4            5          6            7
        [ x, y, steer_angle,   vel     ,  yaw_angle ,  yaw_rate , slip_angle]

    :param extended_state: flat observation extended representation
    :return: flat velocity only state representation
    """
    velocity_only_state = extended_state[[3, 7, 5]].copy()
    return velocity_only_state


def convert_f110_racecar_velocity_only_state_to_extended_representation(
    velocity_only_state: np.ndarray,
) -> np.ndarray:
    """Convert a f110-gym RaceCar inner state representation (velocity only) to a RaceCar
        extended state representation.

        f110-gym RaceCar state representation:
        - state (extended):
              0  1      2           3             4            5          6            7
            [ x, y,      _     , lin_vels_x, poses_theta, ang_vels_z,     _     , lin_vels_y ]
        - state (motion):
              0  1      2           3             4            5          6            7
            [ x, y, poses_theta, lin_vels_x, lin_vels_y , ang_vels_z ]
        - state (pose):
              0  1      2           3             4            5          6            7
            [ x, y, poses_theta ]
        - state (velocity):
                              [ lin_vels_x, lin_vels_y , ang_vels_z ]
    `
        Note that the original RaceCar state representation uses different naming convention in the
        codebase

        - state (original):
              0  1      2           3             4            5          6            7
            [ x, y, steer_angle,   vel     ,  yaw_angle ,  yaw_rate , slip_angle]

        :param velocity_only_state: flat motion velocity only state representation
        :return: flat observation extended representation
    """
    extended_state = velocity_only_state[[0, 0, 0, 0, 0, 2, 2, 1]].copy()
    extended_state[0:3] = 0.0
    extended_state[4] = 0.0
    extended_state[6] = 0.0
    return extended_state
