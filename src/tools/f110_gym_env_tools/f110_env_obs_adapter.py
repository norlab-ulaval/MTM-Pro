# coding=utf-8

import numpy as np


def unpack_f110_env_multiagent_obs_to_sau_representation(obs: dict, ego_idx: int) -> dict:
    """Convert a f110-gym multiagent observation to a Single Agent Unpacked (sau) representation.

    Except for the numpy ndarray in 'scans', all other observation values are numbers.

    :param obs: a standard multiagent f110-gym observation
    :param ego_idx: the index of the agent to unpack
    :return: single agent (unpacked)
    """
    obs["poses_x"] = obs["poses_x"][ego_idx]
    obs["poses_y"] = obs["poses_y"][ego_idx]
    obs["poses_theta"] = obs["poses_theta"][ego_idx]
    obs["linear_vels_x"] = obs["linear_vels_x"][ego_idx]
    obs["linear_vels_y"] = np.float64(obs["linear_vels_y"][ego_idx])
    obs["ego_idx"] = ego_idx
    obs["ang_vels_z"] = obs["ang_vels_z"][ego_idx]
    obs["collisions"] = obs["collisions"][ego_idx]
    obs["lap_counts"] = obs["lap_counts"][ego_idx]
    obs["lap_times"] = obs["lap_times"][ego_idx]
    obs["scans"] = obs["scans"][ego_idx]

    return obs


# ::: Convert f110-gym env SAU observation ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def convert_f110_env_sau_obs_to_flattened_representation(
    obs: dict, dtype=np.float64
) -> np.ndarray:
    """Convert a f110-gym single agent unpacked (sau) dict observation to a flattened
    representation

    The returned flattened observation ndarray values are ordered arbitrarly with the following
    index:
            - obs[0] -> 'poses_x'
            - obs[1] -> 'poses_y'
            - obs[2] -> 'poses_theta'
            - obs[3] -> 'linear_vels_x'
            - obs[4] -> 'linear_vels_y'
            - obs[5] -> 'ang_vels_z'
            - obs[6] -> 'ego_idx'
            - obs[7] -> 'collisions'
            - obs[8] -> 'lap_counts'
            - obs[9] -> 'lap_times'
            - obs[10:1090] -> 'scans'

    :param obs: a single agent unpacked (sau) f110-gym dict observation
    :param dtype:
    :return: the observation in a flattened numpy array
    """
    motion_and_sim_obs = np.array(
        [
            obs["poses_x"],
            obs["poses_y"],
            obs["poses_theta"],
            obs["linear_vels_x"],
            obs["linear_vels_y"],
            obs["ang_vels_z"],
            obs["ego_idx"],
            obs["collisions"],
            obs["lap_counts"],
            obs["lap_times"],
        ],
        dtype=dtype,
    )

    motion_sim_and_scans_obs = np.concatenate((motion_and_sim_obs, obs["scans"]), dtype=dtype)

    return motion_sim_and_scans_obs


def convert_f110_env_sau_obs_to_motion_only_flattened_representation(
    obs: dict,
    dtype=np.float64,
) -> np.ndarray:
    """Convert a f110-gym single agent unpacked (sau) dict observation to a motion only
    flattened representation

    :param obs: a single agent unpacked (sau) f110-gym dict observation
    :param dtype:
    :return: the observation in a flattened numpy array
    """
    motion_only_obs = np.array(
        [
            obs["poses_x"],
            obs["poses_y"],
            obs["poses_theta"],
            obs["linear_vels_x"],
            obs["linear_vels_y"],
            obs["ang_vels_z"],
        ],
        dtype=dtype,
    )

    return motion_only_obs


# ::: Convert f110-gym env flatten observation ::::::::::::::::::::::::::::::::::::::::::::::::::::
def convert_f110_env_flattened_obs_to_sau_representation(obs: np.ndarray) -> dict:
    """Convert a f110-gym flattened observation to a single agent unpacked (sau) dict
    representation

    Except for the numpy ndarray in 'scans', all other observation values are numbers.

    Argument `obs` need to be a motion dynamic only representation with
            - obs[0] <- value for 'poses_x'
            - obs[1] <- value for 'poses_y'
            - obs[2] <- value for 'poses_theta'
            - obs[3] <- value for 'linear_vels_x'
            - obs[4] <- value for 'linear_vels_y'
            - obs[5] <- value for 'ang_vels_z'
            - obs[6] <- value for 'ego_idx'
            - obs[7] <- value for 'collisions'
            - obs[8] <- value for 'lap_counts'
            - obs[9] <- value for 'lap_times'
            - obs[10:1090] <- value for 'scans'

    :param obs: a flattened f110-gym observation
    :return: a f110-gym single agent unpacked (sau) representation
    """
    sau_obs = {
        "poses_x": obs[0],
        "poses_y": obs[1],
        "poses_theta": obs[2],
        "linear_vels_x": obs[3],
        "linear_vels_y": obs[4],
        "ang_vels_z": obs[5],
        "ego_idx": obs[6],
        "collisions": obs[7],
        "lap_counts": obs[8],
        "lap_times": obs[9],
        "scans": obs[10:],
    }

    return sau_obs


def convert_f110_env_flattened_obs_to_multiagent_representation(obs: np.ndarray) -> dict:
    """Convert a f110-gym flattened observation to the original f110-gym multiagent dict
    representation

    :param obs: a flattened f110-gym observation
    :return: a observation in the original f110-gym multiagent representation
    """
    original_f110gym_dict_obs = {
        "poses_x": [obs[0]],
        "poses_y": [obs[1]],
        "poses_theta": [obs[2]],
        "linear_vels_x": [obs[3]],
        "linear_vels_y": [obs[4]],
        "ang_vels_z": [obs[5]],
        "ego_idx": obs[6],
        "collisions": np.array([obs[7]]),
        "lap_counts": np.array([obs[8]]),
        "lap_times": np.array([obs[9]]),
        "scans": [obs[10:]],
    }

    return original_f110gym_dict_obs


def convert_f110_env_flattened_motion_only_obs_to_sau_representation(obs: np.ndarray) -> dict:
    """Convert a f110-gym flattened motion only observation to a single agent unpacked (sau)
    dict representation

    Argument `obs` need to be a motion dynamic only representation with
      - obs[0] <- value for 'poses_x'
      - obs[1] <- value for 'poses_y'
      - obs[2] <- value for 'poses_theta'
      - obs[3] <- value for 'linear_vels_x'
      - obs[4] <- value for 'linear_vels_y'
      - obs[5] <- value for 'ang_vels_z'

    :param obs: a flattened motion dynamic only f110-gym observation
    :return: a f110-gym single agent unpacked (sau) motion dynamic only representation
    """
    sau_motion__only_obs = {
        "poses_x": obs[0],
        "poses_y": obs[1],
        "poses_theta": obs[2],
        "linear_vels_x": obs[3],
        "linear_vels_y": obs[4],
        "ang_vels_z": obs[5],
    }

    return sau_motion__only_obs

def convert_f110_env_flattened_obs_to_flattened_motion_only_obs(obs: np.ndarray) -> np.ndarray:
    """Convert a f110-gym flattened observation to a motion dynamic only observation
    representation

    Argument `obs` need to be a flattened observation representation with the first six ndarray
    index containing the following values:
      - obs[0] <- value for 'poses_x'
      - obs[1] <- value for 'poses_y'
      - obs[2] <- value for 'poses_theta'
      - obs[3] <- value for 'linear_vels_x'
      - obs[4] <- value for 'linear_vels_y'
      - obs[5] <- value for 'ang_vels_z'

    :param obs: a flattened f110-gym observation
    :return: a flattened motion dynamic only f110-gym observation
    """
    motion_only_obs = obs[:6]
    return motion_only_obs


def convert_f110_env_flattened_obs_to_flattened_pose_only_obs(obs: np.ndarray) -> np.ndarray:
    """Convert a f110-gym flattened observation to a motion dynamic pose only observation
    representation

    Argument `obs` need to be a flattened observation representation with the first three ndarray
    index containing the following values:
      - obs[0] <- value for 'poses_x'
      - obs[1] <- value for 'poses_y'
      - obs[2] <- value for 'poses_theta'

    :param obs: a flattened f110-gym observation
    :return: a flattened motion dynamic only f110-gym observation
    """
    pose_only_obs = obs[:3]
    return pose_only_obs


def convert_f110_env_flattened_obs_to_flattened_velocity_only_obs(obs: np.ndarray) -> np.ndarray:
    """Convert a f110-gym flattened observation to a motion dynamic velocity only observation
    representation

    Argument `obs` need to be a flattened observation representation with index three to six
     containing the following values:
      - obs[3] <- value for 'linear_vels_x'
      - obs[4] <- value for 'linear_vels_y'
      - obs[5] <- value for 'ang_vels_z'

    :param obs: a flattened f110-gym observation
    :return: a flattened motion dynamic only f110-gym observation
    """
    vel_only_obs = obs[3:6]
    return vel_only_obs


def convert_f110_env_flattened_obs_to_flattened_velocity_obs_with_delta_time(obs: np.ndarray) -> np.ndarray:
    """Convert a f110-gym flattened observation to a motion dynamic velocity only observation
    representation with delta time observation.

    Argument `obs` need to be a flattened observation representation with index three to six
     containing the following values:
      - obs[3] <- value for 'linear_vels_x'
      - obs[4] <- value for 'linear_vels_y'
      - obs[5] <- value for 'ang_vels_z'
      - obs[6] <- value for delta time

    :param obs: a flattened f110-gym observation with time dimension
    :return: a flattened motion dynamic only f110-gym observation
    """
    vel_only_with_delta_time_obs = obs[3:7]
    return vel_only_with_delta_time_obs


