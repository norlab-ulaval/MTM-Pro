# coding=utf-8
import numpy as np
import torch


def compute_multistep_model_in_size(
    singlestep_obs_len: int, singlestep_act_len: int, multistep_len: int
) -> int:
    """
    Compute the size of the input for a multistep model based on the provided observation
    length, action length, and the number of multisteps.

    :param singlestep_obs_len: Length of the observation vector at a single step.
    :param singlestep_act_len: Length of the action vector at a single step.
    :param multistep_len: Number of steps in the multistep sequence.
    :return: The total input size for the multistep model.
    """
    return singlestep_obs_len * multistep_len + singlestep_act_len * multistep_len


def compute_multistep_model_out_size(
    singlestep_obs_len: int,
    singlestep_act_len: int,
    multistep_len: int,
    learned_reward: bool = False,
) -> int:
    """
    This function calculates the size of the output for a model that processes multiple
    steps of observations and actions. The output size is determined based on the length
    of single-step observations, single-step actions, the number of steps (multistep length),
    and whether a learned reward is included in the model.

    :param singlestep_obs_len: Length of single-step observation data.
    :param singlestep_act_len: Length of single-step action data.
    :param multistep_len: Number of steps in the multi-step sequence.
    :param learned_reward: Whether the model includes a learned reward in the output.
    :return: The computed size of the multi-step model output.
    """
    return (
        singlestep_obs_len * multistep_len
        + (singlestep_act_len * (multistep_len - 1))
        + int(learned_reward)
    )


def convert_compose_next_obs_multistep_obs_to_next_single_step_obs(
    composed_obs: np.ndarray, compose_next_multistep_obs_to_next_singlestep_obs_slice: slice
) -> np.ndarray:
    """Extract the next single step observation from a multistep observation with prediction
    offset.
    Use by MultistepObservationToSinglestepObservationAdapter to define a base fct signature.

    Pre: Assume the multistep_obs is a flatten array

    :param composed_obs: A flattened multistep observation
    :param compose_next_multistep_obs_to_next_singlestep_obs_slice: the next observation
    indexes
    :return: the next single step observation
    """
    return composed_obs[compose_next_multistep_obs_to_next_singlestep_obs_slice]


def convert_multistep_obs_to_latest_single_step_obs(
    multistep_obs: np.ndarray, single_step_obs_len: int
) -> np.ndarray:
    """Extract the latest single step observation from a multistep observation ordered with the
    latest on the right side of the  array .

    Pre: Assume the multistep_obs is a flatten array

    :param multistep_obs: a flattened multistep observation
    :param single_step_obs_len: the length of each single step observations
    :return: the latest single step observation
    """
    return multistep_obs[-single_step_obs_len:]

