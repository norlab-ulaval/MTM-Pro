# coding=utf-8
import warnings
from typing import Callable, Tuple, Union

import numpy as np
import torch
from torch import nn as nn

from tools.model_adapter_tools.ms_to_ss_observation_adapter import (
    MultistepObservationToSinglestepObservationAdapter,
)


class EnsembleDeployHead(nn.Module):
    """Multistep output adapter layer for ensemble models."""

    def __init__(
        self,
        adapter: Union[Callable, MultistepObservationToSinglestepObservationAdapter],
    ):
        super().__init__()
        self._model_to_env_next_obs_adapter = adapter

    # (NICE TO HAVE) ToDo: to consider once the regression problem is fixe (ref task RLRP-510)
    #   but its purely cosmetic
    # def train(self, mode: bool = True) -> "EnsembleDeployHead":
    #     """This module has no trainable parameters; always stay in eval mode."""
    #     return super().train(False)

    def forward(self, x):
        return self._model_to_env_next_obs_adapter(x)


class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# ==== Multistep utilities ========================================================================
def timestep_first_multistep_dim_unflaten_array(
    x: Union[torch.Tensor, np.ndarray],
    singlestep_obs_len: int,
    singlestep_act_len: int,
    sequence_len: int,
    enable_last_action_padding=False,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Reshapes a multistep array (as received by a MS-replaybuffer) into its original
    multi-dimensional structure.

    This function takes an array and reshapes it to match the specified observation,
    action, and sequence lengths. It assumes the last axis of the provided array represents
    a flattened combination

        (..., O[1:Do]_1 + ... +  O[1:Do]_MS + A[1:Da]_1 + ... +  A[1:Da]_MS ) --> (..., MS, O+A)

    with O[1:Do]=observation, A[1:Da]=action dimension len and MS=multistep len i.e., features dim
    are chunked by timesteps from 1 to MS.

        Tensor shape: (... x MS x F)
                             │    └─ Feature dim: obs + act dimensions
                             └────── Multi-step: prediction horizon

    :param x: Input array to be reshaped. Can be either a PyTorch Tensor or a NumPy ndarray.
    :param singlestep_obs_len: Length of the single step observation to be used for reshaping.
    :param singlestep_act_len: Length of the single step action to be used for reshaping.
    :param sequence_len: Length of the sequence to be used for reshaping, i.e., the last target dimension.
     For a composed MS model INPUT this is ``history_len`` (``H``); for a composed MS->MS model
     OUTPUT / next-obs target it is the output window ``W = MultiStepMLP.output_window_len``
     (``== H`` for every legacy family, ``max(H, F)`` for the MS->MS forecast family, RLRP-824).
    :param enable_last_action_padding: Pad action dimension so that they have the same
     trajectory length as observations. Pad a single timestep action dimensions with zeros.
    :return: Reshaped array with trajectory timesteps on the last dimension.
    """
    original_shape = x.shape

    if enable_last_action_padding and singlestep_act_len > 0:
        expected_ms_len_wo_act = (
            singlestep_obs_len + singlestep_act_len
        ) * sequence_len - singlestep_act_len
        assert (
            original_shape[-1] == expected_ms_len_wo_act
        ), f"{original_shape[-1]=} != {expected_ms_len_wo_act=}"
    else:
        expected_ms_len = (singlestep_obs_len + singlestep_act_len) * sequence_len
        assert (
            original_shape[-1]
            == (singlestep_obs_len + singlestep_act_len) * sequence_len
        ), f"{original_shape[-1]=} != {expected_ms_len=}"

    obs = x[..., : singlestep_obs_len * sequence_len]
    obs = obs.reshape(-1, sequence_len, singlestep_obs_len)

    if singlestep_act_len > 0:
        if enable_last_action_padding:
            act = x[..., -singlestep_act_len * (sequence_len - 1) :]
            if isinstance(x, np.ndarray):
                padded_act = np.zeros_like(act[..., -singlestep_act_len:])
                # if singlestep_act_len == 1:
                #     padded_act = np.expand_dims(padded_act, axis=-1)
                act = np.concatenate([act, padded_act], axis=-1)
            else:
                padded_act = torch.zeros_like(act[..., -singlestep_act_len:])
                # if singlestep_act_len == 1:
                #     padded_act = padded_act.unsqueeze(-1)
                act = torch.concatenate([act, padded_act], dim=-1)
        else:
            act = x[..., -singlestep_act_len * sequence_len :]

        act = act.reshape(-1, sequence_len, singlestep_act_len)

        if isinstance(x, np.ndarray):
            x1 = np.concatenate([obs, act], axis=-1)
        else:
            x1 = torch.concatenate([obs, act], dim=-1)
    else:
        x1 = obs

    x1 = x1.reshape(
        *original_shape[:-1], sequence_len, singlestep_obs_len + singlestep_act_len
    )
    return x1


def revert_timestep_first_multistep_dim_unflaten_array(
    x: Union[torch.Tensor, np.ndarray],
    singlestep_obs_len: int = None,
    singlestep_act_len: int = None,
    remove_last_action_padding: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Reverts the operation of `timestep_first_multistep_dim_unflaten_array`.

    Transforms a reshaped multistep array back to its flattened form:
    (..., MS, O+A) --> (..., O[1:Do]_1 + ... + O[1:Do]_MS + A[1:Da]_1 + ... + A[1:Da]_MS)

    :param x: Input array with shape (..., feature_dim, sequence_len) where feature_dim
              is singlestep_obs_len + singlestep_act_len
    :param singlestep_obs_len: Length of single step observation (required)
    :param singlestep_act_len: Length of single step action (required)
    :param remove_last_action_padding: If True, removes the padded action timestep
                                         to match the original format where actions have
                                         one fewer timestep than observations
    :return: Flattened array matching the original input format of
             timestep_first_multistep_dim_unflaten_array
    """
    if singlestep_obs_len is None or singlestep_act_len is None:
        raise ValueError("singlestep_obs_len and singlestep_act_len must be provided")

    # Split observations and actions
    obs_part = x[..., :singlestep_obs_len]  # Shape: (..., sequence_len, obs_len)
    act_part = x[..., singlestep_obs_len:]  # Shape: (..., sequence_len, act_len)

    if remove_last_action_padding:
        # Remove the last timestep from actions (the padded zero timestep)
        act_part = act_part[..., :-1, :]  # Shape: (..., sequence_len-1, act_len)

    # Flatten each part and concatenate
    if isinstance(x, torch.Tensor):
        obs_flattened = obs_part.flatten(start_dim=-2)
        act_flattened = act_part.flatten(start_dim=-2)
        return torch.concatenate([obs_flattened, act_flattened], dim=-1)
    else:
        obs_flattened = obs_part.reshape(*obs_part.shape[:-2], -1)
        act_flattened = act_part.reshape(*act_part.shape[:-2], -1)
        return np.concatenate([obs_flattened, act_flattened], axis=-1)


def dim_first_homogenous_multistep_dim_unflaten_array(
    x: Union[torch.Tensor, np.ndarray],
    singlestep_obs_len: int,
    singlestep_act_len: int,
    sequence_len: int,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Reshapes a multistep array into its original multi-dimensional structure.

    This function takes an array and reshapes it to match the specified observation,
    action, and sequence lengths. It assumes the last axis of the provided array represents
    a flattened combination

        (..., F[1] x MS + F[2] x MS + ... + F[D] x MS) --> (..., MS, F)

    with F[1:D]=observation+action dimension len and MS=multistep len i.e., features are chunked by
    trajectories from dimension 1 to D.

    :param x: Input array to be reshaped. Can be either a PyTorch Tensor or a NumPy ndarray.
    :param singlestep_obs_len: Length of the single step observation to be used for reshaping.
    :param singlestep_act_len: Length of the single step action to be used for reshaping.
    :param sequence_len: Length of the sequence to be used for reshaping, i.e., the target last dimension.
    :return: Reshaped array with trajectory timesteps on the last dimension.
    """
    if x.ndim > 2:
        # Assume x is batched (E, B, MS*F)
        raise NotImplementedError(
            "# (CRITICAL) ToDo: Unittest for batched input casses (E, B, MS*F), current ones only cover casses (MS*F)"
        )

    original_shape = x.shape
    return x.reshape(
        (*original_shape[:-1], singlestep_obs_len + singlestep_act_len, sequence_len)
    ).swapaxes(-2, -1)


def extract_multistep_features_from_array(
    x: Union[np.ndarray, torch.Tensor],
    start_idx: int,
    stop_idx: int,
    single_step_features_size: int,
    vectorized=True,
) -> Union[np.ndarray, torch.Tensor]:
    # (CRITICAL) ToDo: implement test case
    #       (see `_extract_obs_features_from_multistep_composed_model_output` test)
    """Utility for extracting features from an array of multistep model input or output

        (..., F[1] x MS + F[2] x MS + ... + F[D] x MS) --> (..., F, MS)

    with F[1:D]=observation+action dimension len and MS=multistep len.

    :param x: the multistep array
    :param start_idx: the feature start idx
    :param stop_idx: the feature end idx
    :param single_step_features_size: the feature dimension size
    :param vectorized: use vectorize implementation
    :return: A features array of shape (trajectory_len, feature_size, multistep_lens)
    """
    assert x.ndim >= 2, f"{x.ndim=} != 2"

    if isinstance(x, torch.Tensor):
        feature = _extract_multistep_features_from_tensor(
            x, start_idx, stop_idx, single_step_features_size, vectorized
        )
    else:
        feature = _extract_multistep_features_from_ndarray(
            x, start_idx, stop_idx, single_step_features_size, vectorized
        )

    # .... Reshape feature multistep data to third dimension ......................................
    original_shape = x.shape
    # (CRITICAL) ToDo: validate element ordering
    feature = feature.reshape(*original_shape[:-1], single_step_features_size, -1)
    return feature


def _extract_multistep_features_from_ndarray(
    x: np.ndarray,
    start_idx: int,
    stop_idx: int,
    single_step_features_size: int,
    vectorized: bool,
) -> np.ndarray:
    if x.shape[-1] > single_step_features_size:
        if vectorized:
            x = x[..., start_idx:stop_idx]
            x = x.reshape(*x.shape[:-1], -1, single_step_features_size)
            x = np.swapaxes(x, -2, -1)
        else:
            feature = []
            for each_obs_idx in range(start_idx, stop_idx, single_step_features_size):
                feature.append(
                    x[..., each_obs_idx : each_obs_idx + single_step_features_size]
                )
            x = np.stack(feature, axis=-1)

    return x


def _extract_multistep_features_from_tensor(
    x: torch.Tensor,
    start_idx: int,
    stop_idx: int,
    single_step_features_size: int,
    vectorized: bool,
) -> torch.Tensor:
    if x.shape[-1] > single_step_features_size:
        if vectorized:
            x = x[..., start_idx:stop_idx]
            left_side_shape = x.size()[:-1]
            x = x.reshape(*left_side_shape, -1, single_step_features_size)
            x = torch.swapaxes(x, -2, -1)
        else:
            x = _extract_multistep_features_from_tensor_sequential(
                single_step_features_size, start_idx, stop_idx, x
            )

    return x


@torch.jit.script
def _extract_multistep_features_from_tensor_sequential(
    single_step_features_size: int, start_idx: int, stop_idx: int, x: torch.Tensor
) -> torch.Tensor:
    # See https://github.com/vahidk/EffectivePyTorch?tab=readme-ov-file#optimizing-runtime-with
    # -torchscript last code snipet.
    feature = []
    for each_obs_idx in range(start_idx, stop_idx, single_step_features_size):
        feature.append(x[..., each_obs_idx : each_obs_idx + single_step_features_size])
    x = torch.stack(feature, dim=-1)
    return x


# ==== General ====================================================================================


def _case_sampled_with_bootstrap_iterator_false(samples: torch.Tensor) -> bool:
    """Case sample from SequenceTransitionIterator with _bootstrap_iter=False"""
    return samples.ndim == 3


# ==== Batch sequence utilities ===================================================================
def reshape_sequence_dim_into_batch_dim(
    x: torch.Tensor, ensemble_size: int
) -> Tuple[torch.Tensor, int]:
    """
    Reshapes the sequence dimension of a tensor into the batch dimension while preserving
    the feature size, and returns the reshaped tensor alongside the sequence length i.e.:

        4 dimension input: (E, B, S, Idim) --> (E, B * S, Odim)
        3 dimension input: (B, S, Idim) --> (B * S, Odim)

    with E, B, S, Idim and Odim being ensemble size, batch size, sequence len, input dimension
    size and output dimension size respectively.

    :param x: Input tensor whose dimensions will be reshaped. The tensor can have either 3 or
              4 dimensions, where the last two dimensions correspond to sequence length and
              feature size, respectively.
    :param ensemble_size: Integer specifying the size of the ensemble, which determines how
                          the input tensor's 4 dimensions are split during reshaping.
    :return: A tuple containing the reshaped tensor and an integer representing the sequence
             length of the batch (based on the second-to-last dimension of the input tensor).
    """
    batch_sequence_len = x.size(-2)
    feature_size = x.size(-1)
    if x.ndim == 4:
        return x.reshape(ensemble_size, -1, feature_size), batch_sequence_len
    elif x.ndim == 3:
        return x.reshape(-1, feature_size), batch_sequence_len
    else:
        raise ValueError(_received_sequence_batch_error_msg(x))


def revert_reshape_sequence_dim_into_batch_dim(
    x: torch.Tensor, sequence_len: int
) -> torch.Tensor:
    """
    Revert the operation of method `reshape_sequence_dim_into_batch_dim`, i.e.,

        3 dimension input: (E, B * S, Odim) --> (E, B, S, Idim)
        2 dimension input: (B * S, Odim) --> (B, S, Idim)

    with E, B, S, Idim and Odim being ensemble size, batch size, sequence len, input dimension
    size and output dimension size respectively.

    :param x: Input tensor whose dimensions will be reshaped. The tensor can have either 2 or
              3 dimensions, where the last two dimensions correspond to batch X sequence length and
              feature size, respectively.
    :param sequence_len: The length of the sequence dimension into which the tensor should be reshaped.
    :return: A reshaped tensor of dimensions [batch_size, -1, sequence_len, feature_size]
             for 3D input or [batch_size, sequence_len, feature_size] for 2D input.
    """
    assert x.ndim == 2 or x.ndim == 3
    feature_size = x.size(-1)
    if x.ndim == 3:
        return x.reshape(x.size(0), -1, sequence_len, feature_size)
    else:
        return x.reshape(-1, sequence_len, feature_size)


def _received_sequence_batch_error_msg(x: Union[np.ndarray, torch.Tensor]) -> str:
    return (
        "Expect to receive sample of dimension 4 (Ensemble x Batch x Sequence x Idim) or "
        "3 (Batch x Sequence x iDim) when initialized with "
        f"`receive_sequence_batch=True` current sample have {x.ndim=}!"
    )


def _batch_sequence_insert_at_next_timestep_indice(
    x: torch.Tensor, sub_x: torch.Tensor, timestep: int
) -> torch.Tensor:
    # sub_x = sub_x.squeeze()
    if x.ndim == 4:
        x[:, :, timestep + 1, ...] = sub_x
    elif x.ndim == 3:
        x[:, timestep + 1, ...] = sub_x
    elif x.ndim == 2:
        x[timestep + 1, ...] = sub_x
    return x


def _batch_sequence_timestep_view(x: torch.Tensor, timestep: int) -> torch.Tensor:
    if x.ndim == 4:
        return x[:, :, timestep, ...].clone()
    elif x.ndim == 3:
        return x[:, timestep, ...].clone()
    else:
        raise ValueError(_received_sequence_batch_error_msg(x))
