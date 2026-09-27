# coding=utf-8
from typing import Tuple, Union

import numpy as np
import torch

from tools.math_tools.space_conversion_tools.time_to_delta_time import \
    convert_state_time_to_state_delta_time
from tools.math_tools.space_conversion_tools.utils import convert_ndim_data_to_delta_ndim_data


# (Priority) ToDo: RLRP-393

def convert_state_derivatives_to_state_coordinate(
    partial_derivatives: np.ndarray,
    delta_time_space: np.ndarray,
    initiale_coordinates: Union[Tuple[float, ...], np.ndarray],
    time_space_is_delta_time: bool = True,
) -> np.ndarray:
    """
    Converts partial derivatives to state coordinates over time.
    Can handle partial_derivatives array of arbitrary dimension.

    Given `coord[0] = initiale_coordinates`, output
    `coord[t+1] = coord[t] + derivative[t+1] * dt[t+1]` for all `t` timesteps.

    :param partial_derivatives: A numpy array of partial derivatives.
    :param delta_time_space: A numpy array representing time intervals or absolute time points.
    :param initiale_coordinates: A tuple representing the initial coordinates.
    :param time_space_is_delta_time: Indicate if time_space represents delta times.
    :return: A numpy array of state coordinates computed from the partial derivatives.
    """
    assert partial_derivatives.shape[0] == delta_time_space.shape[0]

    dt_partial_derivatives = partial_derivatives.copy()
    if partial_derivatives.ndim == 1:
        dt_partial_derivatives = np.expand_dims(dt_partial_derivatives, 1)

    assert dt_partial_derivatives.shape[-1] == len(initiale_coordinates)

    delta_time = delta_time_space.copy()

    if not time_space_is_delta_time:
        delta_time = convert_state_time_to_state_delta_time(delta_time)

    dt_partial_derivatives *= np.expand_dims(delta_time, 1)

    dt_partial_derivatives[0] = initiale_coordinates
    state_coordinates = np.cumsum(dt_partial_derivatives, axis=0)

    if partial_derivatives.ndim == 1:
        state_coordinates = state_coordinates.squeeze()
    assert state_coordinates.shape == partial_derivatives.shape
    return state_coordinates


def convert_dt_state_derivatives_to_state_coordinate(
    dt_partial_derivatives: Union[np.ndarray, torch.Tensor],
    initiale_coordinates: Union[Tuple[float, ...], np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Converts partial derivatives already multiply by delta time to state coordinates over time.
    Can handle partial_derivatives array of arbitrary dimension.
    Supports both numpy arrays and torch tensors.

    Given `coord[0] = initiale_coordinates`, output
    `coord[t+1] = coord[t] + dt_derivative[t+1]` for all `t` timesteps.

    :param dt_partial_derivatives: A numpy array or torch tensor of partial derivatives.
    :param initiale_coordinates: A tuple representing the initial coordinates.
    :return: State coordinates computed from the partial derivatives (same type as input).
    """
    if isinstance(dt_partial_derivatives, torch.Tensor):
        return _convert_dt_state_derivatives_to_state_coordinate_torch(
            dt_partial_derivatives, initiale_coordinates
        )

    # RLRP-723 M4: accumulate the running sum in float64 (cast in / cast back) so a long-
    # horizon ``cumsum`` does not accrue float32 rounding error, matching the robotic
    # integrator's float64 path (F4). The output dtype is preserved.
    orig_dtype = dt_partial_derivatives.dtype
    dt_p_deriv = dt_partial_derivatives.astype(np.float64, copy=True)
    if dt_partial_derivatives.ndim == 1:
        dt_p_deriv = np.expand_dims(dt_p_deriv, 1)

    assert dt_p_deriv.shape[-1] == len(initiale_coordinates), f"Dimension mismatch: dt_partial_derivatives={dt_p_deriv.shape[-1]} != len(initiale_coordinates)={len(initiale_coordinates)}"

    dt_p_deriv[0] = initiale_coordinates
    state_coordinates = np.cumsum(dt_p_deriv, axis=0).astype(orig_dtype, copy=False)

    if dt_partial_derivatives.ndim == 1:
        state_coordinates = state_coordinates.squeeze()
    assert state_coordinates.shape == dt_partial_derivatives.shape, f"Shape mismatch: state_coordinates={state_coordinates.shape} != dt_partial_derivatives={dt_partial_derivatives.shape}"
    return state_coordinates


def _convert_dt_state_derivatives_to_state_coordinate_torch(
    dt_partial_derivatives: torch.Tensor,
    initiale_coordinates: Union[Tuple[float, ...], np.ndarray, torch.Tensor],
) -> torch.Tensor:
    """Torch-native implementation of convert_dt_state_derivatives_to_state_coordinate."""
    # RLRP-723 M4: accumulate the running sum in float64 (cast in / cast back) so a long-
    # horizon ``cumsum`` does not accrue float32 rounding error, matching the robotic
    # integrator's float64 path (F4). The output dtype is preserved.
    orig_dtype = dt_partial_derivatives.dtype
    compute_dtype = torch.float64
    dt_p_deriv = dt_partial_derivatives.to(compute_dtype)
    if dt_partial_derivatives.ndim == 1:
        dt_p_deriv = dt_p_deriv.unsqueeze(1)

    if isinstance(initiale_coordinates, (tuple, list)):
        init_coords = torch.tensor(
            initiale_coordinates, dtype=compute_dtype, device=dt_p_deriv.device
        )
    elif isinstance(initiale_coordinates, np.ndarray):
        init_coords = torch.from_numpy(initiale_coordinates).to(
            dtype=compute_dtype, device=dt_p_deriv.device
        )
    else:
        init_coords = initiale_coordinates.to(dtype=compute_dtype, device=dt_p_deriv.device)

    assert dt_p_deriv.shape[-1] == len(init_coords), (
        f"Dimension mismatch: dt_partial_derivatives={dt_p_deriv.shape[-1]} "
        f"!= len(initiale_coordinates)={len(init_coords)}"
    )

    dt_p_deriv[0] = init_coords
    state_coordinates = torch.cumsum(dt_p_deriv, dim=0).to(orig_dtype)

    if dt_partial_derivatives.ndim == 1:
        state_coordinates = state_coordinates.squeeze()
    assert state_coordinates.shape == dt_partial_derivatives.shape, (
        f"Shape mismatch: state_coordinates={state_coordinates.shape} "
        f"!= dt_partial_derivatives={dt_partial_derivatives.shape}"
    )
    return state_coordinates


def convert_state_coordinate_to_state_derivatives(
    state_coordinate: np.ndarray,
    delta_time_space: np.ndarray,
    time_space_is_delta_time: bool = True,
) -> np.ndarray:
    """
    Converts state coordinates to partial derivatives over time.
    Can handle state_coordinate array of arbitrary dimension.

    Output `derivative[t+1] = (coord[t+1] - coord[t])/dt[t+1]` for all `t` timesteps.

    :param state_coordinate: A numpy array of partial derivatives.
    :param delta_time_space: A numpy array representing time intervals or absolute time points.
    :param time_space_is_delta_time: Indicate if time_space represents delta times.
    :return: A numpy array of state coordinates computed from the partial derivatives.
    """
    assert state_coordinate.shape[0] == delta_time_space.shape[0], f"Shape mismatch: state_coordinate={state_coordinate.shape[0]} != delta_time_space={delta_time_space.shape[0]}"
    assert np.isfinite(state_coordinate).any(), f"Non-finite values found in state_coordinate: {state_coordinate}"

    delta_time = delta_time_space.copy().astype(state_coordinate.dtype)
    if not time_space_is_delta_time:
        delta_time = convert_state_time_to_state_delta_time(delta_time)

    dt_partial_derivatives = convert_ndim_data_to_delta_ndim_data(state_coordinate)

    if dt_partial_derivatives.ndim == 1:
        dt_partial_derivatives = np.expand_dims(dt_partial_derivatives, 1)

    dt_partial_derivatives[1:] /= np.expand_dims(delta_time[1:], 1)

    assert np.isfinite(dt_partial_derivatives).any(), f"Non-finite values found in dt_partial_derivatives: {dt_partial_derivatives}"
    if state_coordinate.ndim == 1:
        dt_partial_derivatives = dt_partial_derivatives.squeeze()
    assert dt_partial_derivatives.shape == state_coordinate.shape, f"Shape mismatch: dt_partial_derivatives={dt_partial_derivatives.shape} != state_coordinate={state_coordinate.shape}"
    return dt_partial_derivatives
