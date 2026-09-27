# coding=utf-8
import numpy as np


def convert_ndim_data_to_delta_ndim_data(
    ndim_data: np.ndarray,
) -> np.ndarray:
    # (NICE TO HAVE) ToDo: implement test case >> it's curently indirectly tested
    """
    Converts ndim data to ndim delta data over time.

    Output `delta_data[t+1] = data[t+1] - data[t]` for all `t` timesteps.

    :param ndim_data: A numpy array of partial derivatives.
    :return: A numpy array of state coordinates computed from the partial derivatives.
    """
    dt_ndim_data = np.empty_like(ndim_data)
    ndim_data_ = ndim_data.copy()

    if ndim_data.ndim == 1:
        dt_ndim_data = np.expand_dims(dt_ndim_data, 1)
        ndim_data_ = np.expand_dims(ndim_data_, 1)

    for each_dim in np.arange(dt_ndim_data.shape[-1]):
        dt_ndim_data[..., each_dim] = np.ediff1d(ndim_data_[..., each_dim], to_begin=0)

    if ndim_data.ndim == 1:
        dt_ndim_data = dt_ndim_data.squeeze()
    assert dt_ndim_data.shape == ndim_data.shape
    return dt_ndim_data
