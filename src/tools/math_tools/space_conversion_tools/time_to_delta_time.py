# coding=utf-8
import numpy as np


def convert_state_delta_time_to_state_time(delta_time: np.ndarray) -> np.ndarray:
    assert delta_time.ndim == 1
    time_space = np.cumsum(delta_time)
    return time_space


def convert_state_time_to_state_delta_time(time_space: np.ndarray) -> np.ndarray:
    assert time_space.ndim == 1
    delta_time = np.ediff1d(time_space, to_begin=0)
    return delta_time
