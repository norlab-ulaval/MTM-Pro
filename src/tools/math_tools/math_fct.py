# coding=utf-8
import math
from typing import Union

import numpy as np


# ==== Trigonometry ===============================================================================
def csc_theta(theta):
    return 1 / math.sin(theta)


def sec_theta(theta):
    return 1 / math.cos(theta)


def numpy_softplus(
    noise_dynamic_size: Union[float, np.ndarray], beta: float = 1.0
) -> Union[float, np.ndarray]:
    """
    Computes the softplus function using numpy. Smaller beta makes the ramp around 0 smoother. 0 < Beta

    :param noise_dynamic_size: Input value(s), either a float or a numpy array.
    :param beta: Smoothing parameter, controls how sharp the function behaves,
        with a default value of 1.0.
    :return: The computed softplus value, in the same type (float or numpy array)
        as the input value(s).
    """
    return 1 / beta * np.log(1 + np.exp(beta * noise_dynamic_size))
