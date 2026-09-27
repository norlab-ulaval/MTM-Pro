# coding=utf-8
from typing import List, Sequence, Union

import numpy as np
import torch

from tools.math_tools.ndarray_tools.ndarray_checker import check_ndarray_list_composante_shapes_match


def min_max_normalization(
    obs_feature: np.ndarray, scale_min: float = 0.5, scale_max: float = 1.0
) -> np.ndarray:
    # (NICE TO HAVE) ToDo: implement test case
    """
    The function performs min-max normalization on an array of observation features.
    See https://en.wikipedia.org/wiki/Feature_scaling for explanation.

    Return `scale_max` for case where obs_feature.size==1

    :param obs_feature: A numpy array of observation features to be normalized.
    :param scale_min: The minimum value of the desired scale range.
    :param scale_max: The maximum value of the desired scale range.
    :return: A numpy array of the normalized observation features.
    """
    EPS = 1e-08

    if obs_feature.size > 1:
        numerator = (obs_feature - np.min(obs_feature )) * (scale_max - scale_min)
        denominator = np.max(obs_feature) - np.min(obs_feature)
        scaled_feature_weight = scale_min + (numerator / (denominator + EPS))
    else:
        scaled_feature_weight = np.array(
            [
                scale_max,
            ]
        )
    return scaled_feature_weight


def normalize_predictions(
    pred: Sequence[Union[np.ndarray, torch.Tensor]],
) -> List[Union[np.ndarray, torch.Tensor]]:
    """
    Normalize a sequence of predictions using min-max feature scaling.

    Supports both numpy ndarrays and torch tensors. All elements in ``pred``
    must be of the same type.

    :param pred: A list/tuple of numpy ndarrays or torch tensors, where each
        element represents a set of predictions.
    :return: A list of normalized predictions of the same type as the input.
    """
    # (CRITICAL) ToDo: (RLRP-390) better explain what the fct do as its not clear how
    #   `compute_trajectories_prediction_error` use it i.e. normalize given the target...
    # .... Pre-condition ..........................................................................
    assert isinstance(pred, Sequence)
    assert isinstance(pred[0], (np.ndarray, torch.Tensor))
    check_ndarray_list_composante_shapes_match(pred)

    _is_torch = isinstance(pred[0], torch.Tensor)

    # .... Normalize ..............................................................................
    if _is_torch:
        pred_normalization = torch.cat(list(pred), dim=0)
        if pred_normalization.ndim == 1:
            pred_normalization = pred_normalization.unsqueeze(-1)
        pred_normalization_min = pred_normalization.min(dim=0).values
        pred_normalization_min_max = (
            pred_normalization.max(dim=0).values - pred_normalization_min
        )
    else:
        pred_normalization = np.concatenate(pred, axis=0)
        if pred_normalization.ndim == 1:
            pred_normalization = np.expand_dims(pred_normalization, 1)
        pred_normalization_min = pred_normalization.min(axis=0)
        pred_normalization_min_max = (
            pred_normalization.max(axis=0) - pred_normalization_min
        )

    pred = list(pred)
    for pred_idx in range(len(pred)):
        pred[pred_idx] = (
                                 pred[pred_idx] - pred_normalization_min
        ) / pred_normalization_min_max
    return pred
