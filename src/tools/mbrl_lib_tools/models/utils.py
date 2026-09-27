# coding=utf-8
import math
from typing import Tuple, Union

import numpy
import numpy as np
import torch
import warnings

from mbrl.models import Model

from tools.mbrl_lib_tools.models.prediction_statistics import SENTINEL_LOGVAR

# Near-deterministic logvar fill value; single canonical source (RLRP-761 S10.4).
# See prediction_statistics.SENTINEL_LOGVAR / ExponentialFamilyMLP for the rationale.
_LOGVAR_MIN_LIMIT = SENTINEL_LOGVAR


def compute_n_dim_obs_trajectory_probability_statistics_over_ensemble(
    n_dim_obs_pred: torch.Tensor,
    n_dim_obs_pred_logvar: torch.Tensor,
    ensemble_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the mean and standard deviation of ndim obs trajectory probabilities over an ensemble.

    Take tensors of the form `E x TrjLen x obsDim` or `TrjLen x obsDim` and return one of form
    `TrjLen x obsDim` with ensemble size `E`, trajectory length `TrjLen` and number of observation
    dimensions `obsDim`.

    :param n_dim_obs_pred: Tensor containing trajectory predictions.
    :param n_dim_obs_pred_logvar: Tensor containing the log-variance trajectory predictions.
    :param ensemble_size: The size of the ensemble.
    :return: Tuple containing the mean, standard deviation and epistemic uncertainty standard
     deviation arrays of trajectory predictions over the ensemble.
    """
    # ToDo: implement test case (curently indirectly tested)

    assert n_dim_obs_pred.shape == n_dim_obs_pred_logvar.shape

    if ensemble_size == 1 and n_dim_obs_pred.ndim == 2:
        # Add the missing ensemble dimension: B x ndim → E x B x ndim
        n_dim_obs_pred = torch.unsqueeze(n_dim_obs_pred, 0)
        n_dim_obs_pred_logvar = torch.unsqueeze(n_dim_obs_pred_logvar, 0)

        pred_uncertainty_std = torch.sqrt(n_dim_obs_pred_logvar.exp())
        return (
            n_dim_obs_pred,
            pred_uncertainty_std,
            torch.full_like(pred_uncertainty_std, fill_value=_LOGVAR_MIN_LIMIT),
        )
    else:
        pred_ensembles_mean, pred_uncertainty_std, pred_epi_uncertainty_std = (
            compute_trajectory_probability_statistics_over_ensemble(
                n_dim_obs_pred, n_dim_obs_pred_logvar
            )
        )
        return pred_ensembles_mean, pred_uncertainty_std, pred_epi_uncertainty_std


def compute_trajectory_probability_statistics_over_ensemble(
    ensemble_pred: torch.Tensor,
    ensemble_pred_logvar: torch.Tensor,
    to_tensor: bool = True,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Computes the trajectory expectation statistics over an ensemble of predictions.

    It's the user responsability to make sure their first dimension is the ensemble.
     e.g.,: `Ensemble x Batch/trj-len x InputDimension` or `Ensemble x Batch/trj-len`

    :param ensemble_pred: A tensor containing ensemble predictions of shape (ensemble_size, ...,
        output_dim).
    :param ensemble_pred_logvar: A tensor containing the log variance of ensemble predictions.
    :param to_tensor: A boolean indicating whether to return the result as tensors (True) or
        numpy arrays (False). Defaults to True (torch-first).
    :return: A tuple containing the ensemble prediction means, uncertainty standard deviation (std)
        and epistemic uncertainty std, either as tensors or numpy arrays.
    """
    # .... Pre-condition ..........................................................................
    assert ensemble_pred.ndim >= 2
    assert ensemble_pred.shape == ensemble_pred_logvar.shape

    # .... Compute standar deviation over ensemble dimensions .....................................
    pred_ensembles_mean = ensemble_pred.mean(dim=0)
    ensemble_pred_var = ensemble_pred_logvar.exp()
    pred_var_aleatoric = ensemble_pred_var.mean(dim=0)
    if ensemble_pred.size(0) == 1:
        # Case only one ensemble
        pred_var_epistemic = torch.zeros_like(pred_ensembles_mean)
    else:
        pred_var_epistemic = ensemble_pred.var(dim=0)
    pred_epi_uncertainty_std = torch.sqrt(pred_var_epistemic)
    pred_uncertainty_std = torch.sqrt(pred_var_aleatoric) + pred_epi_uncertainty_std

    # .... Setup return format ....................................................................
    if to_tensor:
        return pred_ensembles_mean, pred_uncertainty_std, pred_epi_uncertainty_std
    else:
        return (
            pred_ensembles_mean.cpu().numpy(),
            pred_uncertainty_std.cpu().numpy(),
            pred_epi_uncertainty_std.cpu().numpy(),
        )


def is_model_ensemble(model: Model) -> bool:
    try:
        is_ensemble_model_ = model.num_members > 1
    except AttributeError:
        is_ensemble_model_ = False
    return is_ensemble_model_


def is_probabilistic_model(model: Model) -> bool:
    try:
        is_probabilistic_model_ = not model.deterministic
    except AttributeError:
        is_probabilistic_model_ = False
    return is_probabilistic_model_
