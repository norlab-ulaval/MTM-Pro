# coding=utf-8
from typing import List, Literal, Optional, Sequence, Tuple, Union

import hydra
import mbrl.models
import numpy as np
import omegaconf
import torch

from tools.math_tools.ndarray_tools.ndarray_checker import (
    check_ndarray_list_composante_shapes_match,
)
from .rollout_and_stats_collection import (
    multistep_model_testtime_rollout_and_collect_pred_stats,
    singlestep_model_testtime_rollout_and_collect_pred_stats,
)

from tools.feature_handling_tools.env_handlers import (
    attach_feature_handler_to_deploy,
)
from tools.r2s_motion_model_container_tools.container import R2SMotionModelContainer
from tools.console_tools.format import ConsoleFormat
from tools.console_tools.message import consol_msg_universal_one_liner
from tools.math_tools.normalization import normalize_predictions
from tools.math_tools.space_conversion_tools.coordinate_to_velocity import (
    convert_dt_state_derivatives_to_state_coordinate,
)
from tools.mbrl_lib_tools.models.one_dim_tr_model_v2 import (
    OneDTransitionRewardModelV2,
    assert_is_OneDTransitionRewardModelV2,
)
from tools.mbrl_lib_tools.models.utils import (
    compute_n_dim_obs_trajectory_probability_statistics_over_ensemble,
)
from trajectory_container_tools.dataclasses.erll_trajectory_dataclass import (
    TestMotionTrajectoryDataclass,
    TestTrajectoryDataclass,
)
from algorithm.experience_replay_learning_loop.core.data_classes import PredictionMetric


def model_testtime_rollout_and_compute_prediction_metric(
    cfg: omegaconf.DictConfig,
    dynamic_model: Union[OneDTransitionRewardModelV2, mbrl.models.Model],
    motion_model_container: Optional[R2SMotionModelContainer],
    torch_rng: torch.Generator,
    test_trajectories: Union[
        list[TestTrajectoryDataclass], list[TestMotionTrajectoryDataclass]
    ],
    compounded_predictions_score: bool,
    next_state_deterministic_selection: bool,
    next_state_sampling_size: int = 1,
    ground_truth_feed_warmup_steps: int = 0,
    tutor_and_release=False,
    show_debug_info: bool = False,
) -> PredictionMetric:
    """
    Rollouts a dynamic model over a trajectory from a testing environment, computes predicted
    statistics, and optionally displays debug information.

    This function takes a dynamic model and a testing environment, performs a rollout over the
    given trajectory, and computes trajectory prediction metrics, including mean predictions,
    standard deviations, epistemic uncertainty, and mean absolute error. The function supports
    single-step and multi-step rollouts depending on whether a `motion_model_container`
    is provided.

    Output prediction mean, aleatoric+epistemic model uncertainty std, epistemic model
    uncertainty std, mean-absolute-error. Output ndarray's are of the form `TrjLen x obsDim`
    with trajectory length `TrjLen` and number of observation dimension `obsDim`. Note that the
    mean-absolute-error is `(TrjLen - 1) x obsDim`.


    :param cfg: Configuration object containing environment and model-related settings.
    :param dynamic_model: The dynamic model used for predictions.
    :param motion_model_container: Optional container object for handling multi-step rollouts.
    :param torch_rng: Random number generator used during model rollouts.
    :param test_trajectories: Test environment, either a list of trajectory data class or a list of
        supported gymnasium wrapper.
    :param compounded_predictions_score: Flag to enable compounded predictions scoring during
        rollouts.
    :param next_state_deterministic_selection: Used for determining next state sampling strategy.
    :param next_state_sampling_size: Nb of sample to draw if next_state_deterministic_selection=False
    :param ground_truth_feed_warmup_steps: Nb of step before switching to compounded predictions
    :param tutor_and_release: Optionally, propagate the ground truth in collected predictions
        until the warm-up period end. This for measuring the drift resilence of multiple model
        starting # from an trajectory arbitrary timestep.
    :param show_debug_info: Optional; flag to enable debug info display for non-finite values
        detected.
    :return: A tuple consisting of mean predictions, standard deviation predictions (ale+epi),
        standard deviation predictions (epi only), and mean absolute error of predictions.
    """

    assert_is_OneDTransitionRewardModelV2(dynamic_model)
    assert isinstance(test_trajectories, List)

    pred_world_poses = []
    pred_means = []
    pred_stds = []
    pred_std_epis = []
    pred_maes = []
    pred_l2_norms = []

    # Cache Hydra-instantiated post-processing to avoid re-instantiation per trajectory
    _cached_deploy_post_proc = hydra.utils.instantiate(
        cfg.environment.deploy_rollout_post_processing, cfg, _recursive_=False
    )
    # RLRP-736 S1.5: thread the per-environment feature handler into deploy
    # (level C). No-op unless the deploy object exposes ``set_feature_handler``.
    attach_feature_handler_to_deploy(cfg, _cached_deploy_post_proc)

    for each_test_trajectory in test_trajectories:
        assert isinstance(
            each_test_trajectory,
            (TestTrajectoryDataclass, TestMotionTrajectoryDataclass),
        )

        # .... Rollout model and collect stats ....................................................

        if motion_model_container:
            motion_model_container.set_dynamics_model(dynamic_model)

            pred_world_pose, pred_ensemble, pred_ensemble_logvar = (
                multistep_model_testtime_rollout_and_collect_pred_stats(
                    cfg,
                    each_test_trajectory,
                    motion_model_container,
                    torch_rng,
                    compounded_predictions_score=compounded_predictions_score,
                    next_state_deterministic_selection=next_state_deterministic_selection,
                    next_state_sampling_size=next_state_sampling_size,
                    ground_truth_feed_warmup_steps=ground_truth_feed_warmup_steps,
                    tutor_and_release=tutor_and_release,
                    deploy_rollout_post_processing=_cached_deploy_post_proc,
                )
            )
        else:
            # Fallback for comparing against baseline legacy model
            pred_world_pose, pred_ensemble, pred_ensemble_logvar = (
                singlestep_model_testtime_rollout_and_collect_pred_stats(
                    cfg,
                    each_test_trajectory,
                    dynamic_model,
                    torch_rng,
                    compounded_predictions_score=compounded_predictions_score,
                    next_state_deterministic_selection=next_state_deterministic_selection,
                    next_state_sampling_size=next_state_sampling_size,
                    ground_truth_feed_warmup_steps=ground_truth_feed_warmup_steps,
                    tutor_and_release=tutor_and_release,
                    deploy_rollout_post_processing=_cached_deploy_post_proc,
                )
            )

        # .... Post-process data ..................................................................

        (
            pred_mean,
            pred_std,
            pred_std_epi,
        ) = compute_n_dim_obs_trajectory_probability_statistics_over_ensemble(
            n_dim_obs_pred=pred_ensemble,
            n_dim_obs_pred_logvar=pred_ensemble_logvar,
            ensemble_size=dynamic_model.model.num_members,
        )

        is_finite = torch.isfinite(pred_mean).all().item()
        pred_mean = torch.nan_to_num(pred_mean)
        is_finite = torch.isfinite(pred_std).all().item() and is_finite
        pred_std = torch.nan_to_num(pred_std)
        is_finite = torch.isfinite(pred_std_epi).all().item() and is_finite
        pred_std_epi = torch.nan_to_num(pred_std_epi)

        is_finite = torch.isfinite(pred_world_pose).all().item() and is_finite
        pred_world_pose = torch.nan_to_num(pred_world_pose)

        # RLRP-723 M5: score against the (possibly noisy) measured ``pose`` — the canonical
        # evaluation target in *both* environments (robotic real signal; math noisy signal,
        # mirroring robotic noise-coping). For the robotic env ``pose`` and ``pose_gt`` are the
        # same real signal; for the math env ``pose`` is the noisy path and ``pose_gt`` (clean)
        # is logistical/diagnostic only. This makes call site C consistent with call sites A/B.
        pose_tensor = torch.from_numpy(each_test_trajectory.pose).to(
            dtype=pred_world_pose.dtype, device=pred_world_pose.device
        )
        # RLRP-723 F1: pred_world_pose is already on the GT timestamp grid (RLRP-707), so the
        # error is the *direct* pred[i]-gt[i] (align=False); drop the trivially-zero seeded
        # index 0 via the [1:] slice so a perfect rollout scores exactly 0.
        pred_mae = compute_trajectories_prediction_error(
            pred=pred_world_pose[1:],
            target=pose_tensor[1:],
            align_trajectory_prediction_timestep_with_target_states=False,
            normalize_input_via_feature_scalling=False,
            batch_reduction=True,
            coordinate_error_mode="elementwise",
        )
        pred_l2_norm = compute_trajectories_prediction_error(
            pred=pred_world_pose[1:],
            target=pose_tensor[1:],
            align_trajectory_prediction_timestep_with_target_states=False,
            normalize_input_via_feature_scalling=False,
            batch_reduction=True,
            coordinate_error_mode="l2",
        )

        is_finite = torch.isfinite(pred_mae).all().item() and is_finite
        pred_mae = torch.nan_to_num(pred_mae)
        is_finite = torch.isfinite(pred_l2_norm).all().item() and is_finite
        pred_l2_norm = torch.nan_to_num(pred_l2_norm)

        if not is_finite and show_debug_info:
            print(
                f"\n{ConsoleFormat.MSG_ERROR_FORMAT}"
                f"{consol_msg_universal_one_liner('Non finite value detected!', print_it=False)}"
                f"{ConsoleFormat.MSG_END_FORMAT}\n"
            )

        pred_world_poses.append(pred_world_pose)
        pred_means.append(pred_mean)
        pred_stds.append(pred_std)
        pred_std_epis.append(pred_std_epi)
        pred_maes.append(pred_mae)
        pred_l2_norms.append(pred_l2_norm)

        # .... Memory management  .................................................................
        del pred_ensemble, pred_ensemble_logvar

    # .... Compute average metric over test trajectory set ........................................
    m_pred_world_poses = torch.stack(pred_world_poses, dim=-1).mean(dim=-1)
    m_pred_mean = torch.stack(pred_means, dim=-1).mean(dim=-1)
    m_pred_std = torch.stack(pred_stds, dim=-1).mean(dim=-1)
    m_pred_std_epi = torch.stack(pred_std_epis, dim=-1).mean(dim=-1)
    m_pred_mae = torch.stack(pred_maes, dim=-1).mean(dim=-1)
    m_pred_l2_norm = torch.stack(pred_l2_norms, dim=-1)

    return PredictionMetric(
        pred_obs=m_pred_world_poses,
        mean=m_pred_mean,
        std=m_pred_std,
        std_epi=m_pred_std_epi,
        mae=m_pred_mae,
        l2_norm=m_pred_l2_norm,
    )


def compute_trajectories_prediction_error(
    pred: Union[
        torch.Tensor, Tuple[torch.Tensor, ...], np.ndarray, Tuple[np.ndarray, ...]
    ],
    target: Union[torch.Tensor, np.ndarray],
    align_trajectory_prediction_timestep_with_target_states: bool,
    normalize_input_via_feature_scalling: bool = True,
    trajectory_reduction: bool = False,
    batch_reduction: bool = True,
    coordinate_error_mode: Literal["elementwise", "l2"] = "elementwise",
    coordinate_axis: int = -1,
    coordinate_slice: Optional[slice] = None,
) -> Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]:
    # (NICE TO HAVE) ToDo: stabilization unit-test for 'trajectory_reduction' and 'batch_reduction'
    # (NICE TO HAVE) ToDo: refactor-out to a general env util module
    """
    Compute the absolute error (AE) or mean absolute error (MAE) for trajectory predictions
    by optionally aligning and normalizing predictions and target states so that the error metric
    be meaninfull.

    Torch-first implementation: when ``target`` is a ``torch.Tensor``, all computation is
    performed using torch operations. Numpy ndarrays are still supported for backward
    compatibility.

    Normalization is performed over feature dimensions so that error be easy to visualize since the
    intended usage is for post-training model deployment assessment.

    This function can optionaly aligns the trajectory predictions with their respective target
    states by trimming the last element of prediction arrays and subsequently normalizes
    predictions and targets via feature scaling. It computes the MAE of each prediction relative
    to the normalized target.

    :param pred: Predicted trajectories, which can be a singular tensor/ndarray or a tuple of
        such arrays, with individual predictions potentially differing in dimensionality.
    :param target: Ground truth target trajectories as a tensor or ndarray.
    :param align_trajectory_prediction_timestep_with_target_states: Remove target first timestep
        and last prediction timestep.
    :param normalize_input_via_feature_scalling: feature scalling normalization over both
        predictions and target.
    :param trajectory_reduction: Compute the sum over trajectory length and mean over feature dimensions.
    :param batch_reduction: Compute the batch mean (output a tensor/ndarray instead of a list).
    :param coordinate_error_mode: How to aggregate error across the coordinate axis.
        - ``"elementwise"`` (default): keep current behavior, ``|target - pred|`` per feature.
        - ``"l2"``: L2 (Euclidean) norm over ``coordinate_axis``
          (``sqrt(sum_i (target_i - pred_i)**2)`` over the coordinate components, e.g. xyz).
          Also known as Average Displacement Error (ADE) when later reduced over the
          trajectory length. Preserves physical-unit (e.g. meters) interpretation and is
          invariant to the orientation of the coordinate axes. Output drops the coordinate
          axis. Note: L2 is more sensitive to outliers than the per-feature MAE (it squares
          residuals, like RMSE does along the time axis); prefer the default ``"elementwise"``
          mode for noisy robotic data when per-axis MAE is what you need. The previous L1
          option was dropped because, up to a constant factor of the number of coordinate
          components, it is equivalent to averaging the elementwise MAE across features.
    :param coordinate_axis: Axis along which the coordinate components live (default last axis).
        Only used when ``coordinate_error_mode != "elementwise"``.
    :param coordinate_slice: Optional slice to select a sub-range of coordinate components along
        ``coordinate_axis`` (e.g. ``slice(0, 3)`` to keep only x,y,z of a larger state vector).
    :return: A tensor/ndarray if batch_reduction=True else a list representing the absolute
      error for each predicted trajectory when compared with the normalized target trajectory.
    """
    assert coordinate_error_mode in (
        "elementwise",
        "l2",
    ), f"coordinate_error_mode must be 'elementwise' or 'l2', not {coordinate_error_mode!r}"
    if coordinate_error_mode == "l2" and normalize_input_via_feature_scalling:
        raise ValueError(
            "coordinate_error_mode='l2' is incompatible with "
            "normalize_input_via_feature_scalling=True: feature scaling rescales each "
            "coordinate component independently, which destroys the physical-unit meaning "
            "of the L2 (Euclidean) distance across coordinates. Pass "
            "normalize_input_via_feature_scalling=False."
        )
    assert isinstance(
        target, (torch.Tensor, np.ndarray)
    ), f"target must be a torch.Tensor or np.ndarray, not {type(target)}"

    _is_torch = isinstance(target, torch.Tensor)

    # .... Align trajectory prediction with target states .........................................
    if align_trajectory_prediction_timestep_with_target_states:
        if not isinstance(pred, Sequence):
            pred = [pred[:-1, ...]]
        else:
            pred = list(pred)
            for pred_idx in range(len(pred)):
                pred[pred_idx] = pred[pred_idx][:-1, ...]
        target = target[1:, ...]
        trajectories = [*pred, target]
    else:
        if not isinstance(pred, Sequence):
            trajectories = [pred, target]
            pred = [pred]
        else:
            trajectories = [*pred, target]

    check_ndarray_list_composante_shapes_match(trajectories)

    if normalize_input_via_feature_scalling:
        # Note: Normalize over both target and prediction ndarray list
        trajectories = normalize_predictions(trajectories)
        pred = trajectories[:-1]
        target = trajectories[-1]

    # .... Compute mean absolute error ............................................................
    pred_mae = []
    for each_pred in pred:
        _t = target
        _p = each_pred
        if coordinate_slice is not None:
            # Slice along the coordinate axis (e.g., keep only x,y,z)
            idx = [slice(None)] * _t.ndim
            idx[coordinate_axis] = coordinate_slice
            idx = tuple(idx)
            _t = _t[idx]
            _p = _p[idx]

        if _is_torch:
            diff = _t - _p
            if coordinate_error_mode == "l2":
                # L2 (Euclidean) norm over the coordinate axis -> removes that axis
                absolute_error = torch.linalg.vector_norm(
                    diff, ord=2, dim=coordinate_axis
                )
            else:
                absolute_error = torch.abs(diff)
        else:
            diff = np.subtract(_t, _p)
            if coordinate_error_mode == "l2":
                absolute_error = np.linalg.norm(diff, ord=2, axis=coordinate_axis)
            else:
                absolute_error = np.absolute(diff)

        if trajectory_reduction:
            if _is_torch:
                absolute_error = torch.sum(absolute_error, dim=0)
                absolute_error = torch.mean(absolute_error)
            else:
                absolute_error = np.sum(
                    absolute_error, axis=0
                )  # Sum over trajectory horizon
                absolute_error = np.mean(absolute_error)  # Average over dimensions

        if batch_reduction:
            # Add batch dimension for the reduction step
            if _is_torch:
                absolute_error = absolute_error.unsqueeze(0)
            else:
                absolute_error = np.expand_dims(absolute_error, 0)

        pred_mae.append(absolute_error)

    if batch_reduction:
        if _is_torch:
            pred_mae = torch.cat(pred_mae, dim=0).mean(dim=0)
        else:
            pred_mae = np.mean(np.concatenate(pred_mae, axis=0), axis=0)

    return pred_mae
