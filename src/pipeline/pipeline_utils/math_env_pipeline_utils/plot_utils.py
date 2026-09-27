# coding=utf-8
import warnings
from typing import Any, Union
import gymnasium as gym
import numpy as np
import omegaconf
import torch

from omegaconf import DictConfig

from math_gymnasium.envs.arbitrary_dim_math_continuous import MathContinuousGymnasium
from tools.plot_tools.plot import (
    prediction_epistemic_uncertainty_plot,
    prediction_mae_plot,
    prediction_signal_to_noise_approximate_abs_error_plot,
    prediction_signal_to_noise_ratio_sliding_window_plot,
)
from tools.plot_tools.plot_management import (
    manage_matplotlib_backend,
    manage_matplotlib_warnings,
    show_and_save_plot_helper,
)


def require_physical_stats_space(space: Any, consumer: str) -> None:
    """Refuse a statistics channel that is not in PHYSICAL units (RLRP-761 ``P3``).

    The math-env MAE / SNR surfaces score the prediction **statistics** channel
    against a **physical** target (``state_axes.poses`` /
    ``poses_with_noise``). Before ``P1`` that channel was left in normalized
    target space, so the comparison was inflated by ``1/sigma_target`` — ``1x``
    under ``normalizer_type='standard'`` (which is why it stayed invisible) and
    ``3-40x`` under ``standard_symmetric_innovation``.

    ``P1`` removed the defect at the producer; this is the guard that keeps a
    **pre-``P1`` artifact** (or a future producer that forgets its sibling
    channel) from silently re-entering the plot. It is the interim protection
    while the ``P3.1``/``P3.2`` pose-channel substitution stays deferred.

    :param space: the recorded unit space, or ``None`` for a run-stats tuple
        that predates the field.
    :param consumer: plot / call-site name, used in the message.
    :raises ValueError: on a recorded ``normalized`` (or otherwise non-physical)
        space. A missing space only warns — a legacy artifact cannot be proven
        wrong, but it must not pass unnoticed either.
    """
    if space is None:
        warnings.warn(
            f"[{consumer}] the prediction-statistics run-stats carry no recorded "
            "unit space (legacy artifact, pre-RLRP-761-P4). Assuming 'physical'; "
            "if this run predates P1 with a non-'standard' normalizer_type, the "
            "MAE/SNR plots are inflated by 1/sigma_target — regenerate the deploy "
            "phase (plan stage P5).",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if str(space) == "physical":
        return
    raise ValueError(
        f"[{consumer}] the prediction-statistics channel is in '{space}' space "
        "but these plots compare it against a PHYSICAL target, so the result "
        "would be inflated by 1/sigma_target (1x under "
        "normalizer_type='standard', 3-40x under "
        "standard_symmetric_innovation). Re-run the deploy phase with a "
        "post-RLRP-761-P1 build (plan stage P5)."
    )


def resolve_error_channel(
    pose_channel: Any, stats_channel: Any, consumer: str
) -> Any:
    """Pick the channel an ERROR metric must be scored from (RLRP-761 ``P3.1``).

    Every error metric in the repo (MAE, C-MAE, L2) is scored from the
    **integrated pose** channel; the math-env MAE / SNR surfaces were the only
    exception, scoring the prediction **statistics** channel instead. Since the
    producer now carries the pose alongside the statistics, that exception is
    removed here.

    The statistics channel remains the right input for the UNCERTAINTY surfaces
    (``prediction_epistemic_uncertainty_plot``) — the two are distinct
    quantities, not interchangeable ones.

    :param pose_channel: the integrated pose, or ``None`` for a run-stats tuple
        produced before the field existed.
    :param stats_channel: the prediction-statistics mean (legacy fallback).
    :param consumer: call-site name, used in the fallback warning.
    :return: *pose_channel* when available, else *stats_channel*.
    """
    if pose_channel is not None:
        return pose_channel
    warnings.warn(
        f"[{consumer}] the run-stats carry no pose channel (legacy artifact, "
        "pre-RLRP-761-P3.1); the error surfaces fall back to the prediction "
        "STATISTICS channel. That channel is denormalized since P1, so the "
        "result is numerically sound, but it is a per-step prediction rather "
        "than the integrated trajectory every other error metric uses.",
        RuntimeWarning,
        stacklevel=2,
    )
    return stats_channel


def plot_model_comparaison(
    cfg: DictConfig,
    model1_3d_run_stats: list[Any],
    model1_model_description,
    model2_3d_run_stats: list[Any],
    model2_model_description,
    exp_dir_relative_path: str | Any,
    headless: bool,
):

    # raise NotImplementedError(
    #     "ToDo: implement logic to crawl over `target_env` and "
    #     "`compounded_predictions_score`
    #     runs"
    # )

    for each in zip(model1_3d_run_stats, model2_3d_run_stats):
        # RLRP-761 P3 (interim guard) — the trailing `stats_space` is unpacked with
        # a star so a legacy 6-tuple (produced before the field existed) still
        # loads; `require_physical_stats_space` turns its absence into a warning
        # rather than an IndexError.
        (
            (
                target_env,
                model1_3d_pred_mean,
                model1_3d_pred_std,
                model1_3d_pred_epi,
                target_is_ood,
                compounded_predictions_score,
                *model1_extra,
            ),
            (
                _,
                model2_3d_pred_mean,
                model2_3d_pred_std,
                model2_3d_pred_epi,
                _,
                _,
                *model2_extra,
            ),
        ) = each

        require_physical_stats_space(
            model1_extra[0] if model1_extra else None,
            f"plot_model_comparaison/{model1_model_description}",
        )
        require_physical_stats_space(
            model2_extra[0] if model2_extra else None,
            f"plot_model_comparaison/{model2_model_description}",
        )

        # RLRP-761 P3.1 — the 8th tuple element is the integrated POSE channel.
        model1_pose = model1_extra[1] if len(model1_extra) > 1 else None
        model2_pose = model2_extra[1] if len(model2_extra) > 1 else None

        title_test_env_type = "OoD test env" if target_is_ood else "InD test env"
        file_test_env_type = "ood_test_env" if target_is_ood else "InD_test_env"
        file_compounded_predictions_score = (
            "_compounded_predictions_score" if compounded_predictions_score else ""
        )
        # Torch-first: convert tensors to numpy at plotting boundary
        def _to_np(x):
            return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

        plot_predictions_metrics(
            cfg,
            target_env,
            model2_model_description,
            _to_np(model2_3d_pred_mean),
            _to_np(model2_3d_pred_std),
            _to_np(model2_3d_pred_epi),
            model1_model_description,
            _to_np(model1_3d_pred_mean),
            _to_np(model1_3d_pred_std),
            _to_np(model1_3d_pred_epi),
            exp_dir_relative_path,
            headless,
            _to_np(
                resolve_error_channel(
                    model2_pose, model2_3d_pred_mean, model2_model_description
                )
            ),
            _to_np(
                resolve_error_channel(
                    model1_pose, model1_3d_pred_mean, model1_model_description
                )
            ),
            f"({title_test_env_type}, compounded_predictions_score="
            f"{compounded_predictions_score})",
            f"ensemble_model{file_compounded_predictions_score}_{file_test_env_type}",
        )

    # run_only_once
    plot_noise_metrics(
        cfg,
        target_env,
        model2_model_description,
        _to_np(
            resolve_error_channel(
                model2_pose, model2_3d_pred_mean, model2_model_description
            )
        ),
        model1_model_description,
        _to_np(
            resolve_error_channel(
                model1_pose, model1_3d_pred_mean, model1_model_description
            )
        ),
        exp_dir_relative_path,
        headless,
    )


def plot_predictions_metrics(
    cfg: omegaconf.DictConfig,
    target_env: Union[MathContinuousGymnasium, gym.Env],
    ms_model_name: str,
    ms_3d_pred_mean: np.ndarray,
    ms_3d_pred_std: np.ndarray,
    ms_3d_pred_epi: np.ndarray,
    ss_model_name: str,
    ss_3d_pred_mean: np.ndarray,
    ss_3d_pred_std: np.ndarray,
    ss_3d_pred_epi: np.ndarray,
    exp_dir_relative_path: str,
    headless: bool,
    ms_3d_error_channel: np.ndarray,
    ss_3d_error_channel: np.ndarray,
    title_comment: str,
    file_name_comment: str,
) -> None:
    pipeline_plot_show = (
        cfg.pipeline.plot.show_prediction_mae_plot
        or cfg.pipeline.plot.show_epistemic_uncertainty_plot
    )
    with warnings.catch_warnings():
        manage_matplotlib_warnings()
        manage_matplotlib_backend(pipeline_plot_show, headless)

        # .... Prediction mae plot ..............................................
        # RLRP-761 P3.1/P3.2 — CHANNEL CHOICE, now aligned with the rest of the
        # repo: this MAE is scored from ``*_3d_error_channel``, the INTEGRATED
        # POSE, exactly like ``pred_mae`` / ``pred_l2_norm`` in
        # ``general/train_and_deploy_utils.py``. It used to be scored from the
        # prediction STATISTICS channel (``*_3d_pred_mean``) — the only error
        # metric in the repo that did — which before ``P1`` also compared a
        # NORMALIZED prediction against a PHYSICAL target (inflated by
        # 1/sigma_target: 1x under 'standard', 3-40x under
        # standard_symmetric_innovation).
        # ``resolve_error_channel`` falls back to the statistics channel, with a
        # warning, for a legacy run-stats tuple that carries no pose;
        # ``require_physical_stats_space`` still refuses a pre-``P1`` normalized
        # statistics channel on that fallback path.
        # P3.3/P3.4: :func:`prediction_epistemic_uncertainty_plot` legitimately
        # stays a statistics-channel consumer (an uncertainty band is NOT an
        # error), and the clean (``poses``) vs noise-augmented
        # (``poses_with_noise``) target pair MUST stay intact — it is well posed
        # only because the math env is synthetic (both channels are samples of a
        # known ground truth), which has no equivalent in the robotic-3D env.
        y_lim_cfg = None
        # y_lim_cfg = [(-0.00001, 0.00601), (-0.00001, 0.00601), (-0.00001, 0.00601)]
        pred_mae_fig, _ = prediction_mae_plot(
            target=target_env.trj.state_axes.poses_with_noise,
            pred_mean=(ms_3d_error_channel, ss_3d_error_channel),
            plot_label=(ms_model_name, ss_model_name),
            y_label=("x", "y", "z"),
            y_lim=y_lim_cfg,
            title=f"Prediction Mean Square Error {title_comment}",
            figsize=(cfg.pipeline.plot.figsize[0], 13),
        )
        show_and_save_plot_helper(pred_mae_fig, exp_dir_relative_path,
                                  f"pred_mean_absolute_error_{file_name_comment}", headless,
                                  cfg.pipeline.plot.show_prediction_mae_plot,
                                  cfg.pipeline.plot.save_dpi)

        # .... Prediction epistemic uncertainty plot ..............................................
        y_lim_cfg = None
        # y_lim_cfg = [(-0.01, 0.701), (-0.01, 0.701), (-0.01, 0.701)]
        pred_epistemic_uncertainty_fig, _ = prediction_epistemic_uncertainty_plot(
            pred_std=(ms_3d_pred_epi, ss_3d_pred_epi),
            plot_label=(ms_model_name, ss_model_name),
            y_label=("x", "y", "z"),
            y_lim=y_lim_cfg,
            title=f"Prediction Epistemic Uncertainty {title_comment}",
            figsize=(cfg.pipeline.plot.figsize[0], 13),
        )
        show_and_save_plot_helper(pred_epistemic_uncertainty_fig, exp_dir_relative_path,
                                  f"pred_epistemic_uncertainty_{file_name_comment}", headless,
                                  cfg.pipeline.plot.show_epistemic_uncertainty_plot,
                                  cfg.pipeline.plot.save_dpi)

    return None


def plot_noise_metrics(
    cfg: omegaconf.DictConfig,
    target_env: Union[MathContinuousGymnasium, gym.Env],
    ms_model_name: str,
    ms_3d_error_channel: np.ndarray,
    ss_model_name: str,
    ss_3d_error_channel: np.ndarray,
    exp_dir_relative_path: str,
    headless: bool,
) -> None:
    """Signal-to-noise surfaces, scored from the INTEGRATED POSE channel.

    RLRP-761 ``P3.1``/``P3.2``: like the MAE plot, these three surfaces compare a
    prediction against a **physical** target (``poses_with_noise`` / ``poses``),
    so they take the same channel every other error metric in the repo uses.
    The caller resolves it via :func:`resolve_error_channel`, which falls back to
    the prediction-statistics channel (with a warning) for a legacy run-stats
    tuple that carries no pose.
    """
    pipeline_plot_show = (
        cfg.pipeline.plot.show_approx_sn_abs_error_plot
        or cfg.pipeline.plot.show_approx_sn_abs_error_plot
        or cfg.pipeline.plot.show_sliding_window_snr_plot
    )
    with warnings.catch_warnings():
        manage_matplotlib_warnings()
        manage_matplotlib_backend(pipeline_plot_show, headless)

    # .... Prediction signal to noise approximate abs error plot ..............................
    pred_snr_abs_error_fig_savgol, _ = (
        prediction_signal_to_noise_approximate_abs_error_plot(
            target=target_env.trj.state_axes.poses_with_noise,
            pred_mean=(ms_3d_error_channel, ss_3d_error_channel),
            plot_label=(ms_model_name, ss_model_name),
            title=f"Signal To Noise Abs Error Aprroximation (using Savgol)",
            fft_smoother=False,
            smooth_window_in=10,
            smooth_window_out=400,
            y_label=("x", "y", "z"),
            figsize=(cfg.pipeline.plot.figsize[0], 13),
        )
    )
    show_and_save_plot_helper(pred_snr_abs_error_fig_savgol, exp_dir_relative_path,
                              f"pred_signal_to_noise_abs_error_savgol", headless,
                              cfg.pipeline.plot.show_approx_sn_abs_error_plot,
                              cfg.pipeline.plot.save_dpi)

    pred_snr_abs_error_fig_fft, _ = (
        prediction_signal_to_noise_approximate_abs_error_plot(
            target=target_env.trj.state_axes.poses_with_noise,
            pred_mean=(ms_3d_error_channel, ss_3d_error_channel),
            plot_label=(ms_model_name, ss_model_name),
            title="Signal To Noise Abs Error Aprroximation (using FFT)",
            fft_smoother=True,
            fft_threshold=1,
            smooth_window_out=400,
            # y_lim=[(-0.0001,0.006),(-0.0001,0.006),(-0.0001,0.006),],
            y_label=("x", "y", "z"),
            figsize=(cfg.pipeline.plot.figsize[0], 13),
        )
    )
    show_and_save_plot_helper(pred_snr_abs_error_fig_fft, exp_dir_relative_path,
                              f"pred_signal_to_noise_abs_error_fft", headless,
                              cfg.pipeline.plot.show_approx_sn_abs_error_plot,
                              cfg.pipeline.plot.save_dpi)

    # .... Prediction signal-to-noise ratio sliding window plot ...............................
    pred_sliding_window_snr_fft_fig, _ = (
        prediction_signal_to_noise_ratio_sliding_window_plot(
            target=target_env.trj.state_axes.poses_with_noise,
            target_clean=target_env.trj.state_axes.poses,
            snr_window=200,
            pred_mean=(ms_3d_error_channel, ss_3d_error_channel),
            plot_label=(ms_model_name, ss_model_name),
            title="Sliding Window Signal-to-Noise Ratio",
            fft_smoother=True,
            fft_threshold=1,
            y_label=("x", "y", "z"),
            figsize=(cfg.pipeline.plot.figsize[0], 13),
        )
    )
    show_and_save_plot_helper(pred_sliding_window_snr_fft_fig, exp_dir_relative_path,
                              f"pred_sliding_window_snr", headless,
                              cfg.pipeline.plot.show_sliding_window_snr_plot,
                              cfg.pipeline.plot.save_dpi)

    return None


