# coding=utf-8
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt

from tools.math_tools.signal_analysis import (
    fast_fourier_transform_denoising,
    sliding_window_signal_to_noise_ratio_metric,
)
from algorithm.experience_replay_learning_loop.core.model_testing_utils import (
    compute_trajectories_prediction_error,
)
from tools.math_tools.normalization import normalize_predictions
from tools.plot_tools.plot_general_utils import arbitrary_dimension_array_plot


def loss_plot(
    train_losses: Union[np.ndarray, List[float]],
    val_losses: Union[np.ndarray, List[float]],
    title: str = "Training/Validation Loss",
    training_time: float = None,
    figsize: tuple = (20, 8),
    train_loss_comment: str = "(gaussian nll)",
    val_loss_comment: str = "(mse)",
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=50)  # default dpi=100
    ax[0].plot(train_losses)
    # ax[0].set_xlabel("epoch")
    ax[0].set_ylabel(f"train loss {train_loss_comment}")
    ax[1].plot(val_losses)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel(f"val loss {val_loss_comment}")
    fig.tight_layout(pad=2)
    fig.suptitle(title, size="large", weight="bold")
    if training_time:
        # Note: ``training_time`` is a *duration* in seconds (not an epoch timestamp).
        # Using ``time.localtime`` here would treat it as seconds-since-epoch and a
        # ``"%Mm %Ss"`` format would silently drop the hour/day components, leading to
        # grossly under-reported wall-clock times (e.g. a ~7h training rendered as
        # "03m 39s"). Compute h/m/s manually so the formatting also handles durations
        # longer than 24 hours correctly.
        total_seconds = int(round(float(training_time)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        training_time_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
        fig.text(
            0.95,
            0.90,
            f"Training wall-clock time: {training_time_str}",
            horizontalalignment="right",
        )
    return fig, ax


def prediction_epistemic_uncertainty_plot(
    pred_std: Union[np.ndarray, Tuple[np.ndarray, ...]],
    plot_label: Union[str, Tuple[str, ...]] = "",
    y_label: Tuple[str, ...] = ("y",),
    y_lim: Optional[List[Tuple[float, float]]] = None,
    title: Optional[str] = "Prediction Epistemic Uncertainty",
    figsize: tuple = (20, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = arbitrary_dimension_array_plot(
        data=pred_std,
        plot_label=plot_label,
        y_label=y_label,
        y_lim=y_lim,
        title=title,
        figsize=figsize,
    )
    return fig, ax


def prediction_mae_plot(
    target: Union[np.ndarray, Tuple[np.ndarray, ...]],
    pred_mean: Union[np.ndarray, Tuple[np.ndarray, ...]],
    plot_label: Union[str, Tuple[str, ...]] = "",
    y_label: Tuple[str, ...] = ("y",),
    y_lim: Optional[List[Tuple[float, float]]] = None,
    title: Optional[str] = "Prediction Mean Square Error",
    figsize: tuple = (20, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    pred_mae = compute_trajectories_prediction_error(
        pred_mean,
        target,
        align_trajectory_prediction_timestep_with_target_states=True,
        normalize_input_via_feature_scalling=True,
        batch_reduction=False,
    )
    assert isinstance(pred_mae, List)

    # RLRP-723: the compounded MAE is a cumulative quantity that depends on the
    # trajectory length, so annotate the title to prevent comparing curves
    # computed over different trajectory lengths.
    if title is not None:
        title = f"{title} (len(T) based)"

    fig, ax = arbitrary_dimension_array_plot(
        data=tuple(pred_mae),
        plot_label=plot_label,
        y_label=y_label,
        y_lim=y_lim,
        title=title,
        figsize=figsize,
    )
    return fig, ax


def prediction_signal_to_noise_approximate_abs_error_plot(
    target: np.ndarray,
    pred_mean: Union[np.ndarray, Tuple[np.ndarray, ...]],
    plot_label: Union[str, Tuple[str, ...]] = "",
    smooth_window_out: int = 80,
    fft_smoother: bool = True,
    fft_threshold: float = 1,
    smooth_window_in: int = 2,
    y_label: Tuple[str, ...] = ("y",),
    y_lim: Optional[List[Tuple[float, float]]] = None,
    title: Optional[str] = "Signal To Noise Abs Error Aprroximation",
    figsize: tuple = (20, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate a plot comparing the signal-to-noise absolute prediction error's over time.

        1. Approximate the time-variant noisy signal mean using Fast Fourier Transform or
            Savitzky-Golay filter, i.e. smooth the signal noise.
        2. Compute the mean absolute error (MAE) between the time-variing noisy signal and the
            smoothed signal approximattion.

    :param target: Target signal values.
    :param pred_mean: Predicted mean values, either as a single array or a tuple of arrays.
    :param plot_label: Label(s) for the plot line(s).
    :param smooth_window_in: Window size for approximating the signal (Only valid if
    fft_smoother=False).
    :param smooth_window_out: Window size for smoothing data visualization.
    :param fft_smoother: Use Fast Fourier Transform or Savitzky-Golay filter otherwise
    :param fft_threshold: Frequency domain threshold (Only valid if fft_smoother=True).
    :param y_label: Tuple denoting Y-axis label(s).
    :param y_lim: Optional tuple specifying Y-axis limits for the plot.
    :param title: Title of the plot.
    :param figsize: Size of the figure.
    :return: A tuple containing the generated figure and axes.
    """
    # .... Pre-condition ..........................................................................
    assert smooth_window_in >= 1
    assert smooth_window_out >= 1

    # .... Setup ..................................................................................
    target = target[1:]
    if not isinstance(pred_mean, Sequence):
        pred_mean = [pred_mean[:-1, ...], target]
        plot_label = [plot_label, "target"]
    else:
        pred_mean = list(pred_mean)
        for pred_idx in range(len(pred_mean)):
            pred_mean[pred_idx] = pred_mean[pred_idx][:-1, ...]
        pred_mean = [*pred_mean, target]
        plot_label = [*plot_label, "target"]

    pred_smooth = []
    pred_signal2noise_abs_error = []
    for each in pred_mean:
        pred_smooth.append(np.empty_like(each))
        pred_signal2noise_abs_error.append(np.empty_like(each))

    # .... Normalization via feature scalling .....................................................
    normalized_pred_mean = normalize_predictions(pred_mean)

    for pred_idx in range(len(normalized_pred_mean)):
        # .... Smooth noisy signal to get a proxy-average of the signal ...........................
        if fft_smoother:
            pred_smooth[pred_idx] = fast_fourier_transform_denoising(
                normalized_pred_mean[pred_idx], threshold=fft_threshold
            )
        else:
            pred_smooth[pred_idx] = savgol_filter(
                normalized_pred_mean[pred_idx], smooth_window_in, polyorder=2, axis=0
            )

        # .... Compute absolute error and smooth data for vizualisation ...........................
        pred_signal2noise_abs_error[pred_idx] = np.absolute(
            np.subtract(pred_smooth[pred_idx], normalized_pred_mean[pred_idx])
        )

        pred_signal2noise_abs_error[pred_idx] = savgol_filter(
            pred_signal2noise_abs_error[pred_idx],
            smooth_window_out,
            polyorder=2,
            axis=0,
        )

    fig, ax = arbitrary_dimension_array_plot(
        data=tuple(pred_signal2noise_abs_error),
        plot_label=tuple(plot_label),
        y_label=y_label,
        y_lim=y_lim,
        title=title,
        figsize=figsize,
    )

    return fig, ax


def prediction_signal_to_noise_ratio_sliding_window_plot(
    target: np.ndarray,
    target_clean: np.ndarray,
    snr_window: int,
    pred_mean: Union[np.ndarray, Tuple[np.ndarray, ...]],
    plot_label: Union[str, Tuple[str, ...]] = "",
    fft_smoother: bool = True,
    fft_threshold: float = 1,
    smooth_window_in: int = 2,
    y_label: Tuple[str, ...] = ("y",),
    y_lim: Optional[List[Tuple[float, float]]] = None,
    title: Optional[str] = "Sliding Window Signal-to-Noise Ratio",
    figsize: tuple = (20, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate a plot comparing the signal-to-noise absolute prediction error's over time.

        1. Approximate the time-variant noisy signal mean using Fast Fourier Transform or
            Savitzky-Golay filter, i.e. smooth the signal noise.
        2. Compute the Signal-to-Noise Ratio (SNR) between the time-variing noisy signal and the
            smoothed signal approximattion over trajectory using sliding window.

    :param target: Target signal values (noise version).
    :param target_clean: Target signal values (the reference one).
    :param snr_window: The SNR computation sliding window lenght.
    :param pred_mean: Predicted mean values, either as a single array or a tuple of arrays.
    :param plot_label: Label(s) for the plot line(s).
    :param smooth_window_in: Window size for approximating the signal (Only valid if
    fft_smoother=False).
    :param fft_smoother: Use Fast Fourier Transform or Savitzky-Golay filter otherwise
    :param fft_threshold: Frequency domain threshold (Only valid if fft_smoother=True).
    :param y_label: Tuple denoting Y-axis label(s).
    :param y_lim: Optional tuple specifying Y-axis limits for the plot.
    :param title: Title of the plot.
    :param figsize: Size of the figure.
    :return: A tuple containing the generated figure and axes.
    """
    # .... Pre-condition ..........................................................................
    assert smooth_window_in >= 1
    assert snr_window >= 1

    # .... Setup ..................................................................................
    target = target[1:]
    target_clean = target_clean[1:]
    if not isinstance(pred_mean, Sequence):
        pred_mean = [pred_mean[:-1, ...], target, target_clean]
        plot_label = [plot_label, "target"]
    else:
        pred_mean = list(pred_mean)
        for pred_idx in range(len(pred_mean)):
            pred_mean[pred_idx] = pred_mean[pred_idx][:-1, ...]
        pred_mean = [*pred_mean, target, target_clean]
        plot_label = [*plot_label, "target"]

    pred_smooth = []
    pred_snr = []
    for each in pred_mean[:-2]:
        pred_smooth.append(np.empty_like(each))
    for each in pred_mean[:-1]:
        pred_snr.append(np.empty_like(each))
    pred_snr_rollout = np.empty(len(pred_mean) - 1)

    # .... Normalization via feature scalling .....................................................
    normalized_pred_mean = normalize_predictions(pred_mean)

    for pred_idx in range(len(pred_smooth)):
        # .... Smooth noisy signal to get a proxy-average of the signal ...........................
        if fft_smoother:
            pred_smooth[pred_idx] = fast_fourier_transform_denoising(
                normalized_pred_mean[pred_idx], threshold=fft_threshold
            )
        else:
            pred_smooth[pred_idx] = savgol_filter(
                normalized_pred_mean[pred_idx], smooth_window_in, polyorder=2, axis=0
            )

        # .... Compute predictions sliding window SNR .............................................
        (
            pred_snr_rollout[pred_idx],
            pred_snr[pred_idx],
            _,
        ) = sliding_window_signal_to_noise_ratio_metric(
            normalized_pred_mean[pred_idx],
            pred_smooth[pred_idx],
            snr_window,
            len(y_label),
        )
    else:
        # .... Compute target sliding window SNR ..................................................
        pred_snr_rollout[-1], pred_snr[-1], _ = (
            sliding_window_signal_to_noise_ratio_metric(
                normalized_pred_mean[-2],
                normalized_pred_mean[-1],
                snr_window,
                len(y_label),
            )
        )

    for idx in range(len(pred_snr)):
        pred_snr[idx] = pred_snr[idx].detach().numpy()

    fig, ax = arbitrary_dimension_array_plot(
        data=tuple(pred_snr),
        plot_label=tuple(plot_label),
        y_label=y_label,
        y_lim=y_lim,
        title=title,
        figsize=figsize,
    )

    # .... Show rollout wide SNR ..................................................................
    rollout_snr_text = f"Signal-to-Noise Ratio (Rollout wide):\n"
    for idx in range(len(pred_snr_rollout)):
        rollout_snr_text += f"  - {plot_label[idx]}: {pred_snr_rollout[idx]:>.2f}\n"

    fig.text(
        0.125,
        -0.01,
        rollout_snr_text,
        verticalalignment="top",
        horizontalalignment="left",
    )

    return fig, ax
