# coding=utf-8
from typing import Tuple

import numpy as np
import torch
import torchmetrics
from torchmetrics.audio import SignalNoiseRatio


def sliding_window_signal_to_noise_ratio_metric(
        pred: np.ndarray, target: np.ndarray, snr_window: int, obs_dim: int
        ) -> Tuple[torch.Tensor, torch.Tensor, torchmetrics.Metric]:
    """
    Calculates the Signal-to-Noise Ratio (SNR) over a sliding window for given prediction and
    target arrays.

    Note: Higher value mean less noise.

    :param pred: Prediction array; should have the same shape as `target`.
    :param target: Target array; should have the same shape as `pred`.
    :param snr_window: Size of the sliding window over which the SNR is computed.
    :param obs_dim: Observation dimension for preparing the data, e.g. obs x,y,z ==> 3.
    :return: A tuple containing the overall SNR, the SNR values for each window, and the SNR
     metric object.
    """
    assert pred.shape == target.shape, f"{pred.shape} != {target.shape}"

    snr_metric = SignalNoiseRatio(zero_mean=False)
    values = []
    for idx in range(len(pred) - snr_window):
        p_pred = prep_data(pred[idx: idx + snr_window], obs_dim=obs_dim)
        p_target = prep_data(target[idx: idx + snr_window], obs_dim=obs_dim)
        snr_values = torch.empty((1, obs_dim))
        for each_dim in range(obs_dim):
            snr_values[:, each_dim] = snr_metric(
                    preds=p_pred[each_dim],
                    target=p_target[each_dim],
                    )
        values.append(snr_values)
    snr_all = snr_metric.compute()
    return snr_all, torch.cat(values, dim=0), snr_metric


def prep_data(data: np.ndarray, obs_dim: int) -> torch.Tensor:
    if obs_dim == 1:
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
    if data.shape[-1] == obs_dim:
        data = data.T
    return torch.tensor(data)


def fast_fourier_transform_denoising(noisy_signal: np.ndarray,
                                     threshold: float = 1.0) -> np.ndarray:
    # @formatter:off
    """
    Performs denoising of a 1-dimensional noisy signal using the Fast Fourier Transform (FFT) by filtering out frequency components below a certain threshold.

    Credit: https://lightning.ai/docs/torchmetrics/stable/gallery/audio/signal_to_noise_ratio.html#sphx-glr-gallery-audio-signal-to-noise-ratio-py

    :param noisy_signal: The input array representing the noisy signal.
    :param threshold: The magnitude threshold below which frequency components are filtered out.
    :return: The denoised signal as an array.
    """
    # @formatter:on
    if noisy_signal.ndim == 1:
        np.expand_dims(noisy_signal, 1)
    freq_domain = np.fft.fft(noisy_signal)  # Filter frequencies using FFT
    magnitude = np.abs(freq_domain)
    filtered_freq_domain = freq_domain * (magnitude > threshold)
    return np.fft.ifft(filtered_freq_domain).real  # Perform inverse FFT to reconstruct the signal
