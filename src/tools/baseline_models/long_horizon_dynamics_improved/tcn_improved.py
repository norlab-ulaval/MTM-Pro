"""Improved (copy) of the Temporal Convolutional Network conv stack.

Provenance
----------
This module is a **verbatim copy** of the conv-stack classes in
``src/tools/baseline_models/long_horizon_dynamics/tcn.py`` — itself based on the
codebase (https://github.com/arplaboratory/long-horizon-dynamics) of the paper
*Learning Long-Horizon Predictions for Quadrotor Dynamics*
(https://arxiv.org/abs/2407.12964), published in IROS 2024.

The original ``tcn.py`` is a third-party / other-paper baseline and MUST NOT be
modified (see the RLRP-775 `.junie` plan
``perf_tcn_ms2ss_training_speed_RLRP-775.md`` §1.6). All RLRP-775 conv-stack
improvements live here instead, reached only by the new improved model classes
(``MS2SSProbabilisticTCNFaster`` / ``MS2SSProbabilisticTCNStreaming``); the
paper-faithful ``MS2SSProbabilisticTCN`` keeps importing the original ``tcn.py``.

Deltas vs. the original conv stack
----------------------------------
* **A3** — ``Chomp1d.forward`` returns the narrowed view WITHOUT the trailing
  ``.contiguous()`` copy. ``Conv1d`` / ``BatchNorm`` accept the (non-contiguous)
  narrowed view, so the explicit per-block copy on the autoregressive hot path is
  pure overhead. The returned VALUES are byte-identical to the original, so a
  ``TemporalConvNet`` built here is numerically identical to the original for a
  shared ``state_dict`` (guarded by the conv-stack parity sub-test in
  ``tests/.../test_tcn_faster_ms2ss.py``).
* **receptive_field** — additive read-only property (no behaviour change),
  ``1 + sum_blocks 2*(k-1)*d`` (== 29 for ``k=3``, ``hid_size=[512,256,256]``).

``forward`` is kept byte-identical to the original so warm-up parity is trivially
preserved. The ``A1`` streaming primitives are intentionally NOT implemented here
yet (deferred, see the plan revision log): for the live configuration the TCN
receptive field ``R = 29`` exceeds the sliding-window length ``HI = 20``, so a
carried-context streaming cache cannot be numerically equal to the paper-faithful
per-window (zero-padded) convolution.
"""

import torch
import torch.nn as nn


class TemporalConvNet(nn.Module):

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int | tuple[int] = 2,
        dropout: float = 0.0,
    ):
        super(TemporalConvNet, self).__init__()
        self.kernel_size = kernel_size
        self.num_channels = list(num_channels)
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    @property
    def receptive_field(self) -> int:
        """``1 + sum_blocks 2*(k-1)*d`` (two dilated convs per ``TemporalBlock``).

        Additive read-only helper (no behaviour change): the number of trailing
        input frames the last output timestep depends on. == 29 for ``k=3`` and
        ``hid_size=[512,256,256]`` (dilations 1, 2, 4).
        """
        k = self.kernel_size
        return 1 + sum(2 * (k - 1) * (2**i) for i in range(len(self.num_channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ----------------------------------------------------------------------------
class TemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int],
        dilation: int | tuple[int],
        padding: int,
        dropout: float = 0.0,
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        # Leakly relu
        self.relu1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.LeakyReLU(0.2)

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)


class Chomp1d(nn.Module):

    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # A3 (RLRP-775): drop the trailing ``.contiguous()`` of the original
        # ``tcn.py`` ``Chomp1d``. ``Conv1d`` / ``BatchNorm`` accept the
        # (non-contiguous) narrowed view; the explicit copy was pure overhead on
        # the autoregressive hot path. The returned VALUES are byte-identical.
        return x[:, :, : -self.chomp_size]
