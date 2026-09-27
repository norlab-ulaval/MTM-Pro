"""
Code in this module is based on the codebase (https://github.com/arplaboratory/long-horizon-dynamics)
 of the paper Learning Long-Horizon Predictions for Quadrotor Dynamics (https://arxiv.org/abs/2407.12964)
 published in IROS 2024.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .mlp import MLP



# class TCN(nn.Module):
#     def __init__(self, input_size, encoder_sizes, history_len, decoder_sizes, output_size,
#     kernel_size, dropout, **kwargs):
#         super(TCN, self).__init__()
#         self.encoder = TemporalConvNet(input_size, encoder_sizes, kernel_size=kernel_size,
#         dropout=dropout)
#         self.decoder = MLP(encoder_sizes[-1], history_len, decoder_sizes, output_size, dropout)


#     def forward(self, x,  args=None):
#         x = x.permute(0, 2, 1)  # Transpose input to (batch_size, num_features, history_length)
#         y = self.encoder(x)
#         y = self.decoder(y) # Take the output of the last time step
#         return y


class TCN(nn.Module):

    def __init__(
        self,
        input_size: int,
        encoder_sizes: list[int],
        history_len: int,
        decoder_sizes: list[int],
        output_size: int,
        kernel_size: int | tuple[int],
        dropout: float,
        **kwargs,
    ):
        super(TCN, self).__init__()
        self.encoder = TemporalConvNet(
            input_size, encoder_sizes, kernel_size=kernel_size, dropout=dropout
        )
        self.decoder = MLP(
            encoder_sizes[-1] / history_len,
            history_len,
            decoder_sizes,
            output_size,
            dropout,
        )
        # self.linear = nn.Linear(encoder_sizes[-1], output_size)

    def forward(self, x: torch.Tensor, args=None) -> torch.Tensor:
        x = x.permute(
            0, 2, 1
        )  # Transpose input to (batch_size, num_features, history_length)
        x = self.encoder(x)
        x = self.decoder(x[:, :, -1:])  # Take the output of the last time step
        # x = self.linear(x[:, :, -1])
        return x


class TemporalConvNet(nn.Module):

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int | tuple[int] = 2,
        dropout: float = 0.0,
    ):
        super(TemporalConvNet, self).__init__()
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
        return x[:, :, : -self.chomp_size].contiguous()
