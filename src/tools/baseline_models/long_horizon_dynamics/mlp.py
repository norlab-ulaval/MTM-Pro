"""
Code in this module is based on the codebase (https://github.com/arplaboratory/long-horizon-dynamics)
 of the paper Learning Long-Horizon Predictions for Quadrotor Dynamics (https://arxiv.org/abs/2407.12964)
 published in IROS 2024.
"""

import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):

    def __init__(
        self,
        input_size: float,
        history_len: int,
        decoder_sizes: List[int],
        output_size: int,
        dropout: float,
        **kwargs,
    ) -> None:
        super(MLP, self).__init__()
        self.model = self.make(int(input_size * history_len), decoder_sizes, output_size, dropout)

    def make(
        self,
        input_size: int,
        decoder_sizes: List[int],
        output_size: int,
        dropout: float,
    ) -> nn.Sequential:
        layers = []
        layers.append(nn.Linear(input_size, decoder_sizes[0]))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        for i in range(len(decoder_sizes) - 1):
            layers.append(nn.Linear(decoder_sizes[i], decoder_sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(decoder_sizes[-1], output_size))
        return nn.Sequential(*layers)

    def forward(self, x, args=None):
        x = x.reshape(x.shape[0], -1)
        x = self.model(x)
        return x
