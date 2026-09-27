# coding=utf-8
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import mbrl
import torch
from torch import Tensor, nn as nn
from torch.nn import functional as F
import torch.nn as nn
import numpy as np

from omegaconf import omegaconf

from tools.baseline_models.long_horizon_dynamics.tcn import TemporalConvNet
from tools.multistep_tools.models import AbstractMS2SSAutoRegressive


class MS2SSProbabilisticTCN(AbstractMS2SSAutoRegressive):

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        singlestep_obs_len: int,
        singlestep_act_len: int,
        history_len: int = 20,
        horizon_len: int = 20,
        obs_feature_weights: Union[float, Tuple[float, ...]] = 1.0,
        act_feature_weights: Union[float, Tuple[float, ...]] = 1.0,
        feature_weight_mode: str = "tempered",
        feature_weight_max_ratio: Optional[float] = None,
        num_layers: Optional[int] = None,
        ensemble_size: int = 1,
        hid_size: list[int] = [256, 128, 128],
        deterministic: bool = True,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        distribution_name: str = "gaussian",
        mae_loss: bool = True,
        model_use_double_precision: bool = False,
        dropout: float = 0.0,
        decoder_num_layers: int = -1,
        decoder_hid_size: int = -2,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        ar_enabled: bool = True,
        receive_sequence_batch: bool = False,
        ar_train_horizon_unroll: bool = True,
        decoder_ss_head_num_layers: int = 1,
        kernel_size: int = 2,
        teacher_forcing_decay_stop: int = 0, # Options: >0=decay steps, -1=always_on, 0=always_off
        teacher_forcing_decay_start: int = 0, # Steps before decay begins (warmup)
        teacher_forcing_method: str = "linear", # Options: "linear", "exponential", "always_on" "always_off"
        ar_temporal_weights: float = 1.0,
        ar_temporal_weights_start: float = 1.0,
        ar_temporal_weights_warmup: int = 0,
        ar_temporal_weights_ramp_stop: int = 0,
        ar_unrol_len_probablity_decay_start: int = 0,
        ar_unrol_len_probablity_decay_stop: int = 0,
        ar_unrol_len_decay_method: str = "beta",
        ar_unrol_len_start_horizon_len: int = 1,
        decoder_receive_ms_hidden_size=False,
        description: Optional[str] = None,
        feature_geometry=None,
        internal_orientation=None,
        orientation_singlestep_slots=None,
    ):

        self.kernel_size = kernel_size

        if isinstance(hid_size, int) and num_layers is not None:
            self.encoder_sizes = [hid_size] * num_layers
        elif isinstance(hid_size, int):
            self.encoder_sizes = [hid_size]
        else:
            self.encoder_sizes = hid_size

        if decoder_hid_size == -1:
            self.decoder_hid_size = out_size
        elif decoder_hid_size == -2:
            self.decoder_hid_size = self.encoder_sizes[-1]
        else:
            self.decoder_hid_size = decoder_hid_size

        if decoder_num_layers == -1:
            self.decoder_num_layers = len(self.encoder_sizes)
        else:
            self.decoder_num_layers = decoder_num_layers

        self.history_len = history_len
        self.singlestep_obs_len = singlestep_obs_len
        self.singlestep_act_len = singlestep_act_len

        if model_use_double_precision:
            self.model_dtype = torch.double
        else:
            self.model_dtype = torch.float32

        super().__init__(
            in_size,
            out_size,
            device=device,
            singlestep_obs_len=singlestep_obs_len,
            singlestep_act_len=singlestep_act_len,
            history_len=history_len,
            horizon_len=horizon_len,
            obs_feature_weights=obs_feature_weights,
            act_feature_weights=act_feature_weights,
            feature_weight_mode=feature_weight_mode,
            feature_weight_max_ratio=feature_weight_max_ratio,
            num_layers=len(self.encoder_sizes),
            ensemble_size=ensemble_size,
            deterministic=deterministic,
            propagation_method=propagation_method,
            learn_logvar_bounds=learn_logvar_bounds,
            activation_fn_cfg=activation_fn_cfg,
            distribution_name=distribution_name,
            mae_loss=mae_loss,
            model_use_double_precision=model_use_double_precision,
            dropout=dropout,
            ar_enabled=ar_enabled,
            receive_sequence_batch=receive_sequence_batch,
            ar_train_horizon_unroll=ar_train_horizon_unroll,
            ss_head_num_layers=decoder_ss_head_num_layers,
            teacher_forcing_decay_stop=teacher_forcing_decay_stop,
            teacher_forcing_decay_start=teacher_forcing_decay_start,
            teacher_forcing_method=teacher_forcing_method,
            ar_temporal_weights=ar_temporal_weights,
            ar_temporal_weights_start=ar_temporal_weights_start,
            ar_temporal_weights_warmup=ar_temporal_weights_warmup,
            ar_temporal_weights_ramp_stop=ar_temporal_weights_ramp_stop,
            ar_unrol_len_probablity_decay_start=ar_unrol_len_probablity_decay_start,
            ar_unrol_len_probablity_decay_stop=ar_unrol_len_probablity_decay_stop,
            ar_unrol_len_decay_method=ar_unrol_len_decay_method,
            ar_unrol_len_start_horizon_len=ar_unrol_len_start_horizon_len,
            decoder_receive_ms_hidden_size=decoder_receive_ms_hidden_size,
            description=description,
            feature_geometry=feature_geometry,
            internal_orientation=internal_orientation,
            orientation_singlestep_slots=orientation_singlestep_slots,
        )

    def _build_network(
        self,
        num_layers: int,
        in_size: int,
        hid_size: int,
        out_size: int,
        ensemble_size: int,
        activation_fn_cfg: omegaconf.DictConfig,
        deterministic: bool,
        learn_logvar_bounds: bool,
        instanciate_logvar_bound_module: bool = True,
        dropout: float = 0.0,
    ) -> None:

        assert (
            self.num_members == 1
        ), f"Casse ensemble > 1 is not suported yet ({self.num_members=})."

        # RLRP-736 bespoke-forward plan §3.4: widen the per-block encoder input by
        # ``_ss_block_in_extra`` so the TCN consumes the PER-BLOCK encoded attitude
        # (0 when the by-construction rotation rep is OFF => bit-exact).
        self.ar_encoder = TemporalConvNet(
            num_inputs=self.singlestep_obs_len
            + self.singlestep_act_len
            + self._ss_block_in_extra,
            num_channels=self.encoder_sizes,
            kernel_size=self.kernel_size,
            dropout=dropout if len(self.encoder_sizes) >= 2 else 0.0,
        )

        if self.decoder_receive_ms_hidden_size:
            decoder_in_size = self.encoder_sizes[-1] * self.history_len
        else:
            decoder_in_size = self.encoder_sizes[-1]

        # Note: MS(HI->1) Classic many-to-one AR logic with horizon unroling
        decoder_out_size = self.singlestep_obs_len

        # Note: required for masking act dim
        self.set_deploy_head_multistep_to_singlestep_adapter(lambda x: x)

        super()._build_network(
            num_layers=self.decoder_num_layers,
            in_size=decoder_in_size,
            hid_size=self.decoder_hid_size,
            out_size=decoder_out_size,
            ensemble_size=ensemble_size,
            activation_fn_cfg=activation_fn_cfg,
            deterministic=deterministic,
            learn_logvar_bounds=learn_logvar_bounds,
            instanciate_logvar_bound_module=instanciate_logvar_bound_module,
            dropout=dropout,
        )

        return None

    def _forward_AR_encoder(self, x: Tensor) -> Tensor:

        # Transpose input  (..., MS, F) ->  (..., F, MS)
        x = x.swapaxes(-2, -1)

        e_out = self.ar_encoder(x)
        e_out = e_out.swapaxes(-2, -1)
        # Output shape (B X MS X Hdim) or (MS X Hdim)

        if self.decoder_receive_ms_hidden_size:
            # Note: MS(HI->HO->1) many-to-many-to-one AR logic

            # Revert shape (..., MS, Hdim) -> (..., MS X Hdim)
            e_out = e_out.flatten(start_dim=-2)
        else:
            # Note: MS(HI->1) Classic many-to-one AR logic with horizon unrolling
            # Pick Hdim sequence last timestep
            e_out = e_out[..., -1, :]

        return e_out
