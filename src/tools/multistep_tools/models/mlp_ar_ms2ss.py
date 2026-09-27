# coding=utf-8
import pathlib
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import mbrl
import torch
from mbrl.models import EnsembleLinearLayer
from torch import Tensor, nn as nn
from torch.nn import functional as F
import numpy as np

import torch.nn as nn
from omegaconf import omegaconf

from tools.multistep_tools.models import AbstractMS2SSAutoRegressive


class MS2SSProbabilisticMLPAR(AbstractMS2SSAutoRegressive):
    # ToDo: minimal implementation stability unit-test (ref task RLRP-456)

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
        num_layers: int = 2,
        ensemble_size: int = 1,
        hid_size: int = 256,
        deterministic: bool = True,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        distribution_name: str = "gaussian",
        mae_loss: bool = True,
        model_use_double_precision: bool = False,
        dropout: float = 0.0,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        ar_enabled: bool = True,
        receive_sequence_batch: bool = False,
        ar_train_horizon_unroll: bool = True,
        ss_head_num_layers: int = 1,
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
        description: Optional[str] = None,
        feature_geometry=None,
        internal_orientation=None,
        orientation_singlestep_slots=None,
    ):

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
            num_layers=num_layers,
            ensemble_size=ensemble_size,
            hid_size=hid_size,
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
            ss_head_num_layers=ss_head_num_layers,
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
            decoder_receive_ms_hidden_size=False,
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

        # Note: required for masking act dim
        self.set_deploy_head_multistep_to_singlestep_adapter(lambda x: x)

        super()._build_network(
            num_layers=num_layers,
            in_size=in_size,
            hid_size=hid_size,
            out_size=out_size,
            ensemble_size=ensemble_size,
            activation_fn_cfg=activation_fn_cfg,
            deterministic=deterministic,
            learn_logvar_bounds=learn_logvar_bounds,
            instanciate_logvar_bound_module=instanciate_logvar_bound_module,
            dropout=dropout,
        )

        return None

    def _forward_AR_encoder(self, x: Tensor) -> Tensor:
        return x

    def _default_forward_AR_encoder(self, x: Tensor) -> Tensor:
        # RLRP-736 bespoke-forward plan §3.4: MLP-AR is the degenerate AR case —
        # its "encoder" is the identity (``_forward_AR_encoder`` returns ``x``) and
        # it feeds the RAW COMPOSED window straight into ``hidden_layers`` (built at
        # the rep-expanded ``_trunk_in_size``). So, unlike the GRU/LSTM/TCN
        # per-block path, the attitude input slot(s) are encoded on the WHOLE
        # composed window here via ``_apply_orientation_input_encoding`` (no-op /
        # bit-exact when the by-construction rotation rep is OFF).
        x = self._apply_orientation_input_encoding(x)

        # Quick-hack setup for ensemble (ref task RLRP-456)
        if x.ndim == 4 and self.num_members == 1:
            x = x.squeeze(0)
        #     if self.ar_memory is not None:
        #         self.ar_memory = self.ar_memory.squeeze(1)

        x = x.to(dtype=self.model_dtype)
        return x
