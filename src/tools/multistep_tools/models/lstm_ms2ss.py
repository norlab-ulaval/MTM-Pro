# coding=utf-8
import pathlib
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import mbrl
import torch
from torch import Tensor, nn as nn
from torch.nn import functional as F
import torch.nn as nn
import numpy as np

from omegaconf import omegaconf

from tools.multistep_tools.models import AbstractMS2SSAutoRegressive


class MS2SSProbabilisticLSTM(AbstractMS2SSAutoRegressive):
    """Probabilistic LSTM MS2SS auto-regressive baseline.

    Stateful recurrent AR encoder: it owns an LSTM ``(h, c)`` state
    (``ar_memory`` / ``ar_memory_c``) threaded through the free-running horizon
    unroll (training ``loss()`` U-loop) and the deploy rollout.

    ``ar_recurrent_memory_mode`` (RLRP-767) selects HOW that ``(h, c)`` state is
    threaded, because the compounded-prediction window slides by one frame each
    unroll step and overlaps its predecessor by ``history_len - 1`` frames:

    * ``"single_frame_carry"`` (**default, best practice**): warm ``(h, c)`` on
      the first step with the full history window, then carry it and feed ONLY
      the newest frame on subsequent steps. Canonical stateful AR: each frame is
      ingested exactly once, no history double-counting. This is the ONLY mode
      incompatible with ``decoder_receive_ms_hidden_size=True`` (length-1 recurrent
      sequence cannot fill the many-to-many-to-one decoder; ``sliding_window_reset``
      and ``legacy`` re-feed the full window and stay compatible with that decoder).
    * ``"sliding_window_reset"``: reset ``(h, c)`` each step and re-feed the whole
      sliding window (pure many-to-one, no carry). Also double-count free; more
      expensive but compatible with every decoder configuration.
    * ``"legacy"``: carry ``(h, c)`` AND re-feed the whole sliding window each
      step (historical behaviour); the ``history_len - 1`` overlapping frames are
      re-ingested on top of the carried state, double-counting history. Kept only
      for bit-exact reproduction of pre-RLRP-767 experiments/checkpoints.
    """

    ar_memory_c: Optional[torch.Tensor] = None

    # RLRP-786 AR-baseline extension: the cuDNN ``nn.LSTM`` encoder (``(h, c)`` carried across the
    # unroll, reserve / workspace allocation, ``flatten_parameters`` weight buffer) was validated
    # under ``torch.cuda.CUDAGraph`` capture on the Jetson AGX Orin 2026-09-19
    # (``test_ar_ms2ss_cuda_graph_step_parity_rlrp786.py::test_ar_rnn_cuda_graph_step_is_bit_identical_to_eager_over_k_steps[lstm]``:
    # k = 4 steps, loss / grads / Adam state ``torch.equal`` to eager under ``cudnn.deterministic``,
    # ``single_frame_carry``, ``dropout=0``). A100 confirmation runs in the Narval parity phase.
    # Flip back to ``False`` to report the encoder as a capture blocker (eager fallback + notice).
    _CUDA_GRAPH_ENCODER_VALIDATED: bool = True

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
        decoder_num_layers: int = -1,
        decoder_hid_size: int = -2,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        ar_enabled: bool = True,
        receive_sequence_batch: bool = False,
        ar_train_horizon_unroll: bool = True,
        decoder_ss_head_num_layers: int = 1,
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
        # RLRP-767: recurrent-state threading policy through the AR unroll.
        #   single_frame_carry     -> warm-up then newest-frame-only (canonical,
        #                             no history double-count) [best-practice default]
        #   sliding_window_reset   -> reset state each step, re-feed full window
        #   legacy                 -> carry state + re-feed full window (double-counts
        #                             history; pre-RLRP-767 reproduction only)
        ar_recurrent_memory_mode: str = "single_frame_carry",
        decoder_receive_ms_hidden_size=False,
        description: Optional[str] = None,
        feature_geometry=None,
        internal_orientation=None,
        orientation_singlestep_slots=None,
    ):

        if decoder_hid_size == -1:
            self.decoder_hid_size = out_size
        elif decoder_hid_size == -2:
            self.decoder_hid_size = hid_size
        else:
            self.decoder_hid_size = decoder_hid_size

        if decoder_num_layers == -1:
            self.decoder_num_layers = num_layers
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
            ar_recurrent_memory_mode=ar_recurrent_memory_mode,
            decoder_receive_ms_hidden_size=decoder_receive_ms_hidden_size,
            description=description,
            feature_geometry=feature_geometry,
            internal_orientation=internal_orientation,
            orientation_singlestep_slots=orientation_singlestep_slots,
        )

        self.ar_memory_c = None

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
        # ``_ss_block_in_extra`` so the LSTM consumes the PER-BLOCK encoded attitude
        # (0 when the by-construction rotation rep is OFF => bit-exact).
        self.ar_encoder = nn.LSTM(
            input_size=self.singlestep_obs_len
            + self.singlestep_act_len
            + self._ss_block_in_extra,
            hidden_size=hid_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers >= 2 else 0.0,
            device=self.device,
            dtype=self.model_dtype,
        )

        if dropout > 0.0:
            self.ar_dropout = nn.Dropout(dropout)
        else:
            self.ar_dropout = None

        if self.decoder_receive_ms_hidden_size:
            decoder_in_size = hid_size * self.history_len
        else:
            decoder_in_size = hid_size

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

        if self.ar_train_horizon_unroll:
            # RLRP-757 review follow-up: ``ar_recurrent_memory_mode`` selects the
            # window/carry policy (``legacy`` == carry + full window, bit-exact).
            enc_in, carry_state = self._ar_recurrent_step_inputs(x)
            if carry_state:
                if self.ar_memory is None:
                    e_out, (self.ar_memory, self.ar_memory_c) = self.ar_encoder(enc_in)
                else:
                    e_out, (self.ar_memory, self.ar_memory_c) = self.ar_encoder(
                        enc_in, (self.ar_memory, self.ar_memory_c)
                    )
                # When NOT training (eval/inference ``deploy``) the recurrent (h, c) state
                # must be detached before it is carried into a subsequent call: an eval-mode
                # cuDNN LSTM forward does not retain the reserve space required for backward,
                # so letting the eval-mode hidden state bleed into a later training
                # ``loss().backward()`` surfaces on GPU as
                # "cudnn RNN backward can only be called in training mode". Training keeps
                # the live graph so BPTT across the AR horizon unroll is unchanged. Mirrors
                # the ``deploy`` stash guard in ``CompoundedPredictionMultiStepIterator``.
                if not self.training:
                    if self.ar_memory is not None:
                        self.ar_memory = self.ar_memory.detach()
                    if self.ar_memory_c is not None:
                        self.ar_memory_c = self.ar_memory_c.detach()
            else:
                # Stateless lookback model variation
                # ``sliding_window_reset``: no (h, c) state carried across unroll steps.
                # PyTorch automatically creates and destroys a zero-initialized hidden state
                self.ar_memory = None
                self.ar_memory_c = None
                e_out, _ = self.ar_encoder(enc_in)
        else:
            e_out, _ = self.ar_encoder(x)
        # Output shape (B X MS X Hdim) or (MS X Hdim)

        if self.decoder_receive_ms_hidden_size:
            # Note: MS(HI->HO->1) many-to-many-to-one AR logic

            # Revert shape (..., MS, Hdim) -> (..., MS X Hdim)
            e_out = e_out.flatten(start_dim=-2)
        else:
            # Note: MS(HI->1) Classic many-to-one AR logic with horizon unrolling
            # Pick Hdim sequence last timestep
            e_out = e_out[..., -1, :]

        if self.ar_dropout is not None:
            e_out = self.ar_dropout(e_out)
        return e_out

    def wipe_ar_memory(self) -> None:
        super().wipe_ar_memory()
        self.ar_memory_c = None
        return None

    def _save_state_dict(self) -> dict[str, list[int] | dict[str, Any] | Any]:
        model_dict = super()._save_state_dict()
        model_dict["ar_memory_c"] = self.ar_memory_c
        return model_dict

    def _load_state_dict(self, model_dict):
        # Validate the LSTM-specific ``ar_memory_c`` key with the same strictness the
        # base applies to ``ar_memory`` (``_save_state_dict`` always co-writes it):
        # a missing key indicates an incompatible / corrupt checkpoint.
        if "ar_memory_c" not in model_dict:
            raise KeyError(
                f"{type(self).__name__}._load_state_dict: missing keys in "
                f"model_dict: ar_memory_c"
            )
        loaded_ar_memory_c = model_dict["ar_memory_c"]
        if loaded_ar_memory_c is not None and not isinstance(
            loaded_ar_memory_c, torch.Tensor
        ):
            raise TypeError(
                f"{type(self).__name__}._load_state_dict: 'ar_memory_c' must be "
                f"either None or a torch.Tensor, got {type(loaded_ar_memory_c).__name__}."
            )

        super()._load_state_dict(model_dict)
        self.ar_memory_c = loaded_ar_memory_c
        return None
