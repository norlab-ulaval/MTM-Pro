# coding=utf-8

import pathlib
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import omegaconf
from torch import Tensor
from torch.nn import functional as F

from tools.multistep_tools.models import CompoundedPredictionMultiStepIterator
from tools.multistep_tools.models.weighted_multistep_dual_head_mlp import (
    WeightedMultiStepDualHeadMLP,
)
from tools.multistep_tools.models.utils import (
    reduce_deterministic_compose_loss,
    reduce_probabilistic_compose_loss,
)


class AbstractMS2SSAutoRegressive(
    CompoundedPredictionMultiStepIterator,
    WeightedMultiStepDualHeadMLP,
    metaclass=ABCMeta,
):
    # ToDo: minimal implementation stability unit-test (ref task RLRP-456)

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        singlestep_obs_len: int,
        singlestep_act_len: int,
        history_len: int,
        horizon_len: int,
        obs_feature_weights: Union[float, Tuple[float, ...]] = 1.0,
        act_feature_weights: Union[float, Tuple[float, ...]] = 1.0,
        feature_weight_mode: str = "tempered",
        feature_weight_max_ratio: Optional[float] = None,
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        deterministic: bool = False,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        distribution_name: str = "gaussian",
        mae_loss: bool = True,
        model_use_double_precision: bool = False,
        dropout: float = 0.0,
        ar_enabled: bool = True,
        receive_sequence_batch: bool = False,
        ss_composite_loss_weight: float = 1.0,
        ms_composite_loss_weight: float = 1.0,
        ar_train_horizon_unroll: bool = True,
        ss_head_num_layers: int = 1,
        teacher_forcing_decay_stop: int = 0,  # Options: >0=decay steps, -1=always_on, 0=always_off
        teacher_forcing_decay_start: int = 0,  # Steps before decay begins (warmup)
        teacher_forcing_method: str = "linear",  # Options: "linear", "exponential", "always_on" "always_off"
        ar_temporal_weights: float = 1.0,
        ar_temporal_weights_start: float = 1.0,
        ar_temporal_weights_warmup: int = 0,
        ar_temporal_weights_ramp_stop: int = 0,
        ar_unrol_len_probablity_decay_start: int = 0,
        ar_unrol_len_probablity_decay_stop: int = 0,
        ar_unrol_len_decay_method: str = "beta",
        ar_unrol_len_start_horizon_len: int = 1,
        # RLRP-767: recurrent-state threading policy for stateful AR children (GRU/LSTM).
        # Options: legacy, sliding_window_reset, single_frame_carry. The base keeps
        # ``legacy`` (bit-exact historical behaviour, and the load fallback for pre-RLRP-767
        # checkpoints); the recurrent children (GRU/LSTM) override the constructor default to
        # the best-practice ``single_frame_carry``. Ignored by stateless children (TCN/MLP-AR).
        ar_recurrent_memory_mode: str = "legacy",
        decoder_receive_ms_hidden_size=False,
        description: Optional[str] = None,
        feature_geometry=None,
        internal_orientation=None,
        orientation_singlestep_slots=None,
    ):
        self.decoder_receive_ms_hidden_size = decoder_receive_ms_hidden_size

        # .... Dual-head decoder base (direct inheritance) ........................................
        # Build the single-step dual-head decoder exactly as before by delegating to
        # ``WeightedMultiStepDualHeadMLP.__init__`` directly (no longer through the iterator).
        # This runs ``_build_network`` with the AR dual-head settings and applies the first
        # dtype cast.
        WeightedMultiStepDualHeadMLP.__init__(
            self,
            in_size,
            out_size,
            device,
            singlestep_obs_len=singlestep_obs_len,
            singlestep_act_len=singlestep_act_len,
            history_len=history_len,
            horizon_len=horizon_len,
            temporal_weights=1.0,
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
            ss_composite_loss_weight=ss_composite_loss_weight,
            ms_composite_loss_weight=ms_composite_loss_weight,
            dual_head_shared_input_layer=True,
            enable_multistep_head=False,
            enable_auto_loss_weighting=False,
            ms_head_dropout=0.0,
            ms_head_num_layers=0,
            ss_head_num_layers=ss_head_num_layers,
            description=description,
            feature_geometry=feature_geometry,
            internal_orientation=internal_orientation,
            orientation_singlestep_slots=orientation_singlestep_slots,
        )

        # .... Compounded-prediction AR iterator setup ............................................
        # Initialize the AR state and schedulers (the iterator mixin no longer runs a
        # base ``__init__``); this also applies the final dtype cast.
        self._setup_compounded_prediction_iterator(
            horizon_len=horizon_len,
            model_use_double_precision=model_use_double_precision,
            ar_enabled=ar_enabled,
            receive_sequence_batch=receive_sequence_batch,
            ar_train_horizon_unroll=ar_train_horizon_unroll,
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
        )

        # ``single_frame_carry`` feeds a length-1 window on subsequent unroll steps,
        # so the many-to-many-to-one decoder (which flattens ``history_len`` recurrent
        # states) is structurally incompatible with it.
        if (
            self.ar_recurrent_memory_mode == self.AR_MEM_MODE_SINGLE_FRAME_CARRY
            and self.decoder_receive_ms_hidden_size
        ):
            raise ValueError(
                "ar_recurrent_memory_mode='single_frame_carry' is incompatible with "
                "decoder_receive_ms_hidden_size=True: the newest-frame-only feed "
                "produces a length-1 recurrent sequence, which cannot fill the "
                "many-to-many-to-one decoder's flattened (history_len x hidden) input."
            )

    def _ar_recurrent_step_inputs(self, x: Tensor) -> Tuple[Tensor, bool]:
        """Select the encoder input window and recurrent-state carry policy for one AR unroll step.

        Honours ``ar_recurrent_memory_mode``. The recurrent children (GRU / LSTM)
        call this at the top of ``_forward_AR_encoder`` while horizon unrolling is
        active and thread ``ar_memory`` according to the returned ``carry_state``
        flag. ``x`` is the composed window ``(..., MS, O+A)``.

        Returns a tuple ``(encoder_input, carry_state)``.
        """
        mode = self.ar_recurrent_memory_mode
        if mode == self.AR_MEM_MODE_LEGACY:
            return x, True
        if mode == self.AR_MEM_MODE_SLIDING_RESET:
            return x, False
        if mode == self.AR_MEM_MODE_SINGLE_FRAME_CARRY:
            # First step (fresh state) warms up on the full history window; every
            # subsequent step carries the state and feeds only the newest frame.
            if self.ar_memory is None:
                return x, True
            return x[..., -1:, :], True
        raise ValueError(
            f"Unknown ar_recurrent_memory_mode={mode!r}. "
            f"Expected one of {self.AR_RECURRENT_MEMORY_MODES}."
        )

    @abstractmethod
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
        super()._build_network(
            num_layers,
            in_size,
            hid_size,
            out_size,
            ensemble_size,
            activation_fn_cfg,
            deterministic,
            learn_logvar_bounds,
            instanciate_logvar_bound_module,
            dropout,
        )

    def _forward_ensemble(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # (Standby) TMP Quickhack: ensemble not supported yet
        raise NotImplementedError("Support for ensemble not implemented yet!")
        return self._default_forward(x, only_elite=False)

    def _default_forward(
        self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        e_out = self._default_forward_AR_encoder(x)

        # Run the hidden layers (the backbone) in training mode
        e_out = self.hidden_layers(e_out)

        if self.deterministic:
            ss_mean = self.deploy_head_mean(e_out)
            ss_logvar = None
        else:
            d_mean_and_logvar = self.deploy_head_mean_and_logvar(e_out)

            ss_mean_split, ss_logvar_split = torch.chunk(
                d_mean_and_logvar, chunks=2, dim=-1
            )

            ss_mean = self.deploy_head_mean(ss_mean_split)
            ss_logvar = self.deploy_head_logvar(ss_logvar_split)
            ss_logvar = self._get_deploy_logvar_bound_layer()(ss_logvar)

        # RLRP-736 bespoke-forward plan §3.4: decode the SS deploy-head raw
        # attitude slot(s) to a unit quaternion by construction. The SS decoder is
        # inherited from ``WeightedMultiStepDualHeadMLP`` (architecture B), so the
        # widened ``deploy_head_mean`` + ``_apply_ss_orientation_output_decoding``
        # cover the AR family with no per-child change (no-op / bit-exact when OFF;
        # active path is deterministic-only).
        ss_mean = self._apply_ss_orientation_output_decoding(ss_mean)

        return ss_mean, ss_logvar

    def _default_deploy_head(
        self, x: torch.Tensor, only_elite: bool = True, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        e_out = self._default_forward_AR_encoder(x)

        # Run the hidden layers (the backbone) in deploy mode
        self._maybe_toggle_layers_use_only_elite(only_elite)
        e_out = self.hidden_layers(e_out)
        self._maybe_toggle_layers_use_only_elite(only_elite)

        if self.deterministic:
            ss_mean = self.deploy_head_mean(e_out)
            ss_logvar = None
        else:
            d_mean_and_logvar = self.deploy_head_mean_and_logvar(e_out)

            ss_mean_split, ss_logvar_split = torch.chunk(
                d_mean_and_logvar, chunks=2, dim=-1
            )

            ss_mean = self.deploy_head_mean(ss_mean_split)
            ss_logvar = self.deploy_head_logvar(ss_logvar_split)
            ss_logvar = self._get_deploy_logvar_bound_layer()(ss_logvar)

        # RLRP-736 bespoke-forward plan §3.4: decode the SS deploy-head raw
        # attitude slot(s) to a unit quaternion by construction (inherited B
        # decoder; no-op / bit-exact when OFF). This is the path the real robot
        # consumes AND the per-step output of the CP free-running unroll (each
        # unroll step re-enters here), so the AR feedback is a valid rotation.
        ss_mean = self._apply_ss_orientation_output_decoding(ss_mean)

        # RLRP-736 Item 1 (SS ≡ CP(horizon=1); §2.2A): sign-align the standalone
        # AR-model SS/deploy exposure to the LAST obs frame of the incoming history
        # ``x`` (``⟨q_pred, q_ref⟩ ≥ 0`` + L2-projection to S^3). Applied strictly
        # last; this is the seam the test-time rollout deployer reaches for the
        # ``ms2ss`` AR models (``sample_1d -> _forward_propagation -> self.deploy``)
        # and is NOT covered by the shared CP splice for a single-step deploy.
        # No-op / bit-exact for ``quaternion_legacy`` and the no-attitude cases.
        ss_mean = self._apply_deploy_attitude_continuity(ss_mean, x)

        return ss_mean, ss_logvar

    def _default_forward_AR_encoder(self, x: Tensor) -> Tensor:

        x = self.reshape_tensor_to_ms_f_dim(x)

        # RLRP-736 bespoke-forward plan §3.4: encode the attitude input slot of
        # EACH single-step block ``(..., MS, F)`` BEFORE the specialized AR encoder
        # (GRU/LSTM/TCN cell) so it consumes the continuous internal rep, not the
        # raw discontinuous quaternion. No-op / bit-exact when OFF; the child
        # encoder ``input_size`` is widened by ``_ss_block_in_extra`` to match.
        x = self._apply_ss_orientation_input_encoding(x)

        # Quick-hack setup for ensemble (ref task RLRP-456)
        if x.ndim == 4 and self.num_members == 1:
            x = x.squeeze(0)
        #     if self.ar_memory is not None:
        #         self.ar_memory = self.ar_memory.squeeze(1)

        self._ar_step_dimension_validation(x)

        x = x.to(
            dtype=self.model_dtype
        )  # ToDo: update with new self._maybe_toggle_layers_use_only_elite()

        # Note: expect x of shape (..., MS, F) with MS multistrep history length and F feature length
        return self._forward_AR_encoder(x)

    @abstractmethod
    def _forward_AR_encoder(self, x: Tensor) -> Tensor: ...

    def _ar_step_dimension_validation(self, x: Tensor):
        if not self.training:
            pass  # Note: this is just for debugging

        if self.training and x.ndim != 3:
            raise ValueError(
                f"Expected 3D input after preprocessing when in training mode, got shape {x.shape}"
            )
        elif not self.training and x.ndim == 2:
            raise ValueError(
                f"Expected 3D input after preprocessing when in eval mode, got shape {x.shape}"
            )

        # (☕MINOR) ToDo: validate if still usefull
        if self.ar_memory is not None and x.shape[0] != self.ar_memory.shape[1]:
            raise ValueError(
                f"Batch size mismatch: {x.shape[0]=}, "
                f"{self.ar_memory.shape[1]=}. Reference {x.shape=} and {self.ar_memory.shape}"
            )

    @torch.compiler.disable
    def _deterministic_loss(
        self, model_in: torch.Tensor, ss_target: torch.Tensor, reduce=True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        losses = w1 * ms_t1_losses + w2 * ms_losses
        """

        meta = {}
        model_in, ss_target = self._setup_loss_input(model_in, ss_target)

        # ToDo: assess >> should probably use the forward(...) method instead of the deploy(...) one
        ss_pred_mean, _ = self.deploy(model_in, only_elite=False)

        if self.mae_loss:
            ss_losses = F.l1_loss(ss_pred_mean, ss_target, reduction="none")
        else:
            ss_losses = F.mse_loss(ss_pred_mean, ss_target, reduction="none")

        # .... Apply feature weight ...............................................................
        ss_losses = self.apply_next_obs_feature_weights(
            ss_losses,
            log_space=False,
            pre_process_weight=lambda x: self.multistep_to_singlestep_next_obs_adapter(
                x
            ),
        )

        # ==== Composite loss =====================================================================
        # Reduce over feature dim
        # Sum of log over feature dim imply they are independent random variables
        # e.g., from a multi-variate distribution
        ss_losses = ss_losses.sum(2, keepdim=True)
        ss_losses = self.ss_composite_loss_weight * ss_losses
        # A7 (RLRP-788): gate diagnostic ``meta`` writes behind the meta-collection kill-switch (RLRC meta-collection kill-switch `.junie` plan, ``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
        if self._enable_meta_collection:
            meta["singlestep_loss"] = ss_losses.detach().mean().item()

        losses = ss_losses

        if reduce:
            losses = reduce_deterministic_compose_loss(losses)

        # RLRP-751 (task T5): compose / collect the AR deploy-head per-feature
        # geometry term (no-op / bit-neutral unless the term is active).
        losses = self._compose_ar_feature_geometry(ss_pred_mean, ss_target, losses, meta)

        return losses, meta

    def _compose_ar_feature_geometry(
        self,
        ss_pred_mean: torch.Tensor,
        ss_target: torch.Tensor,
        losses: torch.Tensor,
        meta: Dict[str, Any],
    ) -> torch.Tensor:
        """Compose / collect the AR deploy-head per-feature geometry term.

        RLRP-751 (task T5), inline successor of the removed
        ``_stash_ar_feature_loss_point``. ``AbstractMS2SSAutoRegressive`` has ONLY
        a single-step (deploy) head, and its free-running unroll is driven by
        ``CompoundedPredictionMultiStepIterator.loss``, which calls this loss seam
        ONCE per compounded-prediction unroll step.

        * DURING the AR unroll (``_cp_feature_collect_active``): the inline SS term
          is SUPPRESSED and each step's deploy ``(point, target)`` is appended to
          the iterator's ``_cp_feature_points`` accumulator; the iterator composes
          the *compounded* ``feature_geom_loss_cp`` term ONCE after the unroll
          (attitude drift compounding along the free-running unroll — the
          resilient-controller failure mode).
        * OTHERWISE (``ar_train_horizon_unroll`` off, or the model called directly
          without the iterator loop): compose the single-step ``t=1``
          ``feature_geom_loss`` SS term inline, exactly as every other family.

        No-op / bit-neutral unless the term is active. ``ss_pred_mean`` is the
        point produced by ``deploy``; ``ss_target`` is already the single-step
        target.
        """
        if not self._feature_loss_active():
            return losses
        if getattr(self, "_cp_feature_collect_active", False):
            self._cp_feature_points.append((ss_pred_mean, ss_target))
            return losses
        return self._compose_feature_geometry(losses, meta, ss=(ss_pred_mean, ss_target))

    def _logvar_bound_penalty_specs(self):
        """Deploy/SS bound only (this AR single-step model has no separate encoder NLL
        bound). Identity adapter, matching the historical in-path SS ``bound_losses``
        call (RLRP-718).
        """
        dep = self._get_deploy_logvar_bound_layer()
        return [(dep, None)] if dep is not None else []

    @torch.compiler.disable
    def _probabilistic_loss(
        self, model_in: torch.Tensor, ss_target: torch.Tensor, reduce: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        losses = ss_nll_losses + ms_nll_losses
        """
        meta = {}
        model_in, ss_target = self._setup_loss_input(model_in, ss_target)

        # ToDo: assess >> should probably use the forward(...) method instead of the deploy(...) one
        ss_pred_mean, ss_pred_logvar = self.deploy(model_in, only_elite=False)

        # .... Horizon at t=1 NLL loss (SS head)...................................................
        ss_distribution = self._to_distribution(ss_pred_mean, ss_pred_logvar)
        ss_nll_losses = -ss_distribution.log_prob(ss_target)

        # .... logvar bound penalty: now added ONCE on the composite (RLRP-718) ...................

        # .... Apply feature weight ...............................................................
        ss_nll_losses = self.apply_next_obs_feature_weights(
            ss_nll_losses,
            log_space=True,
            pre_process_weight=lambda x: self.multistep_to_singlestep_next_obs_adapter(
                x
            ),
        )

        # ==== Composite loss =====================================================================
        # Reduce over feature dim
        # Sum of log over feature dim imply they are independent random variables
        # e.g., from a multi-variate distribution
        ss_nll_losses = ss_nll_losses.mean(2, keepdim=True)
        ss_nll_losses = self.ss_composite_loss_weight * ss_nll_losses
        if self._enable_meta_collection:
            meta["singlestep_loss"] = ss_nll_losses.detach().mean().item()

        losses = ss_nll_losses

        # .... Standalone fixed-coefficient logvar bound penalty (RLRP-718) .......................
        losses = losses + self._logvar_bound_penalty()

        if reduce:
            losses = reduce_probabilistic_compose_loss(losses)

        # RLRP-751 (task T5): compose / collect the AR deploy-head per-feature
        # geometry term (distribution mean, never a sample). No-op / bit-neutral
        # unless the term is active.
        losses = self._compose_ar_feature_geometry(ss_pred_mean, ss_target, losses, meta)

        return losses, meta

    def _save_state_dict(self) -> dict[str, list[int] | dict[str, Any] | Any]:
        model_dict = super()._save_state_dict()
        model_dict["ar_memory"] = self.ar_memory  # GRU hidden state

        model_dict["model_config"][
            "decoder_receive_ms_hidden_size"
        ] = self.decoder_receive_ms_hidden_size

        model_dict["model_config"][
            "ar_recurrent_memory_mode"
        ] = self.ar_recurrent_memory_mode

        return model_dict

    def _load_state_dict(self, model_dict: dict[str, list[int] | dict[str, Any] | Any]) -> None:
        # .... Validate required keys before mutating any state ...................................
        if "ar_memory" not in model_dict:
            raise KeyError(
                f"{type(self).__name__}._load_state_dict: missing keys in "
                f"model_dict: ar_memory"
            )

        model_config = model_dict["model_config"]
        if "decoder_receive_ms_hidden_size" not in model_config:
            raise KeyError(
                f"{type(self).__name__}._load_state_dict: missing keys in "
                f"model_dict['model_config']['decoder_receive_ms_hidden_size']"
            )

        # .... Validate AR-specific derived state consistency .....................................
        loaded_ar_memory = model_dict["ar_memory"]
        if loaded_ar_memory is not None and not isinstance(
            loaded_ar_memory, torch.Tensor
        ):
            raise TypeError(
                f"{type(self).__name__}._load_state_dict: 'ar_memory' must be "
                f"either None or a torch.Tensor, got {type(loaded_ar_memory).__name__}."
            )

        loaded_decoder_receive_ms_hidden_size = model_config[
            "decoder_receive_ms_hidden_size"
        ]
        current_decoder_receive_ms_hidden_size = getattr(
            self, "decoder_receive_ms_hidden_size", None
        )
        if (
            current_decoder_receive_ms_hidden_size is not None
            and current_decoder_receive_ms_hidden_size
            != loaded_decoder_receive_ms_hidden_size
        ):
            # This flag is consumed by __init__ to build the network; loading a value
            # different from the one used at construction would leave a structurally
            # stale network silently incompatible with the loaded weights.
            raise ValueError(
                f"{type(self).__name__}._load_state_dict: loaded "
                f"'decoder_receive_ms_hidden_size'={loaded_decoder_receive_ms_hidden_size} "
                f"does not match the value used at construction "
                f"({current_decoder_receive_ms_hidden_size}). The model network was "
                f"built with the constructor value and cannot be safely re-used with "
                f"a different one."
            )

        super()._load_state_dict(model_dict)

        self.ar_memory = loaded_ar_memory
        self.decoder_receive_ms_hidden_size = loaded_decoder_receive_ms_hidden_size
        # ``ar_recurrent_memory_mode`` was added after the first AR checkpoints, so a
        # missing key falls back to the historical ``legacy`` semantics.
        self.ar_recurrent_memory_mode = model_config.get(
            "ar_recurrent_memory_mode", self.AR_MEM_MODE_LEGACY
        )
        return None
