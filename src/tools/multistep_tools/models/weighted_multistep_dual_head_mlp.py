# coding=utf-8
from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F
from mbrl.models import truncated_normal_init

from tools.multistep_tools.models.exponential_family_mlp_utils import (
    LogvarBoundLayer,
    create_activation_,
    create_linear_layer_,
    create_logvar_bound_layer,
    make_layer_bloc_seq,
    zero_init_residual_blocks_,
)

from tools.multistep_tools.models import AbstractFeatureWeightedMultiStepMLP
from tools.multistep_tools.models.utils import (
    reduce_deterministic_compose_loss,
    reduce_probabilistic_compose_loss,
)


class WeightedMultiStepDualHeadMLP(AbstractFeatureWeightedMultiStepMLP):

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        singlestep_obs_len: int,
        singlestep_act_len: int,
        history_len: int,
        horizon_len: int,
        temporal_weights: Union[float, Tuple[float, ...]] = 1.0,
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
        logvar_bound_grad_clip: Optional[float] = None,
        auto_weighting_noise_model: str = "gaussian",
        auto_weighting_scheme: str = "tempered_likelihood",
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        distribution_name: str = "gaussian",
        mae_loss: bool = True,
        model_use_double_precision: bool = False,
        ss_composite_loss_weight: float = 1.0,
        ms_composite_loss_weight: float = 1.0,
        dropout: float = 0.0,
        ms_temporal_weighting_mode: str = "discounted-sum",
        dual_head_shared_input_layer: bool = False,
        enable_multistep_head: bool = True,
        ms_head_dropout: float = 0.0,
        ms_head_num_layers: int = 1,
        ss_head_num_layers: int = 1,
        residual_form: str = "identity_v2",
        layer_bloc=None,  # RLRP-768 per-region layer-bloc selector (forwarded)
        enable_auto_loss_weighting: bool = True,
        ms_probabilities_reduction: str = "independent",
        ms_energy_beta: Union[float, str] = "learned",
        ms_energy_axis: str = "step",
        train_time_domain_randomization: Optional[
            Union[Dict, omegaconf.DictConfig]
        ] = None,
        description: Optional[str] = None,
        feature_geometry=None,
        internal_orientation=None,
        orientation_singlestep_slots=None,
    ):

        self.singlestep_obs_len = singlestep_obs_len
        self.singlestep_act_len = singlestep_act_len
        self.dual_head_shared_input_layer = dual_head_shared_input_layer
        self.enable_multistep_head = enable_multistep_head

        assert ss_head_num_layers >= 1
        self.ss_head_num_layers = ss_head_num_layers

        super().__init__(
            in_size,
            out_size,
            device,
            singlestep_obs_len=singlestep_obs_len,
            singlestep_act_len=singlestep_act_len,
            history_len=history_len,
            horizon_len=horizon_len,
            temporal_weights=temporal_weights,
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
            logvar_bound_grad_clip=logvar_bound_grad_clip,
            auto_weighting_noise_model=auto_weighting_noise_model,
            auto_weighting_scheme=auto_weighting_scheme,
            activation_fn_cfg=activation_fn_cfg,
            distribution_name=distribution_name,
            mae_loss=mae_loss,
            model_use_double_precision=model_use_double_precision,
            dropout=dropout,
            ms_head_dropout=ms_head_dropout,
            ms_head_num_layers=ms_head_num_layers,
            residual_form=residual_form,
            layer_bloc=layer_bloc,
            ms_temporal_weighting_mode=ms_temporal_weighting_mode,
            enable_auto_loss_weighting=enable_auto_loss_weighting,
            ms_probabilities_reduction=ms_probabilities_reduction,
            ms_energy_beta=ms_energy_beta,
            ms_energy_axis=ms_energy_axis,
            train_time_domain_randomization=train_time_domain_randomization,
            description=description,
            feature_geometry=feature_geometry,
            internal_orientation=internal_orientation,
            orientation_singlestep_slots=orientation_singlestep_slots,
        )

        # .... loss related .......................................................................
        # Note: force cast to float
        self.ss_composite_loss_weight = float(ss_composite_loss_weight)
        self.ms_composite_loss_weight = float(ms_composite_loss_weight)

        # .... Final build step ...................................................................
        self.build_network_post(learn_logvar_bounds)
        self.to(dtype=self.model_dtype)

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

        if self.dual_head_shared_input_layer:
            DUAL_HEAD_SPLIT_OUT_LAYER_SIZE = hid_size
            DUAL_HEAD_SPLIT_MS_IN_LAYER_SIZE = DUAL_HEAD_SPLIT_OUT_LAYER_SIZE
            DUAL_HEAD_SPLIT_SS_IN_LAYER_SIZE = DUAL_HEAD_SPLIT_OUT_LAYER_SIZE
        else:
            # DUAL_HEAD_SPLIT_OUT_LAYER_SIZE = (hid_size // 2) * 2 # Narow splitable out layer size
            DUAL_HEAD_SPLIT_OUT_LAYER_SIZE = hid_size * 2  # Wide split out layer size
            DUAL_HEAD_SPLIT_MS_IN_LAYER_SIZE = int(
                DUAL_HEAD_SPLIT_OUT_LAYER_SIZE * 0.75
            )
            DUAL_HEAD_SPLIT_SS_IN_LAYER_SIZE = (
                DUAL_HEAD_SPLIT_OUT_LAYER_SIZE - DUAL_HEAD_SPLIT_MS_IN_LAYER_SIZE
            )

        # 💎 Both head needs big hiddin layer size to perform
        MS_HEAD_HIDDEN_SIZE = DUAL_HEAD_SPLIT_MS_IN_LAYER_SIZE
        SS_HEAD_HIDDEN_SIZE = DUAL_HEAD_SPLIT_SS_IN_LAYER_SIZE

        create_activation = partial(create_activation_, activation_fn_cfg)

        create_linear_layer = partial(create_linear_layer_, ensemble_size)

        # .... Layer-bloc resolvers (RLRP-768: per-region type / layer_norm / dropout) ...........
        # Three independent resolvers: encoder/hidden (region "encoder", uses `dropout`),
        # MS-head (region "ms_head", uses `ms_head_dropout`) and SS deploy-head
        # (region "ss_head", no dropout, as before).
        enc_res_kind, make_enc_res = self._layer_bloc_factory(
            "encoder", ensemble_size, create_activation, dropout
        )
        ms_res_kind, make_ms_res = self._layer_bloc_factory(
            "ms_head", ensemble_size, create_activation, self.ms_head_dropout
        )
        ss_res_kind, make_ss_res = self._layer_bloc_factory(
            "ss_head", ensemble_size, create_activation, 0.0
        )

        # ==== Encoder ============================================================================
        if num_layers >= 2:
            hidden_layers = [
                nn.Sequential(
                    # nn.Dropout(p=dropout),
                    create_linear_layer(in_size, hid_size),
                    create_activation(),
                )
            ]

            # Stack residual layer
            for i in range(num_layers - 2):
                hidden_layers.append(
                    make_layer_bloc_seq(
                        enc_res_kind, make_enc_res, hid_size, create_activation, dropout
                    )
                )

            # Dual head split
            hidden_layers.append(
                nn.Sequential(
                    nn.Dropout(p=dropout),
                    create_linear_layer(hid_size, DUAL_HEAD_SPLIT_OUT_LAYER_SIZE),
                    create_activation(),
                )
            )
        else:
            hidden_layers = [
                nn.Sequential(
                    nn.Dropout(p=dropout),
                    create_linear_layer(in_size, DUAL_HEAD_SPLIT_OUT_LAYER_SIZE),
                    create_activation(),
                )
            ]

        self.hidden_layers = nn.Sequential(*hidden_layers)

        # ==== Decoder ============================================================================
        # .... Multi-Step head ....................................................................
        if self.enable_multistep_head:
            ms_mean_layers = []
            ms_logvar_layers = []

            # Note: Dropout is only on the first ms head layer on purposes and use a dedicated
            # parammeter 'ms_head_dropout' instead of the 'dropout' one which is used for
            # the encoder hiden layers.
            if deterministic:
                ms_mean_layers.append(
                    nn.Sequential(
                        nn.Dropout(p=self.ms_head_dropout),
                        create_linear_layer(
                            DUAL_HEAD_SPLIT_MS_IN_LAYER_SIZE, MS_HEAD_HIDDEN_SIZE
                        ),
                        create_activation(),
                    )
                )
            else:
                self.mean_and_logvar = nn.Sequential(
                    nn.Dropout(p=self.ms_head_dropout),
                    create_linear_layer(
                        DUAL_HEAD_SPLIT_MS_IN_LAYER_SIZE, 2 * MS_HEAD_HIDDEN_SIZE
                    ),
                    create_activation(),
                )

            # Stack multi-step mean and logvar residual layers
            for i in range(self.ms_head_num_layers - 1):
                ms_mean_layers.append(
                    make_layer_bloc_seq(
                        ms_res_kind, make_ms_res, MS_HEAD_HIDDEN_SIZE,
                        create_activation, self.ms_head_dropout,
                    )
                )

                if not self.deterministic:
                    ms_logvar_layers.append(
                        make_layer_bloc_seq(
                            ms_res_kind, make_ms_res, MS_HEAD_HIDDEN_SIZE,
                            create_activation, self.ms_head_dropout,
                        )
                    )

            # Finale multi-step mean layer
            ms_mean_layers.append(
                nn.Sequential(
                    create_linear_layer(MS_HEAD_HIDDEN_SIZE, out_size),
                )
            )
            self.mean_layer = nn.Sequential(*ms_mean_layers)

            if not self.deterministic:
                # Finale multi-step logvar layer
                ms_logvar_layers.append(
                    nn.Sequential(
                        create_linear_layer(MS_HEAD_HIDDEN_SIZE, out_size),
                    )
                )
                self.logvar_layer = nn.Sequential(*ms_logvar_layers)

        # .... Deploy head (i.e., single-step head) ...............................................
        self.deploy_head_mean_adapter = None
        self.deploy_head_logvar_adapter = None

        deploy_mean_layers = []
        deploy_logvar_layers = []

        # Note: No dropout on deploy head
        if deterministic:
            deploy_mean_layers.append(
                nn.Sequential(
                    create_linear_layer(
                        DUAL_HEAD_SPLIT_SS_IN_LAYER_SIZE, SS_HEAD_HIDDEN_SIZE
                    ),
                    create_activation(),
                )
            )
        else:
            self.deploy_head_mean_and_logvar = nn.Sequential(
                create_linear_layer(
                    DUAL_HEAD_SPLIT_SS_IN_LAYER_SIZE, 2 * SS_HEAD_HIDDEN_SIZE
                ),
                create_activation(),
            )

        # Stack deploy mean and logvar residual layers
        for i in range(self.ss_head_num_layers - 1):
            deploy_mean_layers.append(
                make_layer_bloc_seq(
                    ss_res_kind, make_ss_res, SS_HEAD_HIDDEN_SIZE, create_activation, 0.0
                )
            )
            if not self.deterministic:
                deploy_logvar_layers.append(
                    make_layer_bloc_seq(
                        ss_res_kind, make_ss_res, SS_HEAD_HIDDEN_SIZE, create_activation, 0.0
                    )
                )

        # Finale deploy mean layer
        # RLRP-736 bespoke-forward plan §3.2/§5.3: the DEDICATED SS (deploy) head
        # emits ONE single-step obs block; when the by-construction rotation rep is
        # active its attitude slot(s) are widened to
        # ``internal_rep_out_width(rep, external_width)`` raw
        # numbers, so the final layer emits ``_ss_trunk_out_len`` (== singlestep_obs_len
        # when the wiring is OFF => bit-exact) and ``_default_deploy_head`` decodes
        # the raw slot(s) back to a 4-D unit quaternion.
        deploy_mean_layers.append(
            nn.Sequential(
                create_linear_layer(SS_HEAD_HIDDEN_SIZE, self._ss_trunk_out_len),
            )
        )
        self.deploy_head_mean = nn.Sequential(*deploy_mean_layers)

        if not self.deterministic:
            # Finale deploy logvar layer
            deploy_logvar_layers.append(
                nn.Sequential(
                    create_linear_layer(SS_HEAD_HIDDEN_SIZE, self.singlestep_obs_len),
                )
            )
            self.deploy_head_logvar = nn.Sequential(*deploy_logvar_layers)

        # .........................................................................................
        self.apply(truncated_normal_init)
        # Re-zero the identity residual blocks' last linear AFTER the global
        # truncated-normal init so they start as an exact identity (y == x).
        zero_init_residual_blocks_(self)
        self.to(self.device)
        return None

    def build_network_post(self, learn_logvar_bounds: bool) -> None:

        if not self.deterministic:
            # .... Encoder logvar bound ...........................................................
            if self.enable_multistep_head:
                self.logvar_layer.add_module(  # Make the module fetchable by name
                    "logvar_bound",
                    create_logvar_bound_layer(
                        self.out_size,
                        learn_logvar_bounds,
                        # bound_min=weighted_bound_min,
                        # bound_max=weighted_bound_max,
                        bound_min_init=torch.tensor(
                            self._LOGVAR_MIN_BOUND_INIT, device=self.device
                        ),
                        bound_max_init=torch.tensor(
                            self._LOGVAR_MAX_BOUND_INIT, device=self.device
                        ),
                        grad_clip=self.logvar_bound_grad_clip,
                    ),
                )

            # .... Decoder logvar bound ...........................................................
            self.deploy_head_logvar.add_module(  # Make the module fetchable by name
                "logvar_bound",
                create_logvar_bound_layer(
                    self.singlestep_obs_len,
                    learn_logvar_bounds,
                    bound_min_init=torch.tensor(
                        self._LOGVAR_MIN_BOUND_INIT, device=self.device
                    ),
                    bound_max_init=torch.tensor(
                        self._LOGVAR_MAX_BOUND_INIT, device=self.device
                    ),
                    grad_clip=self.logvar_bound_grad_clip,
                ),
            )

        return None

    def _get_deploy_logvar_bound_layer(self) -> Optional[LogvarBoundLayer]:
        return (
            self.deploy_head_logvar.get_submodule("logvar_bound")
            if not self.deterministic
            else None
        )

    def _logvar_bound_penalty_specs(self):
        """Encoder/MS bound (active only when the multistep head is built) + deploy/SS bound
        (always in use; the SS term is always computed). Both use the identity adapter,
        matching the historical in-path ``bound_losses`` calls. Deduped by identity in the
        shared helper (RLRP-718).
        """
        specs = []
        if getattr(self, "enable_multistep_head", True) and not self.deterministic:
            enc = self._get_logvar_bound_layer()
            if enc is not None:
                specs.append((enc, None))
        dep = self._get_deploy_logvar_bound_layer()
        if dep is not None:
            specs.append((dep, None))
        return specs

    def _default_forward(
        self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if self.enable_multistep_head:
            self._maybe_toggle_layers_use_only_elite(only_elite)

            x = self._maybe_cast_to_model_dtype(x)

            if not self.training:
                # Run backbone in loss at train time.
                # RLRP-736 bespoke-forward plan §3.2: at inference the backbone runs
                # on the RAW input here, so the attitude input slot(s) must be
                # encoded to the internal rep first (no-op / bit-exact when OFF). In
                # the training path the loss encodes ``model_in`` once before running
                # ``hidden_layers`` and passes the post-backbone features in, so this
                # branch is skipped and there is no double-encode.
                x = self._apply_orientation_input_encoding(x)
                x = self.hidden_layers(x)

            if self.dual_head_shared_input_layer:
                ms_head = x
            else:
                _, ms_head1, ms_head2, ms_head3 = torch.chunk(x, chunks=4, dim=-1)
                ms_head = torch.concatenate([ms_head1, ms_head2, ms_head3], dim=-1)

            if self.deterministic:
                ms_mean_head = self.mean_layer(ms_head)
                ms_logvar_head = None
            else:
                ms_mean_and_logvar = self.mean_and_logvar(ms_head)

                # Architecture 1
                ms_mean_split, ms_logvar_split = torch.chunk(
                    ms_mean_and_logvar, chunks=2, dim=-1
                )
                ms_mean_head = self.mean_layer(ms_mean_split)
                ms_logvar_head = self.logvar_layer(ms_logvar_split)
                ms_logvar_head = self._get_logvar_bound_layer()(ms_logvar_head)

                # Architecture 2
                # mean_head = self.mean_layer(mean_and_logvar)
                # logvar_head = self.logvar_layer(mean_and_logvar)

            # RLRP-736 bespoke-forward plan §3.2: decode the MS forecast head's raw
            # composed-window attitude slot(s) back to unit quaternion(s) by
            # construction (no-op / bit-exact when the wiring is OFF; the active
            # path is deterministic-only, the probabilistic head is gated at setup).
            ms_mean_head = self._apply_orientation_output_decoding(ms_mean_head)

            self._maybe_toggle_layers_use_only_elite(only_elite)
            return ms_mean_head, ms_logvar_head
        else:
            return None, None

    def deploy(
        self, x: torch.Tensor, only_elite: bool = True, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        x = self._maybe_cast_to_model_dtype(x)

        if not self.training:
            # Run backbone in loss at train time.
            # RLRP-736 bespoke-forward plan §3.2: at inference the backbone runs on
            # the RAW input here, so encode the attitude input slot(s) first (no-op
            # / bit-exact when OFF). The training loss encodes ``model_in`` once and
            # passes post-backbone features in => this branch is skipped, no
            # double-encode.
            self._maybe_toggle_layers_use_only_elite(only_elite)
            x = self._apply_orientation_input_encoding(x)
            x = self.hidden_layers(x)
            self._maybe_toggle_layers_use_only_elite(only_elite)

        if self.dual_head_shared_input_layer:
            ss_head = x
        else:
            ss_head, *_ = torch.chunk(x, chunks=4, dim=-1)

        ss_mean, ss_logvar = self._default_deploy_head(ss_head, only_elite)
        return ss_mean, ss_logvar

    def _default_deploy_head(
        self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._maybe_toggle_layers_use_only_elite(only_elite)

        x = self._maybe_cast_to_model_dtype(x)

        if self.deterministic:
            ss_mean = self.deploy_head_mean(x)
            ss_logvar = None
        else:
            d_mean_and_logvar = self.deploy_head_mean_and_logvar(x)

            ss_mean_split, ss_logvar_split = torch.chunk(
                d_mean_and_logvar, chunks=2, dim=-1
            )

            ss_mean = self.deploy_head_mean(ss_mean_split)
            ss_logvar = self.deploy_head_logvar(ss_logvar_split)
            ss_logvar = self._get_deploy_logvar_bound_layer()(ss_logvar)

        # RLRP-736 bespoke-forward plan §3.2/§5.3: decode the DEDICATED SS (deploy)
        # head's raw single-step attitude slot(s) back to a 4-D unit quaternion by
        # construction, so the deploy output the real robot consumes is a valid
        # rotation (no-op / bit-exact when the wiring is OFF; active path is
        # deterministic-only).
        ss_mean = self._apply_ss_orientation_output_decoding(ss_mean)

        self._maybe_toggle_layers_use_only_elite(only_elite)
        return ss_mean, ss_logvar

    @torch.compiler.disable
    def _deterministic_loss(
        self, model_in: torch.Tensor, target: torch.Tensor, reduce=True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        losses = ss_nll_losses + ms_nll_losses
        """
        meta = {}
        model_in, target = self._setup_loss_input(model_in, target)
        ss_target = self.multistep_to_singlestep_next_obs_adapter(target)

        # Run shared hidden layers (the backbone) in loss at train time to prevent calling
        # it twice in forward and deploy methods.
        # RLRP-736 bespoke-forward plan §3.2: encode the attitude input slot(s) of
        # the RAW ``model_in`` ONCE here before the backbone (no-op / bit-exact when
        # OFF). ``forward``/``deploy`` receive the post-backbone features (training
        # mode) so they do not re-run ``hidden_layers`` and there is no re-encode.
        model_in = self._apply_orientation_input_encoding(model_in)
        x = self.hidden_layers(model_in)

        ss_pred_mean, _ = self.deploy(x, only_elite=False)

        if self.enable_multistep_head:
            ms_pred_mean, _ = self.forward(x, use_propagation=False)

            # .... Horizon Nll Loss (Ms Head) .....................................................
            if self.mae_loss:
                ms_losses = F.l1_loss(ms_pred_mean, target, reduction="none")
            else:
                ms_losses = F.mse_loss(ms_pred_mean, target, reduction="none")

            # .... Apply temporal weight ..........................................................
            ms_losses = self.apply_next_obs_temporal_discount_factor_weights(ms_losses)

            # .... Apply feature weight ...........................................................
            ms_losses = self.apply_next_obs_feature_weights(ms_losses)

            # .... Reduce multistep horizon .......................................................
            ms_losses = self.reduce_multistep_losses_horizon(ms_losses, probabilistic_losses=False)

        # .... Horizon at t=1 NLL loss (SS head)...................................................
        if self.mae_loss:
            ss_losses = F.l1_loss(ss_pred_mean, ss_target, reduction="none")
        else:
            ss_losses = F.mse_loss(ss_pred_mean, ss_target, reduction="none")

        # .... Apply feature weight ...............................................................
        ss_losses = self.apply_next_obs_feature_weights(
            ss_losses,
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

        if self.enable_multistep_head:
            ms_losses = ms_losses.sum(2, keepdim=True)
            ms_losses = self.ms_composite_loss_weight * ms_losses
            if self._enable_meta_collection:
                meta["horizon_loss"] = ms_losses.detach().mean().item()

            ss_losses, meta = self.composite_loss_automatic_weighting(
                ss_losses, "SS", meta
            )
            ms_losses, meta = self.composite_loss_automatic_weighting(
                ms_losses, "MS", meta
            )

            losses = ss_losses + ms_losses
        else:
            losses = ss_losses

        if reduce:
            losses = reduce_deterministic_compose_loss(losses)

        # RLRP-751 (task T5): compose the per-feature geometry term(s) INLINE
        # (replaces the removed stash/drain seam). SS = the deploy-head point;
        # MS = the composed forecast mean (legacy layout), gated on
        # ``enable_multistep_head``. Bit-neutral OFF; MS no-ops when
        # ``horizon_len == 1``.
        losses = self._compose_feature_geometry(
            losses,
            meta,
            ss=(ss_pred_mean, ss_target),
            ms_head=(
                (ms_pred_mean, target) if self.enable_multistep_head else None
            ),
            ms_head_legacy_composed_shape=True,
        )

        # .... memory management ..................................................................
        # (CRITICAL) ToDo: validate grad ok
        del model_in, target
        if self.enable_multistep_head:
            del (ms_pred_mean,)
        del ss_pred_mean, ss_target

        return losses, meta

    @torch.compiler.disable
    def _probabilistic_loss(
        self, model_in: torch.Tensor, target: torch.Tensor, reduce: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        losses = ss_nll_losses + ms_nll_losses
        """
        meta = {}
        model_in, target = self._setup_loss_input(model_in, target)
        ss_target = self.multistep_to_singlestep_next_obs_adapter(target)

        # Run shared hidden layers (the backbone) in loss at train time to prevent calling
        # it twice in forward and deploy methods.
        # RLRP-736 bespoke-forward plan §3.2: encode the attitude input slot(s) of
        # the RAW ``model_in`` ONCE here before the backbone (no-op / bit-exact when
        # OFF; the active by-construction path is deterministic-only, so this
        # probabilistic branch only ever sees the identity encode).
        model_in = self._apply_orientation_input_encoding(model_in)
        x = self.hidden_layers(model_in)

        ss_pred_mean, ss_pred_logvar = self.deploy(x, only_elite=False)

        if self.enable_multistep_head:
            ms_pred_mean, ms_pred_logvar = self.forward(x, use_propagation=False)

            # .... Horizon Nll Loss (Ms Head) .....................................................
            ms_distribution = self._to_distribution(ms_pred_mean, ms_pred_logvar)
            ms_nll_losses = -ms_distribution.log_prob(target)

            # .... logvar bound penalty: now added ONCE on the composite (RLRP-718) ...............

            # .... Apply temporal weight ..........................................................
            ms_nll_losses = self.apply_next_obs_temporal_discount_factor_weights(
                ms_nll_losses,
                log_space=True,
                temporal_mode_aware=True,
            )

            # .... Apply feature weight ...........................................................
            ms_nll_losses = self.apply_next_obs_feature_weights(
                ms_nll_losses,
                log_space=True,
            )

            # .... Reduce multistep horizon .......................................................
            ms_nll_losses = self.reduce_multistep_losses_horizon(ms_nll_losses,
                                                                 probabilistic_losses=True)

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

        if self.enable_multistep_head:
            ms_nll_losses = ms_nll_losses.mean(2, keepdim=True)
            ms_nll_losses = self.ms_composite_loss_weight * ms_nll_losses
            if self._enable_meta_collection:
                meta["horizon_loss"] = ms_nll_losses.detach().mean().item()

            # .... RLRP-528 feat: improve CompositeLossAutomaticWeighting .........................
            # ToDo: remove cdf argument (ref task RLRP-528)
            ss_nll_losses, meta = self.composite_loss_automatic_weighting(
                ss_nll_losses,
                "SS",
                meta,
                are_log_prob_losses=True,
                # cdf=ss_distribution.cdf(ss_target)
            )
            ms_nll_losses, meta = self.composite_loss_automatic_weighting(
                ms_nll_losses,
                "MS",
                meta,
                are_log_prob_losses=True,
                # cdf=ms_distribution.cdf(target)
            )
            # .................. RLRP-528 feat: improve CompositeLossAutomaticWeighting ...(end)...

            losses = ss_nll_losses + ms_nll_losses
        else:
            losses = ss_nll_losses

        # .... Standalone fixed-coefficient logvar bound penalty (RLRP-718) .......................
        losses = losses + self._logvar_bound_penalty()

        if reduce:
            losses = reduce_probabilistic_compose_loss(losses)

        # RLRP-751 (task T5): compose the per-feature geometry term(s) INLINE
        # (replaces the removed stash/drain seam). SS = the deploy-head
        # distribution mean; MS = the composed forecast distribution mean (legacy
        # layout), gated on ``enable_multistep_head``. Bit-neutral OFF; MS no-ops
        # when ``horizon_len == 1``.
        losses = self._compose_feature_geometry(
            losses,
            meta,
            ss=(ss_pred_mean, ss_target),
            ms_head=(
                (ms_pred_mean, target) if self.enable_multistep_head else None
            ),
            ms_head_legacy_composed_shape=True,
        )

        # .... memory management ..................................................................
        # (CRITICAL) ToDo: validate grad ok
        del model_in, target
        if self.enable_multistep_head:
            del (
                ms_pred_mean,
                ms_pred_logvar,
            )
        del ss_pred_mean, ss_pred_logvar, ss_target

        return losses, meta

