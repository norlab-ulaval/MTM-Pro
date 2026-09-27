# coding=utf-8
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import omegaconf
import torch
from torch.nn import functional as F
import math

from tools.multistep_tools.models import AbstractFeatureWeightedMultiStepMLP
from tools.multistep_tools.models.utils import reduce_deterministic_compose_loss, \
    reduce_probabilistic_compose_loss


class WeightedMultiStepMLP(AbstractFeatureWeightedMultiStepMLP):

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
        ms_head_dropout: float = 0.0,
        ms_head_num_layers: int = 1,
        ms_temporal_weighting_mode: str = "discounted-sum",
        enable_auto_loss_weighting: bool = True,
        ms_probabilities_reduction: str = "independent",
        ms_energy_beta: Union[float, str] = "learned",
        ms_energy_axis: str = "step",
        description: Optional[str] = None,
        feature_geometry=None,
        internal_orientation=None,
        orientation_singlestep_slots=None,
    ):
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
            ms_temporal_weighting_mode=ms_temporal_weighting_mode,
            enable_auto_loss_weighting=enable_auto_loss_weighting,
            ms_probabilities_reduction=ms_probabilities_reduction,
            ms_energy_beta=ms_energy_beta,
            ms_energy_axis=ms_energy_axis,
            description=description,
            feature_geometry=feature_geometry,
            internal_orientation=internal_orientation,
            orientation_singlestep_slots=orientation_singlestep_slots,
        )

        # Note: force cast to float
        self.ss_composite_loss_weight = float(ss_composite_loss_weight)
        self.ms_composite_loss_weight = float(ms_composite_loss_weight)

    @torch.compiler.disable
    def _deterministic_loss(
        self, model_in: torch.Tensor, target: torch.Tensor, reduce=True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        losses = w1 * ms_t1_losses + w2 * ms_losses
        """
        meta = {}
        assert model_in.ndim == target.ndim

        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)

        ss_target = self.multistep_to_singlestep_next_obs_adapter(target)

        ms_pred_mean, _ = self.forward(model_in, use_propagation=False)

        if self.mae_loss:
            ms_losses = F.l1_loss(ms_pred_mean, target, reduction="none")
        else:
            ms_losses = F.mse_loss(ms_pred_mean, target, reduction="none")

        # .... Weight loss temporaly ..............................................................
        ms_losses = self.apply_next_obs_temporal_discount_factor_weights(ms_losses)

        # .... Weight loss by feature .............................................................
        ms_losses = self.apply_next_obs_feature_weights(ms_losses)

        # .... Reduce multistep horizon ...........................................................
        ms_losses = self.reduce_multistep_losses_horizon(ms_losses, probabilistic_losses=False)

        # .... Horizon at t=1 NLL loss ............................................................
        ms_t1_pred_mean = self.deploy_head_mean_adapter(ms_pred_mean)

        if self.mae_loss:
            ms_t1_losses = F.l1_loss(ms_t1_pred_mean, ss_target, reduction="none")
        else:
            ms_t1_losses = F.mse_loss(ms_t1_pred_mean, ss_target, reduction="none")

        # .... Weight loss by feature .............................................................
        ms_t1_losses = self.apply_next_obs_feature_weights(
            ms_t1_losses,
            pre_process_weight=lambda x: self.multistep_to_singlestep_next_obs_adapter(
                x
            ),
        )

        # ==== Composite loss =====================================================================
        ms_losses = ms_losses.mean(2, keepdim=True)
        ms_t1_losses = ms_t1_losses.mean(2, keepdim=True)

        ms_losses = self.ms_composite_loss_weight * ms_losses
        ms_t1_losses = self.ss_composite_loss_weight * ms_t1_losses

        losses = ms_t1_losses + ms_losses

        # A7 (RLRP-788): gate diagnostic ``meta`` writes behind the meta-collection kill-switch (RLRC meta-collection kill-switch `.junie` plan, ``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
        if self._enable_meta_collection:
            meta["singlestep_loss"] = ms_t1_losses.detach().mean().item()
            meta["horizon_loss"] = ms_losses.detach().mean().item()

        # RLRP-751 (task T5): compose the per-feature geometry term(s) INLINE,
        # before the reduction, at the same accumulator stage as every other
        # composite term (replaces the removed stash/drain seam). SS = deploy-head
        # (t=1) mean; MS = composed forecast mean (legacy layout). Bit-neutral OFF.
        losses = self._compose_feature_geometry(
            losses,
            meta,
            ss=(ms_t1_pred_mean, ss_target),
            ms_head=(ms_pred_mean, target),
            ms_head_legacy_composed_shape=True,
        )

        if reduce:
            losses = reduce_deterministic_compose_loss(losses)

        # .... memory management ..................................................................
        # (CRITICAL) ToDo: validate grad ok
        del ms_pred_mean, ms_t1_pred_mean, model_in, target, ss_target

        return losses, meta

    @torch.compiler.disable
    def _probabilistic_loss(
        self, model_in: torch.Tensor, target: torch.Tensor, reduce: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        losses = ms_t1_nll_losses + ms_nll_losses
        """
        meta = {}
        model_in, target = self._setup_loss_input(model_in, target)
        ss_target = self.multistep_to_singlestep_next_obs_adapter(target)

        ms_pred_mean, ms_pred_logvar = self.forward(model_in, use_propagation=False)

        # .... Horizon NLL loss ...................................................................
        ms_distribution = self._to_distribution(ms_pred_mean, ms_pred_logvar)
        ms_nll_losses = -ms_distribution.log_prob(target)

        # .... logvar bound penalty: now added ONCE on the composite (RLRP-718) ...................

        # .... Apply temporal weight ..............................................................
        # RLRP-769: `temporal_mode_aware=True` dispatches the discount application on
        # `ms_temporal_weighting_mode` ('log-shift' default = legacy additive-log shift, bit-exact).
        ms_nll_losses = self.apply_next_obs_temporal_discount_factor_weights(
            ms_nll_losses, log_space=True, apply_minus_log=False, temporal_mode_aware=True
        )

        # .... Apply feature weight ...............................................................
        ms_nll_losses = self.apply_next_obs_feature_weights(
            ms_nll_losses, log_space=True, apply_minus_log=False
        )

        # .... Reduce multistep horizon ...........................................................
        ms_nll_losses = self.reduce_multistep_losses_horizon(ms_nll_losses,
                                                             probabilistic_losses=True)

        # .... Horizon at t=1 NLL loss ............................................................
        ms_t1_pred_mean = self.deploy_head_mean_adapter(ms_pred_mean)
        ms_t1_pred_logvar = self.deploy_head_logvar_adapter(ms_pred_logvar)
        ms_t1_distribution = self._to_distribution(ms_t1_pred_mean, ms_t1_pred_logvar)
        ms_t1_nll_losses = -ms_t1_distribution.log_prob(ss_target)

        # .... logvar bound penalty: now added ONCE on the composite (RLRP-718) ...................

        # .... Apply feature weight ...............................................................
        ms_t1_nll_losses = self.apply_next_obs_feature_weights(
            ms_t1_nll_losses,
            log_space=True,
            apply_minus_log=False,
            pre_process_weight=lambda x: self.multistep_to_singlestep_next_obs_adapter(
                x
            ),
        )

        # ==== Composite loss =====================================================================
        # Reduce over feature dim
        ms_nll_losses = ms_nll_losses.mean(2, keepdim=True)
        ms_t1_nll_losses = ms_t1_nll_losses.mean(2, keepdim=True)

        ms_nll_losses = self.ms_composite_loss_weight * ms_nll_losses
        ms_t1_nll_losses = self.ss_composite_loss_weight * ms_t1_nll_losses

        if self._enable_meta_collection:
            meta["horizon_at_t1_nll_losses"] = ms_t1_nll_losses.detach().mean().item()
            meta["horizon_loss"] = ms_nll_losses.detach().mean().item()

        # .... RLRP-528 feat: improve CompositeLossAutomaticWeighting .........................
        # ToDo: remove cdf argument (ref task RLRP-528)
        ms_t1_nll_losses, meta = self.composite_loss_automatic_weighting(
            ms_t1_nll_losses, "SS", meta, are_log_prob_losses=True,
            # cdf=ms_t1_distribution.cdf(ss_target)
        )
        ms_nll_losses, meta = self.composite_loss_automatic_weighting(
            ms_nll_losses, "MS", meta, are_log_prob_losses=True,
            # cdf=ms_t1_distribution.cdf(target)
        )
        # .................. RLRP-528 feat: improve CompositeLossAutomaticWeighting ...(end)...

        losses = ms_t1_nll_losses + ms_nll_losses

        # RLRP-751 (task T5): compose the per-feature geometry term(s) INLINE,
        # before the reduction, at the same (E,B,1) accumulator stage as every
        # other composite term (replaces the removed stash/drain seam). SS =
        # deploy-head (t=1) distribution mean; MS = composed forecast distribution
        # mean (legacy layout). Bit-neutral OFF.
        losses = self._compose_feature_geometry(
            losses,
            meta,
            ss=(ms_t1_pred_mean, ss_target),
            ms_head=(ms_pred_mean, target),
            ms_head_legacy_composed_shape=True,
        )

        # Standalone fixed-coefficient logvar bound penalty (RLRP-718)
        losses = losses + self._logvar_bound_penalty()

        if reduce:
            losses = reduce_probabilistic_compose_loss(losses)

        # .... memory management ..................................................................
        # (CRITICAL) ToDo: validate grad ok
        del (
            ms_pred_mean,
            ms_pred_logvar,
            model_in,
            target,
        )
        del ms_t1_pred_mean, ms_t1_pred_logvar, ss_target

        return losses, meta
