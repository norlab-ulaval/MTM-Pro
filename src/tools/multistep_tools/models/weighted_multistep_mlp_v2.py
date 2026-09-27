# coding=utf-8
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import omegaconf
import torch
from torch.nn import functional as F
import math

from tools.multistep_tools.models import AbstractFeatureWeightedMultiStepMLP
from tools.multistep_tools.models.utils import reduce_deterministic_compose_loss, \
    reduce_probabilistic_compose_loss


class WeightedMultiStepMLPV2(AbstractFeatureWeightedMultiStepMLP):

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
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        distribution_name: str = "gaussian",
        mae_loss: bool = True,
        model_use_double_precision: bool = False,
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
        )

    @torch.compiler.disable
    def _deterministic_loss(
        self, model_in: torch.Tensor, target: torch.Tensor, reduce=True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        losses = ms_losses
        """
        meta = {}
        assert model_in.ndim == target.ndim

        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)

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

        # ==== Composite loss =====================================================================
        # Reduce over feature dim
        ms_losses = ms_losses.mean(2, keepdim=True)

        ms_losses = ms_losses

        losses = ms_losses

        # A7 (RLRP-788): gate diagnostic ``meta`` writes behind the meta-collection kill-switch (RLRC meta-collection kill-switch `.junie` plan, ``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
        if self._enable_meta_collection:
            meta["horizon_loss"] = ms_losses.detach().mean().item()

        if reduce:
            losses = reduce_deterministic_compose_loss(losses)

        # .... memory management ..................................................................
        # (CRITICAL) ToDo: validate grad ok
        del ms_pred_mean, model_in, target

        return losses, meta

    @torch.compiler.disable
    def _probabilistic_loss(
        self, model_in: torch.Tensor, target: torch.Tensor, reduce: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        losses = ms_nll_losses
        """
        meta = {}
        model_in, target = self._setup_loss_input(model_in, target)

        ms_pred_mean, ms_pred_logvar = self.forward(model_in, use_propagation=False)

        # .... Horizon NLL loss ...................................................................
        ms_distribution = self._to_distribution(ms_pred_mean, ms_pred_logvar)
        ms_nll_losses = -ms_distribution.log_prob(target)

        # .... logvar bound penalty: now added ONCE on the composite (RLRP-718) ...................

        # .... Apply temporal weight ..............................................................
        ms_nll_losses = self.apply_next_obs_temporal_discount_factor_weights(ms_nll_losses,
                                                                             log_space=True,
                                                                             apply_minus_log=False,
                                                                             temporal_mode_aware=True)

        # .... Apply feature weight ...............................................................
        ms_nll_losses = self.apply_next_obs_feature_weights(ms_nll_losses, log_space=True,
                                                            apply_minus_log=False)

        # .... Reduce multistep horizon ...........................................................
        ms_nll_losses = self.reduce_multistep_losses_horizon(ms_nll_losses,
                                                             probabilistic_losses=True)

        # ==== Composite loss =====================================================================
        ms_nll_losses = ms_nll_losses.mean(2, keepdim=True)

        losses = ms_nll_losses

        # Standalone fixed-coefficient logvar bound penalty (RLRP-718)
        losses = losses + self._logvar_bound_penalty()

        if self._enable_meta_collection:
            meta["horizon_loss"] = ms_nll_losses.detach().mean().item()

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

        return losses, meta
