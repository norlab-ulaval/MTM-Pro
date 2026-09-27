# coding=utf-8
"""Intermediate multistep model class hosting the train-time domain randomization top layer.

Permanent production module. Introduced by task RLRP-707 of the Training-Pipeline Domain
Randomization `.junie` plan (``feature_training_pipeline_domain_randomization_plan_20260610.md``).

This class is inserted between ``MultiStepMLP`` and ``AbstractTemporalyWeightedMultiStepMLP`` in the
inheritance chain. It owns an optional :class:`TrainTimeDomainRandomizer` top layer that perturbs
the model input during fitting only (a common input-side domain-randomization / data-augmentation
scheme; BYOL/TD-MPC2 cited in RLRP-707 only as intuition, not replicated). The layer is disabled by
default for every parent and child model, so behaviour is unchanged unless explicitly enabled.
"""
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import omegaconf
import torch

from tools.domain_randomization_tools.spec_containers import (
    TrainTimeDomainRandomizationSpec,
)
from tools.domain_randomization_tools.train_time_domain_randomizer import (
    TrainTimeDomainRandomizer,
    train_time_domain_randomizer_factory,
)
from tools.multistep_tools.models.base_multistep_mlp import MultiStepMLP


class TrainTimeDomainRandomizationMultiStepMLP(MultiStepMLP):
    """``MultiStepMLP`` augmented with an optional train-time domain randomization top layer."""

    train_time_domain_randomizer: TrainTimeDomainRandomizer

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        singlestep_obs_len: int,
        singlestep_act_len: int,
        history_len: int,
        horizon_len: int,
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
        dropout: float = 0.0,
        ms_head_dropout: float = 0.0,
        ms_head_num_layers: int = 1,
        residual_form: str = "identity_v2",
        layer_bloc=None,  # RLRP-768 per-region layer-bloc selector (forwarded)
        enable_auto_loss_weighting: bool = False,
        train_time_domain_randomization: Optional[
            Union[Dict, omegaconf.DictConfig, TrainTimeDomainRandomizationSpec]
        ] = None,
        description: Optional[str] = None,
        feature_geometry=None,
        internal_orientation=None,
        orientation_singlestep_slots=None,
    ):
        """Initialize the intermediate model and build the train-time randomization top layer.

        :param train_time_domain_randomization: Optional train-time domain randomization config
            (mapping, ``DictConfig`` or spec). When ``None`` or with ``enable: false`` the top
            layer is a strict no-op (legacy-preserving default).
        """
        super().__init__(
            in_size,
            out_size,
            device,
            singlestep_obs_len=singlestep_obs_len,
            singlestep_act_len=singlestep_act_len,
            history_len=history_len,
            horizon_len=horizon_len,
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
            enable_auto_loss_weighting=enable_auto_loss_weighting,
            description=description,
            feature_geometry=feature_geometry,
            internal_orientation=internal_orientation,
            orientation_singlestep_slots=orientation_singlestep_slots,
        )

        self._train_time_domain_randomization_cfg = train_time_domain_randomization
        # Forward the multistep layout (RLRP-707) so the randomizer is multistep-aware: the
        # flattened model input axis is laid out as per-timestep blocks ``(O_1..O_MS, A_1..A_MS)``
        # with ``MS == history_len`` (the input sequence length), not a single increasing sequence.
        # This enables per-single-step ``per_feature_scale`` tiling and AR(1) noise along the
        # genuine multistep/time axis.
        self.train_time_domain_randomizer = train_time_domain_randomizer_factory(
            train_time_domain_randomization,
            feature_dim=self.in_size,
            device=self.device,
            dtype=self.model_dtype,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            sequence_len=self.history_len,
            horizon_len=self.horizon_len,
        )

    @property
    def train_time_domain_randomization_enabled(self) -> bool:
        """Whether the train-time domain randomization top layer is active.

        :return: ``True`` if the randomizer is enabled, ``False`` otherwise.
        """
        return bool(self.train_time_domain_randomizer.enable)

    def bind_train_time_domain_randomization_normalizer_scale(
        self, sigma_norm
    ) -> None:
        """Late-bind the DR scale to the normalizer output std (RLRP-761 S1.7).

        The randomizer is built in ``__init__``, i.e. **before** any normalizer
        statistic exists, so ``per_feature_scale_mode:
        relative_to_normalized_std`` cannot be resolved at construction time.
        This is called from ``OneDTransitionRewardModelV2.update_normalizer``
        (re-called on every refit, which is correct and required).

        No-op in the default ``absolute`` mode.

        :param sigma_norm: per-single-step-feature normalizer output std, as
            resolved by
            :func:`tools.feature_handling_tools.feature_target_space.resolve_normalizer_output_std`.
        """
        self.train_time_domain_randomizer.bind_normalizer_scale(sigma_norm)
        return None

    def set_train_time_domain_randomization_epoch(self, epoch: int) -> float:
        """Update the randomizer noise-scale multiplier from its scheduler for the given epoch.

        :param epoch: The current training epoch.
        :return: The applied scalar multiplier.
        """
        return self.train_time_domain_randomizer.update_scale_from_scheduler(epoch)

    def get_train_time_domain_randomization_noise_scale(self) -> torch.Tensor:
        """Return the randomizer's current effective per-feature noise scale.

        Mirrors :meth:`set_train_time_domain_randomization_epoch`; used by the RLRP-707 (§4ter)
        epoch callback to record the *current* noise to tensorboard.

        :return: A tensor of shape ``(feature_dim,)`` with the effective per-feature noise scale.
        """
        return self.train_time_domain_randomizer.effective_noise_scale

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply the train-time domain randomization top layer, then delegate to the base loss.

        The randomizer is a strict no-op in eval mode or when disabled, so this only perturbs the
        input during fitting.

        :param model_in: The model input tensor (``E x B x Id`` or ``B x Id``).
        :param target: The optional target tensor.
        :return: The loss tensor and a metadata dict, as returned by the base implementation.
        """
        model_in = self.train_time_domain_randomizer(model_in, segment="input")
        if (
            target is not None
            and self.train_time_domain_randomizer.enable
            and self.training
            and self._resolve_randomize_target()
        ):
            # The model target follows the codebase model-output convention
            # ``out_size = O*MS + A*(MS-1) = in_size - singlestep_act_len`` (the last single-step
            # action block is dropped), which is *not* equal to the randomizer ``feature_dim``
            # (the model-input ``in_size``). The previous ``target.shape[-1] == feature_dim`` guard
            # was therefore always false. The randomizer now resolves the model-output layout for
            # the ``"target"`` segment (correct per-feature scale + last-action padding) and falls
            # back to a strict no-op when the target shape is unsupported (RLRP-707).
            target = self.train_time_domain_randomizer(target, segment="target")
        return super().loss(model_in, target)

    def _resolve_randomize_target(self) -> bool:
        """Resolve whether the target tensor should also be perturbed.

        :return: ``True`` if target randomization is requested, ``False`` otherwise.
        """
        cfg = self._train_time_domain_randomization_cfg
        if cfg is None:
            return False
        if isinstance(cfg, TrainTimeDomainRandomizationSpec):
            return bool(cfg.randomize_target)
        if isinstance(cfg, omegaconf.DictConfig):
            return bool(cfg.get("randomize_target", False))
        if isinstance(cfg, dict):
            return bool(cfg.get("randomize_target", False))
        return False
