# coding=utf-8
"""
Trajectory-Based Model (TBM) MS->MS forecast baseline (RLRP-694).

Permanent production module. Introduced by section 4 of the MS->MS forecast baselines
(End-to-End-TCN / M3 / TBM) ``.junie`` plan
(``feature_ms2ms_forecast_baselines_e2etcn_m3_tbm_plan_RLRP-692_693_694_20260703.md``).
Reproduces Lambert et al. (CDC 2021) "Learning Accurate Long-term Dynamics for Model-based RL"
(``dynamics_model.py`` ``class Net``, ``cfg.model.traj=True``): a plain MLP that predicts
``o_{t+h} = Net([s0, h, params])`` as a direct function of the horizon index (NOT recursive).
Per plan section 0.1, TBM is the only baseline of the three for which model-ensemble and a
probabilistic head are core components, so ``ensemble_size`` and ``distribution_name`` /
``deterministic`` are exposed as first-class configurable knobs.
"""
from typing import Dict, Optional, Sequence, Union

import omegaconf
import torch

from tools.multistep_tools.models.abstract_horizon_indexed_ms2ms_forecast import (
    AbstractHorizonIndexedMS2MSForecast,
)


class MS2MSTrajectoryBasedModel(AbstractHorizonIndexedMS2MSForecast):
    """Trajectory-Based Model (TBM) horizon-indexed forecast baseline (RLRP-694).

    ``o_{t+h} = Net([s0, h, params])`` where (RLRC I/O adaptation, plan section 0.3):
    ``s0`` = the observed multi-step history (full history, Q-G), ``h`` = the scalar horizon index
    ``h / H`` (Q-C), and ``params`` (the reference's policy/control parameters) are folded into the
    history, whose action stream already carries the available control information (Q-D fallback).
    The horizon loop, the anti-compounding "history-only state input" invariant, and the ensemble +
    probabilistic machinery are all inherited unchanged.
    """

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
        auto_weighting_noise_model: str = "gaussian",
        auto_weighting_scheme: str = "tempered_likelihood",
        # RLRP-761 S3: accepted for config-surface compatibility only -- see
        # ``AbstractMS2MSForecast._validate_unsupported_feature_weighting``.
        feature_weight_mode: str = "tempered",
        feature_weight_max_ratio: Optional[float] = None,
        description: Optional[str] = None,
        feature_geometry=None,
        internal_orientation=None,
        orientation_singlestep_slots=None,
        # Config-level future-action-plan conditioning switch (operator follow-up to the
        # post-implementation review of
        # ``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``):
        # ``False`` => the true plan-FREE baseline. Default ``True`` keeps every existing
        # config bit-exact.
        enable_future_action_plan_conditioning: bool = True,
        # RLRP-824 Step 7 (FR9): opt-in autocast around training_step / validation_step
        # (``null`` = bit-exact).
        mixed_precision: Optional[str] = None,
    ):
        super().__init__(
            in_size,
            out_size,
            device=device,
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
            auto_weighting_noise_model=auto_weighting_noise_model,
            auto_weighting_scheme=auto_weighting_scheme,
            feature_weight_mode=feature_weight_mode,
            feature_weight_max_ratio=feature_weight_max_ratio,
            description=description,
            feature_geometry=feature_geometry,
            # RLRP-736 bespoke-forward plan §3.5 (C.3): by-construction rotation rep
            # (defaults quaternion / None => bit-exact; active only on the
            # DETERMINISTIC path, the probabilistic head stays the §4A-B follow-up).
            internal_orientation=internal_orientation,
            orientation_singlestep_slots=orientation_singlestep_slots,
            enable_future_action_plan_conditioning=enable_future_action_plan_conditioning,
            mixed_precision=mixed_precision,
        )

    def _model_family_tag(self) -> str:
        return "TBM"

    # ==== TBM per-horizon action conditioning (RLRP-728) =========================================
    def _per_horizon_extra_conditioning(
        self,
        history_feat: torch.Tensor,
        h: int,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Append the padded action sub-sequence ``a^h`` to the per-horizon input (RLRP-728).

        The reference TBM (``o_{t+h} = Net([s0, h, params])``) originally folded the control
        information implicitly into the flat history only, so this model had **no explicit
        per-horizon action channel** (the base default returned ``None``). RLRP-728 adds one so
        the test-time rollout can feed the horizon's planned future-action sequence: the ``params``
        term of the reference is realised as the same padded action sub-sequence ``a^h`` used by M3
        (``future_actions`` supplied ⇒ built from the internal driving sequence ``a_{t..t+F-1}`` ==
        ``[a_t] ++ plan``, slot 0 == ``a_t``, for every forecast window step; stage ``A4`` of the Fix
        the MS->MS future-action-plan conditioning contract ``.junie`` plan,
        ``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``). This is a
        **permanent architectural addition** (it grows
        :meth:`_per_horizon_in_size`); it is intentionally not bit-exact with the prior
        action-agnostic TBM, which was incomplete and never used in an experiment.
        """
        return self._build_padded_action_subsequence(
            history_feat, h, future_actions=future_actions
        )

    def _per_horizon_extra_conditioning_family(
        self,
        history_feat: torch.Tensor,
        family_len: int,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """One-shot ``(..., H, H * Da)`` stack of the padded action sub-sequences ``{a^h}``.

        RLRP-824 test-time rollout hotfix (2026-09-19): only reached on the opt-in
        ``vectorized_family_forward`` path (``torch.equal`` to the per-``h`` builder above); the
        default per-horizon loop path is untouched, so TBM stays bit-exact by default.
        """
        return self._build_padded_action_subsequence_family(
            history_feat, family_len, future_actions=future_actions
        )

    def _per_horizon_extra_size(self) -> int:
        return self.history_len * self.singlestep_act_len
