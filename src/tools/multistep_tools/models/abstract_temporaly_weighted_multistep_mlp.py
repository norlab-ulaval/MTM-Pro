# coding=utf-8
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import abc
import math
import warnings
import numpy as np
import omegaconf
import torch
from numpy import ndarray
from torch import Tensor, nn as nn
from torch.nn import functional as F
from torch import distributions as dist

from tools.domain_randomization_tools.spec_containers import (
    TrainTimeDomainRandomizationSpec,
)
from tools.multistep_tools.models.train_time_domain_randomization_multistep_mlp import (
    TrainTimeDomainRandomizationMultiStepMLP,
)
from tools.math_tools.weigthing import check_finite, flags_to_operator, weight_values
from tools.multistep_tools.models.precision_bounds import weights_eps

# RLRP-769: how the temporal discount factors w_j (w_j = gamma^j) ENTER the per-horizon-step NLLs,
# and how the horizon is aggregated. Orthogonal to `ms_probabilities_reduction`. See the ctor
# comment block below.
# NOTE: the weights are normalized (feature-averaged horizon sum == 1) for every mode EXCEPT
# 'discounted-sum', which deliberately uses the RAW exponential-moving-average profile gamma^j
# (near step anchored at w_0 = 1).
MS_TEMPORAL_WEIGHTING_MODES = (
    "log-shift",
    "tempered",
    "discounted-sum",
    "joint",
    "uniform-mean",
)


class AbstractTemporalyWeightedMultiStepMLP(
    TrainTimeDomainRandomizationMultiStepMLP, abc.ABC
):
    _temporal_weights: torch.Tensor
    _composed_next_obs_temporal_discount_factors: torch.Tensor
    # Floor for discount-factor weight clamping is dtype-aware, finfo-derived via
    # ``precision_bounds.weights_eps`` (== ``finfo(dtype).eps``: ~1.19e-7 float32 /
    # ~2.22e-16 float64, RLRP-750 -- the relative-precision floor for O(1) weights,
    # replacing the former frozen float32-era 1e-10/1e-15 values); see the clamp site
    # in ``_init_composed_next_obs_temporal_discount_factors`` (RLRP-769 revision 2026-08-08:
    # moved out of ``_init_discount_factors_on_horizon_slice`` and made mode-aware -- the
    # multiplicative modes read the EXACT, unfloored profile).

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
        ms_probabilities_reduction: str = "independent",
        ms_temporal_weighting_mode: str = "discounted-sum",
        ms_energy_beta: Union[float, str] = "learned",
        ms_energy_axis: str = "step",
        train_time_domain_randomization: Optional[
            Union[Dict, omegaconf.DictConfig, TrainTimeDomainRandomizationSpec]
        ] = None,
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
            train_time_domain_randomization=train_time_domain_randomization,
            description=description,
            feature_geometry=feature_geometry,
            internal_orientation=internal_orientation,
            orientation_singlestep_slots=orientation_singlestep_slots,
        )

        # .... Set discount factor's ..............................................................
        # `ms_probabilities_reduction` selects how the per-horizon-step NLLs are reduced into a
        # single multi-step loss (only effective for probabilistic losses):
        #   independent -> assume the horizon steps are independent, p(X,Y) = p(X)p(Y), so the
        #                  joint NLL is the SUM of per-step NLLs (-log prod_j p_j = sum_j -log p_j);
        #   prob-sum    -> aggregate the per-step densities in PROBABILITY space, p(X,Y) = p(X)+p(Y),
        #                  giving the NLL of the (unnormalized) sum-over-steps:
        #                  -log(sum_j p_j) = -logsumexp_j(log p_j) = -logsumexp_j(-nll_j).
        #                  NOTE: despite aggregating the steps it does NOT model genuine inter-step
        #                  dependence p(y_0,...,y_{F-1}) (formerly mis-named 'dependent', RLRP-704;
        #                  a true chain-rule joint is a deferred follow-up);
        #   soft-max-energy -> ENERGY-FUNCTION reading: treat each per-step NLL as a state
        #                  "energy" E_j over the horizon "states" j and take the soft-max
        #                  (log-partition at inverse temperature beta=1) logsumexp_j(E_j) = log Z,
        #                  a robust worst-step penalty ~= max_j nll_j. This is the Gibbs FREE
        #                  ENERGY / log-partition of the horizon energies (no probabilistic-joint
        #                  NLL interpretation over the data space).
        #   energy-density -> genuine (temperature-controlled) energy density / log-partition:
        #                  the free energy F(beta) = (1/beta) * logsumexp_j(beta * E_j) of the
        #                  Gibbs distribution p_j ~ exp(beta * E_j) / Z(beta) over horizon steps.
        #                  `beta` is the inverse temperature controlled by `ms_energy_beta`:
        #                  either the string "learned" (default) to make beta a trainable scalar
        #                  (initialized SMALL/mean-like and ANNEALED mean -> max by training),
        #                  or an explicit float > 0 to fix it; beta -> 0 recovers
        #                  the (shifted) mean, beta -> +inf recovers the hard max. At beta=1 the
        #                  value coincides with `soft-max-energy` (which omits the 1/beta scaling).
        assert ms_probabilities_reduction in {
            "independent",
            "prob-sum",
            "soft-max-energy",
            "energy-density",
        }, (
            f"ms_probabilities_reduction must be 'independent', 'prob-sum', 'soft-max-energy' "
            f"or 'energy-density', got {ms_probabilities_reduction}"
        )
        self.ms_probabilities_reduction = ms_probabilities_reduction

        # `ms_temporal_weighting_mode` selects HOW the temporal discount factors w_j (w_j = gamma^j,
        # ALWAYS normalized so that the feature-averaged horizon sum is one) enter the
        # per-horizon-step NLLs, and how the horizon is aggregated. It is ORTHOGONAL to
        # `ms_probabilities_reduction` (RLRP-769):
        #   log-shift    -> LEGACY (gradient-exact): additive shift in log space,
        #                   nll_j + log w_j, aggregated by SUM. NOTE: sum_j log w_j is a
        #                   DATA-INDEPENDENT CONSTANT, so gamma has NO gradient effect and the
        #                   magnitude scales with the horizon length F. Kept for backward
        #                   compatibility and checkpoint reproduction. (Former default; the
        #                   RLRP-769 T10 ablation showed it diverges on compounded rollouts.)
        #   tempered     -> MULTIPLICATIVE weighting, w_j * nll_j, aggregated by SUM. With
        #                   sum_j w_j = 1 this is a CONVEX COMBINATION of the per-step NLLs, i.e.
        #                   the NLL of the tempered / power-pooled product prod_j p_j^{w_j}
        #                   (product-of-experts). Horizon-scale invariant and gamma is
        #                   gradient-effective. THIS IS THE THEORETICALLY RE-ALIGNED MODE.
        #   joint        -> NO temporal weighting, sum_j nll_j: the EXACT joint NLL under step
        #                   independence, -log prod_j p_j. Principled but undiscounted and O(F).
        #   uniform-mean -> NO temporal weighting, aggregated by MEAN (1/F), i.e. `joint / F`: the
        #                   UNIFORM mean per-step NLL. The control arm that removes the
        #                   horizon-scale inflation with the discount ablated away.
        #   discounted-sum -> MULTIPLICATIVE weighting with the RAW (UNNORMALIZED) exponential
        #                   moving average profile, sum_j gamma^j * nll_j, aggregated by SUM. The
        #                   near step keeps w_0 = 1 (exactly its 'log-shift'/'joint' weight) and
        #                   each further step is damped by gamma -- i.e. the EMA behaviour the
        #                   temporal discount was meant to express. Same mathematical object as
        #                   'tempered' (-log prod_j p_j^{gamma^j}), differing ONLY by the
        #                   per-member constant gain sum_i gamma^i:
        #                       tempered == discounted-sum / sum_i gamma^i.
        #                   Magnitude saturates at 1/(1-gamma) instead of growing with F, so it is
        #                   the MINIMAL behavioural delta from 'log-shift' while making gamma
        #                   gradient-effective (RLRP-769 revision 2026-08-08). THIS IS THE DEFAULT
        #                   (promoted from 'log-shift' after the T10 ablation, RLRP-769).
        assert ms_temporal_weighting_mode in MS_TEMPORAL_WEIGHTING_MODES, (
            f"ms_temporal_weighting_mode must be one of {MS_TEMPORAL_WEIGHTING_MODES}, "
            f"got {ms_temporal_weighting_mode!r}"
        )
        self.ms_temporal_weighting_mode = ms_temporal_weighting_mode

        # `ms_energy_beta` is the inverse temperature beta > 0 used by the 'energy-density'
        # reduction (and ignored by the other reductions). It accepts either the string
        # "learned" (default) to make beta a trainable scalar parameter, kept strictly positive
        # via a softplus reparameterization (raw -> softplus(raw)) and initialized SMALL/mean-like
        # so it anneals mean -> max under training, or an explicit float > 0 to fix it as a buffer.
        ms_energy_beta_learnable = (
            isinstance(ms_energy_beta, str) and ms_energy_beta == "learned"
        )
        if not ms_energy_beta_learnable:
            assert not isinstance(ms_energy_beta, str), (
                f"ms_energy_beta must be the string 'learned' or a float > 0, "
                f"got {ms_energy_beta!r}"
            )
            assert ms_energy_beta > 0.0, (
                f"ms_energy_beta (inverse temperature) must be > 0, got {ms_energy_beta}"
            )
        self.ms_energy_beta = ms_energy_beta
        self.ms_energy_beta_learnable = ms_energy_beta_learnable
        if ms_energy_beta_learnable:
            # Trainable beta initialized SMALL (mean-like) so the 'energy-density' reduction starts
            # close to a horizon MEAN (beta -> 0 limit) and gradually ANNEALS toward a worst-step
            # MAX (beta -> inf limit) under training. The energy-density free energy
            # F(beta) = (1/beta) logsumexp(beta E) is monotonically DECREASING in beta
            # (dF/dbeta = -H(p_beta)/beta^2 <= 0), so minimizing the loss naturally drives beta
            # UPWARD (mean -> max); starting at a small beta gives the intended mean->max curriculum
            # to help the MS forecast path converge. Inverse softplus so that softplus(raw) == init.
            _ms_energy_beta_init = 0.1
            raw_beta = math.log(math.expm1(_ms_energy_beta_init))
            self._ms_energy_raw_beta = nn.Parameter(
                torch.tensor(raw_beta, device=self.device, dtype=self.model_dtype)
            )
        else:
            self.register_buffer(
                "_ms_energy_beta_const",
                torch.tensor(ms_energy_beta, device=self.device, dtype=self.model_dtype),
            )

        # `ms_energy_axis` selects the axis (index set) over which the ENERGY reductions
        # ('soft-max-energy' / 'energy-density') take their log-partition / free-energy:
        #   step    -> partition over the horizon STEPS j (default; the current behaviour). The
        #              per-feature free energy is kept (output (E, B, O+A)).
        #   feature -> partition over the OBSERVATION/feature dims d (O+A), giving a robust
        #              worst-observation-dimension free energy per step, then the per-step free
        #              energies are SUMMED over the (normalized) horizon (matching 'independent').
        #   joint   -> a single partition over the whole (step x feature) lattice (d, j) together.
        # NOTE: 'feature'/'joint' collapse the feature axis INSIDE the reduction, so they are only
        # honoured at the primary-forecast call sites (which opt in via
        # `enable_feature_energy_axis=True` to `reduce_multistep_losses_horizon`); every other caller
        # (e.g. the pre-mixture U-loss, which accumulates a per-feature (E, B, O+A) tensor) keeps the
        # 'step' behaviour regardless of this setting. The axis is also meaningful only for the two
        # energy reductions; for 'independent'/'prob-sum' it must stay 'step'.
        assert ms_energy_axis in {"step", "feature", "joint"}, (
            f"ms_energy_axis must be 'step', 'feature' or 'joint', got {ms_energy_axis}"
        )
        if ms_energy_axis != "step" and ms_probabilities_reduction not in {
            "soft-max-energy",
            "energy-density",
        }:
            raise ValueError(
                f"ms_energy_axis={ms_energy_axis!r} is only valid with an energy reduction "
                f"(ms_probabilities_reduction in {{'soft-max-energy', 'energy-density'}}), "
                f"got ms_probabilities_reduction={ms_probabilities_reduction!r}. Set "
                f"ms_energy_axis='step' for the other reductions."
            )
        self.ms_energy_axis = ms_energy_axis

        if isinstance(temporal_weights, float):
            temporal_weights = (temporal_weights,) * ensemble_size
        else:
            assert isinstance(temporal_weights, Sequence)
            assert len(temporal_weights) == ensemble_size, (
                f"temporal_weights must match ensemble_size, currently "
                f"{temporal_weights=} != {ensemble_size=}"
            )

        # # The RL (temporal) discount factor formulation `0 ≤ gamma ≤ 1.0`
        # # with `0 ⇒ myopic and 1 ⇒ farsighted`.
        # assert np.all(0.0 <= np.array(temporal_weights)) and np.all(
        #     np.array(temporal_weights) <= 1.0
        # ), (
        #     f"All gamma values passed to param `temporal_weights` "
        #     f"must be 0 <= gamma <= 1.0, curently {temporal_weights=}"
        # )
        # The RL (temporal) discount factor formulation `0 ≤ gamma`
        # with `0 ⇒ myopic and 1 ⇒ farsighted`.
        assert np.all(0.0 <= np.array(temporal_weights)), (
            f"All gamma values passed to param `temporal_weights` "
            f"must be 0 <= gamma, curently {temporal_weights=}"
        )

        self._temporal_weights = torch.tensor(
            temporal_weights, device=self.device, dtype=self.model_dtype
        ).unsqueeze(1)
        self._init_composed_next_obs_temporal_discount_factors()

        # .... Final build step ...................................................................
        # self._build_temporaly_weighted_logvar_bound_layer(learn_logvar_bounds)

        if model_use_double_precision:
            self.to(dtype=torch.double)
        else:
            self.to(dtype=self.model_dtype)

    @abc.abstractmethod
    def _probabilistic_loss(
        self, model_in: torch.Tensor, target: torch.Tensor, reduce: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]: ...

    @abc.abstractmethod
    def _deterministic_loss(
        self, model_in: torch.Tensor, target: torch.Tensor, reduce=True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]: ...

    def apply_next_obs_temporal_discount_factor_weights(
        self,
        values: Union[torch.Tensor, np.ndarray],
        log_space: bool = False,
        values_centred_around_zero: bool = False,
        apply_minus_log: bool = False,
        pre_process_weight: Optional[Callable] = None,
        temporal_mode_aware: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Apply temporal discount factor weights to value tensor. Handle negative value case

        :param values: Input tensor.
        :param log_space: Set to True if values are in log space.
        :param values_centred_around_zero: Whether values in normal space are centered around zero
            or not. Only valid with 'log_space=False'.
        :param apply_minus_log: Execute `values - log weights` instead of `values + log weights`.
            Only valid with 'log_space=True'.
        :param pre_process_weight: A function that will be applied to values_weights before weighting.
        :param temporal_mode_aware: opt-in flag for the per-step **NLL** call sites only. When True
            the application of the discount is dispatched on `ms_temporal_weighting_mode`
            ('tempered' -> multiplicative w_j * nll_j, 'joint' / 'uniform-mean' -> no weighting,
            'log-shift' -> the legacy additive-log shift). Default False keeps the legacy
            additive/multiplicative `apply_weights` behaviour bit-exactly, so the non-NLL call
            sites (KL terms, MPPI cost, deterministic losses, prediction reduction) are unaffected.
        :return: A temporal discounted weighted value tensor.
        """
        if temporal_mode_aware:
            assert log_space and not apply_minus_log, (
                "temporal_mode_aware=True is only defined for the per-step NLL path "
                "(log_space=True, apply_minus_log=False)"
            )
            if self.ms_temporal_weighting_mode in ("joint", "uniform-mean"):
                # NO temporal discount at all. 'joint' -> exact independent-steps joint NLL
                # sum_j nll_j; 'uniform-mean' -> the same, divided by F in the horizon reduction
                # (see `reduce_multistep_losses_horizon`).
                return values
            if self.ms_temporal_weighting_mode in ("tempered", "discounted-sum"):
                # Tempered / power-pooled likelihood: w_j * nll_j, i.e. -log p_j^{w_j}.
                # 'tempered'       -> normalized weights (feature-averaged sum_j w_j = 1);
                # 'discounted-sum' -> RAW exponential-moving-average profile gamma^j (w_0 = 1).
                # Both read the EXACT (unfloored) profile: no log of the weight is taken here,
                # so the `weights_eps` floor is unnecessary and would truncate the EMA tail for
                # small `temporal_weights` (RLRP-769 revision 2026-08-08).
                # Tempering MULTIPLIES a log-density, so the `'scale'` operator is the correct
                # one here even though `nll_j` IS a log-space quantity -- the named-operator API
                # (RLRP-769) makes that explicit, where the legacy `apply_weights(log_space=False,
                # values_centred_around_zero=True)` call used to assert something FALSE about the
                # tensor. `'scale'` does NOT run the non-finite guards (only the additive-log
                # operators can create a non-finite out of finite inputs), so we keep them here
                # explicitly to match the `log-shift` path.
                mode_weights = self._temporal_discount_factors_for_mode()
                check_finite(values, "values", False)
                check_finite(mode_weights, "values_weights", False)
                weighted_values = weight_values(
                    values,
                    values_weights=mode_weights,
                    operator="scale",
                    pre_process_weight=pre_process_weight,
                )
                check_finite(weighted_values, "weighted_values", False)
                return weighted_values
            # ONLY 'log-shift' falls through to the legacy log-space shift.

        return weight_values(
            values,
            values_weights=self._composed_next_obs_temporal_discount_factors,
            operator=flags_to_operator(
                log_space=log_space,
                values_centred_around_zero=values_centred_around_zero,
                apply_minus_log=apply_minus_log,
            ),
            pre_process_weight=pre_process_weight,
        )

    def compensate_composed_mean_horizon_scale(self, ms_losses: Tensor) -> Tensor:
        """Restore the `ms_temporal_weighting_mode` horizon scale for call sites that reduce the
        per-step NLLs with a plain `mean()` over the FLATTENED composed axis (MS x (O+A)) instead
        of going through `reduce_multistep_losses_horizon` (RLRP-769 review).

        Such a `mean()` bakes in an implicit `1/F` factor. That factor is only correct for the
        modes whose definition includes it:

          - `log-shift`    -> legacy aggregation, kept AS-IS (bit-neutral);
          - `uniform-mean` -> the `1/F` IS the mode, kept AS-IS;
          - `tempered`     -> the weights already satisfy the feature-averaged `sum_j w_j = 1`,
                              so the `1/F` is a DOUBLE normalization -> multiplied back out;
          - `joint`        -> the exact `-log prod_j p_j` is a SUM, so the `1/F` destroys the
                              joint semantics -> multiplied back out;
          - `discounted-sum` -> the raw EMA `sum_j gamma^j nll_j` is a SUM anchored at
                              `w_0 = 1`, so the `1/F` is multiplied back out as well.

        Without this compensation `tempered` is silently divided by the horizon length and
        `joint` becomes indistinguishable from `uniform-mean` on those models.

        :param ms_losses: A horizon-reduced multi-step loss tensor (post composed-axis `mean()`).
        :return: The loss tensor on the scale defined by `ms_temporal_weighting_mode`.
        """
        if self.ms_temporal_weighting_mode in ("tempered", "discounted-sum", "joint"):
            return ms_losses * float(self.horizon_len)
        return ms_losses

    def _temporal_discount_factors_for_mode(self) -> Tensor:
        """Return the temporal discount weight buffer matching `ms_temporal_weighting_mode`
        (RLRP-769 revision 2026-08-08).

        - `discounted-sum` -> the EXACT, UNNORMALIZED exponential-moving-average profile
          `gamma^j` (near step anchored at `w_0 = 1`);
        - `tempered`       -> the EXACT profile scaled by the (feature-averaged) normalizer, so
          that `sum_j w_j = 1` -- identical to the legacy buffer except that the `weights_eps`
          floor is NOT baked in (it only matters for small `gamma` / long horizons);
        - every other mode / legacy caller -> the FLOORED + normalized buffer, which is the one
          the log-space operators need (`log 0 = -inf`).

        :return: The (num_members, out_size) weight buffer for the active mode.
        """
        if self.ms_temporal_weighting_mode == "discounted-sum":
            return self._composed_next_obs_temporal_discount_factors_exact
        if self.ms_temporal_weighting_mode == "tempered":
            return self._composed_next_obs_temporal_discount_factors_exact_normalized
        return self._composed_next_obs_temporal_discount_factors

    def _init_composed_next_obs_temporal_discount_factors(self) -> None:
        self.register_buffer(
            "_composed_next_obs_temporal_discount_factors",
            torch.ones(
                (
                    self.num_members,
                    self.out_size,
                ),
                device=self.device,
                # dtype=torch.double,
                dtype=self.model_dtype,
            ),
        )

        # .... Init observation horizon ...........................................................
        self._init_discount_factors_on_horizon_slice(
            horizon_slice=self._compose_next_obs_multistep_obs_horizon_slice,
            horizon_t_init=0,
        )
        # .... Init action horizon ................................................................
        self._init_discount_factors_on_horizon_slice(
            horizon_slice=self._compose_next_obs_multistep_act_horizon_slice,
            horizon_t_init=1,
        )

        # .... Mode-aware weight floor (RLRP-769 revision 2026-08-08) .............................
        # The `weights_eps` floor is required ONLY by the additive-log operators (`log 0 = -inf`),
        # i.e. by 'log-shift' and every legacy log-space consumer. The multiplicative modes
        # ('tempered' / 'discounted-sum') never take the log of a weight, so flooring them only
        # TRUNCATES the geometric tail for small `temporal_weights` -- turning the exponential
        # moving average into "decay, then flat". We therefore snapshot the EXACT profile FIRST,
        # then floor the buffer the log-space paths consume.
        self.register_buffer(
            "_composed_next_obs_temporal_discount_factors_exact",
            self._composed_next_obs_temporal_discount_factors.clone(),
        )
        self._warn_on_binding_temporal_weight_floor()
        # NOTE: flooring ONCE here is equivalent to the former per-slice clamping (the obs/act
        # slices are disjoint and `clamp_` is idempotent and element-wise) -> behaviour-preserving.
        self._composed_next_obs_temporal_discount_factors.clamp_(
            min=weights_eps(self.model_dtype)
        )

        # .... Normalize temporal discount weights ................................................
        self._composed_next_obs_temporal_discount_factors_unnormalized = (
            self._composed_next_obs_temporal_discount_factors.clone()
        )

        # .... Normalize temporal discount weights (ALWAYS -- RLRP-769) ...........................
        # Sum over the horizon (per feature) then reduce the feature dim, so that each ensemble
        # member gets its own normalization constant. NOTE: this normalizes the FEATURE-AVERAGED
        # horizon sum, mean_d(sum_j w_jd) = 1 -- NOT each channel's sum (the obs slice carries
        # exponents 0..H-1 and the act slice 1..H, so they differ by a factor gamma). Kept as-is
        # to preserve the legacy `classic_gamma_normalization=True/False` semantics.
        # NOTE: versus the former closed-form branch this is GRADIENT-exact, NOT value-exact --
        # the normalizer differs by the constant (O+A)/(O+A*gamma), which shifts a 'log-shift'
        # loss VALUE by F*log(c) while leaving every gradient unchanged.
        # The former closed-form `(1-gamma)/(1-gamma**H)` branch is dropped (equivalent to this
        # summation up to a per-member constant, and it breaks at gamma >= 1); the former
        # `classic_gamma_normalization is None` (unnormalized) tri-state is removed with the knob.
        tdf = self.unflaten_multistep_composed_array(
            self._composed_next_obs_temporal_discount_factors
        )
        normalization_constant = 1 / (tdf.sum(axis=-1).mean(-1, keepdim=True))
        self._composed_next_obs_temporal_discount_factors *= normalization_constant

        # The 'tempered' mode uses the SAME normalizer but applied to the EXACT (unfloored)
        # profile, so that `sum_j w_j = 1` holds on a true geometric decay even when the floor
        # would have bound (small gamma / long horizon). Identical to the buffer above otherwise.
        self.register_buffer(
            "_composed_next_obs_temporal_discount_factors_exact_normalized",
            self._composed_next_obs_temporal_discount_factors_exact
            * normalization_constant,
        )

        return None

    def _warn_on_binding_temporal_weight_floor(self) -> None:
        """Report (once, at construction) that the `weights_eps` floor BINDS on the raw discount
        profile, i.e. that `gamma ** j` underflows the dtype relative precision before the end of
        the horizon (RLRP-769 revision 2026-08-08).

        When it binds, the log-space modes ('log-shift' and the legacy consumers) see a profile
        that decays and then goes FLAT instead of being geometric. The multiplicative modes
        ('tempered' / 'discounted-sum') are immune -- they read the exact profile -- but the user
        still deserves to know that `temporal_weights` is mis-specified for this `horizon_len`.

        :return: None
        """
        floor = weights_eps(self.model_dtype)
        exact = self._composed_next_obs_temporal_discount_factors_exact
        binding = exact < floor
        if not bool(torch.any(binding)):
            return None

        gammas = self._temporal_weights.flatten().tolist()
        warnings.warn(
            f"RLRP-769: the temporal discount weight floor (weights_eps="
            f"{floor:.3e} for {self.model_dtype}) BINDS on "
            f"{int(binding.sum())}/{binding.numel()} composed weights with "
            f"temporal_weights={gammas} and horizon_len={self.horizon_len}: gamma**j underflows "
            f"the dtype relative precision before the end of the horizon, so the log-space modes "
            f"('log-shift' and the legacy consumers) see a FLAT tail instead of a geometric one. "
            f"The multiplicative modes ('tempered'/'discounted-sum') read the exact profile and "
            f"are unaffected. Consider a larger gamma, a shorter horizon_len or "
            f"model_use_double_precision=True.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    def _init_discount_factors_on_horizon_slice(
        self, horizon_slice: slice, horizon_t_init: int
    ) -> None:
        """
        Information is structured by timestep, meaning that for a 3 dimensions observation,
        the observation are stored and retrive such that

            `{(D1, D2, D3)_t=1, (D1, D2, D3)_t=2, ... , (D1, D2, D3)_t=H }`

        with dimension `D`, timestep `t` and horizon `H`.

        :param horizon_slice:
        :param horizon_t_init:
        :return:None
        """
        for each_obs_horizon_idx in range(
            horizon_slice.start, horizon_slice.stop, horizon_slice.step
        ):
            self._composed_next_obs_temporal_discount_factors[
                :, each_obs_horizon_idx : each_obs_horizon_idx + horizon_slice.step
            ] *= torch.pow(self._temporal_weights, torch.tensor(horizon_t_init))
            horizon_t_init += 1

        # NOTE (RLRP-769 revision 2026-08-08): the `weights_eps` floor formerly applied here is now
        # applied ONCE by the caller, AFTER both horizon slices and AFTER the exact-profile
        # snapshot, so that the multiplicative modes keep a true geometric (EMA) decay. Flooring
        # once at the end is equivalent (disjoint slices, idempotent element-wise clamp).

        # .... Sanity check .......................................................................
        assert torch.all(
            torch.isfinite(self._composed_next_obs_temporal_discount_factors)
        ), f"{self._composed_next_obs_temporal_discount_factors=} as non finite value(s)!"

        return None

    def reduce_multistep_prediction_to_single_step(
        self,
        pred_trj: Union[np.ndarray, torch.Tensor],
        reduction: str = "mean",
        apply_discount_factor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        # (Priority) ToDo: implement test case (its indirectly tested for now)
        # Torch-first: handle both numpy and tensor inputs
        is_numpy = isinstance(pred_trj, np.ndarray)
        if is_numpy:
            feature = torch.from_numpy(pred_trj.copy()).to(self.device)
        else:
            feature = pred_trj.clone()

        if not torch.isfinite(feature).all():
            feature = torch.ones_like(feature)

        if apply_discount_factor:
            # (CRITICAL) ToDo: (RLRP-298) assess using discount factor (or not)
            feature = self.apply_next_obs_temporal_discount_factor_weights(feature)

        feature = self.unflaten_multistep_composed_array(feature)

        if feature.shape[-1] > 1:
            if reduction == "sum":
                feature = torch.sum(feature, dim=-1)
            elif reduction == "mean":
                feature = torch.mean(feature, dim=-1)
            else:
                raise NotImplementedError(
                    f"Reduction method '{reduction}' is not suported. "
                    f"Choose either 'sum' or 'mean'"
                )
        else:
            feature = feature.squeeze(-1)

        if is_numpy:
            return feature.cpu().numpy()
        return feature

    def _to_distribution(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor,
        overwrite_distribution_name: Optional[str] = None,
        scale_are_log_variance: bool = True,
        obs_space_only: bool = False,
        validate_args: Optional[bool] = None,
    ) -> Union[
        dist.Laplace,
        dist.Normal,
        dist.StudentT,
        dist.ExponentialFamily,
        dist.MixtureSameFamily,
    ]:
        """
        Creates a PyTorch distribution based on the provided parameters and an optional
        specified distribution name, with handling for multistep mixture distributions.

        :param mean: Mean tensor of the distribution.
        :param scale: Scale tensor of the distribution.
        :param overwrite_distribution_name: Optional string to overwrite the default
            distribution type. Specific handling is applied for
            "multisteps_to_singlestep_distribution_mixture".
        :param scale_are_log_variance: Boolean indicating whether the scale is provided
            as log-variance. Default is True.
        :param obs_space_only: Boolean indicating whether to restrict handling to
            observation space variables only. Default is False.
            Only affect multisteps_to_singlestep_distribution_mixture distribution.
        :return: A PyTorch distribution object, which could be one of several types
            including Laplace, Normal, StudentT, ExponentialFamily, or a MixtureSameFamily.
        """
        if (
            overwrite_distribution_name
            == "multisteps_to_singlestep_distribution_mixture"
        ):
            assert self.in_size == self.out_size + self.singlestep_act_len, "This implementation only supports legacy output size for mixture distribution"

            # (E x B x MS x O+A)
            ms_pred_mean = self.unflaten_multistep_composed_array(mean).swapaxes(-2, -1)
            ms_pred_scale = self.unflaten_multistep_composed_array(scale).swapaxes(
                -2, -1
            )

            if obs_space_only:
                # (E x B x MS x O+A) -> (E x B x MS x O)
                ms_pred_mean = ms_pred_mean[..., : self.singlestep_obs_len]
                ms_pred_scale = ms_pred_scale[..., : self.singlestep_obs_len]

            ms_distribution = super()._to_distribution(
                ms_pred_mean, ms_pred_scale, None, scale_are_log_variance, validate_args
            )
            ms_distribution = dist.Independent(
                ms_distribution,
                reinterpreted_batch_ndims=1,
                validate_args=validate_args,
            )  # Make the features dim independent

            # (E x O+A x MS)
            mix_gamma_prob = self.unflaten_multistep_composed_array(
                self._composed_next_obs_temporal_discount_factors
            )

            if obs_space_only:
                # (E x O+A x MS) -> (E x O x MS)
                mix_gamma_prob = mix_gamma_prob[:, : self.singlestep_obs_len, :]

            # Reduce Odim but keep dim as a batch dim (E x B=1 x MS)
            mix_gamma_prob = mix_gamma_prob.mean(axis=-2, keepdims=True)

            mix = dist.Categorical(probs=mix_gamma_prob, validate_args=validate_args)
            ms_distribution_mixture = dist.MixtureSameFamily(
                mix, ms_distribution, validate_args
            )

            return ms_distribution_mixture
        else:
            return super()._to_distribution(
                mean,
                scale,
                overwrite_distribution_name,
                scale_are_log_variance,
                validate_args,
            )

    def kl_divergence_dist_to_mixture(
        self,
        ss_mean: torch.Tensor,
        ss_scale: torch.Tensor,
        mixture_means: torch.Tensor,
        mixture_scales: torch.Tensor,
        scale_are_log_variance: bool = True,
        sample_size: Optional[int] = 1,
    ) -> torch.Tensor:
        """
        Estimate KL(SS dist || Mixture of MS dist) using Monte Carlo sampling.

        MC KL = E_p [log p(x) - log q(x)]

        Note: implementing the reverse is not possible since pytorch mixture distribution does not
        support the rsample method.
        """
        # 1. Create the P distribution (the 'source' for samples)
        p_ss_dist = self._to_distribution(
            ss_mean,
            ss_scale,
            scale_are_log_variance=scale_are_log_variance,
        )
        p_ss_dist = dist.Independent(p_ss_dist, 1)

        # 2. Create the Q distribution, i.e., the Mixture in our casse
        q_ms_mix_dist: dist.MixtureSameFamily = self._to_distribution(
            mixture_means,
            mixture_scales,
            overwrite_distribution_name="multisteps_to_singlestep_distribution_mixture",
            scale_are_log_variance=scale_are_log_variance,
            obs_space_only=True,
        )

        # 3. Sample from the P distribution
        # Note: T=MC sample size, K=mixture nb of components (i.e., ms horizon len), E=ensemble size, B=batch size
        T = sample_size

        # Will sample a shape (T x K x E x B)
        samples = p_ss_dist.rsample(
            torch.Size([T])
        )  # Reparametrization trick -> grad flow to the MS model

        # 4. Compute log probabilities
        log_p = p_ss_dist.log_prob(samples)
        log_q = q_ms_mix_dist.log_prob(samples)

        # 5. Standard MC KL = E_p [log p(x) - log q(x)]
        kl_div = log_p - log_q

        # log(1/T * sum^T sum^K log_weights)
        log_sum_exp_reduce_dim = (0, 1)
        assert kl_div.dim() == 3
        kl_div = torch.logsumexp(kl_div, dim=log_sum_exp_reduce_dim) - torch.log(
            torch.tensor(T, device=self.device)
        )

        return kl_div

    def SIW_MP_temporal_mixture_to_dist_loss(
        self,
        mixture_means: torch.Tensor,
        mixture_scales: torch.Tensor,
        ss_mean: torch.Tensor,
        ss_scale: torch.Tensor,
        scale_are_log_variance: bool = True,
        sample_size: Optional[int] = 1,
    ) -> torch.Tensor:
        """
        SIW-MP: Stratified Importance-Weighted mixture-projection loss (legacy ms->ss variant).

        Naming note (RLRP-704): formerly referred to as "SIWAE". As validated for the PME variant,
        this estimator has NO Auto-Encoder component (no decoder ``p(y|z)`` / prior ``r(z)``): the
        "target" is the single-step head ``p`` and the "proposal" is the multi-step mixture ``q``,
        so the importance weight has expectation 1 and the bound is a distribution-to-distribution
        PROJECTION (mass-covering), not a generative SIWAE. Hence "SIW-MP" (Stratified IW
        mixture-projection). NOTE: unlike the PME ``_siw_mp_dist_to_mixture_loss_at_i``, this legacy method
        does NOT include the categorical mixing log-weights ``log pi_k`` and is hard-wired to
        ``history_len == horizon_len`` (uses the static-gamma obs-only mixture); prefer the PME
        per-step estimator for new work.

        Usage note: the SIW-MP loss implementation is not a lower bound, it already is negated.

        Two KL version:
            # ToDo: validate semantic
            Standard MC KL: Samples from p (SS) and computes KL(p || q).
            SIW-MP: Samples from q (MS) and computes KL(q || p).

        These are different objectives:
            # ToDo: validate semantic
            KL(p || q) is mode-seeking for p. It forces the SS head to fit inside one mode of the MS mixture.
            KL(q || p) is mean-covering for p. It forces the SS head to stretch and cover all components of the MS mixture.

        """
        # 1. Setup the SS Head (Target/Prior p)
        p_ss_dist = self._to_distribution(
            ss_mean,
            ss_scale,
            scale_are_log_variance=scale_are_log_variance,
        )
        p_ss_dist = dist.Independent(p_ss_dist, 1)

        # 2. Setup the MS Head Components (Proposal q_m)
        # Unflatten to access individual horizon components: (E x B x MS x Obs_Dim)
        ms_pred_mean = self.unflaten_multistep_composed_array(mixture_means).swapaxes(
            -2, -1
        )
        ms_pred_scale = self.unflaten_multistep_composed_array(mixture_scales).swapaxes(
            -2, -1
        )
        ms_pred_mean = ms_pred_mean[..., : self.singlestep_obs_len]
        ms_pred_scale = ms_pred_scale[..., : self.singlestep_obs_len]

        # SIW-MP Stratified Sampling: Sample 1 point from EACH component i.e., the MS horizon dim
        q_ms_comp_dist = self._to_distribution(
            ms_pred_mean, ms_pred_scale, scale_are_log_variance=scale_are_log_variance
        )
        K = self.horizon_len  # MS horizon

        # 3. Sample from the q distribution
        N = sample_size
        samples = q_ms_comp_dist.rsample(
            torch.Size([N])
        )  # Reparametrization trick -> grad flow to the MS model

        # Note: T=MC sample size, K=mixture nb of components (i.e., ms horizon len), E=ensemble size, B=batch size
        # Permute to (T x K x E x B x Obs_Dim) to treat Horizon as the sample dimension K
        samples = samples.permute(0, 3, 1, 2, 4)

        # 4. Create the Q distribution, i.e., the Mixture in our casse
        q_mix_dist: dist.MixtureSameFamily = self._to_distribution(
            mixture_means,
            mixture_scales,
            overwrite_distribution_name="multisteps_to_singlestep_distribution_mixture",
            scale_are_log_variance=scale_are_log_variance,
            obs_space_only=True,
        )

        # 5. Compute Log Probabilities for the SIW-MP bound
        log_p = p_ss_dist.log_prob(samples)  # (T x K x E x B)
        log_q = q_mix_dist.log_prob(samples)  # (T x K x E x B)

        # 6. SIW-MP Objective Logic: KL = E_q [log (1/T * sum^T sum^K w_k p(x)/q(x))]
        # For k=1 sample per component, the inner sum is over K=Horizon_len.
        log_weights = log_p - log_q

        # log(1/T * sum^T sum^K log_weights)
        assert log_weights.dim() == 4
        SIW_MP_ELBO = torch.logsumexp(log_weights, dim=(0, 1)) - torch.log(
            torch.tensor(N, device=self.device)
        )

        # Turn the evidence lower bound into a minimization objective for PyTorch
        return -SIW_MP_ELBO

    def _resolve_ms_energy_beta(self) -> Tensor:
        """Return the strictly-positive inverse temperature beta used by the 'energy-density'
        reduction. When `ms_energy_beta_learnable` is True it is the softplus of the trainable
        raw parameter (so gradient flows to beta); otherwise it is the fixed buffer value."""
        if self.ms_energy_beta_learnable:
            return F.softplus(self._ms_energy_raw_beta) + weights_eps(self.model_dtype)
        return self._ms_energy_beta_const

    def _energy_axis_reduce(
        self,
        ms_losses: Tensor,
        beta: Union[float, Tensor],
        scale_by_inv_beta: bool,
        sequence_len: int,
        avg_over_sequence_len: bool,
    ) -> Tensor:
        """Energy (free-energy / log-partition) reduction over the configured `ms_energy_axis`.

        Input `ms_losses` is the per-(feature, step) energy tensor E_{d,j} of shape
        (E, B, O+A, MS). Depending on `self.ms_energy_axis`:
          - 'feature' : logsumexp over the feature dim d (O+A) -> per-step free energy (E, B, MS),
                        then SUM over the (normalized) horizon -> (E, B, 1) (keepdim feature axis);
          - 'joint'   : single logsumexp over the whole (d, j) lattice -> (E, B, 1).
        The padded last-step action slots (zero-filled by the composed-array layout) are masked
        OUT of the logsumexp (set to -inf -> 0 contribution) so they do not inflate the partition.
        Returns a (E, B, 1) tensor so the downstream per-feature collapse (`.mean(2, keepdim=True)`)
        is a no-op.
        """
        # Scale by the inverse temperature FIRST, then mask the padded last-step action slots out
        # of the feature/joint partition by setting the SCALED energy to -inf (-> exp(-inf)=0
        # contribution to the partition Z). The composed layout zero-pads the action dims of the
        # LAST horizon step; a 0 energy would otherwise contribute exp(0)=1. Masking AFTER the
        # `beta *` multiply keeps the masked entries a constant (-inf) that does NOT depend on a
        # (possibly learnable) beta, avoiding a 0*(-inf)=NaN gradient back into beta. (Obs dims are
        # never padded, so each feature/joint logsumexp slice retains real entries.)
        scaled = beta * ms_losses
        if self.singlestep_act_len > 0:
            o = self.singlestep_obs_len
            a = self.singlestep_act_len
            mask = torch.zeros_like(scaled, dtype=torch.bool)
            mask[..., o : o + a, -1] = True
            scaled = scaled.masked_fill(mask, float("-inf"))

        if self.ms_energy_axis == "feature":
            # Free energy over the observation/feature dims, per step -> (E, B, MS)
            per_step = torch.logsumexp(scaled, dim=-2, keepdim=False)
            if scale_by_inv_beta:
                per_step = per_step / beta
            # Aggregate the per-step free energies over the (normalized) horizon (matches the
            # 'independent' temporal SUM); `avg_over_sequence_len` adds the optional 1/MS factor.
            reduced = torch.sum(per_step, dim=-1, keepdim=False)
            if avg_over_sequence_len:
                reduced = reduced / sequence_len
        else:  # "joint": one partition over the whole (feature x step) lattice
            flat = scaled.flatten(start_dim=-2)  # (E, B, (O+A)*MS)
            reduced = torch.logsumexp(flat, dim=-1, keepdim=False)
            if scale_by_inv_beta:
                reduced = reduced / beta
            # `avg_over_sequence_len` has no single-axis meaning for the joint lattice -> no-op.
        return reduced.unsqueeze(-1)  # (E, B, 1) keepdim on the collapsed feature axis

    def reduce_multistep_losses_horizon(
        self,
        ms_losses: Tensor,
        probabilistic_losses: bool = True,
        unflaten_composed_array_enabled: bool = True,
        avg_over_sequence_len: bool = False,
        enable_feature_energy_axis: bool = False,
    ) -> Tensor:
        """
        Reduces multistep horizon losses (probabilistic or deterministic losses).

        Given an input tensor of shape (E, B, (MS X Dim)) the function return a tensor of shape
         (E, B, Dim) with ensemble E, batch B, multi-step length MS, and feature size Dim.

        NOTE (RLRP-769): with an ENERGY reduction ('soft-max-energy' / 'energy-density') the
        mode-derived 1/F factor is suppressed (a 1/F on a log-partition is not a mean) and both
        'joint' and 'uniform-mean' skip the temporal weighting -- so those two modes produce
        IDENTICAL losses there. Ablation sweeps should prune the redundant arm.

        This function processes multi-step losses by unflattening and reducing their
        dimensions. It applies different reduction strategies based on the horizon-step
        probability assumption, i.e., the 'ms_probabilities_reduction' parameter, and the
        'ms_temporal_weighting_mode' (which drives the mode-derived 1/F horizon averaging). The
        temporal discount factors are ALWAYS normalized (feature-averaged sum_j w_j = 1, RLRP-769).
        The per-step NLLs are reduced over the horizon as one of:
          - 'independent'     : SUM of per-step NLLs (-log prod_j p_j = sum_j -log p_j);
          - 'prob-sum'        : -logsumexp_j(-nll_j) = -log(sum_j p_j), the NLL of the
                                probability-space sum-over-steps p(X,Y) = p(X)+p(Y) (does NOT model
                                genuine inter-step dependence; formerly 'dependent', RLRP-704);
          - 'soft-max-energy' : logsumexp_j(E_j) = log Z, the Gibbs free energy / log-partition
                                of the per-step energies E_j = nll_j at inverse temperature 1
                                (~= max_j nll_j, a robust worst-step penalty);
          - 'energy-density'  : (1/beta) * logsumexp_j(beta * E_j), the temperature-controlled
                                free energy / log-partition of the Gibbs distribution over
                                horizon steps (beta = `ms_energy_beta`: "learned" or float > 0).

        :param ms_losses: A tensor representing multi-step horizon losses,
            which needs to be reduced horizon wide.
        :param probabilistic_losses: A boolean flag indicating multi-step losses are in log-space.
        :param unflaten_composed_array_enabled: Expect ms_losses to be a flattened composed array
            when True and be an unflaten array of shape (..., O+A, MS) when False.
        :param avg_over_sequence_len: RESERVED / test-only (RLRP-769). When True ALSO divide the
            horizon-reduced loss by the sequence (horizon) length. It is normally DERIVED from
            `ms_temporal_weighting_mode`: forced True for 'uniform-mean' (the mean IS the mode,
            restricted to the sum-family reductions) and False otherwise -- 'tempered' is already
            normalized by the feature-averaged sum_j w_j = 1, so the matching reduction is a pure
            sum, and 'joint'/'log-shift' aggregate by sum by construction. No production call site
            passes True; it survives only for the deterministic/test call path.
            NOTE: the sum-family restriction applies to the DERIVED value ONLY -- an explicit
            True still divides inside the energy branches (legacy behaviour, test-only).
        :param enable_feature_energy_axis: opt-in flag for the primary-forecast call sites only.
            When True AND `ms_energy_axis in {'feature', 'joint'}` AND an energy reduction is used,
            the energy log-partition is taken over the observation/feature axis (and the step axis
            for 'joint'), collapsing the feature dim INSIDE the reduction and returning (E, B, 1).
            Default False keeps the per-feature 'step' behaviour (output (E, B, O+A)) so callers that
            accumulate a per-feature tensor (e.g. the pre-mixture U-loss) are unaffected.
        :return: A tensor containing the reduced multi-step horizon losses.
        """
        if unflaten_composed_array_enabled:
            # Note: unflaten_composed_array_enabled=True also validate the tensor shape
            # (E, B, MS*(O+A)) -> (E, B, O+A, MS). It's the user's responsibility to validate
            # the tensor shape if unflaten_composed_array_enabled=False
            ms_losses = self.unflaten_multistep_composed_array(
                ms_losses, is_model_output=True
            )

        # Every logic branch reduces ms_losses shape (E, B, O+A, MS) -> (E, B, O+A)
        if probabilistic_losses:
            # The horizon `1/F` averaging factor is DERIVED from `ms_temporal_weighting_mode`, it
            # is not a free/config choice (RLRP-769):
            #   uniform-mean             -> 1/F ON  (the mean IS the mode)
            #   log-shift/tempered/joint/discounted-sum -> 1/F OFF ('tempered' is already
            #                               normalized by the feature-averaged sum_j w_j = 1,
            #                               'discounted-sum' is a raw EMA SUM anchored at w_0 = 1,
            #                               and 1/F would destroy the 'joint' -log prod_j p_j
            #                               semantics)
            # The `avg_over_sequence_len` parameter is RESERVED / test-only: the deterministic
            # branch always means over the horizon and ignores it, and no production call site
            # passes True. It is NOT exposed to Hydra.
            # NOTE: restricted to the SUM-family reductions -- a 1/F factor on a logsumexp
            # log-partition (soft-max-energy / energy-density) is not a mean, it is an arbitrary
            # rescaling of a free energy. Mirrors the guard on the sum-family reductions below.
            avg_over_sequence_len = avg_over_sequence_len or (
                self.ms_temporal_weighting_mode == "uniform-mean"
                and self.ms_probabilities_reduction in ("independent", "prob-sum")
            )

            # Note: with `tempered`/`log-shift` the per-step NLLs are reduced
            # (summed/aggregated) over the horizon and `avg_over_sequence_len` only adds an
            # extra 1/MS averaging factor when True (forced on for `uniform-mean`).
            sequence_len = ms_losses.size(-1)
            if self.ms_probabilities_reduction == "independent":
                # We assume the MS horizon steps are independent, p(X,Y) = p(X)p(Y), so the
                # joint NLL is the SUM of per-step NLLs: -log prod_j p_j = sum_j -log p_j.
                ms_losses = torch.sum(ms_losses, dim=-1, keepdim=False)
            elif self.ms_probabilities_reduction == "prob-sum":
                # Aggregate the per-step densities in PROBABILITY space, p(X,Y) = p(X)+p(Y).
                # Its NLL is -log(sum_j p_j) = -logsumexp_j(log p_j) = -logsumexp_j(-nll_j)
                # (nll_j = -log p_j). NOTE: this is a probability-space sum-over-steps, NOT a
                # genuine inter-step-dependent joint p(y_0,...,y_{F-1}) (formerly mis-named
                # 'dependent', RLRP-704).
                ms_losses = -torch.logsumexp(-ms_losses, dim=-1, keepdim=False)
            elif self.ms_probabilities_reduction in (
                "soft-max-energy",
                "energy-density",
            ):
                # ENERGY reductions. 'soft-max-energy' is the Gibbs free energy / log-partition
                # of the per-step energies E_j = nll_j at inverse temperature beta=1
                # (logsumexp_j(E_j) = log Z ~= max_j nll_j, a robust worst-step penalty, NOT a
                # probabilistic-joint NLL). 'energy-density' is the temperature-controlled free
                # energy F(beta) = (1/beta) * logsumexp(beta * E) over the same energies.
                if self.ms_probabilities_reduction == "soft-max-energy":
                    beta = 1.0
                    scale_by_inv_beta = False
                else:  # "energy-density"
                    beta = self._resolve_ms_energy_beta().to(ms_losses.dtype)
                    scale_by_inv_beta = True

                # `ms_energy_axis` selects the partition axis, but only the primary-forecast call
                # sites opt in (`enable_feature_energy_axis=True`); every other caller keeps the
                # per-feature 'step' behaviour (output (E, B, O+A)) so its downstream
                # accumulation/shape is unchanged.
                if enable_feature_energy_axis and self.ms_energy_axis != "step":
                    # 'feature' / 'joint': collapse the feature axis INSIDE the reduction
                    # (with padded-action masking) and return (E, B, 1).
                    ms_losses = self._energy_axis_reduce(
                        ms_losses,
                        beta=beta,
                        scale_by_inv_beta=scale_by_inv_beta,
                        sequence_len=sequence_len,
                        avg_over_sequence_len=avg_over_sequence_len,
                    )
                else:
                    # 'step': partition over the horizon steps j -> (E, B, O+A).
                    ms_losses = torch.logsumexp(
                        beta * ms_losses, dim=-1, keepdim=False
                    )
                    if scale_by_inv_beta:
                        ms_losses = ms_losses / beta
                    if avg_over_sequence_len:
                        ms_losses = ms_losses / sequence_len

            if (
                avg_over_sequence_len
                and self.ms_probabilities_reduction in ("independent", "prob-sum")
            ):
                ms_losses = ms_losses / sequence_len
        else:
            # Deterministic losses (MSE/MAE): always use mean over horizon steps,
            # matching the original per-step averaging regardless of gamma normalization.
            # (NICE TO HAVE) ToDo: implement support for multistep horizon reduction method
            ms_losses = ms_losses.mean(-1, keepdims=False)

        return ms_losses
