# coding=utf-8
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import omegaconf
import torch
import abc
import numpy as np
from torch import Tensor, nn as nn

from tools.console_tools.message import consol_msg_universal_one_liner
from tools.domain_randomization_tools.spec_containers import (
    TrainTimeDomainRandomizationSpec,
)
from tools.math_tools.normalization import min_max_normalization
from tools.multistep_tools.models import AbstractTemporalyWeightedMultiStepMLP
from tools.multistep_tools.models.precision_bounds import weights_eps
from tools.math_tools.weigthing import flags_to_operator, weight_values


#: RLRP-761 S3.4 -- per-feature weighting semantics (see the ``__init__`` note).
_TEMPERED_MODE = "tempered"
_MULTIPLICATIVE_MODE = "multiplicative"
_FEATURE_WEIGHT_MODES = frozenset({_TEMPERED_MODE, _MULTIPLICATIVE_MODE})


def _is_named_criterion(weights) -> bool:
    """Whether *weights* is an UNRESOLVED named criterion (RLRP-761 ``S3.3``).

    ``ms_model.obs_feature_weights`` accepts a criterion NAME (see
    ``feature_loss_weights.CRITERIA``) which can only be turned into numbers once
    the dataset exists -- i.e. at the training-start seam, long after Hydra has
    instantiated the model. The constructor therefore receives the bare string.
    """
    return isinstance(weights, str)


def _validate_named_criterion(weights, field_name: str) -> None:
    """Fail fast on a criterion typo instead of silently running unweighted."""
    from tools.feature_handling_tools.feature_loss_weights import CRITERIA

    if weights not in CRITERIA:
        raise ValueError(
            f"{field_name}={weights!r} is not a known feature-weight criterion; "
            f"expected one of {CRITERIA}, a scalar, or a per-dim sequence."
        )


def _is_neutral_feature_weights(weights) -> bool:
    """Whether *weights* is the neutral (all-ones) specification.

    RLRP-761 S3.6 / blocker ``B2``. Both the scalar ``1.0`` and a uniform vector
    of ones are neutral **specifications**; only the scalar used to be recognized
    as such, so a resolved ``[1.0] * D`` (or an omegaconf ``ListConfig`` of ones)
    silently activated the non-neutral weighting path.

    A NAMED CRITERION is neutral *at construction time*: its numeric value is not
    knowable yet, so the weighting path stays OFF and is switched on later by
    ``set_observation_feature_weights`` / ``set_action_feature_weights`` (both of
    which self-enable it). Iterating the string instead -- the historical
    behaviour -- raised ``could not convert string to float: 'r'``.
    """
    if weights is None:
        return True
    if isinstance(weights, str):
        return True
    if isinstance(weights, (int, float)):
        return float(weights) == 1.0
    try:
        values = [float(w) for w in weights]
    except (TypeError, ValueError):
        return False
    return bool(values) and all(v == 1.0 for v in values)


class AbstractFeatureWeightedMultiStepMLP(
    AbstractTemporalyWeightedMultiStepMLP, abc.ABC
):
    obs_feature_weights: None | Tuple[float, ...]
    act_feature_weights: None | Tuple[float, ...]
    _composed_next_obs_feature_weights: torch.Tensor
    _composed_next_obs_feature_normalization_weights: torch.Tensor
    # Feature-weight clamp floor is dtype-aware, finfo-derived via
    # ``precision_bounds.weights_eps`` (== ``finfo(dtype).eps``: ~1.19e-7 float32 /
    # ~2.22e-16 float64, RLRP-750 -- relative-precision floor for O(1) weights,
    # replacing the former frozen 1e-10/1e-15 values); see the clamp site below.

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
        dropout: float = 0.0,
        ms_head_dropout: float = 0.0,
        ms_head_num_layers: int = 1,
        residual_form: str = "identity_v2",
        layer_bloc=None,  # RLRP-768 per-region layer-bloc selector (forwarded)
        ms_temporal_weighting_mode: str = "discounted-sum",
        enable_auto_loss_weighting: bool = False,
        ms_probabilities_reduction: str = "independent",
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
            temporal_weights=temporal_weights,
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

        # .... composed observation feature weights ...............................................
        # RLRP-761 S3.4: how a per-feature weight enters the loss.
        #   'tempered'       -> `values + log(w)` at the MS/SS/CP call sites. On a
        #                       per-feature NLL that is an ADDITIVE CONSTANT, so
        #                       under the (linear) downstream reduction it leaves
        #                       `dL/dtheta` UNCHANGED: it tempers the reported
        #                       likelihood, it does NOT re-allocate the loss
        #                       budget. Historical behaviour, kept only to
        #                       reproduce past runs.
        #   'multiplicative' -> `w * values`: the actual per-feature loss
        #                       re-weighting (RLRP-761 blocker `B1`).
        if feature_weight_mode not in _FEATURE_WEIGHT_MODES:
            raise ValueError(
                f"feature_weight_mode={feature_weight_mode!r} is unknown; expected "
                f"one of {sorted(_FEATURE_WEIGHT_MODES)}."
            )
        self.feature_weight_mode = feature_weight_mode
        self.feature_weight_max_ratio = feature_weight_max_ratio

        # RLRP-761 S3.3: a named criterion is resolved at the training-start seam
        # (`resolve_and_apply_feature_loss_weights`), not here -- the dataset does
        # not exist yet. Validate the NAME now so a typo fails fast instead of
        # silently degrading to an unweighted run.
        for _spec, _field in (
            (obs_feature_weights, "obs_feature_weights"),
            (act_feature_weights, "act_feature_weights"),
        ):
            if _is_named_criterion(_spec):
                _validate_named_criterion(_spec, _field)

        # RLRP-761 S3.6 (blocker `B2`): a GENUINELY NEUTRAL specification must
        # short-circuit the whole weighting path. The former `!= 1.0` scalar
        # comparison made a resolved `[1.0] * D` vector (or an omegaconf
        # `ListConfig` of ones) activate it -- and activation is not neutral,
        # since `_normalize_feature_weights` divides by the sum and turns uniform
        # ones into `1/out_size`, i.e. a `log w ~ -5.3` shift on every element
        # under the 'tempered' semantic (and a non-trivial coupling to the
        # task-level Kendall auto-weighting).
        self._feature_weighting_enabled = not (
            _is_neutral_feature_weights(obs_feature_weights)
            and _is_neutral_feature_weights(act_feature_weights)
        )

        if self._feature_weighting_enabled:
            if isinstance(obs_feature_weights, float):
                obs_feature_weights = (obs_feature_weights,) * singlestep_obs_len
            else:
                assert isinstance(obs_feature_weights, Sequence)
                assert len(obs_feature_weights) == singlestep_obs_len, (
                    f"len(obs_feature_weights) != " f"singlestep_obs_len"
                )

            if isinstance(act_feature_weights, float):
                act_feature_weights = (act_feature_weights,) * singlestep_act_len
            else:
                assert isinstance(act_feature_weights, Sequence)
                assert len(act_feature_weights) == singlestep_act_len, (
                    f"len(act_feature_weights) != " f"singlestep_act_len"
                )

            # RLRP-761 S3.5 (blocker `B3`): bound the DYNAMIC RANGE, not the
            # absolute values. `_normalize_feature_weights` divides the composed
            # vector by its own sum, so only the ratios survive and an absolute
            # clip is meaningless after the fact.
            obs_feature_weights, act_feature_weights = (
                self._squash_feature_weight_dynamic_range(
                    obs_feature_weights, act_feature_weights
                )
            )

            # Note: composed next obs feature weights := (obs feature weights, act feature weights)
            self.obs_feature_weights = obs_feature_weights
            self.act_feature_weights = act_feature_weights
            self._init_composed_next_obs_feature_weights()
            consol_msg_universal_one_liner(
                f"Feature weigthing enabled (mode={self.feature_weight_mode}, "
                f"obs={tuple(round(float(w), 4) for w in obs_feature_weights)}, "
                f"act={tuple(round(float(w), 4) for w in act_feature_weights)})"
            )
        else:
            self.obs_feature_weights = None
            self.act_feature_weights = None
            consol_msg_universal_one_liner("Feature weigthing disabled")

        # .... Final build step ...................................................................
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

    # :::: Feature weight update utilities ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def apply_next_obs_feature_weights(
        self,
        values: Union[torch.Tensor, np.ndarray],
        log_space: bool = False,
        values_centred_around_zero=False,
        apply_minus_log=False,
        pre_process_weight: Optional[Callable] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Apply feature weights to value tensor. Handle negative value case

        :param values: Input tensor.
        :param log_space: Set to True if values are in log space.
        :param values_centred_around_zero: Whether values in normal space are centered around zero
            or not. Only valid with 'log_space=False'.
        :param apply_minus_log: Execute `values - log weights` instead of `values + log weights`.
            Only valid with 'log_space=True'.
        :param pre_process_weight: A function that will be applied to values_weights before weighting
        :return: A feature weighted value tensor.
        """
        if not self._feature_weighting_enabled:
            return values

        if self.feature_weight_mode == _MULTIPLICATIVE_MODE and log_space:
            # RLRP-761 S3.4 / blocker `B1`. The caller asked for the historical
            # log-space tempering (`values + log w`), which is an ADDITIVE
            # CONSTANT on a per-feature NLL and therefore gradient-inert; the
            # configured semantic is the genuine per-feature re-weighting, so
            # apply `w * values` instead.
            #
            # Deliberately NOT routed through `apply_weights` (blocker `B4`):
            # that helper contains
            #     elif values.shape[-1] == values_weights.shape[-1]:
            #         values_weights = values_weights.mean(dim=0)
            # which, for the 1-D composed weight vector used here, collapses the
            # whole vector to its SCALAR MEAN — silently destroying the very
            # per-feature structure this stage exists to apply. The reduction is
            # meant for a 2-D (per-ensemble) weight tensor; it is left untouched
            # because the temporal-weighting caller shares the helper.
            weights = self._composed_next_obs_feature_weights
            if pre_process_weight:
                weights = pre_process_weight(weights)
            return values * weights.to(
                device=values.device, dtype=values.dtype
            )

        return weight_values(
            values,
            values_weights=self._composed_next_obs_feature_weights,
            operator=flags_to_operator(
                log_space=log_space,
                values_centred_around_zero=values_centred_around_zero,
                apply_minus_log=apply_minus_log,
            ),
            pre_process_weight=pre_process_weight,
        )

    def _squash_feature_weight_dynamic_range(
        self,
        obs_feature_weights: Tuple[float, ...],
        act_feature_weights: Tuple[float, ...],
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """Bound ``max/min`` of the composed weight vector (RLRP-761 ``S3.5``).

        The bound is enforced on the ``obs + act`` concatenation, since that is
        the vector whose ratios reach the loss. A no-op when
        ``feature_weight_max_ratio`` is unset.

        :param obs_feature_weights: The per-single-step obs weights.
        :param act_feature_weights: The per-single-step act weights.
        :return: The (possibly squashed) ``(obs, act)`` weights.
        """
        max_ratio = self.feature_weight_max_ratio
        if max_ratio is None:
            return tuple(obs_feature_weights), tuple(act_feature_weights)

        from tools.feature_handling_tools.feature_loss_weights import (
            squash_dynamic_range,
        )

        composed = list(obs_feature_weights) + list(act_feature_weights)
        squashed, did_bind = squash_dynamic_range(composed, max_ratio=float(max_ratio))
        if did_bind:
            consol_msg_universal_one_liner(
                f"Feature weight dynamic range squashed to "
                f"max/min <= {float(max_ratio)} "
                f"(pre={tuple(round(float(w), 4) for w in composed)}, "
                f"post={tuple(round(float(w), 4) for w in squashed)})"
            )
        n_obs = len(obs_feature_weights)
        return tuple(squashed[:n_obs]), tuple(squashed[n_obs:])

    def compute_ideal_obs_feature_weights_update_from_prediction_std_trajectory(
        self, pred_epi_trj: np.ndarray, scale_min: float = 0.5, scale_max: float = 1.0
    ) -> Tuple[float, ...]:
        """
        Utility for computing the ideal observation feature weight update from an array of
        ensemble model multistep prediction standard deviations
        of shape (feature_1 X timestep + feature_2 X timestep ..., ).
        Returned values are [0:1] normalized.

        :param pred_epi_trj: The predicted standard deviations.
        :param scale_min: feature weight scaling min value
        :param scale_max: feature weight scaling max value
        :return: A tuple containing the ideal observation feature weights update.
        """
        assert pred_epi_trj.ndim == 2, f"{pred_epi_trj.ndim=} != 2"
        pred_epi_trj = pred_epi_trj.copy()

        pred_epi_ss = self.reduce_multistep_prediction_to_single_step(
            pred_epi_trj, reduction="mean", apply_discount_factor=True
        )

        BATCH_INDEX = 0
        batch_pred_epi = np.mean(pred_epi_ss, axis=BATCH_INDEX)

        scaled_feature_weight = min_max_normalization(
            batch_pred_epi, scale_min, scale_max
        )

        if self.horizon_len > 1:
            composed_obs_len = self.singlestep_obs_len + self.singlestep_act_len
        else:
            composed_obs_len = self.singlestep_obs_len

        assert (
            scaled_feature_weight.size == composed_obs_len
        ), f"{scaled_feature_weight.size=} != {composed_obs_len=}"

        assert np.all(
            np.isfinite(scaled_feature_weight)
        ), f"{scaled_feature_weight=} as non finite value(s)!"

        scaled_feature_weight = tuple(scaled_feature_weight.tolist())
        return scaled_feature_weight

    # :::: Composed obs/act feature :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def _init_composed_next_obs_feature_weights(self) -> None:
        self.register_buffer(
            "_composed_next_obs_feature_weights",
            torch.ones((self.out_size,), device=self.device, dtype=self.model_dtype),
        )

        self._init_observation_feature_weights(self.obs_feature_weights)
        self._init_action_feature_weights(self.act_feature_weights)

        self._normalize_feature_weights()

        return None

    def _init_observation_feature_weights(
        self, obs_feature_weights: Tuple[float, ...]
    ) -> None:
        self._init_feature_weights(
            start_idx=0,
            stop_idx=self._compose_next_obs_multistep_obs_horizon_slice.stop,
            single_step_features_size=self._compose_next_obs_multistep_obs_horizon_slice.step,
            feature_weights=obs_feature_weights,
        )
        return None

    def _init_action_feature_weights(
        self, act_feature_weights: Tuple[float, ...]
    ) -> None:
        if self.horizon_len > 1:
            # RLRP-761 S3.6: `stop_idx` is `out_size`, not `out_size - 1`. The
            # loop walks `single_step_features_size`-wide blocks, so the historical
            # `- 1` silently DROPPED the final action block whenever
            # `singlestep_act_len == 1` (with a wider action block the last block
            # start still satisfies `< out_size - 1`, which is why the defect went
            # unnoticed). No composed index is left unweighted now.
            self._init_feature_weights(
                start_idx=self._compose_next_obs_multistep_obs_horizon_slice.stop,
                stop_idx=self.out_size,
                single_step_features_size=self._compose_next_obs_multistep_act_horizon_slice.step,
                feature_weights=act_feature_weights,
            )
        return None

    def set_observation_feature_weights(
        self, feature_weights: Tuple[float, ...]
    ) -> None:
        """Update the multistep model observation feature weights at runtime.

        :param feature_weights: A tuple of floats representing the new obs/act feature weights.
        :return: None
        """
        obs_feature_weight = feature_weights[0 : self.singlestep_obs_len]

        # (CRITICAL) ToDo: validate casse where is not enabled
        if not self._feature_weighting_enabled:
            self._feature_weighting_enabled = True
            self.obs_feature_weights = (1.0,) * self.singlestep_obs_len
            self.act_feature_weights = (1.0,) * self.singlestep_act_len
            self._init_composed_next_obs_feature_weights()

        # .... Sanity check .......................................................................
        assert len(obs_feature_weight) == len(
            self.obs_feature_weights
        ), f"{len(obs_feature_weight)=} != {len(self.obs_feature_weights)=}"
        assert torch.all(
            torch.isfinite(torch.tensor(obs_feature_weight))
        ), f"{obs_feature_weight=} as non finite value(s)!"

        # .... Un-normalize feature weights .......................................................
        self._composed_next_obs_feature_weights *= (
            self._composed_next_obs_feature_normalization_weights
        )

        # .... Execute ............................................................................
        self._reset_feature_weights(
            start_idx=0,
            stop_idx=self._compose_next_obs_multistep_obs_horizon_slice.stop,
            single_step_features_size=self._compose_next_obs_multistep_obs_horizon_slice.step,
        )
        self._init_observation_feature_weights(obs_feature_weight)

        self._normalize_feature_weights()

        return None

    def _normalize_feature_weights(self) -> None:
        normalization_weights = torch.sum(
            self._composed_next_obs_feature_weights, dim=-1, keepdim=True
        )
        # Register on first call so the tensor follows `self.to(device)` together with
        # its sibling buffer `_composed_next_obs_feature_weights`. Without this, the
        # plain-attribute version stays pinned to the init-time device and collides
        # with the (correctly moved) sibling buffer on CUDA.
        if "_composed_next_obs_feature_normalization_weights" not in self._buffers:
            self.register_buffer(
                "_composed_next_obs_feature_normalization_weights",
                normalization_weights,
            )
        else:
            self._composed_next_obs_feature_normalization_weights = normalization_weights
        self._composed_next_obs_feature_weights /= normalization_weights
        return None

    def set_action_feature_weights(self, feature_weights: Tuple[float, ...]) -> None:
        """Update the multistep model action feature weights at runtime.

        :param feature_weights: A tuple of floats representing the new obs/act feature weights.
        :return: None
        """
        act_feature_weights = feature_weights[-self.singlestep_act_len :]

        # (CRITICAL) ToDo: validate casse where is not enabled
        if not self._feature_weighting_enabled:
            self._feature_weighting_enabled = True
            self.obs_feature_weights = (1.0,) * self.singlestep_obs_len
            self.act_feature_weights = (1.0,) * self.singlestep_act_len
            self._init_composed_next_obs_feature_weights()

        if self.horizon_len > 1:
            # .... Sanity check ...................................................................
            assert len(act_feature_weights) == len(
                self.act_feature_weights
            ), f"{len(act_feature_weights)=} != {len(self.act_feature_weights)=}"
            assert torch.all(
                torch.isfinite(torch.tensor(act_feature_weights))
            ), f"{act_feature_weights=} as non finite value(s)!"

            # .... Un-normalize feature weights ...................................................
            self._composed_next_obs_feature_weights *= (
                self._composed_next_obs_feature_normalization_weights
            )

            # .... Execute ........................................................................
            self._reset_feature_weights(
                start_idx=self._compose_next_obs_multistep_obs_horizon_slice.stop,
                stop_idx=self.out_size,
                single_step_features_size=self._compose_next_obs_multistep_act_horizon_slice.step,
            )
            self._init_action_feature_weights(act_feature_weights)

            self._normalize_feature_weights()

        return None

    def _reset_feature_weights(
        self, start_idx: int, stop_idx: int, single_step_features_size: int
    ) -> None:
        # (CRITICAL) ToDo: implement test case
        for each_obs_idx in range(start_idx, stop_idx, single_step_features_size):
            self._composed_next_obs_feature_weights[
                each_obs_idx : each_obs_idx + single_step_features_size
            ] = 1.0

        return None

    def _init_feature_weights(
        self,
        start_idx: int,
        stop_idx: int,
        single_step_features_size: int,
        feature_weights: tuple,
    ) -> None:
        for each_obs_idx in range(start_idx, stop_idx, single_step_features_size):
            self._composed_next_obs_feature_weights[
                each_obs_idx : each_obs_idx + single_step_features_size
            ] *= torch.tensor(feature_weights, device=self.device)

        # .... Set minimum weight for numerical stability .........................................
        self._composed_next_obs_feature_weights.clamp_(min=weights_eps(self.model_dtype))

        # .... Sanity check .......................................................................
        assert torch.all(
            torch.isfinite(self._composed_next_obs_feature_weights)
        ), f"{self._composed_next_obs_feature_weights=} as non finite value(s)!"

        return None
