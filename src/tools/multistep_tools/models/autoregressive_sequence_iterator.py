# coding=utf-8
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import omegaconf
import torch

from tools.multistep_tools.buffer_unit import MultistepBuffer
from tools.multistep_tools.multistep_model_util import (
    _case_sampled_with_bootstrap_iterator_false,
    reshape_sequence_dim_into_batch_dim,
    _batch_sequence_timestep_view,
    timestep_first_multistep_dim_unflaten_array,
)
from tools.math_tools.ndarray_tools.custom_msg import nan_infinity_console_warning

from tools.multistep_tools.models.weighted_multistep_mlp import WeightedMultiStepMLP


class AutoRegressiveSequenceIterator(WeightedMultiStepMLP):
    _debug_mode: bool = (
        False  # Console print tensor shape for fast debugging via run mode.
    )

    ar_memory: Optional[torch.Tensor] = None

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
        ms_temporal_weighting_mode: str = "discounted-sum",
        ar_enabled: bool = True,
        receive_sequence_batch: bool = True,
        ss_composite_loss_weight: float = 1.0,
        ms_composite_loss_weight: float = 1.0,
        ar_train_horizon_unroll: bool = False,
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
            logvar_bound_grad_clip=logvar_bound_grad_clip,
            auto_weighting_noise_model=auto_weighting_noise_model,
            auto_weighting_scheme=auto_weighting_scheme,
            activation_fn_cfg=activation_fn_cfg,
            distribution_name=distribution_name,
            mae_loss=mae_loss,
            model_use_double_precision=model_use_double_precision,
            dropout=dropout,
            ms_temporal_weighting_mode=ms_temporal_weighting_mode,
            ss_composite_loss_weight=ss_composite_loss_weight,
            ms_composite_loss_weight=ms_composite_loss_weight,
            enable_auto_loss_weighting=enable_auto_loss_weighting,
            ms_probabilities_reduction=ms_probabilities_reduction,
            ms_energy_beta=ms_energy_beta,
            ms_energy_axis=ms_energy_axis,
            description=description,
            feature_geometry=feature_geometry,
        )

        # .... input/output related ...............................................................
        self.ar_enabled = ar_enabled
        self.receive_sequence_batch = receive_sequence_batch
        self.ar_train_horizon_unroll = ar_train_horizon_unroll

        self.ar_memory = None

        # .... Final build step ...................................................................
        if model_use_double_precision:
            self.to(dtype=torch.double)
        else:
            self.to(dtype=self.model_dtype)

    def wipe_ar_memory(self) -> None:
        self.ar_memory = None
        if self._debug_mode:
            print(f">>>>> Reset ar_memory (mode {self.training=})")
        return None

    def loss(
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Note:
        # - Expect samples are given by SequenceTransitionIterator with _bootstrap_iter==True
        # - method `Model.loss` is only used by `Model.update` method

        if (
            self.ar_enabled
            and self.ar_train_horizon_unroll
            and not self.receive_sequence_batch
        ):
            # Case: classical auto-regressive setup
            self.wipe_ar_memory()
            self._check_input_and_target_required_dim(model_in, target, 3, 2)

        elif self.receive_sequence_batch:
            warnings.warn(
                "BE ADVISED: loss with receive_sequence_batch=True is still in beta release"
            )
            # iceboxed: implement loss of sequence batch of multisteps observations (hi:ho)

            # Case: sequences batch of multistep history and horizon samples
            self.wipe_ar_memory()
            self._check_input_and_target_required_dim(model_in, target, 4, 3)

            # .... Assess mod inspired from torch.nn.gaussian_nll_loss implementation .............
            # # Original
            # with torch.inference_mode():
            #     self.eval()
            #     processed_model_in = self._process_sequence_batch(model_in.detach_())
            #     self.train()
            #     del model_in  # memory management
            #
            # model_in = processed_model_in.clone().requires_grad_()
            # del processed_model_in  # memory management

            model_in = model_in.clone()
            with torch.inference_mode():
                self.eval()
                with torch.no_grad():
                    model_in = self._process_sequence_batch(model_in)

            model_in = model_in.clone().requires_grad_()
            self.train()

            # Reset memory again before running the super loss
            self.wipe_ar_memory()
            # .............................................................................(end)...

            target, sequence_len = reshape_sequence_dim_into_batch_dim(
                target, self.num_members
            )

        elif not self.ar_enabled and not self.receive_sequence_batch:
            # Case: AR capabilities disabled and receive standard samples shape
            assert model_in.ndim == target.ndim

        # Note: Be advised that mbrl-lib 'loss' method call forward ⚠️
        losses, meta = super().loss(model_in, target)
        return losses, meta

    def eval_score(
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if self.ar_enabled and not self.receive_sequence_batch:
            # Case: classical auto-regressive setup
            self.wipe_ar_memory()
            self._check_input_and_target_required_dim(model_in, target, 2, 2)

        elif self.receive_sequence_batch:
            warnings.warn(
                "BE ADVISED: eval_score with receive_sequence_batch=True is still in beta release"
            )
            # iceboxed: implement eval of sequence batch of multisteps observations (hi:ho)

            with torch.inference_mode():
                self.eval()

                # Case: sequences batch of multistep history and horizon samples
                self.wipe_ar_memory()
                self._check_input_and_target_required_dim(model_in, target, 3, 3)

                # .... Assess mod inspired from torch.nn.gaussian_nll_loss implementation .........
                # # Original
                # processed_model_in = self._process_sequence_batch(model_in.detach_())
                # del model_in  # memory management
                # model_in = processed_model_in.clone()
                # del processed_model_in  # memory management

                model_in = model_in.clone()
                with torch.no_grad():
                    model_in = self._process_sequence_batch(model_in)

                # Reset memory again before running the super eval_score
                self.wipe_ar_memory()
                # .........................................................................(end)...

                target, sequence_len = reshape_sequence_dim_into_batch_dim(
                    target, self.num_members
                )

        elif not self.ar_enabled and not self.receive_sequence_batch:
            # Case: AR capabilities disabled and receive standard samples shape
            ...

        # Note: Be advised that mbrl-lib 'eval_score' method call forward ⚠️
        losses, meta = super().eval_score(model_in, target)
        return losses, meta

    def _check_input_and_target_required_dim(
        self,
        model_in: torch.Tensor,
        target: torch.Tensor | None,
        ensemble_required_dim: int,
        non_ensemble_required_dim: int,
    ):
        if self.num_members > 1:
            assert (
                model_in.ndim == ensemble_required_dim
                and target.ndim == ensemble_required_dim
            ), f"{model_in.ndim=} != {ensemble_required_dim} and/or {target.ndim=} != {ensemble_required_dim}"
        else:
            # Case no model ensemble
            assert (
                model_in.ndim == non_ensemble_required_dim
                and target.ndim == non_ensemble_required_dim
            ), f"{model_in.ndim=} != {non_ensemble_required_dim} and/or {target.ndim=} != {non_ensemble_required_dim}"

    # :::: Sequence processing tools ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def _process_sequence_batch(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Compounded prediction re-implementation of the `GaussianMLP.forward` method.
        Perform prediction in an autoregressive fashion.

        Take input of dimension 2,3 or 4. Input/output shape when setting
        `ar_enabled=True, receive_sequence_batch=True` are either:
         - update case: (E, B, S, Idim) --> (E, B * S, Odim)
         - evaluation case:  (B, S, Idim) --> (B * S, Odim)
         with E, B, S, Idim and Odim being ensemble size, batch size, sequence len, input dimension
         size and output dimension size respectively. Note: that it require data to be feed via a
         `SequenceTransitionIterator` object.

         Otherwize input/output should have shape
         - update case: (E, B, Idim) --> (E, B, Odim)
         - evaluation case: (B, Idim) --> (B, Odim)
        """
        if not torch.all(torch.isfinite(x)):
            nan_infinity_console_warning("_process_sequence_batch --> x")

        # Note: keep the debug variable and console print as its realy helpfull for fast debugging
        if self._debug_mode:
            print(f"\n//=={x.shape=}=============================")

        if self.ar_enabled and self.receive_sequence_batch and x.ndim >= 3:
            sequence_batch_x, _, _ = (
                self._reprocess_input_to_compounded_prediction_input(
                    x,
                    rng,
                    collect_means=False,
                    collect_log_variances=False,
                )
            )
        else:
            if self.receive_sequence_batch and x.ndim > 2:
                sequence_batch_x, _ = reshape_sequence_dim_into_batch_dim(
                    x, self.num_members
                )

            elif self.receive_sequence_batch and x.ndim == 2:
                # Note: Is executing a no-batch + single-step forward step i.e. env sampling step
                # (CRITICAL) ToDo: assess usefullness of handling this case (ref task RLRP-220)
                # (CRITICAL) ToDo: implement test case (ref task RLRP-220)
                return x
            else:
                return x

        if self._debug_mode:
            debug_log = f"{x.shape=}\n{sequence_batch_x.shape}"

            print(f"{'=' * 54}\nRecomputed x\n{debug_log}{'=' * 54}\n")

        if not torch.all(torch.isfinite(sequence_batch_x)):
            nan_infinity_console_warning("_process_sequence_batch --> sequence_batch_x")

        return sequence_batch_x

    def _reprocess_input_to_compounded_prediction_input(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        collect_means: bool = False,
        collect_log_variances: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Take an input tensor of single-step sequence and re-compute the observation dimension
        as compounded prediction observation.

        Take input of dimension 3 or 4. Input/output shape when setting
        `ar_enabled=True, receive_sequence_batch=True` are either:
         - update case: (E, B, S, Idim) --> (E, B * S, Odim)
         - evaluation case:  (B, S, Idim) --> (B * S, Odim)
         with E, B, S, Idim and Odim being ensemble size, batch size, sequence len, input dimension
         size and output dimension size respectively. Note: that it require data to be feed via a
         `SequenceTransitionIterator` object.

        :param x: Input tensor of shape (E, B, S, Idim) or (B, S, Idim).
        :param rng: Random number generator instance for probabilistic computations.
        :param collect_means: Set to True to compute the ensemble means tensor.
        :param collect_log_variances: Set to True to compute the ensemble log variances tensor.
        :return: compounded_x, compounded_pred_means, compounded_pred_logvars of shape
         (E, B * S, Odim) or (B * S, Odim). Tensor compounded_pred_logvars will be set to None
         if param `collect_log_variance=False`.
        """
        assert self.ar_enabled and self.receive_sequence_batch and x.ndim >= 3

        # Note: keep the debug variable and console print as its realy helpfull for fast debugging
        if self._debug_mode:
            print(f"\n//=={x.shape=}=============================")

        # .... Recompute the input tensor x ...................................................
        original_shape = x.shape
        sequence_len = x.size(-2)
        device = x.device

        multistep_pred_buffer = MultistepBuffer(
            multistep_len=self.history_len,
            single_step_len=self.singlestep_obs_len,
            info="Obs pred history",
            consol_log=False,
        )

        timestep_0 = _batch_sequence_timestep_view(x, 0)
        timestep_0_obs_ms = self.extract_obs_features_from_multistep_composed_array(
            timestep_0, is_model_output=False
        )

        initial_obs = timestep_0_obs_ms[..., 0]
        if self._debug_mode:
            print(f"{timestep_0.shape=}")
            print(f"{timestep_0_obs_ms.shape=}")
            print(f"{initial_obs.shape=}")

        multistep_pred_buffer.reset(initial_obs)

        for idx in range(1, self.history_len):
            multistep_pred_buffer.append(timestep_0_obs_ms[..., idx])

        # (CRITICAL) ToDo: validate boostrap check logic
        bootstrap_iterator_false = _case_sampled_with_bootstrap_iterator_false(x)

        step = _batch_sequence_timestep_view(x, 0)
        compounded_x = [step]

        compounded_pred_means = []
        compounded_pred_logvars = []
        assert sequence_len > 1
        for timestep in range(sequence_len):
            mean_t, logvar_t = super().forward(
                step,
                rng,
                None,
                use_propagation=False,  # ignore propagation method for inference time
            )

            # (CRITICAL) ToDo: Maybe missing the loss computation per step!

            # mean_t = mean_t.detach()
            # if collect_log_variances:
            #     logvar_t = logvar_t.detach()

            if self._debug_mode:
                # print(f"{timestep_view_.shape=}")
                print(f"{step.shape=}")
                print(f"{mean_t.shape=}")

            if collect_means:
                compounded_pred_means.append(mean_t)
            if collect_log_variances:
                compounded_pred_logvars.append(logvar_t)

            if timestep >= sequence_len - 1:
                break

            # (PRIORITY) ToDo: refactor to use method `sample_1d` instead
            # Note: ensemble vote
            if bootstrap_iterator_false:
                mean_t = mean_t.mean(0)

            mean_t = self.multistep_to_singlestep_next_obs_adapter(mean_t)
            multistep_pred_buffer.append(mean_t)
            if self._debug_mode:
                print(f"adapted mean_t.shape={mean_t.shape}")

            next_obs = multistep_pred_buffer.get_buffer()
            # .to(device=device)
            next_act = _batch_sequence_timestep_view(x, timestep + 1)[
                ..., -self.singlestep_act_len * self.history_len :
            ]

            if self._debug_mode:
                print(f"{next_obs.shape=}")
                print(f"{next_act.shape=}")

            step = torch.cat([next_obs, next_act], dim=-1)
            compounded_x.append(step)
            if self._debug_mode:
                print(f"{step.shape=}")

            # x = self._batch_sequence_insert_at_next_timestep_indice(x, step, timestep)

        # (CRITICAL) ToDo: validate element ordering
        x = torch.cat(compounded_x, dim=-2)
        # torch.swapaxes(torch.stack(compounded_x, dim=-2, ), -3, -2).reshape(1, -1, 20)

        if self._debug_mode:
            debug_log = f"{original_shape=}\n{len(compounded_x)=}\n"

        if collect_means:
            # (CRITICAL) ToDo: validate element ordering
            compounded_pred_means = torch.cat(compounded_pred_means, dim=-2)
            if self._debug_mode:
                debug_log = f"{compounded_pred_means.shape=}\n"
        else:
            compounded_pred_means = None

        if collect_log_variances:
            # (CRITICAL) ToDo: validate element ordering
            compounded_pred_logvars = torch.cat(compounded_pred_logvars, dim=-2)
            if self._debug_mode:
                debug_log += f"{compounded_pred_logvars.shape=}\n"
            if collect_means:
                assert compounded_pred_means.shape == compounded_pred_logvars.shape
        else:
            compounded_pred_logvars = None

        if self._debug_mode:
            print(f"{'-' * 54}\n{debug_log}{'-' * 54}\n")

        # .... memory management ..................................................................
        del (
            compounded_x,
            step,
            next_obs,
            next_act,
            mean_t,
            logvar_t,
            multistep_pred_buffer,
        )

        # (CRITICAL) ToDo: Maybe missing the sequence loss computation averaged over sequence length!
        return x, compounded_pred_means, compounded_pred_logvars
