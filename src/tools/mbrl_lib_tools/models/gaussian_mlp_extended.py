# coding=utf-8
import math
from typing import Any, Dict, Optional, Tuple, Union

import omegaconf
import torch
from mbrl.models import GaussianMLP

from tools.mbrl_lib_tools.models.prediction_statistics import SENTINEL_LOGVAR
from tools.multistep_tools.models.exponential_family_mlp_utils import \
    fast_distribution_mode_approximation


class GaussianMLPExtended(GaussianMLP):

    # Near-deterministic logvar fill value; single canonical source (RLRP-761 S10.4).
    # See prediction_statistics.SENTINEL_LOGVAR / ExponentialFamilyMLP for the rationale.
    _LOGVAR_MIN_LIMIT = SENTINEL_LOGVAR

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        deterministic: bool = False,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        description: Optional[str] = None,
    ):

        super().__init__(
            in_size,
            out_size,
            device,
            num_layers,
            ensemble_size,
            hid_size,
            deterministic,
            propagation_method,
            learn_logvar_bounds,
            activation_fn_cfg,
        )
        self.description = description

    def forward(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        ensemble_means, ensemble_logvars = super().forward(
            x, rng, propagation_indices, use_propagation
        )

        if ensemble_logvars is None:
            # Quick-hack for casse model has an ensemble of one or is deterministic
            ensemble_logvars = torch.full_like(ensemble_means, self._LOGVAR_MIN_LIMIT)

        return ensemble_means, ensemble_logvars

    def sample_1d(
        self,
        model_input: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        next_state_sampling_size: int = 1,
        rng: Optional[torch.Generator] = None,
        epi_knn=False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        # (CRITICAL) ToDo: implement test case (ref task RLRP-332)
        # (Priority) ToDo: RLRP-354 feat(GaussianMLPExtended): override extend `_forward_ensemble`
        # with 'post_sampling_expectation' logic
        # (Priority) ToDo: RLRP-389 feat: implement multistep sample method logic (Epi k-nearest)
        """Samples an output from the model.

        Note: re-implementation which output ensemble mean and logvariance via the `model_state`

        This method will be used by :class:`ModelEnv` to simulate a transition of the form.
            outputs_t+1, s_t+1 = sample(model_input_t, s_t), where

            - model_input_t: observation and action at time t, concatenated across axis=1.
            - s_t: model state at time t (as returned by :meth:`reset()` or :meth:`sample()`.
            - outputs_t+1: observation and reward at time t+1, concatenated across axis=1.

        The default implementation returns `s_t+1=s_t`.

        :param model_input: the observation and action at.
        :param model_state: the model state st. Must contain a key "propagation_indices" to use
         for uncertainty propagation.
        :param deterministic: if ``True``, the model returns a deterministic "sample"
         (e.g., the mean prediction). Defaults to ``False``.
        :param next_state_sampling_size: Nb of sample draw if deterministic=False
        :param rng: an optional random number generator to use.
        :param epi_knn: epistemic k-nearest sampling

        :return: predicted observation and model state dictionary.
        """
        ensemble_size = self.num_members

        ensemble_means, ensemble_logvars = self._forward_propagation(
            model_input, model_state, rng
        )

        # RLRP-761 P1.5 — third sentinel site (see ``ExponentialFamilyMLP.sample_1d``).
        # The fill is a "no variance" MARKER, not a statistic: flag it so the P1.1
        # variance transport skips it instead of manufacturing a per-dimension
        # pseudo-variance out of a constant (risk ``Q-A``).
        logvars_are_absent = ensemble_logvars is None
        if ensemble_logvars is None:
            # Quick-hack for casse model has an ensemble of one or is deterministic
            ensemble_logvars = torch.full_like(ensemble_means, self._LOGVAR_MIN_LIMIT)

        model_state["ensemble_means"] = ensemble_means
        model_state["ensemble_logvars"] = ensemble_logvars
        model_state["ensemble_logvars_is_absent"] = logvars_are_absent

        if ensemble_size == 1:
            # Case non-model-ensemble NN
            means = ensemble_means.squeeze(dim=0)
            logvars = ensemble_logvars.squeeze(dim=0)
        elif self.propagation_method == "expectation":
            # Finalize expectation computation step from `_forward_propagation()`
            means = ensemble_means.mean(dim=0)
            logvars = ensemble_logvars.mean(dim=0)
        elif self.propagation_method == "post_sampling_expectation":
            means = ensemble_means
            logvars = ensemble_logvars
        elif epi_knn:

            assert ensemble_logvars.ndim == 1
            logvars_T = ensemble_logvars.reshape(ensemble_logvars.size(0), -1)
            distance_matrix = torch.cdist(logvars_T, logvars_T)

            # Example:
            #   >>> aa = torch.arange(1,6, dtype=float).reshape(5, -1)
            #   tensor([[1.],
            #           [2.],
            #           [3.],
            #           [4.],
            #           [5.]], dtype=torch.float64)
            #   >>> dist_matrix = torch.cdist(aa,aa)
            #   >>> dist_matrix
            #   tensor([[0., 1., 2., 3., 4.],
            #           [1., 0., 1., 2., 3.],
            #           [2., 1., 0., 1., 2.],
            #           [3., 2., 1., 0., 1.],
            #           [4., 3., 2., 1., 0.]], dtype=torch.float64)
            #   # Remove matrix indice row/col (margin)
            #   >>> dist_matrix[1:,1:]
            #   tensor([[0., 1., 2., 3.],
            #           [1., 0., 1., 2.],
            #           [2., 1., 0., 1.],
            #           [3., 2., 1., 0.]], dtype=torch.float64)
            #   >>> tril_indices = torch.tril_indices(4,4,-1)
            #   >>> tril_indices
            #   tensor([[1, 2, 2, 3, 3, 3],
            #           [0, 0, 1, 0, 1, 2]])
            #   >>> dist_matrix[tril_indices[0], tril_indices[1]]
            #   tensor([1., 2., 1., 3., 2., 1.], dtype=torch.float64)

            raise NotImplementedError("ToDo: epi k-nearest ")

        else:
            means = ensemble_means
            logvars = ensemble_logvars

        if deterministic or self.deterministic:
            # (CRITICAL) ToDo: validate averaging over ensemble dim for deterministic sampling
            if means.ndim >= 2 and 1 < ensemble_size == means.shape[0]:
                next_obs = torch.mean(means, dim=0)
            else:
                next_obs = means.squeeze(dim=0)
        else:
            variances = logvars.exp()
            stds = torch.sqrt(variances)

            # Note: nan handling is usefull when using this method early in the trainning stage
            stds = torch.nan_to_num(stds)

            # Sample each ensemble models independantly. Output next obs with shape E X OutDim.
            if next_state_sampling_size > 1:
                # Expand means and stds to (N_SAMPLES, ENSEMBLE, BATCH, DIM)
                # This allows torch.randn to broadcast across the new dimension efficiently
                expanded_means = means.unsqueeze(0).expand(
                    next_state_sampling_size, *means.shape
                )
                expanded_stds = stds.unsqueeze(0).expand(
                    next_state_sampling_size, *stds.shape
                )

                next_obs = expanded_means + expanded_stds * torch.randn_like(
                    expanded_means
                )
                # next_obs = torch.mean(next_obs, dim=0)
                # (CRITICAL) ToDo: validate
                next_obs = fast_distribution_mode_approximation(next_obs, ensemble_size)
            else:
                next_obs = means + stds * torch.randn_like(means)

            if ensemble_size > 1:
                if self.propagation_method == "post_sampling_expectation":
                    next_obs = torch.mean(next_obs, dim=0)
            else:
                next_obs = next_obs.squeeze(dim=0)

        # .... Memory management ..................................................................
        del ensemble_means, ensemble_logvars, means
        if not deterministic:
            del stds, variances

        return next_obs, model_state

    def _forward_propagation(
        self,
        model_input: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.propagation_method in ["expectation", "post_sampling_expectation"]:
            # Reproduce the expectation propagation logic but without the applying the mean
            # at the last step i.e., output E x B x Od instead of B x Od

            assert model_input.ndim == 2
            model_len = (
                len(self.elite_models) if self.elite_models is not None else len(self)
            )
            if model_input.shape[0] % model_len != 0:
                raise ValueError(
                    f"GaussianMLP ensemble requires batch size to be a multiple of the "
                    f"number of models. Current batch size is {model_input.shape[0]} for "
                    f"{model_len} models."
                )

            model_input = model_input.unsqueeze(0)
            ensemble_means, ensemble_logvars = self._default_forward(
                model_input, only_elite=True
            )

            # # Merge batch dim
            # ensemble_means = ensemble_means.mean(dim=1, keepdim=True)
            # ensemble_logvars = ensemble_logvars.mean(dim=1, keepdim=True)

            # # Compute expectation
            # means = ensemble_means.mean(dim=0)
            # logvars = ensemble_logvars.mean(dim=0)

        elif self.propagation_method is None:

            # # Note: Equivalent to running '_default_forward()' method with 'only_elite=True'
            self._maybe_toggle_layers_use_only_elite(only_elite=True)
            ensemble_means, ensemble_logvars = self.forward(
                model_input, use_propagation=False
            )
            self._maybe_toggle_layers_use_only_elite(only_elite=True)

        else:

            ensemble_means, ensemble_logvars = self.forward(
                model_input,
                use_propagation=True,
                propagation_indices=model_state["propagation_indices"],
            )

            # # Merge batch dim
            # ensemble_means = ensemble_means.mean(dim=1, keepdim=True)
            # ensemble_logvars = ensemble_logvars.mean(dim=1, keepdim=True)

        return ensemble_means, ensemble_logvars
