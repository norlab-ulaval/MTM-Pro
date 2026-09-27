# coding=utf-8

import warnings
from typing import List, Optional, Union
import numpy as np

import torch
from mbrl.models.util import to_tensor
from mbrl.types import TransitionBatch
from mbrl.util import ReplayBuffer
from omegaconf import omegaconf
from tqdm import tqdm

from tools.r2s_motion_model_container_tools.container import R2SMotionModelContainer
from algorithm.policy.replay_buffer_exploration_policy.base_policy import (
    BaseReplayBufferExplorationPolicy,
)
from tools.console_tools.progressbar_tools import init_progressbar
from tools.math_tools.ease_in_ease_out import easeInOutSine
from tools.mbrl_lib_tools.models.gaussian_mlp_extended import GaussianMLPExtended
from tools.mbrl_lib_tools.models.one_dim_tr_model_v2 import OneDTransitionRewardModelV2, \
    assert_is_OneDTransitionRewardModelV2
from tools.multistep_tools.models import AutoRegressiveSequenceIterator, WeightedMultiStepMLP


class PartitionBasedReplayBufferExplorationPolicy(BaseReplayBufferExplorationPolicy):
    _dynamic_model: OneDTransitionRewardModelV2
    _pred_means: Union[np.ndarray, torch.Tensor]
    _pred_stds: Union[np.ndarray, torch.Tensor]
    _pred_epi: Union[np.ndarray, torch.Tensor]
    _samples_score: np.ndarray
    _window_len_samples_score: np.ndarray
    _uncertainty_sorted_interval_start_idx: np.ndarray
    _progressbar: Optional[tqdm]
    _current_sample_uncertainty_ratio: int = 0.0
    _ramp_in_count = 0
    _ramp_out_count = 0

    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        obs_source_replay_buffer: ReplayBuffer,
        dynamic_model: Union[OneDTransitionRewardModelV2, R2SMotionModelContainer],
        torch_rng: torch.Generator,
        policy_cfg_key: str = "UDER.uder_exploration_policy",
        simulator_cfg_key: str = "environment",
    ):
        """Uncertainty driven replay buffer exploration policy (UDRBE)

        :param cfg: hydra configuration
        :param obs_source_replay_buffer: the observed source replay buffer
        :param dynamic_model: A trained system dynamic model
        :param policy_cfg_key: policy related key to fetch in the cfg object
        :param simulator_cfg_key: simulator related key to fetch in the cfg object
        """
        if isinstance(dynamic_model, R2SMotionModelContainer):
            motion_model_container = dynamic_model
            self._dynamic_model = motion_model_container.dynamics_model
        else:
            self._dynamic_model = dynamic_model

        self._torch_rng = torch_rng
        super().__init__(cfg, obs_source_replay_buffer, policy_cfg_key, simulator_cfg_key)

        # .... Pre-condition ......................................................................
        assert_is_OneDTransitionRewardModelV2(self._dynamic_model)
        ensemble_size = self._dynamic_model.model.num_members
        assert ensemble_size > 1, "Require dynamic_model with ensemble size > 1!"

    # noinspection PyMethodOverriding
    def update_policy(
        self,
        obs_source_replay_buffer: ReplayBuffer,
        dynamic_model: Union[OneDTransitionRewardModelV2, R2SMotionModelContainer],
    ):
        # .... Set sample uncertainty .............................................................
        assert 1 >= self.policy_cfg.sample_uncertainty.in_ratio >= 0
        assert 1 >= self.policy_cfg.sample_uncertainty.out_ratio >= 0
        assert self.policy_cfg.sample_uncertainty.in_ramp >= 0
        assert self.policy_cfg.sample_uncertainty.out_ramp >= 0
        assert self.policy_cfg.sample_uncertainty.in_epoch >= 1
        assert (
            self.policy_cfg.sample_uncertainty.in_epoch
            + self.policy_cfg.sample_uncertainty.in_ramp
            <= self.policy_cfg.sample_uncertainty.out_epoch
        ), (
            f"{self.policy_cfg.sample_uncertainty.in_epoch=} + "
            f"{self.policy_cfg.sample_uncertainty.in_ramp=} "
            f"!<= {self.policy_cfg.sample_uncertainty.out_epoch=}"
        )

        # .... Update policy replay buffer and model ..............................................
        # (CRITICAL) ToDo: implement test case
        if isinstance(dynamic_model, R2SMotionModelContainer):
            self._dynamic_model = dynamic_model.dynamics_model
        else:
            self._dynamic_model = dynamic_model

        return super().update_policy(obs_source_replay_buffer)

    @property
    def _sim_cfg_keys(self) -> Optional[List[str]]:
        return None

    @property
    def _policy_cfg_keys(self) -> List[str]:
        uncertainty_driven_policy_cfg_keys = [
            "sample_uncertainty.initial_ratio",
            "sample_uncertainty.in_epoch",
            "sample_uncertainty.in_ramp",
            "sample_uncertainty.in_ratio",
            "sample_uncertainty.out_epoch",
            "sample_uncertainty.out_ramp",
            "sample_uncertainty.out_ratio",
            "multistep_prediction_compression",
            "sample_score_evaluation_method",
        ]
        return super()._policy_cfg_keys + uncertainty_driven_policy_cfg_keys

    def _init_policy(self) -> None:
        self._set_sample_uncertainty_ratio()

        self._compute_samples_statistics()

        if isinstance(self._dynamic_model.model, WeightedMultiStepMLP):
            if self.policy_cfg.sample_score_evaluation_method == "NLPD":
                window_len_samples_score = (
                    self._evaluate_sample_score_base_on_negative_log_probability_density()
                )
            elif self.policy_cfg.sample_score_evaluation_method == "MaxEpi":
                window_len_samples_score = (
                    self._evaluate_sample_score_base_on_feature_max_epi_uncertainty()
                )
            elif self.policy_cfg.sample_score_evaluation_method == "both":
                nlpd_window_len_samples_score = (
                    self._evaluate_sample_score_base_on_negative_log_probability_density()
                )
                maxepi_window_len_samples_score = (
                    self._evaluate_sample_score_base_on_feature_max_epi_uncertainty()
                )
                window_len_samples_score = (
                    nlpd_window_len_samples_score + maxepi_window_len_samples_score
                )
                self._window_len_samples_score = window_len_samples_score
            else:
                raise NotImplementedError(
                    f"Sample score evaluation method "
                    f"'{self.policy_cfg.sample_score_evaluation_method}' not implemented!"
                )
        elif isinstance(self._dynamic_model.model, GaussianMLPExtended):
            window_len_samples_score = (
                self._evaluate_sample_score_base_on_feature_max_epi_uncertainty()
            )
        else:
            raise NotImplementedError(
                f"Model type {self._dynamic_model.model.__class__.__name__} " f"not suported"
            )

        self._uncertainty_sorted_interval_start_idx = (
            self._sort_sample_indexes_by_increasing_score(window_len_samples_score)
        )

        def max_scored_predicted_samples() -> int:
            """Selects a trajectory start and end index base on epistemic uncertainty and
            updates visited states. Garantee to return any `trajectory_start_idx` only once.
            """
            available_sorted_sample_score_idx = self._uncertainty_sorted_interval_start_idx[
                np.isin(self._uncertainty_sorted_interval_start_idx, self._available_trj_start_idx)
            ]

            random_idx = int(self._rng.choice(self._available_trj_start_idx))

            max_scored_idx = int(available_sorted_sample_score_idx[-1])

            sampled_start_idx = self._rng.choice(
                [max_scored_idx, random_idx],
                p=[
                    self._current_sample_uncertainty_ratio,
                    1.0 - self._current_sample_uncertainty_ratio,
                ],
            )
            return sampled_start_idx

        self.policy = max_scored_predicted_samples
        return None

    def _set_sample_uncertainty_ratio(self) -> None:
        uder_epoch = self._current_uder_epoch
        if uder_epoch == 1:
            self._current_sample_uncertainty_ratio = (
                self.policy_cfg.sample_uncertainty.initial_ratio
            )

        in_epoch = self.policy_cfg.sample_uncertainty.in_epoch
        in_ramp = self.policy_cfg.sample_uncertainty.in_ramp
        in_ratio = self.policy_cfg.sample_uncertainty.in_ratio
        out_epoch = self.policy_cfg.sample_uncertainty.out_epoch
        out_ramp = self.policy_cfg.sample_uncertainty.out_ramp
        out_ratio = self.policy_cfg.sample_uncertainty.out_ratio

        assert in_ratio >= out_ratio

        if (
            in_epoch <= uder_epoch < in_epoch + in_ramp
            and self._current_sample_uncertainty_ratio < 1.0
        ):
            self._ramp_in_count += 1
            self._current_sample_uncertainty_ratio = in_ratio * easeInOutSine(
                self._ramp_in_count * 0.1 * 10 / in_ramp
            )
        elif in_ramp == 0 or in_epoch + in_ramp <= uder_epoch < out_epoch:
            self._current_sample_uncertainty_ratio = in_ratio
            self._ramp_out_count = 0
        elif (
            out_epoch <= uder_epoch < out_epoch + out_ramp
            and self._current_sample_uncertainty_ratio > 0.0
        ):
            self._ramp_out_count += 1
            self._current_sample_uncertainty_ratio = in_ratio - abs(
                (in_ratio - out_ratio) * easeInOutSine(self._ramp_out_count * 0.1 * 10 / out_ramp)
            )
        elif uder_epoch >= out_epoch + out_ramp:
            self._current_sample_uncertainty_ratio = out_ratio
            self._ramp_in_count = 0
            self._ramp_out_count = 0

        return None

    @property
    def current_sample_uncertainty_ratio(self) -> float:
        return self._current_sample_uncertainty_ratio

    def _compute_samples_statistics(self) -> None:
        all_samples: TransitionBatch
        all_samples = self._source_replay_buffer.get_all(shuffle=False)
        prediction_shape = (len(all_samples), self._dynamic_model.model.out_size)

        self._init_progressbar(
            pb_len=1,
            description="(UDRBE) Deploy model and compute prediction statistic on full " "dataset",
        )

        all_samples_obs = all_samples.obs
        all_samples_act = all_samples.act

        if isinstance(self._dynamic_model.model, AutoRegressiveSequenceIterator):
            self._dynamic_model.model.wipe_ar_memory()

            if self.multistep_model_expect_sequence_transition_but_miss_batch_dim(all_samples_obs):
                if isinstance(all_samples_obs, np.ndarray):
                    warnings.warn(
                        "[RLRC torch-first] 'all_samples_obs/act' are numpy ndarrays in "
                        "_compute_samples_statistics. With output_torch=True (default), "
                        "ReplayBuffer.get_all() should return torch.Tensor. "
                        "Update the upstream caller to pass torch tensors directly.",
                        DeprecationWarning,
                        stacklevel=1,
                    )
                    all_samples_obs = to_tensor(all_samples_obs)
                    all_samples_act = to_tensor(all_samples_act)

                # Note: add the ensemble and batch dim => Sequence x Dim -> E x B x Sequence x Dim
                # all_samples_obs = all_samples_obs.unsqueeze(0).repeat(
                # self._dynamic_model.model.num_members, 1,1,1)
                # all_samples_act = all_samples_act.unsqueeze(0).repeat(
                # self._dynamic_model.model.num_members, 1,1,1)

                # Note: add the batch dim => Sequence x Dim -> B x Sequence x Dim
                all_samples_obs = all_samples_obs.unsqueeze(0)
                all_samples_act = all_samples_act.unsqueeze(0)

        (
            self._pred_means,
            self._pred_stds,
            self._pred_epi,
        ) = self._dynamic_model.compute_prediction_and_stats(
            obs=all_samples_obs,
            act=all_samples_act,
            rng=self._torch_rng,
            compute_probability_statistics_over_ensemble=True,
        )

        self._progressbar.update(1)

        # Torch-first: ensure predictions are always tensors
        if isinstance(self._pred_means, np.ndarray):
            self._pred_means = torch.from_numpy(self._pred_means)
        if isinstance(self._pred_stds, np.ndarray):
            self._pred_stds = torch.from_numpy(self._pred_stds)
        if isinstance(self._pred_epi, np.ndarray):
            self._pred_epi = torch.from_numpy(self._pred_epi)
        assert (
            self._pred_means.shape == prediction_shape
        ), f"{self._pred_means.shape} != {prediction_shape}"
        assert (
            self._pred_stds.shape == prediction_shape
        ), f"{self._pred_stds.shape} != {prediction_shape}"
        assert (
            self._pred_epi.shape == prediction_shape
        ), f"{self._pred_epi.shape} != {prediction_shape}"

        self._progressbar.close()
        return None

    def multistep_model_expect_sequence_transition_but_miss_batch_dim(
        self, all_samples: torch.Tensor
    ):
        return self._dynamic_model.model.receive_sequence_batch and all_samples.ndim == 2

    def _compute_prediction_square_error(self, pred_means: torch.Tensor) -> torch.Tensor:
        # (CRITICAL) ToDo: implement test case (its indirectly tested for now)
        # Torch-first: ensure pred_means is a tensor
        if isinstance(pred_means, np.ndarray):
            pred_means = torch.from_numpy(pred_means)
        all_samples: TransitionBatch
        all_samples = self._source_replay_buffer.get_all(shuffle=False)
        prediction_shape = (len(all_samples), self._dynamic_model.model.out_size)

        # Vectorized: compute squared error in a single torch operation
        all_next_obs = all_samples.next_obs  # Already a tensor with output_torch=True
        # Replay-buffer tensors come out CPU-resident; align to `pred_means` device
        # (which follows the dynamic model). No-op when both are already on the same
        # device (CPU/M3 path); avoids CUDA↔CPU mismatch on Orin/V100.
        if all_next_obs.device != pred_means.device:
            all_next_obs = all_next_obs.to(pred_means.device)
        pred_square_error = torch.square(all_next_obs - pred_means)

        assert pred_square_error.shape == prediction_shape

        return pred_square_error

    def _evaluate_sample_score_base_on_feature_max_epi_uncertainty(self) -> np.ndarray:
        # ToDo: implement test case. Function unit are tested separatly only for the moment.

        self._init_progressbar(
            # pb_len=4,
            pb_len=3,
            description="(UDRBE) Compute dataset sample score base on feature max epi "
            "uncertainty",
        )

        ss_pred_epi = self._convert_multistep_array_to_singlestep(
            self._pred_epi, self.policy_cfg.multistep_prediction_compression
        )
        self._progressbar.update(1)

        self._samples_score = self._compute_sample_score_over_feature_dim(
            ss_pred_epi, reduction="sum"
        )
        self._progressbar.update(1)

        self._window_len_samples_score = self._compute_window_len_sample_score(self._samples_score)
        self._progressbar.update(1)

        self._progressbar.close()
        return self._window_len_samples_score

    def _evaluate_sample_score_base_on_negative_log_probability_density(
        self, eps: float = 1e-08
    ) -> np.ndarray:
        """Evaluate feature score using Negative Log-Probaility Density

        Negative log-probaility density penalise overconfidence, i.e for two equal error,
        the one with the less std (more confident) will have a higher score then the one with
        the higher std (less confident).

        Ref paper: "Uncertainty Estimation and Reduction of Pre-trained Models for Text Regression"
        section "5 Uncertainty Evaluation Metrics".

        :param eps: small constant added to epi variance for numerical stability
        :return: window length samples score
        """

        assert isinstance(self._dynamic_model.model, WeightedMultiStepMLP)
        assert 1 >= eps > 0

        pred_square_error = self._compute_prediction_square_error(self._pred_means)

        self._init_progressbar(
            pb_len=4,
            description="(UDRBE) Compute dataset sample score base on feature negative log "
            "probability density",
        )

        # .... Setup data .........................................................................
        # Torch-first: use torch operations for NLPD computation
        pred_epi = self._pred_epi
        if isinstance(pred_epi, np.ndarray):
            pred_epi = torch.from_numpy(pred_epi)
        feature_epi_var = torch.square(pred_epi)  # raw pred epi is a standard deviation

        feature_epi_var = self._dynamic_model.model.unflaten_multistep_composed_array(
            feature_epi_var
        )
        pred_square_error = self._dynamic_model.model.unflaten_multistep_composed_array(
            pred_square_error
        )

        self._progressbar.update(1)

        # .... Compute NLPD .......................................................................
        # Note:
        # - This is the NLPD formula for gaussian predictive distribution
        # - adding a small constant (epsilon) to the variance is required to handle the case
        #   where variance=0 such that log(var=zero) and division by zero does not produce error.

        # (NICE TO HAVE) ToDo: maybe refactor using torch.nn.GaussianNLLLoss
        feature_epi_var += eps
        self._samples_score = (1 / (2 * feature_epi_var.shape[-1])) * torch.sum(
            torch.log(feature_epi_var) + (pred_square_error / feature_epi_var),
            dim=-1,
        )

        batch_len = pred_square_error.shape[0]
        if self._dynamic_model.model.horizon_len > 1:
            composed_obs_len = (
                self._dynamic_model.model.singlestep_obs_len
                + self._dynamic_model.model.singlestep_act_len
            )
        else:
            composed_obs_len = self._dynamic_model.model.singlestep_obs_len

        expected_shape = (
            batch_len,
            composed_obs_len,
        )
        assert self._samples_score.shape == expected_shape, (
            f"{self._samples_score.shape=} != " f"{expected_shape=}"
        )

        self._progressbar.update(1)

        # .... Compute scoring indexes ............................................................
        self._samples_score = self._compute_sample_score_over_feature_dim(
            self._samples_score, reduction="sum"
        )
        self._progressbar.update(1)

        expected_shape = (batch_len,)
        assert self._samples_score.shape == expected_shape, (
            f"{self._samples_score.shape=} != " f"{expected_shape=}"
        )

        self._window_len_samples_score = self._compute_window_len_sample_score(self._samples_score)
        self._progressbar.update(1)

        self._progressbar.close()
        return self._window_len_samples_score

    def _convert_multistep_array_to_singlestep(
        self, x: Union[torch.Tensor, np.ndarray], multistep_compression: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(self._dynamic_model.model, WeightedMultiStepMLP):
            model = self._dynamic_model.model
            if not multistep_compression:
                singlestep_x = model.multistep_to_singlestep_next_obs_adapter(x)
            else:
                singlestep_x = model.reduce_multistep_prediction_to_single_step(
                    x, reduction="mean", apply_discount_factor=True
                )
            return singlestep_x
        else:
            return x

    def _compute_sample_score_over_feature_dim(
        self, x: Union[torch.Tensor, np.ndarray], reduction: str = "mean"
    ) -> Union[torch.Tensor, np.ndarray]:
        assert x.ndim == 2
        # Torch-first: use torch ops when input is a tensor
        if isinstance(x, torch.Tensor):
            if reduction == "mean":
                return torch.mean(x, dim=-1)
            elif reduction == "max":
                return torch.max(x, dim=-1).values
            elif reduction == "sum":
                return torch.sum(x, dim=-1)
            else:
                raise NotImplementedError(
                    f"Reduction method {reduction=} is not implemented. "
                    f"Choose either 'sum', 'mean' or 'max'"
                )
        else:
            if reduction == "mean":
                return np.mean(x, axis=-1)
            elif reduction == "max":
                return np.max(x, axis=-1)
            elif reduction == "sum":
                return np.sum(x, axis=-1)
            else:
                raise NotImplementedError(
                    f"Reduction method {reduction=} is not implemented. "
                    f"Choose either 'sum', 'mean' or 'max'"
                )

    def _compute_window_len_sample_score(
        self, samples_score: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        assert samples_score.ndim == 1
        # Torch-first: convert to numpy at this boundary for downstream np.argsort/np.isin
        if isinstance(samples_score, torch.Tensor):
            samples_score = samples_score.detach().cpu().numpy()

        window_len = self.policy_cfg.scan_window_len
        score_len = len(samples_score)
        trj_start_indices = self._dataset_trj_start_idx

        window_len_samples_score = np.empty(len(trj_start_indices), dtype=samples_score.dtype)
        for i, each_idx in enumerate(trj_start_indices):
            window_end_idx = each_idx + window_len

            # (NICE TO HAVE) ToDo: improve handling the remaining sample score at the array end
            if window_end_idx >= score_len:
                window_end_idx = score_len

            window = samples_score[each_idx:window_end_idx]

            if len(window) < window_len:
                # Handle the end of the sample score array by adding average value
                pad = window_len - len(window)
                window = np.pad(window, (0, pad), "mean")
                assert len(window) == window_len

            window_len_samples_score[i] = np.mean(window)

        assert len(window_len_samples_score) == len(trj_start_indices)

        return window_len_samples_score

    def _sort_sample_indexes_by_increasing_score(self, samples_score: np.ndarray) -> np.ndarray:
        """Acc order idx by max std so that the right most be the most prediction uncertain.
        :param samples_score:
        """
        assert samples_score.ndim == 1
        uncertainty_sorted_interval_start_idx = self._dataset_trj_start_idx[
            np.argsort(samples_score.squeeze())
        ]
        return uncertainty_sorted_interval_start_idx

    def _init_progressbar(self, pb_len: int, description: str) -> None:
        self._progressbar = init_progressbar(pb_len, description)
        return None

    @property
    def prediction_epistemic_uncertainty(self) -> torch.Tensor:
        return self._pred_epi
