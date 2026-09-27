# coding=utf-8
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import hydra
import mbrl.models
import numpy as np
import omegaconf
import torch

from algorithm.experience_replay_learning_loop.core.tensorboard_utils import (
    tensorboard_prediction_error,
)
from mbrl import models as models

from mbrl.util import ReplayBuffer

from tools.console_tools.message import consol_msg_universal_one_liner
from tools.mbrl_lib_tools.models.utils import is_model_ensemble
from tools.r2s_motion_model_container_tools.container import R2SMotionModelContainer
from algorithm.utils import seed_me
from tools.f110_gym_env_tools.plot_tensorboard import OnlineTensorboardWritter
from algorithm.experience_replay_learning_loop.core.model_testing_utils import (
    model_testtime_rollout_and_compute_prediction_metric,
)
from algorithm.experience_replay_learning_loop.core.data_classes import PredictionMetric
from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist
from tools.multistep_tools.models.ms2ms2ss_ar_temporal_mixture_pme_utils import (
    temporal_mixture_head_leaf_parameter_names,
)
from trajectory_container_tools.dataclasses.erll_trajectory_dataclass import (
    TestMotionTrajectoryDataclass,
    TestTrajectoryDataclass,
)


# ToDo: RLRP-290 feat: refactor training_callback module member as class


def setup_trainer_epoch_callback_aggregator(
    cfg: omegaconf.DictConfig,
    cfg_training: omegaconf.DictConfig,
    model_trainer: models.ModelTrainer,
    tensorboard_writer: Optional[OnlineTensorboardWritter],
    register_callbacks: List[
        Callable[
            [
                omegaconf.DictConfig,
                omegaconf.DictConfig,
                models.ModelTrainer,
                Optional[OnlineTensorboardWritter],
            ],
            Callable,
        ]
    ],
    enable_tensorboard_call_method=True,
) -> Callable:
    """
    Sets up a train or replay buffer optimization epoch callback function to be used by either
    the ModelTrainer object or the uncertainty_driven_experience_replay function at each epoch.

    Callback functions passed to `register_callbacks` must respect the following signature:

    >>> def setup_cool_epoch_callback(cfg: omegaconf.DictConfig,
    >>>         cfg_training: omegaconf.DictConfig,
    >>>         model_trainer: models.ModelTrainer,
    >>>         tensorboard_writer: Optional[OnlineTensorboardWritter],
    >>>     ) -> Callable:
    >>>
    >>>     # Add initialisation logic (will be executed once)
    >>>
    >>>     def cool_epoch_callback(
    >>>         model: mbrl.models.Model,
    >>>         train_iteration: int,
    >>>         epoch: int,
    >>>         total_avg_loss: float,
    >>>         eval_score: float,
    >>>         best_val_score: float,
    >>>     ) -> None:
    >>>         if tensorboard_writer is not None:
    >>>             # Add logic to be executed at each epoch
    >>>             pass
    >>>         return None
    >>>
    >>>     return cool_epoch_callback

    :param cfg: Configuration dictionary containing general settings.
    :param cfg_training: Configuration dictionary specific to the model training cfg.
    :param model_trainer: Instance of ModelTrainer responsible for training the model.
    :param tensorboard_writer: Optional tensorboard writer for logging training metrics.
    :param register_callbacks: List of callable functions to register as callbacks.
    :param enable_tensorboard_call_method: execute tensorboard_writer(...) method.
    :return: Callable function to be used as an epoch callback.
    """
    debug_mode = cfg.get("debug_mode", False)

    initialized_callbacks = []
    for callback_init in register_callbacks:
        callback = callback_init(cfg, cfg_training, model_trainer, tensorboard_writer)
        initialized_callbacks.append(callback)

    def epoch_callback(
        model: mbrl.models.Model,
        train_iteration: int,
        epoch: int,
        total_avg_loss: float,
        eval_score: float,
        best_val_score: float,
    ) -> None:
        """Called by `mbrl.models.ModelTrainer.train(...)` method at the end of each epoch."""
        if tensorboard_writer is not None and enable_tensorboard_call_method:
            tensorboard_writer(
                model,
                train_iteration,
                epoch,
                total_avg_loss,
                eval_score,
                best_val_score,
            )

        for callback_execute in initialized_callbacks:
            callback_execute(
                model,
                train_iteration,
                epoch,
                total_avg_loss,
                eval_score,
                best_val_score,
            )

        return None

    return epoch_callback


def setup_batch_callback_aggregator(
    cfg: omegaconf.DictConfig,
    cfg_training: omegaconf.DictConfig,
    model_trainer: models.ModelTrainer,
    tensorboard_writer: Optional[OnlineTensorboardWritter],
    register_callbacks: List[
        Callable[
            [
                omegaconf.DictConfig,
                omegaconf.DictConfig,
                models.ModelTrainer,
                Optional[OnlineTensorboardWritter],
            ],
            Callable,
        ]
    ],
) -> Callable:
    """
    Sets up a train batch callback function to be used by the ModelTrainer object.

    From ModelTrainer doc:
    > this function will be called for every batch with the output of ``model.update()`` (during
    > training), and ``model.eval_score()`` (during evaluation). It will be called with four
    > arguments ``(epoch_index, loss/score, meta, mode)``, where ``mode`` is one of ``"train"``
    > or ``"eval"``, indicating if the callback was called during training or evaluation.

    Callback functions passed to `register_callbacks` must respect the following signature:

    >>> def setup_cool_batch_callback(cfg: omegaconf.DictConfig,
    >>>         cfg_training: omegaconf.DictConfig,
    >>>         model_trainer: models.ModelTrainer,
    >>>         tensorboard_writer: Optional[OnlineTensorboardWritter],
    >>>     ) -> Callable:
    >>>
    >>>     # Add initialisation logic (will be executed once)
    >>>
    >>>     def cool_batch_callback(
    >>>         epoch_index: int,
    >>>         loss: float,
    >>>         meta: Dict,
    >>>         mode: str
    >>>     ) -> None:
    >>>         if tensorboard_writer is not None:
    >>>             # Add logic to be executed at each epoch
    >>>             pass
    >>>         return None
    >>>
    >>>     return cool_batch_callback

    :param cfg: Configuration dictionary containing general settings.
    :param cfg_training: Configuration dictionary specific to the model training cfg.
    :param model_trainer: Instance of ModelTrainer responsible for training the model.
    :param tensorboard_writer: Optional tensorboard writer for logging training metrics.
    :param register_callbacks: List of callable functions to register as callbacks.
    :param enable_tensorboard_call_method: execute tensorboard_writer(...) method.
    :return: Callable function to be used as an epoch callback.
    """
    debug_mode = cfg.get("debug_mode", False)

    initialized_callbacks = []
    for callback_init in register_callbacks:
        callback = callback_init(cfg, cfg_training, model_trainer, tensorboard_writer)
        initialized_callbacks.append(callback)

    def batch_callback(epoch_index: int, loss: float, meta: Dict, mode: str) -> None:
        """Called by `mbrl.models.ModelTrainer.train(...)` method at each update/eval_score
        call."""

        for callback_execute in initialized_callbacks:
            callback_execute(epoch_index, loss, meta, mode)

        return None

    return batch_callback


def setup_erll_epoch_pre_training_callback_aggregator(
    cfg: omegaconf.DictConfig,
    cfg_training: omegaconf.DictConfig,
    model_trainer: models.ModelTrainer,
    tensorboard_writer: Optional[OnlineTensorboardWritter],
    register_callbacks: List[
        Callable[
            [
                omegaconf.DictConfig,
                omegaconf.DictConfig,
                models.ModelTrainer,
                Optional[OnlineTensorboardWritter],
            ],
            Callable,
        ]
    ],
) -> Callable:
    """
    Sets up a train or replay buffer optimization pre-training epoch callback function to be used
     by uncertainty_driven_experience_replay function.

    Callback functions passed to `register_callbacks` must respect the following signature:

    >>> def setup_cool_pre_training_callback(cfg: omegaconf.DictConfig,
    >>>         cfg_training: omegaconf.DictConfig,
    >>>         model_trainer: models.ModelTrainer,
    >>>         tensorboard_writer: Optional[OnlineTensorboardWritter],
    >>>     ) -> Callable:
    >>>
    >>>     # Add initialisation logic (will be executed once)
    >>>
    >>>     def cool_pre_training_callback(
    >>>         model: mbrl.models.Model,
    >>>         uder_epoch: int,
    >>>         global_epoch: int,
    >>>         replay_buffer: ReplayBuffer,
    >>>     ) -> None:
    >>>         if tensorboard_writer is not None:
    >>>             # Add logic to be executed at each epoch
    >>>             pass
    >>>         return None
    >>>
    >>>     return cool_pre_training_callback

    :param cfg: Configuration dictionary containing general settings.
    :param cfg_training: Configuration dictionary specific to the model training cfg.
    :param model_trainer: Instance of ModelTrainer responsible for training the model.
    :param tensorboard_writer: Optional tensorboard writer for logging training metrics.
    :param register_callbacks: List of callable functions to register as callbacks.
    :return: Callable function to be used as an epoch callback.
    """
    debug_mode = cfg.get("debug_mode", False)

    initialized_callbacks = []
    for callback_init in register_callbacks:
        callback = callback_init(cfg, cfg_training, model_trainer, tensorboard_writer)
        initialized_callbacks.append(callback)

    def erll_epoch_pre_training_callback(
        model: mbrl.models.Model,
        uder_epoch: int,
        global_epoch: int,
        replay_buffer: ReplayBuffer,
    ) -> None:
        """Called by `uncertainty_driven_experience_replay` before model training."""
        for callback_execute in initialized_callbacks:
            callback_execute(model, uder_epoch, global_epoch, replay_buffer)
        return None

    return erll_epoch_pre_training_callback


# =================================================================================================


@dataclass()
class EpochStepRecorder:
    epoch_count: Optional[int] = None
    skip_init_epoch: bool = True

    def step_and_is_new_epoch(self, epoch: int) -> bool:

        if self.epoch_count is None or self.epoch_count != epoch:
            self.epoch_count = epoch

            if self.skip_init_epoch and self.epoch_count == 0:
                return False

            return True
        else:
            return False


@dataclass()
class BatchStepRecorder:
    """
    Handles the recording of batch steps and flags when a specified step interval is reached.

    Note:
        Need to execute the ``step`` method before the ``is_n_step`` method since ``step_count=0``
        on instanciation.

    :ivar step_count: The current step count being tracked. Automatically initialized to 0.
    :type step_count: int
    :ivar flag_every_n_step: The interval for flagging every n step. Defaults to 1. Must be positive.
    :type flag_every_n_step: int
    """

    step_count: int = field(default=0, init=False)
    flag_every_n_step: int = 1

    def __post_init__(self):
        if self.flag_every_n_step is not None and self.flag_every_n_step <= 0:
            raise ValueError(
                "flag_every_n_step must be positive, got: {}".format(
                    self.flag_every_n_step
                )
            )

    def step(self):
        self.step_count += 1
        return None

    def is_n_step(self) -> bool:
        if self.flag_every_n_step is not None:

            if self.flag_every_n_step == 0:
                return True

            if self.step_count % self.flag_every_n_step == 0:
                return True

        return False


@dataclass()
class EpochValuesRecorder:
    values: List[torch.Tensor] = field(default=None, init=False)
    epoch_count: Optional[int] = None
    skip_init_epoch: bool = True

    def __post_init__(self):
        self.values = []

    def step_and_is_new_epoch(self, epoch: int) -> bool:
        if self.epoch_count is None or self.epoch_count != epoch:
            self.epoch_count = epoch

            if self.skip_init_epoch and self.epoch_count == 0:
                return False

            return True
        else:
            return False

    def append(self, values: torch.Tensor) -> None:
        self.values.append(values)
        return None

    def total_average_values(self) -> float:
        if len(self.values) == 0:
            return 0.0

        total = 0.0
        count = 0
        for v in self.values:
            f = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            if math.isfinite(f):
                total += f
                count += 1

        if count == 0:
            return 0.0

        return total / count

    def flush(self) -> None:
        self.values.clear()
        return None


def setup_multistep_loss_batch_callback(
    cfg: omegaconf.DictConfig,
    cfg_training: omegaconf.DictConfig,
    model_trainer: models.ModelTrainer,
    tensorboard_writer: Optional[OnlineTensorboardWritter],
) -> Callable:
    epoch_step_recorder = EpochStepRecorder()
    batch_step_recorder = BatchStepRecorder(
        flag_every_n_step=cfg.pipeline.tensorboard.batch_callback_execute_every_n
    )

    ss_loss_epoch_agregator = EpochValuesRecorder()
    ho_loss_epoch_agregator = EpochValuesRecorder()
    ho_at_t1_loss_epoch_agregator = EpochValuesRecorder()
    nll_loss_epoch_agregator = EpochValuesRecorder()
    ms_mix_loss_epoch_agregator = EpochValuesRecorder()
    ms_kl_loss_epoch_agregator = EpochValuesRecorder()
    ms_siw_mp_loss_epoch_agregator = EpochValuesRecorder()
    ms_gms_iwae_loss_epoch_agregator = EpochValuesRecorder()
    ss_kl_loss_epoch_agregator = EpochValuesRecorder()
    encoder_loss_epoch_agregator = EpochValuesRecorder()
    decoder_loss_epoch_agregator = EpochValuesRecorder()
    pre_mixture_u_loss_epoch_agregator = EpochValuesRecorder()
    post_mixture_loss_epoch_agregator = EpochValuesRecorder()
    info_projection_loss_epoch_agregator = EpochValuesRecorder()
    rollout_consistency_loss_epoch_agregator = EpochValuesRecorder()
    # Deploy-history drift residual (DH) CP sub-term (RLRP-731): terminal-step raw-space
    # residual MSE scored inside the CP deploy unroll. Emitted by MTM-Pro as
    # ``meta["ms_deploy_history_drift_loss"]``; only present when
    # ``compounded_prediction_deploy_loss.enable_history_drift_loss`` is true (and CP active).
    dh_loss_epoch_agregator = EpochValuesRecorder()
    projection_vs_horizon_ratio_epoch_agregator = EpochValuesRecorder()
    ms_energy_beta_epoch_agregator = EpochValuesRecorder()
    # Term-balance regression monitor (RLRP-720): one lazily-created aggregator per composite term
    # share key ("<TERM>_composite_share"). See the "improve CompositeLossAutomaticWeighting"
    # .junie plan (refactor_composite_loss_auto_weighting_and_shifted_softplus_plan_20260623.md).
    composite_share_epoch_agregators: Dict[str, EpochValuesRecorder] = {}
    mppi_cost_epoch_agregator = EpochValuesRecorder()
    mppi_unnormalized_pred_ho_iw_epoch_agregator = EpochValuesRecorder()
    mppi_epsilon_epoch_agregator = EpochValuesRecorder()
    ss_loss_auto_weighting_agregator = EpochValuesRecorder()
    ms_loss_auto_weighting_agregator = EpochValuesRecorder()
    ss_kl_loss_auto_weighting_agregator = EpochValuesRecorder()
    ms_kl_loss_auto_weighting_agregator = EpochValuesRecorder()
    ms_siw_mp_loss_auto_weighting_agregator = EpochValuesRecorder()
    ms_RC_loss_auto_weighting_agregator = EpochValuesRecorder()
    ms_gms_iwae_loss_auto_weighting_agregator = EpochValuesRecorder()
    ms_U_loss_auto_weighting_agregator = EpochValuesRecorder()
    ms_MIX_loss_auto_weighting_agregator = EpochValuesRecorder()
    IPROJ_loss_auto_weighting_agregator = EpochValuesRecorder()
    ms_encoder_loss_auto_weighting_agregator = EpochValuesRecorder()
    ms_decoder_loss_auto_weighting_agregator = EpochValuesRecorder()
    cp_loss_auto_weighting_agregator = EpochValuesRecorder()
    dh_loss_auto_weighting_agregator = EpochValuesRecorder()
    # RLRP-751 (task T7): per-feature geometry AUTO-weighting cards. Emitted by
    # ``CompositeLossAutomaticWeighting.forward`` as
    # ``FEAT_GEOM_{SS,MS,CP}_loss_auto_weighting`` only when the geometry term is
    # routed through the composite auto-weighting (``feature_geometry.auto_weighting``
    # on, MTM-Pro family); absent -> neutral (STATIC weighting / other families).
    feature_geom_ss_loss_auto_weighting_agregator = EpochValuesRecorder()
    feature_geom_ms_loss_auto_weighting_agregator = EpochValuesRecorder()
    feature_geom_cp_loss_auto_weighting_agregator = EpochValuesRecorder()
    # Deploy-path training stabilization (RLRP-722, feature (1) forecaster warmup): per-epoch
    # diagnostics emitted by the MTM-Pro model in ``meta`` (``deploy_warmup_active`` ->
    # fraction of the epoch's batches still in the forecast-only warmup phase;
    # ``deploy_ramp`` -> the post-warmup linear deploy-weight ramp factor in [0, 1]). Only
    # present when ``deploy_path_training.warmup.enable`` is true.
    deploy_warmup_active_epoch_agregator = EpochValuesRecorder()
    deploy_ramp_epoch_agregator = EpochValuesRecorder()
    # Deploy-path training stabilization (RLRP-722, feature (3) EMA forecast clone): the current
    # EMA momentum ``d`` (constant or scheduled). Only present when
    # ``deploy_path_training.ema_forecast_clone.enable`` is true.
    ema_forecast_momentum_epoch_agregator = EpochValuesRecorder()
    # Per-environment feature-geometry penalty (RLRP-736): the quaternion geodesic
    # (orientation) auxiliary term. Emitted by ``ExponentialFamilyMLP.loss`` as the
    # scalar ``meta["feature_geom_loss"]`` only when the term is active (a feature
    # handler is registered and ``feature_geometry_loss_weight > 0``); absent -> neutral.
    feature_geom_loss_epoch_agregator = EpochValuesRecorder()
    # Per-environment feature-geometry penalty (RLRP-736) — MS (multi-step forecast
    # horizon) term, emitted as the distinct scalar ``meta["feature_geom_loss_ms"]``
    # by ``ExponentialFamilyMLP.loss`` (the mean geometry penalty over the unrolled
    # forecast horizon). Present only when the term is active AND the model has a
    # multi-step forecast head with ``horizon_len > 1``; absent -> neutral. Kept on
    # its own card so multi-step attitude-drift can be monitored separately from the
    # single-step deploy-head term above.
    feature_geom_loss_ms_epoch_agregator = EpochValuesRecorder()
    # Per-environment feature-geometry penalty (RLRP-736) — CP (compounded-
    # prediction / free-running unroll) term, emitted as the distinct scalar
    # ``meta["feature_geom_loss_cp"]`` by ``ExponentialFamilyMLP.loss`` (the mean
    # geometry penalty over the CP deploy unroll steps). Present only when the term
    # is active AND the model runs a compounded-prediction unroll (e.g. MTM-Pro
    # with the CP deploy path active); absent -> neutral. Kept on its own card so
    # the compounding attitude-drift signal is monitored separately from the
    # single-shot forecast-head (multi-step) term above.
    feature_geom_loss_cp_epoch_agregator = EpochValuesRecorder()

    def multistep_loss_batch_callback(
        epoch_index: int, loss: float, meta: Dict, mode: str
    ) -> None:

        batch_step_recorder.step()

        if (
            tensorboard_writer is not None
            and mode == "train"
            and (
                batch_step_recorder.is_n_step()
                or epoch_step_recorder.step_and_is_new_epoch(epoch_index)
            )
        ):
            if "singlestep_loss" in meta:
                if not ss_loss_epoch_agregator.step_and_is_new_epoch(epoch_index):
                    ss_loss_epoch_agregator.append(meta["singlestep_loss"])
                else:
                    total_avg_singlestep_losses = (
                        ss_loss_epoch_agregator.total_average_values()
                    )
                    ss_loss_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train single-step loss",
                        value=total_avg_singlestep_losses,
                    )

            # Deploy-path training stabilization (RLRP-722, feature (1) forecaster warmup) cards.
            if "deploy_warmup_active" in meta:
                if not deploy_warmup_active_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    deploy_warmup_active_epoch_agregator.append(
                        meta["deploy_warmup_active"]
                    )
                else:
                    total_avg_deploy_warmup_active = (
                        deploy_warmup_active_epoch_agregator.total_average_values()
                    )
                    deploy_warmup_active_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="DeployPathTraining/warmup active (frac)",
                        value=total_avg_deploy_warmup_active,
                    )

            if "deploy_ramp" in meta:
                if not deploy_ramp_epoch_agregator.step_and_is_new_epoch(epoch_index):
                    deploy_ramp_epoch_agregator.append(meta["deploy_ramp"])
                else:
                    total_avg_deploy_ramp = (
                        deploy_ramp_epoch_agregator.total_average_values()
                    )
                    deploy_ramp_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="DeployPathTraining/deploy weight ramp",
                        value=total_avg_deploy_ramp,
                    )

            # Deploy-path training stabilization (RLRP-722, feature (3) EMA forecast clone) card.
            if "ema_forecast_momentum" in meta:
                if not ema_forecast_momentum_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    ema_forecast_momentum_epoch_agregator.append(
                        meta["ema_forecast_momentum"]
                    )
                else:
                    total_avg_ema_forecast_momentum = (
                        ema_forecast_momentum_epoch_agregator.total_average_values()
                    )
                    ema_forecast_momentum_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="DeployPathTraining/EMA forecast momentum",
                        value=total_avg_ema_forecast_momentum,
                    )

            # Per-environment feature-geometry penalty (RLRP-736) card.
            if "feature_geom_loss" in meta:
                if not feature_geom_loss_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    feature_geom_loss_epoch_agregator.append(meta["feature_geom_loss"])
                else:
                    total_avg_feature_geom_loss = (
                        feature_geom_loss_epoch_agregator.total_average_values()
                    )
                    feature_geom_loss_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train feature geometry loss",
                        value=total_avg_feature_geom_loss,
                    )

            # Per-environment feature-geometry penalty (RLRP-736) — MS (multi-step
            # forecast horizon) card, logged in the same fashion as the SS term above.
            if "feature_geom_loss_ms" in meta:
                if not feature_geom_loss_ms_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    feature_geom_loss_ms_epoch_agregator.append(
                        meta["feature_geom_loss_ms"]
                    )
                else:
                    total_avg_feature_geom_loss_ms = (
                        feature_geom_loss_ms_epoch_agregator.total_average_values()
                    )
                    feature_geom_loss_ms_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train feature geometry loss (multi-step)",
                        value=total_avg_feature_geom_loss_ms,
                    )

            # Per-environment feature-geometry penalty (RLRP-736) — CP (compounded-
            # prediction / free-running unroll) card, logged in the same fashion as
            # the SS and MS terms above.
            if "feature_geom_loss_cp" in meta:
                if not feature_geom_loss_cp_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    feature_geom_loss_cp_epoch_agregator.append(
                        meta["feature_geom_loss_cp"]
                    )
                else:
                    total_avg_feature_geom_loss_cp = (
                        feature_geom_loss_cp_epoch_agregator.total_average_values()
                    )
                    feature_geom_loss_cp_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train feature geometry loss (compounded-prediction)",
                        value=total_avg_feature_geom_loss_cp,
                    )

            if "horizon_loss" in meta:
                if not ho_loss_epoch_agregator.step_and_is_new_epoch(epoch_index):
                    ho_loss_epoch_agregator.append(meta["horizon_loss"])
                else:
                    total_avg_ho_losses = ho_loss_epoch_agregator.total_average_values()
                    ho_loss_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train horizon loss",
                        value=total_avg_ho_losses,
                    )

            if "horizon_mixture_loss" in meta:
                if not ms_mix_loss_epoch_agregator.step_and_is_new_epoch(epoch_index):
                    ms_mix_loss_epoch_agregator.append(meta["horizon_loss"])
                else:
                    total_avg_ho_mix_losses = (
                        ms_mix_loss_epoch_agregator.total_average_values()
                    )
                    ms_mix_loss_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train horizon mixture loss",
                        value=total_avg_ho_mix_losses,
                    )

            if "horizon_at_t1_nll_losses" in meta:
                if not ho_at_t1_loss_epoch_agregator.step_and_is_new_epoch(epoch_index):
                    ho_at_t1_loss_epoch_agregator.append(
                        meta["horizon_at_t1_nll_losses"]
                    )
                else:
                    total_avg_ho_at_t1_losses = (
                        ho_at_t1_loss_epoch_agregator.total_average_values()
                    )
                    ho_at_t1_loss_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train horizon at t1 loss",
                        value=total_avg_ho_at_t1_losses,
                    )

            if "nll_losses" in meta:
                if not nll_loss_epoch_agregator.step_and_is_new_epoch(epoch_index):
                    nll_loss_epoch_agregator.append(meta["nll_losses"])
                else:
                    total_avg_nll_losses = (
                        nll_loss_epoch_agregator.total_average_values()
                    )
                    nll_loss_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train nll losses",
                        value=total_avg_nll_losses,
                    )

            if "ms_KL_divergence_loss" in meta:
                if not ms_kl_loss_epoch_agregator.step_and_is_new_epoch(epoch_index):
                    ms_kl_loss_epoch_agregator.append(meta["ms_KL_divergence_loss"])
                else:
                    total_avg_MS_kl_losses = (
                        ms_kl_loss_epoch_agregator.total_average_values()
                    )
                    ms_kl_loss_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train multistep KL losses",
                        value=total_avg_MS_kl_losses,
                    )

            if "ms_siw_mp_loss" in meta:
                if not ms_siw_mp_loss_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    ms_siw_mp_loss_epoch_agregator.append(meta["ms_siw_mp_loss"])
                else:
                    total_avg_ms_siw_mp_losses = (
                        ms_siw_mp_loss_epoch_agregator.total_average_values()
                    )
                    ms_siw_mp_loss_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train multistep SIW-MP losses",
                        value=total_avg_ms_siw_mp_losses,
                    )

            if "ms_gms_iwae_loss" in meta:
                if not ms_gms_iwae_loss_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    ms_gms_iwae_loss_epoch_agregator.append(meta["ms_gms_iwae_loss"])
                else:
                    total_avg_ms_gms_iwae_losses = (
                        ms_gms_iwae_loss_epoch_agregator.total_average_values()
                    )
                    ms_gms_iwae_loss_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train multistep GMS-IWAE losses",
                        value=total_avg_ms_gms_iwae_losses,
                    )

            if "ss_KL_divergence_loss" in meta:
                if not ss_kl_loss_epoch_agregator.step_and_is_new_epoch(epoch_index):
                    ss_kl_loss_epoch_agregator.append(meta["ss_KL_divergence_loss"])
                else:
                    total_avg_SS_kl_losses = (
                        ss_kl_loss_epoch_agregator.total_average_values()
                    )
                    ss_kl_loss_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train single-step KL losses",
                        value=total_avg_SS_kl_losses,
                    )

            if "encoder_loss" in meta:
                if not encoder_loss_epoch_agregator.step_and_is_new_epoch(epoch_index):
                    encoder_loss_epoch_agregator.append(meta["encoder_loss"])
                else:
                    total_avg_encoder_losses = (
                        encoder_loss_epoch_agregator.total_average_values()
                    )
                    encoder_loss_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train encoder losses",
                        value=total_avg_encoder_losses,
                    )

            if "decoder_loss" in meta:
                if not decoder_loss_epoch_agregator.step_and_is_new_epoch(epoch_index):
                    decoder_loss_epoch_agregator.append(meta["decoder_loss"])
                else:
                    total_avg_decoder_losses = (
                        decoder_loss_epoch_agregator.total_average_values()
                    )
                    decoder_loss_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train decoder losses",
                        value=total_avg_decoder_losses,
                    )

            if "ms_pre_mixture_u_loss" in meta:
                if not pre_mixture_u_loss_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    pre_mixture_u_loss_epoch_agregator.append(
                        meta["ms_pre_mixture_u_loss"]
                    )
                else:
                    total_avg_pre_mixture_u_loss_losses = (
                        pre_mixture_u_loss_epoch_agregator.total_average_values()
                    )
                    pre_mixture_u_loss_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train pre-mixture U losses",
                        value=total_avg_pre_mixture_u_loss_losses,
                    )

            if "ms_post_mixture_loss" in meta:
                if not post_mixture_loss_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    post_mixture_loss_epoch_agregator.append(
                        meta["ms_post_mixture_loss"]
                    )
                else:
                    total_avg_post_mixture_loss = (
                        post_mixture_loss_epoch_agregator.total_average_values()
                    )
                    post_mixture_loss_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train post-mixture (MIX) losses",
                        value=total_avg_post_mixture_loss,
                    )

            if "ms_info_projection_loss" in meta:
                if not info_projection_loss_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    info_projection_loss_epoch_agregator.append(
                        meta["ms_info_projection_loss"]
                    )
                else:
                    total_avg_info_projection_loss = (
                        info_projection_loss_epoch_agregator.total_average_values()
                    )
                    info_projection_loss_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train information-projection (IPROJ) losses",
                        value=total_avg_info_projection_loss,
                    )

            if "ms_rollout_consistency_loss" in meta:
                if not rollout_consistency_loss_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    rollout_consistency_loss_epoch_agregator.append(
                        meta["ms_rollout_consistency_loss"]
                    )
                else:
                    total_avg_rollout_consistency_loss = (
                        rollout_consistency_loss_epoch_agregator.total_average_values()
                    )
                    rollout_consistency_loss_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train rollout self-consistency (RC) losses",
                        value=total_avg_rollout_consistency_loss,
                    )

            # Deploy-history drift residual (DH) CP sub-term card (RLRP-731).
            if "ms_deploy_history_drift_loss" in meta:
                if not dh_loss_epoch_agregator.step_and_is_new_epoch(epoch_index):
                    dh_loss_epoch_agregator.append(meta["ms_deploy_history_drift_loss"])
                else:
                    total_avg_dh_loss = dh_loss_epoch_agregator.total_average_values()
                    dh_loss_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train deploy-history drift (DH)",
                        value=total_avg_dh_loss,
                    )

            if "ms_projection_vs_horizon_ratio" in meta:
                if not projection_vs_horizon_ratio_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    projection_vs_horizon_ratio_epoch_agregator.append(
                        meta["ms_projection_vs_horizon_ratio"]
                    )
                else:
                    total_avg_projection_vs_horizon_ratio = (
                        projection_vs_horizon_ratio_epoch_agregator.total_average_values()
                    )
                    projection_vs_horizon_ratio_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train projection-vs-horizon magnitude ratio",
                        value=total_avg_projection_vs_horizon_ratio,
                    )

            if "ms_energy_beta" in meta:
                if not ms_energy_beta_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    ms_energy_beta_epoch_agregator.append(meta["ms_energy_beta"])
                else:
                    total_avg_ms_energy_beta = (
                        ms_energy_beta_epoch_agregator.total_average_values()
                    )
                    ms_energy_beta_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train ms energy-density beta",
                        value=total_avg_ms_energy_beta,
                    )

            # .... Composite term-balance monitor (RLRP-720) ......................................
            # Log the share of each active composite term so a DWARFED (share -> 0) or DOMINATING
            # (share -> 1) term is visible in TensorBoard. Keys are emitted by the model's
            # `_probabilistic_loss` as "<TERM>_composite_share".
            for _share_key in [k for k in meta if k.endswith("_composite_share")]:
                _agg = composite_share_epoch_agregators.setdefault(
                    _share_key, EpochValuesRecorder()
                )
                if not _agg.step_and_is_new_epoch(epoch_index):
                    _agg.append(meta[_share_key])
                else:
                    _total_avg_share = _agg.total_average_values()
                    _agg.flush()
                    _term_name = _share_key[: -len("_composite_share")]
                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag=f"Loss/train composite share {_term_name}",
                        value=_total_avg_share,
                    )

            # .... TW-MS-MPPI .....................................................................
            if "mppi_unnormalized_pred_ho_iw" in meta:
                if not mppi_unnormalized_pred_ho_iw_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    mppi_unnormalized_pred_ho_iw_epoch_agregator.append(
                        meta["mppi_unnormalized_pred_ho_iw"]
                    )
                else:
                    total_avg_mppi_unnormalized_pred_ho_iw_epoch_agregator = (
                        mppi_unnormalized_pred_ho_iw_epoch_agregator.total_average_values()
                    )
                    mppi_unnormalized_pred_ho_iw_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="MPPI/train unnormalized prediction horizon importance weights",
                        value=total_avg_mppi_unnormalized_pred_ho_iw_epoch_agregator,
                    )

            if "mppi_cost" in meta:
                if not mppi_cost_epoch_agregator.step_and_is_new_epoch(epoch_index):
                    mppi_cost_epoch_agregator.append(meta["mppi_cost"])
                else:
                    total_avg_mppi_cost_epoch_agregator = (
                        mppi_cost_epoch_agregator.total_average_values()
                    )
                    mppi_cost_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="MPPI/train cost",
                        value=total_avg_mppi_cost_epoch_agregator,
                    )

            if "mppi_epsilon" in meta:
                if not mppi_epsilon_epoch_agregator.step_and_is_new_epoch(epoch_index):
                    mppi_epsilon_epoch_agregator.append(meta["mppi_epsilon"])
                else:
                    total_avg_mppi_epsilon_epoch_agregator = (
                        mppi_epsilon_epoch_agregator.total_average_values()
                    )
                    mppi_epsilon_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="MPPI/train epsilon (temporally correlated noise)",
                        value=total_avg_mppi_epsilon_epoch_agregator,
                    )
            # .... Composite loss auto weighting ..................................................
            if "SS_loss_auto_weighting" in meta:
                if not ss_loss_auto_weighting_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    ss_loss_auto_weighting_agregator.append(
                        meta["SS_loss_auto_weighting"]
                    )
                else:
                    total_avg_ss_loss_auto_weighting_agregator = (
                        ss_loss_auto_weighting_agregator.total_average_values()
                    )
                    ss_loss_auto_weighting_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train SS loss auto weighting",
                        value=total_avg_ss_loss_auto_weighting_agregator,
                    )

            if "MS_loss_auto_weighting" in meta:
                if not ms_loss_auto_weighting_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    ms_loss_auto_weighting_agregator.append(
                        meta["MS_loss_auto_weighting"]
                    )
                else:
                    total_avg_ms_loss_auto_weighting_agregator = (
                        ms_loss_auto_weighting_agregator.total_average_values()
                    )
                    ms_loss_auto_weighting_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train MS loss auto weighting",
                        value=total_avg_ms_loss_auto_weighting_agregator,
                    )
            if "SS_KL_loss_auto_weighting" in meta:
                if not ss_kl_loss_auto_weighting_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    ss_kl_loss_auto_weighting_agregator.append(
                        meta["SS_KL_loss_auto_weighting"]
                    )
                else:
                    total_avg_ss_kl_loss_auto_weighting_agregator = (
                        ss_kl_loss_auto_weighting_agregator.total_average_values()
                    )
                    ss_kl_loss_auto_weighting_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train SS KL loss auto weighting",
                        value=total_avg_ss_kl_loss_auto_weighting_agregator,
                    )

            if "MS_KL_loss_auto_weighting" in meta:
                if not ms_kl_loss_auto_weighting_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    ms_kl_loss_auto_weighting_agregator.append(
                        meta["MS_KL_loss_auto_weighting"]
                    )
                else:
                    total_avg_ms_kl_loss_auto_weighting_agregator = (
                        ms_kl_loss_auto_weighting_agregator.total_average_values()
                    )
                    ms_kl_loss_auto_weighting_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train MS KL loss auto weighting",
                        value=total_avg_ms_kl_loss_auto_weighting_agregator,
                    )

            if "MS_MIX_loss_auto_weighting" in meta:
                if not ms_MIX_loss_auto_weighting_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    ms_MIX_loss_auto_weighting_agregator.append(
                        meta["MS_MIX_loss_auto_weighting"]
                    )
                else:
                    total_avg_ms_mix_loss_auto_weighting_agregator = (
                        ms_MIX_loss_auto_weighting_agregator.total_average_values()
                    )
                    ms_MIX_loss_auto_weighting_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train MS mixture loss auto weighting",
                        value=total_avg_ms_mix_loss_auto_weighting_agregator,
                    )

            if "MS_SIW_MP_loss_auto_weighting" in meta:
                if not ms_siw_mp_loss_auto_weighting_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    ms_siw_mp_loss_auto_weighting_agregator.append(
                        meta["MS_SIW_MP_loss_auto_weighting"]
                    )
                else:
                    total_avg_ms_siw_mp_loss_auto_weighting_agregator = (
                        ms_siw_mp_loss_auto_weighting_agregator.total_average_values()
                    )
                    ms_siw_mp_loss_auto_weighting_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train MS SIW-MP loss auto weighting",
                        value=total_avg_ms_siw_mp_loss_auto_weighting_agregator,
                    )

            if "MS_RC_loss_auto_weighting" in meta:
                if not ms_RC_loss_auto_weighting_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    ms_RC_loss_auto_weighting_agregator.append(
                        meta["MS_RC_loss_auto_weighting"]
                    )
                else:
                    total_avg_ms_RC_loss_auto_weighting_agregator = (
                        ms_RC_loss_auto_weighting_agregator.total_average_values()
                    )
                    ms_RC_loss_auto_weighting_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train MS RC loss auto weighting",
                        value=total_avg_ms_RC_loss_auto_weighting_agregator,
                    )

            if "MS_GMS_IWAE_loss_auto_weighting" in meta:
                if not ms_gms_iwae_loss_auto_weighting_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    ms_gms_iwae_loss_auto_weighting_agregator.append(
                        meta["MS_GMS_IWAE_loss_auto_weighting"]
                    )
                else:
                    total_avg_ms_gms_iwae_loss_auto_weighting_agregator = (
                        ms_gms_iwae_loss_auto_weighting_agregator.total_average_values()
                    )
                    ms_gms_iwae_loss_auto_weighting_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train MS GMS-IWAE loss auto weighting",
                        value=total_avg_ms_gms_iwae_loss_auto_weighting_agregator,
                    )

            if "U_loss_auto_weighting" in meta:
                if not ms_U_loss_auto_weighting_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    ms_U_loss_auto_weighting_agregator.append(
                        meta["U_loss_auto_weighting"]
                    )
                else:
                    total_avg_ms_U_loss_auto_weighting_agregator = (
                        ms_U_loss_auto_weighting_agregator.total_average_values()
                    )
                    ms_U_loss_auto_weighting_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train U loss auto weighting",
                        value=total_avg_ms_U_loss_auto_weighting_agregator,
                    )

            if "IPROJ_loss_auto_weighting" in meta:
                if not IPROJ_loss_auto_weighting_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    IPROJ_loss_auto_weighting_agregator.append(
                        meta["IPROJ_loss_auto_weighting"]
                    )
                else:
                    total_avg_IPROJ_loss_auto_weighting_agregator = (
                        IPROJ_loss_auto_weighting_agregator.total_average_values()
                    )
                    IPROJ_loss_auto_weighting_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train IPROJ loss auto weighting",
                        value=total_avg_IPROJ_loss_auto_weighting_agregator,
                    )

            if "DECODER_loss_auto_weighting" in meta:
                if not ms_decoder_loss_auto_weighting_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    ms_decoder_loss_auto_weighting_agregator.append(
                        meta["DECODER_loss_auto_weighting"]
                    )
                else:
                    total_avg_ms_decoder_loss_auto_weighting_agregator = (
                        ms_decoder_loss_auto_weighting_agregator.total_average_values()
                    )
                    ms_decoder_loss_auto_weighting_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train DECODER loss auto weighting",
                        value=total_avg_ms_decoder_loss_auto_weighting_agregator,
                    )

            if "ENCODER_loss_auto_weighting" in meta:
                if not ms_encoder_loss_auto_weighting_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    ms_encoder_loss_auto_weighting_agregator.append(
                        meta["ENCODER_loss_auto_weighting"]
                    )
                else:
                    total_avg_ms_encoder_loss_auto_weighting_agregator = (
                        ms_encoder_loss_auto_weighting_agregator.total_average_values()
                    )
                    ms_encoder_loss_auto_weighting_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train ENCODER loss auto weighting",
                        value=total_avg_ms_encoder_loss_auto_weighting_agregator,
                    )

            if "CP_loss_auto_weighting" in meta:
                if not cp_loss_auto_weighting_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    cp_loss_auto_weighting_agregator.append(
                        meta["CP_loss_auto_weighting"]
                    )
                else:
                    total_avg_cp_loss_auto_weighting_agregator = (
                        cp_loss_auto_weighting_agregator.total_average_values()
                    )
                    cp_loss_auto_weighting_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train CP loss auto weighting",
                        value=total_avg_cp_loss_auto_weighting_agregator,
                    )

            if "DH_loss_auto_weighting" in meta:
                if not dh_loss_auto_weighting_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    dh_loss_auto_weighting_agregator.append(
                        meta["DH_loss_auto_weighting"]
                    )
                else:
                    total_avg_dh_loss_auto_weighting_agregator = (
                        dh_loss_auto_weighting_agregator.total_average_values()
                    )
                    dh_loss_auto_weighting_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train DH loss auto weighting",
                        value=total_avg_dh_loss_auto_weighting_agregator,
                    )

            # RLRP-751 (task T7): per-feature geometry AUTO-weighting cards.
            if "FEAT_GEOM_SS_loss_auto_weighting" in meta:
                if not feature_geom_ss_loss_auto_weighting_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    feature_geom_ss_loss_auto_weighting_agregator.append(
                        meta["FEAT_GEOM_SS_loss_auto_weighting"]
                    )
                else:
                    total_avg_feature_geom_ss_loss_auto_weighting = (
                        feature_geom_ss_loss_auto_weighting_agregator.total_average_values()
                    )
                    feature_geom_ss_loss_auto_weighting_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train feature geometry loss auto weighting (single-step)",
                        value=total_avg_feature_geom_ss_loss_auto_weighting,
                    )

            if "FEAT_GEOM_MS_loss_auto_weighting" in meta:
                if not feature_geom_ms_loss_auto_weighting_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    feature_geom_ms_loss_auto_weighting_agregator.append(
                        meta["FEAT_GEOM_MS_loss_auto_weighting"]
                    )
                else:
                    total_avg_feature_geom_ms_loss_auto_weighting = (
                        feature_geom_ms_loss_auto_weighting_agregator.total_average_values()
                    )
                    feature_geom_ms_loss_auto_weighting_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train feature geometry loss auto weighting (multi-step)",
                        value=total_avg_feature_geom_ms_loss_auto_weighting,
                    )

            if "FEAT_GEOM_CP_loss_auto_weighting" in meta:
                if not feature_geom_cp_loss_auto_weighting_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    feature_geom_cp_loss_auto_weighting_agregator.append(
                        meta["FEAT_GEOM_CP_loss_auto_weighting"]
                    )
                else:
                    total_avg_feature_geom_cp_loss_auto_weighting = (
                        feature_geom_cp_loss_auto_weighting_agregator.total_average_values()
                    )
                    feature_geom_cp_loss_auto_weighting_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Loss/train feature geometry loss auto weighting (compounded-prediction)",
                        value=total_avg_feature_geom_cp_loss_auto_weighting,
                    )

        return None

    return multistep_loss_batch_callback


def setup_compounded_pred_unroll_len_batch_callback(
    cfg: omegaconf.DictConfig,
    cfg_training: omegaconf.DictConfig,
    model_trainer: models.ModelTrainer,
    tensorboard_writer: Optional[OnlineTensorboardWritter],
) -> Callable:
    epoch_step_recorder = EpochStepRecorder()
    batch_step_recorder = BatchStepRecorder(
        flag_every_n_step=cfg.pipeline.tensorboard.batch_callback_execute_every_n
    )

    horizon_unroll_len_epoch_agregator = EpochValuesRecorder()
    teacher_forcing_prob_epoch_agregator = EpochValuesRecorder()
    # RLRP-726: dedicated aggregator for the MS *forecast*-path teacher-forcing scheduler
    # (scheduled sampling on `state_history_update`), INDEPENDENT of the CP/deploy
    # `teacher_forcing_prob` above. Emitted by MTM-Pro as `meta["forecast_teacher_forcing_prob"]`.
    forecast_teacher_forcing_prob_epoch_agregator = EpochValuesRecorder()
    ar_temporal_weights_epoch_agregator = EpochValuesRecorder()

    def compounded_pred_unroll_len_batch_callback(
        epoch_index: int, loss: float, meta: Dict, mode: str
    ) -> None:

        batch_step_recorder.step()

        if (
            tensorboard_writer is not None
            and mode == "train"
            and (
                batch_step_recorder.is_n_step()
                or epoch_step_recorder.step_and_is_new_epoch(epoch_index)
            )
        ):
            if "horizon_unroll_len" in meta:
                if not horizon_unroll_len_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    horizon_unroll_len_epoch_agregator.append(
                        meta["horizon_unroll_len"]
                    )
                else:
                    total_avg_horizon_unroll_len = (
                        horizon_unroll_len_epoch_agregator.total_average_values()
                    )
                    horizon_unroll_len_prob_variance = np.var(
                        horizon_unroll_len_epoch_agregator.values
                    ).item()

                    horizon_unroll_len_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Compounded prediction/horizon unroll len (avg)",
                        value=total_avg_horizon_unroll_len,
                    )
                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Compounded prediction/horizon unroll len (variance)",
                        value=horizon_unroll_len_prob_variance,
                    )

            if "teacher_forcing_prob" in meta:
                if not teacher_forcing_prob_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    teacher_forcing_prob_epoch_agregator.append(
                        meta["teacher_forcing_prob"]
                    )
                else:
                    total_avg_teacher_forcing_prob = (
                        teacher_forcing_prob_epoch_agregator.total_average_values()
                    )
                    teacher_forcing_prob_variance = np.var(
                        teacher_forcing_prob_epoch_agregator.values
                    ).item()

                    teacher_forcing_prob_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Compounded prediction/teacher forcing prob (avg)",
                        value=total_avg_teacher_forcing_prob,
                    )
                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Compounded prediction/teacher forcing prob (variance)",
                        value=teacher_forcing_prob_variance,
                    )

            # RLRP-726: MS *forecast*-path teacher-forcing prob (dedicated scheduler, scheduled
            # sampling on `state_history_update`). Independent of the CP/deploy key above; published
            # under its own "Compounded prediction/(MS forecast) teacher forcing prob" cards so the
            # two schedules are grouped together yet never conflated.
            if "forecast_teacher_forcing_prob" in meta:
                if not forecast_teacher_forcing_prob_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    forecast_teacher_forcing_prob_epoch_agregator.append(
                        meta["forecast_teacher_forcing_prob"]
                    )
                else:
                    total_avg_forecast_teacher_forcing_prob = (
                        forecast_teacher_forcing_prob_epoch_agregator.total_average_values()
                    )
                    forecast_teacher_forcing_prob_variance = np.var(
                        forecast_teacher_forcing_prob_epoch_agregator.values
                    ).item()

                    forecast_teacher_forcing_prob_epoch_agregator.flush()

                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Compounded prediction/(MS forecast) teacher forcing prob (avg)",
                        value=total_avg_forecast_teacher_forcing_prob,
                    )
                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Compounded prediction/(MS forecast) teacher forcing prob (variance)",
                        value=forecast_teacher_forcing_prob_variance,
                    )

            if "ar_temporal_weights" in meta:
                if not ar_temporal_weights_epoch_agregator.step_and_is_new_epoch(
                    epoch_index
                ):
                    ar_temporal_weights_epoch_agregator.append(
                        meta["ar_temporal_weights"]
                    )
                else:
                    total_avg_ar_temporal_weights = (
                        ar_temporal_weights_epoch_agregator.total_average_values()
                    )
                    ar_temporal_weights_epoch_agregator.flush()
                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        tag="Compounded prediction/ar temporal weights (avg)",
                        value=total_avg_ar_temporal_weights,
                    )

        return None

    return compounded_pred_unroll_len_batch_callback


def setup_gradient_monitoring_callback(
    cfg: omegaconf.DictConfig,
    cfg_training: omegaconf.DictConfig,
    model_trainer: models.ModelTrainer,
    tensorboard_writer: Optional[OnlineTensorboardWritter],
) -> Callable:
    def gradient_monitoring_callback(
        model: mbrl.models.Model,
        train_iteration: int,
        epoch: int,
        total_avg_loss: float,
        eval_score: float,
        best_val_score: float,
    ) -> None:
        if tensorboard_writer is not None and fetch_cfg_pipeline_tensorboard_key_value(
            cfg, key="enable_gradient_and_parameters_histogram", key_value_default=True
        ):
            # Note: Tensorboard add_histogram can't handle large number
            VIS_CLIP_VALUE = 1e10

            # RLRP-289: histogram step-axis alignment. The `epoch` argument forwarded by
            # `mbrl.models.ModelTrainer.train(...)` is the *inner* Lightning
            # `trainer.current_epoch`, which restarts at 0 on every `train(...)` call, i.e. on
            # every ERLL/UDER pass (a fresh `pl.Trainer` is built per call). Using it as
            # `global_step` made every histogram tag collide with the step range already written
            # by the previous pass: TensorBoard purges the out-of-order tail and keeps rewriting
            # the same slots, so the parameter distribution *looked* clamped/frozen from the
            # second UDER pass onward even though the parameters kept moving.
            #
            # Publish instead on the writer's monotonic global epoch, which is the very same axis
            # already used by `add_scalar_per_epoch_monitoring` below (`Model gradient norm (per
            # epoch mean)/*`) and by `OnlineTensorboardWritter.__call__` (`Loss/*`). The writer
            # `__call__` runs *before* the registered epoch callbacks (see
            # `setup_trainer_epoch_callback_aggregator`), so the counter is already incremented
            # for the current epoch here. Falls back to the inner `epoch` when the writer does not
            # expose the accessor (legacy behaviour, single-pass runs are unaffected).
            histogram_global_step = getattr(
                tensorboard_writer, "current_global_epoch", epoch
            )

            generate_model_parameters_head_leaf = (
                fetch_cfg_pipeline_tensorboard_key_value(
                    cfg, "generate_model_parameters_head_leaf", False
                )
            )
            generate_model_parameters = fetch_cfg_pipeline_tensorboard_key_value(
                cfg, "generate_model_parameters", False
            )

            generate_model_gradients = fetch_cfg_pipeline_tensorboard_key_value(
                cfg, "generate_model_gradients", False
            )

            generate_model_gradient_norm = fetch_cfg_pipeline_tensorboard_key_value(
                cfg, "generate_model_gradient_norm", False
            )

            ms_head_num_layers = (
                model.model.ms_head_num_layers - 1
                if hasattr(model.model, "ms_head_num_layers")
                else -1
            )
            ss_head_num_layers = (
                model.model.ss_head_num_layers - 1
                if hasattr(model.model, "ss_head_num_layers")
                else -1
            )
            projection_head_num_layers = (
                model.model.projection_head_split_num_layers
                if hasattr(model.model, "projection_head_split_num_layers")
                else -1
            )

            if generate_model_parameters_head_leaf:
                # RLRP-735: auto-discover the temporal-mixture logit leaves to monitor. The mixer
                # family now includes COMPOSED variants whose logit-producing parameters are nested
                # (e.g. ``index.mixture_logits``, ``input_head.mixture_logits.weight``) or additive
                # (``index_base``), so the previous single hard-coded name would silently drop them.
                # The helper introspects the mixer subtree by naming convention and returns names
                # relative to the mixer; prefix them to match ``model.named_parameters()`` keys.
                mixture_head_leaf_param_names: List[str] = []
                _mixer = getattr(
                    getattr(model, "model", None), "temporal_mixture_weights", None
                )
                if _mixer is not None:
                    mixture_head_leaf_param_names = [
                        f"model.temporal_mixture_weights.{each}"
                        for each in temporal_mixture_head_leaf_parameter_names(_mixer)
                    ]

                    # RLRP-735: single MERGED histogram of the mixer's composing learnable parameters
                    # (following the additive composition pattern of the mixer's forward/logits, e.g.
                    # φ = γ + ψ + ρ). Gated by the mixer's own `show_mixer_composed_params` toggle so a
                    # variant can opt out. Logged once per epoch (not per-parameter).
                    if getattr(_mixer, "show_mixer_composed_params", False) and hasattr(
                        _mixer, "merged_mixer_parameter_vector"
                    ):
                        merged_mixer_params = _mixer.merged_mixer_parameter_vector()
                        if (
                            merged_mixer_params is not None
                            and merged_mixer_params.numel() > 0
                        ):
                            merged_vis = torch.clamp(
                                merged_mixer_params.detach().cpu(),
                                -VIS_CLIP_VALUE,
                                VIS_CLIP_VALUE,
                            )
                            tensorboard_writer.writer.add_histogram(
                                # RLRP-735: same shortened convention as the head-leaf loop below
                                # (``model.temporal_mixture_weights.*`` -> ``temporal_mix.*``).
                                tag="Model parameters head leaf/temporal_mix.composed_params",
                                values=merged_vis.numpy(),
                                global_step=histogram_global_step,
                            )

            for layer_idx, (name, param) in enumerate(model.named_parameters()):
                if param.requires_grad:

                    try:
                        param_grad = param.grad.clone().cpu()
                        param_vis = torch.clamp(
                            param.detach().cpu(), -VIS_CLIP_VALUE, VIS_CLIP_VALUE
                        )

                        grad_vis = torch.clamp(
                            param_grad, -VIS_CLIP_VALUE, VIS_CLIP_VALUE
                        )

                        if "bias" not in name:

                            # .... Fetch head leaf param only  ....................................
                            if generate_model_parameters_head_leaf:
                                for each in [
                                    f"model.deploy_head_mean.{ss_head_num_layers}.0.weight",
                                    f"model.deploy_head_logvar.{ss_head_num_layers}.0.weight",
                                    f"model.mean_layer.{ms_head_num_layers}.0.weight",
                                    f"model.logvar_layer.{ms_head_num_layers}.0.weight",
                                    f"model.projection_head.mean_head.{projection_head_num_layers}.weight",
                                    f"model.projection_head.logvar_head.{projection_head_num_layers}.weight",
                                    *mixture_head_leaf_param_names,
                                ]:
                                    if each in name:
                                        # RLRP-735: shorten the displayed tensorboard tag. Mixer
                                        # params (``model.temporal_mixture_weights.*``) are long due
                                        # to composition/nesting, so collapse them to
                                        # ``temporal_mix.*``; strip the leading ``model.`` prefix
                                        # from any other head-leaf param for a compact display.
                                        short_name = name
                                        if short_name.startswith(
                                            "model.temporal_mixture_weights."
                                        ):
                                            short_name = short_name.replace(
                                                "model.temporal_mixture_weights.",
                                                "temporal_mix.",
                                                1,
                                            )
                                        elif short_name.startswith("model."):
                                            short_name = short_name[len("model.") :]
                                        tensorboard_writer.writer.add_histogram(
                                            tag=f"Model parameters head leaf/{short_name}",
                                            values=param_vis.numpy(),
                                            global_step=histogram_global_step,
                                        )

                            # .... Fetch param and grad ...........................................
                            if generate_model_parameters:
                                tensorboard_writer.writer.add_histogram(
                                    tag=f"Model parameters/{name}",
                                    values=param_vis.numpy(),
                                    global_step=histogram_global_step,
                                )
                            if generate_model_gradients:
                                tensorboard_writer.writer.add_histogram(
                                    tag=f"Model gradients/{name}",
                                    values=grad_vis.numpy(),
                                    global_step=histogram_global_step,
                                )

                        # tensorboard_writer.add_scalar_per_epoch_monitoring(
                        #     tag=f"Model gradient (per epoch mean)/{name}",
                        #     value=param_grad.abs().mean().item(),
                        # )
                        # Note (RLRP-289): this scalar already publishes on the writer's global
                        # epoch axis; the histograms above now match it.
                        if generate_model_gradient_norm:
                            tensorboard_writer.add_scalar_per_epoch_monitoring(
                                tag=f"Model gradient norm (per epoch mean)/{name}",
                                value=param_grad.data.norm(2).item() ** 2,
                            )
                    except (AttributeError, ValueError) as e:
                        if str(e) == "'NoneType' object has no attribute 'clone'":
                            pass
                        elif (
                            str(e)
                            == "The histogram is empty, please file a bug report."
                        ):
                            pass
                        else:
                            raise e

        return None

    return gradient_monitoring_callback


def setup_tensorboard_arbitrary_dimension_trajectory_prediction_monitor_callback(
    cfg: omegaconf.DictConfig,
    cfg_training: omegaconf.DictConfig,
    model_trainer: models.ModelTrainer,
    tensorboard_writer: Optional[OnlineTensorboardWritter],
    trajectories: Union[
        list[TestTrajectoryDataclass], list[TestMotionTrajectoryDataclass]
    ],
    motion_model_container: Optional[R2SMotionModelContainer] = None,
    execute_every_n_epoch: Optional[int] = 500,
    trajectory_record_step_size: int = 100,
    obs_dim_label: Tuple[str, ...] = ("X", "Y", "Z"),
    rollout_dim_label: Tuple[str, ...] = ("X", "Y", "Z"),
    label: str = "Train",
) -> Callable:
    # RLRP-289 (resolved): every per-epoch publish is aligned on the writer's monotonic global
    # epoch (`OnlineTensorboardWritter.current_global_epoch` / `.total_epoch`) rather than on the
    # inner Lightning `epoch`, which restarts at 0 on every ERLL/UDER pass.
    """ModelTrainer.Train callback for monitoring predictions values per epoch:
    - prediction distribution over trajectory
    - prediction epistemic uncertainty
    - prediction MAE

    Note:
    -----
    - execute_every_n_epoch: set to None to skip callback execution at runtime
    - Minimum required argument:

        >>> partial(
        >>>     setup_tensorboard_arbitrary_dimension_trajectory_prediction_monitor_callback,
        >>>     **{"trajectories": [validation_environment]},
        >>> )

    - For the multistep model case, instanciate the callback using

        >>> partial(
        >>>     setup_tensorboard_arbitrary_dimension_trajectory_prediction_monitor_callback,
        >>>     **{"trajectories": [validation_environment],
        >>>        "motion_model_container": motion_model_container
        >>>     },
        >>> )

    :param label:
    :param cfg:
    :param cfg_training:
    :param model_trainer:
    :param tensorboard_writer:
    :param trajectories:
    :param motion_model_container:
    :param execute_every_n_epoch: Set to `None` to switch callback off
    :param trajectory_record_step_size:
    :param obs_dim_label: Tuple of dimension names to specify which trajectory observed dimensions
    :param rollout_dim_label: Tuple of dimension names to specify which trajectory rollout dimensions
    :return: a callback function
    """
    for each_tt in trajectories:
        assert isinstance(
            each_tt, (TestTrajectoryDataclass, TestMotionTrajectoryDataclass)
        )

    if tensorboard_writer is not None:
        seed, torch_rng = seed_me(cfg, output_torch_rdn_generator=True)

    def tensorboard_arbitrary_dimension_trajectory_prediction_monitor_callback(
        model, train_iteration, epoch, total_avg_loss, eval_score, best_val_score
    ) -> None:
        """Called by `mbrl.models.ModelTrainer.train(...)` method at the end of each epoch."""

        if (
            tensorboard_writer is not None
            and execute_every_n_epoch is not None
            and (tensorboard_writer.total_epoch % execute_every_n_epoch == 0)
        ):
            prediction_metric = model_testtime_rollout_and_compute_prediction_metric(
                cfg,
                model_trainer.model,
                motion_model_container,
                torch_rng,
                trajectories,
                compounded_predictions_score=cfg.deploy.target_experiment.compounded_predictions_score,
                next_state_deterministic_selection=cfg.deploy.model_runtime.next_state_deterministic_selection,
                next_state_sampling_size=cfg.deploy.model_runtime.next_state_sampling_size,
                ground_truth_feed_warmup_steps=cfg.deploy.target_experiment.ground_truth_feed_warmup_steps,
                tutor_and_release=cfg.deploy.target_experiment.get(
                    "tutor_and_release", False
                ),
                show_debug_info=cfg.get("debug_mode", False),
            )

            tensorboard_arbitrary_dimension_prediction_trajectory_writer(
                tensorboard_writer,
                prediction_metric,
                tensorboard_writer.total_epoch,
                trajectory_record_step_size,
                is_model_ensemble=is_model_ensemble(model_trainer.model.model)
                and model_trainer.model.model.num_members > 1,
                label_pre=f"[{label}] ",
                obs_dim_label=obs_dim_label,
                rollout_dim_label=rollout_dim_label,
                show_prediction_distributions=fetch_cfg_pipeline_tensorboard_key_value(
                    cfg, "generate_prediction_distributions", False
                ),
                show_prediction_rollout_mae=fetch_cfg_pipeline_tensorboard_key_value(
                    cfg, "generate_prediction_rollout_mae", False
                ),
            )

            tensorboard_prediction_error(
                cfg,
                f"{label}",
                prediction_metric.mae.sum(),
                prediction_metric.l2_norm.sum(),
                tensorboard_writer,
                # RLRP-723: pass the rollout horizon (number of scored steps) so
                # the trajectory-length normalized ``avg-mae`` / ``avg-L2 norm``
                # variants can be logged alongside the cumulative ones.
                # Read the horizon (number of scored steps) straight from the tensor shape.
                # `np.asarray(...)` would force a device→host copy and raises on a CUDA tensor
                # ("can't convert cuda:0 device type tensor to numpy"); `.shape[0]` is
                # device-agnostic and works for both torch tensors and numpy arrays.
                rollout_horizon=int(prediction_metric.mae.shape[0]),
            )

        return None

    return tensorboard_arbitrary_dimension_trajectory_prediction_monitor_callback


def fetch_cfg_pipeline_tensorboard_key_value(
    cfg: omegaconf.DictConfig, key: str, key_value_default: Any
) -> Any:
    """
    Fetches a specific key-value from the 'pipeline.tensorboard' section of the provided
    configuration.

    This function checks if a specified key exists in the 'pipeline.tensorboard' section of the
    configuration dictionary. If the key exists, its value is returned; if the value is a list,
    it is converted to a tuple. If the key does not exist, a default value is returned.

    :param cfg: Configuration object
    :param key: The specific key to look for within 'cfg.pipeline.tensorboard'.
    :param key_value_default: The default value to return if the key does not exist.
    :return: The value of the specified key if it exists, otherwise the default value.
    """
    if is_cfg_key_exist(cfg, f"pipeline.tensorboard.{key}"):
        key_value = cfg.pipeline.tensorboard.get(key)
        if omegaconf.OmegaConf.is_list(key_value):
            key_value = tuple(omegaconf.OmegaConf.to_object(key_value))
    else:
        key_value = key_value_default

    return key_value


def tensorboard_arbitrary_dimension_prediction_trajectory_writer(
    tensorboard_writer: OnlineTensorboardWritter,
    prediction_metric: PredictionMetric,
    current_epoch: int,
    trajectory_record_step_size: int,
    is_model_ensemble: bool = False,
    label_pre: str = "",
    obs_dim_label: Tuple[str, ...] = ("X", "Y", "Z"),
    rollout_dim_label: Tuple[str, ...] = ("X", "Y", "Z"),
    show_prediction_distributions: bool = False,
    show_prediction_rollout_mae: bool = False,
) -> None:
    """
    Writes prediction trajectory metrics for arbitrary dimensions to
    TensorBoard, including probability distributions, epistemic uncertainty,
    and mean absolute error (MAE) for the specified dimensions, at each epoch.

    Will iterate over `obs_dim_label` tuple. Its the user responsibilty to ensure that
    `obs_dim_label` order match `prediction_metric` field ndarray dimension ordering.

    :param tensorboard_writer: Online tensorboard writer instance used for logging data.
    :param prediction_metric: Contains trajectory statistical metrics such as mean, standard
    deviation, epistemic uncertainty, and MAE.
    :param current_epoch: Current epoch number for tracking the logged data.
    :param trajectory_record_step_size: Step size between trajectory records being written to
    TensorBoard.
    :param is_model_ensemble:
    :param label_pre: Prefix label to distinguish logged metrics.
    :param obs_dim_label: Tuple of dimension names to specify which trajectory observed dimensions
    :param rollout_dim_label: Tuple of dimension names to specify which trajectory rollout dimensions
    should be logged.
    :param show_prediction_distributions:
    :param show_prediction_rollout_mae:
    :return: None
    """
    assert isinstance(obs_dim_label, tuple)
    assert (
        len(obs_dim_label) == prediction_metric.mean.shape[-1]
    ), f"Dimension label length {len(obs_dim_label)} does not match prediction metric shape {prediction_metric.mean.shape[-1]}"
    assert (
        len(rollout_dim_label) == prediction_metric.pred_obs.shape[-1]
    ), f"Dimension label length {len(rollout_dim_label)} does not match prediction metric shape {prediction_metric.pred_obs.shape[-1]}"

    if show_prediction_distributions:
        for dim in range(len(obs_dim_label)):
            assert isinstance(obs_dim_label[dim], str)

            # .... Display trajectory prediction distribution .........................................

            # (Critical) ToDo: this ↓↓ can't represent multi-modal distribution, it should take a torch distribution object instead ⚠️
            if show_prediction_distributions:
                tensorboard_writer.add_sequential_probability_distribution(
                    tag=(
                        f"{label_pre}Prediction distributions ("
                        f"{obs_dim_label[dim]})/epoch "
                        f"{current_epoch}"
                    ),
                    trajectory_means=prediction_metric.mean[..., dim],
                    trajectory_stds=prediction_metric.std[..., dim],
                    step_size=trajectory_record_step_size,
                )

            if is_model_ensemble:
                # .... Display trajectory prediction epistemic uncertainty ............................
                tensorboard_writer.add_sequential_scalar_values(
                    tag=(
                        f"{label_pre}Prediction epistemic uncertainty "
                        f"({obs_dim_label[dim]})/epoch {current_epoch}"
                    ),
                    scalar_value_array=prediction_metric.std_epi[..., dim],
                    step_size=trajectory_record_step_size,
                )

    if show_prediction_rollout_mae:
        for dim in range(len(rollout_dim_label)):
            assert isinstance(rollout_dim_label[dim], str)

            tensorboard_writer.add_sequential_scalar_values(
                tag=(
                    f"{label_pre}Prediction Rollout in world frame "
                    f"({rollout_dim_label[dim]})/epoch {current_epoch}"
                ),
                scalar_value_array=prediction_metric.pred_obs[..., dim],
                step_size=trajectory_record_step_size,
            )

            # .... Display trajectory prediction MAE ..............................................
            # RLRP-723: the MAE is a cumulative quantity that depends on the
            # trajectory length, so annotate the tag accordingly.
            tensorboard_writer.add_sequential_scalar_values(
                tag=(
                    f"{label_pre}Prediction Rollout MAE "
                    f"(pose {rollout_dim_label[dim]}, len(T) based)"
                    f"/epoch {current_epoch}"
                ),
                scalar_value_array=prediction_metric.mae[..., dim],
                step_size=trajectory_record_step_size,
            )
    return None


def setup_trainer_lr_scheduler_callback(
    cfg: omegaconf.DictConfig,
    cfg_training: omegaconf.DictConfig,
    model_trainer: models.ModelTrainer,
    tensorboard_writer: Optional[OnlineTensorboardWritter],
) -> Callable:
    debug_mode = cfg.get("debug_mode", False)

    schedulers_dict = {}
    cfg_lr_schedule = omegaconf.OmegaConf.select(
        cfg_training, "trainer_lr_schedule", default=None
    )

    # RLRP-727 (B1/B2): Law-U opt-in. When `UDER.uder_lr_schedule.law_u_enable` is true, the ERLL
    # loop owns the per-pass LR envelope g(p) and RESTARTS this inner (per-epoch) schedule at every
    # ERLL pass boundary (SGDR-like warm restart) so the realized LR is the clean product
    # `lr(p, e) = base · g(p) · h(e)` with `h` restarting each pass (h(0)=1). To restart, we rebuild
    # the inner schedulers from the optimizer's CURRENT lr (which the loop has just set to
    # `base · g(p)`); the loop triggers the rebuild via the `model_trainer._law_u_reset_inner_lr`
    # handle installed below. Default (false) = the historical single continuous inner schedule.
    law_u_enable = bool(
        omegaconf.OmegaConf.select(
            cfg, "UDER.uder_lr_schedule.law_u_enable", default=False
        )
    )

    def _resolve_config_epoch_horizon() -> float:
        # Shared fallback epoch-horizon resolution from the static config (used by both the cosine
        # `T_max` and the RLRP-770 auto-`gamma` paths when no realized per-pass budget is stashed).
        if (
            is_cfg_key_exist(cfg, "UDER.num_epochs_train_model")
            and cfg.UDER.num_epochs_train_model is not None
        ) and (
            is_cfg_key_exist(cfg, "UDER.final_max_num_epochs_train_model")
            and cfg.UDER.final_max_num_epochs_train_model is not None
        ):
            _t_max = (
                cfg.UDER.num_epochs_train_model
                + cfg.UDER.final_max_num_epochs_train_model
            )
        elif (
            is_cfg_key_exist(cfg, "UDER.final_max_num_epochs_train_model")
            and cfg.UDER.final_max_num_epochs_train_model is None
        ):
            # Unbounded (patience-driven) budget. Under Law-U a non-finite `T_max` is degenerate, so
            # fall back to the finite final-pass budget when available (B2 guard).
            _final = omegaconf.OmegaConf.select(
                cfg, "UDER.final_max_num_epochs_train_model", default=None
            )
            _t_max = math.inf if not law_u_enable else (_final if _final else 100)
        elif (
            is_cfg_key_exist(cfg, "UDER.final_max_num_epochs_train_model")
            and cfg.UDER.final_max_num_epochs_train_model is not None
        ):
            _t_max = cfg.UDER.final_max_num_epochs_train_model
        elif (
            is_cfg_key_exist(cfg, "num_epochs_train_model")
            and cfg.num_epochs_train_model is not None
        ):
            _t_max = cfg_training.get("num_epochs_train_model")
        else:
            raise ValueError(
                "Cosine annealing learning rate require num_epochs_train_model, UDER.num_epochs_train_model or UDER.final_max_num_epochs_train_model to be specified in cfg"
            )
        return _t_max

    def _resolve_cosine_t_max(cfg_cosine_annealing_lr) -> float:
        # RLRP-727 (B2): under Law-U each pass restarts, so size `T_max` from the realized per-pass
        # budget the loop stashes on the trainer (`_law_u_pass_num_epochs`) instead of the
        # cumulative config horizon (which the persistent counter could overrun when
        # `num_epochs_train_model` is null/patience-driven).
        if law_u_enable:
            _pass_budget = getattr(model_trainer, "_law_u_pass_num_epochs", None)
            if _pass_budget is not None and _pass_budget > 0:
                return _pass_budget

        if cfg_cosine_annealing_lr.get("T_max", None) is not None:
            return cfg_cosine_annealing_lr.get("T_max")

        return _resolve_config_epoch_horizon()

    def _resolve_exp_gamma(cfg_exponential_lr) -> float:
        # RLRP-770 (option 3): optionally AUTO-SIZE the `ExponentialLR` decay `gamma` from the
        # realized per-pass epoch budget so the LR always lands on a target end-of-pass fraction
        # `end_fraction` of the pass restart height. This generalizes a hand-tuned fixed `gamma`
        # (which under-decays on long passes -> long frozen tail, and over-decays on short ones):
        # under Law-U the pass budget varies (`_law_u_pass_num_epochs`), so a single `gamma` cannot
        # be right for every pass. Opt-in: set `exponential_lr.end_fraction` in (0, 1). When unset,
        # the explicit `gamma` is used unchanged (historical behaviour).
        end_fraction = cfg_exponential_lr.get("end_fraction", None)
        if end_fraction is None:
            return cfg_exponential_lr.gamma

        if not (0.0 < float(end_fraction) < 1.0):
            raise ValueError(
                "trainer_lr_schedule.exponential_lr.end_fraction must be in the open interval "
                f"(0, 1), got {end_fraction}."
            )

        # Per-pass epoch budget: prefer the realized Law-U per-pass budget the ERLL loop stashes on
        # the trainer; otherwise fall back to the static config horizon.
        _pass_budget = None
        if law_u_enable:
            _pass_budget = getattr(model_trainer, "_law_u_pass_num_epochs", None)
        if _pass_budget is None or _pass_budget <= 0:
            _pass_budget = _resolve_config_epoch_horizon()

        # Degenerate single-epoch (or non-finite/patience-unbounded) pass: no decay length to size,
        # so keep the LR flat within the pass (gamma == 1.0) rather than guessing.
        if _pass_budget is None or not math.isfinite(_pass_budget) or _pass_budget <= 1:
            return 1.0

        # `ExponentialLR` scales the LR by `gamma` once per epoch AFTER epoch 0, so over an
        # N-epoch pass the LR at the last epoch is `restart_height · gamma**(N-1)`. Solve for the
        # `gamma` that makes that equal `end_fraction · restart_height`.
        return float(end_fraction) ** (1.0 / (float(_pass_budget) - 1.0))

    def _build_schedulers(include_warmup: bool = True) -> None:
        # (Re)build the inner per-epoch schedulers into `schedulers_dict` from scratch, capturing
        # the optimizer's CURRENT per-group lr as the base (torch schedulers snapshot `base_lrs` at
        # construction). Called once at setup and — under Law-U — again at each ERLL pass boundary.
        #
        # RLRP-727: the `trainer_lr_schedule.warmup` stage is a ONE-TIME global optimizer
        # stabilization, NOT a per-pass concept. Under Law-U the inner schedule is RESTARTED at
        # every ERLL pass boundary (SGDR-like warm restart); rebuilding the warmup there would
        # re-run it every pass (visible as a repeated warmup plateau on the trainer LR plot). We
        # therefore only build the warmup stage on the ERLL initialization pass
        # (`include_warmup=True`); subsequent per-pass restarts (`include_warmup=False`) skip it and
        # anneal straight from the pass restart height.
        schedulers_dict.clear()
        schedulers_dict["current_epoch"] = 0
        schedulers_dict["warmup_steps"] = 0
        post_warmup_schedulers = []

        cfg_warmup = omegaconf.OmegaConf.select(cfg_lr_schedule, "warmup", default=None)
        # Warmup "disabled" knob: `steps <= 0` skips building the warmup `ConstantLR` entirely, so the
        # post-warmup (e.g. exponential) chain is stepped from epoch 0 — behaviourally identical to
        # `warmup: null`. This is NOT the same as `{steps:1,factor:1.0}` nor even `{steps:0,factor:1.0}`
        # if the scheduler were still built: merely HAVING a warmup `ConstantLR` consumes one epoch
        # (the `current_epoch <= warmup_steps` branch always fires once at epoch 0), which delays the
        # decay onset by one extra epoch. Guarding on `steps > 0` makes `{steps:0}` a real disable that
        # is expressible in a Hydra `choice(...)` sweep (unlike `null`, which the grammar rejects).
        if cfg_warmup is not None and include_warmup and cfg_warmup.steps > 0:
            schedulers_dict["warmup_steps"] = cfg_warmup.steps
            schedulers_dict["warmup"] = torch.optim.lr_scheduler.ConstantLR(
                model_trainer.optimizer,
                factor=cfg_warmup.factor,
                total_iters=cfg_warmup.steps,
            )

        cfg_exponential_lr = omegaconf.OmegaConf.select(
            cfg_lr_schedule, "exponential_lr", default=None
        )
        if cfg_exponential_lr is not None:
            post_warmup_schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    model_trainer.optimizer,
                    gamma=_resolve_exp_gamma(cfg_exponential_lr),
                )
            )

        cfg_cosine_annealing_lr = omegaconf.OmegaConf.select(
            cfg_lr_schedule, "cosine_annealing_lr", default=None
        )
        if cfg_cosine_annealing_lr is not None:
            post_warmup_schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    model_trainer.optimizer,
                    T_max=_resolve_cosine_t_max(cfg_cosine_annealing_lr),
                    # `eta_min` is optional in the config (e.g. E3 comments it out, keeping only
                    # `T_max`). Read it defensively with torch's own default (0) so an absent key
                    # falls back gracefully instead of raising an OmegaConf struct-mode error.
                    eta_min=cfg_cosine_annealing_lr.get("eta_min", 0),
                )
            )

        cfg_lambda_lr = omegaconf.OmegaConf.select(
            cfg_lr_schedule, "lambda_lr", default=None
        )
        if cfg_lambda_lr is not None:
            post_warmup_schedulers.append(
                torch.optim.lr_scheduler.LambdaLR(
                    model_trainer.optimizer,
                    lr_lambda=hydra.utils.instantiate(cfg_lambda_lr),
                )
            )

        cfg_cyclic_lr = omegaconf.OmegaConf.select(
            cfg_lr_schedule, "cyclic_lr", default=None
        )
        if cfg_cyclic_lr is not None:
            post_warmup_schedulers.append(
                torch.optim.lr_scheduler.CyclicLR(
                    model_trainer.optimizer,
                    base_lr=cfg_cyclic_lr.base_lr,
                    max_lr=cfg_cyclic_lr.max_lr,
                    step_size_up=cfg_cyclic_lr.step_size_up,
                    mode=cfg_cyclic_lr.mode,
                    gamma=cfg_cyclic_lr.gamma,
                )
            )

        if post_warmup_schedulers:
            schedulers_dict["chained"] = torch.optim.lr_scheduler.ChainedScheduler(
                post_warmup_schedulers,
                optimizer=model_trainer.optimizer,
            )

    if cfg_lr_schedule is not None:
        _build_schedulers()

        if law_u_enable:
            # Expose a per-pass restart handle to the ERLL loop (Law-U). The loop sets each param
            # group's lr to `base · g(p)` at pass start, then calls this to re-snapshot the inner
            # schedule's `base_lrs` from that value and reset the intra-pass epoch counter. The loop
            # passes `include_warmup` so the one-time `trainer_lr_schedule.warmup` is applied only on
            # the ERLL initialization pass, never re-run on later per-pass restarts (RLRP-727).
            model_trainer._law_u_reset_inner_lr = _build_schedulers

    def lr_scheduler_callback(
        model, train_iteration, epoch, total_avg_loss, eval_score, best_val_score
    ) -> None:
        """Called by `mbrl.models.ModelTrainer.train(...)` method at the end of each epoch."""
        if cfg_lr_schedule is not None and schedulers_dict:
            current_epoch = schedulers_dict["current_epoch"]
            warmup_steps = schedulers_dict["warmup_steps"]

            # Log current LR (RLRP-727 B4: per-param-group. With the RLRP-722 two-timescale split
            # the optimizer carries >1 param group (slow body / fast heads); logging only
            # `param_groups[0]` hid the head LR. Emit one scalar per group, tagged by index, plus a
            # `body`/`head` alias for the canonical 2-group case, while keeping the legacy
            # group-0 scalar tag for backward-compatible dashboards.)
            if tensorboard_writer is not None:
                param_groups = model_trainer.optimizer.param_groups
                current_lr = param_groups[0]["lr"]

                tensorboard_writer.add_scalar_per_epoch_monitoring(
                    "Replay buffer optimization scheduling/LR schedule (trainer)",
                    current_lr,
                )

                if len(param_groups) > 1:
                    _group_alias = (
                        ["body", "head"]
                        if len(param_groups) == 2
                        else [f"group_{i}" for i in range(len(param_groups))]
                    )
                    for _i, _pg in enumerate(param_groups):
                        tensorboard_writer.add_scalar_per_epoch_monitoring(
                            "Replay buffer optimization scheduling/LR schedule (trainer) "
                            f"[{_group_alias[_i]}]",
                            _pg["lr"],
                        )

            # Step appropriate scheduler
            if current_epoch <= warmup_steps and "warmup" in schedulers_dict:
                if current_epoch == warmup_steps:
                    print(
                        f"\n{consol_msg_universal_one_liner('learning rate scheduler warmup phase DONE', print_it=False)}"
                    )
                schedulers_dict["warmup"].step()
            elif "chained" in schedulers_dict:
                schedulers_dict["chained"].step()

            schedulers_dict["current_epoch"] += 1

        return None

    return lr_scheduler_callback


def setup_train_time_domain_randomization_scheduler_callback(
    cfg: omegaconf.DictConfig,
    cfg_training: omegaconf.DictConfig,
    model_trainer: models.ModelTrainer,
    tensorboard_writer: Optional[OnlineTensorboardWritter],
) -> Callable:
    """
    RLRP-707 (§4ter) — steps the train-time domain randomization noise-scale scheduler once per
    model training epoch and records the current noise scale to tensorboard.

    This is a strict no-op unless the inner model exposes
    ``set_train_time_domain_randomization_epoch`` (i.e. a
    ``TrainTimeDomainRandomizationMultiStepMLP`` subclass). Noise recording is additionally gated on
    the randomizer being enabled and a tensorboard writer being present, so disabled (default) runs
    emit nothing new.

    :param cfg: Configuration dictionary containing general settings.
    :param cfg_training: Configuration dictionary specific to the model training cfg.
    :param model_trainer: Instance of ModelTrainer responsible for training the model.
    :param tensorboard_writer: Optional tensorboard writer for logging training metrics.
    :return: Callable function to be used as an epoch callback.
    """
    # RLRP-707 fix: the ``epoch`` argument forwarded by ``mbrl.models.ModelTrainer`` is
    # ``trainer.current_epoch``, which restarts at 0 on every ERLL outer epoch (a fresh
    # PyTorch-Lightning ``Trainer`` is created per ``train()`` call). Driving the noise-scale
    # scheduler with that per-call value keeps it stuck in the warmup phase forever. We therefore
    # keep our own cumulative (global) epoch counter that persists across ``train()`` calls,
    # mirroring ``setup_trainer_lr_scheduler_callback`` (which tracks its own ``current_epoch``).
    scheduler_state = {"global_epoch": 0}

    def train_time_domain_randomization_scheduler_callback(
        model, train_iteration, epoch, total_avg_loss, eval_score, best_val_score
    ) -> None:
        """Called by `mbrl.models.ModelTrainer.train(...)` method at the end of each epoch."""
        # `model.model` is the inner MultiStepMLP (OneDTransitionRewardModel(V2) wrapper); the
        # getattr fallback keeps this safe for non-wrapped models.
        inner = getattr(model, "model", model)
        if not hasattr(inner, "set_train_time_domain_randomization_epoch"):
            return None

        # Step the train-time domain randomization noise-scale scheduler with the cumulative global
        # epoch (not the per-`train()`-call `epoch`, which resets each ERLL epoch), then advance it.
        applied_multiplier = inner.set_train_time_domain_randomization_epoch(
            scheduler_state["global_epoch"]
        )
        scheduler_state["global_epoch"] += 1

        # Record the current noise (only when enabled + writer present).
        if tensorboard_writer is not None and getattr(
            inner, "train_time_domain_randomization_enabled", False
        ):
            noise_scale = (
                inner.get_train_time_domain_randomization_noise_scale().detach().cpu()
            )
            # Per-epoch scalars (grouped tag `<GROUP>/<NAME>`, scheme of
            # `setup_trainer_lr_scheduler_callback`).
            tensorboard_writer.add_scalar_per_epoch_monitoring(
                "Train-time domain randomization/noise scale multiplier",
                float(applied_multiplier),
            )
            tensorboard_writer.add_scalar_per_epoch_monitoring(
                "Train-time domain randomization/noise scale (mean)",
                noise_scale.mean().item(),
            )
            # # Per-feature breakdown for this epoch (sequential-scalar scheme).
            # tensorboard_writer.add_sequential_scalar_values(
            #     tag=(
            #         "Train-time domain randomization/"
            #         f"noise scale per feature (epoch {epoch})"
            #     ),
            #     scalar_value_array=noise_scale.numpy(),
            # )

        return None

    return train_time_domain_randomization_scheduler_callback


def setup_erll_tensorboard_manual_lr_monitor_callback(
    cfg: omegaconf.DictConfig,
    cfg_training: omegaconf.DictConfig,
    model_trainer: models.ModelTrainer,
    tensorboard_writer: Optional[OnlineTensorboardWritter],
) -> Callable:
    """uncertainty_driven_experience_replay callback for monitoring learning rate value per
    epoch"""

    def fetch_current_optimiser_lr():
        return model_trainer.optimizer.param_groups[0]["lr"]

    def tensorboard_manual_lr_monitor_callback(
        model, train_iteration, epoch, total_avg_loss, eval_score, best_val_score
    ) -> None:
        """Called by `uncertainty_driven_experience_replay` method at the end of each dataset
        optimization epoch."""
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar_per_epoch_monitoring(
                "Replay buffer optimization scheduling/lr", fetch_current_optimiser_lr()
            )

            # Deploy-path training stabilization (RLRP-722, feature (2) two-timescale LR): when the
            # optimizer carries the body/head split (2 param groups), log each group's LR so the
            # slow-body / fast-head schedule is visible. Single-group (feature OFF) emits nothing
            # new here.
            param_groups = model_trainer.optimizer.param_groups
            if len(param_groups) > 1:
                _group_names = ("body", "head")
                for group_index, param_group in enumerate(param_groups):
                    group_label = (
                        _group_names[group_index]
                        if group_index < len(_group_names)
                        else f"group{group_index}"
                    )
                    tensorboard_writer.add_scalar_per_epoch_monitoring(
                        f"Replay buffer optimization scheduling/lr ({group_label})",
                        param_group["lr"],
                    )

        return None

    return tensorboard_manual_lr_monitor_callback
