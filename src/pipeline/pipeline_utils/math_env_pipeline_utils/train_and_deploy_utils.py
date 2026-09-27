# coding=utf-8
from functools import partial
from typing import Any, Optional, Union
import omegaconf
from omegaconf import DictConfig

from algorithm.experience_replay_learning_loop.core.model_testing_utils.rollout_and_stats_collection import (
    Deploy_Rollout_PostProcessing,
)
from algorithm.experience_replay_learning_loop.core.training_callback import (
    fetch_cfg_pipeline_tensorboard_key_value,
    setup_erll_epoch_pre_training_callback_aggregator,
    setup_gradient_monitoring_callback,
    setup_tensorboard_arbitrary_dimension_trajectory_prediction_monitor_callback,
    setup_trainer_epoch_callback_aggregator,
    setup_trainer_lr_scheduler_callback,
)
from algorithm.motion_model.env2sim_components.math_gym_toy_s2s.math_env_training_callback import (
    setup_math_env_uder_buffer_sampling_plotter_callback,
)
from pipeline.pipeline_utils.general.setup import (
    get_model_description,
    setup_ms_model_train_callback,
    setup_multistep_step_model_and_trainer,
    setup_single_step_model,
    setup_source_multi_step_replay_buffer,
    setup_source_single_step_replay_buffer,
    setup_tensorboard_writer,
)
from algorithm.experience_replay_learning_loop.core.data_classes import (
    MsTestTimeDeployResult,
)
from pipeline.pipeline_utils.general.train_and_deploy_utils import (
    execute_ms_model_test_time_rollouts,
    execute_ss_legacy_model_test_time_rollouts,
    load_primary_model_in_place,
    resolve_resume_experiments,
    run_epoch_checkpoint_rollouts_if_enabled,
    run_resume_driver,
    train_system_dynamic_model,
)
from tools.mbrl_lib_tools.resume_utils import STAGE_TRAIN, ResumeExperiment
from pipeline.pipeline_utils.general.testtime_trajectory_dataclass_util import TestTrajectoryEntry
from tools.dna_dev_tools.dn_CI_tools import show_plot_unless_CI_server_runned
from tools.dna_dev_tools.dn_pytest_tools import is_pytest_run
from tools.f110_gym_env_tools.plot_tensorboard import OnlineTensorboardWritter
from tools.hydra_apps_tools.r2s2r_apps_utils import R2S2RPipelineHydraApp
from tools.math_tools.space_conversion_tools.coordinate_to_velocity import (
    convert_dt_state_derivatives_to_state_coordinate,
)
from tools.console_tools.message import consol_msg_universal_one_liner
from tools.multistep_tools.window_dataset.pipeline_utils import (
    build_store_from_ss_replay_buffers,
    build_window_data_source,
    build_window_dataset,
    describe_window_path,
    is_dataloader_data_manager,
    move_store_to_training_device,
    stamp_environment_shapes_from_data,
    stamp_original_dataset_size,
)
from trajectory_container_tools.dataclasses import (
    TestMotionTrajectoryDataclass,
)
import torch
import numpy as np


class MathEnvDeployRolloutPostprocessing(Deploy_Rollout_PostProcessing):

    @staticmethod
    def _align_predicted_obs_to_timestamp_grid(
        pred_obs: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Right-shift the predicted-obs sequence onto the integrator's time grid (RLRP-723 M1).

        The forward dynamics model predicts the observation for time ``t+1``, so ``pred_obs[i]``
        is the velocity/displacement *arriving at* ``t_{i+1}``. The integrator
        :func:`convert_dt_state_derivatives_to_state_coordinate` instead expects ``arr[i]`` to be
        the displacement arriving at ``t_i`` (row 0 is a placeholder, overwritten by the initial
        coordinate, and ``coord[i] = init + sum_{j=1..i} arr[j]``). Without this realignment the
        reconstructed pose is left-shifted by one step (``coord[i] = pose[i+1] - Delta_pose_0``
        for a perfect model) — a separate off-by-one from the metric-level RLRP-707/F1 fix that
        the robotic ``Robotic3DDeployRolloutPostprocessing`` already received.

        We shift the sequence right by one: prepend one placeholder row (its value is irrelevant
        because the integrator overwrites row 0 with the seed) and drop the last row to keep the
        length ``N``. Then ``coord[i] = init + sum_{k=0..i-1} pred_obs[k] = pose[i]`` for a
        perfect model.

        :param pred_obs: ``(N, obs_dim)`` (or ``(N,)``) predicted velocities/displacements.
        :return: ``(N, ...)`` displacements aligned so row ``i`` corresponds to ``t_i``.
        """
        if isinstance(pred_obs, torch.Tensor):
            return torch.cat([torch.zeros_like(pred_obs[:1]), pred_obs[:-1]], dim=0)
        pred_obs = np.asarray(pred_obs)
        return np.concatenate([np.zeros_like(pred_obs[:1]), pred_obs[:-1]], axis=0)

    def deploy_adapter(
        self,
        pred_obs: Union[np.ndarray, torch.Tensor],
        test_env: TestMotionTrajectoryDataclass,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        if test_env.obs_are_velocity:
            # RLRP-723 M1: align the predicted obs (model predicts state @ t+1) onto the
            # integrator's time grid before integrating, mirroring the robotic RLRP-707 fix.
            pred_obs = self._align_predicted_obs_to_timestamp_grid(pred_obs)
            pred_world_pose = convert_dt_state_derivatives_to_state_coordinate(
                pred_obs,
                initiale_coordinates=tuple(test_env[0].pose),
            )
        else:
            pred_world_pose = pred_obs
        return pred_world_pose


def math_env_ms_model_train_and_deploy(
    cfg: DictConfig,
    ss_full_time_space_replay_buffers: list[Any],
    state_space_label: list[Any],
    target_InD_rollouts: list[TestTrajectoryEntry],
    target_OOD_rollouts: list[TestTrajectoryEntry],
    val_env_trjs: list[TestMotionTrajectoryDataclass],
    exp_dir_relative_path: str | Any,
    headless: bool,
    torch_rng,
    pipeline_app: Optional[R2S2RPipelineHydraApp] = None,
) -> tuple[float, MsTestTimeDeployResult, Any, Union[OnlineTensorboardWritter, None, Any], float]:
    """Train the multi-step model and deploy it on the InD/OoD test target rollouts.

    Updated by phase 4 of the math_env multi test-trajectory `.junie`
    plan (``feature_math_env_multi_test_trajectory_plan_20260517.md``)
    to accept ``list[TestTrajectoryEntry]`` directly (mirror of the
    robotic env counterpart), removing the legacy single-env wrap.

    RLRP-839: when ``training_common.resume_from_checkpoint`` is set, the path is resolved into
    audited :class:`ResumeExperiment` s (:func:`resolve_resume_experiments`) and every incomplete candidate is
    processed sequentially by :func:`run_resume_driver`: stage ``train`` => the RLRP-824 ERLL resume
    (new cwd); stage ``deploy`` => training skipped, primary model loaded from the candidate root and
    ONLY the missing main / per-epoch rollouts completed IN PLACE under that root. ``null`` => the
    legacy single train -> deploy path, byte-identical.
    """

    # .... Dataset ............................................................................
    # RLRP-824 (FR1): ``pipeline.data_manager`` selects the training data path. ``replay-buffer``
    # (default) is the legacy materialized multistep buffer (bit-exact); ``dataloader`` builds an
    # in-memory ``SingleStepTrajectoryStore`` straight from the generated single-step rollouts (no
    # file I/O, plan KD2) and composes the ``(H, F)`` windows lazily per batch.
    use_dataloader = is_dataloader_data_manager(cfg)
    if use_dataloader:
        stamp_environment_shapes_from_data(
            cfg,
            ss_full_time_space_replay_buffers[0].obs_shape[-1],
            ss_full_time_space_replay_buffers[0].action_shape[-1],
        )
        single_step_store = build_store_from_ss_replay_buffers(cfg, ss_full_time_space_replay_buffers)
        ms_data_buffer_processor_3D = None
        ms_replay_buffer_explorable_region = None
    else:
        single_step_store = None
        (
            ms_data_buffer_processor_3D,
            ms_replay_buffer_explorable_region,
        ) = setup_source_multi_step_replay_buffer(cfg, ss_full_time_space_replay_buffers)

    # .... Setup model ........................................................................
    motion_model_container, ms_trainer = setup_multistep_step_model_and_trainer(cfg)
    if use_dataloader:
        # Store on the training device first (``pipeline.dataloader.store_device``, default
        # ``auto`` -> CUDA model device) so windows are composed on-device (no per-batch H2D copy).
        single_step_store = move_store_to_training_device(
            cfg, single_step_store, motion_model_container.dynamics_model.device
        )
        window_dataset = build_window_dataset(
            cfg, single_step_store, output_window_len=motion_model_container.output_window_len
        )
        motion_model_container.validate_with_window_dataset(window_dataset)
        stamp_original_dataset_size(cfg, len(window_dataset))
        consol_msg_universal_one_liner(describe_window_path(single_step_store, window_dataset))
        erll_source = build_window_data_source(
            cfg, window_dataset, device=motion_model_container.dynamics_model.device
        )
    else:
        motion_model_container.validate_with_ms_data_buffer_processor(
            ms_data_buffer_processor_3D
        )
        erll_source = ms_replay_buffer_explorable_region
    # ms_model_name = motion_model_container.dynamics_model.model.__class__.__name__
    ms_model_description = get_model_description(cfg.ms_model, motion_model_container)

    # .... RLRP-839 resume experiment resolution ....................................................
    resume_experiments = resolve_resume_experiments(
        cfg, motion_model_container, ms_model_description, target_InD_rollouts, target_OOD_rollouts
    )

    def _once(resume_experiment: Optional[ResumeExperiment], container, trainer):
        return _math_env_ms_model_train_and_deploy_once(
            cfg,
            container,
            trainer,
            erll_source,
            ms_model_description,
            state_space_label,
            target_InD_rollouts,
            target_OOD_rollouts,
            val_env_trjs,
            exp_dir_relative_path,
            headless,
            torch_rng,
            pipeline_app,
            resume_experiment,
        )

    if resume_experiments is None:
        return _once(None, motion_model_container, ms_trainer)

    _first_container: list = [(motion_model_container, ms_trainer)]

    def _run_once(resume_experiment: ResumeExperiment):
        # A fresh model/trainer per candidate (FR7); the probe built above serves the first one.
        if _first_container:
            container, trainer = _first_container.pop()
        else:
            container, trainer = setup_multistep_step_model_and_trainer(cfg)
        return _once(resume_experiment, container, trainer)

    result = run_resume_driver(cfg, resume_experiments, _run_once)
    if result is None:
        consol_msg_universal_one_liner(
            "resume: no candidate executed (dry-run, or every candidate complete/fresh) => "
            "returning sentinel objectives."
        )
        # The caller closes the writer unconditionally outside pytest/CI: keep its contract.
        if show_plot_unless_CI_server_runned(True) and not is_pytest_run():
            ms_tensorboard_writer = setup_tensorboard_writer(cfg, ms_model_description)
        else:
            ms_tensorboard_writer = None
        return (
            float("nan"),
            MsTestTimeDeployResult(),
            ms_model_description,
            ms_tensorboard_writer,
            float("nan"),
        )
    return result


def _math_env_ms_model_train_and_deploy_once(
    cfg: DictConfig,
    motion_model_container,
    ms_trainer,
    erll_source,
    ms_model_description: str,
    state_space_label: list[Any],
    target_InD_rollouts: list[TestTrajectoryEntry],
    target_OOD_rollouts: list[TestTrajectoryEntry],
    val_env_trjs: list[TestMotionTrajectoryDataclass],
    exp_dir_relative_path: str | Any,
    headless: bool,
    torch_rng,
    pipeline_app: Optional[R2S2RPipelineHydraApp],
    resume_experiment: Optional[ResumeExperiment],
) -> tuple[float, MsTestTimeDeployResult, Any, Union[OnlineTensorboardWritter, None, Any], float]:
    """One train -> deploy (-> epoch rollouts) run, or one in-place deploy-stage resume (RLRP-839).

    ``resume_experiment is None`` or ``stage == "train"`` => the unchanged legacy sequence (for the train
    stage the ERLL picks the run root up from ``training_common.resume_from_checkpoint``, set by
    :func:`run_resume_driver`). ``stage == "deploy"`` => no ERLL is built: the primary model is
    loaded from the candidate root, the main deploy runs only when incomplete and the per-epoch
    rollouts only for the missing/partial epochs, everything written under the candidate root. The
    TensorBoard writer is not forwarded to resume-driven deploy calls (RLRP-773 R13 isolation) and
    the HPO objectives are NaN (resume mode is not an HPO mode).
    """
    if show_plot_unless_CI_server_runned(True) and not is_pytest_run():
        ms_tensorboard_writer = setup_tensorboard_writer(cfg, ms_model_description)
    else:
        ms_tensorboard_writer = None

    if resume_experiment is not None and resume_experiment.stage != STAGE_TRAIN:
        # .... RLRP-839 deploy stage, in place ..................................................
        root = resume_experiment.candidate.run_root
        load_primary_model_in_place(motion_model_container, root)
        training_time = None
        ms_deploy_result = MsTestTimeDeployResult()
        if not resume_experiment.main_deploy_complete:
            ms_deploy_result = execute_ms_model_test_time_rollouts(
                cfg,
                motion_model_container,
                ms_model_description,
                state_space_label,
                target_InD_rollouts,
                target_OOD_rollouts,
                training_time,
                None,
                exp_dir_relative_path,
                headless,
                torch_rng,
                save_base_dir=root,
            )
        else:
            consol_msg_universal_one_liner(
                f"resume: main deploy already complete under {root}, skipping."
            )
        run_epoch_checkpoint_rollouts_if_enabled(
            cfg,
            motion_model_container,
            ms_model_description,
            state_space_label,
            target_InD_rollouts,
            target_OOD_rollouts,
            training_time,
            None,
            exp_dir_relative_path,
            headless,
            torch_rng,
            save_base_dir=root,
            checkpoints_source_dir=root,
            epochs_override=list(resume_experiment.missing_epoch_rollouts),
            wipe_dirs=resume_experiment.partial_epoch_rollout_dirs,
        )
        return (
            float("nan"),
            ms_deploy_result,
            ms_model_description,
            ms_tensorboard_writer,
            float("nan"),
        )

    # .... Train ..............................................................................
    (
        ms_erll_epoch_post_training_callback,
        ms_erll_epoch_pre_training_callback,
        ms_trainer_batch_callback,
        ms_trainer_epoch_callback,
    ) = setup_ms_model_train_callback(
        cfg,
        ms_tensorboard_writer,
        motion_model_container,
        ms_trainer,
        val_env_trjs=val_env_trjs,
        test_InD_trjs=target_InD_rollouts,
        test_OoD_trjs=target_OOD_rollouts,
        state_space_label=state_space_label,
        exp_dir_relative_path=exp_dir_relative_path,
        headless=headless,
        pipeline_app=pipeline_app,
    )

    best_pred_mae_score, training_time, best_pred_val_loss = train_system_dynamic_model(
        cfg,
        cfg.ms_training,
        erll_source,
        val_env_trjs,
        ms_trainer,
        ms_model_description,
        ms_tensorboard_writer,
        ms_erll_epoch_pre_training_callback,
        ms_trainer_epoch_callback,
        ms_trainer_batch_callback,
        ms_erll_epoch_post_training_callback,
        exp_dir_relative_path,
        headless,
        torch_rng,
        motion_model_container,
    )

    # .... Deploy .............................................................................
    ms_deploy_result = execute_ms_model_test_time_rollouts(
        cfg,
        motion_model_container,
        ms_model_description,
        state_space_label,
        target_InD_rollouts,
        target_OOD_rollouts,
        training_time,
        ms_tensorboard_writer,
        exp_dir_relative_path,
        headless,
        torch_rng,
    )

    # RLRP-773 (R6/G1.1): additionally run per-epoch-checkpoint rollouts when enabled (opt-in).
    # Additive: the single deploy above stays the primary result; no-op when OFF or no
    # `epoch_checkpoints/` tree exists on disk (the ERLL save side wrote it under the exp cwd).
    run_epoch_checkpoint_rollouts_if_enabled(
        cfg,
        motion_model_container,
        ms_model_description,
        state_space_label,
        target_InD_rollouts,
        target_OOD_rollouts,
        training_time,
        ms_tensorboard_writer,
        exp_dir_relative_path,
        headless,
        torch_rng,
    )
    return (
        best_pred_mae_score,
        ms_deploy_result,
        ms_model_description,
        ms_tensorboard_writer,
        best_pred_val_loss,
    )


def math_env_legacy_ss_model_train_and_deploy(
    cfg: DictConfig,
    ss_full_time_space_replay_buffers: list[Any],
    state_space_label: list[Any],
    target_InD_rollouts: list[TestTrajectoryEntry],
    target_OOD_rollouts: list[TestTrajectoryEntry],
    val_env_trjs: list[TestMotionTrajectoryDataclass],
    exp_dir_relative_path: Union[str, Any],
    headless: bool,
    torch_rng: Any,
) -> tuple[float, MsTestTimeDeployResult, Any, Union[OnlineTensorboardWritter, None, Any], float]:
    """Train the legacy single-step model and deploy it on the InD/OoD test target rollouts.

    Updated by phase 4 of the math_env multi test-trajectory `.junie`
    plan (``feature_math_env_multi_test_trajectory_plan_20260517.md``)
    to accept ``list[TestTrajectoryEntry]`` directly.
    """
    # .... Configuration setting validation .......................................................
    if cfg.ss_model.description == "Fully deterministic MLP":
        omegaconf.OmegaConf.update(cfg, "ss_model.deterministic", True)

    if cfg.ss_model.description in ["Fully deterministic MLP", "Probabilistic MLP"]:
        omegaconf.OmegaConf.update(cfg, "ss_model.ensemble_size", 1)

    # .... Dataset ................................................................................
    ss_replay_buffer_explorable_region = setup_source_single_step_replay_buffer(
        cfg, ss_full_time_space_replay_buffers
    )

    # .... Setup model ............................................................................
    ss_1D_transition_model, ss_trainer = setup_single_step_model(cfg, torch_rng)
    # ss_model_name = ss_1D_transition_model.model.__class__.__name__
    ss_model_description = get_model_description(cfg.ss_model, ss_1D_transition_model)

    # .... Train ..................................................................................
    if show_plot_unless_CI_server_runned(True) and not is_pytest_run():
        ss_tensorboard_writer = setup_tensorboard_writer(cfg, ss_model_description)
    else:
        ss_tensorboard_writer = None

    ss_erll_epoch_pre_training_callback = setup_erll_epoch_pre_training_callback_aggregator(
        cfg,
        cfg.ms_training,
        ss_trainer,
        ss_tensorboard_writer,
        register_callbacks=[
            partial(
                setup_math_env_uder_buffer_sampling_plotter_callback,
                **{
                    "test_env": val_env_trjs[0],
                    # Pick one env
                    "state_space_label": state_space_label,
                    "exp_dir_relative_path": exp_dir_relative_path,
                    "headless": headless,
                },
            )
        ],
    )

    ss_trainer_epoch_callback = setup_trainer_epoch_callback_aggregator(
        cfg,
        cfg.ss_training,
        ss_trainer,
        ss_tensorboard_writer,
        register_callbacks=[
            setup_trainer_lr_scheduler_callback,
            setup_gradient_monitoring_callback,
            partial(
                setup_tensorboard_arbitrary_dimension_trajectory_prediction_monitor_callback,
                **{
                    "trajectories": val_env_trjs,
                    "trajectory_record_step_size": fetch_cfg_pipeline_tensorboard_key_value(
                        cfg,
                        key="trajectory_record_step_size",
                        key_value_default=100,
                    ),
                    "execute_every_n_epoch": fetch_cfg_pipeline_tensorboard_key_value(
                        cfg,
                        key="record_trajectory_pred_every_n_train_epoch",
                        key_value_default=None,
                    ),
                    "obs_dim_label": fetch_cfg_pipeline_tensorboard_key_value(
                        cfg,
                        key="obs_dim_label",
                        key_value_default=("X", "Y", "Z"),
                    ),
                    "rollout_dim_label": fetch_cfg_pipeline_tensorboard_key_value(
                        cfg,
                        key="rollout_dim_label",
                        key_value_default=("X", "Y", "Z"),
                    ),
                    "label": "Train",
                },
            ),
        ],
    )

    ss_erll_epoch_post_training_callback = setup_trainer_epoch_callback_aggregator(
        cfg,
        cfg.ms_training,
        ss_trainer,
        ss_tensorboard_writer,
        register_callbacks=[
            # setup_erll_tensorboard_manual_lr_monitor_callback,
        ],
        enable_tensorboard_call_method=False,
    )

    best_pred_mae_score, training_time, best_pred_val_loss = train_system_dynamic_model(
        cfg,
        cfg.ss_training,
        ss_replay_buffer_explorable_region,
        val_env_trjs,
        ss_trainer,
        ss_model_description,
        ss_tensorboard_writer,
        ss_erll_epoch_pre_training_callback,
        ss_trainer_epoch_callback,
        None,
        ss_erll_epoch_post_training_callback,
        exp_dir_relative_path,
        headless,
        torch_rng,
    )

    # .... Deploy .................................................................................
    ss_deploy_result = execute_ss_legacy_model_test_time_rollouts(
        cfg,
        ss_1D_transition_model,
        ss_model_description,
        state_space_label,
        target_InD_rollouts,
        target_OOD_rollouts,
        training_time,
        ss_tensorboard_writer,
        exp_dir_relative_path,
        headless,
        torch_rng,
    )
    return (
        best_pred_mae_score,
        ss_deploy_result,
        ss_model_description,
        ss_tensorboard_writer,
        best_pred_val_loss,
    )
