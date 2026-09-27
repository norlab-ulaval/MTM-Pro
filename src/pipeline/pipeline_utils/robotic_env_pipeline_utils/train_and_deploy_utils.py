# coding=utf-8
from functools import partial
from typing import Any, Optional, Union

import mbrl
import numpy as np
import omegaconf
from omegaconf import DictConfig

from algorithm.experience_replay_learning_loop.core.model_testing_utils.rollout_and_stats_collection import (
    Deploy_Rollout_PostProcessing,
)
from pipeline.pipeline_utils.general.setup import (
    get_model_description,
    setup_ms_model_train_callback,
    setup_multistep_step_model_and_trainer,
    setup_source_multi_step_replay_buffer,
    setup_tensorboard_writer,
)
from algorithm.experience_replay_learning_loop.core.data_classes import (
    MsTestTimeDeployResult,
)
from pipeline.pipeline_utils.general.train_and_deploy_utils import (
    execute_ms_model_test_time_rollouts,
    load_primary_model_in_place,
    resolve_resume_experiments,
    run_epoch_checkpoint_rollouts_if_enabled,
    run_resume_driver,
    train_system_dynamic_model,
)
from tools.console_tools.message import consol_msg_universal_one_liner
from tools.mbrl_lib_tools.resume_utils import STAGE_TRAIN, ResumeExperiment
from pipeline.pipeline_utils.robotic_env_pipeline_utils.robotic_trajectory_dataclass import (
    obs_2_dict,
)
from pipeline.pipeline_utils.general.testtime_trajectory_dataclass_util import (
    TestTrajectoryEntry,
)
from pipeline.pipeline_utils.robotic_env_pipeline_utils.utils import (
    compute_position_from_velocity_and_attitude,
    report_angular_velocity_clamp,
    resolve_max_angular_velocity_from_cfg,
    resolve_training_frame_from_cfg,
)
from tools.dna_dev_tools.dn_CI_tools import show_plot_unless_CI_server_runned
from tools.dna_dev_tools.dn_pytest_tools import is_pytest_run
from tools.f110_gym_env_tools.plot_tensorboard import OnlineTensorboardWritter
from tools.hydra_apps_tools.r2s2r_apps_utils import R2S2RPipelineHydraApp
from tools.multistep_tools.data_buffer_processor.multistep_data_buffer_processor_arbitrary_dim import (
    MultistepDataBufferProcessorArbitraryDimension,
)
from tools.r2s_motion_model_container_tools.container import R2SMotionModelContainer
from trajectory_container_tools.dataclasses import (
    TestMotionTrajectoryDataclass,
)
import torch


class Robotic3DDeployRolloutPostprocessing(Deploy_Rollout_PostProcessing):

    @staticmethod
    def _align_predicted_obs_to_timestamp_grid(
        pred_obs: Union[np.ndarray, torch.Tensor],
        test_env: TestMotionTrajectoryDataclass,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Right-shift the predicted-obs sequence onto the ground-truth time grid.

        Wrong-timestep-index fix (RLRP-707). The forward dynamics model predicts
        the observation for time ``t+1``, so ``pred_obs[i]`` is the prediction
        *for* ``test_env.timestamps[i+1]``. The position integrator
        (:func:`compute_position_from_velocity_and_attitude`) assumes
        ``vel[i]`` / ``quaternions[i]`` are aligned to ``timestamps[i]`` with
        ``positions[0] == initial_position``. Without realignment the integrator
        would apply the velocity/attitude predicted *for* ``t+1`` at integration
        step ``t`` (and silently drop ``pred_obs[0]``, whose increment is zero),
        a persistent one-step index error that biases — i.e. drifts — the
        reconstructed trajectory once the rollout leaves the ground-truth-feed
        warmup window and runs on compounded predictions.

        We therefore shift the sequence right by one and seed index 0 with the
        true initial observation (``test_env.observations[0]``); its velocity is
        unused because the first integration increment is zero, but its attitude
        keeps index 0 consistent with ``initial_position``.

        :param pred_obs: ``(N, obs_dim)`` predicted observations (model predicts
            the state at ``t+1`` at row ``i``).
        :param test_env: The test trajectory (provides the initial observation).
        :return: ``(N, obs_dim)`` observations aligned so row ``i`` corresponds to
            ``timestamps[i]``.
        """
        init_obs = test_env.observations[0]
        if isinstance(pred_obs, torch.Tensor):
            init_row = torch.as_tensor(
                init_obs, dtype=pred_obs.dtype, device=pred_obs.device
            ).reshape(pred_obs[:1].shape)
            return torch.cat([init_row, pred_obs[:-1]], dim=0)
        init_row = (
            np.asarray(init_obs)
            .reshape(pred_obs[:1].shape)
            .astype(pred_obs.dtype, copy=False)
        )
        return np.concatenate([init_row, pred_obs[:-1]], axis=0)

    def deploy_adapter(
        self,
        pred_obs: Union[np.ndarray, torch.Tensor],
        test_env: TestMotionTrajectoryDataclass,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:

        # (CRITICAL) ToDo: validate timsetamps source (original vs post-processed int64) (ref task RLRP-503)
        #       see 'robotic_data_to_test_motion_trajectory_dataclass(...)'

        # Align the predicted obs (model predicts state @ t+1) onto the GT
        # timestamp grid before integrating (wrong-timestep-index fix, RLRP-707).
        pred_obs = self._align_predicted_obs_to_timestamp_grid(pred_obs, test_env)

        # Torch-first: pass tensors directly through the pipeline.
        # compute_position_from_velocity_and_attitude supports both numpy and torch.
        integrator_inputs = obs_2_dict(
            pred_obs, test_env, timestamps=test_env.timestamps, cfg=self._cfg
        )
        # RLRP-791: `quaternions_are_gt` is *metadata* (which source fed the
        # attitude channel), not an integrator argument -> pop it before the `**`
        # splat and surface it so the provenance of a reconstruction is auditable.
        quaternions_are_gt = integrator_inputs.pop("quaternions_are_gt")
        # RLRP-753: `orientation_source` is the three-valued successor of the
        # `quaternions_are_gt` bool (the gravity block is a third source). Both
        # are metadata -> popped before the `**` splat.
        orientation_source = integrator_inputs.pop("orientation_source", None)
        self._report_orientation_source(
            quaternions_are_gt, test_env, orientation_source=orientation_source
        )
        # RLRP-753 `D-deploy`: a gravity run reconstructs the attitude from its
        # own prediction, but the reconstruction ANCHOR is still the ground-truth
        # row 0 (same contract as `initial_position`). Without it there is nothing
        # to seed roll/pitch/yaw with -> fail loud rather than silently anchoring
        # on identity.
        if orientation_source == "gravity_prediction" and test_env.orientation_gt is None:
            raise ValueError(
                "a gravity-configured run needs `test_env.orientation_gt[0]` as the "
                "reconstruction anchor (roll/pitch come from the predicted "
                "`gravity` block and yaw from the predicted angular velocity, but "
                "both are relative to a start attitude), got "
                "`test_env.orientation_gt=None`."
            )
        max_angular_velocity = resolve_max_angular_velocity_from_cfg(self._cfg)

        # Deploy-time physical-plausibility check (owned by the deploy layer).
        # Surfaces physically-implausible *model predictions* (|omega| above the
        # obs-spec cap) explicitly and unconditionally, rather than relying on the
        # integrator's silent numerical-overflow rail. This is the deploy-level
        # decision/report; the integrator clamp remains only a defensive rail.
        self._report_predicted_angular_velocity_plausibility(
            integrator_inputs.get("angular_velocity"),
            max_angular_velocity,
            test_env,
        )

        # RLRP-758 deploy-boundary guard (Rec §7.1): the frame the trajectory's
        # velocity channels carry MUST equal the model's training frame, else the
        # reconstruction silently double-/never-converts. Fail loud.
        #
        # NOTE: `test_env` here is a TCT `TestMotionTrajectoryDataclass`, which
        # exposes a SINGLE `velocity_frame`. At ingestion both RLRC channels are
        # converted to the training frame and stamped equal, so this one field is
        # the frame of BOTH channels post-conversion.
        training_frame = resolve_training_frame_from_cfg(self._cfg)
        # RLRP-758 (merit review): fail-loud SAFETY guard -> a real ``raise`` (NOT
        # an ``assert``, which ``python -O`` strips, re-opening the silent
        # double-/never-convert hazard this guard exists to prevent).
        if test_env.velocity_frame != training_frame:
            raise ValueError(
                f"frame mismatch: trajectory velocity_frame="
                f"{test_env.velocity_frame!r} != ms_model.training_frame="
                f"{training_frame!r}"
            )

        pred_world_pose, _ = compute_position_from_velocity_and_attitude(
            # RLRP-791: the reconstruction is anchored on the trajectory GROUND
            # TRUTH, by contract. Direct field indexing (not `test_env[0].<field>`)
            # avoids the full-trajectory `deepcopy` performed by the TCT
            # `__getitem__` on every rollout, and lets the optional
            # `orientation_gt` be `None`-guarded explicitly.
            initial_position=test_env.pose_gt[0],
            initial_orientation=(
                None if test_env.orientation_gt is None else test_env.orientation_gt[0]
            ),
            linear_velocity_frame=test_env.velocity_frame,
            angular_velocity_frame=test_env.velocity_frame,
            interpolate_quaternions=self._cfg.deploy.trajectory_computation.attitude_propagation,
            quaternion_blend_weight=self._cfg.deploy.trajectory_computation.quaternion_ang_vel_blend,
            max_angular_velocity=max_angular_velocity,
            integration_scheme=self._cfg.deploy.trajectory_computation.get(
                "integration_scheme", "forward_euler"
            ),
            **integrator_inputs,
        )
        return pred_world_pose

    @staticmethod
    def _report_orientation_source(
        quaternions_are_gt: bool,
        test_env: TestMotionTrajectoryDataclass,
        orientation_source: Optional[str] = None,
    ) -> None:
        """Log which source fed the attitude channel of this reconstruction.

        RLRP-791: ``obs_2_dict`` picks the attitude either from the *predicted*
        observation block or from ``test_env.orientation_gt``, driven solely by
        ``cfg.environment.obs_dims``. That choice changes the body->world velocity
        rotation of the whole rollout, so it is reported here (always, regardless
        of ``debug``) instead of staying an invisible configuration side effect.

        Permanent deploy-path contract. Introduced by action `A3` of the RLRC
        Explicit attitude source in deployer trajectory integration `.junie` plan
        (`feat_explicit_attitude_source_deploy_integration_plan_RLRP-791_20260829.md`).

        RLRP-753 adds a **third** label: a 9-D gravity observation space carries no
        quaternion at all, and the attitude is reconstructed from the predicted
        ``gravity`` block (roll/pitch) plus the predicted angular velocity (yaw).
        That regime must be distinguishable from both of the RLRP-791 ones,
        otherwise a gravity run looks like a quaternion run in the log.

        :param quaternions_are_gt: ``True`` when the trajectory ground-truth
            attitude was used, ``False`` when the model prediction was.
        :param test_env: the test trajectory (used to label the report).
        :param orientation_source: the explicit ``obs_2_dict`` label
            (``"ground_truth"`` | ``"attitude_prediction"`` | ``"gravity_prediction"``).
            ``None`` falls back to the RLRP-791 two-valued behaviour.
        :return: None.
        """
        traj_label = getattr(test_env, "trajectory_name", None) or getattr(
            test_env, "feature_name", ""
        )
        if orientation_source == "gravity_prediction":
            source = "model prediction (gravity block + omega)"
        else:
            source = (
                "ground truth (test_env.orientation_gt)"
                if quaternions_are_gt
                else "model prediction (obs attitude block)"
            )
        print(
            f"Robotic3DDeployRolloutPostprocessing.deploy_adapter "
            f"[{traj_label}] attitude source: {source} "
            f"| start attitude anchor: test_env.orientation_gt[0]"
        )

    @staticmethod
    def _report_predicted_angular_velocity_plausibility(
        angular_velocity: Optional[Union[np.ndarray, torch.Tensor]],
        max_angular_velocity: float,
        test_env: TestMotionTrajectoryDataclass,
    ) -> Optional[dict]:
        """Surface physically-implausible predicted body rates at the deploy layer.

        Computes ``|omega| = norm(angular_vels.{x,y,z})`` from the *predicted*
        observations and reports (always, regardless of ``debug``) any sample that
        exceeds the obs-spec saturation cap. This makes a bad multi-step prediction
        visible at deploy time instead of being silently rescaled inside the
        attitude integrator.

        :param angular_velocity: predicted angular-velocity sequence ``(N, 3)`` or
            ``None`` when the obs space carries no angular velocity.
        :param max_angular_velocity: the obs-spec saturation cap (rad/s).
        :param test_env: the test trajectory (used to label the report).
        :return: the violation summary dict, or ``None`` when nothing exceeds the cap.
        """
        if angular_velocity is None:
            return None

        if isinstance(angular_velocity, torch.Tensor):
            omega_magnitudes = torch.norm(angular_velocity, dim=-1)
        else:
            omega_magnitudes = np.linalg.norm(np.asarray(angular_velocity), axis=-1)

        traj_label = getattr(test_env, "trajectory_name", None) or getattr(
            test_env, "label", ""
        )
        context = (
            "Robotic3DDeployRolloutPostprocessing.deploy_adapter "
            "(predicted angular-velocity plausibility check"
            f"{f' [{traj_label}]' if traj_label else ''})"
        )
        return report_angular_velocity_clamp(
            omega_magnitudes, max_angular_velocity, context=context
        )


def robotic_3d_env_ms_model_train_and_deploy(
    cfg: DictConfig,
    ms_replay_buffer,  # ReplayBuffer (legacy) | ERLLDataSource (RLRP-824 ``data_manager: dataloader``)
    motion_model_container: R2SMotionModelContainer,
    ms_model_description: str,
    target_InD_rollouts: list[TestTrajectoryEntry],
    target_OOD_rollouts: list[TestTrajectoryEntry],
    ms_trainer: mbrl.models.ModelTrainer,
    val_env_trjs: list[TestMotionTrajectoryDataclass],
    exp_dir_relative_path: str | Any,
    headless: bool,
    torch_rng,
    pipeline_app: Optional[R2S2RPipelineHydraApp] = None,
) -> tuple[
    float, MsTestTimeDeployResult, Union[OnlineTensorboardWritter, None, Any], float
]:
    """Train the multi-step model and deploy it on the InD/OoD test target rollouts.

    RLRP-839: mirror of the math_env driver. When ``training_common.resume_from_checkpoint`` is set,
    the path is resolved into audited :class:`ResumeExperiment` s and every incomplete candidate is
    processed sequentially (``train`` => RLRP-824 ERLL resume in the new cwd; ``deploy`` => training
    skipped, missing rollouts completed IN PLACE under the candidate root). ``null`` => legacy path.
    """
    resume_experiments = resolve_resume_experiments(
        cfg, motion_model_container, ms_model_description, target_InD_rollouts, target_OOD_rollouts
    )

    def _once(resume_experiment: Optional[ResumeExperiment], container, trainer):
        return _robotic_3d_env_ms_model_train_and_deploy_once(
            cfg,
            ms_replay_buffer,
            container,
            ms_model_description,
            target_InD_rollouts,
            target_OOD_rollouts,
            trainer,
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
        # A fresh model/trainer per candidate (FR7); the caller-built pair serves the first one.
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
        return float("nan"), MsTestTimeDeployResult(), ms_tensorboard_writer, float("nan")
    return result


def _robotic_3d_env_ms_model_train_and_deploy_once(
    cfg: DictConfig,
    ms_replay_buffer,
    motion_model_container: R2SMotionModelContainer,
    ms_model_description: str,
    target_InD_rollouts: list[TestTrajectoryEntry],
    target_OOD_rollouts: list[TestTrajectoryEntry],
    ms_trainer: mbrl.models.ModelTrainer,
    val_env_trjs: list[TestMotionTrajectoryDataclass],
    exp_dir_relative_path: str | Any,
    headless: bool,
    torch_rng,
    pipeline_app: Optional[R2S2RPipelineHydraApp],
    resume_experiment: Optional[ResumeExperiment],
) -> tuple[
    float, MsTestTimeDeployResult, Union[OnlineTensorboardWritter, None, Any], float
]:
    """One train -> deploy (-> epoch rollouts) run, or one in-place deploy-stage resume (RLRP-839).

    See the math_env twin ``_math_env_ms_model_train_and_deploy_once`` for the stage semantics.
    """

    state_space_label = cfg.environment.data.label

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
        return float("nan"), ms_deploy_result, ms_tensorboard_writer, float("nan")

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
        ms_replay_buffer,
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
        ms_tensorboard_writer,
        best_pred_val_loss,
    )
