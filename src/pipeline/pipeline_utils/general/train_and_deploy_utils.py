# coding=utf-8
import contextlib
import os
import shutil
import time
import traceback
import warnings
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import hydra
import mbrl.models
import numpy as np
import omegaconf
import torch
from hydra.core.hydra_config import HydraConfig
from mbrl.util import ReplayBuffer
from omegaconf import DictConfig

from algorithm.experience_replay_learning_loop import (
    PartitionBasedUncertaintyDrivenERLL,
    ProgressiveBatchExperienceReplay,
    SingleGlobalLoopERLL,
)
from algorithm.experience_replay_learning_loop.core.data_classes import (
    _SENTINEL,
    MsTestTimeDeployResult,
    TestTimeRolloutPredictionMetric,
    get_metric_sub_dir,
)
from algorithm.experience_replay_learning_loop.core.erll_epoch_budget import (
    experiment_planned_total_epochs,
)
from algorithm.experience_replay_learning_loop.core.model_testing_utils import (
    compute_trajectories_prediction_error,
    multistep_model_testtime_OPENLOOP_forecast_and_collect_pred_stats,
    multistep_model_testtime_SELFFED_forecast_and_collect_pred_stats,
    multistep_model_testtime_rollout_and_collect_pred_stats,
    singlestep_model_testtime_rollout_and_collect_pred_stats,
)
from pipeline.pipeline_utils.general.plot_utils import (
    _plot_3d_rollout,
)
from tools.feature_handling_tools.env_handlers import (
    attach_feature_handler_to_deploy,
)
from tools.console_tools.message import (
    consol_msg,
    consol_msg_universal_one_liner,
    console_msg_pipeline_footer,
    console_msg_pipeline_header,
)
from tools.f110_gym_env_tools.plot_tensorboard import OnlineTensorboardWritter
from tools.general_utils import sanitize_dirname
from tools.hydra_apps_tools.hydra_utils import get_hydra_experiment_cwd
from tools.math_tools.space_conversion_tools.coordinate_to_velocity import (
    convert_dt_state_derivatives_to_state_coordinate,
)
from tools.mbrl_lib_tools import persistent_checkpoint_utils
from tools.mbrl_lib_tools import resume_utils
from tools.mbrl_lib_tools.models.one_dim_tr_model_v2 import OneDTransitionRewardModelV2
from tools.mbrl_lib_tools.models.utils import (
    compute_n_dim_obs_trajectory_probability_statistics_over_ensemble,
    is_probabilistic_model,
)
from tools.mbrl_lib_tools.setup_utils import setup_saved_model_dir
from tools.plot_tools.plot import loss_plot
from tools.plot_tools.plot_management import (
    manage_matplotlib_backend,
    manage_matplotlib_warnings,
    show_and_save_plot_helper,
)
from tools.r2s_motion_model_container_tools.container import R2SMotionModelContainer
from tools.multistep_tools.window_dataset.erll_data_source import (
    ERLLDataSource,
    validate_constant_batch_size_for_dataloader,
)
from pipeline.pipeline_utils.general.testtime_trajectory_dataclass_util import (
    TestTrajectoryEntry,
)
from trajectory_container_tools.dataclasses import (
    TestMotionTrajectoryDataclass,
    TestTrajectoryDataclass,
)


def train_system_dynamic_model(
    cfg: omegaconf.DictConfig,
    cfg_training: omegaconf.DictConfig,
    source_replay_buffer: Union[ReplayBuffer, ERLLDataSource],
    test_trajectories: Union[
        list[TestTrajectoryDataclass], list[TestMotionTrajectoryDataclass]
    ],
    model_trainer: mbrl.models.ModelTrainer,
    model_name: str,
    tensorboard_writer: Optional[OnlineTensorboardWritter],
    erll_epoch_pre_training_callback: Optional[Callable],
    model_trainer_epoch_callback: Optional[Callable],
    model_trainer_batch_callback: Optional[Callable],
    erll_epoch_post_training_callback: Optional[Callable],
    exp_dir_relative_path: str,
    headless: bool,
    torch_rng,
    motion_model_container=None,
) -> Tuple[float, float, float]:
    console_msg_pipeline_header(f"Begin {model_name} model training", ".")

    # RLRP-727: 3-way ERLL loop selector (replaces the legacy
    # `pipeline.replay_buffer_exploration_policy_enable` boolean). The algorithm config block is
    # always exposed under the `UDER` cfg key (even when the concrete loop is PBER or the single
    # global loop), so the selector `loop_kind` lives there.
    #   - `uder`               → PartitionBasedUncertaintyDrivenERLL (partition/uncertainty driven)
    #   - `pber`               → ProgressiveBatchExperienceReplay (progressive batch schedule)
    #   - `single_global_loop` → SingleGlobalLoopERLL (one fused pass, ERLL disabled / ablation)
    _erll_loop_kind_dispatch = {
        "uder": PartitionBasedUncertaintyDrivenERLL,
        "pber": ProgressiveBatchExperienceReplay,
        "single_global_loop": SingleGlobalLoopERLL,
    }
    loop_kind = omegaconf.OmegaConf.select(cfg, "UDER.loop_kind", default=None)
    if loop_kind not in _erll_loop_kind_dispatch:
        raise ValueError(
            f"Unknown / missing `UDER.loop_kind={loop_kind!r}`. "
            f"Expected one of {sorted(_erll_loop_kind_dispatch)}."
        )
    if isinstance(source_replay_buffer, ERLLDataSource) and not source_replay_buffer.is_replay_buffer:
        # RLRP-824 FR12 / operator decision 7: the lazy window DataLoader path (`pipeline.data_manager:
        # dataloader`) builds ONE loader pair for the whole run, hence a constant batch size only;
        # `uder` needs replay-buffer partition surgery and is rejected outright.
        validate_constant_batch_size_for_dataloader(cfg, loop_kind)
    experience_replay_learning_loop = _erll_loop_kind_dispatch[loop_kind](
        cfg,
        cfg_training,
        model_trainer,
        source_replay_buffer,
        test_trajectories,
        tensorboard_writer,
        erll_epoch_pre_training_callback,
        model_trainer_epoch_callback,
        model_trainer_batch_callback,
        erll_epoch_post_training_callback,
        torch_rng,
        motion_model_container,
        replay_buffer_learning_loop_cfg_key="UDER",
    )

    (
        train_losses,
        val_losses,
        training_time,
        best_pred_mae_score,
        best_pred_val_loss,
    ) = experience_replay_learning_loop.execute()

    # .... Plot loss ..............................................................................
    with warnings.catch_warnings():
        manage_matplotlib_warnings()
        manage_matplotlib_backend(cfg.pipeline.plot.show_loss_plot, headless)

        loss_fig, _ = loss_plot(
            train_losses,
            val_losses,
            f"{model_name} Ensemble Model Training/Validation Loss",
            training_time,
            (cfg.pipeline.plot.figsize[0], 10),
        )

        show_and_save_plot_helper(
            loss_fig,
            exp_dir_relative_path,
            f"{sanitize_dirname(model_name)}_ensemble_model_training_validation_loss",
            headless,
            cfg.pipeline.plot.show_loss_plot,
            cfg.pipeline.plot.save_dpi,
        )

    console_msg_pipeline_footer(
        f"{model_name} model training done in {training_time:.2f} seconds", "."
    )
    return best_pred_mae_score, training_time, best_pred_val_loss


def resolve_stats_unit_contract(dynamics_model: Any) -> Tuple[Optional[str], str]:
    """Resolve ``(normalizer_type, stats_space)`` of a run's prediction statistics.

    RLRP-761 ``P4.2``. The unit space of ``mean`` / ``std`` / ``std_epi`` is known
    only here, at the producer: since ``P1`` the deploy producers denormalize the
    statistics channel alongside ``next_obs``, so the artifact is ``physical`` for
    **every** normalizer type. Recording it explicitly (rather than letting a
    downstream consumer infer it from the normalizer type) is what allows the
    ``P4.4`` aggregation guard to distinguish a post-``P1`` artifact from a
    pre-``P1`` one, which is *not* derivable from the config alone.

    Args:
        dynamics_model: the transition-model wrapper the rollout was produced with.

    Returns:
        ``(normalizer_type, stats_space)``; ``stats_space`` is ``"unknown"`` when
        the wrapper does not expose the contract (never the case in production,
        but test doubles do exist).
    """
    normalizer_type = getattr(dynamics_model, "normalizer_type", None)
    if not hasattr(dynamics_model, "denormalize_predicted_logvar"):
        # A wrapper predating P1 (or a stub): refuse to claim a space.
        return normalizer_type, "unknown"
    return normalizer_type, "physical"


def _per_step_feature_mae(pred_mae) -> np.ndarray:
    """Per-step MAE sequence (one value per scored step) in physical units (meters).

    The feature (coordinate) axis is reduced by ``mean`` so each per-step term is a *true*
    MAE in meters (paper-grade reporting, RLRP-723 F2-(ii)) rather than the feature-summed
    value (inflated by the number of coordinate components). The time axis is **preserved**
    to keep the cumulative (compounded-drift) semantics and to allow a common-horizon
    truncation across InD/OoD before summing (RLRP-723 F2-(i)). Works for torch/numpy.

    :param pred_mae: per-trajectory absolute error of shape ``(steps, features)`` (or a
        already feature-reduced ``(steps,)`` sequence).
    :return: a ``(steps,)`` numpy array of per-step MAE (meters).
    """
    if isinstance(pred_mae, torch.Tensor):
        arr = pred_mae.detach().cpu().numpy()
    else:
        arr = np.asarray(pred_mae)
    arr = np.atleast_1d(arr)
    if arr.ndim == 1:
        # No explicit feature axis: each entry is treated as an already feature-reduced step.
        return arr.astype(float, copy=False)
    return arr.mean(axis=-1).astype(float, copy=False)


def _reduce_pred_mae_to_scalar(pred_mae) -> float:
    """Reduce a per-trajectory MAE tensor/ndarray of shape ``(steps, features)`` to a single
    *cumulative* MAE scalar (meters): **sum over the trajectory steps** (cumulative /
    compounded-drift semantics, RLRP-723 F2) and **mean over the feature axis** so the scalar
    is a true MAE in physical units (meters) instead of the feature-summed value. Works for
    both torch tensors and numpy arrays.
    """
    return float(_per_step_feature_mae(pred_mae).sum())


def _common_horizon(*sequence_groups: "list[np.ndarray]") -> int:
    """Shortest scored horizon across all provided per-step sequences (e.g. InD ∪ OoD).

    Used to make the cumulative MAE comparable across trajectory sets of differing length
    without destroying the cumulative (compounded-drift) semantics (RLRP-723 F2-(i),
    recommendation 1: compare at a common horizon ``H = min`` rather than normalizing by
    length).
    """
    lengths = [len(seq) for group in sequence_groups for seq in group]
    return min(lengths) if lengths else 0


def _cumulative_mae_over_common_horizon(
    per_step_sequences: "list[np.ndarray]", horizon: int
) -> float:
    """Aggregate per-trajectory per-step MAE sequences into one cumulative MAE (meters).

    Each sequence is truncated to a **common horizon** ``[:horizon]`` (RLRP-723 F2-(i)),
    summed over time (cumulative / compounded-drift MAE), then averaged across trajectories.
    Returns ``_SENTINEL`` when there is nothing to aggregate.
    """
    if not per_step_sequences or horizon <= 0:
        return _SENTINEL
    return float(
        sum(float(seq[:horizon].sum()) for seq in per_step_sequences)
        / len(per_step_sequences)
    )


def _mean_or_sentinel(values: list[float]) -> float:
    """Mean over the accumulated per-trajectory scalars, or ``_SENTINEL`` when empty."""
    return float(sum(values) / len(values)) if values else _SENTINEL


def _obs_space_gt_and_error(
    gt_observations, pred_obs_mean
) -> tuple["np.ndarray | None", "np.ndarray | None"]:
    """RLRP-761 ``S8.16`` (option A) — obs-space GT trajectory + per-step obs error.

    The adverse-event ``M2`` slice ranks timesteps by the terrain-vibration /
    weight-transfer triplet magnitude and slices an obs-space error — both of which
    live in OBSERVATION space, not the 3-D world-pose space that ``target`` / ``mae``
    record. This recovers exactly those two arrays from data already in scope at the
    producer (no dataset reload, no re-integration):

    - ``obs_target`` = the ground-truth obs trajectory (``test_env.observations``);
    - ``obs_mae`` = the per-timestep feature-mean absolute error between the physical
      prediction mean (``ms_3d_pred_stats_mean``, physical since ``P1``) and the GT.

    Both are aligned by dropping the trivially-zero seeded index 0 (the same
    convention as the pose MAE), so ``obs_target.shape[0] == obs_mae.shape[0]`` and
    the consumer's time-axis match (`_as_time_major`) holds. Returns ``(None, None)``
    when the two arrays cannot be aligned (defensive; the consumer then skips loudly).
    """
    if gt_observations is None or pred_obs_mean is None:
        return None, None
    gt = np.asarray(
        (
            gt_observations.detach().cpu().numpy()
            if hasattr(gt_observations, "detach")
            else gt_observations
        ),
        dtype=np.float64,
    )
    pred = np.asarray(
        (
            pred_obs_mean.detach().cpu().numpy()
            if hasattr(pred_obs_mean, "detach")
            else pred_obs_mean
        ),
        dtype=np.float64,
    )
    if gt.ndim == 1:
        gt = gt[:, None]
    if pred.ndim == 1:
        pred = pred[:, None]
    if gt.ndim != 2 or pred.ndim != 2 or gt.shape[1] != pred.shape[1]:
        return None, None
    n = min(gt.shape[0], pred.shape[0])
    if n < 2:
        return None, None
    # Drop seeded index 0 (RLRP-723 F1 convention): a perfect rollout scores 0 and the
    # seeded row is never credited.
    gt_aligned = gt[1:n]
    pred_aligned = pred[1:n]
    obs_target = gt_aligned
    obs_mae = np.abs(pred_aligned - gt_aligned).mean(axis=-1)
    return obs_target, obs_mae


def _assemble_and_persist_prediction_stats(
    cfg: DictConfig,
    *,
    model,
    model_description: str,
    metric_mean,
    stats_mean,
    stats_ale_epi_std,
    stats_epi_std,
    pred_world_pose,
    test_trajectories: TestTrajectoryEntry,
    pred_mae,
    pred_l2_norm,
    target_is_ood: bool,
    is_compounded_pred_rollout: bool,
    is_pr_rollout: bool,
    training_time: float,
    rollout_time: float,
    trajectory_short_name: str,
    save_base_dir: str | None,
    obs_target=None,
    obs_mae=None,
    rollout_steps: int | None = None,
    benchmark=None,
) -> tuple:
    """RLRP-761 ``N6`` — the SINGLE post-rollout statistics-assembly block.

    Previously duplicated verbatim between the multistep and the legacy single-step
    deploy paths, which is what made the RLRP-761 denormalization defect exist twice
    and cost ``P1`` / ``P2`` / ``P4`` / ``P5`` a two-site tax each. The SS path is the
    less exercised of the two, so a one-sided edit produced a defect no A/B cell could
    surface (every cell is multistep).

    Two MS/SS asymmetries are PRESERVED deliberately (they are behavioural, not
    incidental) and are therefore explicit parameters rather than derived:

    - ``metric_mean`` — the archived :class:`TestTimeRolloutPredictionMetric` stores the
      per-step prediction STATISTICS mean on the MS path, but the integrated POSE on the
      legacy SS path;
    - ``stats_mean`` — the run-stats tuple always carries the STATISTICS mean, which on
      the SS path is therefore a DIFFERENT array from ``metric_mean``.

    Returns the run-stats tuple, whose element order is the consumer contract:
    ``(trajectories, mean, ale+epi std, epi std, is_ood, is_compounded, stats_space,
    pose)``. The last two are APPENDED (never inserted) so a legacy 6-tuple still
    unpacks at the consumer (``P3``).
    """
    # RLRP-761 P4.2 — stamp the unit space at the PRODUCER, where it is known, rather
    # than leaving a consumer to infer it.
    stats_normalizer_type, stats_space = resolve_stats_unit_contract(model)

    test_time_metric = TestTimeRolloutPredictionMetric(
        mean=metric_mean,
        std=stats_ale_epi_std,
        std_epi=stats_epi_std,
        mae=pred_mae,
        l2_norm=pred_l2_norm,
        target=test_trajectories.pose,
        target_is_ood=target_is_ood,
        compounded_predictions_score=is_compounded_pred_rollout,
        training_wall_clock_time=training_time,
        rollout_wall_clock_time=rollout_time,
        normalizer_type=stats_normalizer_type,
        stats_space=stats_space,
        # RLRP-761 S8.16 (option A) — obs-space GT + obs-space per-step error, so the
        # adverse-event ``M2`` slice is scorable on robotic-3D (where ``target`` / ``mae``
        # are the 3-D world pose). ``None`` on the legacy SS path (kwargs defaulted).
        obs_target=obs_target,
        obs_mae=obs_mae,
        # RLRP-785 A3 — carry the step count (always, cheap metadata) and the optional
        # benchmark metric set (only when instrumentation was ON) so the reporting chain can
        # express a *rate* (Hz == fps) instead of a total-seconds stopwatch (defect D3).
        # ``rollout_steps`` is the divisor for a harness-level rate; ``benchmark`` holds the
        # per-level/regime/timing-pass measurements produced by ``step_timer.report(...)``.
        rollout_steps=rollout_steps,
        benchmark=benchmark,
    )
    test_time_metric.set_name_and_description_from_model(model.model)

    save_dir = save_base_dir if save_base_dir else get_hydra_experiment_cwd(cfg)
    test_time_metric.save(
        os.path.join(
            save_dir,
            "testtime_rollouts",
            trajectory_short_name,
            f"TTRPM_{sanitize_dirname(model_description)}_PR_rollout{is_pr_rollout}",
        )
    )

    return (
        test_trajectories,
        # RLRP-761 P8 — channel choice, now EXPLICIT. These run-stats feed the
        # UNCERTAINTY surfaces (mean +/- aleatoric/epistemic bands), so they take the
        # prediction STATISTICS channel. Every ERROR metric (MAE, C-MAE, L2) is scored
        # from the integrated POSE channel instead. The two are no longer in different
        # unit spaces: since P1 the statistics channel is denormalized at the producer
        # alongside ``next_obs`` (invariant pinned by
        # ``test_prediction_stats_space_invariant.py``). NOTE they remain DISTINCT
        # quantities — the pose is the integrated trajectory, the statistics are the
        # per-step prediction — so this is a documented choice, not an equivalence.
        stats_mean,
        stats_ale_epi_std,
        stats_epi_std,
        target_is_ood,
        is_compounded_pred_rollout,
        # RLRP-761 P3 (interim guard) — the UNIT SPACE of the three statistics arrays
        # above, carried WITH them. The math-env plot consumer scores them against a
        # PHYSICAL target, so it must be able to refuse a normalized channel instead of
        # silently producing a plot inflated by 1/sigma_target.
        stats_space,
        # RLRP-761 P3.1 — the integrated POSE channel, carried alongside the statistics
        # one so the math-env ERROR surfaces (MAE / SNR) can be scored from the SAME
        # channel every other error metric in the repo uses. The statistics channel
        # stays for the UNCERTAINTY surfaces, which is what it is.
        pred_world_pose,
    )


_ASYMMETRIC_WINDOW_FORECAST_STEPS = (
    "test_rollout_openloop_forecast",
    "test_rollout_selffed_forecast",
)


def assert_asymmetric_window_deploy_conditioning(cfg: DictConfig, model) -> None:
    """FR14 (d) deploy-side gate (RLRP-824 Step 6b, operator decision 9): an asymmetric
    ``horizon_len > history_len`` MS->MS model may only be evaluated **plan-conditioned**.

    On the ``W = F > H`` output window the composed action columns ARE the plan
    ``a_{t+1..t+F-1}``: there is no history echo of the right length to fall back on, so
    ``AbstractMS2MSForecast.forecast`` refuses a missing plan. Rather than let that surface as a
    mid-rollout ``ValueError`` after hours of training, this gate fails fast BEFORE the deploy
    stages run: for every ENABLED forecast step (``pipeline.steps.test_rollout_openloop_forecast``
    / ``test_rollout_selffed_forecast``) it requires
    ``pipeline.<step>.future_action_conditioning == "plan"`` and
    ``pipeline.<step>.feed_future_action_plan`` truthy (default ``true``). The compounded
    single-step rollout needs no knob: it always feeds the ground-truth plan on ``W > H``
    (``_asymmetric_window_step_plan``). ``last_action_hold`` / ``obs_only`` /
    ``feed_future_action_plan: false`` on ``F > H`` are deferred to a follow-up task (RLRP-824
    *Deferred follow-ups*). Every ``F <= H`` model passes untouched (all three modes stay legal).

    :param cfg: the resolved experiment configuration.
    :param model: the trained dynamics model (``motion_model_container.dynamics_model.model``).
    :raises ValueError: on a non-plan conditioning of an asymmetric-window model.
    """
    if not bool(getattr(model, "asymmetric_output_window", False)):
        return None
    offending = []
    for step in _ASYMMETRIC_WINDOW_FORECAST_STEPS:
        if not omegaconf.OmegaConf.select(cfg, f"pipeline.steps.{step}", default=False):
            continue
        conditioning = omegaconf.OmegaConf.select(
            cfg, f"pipeline.{step}.future_action_conditioning", default="plan"
        )
        feed_plan = omegaconf.OmegaConf.select(
            cfg, f"pipeline.{step}.feed_future_action_plan", default=True
        )
        if str(conditioning) != "plan" or not bool(feed_plan):
            offending.append(
                f"pipeline.{step}.future_action_conditioning={conditioning!r}, "
                f"pipeline.{step}.feed_future_action_plan={feed_plan!r}"
            )
    if offending:
        raise ValueError(
            f"{type(model).__name__} runs the asymmetric output window (horizon_len="
            f"{model.horizon_len} > history_len={model.history_len}, RLRP-824): the deploy-side "
            "test-time forecast stages must be plan-conditioned -- set "
            "`future_action_conditioning: plan` and `feed_future_action_plan: true` for "
            + "; ".join(offending)
            + ". Non-plan action conditioning (last_action_hold / obs_only / "
            "feed_future_action_plan: false) on F > H is deferred to a follow-up task "
            "(RLRP-824 FR14 (d), plan 'Deferred follow-ups')."
        )
    return None


def _maybe_execute_openloop_forecast(
    cfg: DictConfig,
    motion_model_container: R2SMotionModelContainer,
    ms_model_description,
    test_rollout_env_types: list[tuple[TestTrajectoryEntry, bool]],
    torch_rng,
    save_base_dir: str | None,
) -> None:
    """Opt-in open-loop, control-conditioned per-horizon forecast eval step (RLRP-728 S3).

    Gated by ``cfg.pipeline.steps.test_rollout_openloop_forecast`` (**default OFF** via
    :func:`omegaconf.OmegaConf.select`, so launchers that predate this key still resolve). When ON,
    for every selected InD/OOD test trajectory it drives
    :func:`multistep_model_testtime_OPENLOOP_forecast_and_collect_pred_stats` and records the
    **cumulative per-horizon MAE in target-env observation space** (the "same basis" metric for the
    E2E-TCN / M3 / TBM baselines). The single-step deploy path is untouched.

    The conditioning knob ``cfg.pipeline.test_rollout_openloop_forecast.future_action_conditioning``
    (``plan`` | ``last_action_hold`` | ``obs_only``, default ``plan``) selects whether the horizon's
    true future-action plan is fed. Failures (e.g. the model is not an ``MS->MS`` MultiStepMLP, or
    the target-env adapter is the identity default) surface as the collector's own fail-fast
    assertions -- this step is only meaningful for the control-conditioned MS->MS baselines.

    :param test_rollout_env_types: the already-gated (InD/OOD) list of ``(entry, target_is_ood)``
        produced by :func:`setup_test_rollouts_grid`.
    :param save_base_dir: base directory for artifacts; falls back to the Hydra experiment cwd.
    """
    if not omegaconf.OmegaConf.select(
        cfg, "pipeline.steps.test_rollout_openloop_forecast", default=False
    ):
        return None

    future_action_conditioning = omegaconf.OmegaConf.select(
        cfg,
        "pipeline.test_rollout_openloop_forecast.future_action_conditioning",
        default="plan",
    )

    console_msg_pipeline_header(
        f"Begin {ms_model_description} open-loop per-horizon forecast eval "
        f"(future_action_conditioning={future_action_conditioning})",
        ".",
    )

    _save_dir = save_base_dir if save_base_dir else get_hydra_experiment_cwd(cfg)
    for entry, target_is_ood in test_rollout_env_types:
        per_horizon_mae = (
            multistep_model_testtime_OPENLOOP_forecast_and_collect_pred_stats(
                cfg,
                entry.env,
                motion_model_container,
                torch_rng,
                future_action_conditioning=future_action_conditioning,
            )
        )
        per_horizon_mae_np = per_horizon_mae.detach().cpu().numpy()

        consol_msg_universal_one_liner(
            f"[{entry.short_name}] {'OoD' if target_is_ood else 'InD'} open-loop per-horizon "
            f"target-env MAE (H={per_horizon_mae_np.shape[0]}): {np.array2string(per_horizon_mae_np, precision=5)}"
        )

        _openloop_dir = os.path.join(_save_dir, "testtime_rollouts", entry.short_name)
        os.makedirs(_openloop_dir, exist_ok=True)
        np.save(
            os.path.join(
                _openloop_dir,
                f"OPENLOOP_forecast_per_horizon_mae_{sanitize_dirname(ms_model_description)}"
                f"_{'OoD' if target_is_ood else 'InD'}.npy",
            ),
            per_horizon_mae_np,
        )

    console_msg_pipeline_footer(
        f"{ms_model_description} open-loop per-horizon forecast eval done", "."
    )
    return None


def _maybe_execute_selffed_forecast(
    cfg: DictConfig,
    motion_model_container: R2SMotionModelContainer,
    ms_model_description,
    test_rollout_env_types: list[tuple[TestTrajectoryEntry, bool]],
    torch_rng,
    save_base_dir: str | None,
) -> None:
    """Opt-in compounded **self-fed** MS-forecaster rollout eval step (RLRP-760 S3).

    Gated by ``cfg.pipeline.steps.test_rollout_selffed_forecast`` (**default OFF** via
    :func:`omegaconf.OmegaConf.select`, so launchers that predate this key still resolve). When ON,
    for every selected InD/OOD test trajectory it drives
    :func:`multistep_model_testtime_SELFFED_forecast_and_collect_pred_stats` and records the
    **per-global-step MAE in target-env observation space** -- the deployment-resilience metric for
    the MS->MS forecast baselines (E2E-TCN / M3 / TBM) and the MTM-Pro family.

    Unlike the RLRP-728 open-loop step (which re-seeds every anchor from ground truth and therefore
    measures one-shot forecast *accuracy*), this step feeds the model's own ``F``-step forecast back
    into the history window after window, so the reported drift is directly comparable to the
    compounded single-step / MTM-Pro curves. The single-step deploy path is untouched.

    Knobs (all optional, ``OmegaConf.select`` defaults keep old launchers working):

    - ``cfg.pipeline.test_rollout_selffed_forecast.future_action_conditioning``
      (``plan`` | ``last_action_hold`` | ``obs_only``, default ``plan``);
    - ``cfg.pipeline.test_rollout_selffed_forecast.warmup_steps`` -- when unset, falls back to
      ``cfg.deploy.target_experiment.ground_truth_feed_warmup_steps`` (plan §9-Q4) so the warm-up
      semantics line up with the compounded single-step metric.

    :param test_rollout_env_types: the already-gated (InD/OOD) list of ``(entry, target_is_ood)``
        produced by :func:`setup_test_rollouts_grid`.
    :param save_base_dir: base directory for artifacts; falls back to the Hydra experiment cwd.
    """
    if not omegaconf.OmegaConf.select(
        cfg, "pipeline.steps.test_rollout_selffed_forecast", default=False
    ):
        return None

    future_action_conditioning = omegaconf.OmegaConf.select(
        cfg,
        "pipeline.test_rollout_selffed_forecast.future_action_conditioning",
        default="plan",
    )
    warmup_steps = omegaconf.OmegaConf.select(
        cfg, "pipeline.test_rollout_selffed_forecast.warmup_steps", default=None
    )
    if warmup_steps is None:
        # Plan §9-Q4: reuse the compounded single-step warm-up verbatim.
        warmup_steps = omegaconf.OmegaConf.select(
            cfg, "deploy.target_experiment.ground_truth_feed_warmup_steps", default=0
        )

    console_msg_pipeline_header(
        f"Begin {ms_model_description} compounded self-fed forecast eval "
        f"(future_action_conditioning={future_action_conditioning}, "
        f"warmup_steps={warmup_steps})",
        ".",
    )

    _save_dir = save_base_dir if save_base_dir else get_hydra_experiment_cwd(cfg)
    for entry, target_is_ood in test_rollout_env_types:
        per_step_mae = multistep_model_testtime_SELFFED_forecast_and_collect_pred_stats(
            cfg,
            entry.env,
            motion_model_container,
            torch_rng,
            ground_truth_feed_warmup_steps=int(warmup_steps),
            future_action_conditioning=future_action_conditioning,
        )
        per_step_mae_np = per_step_mae.detach().cpu().numpy()

        consol_msg_universal_one_liner(
            f"[{entry.short_name}] {'OoD' if target_is_ood else 'InD'} compounded self-fed "
            f"per-global-step target-env MAE (steps={per_step_mae_np.shape[0]}, "
            f"cumulative={float(per_step_mae_np.sum()):.5f})"
        )

        _selffed_dir = os.path.join(_save_dir, "testtime_rollouts", entry.short_name)
        os.makedirs(_selffed_dir, exist_ok=True)
        np.save(
            os.path.join(
                _selffed_dir,
                f"SELFFED_forecast_per_step_mae_{sanitize_dirname(ms_model_description)}"
                f"_{'OoD' if target_is_ood else 'InD'}.npy",
            ),
            per_step_mae_np,
        )

    console_msg_pipeline_footer(
        f"{ms_model_description} compounded self-fed forecast eval done", "."
    )
    return None


def execute_ms_model_test_time_rollouts(
    cfg: DictConfig,
    motion_model_container: R2SMotionModelContainer,
    ms_model_description,
    state_space_label,
    target_InD_rollouts: list[TestTrajectoryEntry],
    target_OOD_rollouts: list[TestTrajectoryEntry],
    training_time: float,
    ms_tensorboard_writer: OnlineTensorboardWritter | None | Any,
    exp_dir_relative_path: str | Any,
    headless: bool,
    torch_rng,
    save_base_dir: str | None = None,
    epoch_rollout_label: str | None = None,
) -> MsTestTimeDeployResult:
    console_msg_pipeline_header(f"Begin {ms_model_description} deployment phase", ".")
    # RLRP-824 FR14 (d): an asymmetric ``F > H`` model is deployable plan-conditioned only.
    assert_asymmetric_window_deploy_conditioning(
        cfg, motion_model_container.dynamics_model.model
    )
    ms_3d_run_stats = []

    # Accumulators for the compounded-prediction-mode MAE on the InD/OOD target
    # rollout sets (deterministic cells only), surfaced as HPO objective options.
    # RLRP-723 F2: store per-trajectory per-step MAE *sequences* (feature-mean, meters)
    # rather than pre-summed scalars, so InD/OoD can be aggregated over a common horizon.
    compounded_pred_mae_InD: list[np.ndarray] = []
    compounded_pred_mae_OOD: list[np.ndarray] = []

    compounded_pred_rollouts, probabilistic_rollout, test_rollout_env_types = (
        setup_test_rollouts_grid(
            cfg,
            motion_model_container.dynamics_model.model,
            target_InD_rollouts,
            target_OOD_rollouts,
        )
    )

    # (Priority) ToDo: refactor this bloc (and the single step one above) using function
    #  "model_testtime_rollout_and_compute_prediction_metric()" from
    #  "model_testing_utils/model_performance_tester.py" as it repeat the same logic segmented
    #  over two functions, including the compute and plot fct.

    # Cache Hydra-instantiated post-processing to avoid re-instantiation per trajectory
    _cached_deploy_post_proc = hydra.utils.instantiate(
        cfg.environment.deploy_rollout_post_processing, cfg, _recursive_=False
    )
    # RLRP-736 S1.5: thread the per-environment feature handler into deploy
    # (level C). No-op unless the deploy object exposes ``set_feature_handler``.
    attach_feature_handler_to_deploy(cfg, _cached_deploy_post_proc)

    _effective_base_dir = save_base_dir if save_base_dir else exp_dir_relative_path

    # (Priority) ToDo: assess if its still usefull now that deploy forcaster logic is integrated to multistep_model_testtime_rollout_and_collect_pred_stats function.
    # RLRP-728 (S3): opt-in open-loop, control-conditioned per-horizon forecast eval (default OFF).
    # Deterministic + independent of the probabilistic x compounded grid, so it runs once per
    # selected InD/OOD trajectory before the main deploy grid below.
    _maybe_execute_openloop_forecast(
        cfg,
        motion_model_container,
        ms_model_description,
        test_rollout_env_types,
        torch_rng,
        save_base_dir,
    )

    # (Priority) ToDo: assess if its still usefull now that deploy forcaster logic is integrated to multistep_model_testtime_rollout_and_collect_pred_stats function.
    # RLRP-760 (S3): opt-in compounded self-fed forecast eval (default OFF). Same placement
    # rationale as the open-loop step above: deterministic and independent of the
    # probabilistic x compounded grid, so it runs once per selected InD/OOD trajectory.
    _maybe_execute_selffed_forecast(
        cfg,
        motion_model_container,
        ms_model_description,
        test_rollout_env_types,
        torch_rng,
        save_base_dir,
    )

    # RLRP-785 (A3 / A2.b): config-gated, OFF-by-default per-step benchmark. Resolved ONCE here;
    # when disabled (the deploy default) no benchmark kwargs are forwarded to the rollout, so it
    # stays byte-identical (gate G3). The step count is recorded UNCONDITIONALLY below (cheap
    # metadata), so the plot layer can always fall back to a harness-level rate even OFF.
    _bench_cfg = cfg.deploy.get("benchmark", None)
    _bench_enabled = (
        bool(_bench_cfg.get("enabled", False)) if _bench_cfg is not None else False
    )
    _bench_levels = ()
    _bench_timing_pass = None
    _bench_discard_warmup = 0
    # RLRP-803 (item 1 / gate ``V10``): opt-in raw-latency-sample retention on the DEPLOY path.
    # Without it the persisted metric carries ``latency_samples_ms = None``, so the taint split
    # can be counted but its p95/p99 tightening cannot be evidenced.
    _bench_keep_raw_samples = False
    _bench_max_raw_samples = 10_000
    if _bench_enabled:
        from tools.benchmark_tools.benchmark_metric import BenchmarkLevel, TimingPass

        _bench_levels = tuple(
            BenchmarkLevel(str(_l)) for _l in _bench_cfg.get("levels", ["control_loop_step"])
        )
        _bench_timing_pass = TimingPass(
            str(_bench_cfg.get("timing_pass", "deployable_latency"))
        )
        _bench_discard_warmup = int(_bench_cfg.get("discard_warmup_steps", 0))
        _bench_keep_raw_samples = bool(_bench_cfg.get("keep_raw_samples", False))
        _bench_max_raw_samples = int(_bench_cfg.get("max_raw_samples", 10_000))

    for each_is_pr_rollout in probabilistic_rollout:
        for entry, target_is_ood in test_rollout_env_types:
            test_trajectories = entry.env
            trajectory_short_name = entry.short_name

            for is_compounded_pred_rollout in compounded_pred_rollouts:

                consol_msg_universal_one_liner(
                    f"[{trajectory_short_name}] {'OoD' if target_is_ood else 'InD'}, {'' if is_compounded_pred_rollout else 'non-'}compounded predictions, {'probabilistic' if each_is_pr_rollout else 'deterministic'} rollout"
                )
                # RLRP-785 (A3): a fresh per-rollout container to receive the metric set; ``None``
                # when OFF so the rollout signature default path is taken (byte-identical).
                _benchmark_out = [] if _bench_enabled else None
                rollout_start_time = time.time()

                (
                    ms_pred_world_pose,
                    ms2ss_3d_ensemble_pred_mean,
                    ms2ss_3d_ensemble_pred_logvar,
                ) = multistep_model_testtime_rollout_and_collect_pred_stats(
                    cfg,
                    test_trajectories,
                    motion_model_container,
                    torch_rng,
                    is_compounded_pred_rollout,
                    next_state_deterministic_selection=not each_is_pr_rollout,
                    next_state_sampling_size=cfg.deploy.model_runtime.next_state_sampling_size,
                    ground_truth_feed_warmup_steps=cfg.deploy.target_experiment.ground_truth_feed_warmup_steps,
                    tutor_and_release=cfg.deploy.target_experiment.get(
                        "tutor_and_release", False
                    ),
                    deploy_rollout_post_processing=_cached_deploy_post_proc,
                    benchmark_step_timing=_bench_enabled,
                    benchmark_levels=_bench_levels,
                    benchmark_timing_pass=_bench_timing_pass,
                    benchmark_discard_warmup_steps=_bench_discard_warmup,
                    benchmark_keep_raw_samples=_bench_keep_raw_samples,
                    benchmark_max_raw_samples=_bench_max_raw_samples,
                    benchmark_out=_benchmark_out,
                )
                rollout_time = time.time() - rollout_start_time
                # RLRP-785 (A3): the ``BenchmarkMetricSet`` produced by ``step_timer.report(...)``
                # inside the rollout (``None`` when instrumentation was OFF), plus the step count
                # (always available -- the divisor for a harness-level rate).
                _rollout_benchmark = _benchmark_out[0] if _benchmark_out else None
                _rollout_steps = getattr(test_trajectories, "trajectory_len", None)
                _rollout_steps = (
                    int(_rollout_steps) if _rollout_steps is not None else None
                )

                ms_pred_world_pose = ms_pred_world_pose.cpu().numpy()

                # //// INLINE REFACTORED BLOC /////////////////////////////////////////////////////////////////

                (
                    ms_3d_pred_stats_mean,
                    ms_3d_pred_stats_ale_epi_std,
                    ms_3d_pred_stats_epi,
                ) = compute_n_dim_obs_trajectory_probability_statistics_over_ensemble(
                    n_dim_obs_pred=ms2ss_3d_ensemble_pred_mean,
                    n_dim_obs_pred_logvar=ms2ss_3d_ensemble_pred_logvar,
                    ensemble_size=cfg.ms_model.ensemble_size,
                )

                _plot_3d_rollout(
                    cfg,
                    motion_model_container.dynamics_model.model,
                    test_trajectories,
                    pred_3d_ale_epi_std=ms_3d_pred_stats_ale_epi_std.cpu().numpy(),
                    pred_3d_epi_std=ms_3d_pred_stats_epi.cpu().numpy(),
                    pred_3d_mean_coord=ms_pred_world_pose,
                    # RLRP-741: forward the obs-space per-feature prediction mean
                    # for the optional arbitrary feature-dim subplots.
                    pred_feature_mean=ms_3d_pred_stats_mean.cpu().numpy(),
                    tensorboard_writer=ms_tensorboard_writer,
                    exp_dir_relative_path=_effective_base_dir,
                    state_space_label=state_space_label,
                    show_interval_subplot=cfg.pipeline.plot.show_deploy_prediction_1d_subplot_interval,
                    probabilistic_rollout=each_is_pr_rollout,
                    compounded_predictions_score=is_compounded_pred_rollout,
                    test_env_is_OoD=target_is_ood,
                    headless=headless,
                    trajectory_subdir=trajectory_short_name,
                    epoch_rollout_label=epoch_rollout_label,
                )
                # ////////////////////////////////////////////////////////// INLINE REFACTORED BLOC ///(end)///

                # RLRP-723 F1: the RLRP-707 grid-shift already aligns pred_world_pose[i]
                # to t_i (row 0 == initial pose), so the error is the *direct* pred[i]-gt[i]
                # (align=False). Drop the trivially-zero seeded index 0 via the [1:] slice so
                # a perfect rollout scores exactly 0 and the seeded row is never credited.
                pred_mae = compute_trajectories_prediction_error(
                    pred=ms_pred_world_pose[1:],
                    target=test_trajectories.pose[1:],
                    align_trajectory_prediction_timestep_with_target_states=False,
                    normalize_input_via_feature_scalling=False,
                    batch_reduction=True,
                    coordinate_error_mode="elementwise",
                )

                # Surface the compounded-prediction-mode MAE (deterministic cells only) on the
                # InD/OOD target rollout sets, using the same reduction as best_pred_mae_score.
                if is_compounded_pred_rollout and not each_is_pr_rollout:
                    # RLRP-723 F2: keep the per-step (feature-mean, meters) MAE sequence so
                    # InD/OoD can later be summed over a *common horizon* (recommendation 1),
                    # preserving the cumulative compounded-drift semantics while staying
                    # comparable across trajectory sets of differing length.
                    _pred_mae_seq = _per_step_feature_mae(pred_mae)
                    if target_is_ood:
                        compounded_pred_mae_OOD.append(_pred_mae_seq)
                    else:
                        compounded_pred_mae_InD.append(_pred_mae_seq)

                # RLRP-723 F1: direct alignment (see MAE above) — drop seeded index 0.
                pred_l2_norm = compute_trajectories_prediction_error(
                    pred=ms_pred_world_pose[1:],
                    target=test_trajectories.pose[1:],
                    align_trajectory_prediction_timestep_with_target_states=False,
                    normalize_input_via_feature_scalling=False,
                    batch_reduction=True,
                    coordinate_error_mode="l2",
                )

                # RLRP-761 S8.16 (option A): record the obs-space GT + per-step obs
                # error so the adverse-event M2 slice is scorable on robotic-3D. Both
                # arrays are already in scope here (obs-space GT trajectory + physical
                # per-feature prediction mean) — no dataset reload, no re-integration.
                obs_target, obs_mae = _obs_space_gt_and_error(
                    getattr(test_trajectories, "observations", None),
                    ms_3d_pred_stats_mean,
                )

                ms_3d_run_stats.append(
                    _assemble_and_persist_prediction_stats(
                        cfg,
                        model=motion_model_container.dynamics_model,
                        model_description=ms_model_description,
                        metric_mean=ms_3d_pred_stats_mean,
                        stats_mean=ms_3d_pred_stats_mean,
                        stats_ale_epi_std=ms_3d_pred_stats_ale_epi_std,
                        stats_epi_std=ms_3d_pred_stats_epi,
                        pred_world_pose=ms_pred_world_pose,
                        test_trajectories=test_trajectories,
                        pred_mae=pred_mae,
                        pred_l2_norm=pred_l2_norm,
                        target_is_ood=target_is_ood,
                        is_compounded_pred_rollout=is_compounded_pred_rollout,
                        is_pr_rollout=each_is_pr_rollout,
                        training_time=training_time,
                        rollout_time=rollout_time,
                        trajectory_short_name=trajectory_short_name,
                        save_base_dir=save_base_dir,
                        obs_target=obs_target,
                        obs_mae=obs_mae,
                        # RLRP-785 (A3): step count (always) + benchmark set (when ON).
                        rollout_steps=_rollout_steps,
                        benchmark=_rollout_benchmark,
                    )
                )

    console_msg_pipeline_footer(f"{ms_model_description} deployment phase done", ".")
    # RLRP-723 F2: aggregate the cumulative L1 MAE over a *common horizon* H = min over all
    # scored InD/OoD trajectories, so the InD-vs-OoD comparison is not biased by horizon
    # length while keeping the cumulative (compounded-drift) semantics.
    _common_H = _common_horizon(compounded_pred_mae_InD, compounded_pred_mae_OOD)
    return MsTestTimeDeployResult(
        run_stats=tuple(ms_3d_run_stats),
        compounded_pred_mae_target_InD=_cumulative_mae_over_common_horizon(
            compounded_pred_mae_InD, _common_H
        ),
        compounded_pred_mae_target_OOD=_cumulative_mae_over_common_horizon(
            compounded_pred_mae_OOD, _common_H
        ),
    )


def execute_ms_model_test_time_rollouts_over_epoch_checkpoints(
    cfg: DictConfig,
    motion_model_container: R2SMotionModelContainer,
    ms_model_description,
    state_space_label,
    target_InD_rollouts: list[TestTrajectoryEntry],
    target_OOD_rollouts: list[TestTrajectoryEntry],
    training_time: float,
    ms_tensorboard_writer: OnlineTensorboardWritter | None | Any,
    exp_dir_relative_path: str | Any,
    headless: bool,
    torch_rng,
    save_base_dir: str | None = None,
    checkpoints_source_dir: str | None = None,
    epoch_stride: int | None = None,
    epochs: list[int] | None = None,
) -> list[tuple[int, MsTestTimeDeployResult]]:
    """Run the test-time rollout ONCE PER epoch checkpoint, preserving each in its own subtree.

    Thin wrapper around :func:`execute_ms_model_test_time_rollouts` for the RLRP-773 epoch
    checkpoint feature (R6). It discovers the ``epoch_checkpoints/`` tree written on the save
    side, optionally narrows it (R12 subset selection), loads each checkpoint's weights IN PLACE
    into the already-built container (avoiding ``setup_multistep_step_model``'s
    ``.hydra/config.yaml``-in-parent requirement, which the ``epoch_checkpoints/`` tree does not
    provide), and runs the rollout into a distinct ``epoch_checkpoints_rollouts/epoch_<E>/`` subtree
    so each epoch's results are preserved (R2).

    The wrapper is legacy-safe and strictly additive: when no epoch checkpoints exist it is a no-op
    returning ``[]`` (the caller keeps its normal single deploy as the primary result).

    Container safety (R13): loading a checkpoint overwrites the live weights/normalizers and puts
    the model in ``eval`` mode, and the container is returned to the caller, so the loop is wrapped
    in ``try/finally`` that reloads the run's primary (best-metric) model and restores the
    ``train``/``eval`` mode on exit. TensorBoard isolation (R13): the per-epoch calls are given
    ``ms_tensorboard_writer=None`` so the N identically-tagged scalar series never collide with the
    primary deploy's TensorBoard record.

    :param cfg: the Hydra config.
    :param motion_model_container: the already-built container whose ``dynamics_model`` is loaded
        in place for each epoch.
    :param ms_model_description: the model description string (passed through).
    :param state_space_label: the state-space label (passed through).
    :param target_InD_rollouts: the InD target rollouts (passed through).
    :param target_OOD_rollouts: the OoD target rollouts (passed through).
    :param training_time: the training time (passed through).
    :param ms_tensorboard_writer: accepted for signature parity; NOT forwarded to the per-epoch
        rollouts (see R13 above).
    :param exp_dir_relative_path: the experiment cwd (default discovery/output base).
    :param headless: rendering flag (passed through).
    :param torch_rng: the torch RNG (passed through).
    :param save_base_dir: optional output base override; when given, the per-epoch rollout subtree
        is written under it instead of ``exp_dir_relative_path``.
    :param checkpoints_source_dir: optional dir holding the ``epoch_checkpoints/`` tree AND the
        primary ``model_<ClassName>/`` dir (e.g. a deploy-only run's experiment dir); defaults to
        the output base.
    :param epoch_stride: R12 subset — keep every ``epoch_stride``-th discovered checkpoint.
    :param epochs: R12 subset — explicit epoch subset (wins over ``epoch_stride``).
    :return: a list of ``(epoch, MsTestTimeDeployResult)``, ascending by epoch (empty when no
        epoch checkpoints are found).
    """
    base_dir = save_base_dir if save_base_dir else exp_dir_relative_path
    discovery_dir = checkpoints_source_dir if checkpoints_source_dir else base_dir

    discovered = persistent_checkpoint_utils.discover_epoch_checkpoints(discovery_dir)
    if not discovered:
        # Legacy-safe no-op: no epoch checkpoints on disk => caller keeps its single deploy.
        return []
    selected = persistent_checkpoint_utils.select_epoch_checkpoints(
        discovered, stride=epoch_stride, epochs=epochs
    )

    console_msg_pipeline_header(
        f"Begin {ms_model_description} per-epoch-checkpoint deployment "
        f"({len(selected)} of {len(discovered)} epoch checkpoints)",
        ".",
    )

    model = motion_model_container.dynamics_model
    was_training = bool(getattr(model, "training", False))

    per_epoch_results: list[tuple[int, MsTestTimeDeployResult]] = []
    # RLRP-773: the label denominator is the LARGEST selected training epoch (not the snapshot
    # count), so the title carries the checkpoint's REAL training epoch (see below).
    _max_selected_epoch = max(epoch for epoch, _ in selected)
    try:
        for epoch, ckpt_dir in selected:
            persistent_checkpoint_utils.load_epoch_checkpoint_model(model, ckpt_dir)
            per_epoch_out_dir = (
                persistent_checkpoint_utils.epoch_checkpoint_rollouts_dir(
                    base_dir, epoch
                )
            )
            os.makedirs(per_epoch_out_dir, exist_ok=True)
            consol_msg_universal_one_liner(
                f"epoch checkpoint {epoch} -> {per_epoch_out_dir}"
            )
            # RLRP-773: surface the checkpoint's REAL training epoch in the per-epoch rollout plot
            # title (e.g. "(epoch 80/560)") — the completed global inner-epoch count of THIS
            # snapshot over the largest selected epoch — so each of the N rendered figures is
            # identifiable by its actual training epoch (not a selection ordinal).
            _epoch_rollout_label = f"{epoch}/{_max_selected_epoch}"
            epoch_result = execute_ms_model_test_time_rollouts(
                cfg,
                motion_model_container,
                ms_model_description,
                state_space_label,
                target_InD_rollouts,
                target_OOD_rollouts,
                training_time,
                None,  # R13: never share the primary TensorBoard writer across epochs.
                exp_dir_relative_path,
                headless,
                torch_rng,
                save_base_dir=per_epoch_out_dir,
                epoch_rollout_label=_epoch_rollout_label,
            )
            per_epoch_results.append((epoch, epoch_result))
    finally:
        # R13: restore the container to a state equivalent to entry — reload the primary
        # (best-metric) model that the caller's single deploy used, then restore train/eval mode.
        primary_model_dir = os.path.join(discovery_dir, setup_saved_model_dir(model))
        if os.path.isdir(primary_model_dir):
            model.load(primary_model_dir)
        else:
            warnings.warn(
                "execute_ms_model_test_time_rollouts_over_epoch_checkpoints: could not locate the "
                f"primary model dir {primary_model_dir!r} to restore the container; it is left "
                "holding the last-iterated epoch checkpoint weights.",
                stacklevel=2,
            )
        if was_training and hasattr(model, "train"):
            model.train()

    console_msg_pipeline_footer(
        f"{ms_model_description} per-epoch-checkpoint deployment done", "."
    )
    return per_epoch_results


def run_epoch_checkpoint_rollouts_if_enabled(
    cfg: DictConfig,
    motion_model_container: R2SMotionModelContainer,
    ms_model_description,
    state_space_label,
    target_InD_rollouts: list[TestTrajectoryEntry],
    target_OOD_rollouts: list[TestTrajectoryEntry],
    training_time: float | None,
    ms_tensorboard_writer: OnlineTensorboardWritter | None | Any,
    exp_dir_relative_path: str | Any,
    headless: bool,
    torch_rng,
    save_base_dir: str | None = None,
    checkpoints_source_dir: str | None = None,
    epochs_override: Sequence[int] | None = None,
    wipe_dirs: Sequence[str] = (),
) -> list[tuple[int, MsTestTimeDeployResult]]:
    """Opt-in gate for per-epoch-checkpoint rollouts, shared by all deploy paths (RLRP-773 R12).

    Reads the ``deploy.epoch_checkpoint_rollouts`` block and only runs
    :func:`execute_ms_model_test_time_rollouts_over_epoch_checkpoints` when ``enable`` is truthy.
    Centralising the gate here keeps the four call sites (the two ``*_ms_model_train_and_deploy``
    utils and the two ``*_deploy_only_pipeline`` modules) consistent. Default OFF => legacy cost,
    zero new work; also a no-op when no ``epoch_checkpoints/`` tree exists on disk.

    :param save_base_dir: forwarded output base override (see the wrapper).
    :param checkpoints_source_dir: forwarded checkpoints/primary-model source dir (see the wrapper).
    :param epochs_override: RLRP-839 deploy-stage resume — the exact epoch subset to (re)run; wins
        over the cfg ``epoch_stride`` / ``epochs`` selection. An EMPTY sequence (nothing missing)
        short-circuits to ``[]``; ``None`` => cfg selection (legacy).
    :param wipe_dirs: RLRP-839 — existing-but-partial ``epoch_checkpoints_rollouts/epoch_<E>/`` dirs
        removed (``shutil.rmtree``, logged) BEFORE the wrapper runs so a re-run starts clean.
    :return: the per-epoch ``(epoch, result)`` list, or ``[]`` when disabled / nothing to do.
    """
    epoch_ckpt_cfg = omegaconf.OmegaConf.select(
        cfg, "deploy.epoch_checkpoint_rollouts", default=None
    )
    if epoch_ckpt_cfg is None or not bool(
        omegaconf.OmegaConf.select(epoch_ckpt_cfg, "enable", default=False)
    ):
        return []

    if epochs_override is not None:
        epochs_override = [int(each) for each in epochs_override]
        if not epochs_override:
            consol_msg_universal_one_liner(
                "per-epoch-checkpoint deployment: no missing epoch rollout to (re)run, skipping."
            )
            return []

    for each_dir in wipe_dirs:
        if os.path.isdir(each_dir):
            consol_msg_universal_one_liner(
                f"resume: removing PARTIAL epoch rollout dir before re-run -> {each_dir}"
            )
            shutil.rmtree(each_dir)

    if epochs_override is not None:
        epoch_stride, epochs = None, epochs_override
    else:
        epoch_stride = omegaconf.OmegaConf.select(epoch_ckpt_cfg, "epoch_stride", default=None)
        epochs = omegaconf.OmegaConf.select(epoch_ckpt_cfg, "epochs", default=None)

    return execute_ms_model_test_time_rollouts_over_epoch_checkpoints(
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
        save_base_dir=save_base_dir,
        checkpoints_source_dir=checkpoints_source_dir,
        epoch_stride=epoch_stride,
        epochs=epochs,
    )


# //// RLRP-839: resume at the deployment stage ////////////////////////////////////////////////////
_RESUME_WHO_AM_I = "resume_from_checkpoint"


def expected_testtime_rollout_metric_relpaths(
    cfg: DictConfig,
    model: mbrl.models.Model,
    ms_model_description: str,
    target_InD_rollouts: list[TestTrajectoryEntry],
    target_OOD_rollouts: list[TestTrajectoryEntry],
) -> list[str]:
    """Every metric leaf a COMPLETE deploy writes, relative to its base dir (RLRP-839 FR3).

    Reproduces the exact producer path (:func:`_assemble_and_persist_prediction_stats` +
    :meth:`TestTimeRolloutPredictionMetric.save`) over the :func:`setup_test_rollouts_grid` grid::

        testtime_rollouts/<traj short_name>/TTRPM_<sanitized desc>_PR_rollout<pr>/
            <InD|OOD>_compounded_<c>/TestTimeRolloutPredictionMetric.pkl

    The list is sorted so audits are reproducible. The same list is valid for the run root (main
    deploy) and for each ``epoch_checkpoints_rollouts/epoch_<E>/`` dir (per-epoch deploy).
    """
    compounded_pred_rollouts, probabilistic_rollout, test_rollout_env_types = (
        setup_test_rollouts_grid(cfg, model, target_InD_rollouts, target_OOD_rollouts)
    )
    leaves: list[str] = []
    for is_pr_rollout in probabilistic_rollout:
        for entry, target_is_ood in test_rollout_env_types:
            for is_compounded in compounded_pred_rollouts:
                leaves.append(
                    os.path.join(
                        resume_utils.TESTTIME_ROLLOUTS_ROOT_NAME,
                        entry.short_name,
                        f"TTRPM_{sanitize_dirname(ms_model_description)}_PR_rollout{is_pr_rollout}",
                        get_metric_sub_dir(is_compounded, target_is_ood),
                        "TestTimeRolloutPredictionMetric.pkl",
                    )
                )
    return sorted(set(leaves))


def load_primary_model_in_place(
    motion_model_container: R2SMotionModelContainer, run_root: str
) -> str:
    """Load ``<run_root>/model_<ClassName>/saved_dynamic_model`` into the live container (FR5).

    :return: the loaded saved-model dir.
    :raises FileNotFoundError: when the primary model dir is absent (explicit, SLURM-safe).
    """
    model = motion_model_container.dynamics_model
    saved_model_dir = os.path.join(run_root, setup_saved_model_dir(model))
    if not os.path.isdir(saved_model_dir):
        raise FileNotFoundError(
            f"resume (deploy stage): primary model dir `{saved_model_dir}` not found under the "
            f"candidate run root `{run_root}`. The run cannot be deployed in place; resume it at the "
            "training stage instead (point `training_common.resume_from_checkpoint` at its "
            "`epoch_checkpoints/epoch_<E>/` dir)."
        )
    model.load(saved_model_dir)
    if hasattr(model, "eval"):
        model.eval()
    consol_msg_universal_one_liner(f"resume: loaded primary model in place from {saved_model_dir}")
    return saved_model_dir


def _resolve_resume_root(cfg: DictConfig, resume_from: str) -> str:
    """Resolve a relative ``resume_from_checkpoint`` against the project root / original cwd.

    Hydra ``chdir: true`` moves the process into the NEW job dir, so a path such as
    ``artifact/ICRA2026/...`` (relative to the project root, as documented) would not resolve
    against ``os.getcwd()``. The first existing candidate wins; otherwise the raw path is returned
    so the scan raises its explicit ``FileNotFoundError``.
    """
    resume_from = os.path.expanduser(str(resume_from))
    if os.path.isabs(resume_from):
        return resume_from
    bases: list[str] = []
    project_root = omegaconf.OmegaConf.select(cfg, "project_root_path", default=None)
    if project_root:
        bases.append(str(project_root))
    try:
        bases.append(str(HydraConfig.get().runtime.cwd))
    except Exception:  # not under Hydra (unit tests)
        pass
    bases.append(os.getcwd())
    for base in bases:
        candidate = os.path.join(base, resume_from)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return resume_from


def _current_hydra_launch_context() -> tuple[list[str], Optional[dict], list[str], int]:
    """``(task_overrides, runtime_choices, own_output_roots, job_num)`` of the current launch."""
    try:
        hydra_cfg = HydraConfig.get()
    except Exception:  # not under Hydra (unit tests / notebooks)
        return [], None, [], 0
    task_overrides = [str(each) for each in (hydra_cfg.overrides.task or [])]
    choices = omegaconf.OmegaConf.select(hydra_cfg, "runtime.choices", default=None)
    choices = dict(choices) if choices is not None else None
    # The own output roots are derived from the already-resolved `runtime.output_dir` (NOT from
    # `sweep.dir` / `run.dir`, whose `${now:...}` interpolations may re-resolve to a later instant).
    # In MULTIRUN mode the parent of the job dir is this launch's sweep dir (only its own jobs), so
    # it is excluded too; in RUN mode the parent is a shared date dir and must stay scannable.
    own_roots: list[str] = []
    output_dir = omegaconf.OmegaConf.select(hydra_cfg, "runtime.output_dir", default=None)
    if output_dir:
        output_dir = str(output_dir).rstrip(os.sep)
        own_roots.append(output_dir)
        mode = omegaconf.OmegaConf.select(hydra_cfg, "mode", default=None)
        if mode is not None and "MULTIRUN" in str(mode).upper():
            # The sweep dir is the output dir minus its `job.override_dirname` leaf. NOT
            # `os.path.dirname`: the override dirname may itself contain `/` (e.g. the value of
            # `training_common.resume_from_checkpoint=artifact/...`) and would be cut short.
            subdir = str(
                omegaconf.OmegaConf.select(hydra_cfg, "job.override_dirname", default="") or ""
            ).rstrip(os.sep)
            if subdir and output_dir.endswith(subdir):
                own_roots.append(output_dir[: -len(subdir)].rstrip(os.sep))
            else:
                own_roots.append(os.path.dirname(output_dir))
    job_num = int(omegaconf.OmegaConf.select(hydra_cfg, "job.num", default=0) or 0)
    return task_overrides, choices, own_roots, job_num


def resolve_resume_experiments(
    cfg: DictConfig,
    motion_model_container: R2SMotionModelContainer,
    ms_model_description: str,
    target_InD_rollouts: list[TestTrajectoryEntry],
    target_OOD_rollouts: list[TestTrajectoryEntry],
    current_task_overrides: Optional[Sequence[str]] = None,
    current_choices: Optional[dict] = None,
    exclude_roots: Optional[Sequence[str]] = None,
    job_num: Optional[int] = None,
) -> Optional[list[resume_utils.ResumeExperiment]]:
    """Turn ``training_common.resume_from_checkpoint`` into the audited, ordered resume experiments.

    ``None`` when the key is null (legacy single run). Otherwise (RLRP-839 FR1-FR4, FR7-FR10):
    scan the path (excluding the current launch's own output roots), fail fast on any illegal
    candidate, audit every candidate against the static epoch budget and the expected deploy metric
    leaves, print the candidate table, apply ``training_common.resume.candidate_filter`` and honour
    ``training_common.resume.dry_run`` (table only => ``[]``). Non-interactive by design.

    The ``current_*`` / ``exclude_roots`` / ``job_num`` parameters default to the live
    :class:`HydraConfig` and exist so unit tests can run the resolution outside Hydra.

    :raises resume_utils.ResumeLegalityError: on any candidate/launch mismatch.
    :raises RuntimeError: scan mode with several candidates launched as a multi-job sweep
        (``hydra.job.num > 0``) — duplicate in-place writes; launch with ``--multirun trial_nb=1``.
    :raises ValueError: when the planned epoch budget is patience-driven (cannot audit).
    """
    resume_from = omegaconf.OmegaConf.select(
        cfg, "training_common.resume_from_checkpoint", default=None
    )
    if not resume_from:
        return None

    hydra_overrides, hydra_choices, hydra_own_roots, hydra_job_num = _current_hydra_launch_context()
    if current_task_overrides is None:
        current_task_overrides = hydra_overrides
    if current_choices is None:
        current_choices = hydra_choices
    if exclude_roots is None:
        exclude_roots = hydra_own_roots
    if job_num is None:
        job_num = hydra_job_num

    model = motion_model_container.dynamics_model
    primary_model_dirname = os.path.dirname(setup_saved_model_dir(model))
    scan_root = _resolve_resume_root(cfg, str(resume_from))
    candidates = resume_utils.scan_resume_candidates(
        scan_root, exclude_roots=list(exclude_roots), primary_model_dirname=primary_model_dirname
    )
    scan_mode = not (
        resume_utils.is_epoch_checkpoint_dir(scan_root) or resume_utils.is_job_dir(scan_root)
    )
    consol_msg(
        who_am_i=_RESUME_WHO_AM_I,
        msg=(
            f"{'scanned' if scan_mode else 'resolved'} `{scan_root}` -> {len(candidates)} "
            f"candidate(s); excluded own output roots: {list(exclude_roots) or '-'}"
        ),
        space_after=False,
    )

    if scan_mode:
        # FR2: a scanned candidate must be provably from the SAME experiment as this launch.
        resume_utils.validate_candidates_legality(candidates, current_task_overrides, current_choices)
    else:
        consol_msg_universal_one_liner(
            "resume: explicit epoch dir / run root given => legality check vs the current launch "
            "skipped (RLRP-824 semantics)."
        )

    candidate_filter = omegaconf.OmegaConf.select(
        cfg, "training_common.resume.candidate_filter", default=None
    )
    filtered = resume_utils.filter_candidates(candidates, candidate_filter)
    if candidate_filter and len(filtered) != len(candidates):
        consol_msg_universal_one_liner(
            f"resume: candidate_filter={candidate_filter!r} keeps {len(filtered)}/{len(candidates)}"
        )
    if not filtered:
        raise FileNotFoundError(
            f"training_common.resume.candidate_filter={candidate_filter!r} matches none of the "
            f"{len(candidates)} candidate(s) found under `{scan_root}`."
        )

    total_epochs = experiment_planned_total_epochs(cfg)
    expected_leaves = expected_testtime_rollout_metric_relpaths(
        cfg, model.model, ms_model_description, target_InD_rollouts, target_OOD_rollouts
    )
    epoch_ckpt_cfg = omegaconf.OmegaConf.select(
        cfg, "deploy.epoch_checkpoint_rollouts", default=None
    )
    epoch_rollouts_enabled = bool(
        omegaconf.OmegaConf.select(epoch_ckpt_cfg, "enable", default=False)
    ) if epoch_ckpt_cfg is not None else False
    experiments = [
        resume_utils.build_resume_experiment(
            candidate,
            experiment_planned_total_epochs=total_epochs,
            expected_relative_metric_paths=expected_leaves,
            epoch_rollouts_enabled=epoch_rollouts_enabled,
            epoch_stride=(
                omegaconf.OmegaConf.select(epoch_ckpt_cfg, "epoch_stride", default=None)
                if epoch_ckpt_cfg is not None else None
            ),
            epochs=(
                omegaconf.OmegaConf.select(epoch_ckpt_cfg, "epochs", default=None)
                if epoch_ckpt_cfg is not None else None
            ),
        )
        for candidate in filtered
    ]

    consol_msg(
        who_am_i=_RESUME_WHO_AM_I,
        msg=(
            f"candidate table (experiment_planned_total_epochs={total_epochs}, "
            f"epoch_rollouts_enabled={epoch_rollouts_enabled}, "
            f"{len(expected_leaves)} expected metric leaves per deploy):\n"
            + resume_utils.format_candidate_table(experiments)
        ),
    )

    if omegaconf.OmegaConf.select(cfg, "training_common.resume.dry_run", default=False):
        consol_msg(
            who_am_i=_RESUME_WHO_AM_I,
            msg="training_common.resume.dry_run=true => table printed, nothing will be executed.",
        )
        return []

    if scan_mode and len(experiments) > 1 and int(job_num) > 0:
        # FR9: several driver jobs would complete the same candidates IN PLACE concurrently.
        raise RuntimeError(
            f"resume scan found {len(experiments)} candidates but this launch is Hydra job #{job_num} of a "
            "multi-job sweep: several jobs would write the same run roots in place. Launch a SINGLE "
            "driver job by collapsing the `trial_nb` sweep: put `trial_nb=1` among the overrides, with "
            "`--multirun` placed BEFORE every `key=value` override (Hydra's argparse rejects a trailing "
            "`--multirun trial_nb=1` as 'unrecognized arguments'), e.g. "
            "`launcher/math_env_main.py --config-name=... --config-dir=... --multirun trial_nb=1 "
            "training_common.resume_from_checkpoint=<root>`; shard by hand with "
            "`training_common.resume.candidate_filter=<multirun id>` if needed."
        )
    return experiments


@contextlib.contextmanager
def resume_cfg_override(cfg: DictConfig, experiment: resume_utils.ResumeExperiment):
    """Point ``training_common.resume_from_checkpoint`` at *experiment*'s run root for one candidate (FR6).

    The ERLL reads that key at construction (RLRP-824), so the train stage of a scanned candidate
    is fed its resolved run root (=> latest cadence checkpoint). When the operator gave an explicit
    ``epoch_<E>`` dir that belongs to the same run root, that more specific value is kept
    (RLRP-824 semantics). The original value is restored on exit.
    """
    original = omegaconf.OmegaConf.select(
        cfg, "training_common.resume_from_checkpoint", default=None
    )
    target = experiment.candidate.run_root
    if original:
        resolved_original = _resolve_resume_root(cfg, str(original))
        if resume_utils.is_epoch_checkpoint_dir(resolved_original) and os.path.abspath(
            os.path.dirname(os.path.dirname(resolved_original))
        ) == os.path.abspath(target):
            target = os.path.abspath(resolved_original)
    with omegaconf.read_write(cfg), omegaconf.open_dict(cfg):
        omegaconf.OmegaConf.update(
            cfg, "training_common.resume_from_checkpoint", target, force_add=True
        )
    try:
        yield target
    finally:
        with omegaconf.read_write(cfg), omegaconf.open_dict(cfg):
            omegaconf.OmegaConf.update(
                cfg, "training_common.resume_from_checkpoint", original, force_add=True
            )


def run_resume_driver(
    cfg: DictConfig,
    experiments: Sequence[resume_utils.ResumeExperiment],
    run_once: Callable[[resume_utils.ResumeExperiment], Any],
) -> Optional[Any]:
    """Sequential, non-interactive driver over the audited resume experiments (RLRP-839 FR7/FR8/FR10).

    ``complete`` / ``fresh`` experiments are skipped with a log line; each ``train`` / ``deploy`` experiment is
    executed by *run_once* under :func:`resume_cfg_override`, then a ``resume_log.json`` entry is
    appended at the candidate root. A failing candidate is reported and the loop CONTINUES; once
    every candidate has been visited the failures are re-raised as one ``RuntimeError`` (non-zero
    exit, readable ``console.log``).

    :return: the result of the LAST executed *run_once* (``None`` when nothing was executed).
    """
    hydra_overrides, _, hydra_own_roots, _ = _current_hydra_launch_context()
    launcher_cwd = hydra_own_roots[0] if hydra_own_roots else os.getcwd()
    last_result: Optional[Any] = None
    failures: list[tuple[str, BaseException]] = []
    executed = 0
    for idx, experiment in enumerate(experiments, start=1):
        root = experiment.candidate.run_root
        if not experiment.needs_work:
            consol_msg(
                who_am_i=_RESUME_WHO_AM_I,
                msg=(
                    f"[{idx}/{len(experiments)}] stage={experiment.stage} => skipping `{root}`"
                    + (
                        " (no epoch checkpoint and no primary model on disk: nothing to resume)"
                        if experiment.stage == resume_utils.STAGE_FRESH
                        else ""
                    )
                ),
                space_before=False,
                space_after=False,
            )
            continue
        consol_msg(
            who_am_i=_RESUME_WHO_AM_I,
            msg=(
                f"[{idx}/{len(experiments)}] stage={experiment.stage} => resuming `{root}` "
                f"(main_deploy_complete={experiment.main_deploy_complete}, "
                f"missing_epoch_rollouts={list(experiment.missing_epoch_rollouts) or '-'})"
            ),
        )
        entry = {
            "stage": experiment.stage,
            "main_deploy_rerun": not experiment.main_deploy_complete,
            "epochs_rerun": list(experiment.missing_epoch_rollouts),
            "wiped_dirs": list(experiment.partial_epoch_rollout_dirs),
            "launcher_cwd": launcher_cwd,
            "launcher_overrides": list(hydra_overrides),
            "experiment_planned_total_epochs": experiment.experiment_planned_total_epochs,
            "latest_ckpt_epoch": experiment.candidate.latest_ckpt_epoch,
        }
        try:
            with resume_cfg_override(cfg, experiment):
                last_result = run_once(experiment)
            executed += 1
            entry["status"] = "ok"
        except Exception as error:  # noqa: BLE001 - reported, loop continues (FR7)
            failures.append((root, error))
            entry["status"] = f"failed: {type(error).__name__}: {error}"
            consol_msg(
                who_am_i=_RESUME_WHO_AM_I,
                msg=(
                    f"[{idx}/{len(experiments)}] FAILED `{root}`: {type(error).__name__}: {error}\n"
                    + traceback.format_exc()
                ),
            )
        try:
            resume_utils.append_resume_log(root, entry)
        except OSError as error:  # the audit trail must never mask the run outcome
            warnings.warn(f"resume: could not append resume_log.json at {root!r}: {error}", stacklevel=2)

    consol_msg(
        who_am_i=_RESUME_WHO_AM_I,
        msg=(
            f"driver done: {executed} executed, {len(experiments) - executed - len(failures)} skipped, "
            f"{len(failures)} failed out of {len(experiments)} candidate(s)."
        ),
    )
    if failures:
        details = "\n".join(f"- {root}: {type(error).__name__}: {error}" for root, error in failures)
        raise RuntimeError(
            f"resume driver: {len(failures)}/{len(experiments)} candidate(s) failed (see console.log for "
            f"the full tracebacks):\n{details}"
        )
    return last_result


def execute_ss_legacy_model_test_time_rollouts(
    cfg: DictConfig,
    ss_1D_transition_model: OneDTransitionRewardModelV2,
    ss_model_description,
    state_space_label,
    target_InD_rollouts: list[TestTrajectoryEntry],
    target_OOD_rollouts: list[TestTrajectoryEntry],
    training_time: float,
    ss_tensorboard_writer: OnlineTensorboardWritter | None | Any,
    exp_dir_relative_path: str | Any,
    headless: bool,
    torch_rng,
    save_base_dir: str | None = None,
) -> MsTestTimeDeployResult:
    console_msg_pipeline_header(f"Begin {ss_model_description} deployment phase", ".")
    ss_3d_run_stats = []

    # Accumulators for the compounded-prediction-mode MAE on the InD/OOD target
    # rollout sets (deterministic cells only), surfaced as HPO objective options.
    # RLRP-723 F2: store per-trajectory per-step MAE *sequences* (feature-mean, meters)
    # rather than pre-summed scalars, so InD/OoD can be aggregated over a common horizon.
    compounded_pred_mae_InD: list[np.ndarray] = []
    compounded_pred_mae_OOD: list[np.ndarray] = []

    compounded_pred_rollouts, probabilistic_rollout, test_rollout_env_types = (
        setup_test_rollouts_grid(
            cfg, ss_1D_transition_model.model, target_InD_rollouts, target_OOD_rollouts
        )
    )

    # .... Begin rollouts .........................................................................
    # (Priority) ToDo: refactor this bloc (and the multi-steps one below) using function
    #  "model_testtime_rollout_and_compute_prediction_metric()" from
    #  "model_testing_utils/model_performance_tester.py" as it repeat the same logic segmented
    #  over two functions, including the compute and plot fct.

    # Cache Hydra-instantiated post-processing to avoid re-instantiation per trajectory
    _cached_deploy_post_proc = hydra.utils.instantiate(
        cfg.environment.deploy_rollout_post_processing, cfg, _recursive_=False
    )
    # RLRP-736 S1.5: thread the per-environment feature handler into deploy
    # (level C). No-op unless the deploy object exposes ``set_feature_handler``.
    attach_feature_handler_to_deploy(cfg, _cached_deploy_post_proc)

    _effective_base_dir = save_base_dir if save_base_dir else exp_dir_relative_path

    for each_is_pr_rollout in probabilistic_rollout:
        if each_is_pr_rollout:
            rollout_type_str = f"Deterministic rollout\n"
        else:
            rollout_type_str = f"Probabilistic rollout\n"
        consol_msg_universal_one_liner(rollout_type_str)

        for entry, target_is_ood in test_rollout_env_types:
            test_trajectories = entry.env
            trajectory_short_name = entry.short_name

            for is_compounded_pred_rollout in compounded_pred_rollouts:
                rollout_start_time = time.time()

                (
                    ss_pred_world_pose,
                    ss_3d_ensemble_pred_mean,
                    ss_3d_ensemble_pred_logvar,
                ) = singlestep_model_testtime_rollout_and_collect_pred_stats(
                    cfg,
                    test_trajectories,
                    ss_1D_transition_model,
                    torch_rng,
                    is_compounded_pred_rollout,
                    next_state_deterministic_selection=not each_is_pr_rollout,
                    next_state_sampling_size=cfg.deploy.model_runtime.next_state_sampling_size,
                    ground_truth_feed_warmup_steps=cfg.deploy.target_experiment.ground_truth_feed_warmup_steps,
                    tutor_and_release=cfg.deploy.target_experiment.get(
                        "tutor_and_release", False
                    ),
                    deploy_rollout_post_processing=_cached_deploy_post_proc,
                )
                rollout_time = time.time() - rollout_start_time

                # //// INLINE REFACTORED BLOC /////////////////////////////////////////////////////////////////
                ss_pred_world_pose = ss_pred_world_pose.cpu().numpy()

                (
                    ss_3d_pred_mean,
                    ss_3d_pred_ale_epi_std,
                    ss_3d_pred_3d_epi_std,
                ) = compute_n_dim_obs_trajectory_probability_statistics_over_ensemble(
                    n_dim_obs_pred=ss_3d_ensemble_pred_mean,
                    n_dim_obs_pred_logvar=ss_3d_ensemble_pred_logvar,
                    ensemble_size=cfg.ss_model.ensemble_size,
                )

                _plot_3d_rollout(
                    cfg,
                    ss_1D_transition_model.model,
                    test_trajectories,
                    pred_3d_ale_epi_std=ss_3d_pred_ale_epi_std.cpu().numpy(),
                    pred_3d_epi_std=ss_3d_pred_3d_epi_std.cpu().numpy(),
                    pred_3d_mean_coord=ss_pred_world_pose,
                    # RLRP-741: forward the obs-space per-feature prediction mean
                    # for the optional arbitrary feature-dim subplots.
                    pred_feature_mean=ss_3d_pred_mean.cpu().numpy(),
                    tensorboard_writer=ss_tensorboard_writer,
                    exp_dir_relative_path=_effective_base_dir,
                    state_space_label=state_space_label,
                    show_interval_subplot=cfg.pipeline.plot.show_deploy_prediction_1d_subplot_interval,
                    probabilistic_rollout=each_is_pr_rollout,
                    compounded_predictions_score=is_compounded_pred_rollout,
                    test_env_is_OoD=target_is_ood,
                    headless=headless,
                    trajectory_subdir=trajectory_short_name,
                )
                # ////////////////////////////////////////////////////////// INLINE REFACTORED BLOC ///(end)///

                # RLRP-723 F1: direct alignment over the grid-shifted pose; drop seeded index 0.
                pred_mae = compute_trajectories_prediction_error(
                    pred=ss_pred_world_pose[1:],
                    target=test_trajectories.pose[1:],
                    align_trajectory_prediction_timestep_with_target_states=False,
                    normalize_input_via_feature_scalling=False,
                    batch_reduction=True,
                    coordinate_error_mode="elementwise",
                )

                # Surface the compounded-prediction-mode MAE (deterministic cells only) on the
                # InD/OOD target rollout sets, using the same reduction as best_pred_mae_score.
                if is_compounded_pred_rollout and not each_is_pr_rollout:
                    # RLRP-723 F2: keep the per-step (feature-mean, meters) MAE sequence so
                    # InD/OoD can later be summed over a *common horizon* (recommendation 1),
                    # preserving the cumulative compounded-drift semantics while staying
                    # comparable across trajectory sets of differing length.
                    _pred_mae_seq = _per_step_feature_mae(pred_mae)
                    if target_is_ood:
                        compounded_pred_mae_OOD.append(_pred_mae_seq)
                    else:
                        compounded_pred_mae_InD.append(_pred_mae_seq)

                # RLRP-723 F1: direct alignment (see MAE above) — drop seeded index 0.
                pred_l2_norm = compute_trajectories_prediction_error(
                    pred=ss_pred_world_pose[1:],
                    target=test_trajectories.pose[1:],
                    align_trajectory_prediction_timestep_with_target_states=False,
                    normalize_input_via_feature_scalling=False,
                    batch_reduction=True,
                    coordinate_error_mode="l2",
                )

                ss_3d_run_stats.append(
                    _assemble_and_persist_prediction_stats(
                        cfg,
                        model=ss_1D_transition_model,
                        model_description=ss_model_description,
                        # LEGACY SS asymmetry, preserved: the archived metric stores the
                        # integrated POSE as ``mean`` here, while the run-stats tuple
                        # below carries the prediction STATISTICS mean.
                        metric_mean=ss_pred_world_pose,
                        stats_mean=ss_3d_pred_mean,
                        stats_ale_epi_std=ss_3d_pred_ale_epi_std,
                        stats_epi_std=ss_3d_pred_3d_epi_std,
                        pred_world_pose=ss_pred_world_pose,
                        test_trajectories=test_trajectories,
                        pred_mae=pred_mae,
                        pred_l2_norm=pred_l2_norm,
                        target_is_ood=target_is_ood,
                        is_compounded_pred_rollout=is_compounded_pred_rollout,
                        is_pr_rollout=each_is_pr_rollout,
                        training_time=training_time,
                        rollout_time=rollout_time,
                        trajectory_short_name=trajectory_short_name,
                        save_base_dir=save_base_dir,
                    )
                )

    console_msg_pipeline_footer(f"{ss_model_description} deployment phase done", ".")
    # RLRP-723 F2: aggregate the cumulative L1 MAE over a *common horizon* H = min over all
    # scored InD/OoD trajectories (see execute_ms_model_test_time_rollouts).
    _common_H = _common_horizon(compounded_pred_mae_InD, compounded_pred_mae_OOD)
    return MsTestTimeDeployResult(
        run_stats=tuple(ss_3d_run_stats),
        compounded_pred_mae_target_InD=_cumulative_mae_over_common_horizon(
            compounded_pred_mae_InD, _common_H
        ),
        compounded_pred_mae_target_OOD=_cumulative_mae_over_common_horizon(
            compounded_pred_mae_OOD, _common_H
        ),
    )


def setup_test_rollouts_grid(
    cfg: DictConfig,
    model: mbrl.models.Model,
    target_InD_rollouts: list[TestTrajectoryEntry],
    target_OOD_rollouts: list[TestTrajectoryEntry],
) -> tuple[list[bool], list[bool], list[tuple[TestTrajectoryEntry, bool]]]:

    # .... Setup grid .............................................................................
    test_rollout_env_types: list[tuple[TestTrajectoryEntry, bool]] = []
    if cfg.pipeline.steps.test_rollout_InD:
        for entry in target_InD_rollouts:
            test_rollout_env_types.append((entry, False))

    if cfg.pipeline.steps.test_rollout_OOD:
        for entry in target_OOD_rollouts:
            test_rollout_env_types.append((entry, True))

    probabilistic_rollout = []
    if cfg.pipeline.steps.test_rollout_deterministic:
        probabilistic_rollout.append(False)

    if is_probabilistic_model(model) and cfg.pipeline.steps.test_rollout_probabilistic:
        probabilistic_rollout.append(True)

    compounded_pred_rollouts = []
    if cfg.pipeline.steps.test_rollout_ground_truth:
        compounded_pred_rollouts.append(False)

    if cfg.pipeline.steps.test_rollout_compounded_pred:
        compounded_pred_rollouts.append(True)

    # .... Sanity checks ..........................................................................
    def error_msg(_str):
        return f"pipeline cfg {_str} are set to False! Won't show any rollout"

    if len(test_rollout_env_types) == 0:
        raise ValueError(error_msg("test_rollout_InD and test_rollout_OOD"))

    if len(probabilistic_rollout) == 0:
        raise ValueError(
            error_msg("test_rollout_deterministic and test_rollout_probabilistic")
        )

    if len(compounded_pred_rollouts) == 0:
        raise ValueError(
            error_msg("test_rollout_ground_truth and test_rollout_compounded_pred")
        )

    return compounded_pred_rollouts, probabilistic_rollout, test_rollout_env_types
