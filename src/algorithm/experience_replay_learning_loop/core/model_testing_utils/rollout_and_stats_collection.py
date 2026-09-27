# coding=utf-8
import abc
from typing import Any, Optional, Sequence, Tuple, Union

import hydra
import numpy as np
import omegaconf
import torch
from numpy import ndarray
from torch import Tensor

from tools.feature_handling_tools.env_handlers import (
    attach_feature_handler_to_deploy,
)
from tools.r2s_motion_model_container_tools.container import R2SMotionModelContainer
from tools.mbrl_lib_tools.models.one_dim_tr_model_v2 import (
    OneDTransitionRewardModelV2,
    assert_is_OneDTransitionRewardModelV2,
)
from tools.multistep_tools.models import MultiStepMLP
from tools.multistep_tools.models.abstract_ms2ms_forecast import (
    AbstractMS2MSForecast,
)
from tools.multistep_tools.deployer.multistep_motion_model_testtime_rollout_deployer import (
    MultistepMotionModelTestTimeRolloutDeployer,
)
from algorithm.experience_replay_learning_loop.core.model_testing_utils.control_loop_step import (
    control_loop_step,
)
from tools.model_adapter_tools.deployer_adapter import (
    Model2EnvNextObservationAdapter,
)
from tools.r2s_motion_model_container_tools.utils import DeployerAdapter
from trajectory_container_tools.dataclasses.erll_trajectory_dataclass import (
    TestMotionTrajectoryDataclass,
    TestTrajectoryDataclass,
)


def singlestep_model_testtime_rollout_and_collect_pred_stats(
    cfg,
    test_env: Union[TestTrajectoryDataclass, TestMotionTrajectoryDataclass],
    single_step_model: OneDTransitionRewardModelV2,
    torch_rng: torch.Generator,
    compounded_predictions_score: bool = False,
    next_state_deterministic_selection: bool = False,
    next_state_sampling_size: int = 1,
    ground_truth_feed_warmup_steps: int = 0,
    deploy_rollout_post_processing: Optional["Deploy_Rollout_PostProcessing"] = None,
    tutor_and_release: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Executes a test-time rollout using a single-step model in a given test environment,
    while collecting statistics of predicted states. It's meant to be used for comparing the
    multistep model framework against the single step framework baseline.

    This function evaluates a single-step transition model by performing a rollout based on the
    given test environment's trajectory data. Predictions are collected at each step along with
    their mean and logarithmic variance statistics. It allows for optional compounded predictions
    and deterministic sampling for the next state.

    :param cfg: hydra configuration
    :param test_env: Test environment containing trajectory data required for model predictions.
        Must be an instance of the `TestTrajectoryDataclass` or `TestMotionTrajectoryDataclass`.
    :param single_step_model: The single step transition-reward model based on 1D input/output,
        used for making predictions. Must be an instance of `OneDTransitionRewardModelV2`.
    :param torch_rng: Random number generator from PyTorch used to control sampling behavior.
    :param compounded_predictions_score: A boolean flag indicating whether to feed model
        predictions back as inputs for compounding predictions across steps.
    :param next_state_deterministic_selection: A boolean flag to determine whether next states
        should be deterministically sampled, overriding stochastic behavior.
    :param next_state_sampling_size: The number of samples draws if next_state_deterministic_selection=False
    :param ground_truth_feed_warmup_steps: Nb of step before switching to compounded predictions
    :param tutor_and_release: Optionally, propagate the ground truth in collected predictions
        until the warm-up period end. This for measuring the drift resilence of multiple model
        starting # from an trajectory arbitrary timestep.
    :return: A tuple containing a tensor of collected mean predictions made by the model for each
        step of the test trajectory and a tensor of collected logarithmic variance predictions made
         by the model for each step of the test trajectory.
    """

    device = single_step_model.device
    assert isinstance(
        test_env, (TestTrajectoryDataclass, TestMotionTrajectoryDataclass)
    )
    assert not isinstance(single_step_model.model, MultiStepMLP)
    assert_is_OneDTransitionRewardModelV2(single_step_model)

    _model_original_training_state = single_step_model.training

    with torch.inference_mode():
        single_step_model.model.eval()

        # .... Construct input tensor .............................................................
        # C3 — non_blocking H→D copies on the test-time rollout hot path.
        # No-op on CPU (M3 DNA Docker) and MPS; overlaps with compute on CUDA.
        state_tensor = torch.from_numpy(test_env.observations).to(
            device, non_blocking=True
        )
        act_tensor = torch.from_numpy(test_env.actions).to(device, non_blocking=True)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(1)
        if act_tensor.ndim == 1:
            act_tensor = act_tensor.unsqueeze(1)

        # .... Rollout over the entire space ......................................................
        ensemble_size = single_step_model.model.num_members
        rollout_len = test_env.trajectory_len
        ss_pred_obs_ = []
        ss_pred_mean_ = []
        ss_pred_logvar_ = []
        obs = state_tensor[0]
        act = act_tensor[0]

        for each_step in range(0, rollout_len):
            pred_next_obs, model_state = single_step_model.sample_1d_direct_model_call(
                obs=obs,
                act=act,
                rng=torch_rng,
                deterministic=next_state_deterministic_selection,
                next_state_sampling_size=next_state_sampling_size,
            )

            # (Priority) ToDo: implement (ref task RLRP-754 feat: improve deployer initial state history logic)
            if (
                compounded_predictions_score
                and tutor_and_release
                and each_step < ground_truth_feed_warmup_steps
            ):
                # Optionally, propagate the ground truth in collected predictions until the warm-up
                # period end. This for measuring the drift resilence of multiple model starting
                # from an trajectory arbitrary timestep.
                pred_next_obs = state_tensor[each_step + 1]

            ss_pred_obs_.append(pred_next_obs)
            ss_pred_mean_.append(model_state["ensemble_means"])
            ss_pred_logvar_.append(model_state["ensemble_logvars"])

            if each_step + 1 < rollout_len:
                if (
                    compounded_predictions_score
                    and each_step >= ground_truth_feed_warmup_steps
                ):
                    obs = pred_next_obs
                else:
                    obs = state_tensor[each_step + 1]

                act = act_tensor[each_step + 1]

            else:
                break

        batch_dim_idx = 1
        ss_pred_obs = torch.stack(ss_pred_obs_, dim=batch_dim_idx - 1)
        ss_pred_mean = torch.stack(ss_pred_mean_, dim=batch_dim_idx)
        ss_pred_logvar = torch.stack(ss_pred_logvar_, dim=batch_dim_idx)

        if ss_pred_obs.ndim == 1:  # Handle case single_step_features_size=1
            ss_pred_obs = ss_pred_obs.unsqueeze(-1)

        if ss_pred_mean.ndim == 2:  # Handle case single_step_features_size=1
            ss_pred_mean = ss_pred_mean.unsqueeze(-1)
            ss_pred_logvar = ss_pred_logvar.unsqueeze(-1)

        if deploy_rollout_post_processing is None:
            deploy_rollout_post_processing = hydra.utils.instantiate(
                cfg.environment.deploy_rollout_post_processing, cfg, _recursive_=False
            )
            # RLRP-736 S1.5: thread the per-environment feature handler into the
            # locally-instantiated deploy postprocessing (level C). No-op unless
            # the deploy object exposes ``set_feature_handler``.
            attach_feature_handler_to_deploy(cfg, deploy_rollout_post_processing)
        ss_pred_world_pose = deploy_rollout_post_processing(ss_pred_obs, test_env)

        # .... Sanity check .......................................................................
        model_out_size = single_step_model.model.out_size
        assert (
            ss_pred_mean.shape[-1] == model_out_size
        ), f"{ss_pred_mean.shape=} last dimension != {model_out_size=}"
        assert (
            ss_pred_logvar.shape[-1] == model_out_size
        ), f"{ss_pred_logvar.shape=} last dimension != {model_out_size=}"

        # .... teardown ...........................................................................
        single_step_model.train(_model_original_training_state)

        # .... Memory management ..................................................................
        del state_tensor, act_tensor, pred_next_obs, model_state, obs, act
        del ss_pred_obs, ss_pred_obs_, ss_pred_mean_, ss_pred_logvar_

        return ss_pred_world_pose, ss_pred_mean, ss_pred_logvar


def _asymmetric_window_step_plan(
    one_d_tr_model: OneDTransitionRewardModelV2,
    act_tensor: torch.Tensor,
    step: int,
) -> Optional[torch.Tensor]:
    """Ground-truth plan ``a_{t+1..t+F-1}`` fed to the single-step deploy head at ``t = step``
    when -- and only when -- the model runs the asymmetric ``W = F > H`` output window.

    RLRP-824 Step 6b (operator decision 14, "Feed ground-truth plan"): on ``W > H`` the composed
    action columns of the model output ARE the plan, so the plan-free single-step deploy path
    cannot lay the window out (``AbstractMS2MSForecast.forecast`` raises, FR14). The compounded
    rollout therefore feeds the test trajectory's future actions -- the ACTION channel is the
    ground-truth channel of a deployment, exactly as in the ``plan`` branch of
    :func:`multistep_model_testtime_SELFFED_forecast_and_collect_pred_stats`. Near the trajectory
    end the plan is completed by holding the last recorded action (``a_{T-1}``) so the window
    keeps its ``F-1`` slots; the collected metric only ever reads ``ô_{t+1}``.

    Returns ``None`` for every ``F <= H`` model (legacy bit-exact: the compounded single-step
    rollout stays plan-unconditioned there) and for non-forecaster models.
    """
    model = one_d_tr_model.model
    if not (
        isinstance(model, AbstractMS2MSForecast)
        and bool(getattr(model, "asymmetric_output_window", False))
    ):
        return None
    plan_len = int(model.horizon_len) - 1
    if plan_len <= 0:
        return None
    plan = act_tensor[step + 1 : step + 1 + plan_len]
    missing = plan_len - int(plan.shape[0])
    if missing > 0:
        last = act_tensor[-1].unsqueeze(0).expand(missing, -1)
        plan = torch.cat([plan, last], dim=0) if plan.shape[0] > 0 else last
    return plan


def _reanchor_ms2ms_forecaster_on_ground_truth(
    deployer: MultistepMotionModelTestTimeRolloutDeployer,
    state_tensor: torch.Tensor,
    act_tensor: torch.Tensor,
    anchor_step: int,
    history_capacity: int,
    one_d_tr_model: Optional[OneDTransitionRewardModelV2] = None,
) -> None:
    """Anchor a multistep deployer on the ground-truth history **at the warm-up boundary**.

    .. important::
        **Warm-up only.** This helper may be called **once per rollout**, at
        ``t = ground_truth_feed_warmup_steps`` -- i.e. at the hand-over from the
        ground-truth-fed warm-up phase to the self-fed phase. It must **never** be called
        again afterwards.

        The goal of the test-time rollout collectors is to reproduce robotic **deployment**
        conditions: the only ground-truth observations a deployed model ever sees are the ones
        of the start-up / warm-up window. After that hand-over the **actions** are the only
        ground-truth channel left (they are the commands actually executed), and every
        observation the model consumes must be its own prediction -- otherwise the drift curve
        is bounded by the re-anchoring period instead of measuring compounding error.

        Superseded premise (do not restore): an earlier revision re-anchored an
        :class:`AbstractMS2MSForecast` on the ground truth at **every** forecast-window
        boundary (``+ horizon_len``), on the rationale that a non-autoregressive forecaster
        must restart each window from a truthful history. That rationale is wrong for a
        deployment-conditions metric: whether a baseline is autoregressive at *training* time
        is irrelevant to what it is allowed to be fed at *test* time, and the periodic re-feed
        capped the self-fed depth at ``F-1`` steps, turning the published "compounded
        predictions score" of E2E-TCN / M3 / TBM into a bounded ``1..F``-step sawtooth.
        RLRP-757 test-time rollout review follow-up (2026-09-12 operator ruling).

    This helper rebuilds the deployer's history buffer so the prediction issued at
    ``anchor_step`` is conditioned on the **true** ground-truth history window
    ``[anchor_step - history_capacity + 1 .. anchor_step - 1]`` (the anchor obs/action
    itself is appended by the subsequent ``predict_next_state`` call in the rollout
    loop). It resets the deployer with the earliest ground-truth step of the window and
    replays the remaining ground-truth steps through ``predict_next_state`` (interim
    single-step outputs are discarded — only the buffer slide matters).

    :param deployer: the active multistep test-time rollout deployer.
    :param state_tensor: full target-env observation trajectory ``(trajectory_len, obs)``.
    :param act_tensor: full action trajectory ``(trajectory_len, act)``.
    :param anchor_step: the warm-up boundary timestep to anchor on (never a later
        forecast-window boundary -- see the ``.. important::`` note above).
    :param history_capacity: the deployer history buffer capacity (multistep length).
    :param one_d_tr_model: when given, the buffer-sliding ``predict_next_state`` calls receive
        the ground-truth plan of each replayed step on an asymmetric ``W > H`` model
        (:func:`_asymmetric_window_step_plan`; ``None`` on every ``F <= H`` model, bit-exact).
    """
    adapter = deployer.model_adapter
    window_start = max(0, anchor_step - history_capacity + 1)
    deployer.reset(
        state_initialization_value=adapter.target_env_to_model_ss_in_obs(
            state_tensor[window_start]
        ),
        action_initialization_value=act_tensor[window_start],
    )
    # Slide the buffer with the true ground-truth history up to (but excluding) the
    # anchor step; the anchor obs/action is appended by the caller's predict_next_state.
    for gt_step in range(window_start + 1, anchor_step):
        plan = (
            None
            if one_d_tr_model is None
            else _asymmetric_window_step_plan(one_d_tr_model, act_tensor, gt_step)
        )
        if plan is None:
            deployer.predict_next_state(
                adapter.target_env_to_model_ss_in_obs(state_tensor[gt_step]),
                act_tensor[gt_step],
            )
        else:
            deployer.predict_next_state(
                adapter.target_env_to_model_ss_in_obs(state_tensor[gt_step]),
                act_tensor[gt_step],
                future_actions=plan,
            )
    return None


def multistep_model_testtime_rollout_and_collect_pred_stats(
    cfg: omegaconf.DictConfig,
    test_env: Union[TestTrajectoryDataclass, TestMotionTrajectoryDataclass],
    motion_model_container: R2SMotionModelContainer,
    torch_rng: torch.Generator,
    compounded_predictions_score: bool = False,
    next_state_deterministic_selection: bool = False,
    next_state_sampling_size: int = 10,
    ground_truth_feed_warmup_steps: int = 0,
    tutor_and_release: bool = False,
    deploy_rollout_post_processing: Optional["Deploy_Rollout_PostProcessing"] = None,
    *,
    benchmark_step_timing: bool = False,
    benchmark_levels: Sequence[Any] = (),
    benchmark_timing_pass: Any = None,
    benchmark_discard_warmup_steps: int = 0,
    benchmark_exclude_tainted_steps: bool = True,
    benchmark_keep_raw_samples: bool = False,
    benchmark_max_raw_samples: int = 10_000,
    benchmark_out: Optional[list] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs multistep rollout and collects prediction statistics during test time. This
    function is meant for evaluating model performance over multistep prediction horizons.

    The function takes in an environment object defining the test
    trajectory, processes the observations and actions tensors, and uses the
    motion model to predict and record these statistics step-by-step. Users can
    choose whether to compound predictions during rollout.

    .. warning::
        **What ``compounded_predictions_score=True`` measures for an MS->MS forecast baseline
        (E2E-TCN / M3 / TBM) -- and what it does NOT.** Established by the RLRP-757 test-time
        rollout review and its 2026-09-12 follow-up ruling
        (``review_e2e_tcn_testtime_selffed_rollout_report_RLRP-757_20260912.md``).

        The curve **is** a genuine compounding (self-fed) curve -- the ground truth enters the
        deployer history **only during the warm-up phase** -- but it is measured through a
        different model surface than the forecast head:

        - the rollout walks the trajectory **one step at a time** through
          :meth:`MultistepMotionModelTestTimeRolloutDeployer.predict_next_state`, i.e. through
          the **single-step** deploy head -- the model's ``F``-step forecast head is never used;
        - no ``future_actions`` are passed for any ``F <= H`` model, so it is
          **plan-unconditioned** there. **Exception (RLRP-824 Step 6b, operator decision 14)**:
          on the asymmetric ``W = F > H`` output window the composed action columns ARE the plan
          and the plan-free deploy head cannot lay the window out, so every single step feeds the
          test trajectory's ground-truth plan ``a_{t+1..t+F-1}`` (the action channel is the
          ground-truth channel of a deployment; see :func:`_asymmetric_window_step_plan`).

        Fixed on 2026-09-12 (operator ruling, RLRP-757 test-time rollout review follow-up): the
        previous revision re-anchored an :class:`AbstractMS2MSForecast` on the **ground-truth**
        history at every ``horizon_len`` window boundary, capping the self-fed depth at ``F-1``
        single steps and producing a bounded ``1..F``-step sawtooth that was published as a
        compounding curve. Ground-truth anchoring now happens **once**, at
        ``t = ground_truth_feed_warmup_steps`` (see
        ``_reanchor_ms2ms_forecaster_on_ground_truth``), so after the hand-over the **actions**
        are the only ground-truth channel -- as in a real deployment. **Every
        ``compounded_predictions_score`` artifact produced for an MS->MS baseline before that
        date is void and must be re-collected.**

        The headline **forecast-head** self-fed drift curve for those baselines still comes from
        :func:`multistep_model_testtime_SELFFED_forecast_and_collect_pred_stats` (one-shot
        ``F``-step forecast, stride-``F`` re-anchoring on the model's OWN ``ô_{t+F}``, the action
        plan being the only ground-truth channel after warm-up), while
        :func:`multistep_model_testtime_OPENLOOP_forecast_and_collect_pred_stats` measures
        one-shot forecast *accuracy* from a ground-truth-rebuilt history.

    :param cfg: hydra configuration
    :param test_env: Environment object containing test trajectory data, which must
        include observations and actions of the entire trajectory, as well as the
        length of the trajectory.
    :param motion_model_container: Container object for the motion model, which
        includes the dynamics model configured for multitimestep prediction and
        its model adapter.
    :param torch_rng: Torch random number generator used to ensure reproducibility
        of model predictions during stochastic operations in the motion model.
    :param compounded_predictions_score: Indicator for whether to compound model
        predictions during the rollout for state propagation, where the following
        states are recursively predicted from prior predictions instead of using
        ground-truth observations.
    :param next_state_deterministic_selection: Flag to toggle deterministic sampling
        of the next state during the rollout. If enabled, sampling outputs will
        strictly follow deterministic rules set by the motion model.
    :param next_state_sampling_size: The number of samples draws if next_state_deterministic_selection=False
    :param ground_truth_feed_warmup_steps: Nb of step before switching to compounded predictions
    :param tutor_and_release: Optionally, propagate the ground truth in collected predictions
        until the warm-up period end. This for measuring the drift resilence of multiple model
        starting # from an trajectory arbitrary timestep.
    :param deploy_rollout_post_processing:
    :param benchmark_keep_raw_samples: action ``RLRP-803-1`` -- carry the per-step latency
        population into the emitted metrics so the plotter draws TRUE quartiles and the taint
        filtering (``V10``) can be evidenced by comparing the untainted p95/p99 against the
        unfiltered one. Ignored when ``benchmark_step_timing`` is False (the deploy default).
    :param benchmark_max_raw_samples: cap on that population (uniform time-ordered decimation
        above it), so a long rollout cannot blow up the persisted artifact.
    :return: The 3-tuple ``(pred_world_pose, ensemble_means, ensemble_logvars)``.
    """

    ms_model2env = MultistepMotionModelTestTimeRolloutDeployer(
        motion_model_container,
        next_state_deterministic_selection=next_state_deterministic_selection,
        next_state_sampling_size=next_state_sampling_size,
        consol_log=False,
        rng=torch_rng,
    )

    one_d_tr_model: OneDTransitionRewardModelV2 = motion_model_container.dynamics_model
    device = one_d_tr_model.device
    assert_is_OneDTransitionRewardModelV2(one_d_tr_model)
    assert isinstance(
        one_d_tr_model.model, MultiStepMLP
    ), f"Expected MultiStepMLP, got {type(one_d_tr_model.model)=} instead"

    # (CRITICAL) ToDo: validate deleting post_sampling_expectation check (ref task RLRP-456)
    # assert one_d_tr_model.model.propagation_method is not None

    assert isinstance(
        test_env, (TestTrajectoryDataclass, TestMotionTrajectoryDataclass)
    )

    _model_original_training_state = one_d_tr_model.training

    with torch.inference_mode():
        one_d_tr_model.eval()

        # .... Construct input tensor .............................................................
        state_tensor = ndarray_2_torch_tensor(test_env.observations, device)
        act_tensor = ndarray_2_torch_tensor(test_env.actions, device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(1)
        if act_tensor.ndim == 1:
            act_tensor = act_tensor.unsqueeze(1)

        # .... Rollout over the entire space ......................................................
        rollout_len = test_env.trajectory_len

        # .... RLRP-785 (A2.b): optional per-step benchmark timer, OFF by default .................
        # When ``benchmark_step_timing`` is False (the deploy default) ``step_timer`` stays
        # ``None`` and every timer interaction below is a zero-cost no-op, so the rollout is
        # byte-identical (gate G3). ``benchmark_tools`` is imported lazily, only when ON.
        step_timer = None
        if benchmark_step_timing:
            from tools.benchmark_tools.step_timer import DeployStepTimer
            from tools.benchmark_tools.benchmark_metric import BenchmarkLevel, TimingPass

            _levels = tuple(benchmark_levels) or (BenchmarkLevel.CONTROL_LOOP_STEP,)
            _timing_pass = benchmark_timing_pass or TimingPass.DEVICE_TIME
            step_timer = DeployStepTimer(
                enabled=True,
                device=device,
                capacity=rollout_len,
                enabled_levels=_levels,
                timing_pass=_timing_pass,
                keep_raw_samples=bool(benchmark_keep_raw_samples),
                max_raw_samples=int(benchmark_max_raw_samples),
            )
            ms_model2env.step_timer = step_timer  # A8 hook -> enables the ``model_call`` level

        ms_pred_obs_ = []
        ms_pred_mean_ = []
        ms_pred_logvar_ = []
        obs = ms_model2env.model_adapter.target_env_to_model_ss_in_obs(state_tensor[0])
        act = act_tensor[0]

        ms_model2env.reset(
            state_initialization_value=obs, action_initialization_value=act
        )

        if not compounded_predictions_score:
            ground_truth_feed_warmup_steps = np.inf

        # .... MS->MS forecaster ground-truth anchoring, WARM-UP BOUNDARY ONLY ...................
        # The MS->MS forecast baselines (E2E-TCN / M3 / TBM) consume a history WINDOW rather than
        # a single state, so at the warm-up hand-over their deployer rings must hold the TRUE
        # ground-truth window (they are prefilled by ``reset`` with the trajectory's first step,
        # which is not the window preceding ``ground_truth_feed_warmup_steps``). This fires
        # EXACTLY ONCE, at ``t == ground_truth_feed_warmup_steps``.
        #
        # 2026-09-12 operator ruling (RLRP-757 test-time rollout review follow-up): the previous
        # revision re-anchored again at every ``+ horizon_len`` window boundary. That was WRONG --
        # the collector must reproduce deployment conditions, where the only ground-truth
        # OBSERVATIONS are the warm-up ones and the only ground truth left afterwards is the
        # ACTION channel. Whether a baseline is autoregressive at training time does not entitle
        # it to a truthful history at test time; the periodic re-feed capped the self-fed depth at
        # ``F-1`` steps and turned the "compounded predictions score" into a bounded sawtooth.
        # See ``_reanchor_ms2ms_forecaster_on_ground_truth``.
        is_ms2ms_forecaster = isinstance(one_d_tr_model.model, AbstractMS2MSForecast)
        do_forecaster_gt_anchor = is_ms2ms_forecaster and compounded_predictions_score
        history_capacity = int(motion_model_container.multistep_len)

        for each_step in range(0, rollout_len):

            # .... Forecaster logic ...............................................................
            # Anchor the forecaster on the true ground-truth history ONCE, at the warm-up
            # boundary ``t == ground_truth_feed_warmup_steps``. NEVER at a later window
            # boundary: after the hand-over the actions are the only ground-truth channel.
            _gt_anchor_fired = (
                do_forecaster_gt_anchor and each_step == ground_truth_feed_warmup_steps
            )
            if _gt_anchor_fired:
                _reanchor_ms2ms_forecaster_on_ground_truth(
                    ms_model2env,
                    state_tensor,
                    act_tensor,
                    anchor_step=each_step,
                    history_capacity=history_capacity,
                    one_d_tr_model=one_d_tr_model,
                )
                obs = ms_model2env.model_adapter.target_env_to_model_ss_in_obs(
                    state_tensor[each_step]
                )
                act = act_tensor[each_step]

            # .... Step model + target-env output adapter (rev. 5, A2.a) ..........................
            # ``control_loop_step`` is the SHARED callable extracted from the former inlined
            # :383-410 region (plan section 3.12). The standalone bench (A5) calls this same
            # function, so there is exactly one control-loop body in the codebase and the
            # benchmark can never drift from what production executes.
            _tutoring_step = (
                compounded_predictions_score
                and tutor_and_release
                and each_step < ground_truth_feed_warmup_steps
            )
            # rev. 4 (R4): a step that anchors the forecaster on the ground truth (the warm-up
            # boundary step above) or takes the ground-truth-tutoring branch executes work no
            # deployed node executes on that step. It still RUNS (bit-exactness), it is just not
            # RECORDED by the timer.
            record_this_step = (not benchmark_exclude_tainted_steps) or not (
                _gt_anchor_fired or _tutoring_step
            )
            step_result = control_loop_step(
                ms_model2env,
                env_to_model_adapter=ms_model2env.model_adapter.target_env_to_model_ss_in_obs,
                model_to_env_adapter=ms_model2env.model_adapter.model_ss_out_to_target_env_next_obs,
                model_ss_in_obs=obs,
                action=act,
                tutoring_env_next_obs=(
                    state_tensor[each_step + 1] if _tutoring_step else None
                ),
                step_timer=step_timer,  # None-safe: zero cost when instrumentation is OFF
                record=record_this_step,  # rev. 4 (R4) taint flag, forwarded
                # RLRP-824 6b: ground-truth plan on the asymmetric W > H window ONLY (None else)
                future_actions=_asymmetric_window_step_plan(
                    one_d_tr_model, act_tensor, each_step
                ),
            )
            pred_next_obs = step_result.pred_next_obs
            env_pred_next_obs = step_result.env_pred_next_obs
            model_state = step_result.model_state

            # (Priority) ToDo: implement (ref task RLRP-754 feat: improve deployer initial state history logic)
            ms_pred_obs_.append(env_pred_next_obs)
            ms_pred_mean_.append(step_result.pred_mean)
            ms_pred_logvar_.append(step_result.pred_logvar)

            if each_step + 1 < rollout_len:
                if (
                    compounded_predictions_score
                    and each_step >= ground_truth_feed_warmup_steps
                ):
                    obs = pred_next_obs
                else:
                    obs = ms_model2env.model_adapter.target_env_to_model_ss_in_obs(
                        state_tensor[each_step + 1]
                    )

                act = act_tensor[each_step + 1]

            else:
                break

        # .... RLRP-785 (A2.b): reduce the per-step timings into a BenchmarkMetricSet ............
        # The populations are SPLIT at ``ground_truth_feed_warmup_steps`` (pre = ESTIMATOR-fed,
        # post = AUTOREGRESSIVE); never a single ``MIXED`` blob (plan R3). The report is handed
        # back through the caller-supplied ``benchmark_out`` container so the return tuple stays
        # unchanged (gate G3 -- OFF path byte-identical).
        if step_timer is not None:
            from tools.benchmark_tools.benchmark_metric import FeedbackRegime

            _split_index = (
                int(ground_truth_feed_warmup_steps)
                if (
                    compounded_predictions_score
                    and np.isfinite(ground_truth_feed_warmup_steps)
                )
                else None
            )
            _report = step_timer.report(
                feedback_regime=FeedbackRegime.ESTIMATOR,
                regime_split_index=_split_index,
                discard_warmup_steps=int(benchmark_discard_warmup_steps),
                provenance={
                    "rollout_len": int(rollout_len),
                    "ground_truth_feed_warmup_steps": (
                        None
                        if not np.isfinite(ground_truth_feed_warmup_steps)
                        else int(ground_truth_feed_warmup_steps)
                    ),
                },
            )
            if benchmark_out is not None:
                benchmark_out.append(_report)
            ms_model2env.step_timer = None  # rev. 4 (R11): drop the hook before teardown
            step_timer.close()  # release the pre-allocated event pools deterministically

        batch_dim_idx = 1
        ms_pred_obs = torch.stack(ms_pred_obs_, dim=batch_dim_idx - 1)
        ms_pred_mean = torch.stack(ms_pred_mean_, dim=batch_dim_idx)
        ms_pred_logvar = torch.stack(ms_pred_logvar_, dim=batch_dim_idx)

        if ms_pred_obs.ndim == 1:  # Handle case single_step_features_size=1
            ms_pred_obs = ms_pred_obs.unsqueeze(-1)
        if ms_pred_mean.ndim == 2:  # Handle case single_step_features_size=1
            ms_pred_mean = ms_pred_mean.unsqueeze(-1)
            ms_pred_logvar = ms_pred_logvar.unsqueeze(-1)

        if deploy_rollout_post_processing is None:
            deploy_rollout_post_processing = hydra.utils.instantiate(
                cfg.environment.deploy_rollout_post_processing, cfg, _recursive_=False
            )
            # RLRP-736 S1.5: thread the per-environment feature handler into the
            # locally-instantiated deploy postprocessing (level C). No-op unless
            # the deploy object exposes ``set_feature_handler``.
            attach_feature_handler_to_deploy(cfg, deploy_rollout_post_processing)
        ms_pred_world_pose = deploy_rollout_post_processing(ms_pred_obs, test_env)

        # .... teardown ...........................................................................
        one_d_tr_model.train(_model_original_training_state)

        # .... Memory management ..................................................................
        # ToDo: on task end >> UN-mute next bloc ↓↓
        del state_tensor, act_tensor, pred_next_obs, model_state, obs, act
        del ms_pred_obs_, ms_pred_mean_, ms_pred_logvar_, ms_model2env

        return ms_pred_world_pose, ms_pred_mean, ms_pred_logvar


def assert_target_env_next_obs_adapter_configured(
    adapter: DeployerAdapter,
) -> None:
    """Fail-fast T1 guard (RLRP-728 plan section 3.2.2): the target-env metric requires a
    properly-configured, **non-identity** ``model_ss_out_to_target_env_next_obs`` adapter.

    The :class:`DeployerAdapter` default for that field is the bare identity lambda
    ``lambda next_obs, info, last_env_state: (next_obs, info)``. If it is left at that default,
    "target-env observation space" silently collapses to the model output space and the
    cross-model "same basis" requirement becomes ill-defined -- so we **raise** here. A properly
    configured adapter (even one whose mapping happens to be numerically identity) is a
    :class:`Model2EnvNextObservationAdapter` instance and passes this guard.

    :param adapter: the container deployer adapter to audit.
    :raises AssertionError: when ``model_ss_out_to_target_env_next_obs`` is not a
        :class:`Model2EnvNextObservationAdapter` (i.e. still the identity default / a bare callable).
    """
    assert isinstance(
        adapter.model_ss_out_to_target_env_next_obs, Model2EnvNextObservationAdapter
    ), (
        "The target-env-space per-horizon metric (RLRP-728 S7) requires "
        "`DeployerAdapter.model_ss_out_to_target_env_next_obs` to be a properly-configured "
        "`Model2EnvNextObservationAdapter` (NOT the identity default). Got "
        f"{type(adapter.model_ss_out_to_target_env_next_obs)!r}. Without it, target-env space "
        "collapses to model-output space and models are not comparable on the same basis."
    )
    return None


class CumulativePerHorizonMAE:
    """Step-driven cumulative per-horizon MAE accumulator (RLRP-728 plan section 3.2.2, T3).

    Holds a running absolute-error **sum** and **count** per forecast step ``h`` across all
    anchors. Finalized on termination into the cumulative per-step MAE vector ``(horizon_len,)``
    in target-env observation units. This matches the operator directive: *execute the MS forecast
    by forecast step and, on termination, compute the cumulative MAE step by step*.

    RLRP-760 S0: the accumulator axis is the ``horizon_len`` (``F``) **forecast** axis, not the
    ``history_len`` (``H``) output-window axis (the two coincide only when ``F == H``).
    """

    def __init__(self, horizon_len: int, device: Union[str, torch.device]) -> None:
        self._horizon_len = horizon_len
        self._sum = torch.zeros(horizon_len, device=device)
        self._cnt = torch.zeros(horizon_len, device=device)
        self._num_updates = 0

    def update(
        self, h: int, pred_env_obs: torch.Tensor, gt_env_obs: torch.Tensor
    ) -> None:
        """Accumulate ``|pred - gt|`` for forecast step ``h`` (1-based), in target-env space."""
        idx = h - 1
        self._sum[idx] = self._sum[idx] + (pred_env_obs - gt_env_obs).abs().sum()
        self._cnt[idx] = self._cnt[idx] + pred_env_obs.numel()
        self._num_updates += 1

    @property
    def is_empty(self) -> bool:
        return self._num_updates == 0

    def finalize(self) -> torch.Tensor:
        """Return the cumulative per-step MAE ``(horizon_len,)`` (per-step sum / count).

        Empty (no anchor processed) -> an empty tensor, matching the collector contract.
        """
        if self.is_empty:
            return torch.empty(0, device=self._sum.device)
        # Steps with a zero count would only occur if never updated; clamp to avoid 0/0 NaN.
        return self._sum / self._cnt.clamp_min(1)


def multistep_model_testtime_OPENLOOP_forecast_and_collect_pred_stats(
    cfg: omegaconf.DictConfig,
    test_env: Union[TestTrajectoryDataclass, TestMotionTrajectoryDataclass],
    motion_model_container: R2SMotionModelContainer,
    torch_rng: torch.Generator,
    *,
    feed_future_action_plan: bool = True,
    future_action_conditioning: str = "plan",
) -> torch.Tensor:
    """Open-loop, control-conditioned per-horizon forecast evaluation (RLRP-728).

    Sibling of :func:`multistep_model_testtime_rollout_and_collect_pred_stats` (which advances the
    model one single-step at a time). This collector instead slides a window over the test
    trajectory and, at every valid anchor ``t``, queries the model's **full per-horizon** forecast
    ``ô_{t+1..t+F}`` (``F = horizon_len``, the forecast length -- RLRP-760 S0; the anchor window
    still needs ``H = history_len`` past steps) conditioned on the horizon's **planned future
    actions** ``a_{t+1..t+F-1}`` (shape ``(horizon_len - 1, singlestep_act_len)``), then reports
    the per-horizon MAE against the ground-truth trajectory.

    Permanent: the plan channel is ``F-1``-long because the driving action sequence is
    ``[a_t taken from the history] ++ plan``; ``a_{t+F}`` is never needed. ``F == 1`` therefore
    means an empty plan (``None``). Introduced by stage `A1` of the Fix the MS→MS
    future-action-plan conditioning contract `.junie` plan
    (fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md).

    Design (RLRP-728, REV 4 target-env-space metric / S7):

    - The single-step deploy path (:meth:`predict_next_state`) is untouched; this drives the new
      step-driven :meth:`MultistepMotionModelTestTimeRolloutDeployer.forecast_horizon_target_env`
      session **one forecast-step at a time**.
    - The metric is a **HARD REQUIREMENT in target-env observation space** (the raw units of
      ``test_env.observations``), so every model (E2E-TCN / M3 / TBM and any future baseline) is
      compared on the **same basis** (plan section 3.2.2, supersedes the REV 3 model-obs-space
      option-2 metric). Each model single-step output obs is mapped to target-env space via the
      asymmetric ``DeployerAdapter.model_ss_out_to_target_env_next_obs`` adapter (fail-fast T1:
      :func:`assert_target_env_next_obs_adapter_configured`); ground truth is the **raw**
      ``test_env.observations[t+h]`` (no mapping needed -- that is the same basis).
    - The per-step absolute error is accumulated across all anchors by
      :class:`CumulativePerHorizonMAE` and finalized on termination into the cumulative per-step
      MAE (T3).
    - Tail policy (plan section 3.3): anchors with fewer than ``horizon_len`` future steps
      remaining are **skipped** (deterministic MAE-vs-ground-truth).

    :param feed_future_action_plan: when ``False`` the plan channel is disabled (history-echo /
        ``obs_only`` baseline), regardless of ``future_action_conditioning``.
    :param future_action_conditioning: one of ``{"plan", "last_action_hold", "obs_only"}`` (plan
        section 3.3 fallbacks). ``"plan"`` feeds the true future actions ``a_{t+1..t+F-1}``
        (shape ``(horizon_len - 1, singlestep_act_len)``); ``"last_action_hold"`` repeats ``a_t``
        ``horizon_len - 1`` times (length-consistent with ``"plan"`` -- required for the RLRP-760
        comparison to stay meaningful); ``"obs_only"`` feeds no plan (history echo). With
        ``horizon_len == 1`` both ``"plan"`` and ``"last_action_hold"`` degenerate to ``None``.
    :return: the cumulative per-horizon MAE tensor of shape ``(horizon_len,)`` in **target-env**
        observation units. Empty trajectories (no valid anchor) yield an empty tensor.

    PERF (deferred, RLRP-730): readability-first per-anchor Python loop with per-anchor buffer
    slide; batch the anchors once profiling justifies it.
    """
    assert future_action_conditioning in ("plan", "last_action_hold", "obs_only"), (
        f"future_action_conditioning must be one of plan|last_action_hold|obs_only, "
        f"got {future_action_conditioning!r}."
    )
    assert isinstance(
        test_env, (TestTrajectoryDataclass, TestMotionTrajectoryDataclass)
    )

    ms_model2env = MultistepMotionModelTestTimeRolloutDeployer(
        motion_model_container,
        next_state_deterministic_selection=True,
        next_state_sampling_size=1,
        consol_log=False,
        rng=torch_rng,
    )

    one_d_tr_model: OneDTransitionRewardModelV2 = motion_model_container.dynamics_model
    device = one_d_tr_model.device
    assert_is_OneDTransitionRewardModelV2(one_d_tr_model)
    assert isinstance(
        one_d_tr_model.model, MultiStepMLP
    ), f"Expected MultiStepMLP, got {type(one_d_tr_model.model)=} instead"

    history_len = one_d_tr_model.model.history_len
    horizon_len = one_d_tr_model.model.horizon_len
    adapter = ms_model2env.model_adapter

    # T1 fail-fast: target-env space is only well-defined with a configured (non-identity)
    # model_ss_out_to_target_env_next_obs adapter (plan section 3.2.2).
    assert_target_env_next_obs_adapter_configured(adapter)

    _model_original_training_state = one_d_tr_model.training

    with torch.inference_mode():
        one_d_tr_model.eval()

        state_tensor = ndarray_2_torch_tensor(test_env.observations, device)
        act_tensor = ndarray_2_torch_tensor(test_env.actions, device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(1)
        if act_tensor.ndim == 1:
            act_tensor = act_tensor.unsqueeze(1)

        trajectory_len = test_env.trajectory_len

        def _model_ss_obs(step_idx: int) -> torch.Tensor:
            return adapter.target_env_to_model_ss_in_obs(state_tensor[step_idx])

        ms_model2env.reset(
            state_initialization_value=_model_ss_obs(0),
            action_initialization_value=act_tensor[0],
        )

        accumulator = CumulativePerHorizonMAE(horizon_len, device)
        for t in range(0, trajectory_len):
            # An anchor is valid once the buffer holds ``history_len`` real steps AND the full
            # ground-truth forecast ``t+1..t+horizon_len`` exists (tail policy: else skip).
            is_valid_anchor = (t >= history_len - 1) and (
                t + horizon_len <= trajectory_len - 1
            )

            if not is_valid_anchor:
                # Warm up / slide the history buffer by one without collecting. RLRP-824 6b: an
                # asymmetric ``W > H`` model needs its ground-truth plan even for this discarded
                # step (``None`` on every ``F <= H`` model, bit-exact).
                ms_model2env.predict_next_state(
                    _model_ss_obs(t),
                    act_tensor[t],
                    future_actions=_asymmetric_window_step_plan(one_d_tr_model, act_tensor, t),
                )
                continue

            # Permanent: the plan channel is ``a_{t+1..t+F-1}`` i.e. ``horizon_len - 1`` actions
            # (``a_t`` already comes from the history), and it is empty (``None``) when
            # ``horizon_len == 1``. Both the ``plan`` and ``last_action_hold`` modes MUST stay
            # length-consistent. Introduced by stage `A1` of the Fix the MS→MS future-action-plan
            # conditioning contract `.junie` plan
            # (fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md).
            if (
                not feed_future_action_plan
                or future_action_conditioning == "obs_only"
                or horizon_len == 1
            ):
                future_actions = None
            elif future_action_conditioning == "last_action_hold":
                future_actions = act_tensor[t].unsqueeze(0).repeat(horizon_len - 1, 1)
            else:  # "plan"
                future_actions = act_tensor[t + 1 : t + horizon_len]

            # Step-driven target-env forecast (T2/T3): the model stays open-loop / non-AR; only the
            # coordinate reconstruction is chained per step. Ground truth is the RAW target-env obs
            # ``state_tensor[t+h]`` (same basis -- no mapping).
            for h, env_obs_h in ms_model2env.forecast_horizon_target_env(
                _model_ss_obs(t),
                act_tensor[t],
                future_actions,
                anchor_env_state=state_tensor[t],
            ):
                accumulator.update(h, env_obs_h, state_tensor[t + h])

        one_d_tr_model.train(_model_original_training_state)

        # Cumulative per-step MAE, finalized on termination (T3). Empty -> empty tensor.
        return accumulator.finalize()


def multistep_model_testtime_SELFFED_forecast_and_collect_pred_stats(
    cfg: omegaconf.DictConfig,
    test_env: Union[TestTrajectoryDataclass, TestMotionTrajectoryDataclass],
    motion_model_container: R2SMotionModelContainer,
    torch_rng: torch.Generator,
    *,
    ground_truth_feed_warmup_steps: int = 0,
    feed_future_action_plan: bool = True,
    future_action_conditioning: str = "plan",
) -> torch.Tensor:
    """Compounded, **self-fed** MS-forecaster rollout evaluation (RLRP-760).

    The deployment-resilience metric for the MS->MS forecast baselines (E2E-TCN / M3 / TBM) and
    the MTM-Pro family. Unlike its two siblings:

    - :func:`multistep_model_testtime_OPENLOOP_forecast_and_collect_pred_stats` (RLRP-728) always
      rebuilds the history from **ground truth**, so it measures one-shot forecast *accuracy*, not
      closed-loop resilience;
    - :func:`multistep_model_testtime_rollout_and_collect_pred_stats` is **also** self-fed after
      the warm-up hand-over (ground truth anchors the deployer **once**, at
      ``t = ground_truth_feed_warmup_steps``), but it advances **one single step at a time**
      through the **single-step** deploy head and passes **no action plan** -- so it never
      exercises the ``F``-step forecast head this collector measures.

    this collector rolls the forecaster **closed-loop**: at each window anchor ``t`` it emits the
    full ``F``-step forecast ``ô_{t+1..t+F}`` (``F = horizon_len``, RLRP-760 S0) and feeds those
    predictions -- together with the planned actions ``a_{t+1..t+F-1}``, shape
    ``(horizon_len - 1, singlestep_act_len)`` -- **back into the history
    buffers** before hopping the window by ``F`` (plan §3.2 option A, full-forecast hop). Ground
    truth is used **only** for the warm-up prefix; there is **no** per-``horizon_len`` GT
    re-anchoring, so drift compounds exactly as it would under mobile-robotic deployment.

    Concretely, with ``H = 20`` / ``F = 10``: the history holds ``o_{t-19..t}`` paired with
    ``a_{t-19..t}``, the model forecasts ``ô_{t+1..t+10}``, the window shifts LEFT by 10 and the
    10 **predicted** observations (never the ground-truth ones) are appended, paired with the
    executed plan ``a_{t+1..t+10}`` -- ``a_{t+1..t+9}`` from the plan channel and ``a_{t+10}``
    as the next window's anchor action -- and the next forecast is issued from that fully
    self-fed history. Ground truth enters **only** through the warm-up prefix and the **action**
    channel (the plan ``a_{t+1..t+F-1}`` plus the next anchor action ``a_{t+F}`` -- the commands
    actually executed); no ground-truth **observation** is ever re-fed after the hand-over, as in
    a real robotic deployment. Hence the hop is ``F`` steps, unlike
    :func:`multistep_model_testtime_rollout_and_collect_pred_stats`, which hops by 1.

    The model forward itself stays **non-autoregressive within a window** (the anti-compounding
    property of the reference papers is preserved); the compounding is *across* windows.

    Scoring (plan §3.3):

    - **HARD REQUIREMENT: target-env observation space** (the raw units of
      ``test_env.observations``) so every model is compared on the **same basis**; enforced by
      :func:`assert_target_env_next_obs_adapter_configured`. Each predicted target-env obs is
      computed **once** and reused for scoring, for the ``last_env_state`` reconstruction chain and
      for the model-input feedback (plan §3.1.1).
    - The reported metric is the **per-global-trajectory-step** MAE sequence (feature axis reduced
      by ``mean``, i.e. the same reduction as
      ``pipeline.pipeline_utils.general.train_and_deploy_utils._per_step_feature_mae``), NOT the
      per-``h`` :class:`CumulativePerHorizonMAE`. Only that axis is genuinely comparable to the
      existing compounded single-step / MTM-Pro drift curves, which is the whole point of RLRP-760.

    Tail policy (plan §3.4): the rollout **stops** once fewer than ``F`` ground-truth steps remain
    for the current hop; no right-padding.

    Determinism (plan §3.5): the deployer is forced to
    ``next_state_deterministic_selection=True`` / ``next_state_sampling_size=1``.

    :param ground_truth_feed_warmup_steps: number of leading ground-truth steps ``W`` fed to the
        history before the model starts consuming its own forecast. The first self-fed window is
        anchored at ``t = max(W, history_len - 1)`` (the buffer must hold ``history_len`` real
        steps first) and the hop schedule is then ``t, t+F, t+2F, ...``.
    :param feed_future_action_plan: when ``False`` the plan channel is disabled (history-echo /
        ``obs_only`` baseline), regardless of ``future_action_conditioning``.
    :param future_action_conditioning: one of ``{"plan", "last_action_hold", "obs_only"}``.
        ``"plan"`` feeds ``a_{t+1..t+F-1}`` (shape ``(horizon_len - 1, singlestep_act_len)``),
        ``"last_action_hold"`` repeats ``a_t`` ``horizon_len - 1`` times (length-consistent with
        ``"plan"``), ``"obs_only"`` feeds no plan. With ``horizon_len == 1`` both plan modes
        degenerate to ``None``.

        Permanent: Introduced by stage `A1` of the Fix the MS→MS future-action-plan conditioning
        contract `.junie` plan
        (fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md).
    :return: the per-global-step MAE sequence in **target-env** observation units, shape
        ``(num_scored_steps,)`` where entry ``i`` scores global trajectory step ``t_0 + 1 + i``.
        Trajectories too short to host a single full window yield an empty tensor.

    PERF (deferred, RLRP-730): readability-first per-window Python loop.
    """
    assert future_action_conditioning in ("plan", "last_action_hold", "obs_only"), (
        f"future_action_conditioning must be one of plan|last_action_hold|obs_only, "
        f"got {future_action_conditioning!r}."
    )
    assert isinstance(
        test_env, (TestTrajectoryDataclass, TestMotionTrajectoryDataclass)
    )

    ms_model2env = MultistepMotionModelTestTimeRolloutDeployer(
        motion_model_container,
        next_state_deterministic_selection=True,
        next_state_sampling_size=1,
        consol_log=False,
        rng=torch_rng,
    )

    one_d_tr_model: OneDTransitionRewardModelV2 = motion_model_container.dynamics_model
    device = one_d_tr_model.device
    assert_is_OneDTransitionRewardModelV2(one_d_tr_model)
    assert isinstance(
        one_d_tr_model.model, MultiStepMLP
    ), f"Expected MultiStepMLP, got {type(one_d_tr_model.model)=} instead"

    history_len = int(one_d_tr_model.model.history_len)
    horizon_len = int(one_d_tr_model.model.horizon_len)
    adapter = ms_model2env.model_adapter

    # T1 fail-fast: target-env space is only well-defined with a configured (non-identity) adapter.
    assert_target_env_next_obs_adapter_configured(adapter)

    _model_original_training_state = one_d_tr_model.training

    with torch.inference_mode():
        one_d_tr_model.eval()

        state_tensor = ndarray_2_torch_tensor(test_env.observations, device)
        act_tensor = ndarray_2_torch_tensor(test_env.actions, device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(1)
        if act_tensor.ndim == 1:
            act_tensor = act_tensor.unsqueeze(1)

        trajectory_len = test_env.trajectory_len

        def _model_ss_obs(step_idx: int) -> torch.Tensor:
            return adapter.target_env_to_model_ss_in_obs(state_tensor[step_idx])

        ms_model2env.reset(
            state_initialization_value=_model_ss_obs(0),
            action_initialization_value=act_tensor[0],
        )

        # .... Warm-up prefix (ground truth only) .................................................
        # The buffer must hold ``history_len`` real steps before the first forecast, so the anchor
        # is pushed out to ``history_len - 1`` when the requested warm-up is shorter.
        anchor_step = max(int(ground_truth_feed_warmup_steps), history_len - 1)
        for gt_step in range(0, min(anchor_step, trajectory_len)):
            # Slides the history ring with ground truth; the single-step output is discarded.
            # RLRP-824 6b: an asymmetric ``W > H`` model needs its ground-truth plan even here.
            ms_model2env.predict_next_state(
                _model_ss_obs(gt_step),
                act_tensor[gt_step],
                future_actions=_asymmetric_window_step_plan(
                    one_d_tr_model, act_tensor, gt_step
                ),
            )

        # .... Self-fed windows (no ground-truth re-anchoring) ....................................
        per_step_mae = []
        is_first_window = True
        anchor_model_obs = None
        anchor_env_obs = None
        while anchor_step + horizon_len <= trajectory_len - 1:
            if is_first_window:
                anchor_model_obs = _model_ss_obs(anchor_step)
                anchor_env_obs = state_tensor[anchor_step]

            # Permanent: the plan channel is ``a_{t+1..t+F-1}`` i.e. ``horizon_len - 1`` actions
            # (``a_t`` already comes from the history), and it is empty (``None``) when
            # ``horizon_len == 1``. Both the ``plan`` and ``last_action_hold`` modes MUST stay
            # length-consistent. Introduced by stage `A1` of the Fix the MS→MS future-action-plan
            # conditioning contract `.junie` plan
            # (fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md).
            if (
                not feed_future_action_plan
                or future_action_conditioning == "obs_only"
                or horizon_len == 1
            ):
                future_actions = None
            elif future_action_conditioning == "last_action_hold":
                future_actions = (
                    act_tensor[anchor_step].unsqueeze(0).repeat(horizon_len - 1, 1)
                )
            else:  # "plan"
                future_actions = act_tensor[anchor_step + 1 : anchor_step + horizon_len]

            # ``act_tensor[anchor_step]`` is the anchor action ``a_t`` of THIS window: for a
            # subsequent window that is the ground-truth action executed at the step whose
            # observation the model itself predicted. The deployer pushes that pair (predicted
            # obs + true anchor action) into the rings, which is what keeps the LAST history
            # action slot -- the one the model reads as ``a_t`` -- correct across the hop.
            for h, env_obs_h in ms_model2env.forecast_horizon_selffed_target_env(
                anchor_model_obs,
                act_tensor[anchor_step],
                future_actions,
                anchor_env_state=anchor_env_obs,
            ):
                gt_env_obs = state_tensor[anchor_step + h]
                # Feature-mean absolute error => a true per-step MAE in target-env units.
                per_step_mae.append(
                    (env_obs_h.ravel() - gt_env_obs.ravel()).abs().mean()
                )
                # The last forecast step becomes the next window's anchor (self-fed): steps
                # ``1..F-1`` were pushed into the rings as feedback, and this ``F``-th one is
                # pushed by the NEXT window's anchor append -- paired there with the genuine
                # anchor action ``a_{t+F}`` instead of a held plan entry.
                anchor_env_obs = env_obs_h
                anchor_model_obs = adapter.target_env_to_model_ss_in_obs(env_obs_h)

            is_first_window = False
            anchor_step += horizon_len

        one_d_tr_model.train(_model_original_training_state)

        if not per_step_mae:
            return torch.empty(0, device=device)
        return torch.stack(per_step_mae)


class Deploy_Rollout_PostProcessing(abc.ABC):

    def __init__(self, cfg, device, feature_handler=None, **kwargs):
        self._cfg = cfg
        self._device = device
        # RLRP-736 S1.4: optional per-environment feature handler. When set, its
        # group-wise ``deploy_reconstruct`` fixups (e.g. quaternion unit-norm
        # re-projection) run BEFORE the env-specific ``deploy_adapter``
        # integrator. ``None`` (default) keeps the deploy path byte-identical to
        # the legacy behaviour (``deploy_reconstruct`` is identity by default).
        self._feature_handler = feature_handler

    def set_feature_handler(self, feature_handler) -> None:
        """Register the per-environment feature handler (RLRP-736 S1.4)."""
        self._feature_handler = feature_handler

    def get_feature_handler(self):
        """Return the registered per-environment feature handler (or ``None``)."""
        return self._feature_handler

    @abc.abstractmethod
    def deploy_adapter(
        self,
        pred_obs: Union[np.ndarray, torch.Tensor],
        test_env: TestMotionTrajectoryDataclass,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        pass

    def __call__(
        self,
        pred_obs: Union[np.ndarray, torch.Tensor],
        test_env: TestMotionTrajectoryDataclass,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(pred_obs, (np.ndarray, torch.Tensor))
        assert isinstance(test_env, TestMotionTrajectoryDataclass)

        # RLRP-736 S1.4: per-group deploy fixups (model-external reconstruction,
        # level C) run before the env-specific integrator. Neutral (identity)
        # unless a handler declares a non-trivial ``deploy_reconstruct``; only
        # applied to torch tensors so numpy-only adapters are untouched.
        if self._feature_handler is not None and isinstance(pred_obs, torch.Tensor):
            pred_obs = self._feature_handler.deploy_reconstruct(pred_obs)

        # Torch-first: pass tensors directly to the adapter. Adapters that support
        # torch tensors natively (e.g., MathEnv) will operate without conversion.
        # Adapters that require numpy (e.g., Quadcopter) should convert internally.
        pred_obs = self.deploy_adapter(pred_obs, test_env, **kwargs)

        if isinstance(pred_obs, np.ndarray):
            pred_obs = ndarray_2_torch_tensor(pred_obs, self._device)
        return pred_obs


def ndarray_2_torch_tensor(numpy_array: ndarray, device) -> Tensor:
    # C3 — non_blocking H→D copy; no-op on CPU/MPS, overlaps with
    # compute on CUDA when the host tensor is in pinned memory.
    return torch.from_numpy(numpy_array).to(device, non_blocking=True)


def torch_tensor_2_ndarray(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().numpy()
