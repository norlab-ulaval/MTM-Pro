# coding=utf-8
"""The one and only control-loop step of the test-time rollout deploy path.

Extracted by action ``A2.a`` of the RLRC test-time rollout deployer benchmarking `.junie` plan
(``perf_RLRP-785_testtime_rollout_benchmarking_plan_20260903.md``, rev. 5) from
``rollout_and_stats_collection.py:383-410`` (the model step + the target-env output adapter,
including the ground-truth tutoring branch).

Both the production rollout (``A2``) and the standalone fake-input benchmark (``A5``) call this
function, so the benchmark can never drift away from what production actually executes (the
operator's ``Q2`` decision -- *"use the same component so the benchmark logic evolves along with
the rest of the codebase"*).

Design notes (plan section 3.5 / 3.12):
  * The ``ControlLoopStepResult`` return type encodes the ``R2`` distinction in the type system
    instead of in prose: ``env_pred_next_obs`` is the **published** value (environment frame),
    ``pred_next_obs`` is the **fed-back** value (model single-step space, origin ``:422``). The
    auto-regressive-vs-estimator feedback *selection* remains the caller's responsibility because
    it is loop-index / regime logic; the callable only *produces* the model-space prediction that
    the caller feeds back.
  * ``step_timer=None`` is the production default and costs nothing -- no context manager is
    entered, no branch runs inside a tensor op (invariant (i), the zero-cost OFF path).
  * ``A2.a`` is **pure code motion**: same calls, same order, same tensors, no re-ordering. It is
    guarded by gate ``G14`` and tests ``T23``/``T24``.
"""
from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch


@dataclasses.dataclass(frozen=True)
class ControlLoopStepResult:
    """Result of one control-loop step. Encodes the ``R2`` published/fed-back distinction."""

    env_pred_next_obs: torch.Tensor  # PUBLISHED -- environment frame (origin :406-410)
    pred_next_obs: torch.Tensor  # FED BACK  -- model single-step space (origin :422)
    pred_mean: torch.Tensor  # model_state["ensemble_means"]
    pred_logvar: torch.Tensor  # model_state["ensemble_logvars"]
    model_state: Any  # deployer info payload, forwarded verbatim


@contextmanager
def _window(step_timer, level, *, record: bool) -> Iterator[None]:
    """Enter the timer window for ``level`` -- a true no-op when instrumentation is OFF.

    When ``step_timer is None`` (the production default) nothing but a bare ``yield`` runs, so the
    deploy path stays byte-identical (plan section 3.12 contract 1).
    """
    if step_timer is None:
        yield
        return
    with step_timer.step(level, record=record):
        yield


def control_loop_step(
    deployer,  # MultistepMotionModelTestTimeRolloutDeployer
    *,
    env_to_model_adapter,  # target_env_to_model_ss_in_obs
    model_to_env_adapter,  # model_ss_out_to_target_env_next_obs
    model_ss_in_obs,  # model single-step space: from the estimator OR the previous step
    action,  # controller action
    tutoring_env_next_obs: Optional[torch.Tensor] = None,  # ground-truth-tutoring override (:396-404)
    step_timer=None,  # DeployStepTimer | None -> ZERO cost when None
    record: bool = True,  # rev. 4 (R4) per-step taint flag, forwarded to the timer
    future_actions: Optional[torch.Tensor] = None,  # RLRP-824 6b: plan a_{t+1..t+F-1} (W > H)
) -> ControlLoopStepResult:
    """Run one control-loop step: observation + action -> next environment-frame observation.

    Extracted verbatim from ``rollout_and_stats_collection.py:383-410``. The statement order and
    every call argument are preserved exactly, so the extraction is bit-exact.

    :param deployer: the test-time rollout deployer; its ``predict_next_state`` is the deployer
        step (which itself contains the model call).
    :param env_to_model_adapter: ``model_adapter.target_env_to_model_ss_in_obs`` -- used only on
        the ground-truth-tutoring branch to map the true next env obs into model space.
    :param model_to_env_adapter: ``model_adapter.model_ss_out_to_target_env_next_obs`` -- maps the
        model-space prediction into the environment frame (the *published* value).
    :param model_ss_in_obs: the current observation in model single-step space (from the estimator
        / ground truth, or carried over from the previous prediction in the compounded regime).
    :param action: the controller action at this step.
    :param tutoring_env_next_obs: when not ``None``, the ground-truth-tutoring branch is taken
        (``:388-404``): the *true* next environment observation is mapped into the environment
        frame instead of the model prediction. ``None`` (the deploy default) takes the normal
        branch (``:405-410``).
    :param step_timer: optional :class:`~tools.benchmark_tools.step_timer.DeployStepTimer`. When
        ``None`` the call is byte-identical to the legacy inlined region.
    :param record: per-step taint flag forwarded to the timer (a re-anchor / tutoring step still
        runs, but its latency is not recorded -- plan ``R4``).
    :param future_actions: optional RAW planned future-action sequence ``a_{t+1..t+F-1}``
        forwarded to ``deployer.predict_next_state`` (RLRP-824 Step 6b, operator decision 14:
        an :class:`AbstractMS2MSForecast` on the asymmetric ``W = F > H`` window can only be
        stepped with its plan). ``None`` (default) is the historical call, byte-identical.
    :return: a :class:`ControlLoopStepResult`.
    """
    with _window(step_timer, _CONTROL_LOOP_STEP, record=record):
        # .... Step model (origin :383-386) -- the deployer step contains the model call .........
        with _window(step_timer, _DEPLOYER_STEP, record=record):
            if future_actions is None:
                (
                    pred_next_obs,
                    model_state,
                ) = deployer.predict_next_state(model_ss_in_obs, action)
            else:
                (
                    pred_next_obs,
                    model_state,
                ) = deployer.predict_next_state(
                    model_ss_in_obs, action, future_actions=future_actions
                )

        # .... Map the prediction into the environment frame (origin :388-410) .....................
        if tutoring_env_next_obs is not None:
            # Ground-truth-tutoring branch (:396-404): propagate the ground truth in the collected
            # predictions until the warm-up period ends (drift-resilience measurement).
            env_pred_next_obs, _ = model_to_env_adapter(
                env_to_model_adapter(tutoring_env_next_obs),
                info=model_state,
                last_env_state=model_ss_in_obs,
            )
        else:
            env_pred_next_obs, _ = model_to_env_adapter(
                pred_next_obs, info=model_state, last_env_state=model_ss_in_obs
            )

    return ControlLoopStepResult(
        env_pred_next_obs=env_pred_next_obs,
        pred_next_obs=pred_next_obs,
        pred_mean=model_state["ensemble_means"],
        pred_logvar=model_state["ensemble_logvars"],
        model_state=model_state,
    )


# Lazy level handles: imported at call time so that importing this module never requires the
# benchmark_tools package (keeps the production import graph unchanged when timing is OFF).
try:  # pragma: no cover - trivial import guard
    from tools.benchmark_tools.benchmark_metric import BenchmarkLevel as _BenchmarkLevel

    _CONTROL_LOOP_STEP = _BenchmarkLevel.CONTROL_LOOP_STEP
    _DEPLOYER_STEP = _BenchmarkLevel.DEPLOYER_STEP
except Exception:  # pragma: no cover - benchmark_tools optional at import time
    _CONTROL_LOOP_STEP = "control_loop_step"
    _DEPLOYER_STEP = "deployer_step"
