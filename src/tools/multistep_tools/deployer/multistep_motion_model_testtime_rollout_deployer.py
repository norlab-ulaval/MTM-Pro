# coding=utf-8
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import numpy as np
import torch

from tools.multistep_tools.deployer.abstract import (
    AbstractMultistepMotionModelDeployer,
    CUDA_STREAM,
    use_cuda_stream,
)
from tools.r2s_motion_model_container_tools.container import R2SMotionModelContainer


@contextmanager
def _maybe_time_model_call(step_timer) -> Iterator[None]:
    """Open the ``MODEL_CALL`` timer window -- a true no-op when instrumentation is OFF.

    RLRP-785 (A8): when ``step_timer is None`` (the production default) nothing but a bare
    ``yield`` runs, so the deploy path stays byte-identical and never imports ``benchmark_tools``.
    The ``BenchmarkLevel`` handle is imported lazily so importing this deployer never requires the
    benchmark package.
    """
    if step_timer is None:
        yield
        return
    from tools.benchmark_tools.benchmark_metric import BenchmarkLevel

    with step_timer.step(BenchmarkLevel.MODEL_CALL):
        yield


class MultistepMotionModelTestTimeRolloutDeployer(AbstractMultistepMotionModelDeployer):
    cuda_stream: None

    def __init__(
        self,
        motion_model_container: R2SMotionModelContainer,
        next_state_deterministic_selection: bool = True,
        next_state_sampling_size: int = True,
        consol_log: bool = True,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__(
            motion_model_container,
            next_state_deterministic_selection,
            next_state_sampling_size,
            consol_log,
            rng,
        )

    # @use_cuda_stream(CUDA_STREAM)
    def predict_next_state(
        self,
        env_state: Union[torch.Tensor, np.ndarray],
        action: Union[torch.Tensor, np.ndarray],
        future_actions: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Predict next single-step environment state

        Handle the multistep buffers and the model-2-model adapter.

        :param env_state: The current environment state at timestep t.
        :param action: The action executed at timestep t.
        :param future_actions: optional RAW planned future-action sequence ``a_{t+1..t+F-1}`` of
            shape ``(horizon_len - 1, singlestep_act_len)`` (RLRP-824 Step 6b, operator decision
            14). Required by an :class:`AbstractMS2MSForecast` on the asymmetric ``W = F > H``
            output window (its composed action columns are the plan); forwarded to
            ``OneDTransitionRewardModelV2.sample_1d_direct_model_call``, which normalizes it like
            the training path. ``None`` (default) keeps the historical plan-free single-step
            deploy path byte-identical.
        :return: The single-step next state prediction array and the mbrl-model state dictionary.
        """
        with torch.inference_mode():
            self.one_d_tr_model.eval()

            # Torch-first: convert numpy inputs to tensors if needed
            if isinstance(env_state, np.ndarray):
                env_state = self._ndarray_to_tensor(env_state)
            if isinstance(action, np.ndarray):
                action = self._ndarray_to_tensor(action)
            if isinstance(future_actions, np.ndarray):
                future_actions = self._ndarray_to_tensor(future_actions)

            self._multistep_state_buffer.append(env_state)
            self._multistep_act_buffer.append(action)

            multistep_state = self._multistep_state_buffer.get_buffer()
            multistep_act = self._multistep_act_buffer.get_buffer()

            # RLRP-785 (A8): the ``model_call`` level -- the pure model inference, the region
            # RLRP-786/787 accelerate. ``_maybe_time_model_call(None)`` is a zero-cost no-op, so
            # this stays byte-identical on the default deploy path.
            with _maybe_time_model_call(self.step_timer):
                (
                    next_state_pred,
                    model_state_info,
                ) = self.one_d_tr_model.sample_1d_direct_model_call(
                    obs=multistep_state,
                    act=multistep_act,
                    rng=self.mbrl_rng,
                    propagation_indices=None,
                    deterministic=self.next_state_deterministic_selection,
                    next_state_sampling_size=self.next_state_sampling_size,
                    future_actions=future_actions,
                )

            next_state_pred, model_state_info = self._adapter.model_to_model_ss_obs(
                next_state_pred.ravel(), info=model_state_info
            )

            return (
                next_state_pred,
                model_state_info,
            )

    def forecast_horizon(
        self,
        env_state: Union[torch.Tensor, np.ndarray],
        action: Union[torch.Tensor, np.ndarray],
        future_actions: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> torch.Tensor:
        """Open-loop per-horizon observation forecast (RLRP-728).

        Companion to :meth:`predict_next_state`: instead of the single-step (``h == 1``) next
        state, returns the model's **full per-horizon** observation forecast, optionally
        conditioned on a planned ``future_actions`` sequence. :meth:`predict_next_state` is left
        untouched (bit-exact single-step deploy path).

        The state input reuses the same history buffers as :meth:`predict_next_state` (the current
        ``env_state`` / ``action`` are appended to the multistep buffers first). The per-horizon
        predictions are returned in the model's single-step observation space (RLRP-728 option 2;
        no target-env per-horizon adapter -- see plan section 3.2.1 deviation).

        :param env_state: current environment state at timestep t.
        :param action: action executed at timestep t.
        :param future_actions: optional planned future-action sequence ``a_{t+1..t+F-1}`` of shape
            ``(horizon_len - 1, singlestep_act_len)``; ``None`` keeps the history-echo behaviour.
            The plan excludes ``a_t`` (already in the history) and ``a_{t+F}`` (never needed), so
            ``horizon_len == 1`` implies an empty plan / ``None``.

            Permanent: Introduced by stage `A1` of the Fix the MS→MS future-action-plan
            conditioning contract `.junie` plan
            (fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md).

            **Normalization space: pass RAW env-space actions.** The plan is carried into the
            model input normalization space by
            ``OneDTransitionRewardModelV2._normalize_future_action_plan`` -- the same primitive
            the TRAINING path uses -- so a caller must NOT pre-normalize it. Feeding an
            already-normalized (or otherwise mis-scaled) plan puts the conditioning channel off
            distribution and is refused by the fail-loud guard on the affine normalizer types.
            RLRP-757 test-time rollout review
            (``review_e2e_tcn_testtime_selffed_rollout_report_RLRP-757_20260912.md``).
        :return: per-horizon observation forecast, shape ``(horizon_len, singlestep_obs_len)``
            with index ``h - 1`` == ``ô_{t+h}`` (RLRP-760 S0 forecast axis).
        """
        with torch.inference_mode():
            self.one_d_tr_model.eval()

            if isinstance(env_state, np.ndarray):
                env_state = self._ndarray_to_tensor(env_state)
            if isinstance(action, np.ndarray):
                action = self._ndarray_to_tensor(action)
            if isinstance(future_actions, np.ndarray):
                future_actions = self._ndarray_to_tensor(future_actions)

            self._multistep_state_buffer.append(env_state)
            self._multistep_act_buffer.append(action)

            multistep_state = self._multistep_state_buffer.get_buffer()
            multistep_act = self._multistep_act_buffer.get_buffer()

            return self.one_d_tr_model.forecast_open_loop_per_horizon(
                obs=multistep_state,
                act=multistep_act,
                future_actions=future_actions,
            )

    def forecast_horizon_target_env(
        self,
        env_state: Union[torch.Tensor, np.ndarray],
        action: Union[torch.Tensor, np.ndarray],
        future_actions: Optional[Union[torch.Tensor, np.ndarray]] = None,
        *,
        anchor_env_state: Optional[Union[torch.Tensor, np.ndarray]] = None,
        chain_last_env_state: bool = True,
    ) -> Iterator[Tuple[int, torch.Tensor]]:
        """Yield the per-horizon forecast in **target-env** observation space, one step at a time.

        Step-driven companion to :meth:`forecast_horizon` (RLRP-728, plan section 3.2.2 / S7). The
        model prediction itself stays **open-loop / non-autoregressive** (each ``ô_{t+h}`` is
        predicted directly from the history at ``t`` -- the anti-compounding invariant is
        untouched); only the **coordinate reconstruction** into target-env space is applied
        sequentially so a **relative/stateful** ``model_ss_out_to_target_env_next_obs`` adapter
        (e.g. body-frame->world-frame or delta-integrating) can chain ``last_env_state`` correctly.

        Forecast axis (RLRP-760 S0): ``h`` ranges over ``[1, horizon_len]`` (``F``), NOT
        ``history_len``; see ``OneDTransitionRewardModelV2.forecast_open_loop_per_horizon``.

        Reliability crux (the ``last_env_state`` anchor):

        - ``h == 1``: ``last_env_state = anchor_env_state`` (the current anchor env obs, in
          target-env space) -- falls back to ``env_state`` when ``anchor_env_state`` is ``None``.
        - ``h > 1``: when ``chain_last_env_state`` (the safe default), ``last_env_state`` becomes the
          previous **predicted** target-env obs ``ô^env_{t+h-1}`` so the frame reconstruction chains;
          an **absolute** adapter simply ignores the threaded value.

        The state input reuses the same history buffers as :meth:`predict_next_state` /
        :meth:`forecast_horizon` (``env_state`` / ``action`` are appended first). Because this is a
        generator, the model forward runs on the first ``next(...)`` and the target-env mapping is
        produced lazily per step.

        :param env_state: current environment state at timestep t (model single-step in-obs layout,
            used for the history buffer -- mirrors :meth:`forecast_horizon`).
        :param action: action executed at timestep t.
        :param future_actions: optional planned future-action sequence ``a_{t+1..t+F-1}`` of shape
            ``(horizon_len - 1, singlestep_act_len)``; ``None`` keeps the history-echo behaviour.
            The plan excludes ``a_t`` (already in the history) and ``a_{t+F}`` (never needed), so
            ``horizon_len == 1`` implies an empty plan / ``None``.

            Permanent: Introduced by stage `A1` of the Fix the MS→MS future-action-plan
            conditioning contract `.junie` plan
            (fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md).

            **Normalization space: pass RAW env-space actions.** The plan is carried into the
            model input normalization space by
            ``OneDTransitionRewardModelV2._normalize_future_action_plan`` -- the same primitive
            the TRAINING path uses -- so a caller must NOT pre-normalize it. Feeding an
            already-normalized (or otherwise mis-scaled) plan puts the conditioning channel off
            distribution and is refused by the fail-loud guard on the affine normalizer types.
            RLRP-757 test-time rollout review
            (``review_e2e_tcn_testtime_selffed_rollout_report_RLRP-757_20260912.md``).
        :param anchor_env_state: the ``h == 1`` anchor in **target-env** obs space (raw env obs at
            t). Required for a relative/stateful adapter; defaults to ``env_state`` when ``None``.
        :param chain_last_env_state: when ``True`` (default = chain, the safe superset) thread the
            previous predicted target-env obs as ``last_env_state`` for ``h > 1``; set ``False`` to
            always reuse the anchor (absolute adapters).
        :yield: ``(h, env_obs_h)`` with ``h in [1, horizon_len]`` and ``env_obs_h`` in target-env
            observation space.
        """
        with torch.inference_mode():
            self.one_d_tr_model.eval()

            if isinstance(env_state, np.ndarray):
                env_state = self._ndarray_to_tensor(env_state)
            if isinstance(action, np.ndarray):
                action = self._ndarray_to_tensor(action)
            if isinstance(future_actions, np.ndarray):
                future_actions = self._ndarray_to_tensor(future_actions)
            if isinstance(anchor_env_state, np.ndarray):
                anchor_env_state = self._ndarray_to_tensor(anchor_env_state)

            self._multistep_state_buffer.append(env_state)
            self._multistep_act_buffer.append(action)

            multistep_state = self._multistep_state_buffer.get_buffer()
            multistep_act = self._multistep_act_buffer.get_buffer()

            # (horizon_len, singlestep_obs_len) in the model's single-step OUTPUT obs space.
            model_obs_steps = self.one_d_tr_model.forecast_open_loop_per_horizon(
                obs=multistep_state,
                act=multistep_act,
                future_actions=future_actions,
            )

        yield from self._iter_target_env_forecast_steps(
            model_obs_steps,
            anchor=anchor_env_state if anchor_env_state is not None else env_state,
            chain_last_env_state=chain_last_env_state,
        )

    def _iter_target_env_forecast_steps(
        self,
        model_obs_steps: torch.Tensor,
        anchor: Any,
        chain_last_env_state: bool = True,
    ) -> Iterator[Tuple[int, torch.Tensor]]:
        """Reconstruct the per-step forecast into target-env space, one step at a time.

        Shared core of :meth:`forecast_horizon_target_env` and
        :meth:`forecast_horizon_selffed_target_env` (RLRP-760 S1; a first step towards the plan
        \u00a76.4 API consolidation). Applies the asymmetric
        ``model_ss_out_to_target_env_next_obs`` adapter per step and chains ``last_env_state`` so
        relative/stateful adapters reconstruct correctly (absolute adapters ignore it).

        :param model_obs_steps: ``(horizon_len, singlestep_obs_len)`` in model single-step OUTPUT
            obs space.
        :param anchor: the ``h == 1`` ``last_env_state`` anchor, in target-env obs space.
        :yield: ``(h, env_obs_h)`` with ``h in [1, horizon_len]``.
        """
        last_env_state: Any = anchor
        for h in range(1, model_obs_steps.shape[0] + 1):
            # Map each model single-step OUTPUT obs to target-env next-obs space via the
            # asymmetric ``model_ss_out_to_target_env_next_obs`` adapter (plan section 3.2.2).
            env_obs_h, _info = self._adapter.model_ss_out_to_target_env_next_obs(
                model_obs_steps[h - 1], info={}, last_env_state=last_env_state
            )
            if chain_last_env_state:
                # Chain only matters for relative/stateful adapters; absolute adapters ignore it.
                last_env_state = env_obs_h
            yield h, env_obs_h

    def forecast_horizon_selffed_target_env(
        self,
        env_state: Union[torch.Tensor, np.ndarray],
        action: Union[torch.Tensor, np.ndarray],
        future_actions: Optional[Union[torch.Tensor, np.ndarray]] = None,
        *,
        anchor_env_state: Optional[Union[torch.Tensor, np.ndarray]] = None,
        chain_last_env_state: bool = True,
    ) -> Iterator[Tuple[int, torch.Tensor]]:
        """Closed-loop (**self-fed**) ``F``-step forecast window (RLRP-760 S1).

        Same single forward as :meth:`forecast_horizon_target_env` -- the model stays
        **non-autoregressive within the window** (the anti-compounding invariant of the MS->MS
        baselines is untouched) -- but, as each forecast step is produced, the predicted
        observation and its planned action are **pushed back into the deployer history rings**.
        After the full ``F``-step hop the history window therefore holds ``F`` model predictions
        (plus the ``H - F`` preceding steps), so the **next** window is conditioned on the model's
        own output: the compounding happens **across** windows, which is the deployment condition
        RLRP-760 measures. **No ground-truth re-anchoring** is performed here.

        Feedback contents per step ``h`` (plan \u00a73.1.1, both channels):

        - **obs**: the predicted target-env obs ``\u00f4^env_{t+h}`` (computed **once** and reused for
          the caller's scoring, for the ``last_env_state`` chain and for the feedback) mapped back
          to model single-step **input** space via ``target_env_to_model_ss_in_obs``.
        - **act**: the planned action ``a_{t+h}`` (``future_actions[h - 1]``), or the anchor action
          ``a_t`` held when no plan is supplied (the ``obs_only`` baseline).

        Ring bookkeeping -- ``F`` slides per hop, and **not** a stale action slot:

        The feedback covers ``h = 1 .. F-1`` **only**. The ``F``-th predicted step is deliberately
        NOT pushed here: it is the **next window's anchor** and is pushed by that window's own
        anchor append, paired with **its** anchor action ``a_{t+F}``. That action is the one the
        model reads as ``a_t`` (the composed input action block is ``a_{t-H+1..t}``, so the LAST
        action slot drives ``\u00f4_{t+1}`` -- see
        ``AbstractMS2MSForecast._build_driving_action_sequence``). Pushing the ``F``-th step here
        with a *held* plan action instead left that slot holding ``a_{t+F-1}`` (and duplicated it
        with the slot before), i.e. the next window was driven by the **previous** step's action --
        silently degrading exactly the conditioning RLRP-781 restored. Anchor (1) + feedback
        (``F-1``) still slides the rings by exactly ``F`` per hop.

        Permanent: contract fixed by the MS->MS test-time rollout review (2026-09-11).

        :param env_state: current anchor obs at ``t`` in model single-step **input** obs layout
            (appended to the history ring first -- mirrors :meth:`forecast_horizon`).
        :param action: action ``a_t`` executed at the anchor.
        :param future_actions: planned actions ``a_{t+1..t+F-1}``, shape
            ``(horizon_len - 1, singlestep_act_len)``; ``None`` keeps the history-echo behaviour
            and holds ``a_t`` in the action feedback. The plan excludes ``a_t`` (already in the
            history) and ``a_{t+F}`` (never needed), so ``horizon_len == 1`` implies an empty plan
            / ``None``.

            Permanent: Introduced by stage `A1` of the Fix the MS→MS future-action-plan
            conditioning contract `.junie` plan
            (fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md).

            **Normalization space: pass RAW env-space actions.** The plan is carried into the
            model input normalization space by
            ``OneDTransitionRewardModelV2._normalize_future_action_plan`` -- the same primitive
            the TRAINING path uses -- so a caller must NOT pre-normalize it. Feeding an
            already-normalized (or otherwise mis-scaled) plan puts the conditioning channel off
            distribution and is refused by the fail-loud guard on the affine normalizer types.
            RLRP-757 test-time rollout review
            (``review_e2e_tcn_testtime_selffed_rollout_report_RLRP-757_20260912.md``).
        :param anchor_env_state: the ``h == 1`` anchor in **target-env** obs space; defaults to
            ``env_state``.
        :param chain_last_env_state: thread the previous predicted target-env obs for ``h > 1``
            (default ``True``, the safe superset for relative/stateful adapters).
        :yield: ``(h, env_obs_h)`` with ``h in [1, horizon_len]``, in target-env obs space.
        """
        with torch.inference_mode():
            self.one_d_tr_model.eval()

            if isinstance(env_state, np.ndarray):
                env_state = self._ndarray_to_tensor(env_state)
            if isinstance(action, np.ndarray):
                action = self._ndarray_to_tensor(action)
            if isinstance(future_actions, np.ndarray):
                future_actions = self._ndarray_to_tensor(future_actions)
            if isinstance(anchor_env_state, np.ndarray):
                anchor_env_state = self._ndarray_to_tensor(anchor_env_state)

            # The anchor pair ALWAYS slides in: for the first window it is the ground-truth
            # (o_t, a_t); for every subsequent window it is the previous window's ``F``-th
            # prediction paired with the genuine anchor action ``a_t`` of THIS window.
            self._multistep_state_buffer.append(env_state)
            self._multistep_act_buffer.append(action)

            model_obs_steps = self.one_d_tr_model.forecast_open_loop_per_horizon(
                obs=self._multistep_state_buffer.get_buffer(),
                act=self._multistep_act_buffer.get_buffer(),
                future_actions=future_actions,
            )

        horizon_steps = int(model_obs_steps.shape[0])
        for h, env_obs_h in self._iter_target_env_forecast_steps(
            model_obs_steps,
            anchor=anchor_env_state if anchor_env_state is not None else env_state,
            chain_last_env_state=chain_last_env_state,
        ):
            # Permanent: feed back ``h = 1 .. F-1`` ONLY. The ``F``-th step is the next window's
            # anchor and is pushed by that window's anchor append, paired with its true anchor
            # action ``a_{t+F}`` -- see the ring-bookkeeping note in the method docstring. The
            # plan is ``F-1``-long, so ``future_actions[h - 1]`` is always in bounds here.
            if h < horizon_steps:
                if future_actions is not None and future_actions.shape[0] > 0:
                    feedback_action = future_actions[h - 1]
                else:
                    feedback_action = action
                self.append_feedback_step(
                    env_obs_h,
                    feedback_action,
                )
            yield h, env_obs_h

    def append_feedback_step(
        self,
        predicted_env_obs: Union[torch.Tensor, np.ndarray],
        planned_action: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Push one predicted ``(obs, action)`` pair into the history rings (RLRP-760 S1).

        The self-fed counterpart of the buffer append that :meth:`predict_next_state` performs on
        *incoming* environment steps: it slides the history window **without** running a spurious
        model forward. ``predicted_env_obs`` is given in **target-env** obs space and is mapped to
        the model single-step **input** space via ``target_env_to_model_ss_in_obs`` (the
        ``out -> target-env -> in`` round trip of plan \u00a73.1.1).

        :param predicted_env_obs: predicted observation in target-env obs space.
        :param planned_action: the action planned/executed at that same timestep.
        """
        if isinstance(predicted_env_obs, np.ndarray):
            predicted_env_obs = self._ndarray_to_tensor(predicted_env_obs)
        if isinstance(planned_action, np.ndarray):
            planned_action = self._ndarray_to_tensor(planned_action)
        model_in_obs = self._adapter.target_env_to_model_ss_in_obs(predicted_env_obs)
        self._multistep_state_buffer.append(model_in_obs.ravel())
        self._multistep_act_buffer.append(planned_action.ravel())
        return None
