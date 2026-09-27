# coding=utf-8
import math
import os.path
import contextlib
import pathlib
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from deprecated import deprecated

import numpy as np
import torch
from mbrl.models import Model, OneDTransitionRewardModel
from mbrl.models.model import _log_metrics_eager
from mbrl import types as mbrl_types
from numpy import ndarray
from torch import Tensor
from torch._dynamo import OptimizedModule as torch_OptimizedModule
import mbrl.models.util as model_util

from tools.mbrl_lib_tools.models.gaussian_mlp_extended import GaussianMLPExtended
from tools.mbrl_lib_tools.models.prediction_statistics import (
    SENTINEL_LOGVAR,
    SPACE_PHYSICAL,
    SPACE_UNKNOWN,
    VARIANCE_EXACT,
    VARIANCE_LOCAL_LINEAR,
    PredictionStatistics,
)
from tools.mbrl_lib_tools.models.utils import (
    compute_trajectory_probability_statistics_over_ensemble,
)
from tools.multistep_tools.models import (
    ExponentialFamilyMLP,
    MultiStepMLP,
    MS2MS2SSArTemporalMixturePME,
)
from tools.multistep_tools.multistep_model_util import (
    timestep_first_multistep_dim_unflaten_array,
)

from tools.console_tools.message import consol_msg_universal_one_liner
from tools.mbrl_lib_tools.models.precision_diagnostics import (
    log_precision_summary_once,
    warn_once_float32_downcast,
    warn_once_precision_mismatch,
)


# Fail-loud bound on the NORMALIZED test-time future-action plan, in sigma (RLRP-757 test-time
# rollout review). A plan mistakenly left in RAW env units lands ~380 sigma off on the Neurobem
# motor actions, while legitimate (even aggressive) control stays within a few sigma; the bound is
# kept generous so only a genuine space mismatch trips it.
# See OneDTransitionRewardModelV2._assert_normalized_plan_is_plausible.
_MAX_ABS_NORMALIZED_ACTION_PLAN_SIGMA = 100.0


def _warn_numpy_input(var_name: str, caller: str) -> None:
    """Emit a deprecation warning when a numpy array is passed where a torch.Tensor is expected.

    This is an RLRC-level safety net during the torch-first transition period.
    Once all call sites pass torch tensors, these guards can be removed.
    """
    warnings.warn(
        f"[RLRC torch-first] '{var_name}' is a numpy ndarray in {caller}. "
        f"With output_torch=True (default), data should already be torch.Tensor. "
        f"Update the upstream caller to pass torch tensors directly.",
        DeprecationWarning,
        stacklevel=3,
    )


class OneDTransitionRewardModelV2(OneDTransitionRewardModel):
    """V2 wrapper for 1-D dynamics models with robust normalizer support.

    Extends :class:`OneDTransitionRewardModel` with:

    - **Robust normalization** (winsorized / quantile) via separate obs and act normalizers that
     decompose composed multi-step observations into per-block statistics.  Target normalization
     is handled transparently through the same ``obs_normalizer``, no dedicated target normalizer is needed.
    - **Double-precision** support (``normalize_double_precision``).
    - **Ensemble broadcasting** for single-step sampling (``_single_step_sample_2_model_ensemble``).
    - **Prediction statistics** convenience method (``compute_prediction_and_stats``).

    Args:
        model: The dynamics model to wrap.  Must expose ``in_size``, ``out_size``, ``num_members``
            and, for multi-step models, ``singlestep_obs_len`` / ``singlestep_act_len``.
        target_is_delta: If ``True`` the model predicts observation *deltas* rather than absolute
            next observations.  Per-dimension exceptions are listed in ``no_delta_list``.
        normalize: If ``True`` an input normalizer is created (type selected by ``normalizer_type``).
        normalize_double_precision: Force the normalizer dtype.  When ``None`` (default) the dtype
            is inferred from the wrapped model's ``model_use_double_precision`` attribute.
        learned_rewards: If ``True`` the last output dimension is treated as a reward prediction.
        obs_process_fn: Optional callable applied to observations before they enter the model.
        no_delta_list: Dimension indices excluded from delta prediction.
        num_elites: Number of elite ensemble members.  ``None`` keeps all.
        normalizer_type: ``"winsorized"`` (default), ``"quantile"``, ``"standard"``
            (asymmetric Z-score: input-only normalization, raw target — used by upstream mbrl
            PETS / MBPO), or ``"standard_symmetric"`` (RLRP-684 A1: block-shared Z-score on both
            input and target, so the AR loop stays entirely in normalized space).
        obs_dim: Single-step observation dimensionality.  Inferred from the model when ``None``.
        act_dim: Single-step action dimensionality.  Inferred from the model when ``None``.
        normalizer_kwargs: Extra keyword arguments forwarded to the normalizer factory (
            e.g. ``clip_range``, ``winsor_percentile``, ``soft_clip_iqr_mult``).
    """

    _ONEDT_FNAME = "onedt_model.pth"

    # Near-deterministic logvar fill value; single canonical source (RLRP-761 S10.4).
    # See prediction_statistics.SENTINEL_LOGVAR / ExponentialFamilyMLP for the rationale.
    _LOGVAR_MIN_LIMIT = SENTINEL_LOGVAR
    model_use_double_precision: bool = False

    def __init__(
        self,
        model: Model,
        target_is_delta: bool = False,
        normalize: bool = False,
        normalize_double_precision: Optional[bool] = None,
        learned_rewards: bool = False,
        obs_process_fn: Optional[mbrl_types.ObsProcessFnType] = None,
        no_delta_list: Optional[List[int]] = None,
        num_elites: Optional[int] = None,
        normalizer_type: str = "winsorized",
        obs_dim: Optional[int] = None,
        act_dim: Optional[int] = None,
        normalizer_kwargs: Optional[Dict[str, Any]] = None,
        allow_contract_drop: bool = False,
    ):

        if hasattr(model, "model_use_double_precision"):
            self.model_use_double_precision = model.model_use_double_precision

        # Resolve normalizer dtype: when None, align with model precision
        if normalize_double_precision is None:
            normalize_double_precision = self.model_use_double_precision

        if normalize and normalize_double_precision:
            consol_msg_universal_one_liner(f"Normalizer use double precision")

        super().__init__(
            model,
            target_is_delta,
            normalize,
            normalize_double_precision,
            learned_rewards,
            obs_process_fn,
            no_delta_list,
            num_elites,
            normalizer_type=normalizer_type,
            obs_dim=obs_dim,
            act_dim=act_dim,
            normalizer_kwargs=normalizer_kwargs,
            allow_contract_drop=allow_contract_drop,
        )

        # Propagate clip_range from normalizer_kwargs to standard input_normalizer.
        # Clipping is handled inside ZScoreNormalizer.normalize() so _get_model_input
        # does not need an extra clamp step.
        _kwargs_clip = (normalizer_kwargs or {}).get("clip_range", None)
        if (
            self.input_normalizer is not None
            and self.input_normalizer.single is not None
            and _kwargs_clip is not None
        ):
            self.input_normalizer.single.clip_range = _kwargs_clip

        # Align the wrapper's declared dtype explicitly with the wrapped dynamics model.
        # The wrapped model (e.g. WeightedMultiStepDualHeadMLP / the MTM-Pro family /
        # ExponentialFamilyMLP) exposes ``model_dtype``; the wrapper used to fetch the
        # ``model_use_double_precision`` flag but never declared its own dtype. Setting it
        # here makes downstream logic that inspects the wrapper (``model``) instead of the
        # wrapped model (``model.model``) see a consistent dtype, avoiding silent mismatch.
        self.model_dtype = self._resolve_wrapped_model_dtype()

        self._set_model_normalizer_handle()

        self._emit_precision_summary_once(self.model_dtype)
        self._maybe_warn_ar_precision_recommendation()

        # RLRP-684 WS-C (C2): transient-model-output non-finite telemetry.
        # Early in training the model can legitimately emit NaN/Inf predictions;
        # unlike input data (which fails fast via ``strict_finite``), these are
        # tolerated (``nan_to_num``) but COUNTED so divergence is observable
        # instead of silently swallowed. ``_non_finite_output_calls`` is the
        # number of forward calls that produced any non-finite output.
        self._non_finite_output_calls: int = 0
        self._non_finite_output_entries: int = 0

        # RLRP-786 (FR5): the opt-in CUDA-graph captured training step (see
        # :meth:`enable_cuda_graph_training_step`). ``None`` = today's eager Lightning path.
        self._cuda_graph_step = None
        self._cuda_graph_eager_fallbacks: int = 0
        # AR-baseline extension (FR4): the autocast dtype the captured region runs under when the
        # exploratory ``allow_autocast`` opt-in is active (``None`` = fp32 capture / eager path).
        self._cuda_graph_autocast_dtype: Optional[torch.dtype] = None

    # ==== RLRP-786: opt-in CUDA-graph captured training step (production Lightning path) ========
    def cuda_graph_training_step_blockers(
        self, optimizer: torch.optim.Optimizer, allow_autocast: bool = False
    ) -> List[str]:
        """Why the captured step cannot be enabled on this wrapper (empty = go). FR5 of
        ``perf_RLRP-786_cuda_graph_train_step_plan_20260919.md``: CUDA device, the wrapped model
        reports no capture blocker (sampling-free MTM-Pro variant with ``cuda_graph_capture_ready``,
        or an AR MS2SS model on a frozen-scheduler config; ``performance_mode=fast``, meta
        collection off...), a ``capturable`` optimizer, no autocast.

        :param allow_autocast: AR-baseline extension (FR4 of
            ``perf_RLRP-786_ar_tcn_cuda_graph_fp32_baseline_plan_20260919.md``): accept a
            ``mixed_precision`` autocast step -- EXPLORATORY. The captured region then runs under
            ``torch.autocast(..., cache_enabled=False)`` (the autocast weight-cast cache is not
            replayable), which re-casts every weight at every unroll step instead of once per
            step: NOT the eager kernel sequence, so parity is ``allclose``, not ``torch.equal``.
        """
        blockers: List[str] = []
        if torch.device(self.device).type != "cuda":
            blockers.append(f"device {self.device} is not CUDA")
        report = getattr(self.model, "cuda_graph_capture_blockers", None)
        if report is None:
            blockers.append(
                f"{type(self.model).__name__} does not implement cuda_graph_capture_blockers "
                "(MTM-Pro / AR MS2SS families)"
            )
        else:
            blockers.extend(report())
        if not all(g.get("capturable", False) for g in optimizer.param_groups):
            blockers.append(
                "optimizer is not capturable (pipeline.optimizer.adam_capturable: true)"
            )
        if self.resolve_autocast_dtype() is not None and not allow_autocast:
            blockers.append(
                "mixed_precision autocast is on (the captured step is fp32 only; "
                "pipeline.cuda_graph_allow_autocast: true opts into the exploratory "
                "cache_enabled=False capture)"
            )
        return blockers

    def enable_cuda_graph_training_step(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int = 3,
        allow_autocast: bool = False,
    ) -> bool:
        """Route :meth:`training_step` through a ``CudaGraphTrainStep`` (RLRP-786 FR5).

        Returns ``True`` when enabled. On any blocker the wrapper stays on the eager path and a
        one-line notice names the reasons (never raises: the opt-in must not kill a run).
        Enabling switches the Lightning module to MANUAL optimisation: the captured region
        (``loss_from_processed -> backward -> optimizer.step``) replaces Lightning's automatic
        ``backward`` / ``optimizer.step``; the eager fallback (ragged tail batch) performs the
        same three calls explicitly. ``optimizer`` must be the instance ``configure_optimizers``
        returns (``ModelTrainer.optimizer``).

        :param allow_autocast: see :meth:`cuda_graph_training_step_blockers` -- when the wrapper
            resolves a ``mixed_precision`` dtype the captured ``loss_from_processed`` runs under
            ``torch.autocast(device_type='cuda', dtype=dt, cache_enabled=False)`` (exits before the
            backward, like the eager ``training_step``); the eager fallback keeps the regular
            autocast context. EXPLORATORY: ``allclose`` parity only.
        """
        blockers = self.cuda_graph_training_step_blockers(
            optimizer, allow_autocast=allow_autocast
        )
        if blockers:
            consol_msg_universal_one_liner(
                "RLRP-786 pipeline.cuda_graph_training_step requested but NOT capturable -> eager "
                "training step kept: " + "; ".join(blockers),
                caller_name="OneDTransitionRewardModelV2",
            )
            self._cuda_graph_step = None
            return False
        from tools.torch_tools.cuda_graph_train_step import CudaGraphTrainStep

        autocast_dtype = self.resolve_autocast_dtype() if allow_autocast else None
        self._cuda_graph_autocast_dtype = autocast_dtype

        def _graph_loss_fn(inputs: Tuple[torch.Tensor, ...]):
            if autocast_dtype is not None:
                # The autocast cast cache keeps bf16 copies of the weights across the autocast
                # region; a replayed graph cannot refresh them after an optimizer step -> the
                # cache must be OFF inside the captured region (torch.cuda.make_graphed_callables
                # rule). Consequence: one weight re-cast per unroll step (not the eager kernels).
                ctx = torch.autocast(
                    device_type="cuda", dtype=autocast_dtype, cache_enabled=False
                )
            else:
                ctx = contextlib.nullcontext()
            with ctx:
                loss, meta = self.loss_from_processed(*inputs)
            if isinstance(loss, tuple):
                loss = loss[0]
            return loss, meta

        self._cuda_graph_step = CudaGraphTrainStep(
            loss_fn=_graph_loss_fn,
            optimizer=optimizer,
            device=torch.device(self.device),
            before_replay=self.model.advance_host_step_state,
            warmup_iters=warmup_iters,
        )
        self.automatic_optimization = False
        if autocast_dtype is None:
            _how = "(same kernels, one launch)"
        else:
            _how = (
                f"under torch.autocast({autocast_dtype}, cache_enabled=False) -- EXPLORATORY: "
                "per-unroll-step weight re-casts, NOT the eager kernel sequence, allclose parity only"
            )
        consol_msg_universal_one_liner(
            "RLRP-786 pipeline.cuda_graph_training_step ON: the training step is recorded in a "
            f"torch.cuda.CUDAGraph after {warmup_iters} eager warm-up step(s) and replayed "
            f"{_how}; ragged / non-matching batches run eagerly",
            caller_name="OneDTransitionRewardModelV2",
        )
        return True

    @property
    def cuda_graph_training_step(self):
        """The active ``CudaGraphTrainStep`` (``None`` on the eager path)."""
        return self._cuda_graph_step

    def training_step(self, batch: mbrl_types.TransitionBatch, batch_idx: int):
        """Lightning training step. Eager (inherited) unless :meth:`enable_cuda_graph_training_step`
        succeeded; then MANUAL optimisation with the captured step and an explicit eager fallback.
        NaN stop and metric logging are those of the inherited step in both cases."""
        step = self._cuda_graph_step
        if step is None:
            return super().training_step(batch, batch_idx)
        # Host half of the loss (normalisation) runs eagerly; the captured half consumes tensors.
        with torch.no_grad():
            inputs = self.process_batch_for_loss(batch)
        if self.training and step.is_capturable(inputs):
            loss, meta = step.step(inputs)
            # ``loss`` is the graph's STATIC output buffer (overwritten by the next replay):
            # Lightning's ``log`` keeps a reference for the epoch reduction -> snapshot it
            # (device-to-device copy, no host sync).
            loss = loss.detach().clone()
        else:
            # Ragged tail batch / contract mismatch: the SAME three calls Lightning would issue.
            self._cuda_graph_eager_fallbacks += 1
            opt = self.optimizers()
            opt.zero_grad(set_to_none=True)
            # (no-op context unless ``mixed_precision`` is set AND the autocast capture was allowed;
            # without the opt-in the autocast blocker keeps the whole wrapper on the eager path)
            with self._autocast_context():
                loss, meta = self.loss_from_processed(*inputs)
            if isinstance(loss, tuple):
                loss = loss[0]
            self.manual_backward(loss)
            opt.step()
        loss = loss.detach()
        if loss.dtype in (torch.bfloat16, torch.float16):
            loss = loss.float()
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: train loss is {loss.item()}. Stopping training.")
            self.trainer.should_stop = True
        _log_metrics_eager(self, "train", loss, meta, len(batch))
        return {**meta, "loss": loss}

    def _maybe_warn_ar_precision_recommendation(self) -> None:
        """Recommend the precision-optimal AR setup for double-precision multi-step models.

        For autoregressive multi-step models (those exposing ``state_history_update``,
        e.g. the MTM-Pro family), the per-step ``denormalize -> shift -> renormalize``
        round-trip is only performed for the asymmetric ``standard`` normalizer; it is a
        no-op for the robust/block-facade normalizers. Under double precision that
        round-trip is the main remaining source of catastrophic cancellation in long
        rollouts, and ``.sample()``-based propagation injects Monte-Carlo noise that
        compounds across the horizon.

        We therefore emit a one-liner (RLRC precision hardening, Gate D / Phase 3.2-3.3)
        recommending a robust normalizer + the sampling-free PME variant when a
        double-precision AR model is wrapped with the asymmetric ``standard`` normalizer.
        """
        is_ar_multistep = hasattr(self.model, "state_history_update")
        if (
            self.model_dtype == torch.float64
            and is_ar_multistep
            and not self._uses_block_facade
        ):
            consol_msg_universal_one_liner(
                "[RLRC precision] Double-precision AR multi-step model using the "
                "asymmetric 'standard' normalizer: the per-step denormalize->renormalize "
                "round-trip is a catastrophic-cancellation source that compounds over the "
                "horizon. For maximum precision, prefer a robust normalizer "
                "(normalizer_type: standard_symmetric / winsorized / quantile, which makes "
                "the round-trip a no-op) and the sampling-free PME variant "
                "(ms2ms2ss_ar_temporal_mixture_sampling_free_pme*) to avoid compounding "
                "Monte-Carlo sampling noise.",
                caller_name="OneDTransitionRewardModelV2",
            )

    def _resolve_wrapped_model_dtype(self) -> torch.dtype:
        """Resolve the dtype of the wrapped dynamics model, following its own convention.

        Preference order:
            1. the wrapped model's explicit ``model_dtype`` attribute (set by every
               multi-step / ExponentialFamilyMLP model);
            2. the dtype of its first parameter;
            3. ``torch.double`` / ``torch.float32`` derived from
               ``model_use_double_precision`` as a last resort.
        """
        wrapped_dtype = getattr(self.model, "model_dtype", None)
        if isinstance(wrapped_dtype, torch.dtype):
            return wrapped_dtype
        try:
            return next(self.model.parameters()).dtype
        except (StopIteration, AttributeError):
            return torch.double if self.model_use_double_precision else torch.float32

    @property
    def dtype(self) -> torch.dtype:
        """The wrapper's compute dtype, aligned with the wrapped dynamics model.

        Exposed so downstream code can query ``one_d_transition_reward_model.dtype``
        (the wrapper) and get the same answer as ``one_d_transition_reward_model.model``
        (the wrapped model, e.g. ``WeightedMultiStepDualHeadMLP`` / the MTM-Pro family).

        NOTE: we intentionally do NOT call ``self.to(dtype=self.model_dtype)`` in the
        constructor. The wrapper holds no parameters of its own (the wrapped model
        already lives at ``model_dtype``), and a blanket cast would force the
        normalizer statistics to the model dtype, clobbering the *independently*
        controlled normalizer precision (``normalize_double_precision``). Keeping the
        two precisions decoupled is required by the aligned-precision design
        (see investigation_normalizer_precision_alignment_report_20260307.md).
        """
        return self.model_dtype

    def _set_model_normalizer_handle(self):
        """Propagate the asymmetric-standard normalizer handle to the wrapped model.

        RLRP-684 (Amendment A1): only the asymmetric ``"standard"`` Z-score path
        needs a handle on the wrapped model — the AR loop uses it for the
        ``denormalize -> shift -> renormalize`` round-trip that splices the
        RAW-space predicted next-obs into the normalized model input.  Every
        block facade (``"standard_symmetric"`` / ``"winsorized"`` /
        ``"quantile"``) keeps the AR loop entirely in normalized space, so it
        propagates ``None`` and the round-trip becomes a no-op.

        RLRP-731 (action ``B3-handle``): additionally propagate a SEPARATE,
        obs-space-only DENORM handle used by the deploy-history drift residual
        (DH) loss to score in RAW obs space. It is needed exactly where the
        input-normalizer handle is ``None`` but targets are NORMALIZED, i.e.
        the block facades: the handle wraps the layout-aware
        ``denormalize_predicted_obs`` (obs block only, output/horizon layout).
        Asymmetric ``standard`` (targets already raw) and disabled
        normalization propagate ``None`` (identity in the model). Never reuses
        ``set_one_d_trj_model_input_normalizer`` (RLRP-684 ``None``-handle
        invariant preserved).
        """
        if isinstance(self.model, ExponentialFamilyMLP):
            if (
                self.input_normalizer is not None
                and self.input_normalizer.single is not None
            ):
                single = self.input_normalizer.single
            else:
                single = None
            self.model.set_one_d_trj_model_input_normalizer(single)
            if single is None and self.output_normalizer is not None:
                # Block facade (robust normalizer): normalized targets/predictions -> DH
                # needs the obs-block denorm map.
                self.model.set_one_d_trj_model_obs_denorm_handle(
                    self.denormalize_predicted_obs
                )
            else:
                # Asymmetric 'standard' (raw targets) or normalization disabled -> identity.
                self.model.set_one_d_trj_model_obs_denorm_handle(None)

            # RLRP-761 S4.4/S4.9: the AR splice re-injects a TARGET-space
            # prediction into an INPUT-space window. Those two spaces coincide
            # for every legacy type (``ar_bridge_gain`` is then ``None`` =>
            # identity), and differ only under
            # ``standard_symmetric_innovation``.
            setter = getattr(
                self.model, "set_one_d_trj_model_ar_bridge_gain", None
            )
            if setter is not None:
                setter(self.ar_bridge_gain)

            # RLRP-761 S12.4: the act block re-injected by the MS forecast
            # self-feed crosses the same target->input boundary. Push the act
            # gain too; ``None`` (identity) for every type whose act facades are
            # shared, so legacy paths stay bit-exact.
            setter_act = getattr(
                self.model, "set_one_d_trj_model_ar_bridge_gain_act", None
            )
            if setter_act is not None:
                setter_act(self.ar_bridge_gain_act)

    @torch.compiler.disable
    def _emit_precision_summary_once(self, model_dtype: torch.dtype) -> None:
        """Emit a one-shot device/dtype summary for the assembled model input.

        Confirms (once per process) that ``float64`` survives end-to-end across the
        wrapped model's parameters, the normalizer statistics and the model input on
        the active device (``cpu`` / ``cuda``).
        """
        try:
            param_dtype = next(self.model.parameters()).dtype
        except (StopIteration, AttributeError):
            param_dtype = None

        normalizer_dtype = None
        if (
            self.input_normalizer is not None
            and self.input_normalizer.single is not None
        ):
            _mean = getattr(self.input_normalizer.single, "mean", None)
            if isinstance(_mean, torch.Tensor):
                normalizer_dtype = _mean.dtype

        log_precision_summary_once(
            f"{type(self).__name__}._get_model_input",
            device=self.device,
            param_dtype=param_dtype,
            normalizer_dtype=normalizer_dtype,
            model_in_dtype=model_dtype,
        )

    @torch.compiler.disable
    def _get_model_input(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        normalize: bool = True,
        normalize_strict_finite: Optional[bool] = None,
    ) -> torch.Tensor:
        """Re-implemented from the orginal with support for double precision and optional normalization step.

        ``normalize_strict_finite`` is an optional per-call override of the input
        normalizer's ``strict_finite`` fail-fast policy (RLRP-684 WS-C). ``None``
        (default) keeps the strict instance policy used for genuine input data /
        training. The deploy/test-time rollout passes ``False`` because the ``obs``
        it feeds here is a *model output* fed back auto-regressively: a diverged
        early-training prediction can be non-finite or huge-but-finite (the latter
        overflowing ``(val-mean)/std`` to non-finite), and must be *clamped*
        rather than crash the loop (it is still surfaced via the output-sanitize
        telemetry). Genuine corrupt input data keeps failing fast."""

        # .... refactored from the original without forced float casting ..........................
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)

        if isinstance(obs, np.ndarray):
            _warn_numpy_input("obs", "_get_model_input")
        if isinstance(action, np.ndarray):
            _warn_numpy_input("action", "_get_model_input")

        obs = model_util.to_tensor(obs)
        action = model_util.to_tensor(action)

        if self.device.type == "mps":
            # Defensive: the Apple-Silicon mps backend is NOT reachable from inside the
            # DNA container (runs are cpu local / cuda on Valeria / Jetson-AGX-Orin), so
            # this branch is expected to be dead code. The warn-once is kept as a safety
            # net in case the execution environment ever changes.
            if obs.dtype == torch.float64:
                warn_once_float32_downcast(
                    "OneDTransitionRewardModelV2._get_model_input:mps",
                    tensor_name="obs",
                    from_dtype=obs.dtype,
                    reason="mps backend does not support float64",
                )
                obs = obs.float()
            if action.dtype == torch.float64:
                warn_once_float32_downcast(
                    "OneDTransitionRewardModelV2._get_model_input:mps",
                    tensor_name="action",
                    from_dtype=action.dtype,
                    reason="mps backend does not support float64",
                )
                action = action.float()

        # Training Speed & Efficiency plan — C3: use non_blocking=True
        # so pinned-memory H→D copies can overlap with compute on CUDA.
        # No-op on CPU (M3 DNA Docker) and on MPS; kept bit-exact.
        obs = obs.to(self.device, non_blocking=True)
        action = action.to(self.device, non_blocking=True)

        if self._uses_block_facade and normalize:
            # Robust normalizer path: decompose composed obs, normalize per-block, reassemble
            compute_dtype = (
                torch.double
                if self.model_use_double_precision
                and isinstance(self.model, ExponentialFamilyMLP)
                else torch.float
            )
            obs_norm = (
                # RLRP-761 S4.9: this is the model INPUT, so it must use the
                # input-space obs scale. Identical to the target scale for every
                # type except ``standard_symmetric_innovation``, whose input
                # keeps the well-conditioned state std while its target is
                # innovation-scaled.
                self._normalize_composed_obs(
                    obs, strict_finite=normalize_strict_finite, space="input"
                )
                .to(compute_dtype)
                .to(self.device, non_blocking=True)
            )
            act_norm = (
                # RLRP-761 S12.2: this is the model INPUT, so the act block must
                # use the input-space (state-std) scale. Identical to the target
                # scale for every type except ``standard_symmetric_innovation``,
                # whose act input keeps the well-conditioned state std while its
                # act target is innovation-scaled (mirrors the obs ``space``).
                self._normalize_composed_act(
                    action, strict_finite=normalize_strict_finite, space="input"
                )
                .to(compute_dtype)
                .to(self.device, non_blocking=True)
            )
            model_in = torch.cat([obs_norm, act_norm], dim=obs.ndim - 1)
            return model_in

        model_in = torch.cat([obs, action], dim=obs.ndim - 1)

        if self.model_use_double_precision and isinstance(
            self.model, ExponentialFamilyMLP
        ):
            model_in = model_in.double()
        else:
            warn_once_precision_mismatch(
                "OneDTransitionRewardModelV2._get_model_input:model_in",
                requested_double_precision=self.model_use_double_precision,
                resolved_dtype=torch.float32,
                device=self.device,
            )
            model_in = model_in.float()

        if self.input_normalizer and normalize:
            # Normalizer now preserves input dtype, so the cast above is
            # maintained through normalization without extra re-casting.
            # Clip-range clamping (if configured) is applied inside
            # ZScoreNormalizer.normalize() via self.input_normalizer.clip_range.
            model_in = self.input_normalizer.single.normalize(
                model_in, strict_finite=normalize_strict_finite
            ).to(self.device, non_blocking=True)

        return model_in

    @torch.compiler.disable
    def _process_batch(
        self,
        batch: mbrl_types.TransitionBatch,
        _as_float: bool = False,
        return_raw_target: bool = False,
        return_action_plan: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Build ``(model_input, target)`` from a transition batch.

        Re-implemented from the original with support for double precision and
        robust normalizers.

        When a robust normalizer is active, targets are expressed in normalized
        observation space (via ``_normalize_composed_obs``).  For the standard
        normalizer path, targets are in raw observation space (the standard
        ``input_normalizer`` only normalizes inputs).

        Args:
            batch: A :class:`~mbrl.types.TransitionBatch` sampled from the
                replay buffer.
            _as_float: If ``True`` the returned target tensor is cast to
                ``float32`` regardless of the model precision setting.
            return_raw_target: RLRP-731 batch ``B4-raw`` (Option 1). If ``True`` a THIRD tensor
                is returned: the obs-block RAW (non-normalized) target consumed by the
                deploy-history drift (DH) ``raw_passthrough`` mode. Defaults to ``False`` so every
                existing caller (``loss`` / ``update`` / ``eval_score`` / ``get_output_and_targets``
                / trainer sites) keeps the exact 2-tuple contract and byte-identical behaviour /
                memory footprint (no raw tensor is ever materialized on the non-opted-in path).

            return_action_plan: Stage ``A2`` of the Fix the MS->MS future-action-plan
                conditioning contract ``.junie`` plan
                (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).
                If ``True`` a THIRD tensor is returned: the **future-action plan**
                ``a_{t+1..t+F-1}`` of shape ``(..., horizon_len - 1, singlestep_act_len)``,
                sliced from the **RAW** ``batch.next_obs`` action block and expressed in the model
                **INPUT** normalization space (see
                :meth:`_extract_action_plan_from_composed_next_obs`). Consumed by the
                control-conditioned MS->MS forecast baselines (E2E-TCN / M3 / TBM) so that the
                plan the model is conditioned on at TRAINING time is the same one it gets at test
                time (RLRP-781). Defaults to ``False`` so every existing caller keeps the exact
                2-tuple contract and byte-identical behaviour.

                Mutually exclusive with ``return_raw_target`` (no caller needs both: the former is
                the MTM-Pro deploy-history-drift path, the latter the ms2ms conditioning path).

        Returns:
            ``(model_input, target)`` (default), ``(model_input, target, target_raw_obs)`` when
            ``return_raw_target=True``, or ``(model_input, target, action_plan)`` when
            ``return_action_plan=True``, ready for the wrapped model's loss / update methods.
        """

        # .... refactored from the original with optional float casting ...........................
        obs, action, next_obs, reward, _, _ = batch.astuple()

        # .... RLRP-731 batch ``B4-raw``: obs-block RAW target scope guard (Option 1) ............
        # ``raw_passthrough`` threads a SEPARATE raw obs target. Under a robust/block-facade
        # normalizer with ``target_is_delta`` the normalized target is a NORMALIZED-space delta
        # whose raw counterpart is NOT ``next_obs`` (it would need the affine shift), so fail fast
        # (mirrors the removed model-side guard; ``inline_denorm`` shares this delta caveat but it
        # is out of B4 scope). Every non-opted-in caller keeps the 2-tuple contract untouched.
        if return_raw_target and self.target_is_delta and self._uses_block_facade:
            raise NotImplementedError(
                "history_drift_target_mode='raw_passthrough' with target_is_delta under a "
                "robust/block-facade normalizer is out of scope (RLRP-731 batch B4): the "
                "normalized target is a normalized-space delta with no next_obs raw counterpart."
            )

        # .... Stage ``A2`` action-plan scope guards (risks R2 / R7) ..............................
        # Introduced by stage ``A2`` of the Fix the MS->MS future-action-plan conditioning
        # contract ``.junie`` plan
        # (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).
        if return_action_plan:
            if return_raw_target:
                raise NotImplementedError(
                    "_process_batch(return_raw_target=True, return_action_plan=True) is not "
                    "supported: both opt-ins claim the same THIRD return slot and no model "
                    "needs them together (raw_passthrough is the MTM-Pro deploy-history-drift "
                    "path; the action plan is the ms2ms conditioning path)."
                )
            if self.target_is_delta:
                # R2: with ``target_is_delta`` the composed next-obs action columns are
                # NORMALIZED-space DELTAS, not actions, so slicing a plan out of them would
                # silently feed garbage into the conditioning channel. Fail fast instead.
                raise NotImplementedError(
                    "_process_batch(return_action_plan=True) with target_is_delta is out of "
                    "scope (Fix the MS->MS future-action-plan conditioning contract .junie "
                    "plan, stage A2 / risk R2): the next_obs action columns would be deltas, "
                    "not the planned actions a_{t+1..t+F-1}. Note that MS-obs/MS-next-obs "
                    "training with target_is_delta is already unsupported (see the "
                    "MultiStepMLP NotImplementedError gate below)."
                )

        if isinstance(obs, np.ndarray):
            _warn_numpy_input("obs", "_process_batch")
        if isinstance(action, np.ndarray):
            _warn_numpy_input("action", "_process_batch")
        if isinstance(next_obs, np.ndarray):
            _warn_numpy_input("next_obs", "_process_batch")
        obs = model_util.to_tensor(obs)
        action = model_util.to_tensor(action)
        next_obs = model_util.to_tensor(next_obs)

        if self._uses_block_facade:
            # Robust normalizer path: compute target in normalized observation space
            obs_t = obs.to(self.device, non_blocking=True)
            next_obs_t = next_obs.to(self.device, non_blocking=True)
            if self.target_is_delta:
                if isinstance(self.model, MultiStepMLP):
                    raise NotImplementedError(
                        "Can't work in training via _process_batch method in the casse of MS-obs MS-next-obs. Would require data preprocessing at the replaybuffer level. Only keep the sample method related logic for inference."
                    )
                target_obs = self._normalize_composed_obs(
                    next_obs_t, layout="output"
                ) - self._normalize_composed_obs(obs_t)
                for dim in self.no_delta_list:
                    target_obs[..., dim] = self._normalize_composed_obs(
                        next_obs_t, layout="output"
                    )[..., dim]
            else:
                # RLRP-824: the composed next_obs TARGET spans the OUTPUT window
                # ``W = output_window_len`` obs blocks (``== history_len`` for every legacy
                # config, hence bit-exact); the explicit ``layout`` makes the ``W != H`` case
                # unambiguous (plan risk R2).
                target_obs = self._normalize_composed_obs(next_obs_t, layout="output")

            if self.learned_rewards:
                if isinstance(reward, np.ndarray):
                    _warn_numpy_input("reward", "_process_batch")
                reward_t = model_util.to_tensor(reward)
                reward_t = reward_t.to(self.device, non_blocking=True).unsqueeze(
                    reward_t.ndim
                )
                target = torch.cat([target_obs, reward_t], dim=obs_t.ndim - 1)
            else:
                target = target_obs

            model_in = self._get_model_input(obs, action)

            compute_dtype = (
                torch.double
                if self.model_use_double_precision
                and isinstance(self.model, ExponentialFamilyMLP)
                else torch.float
            )
            target = target.to(compute_dtype)

            # RLRP-731 batch ``B4-raw``: obs-block RAW (pre-normalization) target for the DH
            # ``raw_passthrough`` mode. Obs block only (no reward concat — DH slices
            # ``[: singlestep_obs_len]``), cast with the SAME ``compute_dtype`` rule as ``target``
            # so the terminal-step residual subtraction stays in one dtype. ``target_is_delta``
            # was already rejected by the scope guard above.
            target_raw_obs = next_obs_t.to(compute_dtype) if return_raw_target else None

            if _as_float:
                warn_once_float32_downcast(
                    "OneDTransitionRewardModelV2._process_batch:robust:_as_float",
                    tensor_name="target",
                    from_dtype=target.dtype,
                    reason="_as_float=True forces a float32 target",
                )
                target = target.float()
                if target_raw_obs is not None:
                    target_raw_obs = target_raw_obs.float()

            # Stage ``A2`` of the Fix the MS->MS future-action-plan conditioning contract
            # ``.junie`` plan
            # (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``):
            # the plan is sliced from the RAW ``next_obs`` (NOT from the normalized ``target``,
            # which lives in the composed-next-obs target space -- decision ``D3`` / risk ``R1``).
            if return_action_plan:
                action_plan = self._extract_action_plan_from_composed_next_obs(
                    obs, next_obs_t
                ).to(compute_dtype)
                return model_in, target, action_plan

            if return_raw_target:
                return model_in, target, target_raw_obs
            return model_in, target

        # Standard normalizer path
        if self.target_is_delta:
            if isinstance(self.model, MultiStepMLP):
                raise NotImplementedError(
                    "Can't work in training via _process_batch method in the casse of MS-obs MS-next-obs. Would require data preprocessing at the replaybuffer level. Only keep the sample method related logic for inference."
                )

            if next_obs.dtype != torch.double or obs.dtype != torch.double:
                obs = obs.double()
                next_obs = next_obs.double()

            target_obs = next_obs - obs
            for dim in self.no_delta_list:
                target_obs[..., dim] = next_obs[..., dim]

        else:
            target_obs = next_obs

        # C3 — non_blocking H→D copies on the legacy-normalizer hot path.
        target_obs = target_obs.to(self.device, non_blocking=True)

        if self.learned_rewards:
            if isinstance(reward, np.ndarray):
                _warn_numpy_input("reward", "_process_batch")
            reward = model_util.to_tensor(reward)
            reward = reward.to(self.device, non_blocking=True).unsqueeze(reward.ndim)
            target = torch.cat([target_obs, reward], dim=obs.ndim - 1)
        else:
            target = target_obs

        model_in = self._get_model_input(obs, action)

        if self.model_use_double_precision and isinstance(
            self.model, ExponentialFamilyMLP
        ):
            target = target.double()
        else:
            warn_once_precision_mismatch(
                "OneDTransitionRewardModelV2._process_batch:target",
                requested_double_precision=self.model_use_double_precision,
                resolved_dtype=torch.float32,
                device=self.device,
            )
            target = target.float()

        if _as_float:
            warn_once_float32_downcast(
                "OneDTransitionRewardModelV2._process_batch:_as_float",
                tensor_name="target",
                from_dtype=target.dtype,
                reason="_as_float=True forces a float32 target",
            )
            target = target.float()

        # Stage ``A2`` of the Fix the MS->MS future-action-plan conditioning contract ``.junie``
        # plan (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).
        # On this (standard-normalizer) path the target is already RAW, but the plan is STILL
        # sliced from ``next_obs`` and pushed through the INPUT normalizer -- the input actions
        # are normalized while the target is not, so reusing the target tensor would put the
        # conditioning channel in the wrong space (decision ``D3`` / risk ``R1``).
        if return_action_plan:
            action_plan = self._extract_action_plan_from_composed_next_obs(
                obs, next_obs
            ).to(target.dtype)
            return model_in, target, action_plan

        if return_raw_target:
            # RLRP-731 batch ``B4-raw``: the asymmetric ``standard`` normalizer keeps targets in
            # RAW space, so ``raw_passthrough`` degenerates to ``inline_denorm`` (denorm no-op).
            # Return the obs-block target (no reward), matching ``target``'s dtype.
            target_raw_obs = target_obs.to(target.dtype)
            return model_in, target, target_raw_obs
        return model_in, target

    @torch.compiler.disable
    def _extract_action_plan_from_composed_next_obs(
        self, obs: torch.Tensor, next_obs_raw: torch.Tensor
    ) -> torch.Tensor:
        """Slice the future-action plan ``a_{t+1..t+F-1}`` out of the RAW composed ``next_obs``.

        Permanent training-path plumbing. Introduced by stage ``A2`` of the Fix the MS->MS
        future-action-plan conditioning contract ``.junie`` plan
        (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).

        **Why this is plumbing and not a refactor.** The planned actions are NOT absent from a
        multistep training batch: ``MultistepDataBufferProcessorAbstract.get_compose_next_obs``
        composes the target as ``[o_{t+F-H+1..t+F}, a_{t+F-H+1..t+F-1}]`` -- its ``H-1`` action
        columns **are** the plan (a superset of it when ``F < H``). Only the *input-side routing*
        was missing (the RLRP-781 defect). No replay-buffer refactor is required.

        **Layout (derived, never hard-coded to a magic number).** The composed next-obs action
        block starts right after the ``W`` observation steps and holds ``W-1`` action steps, slot
        ``k`` being ``a_{t+F-W+1+k}``, with ``W = output_window_len = max(H, F)`` (RLRP-824;
        ``W == H`` for every legacy config). The plan ``a_{t+1..t+F-1}`` is therefore the
        **trailing** ``F-1`` action steps, i.e. slots ``W-F .. W-2`` (the whole block when
        ``W == F``). Extracted through the canonical
        ``extract_act_features_from_multistep_composed_array(is_model_output=True)`` (FR10).

        **Normalization space (decision D3 / risk R1).** The plan is sliced from the **RAW**
        ``next_obs`` and pushed through :meth:`_get_model_input` -- the *same* primitive that
        normalizes ``a_t`` into the model input -- so each plan step ends up in the **input**
        normalization space, byte-consistent with the action columns the model already sees. It is
        deliberately NOT taken from the normalized ``target``: the composed-next-obs target lives
        in a different (target / innovation-scaled) space than the input, which would inject a new
        silent distribution shift in place of the old one. Routing through
        :meth:`_get_model_input` also keeps this correct for BOTH normalizer regimes (the
        block-facade ``_normalize_composed_act(space="input")`` path and the flat
        ``input_normalizer.single`` path) without ever reaching into normalizer statistics by
        column offset.

        **Label-leakage discipline (risk R3).** Only ACTION columns are ever read; the slice width
        is asserted to be exactly ``(F-1) * singlestep_act_len`` so no observation column (the very
        quantity the model must predict) can leak into the conditioning channel.

        PERF (RLRP-824, Valeria A100 training-step benchmark of 2026-09-17): the former
        per-plan-step :meth:`_get_model_input` loop re-normalized the obs block ``F-1`` times
        and, worse, paid the ``strict_finite`` device sync ``F-1`` times per step -- on the
        ULH windows (``F=500``) that was ~1500 ``cudaStreamSynchronize`` per training step and
        the bulk of the ~400-600 ms/batch floor common to the NeuroBEM and Husky ``F=500``
        rows (E2E-TCN H20/F500 with the geometry term off: 605 ms/batch, GPU kernel-busy
        13 %). The ``F-1`` steps are now normalized in ONE :meth:`_get_model_input` call: the
        obs history is broadcast along a new step axis, the ``(N, F-1)`` leading dims are
        flattened to rows (the normalizers are per-column, so a row is a row) and the
        ``a_t`` slot is read back per row. Same primitive, same space, identical values.

        :param obs: the composed observation/action history of the batch (model input layout).
        :param next_obs_raw: the RAW (non-normalized) composed ``next_obs`` target of the batch.
        :return: the plan ``a_{t+1..t+F-1}``, shape ``(..., horizon_len - 1, singlestep_act_len)``,
            in the model INPUT normalization space. Zero-length step axis when ``horizon_len == 1``
            (the legitimate empty-plan no-op of the normative contract).
        """
        history_len = int(self.model.history_len)
        horizon_len = int(self.model.horizon_len)
        obs_len = int(self.model.singlestep_obs_len)
        act_len = int(self.model.singlestep_act_len)
        # RLRP-824 (KD1 / FR14 c): the composed next_obs is ``[obs x W][act x (W-1)]`` with
        # ``W = output_window_len = max(H, F)`` -- ``H`` for every legacy config (bit-exact),
        # ``F`` on the asymmetric window, where the ``W-1 == F-1`` action columns ARE the plan.
        output_window_len = int(
            getattr(self.model, "output_window_len", history_len)
        )

        plan_steps = horizon_len - 1
        if plan_steps == 0 or act_len == 0:
            # ``F == 1`` => the empty plan (all the driving information, ``a_t``, is already in
            # the history). Return an explicit zero-length step axis rather than ``None`` so the
            # 3-tuple contract stays shape-typed.
            return next_obs_raw.new_zeros(
                (*next_obs_raw.shape[:-1], plan_steps, act_len)
            ).to(self.device, non_blocking=True)

        act_block_start = obs_len * output_window_len
        act_block_end = act_block_start + (output_window_len - 1) * act_len
        assert next_obs_raw.shape[-1] >= act_block_end, (
            f"composed next_obs width {next_obs_raw.shape[-1]} is smaller than the expected "
            f"multistep layout width {act_block_end} ({history_len=}, {horizon_len=}, "
            f"{output_window_len=}, {obs_len=}, {act_len=})."
        )

        # R3 / RLRP-824 FR10: ACTION columns only, through the canonical composed-observation
        # act-block extractor (``(..., Da, W-1)`` act steps ``a_{t+F-W+1..t+F-1}``), never an
        # open-coded offset into the flat row. The plan ``a_{t+1..t+F-1}`` is the TRAILING
        # ``F-1`` of those ``W-1`` action steps (all of them when ``W == F``). Observation
        # columns (the very quantity the model must predict) can therefore never leak into the
        # conditioning channel: the extractor is bounded to ``[Do*W, out_size)`` by construction.
        act_steps_raw = self.model.extract_act_features_from_multistep_composed_array(
            next_obs_raw[..., :act_block_end], is_model_output=True, vectorized=True
        )
        assert act_steps_raw is not None and act_steps_raw.shape[-1] == output_window_len - 1, (
            f"composed next_obs ACTION block yielded {None if act_steps_raw is None else act_steps_raw.shape[-1]} "
            f"steps, expected W-1 = {output_window_len - 1} ({history_len=}, {horizon_len=})."
        )
        # (..., Da, W-1) -> (..., W-1, Da) -> trailing F-1 steps == the plan.
        plan_raw = act_steps_raw.transpose(-2, -1)[..., -plan_steps:, :]
        assert plan_raw.shape[-2:] == (plan_steps, act_len)

        # Normalize every plan step through the SAME input primitive as ``a_t`` (D3), by feeding
        # it in the single-step action slot and reading that slot back from the normalized input.
        # The slot lives in the model INPUT layout ``[obs x H][act x H]`` (NOT the ``W``-window
        # output layout): ``a_t`` is the LAST of the ``H`` input action steps.
        input_act_block_start = obs_len * history_len
        a_t_slot = slice(
            input_act_block_start + (history_len - 1) * act_len,
            input_act_block_start + history_len * act_len,
        )
        # RLRP-824: ONE normalization call for the F-1 steps instead of a Python loop of F-1
        # (each with its own strict-finite device sync). Broadcast the obs history along the
        # step axis, flatten ``(..., F-1)`` to rows, normalize, read the ``a_t`` slot back.
        lead_shape = tuple(plan_raw.shape[:-2])
        obs_rows = (
            obs.unsqueeze(-2)
            .expand(*lead_shape, plan_steps, obs.shape[-1])
            .reshape(-1, obs.shape[-1])
        )
        plan_rows = plan_raw.reshape(-1, act_len)
        assert obs_rows.shape[0] == plan_rows.shape[0], (
            f"obs history rows {obs_rows.shape[0]} != plan rows {plan_rows.shape[0]} "
            f"({tuple(obs.shape)=}, {tuple(plan_raw.shape)=})."
        )
        normalized_rows = self._get_model_input(obs_rows, plan_rows)[..., a_t_slot]
        return normalized_rows.reshape(*lead_shape, plan_steps, act_len)

    @torch.compiler.disable
    def _normalize_future_action_plan(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Carry a RAW test-time action plan into the model INPUT normalization space.

        Permanent deploy-path plumbing. Introduced by the RLRP-757 test-time rollout review
        (``review_e2e_tcn_testtime_selffed_rollout_report_RLRP-757_20260912.md``), which traced the
        physically-impossible ``SELFFED`` / ``OPENLOOP`` per-step MAE of the E2E-TCN arms
        (MAE ``28..170`` on a state space bounded by ``+-16``, blowing up at ``h == 2`` of the
        very first, fully ground-truth-anchored window) to a **train/test space mismatch** on the
        conditioning channel:

        - at TRAINING time the plan is pushed through :meth:`_get_model_input` (see
          :meth:`_extract_action_plan_from_composed_next_obs`, decision ``D3``), so the model
          learns a plan expressed in the **input** normalization space;
        - at TEST time :meth:`forecast_open_loop_per_horizon` used to hand ``future_actions``
          to ``model.forecast`` **verbatim**, i.e. in RAW env units, while the ``a_t`` the plan
          is concatenated to (``_build_driving_action_sequence``) came from the *normalized*
          ``model_input``. On the Neurobem quadrotor (motor-action ``mean ~ 1330``,
          ``std ~ 350``) that put the conditioning channel ~380 sigma off distribution.

        ``o_{t+1}`` survived the defect (it is driven by ``a_t`` alone, by causality), which is
        exactly the observed signature: a clean ``h == 1`` and garbage from ``h == 2`` on.

        **Same primitive as training, on purpose.** Each plan step is normalized by feeding it in
        the single-step ``a_t`` slot of :meth:`_get_model_input` and reading that slot back --
        byte-for-byte the training-side operation -- rather than by reaching into normalizer
        statistics by column offset. This keeps it correct for BOTH normalizer regimes (the
        block-facade ``_normalize_composed_act(space="input")`` path and the flat
        ``input_normalizer.single`` path) and for the non-affine (``winsorized`` / ``quantile``)
        types whose clip-range clamping makes the transform non-linear.

        The ``a_t`` slot is the **trailing** ``singlestep_act_len`` columns of the composed model
        input (the action block ends with ``a_t``), which holds for both ``(obs, act)`` splits
        used by the call sites (deploy: composed obs block + composed action block; train:
        composed obs+act history + single ``a_t``). A single probe row is enough because the
        per-column normalization does not depend on the companion values.

        **Vectorized over the plan steps (RLRP-824 test-time rollout hotfix, 2026-09-19).** This
        method runs on EVERY control-loop step of the test-time rollouts (compounded / self-fed /
        open-loop, primary deploy AND each ``epoch_checkpoints_rollouts/epoch_<E>/`` pass). The
        previous revision normalized the plan with a Python loop of ``F-1`` sequential
        :meth:`_get_model_input` calls (clone + slot assignment + normalizer forward on a 1-row
        tensor each): on the ULH rows (``F=1000`` Lorenz, ``F=500`` PI-TCN / NeuroBEM) that is
        ~1000 tiny kernel launches per rollout step -- CPU-dispatch-bound with the GPU idle
        (``nvtop`` shows the rollouts "running on the CPU"), ~180 ms/step, i.e. ~3 h per
        epoch-checkpoint deploy on the 10k-step Lorenz trajectories. The ``F-1`` steps are now
        normalized in ONE :meth:`_get_model_input` call, mirroring the training-side
        :meth:`_extract_action_plan_from_composed_next_obs`: the ``(o, a)`` probe row is
        broadcast along the step axis, the ``(..., F-1)`` leading dims are flattened to rows (the
        normalizers are per-column, so a row is a row) and the ``a_t`` slot is read back per row.
        Same primitive, same space, identical values.

        :param obs: the ``obs`` argument as handed to the forecast head (RAW, model input layout).
        :param act: the ``act`` argument as handed to the forecast head (RAW); its trailing
            ``singlestep_act_len`` columns are the ``a_t`` slot.
        :param future_actions: the RAW planned sequence ``a_{t+1..t+F-1}``, shape
            ``(..., horizon_len - 1, singlestep_act_len)``.
        :return: the same-shaped plan in the model INPUT normalization space (and in the
            ``model_input`` dtype so it concatenates cleanly inside the model). A no-op
            pass-through when no input normalizer is active (``normalize=False`` wrappers), which
            keeps the plan-free / unnormalized test fixtures bit-exact.
        """
        act_len = int(self.model.singlestep_act_len)
        plan_raw = model_util.to_tensor(future_actions).to(
            self.device, non_blocking=True
        )

        if act_len == 0 or plan_raw.ndim < 2 or plan_raw.shape[-2] == 0:
            return plan_raw
        if not self._input_normalization_active():
            # No normalizer: the model sees RAW obs/act, so a RAW plan is already in the
            # right space (this is the `normalize=False` unit-test configuration).
            return plan_raw

        assert plan_raw.shape[-1] == act_len, (
            f"the action plan last dim {plan_raw.shape[-1]} is not the single-step action width "
            f"{act_len}: refusing to normalize a plan whose layout is unknown."
        )
        assert act.shape[-1] >= act_len, (
            f"the composed action history width {act.shape[-1]} is smaller than one action step "
            f"{act_len}: the trailing `a_t` slot used as the normalization probe does not exist."
        )
        if not bool(torch.isfinite(plan_raw).all()):
            raise ValueError(
                "the test-time future-action plan contains non-finite values; refusing to "
                "normalize it into the model conditioning channel."
            )

        plan_steps = int(plan_raw.shape[-2])
        lead_shape = tuple(plan_raw.shape[:-2])
        # ONE probe row per plan step: ``(..., F-1, Da)`` flattened row-major to
        # ``(N*(F-1), Da)`` so the reshape back to ``(*lead_shape, F-1, Da)`` below is exact.
        plan_rows = plan_raw.reshape(-1, act_len)
        n_rows = plan_rows.shape[0]

        obs_probe = obs.reshape(-1, obs.shape[-1])[:1].expand(n_rows, -1)
        act_probe = act.reshape(-1, act.shape[-1])[:1].expand(n_rows, -1).clone()
        act_probe[:, -act_len:] = plan_rows.to(act_probe.dtype)

        # Single normalizer call for every plan step (see the docstring: the per-step loop was
        # the CPU-dispatch-bound hot spot of the ULH test-time rollouts).
        plan_norm = self._get_model_input(
            obs=obs_probe,
            action=act_probe,
            normalize_strict_finite=False,
        )[..., -act_len:]
        plan_norm = plan_norm.reshape(*lead_shape, plan_steps, act_len)

        self._assert_normalized_plan_is_plausible(plan_norm)
        return plan_norm

    def _input_normalization_active(self) -> bool:
        """Whether :meth:`_get_model_input` actually normalizes (either regime)."""
        return bool(self.input_normalizer) or bool(self._uses_block_facade)

    @staticmethod
    def _assert_normalized_plan_is_plausible(plan_norm: torch.Tensor) -> None:
        """Fail loud when the normalized conditioning channel is off-distribution.

        Regression pin for the RLRP-757 train/test plan-space mismatch: a plan still expressed in
        RAW env units lands dozens-to-hundreds of sigma away from the ``N(0, 1)``-ish channel the
        model was trained on (measured: ~380 sigma on the Neurobem motor actions). A silent
        garbage forecast is far worse than a crash here, so refuse it instead of publishing it.

        The bound is deliberately generous: legitimate out-of-distribution *control* (an
        aggressive plan) stays within a few sigma, and the non-affine normalizer types clamp to
        their clip range anyway.
        """
        if plan_norm.numel() == 0:
            return
        max_abs = float(torch.max(torch.abs(plan_norm)).item())
        if max_abs > _MAX_ABS_NORMALIZED_ACTION_PLAN_SIGMA:
            raise ValueError(
                f"the normalized future-action plan reaches {max_abs:.1f} sigma "
                f"(> {_MAX_ABS_NORMALIZED_ACTION_PLAN_SIGMA:.0f}), i.e. far outside the input "
                "normalization space the model was trained on. Most likely the plan handed to "
                "`forecast_open_loop_per_horizon` is NOT in raw env action units (it may have "
                "been normalized twice), or the input normalizer statistics do not match the "
                "deployed action space. Refusing to produce a forecast conditioned on an "
                "off-distribution control channel (RLRP-757 test-time rollout review)."
            )

    def _action_plan_needed(self) -> bool:
        """Whether the wrapped model consumes the train-time future-action plan (stage ``A3``).

        Permanent capability gate. Introduced by stage ``A3`` of the Fix the MS->MS
        future-action-plan conditioning contract ``.junie`` plan
        (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).

        Gated by the model-side ``consumes_future_action_plan`` capability flag (decision ``D7``),
        declared ``True`` by :class:`AbstractMS2MSForecast` only -- and now **per instance**: that
        family's ``enable_future_action_plan_conditioning=False`` config knob (operator follow-up
        to the post-implementation review) turns it ``False`` to obtain the genuinely plan-FREE
        baseline, which must therefore never be handed a plan at train time either. Every other
        family reports ``False`` via the ``getattr`` default, so ``_process_batch`` stays a
        2-tuple and their train / eval paths stay bit-exact (risk ``R5``). ``target_is_delta`` is
        excluded here too, mirroring the ``A2`` fail-fast guard (risk ``R2``): such a configuration
        must not silently fall back to the defective plan-free training either, so it keeps failing
        in ``_process_batch``.
        """
        return bool(
            getattr(self.model, "consumes_future_action_plan", False)
            and not self.target_is_delta
        )

    def _dh_raw_target_needed(self) -> bool:
        """Whether the DH ``raw_passthrough`` obs-block RAW target must be threaded (RLRP-731 B4).

        MTM-Pro-family-ONLY support (operator note, 2026-07-05): the extra ``target_raw_HOxDoa``
        kwarg is materialized and threaded ONLY when the wrapped model is a
        :class:`MS2MS2SSArTemporalMixturePME` with the DH sub-term ON and
        ``history_drift_target_mode == 'raw_passthrough'``. Non-family models (and every other DH
        mode) never see the kwarg — the memory / code path is byte-identical to today (the 2-tuple
        ``_process_batch`` contract is preserved), which IS the memory-gating guarantee.
        """
        return (
            isinstance(self.model, MS2MS2SSArTemporalMixturePME)
            and getattr(self.model, "enable_history_drift_loss", False)
            and getattr(self.model, "history_drift_target_mode", "inline_denorm")
            == "raw_passthrough"
        )

    def _assert_train_path_extensions_are_exclusive(self) -> None:
        """Refuse a model that needs BOTH train-path ``_process_batch`` extensions at once.

        Permanent fail-fast guard. Introduced by the operator follow-up to the
        post-implementation review of the Fix the MS->MS future-action-plan conditioning contract
        ``.junie`` plan
        (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).

        :meth:`loss` and :meth:`update` test :meth:`_dh_raw_target_needed` FIRST and return from
        that branch, so a model that ever reported both capabilities would **silently** lose the
        future-action plan -- i.e. it would fall back to exactly the RLRP-781 defect, with nothing
        but an unexplained metric regression to show for it. The two families are disjoint today
        (``MS2MS2SSArTemporalMixturePME`` is not an :class:`AbstractMS2MSForecast`), so this guard
        is unreachable by construction; it exists to make that *assumption* explicit and to fail
        loud the day it stops holding, instead of degrading the conditioning in silence.

        Both extensions also claim the SAME third return slot of ``_process_batch``
        (``return_raw_target`` / ``return_action_plan`` already reject each other there), so
        supporting both would require a real design decision, not a branch reorder.
        """
        assert not (self._dh_raw_target_needed() and self._action_plan_needed()), (
            "the wrapped model reports BOTH the DH raw-target capability (RLRP-731 B4-raw) and "
            "the future-action-plan capability (RLRP-781 A3), but `loss`/`update` can only "
            "thread ONE of the two `_process_batch` third returns: the plan would be silently "
            "dropped by the raw-target branch and the model would train UNCONDITIONED. Decide "
            "how the two conditionings compose (and widen `_process_batch`) before enabling "
            f"both on {type(self.model).__name__}."
        )

    def loss(
        self,
        batch: mbrl_types.TransitionBatch,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the model loss over a batch of transitions.

        RLRP-731 batch ``B4-raw`` (Option 1): when the wrapped MTM-Pro model selects
        ``history_drift_target_mode='raw_passthrough'`` (see :meth:`_dh_raw_target_needed`),
        additionally thread the obs-block RAW target to the model's ``loss``. Every other case
        falls back to the byte-identical inherited (2-tuple) behaviour.
        """
        assert target is None
        self._assert_train_path_extensions_are_exclusive()
        if self._dh_raw_target_needed():
            model_in, target, target_raw = self._process_batch(
                batch, return_raw_target=True
            )
            return self.model.loss(
                model_in, target=target, target_raw_HOxDoa=target_raw
            )
        if self._action_plan_needed():
            # Stage ``A3`` of the Fix the MS->MS future-action-plan conditioning contract
            # ``.junie`` plan
            # (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``):
            # route the training-batch plan ``a_{t+1..t+F-1}`` into the forward pass so the model
            # is TRAINED under the same conditioning it gets at test time (RLRP-781).
            model_in, target, action_plan = self._process_batch(
                batch, return_action_plan=True
            )
            return self.model.loss(
                model_in, target=target, future_actions=action_plan
            )
        model_in, target = self._process_batch(batch)
        return self.model.loss(model_in, target=target)

    # ==== RLRP-786: the two halves of :meth:`loss` for the CUDA-graph captured step ==============
    def process_batch_for_loss(
        self, batch: mbrl_types.TransitionBatch
    ) -> Tuple[torch.Tensor, ...]:
        """HOST half of :meth:`loss`: ``_process_batch`` (normalisation, target composition) ->
        the positional tensor tuple :meth:`loss_from_processed` consumes.

        RLRP-786 (``perf_RLRP-786_cuda_graph_train_step_plan_20260919.md``): the captured region is
        ``loss_from_processed -> backward -> optimizer.step``; this half stays EAGER because the
        mbrl-lib normalizer guards every ``normalize`` with a host-synchronising
        ``if not torch.isfinite(result).all()`` (``utilities/mbrl-lib`` is out of the plan's blast
        radius). Same three branches as :meth:`loss`, same order, so
        ``loss_from_processed(*process_batch_for_loss(b)) == loss(b)`` bit for bit.
        """
        self._assert_train_path_extensions_are_exclusive()
        if self._dh_raw_target_needed():
            return tuple(self._process_batch(batch, return_raw_target=True))
        if self._action_plan_needed():
            return tuple(self._process_batch(batch, return_action_plan=True))
        return tuple(self._process_batch(batch))

    def loss_from_processed(
        self, model_in: torch.Tensor, target: torch.Tensor, extra: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """DEVICE half of :meth:`loss` (see :meth:`process_batch_for_loss`): the wrapped model's
        ``loss`` on pre-processed tensors -- the body recorded in the CUDA graph."""
        if self._dh_raw_target_needed():
            return self.model.loss(model_in, target=target, target_raw_HOxDoa=extra)
        if self._action_plan_needed():
            return self.model.loss(model_in, target=target, future_actions=extra)
        return self.model.loss(model_in, target=target)

    @deprecated(
        reason=(
            "Model.update is deprecated and will be removed in a future version. "
            "Please use `training_step` or `pytorch_lightning.Trainer` instead."
        )
    )
    def update(
        self,
        batch: mbrl_types.TransitionBatch,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Updates the model given a batch of transitions and an optimizer.

        RLRP-731 batch ``B4-raw`` (Option 1): same ``raw_passthrough`` threading as :meth:`loss`
        (the mbrl ``Model.update`` calls ``self.loss`` positionally, so the family provides its own
        ``update`` override that accepts the extra ``target_raw_HOxDoa`` kwarg). Every other case
        falls back to the byte-identical inherited behaviour.
        """
        assert target is None
        self._assert_train_path_extensions_are_exclusive()
        if self._dh_raw_target_needed():
            model_in, target, target_raw = self._process_batch(
                batch, return_raw_target=True
            )
            return self.model.update(
                model_in, optimizer, target=target, target_raw_HOxDoa=target_raw
            )
        if self._action_plan_needed():
            # Stage ``A3`` (see :meth:`loss`).
            model_in, target, action_plan = self._process_batch(
                batch, return_action_plan=True
            )
            return self.model.update(
                model_in, optimizer, target=target, future_actions=action_plan
            )
        model_in, target = self._process_batch(batch)
        return self.model.update(model_in, optimizer, target=target)

    def eval_score(
        self,
        batch: mbrl_types.TransitionBatch,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Evaluates the model score over a batch of transitions.

        Permanent contract change. Introduced by stage ``A3`` (decision ``A3.5`` / risk ``R9``) of
        the Fix the MS->MS future-action-plan conditioning contract ``.junie`` plan
        (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).

        The validation metric MUST be measured under the **same conditioning as the training
        objective**: a plan-conditioned :meth:`loss` paired with a plan-free ``eval_score`` would
        replace the train/test conditioning shift this plan fixes with a subtler train/validation
        one, and the validation score is what ranks checkpoints (model selection / early stopping).

        Every non-opted-in model keeps the inherited 2-tuple path, byte-for-byte unchanged.
        """
        assert target is None
        self._assert_train_path_extensions_are_exclusive()
        if self._action_plan_needed():
            with torch.no_grad():
                model_in, target, action_plan = self._process_batch(
                    batch, return_action_plan=True
                )
                return self.model.eval_score(
                    model_in, target=target, future_actions=action_plan
                )
        return super().eval_score(batch, target=target)

    def update_normalizer(self, batch: mbrl_types.TransitionBatch):
        """Update normalizer statistics from a full replay-buffer batch.

        Delegates to the parent class which handles both the standard
        ``input_normalizer`` and the robust ``obs_normalizer`` /
        ``act_normalizer``.  After the update, normalizer references are
        re-propagated to the wrapped model via
        :meth:`_set_model_normalizer_handle`.
        """
        super().update_normalizer(batch)
        if self.input_normalizer:
            consol_msg_universal_one_liner("Input normalizer update DONE")
        if self._uses_block_facade:
            consol_msg_universal_one_liner(
                f"Robust ({self.normalizer_type}) obs/act normalizer update DONE"
            )

        self._set_model_normalizer_handle()
        self._bind_train_time_dr_normalizer_scale(batch)
        return None

    def _bind_train_time_dr_normalizer_scale(self, batch) -> None:
        """Resolve the train-time DR scale against the fitted statistics.

        RLRP-761 S1.7 (risk ``R-I``). ``train_time_domain_randomization``
        perturbs tensors that are already normalized, so with
        ``per_feature_scale_mode: relative_to_normalized_std`` the configured
        vector is DIMENSIONLESS and must be multiplied by the normalizer's own
        per-dim output std. That std only exists once the statistics are fitted,
        hence this late binding, re-applied on every refit.

        Strict no-op in the default ``absolute`` mode and whenever the model
        carries no randomizer.
        """
        binder = getattr(
            self.model,
            "bind_train_time_domain_randomization_normalizer_scale",
            None,
        )
        if binder is None:
            return None
        randomizer = getattr(self.model, "train_time_domain_randomizer", None)
        if randomizer is None:
            return None
        # RLRP-761 S4.10: the "target" DR segment lives in the TARGET obs space,
        # which differs from the input one only under
        # ``standard_symmetric_innovation``. Bound BEFORE the relative-scale
        # binding below, which rebuilds the target vector from the input one.
        gain_binder = getattr(randomizer, "bind_target_space_obs_gain", None)
        if gain_binder is not None:
            gain_binder(self.ar_bridge_gain)
        if (
            getattr(randomizer, "per_feature_scale_mode", "absolute")
            != "relative_to_normalized_std"
        ):
            if bool(getattr(randomizer, "enable", False)):
                # RLRP-761 risk R-I: an ABSOLUTE per_feature_scale is expressed
                # in normalizer-output units, so it is silently invalidated by
                # any change to the normalization contract (the S1.5 handler
                # flip alone shifts the effective dt noise by ~x90).
                warnings.warn(
                    "Train-time domain randomization is ENABLED with "
                    "per_feature_scale_mode='absolute': the configured scale is "
                    "expressed in NORMALIZER-OUTPUT units and is NOT invariant to "
                    "the normalization contract. Re-calibrate it, or switch to "
                    "per_feature_scale_mode='relative_to_normalized_std' "
                    "(RLRP-761 S1.7 / risk R-I).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return None
        # RLRP-761 S10.2: the scale primitive now lives in the dedicated
        # lower-layer module (this DR late-binding is its third consumer,
        # alongside the diagnostic and the loss-weight resolver).
        from tools.feature_handling_tools.feature_target_space import (
            resolve_normalizer_output_std,
        )

        sigma_norm = resolve_normalizer_output_std(self, batch)
        if sigma_norm is None:
            warnings.warn(
                "OneDTransitionRewardModelV2: train-time domain randomization is "
                "configured with per_feature_scale_mode="
                "'relative_to_normalized_std' but the normalizer output std could "
                "not be resolved; the configured DIMENSIONLESS scale is used as-is "
                "(RLRP-761 S1.7).",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        binder(sigma_norm)
        return None

    # ---- RLRP-761 P1 — prediction-statistics space ------------------------------------------
    #
    # Key names of the unit-space contract carried alongside the statistics in
    # ``model_state`` (P7.4, the "cheap interim" variant of the typed container:
    # the mapping is kept, but the space becomes an EXPLICIT, asserted key).
    STATS_SPACE_KEY: str = "stats_space"
    STATS_VARIANCE_APPROXIMATION_KEY: str = "variance_approximation"
    STATS_LOGVAR_ABSENT_KEY: str = "ensemble_logvars_is_absent"

    def _denormalize_prediction_statistics(
        self,
        model_state: Optional[Dict[str, torch.Tensor]],
        next_obs: torch.Tensor,
        *,
        add_obs_baseline_norm: Optional[torch.Tensor] = None,
        site: str = "",
    ) -> None:
        """Carry ``model_state`` prediction statistics to PHYSICAL space, in place.

        RLRP-761 ``P1.10``-``P1.16``. Every deploy-time prediction is returned
        twice: as ``next_obs`` (denormalized) and as
        ``model_state["ensemble_means"/"ensemble_logvars"]``. Historically only
        the first was denormalized, so eight consumers (rollout figures, the
        persisted ``TTRPM*.pkl``, the multirun aggregation, the TensorBoard
        uncertainty panels and the math-env SNR/MAE plots) read a NORMALIZED
        tensor as if it were physical — inflated by ``1/sigma_target``, i.e. ``1``
        under ``normalizer_type='standard'`` (why it stayed invisible) and
        ``3-40x`` under ``standard_symmetric_innovation``.

        What this does, in the order the producers do it for ``next_obs``:

        1. drop the reward column, if any (``P1.9`` — ``next_obs`` is
           ``preds[:, :-1]`` while the statistics channel is **not** sliced);
        2. re-baseline the mean for a delta target (``P1.14``);
        3. denormalize the mean, and transport the log-variance with the
           **squared** slope (``P1.1``), skipping the "no variance" sentinel
           (``P1.6``);
        4. stamp the resulting unit space and the variance-approximation regime.

        Args:
            model_state: the mutated statistics mapping (no-op when ``None``).
            next_obs: the already-denormalized observation, used as the layout
                reference of the width assert.
            add_obs_baseline_norm: the NORMALIZED ``O_t`` baseline, required iff
                ``self.target_is_delta``.
            site: caller name, used in error messages.
        """
        if model_state is None:
            return
        means = model_state.get("ensemble_means", None)
        if means is None:
            return
        logvars = model_state.get("ensemble_logvars", None)
        logvars_absent = bool(model_state.get(self.STATS_LOGVAR_ABSENT_KEY, False))

        if not self._uses_block_facade:
            # No output normalizer (``normalizer_type='standard'``): the model
            # target IS raw, so the statistics are already physical. Stamping is
            # still required — a consumer must not have to guess (P4/P7).
            model_state[self.STATS_SPACE_KEY] = "physical"
            model_state[self.STATS_VARIANCE_APPROXIMATION_KEY] = "exact"
            self._attach_prediction_statistics_container(model_state)
            return

        # -- 1. reward slice (P1.9) -------------------------------------------
        obs_width = next_obs.shape[-1]
        expected = obs_width + int(bool(self.learned_rewards))
        if means.shape[-1] != expected:
            raise ValueError(
                f"{site or type(self).__name__}: prediction-statistics width "
                f"{means.shape[-1]} != next_obs width {obs_width} + "
                f"learned_rewards {int(bool(self.learned_rewards))} = {expected}. "
                "The statistics channel must be sliced exactly like `next_obs` "
                "before denormalization (RLRP-761 P1.8/P1.9)."
            )
        reward_part = means[..., obs_width:] if self.learned_rewards else None
        means_obs = means[..., :obs_width]
        logvars_obs = (
            logvars[..., :obs_width]
            if (logvars is not None and logvars.shape[-1] == expected)
            else logvars
        )

        # -- 2. delta re-baseline (P1.14) --------------------------------------
        if self.target_is_delta:
            if add_obs_baseline_norm is None:
                raise ValueError(
                    f"{site}: target_is_delta requires the NORMALIZED baseline to "
                    "re-baseline the statistics channel (RLRP-761 P1.14)."
                )
            rebaselined = means_obs + add_obs_baseline_norm
            for dim in self.no_delta_list:
                rebaselined[..., dim] = means_obs[..., dim]
            means_obs = rebaselined
            # P1.15 — the log-variance takes NO baseline term: under an additive
            # DETERMINISTIC baseline the variance of the delta equals the variance
            # of the state, so only the P1.1 scale transform applies below.

        # -- 3. denormalize (P1.1) --------------------------------------------
        # ``horizon_steps`` is passed EXPLICITLY: inferring it from the tensor
        # width silently mis-slices a composed layout (risk ``Q-B``).
        means_phys = self.denormalize_predicted_obs(means_obs, horizon_steps=1)
        if logvars_obs is not None and not logvars_absent:
            logvars_phys = self.denormalize_predicted_logvar(
                logvars_obs, mean_norm=means_obs, horizon_steps=1
            )
        else:
            # P1.6 — the sentinel is a "no variance" MARKER, not a statistic:
            # rescaling it would manufacture a per-dimension pseudo-variance.
            logvars_phys = logvars_obs

        if reward_part is not None:
            # The reward column is not part of the obs facade; keep it as-is so the
            # channel stays layout-identical to what the producer emitted.
            means_phys = torch.cat([means_phys, reward_part], dim=-1)
            if logvars_phys is not None and logvars is not None:
                logvars_phys = torch.cat(
                    [logvars_phys, logvars[..., obs_width:]], dim=-1
                )

        model_state["ensemble_means"] = means_phys
        if logvars_phys is not None:
            model_state["ensemble_logvars"] = logvars_phys

        # -- 4. declare the space and the approximation regime (P7.4) ----------
        model_state[self.STATS_SPACE_KEY] = "physical"
        model_state[self.STATS_VARIANCE_APPROXIMATION_KEY] = (
            "exact" if self.output_variance_transport_is_exact() else "local_linear"
        )
        self._attach_prediction_statistics_container(model_state)

    def _attach_prediction_statistics_container(
        self, model_state: Dict[str, torch.Tensor]
    ) -> None:
        """Expose the statistics under the typed container key (RLRP-761 ``P7.1``/``P7.3``).

        The mapping keys stay exactly as they were — the container is an
        ADDITIONAL entry — so every historical consumer is untouched while a new
        one can assert the unit contract at its boundary via
        :meth:`PredictionStatistics.require_space` instead of guessing.
        """
        means = model_state.get("ensemble_means", None)
        if means is None:
            return
        PredictionStatistics(
            mean=means,
            logvar=model_state.get("ensemble_logvars", None),
            space=str(model_state.get(self.STATS_SPACE_KEY, SPACE_UNKNOWN)),
            variance_approximation=str(
                model_state.get(self.STATS_VARIANCE_APPROXIMATION_KEY, VARIANCE_EXACT)
            ),
            variance_absent=bool(model_state.get(self.STATS_LOGVAR_ABSENT_KEY, False)),
            normalizer_type=getattr(self, "normalizer_type", None),
        ).attach_to(model_state)

    def _sanitize_and_monitor_model_output(
        self, tensor: torch.Tensor, name: str = "model_output"
    ) -> torch.Tensor:
        """Replace non-finite model-output entries with finite values, counting them.

        RLRP-684 WS-C (C2): model *outputs* (predictions) may legitimately be
        non-finite early in training, so — unlike input data / statistics, which
        fail fast under ``strict_finite`` — they are tolerated via
        :func:`torch.nan_to_num`.  To keep divergence *observable* rather than
        silently swallowed, this helper counts non-finite entries, accumulates
        per-instance telemetry (``_non_finite_output_calls`` /
        ``_non_finite_output_entries``), and emits a ``RuntimeWarning`` on the
        offending call.

        Overflow-safety: all non-finite entries (``NaN``, ``+inf`` **and**
        ``-inf``) are mapped to ``0.0`` rather than to the ``float`` extrema
        (``torch.nan_to_num``'s default ``±3.4e38`` for ``inf``). A sanitized
        prediction can be fed back auto-regressively as the *next* model input
        (see the deploy rollout), where it is re-normalized as
        ``(val - mean) / std``; a ``±3.4e38`` value divided by a small (floored)
        per-feature ``std`` overflows back to non-finite and would re-trip the
        strict ``normalize`` guard. Mapping the (already garbage) diverged
        prediction to ``0.0`` keeps the downstream normalized value finite while
        the divergence is still surfaced via the telemetry counters/warning.

        Args:
            tensor: the raw model-output tensor.
            name: a short label for diagnostics.

        Returns:
            The sanitized tensor (non-finite entries replaced by ``0.0``).
        """
        finite_mask = torch.isfinite(tensor)
        if not bool(finite_mask.all()):
            non_finite_count = int((~finite_mask).sum().item())
            self._non_finite_output_calls += 1
            self._non_finite_output_entries += non_finite_count
            warnings.warn(
                f"OneDTransitionRewardModelV2: '{name}' contained "
                f"{non_finite_count} non-finite value(s) (tolerated telemetry; "
                f"cumulative calls={self._non_finite_output_calls}, "
                f"entries={self._non_finite_output_entries}). This is expected "
                f"early in training but persistent occurrences indicate divergence.",
                RuntimeWarning,
                stacklevel=2,
            )
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_prediction_and_stats(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Union[np.ndarray, torch.Tensor],
        rng: torch.Generator,
        compute_probability_statistics_over_ensemble: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run a model forward call and compute prediction probability statistics:
         - `compute_probability_statistics_over_ensemble=True` -> `(pred_mean, pred_std, pred_epi)`
         - `compute_probability_statistics_over_ensemble=False` ->
            `(pred_mean_per_ensemble, log_variance_per_ensemble, zeros)`

        :param obs: The observation data.
        :param act: The action taken.
        :param rng: Random number generator to be used.
        :param compute_probability_statistics_over_ensemble:
        :return: A tuple containing the prediction means, log variances and a zeros array
         or prediction means, std and epistemic uncertainty over ensembles.
        """
        assert rng is not None
        model_input = self._get_model_input(obs=obs, action=act)

        with torch.inference_mode():
            self.model.eval()
            ensemble_means, ensemble_logvars = self.forward(
                model_input, rng=rng, use_propagation=False
            )

            # Note: nan handling is usefull when using this method early in the trainning stage.
            # RLRP-684 WS-C (C2): sanitize + COUNT non-finite model outputs (tolerant
            # telemetry) rather than silently swallowing them via a bare nan_to_num.
            ensemble_means = self._sanitize_and_monitor_model_output(
                ensemble_means, name="ensemble_means"
            )
            ensemble_logvars = torch.nan_to_num(ensemble_logvars)

            if self._uses_block_facade:
                # RLRP-684 WS-A: model OUTPUT tensor -> layout-aware denorm.
                # ``ensemble_means`` is a prediction in NORMALIZED output space
                # (obs/horizon layout, here obs-only single-step, ``W = Do``), so
                # route it through the layout-aware ``denormalize_predicted_obs``
                # (infers step count ``k = W // Do``) instead of the input-layout
                # shim ``_denormalize_composed_obs`` (which hardcodes
                # ``Do * history_len`` and crashed on the obs/horizon layout).
                # RLRP-761 P1.12 — the log-variance used to be left in NORMALIZED
                # space here while the mean was denormalized, i.e. this "good" path
                # was only half-correct. Transport it with the SQUARED slope, using
                # the SAME step count as the mean. ``k`` is resolved once, from the
                # mean's own width, and passed EXPLICITLY to both calls so the two
                # channels can never be sliced differently (RLRP-761 P1.8): a
                # multistep model reaches this method with a COMPOSED output whose
                # width is not ``Do``.
                stats_horizon_steps = ensemble_means.shape[-1] // self._Do
                ensemble_logvars = self.denormalize_predicted_logvar(
                    ensemble_logvars,
                    mean_norm=ensemble_means,
                    horizon_steps=stats_horizon_steps,
                )
                ensemble_means = self.denormalize_predicted_obs(
                    ensemble_means, horizon_steps=stats_horizon_steps
                )

            if (
                compute_probability_statistics_over_ensemble
                and self.model.num_members > 1
            ):
                assert ensemble_means.ndim >= 3
                assert ensemble_means.size(0) == self.model.num_members

                (
                    means,
                    pred_std,
                    pred_epi,
                ) = compute_trajectory_probability_statistics_over_ensemble(
                    ensemble_pred=ensemble_means,
                    ensemble_pred_logvar=ensemble_logvars,
                    to_tensor=True,
                )

                # .... Memory optimization ........................................................
                del ensemble_means, ensemble_logvars, model_input, obs, act

                return means, pred_std, pred_epi
            else:
                # .... Memory optimization ........................................................
                del model_input, obs, act

                # ToDo: assessment >> shouldn't it return pred_epi=None instead?
                #   Being explicit in this case is maybe conceptualy more intuitive then returning
                #   a meaning-less array however its more error prone programaticaly then returning
                #   a tensor which is the expected output upstream.
                if self.model.num_members > 1:
                    return (
                        ensemble_means,
                        ensemble_logvars,
                        torch.full_like(ensemble_means, self._LOGVAR_MIN_LIMIT),
                    )
                else:
                    return (
                        ensemble_means.squeeze(0),
                        ensemble_logvars.squeeze(0),
                        torch.full_like(
                            ensemble_means.squeeze(0), self._LOGVAR_MIN_LIMIT
                        ),
                    )

    def sample_1d_direct_model_call(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rng: torch.Generator,
        propagation_indices: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        next_state_sampling_size: int = 1,
        epi_knn=False,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Convenient function that give direct access to the `model.sample_1d` method of the dynamic
        model wrapped by OneDTransitionRewardModelV2 and execute some of the deployment logic
        expected of the `OneDTransitionRewardModelV2.sample` method. Take care of converting the
        `obs` and `act` to the format required by `model.sample_1d`.

        :param act: Tensor containing the current action input used for sampling.
        :param obs: Tensor containing the current observation input used for sampling.
        :param rng: Random generator object used for stochastic sampling from the model.
        :param propagation_indices: Optional tensor specifying indices for ensemble propagation.
        :param deterministic: Boolean flag determining whether the samples are deterministic
                              or stochastic.
        :param next_state_sampling_size: Nb of sample to draw if deterministic=False
        :param epi_knn: Boolean flag activating the Epi k-nearest logic for sampling.
        :param future_actions: optional RAW (env-space) planned future-action sequence
            ``a_{t+1..t+F-1}`` of shape ``(horizon_len - 1, singlestep_act_len)`` for an
            :class:`AbstractMS2MSForecast` model. RLRP-824 Step 6b (operator decision 14): on the
            asymmetric ``W = F > H`` output window the composed action columns ARE the plan, so
            the single-step deploy head (``sample_1d`` -> ``deploy`` -> ``_default_forward`` ->
            ``forecast``) can only be evaluated with one; it is normalized through
            :meth:`_normalize_future_action_plan` (the TRAINING-time primitive) and bound as the
            forward-scoped plan context for the duration of this call. ``None`` (default) is
            byte-identical to the historical plan-free single-step deploy path (every ``F <= H``
            caller is unchanged).
        :return: A tuple containing the next observations tensor and a dictionary with
                 model state statistics.
        """
        assert isinstance(self.model, (GaussianMLPExtended, ExponentialFamilyMLP))
        assert rng is not None

        e_obs, e_act = self._single_step_sample_2_model_ensemble(obs, act)
        # RLRP-684 WS-C: this is a deploy/test-time rollout step whose ``obs`` may
        # be a model prediction fed back auto-regressively; normalize it
        # tolerantly (clamp diverged / overflowing values) instead of failing
        # fast. Genuine data corruption is still caught in the training paths.
        model_input = self._get_model_input(
            obs=e_obs, action=e_act, normalize_strict_finite=False
        )
        model_state = {"propagation_indices": propagation_indices}

        # Note: Assume sample_1d output single-step prediction with reduced ensemble dim
        # i.e., shape=(oDim, )
        with self._bound_deploy_action_plan(e_obs, e_act, future_actions):
            pred_next_obs, model_state = self.model.sample_1d(
                model_input,
                model_state,
                deterministic=deterministic,
                next_state_sampling_size=next_state_sampling_size,
                rng=rng,
                epi_knn=epi_knn,
            )

        pred_next_obs, next_model_state = _reduce_batch_dim(pred_next_obs, model_state)

        next_obs = pred_next_obs[:, :-1] if self.learned_rewards else pred_next_obs
        # RLRP-761 P1.10 — the NORMALIZED baseline used to re-base the statistics
        # channel exactly like ``next_obs`` (``None`` unless ``target_is_delta``).
        stats_baseline_norm: Optional[torch.Tensor] = None

        if self._uses_block_facade:
            # Robust normalizer path: model output is in normalized obs space
            if self.target_is_delta:
                if hasattr(self.model, "history_len"):
                    # obs_at_t = self._fetch_latest_obs_from_ms_obs_act_history(obs, act)
                    # norm_obs_at_t = self._normalize_composed_obs(obs_at_t)
                    # RLRP-761 P1.16 — DEFERRED. When this MS+delta branch is
                    # implemented, the prediction-STATISTICS channel
                    # (``model_state["ensemble_means"/"ensemble_logvars"]``) MUST be
                    # re-baselined and denormalized alongside ``next_obs``, by
                    # passing the normalized baseline to
                    # ``_denormalize_prediction_statistics``. Omitting it leaves the
                    # statistics a physical *delta* instead of a physical next-obs —
                    # the exact class of defect P1 exists to remove.
                    raise NotImplementedError(
                        "ToDo: implement support for MS prediction residual"
                    )
                else:
                    # RLRP-684 WS-C: fed-back deploy prediction -> tolerant normalize.
                    norm_obs_at_t = self._normalize_composed_obs(
                        obs, strict_finite=False
                    )
                # RLRP-761 P1.14 — the statistics channel must be re-baselined
                # with the SAME baseline, otherwise denormalizing it alone yields a
                # physical *delta* rather than a physical next-obs.
                stats_baseline_norm = norm_obs_at_t
                next_obs_norm = next_obs + norm_obs_at_t
                for dim in self.no_delta_list:
                    next_obs_norm[..., dim] = next_obs[..., dim]
                # RLRP-684 WS-A/A3: single-step delta deploy output -> layout-aware
                # denorm (auto ``k = W // Do`` = 1). The MS+delta residual branch
                # above still raises NotImplementedError; when it is implemented it
                # must likewise route through ``denormalize_predicted_obs`` (with the
                # obs/horizon layout), NOT the input-layout ``_denormalize_composed_obs``.
                next_obs = self.denormalize_predicted_obs(next_obs_norm)
            else:
                # RLRP-684 WS-A: non-delta deploy output (obs-only single-step,
                # ``W = Do``) -> layout-aware denorm (auto ``k = W // Do``).
                next_obs = self.denormalize_predicted_obs(next_obs)
        else:
            if self.target_is_delta:
                if hasattr(self.model, "history_len"):
                    obs_at_t = self._fetch_latest_obs_from_ms_obs_act_history(obs, act)
                    tmp_ = next_obs + obs_at_t
                else:
                    tmp_ = next_obs + obs
                for dim in self.no_delta_list:
                    tmp_[..., dim] = next_obs[..., dim]
                next_obs = tmp_

        # RLRP-684 WS-C (C2): the predicted ``next_obs`` is a model OUTPUT that is
        # fed back auto-regressively as the next rollout INPUT (see
        # ``singlestep_model_testtime_rollout_and_collect_pred_stats``). Early in
        # training the model can legitimately emit non-finite predictions; unlike
        # input data (which fails fast via ``strict_finite``), these are tolerated
        # (``nan_to_num``) but COUNTED so divergence stays observable. Sanitizing
        # here prevents a transient non-finite prediction from re-entering the
        # strict ``ZScoreNormalizer.normalize`` guard on the next rollout step.
        next_obs = self._sanitize_and_monitor_model_output(
            next_obs, name="sample_1d_direct_model_call.next_obs"
        )
        # RLRP-761 P1.10 — carry the SIBLING statistics channel to physical space
        # too. Denormalizing only ``next_obs`` left every uncertainty consumer
        # reading a normalized tensor as if it were physical.
        self._denormalize_prediction_statistics(
            next_model_state,
            next_obs,
            add_obs_baseline_norm=stats_baseline_norm,
            site="sample_1d_direct_model_call",
        )
        next_model_state["obs"] = next_obs

        # .... Memory optimization ................................................................
        del model_input, obs, act, e_obs, e_act, model_state

        return next_obs, next_model_state

    @contextlib.contextmanager
    def _bound_deploy_action_plan(
        self,
        e_obs: torch.Tensor,
        e_act: torch.Tensor,
        future_actions: Optional[torch.Tensor],
    ) -> Iterator[None]:
        """Bind a RAW deploy-time plan as the model's forward-scoped plan context (RLRP-824 6b).

        ``None`` -> a bare ``yield`` (the historical plan-free single-step deploy path, byte-exact).
        Otherwise the plan is normalized with :meth:`_normalize_future_action_plan` -- the same
        primitive the training batch and :meth:`forecast_open_loop_per_horizon` use -- and bound
        through ``AbstractMS2MSForecast._active_future_action_plan`` so that the inherited
        ``sample_1d`` -> ``deploy`` -> ``_default_forward`` -> ``forecast`` chain reads it without
        any signature change on the shared families. Only the MS->MS forecast family can consume
        a plan; any other model receiving one fails loud.
        """
        if future_actions is None:
            yield
            return
        from tools.multistep_tools.models.abstract_ms2ms_forecast import (
            AbstractMS2MSForecast,
        )

        if not isinstance(self.model, AbstractMS2MSForecast):
            raise TypeError(
                "future_actions was supplied to the single-step deploy head of a "
                f"{type(self.model).__name__}, but only the AbstractMS2MSForecast family "
                "(E2E-TCN / M3 / TBM) consumes a future-action plan."
            )
        plan_in_model_space = self._normalize_future_action_plan(
            obs=e_obs, act=e_act, future_actions=future_actions
        )
        with self.model._active_future_action_plan(plan_in_model_space):
            yield

    def forecast_open_loop_per_horizon(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        future_actions: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, PredictionStatistics]]:
        """Open-loop per-horizon observation forecast (RLRP-728, model-obs space).

        Companion to :meth:`sample_1d_direct_model_call` for the control-conditioned MS->MS
        forecast baselines (E2E-TCN / M3 / TBM). Instead of returning the single-step (``h == 1``)
        next observation, this returns the **full per-horizon** observation forecast in the model's
        (denormalized) single-step observation space, optionally conditioned on a planned
        ``future_actions`` sequence.

        Scope note (RLRP-728, option 2): the per-horizon predictions are returned in **model
        single-step obs space** (comparable to ground-truth env observations mapped through
        ``DeployerAdapter.target_env_to_model_ss_in_obs``); a target-env-space per-horizon adapter
        is intentionally **out of scope** here (see the RLRP-728 plan section 3.2.1 deviation).

        Forecast axis (RLRP-760 S0): the composed multistep output window is **always**
        ``history_len`` (``H``) steps long -- ``MultistepDataBufferProcessorAbstract`` keeps the
        next-obs window at history length "to simplify handling the case where the
        ``horizon_len < history_len``" -- and that window spans ``o_{t+F-H+1..t+F}``. The
        **forecast** ``ô_{t+1..t+F}`` is therefore the **LAST** ``F = horizon_len`` steps of the
        window (corroborated by ``_build_multistep_derived_state``, whose obs horizon slice starts
        at ``singlestep_obs_len * (H - F)`` and whose single-step deploy head reads the first step
        of that horizon slice). This method returns exactly those ``F`` steps, so index ``h - 1``
        is ``ô_{t+h}`` for **any** ``F <= H`` (before RLRP-760 it returned all ``H`` window steps,
        an alignment that silently held only for ``F == H``).

        Horizon uncertainty (RLRP-761 ``P6``). The forecast head historically discarded its
        variance (``f_mean, _``), which for the baselines cost a calibration metric and for
        **MTM-Pro** discarded the very signal the family relies on — there the forecast head is
        an *informative horizon-uncertainty* channel while the **deploy head is the only
        legitimate rollout surface** (hence the :class:`AbstractMS2MSForecast` guard below, which
        stays). Pass ``return_uncertainty=True`` to also get the per-horizon log-variance,
        **denormalized through the same ``P1.1`` primitive as the mean** and wrapped in a
        :class:`PredictionStatistics` that declares its unit space and its variance-transport
        regime (``local_linear`` for the non-affine ``winsorized`` / ``quantile`` types).

        :param obs: multistep-composed observation history (single anchor), model input layout.
        :param act: multistep-composed action history (single anchor), model input layout.
        :param future_actions: optional planned future-action sequence ``a_{t+1..t+F-1}``, shape
            ``(horizon_len - 1, singlestep_act_len)``; ``None`` keeps the history-echo behaviour.

            Permanent contract change. Introduced by stage ``A1`` of the Fix the MS->MS
            future-action-plan conditioning contract ``.junie`` plan
            (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``): the
            channel was re-axed ``F -> F-1`` because ``a_t`` is already carried by the ``act``
            history and ``a_{t+F}`` drives no predicted step. ``F == 1`` => the plan MUST be empty
            (pass ``None``). Validated by
            ``AbstractMS2MSForecast._validate_future_actions``, which rejects the old ``F`` axis
            with an actionable message.
        :param return_uncertainty: when ``True``, return
            ``(obs_steps, PredictionStatistics)`` instead of ``obs_steps``. Default ``False``
            keeps every existing call site byte-for-byte identical.
        :return: per-horizon denormalized observation steps, shape
            ``(horizon_len, singlestep_obs_len)``; optionally paired with the per-horizon
            statistics (same shape) when *return_uncertainty*.
        """
        from tools.multistep_tools.models.abstract_ms2ms_forecast import (
            AbstractMS2MSForecast,
        )

        assert isinstance(self.model, AbstractMS2MSForecast), (
            "forecast_open_loop_per_horizon requires an AbstractMS2MSForecast model "
            f"(E2E-TCN / M3 / TBM); got {type(self.model)=}."
        )
        if self.target_is_delta:
            # Mirrors the MS+delta residual gate in ``sample_1d_direct_model_call``.
            raise NotImplementedError(
                "forecast_open_loop_per_horizon does not support target_is_delta MS models yet "
                "(ToDo: MS prediction residual, see sample_1d_direct_model_call)."
            )

        with torch.inference_mode():
            self.model.eval()

            e_obs, e_act = self._single_step_sample_2_model_ensemble(obs, act)
            model_input = self._get_model_input(
                obs=e_obs, action=e_act, normalize_strict_finite=False
            )

            # RLRP-757 test-time rollout review: the plan MUST enter the network in the same
            # space as the action columns of ``model_input`` -- i.e. the model INPUT
            # normalization space, exactly like the TRAINING-time plan
            # (``_extract_action_plan_from_composed_next_obs``, decision ``D3``). It used to be
            # forwarded RAW, which silently destroyed every ``h >= 2`` forecast step (see
            # :meth:`_normalize_future_action_plan`).
            plan_in_model_space = (
                None
                if future_actions is None
                else self._normalize_future_action_plan(
                    obs=e_obs, act=e_act, future_actions=future_actions
                )
            )

            # Full flat (normalized) multistep forecast, control-conditioned on the plan.
            f_mean, f_logvar = self.model.forecast(
                model_input, only_elite=True, future_actions=plan_in_model_space
            )

            # Extract the OBS-only per-step columns (avoids the composed obs+act interleave):
            # (..., singlestep_obs_len, history_len) -> (..., history_len, singlestep_obs_len).
            obs_steps_norm = self.model.extract_obs_features_from_multistep_composed_array(
                f_mean, is_model_output=True, vectorized=True
            ).transpose(-2, -1)

            # Layout-aware per-step denorm (each step is obs-only, k == 1).
            obs_steps = self.denormalize_predicted_obs(
                obs_steps_norm, horizon_steps=1
            )

            # RLRP-761 P6.1 — the horizon variance, carried to the SAME space as the
            # mean. It lives in the composed forecast exactly like the mean, so it is
            # extracted with the same helper and transported with the SQUARED slope
            # (``denormalize_predicted_logvar``), which inherits the regime-A/B split:
            # exact for the affine types, ``local_linear`` for winsorized/quantile.
            # ``f_logvar is None`` for a deterministic model — that is an ABSENT
            # variance, not a zero one, and must NOT be turned into a number
            # (RLRP-761 ``Q-A`` / ``P1.5``).
            logvar_steps: Optional[torch.Tensor] = None
            if return_uncertainty and f_logvar is not None:
                logvar_steps_norm = (
                    self.model.extract_obs_features_from_multistep_composed_array(
                        f_logvar, is_model_output=True, vectorized=True
                    ).transpose(-2, -1)
                )
                logvar_steps = self.denormalize_predicted_logvar(
                    logvar_steps_norm, mean_norm=obs_steps_norm, horizon_steps=1
                )

            # Single anchor: reduce any leading (ensemble / batch) dims to a deterministic mean,
            # leaving the (history_len, singlestep_obs_len) forecast.
            # PERF (deferred, RLRP-730): per-anchor reduce; batch anchors once profiling justifies.
            while obs_steps.ndim > 2:
                obs_steps = obs_steps.mean(dim=0)

            if logvar_steps is not None:
                # P6.1 — reduce in VARIANCE space, not in log space: the mean of the
                # logs is the log of the GEOMETRIC mean, which understates an
                # ensemble's spread. ``logsumexp - log(n)`` is the log of the
                # ARITHMETIC mean variance, the quantity that pairs with the
                # arithmetic mean taken on ``obs_steps`` above.
                while logvar_steps.ndim > 2:
                    n_members = logvar_steps.shape[0]
                    logvar_steps = torch.logsumexp(logvar_steps, dim=0) - math.log(
                        float(n_members)
                    )

            # RLRP-760 S0: keep ONLY the forecast steps (the trailing ``F`` of the ``H``-step
            # composed output window) so ``obs_steps[h - 1] == ô_{t+h}`` holds for any F <= H.
            # No-op when ``F == H`` (the historical RLRP-728 configuration).
            horizon_len = int(self.model.horizon_len)
            obs_steps = obs_steps[..., -horizon_len:, :]

            obs_steps = self._sanitize_and_monitor_model_output(
                obs_steps, name="forecast_open_loop_per_horizon.obs_steps"
            )
            if logvar_steps is not None:
                logvar_steps = logvar_steps[..., -horizon_len:, :]
                logvar_steps = self._sanitize_and_monitor_model_output(
                    logvar_steps, name="forecast_open_loop_per_horizon.logvar_steps"
                )

        del model_input, e_obs, e_act, f_mean, obs_steps_norm
        if not return_uncertainty:
            return obs_steps
        # P6.2/P6.3 — the statistics are returned as an INTERNAL DIAGNOSTIC
        # (horizon uncertainty / calibration), never as an MTM-Pro rollout score:
        # the ``AbstractMS2MSForecast`` assert above keeps this surface off the
        # MTM-Pro family, whose only legitimate rollout head is the deploy head.
        return obs_steps, PredictionStatistics(
            mean=obs_steps,
            logvar=logvar_steps,
            space=SPACE_PHYSICAL,
            variance_approximation=(
                VARIANCE_EXACT
                if self.output_variance_transport_is_exact()
                else VARIANCE_LOCAL_LINEAR
            ),
            variance_absent=logvar_steps is None,
            normalizer_type=getattr(self, "normalizer_type", None),
        )

    def sample(
        self,
        act: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
    ]:
        if not hasattr(self.model, "sample_1d"):
            raise RuntimeError(
                "OneDTransitionRewardModel requires wrapped model to define method sample_1d"
            )

        e_obs, e_act = self._single_step_sample_2_model_ensemble(
            model_state["obs"], act
        )
        # RLRP-684 WS-C: deploy inference on possibly-fed-back model predictions ->
        # normalize tolerantly (see ``sample_1d_direct_model_call``).
        model_in = self._get_model_input(e_obs, e_act, normalize_strict_finite=False)

        if self.target_is_delta:
            if isinstance(model_state["obs"], np.ndarray):
                _warn_numpy_input("model_state['obs']", "sample")
            obs = model_util.to_tensor(model_state["obs"]).to(
                self.device, non_blocking=True
            )

        # Note: Assume sample_1d output single-step prediction with reduced ensemble dim
        # i.e., shape=(oDim, )
        preds, next_model_state = self.model.sample_1d(
            model_in, model_state, rng=rng, deterministic=deterministic
        )
        next_obs = preds[:, :-1] if self.learned_rewards else preds
        # RLRP-761 P1.10 — the NORMALIZED baseline used to re-base the statistics
        # channel exactly like ``next_obs`` (``None`` unless ``target_is_delta``).
        stats_baseline_norm: Optional[torch.Tensor] = None

        if self._uses_block_facade:
            # Robust normalizer path: model output is in normalized obs space
            if self.target_is_delta:
                if hasattr(self.model, "history_len"):
                    # obs_at_t = self._fetch_latest_obs_from_ms_obs_act_history(obs, act)
                    # norm_obs_at_t = self._normalize_composed_obs(obs_at_t)
                    # RLRP-761 P1.16 — DEFERRED. When this MS+delta branch is
                    # implemented, the prediction-STATISTICS channel
                    # (``model_state["ensemble_means"/"ensemble_logvars"]``) MUST be
                    # re-baselined and denormalized alongside ``next_obs``, by
                    # passing the normalized baseline to
                    # ``_denormalize_prediction_statistics``. Omitting it leaves the
                    # statistics a physical *delta* instead of a physical next-obs —
                    # the exact class of defect P1 exists to remove.
                    raise NotImplementedError(
                        "ToDo: implement support for MS prediction residual"
                    )
                else:
                    # RLRP-684 WS-C: fed-back deploy prediction -> tolerant normalize.
                    norm_obs_at_t = self._normalize_composed_obs(
                        obs, strict_finite=False
                    )
                # RLRP-761 P1.14 — the statistics channel must be re-baselined
                # with the SAME baseline, otherwise denormalizing it alone yields a
                # physical *delta* rather than a physical next-obs.
                stats_baseline_norm = norm_obs_at_t
                next_obs_norm = next_obs + norm_obs_at_t
                for dim in self.no_delta_list:
                    next_obs_norm[..., dim] = next_obs[..., dim]
                # RLRP-684 WS-A/A3: single-step delta deploy output -> layout-aware
                # denorm (auto ``k = W // Do`` = 1). The MS+delta residual branch
                # above still raises NotImplementedError; when it is implemented it
                # must likewise route through ``denormalize_predicted_obs`` (with the
                # obs/horizon layout), NOT the input-layout ``_denormalize_composed_obs``.
                next_obs = self.denormalize_predicted_obs(next_obs_norm)
            else:
                # RLRP-684 WS-A: non-delta deploy output (obs-only single-step,
                # ``W = Do``) -> layout-aware denorm (auto ``k = W // Do``).
                next_obs = self.denormalize_predicted_obs(next_obs)
        else:
            if self.target_is_delta:
                if hasattr(self.model, "history_len"):
                    obs_at_t = self._fetch_latest_obs_from_ms_obs_act_history(obs, act)
                    tmp_ = next_obs + obs_at_t
                else:
                    tmp_ = next_obs + obs
                for dim in self.no_delta_list:
                    tmp_[..., dim] = next_obs[..., dim]
                next_obs = tmp_

        rewards = preds[:, -1:] if self.learned_rewards else None
        # RLRP-684 WS-C (C2): sanitize + COUNT the predicted ``next_obs`` (a model
        # OUTPUT re-fed auto-regressively as the next rollout INPUT). Transient
        # early-training non-finite predictions are tolerated here rather than
        # crashing the strict ``ZScoreNormalizer.normalize`` input guard on the
        # following step; genuine input-data corruption still fails fast.
        next_obs = self._sanitize_and_monitor_model_output(
            next_obs, name="sample.next_obs"
        )
        # RLRP-761 P1.11 — see ``sample_1d_direct_model_call``.
        self._denormalize_prediction_statistics(
            next_model_state,
            next_obs,
            add_obs_baseline_norm=stats_baseline_norm,
            site="sample",
        )
        next_model_state["obs"] = next_obs
        pred_terminals = None
        return next_obs, rewards, pred_terminals, next_model_state

    def _fetch_latest_obs_from_ms_obs_act_history(
        self, ms_obs: Tensor, ms_act: Tensor
    ) -> Tensor:
        """
        Transforms the observation and action history into the observation at time step t i.e., the latest observation from the history.

        This function processes a multi-step observation and action history tensor, reconstructs the
        history for the specified sequence length, and extracts the observation at the latest time
        step within the given history.

        :param ms_obs: Multi-step observation tensor.
        :param ms_act: Multi-step action tensor.
        :return: The observation tensor extracted at the latest time step.
        :rtype: Tensor
        """
        ms_history = timestep_first_multistep_dim_unflaten_array(
            self._get_model_input(
                obs=ms_obs, action=ms_act, normalize=False
            ),  # Note: use obs and act as they are not normalized
            singlestep_obs_len=self.model.singlestep_obs_len,
            singlestep_act_len=self.model.singlestep_act_len,
            sequence_len=self.model.history_len,
            enable_last_action_padding=False,
        )
        obs_at_t = ms_history[..., -1, : self.model.singlestep_obs_len]
        return obs_at_t

    def _single_step_sample_2_model_ensemble(
        self, obs: Optional[torch.Tensor], act: Optional[torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Broadcast obs/act to match ensemble size: ``(D,) → (E, D)``.

        Works for both 1-D ``(D,)`` and 2-D ``(1, D)`` inputs.
        """
        ensemble_size = self.model.num_members

        def _broadcast(t: torch.Tensor) -> torch.Tensor:
            if t.ndim == 1:
                return t.unsqueeze(0).expand(ensemble_size, -1)
            if 1 < ensemble_size != t.shape[0]:
                return t.expand(ensemble_size, *t.shape[1:])
            return t

        if obs is not None and (obs.ndim == 1 or (1 < ensemble_size != obs.shape[0])):
            obs = _broadcast(obs)
            if act is None:
                return obs
        if act.ndim == 1 or (1 < ensemble_size != act.shape[0]):
            act = _broadcast(act)
            if obs is None:
                return act
        return obs, act

    def save(
        self, save_dir: Union[str, pathlib.Path], include_normalizers: bool = True
    ) -> None:
        """Persist the wrapped model and normalizer state to *save_dir*.

        :param include_normalizers: RLRP-824 (FR6): ``False`` skips the ``input_normalizer/`` /
            ``output_normalizer/`` sub-dirs (per-epoch checkpoints under
            ``training_common.checkpoint.save_normalizer_every_epoch: false``; the statistics then
            live in the run-root ``normalizers/`` snapshot). Default ``True`` = legacy layout.
        """
        self.model.save(save_dir)
        save_dir_p = pathlib.Path(save_dir)

        if include_normalizers:
            self.save_normalizers(save_dir_p)

        onedt_state_dict = {
            "learned_rewards": self.learned_rewards,
            "target_is_delta": self.target_is_delta,
            "no_delta_list": self.no_delta_list,
            "obs_process_fn": self.obs_process_fn,
            "num_elites": self.num_elites,
            "normalizer_type": self.normalizer_type,
        }

        torch.save(onedt_state_dict, save_dir_p / self._ONEDT_FNAME)
        return None

    def save_normalizers(self, save_dir: Union[str, pathlib.Path]) -> None:
        """Persist ONLY the normalizer statistics (``input_normalizer/`` + ``output_normalizer/``).

        RLRP-824 (FR6): used for the run-root ``normalizers/`` snapshot written right after
        ``update_normalizer``; the layout is the one :meth:`save` nests under ``save_dir``.
        """
        save_dir_p = pathlib.Path(save_dir)
        save_dir_p.mkdir(parents=True, exist_ok=True)
        if self.input_normalizer is not None:
            self.input_normalizer.save(save_dir_p / "input_normalizer")
        if self.output_normalizer is not None:
            self.output_normalizer.save(save_dir_p / "output_normalizer")
        return None

    def load_normalizers(self, load_dir: Union[str, pathlib.Path]) -> None:
        """Restore ONLY the normalizer statistics written by :meth:`save_normalizers` / :meth:`save`
        and re-propagate the handles to the wrapped model (RLRP-824 FR6/FR7)."""
        load_dir_p = pathlib.Path(load_dir)
        self._check_no_legacy_normalizer_layout(load_dir_p)
        if self.input_normalizer is not None:
            self.input_normalizer.load(load_dir_p / "input_normalizer")
        if self.output_normalizer is not None:
            self.output_normalizer.load(load_dir_p / "output_normalizer")
        self._set_model_normalizer_handle()
        return None

    def load(
        self, load_dir: Union[str, pathlib.Path], include_normalizers: bool = True
    ) -> None:
        """Restore the wrapped model and normalizer state from *load_dir*.

        :param include_normalizers: RLRP-824 (FR7): ``False`` restores the weights only and keeps
            the normalizer statistics currently held by the model (resume-from-checkpoint of an
            epoch dir written with ``save_normalizer_every_epoch: false``).

        Reads ``onedt_model.pth`` metadata **before** loading the wrapped model
        so that ``normalizer_type`` (and therefore which normalizer handles are
        registered on the model) is correct when ``model.load_state_dict`` runs.
        This ensures backward compatibility with checkpoints saved before the
        robust-normalizer feature was introduced (they lack ``normalizer_type``
        and are treated as ``"standard"``).
        """
        load_dir_p = pathlib.Path(load_dir)

        # --- 1. Restore wrapper metadata first --------------------------------
        from mbrl.util.common import resolve_load_map_location

        onedt_model_dict = torch.load(
            load_dir_p / self._ONEDT_FNAME,
            weights_only=False,
            map_location=resolve_load_map_location(),
        )
        self.learned_rewards = onedt_model_dict["learned_rewards"]
        self.target_is_delta = onedt_model_dict["target_is_delta"]
        self.no_delta_list = onedt_model_dict["no_delta_list"]
        self.obs_process_fn = onedt_model_dict["obs_process_fn"]
        self.num_elites = onedt_model_dict["num_elites"]

        saved_normalizer_type = onedt_model_dict.get("normalizer_type", "standard")
        if saved_normalizer_type != self.normalizer_type:
            # The checkpoint was saved with a different normalizer type.
            # Reconfigure normalizer handles so the model state_dict matches.
            self.normalizer_type = saved_normalizer_type

        # --- 2. Clear stale normalizer handles on the wrapped model -----------
        #     so model.load_state_dict() does not encounter unexpected keys.
        if isinstance(self.model, ExponentialFamilyMLP):
            self.model.set_one_d_trj_model_input_normalizer(None)
            # RLRP-731: also drop the DH obs-denorm handle; re-propagated in step 5.
            self.model.set_one_d_trj_model_obs_denorm_handle(None)

        # --- 3. Load wrapped model state dict ---------------------------------
        #     Filter legacy normalizer keys from old checkpoints that stored
        #     ms_target_normalizer / ss_target_normalizer on the model.
        model_path = load_dir_p / self.model._MODEL_FNAME
        if model_path.exists() and isinstance(self.model, ExponentialFamilyMLP):
            model_dict = torch.load(
                model_path,
                weights_only=False,
                map_location=resolve_load_map_location(),
            )
            _legacy_prefixes = (
                "_one_d_trj_model_ms_target_normalizer.",
                "_one_d_trj_model_ss_target_normalizer.",
            )
            filtered_sd = {
                k: v
                for k, v in model_dict["state_dict"].items()
                if not k.startswith(_legacy_prefixes)
            }
            model_dict["state_dict"] = filtered_sd
            self.model.load_state_dict(filtered_sd, strict=False)
            self.model.elite_models = model_dict["elite_models"]
        else:
            self.model.load(load_dir)

        # --- 4. Load normalizer state -----------------------------------------
        #     RLRP-684: unified facade layout (input_normalizer/ + output_normalizer/).
        #     Legacy obs_normalizer/ + act_normalizer/ layouts are rejected.
        if include_normalizers:
            self._check_no_legacy_normalizer_layout(load_dir_p)
            if self.input_normalizer is not None:
                self.input_normalizer.load(load_dir_p / "input_normalizer")
            if self.output_normalizer is not None:
                self.output_normalizer.load(load_dir_p / "output_normalizer")

        # --- 5. Re-propagate handles to the wrapped model ---------------------
        self._set_model_normalizer_handle()
        self.eval()
        return None


def assert_is_OneDTransitionRewardModelV2(
    one_d_tr_model: Union[OneDTransitionRewardModelV2, torch_OptimizedModule],
) -> None:
    """
    Validates whether the given object is an instance of OneDTransitionRewardModelV2.
    Support casse where the model was torch compiled.

    :param one_d_tr_model: The object to validate. It should be an instance of
        OneDTransitionRewardModelV2 directly or wrapped within a torch_OptimizedModule.
    :return: None
    """
    if isinstance(one_d_tr_model, torch_OptimizedModule):
        one_d_tr_model_check = one_d_tr_model._modules["_orig_mod"]
    else:
        one_d_tr_model_check = one_d_tr_model

    assert isinstance(
        one_d_tr_model_check, OneDTransitionRewardModelV2
    ), f"Expected OneDTransitionRewardModelV2, got {type(one_d_tr_model)=} instead"
    return None


def _reduce_batch_dim(
    next_obs: Optional[torch.Tensor], model_state: Optional[Dict[str, torch.Tensor]]
) -> Union[
    Union[torch.Tensor, Dict[str, torch.Tensor]],
    Tuple[torch.Tensor, Dict[str, torch.Tensor]],
]:
    """
    Reduces batch dimensions of observation tensors and model state tensors, if applicable.

    :param next_obs: Optional tensor containing the next observations. If provided,
        the mean is computed over its dimensions as specified, based on the tensor's
        number of dimensions.
    :param model_state: Optional dictionary containing model state tensors. If provided,
        keys `ensemble_means` and `ensemble_logvars` are updated to compute their means
        over specified dimensions.
    :return: Depending on provided inputs:
        - If only `next_obs` is given, returns the reduced `next_obs`.
        - If only `model_state` is given, returns the updated `model_state` after processing.
        - If both are provided, returns a tuple containing the reduced `next_obs`
          and the updated `model_state`.
    """
    if next_obs is not None:
        if next_obs.ndim == 3:
            next_obs = next_obs.mean(dim=1)
        elif next_obs.ndim == 2:
            next_obs = next_obs.mean(dim=0)
        if model_state is None:
            return next_obs

    if model_state is not None:
        ensemble_means = model_state["ensemble_means"]
        ensemble_logvars = model_state["ensemble_logvars"]

        if ensemble_means.ndim == 3:
            ensemble_means = ensemble_means.mean(dim=1)
            ensemble_logvars = ensemble_logvars.mean(dim=1)
        elif ensemble_means.ndim == 2:
            ensemble_means = ensemble_means.mean(dim=0)
            ensemble_logvars = ensemble_logvars.mean(dim=0)

        model_state["ensemble_means"] = ensemble_means
        model_state["ensemble_logvars"] = ensemble_logvars

        if next_obs is None:
            return model_state

    return next_obs, model_state
