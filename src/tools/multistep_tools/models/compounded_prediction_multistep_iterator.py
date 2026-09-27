# coding=utf-8
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import omegaconf
import torch
from mbrl.types import ModelInput
from numpy import dtype, ndarray
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm

from tools.console_tools.message import (
    consol_msg_universal,
    consol_msg_universal_one_liner,
)
from tools.console_tools.progressbar_tools import init_progressbar
from tools.multistep_tools.buffer_unit import MultistepBuffer
from tools.multistep_tools.models import precision_bounds
from tools.multistep_tools.models.compounded_prediction_multistep_iterator_utils import (
    HorizonUnrollLengthDecaySampler,
    TeacherForcingScheduler,
    TemporalWeightScheduler,
)
from tools.multistep_tools.models.performance_mode_mixin import PerformanceModeMixin
from tools.torch_tools.cuda_graph_train_step import is_recording_cuda_graph
from tools.multistep_tools.multistep_model_util import (
    _case_sampled_with_bootstrap_iterator_false,
    reshape_sequence_dim_into_batch_dim,
    _batch_sequence_timestep_view,
    revert_timestep_first_multistep_dim_unflaten_array,
    timestep_first_multistep_dim_unflaten_array,
)
from tools.math_tools.ndarray_tools.custom_msg import nan_infinity_console_warning
from tools.feature_handling_tools.feature_spec import (
    InternalOrientationRep,
    NormStrategy,
)
from tools.feature_handling_tools.orientation_heads import (
    align_quaternion_slots_to_reference,
    project_slots_to_unit_norm,
)

# from tools.multistep_tools.models.weighted_multistep_mlp import WeightedMultiStepMLP


class CompoundedPredictionMultiStepIterator(PerformanceModeMixin):
    """Base-agnostic mixin providing compounded-prediction autoregressive (AR) iteration.

    This mixin is **not** a standalone model: it provides only the AR iteration
    behaviour (``forward``/``deploy``/``loss``/``eval_score`` overrides, the
    horizon-unroll / teacher-forcing / temporal-weight schedulers and the
    ``ar_memory`` plumbing). It is meant to be **mixed into** a host model class
    that subclasses ``AbstractFeatureWeightedMultiStepMLP`` (e.g.
    ``WeightedMultiStepDualHeadMLP`` for the AR family today, and the MTM-Pro
    family in the future). Every ``super().*`` call therefore resolves through
    the host's MRO to that common ancestor and never assumes
    ``WeightedMultiStepDualHeadMLP`` is present.

    The host is responsible for running the model base ``__init__`` (which builds
    the network) and then calling ``_setup_compounded_prediction_iterator(...)``
    to initialize the AR state and schedulers.
    """

    _debug_mode: bool = (
        False  # Console print tensor shape for fast debugging via run mode.
    )

    ar_memory: Optional[torch.Tensor] = None

    # .... AR recurrent-memory semantics (RLRP-757 review follow-up) .............................
    # Selects how a *stateful* recurrent AR encoder (GRU / LSTM) threads its hidden
    # state through the free-running horizon unroll (training loss U-loop AND the
    # deploy rollout). Stateless children (TCN / MLP-AR) ignore it. Values:
    #   * ``"legacy"``              : carry the recurrent state AND re-feed the WHOLE
    #                                 sliding window every unroll step (historical
    #                                 behaviour, kept bit-exact by default; the
    #                                 ``MS-1`` overlapping frames are re-ingested each
    #                                 step on top of the carried state).
    #   * ``"sliding_window_reset"``: reset the recurrent state each unroll step and
    #                                 re-feed the whole sliding window (pure
    #                                 many-to-one, no carry).
    #   * ``"single_frame_carry"``  : warm the recurrent state on the first step with
    #                                 the full history window, then carry it and feed
    #                                 ONLY the newest frame on subsequent steps
    #                                 (canonical stateful AR; incompatible with
    #                                 ``decoder_receive_ms_hidden_size=True``).
    AR_MEM_MODE_LEGACY: str = "legacy"
    AR_MEM_MODE_SLIDING_RESET: str = "sliding_window_reset"
    AR_MEM_MODE_SINGLE_FRAME_CARRY: str = "single_frame_carry"
    AR_RECURRENT_MEMORY_MODES: Tuple[str, ...] = (
        AR_MEM_MODE_LEGACY,
        AR_MEM_MODE_SLIDING_RESET,
        AR_MEM_MODE_SINGLE_FRAME_CARRY,
    )
    ar_recurrent_memory_mode: str = AR_MEM_MODE_LEGACY

    # R4 opt-in switch: when True, ``ar_memory`` is repurposed as a growing buffer of
    # deploy-time single-step predictions (MTM-Pro semantics). When False (default), the
    # host owns ``ar_memory`` (e.g. the GRU/LSTM AR family stores the encoder hidden state
    # there), so the buffer logic stays disabled and that family is unchanged.
    _ar_memory_records_deploy_predictions: bool = False
    # Option C2 (RLRP-708) opt-out: when True, the host model computes the
    # compounded-prediction objective itself (e.g. MTM-Pro adds a standalone
    # deploy-path ``CP`` term inside its own ``_probabilistic_loss``), so this mixin's
    # ``loss`` must NOT run its single-step U-loop and instead delegate straight to
    # ``super().loss(...)``. Default False preserves the ``MS2SS*`` AR family behaviour
    # (which relies on the mixin's U-loop) byte-for-byte.
    _compounded_prediction_handled_internally: bool = False
    _last_pred_mean: Optional[Tensor] = None
    _last_pred_logvar: Optional[Tensor] = None
    _teacher_forcing_scheduler: Optional[TeacherForcingScheduler] = None

    # A4 (RLRP-783): amortised backing store for the R4 ``ar_memory`` deploy-prediction buffer.
    # ``ar_memory`` is exposed as a ``narrow`` VIEW onto ``_ar_memory_buffer`` (capacity doubling)
    # instead of a per-step ``torch.cat`` (which was O(N^2) in copied bytes over an N-step
    # rollout). See the RLRC MTM-Pro models code optimization `.junie` plan
    # (``perf_RLRP-783_mtm_pro_models_code_optimization_plan_20260827.md``). These stay unused
    # (``None``/``0``) when ``_ar_memory_records_deploy_predictions`` is False (AR/LSTM family).
    _ar_memory_buffer: Optional[Tensor] = None
    _ar_memory_len: int = 0
    _AR_MEMORY_INITIAL_CAPACITY: int = 64

    # RLRP-753 G-12: memoisation flag of the one-shot absolute-target / no_delta
    # guard (``_assert_absolute_targets_are_no_delta``), evaluated lazily on the
    # first AR splice because the feature handler is registered post-construction.
    _absolute_target_no_delta_checked: bool = False

    # RLRP-786 AR-baseline extension (``perf_RLRP-786_ar_tcn_cuda_graph_fp32_baseline_plan_20260919.md``,
    # Key Decision 1): per-model CUDA-graph capture validation of the ``ar_encoder``. ``True`` once
    # the host's parity cell of ``test_ar_ms2ss_cuda_graph_step_parity_rlrp786.py`` passed on a
    # GPU (Orin 2026-09-19: TCN, MLP-AR det / prob, GRU, LSTM -- all ``torch.equal`` to eager).
    # A host sets it to ``False`` to report its encoder as a capture blocker (never raised).
    _CUDA_GRAPH_ENCODER_VALIDATED: bool = True

    def _setup_compounded_prediction_iterator(
        self,
        *,
        horizon_len: int,
        model_use_double_precision: bool = False,
        ar_enabled: bool = True,
        receive_sequence_batch: bool = False,
        ar_train_horizon_unroll: bool = True,
        teacher_forcing_decay_stop: int = 0,  # Options: >decay_start=abs stop step, -1=always_on, 0=always_off
        teacher_forcing_decay_start: int = 0,  # Steps before decay begins (warmup)
        teacher_forcing_method: str = "linear",  # Options: "linear", "exponential", "always_on" "always_off"
        ar_temporal_weights: float = 1.0,
        ar_temporal_weights_start: float = 1.0,
        ar_temporal_weights_warmup: int = 0,
        ar_temporal_weights_ramp_stop: int = 0,
        ar_unrol_len_probablity_decay_start: int = 0,
        ar_unrol_len_probablity_decay_stop: int = 0,
        ar_unrol_len_decay_method: str = "beta",  # Options: beta, exponential, linear
        ar_unrol_len_start_horizon_len: int = 1,  # Short unroll length the curriculum grows from
        ar_recurrent_memory_mode: str = AR_MEM_MODE_LEGACY,  # Options: legacy, sliding_window_reset, single_frame_carry
    ) -> None:
        """Initialize the AR iteration state and schedulers.

        The host class is responsible for running the model base ``__init__``
        (which builds the network and sets ``self.model_dtype``) **before**
        calling this method. This method does **not** call ``super().__init__``;
        it only sets the AR-specific state, builds the horizon-unroll /
        temporal-weight / teacher-forcing schedulers and applies the final dtype
        cast — exactly as the iterator's former ``__init__`` did.
        """

        # .... input/output related ...............................................................

        self.ar_enabled = ar_enabled
        self.receive_sequence_batch = receive_sequence_batch
        self.ar_memory = None
        # A4 (RLRP-783): amortised ``ar_memory`` capacity buffer state (see the class attributes).
        self._ar_memory_buffer = None
        self._ar_memory_len = 0
        self.ar_train_horizon_unroll = ar_train_horizon_unroll

        if ar_recurrent_memory_mode not in self.AR_RECURRENT_MEMORY_MODES:
            raise ValueError(
                f"Unknown ar_recurrent_memory_mode={ar_recurrent_memory_mode!r}. "
                f"Expected one of {self.AR_RECURRENT_MEMORY_MODES}."
            )
        self.ar_recurrent_memory_mode = ar_recurrent_memory_mode
        # Second-AR-stage unroll horizon. For the ``MS2SS*`` AR family this equals the model
        # ``horizon_len``; hosts that decouple the compounded-prediction horizon from the MS
        # forecast horizon (e.g. MTM-Pro's ``compounded_prediction_deploy_loss.horizon_len``) pass a
        # dedicated value here. The shared unroll helpers key off ``ar_horizon_len`` so the
        # CP unroll length / temporal-discount normalisation stay decoupled from MS forecasting.
        self.ar_horizon_len = horizon_len

        self.ar_unrol_len_probablity_decay_stop = (
            ar_unrol_len_probablity_decay_stop
        )
        self.ar_unrol_len_probablity_decay_start = ar_unrol_len_probablity_decay_start
        self.ar_unrol_len_decay_method = ar_unrol_len_decay_method
        self.ar_unrol_len_start_horizon_len = ar_unrol_len_start_horizon_len
        if self.ar_unrol_len_probablity_decay_stop > 0:
            self._unroll_len_sampler = HorizonUnrollLengthDecaySampler(
                decay_stop=self.ar_unrol_len_probablity_decay_stop,
                horizon_len=horizon_len,
                decay_start=ar_unrol_len_probablity_decay_start,
                method=self.ar_unrol_len_decay_method,
                start_horizon_len=ar_unrol_len_start_horizon_len,
            )
        else:
            consol_msg_universal_one_liner("HorizonUnrollLengthDecaySampler disabled")
            self._unroll_len_sampler = None

        if horizon_len > 1:
            # assert 0.0 <= ar_temporal_weights < 1.0
            assert 0.0 <= ar_temporal_weights
            assert 0.0 <= ar_temporal_weights_start

        self.ar_temporal_weights_start = ar_temporal_weights_start
        self.ar_temporal_weights = ar_temporal_weights
        self.ar_temporal_weights_warmup = ar_temporal_weights_warmup
        self.ar_temporal_weights_ramp_stop = ar_temporal_weights_ramp_stop

        if horizon_len > 1 and ar_temporal_weights_ramp_stop > 0:
            self._temporal_weight_scheduler = TemporalWeightScheduler(
                target_weight=ar_temporal_weights,
                start_weight=ar_temporal_weights_start,
                warmup_steps=ar_temporal_weights_warmup,
                ramp_stop=ar_temporal_weights_ramp_stop,
                method="ease_in_out",  # Smooth S-curve ramp
            )
        else:
            consol_msg_universal_one_liner("TemporalWeightScheduler disabled")
            self._temporal_weight_scheduler = None

        # .... Teacher forcing ...............................................................
        self.teacher_forcing_decay_start = teacher_forcing_decay_start
        self.teacher_forcing_decay_stop = teacher_forcing_decay_stop
        self.teacher_forcing_method = teacher_forcing_method
        if teacher_forcing_method != "always_off" or teacher_forcing_decay_stop > 0:
            self._teacher_forcing_scheduler = TeacherForcingScheduler(
                decay_start=teacher_forcing_decay_start,
                decay_stop=teacher_forcing_decay_stop,
                method=teacher_forcing_method,
            )
        else:
            consol_msg_universal_one_liner("TeacherForcingScheduler disabled")
            self._teacher_forcing_scheduler = None

        # .... Final build step ...................................................................
        if model_use_double_precision:
            self.to(dtype=torch.double)
        else:
            self.to(dtype=self.model_dtype)

    @property
    def _DETACH_FORWARD_PRED(self):
        """
        Prioritize training stability (True) vs. sequence-level optimization (False)

        Indicates whether to enable detach scheduling for the forward prediction process. This can
        be applied for scenarios like teacher forcing stages, free-running stages, or during gradual
        transition phases (e.g., warm-up stages).

        :return: Current state of detach scheduling as a boolean value.
        """
        # (NICE TO HAVE) ToDo: implement detach scheduling e.g., detach for teacher forcing stage and detach for free-running stage, detach during warmup and graduallly attach more
        return False  # ⚠️ <--

    def wipe_ar_memory(self) -> None:
        self.ar_memory = None
        # A4 (RLRP-783): drop the amortised capacity buffer so the next rollout starts empty
        # (mirrors the historical ``ar_memory = None`` reset).
        self._ar_memory_buffer = None
        self._ar_memory_len = 0
        self._last_pred_mean = None
        self._last_pred_logvar = None
        if self._debug_mode:
            print(f">>>>> Reset ar_memory (mode {self.training=})")
        return None

    def _append_ar_memory(self, ss_pred_mean: Tensor) -> None:
        """Record a deploy-time single-step prediction into ``ar_memory``.

        R4: ``ar_memory`` is a **growing buffer** of the single-step deploy
        predictions produced over a deployment rollout, stacked along a new
        leading ``deployed-step`` axis. It is detached (no graph retention) and
        wiped at the start of each rollout via ``reset_1d`` / ``wipe_ar_memory``.

        A4 (RLRP-783): growth is AMORTISED (capacity doubling + ``narrow`` view)
        instead of a per-step ``torch.cat``, which was ``O(N^2)`` in copied bytes
        over an ``N``-step rollout. Introduced by action ``A4`` of the RLRC MTM-Pro
        models code optimization `.junie` plan
        (``perf_RLRP-783_mtm_pro_models_code_optimization_plan_20260827.md``).
        ``self.ar_memory`` remains a ``Tensor`` with an IDENTICAL shape/dtype/values
        contract (it is a view onto the capacity buffer), so the sibling AR/LSTM
        ``ar_memory`` semantics and the checkpoint round-trip are unaffected.
        """
        pred = ss_pred_mean.detach().unsqueeze(0)
        if self._ar_memory_buffer is None:
            # Uninitialised capacity: every slot is written before it is ever exposed by the
            # ``narrow`` view below, so ``empty`` is safe and avoids a pointless fill.
            self._ar_memory_buffer = torch.empty(
                (self._AR_MEMORY_INITIAL_CAPACITY, *pred.shape[1:]),
                dtype=pred.dtype,
                device=pred.device,
            )
            self._ar_memory_len = 0
        elif self._ar_memory_len == self._ar_memory_buffer.shape[0]:
            # amortised O(1): double the capacity, so O(log N) reallocations for N steps.
            self._ar_memory_buffer = torch.cat(
                [self._ar_memory_buffer, torch.empty_like(self._ar_memory_buffer)], dim=0
            )
        self._ar_memory_buffer[self._ar_memory_len] = pred.to(self._ar_memory_buffer)
        self._ar_memory_len += 1
        self.ar_memory = self._ar_memory_buffer.narrow(0, 0, self._ar_memory_len)
        return None

    def reset_1d(
        self, obs: torch.Tensor, rng: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        # Note: Clean memory on the beginning of deployment rollout
        self.wipe_ar_memory()
        return super().reset_1d(obs, rng)

    def forward(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # (CRITICAL) ToDo: validate change to forward instead of _default_forward
        # mean, logvar = self._default_forward(
        #     x, rng, propagation_indices, use_propagation
        # )
        mean, logvar = super().forward(
            x, rng, propagation_indices, use_propagation
        )

        if self.ar_enabled and self.ar_train_horizon_unroll:
            # Quick-hack to pass the predictions to the AR compounded prediction logic
            if self._DETACH_FORWARD_PRED:
                self._last_pred_mean = mean.detach()
                if logvar is not None:
                    self._last_pred_logvar = logvar.detach()
            else:
                self._last_pred_mean = mean
                if logvar is not None:
                    self._last_pred_logvar = logvar
        return mean, logvar

    def deploy(
        self, x: torch.Tensor, only_elite: bool = True, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # Forward any host-specific deploy kwargs (e.g. MTM-Pro's
        # ``deploy_head_mode_override``) so this override does not silently drop
        # them when the mixin shadows the host ``deploy``. The AR family's
        # ``_default_deploy_head`` ignores unknown kwargs (``**_kwargs``), so this
        # is behaviour-preserving for it.
        ss_mean, ss_logvar = self._default_deploy_head(x, only_elite, **_kwargs)

        # R4: at deploy time (eval rollout, not training) grow the ar_memory buffer
        # with each single-step prediction. Opt-in (MTM-Pro only) and gated off during
        # training so the OFF-path / AR-family behaviour stays byte-for-byte unchanged.
        if self._ar_memory_records_deploy_predictions and self.ar_enabled and not self.training:
            self._append_ar_memory(ss_mean)

        if self.ar_enabled and self.ar_train_horizon_unroll:
            # Quick-hack to pass the predictions to the AR compounded prediction logic.
            # ``deploy`` is also an eval/inference entry point: when NOT training the
            # stashed prediction must be detached so an eval-mode graph (e.g. a cuDNN
            # RNN forward, which does not retain the reserve space needed for backward)
            # cannot bleed into a *subsequent* training ``loss().backward()`` via the
            # free-running splice (``compouned_prediction_model_in_update`` reuses
            # ``self._last_pred_mean``). On GPU this otherwise surfaces as
            # "cudnn RNN backward can only be called in training mode".
            if self._DETACH_FORWARD_PRED or not self.training:
                self._last_pred_mean = ss_mean.detach()
                if ss_logvar is not None:
                    self._last_pred_logvar = ss_logvar.detach()
            else:
                self._last_pred_mean = ss_mean
                if ss_logvar is not None:
                    self._last_pred_logvar = ss_logvar
        return ss_mean, ss_logvar

    # ==== Second-AR-stage unroll building blocks ===============================================
    # The following helpers factor out the per-loss-call scaffolding shared by the mixin
    # ``loss`` U-loop AND host-internal compounded-prediction objectives (e.g. MTM-Pro's CP
    # deploy term in ``MS2MS2SSArTemporalMixturePME._compounded_deploy_nll``). Keeping this
    # logic in ONE place avoids duplicating the unroll-length policy, the temporal-discount
    # weighting, the horizon normalisation and the scheduler stepping across subclasses.

    def _sample_ar_horizon_unroll_len(self) -> int:
        """Sample the second-AR-stage unroll length for one loss call.

        Honours the (optional) unroll-length sampler / probability decay schedule and steps
        the sampler so the distribution advances for the next iteration.
        """
        if self._unroll_len_sampler is not None:
            horizon_unroll_len = self._unroll_len_sampler.sample_unroll_len()
            self._unroll_len_sampler.step()  # Update distribution for next iteration
            return horizon_unroll_len
        if self.ar_horizon_len > 1 and self.ar_unrol_len_probablity_decay_stop == 0:
            return self.ar_horizon_len
        return 1

    def _current_ar_temporal_gamma(self) -> float:
        """Current temporal-discount ``gamma`` (scheduler-driven when configured)."""
        if self._temporal_weight_scheduler is not None:
            return self._temporal_weight_scheduler.current_weight
        return self.ar_temporal_weights

    def _accumulate_ar_temporal_weighted(
        self,
        accumulated: Optional[Tensor],
        step_value: Tensor,
        each_u_t: int,
        gamma: float,
    ) -> Tensor:
        """Add unroll-step ``each_u_t``'s ``step_value`` with the ``gamma**k`` discount.

        ``accumulated`` may be ``None`` (per-feature accumulators that start empty) or a
        running tensor (the scalar mixin loss). When ``gamma == 1.0`` (or ``horizon_len == 1``)
        the step value is added undiscounted, preserving the AR family behaviour byte-for-byte.
        """
        if gamma != 1.0 and self.ar_horizon_len > 1:
            contribution = step_value * (gamma**each_u_t)
        else:
            contribution = step_value
        if accumulated is None:
            return contribution
        return accumulated + contribution

    def _normalize_ar_horizon(
        self, accumulated: Tensor, gamma: float, horizon_unroll_len: int
    ) -> Tensor:
        """Normalise an accumulated AR loss over the unroll horizon.

        Geometric-series normalisation under a temporal discount, else the plain horizon mean.
        """
        if gamma != 1.0 and self.ar_horizon_len > 1:
            return accumulated * (1.0 - gamma) / (1.0 - gamma**horizon_unroll_len)
        return accumulated / horizon_unroll_len

    def _step_ar_schedulers(self) -> None:
        """Advance the teacher-forcing + temporal-weight schedulers once per loss call."""
        if self._teacher_forcing_scheduler is not None:
            self._teacher_forcing_scheduler.step()
        # Step the scheduler once per loss call:
        if self._temporal_weight_scheduler is not None:
            self._temporal_weight_scheduler.step()

    # ==== RLRP-786: CUDA-graph capture seam of the AR MS2SS family ==============================
    def advance_host_step_state(self) -> None:
        """Replay the HOST-side per-step bookkeeping of one :meth:`loss` call (RLRP-786).

        On the CUDA-graph captured path the Python body of :meth:`loss` does not run, so the
        CPU-side state it advances once per training step must be advanced by the graph
        ``before_replay`` hook instead. For the AR MS2SS family that is exactly
        :meth:`_step_ar_schedulers` (teacher-forcing + temporal-weight scheduler ``step``),
        which :meth:`loss` skips while the stream is being recorded
        (``is_recording_cuda_graph``) so the capture batch is counted exactly once; the
        unroll-length sampler is stepped inside :meth:`_sample_ar_horizon_unroll_len` but an
        ACTIVE sampler is reported as a capture blocker, so it is never live here.
        ``MS2MS2SSArTemporalMixturePME`` shadows this method with its own (forecast
        counter + TF mask + CP schedulers).
        """
        self._step_ar_schedulers()

    def cuda_graph_capture_blockers(self) -> List[str]:
        """Reasons why recording this AR model's training step in a CUDA graph would be UNSOUND.

        Empty list = capturable (``MS2SSProbabilisticTCN`` on the paper config
        ``B_tcn_deterministic`` with ``pipeline.performance_mode: fast``). A graph replays the
        kernels recorded once, so every piece of HOST-side control flow that changes the
        computation from one step to the next must be frozen: the teacher-forcing coin
        (``always_off`` / ``always_on`` only), the unroll-length curriculum, the temporal-weight
        ramp, the legacy ``ZScoreNormalizer`` denorm / renorm cycle of the unroll (host
        ``isfinite`` guard) and the ``.item()`` meta writes. Anything listed here is REPORTED and
        the caller falls back to eager with a one-line notice (RLRP-786 FR5) -- never raised.
        ``MS2MS2SSArTemporalMixturePME`` shadows this method with its own list.
        """
        blockers: List[str] = []
        if not self._CUDA_GRAPH_ENCODER_VALIDATED:
            blockers.append(
                f"{type(self).__name__} encoder (cuDNN RNN) not validated under CUDA-graph capture"
            )
        if self._performance_mode != "fast":
            blockers.append("pipeline.performance_mode is not 'fast' (dev instrumentation syncs)")
        if getattr(self, "_enable_meta_collection", False):
            blockers.append("enable_meta_collection is True (.item() meta writes inside the loss)")
        if not (self.ar_enabled and self.ar_train_horizon_unroll):
            blockers.append("ar_train_horizon_unroll is off (single-step path not validated)")
        if self._unroll_len_sampler is not None:
            blockers.append("AR unroll-length curriculum is active (host-sampled unroll length)")
        if self._temporal_weight_scheduler is not None:
            blockers.append("AR temporal-weight ramp is active (gamma changes per step)")
        _tf = self._teacher_forcing_scheduler
        if _tf is not None and _tf.method not in ("always_off", "always_on"):
            blockers.append(
                "AR teacher forcing is a scheduled Python coin (only always_off / always_on capture)"
            )
        _get_norm = getattr(self, "get_one_d_trj_model_input_normalizer", None)
        if _get_norm is not None and _get_norm() is not None:
            blockers.append(
                "legacy ZScore input_normalizer denorm/renorm cycle inside the unroll (host isfinite guard)"
            )
        return blockers

    def loss(
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if self._compounded_prediction_handled_internally:
            # Option C2 (RLRP-708): the host (MTM-Pro) owns the compounded-prediction
            # objective inside its own forecast loss. Delegate so the host loss flow runs
            # (and the host target is never sliced to an obs-only single step here).
            return super().loss(model_in, target)

        if self.ar_enabled and self.ar_train_horizon_unroll:
            meta = {}
            inner_meta: Dict[str, Any] = {}
            # RLRP-786 (AR-baseline extension): ``torch.zeros`` (device fill kernel) instead of
            # ``torch.tensor(0.0, device=...)`` (pageable host->device memcpy, NOT permitted while a
            # CUDA stream is being captured). Same value / dtype / device -> bit-exact eager loss.
            ar_losses = torch.zeros((), device=model_in.device, dtype=torch.float64)

            target = self.reshape_tensor_to_ms_f_dim(target, is_target=True)
            target = target[..., -self.horizon_len :, :]

            self.wipe_ar_memory()

            horizon_unroll_len = self._sample_ar_horizon_unroll_len()

            # Note: losses are already reduced
            gamma = self._current_ar_temporal_gamma()

            # Register values for tensorboard
            # A7 (RLRP-788): pure diagnostic writes -> gated behind the meta-collection
            # kill-switch (``horizon_unroll_len`` / ``gamma`` are still computed above because
            # the unroll below consumes them). RLRC meta-collection kill-switch `.junie` plan
            # (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
            if self._enable_meta_collection:
                meta["horizon_unroll_len"] = horizon_unroll_len
                meta["teacher_forcing_prob"] = (
                    self._teacher_forcing_scheduler.current_probability if self._teacher_forcing_scheduler is not None else 1.0
                )
                meta["ar_temporal_weights"] = gamma

            # RLRP-751 (task T5): a MS2SS AR model (GRU / LSTM / TCN /
            # MS2SSProbabilisticMLPAR) has ONLY a single-step deploy head; its
            # free-running unroll is driven HERE, calling the shared per-step
            # ``super().loss`` once per unroll step. To collect the *compounded*
            # per-feature geometry loss (attitude drift compounding along the
            # free-running unroll) rather than only the ``t=1`` term, we ACCUMULATE
            # each step's deploy ``(point, target)`` into ``_cp_feature_points``
            # (see ``AbstractMS2SSAutoRegressive._compose_ar_feature_geometry``,
            # which suppresses its inline SS term while collecting) and compose the
            # CP term ONCE below into the distinct ``feature_geom_loss_cp`` term.
            # No-op / bit-neutral unless the term is active. Guarded with
            # ``getattr`` since this iterator may be used by a host that does not
            # mix in the feature-geometry seam.
            cp_collect = bool(
                getattr(self, "_feature_loss_active", None)
            ) and self._feature_loss_active()
            if cp_collect:
                self._cp_feature_points = []
                self._cp_feature_collect_active = True

            try:
                for each_u_t in range(horizon_unroll_len):

                    ss_target_t = target[..., each_u_t, : self.singlestep_obs_len]
                    # R3 (merge-last): keep the host forecast meta from the *final* unroll
                    # step so downstream TensorBoard logging (e.g. MTM-Pro forecast-loss
                    # diagnostics) is not lost when the second AR stage is ON.
                    loss, inner_meta = super().loss(model_in, ss_target_t)

                    # Use gamma in weighting:
                    ar_losses = self._accumulate_ar_temporal_weighted(
                        ar_losses, loss, each_u_t, gamma
                    )

                    if each_u_t < horizon_unroll_len - 1:
                        model_in = self.compouned_prediction_model_in_update(
                            each_u_t, model_in, target
                        )
            finally:
                # Always clear the collect flag so a later (non-AR) ``loss`` call is
                # never left in the collecting state, even on an exception.
                if cp_collect:
                    self._cp_feature_collect_active = False

            # .... Normalize horizon ..............................................................
            ar_losses = self._normalize_ar_horizon(
                ar_losses, gamma, horizon_unroll_len
            )

            # .... Compounded-prediction (CP) per-feature geometry term ...........................
            # RLRP-751 (task T5): compose the CP channel accumulated across the
            # free-running unroll above ONCE, INLINE, as the distinct
            # ``feature_geom_loss_cp`` term (mean geometry penalty over the unrolled
            # deploy steps), added on top of the AR NLL/MAE loss and surfaced as its
            # own scalar metric / TensorBoard card. Empty / no-op unless the term is
            # active.
            if cp_collect:
                cp_sequence = self._cp_feature_points
                self._cp_feature_points = []
                if cp_sequence:
                    ar_losses = self._compose_feature_geometry(
                        ar_losses, meta, cp_sequence=cp_sequence
                    )

            # .... Teardown .......................................................................
            # Verify gradient flow
            assert ar_losses.requires_grad, "Loss should require gradients"

            # RLRP-786: not during a CUDA-graph recording pass -- the recorded body runs once without
            # executing; ``advance_host_step_state`` (``before_replay`` hook) owns this step then.
            if not is_recording_cuda_graph():
                self._step_ar_schedulers()

            # Reset memory again
            self.wipe_ar_memory()

            # .... R3: merge-last meta propagation ................................................
            # Preserve the host's (e.g. MTM-Pro) forecast meta from the last unroll step,
            # then overlay the AR diagnostic keys. The AR keys MUST NOT collide with any
            # host meta key, otherwise a host signal would be silently overridden.
            if inner_meta:
                ar_meta_keys = (
                    "horizon_unroll_len",
                    "teacher_forcing_prob",
                    "ar_temporal_weights",
                )
                collisions = [k for k in ar_meta_keys if k in inner_meta]
                assert not collisions, (
                    f"CompoundedPredictionMultiStepIterator AR meta keys collide with "
                    f"host forecast meta keys {collisions!r}; the host signal would be "
                    f"silently overridden. Rename the AR meta keys or the host keys."
                )
                merged_meta = dict(inner_meta)
                merged_meta.update(meta)
                meta = merged_meta
            return ar_losses, meta
        else:
            return super().loss(model_in, target)

    def eval_score(
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if self.ar_enabled and self.ar_train_horizon_unroll:

            assert (
                model_in.ndim == 2 and target.ndim == 2
            ), f"{model_in.ndim=} != 2 and/or {target.ndim=} != 2"

            target = target.repeat((self.num_members, 1, 1))
            target = self.reshape_tensor_to_ms_f_dim(target, is_target=True)
            target = target[..., -self.horizon_len :, :]

            with torch.inference_mode():
                self.eval()

                self.wipe_ar_memory()
                sequence_losses = []
                for each_u_t in range(self.horizon_len):
                    ss_pred_mean, ss_log_var = self.deploy(model_in)

                    # //// DEV: assess ////////////////////////////////////////////////////////////
                    # ... Option 1...
                    ss_next_obs = ss_pred_mean

                    # ... Option 2...
                    # if self.deterministic:
                    #     ss_next_obs = ss_pred_mean
                    # else:
                    #     ss_next_obs = self._sample_1d_next_obs_per_ensemble_probabilisitic_decision(
                    #         ss_pred_mean, ss_log_var)
                    # /////////////////////////////////////////////////// DEV: assess ///(end)/////

                    ss_target = target[..., each_u_t, : self.singlestep_obs_len]
                    if self.mae_loss:
                        ss_losses = F.l1_loss(ss_next_obs, ss_target, reduction="none")
                    else:
                        ss_losses = F.mse_loss(ss_next_obs, ss_target, reduction="none")

                    sequence_losses.append(ss_losses)

                    if each_u_t < self.horizon_len - 1:
                        # (CRITICAL) ToDo: implement target-is-delta logic for internal use in AR i.e., not related to deployment. Otherwise the target_is_delta flag in OneDTransitionRewardModelV2 is unusable <--
                        #   Note (RLRP-753 G-12): still open, but the ABSOLUTE-target
                        #   case (attitude S^3 / gravity S^2) is now FENCED by
                        #   ``_assert_absolute_targets_are_no_delta`` — it raises instead
                        #   of silently compounding an off-manifold delta target.
                        model_in = self.compouned_prediction_model_in_update(
                            each_u_t, model_in, target, next_obs=ss_next_obs
                        )

                ar_loss = torch.dstack(sequence_losses).reshape(
                    *sequence_losses[0].size(), -1
                )
                # Note: eval score should return a non-reduced loss so only average
                # over AR sequence length
                ar_loss = ar_loss.mean(dim=-1)

                # Reset memory againloss
                self.wipe_ar_memory()
                meta = {}
                return ar_loss, meta
        else:
            return super().eval_score(model_in, target)

    def _quaternion_ar_continuity_active(self) -> bool:
        """Whether the history-relative AR-splice sign-continuity pass fires (RLRP-736 Item 1).

        Ablation-lever gate: the attitude obs block must be present, the
        ``ms_model.internal_orientation.enforce_continuity`` config flag must be ON
        (RLRP-744 item E), AND the representation must NOT be ``quaternion_legacy``
        — i.e. enabled for ``quaternion`` / ``sixd`` / ``nine_d_svd``, disabled for
        ``quaternion_legacy`` (which stays the byte-for-byte pre-RLRP-736 A/B
        baseline). NOT gated on ``_internal_orientation_rep_active`` (that would wrongly exclude
        the sign-SENSITIVE plain ``quaternion`` rep, the case the AR seam needs
        most). Bit-exact no-op when there is no attitude slot (math env,
        angular-vel-only obs, attitude disabled).

        RLRP-744 item E — the SAME ``enforce_continuity`` config key that gates the
        ingestion continuity pass (``quaternion_enforce_continuity`` via
        ``FeatureGroupSpec.enforce_continuity``) now ALSO gates every model-side
        reference-relative alignment routed through this predicate (AR splice /
        deploy / sampling-free / resampled / trsf + the DH residual), so a single
        config value governs ingestion AND model-graph continuity. ``getattr(...,
        True)`` keeps a model that never ran ``_setup_orientation_rep`` on the
        historical default. Orthogonal to the in-graph unit decode
        (``_unit_decode``) and the conversion-internal ``w >= 0`` in
        ``quaternion_to_axis_angle`` (both excluded by design).
        """
        slots = getattr(self, "_orientation_singlestep_slots", ())
        if not slots:
            return False
        if not bool(getattr(self, "_orientation_enforce_continuity", True)):
            return False
        rep = getattr(self, "_orientation_rep", None)
        return rep is not InternalOrientationRep.QUATERNIONLEGACY

    def _apply_attitude_ar_continuity(
        self, next_obs_and_act: Tensor, history: Tensor
    ) -> Tensor:
        """Sign-align the re-injected obs attitude slot(s) to the last history frame.

        RLRP-736 Item 1 (root-cause). ``next_obs_and_act`` is the appended
        single-step ``(..., 1, O+A)`` prediction frame; ``history`` is the retained
        ``(..., MS-1, O+A)`` window it will follow. Using the LAST history obs frame
        as the sign reference ``q_ref``, each tracked single-step attitude slot of
        the prediction is flipped into ``q_ref``'s hemisphere
        (``⟨q_pred, q_ref⟩ ≥ 0``) and L2-projected onto ``S^3``
        (:func:`align_quaternion_slots_to_reference`). This is applied STRICTLY LAST
        on the free-running branch so it OVERRIDES any prior memoryless ``w >= 0``
        canonicalisation, so the composed trajectory stays hemisphere-continuous
        even when ``q_ref`` lives in the ``w < 0`` hemisphere. The attitude slot
        base indices are the rep-agnostic ``self._orientation_singlestep_slots`` (they
        index the obs sub-block, which starts at 0 of the ``O+A`` layout).
        """
        # ``q_ref``: last retained history frame, kept as (..., 1, O+A) so it
        # broadcasts against the single appended (..., 1, O+A) frame.
        q_ref = history[..., -1:, :]
        return align_quaternion_slots_to_reference(
            next_obs_and_act, q_ref, self._orientation_singlestep_slots
        )

    def _unit_norm_projection_only_slots(self):
        """Base indices of the ``UNIT_NORM`` obs groups that get PROJECTION ONLY.

        Task ``G-10`` of the RLRP-753 gravity-vector plan
        (``feat_gravity_vector_orientation_representation_plan_RLRP-753_20260724.md``).
        Reads the registered feature handler instead of hard-coding a block:
        every observation group whose ``norm_strategy`` is
        :attr:`NormStrategy.UNIT_NORM` and whose ``enforce_continuity`` is
        ``False`` lives on a manifold with NO double cover (the 3-D
        ``gravity.{x,y,z}`` direction, ruling ``S5.6``), so the AR seam must
        re-project it onto the sphere WITHOUT the hemisphere sign flip. Groups
        with ``enforce_continuity=True`` (the 4-D quaternion) are excluded here:
        they stay on the untouched :meth:`_apply_attitude_ar_continuity` route.

        Only CONTIGUOUS groups are returned (the only layout the resolved
        handler can produce for a declared dim block); a non-contiguous group
        would be mis-projected by a base+width slice, so it is skipped rather
        than silently mangled.

        :return: ``((base_index, block_width), ...)``, empty (hence a bit-exact
            no-op) when no handler is registered or no such group exists — which
            is EVERY currently shipping configuration.
        """
        getter = getattr(self, "get_feature_handler", None)
        handler = None if getter is None else getter()
        if handler is None:
            return ()
        slots = []
        for group in getattr(handler, "obs_groups", ()) or ():
            if group.norm_strategy is not NormStrategy.UNIT_NORM:
                continue
            if group.enforce_continuity:
                continue
            indices = tuple(group.indices)
            if not indices:
                continue
            if tuple(range(indices[0], indices[0] + len(indices))) != indices:
                continue
            slots.append((indices[0], len(indices)))
        return tuple(slots)

    def _apply_unit_norm_ar_projection(
        self, next_obs_and_act: Tensor, history: Tensor
    ) -> Tensor:
        """Re-project every ``UNIT_NORM`` obs block of the AR splice onto its manifold.

        Task ``G-10`` of the RLRP-753 gravity-vector plan
        (``feat_gravity_vector_orientation_representation_plan_RLRP-753_20260724.md``).
        Strategy-driven generalisation of :meth:`_apply_attitude_ar_continuity`:
        the free-running branch must keep EVERY unit-norm observation block on
        its manifold across the whole compounded-prediction horizon, not just the
        quaternion. Two behaviours, keyed off the group's ``enforce_continuity``:

        - ``S^3`` / quaternion (``enforce_continuity=True``): hemisphere flip +
          unit projection, delegated UNCHANGED to
          :meth:`_apply_attitude_ar_continuity` (hence
          :func:`align_quaternion_slots_to_reference`) so the RLRP-736 Item 1
          behaviour stays **bit-exact**.
        - ``S^2`` / gravity (``enforce_continuity=False``, width 3): projection
          ONLY, via :func:`project_slots_to_unit_norm`. ``g_hat^B`` has no double
          cover, so ``g`` and ``-g`` are physically distinct and a sign flip
          would CORRUPT the prediction (ruling ``S5.6``).

        Bit-exact no-op for an observation space with no gravity block (every
        currently shipping configuration): the projection loop then iterates over
        an empty slot tuple and the quaternion route is byte-for-byte the
        pre-RLRP-753 call.

        :param next_obs_and_act: The appended single-step ``(..., 1, O+A)``
            prediction frame.
        :param history: The retained ``(..., MS-1, O+A)`` window it will follow
            (the sign reference source of the ``S^3`` route).
        :return: ``next_obs_and_act`` with every unit-norm block back on its manifold.
        """
        out = next_obs_and_act
        if self._quaternion_ar_continuity_active():
            out = self._apply_attitude_ar_continuity(out, history)
        for base, block_width in self._unit_norm_projection_only_slots():
            out = project_slots_to_unit_norm(out, (base,), slot_width=block_width)
        return out

    def _assert_absolute_targets_are_no_delta(
        self, target_is_delta: Optional[bool] = None
    ) -> None:
        """Fence the ``target_is_delta`` AR gap for ABSOLUTE-target blocks (``G-12``).

        Task ``G-12`` of the RLRP-753 gravity-vector plan
        (``feat_gravity_vector_orientation_representation_plan_RLRP-753_20260724.md``).
        A block flagged ``is_absolute_target`` (the ``S^3`` attitude and the
        ``S^2`` gravity direction) is a POINT ON A MANIFOLD, not a quantity a
        delta can be added to: predicting ``obs_t + delta`` for it leaves the
        manifold immediately and the AR unroll then compounds an off-manifold
        state. ``is_absolute_target`` is honoured today ONLY through
        :meth:`EnvFeatureHandler.no_delta_list`, so this guard verifies that the
        wrapper's ``no_delta_list`` really does cover every such index before an
        AR unroll starts.

        This deliberately FENCES the standing ``target-is-delta`` AR ``ToDo``
        (see :meth:`eval_score`) rather than implementing it: the unsupported
        combination now crashes loudly instead of training on a silently wrong
        target.

        :param target_is_delta: Explicit override; ``None`` resolves from the
            model / registered ``OneDTransitionRewardModelV2`` state and defaults
            to ``False`` (i.e. the guard is inert on the absolute-target path
            every shipping configuration runs).
        :raises ValueError: when ``target_is_delta`` is enabled and at least one
            ``is_absolute_target`` index is missing from ``no_delta_list``. A real
            ``raise``, NEVER an ``assert`` (``python -O`` strips those) — repository
            rule for fail-loud safety guards.
        """
        if target_is_delta is None:
            target_is_delta = bool(
                getattr(
                    self,
                    "target_is_delta",
                    getattr(self, "_one_d_trj_model_target_is_delta", False),
                )
            )
        if not target_is_delta:
            return None
        getter = getattr(self, "get_feature_handler", None)
        handler = None if getter is None else getter()
        if handler is None:
            return None
        no_delta = set(handler.no_delta_list())
        missing = sorted(
            index
            for group in getattr(handler, "obs_groups", ()) or ()
            if group.is_absolute_target
            for index in group.indices
            if index not in no_delta
        )
        if missing:
            raise ValueError(
                f"CompoundedPredictionMultiStepIterator: target_is_delta=True but the "
                f"absolute-target observation indices {missing} are ABSENT from the "
                f"handler no_delta_list {sorted(no_delta)}. An `is_absolute_target` "
                f"block (attitude S^3 / gravity S^2) is a point on a manifold, so a "
                f"delta target would take the compounded-prediction unroll off that "
                f"manifold (RLRP-753 G-12). Set target_is_delta=False or add the "
                f"indices to no_delta_list."
            )
        return None

    def _apply_deploy_attitude_continuity(
        self, ss_mean: Tensor, x: Tensor
    ) -> Tensor:
        """Sign-align an EXPOSED single-step deploy prediction to its own history ``x``.

        RLRP-736 Item 1 (SS ≡ CP(horizon=1); §2.2A). The standalone SS / deploy
        exposure does NOT go through the shared CP splice, yet the test-time rollout
        deployer re-injects exactly this prediction. ``x`` is the incoming flattened
        composed multistep obs+act history the deploy head consumes, so the sign
        reference ``q_ref`` (the attitude slot of its LAST obs frame) is already in
        scope. This flips the exposed attitude slot(s) of ``ss_mean`` into
        ``q_ref``'s hemisphere (``⟨q_pred, q_ref⟩ ≥ 0``) and L2-projects onto
        ``S^3`` — applied STRICTLY LAST so it overrides any prior memoryless
        ``w >= 0`` canonicalisation, chaining hemisphere continuity across the
        external rollout by construction. Bit-exact no-op for ``quaternion_legacy``
        and the no-attitude cases.

        :param ss_mean: The exposed single-step mean, either ``(..., 1, O+A)``
            (MTM-Pro composed layout) or ``(..., O)`` (``ms2ss`` obs block).
        :param x: The flattened composed multistep obs+act history the deploy head
            received (the sign reference source).
        :return: ``ss_mean`` sign-aligned to the last obs frame of ``x``.
        """
        if not self._quaternion_ar_continuity_active():
            return ss_mean
        # Reshape the flattened composed history to (..., MS, O+A) and take the
        # LAST obs frame as the sign reference (keyed off singlestep_obs_len via the
        # shared reshape, so the reference cannot be mis-indexed if the layout /
        # horizon changes — §2.2A.2).
        x_ms = self.reshape_tensor_to_ms_f_dim(x)
        q_ref = x_ms[..., -1, :]  # (..., O+A), leading dims == those of ``x``
        # ``ss_mean`` and ``q_ref`` can carry DIFFERENT leading-dim layouts (e.g. an
        # extra ensemble axis at the front for the ``ms2ss`` deploy vs. an extra
        # single-step axis at ``-2`` for the MTM-Pro deploy), so a naive broadcast
        # against the trailing feature axis is unsafe. Flatten both to (N, F) — the
        # size-1 extra axes fold away and the per-row batch correspondence is
        # preserved — align row-wise, then restore ``ss_mean``'s shape.
        _feat = ss_mean.shape[-1]
        ss_flat = ss_mean.reshape(-1, _feat)
        ref_flat = q_ref.reshape(-1, q_ref.shape[-1])
        if ref_flat.shape[0] != ss_flat.shape[0]:
            if ss_flat.shape[0] % ref_flat.shape[0] == 0:
                ref_flat = ref_flat.repeat(ss_flat.shape[0] // ref_flat.shape[0], 1)
            else:
                # Non-divisible row counts -> cannot map the reference safely; leave
                # the prediction untouched rather than risk a wrong alignment.
                return ss_mean
        aligned = align_quaternion_slots_to_reference(
            ss_flat, ref_flat, self._orientation_singlestep_slots
        )
        return aligned.reshape(ss_mean.shape)

    def _ar_bridge_target_to_input(self, obs_norm: Tensor) -> Tensor:
        """Map a TARGET-space obs tensor into INPUT space (RLRP-761 ``S4.4``).

        Both spaces are centred on the same ``mu``, so the location cancels and
        the conversion is a single diagonal multiply by
        ``s / sigma_state`` (see
        :attr:`OneDTransitionRewardModel.ar_bridge_gain`) -- strictly cheaper
        and exactly invertible compared with the ``standard`` path's
        ``denormalize -> shift -> renormalize`` pair.

        **Strict identity** (the tensor is returned untouched, not merely
        multiplied by a vector of ones) whenever no gain is registered, which is
        every normalizer type but ``standard_symmetric_innovation`` -- that is
        what keeps measure ``M5`` bit-exact.

        :param obs_norm: an obs tensor whose last axis is a whole number of
            ``singlestep_obs_len``-wide single-step blocks.
        """
        return self._ar_bridge_apply(
            obs_norm,
            getter_name="get_one_d_trj_model_ar_bridge_gain",
            single_width=self.singlestep_obs_len,
            cache_attr="_ar_bridge_gain_cache",
            who="_ar_bridge_target_to_input",
        )

    def _ar_bridge_act_target_to_input(self, act_norm: Tensor) -> Tensor:
        """Map a TARGET-space **act** tensor into INPUT space (RLRP-761 ``S12.4``).

        Act analogue of :meth:`_ar_bridge_target_to_input`. Strict identity
        (tensor returned untouched) whenever no act gain is registered, which is
        every normalizer type whose act facades are shared — all legacy types
        and pre-``S12`` innovation runs — keeping measure ``M5`` bit-exact.

        :param act_norm: an act tensor whose last axis is a whole number of
            ``singlestep_act_len``-wide single-step blocks.
        """
        return self._ar_bridge_apply(
            act_norm,
            getter_name="get_one_d_trj_model_ar_bridge_gain_act",
            single_width=self.singlestep_act_len,
            cache_attr="_ar_bridge_gain_act_cache",
            who="_ar_bridge_act_target_to_input",
        )

    def _ar_bridge_step_target_to_input(self, step_norm: Tensor) -> Tensor:
        """Bridge a full composed single step (obs || act) TARGET -> INPUT.

        RLRP-761 ``S12.9a`` — the single source of truth for the target->input
        coordinate change of a re-injected forecast step, applying BOTH the obs
        gain (``S4.4``) and the act gain (``S12.4``). Returns the tensor
        untouched when both gains are absent (every legacy type, ``M5``).

        :param step_norm: a tensor whose last axis is exactly
            ``singlestep_obs_len + singlestep_act_len`` (one composed step).
        """
        obs = self._ar_bridge_target_to_input(step_norm[..., : self.singlestep_obs_len])
        act = self._ar_bridge_act_target_to_input(step_norm[..., self.singlestep_obs_len :])
        return torch.cat([obs, act], dim=-1)

    def _ar_bridge_logvar_target_to_input(self, logvar_norm: Tensor) -> Tensor:
        """Map a TARGET-space obs **log-variance** into INPUT space (RLRP-761 ``F-2``).

        Additive sibling of :meth:`_ar_bridge_target_to_input`. Because the mean
        bridge is the diagonal multiply ``x_in = gain * x_tg``, the matching
        second-moment transform is ``var_in = gain^2 * var_tg``, i.e.

            ``logvar_in = logvar_tg + 2 * log(gain)``.

        Omitting this shift while bridging the mean (the ``F-2`` defect) leaves
        the self-fed moment pair on two different scales.

        **Strict identity** (the tensor is returned untouched, not shifted by a
        vector of zeros) whenever no obs gain is registered -- every normalizer
        type but ``standard_symmetric_innovation`` -- keeping measure ``M5``
        bit-exact.

        :param logvar_norm: a log-variance tensor whose last axis is a whole
            number of ``singlestep_obs_len``-wide single-step blocks.
        """
        expanded = self._ar_bridge_expanded_gain(
            logvar_norm,
            getter_name="get_one_d_trj_model_ar_bridge_gain",
            single_width=self.singlestep_obs_len,
            # Distinct cache namespace: this cache holds ``2*log(gain)``, NOT
            # ``gain``. Sharing ``_ar_bridge_gain_cache`` would let a
            # multiplicative lookup return the additive tensor for the same key.
            cache_attr="_ar_bridge_logvar_shift_cache",
            who="_ar_bridge_logvar_target_to_input",
            transform=self._ar_bridge_logvar_shift_of,
        )
        if expanded is None:
            return logvar_norm
        return logvar_norm + expanded

    def _ar_bridge_logvar_act_target_to_input(self, logvar_norm: Tensor) -> Tensor:
        """Act analogue of :meth:`_ar_bridge_logvar_target_to_input` (RLRP-761 ``F-2``).

        Strict identity whenever no act gain is registered (``M5``).

        :param logvar_norm: a log-variance tensor whose last axis is a whole
            number of ``singlestep_act_len``-wide single-step blocks.
        """
        expanded = self._ar_bridge_expanded_gain(
            logvar_norm,
            getter_name="get_one_d_trj_model_ar_bridge_gain_act",
            single_width=self.singlestep_act_len,
            cache_attr="_ar_bridge_logvar_shift_act_cache",
            who="_ar_bridge_logvar_act_target_to_input",
            transform=self._ar_bridge_logvar_shift_of,
        )
        if expanded is None:
            return logvar_norm
        return logvar_norm + expanded

    def _ar_bridge_logvar_step_target_to_input(self, logvar_norm: Tensor) -> Tensor:
        """Bridge a full composed single-step **log-variance** (obs || act) TARGET -> INPUT.

        Second-moment counterpart of :meth:`_ar_bridge_step_target_to_input`,
        applying BOTH additive shifts. Returns the tensor untouched when both
        gains are absent (every legacy type, ``M5``).

        :param logvar_norm: a tensor whose last axis is exactly
            ``singlestep_obs_len + singlestep_act_len`` (one composed step).
        """
        obs = self._ar_bridge_logvar_target_to_input(
            logvar_norm[..., : self.singlestep_obs_len]
        )
        act = self._ar_bridge_logvar_act_target_to_input(
            logvar_norm[..., self.singlestep_obs_len :]
        )
        return torch.cat([obs, act], dim=-1)

    @staticmethod
    def _ar_bridge_logvar_shift_of(gain: Tensor) -> Tensor:
        """``2 * log(gain)``, guarded against a degenerate (zero) gain.

        The gain is clamped to the dtype-aware ``std_floor`` before ``log`` so a
        zero/negative entry cannot emit ``-inf`` / ``nan`` into the encoder input.
        """
        floor = precision_bounds.std_floor(gain.dtype)
        return 2.0 * torch.log(gain.clamp_min(floor))

    def _ar_bridge_apply(
        self,
        tensor: Tensor,
        *,
        getter_name: str,
        single_width: int,
        cache_attr: str,
        who: str,
    ) -> Tensor:
        """Shared diagonal TARGET->INPUT bridge for obs / act (RLRP-761 ``S12.4``).

        Multiplies ``tensor`` by the registered ``(single_width,)`` gain, tiled
        over the last axis. Strict identity (untouched) when the gain is ``None``
        so every non-decoupled path stays bit-exact. Caches the expanded/moved
        gain keyed by its object identity + device/dtype/width (``S11.4``).
        """
        expanded = self._ar_bridge_expanded_gain(
            tensor,
            getter_name=getter_name,
            single_width=single_width,
            cache_attr=cache_attr,
            who=who,
        )
        if expanded is None:
            return tensor
        return tensor * expanded

    def _ar_bridge_expanded_gain(
        self,
        tensor: Tensor,
        *,
        getter_name: str,
        single_width: int,
        cache_attr: str,
        who: str,
        transform: Optional[Any] = None,
    ) -> Optional[Tensor]:
        """Resolve / tile / cache the bridge factor for ``tensor`` (RLRP-761 ``S12.9b``).

        Single source of truth shared by the multiplicative
        (:meth:`_ar_bridge_apply`) and the additive
        (:meth:`_ar_bridge_logvar_target_to_input`) bridges so the two cannot
        drift. Returns ``None`` when no gain is registered -- callers MUST then
        return their input tensor **untouched** (strict identity, ``M5``).

        :param transform: optional elementwise map applied to the raw gain before
            tiling / caching (e.g. ``2*log(gain)`` for the log-variance bridge).
            Each distinct ``transform`` MUST be given its own ``cache_attr``.
        """
        getter = getattr(self, getter_name, None)
        gain = None if getter is None else getter()
        if gain is None:
            return None
        width = tensor.shape[-1]
        if width % single_width != 0:
            raise ValueError(
                f"{who}: last-axis width {width} is not a whole multiple of "
                f"single-step width={single_width}; the per-feature bridge gain "
                f"would be mis-aligned."
            )
        key = (id(gain), str(tensor.device), tensor.dtype, width)
        cached = getattr(self, cache_attr, None)
        if cached is not None and cached[0] == key:
            return cached[1]
        expanded = gain.to(device=tensor.device, dtype=tensor.dtype)
        if transform is not None:
            expanded = transform(expanded)
        if width != single_width:
            expanded = expanded.repeat(width // single_width)
        setattr(self, cache_attr, (key, expanded))
        return expanded

    @torch.compiler.disable
    def compouned_prediction_model_in_update(
        self,
        each_u_t: int,
        model_in: Tensor,
        target: Tensor,
        next_obs: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Update model input for the next autoregressive step.

        Uses teacher forcing scheduler to decide whether to use ground truth
        or model predictions for the next input.
        """
        # RLRP-753 G-12: fence the open target-is-delta AR ToDo for ABSOLUTE-target
        # blocks. Checked lazily (the feature handler is registered AFTER
        # ``_setup_compounded_prediction_iterator``) and memoised, so the AR hot loop
        # pays it once per model.
        if not getattr(self, "_absolute_target_no_delta_checked", False):
            self._assert_absolute_targets_are_no_delta()
            self._absolute_target_no_delta_checked = True

        # Decide whether to use teacher forcing for this step
        if self._teacher_forcing_scheduler is not None:
            use_teacher_forcing = (
                self.training
                and self._teacher_forcing_scheduler.should_use_teacher_forcing()
            )
        else:
            use_teacher_forcing = False

        if use_teacher_forcing:
            # Teacher forcing: use ground truth
            gt_step = target[..., each_u_t, :]
            # RLRP-761 S4.4 / S12.4: the GROUND TRUTH is expressed in TARGET
            # space too (``_process_batch`` normalizes it with the output
            # facade), so BOTH its obs slice AND its act tail need the bridge.
            # Under ``S12`` the act sub-normalizers are decoupled, so the act
            # gain is non-``None`` and the act tail is re-scaled; for every other
            # type the act gain is ``None`` and the tail is untouched (``M5``).
            gt_step = self._ar_bridge_step_target_to_input(gt_step)
            next_obs_and_act = self.add_ms_dim_and_squeeze_ensemble_dim(gt_step)
            # Update scheduler only during training
        else:
            # Free-running: use model predictions
            act_gt = target[..., each_u_t, -self.singlestep_act_len :]

            if next_obs is None:

                # //// DEV: assess ////////////////////////////////////////////////////////////////
                # ... Option 1...
                next_obs = self._last_pred_mean

                # ... Option 2...
                # if self.deterministic:
                #     next_obs = self._last_pred_mean
                # else:
                #     next_obs = self._sample_1d_next_obs_per_ensemble_probabilisitic_decision(
                #         self._last_pred_mean,
                #         self._last_pred_logvar,
                #         next_state_sampling_size=1 # ToDo: validate nb of sample
                #     )
                #     # next_obs = next_obs.mean(dim=(0, 1)) # ToDo: validate averaging nb samples

                # /////////////////////////////////////////////////////// DEV: assess ///(end)/////

            # Note: with robust normalizers, model_in, model predictions
            # (next_obs) and target (act_gt) are all in normalized space —
            # no denormalization needed before the sliding-window update.
            # The denormalize→shift→renormalize cycle via input_normalizer
            # (below) only applies to the standard normalizer path.

            # RLRP-761 S12.4: ``act_gt`` is the ground-truth act in TARGET space
            # (``target[..., each_u_t, -Da:]``), spliced into the INPUT-space
            # window, so it crosses the act bridge exactly like the obs
            # prediction above. Strict identity for every non-decoupled type.
            next_obs_and_act = torch.concatenate(
                [
                    self.add_ms_dim_and_squeeze_ensemble_dim(
                        self._ar_bridge_target_to_input(next_obs)
                    ),
                    self.add_ms_dim_and_squeeze_ensemble_dim(
                        self._ar_bridge_act_target_to_input(act_gt)
                    ),
                ],
                dim=-1,
            )

        # RLRP-619 resolution: the denormalize->shift->normalize cycle (step 1
        # below + step 2 further down) is STILL REQUIRED for the legacy
        # ``ZScoreNormalizer`` (``input_normalizer``) path because that path
        # is asymmetric: model **input** is normalized via
        # ``input_normalizer.normalize`` (see ``OneDTransitionRewardModelV2.
        # _get_model_input``), but the **target** is left in raw data space
        # (``_process_batch`` keeps ``target_obs = next_obs_t`` when
        # ``target_is_delta=False`` and ``_uses_block_facade=False``).
        # As a consequence, ``next_obs`` (either ``self._last_pred_mean`` from
        # the previous AR step, or the teacher-forcing ground-truth target)
        # is in raw data space, while ``model_in`` arrives in normalized
        # space, so one coordinate change is unavoidable to splice them
        # coherently in the sliding window.
        #
        # For every **block-facade** normalizer regime — ``SoftWinsorizedNormalizer``,
        # ``QuantileNormalizer`` and the RLRP-684 (A1) symmetric Z-score path
        # ``normalizer_type="standard_symmetric"`` — the target IS normalized too
        # (``_process_batch`` calls ``_normalize_composed_obs(next_obs_t)``),
        # so ``model_in``, ``next_obs`` and ``act_gt`` are all in normalized
        # space and the cycle becomes a no-op. The
        # ``if self.get_one_d_trj_model_input_normalizer():`` guard correctly
        # short-circuits the round-trip in that case (``input_normalizer``
        # is ``None`` for every block facade; it is non-``None`` only for the
        # asymmetric ``"standard"`` Z-score path, which genuinely needs the
        # round-trip because its training target stays in RAW space). See also
        # ``.junie/ai_artifact/reports/report_rlrp530_denorm_norm_skip_rationale.md``
        # and the mirror comment in
        # ``MS2MS2SSArTemporalMixturePME.state_history_update``.

        # Denormalize->normalize model input (step 1)
        if self.get_one_d_trj_model_input_normalizer():
            model_in = self.get_one_d_trj_model_input_normalizer().denormalize(model_in)
            # Match next_obs leading dimensions to state_history
            if model_in.dim() > 1 and model_in.shape[0] == self.num_members:
                model_in = model_in.squeeze(dim=0)

        # (..., O[1:Do]_1 + ... + O[1:Do]_MS + A[1:Da]_1 + ... + A[1:Da]_MS ) -->  (..., MS, O+A)
        model_in = self.reshape_tensor_to_ms_f_dim(model_in)
        model_in_next = model_in[..., 1:, :]

        # RLRP-736 Item 1: on the FREE-RUNNING branch, sign-align the re-injected
        # attitude prediction to the last retained history frame so the composed
        # window stays hemisphere-continuous at the AR seam (replaces the memoryless
        # ``w >= 0`` rule; applied strictly last). Teacher forcing already injects
        # the continuity-enforced ground truth, so it is left untouched.
        # RLRP-753 G-10: generalized from the attitude-only pass to a
        # strategy-driven ``NormStrategy.UNIT_NORM`` projection so the 3-D gravity
        # direction also stays on ``S^2`` across the whole AR horizon (projection
        # ONLY there — no sign flip, ``S^2`` has no double cover). Bit-exact for an
        # obs space without a gravity block.
        if not use_teacher_forcing:
            next_obs_and_act = self._apply_unit_norm_ar_projection(
                next_obs_and_act, model_in_next
            )

        model_in = torch.cat((model_in_next, next_obs_and_act), dim=-2)

        # (..., MS, O+A) --> (..., O[1:Do]_1 + ... + O[1:Do]_MS + A[1:Da]_1 + ... + A[1:Da]_MS)
        model_in = self.revert_reshape_tensor_to_ms_f_dim(model_in)

        # Denormalize->normalize model input (step 2)
        if self.get_one_d_trj_model_input_normalizer():
            model_in = self.get_one_d_trj_model_input_normalizer().normalize(model_in)
            # Match next_obs leading dimensions to state_history
            if model_in.dim() > 1 and model_in.shape[0] == self.num_members:
                model_in = model_in.squeeze(dim=0)
        return model_in

    def add_ms_dim_and_squeeze_ensemble_dim(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-2)
        if self.num_members == 1 and x.size(0) == 1:
            x = x.squeeze(0)
        return x

    def reshape_tensor_to_ms_f_dim(self, x: Tensor, is_target: bool = False) -> Tensor:
        """
        Reshapes the input tensor to a multi-step format suitable for forecasting:

            (..., O[1:Do]_1 + ... + O[1:Do]_MS + A[1:Da]_1 + ... + A[1:Da]_MS ) -->  (..., MS, O+A)

        :param x: The input tensor that needs to be reshaped.
        :param is_target: whether x is a model input tensor or a target tensor
        :return: Reshaped input data as a Tensor with altered dimensions for multi-step forecasting.
        :raises NotImplementedError: If AR model-ensemble support is required (more than one member) as
            it is not implemented.
        """
        if self.ar_enabled and not self.receive_sequence_batch:
            if self.num_members > 1:
                raise NotImplementedError(
                    "Support for AR model-ensemble is not implemented yet"
                )

            # Reshape data to
            # (..., O[1:Do]_1 + ... + O[1:Do]_MS + A[1:Da]_1 + ... + A[1:Da]_MS ) --> (..., MS, O+A)
            x = timestep_first_multistep_dim_unflaten_array(
                x,
                singlestep_obs_len=self.singlestep_obs_len,
                singlestep_act_len=self.singlestep_act_len,
                sequence_len=self.history_len,
                enable_last_action_padding=is_target,
            )
            # Dev ALT
            # x = self.unflaten_multistep_composed_array(x, is_model_output=True)
        return x

    def revert_reshape_tensor_to_ms_f_dim(
        self, x: Tensor, is_target: bool = False
    ) -> Tensor:
        """
        Revert reshape_tensor_to_ms_f_dim()

            (..., MS, O+A) --> (..., O[1:Do]_1 + ... + O[1:Do]_MS + A[1:Da]_1 + ... + A[1:Da]_MS)

        :param x: The input tensor that needs to be reshaped.
        :param is_target: whether x is a model input tensor or a target tensor
        :return: Reshaped input data as a Tensor with altered dimensions for multi-step forecasting.
        :raises NotImplementedError: If AR model-ensemble support is required (more than one member) as
            it is not implemented.
        """
        if self.ar_enabled and not self.receive_sequence_batch:
            if self.num_members > 1:
                raise NotImplementedError(
                    "Support for AR model-ensemble is not implemented yet"
                )

            # Reshape data to
            # (..., MS, O+A) --> (..., O[1:Do]_1 + ... + O[1:Do]_MS + A[1:Da]_1 + ... + A[1:Da]_MS)
            x = revert_timestep_first_multistep_dim_unflaten_array(
                x,
                singlestep_obs_len=self.singlestep_obs_len,
                singlestep_act_len=self.singlestep_act_len,
                remove_last_action_padding=is_target,
            )
        return x
