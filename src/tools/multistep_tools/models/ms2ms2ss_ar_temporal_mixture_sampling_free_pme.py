# coding=utf-8
import math
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import omegaconf
import torch
from torch import Tensor, nn as nn

from tools.multistep_tools.multistep_model_util import (
    revert_timestep_first_multistep_dim_unflaten_array,
    timestep_first_multistep_dim_unflaten_array,
)
from tools.multistep_tools.utils import (
    compute_multistep_model_in_size,
    compute_multistep_model_out_size,
)

from tools.multistep_tools.models import MS2MS2SSArTemporalMixturePME, StateHistory
from tools.multistep_tools.models.precision_bounds import (
    _VARIANCE_FLOOR_F32,
    variance_floor,
)
from tools.feature_handling_tools.orientation_heads import (
    align_quaternion_slots_to_reference,
)


class MS2MS2SSArTemporalMixtureSamplingFreePME(MS2MS2SSArTemporalMixturePME):
    """MS2MS2SS model with temporal mixture PME — sampling-free variant
    (Codename Distributional-MTM-Pro).

    Replaces the parent's AR ``.sample()``-based moment propagation with a
    fully deterministic, sampling-free moment propagation that carries both
    ``(h_mean, h_logvar)`` through the unroll. Implements the V4.B formal
    spec attached to YouTrack issue RLRP-530.

    Architecture choice (Solution A): the per-history-slot moments are
    concatenated along the last dim before the encoder; the first
    ``create_linear_layer`` of the parent's network is therefore widened to
    ``2 * in_size`` (the doubling is internal to the encoder — external shape
    contracts using ``compute_multistep_model_in_size`` keep using the
    un-doubled ``in_size``).

    See ``.junie/active_plans/implement_ms2ms2ss_ar_temporal_mixture_sampling_free_pme_RLRP-530.md``
    for the full design rationale.
    """

    _GLOBAL_DEBUG = False

    # RLRP-786 (``perf_RLRP-786_cuda_graph_train_step_plan_20260919.md``, Key Decision 6): the
    # sampling-free forecast has NO device RNG, so its training step can be recorded in a CUDA
    # graph and replayed ``torch.equal`` to eager. Enables the ``cuda_graph_capture_ready`` mask
    # path of ``state_history_update`` below (the particle variants keep the Python coin).
    _CUDA_GRAPH_CAPTURABLE_VARIANT: bool = True

    # RLRP-736 §4A-B P1.2 / RLRP-738 (item c): the by-construction PROBABILISTIC
    # attitude head is now wired for the Distributional variant. This variant carries
    # BOTH moments and concatenates ``[h_mean ‖ h_logvar]`` before the encoder (Solution
    # A doubled input width) in its ``_default_forward_u`` override. The by-construction
    # attitude input-encode ``q -> R`` is applied to the MEAN-HALF ONLY (``h_mean``); the
    # logvar-half is left in the external layout (a log-variance is representation-
    # agnostic and carries no attitude to canonicalise). The doubled encoder input width
    # is therefore re-derived as ``_trunk_in_size + in_size`` (rep-expanded mean-half
    # ‖ external logvar-half) in ``_build_network`` — which collapses to the legacy
    # ``2 * in_size`` when the rep is OFF (``_trunk_in_size == in_size``), so the path
    # stays bit-exact. The rep-expanded mean head output is decoded back to a unit
    # quaternion per unroll block (shared base ``_apply_unroll_orientation_output_decoding``).
    _supports_probabilistic_by_construction_orientation = True

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        singlestep_obs_len: int,
        singlestep_act_len: int,
        history_len: int,
        horizon_len: int,
        unrol_len: int,
        temporal_weights: Union[float, Tuple[float, ...]] = 1.0,
        obs_feature_weights: Union[float, Tuple[float, ...]] = 1.0,
        act_feature_weights: Union[float, Tuple[float, ...]] = 1.0,
        feature_weight_mode: str = "tempered",
        feature_weight_max_ratio: Optional[float] = None,
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        logvar_bound_grad_clip: Optional[float] = None,
        auto_weighting_noise_model: str = "gaussian",
        auto_weighting_scheme: str = "tempered_likelihood",
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        distribution_name: str = "gaussian",
        mae_loss: bool = True,
        model_use_double_precision: bool = False,
        ms_composite_loss_weight: float = 1.0,
        dropout: float = 0.0,
        ms_temporal_weighting_mode: str = "discounted-sum",
        ms_head_dropout: float = 0.0,
        ms_head_num_layers: int = 1,
        residual_form: str = "identity_v2",
        layer_bloc=None,  # RLRP-768 per-region layer-bloc selector (forwarded)
        enable_auto_loss_weighting: bool = True,
        ms_probabilities_reduction: str = "independent",
        ms_energy_beta: Union[float, str] = "learned",
        ms_energy_axis: str = "step",
        ms_probabilistic_loss_mode: str = "true_mixture",  # Options: true_mixture or moment_matched
        # ---- Grouped loss-term / projection / deploy configs (nested in YAML, RLRP-704) ----------
        single_step_loss: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        pre_mixture_u_loss: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        post_mixture_loss: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        info_projection_loss: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        siw_mp_loss: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        gms_iwae_loss: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        projection: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        deploy_head: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        rollout_consistency_loss: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        temporal_mixture_weights: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        ar_sample_clamp: Optional[float] = None,
        safe_zero_variance_initialization: Optional[
            float
        ] = 1e-3,  # Historically 1e-3 → input logvar ≈ -6.9078
        train_time_domain_randomization: Optional[
            Union[Dict, omegaconf.DictConfig]
        ] = None,
        description: Optional[str] = None,
        detach_state_history_after_n_steps: Optional[int] = None,
        # Second AR stage / compounded-prediction (CP) deploy loss (RLRP-708, Option C2).
        # Explicitly surfaced (mirrors the base ``MS2MS2SSArTemporalMixturePME`` signature)
        # rather than relying on ``**kwargs`` pass-through, so the CP config group is
        # discoverable on this subclass. ``None`` -> base defaults (CP OFF, no behaviour change).
        compounded_prediction_deploy_loss: Optional[
            Union[Dict, omegaconf.DictConfig]
        ] = None,
        # Dedicated forecast-path teacher-forcing scheduler (RLRP-726). Surfaced explicitly
        # (mirrors the base ``MS2MS2SSArTemporalMixturePME`` signature) so the group is
        # discoverable on this subclass, INDEPENDENT of the CP/deploy ``teacher_forcing_*`` knobs
        # nested in ``compounded_prediction_deploy_loss``. ``None`` -> base default (default-OFF).
        forecast_teacher_forcing: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        # Deploy-path training stabilization (RLRP-722). Surfaced explicitly (mirrors the base
        # signature) so the nested group is discoverable on this subclass. ``None`` -> base
        # defaults (all three mechanisms OFF, no behaviour change).
        deploy_path_training: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        # Wider-scope flag (future sequence-enabled replay-buffer path: non-flat multistep obs
        # ``(E, B, S, DimOA)``); top-level on the base, surfaced here for parity.
        receive_sequence_batch: bool = False,
        # RLRP-736 by-construction orientation contract. Explicitly surfaced (mirrors the base
        # ``MS2MS2SSArTemporalMixturePME`` signature) rather than relying on ``**kwargs``
        # pass-through, so the orientation config group is discoverable on this subclass (which
        # overrides the orientation handling via its mean-half encode seam). Defaults reproduce
        # the base defaults exactly -> bit-exact, no behaviour change.
        feature_geometry=None,
        internal_orientation=None,
        orientation_singlestep_slots=None,
        **kwargs,
    ):
        """
        :param detach_state_history_after_n_steps: Optional safety valve to cap
            the autograd-tape depth of the AR unroll.

            When set to ``k > 0``, gradients stop flowing through the
            state-history feedback after ``k`` AR steps: the propagated
            ``StateHistory`` is ``.detach()``-ed inside
            ``state_history_update`` whenever ``horizon_index + 1 >= k``.
            This caps the autograd-tape depth without changing the forward
            numerics.

            Disabled by default (``None``) — full BPTT through the AR loop,
            which matches the V4.B spec. Set to a small int (e.g. 4–8) only
            if memory pressure or long-range gradient pathology is observed
            in training; otherwise leave ``None`` for spec-faithful behavior.
        """
        if detach_state_history_after_n_steps is not None:
            assert isinstance(detach_state_history_after_n_steps, int), (
                f"detach_state_history_after_n_steps must be int or None, "
                f"got {type(detach_state_history_after_n_steps).__name__}"
            )
            assert 1 <= detach_state_history_after_n_steps <= horizon_len, (
                f"detach_state_history_after_n_steps must be in "
                f"[1, horizon_len={horizon_len}], got "
                f"{detach_state_history_after_n_steps}"
            )
        self._detach_state_history_after_n_steps = detach_state_history_after_n_steps

        super().__init__(
            in_size=in_size,
            out_size=out_size,
            device=device,
            singlestep_obs_len=singlestep_obs_len,
            singlestep_act_len=singlestep_act_len,
            history_len=history_len,
            horizon_len=horizon_len,
            unrol_len=unrol_len,
            temporal_weights=temporal_weights,
            obs_feature_weights=obs_feature_weights,
            act_feature_weights=act_feature_weights,
            feature_weight_mode=feature_weight_mode,
            feature_weight_max_ratio=feature_weight_max_ratio,
            num_layers=num_layers,
            ensemble_size=ensemble_size,
            hid_size=hid_size,
            propagation_method=propagation_method,
            learn_logvar_bounds=learn_logvar_bounds,
            logvar_bound_grad_clip=logvar_bound_grad_clip,
            auto_weighting_noise_model=auto_weighting_noise_model,
            auto_weighting_scheme=auto_weighting_scheme,
            activation_fn_cfg=activation_fn_cfg,
            distribution_name=distribution_name,
            mae_loss=mae_loss,
            model_use_double_precision=model_use_double_precision,
            ms_composite_loss_weight=ms_composite_loss_weight,
            dropout=dropout,
            ms_temporal_weighting_mode=ms_temporal_weighting_mode,
            ms_head_dropout=ms_head_dropout,
            ms_head_num_layers=ms_head_num_layers,
            residual_form=residual_form,
            layer_bloc=layer_bloc,
            enable_auto_loss_weighting=enable_auto_loss_weighting,
            ms_probabilities_reduction=ms_probabilities_reduction,
            ms_energy_beta=ms_energy_beta,
            ms_energy_axis=ms_energy_axis,
            ms_probabilistic_loss_mode=ms_probabilistic_loss_mode,
            single_step_loss=single_step_loss,
            pre_mixture_u_loss=pre_mixture_u_loss,
            post_mixture_loss=post_mixture_loss,
            info_projection_loss=info_projection_loss,
            siw_mp_loss=siw_mp_loss,
            gms_iwae_loss=gms_iwae_loss,
            projection=projection,
            deploy_head=deploy_head,
            rollout_consistency_loss=rollout_consistency_loss,
            temporal_mixture_weights=temporal_mixture_weights,
            ar_sample_clamp=ar_sample_clamp,
            train_time_domain_randomization=train_time_domain_randomization,
            compounded_prediction_deploy_loss=compounded_prediction_deploy_loss,
            forecast_teacher_forcing=forecast_teacher_forcing,
            deploy_path_training=deploy_path_training,
            receive_sequence_batch=receive_sequence_batch,
            description=description,
            feature_geometry=feature_geometry,
            internal_orientation=internal_orientation,
            orientation_singlestep_slots=orientation_singlestep_slots,
            **kwargs,
        )

        # Spec V4.B (RLRP-530, image2.png): the un-predicted state-history prefix
        # is treated as observed, i.e. ~zero variance. We use a small numerical
        # floor instead of strict 0 so the network input is well-conditioned. NLL
        # is not computed on the input, so the parent's "logvar=0 (var=1) to avoid
        # NLL explosion" reasoning (used at the *output* remaining-history
        # boundary) does NOT apply here.
        # (CRITICAL) ToDo: validate (ref task RLRP-761)
        if safe_zero_variance_initialization is not None:
            self.safe_zero_variance_floor = safe_zero_variance_initialization
        else:
            self.safe_zero_variance_floor = variance_floor(self.model_dtype)

    # ============================================================================================
    # RLRP-761 T2.3 — the UNIFIED certainty-logvar injection point
    # ============================================================================================

    def certainty_logvar(self) -> float:
        """The single definition of the "this slot is an observation" log-variance.

        RLRP-761 finding ``F-2`` catalogued **four** regimes injecting a
        near-zero-variance marker into the encoder's ``h_logvar`` channel:

        1. the observed-history prefix (:meth:`setup_state_history`),
        2. the teacher-forced splice (:meth:`state_history_update`),
        3. the fast deploy path (:meth:`_default_deploy_head`),
        4. (contrast) the free-running propagated moment, which is a genuine
           forecast and is instead **bridged** by
           ``_ar_bridge_logvar_step_target_to_input``.

        Regimes 1-3 already shared the same *value* but each re-derived it with
        its own inline ``math.log(self.safe_zero_variance_floor)``, so there was
        no single place to state the contract and no guard against one site
        drifting. Operator decision (2026-08-05): **unify** them here.

        **Space contract** — this is a RAW certainty marker, deliberately
        expressed directly in the encoder's INPUT space. It is NOT a model
        prediction, so it must **never** be routed through the
        ``target -> input`` log-variance bridge: doing so would shift a constant
        that has no target-space provenance by ``2*log(gain)``. This is exactly
        why the bridge in :meth:`state_history_update` is applied *before* the
        teacher-forcing overwrite.

        :return: ``log(safe_zero_variance_floor)``.

        .. important:: **Value-preserving by construction.** The configured floor
           is used verbatim; the only guard is against a degenerate
           (non-positive / non-finite) configuration, which would otherwise
           inject ``-inf`` into an encoder input. In particular the currently
           live ``O-4`` anchoring (``1e-15`` -> ``logvar ~ -34.5``) is passed
           through unchanged: re-anchoring that constant is the pending
           ``VT-1`` A/B decision and must NOT be smuggled in as a side effect of
           this unification (which is required to be numerically inert).
        """
        floor = float(self.safe_zero_variance_floor)
        if not math.isfinite(floor) or floor <= 0.0:
            # Degenerate configuration: fall back to the dtype's representable
            # variance floor rather than emitting -inf / NaN into the encoder.
            floor = variance_floor(self.model_dtype)
        return math.log(floor)

    def certainty_logvar_like(self, reference: Tensor) -> Tensor:
        """:meth:`certainty_logvar` materialised with ``reference``'s shape/dtype/device."""
        return torch.full_like(reference, self.certainty_logvar())

    # ============================================================================================
    # Network override (Solution A: doubled input width for [h_mean ‖ h_logvar])
    # ============================================================================================

    def _build_network(
        self,
        num_layers: int,
        in_size: int,
        hid_size: int,
        out_size: int,
        ensemble_size: int,
        activation_fn_cfg: omegaconf.DictConfig,
        deterministic: bool,
        learn_logvar_bounds: bool,
        instanciate_logvar_bound_module: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Override: widen the first linear layer to ``2 * in_size``.

        The doubling is internal to the encoder and accounts for the
        ``[h_mean ‖ h_logvar]`` concatenation done in ``_default_forward_u``.
        Heads, residual stack, and bound layers are inherited unchanged.
        External shape contracts (``compute_multistep_model_in_size``, debug
        asserts in ``state_history_update``) keep using the un-doubled
        ``in_size``.

        RLRP-736 §4A-B P1.2 / RLRP-738 (item c): the incoming ``in_size`` is the
        rep-expanded ``_trunk_in_size`` (the base passes ``_trunk_in_size`` here). The
        by-construction attitude input-encode ``q -> R`` is applied to the MEAN-HALF
        ONLY in ``_default_forward_u``, so the concatenated encoder input is
        ``[rep-expanded h_mean ‖ external h_logvar]`` of width
        ``_trunk_in_size + self.in_size`` — NOT ``2 * _trunk_in_size``. This collapses
        to the legacy ``2 * in_size`` when the rep is OFF (``_trunk_in_size ==
        self.in_size``), keeping the OFF path bit-exact.
        """
        return super()._build_network(
            num_layers=num_layers,
            # ⬅ Solution A: concat moments at input. Mean-half is rep-expanded
            # (``in_size == _trunk_in_size``), logvar-half stays external (``self.in_size``).
            in_size=in_size + self.in_size,
            hid_size=hid_size,
            out_size=out_size,
            ensemble_size=ensemble_size,
            activation_fn_cfg=activation_fn_cfg,
            deterministic=deterministic,
            learn_logvar_bounds=learn_logvar_bounds,
            instanciate_logvar_bound_module=instanciate_logvar_bound_module,
            dropout=dropout,
        )

    def setup_state_history(self, x: Tensor) -> tuple[StateHistory, int]:
        """Initialize a ``StateHistory`` carrying both moments.

        Per spec V4.B (RLRP-530, image2.png pseudocode), the input state-history
        prefix is initialized as ``s̄^H_τ ← (s^H_τ, 0_{H×D} + EPS)`` — i.e.
        ~zero variance. We use a small numerical floor (``safe_zero_variance_floor``)
        rather than strict 0 to keep the network input well-conditioned.

        This deviates from a naive ``zeros_like(x)`` (which would mean
        ``var = 1``) on purpose: the parent's "logvar=0" choice was made for
        the *output* remaining-history boundary that feeds NLL, irrelevant for
        an input feature.
        """
        # RLRP-761 T2.3: regime 1 of the unified certainty marker.
        h_logvar = self.certainty_logvar_like(x)
        state_history = StateHistory(h_mean=x, h_logvar=h_logvar)

        if state_history.h_mean.dim() == 2 and not self.training:
            # eval mode (B x ...)
            batch_size = state_history.h_mean.shape[0]
        elif state_history.h_mean.dim() == 3 and not self.training:
            # eval mode (B x ...) using deploy
            batch_size = state_history.h_mean.shape[0]
        elif state_history.h_mean.dim() == 3 and self.training:
            # train mode (E x B x ...)
            batch_size = state_history.h_mean.shape[1]
        else:
            raise NotImplementedError(
                f"Support for {state_history.h_mean.dim()}D input and torch "
                f"training {self.training} not implemented"
            )

        # RLRP-735: capture the frozen means-only ``s_τ^H`` token + cached ψ for the
        # index_and_history_dependent mixer (flag-gated no-op otherwise). This model builds its
        # own ``StateHistory`` (no ``super().setup_state_history`` call), so the base capture
        # helper is invoked explicitly here.
        self._maybe_capture_forecast_history_token(x, state_history)
        return state_history, batch_size

    def state_history_remaining(self, state_history: StateHistory) -> Tensor:
        """Return the un-predicted tail of the state-history mean.

        Mirrors the parent (mean only): the AR loop's ``torch.dstack`` site in
        ``_default_forward`` already pairs the returned mean tensor with
        ``torch.zeros_like(...)`` for the output logvar slot. Returning only
        the mean therefore preserves the parent's output-boundary contract
        and avoids an invasive override of ``_default_forward``.
        """
        # Reshape (..., O[1:Do]_1 + ... + O[1:Do]_MS + A[1:Da]_1 + ... + A[1:Da]_MS)
        # → (..., MS, O+A)
        state_history_mean: Tensor = timestep_first_multistep_dim_unflaten_array(
            state_history.h_mean,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            sequence_len=self.history_len,
            enable_last_action_padding=False,
        )

        remaining_len = self.history_len - self.horizon_len
        remaining_state_history = state_history_mean[..., -remaining_len:, :]

        return remaining_state_history

    @torch.compiler.disable
    def _temporal_weighted_moments_at_i(
        self,
        forecast_mean_accumulator: Tensor,  # (E, B, F, F, O+A)
        forecast_logvar_accumulator: Tensor,  # (E, B, F, F, O+A)
        horizon_index: int,
    ) -> Tuple[Tensor, Tensor]:
        """Spec V4.B weighted-average moments for state-history feedback.

        Implements::

            μ_i  = Σ_{k=max(i-U+1,0)}^{i} α^{(i)}_k · μ^{(i)}_{k,i}
            σ²_i = Σ_{k=max(i-U+1,0)}^{i} α^{(i)}_k · σ^{(i)2}_{k,i}

        Note: this is a CONVEX COMBINATION of component variances and is
        intentionally distinct from ``MixtureSameFamily.variance`` (used by
        ``_default_forward_mixture_dist`` for the supervised forecast), which
        also adds the between-component variance term
        ``Σ_k α_k · (μ_k − μ̄)²``. Conflating the two would silently inflate
        the fed-back uncertainty by the component-disagreement term — see
        plan §0.b correction 1.

        Output shapes: ``(E, B, O+A)`` for both ``mu_i`` and ``logvar_i``.
        """
        U = self.unroll_len
        i = horizon_index
        lo = max(i - U + 1, 0)  # 0-based; spec uses 1-based indexing
        hi = i + 1

        # (..., k_count, O+A) — diagonal column at horizon i for components k ∈ [lo, hi).
        pred_mean_i = forecast_mean_accumulator[..., lo:hi, i, :]
        pred_logvar_i = forecast_logvar_accumulator[..., lo:hi, i, :]

        if self._mixer_is_index_and_history_dependent:
            # RLRP-735: index + input-history conditioned mixer. Per-batch weights from the
            # ensemble-aware residual head over the FROZEN means-only ``s_τ^H`` token (D2 — the
            # log-variance branch is NOT required), built once in setup_state_history with its
            # head output ψ cached. Output (E, B, k_count) → broadcast to (E, B, k_count, 1)
            # (per-batch path, NOT the index else branch that unsqueezes a missing batch dim).
            assert self._forecast_history_token is not None, (
                "index_and_history_dependent mixer requires the frozen s_τ^H token to be built "
                "by setup_state_history before the mixer call."
            )
            # The combined ``additive_index_input_and_history_dependent`` kind also consumes the
            # per-step tokens cat([mean, logvar]); the other history kinds ignore
            # ``input_tokens=None`` via the base-class no-op kwarg (single branch, all kinds).
            input_tokens = None
            if self._mixer_uses_input_tokens:
                input_tokens = torch.cat([pred_mean_i, pred_logvar_i], dim=-1)
            alpha = self.temporal_mixture_weights(
                self._forecast_history_token,
                i,
                log_prob=False,
                psi=self._forecast_history_psi,
                input_tokens=input_tokens,
            )
            alpha = alpha.unsqueeze(-1)
        elif self._mixer_is_input_and_index_dependent:
            # Index + input mixer (γ + ρ, no history): per-step weights from the full-width
            # component tokens -> (E, B, k_count); broadcast to (E, B, k_count, 1).
            input_tokens = torch.cat([pred_mean_i, pred_logvar_i], dim=-1)
            alpha = self.temporal_mixture_weights(input_tokens, i, log_prob=False)
            alpha = alpha.unsqueeze(-1)
        elif self._mixer_is_input_dependent:
            # Input-dependent mixer: per-step weights from the full-width component tokens
            # cat([mean, logvar]) -> (E, B, k_count, 2(O+A)); output already batched
            # (E, B, k_count) -> broadcast to (E, B, k_count, 1).
            input_tokens = torch.cat([pred_mean_i, pred_logvar_i], dim=-1)
            alpha = self.temporal_mixture_weights(input_tokens, log_prob=False)
            alpha = alpha.unsqueeze(-1)
        else:
            # Index mixer: batch-agnostic weights (E, k_count) → broadcast to
            # (E, 1, k_count, 1) for the (E, B, k_count, O+A) component tensors.
            alpha = self.temporal_mixture_weights(i, log_prob=False)
            alpha = alpha.unsqueeze(1).unsqueeze(-1)

        mu_i = (alpha * pred_mean_i).sum(dim=-2)  # (E, B, O+A)
        var_i = (alpha * pred_logvar_i.exp()).sum(dim=-2)  # weighted-avg variance

        # The weighted-avg variance is bounded above by max_k σ²_k (convex
        # combination), so mostly we need the lower floor. We still apply
        # the upper clamp as a paranoid guard (also matches the parent's
        # _MIX_LOGVAR_SAFE_MAX contract used in _default_forward_mixture_dist).
        logvar_i = torch.log(var_i.clamp(min=variance_floor(self.model_dtype)))
        logvar_i = logvar_i.clamp(max=self._MIX_LOGVAR_SAFE_MAX)

        return mu_i, logvar_i

    @torch.compiler.disable
    def state_history_update(
        self,
        state_history: StateHistory,
        forecast_mean_accumulator: Tensor,
        forecast_logvar_accumulator: Tensor,
        horizon_index: int,
        debug: bool = False,
    ) -> StateHistory:
        """Sampling-free moment propagation of the state-history.

        Drops the parent's ``.sample()`` and ``_AR_SAMPLE_CLAMP`` heuristics
        (the latter only guarded ``.sample()`` outliers, irrelevant here).
        Both moments are kept in *normalized* space, so the parent's
        denormalize→shift→renormalize round-trip through RAW space is
        intentionally skipped.

        **Normalisation contract — AMENDED by RLRP-761** (the RLRP-530 rationale
        below predates the ``S4``/``S12`` decoupled innovation facades and is
        stale on that point): "normalized space" is NOT a single space once
        ``standard_symmetric_innovation`` gives the input and target facades
        different scales. The head output is TARGET-space, the state-history is
        INPUT-space, so BOTH propagated moments must cross the diagonal
        coordinate change:

        - mean:    ``mu_in     = gain * mu_tg``   (``_ar_bridge_step_target_to_input``)
        - logvar:  ``logvar_in = logvar_tg + 2*log(gain)``
          (``_ar_bridge_logvar_step_target_to_input``, RLRP-761 ``F-2``)

        Both are **strict identities** (bit-exact, tensor returned untouched) for
        every shared-facade type, which is why the RLRP-530 conclusion remains
        valid *for the configuration it was written against*.

        Original rationale (RLRP-530, superseded on the point above):
        ``.junie/ai_artifact/reports/report_rlrp530_denorm_norm_skip_rationale.md``
        """
        debug = debug or self._GLOBAL_DEBUG

        # Reshape (..., N_in) → (..., MS, O+A) for both moments.
        h_mean = timestep_first_multistep_dim_unflaten_array(
            state_history.h_mean,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            sequence_len=self.history_len,
            enable_last_action_padding=False,
        )
        h_logvar = timestep_first_multistep_dim_unflaten_array(
            state_history.h_logvar,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            sequence_len=self.history_len,
            enable_last_action_padding=False,
        )

        # Drop the oldest timestep — will be replaced by the AR-predicted slot.
        h_mean = h_mean[..., 1:, :]
        h_logvar = h_logvar[..., 1:, :]

        # Spec V4.B weighted-average moments — NOT MixtureSameFamily.variance.
        next_mean, next_logvar = self._temporal_weighted_moments_at_i(
            forecast_mean_accumulator,
            forecast_logvar_accumulator,
            horizon_index,
        )
        next_mean = next_mean.unsqueeze(-2)  # (..., 1, O+A)
        next_logvar = next_logvar.unsqueeze(-2)

        # RLRP-761 F-2: the propagated SECOND moment must cross the same
        # target->input coordinate change as the mean. The mean bridge is the
        # diagonal multiply ``x_in = gain * x_tg`` (applied further down, after
        # the teacher-forcing block), so the matching log-variance transform is
        # the additive ``logvar_in = logvar_tg + 2*log(gain)``. Omitting it left
        # the self-fed moment pair on two different scales under
        # ``standard_symmetric_innovation`` (self-fed sigma off by the gain).
        #
        # PLACEMENT (deliberate): applied HERE, on the raw weighted-moment head
        # output, i.e. BEFORE the teacher-forcing block below. The TF branch
        # overwrites ``next_logvar`` with the ``safe_zero_variance_floor``
        # constant, which is a raw certainty marker and must NOT be shifted; by
        # bridging first, the free-running-only semantics fall out of the control
        # flow with no boolean flag. It is likewise before the eval-mode
        # ``logsumexp`` reduction, which commutes with an additive constant.
        # Strict identity (bit-exact) for every non-decoupled type (``M5``).
        next_logvar = self._ar_bridge_logvar_step_target_to_input(next_logvar)

        # .... Forecast-path teacher forcing (scheduled sampling, RLRP-726) .......................
        # DEFAULT-OFF: mirrors the parent splice but adapted to the sampling-free moment
        # propagation. On a "teacher" step (train-only, GT target stashed, dedicated scheduler
        # fires) the ground-truth obs+act `self._forecast_tf_target[..., horizon_index, :]`
        # replaces the free-running weighted mean, and the paired logvar is pinned to the same
        # near-zero-variance floor used for the observed-history prefix (`safe_zero_variance_floor`):
        # a spliced ground-truth slot is a certainty, not a forecast. The GT target rides in the
        # SAME (normalized) space as the propagated moments here, so no round-trip is needed. When
        # disabled (`always_off`, or eval, or no GT target) this branch is skipped and the forecast
        # is byte-for-byte identical to the free-running rollout.
        if self.cuda_graph_capture_ready:
            # RLRP-786 FR6 (``perf_RLRP-786_cuda_graph_train_step_plan_20260919.md``): the
            # graph-capturable form of the splice below. The per-step CPU coin is read from the
            # static ``(F,)`` bool device mask filled ONCE per training step by
            # ``refill_forecast_tf_mask`` (eager: in ``_probabilistic_loss``; captured: the graph
            # ``before_replay`` hook), and the branch becomes a ``torch.where`` SELECTION on both
            # moments -- no arithmetic, so the result is ``torch.equal`` to the Python branch (and
            # autograd routes the gradient to the selected operand only, as the branch did). The
            # guard mirrors the Python branch's static conditions (train-only, GT target stashed,
            # scheduler present); under those the mask is all-``False`` when the scheduler is
            # ``always_off`` / decay-complete, which reproduces the skipped branch bit for bit.
            if self._forecast_tf_coins_consumed():
                if (
                    self._forecast_tf_mask is None
                    or self._forecast_tf_mask.shape[0] != self.horizon_len
                ):
                    # Defensive allocation (the eager loss refills before the forward).
                    self._forecast_tf_mask = torch.zeros(
                        self.horizon_len, dtype=torch.bool, device=next_mean.device
                    )
                teacher_step = self._forecast_tf_mask[horizon_index]
                gt_mean = (
                    self._forecast_tf_target[..., horizon_index, :]
                    .to(dtype=next_mean.dtype)
                    .unsqueeze(-2)
                )
                next_mean = torch.where(teacher_step, gt_mean, next_mean)
                next_logvar = torch.where(
                    teacher_step, self.certainty_logvar_like(next_logvar), next_logvar
                )
        elif (
            self.training
            and self._forecast_tf_target is not None
            and self._forecast_teacher_forcing_scheduler is not None
            and self._forecast_teacher_forcing_scheduler.should_use_teacher_forcing()
        ):
            next_mean = (
                self._forecast_tf_target[..., horizon_index, :]
                .to(dtype=next_mean.dtype)
                .unsqueeze(-2)
            )
            # RLRP-761 T2.3: regime 2 of the unified certainty marker. Note this
            # OVERWRITES the value the log-variance bridge produced a few lines
            # above -- deliberately, since a spliced GT slot is an observation,
            # not a forecast, and its marker is already INPUT-space.
            next_logvar = self.certainty_logvar_like(next_logvar)

        # Eval-mode ensemble reduction — only triggered when the deploy path
        # passes a (B, ...) input (state_history is dropped one rank vs.
        # the forecast accumulators which always carry the E dim).
        if not self.training and next_mean.dim() > h_mean.dim():
            # Mean: arithmetic mean over E (parent's quick-hack — keep as-is).
            next_mean = next_mean.mean(dim=0)
            # Logvar: average **variances** (logsumexp(logvar) - log(E)),
            # NOT log-variances (which would be a geometric mean of
            # variances and systematically under-estimate uncertainty).
            next_logvar = torch.logsumexp(next_logvar, dim=0) - math.log(
                self.num_members
            )
            # Re-clamp after reduction for paranoia.
            next_logvar = next_logvar.clamp(max=self._MIX_LOGVAR_SAFE_MAX)

        # RLRP-761 S12.9: the propagated ``next_mean`` (obs+act) lives in TARGET
        # space and is re-injected into the INPUT-space history, so it crosses
        # the diagonal obs+act bridge — exactly as the base model's sampled
        # ``next_obs`` does. This override previously applied NO bridge, so under
        # ``standard_symmetric_innovation`` the self-fed obs (and now act) were
        # spliced at the wrong scale (risk ``R-O``); this closes that gap on the
        # sampling-free reference model. Strict identity for every non-decoupled
        # type (``M5``). Applied BEFORE the attitude continuity alignment, which
        # compares against the input-space ``h_mean`` (the gain is 1 on
        # ``unit_norm`` dims, so a quaternion is untouched either way).
        next_mean = self._ar_bridge_step_target_to_input(next_mean)

        # RLRP-736 Item 1: sign-align the propagated attitude mean to the LAST
        # retained history frame (``⟨q_pred, q_ref⟩ ≥ 0`` + L2-projection to S^3) so
        # the sampling-free moment self-feed stays hemisphere-continuous. Applied
        # strictly last on the covered reps; ``quaternion_legacy`` / no-attitude are
        # a bit-exact no-op. Only the MEAN carries the attitude quaternion (the
        # logvar rides the 4-D obs layout untouched).
        if self._quaternion_ar_continuity_active():
            q_ref = h_mean[..., -1:, :]
            next_mean = align_quaternion_slots_to_reference(
                next_mean, q_ref, self._orientation_singlestep_slots
            )

        next_state_history_mean = torch.concatenate([h_mean, next_mean], dim=-2)
        next_state_history_logvar = torch.concatenate([h_logvar, next_logvar], dim=-2)

        # Reshape back to flat layout for both moments.
        next_state_history_mean = revert_timestep_first_multistep_dim_unflaten_array(
            next_state_history_mean,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            remove_last_action_padding=False,
        )
        next_state_history_logvar = revert_timestep_first_multistep_dim_unflaten_array(
            next_state_history_logvar,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            remove_last_action_padding=False,
        )

        # NOTE (intentional deviation from parent): skip the
        # denormalize→shift→renormalize cycle. Both forecast accumulators and
        # the input state-history live in normalized space already — no
        # round-trip needed. Full rationale (RLRP-530):
        # .junie/ai_artifact/reports/report_rlrp530_denorm_norm_skip_rationale.md

        # Optional safety valve: detach to cap autograd-tape depth.
        if (
            self._detach_state_history_after_n_steps is not None
            and (horizon_index + 1) >= self._detach_state_history_after_n_steps
        ):
            next_state_history_mean = next_state_history_mean.detach()
            next_state_history_logvar = next_state_history_logvar.detach()

        if debug:
            _expected_compose_obs_len = compute_multistep_model_in_size(
                singlestep_obs_len=self.singlestep_obs_len,
                singlestep_act_len=self.singlestep_act_len,
                multistep_len=self.history_len,
            )
            expected_shape = f"(..., {_expected_compose_obs_len})"
            for _name, _t in (
                ("mean", next_state_history_mean),
                ("logvar", next_state_history_logvar),
            ):
                assert _t.dim() >= 2, (
                    f"next_state_history_{_name} dimension should be >= 2, "
                    f"got {_t.dim()} with {_t.shape}"
                )
                assert _t.shape[-1] == _expected_compose_obs_len, (
                    f"next_state_history_{_name} expected shape {expected_shape}, "
                    f"got (..., {_t.shape[-1]})"
                )
                assert not torch.isnan(
                    _t
                ).any(), f"next_state_history_{_name} contains 'nan'"

        return StateHistory(
            h_mean=next_state_history_mean,
            h_logvar=next_state_history_logvar,
        )

    def _default_forward_u(
        self, state_history: StateHistory, only_elite: bool
    ) -> tuple[Tensor, Tensor]:
        """Forward U steps from the (mean, logvar) state-history.

        Solution A: concat ``[h_mean ‖ h_logvar]`` along the feature dim
        before the encoder. The first hidden layer was widened by
        ``_build_network`` to absorb this (``_trunk_in_size + self.in_size``
        under an active orientation rep, ``2 * in_size`` when OFF).
        """
        self._maybe_toggle_layers_use_only_elite(only_elite)

        h_mean = self._maybe_cast_to_model_dtype(state_history.h_mean)
        h_logvar = self._maybe_cast_to_model_dtype(state_history.h_logvar)

        # RLRP-736 §4A-B P1.2 / RLRP-738 (item c): encode the attitude input slot(s) of
        # the MEAN-HALF ONLY to the internal rep BEFORE the trunk (mirrors the base
        # ``_default_forward_u``). The logvar-half is left in the external layout: a
        # log-variance carries no attitude to canonicalise and is representation-
        # agnostic. The first encoder layer was widened to ``_trunk_in_size +
        # self.in_size`` in ``_build_network`` to absorb the rep-expanded mean-half.
        # No-op / bit-exact when the rep is OFF.
        h_mean = self._apply_orientation_input_encoding(h_mean)

        x_in = torch.cat([h_mean, h_logvar], dim=-1)

        u_ms_head = self.hidden_layers(x_in)
        u_ms_mean_and_logvar = self.mean_and_logvar(u_ms_head)

        # Architecture 1 (mirrors parent)
        u_ms_mean_split, u_ms_logvar_split = torch.chunk(
            u_ms_mean_and_logvar, chunks=2, dim=-1
        )
        u_ms_mean_head = self.mean_layer(u_ms_mean_split)
        # RLRP-736 §4A-B P1.2 / RLRP-738 (item c): decode every per-unroll-block attitude
        # slot of the rep-expanded mean head back to a unit quaternion by construction
        # (shared base helper). No-op / bit-exact when the rep is OFF.
        u_ms_mean_head = self._apply_unroll_orientation_output_decoding(u_ms_mean_head)
        u_ms_logvar_head = self.logvar_layer(u_ms_logvar_split)
        u_ms_logvar_head = self._get_logvar_bound_layer()(u_ms_logvar_head)

        self._maybe_toggle_layers_use_only_elite(only_elite)
        return u_ms_mean_head, u_ms_logvar_head

    @torch.compiler.disable
    def _default_deploy_head(
        self, x: Tensor, only_elite: bool = False,
        deploy_head_mode_override: Optional[str] = None,
        **_kwargs: Any
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Local override of the parent's V2 fast deploy path.

        The parent constructs ``StateHistory(h_mean=x)`` (no logvar). For the
        sampling-free model we need a paired ``h_logvar``; we inject a
        near-zero-variance tensor here (matches the dim of ``x``) so the
        doubled-input encoder receives a well-shaped input.

        At deploy time, the state-history input ``x`` is a *realization that
        we observed*: it is a certainty, so the associated variance must be
        ~0. We therefore reuse the same ``safe_zero_variance_floor`` floor as
        ``setup_state_history`` (logvar ≈ -6.9 → var ≈ 1e-3) rather than
        ``zeros_like`` (which would imply var=1, i.e. unit uncertainty on
        an observed quantity — semantically wrong at deploy).

        Spec (RLRP-530 V4.B): the un-predicted history prefix is treated as
        observed with ``σ² ≈ 0+EPS``; deploy is exactly that case (no AR
        feedback has run yet, the full history is observed).
        """
        x = self._maybe_cast_to_model_dtype(x)

        # RLRP-761 T2.3: regime 3 of the unified certainty marker.
        h_logvar = self.certainty_logvar_like(x)
        _deploy_state_history = StateHistory(h_mean=x, h_logvar=h_logvar)
        u_ms_mean, u_ms_logvar = self._default_forward_u(
            _deploy_state_history, only_elite
        )

        # RLRP-735: this fast deploy override builds ``StateHistory`` inline (it does NOT go
        # through ``setup_state_history``), so the frozen ``s_τ^H`` token for the
        # index_and_history_dependent mixer must be captured here too — otherwise the mixer call
        # below asserts on a missing token. Flag-gated -> no-op for the other kinds.
        self._maybe_capture_forecast_history_token(x, _deploy_state_history)

        if self.unroll_len > 1 or u_ms_mean.dim() == 4:
            # (E x B x MS x O+A) or (E x MS x O+A)
            u_ms_mean = timestep_first_multistep_dim_unflaten_array(
                u_ms_mean,
                self.singlestep_obs_len,
                self.singlestep_act_len,
                sequence_len=self.unroll_len,
                enable_last_action_padding=True,
            )
            u_ms_logvar = timestep_first_multistep_dim_unflaten_array(
                u_ms_logvar,
                self.singlestep_obs_len,
                self.singlestep_act_len,
                sequence_len=self.unroll_len,
                enable_last_action_padding=True,
            )
            u_ms_mean = u_ms_mean[..., 0, :]
            u_ms_logvar = u_ms_logvar[..., 0, :]

        ss_mean = u_ms_mean.unsqueeze(-2)
        ss_logvar = u_ms_logvar.unsqueeze(-2)

        # Note: Calling the temporal mixture is required even though calling it with i=0 is a
        # mixture of one in the index mixture casse, because we also support the MLP and
        # Transformer mixture casse who take mean/logvar obs as input.
        ss_dist_mixture = self._temporal_mixture_distribution_at_i(
            ss_mean,
            ss_logvar,
            0,
            scale_are_log_variance=True,
        )

        if self.training and (
            self.multitask_loss_singlestep_term_enable
            # Option C2 (RLRP-708): the compounded-prediction CP term needs the deploy
            # mixture in ``true_mixture`` mode even when the SS term is disabled.
            or (self.ar_enabled and self.ar_train_horizon_unroll)
        ):
            self._latest_deploy_ss_dist_mixture = ss_dist_mixture

        LOCAL_DEBUG = False
        if LOCAL_DEBUG:
            assert torch.equal(ss_dist_mixture.mean, u_ms_mean)
            assert torch.equal(torch.log(ss_dist_mixture.variance), u_ms_logvar)
            # The mixture mean are the same as the mean of the first step of the multi-step model but the logvar are not.
            # Question: does it mather for SS deployment?

        ss_mean = ss_dist_mixture.mean

        # RLRP-736 Item 1 (SS ≡ CP(horizon=1); §2.2A): sign-align the exposed deploy
        # attitude mean to the LAST obs frame of the incoming history ``x`` (applied
        # strictly last). Covers the test-time rollout deployer for this variant with
        # no change to ``OneDTransitionRewardModelV2``; no-op OFF / legacy.
        ss_mean = self._apply_deploy_attitude_continuity(ss_mean, x)

        # RLRP-530 (validated, retained): these clamps act on the MIXTURE variance, NOT on a raw
        # network logvar, so they are distinct from (and not redundant with) the per-component
        # `LogvarBoundLayer` already applied in `_default_forward_u`. The mixture variance =
        # within-component variance + between-component spread `Sum_k alpha_k (mu_k - mu_bar)^2`;
        # that spread term is NOT bounded by the per-component bound and can exceed
        # `exp(logvar_max)`, so `_MIX_LOGVAR_SAFE_MAX` is a real upper guard. The lower `variance_floor`
        # is mandatory numerical safety for the `torch.log(...)` (avoids `log(0) -> -inf/NaN`).
        ss_logvar = torch.log(
            ss_dist_mixture.variance.clamp(min=variance_floor(self.model_dtype))
        )
        ss_logvar = ss_logvar.clamp(max=self._MIX_LOGVAR_SAFE_MAX)
        # ---------------------------------------------------------------- RLRP-680 ---(end)---

        # Extract obs-only dimensions (mirrors parent for eval_score parity).
        ss_mean = ss_mean[..., : self.singlestep_obs_len]
        ss_logvar = ss_logvar[..., : self.singlestep_obs_len]

        # Stage B (RLRP-711+): mirror the parent's gated deploy-head switch
        # (`deploy_head_mode`/`deploy_head_source`) with the safe moment-matched fallback.
        return self._maybe_apply_deploy_projection_head(
            ss_mean, ss_logvar, only_elite=only_elite,
            deploy_head_mode_override=deploy_head_mode_override
        )
