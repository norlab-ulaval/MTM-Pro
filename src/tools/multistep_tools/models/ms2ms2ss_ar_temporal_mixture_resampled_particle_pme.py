# coding=utf-8
"""MS2MS2SS AR temporal mixture (resampled-particle history update).

Implements the "history re-sampling mixture" variant of the auto-regressive
temporal-mixture PME AKA MTM-Pro (YouTrack RLRP-647). It differs from the particle history update version
(:class:`MS2MS2SSArTemporalMixturePME`, the parent) by the *history update rule*:

- Particle-MTM-Pro (sliding-window single-sample):
  ``s̄^H_{τ+i} = s̄^H_{τ+i-1}[1:] ⌢ (ξ_{τ+i})`` with ``ξ_{τ+i} ~ ρ_φ(F̂, i)``
  — drop the oldest slot, append one fresh sample.

- Re-sampled-Particle-MTM-Pro (resampled-particle / anchored at τ):
  ``s̄^H_{τ+i} = s^H_τ[i:] ⌢ (ξ_1, …, ξ_i)`` with ``ξ_r ~ ρ_φ(F̂, r)`` for
  ``r ∈ [1:i]`` — keep the still-pristine *original* observed prefix
  (length ``H − i``), re-sample the **entire** predicted suffix from the LIVE
  forecast mixture at every AR step.

The key implication: every AR step needs access to the frozen original observed
history ``s^H_τ`` (captured at AR-loop entry by :py:meth:`setup_state_history`),
which is propagated through the loop via the
``StateHistory.h_obs_original`` field (added in the shared dataclass for this
purpose).

Architecture inherited from `MS2MS2SSArTemporalMixturePME`: mean-only encoder input
(Architecture 1, no ``h_logvar`` concat — distinguished from Distributional-MTM-Pro
:class:`MS2MS2SSArTemporalMixtureSamplingFreePME` which doubles the input
width).

Reference:
- YouTrack RLRP-647 (sketch attachments image.png … image2.png).
- Sibling: ``ms2ms2ss_ar_temporal_mixture_sampling_free_pme.py`` (Distributional-MTM-Pro).
- Parent:  ``ms2ms2ss_ar_temporal_mixture_pme.py`` (Particle-MTM-Pro).
"""
from typing import Dict, Optional, Sequence, Tuple, Union

import omegaconf
import torch
from torch import Tensor

from tools.multistep_tools.models import (
    MS2MS2SSArTemporalMixturePME,
    StateHistory,
)
from tools.multistep_tools.multistep_model_util import (
    revert_timestep_first_multistep_dim_unflaten_array,
    timestep_first_multistep_dim_unflaten_array,
)
from tools.multistep_tools.utils import compute_multistep_model_in_size
from tools.feature_handling_tools.orientation_heads import (
    align_quaternion_slots_to_reference,
)


class MS2MS2SSArTemporalMixtureResampledParticlePME(MS2MS2SSArTemporalMixturePME):
    """MS2MS2SS model with temporal mixture PME — resampled-particle history update
    (Codename Resampled-Particle-MTM-Pro).

    At each AR step ``i ∈ [1, F]`` (1-based, ``horizon_index = i - 1``):

    1. Keep the **pristine** observed prefix ``s^H_τ[i:]`` (length ``H − i``)
       — sliced from the frozen snapshot stored in
       :py:attr:`StateHistory.h_obs_original`.
    2. Draw ``i`` fresh samples ``ξ_r ~ ρ_φ(F̂, r)`` for ``r ∈ [1, i]`` from
       the **up-to-date** forecast mixture (every step re-samples the whole
       predicted suffix; nothing is reused from the previous step's history).
    3. Concatenate ``s^H_τ[i:] ⌢ (ξ_1, …, ξ_i)`` → new history of length ``H``.

    The only behavioural delta vs. the parent is :py:meth:`state_history_update`;
    :py:meth:`setup_state_history` is overridden only to capture the frozen
    snapshot. All other methods (``_build_network``, ``_default_forward_u``,
    ``_default_deploy_head``, ``state_history_remaining``, the loss path) are
    inherited unchanged from :class:`MS2MS2SSArTemporalMixturePME`.

    Notes
    -----
    - **Denormalize→renormalize round-trip is retained** (see RLRP-619 rationale
      in :py:meth:`MS2MS2SSArTemporalMixturePME.state_history_update`): because
      ``ρ_φ.sample()`` produces draws in raw data space while the kept
      observed prefix lives in normalized space (legacy ``ZScoreNormalizer``
      path). The robust-normalizer path short-circuits the cycle naturally
      (``get_one_d_trj_model_input_normalizer()`` returns ``None``).
    - **Performance**: the inner sampling loop runs ``Σ_{i=1..F} i = F(F+1)/2``
      ``MixtureSameFamily`` builds per forward. Naive implementation is kept
      here for clarity; a vectorized batched-mixture build is deferred to a
      perf-pass (see plan §5 / Q5).
    - **Gradients**: ``MixtureSameFamily.sample()`` is non-reparameterized →
      no autograd flows through the samples themselves into the encoder; the
      learning signal reaches the encoder only through the forecast-head
      supervised path, identical to the parent.
    """

    _GLOBAL_DEBUG = False

    # RLRP-736 §4A-B P1.2 / RLRP-738 (item b): the by-construction PROBABILISTIC
    # attitude head is now wired for the Re-Particle variant. Its
    # ``state_history_update`` re-injects RAW ``mixture.sample()`` draws into the history
    # buffer, whose attitude sub-vector is a generic 4-D vector — NOT a unit quaternion —
    # so each re-sampled slot is projected back onto the unit-quaternion double-cover
    # (L2-normalise only; the memoryless ``w >= 0`` sign step was removed by
    # RLRP-744 action B-4) via the shared base helper
    # ``_canonicalize_reinjected_quaternion`` BEFORE re-injection, so the inherited
    # ``_default_forward_u`` input-encode ``q -> R`` receives a valid rotation. The path
    # stays bit-exact when the rep is OFF (``quaternion``).
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
        compounded_prediction_deploy_loss: Optional[
            Union[Dict, omegaconf.DictConfig]
        ] = None,
        temporal_mixture_weights: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        ar_sample_clamp: Optional[float] = None,
        train_time_domain_randomization: Optional[
            Union[Dict, omegaconf.DictConfig]
        ] = None,
        forecast_teacher_forcing: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        deploy_path_training: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        # Wider-scope flag (future sequence-enabled replay-buffer path: non-flat multistep obs
        # ``(E, B, S, DimOA)``); top-level on the base, surfaced here for parity.
        receive_sequence_batch: bool = False,
        description: Optional[str] = None,
        # RLRP-736 by-construction orientation contract. Explicitly surfaced (mirrors the base
        # ``MS2MS2SSArTemporalMixturePME`` signature) rather than relying on ``**kwargs``
        # pass-through, so the orientation config group is discoverable on this subclass (which
        # overrides the orientation handling via its re-sampled-suffix AR re-injection
        # canonicalisation). Defaults reproduce the base defaults exactly -> bit-exact.
        feature_geometry=None,
        internal_orientation=None,
        orientation_singlestep_slots=None,
        **kwargs,
    ):
        # This subclass uses the base (index-based) temporal mixer, which supports the obs-only
        # mixture path, so the true-mixture / post-mixture / projection objectives are all
        # available here and forwarded to the parent.
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

    def setup_state_history(self, x: Tensor) -> tuple[StateHistory, int]:
        """Capture the frozen snapshot of ``s^H_τ`` for Stage 3 re-sampling.

        The snapshot is stored as a detached + cloned tensor on
        :py:attr:`StateHistory.h_obs_original`, guaranteeing it survives the
        AR loop untouched while autograd does not back-propagate through it
        (the original observed history is *data*, not a learnable signal).
        """
        state_history, batch_size = super().setup_state_history(x)
        # Frozen snapshot of the original observed history s^H_τ (normalized
        # space, same coordinate frame as the model input).
        state_history.h_obs_original = x.detach().clone()
        return state_history, batch_size

    @torch.compiler.disable
    def state_history_update(
        self,
        state_history: StateHistory,
        forecast_mean_accumulator: Tensor,
        forecast_logvar_accumulator: Tensor,
        horizon_index: int,
        debug: bool = False,
    ) -> StateHistory:
        """Stage 3 history update: ``s̄^H_{τ+i} = s^H_τ[i:] ⌢ (ξ_1, …, ξ_i)``.

        Drops the oldest ``i`` slots from the **frozen** original observed
        history (NOT from the previous step's ``h_mean``), then appends ``i``
        fresh samples drawn from the up-to-date forecast mixture
        ``ρ_φ(F̂, r)`` for ``r ∈ [1, i]``. The previous step's predicted
        suffix is *discarded* (this is the defining property of Stage 3 vs.
        Stage 1).

        Denormalize→renormalize round-trip
        ----------------------------------
        Retained from the parent (RLRP-619 rationale): mixture ``.sample()``
        draws live in raw data space while the kept observed prefix arrives
        in normalized space, so one coordinate change is unavoidable when
        the legacy ``ZScoreNormalizer`` path is active.
        """
        debug = debug or self._GLOBAL_DEBUG
        H = self.history_len
        i_one_based = horizon_index + 1  # spec uses 1-based i; we get 0-based.

        assert state_history.h_obs_original is not None, (
            "Stage 3 (MS2MS2SSArTemporalMixtureResampledParticlePME) requires "
            "StateHistory.h_obs_original to be populated by setup_state_history."
        )

        normalizer = self.get_one_d_trj_model_input_normalizer()

        # ── 1. Materialize the kept observed prefix s^H_τ[i:] ────────────────
        # Read the FROZEN snapshot (do NOT mutate h_obs_original).
        s_obs_normalized = state_history.h_obs_original
        if normalizer:
            # Denormalize → raw space (RLRP-619): samples are produced in raw
            # space below, so the prefix must follow.
            s_obs_work = normalizer.denormalize(s_obs_normalized)
        else:
            # Robust-normalizer path: input == sample space, no round-trip.
            s_obs_work = s_obs_normalized

        # (..., O[1:Do]_1 + ... + A[1:Da]_MS) → (..., MS=H, O+A)
        s_obs_unflat = timestep_first_multistep_dim_unflaten_array(
            s_obs_work,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            sequence_len=H,
            enable_last_action_padding=False,
        )

        # Drop the oldest i slots (defensive ``min`` for the unusual case F > H).
        prefix_drop = min(i_one_based, H)
        kept_prefix = s_obs_unflat[..., prefix_drop:, :]  # (..., H - prefix_drop, O+A)

        # ── 2. Re-sample i fresh slots from ρ_φ(F̂, r) for r ∈ [0..i-1] ──────
        #     (0-based r ↔ spec's 1-based r ∈ [1..i]; spec's ρ_φ(F̂, r=1)
        #      corresponds to _temporal_mixture_at_i(..., horizon_index=0).)
        # RLRP-736 Item 1: seed the history-relative sign reference with the last
        # kept-prefix frame (already ingestion-continuity-enforced) so the re-sampled
        # suffix chains hemisphere-continuously onto it. ``None`` when the prefix is
        # empty (the first fresh slot then seeds the chain against itself).
        _ar_continuity = self._quaternion_ar_continuity_active()
        if _ar_continuity and kept_prefix.shape[-2] > 0:
            _prev_attitude_frame = kept_prefix[..., -1, :]
        else:
            _prev_attitude_frame = None

        fresh_samples = []
        for r in range(i_one_based):
            rho_r = self._temporal_mixture_at_i(
                forecast_mean_accumulator,
                forecast_logvar_accumulator,
                r,
                obs_space_only=False,
            )
            xi_r = rho_r.sample()  # raw data space (parent convention)

            # (CRITICAL) ToDo: validate ↓↓ (ref task RLRP-530)
            # RLRP-530 sample guard: clamp extreme draws & substitute mean for
            # any non-finite entries (mixture variance can blow up when the
            # components strongly disagree). RLRP-750: dtype-aware, finfo-derived
            # ceiling via the shared resolver (no longer the tight ``1e2`` bias).
            xi_r = xi_r.clamp(-self.ar_sample_clamp, self.ar_sample_clamp)
            if not torch.isfinite(xi_r).all():
                xi_r = torch.where(torch.isfinite(xi_r), xi_r, rho_r.mean)

            # .... Forecast-path teacher forcing (scheduled sampling, RLRP-726) ...................
            # DEFAULT-OFF: mirrors the parent splice. Stage 3 re-samples the WHOLE suffix each
            # call (r ∈ [0..i-1]), so on a "teacher" step (train-only, GT target stashed, dedicated
            # scheduler fires) each re-sampled slot ξ_r is replaced by the ground-truth obs+act
            # `self._forecast_tf_target[..., r, :]`. The GT target rides in the SAME space as the
            # mixture sample (raw for the asymmetric `standard` normalizer, normalized for the
            # block-facade regimes), so it follows the same renormalization round-trip below. When
            # disabled (`always_off`, or eval, or no GT target) this branch is skipped and the
            # suffix stays free-running (byte-for-byte parity). The per-step coin is drawn at the
            # schedule frozen once per loss in `_probabilistic_loss` (never stepped here).
            if (
                self.training
                and self._forecast_tf_target is not None
                and self._forecast_teacher_forcing_scheduler is not None
                and r < self._forecast_tf_target.shape[-2]
                and self._forecast_teacher_forcing_scheduler.should_use_teacher_forcing()
            ):
                xi_r = self._forecast_tf_target[..., r, :].to(dtype=xi_r.dtype)

            # RLRP-761 S12.9: ``xi_r`` (obs+act) is sampled in TARGET space and
            # re-injected into the INPUT-space history, so it crosses the
            # diagonal obs+act bridge before it is chained onto the kept prefix.
            # This override previously applied NO bridge (risk ``R-O``); this
            # closes the gap on the resampled-particle path. Applied BEFORE the
            # attitude continuity alignment (which references the input-space
            # kept prefix); strict identity for every non-decoupled type (``M5``).
            xi_r = self._ar_bridge_step_target_to_input(xi_r)

            # RLRP-736 Item 1: sign-align each re-sampled slot to the PREVIOUS frame
            # in the assembled sequence (``⟨q_r, q_{r-1}⟩ ≥ 0`` + L2-projection to
            # S^3), chaining hemisphere continuity across the whole re-sampled suffix
            # and onto the kept prefix. This REPLACES the memoryless ``w >= 0``
            # canonicaliser for the covered reps (applied strictly last);
            # ``quaternion_legacy`` / no-attitude fall back to the byte-for-byte guard.
            if _ar_continuity:
                _ref = _prev_attitude_frame if _prev_attitude_frame is not None else xi_r
                xi_r = align_quaternion_slots_to_reference(
                    xi_r, _ref, self._orientation_singlestep_slots
                )
                _prev_attitude_frame = xi_r
            else:
                xi_r = self._canonicalize_reinjected_quaternion(xi_r)

            xi_r = xi_r.unsqueeze(dim=-2)  # (..., 1, O+A)

            # Eval-mode ensemble collapse (mirrors parent's quick-hack).
            if not self.training and xi_r.dim() > kept_prefix.dim():
                xi_r = xi_r.mean(dim=0)

            fresh_samples.append(xi_r)

        predicted_suffix = torch.cat(fresh_samples, dim=-2)  # (..., i_one_based, O+A)

        # ── 3. Concatenate: kept_prefix ⌢ predicted_suffix ────────────────────
        next_unflat = torch.cat([kept_prefix, predicted_suffix], dim=-2)
        # Defensive trim: if (rare) F > H, the spec's length is > H — keep
        # only the most recent H slots to honour the history-length invariant.
        if next_unflat.shape[-2] > H:
            next_unflat = next_unflat[..., -H:, :]

        # (..., MS=H, O+A) → (..., O[1:Do]_1 + ... + A[1:Da]_MS)
        # (CRITICAL) ToDo: validate switching to 'remove_last_action_padding=True' (ref task RLRP-681)
        next_h_mean = revert_timestep_first_multistep_dim_unflaten_array(
            next_unflat,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            remove_last_action_padding=False,
        )

        if normalizer:
            # Renormalize → encoder input space (RLRP-619 step 2).
            next_h_mean = normalizer.normalize(next_h_mean)

        if debug:
            assert next_h_mean.dim() >= 2, (
                f"next_h_mean dimension should be >= 2, "
                f"got {next_h_mean.dim()} with {next_h_mean.shape}"
            )
            _expected_compose_obs_len = compute_multistep_model_in_size(
                singlestep_obs_len=self.singlestep_obs_len,
                singlestep_act_len=self.singlestep_act_len,
                multistep_len=H,
            )
            assert next_h_mean.shape[-1] == _expected_compose_obs_len, (
                f"next_h_mean expected shape (..., {_expected_compose_obs_len}), "
                f"got (..., {next_h_mean.shape[-1]})"
            )
            assert not torch.isnan(next_h_mean).any(), (
                "next_h_mean contains 'nan'"
            )

        # Propagate the frozen original observed history unchanged so the next
        # AR iteration can again slice s^H_τ[i+1:].
        return StateHistory(
            h_mean=next_h_mean,
            h_obs_original=state_history.h_obs_original,
        )
