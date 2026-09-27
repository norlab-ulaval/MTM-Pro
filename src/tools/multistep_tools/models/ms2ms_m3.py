# coding=utf-8
"""
M3 (Multi-step Model) MS->MS forecast baseline (RLRP-693).

Permanent production module. Introduced by section 5 of the MS->MS forecast baselines
(End-to-End-TCN / M3 / TBM) ``.junie`` plan
(``feature_ms2ms_forecast_baselines_e2etcn_m3_tbm_plan_RLRP-692_693_694_20260703.md``).
Reproduces Asadi, Misra, Kim, Littman (arXiv:1905.13320) "Combating the Compounding-Error Problem
with a Multi-step Model": a family of maps ``{T_j}_{j=1..F}`` each predicting ``s_{t+j}`` directly
from the starting state ``s_t`` and the action sub-sequence ``a^j = (a_t, ..., a_{t+j-1})`` (never
feeding back intermediate predictions). Per plan section 0.1, M3 is a deterministic single net:
``ensemble_size`` is hard-fixed to 1, ``deterministic`` to True, and no ``distribution_name`` knob
is exposed.

Made **paper-faithful** by RLRP-821 (single-state conditioning, family indexed by the look-ahead
depth, masked objective) -- see the class docstring and
``rlrp-821-m3-paper-faithful-plan-20260911.md``. ⚠️ All pre-RLRP-821 M3 checkpoints and results
are void.
"""
from typing import Dict, Optional, Union

import omegaconf
import torch

from tools.multistep_tools.models.abstract_horizon_indexed_ms2ms_forecast import (
    AbstractHorizonIndexedMS2MSForecast,
)


class MS2MSMultiStepModel(AbstractHorizonIndexedMS2MSForecast):
    """M3 multi-step model horizon-indexed forecast baseline (RLRP-693 / **RLRP-821**).

    ``T_j([o_t, enc(j), pad(a^j)])`` for ``j = 1..F`` (``F == horizon_len``): each map predicts
    ``ô_{t+j}`` directly from the **single starting observation** ``o_t``, the normalised
    look-ahead index ``enc(j) = j/F`` and the action sub-sequence ``a^j = a_{t..t+j-1}``. No
    predicted state ever re-enters an input *within* a window -- the paper's anti-compounding
    property.

    **PAPER-FAITHFUL since RLRP-821** (plan: ``rlrp-821-m3-paper-faithful-plan-20260911.md``).
    The three departures that previously disqualified this class as *the* M3 baseline are
    resolved **unconditionally** -- there is no flag, no legacy code path and no ablation alias
    reintroducing them (operator decision: the pre-fix behaviour is *wrong*, not an alternative):

    1. *State surrogate* -- was the full flattened history ``(o, a)_{t-H+1..t}`` (ex operator
       decision ``Q-G``), now the paper's single Markov state: :meth:`_history_features` returns
       ``o_t`` **only**.
    2. *Family index* -- was the OUTPUT WINDOW ``H`` (``H`` maps, ``enc(h) = h/H``, ``H*Da``
       action slots), now the **look-ahead depth**: :meth:`_family_len` is ``F``, so the family is
       ``{T_j}_{j=1..F}`` with ``F-1`` head siblings and ``F*Da`` action slots (none of them
       identically zero).
    3. *Loss dilution* -- with ``F < H`` the leading ``H-F`` composed-window steps used to
       re-predict already-observed timesteps under full weight ``1.0``, diluting the forecast
       objective by ``(H-F)/H`` (``1/2`` on the shipped ``{HI:20, HO:10}`` arm). They are now the
       **observed-history echo** (:meth:`AbstractHorizonIndexedMS2MSForecast._echo_history_obs_steps`,
       detached) and are **masked out of the objective** (:meth:`_forecast_step_loss_mask`).

    The ``H``-step input/output sequence length is deliberately unchanged (decision ``KD1``): it
    is the shared replay-buffer / data-buffer-processor contract, so fidelity is achieved
    **internally** by selecting the relevant timesteps and masking the irrelevant ones. The buffer
    processor, the deploy head and every reporting path are untouched.

    ⚠️ **Every pre-RLRP-821 M3 checkpoint, metric and sweep-arm output is VOID** and must be
    re-run: the per-map input width and the family size both changed, so ``load_state_dict``
    fails loud on a shape mismatch (intended -- a silent re-init would poison a baseline). No
    migration shim is provided.

    **Remaining, deliberate departures** (caption them in any write-up):

    - *Output contract* (plan section 0.3): the composed forecast echoes the observed **action**
      columns; the paper predicts states only. Not supervised as a forecast, orthogonal to the
      fidelity of ``{T_j}``, and explicitly out of scope.
    - *Rollout*: the paper's ``{T_h}`` rollout re-samples each action from the policy at the
      intermediate predicted state. :meth:`forecast` is instead a one-shot stacked forward
      against a GIVEN plan -- and that **is the intended deployment proxy** (operator decision):
      it measures model drift under a *frozen* action sequence, with compounding measured ACROSS
      windows by the self-fed stride-``F`` loop below, not inside one.

    **Deployment contract** (validated against the deploy path; RLRP-821 FR10 / FR11):

    - **One-shot**: ``OneDTransitionRewardModelV2.forecast_open_loop_per_horizon`` issues exactly
      ONE :meth:`forecast` call and returns the trailing ``(F, Do)`` steps, index ``h-1`` ==
      ``ô_{t+h}``. The plan channel is ``a_{t+1..t+F-1}`` (``F-1`` entries); ``a_t`` arrives
      through the action history, so the driving sequence is ``[a_t] ++ plan``.
    - **Self-fed, stride ``F``**: at test time that call is driven by
      ``MultistepMotionModelTestTimeRolloutDeployer.forecast_horizon_selffed_target_env`` +
      ``multistep_model_testtime_SELFFED_forecast_and_collect_pred_stats`` -- **the** reporting
      path for the M3 drift curve. After the ground-truth warm-up the ONLY ground truth refed to
      the model is the **action** channel (the plan plus the next anchor action ``a_{t+F}``); the
      next anchor observation is the model's own ``ô_{t+F}``, and the reference trajectory
      advances by exactly ``F`` so predicted step ``h`` stays scored against GT step ``t+h``.
      Contrast with the sibling collectors, which measure *different* quantities: ``..._OPENLOOP_...``
      is one-shot forecast accuracy (GT-rebuilt history, stride 1) and
      ``multistep_model_testtime_rollout_and_collect_pred_stats`` is single-step compounding.
    - ``target_is_delta`` must stay ``False``: ``forecast_open_loop_per_horizon`` raises
      ``NotImplementedError`` in delta mode (pre-existing limitation).
    - The deployer still needs ``history_len`` real steps in its ring buffers before the first
      anchor is valid. That warm-up is an artifact of the shared ``H``-step buffer contract, **not**
      evidence of history conditioning -- it is the only place ground-truth observations enter.
    - The self-fed loop keeps pushing the ``h = 1..F-1`` predicted observations back into the
      rings; those slots are **inert for M3** (it reads only the last obs slot, i.e. the anchor)
      and exist for the history-consuming families sharing the deployer.

    **Two realisations of the family ``{T_j}``** (``per_horizon_heads``, RLRP-729):

    - ``True`` (default since RLRP-729): the paper's LITERAL definition -- ``F`` independent maps
      ``T_j``, one per look-ahead depth, sharing no weights (the paper says "different
      functions"). Use this to report M3 as-defined; see
      :meth:`AbstractHorizonIndexedMS2MSForecast._build_per_horizon_head_siblings` for the cost
      (``~F x`` parameters, ``1/F`` of the queries per map and hence a sample-efficiency trade)
      and for the checkpoint-incompatibility warning. Requires ``F >= 2``.
    - ``False`` (ex default, operator decision ``Q-H``): ONE shared network conditioned on the
      index ``j``. Parameter-efficient, but the weight tying across horizons IS a capacity and
      inductive-bias reduction w.r.t. the paper -- it is *not* an equivalence, and any write-up
      must say so. Also the only valid realisation at ``F == 1``.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        singlestep_obs_len: int,
        singlestep_act_len: int,
        history_len: int,
        horizon_len: int,
        num_layers: int = 4,
        hid_size: int = 200,
        ensemble_size: int = 1,
        propagation_method: Optional[str] = None,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        mae_loss: bool = True,
        model_use_double_precision: bool = False,
        dropout: float = 0.0,
        ms_head_dropout: float = 0.0,
        ms_head_num_layers: int = 1,
        residual_form: str = "identity_v2",
        layer_bloc=None,  # RLRP-768 per-region layer-bloc selector (forwarded)
        # Realisation of the paper's family ``{T_h}`` (RLRP-729; ex operator decision ``Q-H``).
        # ``True`` (default) -> the paper's LITERAL ``H`` independent maps ``T_h``. ``False`` ->
        # one shared net conditioned on ``h`` (parameter tying across horizons; the pre-RLRP-729
        # default, i.e. the bit-exact legacy behaviour).
        per_horizon_heads: bool = True,
        # RLRP-824 Step 7 (profiling-gated, GO on every ULH dataset row): evaluate the SHARED net
        # once over the stacked ``F`` horizon queries instead of the per-horizon Python loop.
        # Same weights / same math up to floating-point reassociation (``allclose``, not
        # bit-exact). Default ``False`` keeps the loop byte-for-byte; rejected with
        # ``per_horizon_heads=True``.
        vectorized_family_forward: bool = False,
        # RLRP-761 S3: accepted for config-surface compatibility only -- see
        # ``AbstractMS2MSForecast._validate_unsupported_feature_weighting``.
        feature_weight_mode: str = "tempered",
        feature_weight_max_ratio: Optional[float] = None,
        description: Optional[str] = None,
        feature_geometry=None,
        internal_orientation=None,
        orientation_singlestep_slots=None,
        # Config-level future-action-plan conditioning switch (operator follow-up to the
        # post-implementation review of
        # ``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``):
        # ``False`` => the true plan-FREE baseline. Default ``True`` keeps every existing
        # config bit-exact.
        enable_future_action_plan_conditioning: bool = True,
        # RLRP-824 Step 7 (FR9): opt-in autocast around training_step / validation_step. The
        # profiling gate found AMP a NO-GO on the dispatch-bound per-horizon loop; re-evaluate
        # with ``vectorized_family_forward: true``. ``null`` = bit-exact.
        mixed_precision: Optional[str] = None,
    ):
        # Per plan section 0.1, M3 is a deterministic single net: no ensemble, no probabilistic
        # head. These capabilities are therefore NOT exposed as knobs; they are hard-fixed here.
        # ``ensemble_size`` is accepted only for pipeline/config-surface compatibility (every
        # ``ms_model`` config declares it and the reporting path reads ``cfg.ms_model.ensemble_size``,
        # mirroring the ``B_tcn_deterministic`` convention); only ``ensemble_size == 1`` is valid.
        if ensemble_size != 1:
            raise ValueError(
                "MS2MSMultiStepModel is a deterministic single net (plan section 0.1); "
                f"ensemble_size must be 1, got {ensemble_size}."
            )
        # Permanent capability. Introduced by RLRP-729 (the literal per-horizon-head variant
        # deferred by operator decision ``Q-H`` of the MS->MS forecast baselines
        # (End-to-End-TCN / M3 / TBM) ``.junie`` plan,
        # ``feature_ms2ms_forecast_baselines_e2etcn_m3_tbm_plan_RLRP-692_693_694_20260703.md``).
        # MUST be set BEFORE ``super().__init__``: the shared base reads it from
        # ``_build_network``, which runs inside the parent constructor.
        self.per_horizon_heads = bool(per_horizon_heads) # Enable `_per_horizon_heads_active` in parent
        self.vectorized_family_forward = bool(vectorized_family_forward)
        if self.vectorized_family_forward and self.per_horizon_heads:
            raise ValueError(
                "MS2MSMultiStepModel: vectorized_family_forward=True is incompatible with "
                "per_horizon_heads=True (the literal {T_j} realisation has one independent map per "
                "horizon); use the shared-j-conditioned realisation (per_horizon_heads=False)."
            )
        if self.per_horizon_heads and int(horizon_len) < 2:
            # RLRP-821: the family is indexed by the LOOK-AHEAD DEPTH ``F == horizon_len``, so it
            # is ``F`` -- not ``H`` -- that must exceed 1. ``F == 1`` collapses ``{T_j}`` to a
            # single map, making the two realisations coincide; fail loud rather than silently
            # reporting a "per-horizon-head" run that is the shared net.
            raise ValueError(
                "MS2MSMultiStepModel: per_horizon_heads=True is meaningless with "
                f"horizon_len={horizon_len} (the family {{T_j}} has a single member); use the "
                "shared-j-conditioned realisation (per_horizon_heads=False)."
            )

        super().__init__(
            in_size,
            out_size,
            device=device,
            singlestep_obs_len=singlestep_obs_len,
            singlestep_act_len=singlestep_act_len,
            history_len=history_len,
            horizon_len=horizon_len,
            num_layers=num_layers,
            ensemble_size=1,  # hard-fixed (deterministic single net, plan section 0.1)
            hid_size=hid_size,
            deterministic=True,  # hard-fixed (deterministic single net, plan section 0.1)
            propagation_method=propagation_method,
            learn_logvar_bounds=False,
            activation_fn_cfg=activation_fn_cfg,
            mae_loss=mae_loss,
            model_use_double_precision=model_use_double_precision,
            dropout=dropout,
            ms_head_dropout=ms_head_dropout,
            ms_head_num_layers=ms_head_num_layers,
            residual_form=residual_form,
            layer_bloc=layer_bloc,
            feature_weight_mode=feature_weight_mode,
            feature_weight_max_ratio=feature_weight_max_ratio,
            enable_auto_loss_weighting=False,
            description=description,
            feature_geometry=feature_geometry,
            # RLRP-736 bespoke-forward plan §3.5 (C.3): by-construction rotation rep
            # (defaults quaternion / None => bit-exact).
            internal_orientation=internal_orientation,
            orientation_singlestep_slots=orientation_singlestep_slots,
            enable_future_action_plan_conditioning=enable_future_action_plan_conditioning,
            mixed_precision=mixed_precision,
        )

    def _model_family_tag(self) -> str:
        return "M3"

    # ==== Paper-faithful single-state conditioning (RLRP-821, FR1) ==============================
    def _history_features(self, history_flat: torch.Tensor) -> torch.Tensor:
        """The paper's single starting state: the LAST observed observation ``o_t`` only.

        Asadi, Misra, Kim & Littman (arXiv:1905.13320) condition every ``T_j`` on ONE Markov
        state ``s_t``. The pre-RLRP-821 implementation passed the whole flattened ``H``-step
        history ``(o, a)_{t-H+1..t}`` (ex operator decision ``Q-G``), which made the model a
        *history-conditioned* multi-step map -- one of the three departures that disqualified it
        as the M3 baseline. It is now unconditionally ``o_t``: there is **no** knob restoring the
        full-history surrogate (FR7); the discarded behaviour, and every checkpoint / metric it
        produced, is void.

        The ``H``-step input window itself is unchanged (the replay-buffer contract requires equal
        input/output sequence length, decision ``KD1``): the remaining timesteps are simply not
        consulted. At deploy time the last obs slot IS the self-fed anchor ``ô_{t+F}`` of the
        stride-``F`` rollout, so nothing downstream needs to change either.

        See ``rlrp-821-m3-paper-faithful-plan-20260911.md``.
        """
        # (..., obs_len, history_len) -> the trailing timestep ``o_t``, (..., obs_len)
        obs_feature = self.extract_obs_features_from_multistep_composed_array(
            history_flat, is_model_output=False, vectorized=True
        )
        return obs_feature[..., -1]

    def _history_feature_size(self) -> int:
        """``singlestep_obs_len``: the per-map state input is the single observation ``o_t``."""
        return self.singlestep_obs_len

    def _per_horizon_in_size(self) -> int:
        """Per-map input width on the SINGLE-STEP input-encode budget (RLRP-821, R5).

        The inherited width adds ``_trunk_in_size - in_size`` -- the by-construction orientation
        (RLRP-736) INPUT-encode widening measured on the **composed ``H``-step window**. Under
        the single-state surrogate the per-map input carries exactly ONE single-step obs block, so
        the correct budget is the per-block one, ``_ss_block_in_extra`` (RLRP-736 bespoke-forward
        plan §3.4), paired with the single-step encode of
        :meth:`_apply_orientation_input_encoding` below. Reusing the composed delta would
        over-size the trunk by ``(H - 1)`` slots' worth of expansion.

        Both quantities are ``0`` for the neutral (``quaternion``) rep, so that path is
        bit-identical either way.
        """
        return (
            self._history_feature_size()
            + int(getattr(self, "_ss_block_in_extra", 0))
            + 1
            + self._per_horizon_extra_size()
        )

    def _apply_orientation_input_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the attitude slot(s) of the SINGLE observation ``o_t`` (RLRP-821, R5).

        Mirrors -- on the input side -- what
        :meth:`AbstractHorizonIndexedMS2MSForecast._apply_orientation_output_decoding` already
        does on the output side: the per-map tensor is
        ``[o_t (Do) , enc(j) (1) , pad(a^j) (F*Da)]``, i.e. ONE single-step obs block followed by
        conditioning columns that carry no attitude. The composed-window slot set
        (``_ori_in_slots``, one entry per window step) would therefore mis-address it; the
        per-block set ``_ss_ori_in_slots`` is the right one, and the conditioning columns are
        copied through verbatim by the splice.

        No-op / bit-exact when the by-construction rotation rep is OFF.
        """
        return self._apply_ss_orientation_input_encoding(x)

    # ==== Paper-faithful family axis (RLRP-821, FR2-FR3) ========================================
    def _family_len(self) -> int:
        """``F``: one map ``T_j`` per LOOK-AHEAD DEPTH, as the paper indexes the family."""
        return int(self.horizon_len)

    def _window_lead(self) -> int:
        """``W - F``: the leading composed-window steps that are NOT forecasts.

        They are re-predictions of already-observed timesteps ``o_{t+F-H+1..t}`` in the shared
        ``H``-step buffer layout. They are filled from the observed history
        (``_echo_history_obs_steps``, detached) and excluded from the objective by
        :meth:`_forecast_step_loss_mask`, so they neither consume family capacity nor dilute the
        forecast loss (the pre-RLRP-821 behaviour spent ``(H-F)/H`` of both on them).

        RLRP-824: ``W = output_window_len = max(H, F)``, so the lead is ``H - F`` for every
        ``F <= H`` config (unchanged) and ``0`` on the asymmetric ``F > H`` window, where every
        one of the ``W == F`` steps is a genuine ``T_j`` forecast.
        """
        return int(self.output_window_len) - int(self.horizon_len)

    def _forecast_step_loss_mask(self) -> Optional[torch.Tensor]:
        """Zero the leading ``H - F`` composed-window obs steps out of the objective (FR5).

        ``None`` when ``F == H`` (nothing to mask: every window step is a genuine forecast), which
        keeps that configuration on the verbatim inherited loss.
        """
        lead = self._window_lead()
        if lead <= 0:
            return None
        mask = torch.ones(int(self.output_window_len), device=self.device)
        mask[:lead] = 0.0
        return mask

    # ==== M3 action sub-sequence conditioning ====================================================
    def _per_horizon_extra_conditioning(
        self,
        history_feat: torch.Tensor,
        h: int,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return the padded action sub-sequence ``a^j = (a_t, ..., a_{t+j-1})`` (Q-E).

        The ``j`` driving actions are placed at the front of a fixed-length
        ``horizon_len * singlestep_act_len`` vector (RLRP-821: the LOOK-AHEAD axis, so the
        deepest query ``j = F`` fills every slot), the remaining slots zero-padded. This is the
        continuous-action generalisation of the paper's discrete-action ``a^j`` input (plan
        section 0.3 permitted I/O adaptation); the paper's body prescribes no action-sequence
        encoding, so the padded layout is ours.

        When ``future_actions`` (the internal driving sequence ``a_{t..t+F-1}`` == ``[a_t] ++ plan``)
        is supplied, the sub-sequence is built from it, with slot 0 == ``a_t``. Without a plan the
        channel keeps the SAME layout with the unknown future blanked (``[a_t, 0, ..., 0]``) -- on
        this axis slot ``j-1`` MEANS ``a_{t+j-1}``, so an oldest-actions history echo would be a
        layout violation. See
        :meth:`AbstractHorizonIndexedMS2MSForecast._build_padded_action_subsequence` (stage ``A4`` of
        the Fix the MS->MS future-action-plan conditioning contract ``.junie`` plan,
        ``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).
        """
        return self._build_padded_action_subsequence(
            history_feat, h, future_actions=future_actions
        )

    def _per_horizon_extra_conditioning_family(
        self,
        history_feat: torch.Tensor,
        family_len: int,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """One-shot ``(..., F, F * Da)`` stack of the padded action sub-sequences ``{a^j}``.

        RLRP-824 test-time rollout hotfix (2026-09-19): the vectorized family forward composes
        its ``F`` inputs at once (``torch.equal`` to the per-``j`` builder above), which removes
        the ~1000-launch composition loop that dominated a ``B=1`` deploy step at ``F=1000``.
        """
        return self._build_padded_action_subsequence_family(
            history_feat, family_len, future_actions=future_actions
        )

    def _per_horizon_extra_size(self) -> int:
        # RLRP-821 (FR4): ``F * Da`` -- the action sub-sequence axis is the LOOK-AHEAD depth, so
        # the deepest query ``j = F`` fills every slot and no slot is zero-by-construction (the
        # pre-fix ``H * Da`` width left the trailing ``(H-F) * Da`` columns identically zero).
        return self.horizon_len * self.singlestep_act_len
