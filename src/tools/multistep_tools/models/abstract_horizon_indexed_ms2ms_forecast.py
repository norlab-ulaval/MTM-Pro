# coding=utf-8
"""
Shared base for the horizon-indexed ``ms2ms`` forecast baselines (M3 + TBM).

Permanent production module. Introduced by section 2.2 of the MS->MS forecast baselines
(End-to-End-TCN / M3 / TBM) ``.junie`` plan
(``feature_ms2ms_forecast_baselines_e2etcn_m3_tbm_plan_RLRP-692_693_694_20260703.md``).
Encapsulates the "predict ``o_{t+h}`` as a direct function of ``(history, h, [actions])`` then
stack over ``h``" pattern shared by M3 (RLRP-693) and TBM (RLRP-694): property (2) is built by
querying the network *per horizon index*, not by one-shot decoding and not by recursion.
"""
import copy
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

import omegaconf
import torch
from mbrl.models import truncated_normal_init
from torch import nn as nn

from tools.multistep_tools.models.abstract_ms2ms_forecast import AbstractMS2MSForecast
from tools.multistep_tools.models.exponential_family_mlp import ExponentialFamilyMLP
from tools.multistep_tools.models.exponential_family_mlp_utils import (
    zero_init_residual_blocks_,
)
from tools.multistep_tools.models.utils import (
    reduce_deterministic_compose_loss,
    reduce_probabilistic_compose_loss,
)


class AbstractHorizonIndexedMS2MSForecast(AbstractMS2MSForecast):
    """Abstract base for the horizon-indexed forecast baselines (M3 and TBM).

    Both baselines predict ``o_{t+h}`` directly from ``(history, h, [actions])`` and stack the
    per-``h`` single-step predictions into the multistep forecast; crucially **no predicted state
    is ever fed back** into the network (the only state input is the observed history), which is
    the defining anti-compounding property of both papers. The per-``h`` network is a plain
    ``ExponentialFamilyMLP`` built by :meth:`_build_network` (mapping the concatenated per-horizon
    input to ``singlestep_obs_len``), and this base owns the shared per-``h`` loop and the
    horizon-index encoding.

    RLRP-736 bespoke-forward plan §3.5 (C.3). This family HAS wired the
    by-construction rotation head: because each per-``h`` prediction routes through
    the base ``ExponentialFamilyMLP._default_forward`` (see :meth:`_predict_at_horizon`),
    the base already (a) encodes the attitude input slot(s) of the history window to
    the continuous internal rep (``R(q)=R(-q)`` sign-invariance; Geist et al. 2024)
    and (b) decodes the per-step obs attitude slot back to a unit quaternion. The
    two additive hooks below make the per-``h`` head rep-aware: the head OUTPUT is
    widened to the single-step ``_ss_trunk_out_len`` (rep-expanded) and decoded via
    the single-step :meth:`_apply_ss_orientation_output_decoding` (NOT the composed
    MS decode), and the per-``h`` INPUT size grows by the input-encode width delta.
    Neutral (``quaternion``) keeps everything byte-for-byte identical.

    **Two family axes (RLRP-821).** :meth:`_family_len` / :meth:`_window_lead` select which axis
    indexes ``{T_h}``:

    - the **output window** ``H`` (the defaults): one family member per composed-window step. This
      is ``MS2MSTrajectoryBasedModel``'s (TBM) contract and the only correct axis for it -- the
      defaults are NOT a legacy escape hatch.
    - the **look-ahead depth** ``F`` (``MS2MSMultiStepModel`` / M3): one map per *forecast* step,
      as Asadi, Misra, Kim & Littman (arXiv:1905.13320) define the family. The leading ``H-F``
      composed-window steps are then the observed-history echo
      (:meth:`_echo_history_obs_steps`, detached) and are excluded from the objective by
      :meth:`_forecast_step_loss_mask`.

    ⚠️ **Blast-radius containment (operator constraint, RLRP-821 FR9).** The masked objective is
    implemented as **overrides of INHERITED methods on THIS class**
    (:meth:`_deterministic_loss` / :meth:`_probabilistic_loss`), *precisely so that*
    ``AbstractMS2MSForecast`` and every one of its parents (``ExponentialFamilyMLP``,
    ``MultiStepMLP``, the mbrl ``Model`` chain) stay byte-for-byte unchanged: ``MS2MSEndToEndTCN``
    (E2E-TCN) inherits ``AbstractMS2MSForecast`` **directly**, as do the MTM-Pro / dual-head / AR
    families, and their running experiments must remain valid. Nothing may be added to the shared
    base to serve M3. See ``rlrp-821-m3-paper-faithful-plan-20260911.md``.
    """

    # By-construction rotation head IS wired for this family (see class docstring);
    # un-gates the ``AbstractMS2MSForecast.forecast`` fail-loud guard.
    _supports_by_construction_orientation: bool = True

    # Permanent architectural constant (RLRP-729). The module attributes that together
    # constitute ONE per-horizon map ``T_h``: the trunk plus the mean/logvar head stacks built
    # by ``ExponentialFamilyMLP._build_network``. The literal per-horizon-head variant
    # replicates exactly these and nothing else (the orientation encode/decode bookkeeping,
    # the normalizer handles and the loss surface are horizon-agnostic and stay shared).
    _PER_HORIZON_HEAD_MODULE_NAMES: Tuple[str, ...] = (
        "hidden_layers",
        "mean_and_logvar",
        "mean_layer",
        "logvar_layer",
    )

    # ==== Network build ==========================================================================
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
        """Build the per-horizon ``ExponentialFamilyMLP`` head.

        The model-level ``in_size`` / ``out_size`` are ignored: the per-horizon network maps the
        concatenated ``[history_features, horizon_index_encoding, extra_conditioning]`` input to a
        single-step ``singlestep_obs_len`` observation prediction (the shared observation forecast
        surface of ``AbstractMS2MSForecast``). This mirrors how ``tcn_ms2ss`` defers its head to
        ``super()._build_network`` with model-specific sizes.
        """
        # Delegate to the ExponentialFamilyMLP head builder directly: the intermediate
        # ``AbstractMS2MSForecast._build_network`` is abstract, so we bypass the MRO stub.
        # RLRP-736 bespoke-forward plan §3.5: build the per-horizon head at the
        # rep-expanded SINGLE-STEP width ``_ss_trunk_out_len`` (== singlestep_obs_len
        # when the by-construction rotation rep is OFF => bit-exact) so the raw
        # attitude slot can be decoded to a unit quaternion per step.
        ExponentialFamilyMLP._build_network(
            self,
            num_layers=num_layers,
            in_size=self._per_horizon_in_size(),
            hid_size=hid_size,
            out_size=self._ss_trunk_out_len,
            ensemble_size=ensemble_size,
            activation_fn_cfg=activation_fn_cfg,
            deterministic=deterministic,
            learn_logvar_bounds=learn_logvar_bounds,
            instanciate_logvar_bound_module=instanciate_logvar_bound_module,
            dropout=dropout,
        )
        if self._per_horizon_heads_active():
            self._build_per_horizon_head_siblings()
        return None

    # ==== Literal per-horizon heads ``{T_h}`` (RLRP-729) =========================================
    def _per_horizon_heads_active(self) -> bool:
        """``True`` iff this instance realises ``{T_h}`` as ``H`` INDEPENDENT maps.

        Attribute-driven (``self.per_horizon_heads``, set by the subclass BEFORE
        ``super().__init__`` so it is readable from :meth:`_build_network`) and defaulting to
        ``False``, so every subclass that does not expose the knob -- and every existing config --
        keeps the shared-``h``-conditioned net **bit-exact**.
        """
        return bool(getattr(self, "per_horizon_heads", False))

    def _build_per_horizon_head_siblings(self) -> None:
        """Build the ``H - 1`` independent sibling maps of the literal ``{T_h}`` variant (RLRP-729).

        Permanent production surface. Implements the literal per-horizon-head realisation of
        Asadi, Misra, Kim & Littman (arXiv:1905.13320), whose M3 is defined as ``H`` **different
        functions** ``{T_h}_{h=1..H}``; the shared-``h``-conditioned single net remains the adopted
        default (operator decision ``Q-H`` of the MS->MS forecast baselines (End-to-End-TCN / M3 /
        TBM) ``.junie`` plan,
        ``feature_ms2ms_forecast_baselines_e2etcn_m3_tbm_plan_RLRP-692_693_694_20260703.md``,
        deferred as RLRP-729).

        **Topology.** Horizon ``h == 1`` KEEPS the canonical modules built by
        ``ExponentialFamilyMLP._build_network`` (``self.hidden_layers`` / ``self.mean_layer`` /
        ...); horizons ``h = 2..H`` get one sibling each, held in
        ``self.per_horizon_head_siblings[h - 2]``. Deliberately NOT an alias of the canonical
        modules for ``h == 1``: aliasing would emit every canonical tensor twice in
        ``state_dict()`` (same storage, two key sets), which silently breaks any checkpoint key
        audit. The asymmetry is cosmetic -- all ``H`` maps are structurally identical and
        independently initialized.

        **Independence.** Each sibling is a deep copy of the canonical topology (so the layer-bloc
        / residual-form / dropout resolution of RLRP-768 is reproduced EXACTLY, by construction
        rather than by a duplicated builder) followed by a fresh ``truncated_normal_init`` draw and
        the residual-bloc identity re-zeroing. Fresh draws are what make the ``H`` maps genuinely
        different functions rather than ``H`` copies of one.

        ⚠️ **Cost and checkpoint compatibility.** Trainable parameters scale by ``~H`` (``H = 20``
        in the shipped configs) and each map sees ``1/H`` of the horizon queries per batch, so the
        variant trades the shared net's cross-horizon parameter tying (a capacity AND
        inductive-bias reduction w.r.t. the paper) for capacity at the cost of sample efficiency.
        Checkpoints are NOT interchangeable with the shared-net variant in either direction.
        """
        sibling_count = int(self._family_len()) - 1
        siblings = []
        for _ in range(max(sibling_count, 0)):
            sibling = nn.ModuleDict(
                {
                    name: copy.deepcopy(module)
                    for name, module in self._canonical_per_horizon_modules().items()
                }
            )
            # Fresh init draw per sibling => ``H`` genuinely independent maps. Mirrors the tail of
            # ``ExponentialFamilyMLP._build_network`` (global truncated-normal init, then re-zero
            # the identity residual blocs so they start as an exact identity ``y == x``).
            sibling.apply(truncated_normal_init)
            zero_init_residual_blocks_(sibling)
            siblings.append(sibling)

        self.per_horizon_head_siblings = nn.ModuleList(siblings)
        self.to(self.device)
        return None

    def _canonical_per_horizon_modules(self) -> dict:
        """The canonical (``h == 1``) trunk/head modules, keyed by attribute name.

        Only the names actually instantiated by ``ExponentialFamilyMLP._build_network`` for this
        capability set are returned (e.g. ``mean_and_logvar`` / ``logvar_layer`` exist only on the
        probabilistic path), so the sibling maps carry exactly the canonical module set.
        """
        modules = {}
        for name in self._PER_HORIZON_HEAD_MODULE_NAMES:
            module = self._modules.get(name)
            if isinstance(module, nn.Module):
                modules[name] = module
        return modules

    @contextmanager
    def _bound_per_horizon_head(self, h: int) -> Iterator[None]:
        """Temporarily rebind the trunk/head attributes to the sibling map ``T_h`` (RLRP-729).

        Rebinding ``self._modules`` for the duration of ONE per-horizon query (instead of
        duplicating the forward) is what lets the literal variant reuse
        ``ExponentialFamilyMLP._default_forward`` verbatim -- and therefore keeps the orientation
        input-encode / output-decode wiring (RLRP-736), the model-dtype cast and the probabilistic
        head split identical across the two variants. The rebind lives strictly inside the
        forward, so ``state_dict()`` / ``named_parameters()`` are never observed in the swapped
        state.

        :param h: 1-based horizon index; ``h == 1`` is the canonical map and is never rebound.
        """
        sibling = self.per_horizon_head_siblings[h - 2]
        saved = {}
        try:
            for name in self._PER_HORIZON_HEAD_MODULE_NAMES:
                if name in sibling:
                    saved[name] = self._modules[name]
                    self._modules[name] = sibling[name]
            yield
        finally:
            for name, module in saved.items():
                self._modules[name] = module

    # ==== Family axis (output window vs. look-ahead depth) =======================================
    def _family_len(self) -> int:
        """Number of members of the horizon-indexed family ``{T_h}`` queried per forward.

        **Default: the OUTPUT WINDOW length ``H``** (``history_len``) -- the axis
        :class:`~tools.multistep_tools.models.ms2ms_tbm.MS2MSTrajectoryBasedModel` (TBM) is built
        on and the only correct axis for it, since TBM predicts the whole composed window. The
        default is therefore **not** a legacy escape hatch; it is TBM's contract and keeps TBM
        byte-for-byte identical.

        ``MS2MSMultiStepModel`` (M3, RLRP-821) overrides it to the **look-ahead depth** ``F``
        (``horizon_len``), which is how Asadi, Misra, Kim & Littman (arXiv:1905.13320) index the
        family: one map per *forecast* step. See
        ``rlrp-821-m3-paper-faithful-plan-20260911.md``.
        """
        return int(self.history_len)

    def _window_lead(self) -> int:
        """Number of leading composed-window obs steps that are NOT network queries.

        **Default: ``0``** -- every one of the ``H`` composed-window steps is produced by a family
        member (TBM's output-window axis).

        On the look-ahead axis (M3, RLRP-821) the family has only ``F`` members, so the leading
        ``H - F`` window steps -- which are re-predictions of **already observed** timesteps
        ``o_{t+F-H+1..t}`` -- are instead filled from the observed history by
        :meth:`_echo_history_obs_steps` and excluded from the objective by
        :meth:`_forecast_step_loss_mask`. Must satisfy ``_window_lead() + _family_len() == H``.
        """
        return 0

    @contextmanager
    def _bound_history_window(self, history_flat: torch.Tensor) -> Iterator[None]:
        """Expose the RAW ``H``-step composed history window for the duration of one forward.

        :meth:`_build_padded_action_subsequence` reads the observed **action** columns out of the
        composed history layout. Historically it could read them straight from the first argument
        of the per-horizon conditioning hook, because :meth:`_history_features` returned the whole
        ``history_flat``. Since M3 (RLRP-821) reduces its state features to the single observation
        ``o_t``, that argument is no longer the composed window, so the window is bound here and
        resolved by :meth:`_resolve_history_window`.

        Deliberately a scoped binding (same pattern as :meth:`_bound_per_horizon_head`) rather
        than a signature change: the per-horizon conditioning hook is overridden by BOTH M3 and
        TBM, and TBM is out of scope for this change (it must stay bit-exact).
        """
        saved = getattr(self, "_history_window_flat", None)
        try:
            self._history_window_flat = history_flat
            yield
        finally:
            self._history_window_flat = saved

    def _resolve_history_window(self, history_feat: torch.Tensor) -> torch.Tensor:
        """The raw composed ``H``-step history window, falling back to ``history_feat``.

        The fallback keeps every direct (out-of-forward) call of
        :meth:`_build_padded_action_subsequence` -- and therefore every existing test -- working
        unchanged, and is exactly equivalent for any subclass whose
        :meth:`_history_features` is the identity (TBM).
        """
        window = getattr(self, "_history_window_flat", None)
        return history_feat if window is None else window

    def _echo_history_obs_steps(
        self, history_flat: torch.Tensor, count: int
    ) -> torch.Tensor:
        """Echo the ``count`` trailing OBSERVED obs steps of the input history (RLRP-821).

        Observation analogue of ``AbstractMS2MSForecast._extract_history_action_steps``: returns
        ``o_{t-count+1..t}`` with shape ``(..., count, singlestep_obs_len)``, ``detach()``ed so no
        gradient can flow through a slot that is not a prediction.

        Used to fill the leading ``H - F`` composed-window steps on the look-ahead family axis
        (:meth:`_window_lead`), which are re-predictions of already-observed timesteps in the
        window layout ``o_{t+F-H+1..t+F}``: window step ``h <= H-F`` is the observed ``o_{t+h-(H-F)}``
        -- i.e. exactly the ``count`` trailing history observations, in order. Keeping the composed
        window well-formed matters for the geometry / decompose / reporting paths; the objective
        excludes those slots through :meth:`_forecast_step_loss_mask`.
        """
        # (..., obs_len, history_len) -> (..., history_len, obs_len)
        obs_feature = self.extract_obs_features_from_multistep_composed_array(
            history_flat, is_model_output=False, vectorized=True
        )
        obs_steps = obs_feature.transpose(-2, -1)
        return obs_steps[..., -int(count) :, :].detach()

    # ==== Encoder (shared per-horizon loop) ======================================================
    def _forward_encoder(
        self,
        history_flat: torch.Tensor,
        only_elite: bool = False,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Stack the per-horizon single-step predictions into the ``H``-step composed window.

        The family is queried over ``step in [1, _family_len()]``; when :meth:`_window_lead` is
        non-zero the leading ``H - _family_len()`` window slots are **not** network queries but the
        observed-history echo of :meth:`_echo_history_obs_steps` (RLRP-821). With the defaults
        (``_family_len() == H``, ``_window_lead() == 0``) this is the historical behaviour,
        byte-for-byte.

        When ``future_actions`` is provided it is threaded to the per-horizon conditioning hook so
        control-conditioned subclasses (M3 / TBM) consume the **internal driving sequence**
        ``a_{t..t+F-1}`` (``[a_t] ++ plan``) instead of the history actions. ``None`` keeps the
        history-echo path (bit-exact).

        Stage ``A4`` of the Fix the MS->MS future-action-plan conditioning contract ``.junie`` plan
        (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``) settled
        the slot convention: see :meth:`_build_padded_action_subsequence` for the window-step to
        forecast-step mapping.
        """
        history_feat = self._history_features(history_flat)
        family_len = int(self._family_len())
        window_lead = int(self._window_lead())
        # RLRP-824: the composed output window is ``W = output_window_len`` (``== history_len``
        # for ``F <= H``; ``== horizon_len`` on the asymmetric window, where the lead is 0).
        output_window_len = int(self.output_window_len)
        if window_lead + family_len != output_window_len:
            raise ValueError(
                f"{type(self).__name__}: _window_lead()={window_lead} + "
                f"_family_len()={family_len} must equal the composed output window "
                f"output_window_len={output_window_len} (history_len={self.history_len}, "
                f"horizon_len={self.horizon_len})."
            )
        if window_lead > 0 and not self.deterministic:
            # The echoed slots are observations, not distributions: there is no logvar to emit
            # for them. Fail loud instead of inventing one (only M3 uses a non-zero lead, and it
            # hard-fixes ``deterministic=True``).
            raise NotImplementedError(
                f"{type(self).__name__}: a non-zero _window_lead() is only supported on the "
                "deterministic path (the echoed leading window steps carry no variance)."
            )

        if self._vectorized_family_forward_active():
            # RLRP-824 (Step 7, profiling-gated GO on every dataset row): ONE batched evaluation
            # of the shared net over the stacked per-horizon inputs instead of ``F`` sequential
            # launches. Same weights, same math (numerically ``allclose``, not bit-exact, because
            # the GEMM reassociates); default OFF keeps the loop below byte-for-byte.
            with self._bound_history_window(history_flat):
                obs_mean_steps, obs_logvar_steps = self._predict_family_vectorized(
                    history_feat, family_len, only_elite=only_elite, future_actions=future_actions
                )
        else:
            mean_steps = []
            logvar_steps = [] if not self.deterministic else None

            # Readability-first per-horizon Python loop (the legacy, bit-exact path). The
            # dispatch-bound cost of this loop is what ``vectorized_family_forward`` removes.
            with self._bound_history_window(history_flat):
                for step in range(1, family_len + 1):
                    mean_h, logvar_h = self._predict_at_horizon(
                        history_feat,
                        step,
                        only_elite=only_elite,
                        future_actions=future_actions,
                    )
                    mean_steps.append(mean_h)
                    if logvar_steps is not None:
                        logvar_steps.append(logvar_h)

            # (..., _family_len(), singlestep_obs_len)
            obs_mean_steps = torch.stack(mean_steps, dim=-2)
            obs_logvar_steps = (
                torch.stack(logvar_steps, dim=-2) if logvar_steps is not None else None
            )

        if window_lead > 0:
            echo = self._echo_history_obs_steps(history_flat, window_lead)
            echo = echo.expand(
                (*obs_mean_steps.shape[:-2], window_lead, echo.shape[-1])
            ).to(dtype=obs_mean_steps.dtype)
            # (..., history_len, singlestep_obs_len): observed lead ++ the F genuine forecasts.
            obs_mean_steps = torch.cat([echo, obs_mean_steps], dim=-2)

        return obs_mean_steps, obs_logvar_steps

    # ==== Vectorized family forward (RLRP-824 Step 7) ===========================================
    def _vectorized_family_forward_active(self) -> bool:
        """``True`` iff the shared net is evaluated ONCE over the stacked family axis.

        Attribute-driven (``self.vectorized_family_forward``, set by the subclass BEFORE
        ``super().__init__``) and defaulting to ``False`` so every existing config keeps the
        per-horizon Python loop **bit-exact**. Incompatible with the literal ``{T_h}`` realisation
        (``per_horizon_heads=True``): the ``F`` independent maps cannot share one launch.
        """
        active = bool(getattr(self, "vectorized_family_forward", False))
        if active and self._per_horizon_heads_active():
            raise ValueError(
                f"{type(self).__name__}: vectorized_family_forward=True is incompatible with "
                "per_horizon_heads=True (the literal {T_h} realisation has one independent map "
                "per horizon and cannot be batched into a single shared-net evaluation)."
            )
        return active

    def _per_horizon_input(
        self,
        history_feat: torch.Tensor,
        h: int,
        future_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The per-horizon network input ``[history_feat, enc(h), extra(h)]`` (1-based ``h``)."""
        h_enc = self._encode_horizon_index(history_feat, h)
        parts = [history_feat, h_enc]
        extra = self._per_horizon_extra_conditioning(
            history_feat, h, future_actions=future_actions
        )
        if extra is not None:
            parts.append(extra)
        return torch.cat(parts, dim=-1)

    def _predict_family_vectorized(
        self,
        history_feat: torch.Tensor,
        family_len: int,
        only_elite: bool = False,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Evaluate the shared per-horizon net ONCE over all ``family_len`` horizon queries.

        The ``F`` per-horizon inputs (cheap tensor composition, no network) are stacked on a new
        family axis ``(..., B, F, in_h)``, the family axis is folded into the batch axis
        (``(..., B*F, in_h)`` -- the layout the shared ``ExponentialFamilyMLP`` head sees on the
        loop path, one row per query) for a single forward, and the outputs are unfolded back to
        ``(..., B, F, singlestep_obs_len)``. Row-wise ops (orientation encode / decode, dtype
        cast, probabilistic split) are unchanged by the folding, so the result matches the loop up
        to floating-point reassociation (``allclose``). Introduced by Step 7 of the RLRP-824 plan
        after the FR15 profiling gate showed the per-horizon loop dominates on every target row.
        """
        # RLRP-824 test-time rollout hotfix (2026-09-19): the ``F`` per-horizon inputs are
        # composed in ONE shot (:meth:`_per_horizon_inputs_family`) instead of ``F`` sequential
        # ``_per_horizon_input`` calls. The per-query composition (scalar encode + padded action
        # sub-sequence, ~10 tiny launches each) was the whole cost of a ``B=1`` deploy step on
        # the ULH rows (F=1000: ~250 ms/step on the Orin, GPU idle). Pure copying, so bit-exact.
        x_all = self._per_horizon_inputs_family(
            history_feat, int(family_len), future_actions=future_actions
        )  # (..., B, F, in_h)
        if x_all.ndim < 3:
            raise ValueError(
                f"{type(self).__name__}: the vectorized family forward needs a batched history "
                f"(..., B, in); got a stacked family tensor of shape {tuple(x_all.shape)} (the "
                "composed-observation extractors require a batch axis on the loop path too)."
            )
        batch_shape = x_all.shape[:-2]
        folded = x_all.reshape(*batch_shape[:-1], batch_shape[-1] * int(family_len), x_all.shape[-1])
        mean, logvar = ExponentialFamilyMLP._default_forward(self, folded, only_elite=only_elite)

        def _unfold(y: torch.Tensor) -> torch.Tensor:
            # Keep whatever leading dims the head returned (it may prepend the ensemble axis, as
            # it does on the loop path) and split only the folded ``B*F`` axis back to ``(B, F)``.
            return y.reshape(*y.shape[:-2], batch_shape[-1], int(family_len), y.shape[-1])

        mean = _unfold(mean)
        if logvar is not None:
            logvar = _unfold(logvar)
        return mean, logvar

    # ==== Family-wide input composition (RLRP-824 test-time rollout hotfix, 2026-09-19) ==========
    def _per_horizon_inputs_family(
        self,
        history_feat: torch.Tensor,
        family_len: int,
        future_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compose the ``family_len`` per-horizon inputs ``[history_feat, enc(h), extra(h)]`` at once.

        The stacked equivalent of ``torch.stack([_per_horizon_input(h) for h in 1..F], dim=-2)``,
        built with ONE broadcast per part instead of ``F`` sequential compositions. Every part is a
        copy / broadcast of existing values (no arithmetic on the data), so the result is
        ``torch.equal`` to the per-step stack -- pinned by the conditioning test suite.

        Motivation: ``_predict_family_vectorized`` (Step 7 of the RLRP-824 plan) removed the
        ``F`` network launches but still composed its inputs per query; on a ``B=1`` test-time
        deploy step with ``F=1000`` that composition (~10 tiny launches per query) was the entire
        ~250 ms/step measured on the Jetson Orin, i.e. hours of GPU-idle epoch-checkpoint rollouts.

        :param history_feat: ``(..., B, Dh)`` state/history features.
        :param family_len: the number of family queries ``F``.
        :param future_actions: the internal driving sequence ``a_{t..t+F-1}`` or ``None``.
        :return: ``(..., B, F, in_h)``.
        """
        family_len = int(family_len)
        history = history_feat.unsqueeze(-2).expand(
            *history_feat.shape[:-1], family_len, history_feat.shape[-1]
        )
        parts = [history, self._encode_horizon_index_family(history_feat, family_len)]
        extra = self._per_horizon_extra_conditioning_family(
            history_feat, family_len, future_actions=future_actions
        )
        if extra is not None:
            parts.append(extra)
        return torch.cat(parts, dim=-1)

    def _encode_horizon_index_family(
        self, history_feat: torch.Tensor, family_len: int
    ) -> torch.Tensor:
        """``(..., B, F, 1)`` stack of :meth:`_encode_horizon_index` for ``h in 1..F``.

        Bit-exact with the per-step encode: ``float(h) / float(_family_len())`` is a float64
        division there and here (``torch.arange`` in float64), both rounded once to the
        ``history_feat`` dtype.
        """
        values = torch.arange(
            1, int(family_len) + 1, dtype=torch.float64, device=history_feat.device
        ) / float(self._family_len())
        values = values.to(history_feat.dtype).reshape(int(family_len), 1)
        return values.expand(*history_feat.shape[:-1], int(family_len), 1)

    def _per_horizon_extra_conditioning_family(
        self,
        history_feat: torch.Tensor,
        family_len: int,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """``(..., B, F, extra)`` stack of :meth:`_per_horizon_extra_conditioning`, or ``None``.

        Default: the generic per-step stack (correct for ANY override of the per-step hook, at
        the per-step cost). M3 and TBM override it with
        :meth:`_build_padded_action_subsequence_family`, the one-shot builder of their shared
        padded action sub-sequence.
        """
        extras = [
            self._per_horizon_extra_conditioning(
                history_feat, step, future_actions=future_actions
            )
            for step in range(1, int(family_len) + 1)
        ]
        if all(extra is None for extra in extras):
            return None
        return torch.stack(extras, dim=-2)

    def _build_padded_action_subsequence_family(
        self,
        history_feat: torch.Tensor,
        family_len: int,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """One-shot ``(..., B, F, _family_len() * Da)`` stack of :meth:`_build_padded_action_subsequence`.

        Reproduces, for every query ``h in 1..F`` at once, the exact branch structure of the
        per-step builder (read that docstring for the slot semantics):

        - leading non-forecast window steps (``j = h - lead < 1``, output-window axis only):
          history echo ``padded[:h] = a_{t-H+1..t-H+h}``;
        - no plan on a plan-conditioned / look-ahead instance: slot 0 ``= a_t``, rest zero;
        - no plan on a plan-free instance: history echo ``padded[:h]``;
        - plan supplied: ``padded[:j] = a_{t..t+j-1}`` (the internal driving sequence).

        Built with a single ``copy_`` + ``masked_fill_`` per query group (no per-query launches,
        no transient copies beyond the result itself), so it is ``torch.equal`` to the per-step
        stack -- the values are copied, never recomputed.
        """
        if self.singlestep_act_len == 0:
            return None

        act_len = int(self.singlestep_act_len)
        act_feature = self.extract_act_features_from_multistep_composed_array(
            self._resolve_history_window(history_feat),
            is_model_output=False,
            vectorized=True,
        )
        act_steps = act_feature.transpose(-2, -1)  # (..., B, H, Da)
        lead_shape = tuple(act_steps.shape[:-2])
        slots = int(self._family_len())
        queries = int(family_len)
        device = act_steps.device

        lookahead_axis = int(self._window_lead()) > 0
        window_lead = (
            0
            if lookahead_axis
            else int(self.output_window_len) - int(self.horizon_len)
        )
        # Queries ``h <= window_lead`` are non-forecast window steps (history echo, Case A);
        # they are the LEADING queries, so the two groups are contiguous slices of the F axis.
        n_lead = min(max(window_lead, 0), queries)

        padded = act_steps.new_zeros((*lead_shape, queries, slots, act_len))
        s_idx = torch.arange(slots, device=device)

        def _echo_source() -> torch.Tensor:
            # slot ``s`` <- ``a_{t-H+1+s}`` (zero beyond the ``H`` history steps).
            echo = act_steps.new_zeros((*lead_shape, slots, act_len))
            n_echo = min(slots, int(act_steps.shape[-2]))
            echo[..., :n_echo, :] = act_steps[..., :n_echo, :]
            return echo.unsqueeze(-3)  # (..., B, 1, S, Da)

        def _fill_group(view: torch.Tensor, source: torch.Tensor, keep_below: torch.Tensor) -> None:
            # ``view``: (..., B, G, S, Da) slice of ``padded``; ``source`` broadcasts over it;
            # ``keep_below``: (G,) per-query slot count -- slots ``s >= keep_below[g]`` are zero.
            view.copy_(source.expand_as(view))
            zero_mask = s_idx.unsqueeze(0) >= keep_below.unsqueeze(1)  # (G, S)
            view.masked_fill_(zero_mask.reshape(*([1] * len(lead_shape)), -1, slots, 1), 0.0)

        h_idx = torch.arange(1, queries + 1, device=device)

        if n_lead > 0:
            # Case A -- history echo of the first ``h`` window actions.
            _fill_group(padded[..., :n_lead, :, :], _echo_source(), h_idx[:n_lead])

        if n_lead < queries:
            group = padded[..., n_lead:, :, :]
            if future_actions is None:
                if lookahead_axis or self._plan_conditioned_layout_active():
                    # Case B -- slot 0 is ``a_t`` (the last input-history action), rest zero.
                    group[..., 0, :] = act_steps[..., -1, :].unsqueeze(-2).expand(
                        *lead_shape, queries - n_lead, act_len
                    )
                else:
                    # Case C -- plan-free baseline / ``F == 1``: the history echo IS the layout.
                    _fill_group(group, _echo_source(), h_idx[n_lead:])
            else:
                # Case D -- the ``j`` driving actions ``a_{t..t+j-1}`` of forecast step ``j``.
                drive = act_steps.new_zeros((*lead_shape, slots, act_len))
                n_drive = min(slots, int(future_actions.shape[-2]))
                drive[..., :n_drive, :] = future_actions[..., :n_drive, :].expand(
                    *lead_shape, n_drive, act_len
                )
                j_idx = h_idx[n_lead:] - window_lead
                _fill_group(group, drive.unsqueeze(-3), j_idx)

        return padded.flatten(start_dim=-2)

    def _predict_at_horizon(
        self,
        history_feat: torch.Tensor,
        h: int,
        only_elite: bool = False,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict the single-step observation at horizon index ``h`` (1-based).

        Builds the per-horizon input ``[history_feat, enc(h), extra(h)]`` and runs the per-horizon
        ``ExponentialFamilyMLP`` head. The state input is *always* the observed history features
        (never a predicted state), enforcing the anti-compounding invariant of both papers.
        ``future_actions``, when supplied, is the internal driving sequence ``a_{t..t+F-1}``
        (``[a_t] ++ plan``) and is forwarded to the conditioning hook.
        """
        x_h = self._per_horizon_input(history_feat, h, future_actions=future_actions)
        if self._per_horizon_heads_active() and h > 1:
            # Literal ``{T_h}`` variant (RLRP-729): route this query through its OWN map. The
            # horizon-index encoding is still concatenated -- it is a constant per map and hence
            # inert up to the first layer's bias, which keeps the two variants' input contract
            # (and every conditioning test) identical.
            with self._bound_per_horizon_head(h):
                return ExponentialFamilyMLP._default_forward(
                    self, x_h, only_elite=only_elite
                )
        return ExponentialFamilyMLP._default_forward(self, x_h, only_elite=only_elite)

    # ==== Horizon-index encoding + conditioning hooks ============================================
    def _encode_horizon_index(
        self, history_feat: torch.Tensor, h: int
    ) -> torch.Tensor:
        """Encode the (1-based) horizon index ``h`` as the scalar ``h / _family_len()``.

        Q-C (plan section 9) adopted the scalar ``h / H`` encoding as the configurable default;
        the normaliser is the **family size**, which is ``H`` on the default output-window axis
        (TBM) and the look-ahead depth ``F`` for M3 (RLRP-821), so the encoding always spans
        ``1/|family| .. 1``.
        Returns a ``(..., 1)`` tensor broadcastable against ``history_feat``.
        """
        value = float(h) / float(self._family_len())
        return history_feat.new_full((*history_feat.shape[:-1], 1), value)

    def _history_features(self, history_flat: torch.Tensor) -> torch.Tensor:
        """Return the state/history conditioning features (default: the whole flat history).

        Q-G (plan section 9) adopted the *full history* as the "initial state" surrogate.
        """
        return history_flat

    def _history_feature_size(self) -> int:
        """Size of the vector returned by :meth:`_history_features` (default ``in_size``)."""
        return self.in_size

    def _per_horizon_extra_conditioning(
        self,
        history_feat: torch.Tensor,
        h: int,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Optional extra per-horizon conditioning appended to ``[history, enc(h)]``.

        Default: none. M3 and TBM override this to append the padded action sub-sequence ``a_{1:h}``.
        ``future_actions``, when supplied, is the internal driving sequence ``a_{t..t+F-1}``
        (``[a_t] ++ plan``) and lets the override build that sub-sequence from the planned actions
        instead of the history actions (stage ``A4`` of the Fix the MS->MS future-action-plan
        conditioning contract ``.junie`` plan,
        ``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).
        """
        return None

    def _per_horizon_extra_size(self) -> int:
        """Size of the vector returned by :meth:`_per_horizon_extra_conditioning` (default 0)."""
        return 0

    def _build_padded_action_subsequence(
        self,
        history_feat: torch.Tensor,
        h: int,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Build the padded action sub-sequence ``a^h`` shared by M3 and TBM (RLRP-728).

        The action steps that drive the prediction at index ``h`` are placed at the front of a
        fixed-length ``_family_len() * singlestep_act_len`` vector, the remaining slots
        zero-padded (Q-E; the padded layout is ours -- the paper's body prescribes no
        action-sequence encoding).

        - ``future_actions`` supplied: the sub-sequence is built from the **internal driving
          sequence** ``a_{t..t+F-1}`` (``[a_t] ++ plan``), so **slot 0 is ``a_t``** -- the action
          that actually drives ``ô_{t+1}``.
        - ``future_actions is None`` on a **plan-conditioned** instance
          (:meth:`AbstractMS2MSForecast._plan_conditioned_layout_active`): the SAME driving
          layout with the unknown future blanked -- slot 0 is ``a_t`` (read from the input
          history), slots ``1..j-1`` are zero. A missing plan is an *information removal*, not a
          *layout switch*; see the in-body note and the review entry below.
        - ``future_actions is None`` on a **plan-FREE** instance
          (``enable_future_action_plan_conditioning=False``) or with ``horizon_len == 1``, on the
          **output-window** axis: the first ``h`` **input-history** action steps
          ``a_{t-H+1..t-H+h}``, reproducing the historical behaviour byte-for-byte. That echo IS
          what those instances are trained on, so it is the correct -- and bit-exact -- layout for
          them. On the **look-ahead** axis (M3) the blank-the-unknown-future layout above is used
          unconditionally instead: slot ``j - 1`` means ``a_{t+j-1}`` there by construction, and
          only ``a_t`` is ever known from the history, so an oldest-actions echo would be a pure
          layout violation rather than a training-faithful fallback.

        **Causal identity guaranteed by the fallback.** ``ô_{t+1}`` is driven by ``a_t`` alone,
        and ``a_t`` always reaches the model through the input history, so forecast step ``j = 1``
        MUST be identical with and without a plan. The pre-fix echo fallback broke that identity
        (it did not even place ``a_t`` in the channel), which is what corrupted the single-step
        deploy head. Pinned by the conditioning test suite.

        Permanent contract change. Introduced by stage ``A4`` of the Fix the MS->MS
        future-action-plan conditioning contract ``.junie`` plan
        (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).

        **The fix (plan sections 2.2 / 4.4).** The pre-fix code wrote
        ``padded[..., :h, :] = act_steps[..., :h, :]`` with ``act_steps == a_{t-H+1..t}``, so slot
        ``0`` was ``a_{t-H+1}`` -- the *oldest* window action -- while this docstring claimed it was
        ``a_t``; it then overwrote the **leading** slots ``1..h-1`` from a plan tensor that
        ``_expand_future_actions_to_output_axis`` had filled **trailing**-first. For ``F != H`` the
        two conventions disagreed and "plan conditioning" silently degraded to a history echo
        (identity, hence unnoticed, when ``F == H``).

        **Index-to-forecast-step mapping (axis-aware, RLRP-821).**

        - *Output-window axis* (:meth:`_window_lead` ``== 0``, the default -- **TBM**): the encoder
          loop runs over the ``H``-step composed output window ``o_{t+F-H+1..t+F}``, of which only
          the trailing ``F`` steps are the forecast, so window step ``h`` (1-based) is forecast
          step ``j = h - (H - F)``:

          - ``j >= 1``: the prediction is ``ô_{t+j}``, driven by ``a_{t..t+j-1}`` ==
            ``future_actions[..., :j, :]``.
          - ``j < 1``: the window step is a re-prediction of an already-observed timestep
            (``h <= H - F``); **no planned action applies to it**, so it keeps the input-history
            echo (the RLRP-760 S0 b′ property). This branch is reachable **only** on this axis.

          With ``F == H`` the lead is 0 and ``j == h``, so the sub-sequence is ``a_{t..t+h-1}``.
          ⚠️ Slot 0 being ``a_t`` is an **intended** numerical change w.r.t. the pre-RLRP-781
          behaviour (slot 0 was ``a_{t-H+1}``), NOT a regression.

        - *Look-ahead axis* (:meth:`_window_lead` ``> 0`` -- **M3**, RLRP-821): the family has one
          map per forecast step, so the incoming index **is** the forecast step (``j == h``, no
          remap) and the padded vector is ``F * Da`` wide. The leading ``H - F`` window steps are
          never queried at all (they are the observed-history echo of
          :meth:`_echo_history_obs_steps`), so the ``j < 1`` branch above cannot be reached.

        :param future_actions: the internal driving sequence ``a_{t..t+F-1}`` of shape
            ``(..., horizon_len, singlestep_act_len)`` built by
            ``AbstractMS2MSForecast._build_driving_action_sequence`` (slot ``j`` == ``a_{t+j}``);
            ``None`` keeps the pure history echo.
        """
        if self.singlestep_act_len == 0:
            return None

        # (..., act_len, history_len) -> (..., history_len, act_len). Read from the RAW composed
        # history window: ``history_feat`` is no longer guaranteed to be it (M3 conditions on
        # ``o_t`` only since RLRP-821).
        act_feature = self.extract_act_features_from_multistep_composed_array(
            self._resolve_history_window(history_feat),
            is_model_output=False,
            vectorized=True,
        )
        act_steps = act_feature.transpose(-2, -1)

        # PERF (deferred): readability-first per-horizon re-extraction + masked copy; see plan
        # section 11 / RLRP-730. TODO(perf): precompute the action steps once per forward and
        # slice/mask over h to avoid re-extracting them at every horizon query.
        lookahead_axis = int(self._window_lead()) > 0
        padded = act_steps.new_zeros(
            (*act_steps.shape[:-2], int(self._family_len()), self.singlestep_act_len)
        )

        if lookahead_axis:
            # Look-ahead axis (M3): the index IS the forecast step -- no window remap.
            forecast_step = h
        else:
            # RLRP-824: the window lead is ``W - F`` with ``W = output_window_len`` (``H - F``
            # for every ``F <= H`` config, bit-exact; ``0`` on the asymmetric window).
            forecast_step = h - (int(self.output_window_len) - int(self.horizon_len))
            if forecast_step < 1:
                # Non-forecast leading window steps: history echo (identical on both paths).
                # Output-window axis only; unreachable on the look-ahead axis, where those steps
                # are never queried.
                padded[..., :h, :] = act_steps[..., :h, :]
                return padded.flatten(start_dim=-2)

        if future_actions is None:
            if lookahead_axis or self._plan_conditioned_layout_active():
                # Permanent: a plan-conditioned model asked to forecast WITHOUT a plan must keep
                # the layout it was TRAINED on and blank only the genuinely unknown future --
                # slot 0 stays ``a_t`` (known: it is the LAST input-history action slot), slots
                # ``1..j-1`` stay zero. Falling back to the history echo here (the pre-fix
                # behaviour) switched the *layout* instead of removing *information*:
                #
                #   * it put ``a_{t-H+1..t-H+h}`` -- stale PAST actions, not even containing
                #     ``a_t`` -- into the slots that carry ``a_{t..t+j-1}`` at training time;
                #   * it filled ``h`` slots instead of ``j``, activating channel slots that are
                #     identically zero throughout training whenever ``F < H`` (an OOD input);
                #   * it broke the causal identity below, even though ``ô_{t+1}`` is driven by
                #     ``a_t`` ALONE and therefore needs no plan at all.
                #
                # That corrupted the single-step deploy head (``predict_next_state``), which is
                # the path behind the reported ``best_pred_mae_score``, and made the
                # ``obs_only`` mode a train/test layout mismatch rather than the deprivation
                # ablation it is captioned as. Introduced by the MS->MS conditioning-fallback
                # review of the M3 / TBM baselines (2026-09-11); mirrors the graceful
                # zero-action degradation of the E2E-TCN future block
                # (``MS2MSEndToEndTCN._compose_history_and_future_blocks``).
                padded[..., :1, :] = act_steps[..., -1:, :]
                return padded.flatten(start_dim=-2)
            # Plan-FREE baseline / ``F == 1``: the history echo IS the training layout.
            padded[..., :h, :] = act_steps[..., :h, :]
            return padded.flatten(start_dim=-2)

        # Plan-conditioned path: the ``j`` driving actions ``a_{t..t+j-1}`` of forecast step ``j``,
        # slot 0 == ``a_t``. ``j <= F <= H`` so the slice always fits the padded vector.
        driving = future_actions[..., :forecast_step, :]
        padded[..., :forecast_step, :] = driving.expand(
            (*padded.shape[:-2], forecast_step, self.singlestep_act_len)
        )
        return padded.flatten(start_dim=-2)

    def _per_horizon_in_size(self) -> int:
        """Input size of the per-horizon network: history features + 1 (scalar h) + extra.

        RLRP-736 bespoke-forward plan §3.5: when the by-construction rotation rep is
        active the base ``_default_forward`` encodes the attitude input slot(s) of the
        history window (``_apply_orientation_input_encoding`` over ``_ori_in_slots``),
        widening the per-``h`` input by exactly ``_trunk_in_size - in_size`` (0 when
        OFF => bit-exact). The action/horizon conditioning columns carry no attitude
        and are unchanged.
        """
        input_encode_extra = self._trunk_in_size - self.in_size
        return (
            self._history_feature_size()
            + input_encode_extra
            + 1
            + self._per_horizon_extra_size()
        )

    def _apply_orientation_output_decoding(self, mean: torch.Tensor) -> torch.Tensor:
        """Decode a per-horizon SINGLE-STEP obs prediction to a unit quaternion.

        RLRP-736 bespoke-forward plan §3.5 (C.3). The per-``h`` head emits ONE
        single-step obs block (width ``_ss_trunk_out_len``), NOT the composed
        multi-step window, so this family overrides the base composed decode to the
        single-step :meth:`_apply_ss_orientation_output_decoding` (``_ss_ori_out_slots``).
        Called by the base ``ExponentialFamilyMLP._default_forward`` on each per-``h``
        prediction; the per-step decoded quaternions are then stacked and composed
        into the multistep forecast. No-op / bit-exact when the by-construction
        rotation rep is OFF.
        """
        return self._apply_ss_orientation_output_decoding(mean)

    # ==== Masked objective (RLRP-821) ============================================================
    #
    # ⚠️ WHY THIS SEAM LIVES HERE AND NOT ON THE SHARED BASE (hard operator constraint, FR9).
    # ``AbstractMS2MSForecast`` and every one of its parents (``ExponentialFamilyMLP``,
    # ``MultiStepMLP``, the mbrl ``Model`` chain) are OFF-LIMITS: ``MS2MSEndToEndTCN`` (E2E-TCN)
    # inherits ``AbstractMS2MSForecast`` **directly**, as do the MTM-Pro / dual-head / AR
    # families, and their running experiments must stay bit-exact. The per-step objective mask is
    # therefore obtained by OVERRIDING the inherited loss methods on THIS class -- which only M3
    # and TBM inherit -- instead of adding a hook to the shared base. Anything placed here is
    # structurally invisible to E2E-TCN. See ``rlrp-821-m3-paper-faithful-plan-20260911.md``.
    def _forecast_step_loss_mask(self) -> Optional[torch.Tensor]:
        """Per-composed-window-step objective weights, or ``None`` for the uniform objective.

        **Default ``None``** => the inherited loss is called verbatim, so TBM (and any
        ``F == H`` instance) is byte-for-byte unchanged.

        M3 (RLRP-821) returns a ``(history_len,)`` tensor of ``0``/``1`` zeroing the leading
        ``H - F`` window steps: those are NOT forecasts (they are the observed-history echo of
        :meth:`_echo_history_obs_steps`), and supervising them with weight ``1.0`` diluted the
        forecast objective by ``(H - F) / H``.

        :return: ``(history_len,)`` per-step weights, or ``None``.
        """
        return None

    def _broadcast_step_mask_to_out_layout(self, mask: torch.Tensor) -> torch.Tensor:
        """Expand a ``(history_len,)`` per-step mask over the composed flat ``out_size`` layout.

        Built by pushing the mask through :meth:`AbstractMS2MSForecast._compose_flat_forecast`,
        i.e. through the very composition that produced the prediction, so the result is
        **layout-agnostic by construction** (no duplicated index arithmetic).

        The **action columns keep weight 1**: they are the input-history / plan echo, not a
        forecast, and re-weighting them is explicitly out of scope (the action-column echo is a
        documented I/O departure, orthogonal to the fidelity of ``{T_h}``).

        :param mask: ``(history_len,)`` per-step weights (dtype/device of the loss tensor).
        :return: ``(out_size,)`` per-element weights.
        """
        # RLRP-824: the per-step mask spans the composed output window ``W`` (``== history_len``
        # for every ``F <= H`` config).
        window_len = int(self.output_window_len)
        obs_mask = mask.reshape(window_len, 1).expand(
            window_len, self.singlestep_obs_len
        )
        act_mask = mask.new_ones((window_len, self.singlestep_act_len))
        return self._compose_flat_forecast(obs_mask, act_mask)

    def _resolve_out_layout_loss_weights(
        self, reference: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """The ``(out_size,)`` objective weights for ``reference``'s dtype/device, or ``None``."""
        mask = self._forecast_step_loss_mask()
        if mask is None:
            return None
        mask = torch.as_tensor(
            mask, dtype=reference.dtype, device=reference.device
        )
        return self._broadcast_step_mask_to_out_layout(mask)

    @torch.compiler.disable
    def _deterministic_loss(
        self, model_in: torch.Tensor, target: torch.Tensor, reduce=True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Inherited deterministic loss, restricted to the genuine forecast steps (RLRP-821).

        ``_forecast_step_loss_mask() is None`` (TBM, and M3 with ``F == H``) => a **verbatim**
        ``super()`` call, bit-exact.

        Mask active => ``super()`` is run UNREDUCED, the per-element losses are multiplied by the
        broadcast step mask, and the inherited
        :func:`~tools.multistep_tools.models.utils.reduce_deterministic_compose_loss` is applied.

        **No ``H/F`` rescale on this path (deviation from the plan's KD4, deliberate).** The
        deterministic reducer SUMS the feature axis, so dropping the leading ``H - F`` obs columns
        already leaves every *surviving* forecast step at weight ``1.0`` -- exactly its weight in
        an ``F == H`` run. Multiplying by ``H/F`` on a sum-reduction would instead inflate each
        forecast step's weight by ``H/F``, i.e. change the gradient scale rather than preserve it.
        The ``H/F``-style mask normalisation belongs to the MEAN-reduced probabilistic path below.

        ⚠️ Any composite term the inherited loss adds BEFORE the reduction (the feature-geometry
        SS/MS channels) is masked along with the data term when a mask is active -- which is the
        intent: its leading-step contribution must not re-introduce the non-forecast steps. Both
        are bit-neutral OFF (no handler / weight 0).
        """
        if self._forecast_step_loss_mask() is None:
            return super()._deterministic_loss(model_in, target, reduce=reduce)

        losses, meta = super()._deterministic_loss(model_in, target, reduce=False)
        losses = losses * self._resolve_out_layout_loss_weights(losses)
        if reduce:
            losses = reduce_deterministic_compose_loss(losses)
        return losses, meta

    @torch.compiler.disable
    def _probabilistic_loss(
        self, model_in: torch.Tensor, target: torch.Tensor, reduce: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Inherited probabilistic loss, restricted to the genuine forecast steps (RLRP-821).

        Symmetric to :meth:`_deterministic_loss`; ``None`` mask => verbatim ``super()`` call
        (TBM, deterministic **and** probabilistic, stays bit-exact).

        Mask active => the per-element NLL is masked and then **renormalised by the active
        fraction** (``mask.numel() / mask.sum()``) because the probabilistic reducer MEANS the
        feature axis: without it the zeroed columns would drag the per-dimension NLL down and the
        fixed-coefficient companion terms (logvar bound, composite auto-weighting) would silently
        gain relative weight.

        In practice a **formality**: the only masked family is M3, which hard-fixes
        ``deterministic=True``, and TBM never returns a mask. Provided so a future masked
        probabilistic subclass cannot silently fall back to the unmasked objective.
        """
        if self._forecast_step_loss_mask() is None:
            return super()._probabilistic_loss(model_in, target, reduce=reduce)

        losses, meta = super()._probabilistic_loss(model_in, target, reduce=False)
        weights = self._resolve_out_layout_loss_weights(losses)
        weights = weights * (float(weights.numel()) / weights.sum())
        losses = losses * weights
        if reduce:
            losses = reduce_probabilistic_compose_loss(losses)
        return losses, meta

    # NOTE -- ``eval_score`` is deliberately NOT overridden (deviation from the plan's KD4,
    # validated against the code). For this family the inherited
    # ``MultiStepMLP.eval_score`` scores the SINGLE-STEP deploy point only: it runs
    # ``self.deploy(model_in)`` and compares against
    # ``multistep_to_singlestep_next_obs_adapter(target)``, i.e. a ``(E, B, singlestep_obs_len)``
    # tensor holding the ``t = 1`` forecast ``ô_{t+1}`` -- a GENUINE forecast step on both family
    # axes, never an echoed leading window step. There is consequently nothing to mask there, and
    # applying an ``(H,)``/``out_size`` mask to that single-step tensor would be a shape/semantic
    # error. The ``A3.5`` / ``R9`` train-vs-validation conditioning symmetry is untouched (the
    # plan binding stays in ``AbstractMS2MSForecast.eval_score``).

    # ==== Abstract surface (re-declared for clarity) =============================================
    @abstractmethod
    def _model_family_tag(self) -> str:
        """Short subclass identifier (used only for messages / introspection)."""
        raise NotImplementedError
