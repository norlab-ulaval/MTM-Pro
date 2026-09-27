# coding=utf-8
"""
Shared base for the multistep-history-in -> multistep-forecast-out (``ms2ms``) baselines.

Permanent production module. Introduced by section 2.1 of the MS->MS forecast baselines
(End-to-End-TCN / M3 / TBM) ``.junie`` plan
(``feature_ms2ms_forecast_baselines_e2etcn_m3_tbm_plan_RLRP-692_693_694_20260703.md``).
Common home for the three non-auto-regressive forecast baselines (RLRP-692 / RLRP-693 /
RLRP-694): it keeps ``MultiStepMLP``'s proven flat multistep-composed I/O contract while
letting each subclass provide only its own encoder architecture.
"""
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union

import omegaconf
import torch

from tools.multistep_tools.models.base_multistep_mlp import MultiStepMLP
from tools.multistep_tools.multistep_model_util import (
    revert_timestep_first_multistep_dim_unflaten_array,
)


class AbstractMS2MSForecast(MultiStepMLP):
    """Abstract shared base for the ``ms2ms`` (multistep history -> multistep forecast) family.

    The three baselines (End-to-End-TCN, M3, TBM) share the property (1)+(2) family definition
    (multistep history in, multistep forecast out) but are *not* auto-regressive (no self-feed of
    predictions during the forward). This base therefore delegates the whole forecast computation
    to a single :meth:`forecast` public method (backed by the subclass :meth:`_forward_encoder`)
    and reuses the inherited ``MultiStepMLP`` loss / deploy / eval_score plumbing unchanged so the
    three baselines stay directly comparable to the rest of the family on the same metric.

    Fidelity-first directive (plan section 0.3): each subclass reproduces its reference paper's
    internal computation faithfully; the *only* permitted departure is adapting the model I/O to
    the RLRC flat multistep-composed contract. This base owns that single I/O departure: the
    encoder predicts the per-step observations and the (known/planned) action columns are echoed
    from the input history into the composed-forecast action slots.
    """

    # RLRP-736 bespoke-forward plan §3.5 (C.3). Whether this concrete forecast
    # family has wired the by-construction rotation head into its per-step obs
    # prediction. Default False: :meth:`forecast` fails loud on an active
    # non-quaternion rep (the per-step obs head would otherwise emit a
    # non-unit-quaternion / mismatched-width attitude slot). The horizon-indexed
    # family (M3 / TBM) sets it True (see
    # ``AbstractHorizonIndexedMS2MSForecast``); the one-shot E2E-TCN encoder stays
    # False pending its per-child follow-up.
    _supports_by_construction_orientation: bool = False

    # Permanent model-capability flag. Introduced by stage ``A3`` of the Fix the
    # MS->MS future-action-plan conditioning contract ``.junie`` plan
    # (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).
    # ``True`` for this family ONLY: it tells ``OneDTransitionRewardModelV2.loss`` /
    # ``update`` / ``eval_score`` to request the opt-in ``action_plan`` third return
    # from ``_process_batch`` and thread it here. Every other model family leaves it
    # absent/False, so their ``_process_batch`` stays a 2-tuple and their whole train
    # path stays bit-exact (decision ``D7``, risk ``R5``).
    #
    # Family DEFAULT only: ``__init__`` shadows it with a per-instance value driven by
    # the ``enable_future_action_plan_conditioning`` config knob (operator follow-up to
    # the post-implementation review, 2026-09-11), so a genuinely plan-free MS->MS
    # baseline can be CONFIGURED instead of being faked by depriving a plan-trained
    # model of its plan at test time.
    consumes_future_action_plan: bool = True

    # Train/eval-time plan context (see :meth:`_active_future_action_plan`). ``None``
    # => the plan-agnostic (bit-exact) behaviour.
    _active_action_plan: Optional[torch.Tensor] = None

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
        ensemble_size: int = 1,
        hid_size: int = 200,
        deterministic: bool = False,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        logvar_bound_grad_clip: Optional[float] = None,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        distribution_name: str = "gaussian",
        mae_loss: bool = True,
        model_use_double_precision: bool = False,
        dropout: float = 0.0,
        ms_head_dropout: float = 0.0,
        ms_head_num_layers: int = 1,
        residual_form: str = "identity_v2",
        layer_bloc=None,  # RLRP-768 per-region layer-bloc selector (forwarded)
        enable_auto_loss_weighting: bool = False,
        auto_weighting_noise_model: str = "gaussian",
        auto_weighting_scheme: str = "tempered_likelihood",
        feature_weight_mode: str = "tempered",
        feature_weight_max_ratio: Optional[float] = None,
        description: Optional[str] = None,
        feature_geometry=None,
        internal_orientation=None,
        orientation_singlestep_slots=None,
        enable_future_action_plan_conditioning: bool = True,
        # RLRP-824 Step 7 (FR9): opt-in ``torch.autocast`` around the Lightning ``training_step`` /
        # ``validation_step`` (read by the mbrl ``Model`` base through ``resolve_autocast_dtype``).
        # ``None`` (default) = the exact fp32 / fp64 path (bit-exact); ``"bf16"`` is the documented
        # ULH opt-in for E2E-TCN (profiling gate: 2.5-3.5x on every dataset row); ``"fp16"`` runs
        # WITHOUT a GradScaler (CUDA only, at the operator's risk).
        mixed_precision: Optional[str] = None,
    ):
        self.mixed_precision = self._coerce_mixed_precision(mixed_precision)
        # These multistep attributes are read by ``_build_network`` (and the horizon-indexed
        # size helpers), which the base ``ExponentialFamilyMLP.__init__`` calls *before*
        # ``MultiStepMLP.__init__`` assigns them; set them up-front (mirrors ``tcn_ms2ss``).
        self.singlestep_obs_len = singlestep_obs_len
        self.singlestep_act_len = singlestep_act_len
        self.history_len = history_len
        self.horizon_len = self._coerce_horizon_len(history_len, horizon_len)

        # RLRP-824 FR14 (operator instruction 2026-09-15): the asymmetric ``F > H`` output window
        # is supported for the future-action-plan conditioned realisation ONLY. When ``W = F > H``
        # the ``W-1`` composed action columns ARE the plan ``a_{t+1..t+F-1}`` -- there is no
        # history lead to echo -- so a plan-FREE model would have no training-faithful layout for
        # them. The plan-free / ``last_action_hold`` / ``obs_only`` asymmetric cases are deferred
        # to a follow-up task (see ``ultra_long_horizon_dataloader_plan_RLRP-824_20260915.md``,
        # *Deferred follow-ups*). Raised BEFORE the network is built so the failure is immediate.
        if int(self.horizon_len) > int(history_len) and not bool(
            enable_future_action_plan_conditioning
        ):
            raise ValueError(
                f"{type(self).__name__}: horizon_len={self.horizon_len} > history_len="
                f"{history_len} (asymmetric ultra-long-horizon output window, RLRP-824) is "
                "supported ONLY with enable_future_action_plan_conditioning=True: the composed "
                "forecast action columns are the planned actions a_{t+1..t+F-1} and there is no "
                "history echo to fall back on. The plan-free asymmetric case is deferred to a "
                "follow-up task (RLRP-824 deferred follow-ups: non-plan action conditioning on "
                "F > H)."
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
            ensemble_size=ensemble_size,
            hid_size=hid_size,
            deterministic=deterministic,
            propagation_method=propagation_method,
            learn_logvar_bounds=learn_logvar_bounds,
            logvar_bound_grad_clip=logvar_bound_grad_clip,
            activation_fn_cfg=activation_fn_cfg,
            distribution_name=distribution_name,
            mae_loss=mae_loss,
            model_use_double_precision=model_use_double_precision,
            dropout=dropout,
            ms_head_dropout=ms_head_dropout,
            ms_head_num_layers=ms_head_num_layers,
            residual_form=residual_form,
            layer_bloc=layer_bloc,
            enable_auto_loss_weighting=enable_auto_loss_weighting,
            auto_weighting_noise_model=auto_weighting_noise_model,
            auto_weighting_scheme=auto_weighting_scheme,
            # RLRP-761 S3: this family is NOT feature-weighted (it does not even
            # accept ``temporal_weights``); the knobs are forwarded to the base
            # purely so a shared `ms_model` node stays instantiable, and the base
            # WARNS on a non-default value rather than silently ignoring it.
            feature_weight_mode=feature_weight_mode,
            feature_weight_max_ratio=feature_weight_max_ratio,
            description=description,
            feature_geometry=feature_geometry,
            # RLRP-736 bespoke-forward plan §3.5 (C.3): thread the by-construction
            # rotation-rep params to the base so the horizon-indexed family (M3 /
            # TBM) can be activated from Hydra. Defaults (quaternion / None) keep
            # every existing forecast config bit-exact.
            internal_orientation=internal_orientation,
            orientation_singlestep_slots=orientation_singlestep_slots,
        )

        # Permanent config-level conditioning switch. Introduced by the operator follow-up to
        # the post-implementation review of the Fix the MS->MS future-action-plan conditioning
        # contract ``.junie`` plan
        # (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).
        #
        # Rationale (review finding, 2026-09-11): with the capability declared as a hard class
        # constant, EVERY MS->MS model trained plan-conditioned, so a test-time ``obs_only`` run
        # was no longer a plan-free BASELINE but a *deprivation ablation* (a plan-trained model
        # run without its plan == a train/test mismatch). Both are legitimate experiments, but
        # they answer different questions and must not be captioned interchangeably. Turning this
        # knob OFF yields the true plan-free baseline: the plan is neither requested at train time
        # (``OneDTransitionRewardModelV2._action_plan_needed`` reads this very attribute) nor
        # accepted at test time (:meth:`forecast` fails loud rather than silently conditioning).
        self.consumes_future_action_plan = bool(enable_future_action_plan_conditioning)

    # ==== Mixed precision (RLRP-824 FR9) =========================================================
    _MIXED_PRECISION_ALIASES = {
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
    }

    @classmethod
    def _coerce_mixed_precision(cls, value) -> Optional[str]:
        """Normalise the ``mixed_precision`` knob to ``None | "bf16" | "fp16"`` (fail fast)."""
        if value is None or value is False or (isinstance(value, str) and value.lower() in ("", "none", "null", "false", "fp32", "float32")):
            return None
        key = str(value).lower()
        if key not in cls._MIXED_PRECISION_ALIASES:
            raise ValueError(
                f"{cls.__name__}: unsupported mixed_precision={value!r}; expected null | bf16 | fp16."
            )
        return cls._MIXED_PRECISION_ALIASES[key]

    # ==== Output window (RLRP-824) ===============================================================
    @classmethod
    def resolve_output_window_len(cls, history_len: int, horizon_len: int) -> int:
        """``W = max(H, F)``: the MS->MS forecast family emits a ``W``-block composed window.

        Permanent contract change. Introduced by the Ultra-long-horizon MS->MS training via a
        lazy window DataLoader ``.junie`` plan (RLRP-824,
        ``ultra_long_horizon_dataloader_plan_RLRP-824_20260915.md``, KD1 / FR2).

        - ``F <= H`` => ``W == H``: the legacy layout ``obs*H + act*(H-1)`` whose trailing ``F``
          blocks are the forecast (byte-for-byte unchanged).
        - ``F > H`` => ``W == F``: the window IS the forecast ``ô_{t+1..t+F}`` and its ``F-1``
          action columns are the plan ``a_{t+1..t+F-1}`` (plan-conditioned only, FR14).

        Rationale: E2E-TCN already emits ``H + F >= W`` causal per-step outputs and M3 already
        indexes its family by ``F``, so the asymmetric window is a slice/size change, not a new
        head. A fractional ``horizon_len`` (ratio of ``history_len``) is coerced exactly like
        ``MultiStepMLP.__init__`` does.
        """
        return max(
            int(history_len), int(cls._coerce_horizon_len(history_len, horizon_len))
        )

    @property
    def asymmetric_output_window(self) -> bool:
        """``True`` iff ``horizon_len > history_len`` (``W == F > H``; RLRP-824)."""
        return int(self.horizon_len) > int(self.history_len)

    # ==== Abstract subclass surface ==============================================================
    @abstractmethod
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
        """Build the subclass-specific network (TCN body vs. horizon-indexed MLP).

        Called by ``ExponentialFamilyMLP.__init__`` with the *model-level* ``in_size`` / ``out_size``.
        Subclasses are free to ignore those and wire their own internal sizes (e.g. a per-horizon
        MLP that maps ``[history, h, ...] -> singlestep_obs_len``), mirroring how ``tcn_ms2ss``
        builds its encoder then defers the head to ``super()._build_network``.
        """
        raise NotImplementedError

    @abstractmethod
    def _forward_encoder(
        self,
        history_flat: torch.Tensor,
        only_elite: bool = False,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode the flat history into per-step observation predictions.

        :param history_flat: flat multistep-composed history of shape ``(..., in_size)``.
        :param only_elite: forward only through the elite ensemble members (deploy path).
        :param future_actions: optional **internal driving action sequence**
            ``a_{t..t+F-1}`` = ``[a_t from the history] ++ plan`` of shape
            ``(..., horizon_len, singlestep_act_len)`` with **slot 0 == ``a_t``** (the action that
            drives ``ô_{t+1}``). Built by :meth:`_build_driving_action_sequence` from the public
            ``F-1``-long plan ``a_{t+1..t+F-1}``.

            Permanent contract change. Introduced by stage ``A1`` of the Fix the MS->MS
            future-action-plan conditioning contract ``.junie`` plan
            (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).
            This replaces the former ``history_len``-axis expanded layout (RLRP-728 / RLRP-760 S0),
            whose leading-vs-trailing slot convention disagreed with the per-horizon consumer and
            silently degraded plan conditioning to a history echo for ``F != H`` (plan section 2.2).
            The ``H``-axis layout is now used ONLY for the composed action-column echo
            (:meth:`_expand_future_actions_to_output_axis`).

            ``None`` (default) keeps the history-echo behaviour. Every control-conditioned encoder
            (E2E-TCN after stage ``B2`` / M3 / TBM) consumes this sequence.
        :return: a ``(obs_mean_steps, obs_logvar_steps)`` tuple where each tensor has shape
            ``(..., history_len, singlestep_obs_len)`` (``obs_logvar_steps`` is ``None`` when the
            model is deterministic).
        """
        raise NotImplementedError

    # ==== Forecast surface (shared) ==============================================================
    def forecast(
        self,
        history_flat: torch.Tensor,
        only_elite: bool = False,
        *,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return the flat multistep-composed forecast ``(F_mean, F_logvar)``.

        All three baselines expose this identical forecast surface regardless of *how* they
        compute it internally. The observation stream is predicted by :meth:`_forward_encoder`;
        the action stream is echoed from the input history (the RLRC I/O-contract departure of
        plan section 0.3 -- the reference papers predict states only).

        :param future_actions: optional, keyword-only planned future-action sequence of shape
            ``(..., horizon_len - 1, singlestep_act_len)`` -- the plan ``a_{t+1..t+F-1}``.

            Permanent contract change. Introduced by stage ``A1`` of the Fix the MS->MS
            future-action-plan conditioning contract ``.junie`` plan
            (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).
            The channel was re-axed ``F -> F-1``: the forecast ``ô_{t+1..t+F}`` is conditioned on
            the history ``(o, a)_{t-H+1..t}`` -- which **already contains** ``a_t`` -- **plus** the
            plan ``a_{t+1..t+F-1}``. ``a_{t+F}`` is dropped: no predicted step is driven by it and
            it is NOT available in a training batch (``get_compose_next_obs`` stops at
            ``a_{t+F-1}``), so keeping it would make the train/test conditioning layouts impossible
            to align -- which is exactly the RLRP-781 defect. ``F == 1`` => the plan is empty
            (``None`` / zero-length): a legitimate no-op, since all the driving information
            (``a_t``) is already in the history.

            ``None`` (default) reproduces the history-echo behaviour byte-for-byte, and a
            **zero-length** plan is canonicalized to exactly that same path (an empty plan is a
            no-op by contract, so it must not be a second, numerically divergent behaviour).
            When a non-empty plan is provided, the planned actions both (a) replace the echoed
            history action columns in the composed forecast (via
            :meth:`_expand_future_actions_to_output_axis`) and (b) are threaded to the
            control-conditioned encoders as the internal driving sequence ``[a_t] ++ plan`` (via
            :meth:`_build_driving_action_sequence`).

            **Config-level opt-out** (operator follow-up, 2026-09-11): a model instantiated with
            ``enable_future_action_plan_conditioning=False`` is the plan-FREE baseline and
            REJECTS a non-empty plan here rather than consuming it (it was never trained with
            one). A zero-length plan stays legal, being a contractual no-op.
        """
        # RLRP-736 bespoke-forward plan §3.5: the by-construction rotation head
        # decodes each per-step obs prediction to a unit quaternion. This is wired
        # for the horizon-indexed family (M3 / TBM) whose per-``h`` head routes
        # through ``ExponentialFamilyMLP._default_forward`` (see
        # ``AbstractHorizonIndexedMS2MSForecast``). Families that have NOT wired it
        # (e.g. the one-shot E2E-TCN encoder, which builds its per-step obs head at
        # the EXTERNAL single-step width) fail loud here rather than silently
        # emitting a non-unit-quaternion / mismatched-width forecast. Neutral
        # (``quaternion``) keeps the forecast byte-for-byte identical.
        if getattr(self, "_internal_orientation_rep_active", False) and not getattr(
            self, "_supports_by_construction_orientation", False
        ):
            raise NotImplementedError(
                "By-construction internal_orientation_rep "
                f"({getattr(self, '_orientation_rep', None)}) is not yet wired into "
                "this ms2ms forecast model (its per-step forecast composition needs "
                "child-specific encoder-width plumbing; RLRP-736 bespoke-forward plan "
                "§3.5 C.3 follow-up). The horizon-indexed family (M3 / TBM) IS wired; "
                "the one-shot E2E-TCN encoder is not yet. Use 'quaternion' for this "
                "model, or the dual-head / MS2SS-AR / M3 / TBM families for the "
                "by-construction rotation head."
            )

        history_flat = self._maybe_cast_to_model_dtype(history_flat)
        driving_actions = None
        echoed_actions = None
        if future_actions is not None:
            future_actions = self._validate_future_actions(future_actions)
            if future_actions.shape[-2] == 0:
                # ``F == 1`` (or any zero-length plan): the normative contract declares the empty
                # plan a **no-op** -- every action driving the single forecast step (``a_t``) is
                # already carried by the input history. Canonicalize it to the ``None`` path so
                # that "empty plan" and "no plan" are ONE behaviour instead of two numerically
                # divergent code paths (the plan-conditioned routing would otherwise re-slot the
                # driving sequence for zero informational gain). Stage ``A1`` of the Fix the
                # MS->MS future-action-plan conditioning contract ``.junie`` plan
                # (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).
                future_actions = None

        if future_actions is None and self.asymmetric_output_window:
            # RLRP-824 FR14 (b): on the asymmetric ``W = F > H`` window the composed action
            # columns are the plan and nothing else -- there is no history echo of the right
            # length to fall back on (the echo path stays reachable, and bit-exact, for
            # ``W == H`` only). Fail loud rather than emit a mis-laid-out window.
            raise ValueError(
                f"{type(self).__name__}: forecast() on an asymmetric output window "
                f"(horizon_len={self.horizon_len} > history_len={self.history_len}, RLRP-824) "
                "requires a non-empty future-action plan a_{t+1..t+F-1} of shape "
                f"(..., {self.horizon_len - 1}, {self.singlestep_act_len}); got None / a "
                "zero-length plan. Plan-free forecasting on F > H (last_action_hold / obs_only "
                "/ no plan) is deferred to a follow-up task (RLRP-824 deferred follow-ups)."
            )

        if future_actions is not None and not self.consumes_future_action_plan:
            # Permanent fail-fast guard of the config-level conditioning switch (see
            # ``enable_future_action_plan_conditioning`` in :meth:`__init__`). Placed AFTER the
            # empty-plan canonicalization on purpose: a zero-length plan is a contractual no-op
            # (``F == 1``) and stays legal for the plan-free baseline too. A model configured as
            # that baseline has never been trained to read a plan, so silently consuming one here
            # -- or silently dropping it -- would produce a figure whose caption cannot be
            # trusted, which is exactly the failure mode this switch exists to prevent.
            raise ValueError(
                "future_actions was supplied to a model configured with "
                "enable_future_action_plan_conditioning=False (the plan-FREE baseline): this "
                "model is neither trained nor wired to consume a future-action plan, so "
                "conditioning it now would measure a train/test mismatch, not a baseline. "
                "Either run this model under the 'obs_only' test-time conditioning mode, or "
                "instantiate it with enable_future_action_plan_conditioning=True (the family "
                "default) and retrain it."
            )

        if future_actions is not None:
            # Permanent routing split. Introduced by stage ``A1`` of the Fix the MS->MS
            # future-action-plan conditioning contract ``.junie`` plan
            # (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).
            # The public ``F-1`` plan feeds TWO structurally different consumers, which the
            # pre-fix single ``history_len``-axis expansion conflated (plan section 2.2):
            #   - the ENCODERS consume the internal driving sequence ``a_{t..t+F-1}``
            #     (``[a_t] ++ plan``, slot 0 == ``a_t``) on the ``F`` forecast axis;
            #   - the composed action-column ECHO consumes the ``H``-step output-window layout.
            driving_actions = self._build_driving_action_sequence(
                future_actions, history_flat
            )
            echoed_actions = self._expand_future_actions_to_output_axis(
                future_actions, history_flat
            )

        obs_mean_steps, obs_logvar_steps = self._forward_encoder(
            history_flat, only_elite=only_elite, future_actions=driving_actions
        )
        act_steps = self._resolve_action_steps(
            history_flat, obs_mean_steps, future_actions=echoed_actions
        )

        # The encoder may add a leading ensemble dim (e.g. an ensemble deploy from a 2D input
        # where ``EnsembleLinearLayer`` broadcasts) that the echoed action stream lacks; align the
        # action leading dims to the observation leading dims before composing.
        act_steps = self._broadcast_leading_dims(act_steps, obs_mean_steps)

        f_mean = self._compose_flat_forecast(obs_mean_steps, act_steps)

        if self.deterministic:
            return f_mean, None

        # Known/echoed action columns carry no learned variance: fill with logvar=0 (variance=1).
        act_logvar_steps = torch.zeros_like(act_steps)
        f_logvar = self._compose_flat_forecast(obs_logvar_steps, act_logvar_steps)
        return f_mean, f_logvar

    # ==== Train/eval-time plan threading (stage A3) ==============================================
    @contextmanager
    def _active_future_action_plan(
        self, future_actions: Optional[torch.Tensor]
    ) -> Iterator[None]:
        """Bind ``future_actions`` for the duration of a train/eval forward pass.

        Permanent train-path plumbing. Introduced by stage ``A3`` of the Fix the MS->MS
        future-action-plan conditioning contract ``.junie`` plan
        (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).

        The inherited loss chain (``loss`` -> ``_deterministic_loss`` / ``_probabilistic_loss`` ->
        ``forward`` -> :meth:`_default_forward`) is shared with every other model family and its
        signatures carry no plan argument. Rather than re-plumb that whole chain (which would
        touch the MTM-Pro / dual-head / AR families and put risk ``R5`` bit-exactness at stake),
        the plan is bound as a **forward-scoped context** that only
        :meth:`_default_forward` of THIS family reads. ``None`` restores the plan-agnostic
        behaviour, so every non-opted-in path is byte-for-byte unchanged.
        """
        previous = self._active_action_plan
        self._active_action_plan = future_actions
        try:
            yield
        finally:
            self._active_action_plan = previous

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Inherited loss, optionally conditioned on the planned future actions.

        Permanent contract change. Introduced by stage ``A3`` of the Fix the MS->MS
        future-action-plan conditioning contract ``.junie`` plan
        (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).

        ``future_actions`` is the training-batch plan ``a_{t+1..t+F-1}`` extracted by
        ``OneDTransitionRewardModelV2._process_batch(return_action_plan=True)`` (already in the
        model **input** normalization space). This closes the RLRP-781 defect: before this stage
        the conditioning slots were filled at train time with a deterministic slice of the history
        that was already present in the same input vector (a gradient-starved duplicate), while at
        test time they carried information present nowhere else.

        ``None`` (default) => byte-for-byte the pre-stage behaviour.
        """
        with self._active_future_action_plan(future_actions):
            return super().loss(model_in, target=target)

    def eval_score(  # type: ignore[override]
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Inherited validation score, conditioned on the plan **exactly like** :meth:`loss`.

        Permanent contract change. Introduced by stage ``A3`` (decision ``A3.5``) of the Fix the
        MS->MS future-action-plan conditioning contract ``.junie`` plan
        (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).

        **Decision (A3.5 / risk R9): the plan IS threaded through the validation metric.** Leaving
        ``eval_score`` plan-free while ``loss`` became plan-conditioned would measure the
        validation loss under a *different* conditioning than the training loss -- trading the
        train/test shift this plan fixes for a subtler train/validation one, which would in turn
        corrupt model selection / early stopping (the validation score is what ranks checkpoints).
        Conditioning symmetry between the training and the validation objective is therefore
        mandatory, and it is the whole point of metric ``M2``.

        ``None`` (default) => byte-for-byte the pre-stage behaviour.
        """
        with self._active_future_action_plan(future_actions):
            return super().eval_score(model_in, target=target)

    def update(  # type: ignore[override]
        self,
        model_in: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Inherited (deprecated) update, plan-conditioned like :meth:`loss`.

        Permanent contract change. Introduced by stage ``A3`` of the Fix the MS->MS
        future-action-plan conditioning contract ``.junie`` plan
        (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).
        The inherited ``Model.update`` calls ``self.loss(model_in, target)`` **positionally**, so
        the plan cannot be forwarded as an argument; it is bound as a forward-scoped context
        instead (mirrors the MTM-Pro-family ``target_raw_HOxDoa`` override precedent).
        """
        with self._active_future_action_plan(future_actions):
            return super().update(model_in, optimizer, target=target)

    def _default_forward(
        self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Route the inherited train/deploy forward path through :meth:`forecast`.

        Forwards the **train/eval-time plan context** bound by
        :meth:`_active_future_action_plan` (stage ``A3`` of the Fix the MS->MS future-action-plan
        conditioning contract ``.junie`` plan,
        ``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).

        **History (RLRP-728, rewritten -- not deleted -- by stage A3.3).** This method used to be
        deliberately plan-agnostic: ``future_actions`` was never forwarded, so the inherited train /
        single-step-deploy forward stayed byte-for-byte identical to the pre-RLRP-728 code and the
        plan channel was reachable only through an explicit :meth:`forecast` call (the deployer
        ``forecast_horizon`` entry point). That "default OFF => bit-exact" constraint is precisely
        what created the **train/test conditioning asymmetry this plan fixes**: the model was
        advertised as control-conditioned while never being *trained* with a plan, so the published
        ``plan`` vs ``obs_only`` comparison measured mostly noise injection (RLRP-781).

        The bit-exactness guarantee is preserved where it is legitimate: with no plan bound
        (``_active_action_plan is None`` -- every non-MS path, every single-step deploy, and every
        caller that does not opt in) the call is unchanged.
        """
        return self.forecast(
            x, only_elite=only_elite, future_actions=self._active_action_plan
        )

    # ==== Per-feature geometry loss: MS (multi-step forecast) head wiring =========================
    def _maybe_compose_feature_geometry_ms_head(
        self,
        losses: torch.Tensor,
        pred_mean: torch.Tensor,
        target: torch.Tensor,
        meta: Dict,
    ) -> torch.Tensor:
        """Compose the MS (multi-step forecast) head per-feature geometry term.

        RLRP-751 (task T5), inline successor of the removed
        ``_maybe_stash_feature_loss_ms_head_from_base_loss`` stash override. The
        ``ms2ms`` forecast baselines (End-to-End-TCN / M3 / TBM) reuse the
        inherited ``ExponentialFamilyMLP._deterministic_loss`` /
        ``_probabilistic_loss``; those compose ONLY the single-step deploy-head
        (``t=1``) SS term. Because these models emit a full multi-step forecast
        (their ``forward`` / :meth:`forecast` returns the flat composed multistep
        next-obs window in the LEGACY history-length layout), the geometry term
        must ALSO supervise the forecast head.

        Forwards the composed forecast prediction (``pred_mean`` — the
        deterministic mean / distribution mean, NEVER a sample) + matching composed
        ``target`` to :meth:`FeatureGeometryLossMixin._compose_feature_geometry`
        (``ms_head=``), which decomposes the composed window into its per-horizon
        single-step observations and composes the ``FEAT_GEOM_MS`` channel INLINE
        into ``losses``. No-op / bit-neutral unless the term is active AND
        ``horizon_len > 1``.

        :param losses: the running (pre-reduction) loss accumulator.
        :param pred_mean: composed multistep forecast point estimate, ``(..., out_size)``.
        :param target: matching composed multistep next-obs target, ``(..., out_size)``.
        :param meta: loss metadata dict (mutated in place).
        """
        return self._compose_feature_geometry(
            losses,
            meta,
            ms_head=(pred_mean, target),
            ms_head_legacy_composed_shape=True,
        )

    def _feature_geometry_ss_channel(
        self, pred_mean: torch.Tensor, target: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Feed the SS geometry channel this family's genuine ``t=1`` forecast step.

        This family has NO separate single-step deploy head: its ``forward`` /
        :meth:`forecast` emits the flat composed multistep next-obs window
        (``obs*history_len + act*(history_len-1)``), and the inherited base loss used to
        pass that window straight to the SS channel. That was a layout defect, not a
        weighting choice:

        * the geometry term is defined on the SINGLE-STEP obs layout
          (``EnvFeatureHandler.extra_loss`` indexes single-step obs feature indices on the
          last axis), so on a composed window it silently scored the LEADING window
          timestep -- which, for ``horizon_len < history_len``, is a PAST observation
          ``o_{t+F-H+1}``, not a prediction at all -- under the ``feature_geom_loss`` label;
        * as soon as the wrapper registers the block-facade obs denorm handle
          (``OneDTransitionRewardModelV2`` with a robust/normalizing facade), that window's
          echoed-action tail trips the obs-only width guard of
          ``_output_primitive`` and the run dies with
          ``expected width Do*obs_steps + Da*act_steps ... but got <out_size>``.

        The ``t=1`` step of the forecast IS this family's deploy point (index 0 of
        :meth:`decompose_composed_next_obs_to_singlestep_horizon`, which returns the
        trailing ``horizon_len`` timesteps in horizon order), so the SS channel keeps its
        documented meaning and its ``feature_geom_loss`` metric stays comparable with the
        single-step families -- it is simply given the right slice.

        :param pred_mean: composed multistep forecast point estimate, ``(..., out_size)``.
        :param target: matching composed multistep next-obs target.
        :return: the ``t=1`` single-step ``(point, target)`` pair, ``(..., Do)``.
        """
        pred_steps = self.decompose_composed_next_obs_to_singlestep_horizon(
            pred_mean, legacy_composed_shape=True
        )
        target_steps = self.decompose_composed_next_obs_to_singlestep_horizon(
            target, legacy_composed_shape=True
        )
        if not pred_steps or not target_steps:
            return None
        return pred_steps[0], target_steps[0]

    def _validate_future_actions(self, future_actions: torch.Tensor) -> torch.Tensor:
        """Cast + shape-check a planned future-action tensor.

        Permanent contract change. Introduced by stage ``A1`` of the Fix the MS->MS
        future-action-plan conditioning contract ``.junie`` plan
        (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).

        Expected shape ``(..., horizon_len - 1, singlestep_act_len)`` == the plan
        ``a_{t+1..t+F-1}``: the forecast ``ô_{t+1..t+F}`` is conditioned on the history
        ``(o, a)_{t-H+1..t}`` (which ALREADY carries ``a_t``, the action driving ``ô_{t+1}``) plus
        those ``F-1`` planned actions. ``a_{t+F}`` drives no predicted step and is absent from a
        training batch, so it is NOT part of the channel.

        History of this axis: RLRP-728 required the ``history_len`` (``H``) axis because
        ``_forward_encoder`` emits ``H`` steps (conflating the output-window axis with the forecast
        axis); RLRP-760 S0 re-axed it to ``F`` (``a_{t+1..t+F}``); this plan re-axes it to the
        normative ``F-1``.

        ``horizon_len == 1`` => the plan carries no new information and MUST be empty: ``None``
        (handled by the caller) or a zero-length step axis. A non-empty plan is rejected.
        """
        future_actions = self._maybe_cast_to_model_dtype(future_actions)
        expected_plan_len = self.horizon_len - 1
        if future_actions.shape[-2] != expected_plan_len:
            if expected_plan_len == 0:
                raise ValueError(
                    "future_actions must be EMPTY when horizon_len == 1 (expected "
                    f"shape[-2] == 0, got {future_actions.shape[-2]}): the single forecast "
                    "step ô_{t+1} is driven by a_t, which is already carried by the input "
                    "history, so a F==1 plan a_{t+1..t+F-1} is the empty sequence. Pass "
                    "future_actions=None (or a zero-length step axis) instead."
                )
            raise ValueError(
                "future_actions must be indexed by the (horizon_len - 1) plan axis "
                f"(expected shape[-2] == horizon_len - 1 == {expected_plan_len}, got "
                f"{future_actions.shape[-2]}): it is the plan a_{{t+1..t+F-1}} (F-1 actions) "
                f"conditioning the F-step forecast ô_{{t+1..t+F}} with F == horizon_len == "
                f"{self.horizon_len}. NOT the full a_{{t+1..t+F}} horizon_len axis (a_{{t+F}} "
                "drives no predicted step and is unavailable at training time) and NOT the "
                "history_len output-window axis."
            )
        if future_actions.shape[-1] != self.singlestep_act_len:
            raise ValueError(
                "future_actions last dim must equal singlestep_act_len "
                f"({self.singlestep_act_len}), got {future_actions.shape[-1]}."
            )
        return future_actions

    def _plan_conditioned_layout_active(self) -> bool:
        """Whether this instance's TRAINING conditioning layout is the plan-conditioned one.

        Permanent contract helper. Introduced by the MS->MS conditioning-fallback review of the
        M3 / TBM baselines (2026-09-11), which found that a missing plan must be a graceful
        **information removal**, never a **layout switch**.

        A plan is supplied on the training path for every instance reporting
        ``consumes_future_action_plan`` (see ``OneDTransitionRewardModelV2._action_plan_needed``
        / ``loss``), so for those instances the plan-conditioned layout IS the layout the weights
        were fitted on. Test-time callers that legitimately have no plan -- the single-step deploy
        head :meth:`predict_next_state` (which reports the headline ``best_pred_mae_score``) and
        the ``obs_only`` deprivation ablation -- must therefore keep that same layout and merely
        blank the genuinely unknown future actions, instead of falling back to the pre-plan
        history echo (a completely different filling of the same conditioning slots, i.e. an
        out-of-distribution input).

        Two cases legitimately stay on the history echo, because the echo is what they were
        TRAINED on:

        - ``enable_future_action_plan_conditioning=False`` -- the genuinely plan-FREE baseline;
        - ``horizon_len == 1`` -- the plan ``a_{t+1..t+F-1}`` is empty and
          :meth:`forecast` canonicalizes it to ``None``, so training itself runs the echo path.
        """
        return bool(self.consumes_future_action_plan) and self.horizon_len > 1

    def _build_driving_action_sequence(
        self, future_actions: torch.Tensor, history_flat: torch.Tensor
    ) -> torch.Tensor:
        """Build the internal driving action sequence ``a_{t..t+F-1}`` == ``[a_t] ++ plan``.

        Permanent contract helper. Introduced by stage ``A1`` of the Fix the MS->MS
        future-action-plan conditioning contract ``.junie`` plan
        (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``);
        consumed by every control-conditioned encoder (stage ``A4`` for the horizon-indexed M3 /
        TBM family, stage ``B1`` for the E2E-TCN future block).

        This is the ONLY action sequence the encoders consume internally. Its **slot 0 is ``a_t``**
        -- the action that actually drives ``ô_{t+1}`` -- taken from the **last** action slot of the
        input history (``_extract_history_action_steps`` returns ``a_{t-H+1..t}``, so ``a_t`` is the
        trailing entry, NOT the leading one). Slots ``1..F-1`` are the public plan
        ``a_{t+1..t+F-1}``. ``F == 1`` => the sequence is the single step ``[a_t]``.

        :param future_actions: the validated ``F-1``-long plan ``a_{t+1..t+F-1}``.
        :param history_flat: the flat multistep-composed history the forecast is conditioned on.
        :return: ``(..., horizon_len, singlestep_act_len)``, slot ``j`` == ``a_{t+j}``.
        """
        if self.singlestep_act_len == 0:
            return future_actions

        history_act_steps = self._extract_history_action_steps(
            history_flat, future_actions
        )
        act_len = self.singlestep_act_len
        # ``a_t`` is the LAST history action slot (see the method docstring).
        act_at_t = history_act_steps[..., -1:, :]

        # The plan and the history may carry different (ensemble/batch) leading dims -- e.g. a
        # per-anchor deploy plan ``(F-1, A)`` against a batched history ``(1, in_size)``. Align
        # both on the broadcast leading shape before concatenating on the step axis.
        leading = torch.broadcast_shapes(
            act_at_t.shape[:-2], future_actions.shape[:-2]
        )
        act_at_t = act_at_t.expand((*leading, 1, act_len))
        if self.horizon_len == 1:
            return act_at_t
        plan = future_actions.expand((*leading, self.horizon_len - 1, act_len))
        return torch.cat([act_at_t, plan], dim=-2)

    def _expand_future_actions_to_output_axis(
        self, future_actions: torch.Tensor, history_flat: torch.Tensor
    ) -> torch.Tensor:
        """Lay an ``F-1``-long plan onto the ``H``-step output-window ECHO axis.

        Permanent contract change. Introduced by stage ``A1`` of the Fix the MS->MS
        future-action-plan conditioning contract ``.junie`` plan
        (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``). This
        resolves the pre-fix ``# (CRITICAL) ToDo ... RLRP-781`` marker: the helper is now used
        **only** by the composed action-column echo (:meth:`_resolve_action_steps`); the encoders
        consume :meth:`_build_driving_action_sequence` instead, which removes the
        leading-vs-trailing slot disagreement of plan section 2.2.

        The composed multistep output window is **always** ``history_len`` steps long (see
        ``MultistepDataBufferProcessorAbstract.get_compose_next_obs``: it "always returns an array
        of multi-steps length equal to the history length to simplify handling the case where the
        ``horizon_len < history_len``"), and the ``F`` forecast steps ``ô_{t+1..t+F}`` are its
        **LAST** ``F`` entries (the window spans ``o_{t+F-H+1..t+F}``; corroborated by the
        ``obs_horizon_slice`` start index ``obs_len * (H - F)``).

        The window's action columns therefore hold ``a_{t+F-H+1..t+F-1}``: slot ``k`` is
        ``a_{t+F-H+1+k}``, and the **last** slot (``k == H-1``, i.e. ``a_{t+F}``) is dropped by
        :meth:`_compose_flat_forecast` (``remove_last_action_padding=True``) since the composed
        layout only carries ``H-1`` action steps. The plan ``a_{t+1..t+F-1}`` consequently maps
        **exactly** onto slots ``H-F .. H-2``; the leading ``H-F`` slots keep the input-history
        echo (those window steps are re-predictions of already-observed timesteps, so no planned
        action applies to them -- RLRP-760 S0 b′ property, preserved), and the dropped trailing
        slot keeps the history echo because no plan entry exists for it.

        This layout is **numerically identical** to the pre-fix ``F``-axis expansion on every
        column that survives the composition: the pre-fix version wrote ``a_{t+1..t+F}`` into the
        trailing ``F`` slots, whose only extra entry (``a_{t+F}``) landed in the dropped slot.

        ``F == 1`` => there is no plan entry at all and the result is the pure history echo (the
        legitimate conditioning no-op of the normative contract).
        """
        if self.singlestep_act_len == 0:
            return future_actions

        history_act_steps = self._extract_history_action_steps(
            history_flat, future_actions
        )
        act_len = self.singlestep_act_len
        output_window_len = int(self.output_window_len)

        # The plan and the history echo may carry different (ensemble/batch) leading dims -- e.g.
        # a per-anchor deploy plan ``(F-1, A)`` against a batched history ``(1, in_size)``. Align
        # both on the broadcast leading shape before writing the plan slots.
        leading = torch.broadcast_shapes(
            history_act_steps.shape[:-2], future_actions.shape[:-2]
        )
        if self.asymmetric_output_window:
            # RLRP-824 (KD1 / FR14): ``W == F > H`` -- the window has NO history lead, so its
            # ``W`` action slots are ``[plan a_{t+1..t+F-1} | a_{t+F}]`` where the trailing slot
            # is dropped by :meth:`_compose_flat_forecast` (``remove_last_action_padding``).
            # Nothing is echoed: start from a zero window (the dropped slot's value is
            # irrelevant) and place the plan at slots ``0 .. F-2``.
            echoed = history_act_steps.new_zeros((*leading, output_window_len, act_len))
        else:
            echoed = history_act_steps.expand(
                (*leading, output_window_len, act_len)
            ).clone()
        if self.horizon_len == 1:
            return echoed
        plan = future_actions.expand((*leading, self.horizon_len - 1, act_len))
        start = output_window_len - self.horizon_len
        echoed[..., start : start + self.horizon_len - 1, :] = plan
        return echoed

    def _resolve_action_steps(
        self,
        history_flat: torch.Tensor,
        reference_obs_steps: torch.Tensor,
        future_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the per-step action columns to compose into the forecast (RLRP-728).

        Default (``future_actions is None``): echo the input-history action stream, reproducing
        the historical behaviour exactly. When a plan is supplied, echo it instead.

        ``future_actions`` is the ``(..., history_len, singlestep_act_len)`` **output-window echo
        layout** produced by :meth:`_expand_future_actions_to_output_axis` (plan slots ``H-F..H-2``,
        history echo elsewhere) -- NOT the public ``F-1`` plan and NOT the internal driving
        sequence. Introduced by stage ``A1`` of the Fix the MS->MS future-action-plan conditioning
        contract ``.junie`` plan
        (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).
        """
        if future_actions is None:
            return self._extract_history_action_steps(history_flat, reference_obs_steps)
        if self.singlestep_act_len == 0:
            return reference_obs_steps.new_zeros((*reference_obs_steps.shape[:-1], 0))
        return future_actions

    # ==== I/O-contract helpers (shared) ==========================================================
    def _extract_history_action_steps(
        self, history_flat: torch.Tensor, reference_obs_steps: torch.Tensor
    ) -> torch.Tensor:
        """Echo the input-history action stream as per-step action columns.

        Returns the ``history_len`` action steps of shape ``(..., history_len, singlestep_act_len)``
        extracted from the flat history so they can be composed back into the forecast's action
        slots. This is the I/O-contract departure permitted by plan section 0.3 (the models predict
        observations, not actions).
        """
        if self.singlestep_act_len == 0:
            # No action channel: build an empty (..., history_len, 0) tensor.
            return reference_obs_steps.new_zeros(
                (*reference_obs_steps.shape[:-1], 0)
            )
        # ``extract_act_features_from_multistep_composed_array`` returns (..., act_len, history_len).
        act_feature = self.extract_act_features_from_multistep_composed_array(
            history_flat, is_model_output=False, vectorized=True
        )
        # (..., act_len, history_len) -> (..., history_len, act_len)
        return act_feature.transpose(-2, -1)

    @staticmethod
    def _broadcast_leading_dims(
        source: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """Expand ``source`` so its leading dims match ``reference`` (trailing feature dim kept).

        ``source`` and ``reference`` share the trailing ``(history_len, feature)`` layout; only the
        leading (ensemble/batch) dims may differ. ``torch.Tensor.expand`` prepends the missing
        leading dims without copying data.
        """
        target_shape = (*reference.shape[:-1], source.shape[-1])
        if source.shape == target_shape:
            return source
        return source.expand(target_shape)

    def _compose_flat_forecast(
        self, obs_steps: torch.Tensor, act_steps: torch.Tensor
    ) -> torch.Tensor:
        """Compose per-step ``(obs, act)`` predictions into the flat multistep-composed layout.

        ``obs_steps`` and ``act_steps`` have shape ``(..., W, obs|act)`` with
        ``W = output_window_len`` (``== history_len`` for ``F <= H``, ``== horizon_len`` for the
        RLRP-824 asymmetric window); the result has the flat ``out_size`` layout
        (``obs*W + act*(W-1)``) expected by the inherited loss/deploy. Composed-observation
        compliant: the flattening is the canonical
        :func:`revert_timestep_first_multistep_dim_unflaten_array` (FR10).
        """
        per_step = torch.cat([obs_steps, act_steps], dim=-1)
        flat = revert_timestep_first_multistep_dim_unflaten_array(
            per_step,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            remove_last_action_padding=True,
        )
        return flat
