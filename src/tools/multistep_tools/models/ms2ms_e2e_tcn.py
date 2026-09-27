# coding=utf-8
"""
End-to-End-TCN MS->MS forecast baseline (RLRP-692).

Permanent production module. Introduced by section 3 of the MS->MS forecast baselines
(End-to-End-TCN / M3 / TBM) ``.junie`` plan
(``feature_ms2ms_forecast_baselines_e2etcn_m3_tbm_plan_RLRP-692_693_694_20260703.md``).
Reproduces Looper & Waslander (CRV 2022) "Temporal Convolutions for Multi-Step Quadrotor Motion
Prediction" (``End2EndNet.py``): a fully-convolutional causal dilated TCN that emits the whole
horizon in one forward pass (the DIR / one-shot property). Per plan section 0.1 the reference is a
single deterministic net (L1/MSE, no probabilistic head), so ``ensemble_size`` is hard-fixed to 1
and ``deterministic`` to True. Per plan section 0.3 the ``End2EndNet`` block wiring is reproduced
exactly (``Conv1d``+``Chomp1d`` causal pair, ``BatchNorm1d``+``ReLU``, residual every other block,
``K=5``, ``dilations=[1,2,4,8]``); only the paper-faithful ``Chomp1d`` primitive is reused from
``tools/baseline_models/long_horizon_dynamics/tcn.py`` (its ``TemporalBlock`` deviates and is NOT
reused).
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import omegaconf
import torch
from torch import nn as nn

from tools.baseline_models.long_horizon_dynamics.tcn import Chomp1d
from tools.multistep_tools.models.abstract_ms2ms_forecast import AbstractMS2MSForecast
from tools.multistep_tools.models.exponential_family_mlp_utils import (
    create_activation_,
)
from tools.multistep_tools.multistep_model_util import (
    timestep_first_multistep_dim_unflaten_array,
)


class _TConv(nn.Module):
    """A single causal (truncated) 1D convolution, mirroring ``End2EndNet.py::TConv``."""

    def __init__(
        self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=1, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.init_weights()

    def init_weights(self) -> None:
        # Mirror the reference small-positive-variance init.
        self.conv1.weight.data.normal_(0, 0.01)
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TConvBlock(nn.Module):
    """Temporal convolution block, mirroring ``End2EndNet.py::TConvBlock``.

    A sequence of causal convolutions with exponentially increasing dilations, wrapped by an
    internal residual connection (with a 1x1 downsample conv when the channel count changes).
    """

    def __init__(
        self, c_in: int, c_out: int, kernel_size: int, dilations: Tuple[int, ...]
    ):
        super().__init__()
        self.dsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        layers = []
        for i, d in enumerate(dilations):
            layer_in = c_in if i == 0 else c_out
            layers.append(_TConv(layer_in, c_out, kernel_size, dilation=d))
        self.network = nn.Sequential(*layers)
        self.lookback = sum((kernel_size - 1) * d for d in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        res = x if self.dsample is None else self.dsample(x)
        return out + res

class MS2MSEndToEndTCN(AbstractMS2MSForecast):
    """End-to-End-TCN (DIR one-shot) MS->MS forecast baseline (RLRP-692).

    The flat history is reshaped to a per-step sequence view, the ``F``-step future (plan) block is
    concatenated on the **time axis**, and the resulting channels-first ``(..., obs+act+mask, H+F)``
    tensor is run through a faithful ``End2EndNet`` body (stacked ``TConvBlock``s with
    ``BatchNorm1d`` + activation (paper: ``ReLU``; configurable via ``activation_fn_cfg``,
    section 3.4) and an outer residual skip every other block; the final block is bare and projects
    to ``singlestep_obs_len``). The per-step observation predictions are returned to the shared
    ``AbstractMS2MSForecast.forecast`` which composes them with the echoed action stream (the single
    permitted I/O departure, plan section 0.3).

    Plan conditioning (RLRP-790). Stages ``B1`` / ``B2`` of the Fix the MS->MS future-action-plan
    conditioning contract ``.junie`` plan
    (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``) made this
    encoder genuinely plan-conditioned -- it previously accepted ``future_actions`` and discarded
    them, which made the whole ``plan`` mode a no-op on the reported metric and left this baseline
    not comparable to the plan-conditioned M3 / TBM. See :meth:`_forward_encoder` for the defect
    analysis, the operator-mandated encoding decision (``D6``) and the trailing-output derivation.

    **Decision ``B2.4`` (echoed-action OUTPUT columns): KEPT.** Now that the plan is a first-class
    *input*, predicting the action columns back out is redundant. They are nevertheless kept for
    now: dropping them would change ``out_size``, the normalizer block facade, the deploy adapters
    and the composed target layout shared with every other multistep family -- a contract change
    far wider than RLRP-790, and one that touches the RLRC I/O-contract departure documented in
    ``AbstractMS2MSForecast``. Blast radius beats marginal redundancy here; their removal is raised
    as a follow-up (plan section 12) rather than executed in this stage.

    RLRP-736 bespoke-forward plan §3.5 (C.3): the one-shot TCN encoder IS wired for
    the by-construction rotation head. The per-step attitude input slot of each
    history block is encoded to the continuous internal rep before the conv body
    (``R(q)=R(-q)`` sign-invariance; Geist et al. 2024), the final block projects
    to the rep-expanded single-step width ``_ss_trunk_out_len``, and every per-step
    obs prediction is decoded back to a unit quaternion via the single-step
    :meth:`_apply_ss_orientation_output_decoding`. Neutral (``quaternion``) keeps
    the body byte-for-byte identical.
    """

    # By-construction rotation head IS wired for this family (see class docstring);
    # un-gates the ``AbstractMS2MSForecast.forecast`` fail-loud guard.
    _supports_by_construction_orientation: bool = True

    # Permanent architectural constant. Introduced by stage ``B1`` of the Fix the MS->MS
    # future-action-plan conditioning contract ``.junie`` plan
    # (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``),
    # then DEFAULTED OFF by the E2E-TCN paper-fidelity review (2026-09-11, operator decision).
    # Width of the optional explicit *is-future* mask appended to every per-step input vector.
    # ``0`` == the reference ``End2EndNet`` input layout (bare zero fill for the unknown future
    # observations, WITH its ambiguity flaw); shadowed per instance by the constructor knob
    # ``enable_is_future_mask_channel``.
    _IS_FUTURE_MASK_CHANNELS: int = 0

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
        hid_size: int = 32,
        kernel_size: int = 5,
        dilations: Tuple[int, ...] = (1, 2, 4, 8),
        ensemble_size: int = 1,
        propagation_method: Optional[str] = None,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        mae_loss: bool = True,
        model_use_double_precision: bool = False,
        dropout: float = 0.0,
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
        # config bit-exact. NOTE: the ``H+F`` input encoding (stage ``B1``) is UNCHANGED by
        # this knob -- the future block is still appended, its action channels simply carry
        # the history echo instead of a plan, so the architecture stays paper-faithful and
        # the two runs remain parameter-comparable.
        enable_future_action_plan_conditioning: bool = True,
        # Paper-fidelity switch for the explicit *is-future* mask input channel (E2E-TCN
        # paper-fidelity review, 2026-09-11 -- supersedes the mask half of operator decision
        # ``D6`` in ``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).
        # ``False`` (default) reproduces the reference ``End2EndNet`` input layout EXACTLY: the
        # unknown future observations are a BARE ZERO FILL, so the network genuinely cannot tell
        # "observation unknown" from "observation == 0" (an ordinary value in normalized space).
        # That ambiguity is the ORIGINAL AUTHORS' flaw and it is reproduced deliberately: this is
        # a *reference baseline*, so fidelity outranks local optimality -- feeding it information
        # the authors did not have would make the "reproduced SoTA baseline" claim indefensible
        # even if it measured better. ``True`` opts into the strictly-more-informative RLRC
        # variant as an explicitly-captioned ABLATION (it widens the first conv layer, so
        # checkpoints are NOT interchangeable between the two settings).
        enable_is_future_mask_channel: bool = False,
        # RLRP-824 Step 7 (FR9): opt-in autocast around training_step / validation_step
        # (``null`` = bit-exact fp32 path; ``bf16`` = the profiling-gated ULH opt-in).
        mixed_precision: Optional[str] = None,
    ):
        # ``ensemble_size`` is accepted for pipeline/config-surface compatibility (every
        # ``ms_model`` config declares it and the reporting path reads ``cfg.ms_model.ensemble_size``,
        # mirroring the ``B_tcn_deterministic`` convention), but per plan section 0.1 End-to-End-TCN
        # is a deterministic *single* net: only ``ensemble_size == 1`` is valid.
        if ensemble_size != 1:
            raise ValueError(
                "MS2MSEndToEndTCN is a deterministic single net (plan section 0.1); "
                f"ensemble_size must be 1, got {ensemble_size}."
            )

        # Store TCN-specific hyper-parameters BEFORE ``super().__init__`` since the base constructor
        # calls ``_build_network`` (which reads them) during initialisation.
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.tcn_channel_width = hid_size
        self.num_tcn_blocks = num_layers
        # Instance-level mask width (shadows the class default); read by ``_build_network``,
        # which ``super().__init__`` calls, hence set here.
        self.enable_is_future_mask_channel = bool(enable_is_future_mask_channel)
        self._IS_FUTURE_MASK_CHANNELS = 1 if self.enable_is_future_mask_channel else 0

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
            enable_auto_loss_weighting=False,
            feature_weight_mode=feature_weight_mode,
            feature_weight_max_ratio=feature_weight_max_ratio,
            description=description,
            feature_geometry=feature_geometry,
            # RLRP-736 bespoke-forward plan §3.5 (C.3): by-construction rotation rep
            # (defaults quaternion / None => bit-exact). The one-shot TCN encoder is
            # now wired (see class flag + _build_network / _forward_encoder below).
            internal_orientation=internal_orientation,
            orientation_singlestep_slots=orientation_singlestep_slots,
            enable_future_action_plan_conditioning=enable_future_action_plan_conditioning,
            mixed_precision=mixed_precision,
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
        learn_logvar_bounds: bool = False,
        instanciate_logvar_bound_module: bool = False,
        dropout: float = 0.0,
    ) -> None:
        """Build the faithful ``End2EndNet`` body (plan section 3.3)."""
        assert (
            self.num_tcn_blocks >= 2
        ), f"End-to-End-TCN needs at least 2 TConvBlocks, got {self.num_tcn_blocks}."

        # RLRP-736 bespoke-forward plan §3.5 (C.3): widen the input channel count by
        # the per-single-step-block input-encode delta ``_ss_block_in_extra`` (0 when
        # the by-construction rotation rep is OFF => bit-exact) and project the final
        # block to the rep-expanded single-step width ``_ss_trunk_out_len`` (==
        # singlestep_obs_len when OFF) so each per-step attitude slot can be decoded
        # to a unit quaternion.
        # Stage ``B1`` of the Fix the MS->MS future-action-plan conditioning contract ``.junie``
        # plan (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``):
        # the conv sequence is the paper-faithful time-axis concatenation of the ``H``-step
        # history block and the ``F``-step future (plan) block. ``_IS_FUTURE_MASK_CHANNELS`` is
        # ``0`` by default, which makes the per-step channel count identical to the reference
        # ``End2EndNet`` (bare zero fill for the unknown future observations, ambiguity flaw
        # included -- see the constructor knob ``enable_is_future_mask_channel``); the opt-in
        # mask adds exactly one channel.
        channel_in = (
            self.singlestep_obs_len
            + self.singlestep_act_len
            + self._ss_block_in_extra
            + self._IS_FUTURE_MASK_CHANNELS
        )
        width = self.tcn_channel_width
        # channels: [C_in, w, w, ..., w, _ss_trunk_out_len] with ``num_tcn_blocks`` blocks.
        channels: List[int] = (
            [channel_in]
            + [width] * (self.num_tcn_blocks - 1)
            + [self._ss_trunk_out_len]
        )

        blocks = []
        batch_norms = []
        for i in range(self.num_tcn_blocks):
            blocks.append(
                _TConvBlock(
                    channels[i], channels[i + 1], self.kernel_size, self.dilations
                )
            )
            # Every block except the final one is followed by BatchNorm1d + activation
            # (paper: ReLU; configurable via activation_fn_cfg, plan section 3.4).
            if i < self.num_tcn_blocks - 1:
                batch_norms.append(nn.BatchNorm1d(channels[i + 1]))
        self.tcn_blocks = nn.ModuleList(blocks)
        self.tcn_batch_norms = nn.ModuleList(batch_norms)
        # Activation (plan section 3.4). The reference ``End2EndNet`` (Looper & Waslander, CRV 2022)
        # uses ``ReLU`` -- this stays the paper-faithful default because ``create_activation_``
        # maps ``activation_fn_cfg is None`` -> ``nn.ReLU()``. Passing an explicit ``activation_fn_cfg``
        # (e.g. ``torch.nn.GELU``) is an *experimental liberty* that departs from the paper. A single
        # shared (stateless) activation instance is reused across blocks, mirroring the prior
        # hardcoded ``nn.ReLU`` and the MLP-family ``ExponentialFamilyMLP`` convention.
        self.tcn_activation = create_activation_(activation_fn_cfg)

        # Receptive-field guard (plan section 11), re-derived by stage ``B1.3`` of the Fix the
        # MS->MS future-action-plan conditioning contract ``.junie`` plan
        # (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).
        # The conv sequence is now ``H + F`` steps long, and the LAST output slot must still see
        # the FIRST history step: a guard that only checked ``H`` would silently permit a stack
        # whose far-horizon slots lose sight of the start of the conditioning window (and of
        # ``o_t``), i.e. a forecast conditioned on a truncated history.
        required_field = self.history_len + self.horizon_len
        receptive_field = self.num_tcn_blocks * blocks[0].lookback + 1
        assert receptive_field >= required_field, (
            f"TCN receptive field {receptive_field} < history_len + horizon_len "
            f"({self.history_len} + {self.horizon_len} = {required_field}); the trailing "
            f"forecast slots would not see the whole conditioning window. Increase "
            f"num_layers (currently {self.num_tcn_blocks}), kernel_size (currently "
            f"{self.kernel_size}) or dilations (currently {self.dilations}): the field is "
            f"num_layers * sum((kernel_size - 1) * d for d in dilations) + 1."
        )

        self.to(self.device)
        return None

    def _tcn_body_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ``End2EndNet`` block stack on a channels-first ``(N, C, H)`` tensor.

        Reproduces the reference forward: ``act(bn(tconv(.)))`` per block (``act`` = the configured
        activation, paper default ``ReLU``; plan section 3.4), an outer residual skip on every other
        (0-based odd, non-final) block, and a bare final block.
        """
        for i, block in enumerate(self.tcn_blocks):
            y = block(x)
            is_last = i == (self.num_tcn_blocks - 1)
            if not is_last:
                y = self.tcn_activation(self.tcn_batch_norms[i](y))
                # Outer residual skip every other block (x2 = x1 + act(bn(tconv2(x1))), ...).
                if i % 2 == 1:
                    y = y + x
            x = y
        return x

    def _compose_history_and_future_blocks(
        self, history_seq: torch.Tensor, future_actions: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Concatenate the ``H``-step history block and the ``F``-step future block on the time axis.

        Permanent architecture helper. Introduced by stage ``B1`` of the Fix the MS->MS
        future-action-plan conditioning contract ``.junie`` plan
        (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``).

        Per-step channel layout (identical for both blocks, which is what makes this the
        *paper-faithful time-axis concatenation* rather than a bespoke fusion branch)::

            [ obs channels | act channels | is-future mask ]

        The trailing ``is-future`` mask channel is **present only when**
        ``enable_is_future_mask_channel=True``; the default (paper-faithful) per-step vector is
        just ``[ obs channels | act channels ]``, i.e. the reference ``End2EndNet`` layout.

        - **history block** (``H`` steps): the real ``(o, a)_{t-H+1..t}`` channels, mask ``0``.
        - **future block** (``F`` steps): observation channels **zero-filled** (they are genuinely
          unknown), action channels = the driving sequence ``a_{t..t+F-1}``, mask ``1``.
          With **no plan** on a plan-conditioned instance, slot 0 still carries the KNOWN ``a_t``
          and only slots ``1..F-1`` are blanked (2026-09-11 conditioning-fallback review).

        On the reference (default) layout the zero fill is **ambiguous**: a normalized observation
        of exactly ``0`` is an ordinary value, so the network cannot tell "this observation is
        unknown" from "this observation is zero". That is the ORIGINAL AUTHORS' flaw and it is
        reproduced on purpose (E2E-TCN paper-fidelity review, 2026-09-11); the optional mask
        channel removes it and is therefore an explicitly-captioned ablation, NOT the baseline.

        :param history_seq: ``(..., history_len, C)`` per-step history blocks (already
            orientation-encoded), where the action channels are the **trailing**
            ``singlestep_act_len`` columns.
        :param future_actions: the driving sequence ``a_{t..t+F-1}`` of shape
            ``(..., horizon_len, singlestep_act_len)``; ``None`` => the ``obs_only`` ablation, in
            which only the genuinely unknown future actions ``a_{t+1..t+F-1}`` are blanked (slot
            0 keeps ``a_t``, which the history already carries).
        :return: ``(..., history_len + horizon_len, C + _IS_FUTURE_MASK_CHANNELS)``.
        """
        act_len = self.singlestep_act_len
        horizon_len = self.horizon_len
        mask_width = self._IS_FUTURE_MASK_CHANNELS

        # History block: real channels (+ mask 0 when the mask channel is enabled).
        if mask_width:
            hist_mask = history_seq.new_zeros((*history_seq.shape[:-1], mask_width))
            history_block = torch.cat([history_seq, hist_mask], dim=-1)
        else:
            history_block = history_seq

        # Future block: zeros everywhere (+ mask 1), then the driving actions written into the
        # ``act_len`` channels sitting just before the (optional) mask channel.
        block_width = history_block.shape[-1]
        future_block = history_block.new_zeros(
            (*history_block.shape[:-2], horizon_len, block_width)
        )
        if mask_width:
            future_block[..., -mask_width:] = 1.0

        # The action channels are the trailing ``act_len`` columns BEFORE the mask channel
        # (the orientation input-encoding widens the OBS slots only, so the act block stays
        # at the end of the per-step vector).
        act_end = block_width - mask_width
        act_start = act_end - act_len

        if future_actions is not None and act_len > 0:
            driving = future_actions.to(future_block.dtype).expand(
                (*future_block.shape[:-2], horizon_len, act_len)
            )
            future_block[..., act_start:act_end] = driving
        elif act_len > 0 and self._plan_conditioned_layout_active():
            # Permanent: with NO plan, future slot 0 still carries ``a_t`` -- the action driving
            # ``ô_{t+1}``, which is KNOWN (it is the last input-history action step) and is
            # present in that slot at every training step. Leaving it at zero here made the
            # single-step deploy head (``predict_next_state``) and the ``obs_only`` ablation
            # violate the causal identity ``ô_{t+1} = f(history, a_t)``: the plan carries
            # ``a_{t+1..}`` only, so step 1 must be unchanged by its absence. Only slots
            # ``1..F-1`` are genuinely unknown and stay zero (the is-future mask keeps that
            # unambiguous when the mask channel is enabled). Introduced by the MS->MS
            # conditioning-fallback review of the M3 / TBM baselines (2026-09-11), applied here
            # for cross-baseline symmetry.
            future_block[..., 0, act_start:act_end] = history_seq[..., -1, -act_len:]

        return torch.cat([history_block, future_block], dim=-2)

    # ==== Encoder ================================================================================
    def _forward_encoder(
        self,
        history_flat: torch.Tensor,
        only_elite: bool = False,
        future_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the ``H+F``-step plan-conditioned TCN and return the per-step observations.

        Permanent architecture change. Introduced by stages ``B1`` / ``B2`` of the Fix the MS->MS
        future-action-plan conditioning contract ``.junie`` plan
        (``fix_ms2ms_future_action_plan_conditioning_plan_RLRP-781_RLRP-790_20260830.md``),
        YouTrack RLRP-790.

        **The two defects this replaces.**

        1. *The plan was discarded, not mis-indexed.* The pre-fix encoder accepted
           ``future_actions`` and never read it; only the *echoed action* columns of the composed
           forecast changed when a plan was supplied, and those are discarded before the metric is
           computed. The whole ``plan`` conditioning mode was therefore a **no-op on the reported
           metric** for this baseline (and a test asserted that no-op as intended behaviour).
        2. *The output window was temporally mis-labelled.* The stack is strictly causal
           (``_TConv`` = ``Conv1d`` + ``Chomp1d``), so with an ``H``-step input, output slot ``k``
           predicted ``o_{t+1+k}`` from inputs only up to ``(o,a)_{t-H+1+k}``: a **uniform
           ``H``-step-lag sequence-to-sequence map**, not an ``F``-step plan-conditioned forecast.
           Slot 0 had to predict ``o_{t+1}`` from a SINGLE history step (information allocation
           inverted w.r.t. difficulty) and **no slot ever saw an action beyond its own timestep**
           => control-unconditioned by construction. It trained and produced a finite MAE, which
           is why it went unnoticed; but it was not the E2E-TCN of Looper & Waslander
           (arXiv 2110.04182 p.3) and was **not comparable** to the plan-conditioned M3 / TBM
           (a fairness-of-baseline defect for the RLRP-667 claim).

        **Encoding (operator decision ``D6`` -- fidelity, not tuning).** The paper-faithful
        **time-axis concatenation** is used: the conv sequence is extended from ``H`` to ``H + F``
        steps, where the ``F`` future steps carry the driving actions ``a_{t..t+F-1}`` with their
        **observation channels zero-filled** -- exactly the reference ``End2EndNet`` input layout,
        INCLUDING its ambiguity flaw ("obs unknown" is indistinguishable from "obs == 0", which is
        an ordinary value in normalized space). The ``D6`` is-future mask channel that removed
        that ambiguity is now an OPT-IN ablation (``enable_is_future_mask_channel``, default
        ``False``) per the E2E-TCN paper-fidelity review (2026-09-11): feeding the baseline
        information the original authors did not have makes it an improved model, not a
        reproduced baseline. The alternative -- a separate plan branch fused into the head -- is
        **rejected**: E2E-TCN is a *reference* baseline, so fidelity to the paper outranks local
        optimality; a bespoke fusion branch would make the "reproduced SoTA baseline" claim
        indefensible even if it measured better.

        **Why the trailing outputs are the forecast (derivation).** With a strictly causal stack,
        output slot ``H + k`` (0-based ``k`` over the future block) sees every input up to future
        step ``k``, i.e. the **whole** ``H``-step history **plus** the driving actions
        ``a_{t..t+k}`` -- and **no observation beyond ``o_t``**. That is exactly the conditioning
        the normative contract requires for ``ô_{t+k+1}``. Reading the trailing ``F`` outputs as
        ``ô_{t+1..t+F}`` therefore repairs defect (2) directly, and it also repairs the
        context-starvation of the near-horizon slots: every forecast slot now sees the full
        history. The composed output window stays ``H`` steps (``out_size`` unchanged, decision
        ``D4``), so the trailing ``H`` outputs are returned: its last ``F`` entries are the
        forecast and the leading ``H-F`` entries keep their prior meaning (re-predictions of
        already-observed window steps).

        **No plan bound (``future_actions is None``).** The future block is still appended (the
        architecture is fixed at build time) with its action channels zeroed and the is-future
        mask set: "no control information is known about the future", which is precisely the
        ``obs_only`` ablation semantics. ``F == 1`` remains legitimate: the future block is then a
        single step carrying ``a_t``.

        **Cost (``B1.5`` / risk ``R12``).** The sequence grows ``H -> H+F`` (up to ~2x encoder
        cost at ``F == H``) and the receptive-field requirement grows with it. That is a
        **perf** concern routed to RLRP-730 -- never grounds to weaken ``D6`` or the guard.

        :param future_actions: the internal driving action sequence ``a_{t..t+F-1}`` ==
            ``[a_t] ++ plan`` of shape ``(..., horizon_len, singlestep_act_len)`` (slot 0 ==
            ``a_t``), built by ``AbstractMS2MSForecast._build_driving_action_sequence``.
        """
        # (..., in_size) -> (..., H, obs+act)
        seq = timestep_first_multistep_dim_unflaten_array(
            history_flat,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            sequence_len=self.history_len,
        )
        # RLRP-736 bespoke-forward plan §3.5 (C.3): encode the attitude input slot(s)
        # WITHIN each single-step history block (last dim == obs+act) to the
        # continuous internal rep (``R(q)=R(-q)`` sign-invariance; Geist et al. 2024)
        # before the channels-first transpose, widening the last dim by exactly
        # ``_ss_block_in_extra`` (0 when the rep is OFF => bit-exact); ``channel_in``
        # in ``_build_network`` accounts for the same delta.
        seq = self._apply_ss_orientation_input_encoding(seq)

        # Stage ``B1``: build the paper-faithful ``H + F`` time-axis concatenation.
        seq = self._compose_history_and_future_blocks(seq, future_actions)
        sequence_len = self.history_len + self.horizon_len

        # (..., H+F, C) -> (..., C, H+F)
        seq = seq.transpose(-2, -1)

        # PERF (deferred): readability-first reshape/transpose to satisfy Conv1d/BatchNorm1d's
        # strict (N, C, L) contract; see plan section 3.3 / 11. TODO(perf): fuse the flatten/
        # transpose into a single pre-allocated channels-first path once profiling justifies it.
        leading_shape = seq.shape[:-2]
        channels = seq.shape[-2]
        conv_in = seq.reshape(-1, channels, sequence_len)

        conv_out = self._tcn_body_forward(conv_in)  # (N, _ss_trunk_out_len, H+F)

        conv_out = conv_out.reshape(
            *leading_shape, self._ss_trunk_out_len, sequence_len
        )
        # (..., _ss_trunk_out_len, H+F) -> (..., H+F, _ss_trunk_out_len)
        obs_steps = conv_out.transpose(-2, -1)
        # Stage ``B2``: read the TRAILING ``W = output_window_len`` outputs as the composed output
        # window ``ô_{t+F-W+1..t+F}``; its last ``F`` entries are the forecast ``ô_{t+1..t+F}``
        # (see the derivation in the method docstring). ``W == H`` for ``F <= H`` (``out_size``
        # unchanged, bit-exact); ``W == F`` on the RLRP-824 asymmetric window, where the trailing
        # ``F`` conv outputs -- the whole future block -- ARE the window.
        obs_steps = obs_steps[..., -self.output_window_len :, :]

        # RLRP-736 bespoke-forward plan §3.5 (C.3): decode each per-step obs
        # prediction's rep-expanded attitude slot(s) back to a 4-D unit quaternion by
        # construction (``_ss_trunk_out_len`` -> ``singlestep_obs_len``). No-op /
        # bit-exact when the by-construction rotation rep is OFF.
        obs_steps = self._apply_ss_orientation_output_decoding(obs_steps)

        # Family convention: a 2D ``(batch, in_size)`` input to an ensemble model yields an output
        # with a leading ensemble dim (mirrors ``EnsembleLinearLayer`` broadcasting used by the
        # MLP-based baselines). ``num_members`` is 1 here (deterministic single net).
        if history_flat.dim() == 2:
            obs_steps = obs_steps.unsqueeze(0)
        return obs_steps, None
