# coding=utf-8
import contextlib
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import omegaconf
from deprecated import deprecated

import torch
from mbrl.types import TransitionBatch
from torch import Tensor, nn as nn
from torch.distributions import MixtureSameFamily
from torch import distributions as dist
from mbrl.models import EnsembleLinearLayer, truncated_normal_init
import numpy as np

from tools.console_tools.format import ConsoleFormat
from tools.console_tools.message import consol_msg_universal_one_liner
from tools.multistep_tools.models.exponential_family_mlp_utils import (
    EnsembleResidualBlock,
    LogvarBoundLayer,
    create_activation_,
    create_linear_layer_,
    create_logvar_bound_layer,
    create_layer_bloc_factory,
    make_layer_bloc_seq,
    zero_init_residual_blocks_,
)
from tools.multistep_tools.models.ms2ms2ss_ar_temporal_mixture_pme_utils import (
    AdditiveIndexAndHistoryDependentTemporalMixtureWeighs,
    AdditiveIndexInputAndHistoryDependentTemporalMixtureWeighs,
    IndexAndHistoryDependentTemporalMixtureWeighs,
    InputAndIndexDependentTemporalMixtureWeighs,
    InputDependentTemporalMixtureWeighs,
    TemporalMixtureWeighs,
)
from tools.multistep_tools.models.precision_bounds import (
    logvar_safe_max,
    variance_floor,
    ar_sample_clamp_ceil,
)


from tools.multistep_tools.models import (
    AbstractFeatureWeightedMultiStepMLP,
    CompoundedPredictionMultiStepIterator,
)
from tools.multistep_tools.models.compounded_prediction_multistep_iterator_utils import (
    TeacherForcingScheduler,
)
from tools.multistep_tools.models.utils import reduce_probabilistic_compose_loss
from tools.feature_handling_tools.feature_spec import (
    InternalOrientationRep,
    is_s2_orientation_rep,
)
from tools.feature_handling_tools.orientation_heads import (
    DEFAULT_EXTERNAL_ORIENTATION_WIDTH,
    align_quaternion_slots_to_reference,
    decode_internal_rep_to_orientation_slot,
    internal_rep_out_width,
    rotation_tangent_nll,
)
from tools.multistep_tools.multistep_model_util import (
    revert_timestep_first_multistep_dim_unflaten_array,
    timestep_first_multistep_dim_unflaten_array,
)
from tools.multistep_tools.utils import (
    compute_multistep_model_in_size,
    compute_multistep_model_out_size,
)

# A3 (RLRP-783): per-step flush of the latched, on-device non-finite guard (see the RLRC
# MTM-Pro models code optimization `.junie` plan
# ``perf_RLRP-783_mtm_pro_models_code_optimization_plan_20260827.md``).
from tools.math_tools.weigthing import flush_non_finite_reports

# RLRP-786 (``perf_RLRP-786_cuda_graph_train_step_plan_20260919.md``): the recording pass of
# ``CudaGraphTrainStep.capture`` runs the Python body of the loss ONCE without executing it and
# without training on the batch; the HOST-side per-step bookkeeping (training-step counter,
# scheduler ``step``s, the teacher-forcing coin draw + H2D mask refill) must therefore be skipped
# during that pass -- it is replayed by ``advance_host_step_state`` from the graph ``before_replay``
# hook, exactly once per trained batch. Shared with the AR MS2SS family (AR-baseline extension).
from tools.torch_tools.cuda_graph_train_step import (
    is_recording_cuda_graph as _is_recording_cuda_graph,
)


# Default values for the nested loss-term / projection / deploy config groups (RLRP-704).
# These groups replace the former flat ``__init__`` kwargs; the YAML configs nest the keys
# under these group names and Hydra passes each group as a dict. Internally the groups are
# unpacked back into the historical flat local variables, so the rest of the model code and the
# ``self.*`` attribute names are unchanged.
_LOSS_CFG_GROUP_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # Single-step (SS) composite loss term (formerly the flat ``ss_composite_loss_weight`` kwarg,
    # where 0.0 meant "disabled"). Now an explicit on/off switch with its own static weight and a
    # dedicated auto-weighting toggle.
    #   enable        -> add the single-step head NLL to the composite objective.
    #   loss_weight   -> static SS weight (applied with AND without auto-weighting).
    #   auto_weighting-> route the SS term through the composite uncertainty auto-weighting
    #                    (dedicated "SS" slot); when False it enters the sum at its static weight.
    "single_step_loss": {
        "enable": False,
        "loss_weight": 1.0,
        "auto_weighting": True,
    },
    "pre_mixture_u_loss": {"enable": False},
    "post_mixture_loss": {
        "enable": False,
        "backprop_mode": "full_network",
        "loss_weight": 1.0,
    },
    "info_projection_loss": {
        "enable": False,
        "num_samples": 4,
        "head_data_nll": False,
        "loss_weight": 1.0,
    },
    "siw_mp_loss": {
        "enable": False,
        "num_samples": 1,
        "loss_weight": 1.0,
    },
    # Generative Mode-Seeking IWAE (GMS-IWAE) loss. Mutually exclusive with siw_mp_loss and
    # info_projection_loss (all three share the single projection head q_theta). The ``decoder``
    # sub-block configures the generative observation-likelihood p(y|z); it is kept as a FIXED
    # identity (mean = z) observation-noise model, since the decoder is a training-time anchor and
    # is NOT deployed — the deployable expressiveness lives on the projection head q_theta (see the
    # ``projection.head_*`` keys), so a learned decoder would only turn z into a non-deployable
    # latent and break the "deploy q_theta = obs prediction" assumption.
    "gms_iwae_loss": {
        "enable": False,
        "num_samples": 5,
        "loss_weight": 1.0,
        "decoder": {
            # Identity-only p(y|z) = f(z, sigma_obs): mean = z, scale from the obs-noise model.
            "obs_noise": 0.1,
            "learnable_noise": True,
        },
    },
    "projection": {
        # backprop_mode: {full_network, mixing_only, head_only}. head_only also detaches the mixer
        # log-weights so ONLY q_theta (+ decoder params) train (distil a frozen forecast+mixer).
        "backprop_mode": "mixing_only",
        "per_step_conditioning": False,
        # Shared auto-weighting toggle for the (mutually exclusive) IPROJ / SIW-MP / GMS-IWAE
        # projection objectives. When True the active projection term is routed through the
        # composite uncertainty auto-weighting (a dedicated learnable slot); when False it enters
        # the composite sum at its static `loss_weight` (regularisation-term default).
        "auto_weighting": False,
        # Unified temporal-discount: a single float gamma for the per-step IPROJ/SIW-MP/RC reduction.
        # The default 1.0 means DISABLED (uniform / flat mean, i.e. every step equally weighted);
        # any value != 1.0 enables the gamma^i horizon discount.
        "temporal_weights": 1.0,
        # Deployable summary head q_theta (the object emitted at deploy): an ensemble-aware
        # `_build_network`-style residual trunk + split mean/logvar heads (deploy-quality so the
        # single-step prediction is not bottlenecked). The legacy `head_arch` selector was removed
        # (only the residual "network" head remains).
        #   head_hidden      -> hidden width of the shared projection head q_theta.
        #   head_num_layers  -> residual blocks in the shared trunk.
        #   head_split_num_layers -> residual blocks per mean/logvar split head.
        #   head_dropout     -> dropout probability applied INSIDE every residual block of BOTH
        #                       the shared trunk AND the mean/logvar split heads (the block-owned
        #                       dropout of :class:`EnsembleResidualBlock`). 0.0 -> disabled.
        "head_hidden": 64,
        "head_num_layers": 2,
        "head_split_num_layers": 1,
        "head_dropout": 0.0,
    },
    # Temporal mixing-function variant (formerly the flat ``temporal_mixture_weights_kind`` kwarg).
    #   kind: {index, input_dependent, index_and_history_dependent,
    #   additive_index_and_history_dependent, input_and_index_dependent,
    #   additive_index_input_and_history_dependent}. ``index`` -> step-indexed learnable gammas
    #   (TemporalMixtureWeighs); ``input_dependent`` -> an ensemble-aware residual MLP over per-step
    #   component tokens cat([mean, logvar]) (InputDependentTemporalMixtureWeighs, RLRP-735
    #   refactor: zero-init -> uniform warm start, honours head_size/num_layers/dropout);
    #   ``index_and_history_dependent`` (RLRP-735) -> an ensemble-aware residual MLP over the frozen
    #   input-history realisation ``s_τ^H`` (IndexAndHistoryDependentTemporalMixtureWeighs, PURE
    #   history head); ``additive_index_and_history_dependent`` (RLRP-735 P3) -> the same history
    #   head PLUS a zero-init per-(i,k) additive index-γ base (φ_i^k = γ[i,k] + ψ_i^k(s_τ^H);
    #   AdditiveIndexAndHistoryDependentTemporalMixtureWeighs, byte-for-byte init parity);
    #   ``input_and_index_dependent`` -> step-indexed γ PLUS the per-step input head ρ, no history
    #   (φ_i^k = γ[i,k] + ρ_i^k(token_i); InputAndIndexDependentTemporalMixtureWeighs);
    #   ``additive_index_input_and_history_dependent`` (RLRP-735 P3 combined) -> γ index base +
    #   ψ history head + ρ per-step input head (φ_i^k = γ[i,k] + ψ_i^k(s_τ^H) + ρ_i^k(token_i);
    #   AdditiveIndexInputAndHistoryDependentTemporalMixtureWeighs, uniform init).
    #   head_size: hidden width of every non-index mixer head (used by all kinds except ``index``).
    #   num_layers: number of EnsembleResidualBlocks in every non-index mixer head trunk (default
    #     2). Ignored by ``index`` (parity).
    #   dropout: block-owned dropout probability inside every residual block of every non-index
    #     mixer head (default 0.0 -> disabled). Ignored by ``index`` (parity).
    #   NOTE: for the combined ``additive_index_input_and_history_dependent`` kind the SAME
    #     head_size/num_layers/dropout drive BOTH the history head and the input head (P-plan B.4).
    "temporal_mixture_weights": {
        "kind": "index",
        "head_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
    },
    # deploy_head.mode: {forecast (default, the moment-matched / true-mixture forecast output),
    # projection (the trained compressed projection head)}.
    "deploy_head": {"mode": "forecast"},
    "rollout_consistency_loss": {
        "enable": False,
        "num_steps": 0,
        "backprop_mode": "mixing_only",
        "loss_weight": 1.0,
        # Dedicated RC auto-weighting toggle (independent from the projection objectives'
        # `projection.auto_weighting`). When True the RC term is routed through the composite
        # auto-weighting (dedicated slot); when False it enters at its static `loss_weight`.
        "auto_weighting": False,
    },
    # Compounded-prediction deploy loss (CP) — the MTM-Pro second AR stage (RLRP-708,
    # Option C2). A free-running AR NLL of the single-step DEPLOY prediction vs the per-step
    # obs target: `CP = w_cp * (1/Z) * sum_k gamma^k * NLL(deploy(x_k), y_obs_k)`. It trains
    # ONLY the deploy path (the multistep forecast term is untouched) and is additionally
    # gated by the second-AR-stage toggles (`ar_enabled and ar_train_horizon_unroll`). When
    # SS is enabled, CP SUBSUMES the step-0 SS term (its k=0 term == the SS NLL) to avoid
    # double-counting. CP trains the projection head when `deploy_head.mode=projection`, and
    # the forecast/mixing path when `deploy_head.mode=forecast` (no extra knob). Distinct from
    # `rollout_consistency_loss` (a forecast<->projection divergence, not a target NLL).
    #   enable        -> add the CP term to the composite objective. Task 2a (RLRP-708): this
    #                    single switch ALSO drives the second-AR-stage gating for MTM-Pro
    #                    (the former flat ``ar_enabled`` / ``ar_train_horizon_unroll`` kwargs are
    #                    derived from it), so CP is controlled by ONE knob.
    #   loss_weight   -> static CP weight (applied with AND without auto-weighting).
    #   auto_weighting-> route CP through the composite uncertainty auto-weighting ("CP" slot);
    #                    when False it enters the sum at its static weight.
    #   horizon_len   -> CP unroll horizon, DECOUPLED from the MS forecast ``horizon_len`` /
    #                    ``unrol_len`` (Task 2c, RLRP-708). ``None`` -> fall back to the MS
    #                    forecast ``horizon_len`` (preserves the historical behaviour).
    # The remaining keys are the second-AR-stage scheduler / teacher-forcing / temporal-discount
    # knobs inherited from ``CompoundedPredictionMultiStepIterator`` (formerly flat ``ar_*`` /
    # ``teacher_forcing_*`` kwargs), grouped here so the MTM-Pro hydra config stays organised
    # (Task 2b, RLRP-708).
    #
    # Deploy-history drift residual (DH) sub-term (RLRP-731, action ``B1-cfg`` of the
    # feat_deploy_history_drift_residual_loss_plan_20260705.md `.junie` plan): a terminal-step
    # residual MSE scored INSIDE the CP deploy unroll (same trajectory, same sampled horizon,
    # same teacher-forcing schedule), computed in RAW (denormalized) obs space:
    #   `DH = w_dh * mean_i( (denorm(pred_F) - denorm(target_F))^2 )_i`.
    # DH inherits the CP gate (`_cp_active`) by construction: with CP OFF, DH is OFF.
    #   enable_history_drift_loss  -> add the DH sub-term to the composite objective (requires
    #                                 the CP term to be active). OFF by default (parity).
    #   history_drift_loss_weight  -> static DH weight (applied with AND without auto-weighting).
    #   history_drift_auto_weighting -> route DH through the composite auto-weighting ("DH" slot).
    #   history_drift_target_mode  -> `inline_denorm` (default): denormalize BOTH the prediction
    #                                 and the normalized target inside the loss (no-op under the
    #                                 asymmetric 'standard' normalizer, whose targets are already
    #                                 raw); `raw_passthrough`: consume the raw target threaded
    #                                 from the wrapper (RLRP-731 batch B4-raw, Option 1 —
    #                                 IMPLEMENTED; the obs-block raw target is threaded via
    #                                 `OneDTransitionRewardModelV2.loss`/`update`).
    "compounded_prediction_deploy_loss": {
        "enable": False,
        "loss_weight": 1.0,
        "auto_weighting": False,
        "horizon_len": None,
        "enable_history_drift_loss": False,
        "history_drift_loss_weight": 1.0,
        "history_drift_auto_weighting": False,
        "history_drift_target_mode": "inline_denorm",
        "ar_temporal_weights": 1.0,
        "ar_temporal_weights_start": 1.0,
        "ar_temporal_weights_warmup": 0,
        "ar_temporal_weights_ramp_stop": 0,
        "ar_unrol_len_probablity_decay_start": 0,
        "ar_unrol_len_probablity_decay_stop": 0,
        "ar_unrol_len_decay_method": "beta",
        "ar_unrol_len_start_horizon_len": 1,
        "teacher_forcing_decay_stop": 0,
        "teacher_forcing_decay_start": 0,
        "teacher_forcing_method": "always_off",
    },
    # Forecast-path teacher-forcing scheduler (RLRP-726). A DEDICATED scheduled-sampling
    # (Bengio et al., 2015) instance for the MTM-Pro MS *forecast* self-feed
    # (`state_history_update`), INDEPENDENT of the CP/deploy `teacher_forcing_*` knobs above. On a
    # "teacher" step the forecast unroll splices the ground-truth obs+act
    # (`target_HOxDoa[..., horizon_index, :]`) into the next window instead of the free-running
    # mixture sample. Default-OFF (`method="always_off"`, `decay_stop=0` -> `is_fully_disabled`),
    # so disabled runs are byte-for-byte unchanged; RC re-forwards / eval / deploy stay free-running.
    #   method        -> {"linear", "exponential", "always_on", "always_off"}.
    #   decay_start   -> warmup steps held at prob=1.0 before decay begins.
    #   decay_stop    -> ABSOLUTE stop step (>decay_start active, -1 always_on, 0 always_off).
    "forecast_teacher_forcing": {
        "method": "always_off",
        "decay_start": 0,
        "decay_stop": 0,
    },
    # Deploy-path training stabilization (RLRP-722) — three INDEPENDENT, OFF-by-default
    # mechanisms that reduce the non-stationarity the deploy-path heads (SS / CP / projection /
    # RC) experience while chasing the still-moving multistep (MS) forecast body. The YAML group
    # is intentionally nested into three readable sub-blocks; each sub-block is registered as its
    # own ``deploy_path_training.<sub>`` defaults entry and resolved with one
    # ``_resolve_loss_cfg_group(...)`` call (top-level-key validation per sub-block).
    #
    # (1) warmup — suppress the deploy-path terms (only those; the MS forecast term keeps
    #     training) for the first ``forecast_only_steps`` optimizer steps (batches), then enable
    #     them, optionally linearly ramping their static weight over ``ramp_steps``.
    "deploy_path_training.warmup": {
        "enable": False,
        "forecast_only_steps": 0,
        "ramp_steps": 0,
    },
    # (2) two_timescale_lr — give the forecast body a slower LR and the deploy heads a faster LR
    #     (single optimizer, two parameter groups; per-group weight decay too). Wired in
    #     ``optimizer_instantiation.change_optimizer`` via ``build_mtm_pro_param_groups``.
    "deploy_path_training.two_timescale_lr": {
        "enable": False,
        "body_lr_mult": 0.5,
        "head_lr_mult": 1.0,
        "body_wd_mult": 1.0,
        "head_wd_mult": 1.0,
    },
    # (3) ema_forecast_clone — maintain a second, non-trainable EMA copy of the forecast
    #     sub-network (``theta_ema <- d*theta_ema + (1-d)*theta``) with an optionally scheduled
    #     momentum ``d``. The deploy terms named in ``feed_to`` read this STABLE forecast instead
    #     of the live, fast-moving body (teacher-network / two-timescale-via-target idea).
    "deploy_path_training.ema_forecast_clone": {
        "enable": False,
        "momentum": 0.999,
        "momentum_schedule": {
            "kind": "none",  # none | linear | cosine
            "start": 0.99,
            "end": 0.9999,
            "horizon_steps": 0,
        },
        "feed_to": ["cp", "ss", "projection"],  # subset of {cp, ss, projection, rc}
    },
}


def _resolve_loss_cfg_group(
    name: str,
    provided: Optional[Union[Dict, omegaconf.DictConfig]],
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge a (possibly ``None`` / ``DictConfig``) nested config group with its defaults.

    Unknown keys raise so that config typos surface immediately instead of being silently
    ignored. Returns a plain ``dict`` with every default key present.
    """
    merged = dict(defaults)
    if provided is None:
        return merged
    provided = dict(provided)
    unknown = set(provided) - set(defaults)
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in '{name}' config group; "
            f"allowed keys: {sorted(defaults)}"
        )
    merged.update(provided)
    return merged


@dataclass()
class StateHistory:
    h_mean: Tensor
    h_logvar: Optional[Tensor] = None
    # RLRP-647 (Stage 3 — resampled-particle history update): frozen snapshot of
    # the original observed history ``s^H_τ`` taken at AR-loop entry. Stage 3
    # needs to slice ``s^H_τ[i:]`` at every AR step ``i`` to rebuild the kept
    # observed prefix (Stages 1 & 2 do not need this — they recurse on the
    # previous step's history). Default ``None`` keeps Stages 1 & 2 unaffected.
    h_obs_original: Optional[Tensor] = None


class _ProjectionNetworkHead(nn.Module):
    """Expressive, deploy-quality summary head q_theta: ``2O -> trunk -> {mean, logvar} -> O, O``.

    Permanent ensemble-aware deploy head. Introduced by Items 2 & 3 of the PME Projection-Head
    Consolidation `.junie` plan
    (`refactor_pme_projection_head_consolidation_ensemble_plan_20260622.md`).

    A ``_build_network``-style head (the capacity formerly placed in the GMS decoder, which is NOT
    deployed): an input projection, a stack of identity-preserving residual blocks (trunk), and two
    SEPARATE residual split heads producing the obs-only mean and log-variance. Since q_theta is the
    object actually emitted at deploy (the decoder p(y|z) is a training-time anchor only), making it
    expressive is the correct lever for the single-step prediction quality.

    ENSEMBLE-aware (Item 3): the head reuses :class:`EnsembleLinearLayer` /
    :class:`EnsembleResidualBlock` (per-member weights + elite forwarding), exactly like the
    encoder MS stacks, instead of the former non-ensembled ``nn.Linear`` (which shared a single
    weight across the ensemble dim). The forward keeps the ensemble dim as axis 0 and flattens any
    intermediate leading dims into the per-member batch axis so the ``EnsembleLinearLayer``
    ``x.matmul(weight)`` contract (``weight (E, in, out)``) holds.

    TUPLE I/O (Item 2): :meth:`forward` returns a ``(mean, logvar)`` tuple (obs-only, width O each)
    rather than the legacy concatenated width-2O tensor, removing the boundary ``cat``/``chunk``.
    The last linear of each residual block is zero-initialised (``zero_init_last``) so the block
    starts as an exact identity; :func:`zero_init_residual_blocks_` re-applies this after the
    model-level ``truncated_normal_init``.

    DROPOUT: the ``dropout`` probability is forwarded to the block-owned dropout of EVERY
    :class:`EnsembleResidualBlock` in BOTH the shared trunk AND the two mean/logvar split heads
    (a single shared ``projection.head_dropout`` key drives both regions). ``0.0`` -> disabled.
    """

    def __init__(
        self,
        num_members: int,
        obs_len: int,
        hid: int,
        num_layers: int,
        split_num_layers: int,
        activation_factory: Callable[[], nn.Module],
        make_bloc: Callable[[int], nn.Module],
        dropout: float = 0.0,
    ) -> None:
        # RLRP-768 (B4b): the trunk + both split heads are now built through the
        # per-region layer-bloc factory (`region="projection"`, default `dense`)
        # via the injected ``make_bloc`` builder, instead of a hard-coded
        # :class:`EnsembleResidualBlock`. Every bloc kind exposes the same
        # ``set_elite`` / ``toggle_use_only_elite`` contract, so elite forwarding
        # is unchanged.
        super().__init__()
        _o = obs_len
        self.num_members = num_members
        self.dropout = float(dropout)
        self.in_proj = EnsembleLinearLayer(num_members, 2 * _o, hid)
        self.in_act = activation_factory()
        self.trunk = nn.Sequential(*[make_bloc(hid) for _ in range(max(num_layers, 0))])

        def _split_head() -> nn.Sequential:
            layers: list = [make_bloc(hid) for _ in range(max(split_num_layers, 0))]
            layers.append(EnsembleLinearLayer(num_members, hid, _o))
            return nn.Sequential(*layers)

        self.mean_head = _split_head()
        self.logvar_head = _split_head()

        # Flat list of the top-level ensemble layers/blocks for elite forwarding. Each entry is an
        # `EnsembleLinearLayer` OR an `EnsembleResidualBlock` (the latter forwards the toggle to its
        # OWN two inner layers); none are nested, so iterating this list toggles every member-aware
        # layer EXACTLY once (iterating `self.modules()` would double-toggle the blocks' inner
        # layers and silently cancel the elite switch).
        self._elite_layers: list = [
            self.in_proj,
            *self.trunk,
            *self.mean_head,
            *self.logvar_head,
        ]

    def forward(self, mean: Tensor, logvar: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.cat([mean, logvar], dim=-1)  # (E, *, 2O)
        lead = x.shape[:-1]
        x = x.reshape(
            lead[0], -1, x.shape[-1]
        )  # (E, B', 2O), ensemble dim kept as axis 0
        h = self.trunk(self.in_act(self.in_proj(x)))
        q_mean = self.mean_head(h).reshape(*lead, -1)
        q_logvar = self.logvar_head(h).reshape(*lead, -1)
        return q_mean, q_logvar

    def set_elite(self, elite_models: Sequence[int]) -> None:
        for layer in self._elite_layers:
            layer.set_elite(elite_models)

    def toggle_use_only_elite(self) -> None:
        for layer in self._elite_layers:
            layer.toggle_use_only_elite()


def _canonicalize_quaternion_slots(
    x: Tensor, slots: Sequence[int], eps: float
) -> Tensor:
    """Project every 4-D attitude slot of ``x`` back onto the unit-quaternion double-cover.

    RLRP-736 §4A-B P3.1 (RLRP-738). A moment-matched weighted average of unit
    quaternions is generically NON-unit; a chordal-L2 projection (normalise) returns a
    valid rotation so the exposed / CP-re-injected mean lives on the manifold (Forster
    et al. 2016; Geist et al. 2024). Bit-exact no-op when ``slots`` is empty.

    The memoryless ``w >= 0`` sign step was removed by action B-3 of the Quaternion
    manifold upgrade + memoryless ``w >= 0`` removal `.junie` plan
    (``rlrp-744-quaternion-manifold-upgrade-and-memoryless-wge0-removal-plan-20260720.md``);
    hemisphere continuity is owned by the reference-relative
    ``align_quaternion_slots_to_reference`` pass (RLRP-736 Item 1), applied last.

    **Single-step precondition (RLRP-747 ticket TODO 2, audit of 2026-08-27).**
    ``x`` is NOT a composed multi-step observation. Both production call sites are
    :class:`_ManifoldMeanMixtureSameFamily` ``.mean`` / ``.sample``, which pass
    ``_attitude_out_slots = tuple(sorted(self._ss_ori_out_slots))`` — the
    SINGLE-STEP slot set — applied to a single-step-wide mixture mean / sample.
    Consequently ``len(slots) == 1`` in the robotic-3D configs. Would a caller ever
    hand over a composed window, it MUST pass the composed slot expansion
    ``k * singlestep_obs_len + b``; passing the single-step set against a composed
    tensor would silently canonicalise only the first horizon block. The historical
    ``s + 4 > width`` skip is preserved (it is a caller bug, kept non-fatal for
    bit-exact backward compatibility).

    **RLRP-747: the per-slot loop below is KEPT, on measurement.** The vectorized
    (gather -> normalise -> single scatter) form was implemented and then
    **reverted**: the benchmark
    (``.junie/ai_artifact/scripts/bench_rlrp747_orientation_vectorization.py``,
    which still carries both variants) measured it SLOWER on **all three**
    platforms, with a dispatched-op (kernel-launch) count that goes UP rather than
    down (``7 -> 11``). Speed-up factors, loop-relative (``> 1`` = vectorized wins):

    =============================  ==========  ==========  ==========  ====  ====
    shape (E x B*C x F)            MacBook     Orin CPU    Orin CUDA   loop  vec
                                   CPU         (arm64)     (sm_87)     ops   ops
    =============================  ==========  ==========  ==========  ====  ====
    5 x 1280 x 10   (S=1, prod)         0.66x       0.62x       0.66x     7    11
    5 x 5120 x 10   (S=1, prod)         0.45x       0.73x       0.70x     7    11
    5 x 1280 x 80   (S=8, hypoth.)      2.46x       2.98x       1.93x    56    25
    =============================  ==========  ==========  ==========  ====  ====

    Root cause: ``slots`` is the SINGLE-STEP set (see the precondition above), so
    ``S == 1`` — there is no loop to collapse, and the gather adds a copy on top of
    identical arithmetic. The ``S = 8`` row shows the crossover: gather-isation is a
    ~2-3x WIN once there really are several slots. Per the plan's ROI table this
    item was **benchmark-gated** and the gate closed it. The call FREQUENCY (once
    per mixture ``.mean`` / ``.sample``) does not change the verdict: a per-call
    regression multiplied by the same frequency is still a regression.
    """
    if not slots:
        return x
    out = x
    width = out.shape[-1]
    for s in sorted(slots):
        if s + 4 > width:
            continue
        q = out[..., s : s + 4]
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
        out = torch.cat([out[..., :s], q, out[..., s + 4 :]], dim=-1)
    return out


class _ManifoldMeanMixtureSameFamily(MixtureSameFamily):
    """A :class:`MixtureSameFamily` whose exposed ``.mean`` is manifold-consistent.

    RLRP-736 §4A-B P3.1 (RLRP-738; ★ compounded-error lever). For the MTM-Pro
    temporal mixture the moment-matched ``.mean`` is a weighted average of the
    component quaternions and is therefore generically NOT a unit quaternion. When
    the by-construction PROBABILISTIC attitude split is active, this subclass
    canonicalises each attitude output slot of the exposed mean back onto S^3
    (chordal-L2 projection; the memoryless ``w >= 0`` sign step was removed by
    RLRP-744 action B-3), so the value that is re-injected as
    ``next_obs`` in the free-running CP unroll (``_default_forward_mixture_dist`` ->
    ``ss_mean``) and consumed downstream stays a valid rotation and the per-step
    uncertainty propagates manifold-consistently (Forster et al. 2016; Geist et al.
    2024).

    It ALSO carries, on the mixture object, the per-slot 3-D ``so3``-tangent
    log-variance produced at COMPONENT-CONSTRUCTION time by the dedicated ``4 -> 3``
    head (``_attitude_tangent_log_var_by_slot``), so scoring consumes the tangent the
    component was built with rather than re-deriving it (P3.1: the head is moved OUT
    of the scoring-only path and INTO the component construction).

    Bit-exact OFF: this subclass is only instantiated when the by-construction
    mixture-orientation split is active; otherwise the plain
    :class:`torch.distributions.MixtureSameFamily` is used unchanged.
    """

    _attitude_out_slots: Tuple[int, ...] = ()
    _attitude_variance_eps: float = 1e-12
    _attitude_tangent_log_var_by_slot: Optional[Dict[int, Tensor]] = None

    @property
    def mean(self) -> Tensor:
        base_mean = super().mean
        return _canonicalize_quaternion_slots(
            base_mean, self._attitude_out_slots, self._attitude_variance_eps
        )

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        """Draw a mixture sample whose attitude slots are unit quaternions by construction.

        RLRP-736 §4A-B P3.1 residual (RLRP-738; ★ compounded-error lever). The mixture
        components live in 4-D quaternion space, so a raw ``MixtureSameFamily.sample`` draw
        is generically NON-unit in each attitude slot. This retracts every attitude output
        slot of the drawn sample back onto the unit-quaternion double-cover (chordal-L2
        projection, via :func:`_canonicalize_quaternion_slots`; the memoryless ``w >= 0``
        sign step was removed by RLRP-744 action B-3), so a
        SAMPLED prediction re-injected as ``next_obs`` in the free-running CP unroll is a
        valid rotation by construction rather than relying solely on the downstream
        ``_canonicalize_reinjected_quaternion`` guard (Forster et al. 2016; Geist et al.
        2024). Bit-exact OFF: this subclass is only built when the split is active; with no
        attitude slots the projection is a no-op.
        """
        raw_sample = super().sample(sample_shape)
        return _canonicalize_quaternion_slots(
            raw_sample, self._attitude_out_slots, self._attitude_variance_eps
        )


class MS2MS2SSArTemporalMixturePME(
    CompoundedPredictionMultiStepIterator,
    AbstractFeatureWeightedMultiStepMLP,
):
    """
    This class represents a multi-step temporal mixture predictive model with feature weighting,
    supporting both single-step and multi-step forecasting (Codename Particle-MTM-Pro).

    MAIN GOAL: MTM-Pro models the UNCERTAINTY of a multi-step forecast while MITIGATING the
    COMPOUNDED PREDICTION ERROR problem inherent to free-running auto-regressive (AR /
    compounded-prediction) unrolls, where each step is re-injected as the next input and small
    per-step errors accumulate over the horizon. It does so with a temporal mixture over particles
    (uncertainty forecast) plus dedicated single-step and compounded-prediction-deploy loss terms
    that regularise the free-running unroll.

    The `MS2MS2SSArTemporalMixturePME` class is designed for tasks that involve multi-step
    temporal predictions while incorporating feature weighting and ensemble modeling. It supports
    configurable activation functions, dropout mechanisms, and loss weighting, and it can operate
    with both deterministic and probabilistic distributions.

    """

    _GLOBAL_DEBUG = False  # <--

    # Numerical safety bounds for mixture log-variance.
    # The mixture variance (mean of component variances + between-component
    # variance) is unbounded and can overflow float32 when passed through
    # torch.log → _to_distribution → exp(logvar * 0.5).
    #   Lower variance floor: dtype-aware, finfo-derived ``precision_bounds.variance_floor``
    #       (``16*finfo.tiny`` -> ~1.9e-37 float32 / ~3.6e-307 float64, RLRP-750);
    #       applied at every log() call site.
    #   _MIX_LOGVAR_SAFE_MAX: ceiling for logvar to keep distribution scale
    #       numerically tractable (exp(20/2) = exp(10) ≈ 22026 as scale).
    # Consolidated in ``precision_bounds`` (``logvar_safe_max()`` / ``LOGVAR_SAFE_MAX``);
    # the class attribute is kept as a thin alias so subclasses / tests can still read
    # ``self._MIX_LOGVAR_SAFE_MAX``.
    _MIX_LOGVAR_SAFE_MAX = logvar_safe_max()

    _latest_deploy_ss_dist_mixture = None

    # RLRP-736 §4A-B P1.2 (Particle-MTM-Pro, ensemble size == 1): opt IN to the
    # by-construction PROBABILISTIC attitude head so ``_setup_orientation_rep`` no
    # longer fails loud on an active (non-``quaternion``) rep for this family.
    #
    # PURPOSE (why this family exists): MTM-Pro models the UNCERTAINTY of a
    # multi-step forecast while MITIGATING the COMPOUNDED PREDICTION ERROR problem
    # of free-running auto-regressive (AR / CP) unrolls. The by-construction
    # orientation wiring below serves that goal end-to-end: the attitude mean stays
    # ON the unit-quaternion manifold and its uncertainty is carried on the ``so3``
    # tangent, so error does not accumulate off-manifold as the mean is re-injected
    # as ``next_obs`` through the compounded-prediction unroll.
    #
    # LANDED (P1.2 + P3.1, ensemble size == 1): the shared MS/deploy head
    # (``_default_forward_u``) encodes the attitude input slot(s) to the internal
    # rep (``sixd``/``nine_d_svd``; Geist et al. 2024, Zhou et al. 2019) and DECODES
    # every per-unroll-block attitude prediction back to a unit quaternion. The
    # (log-)variance is carried on the 3-D ``so3`` tangent (Forster et al. 2016) via
    # the dedicated ``4→3`` head built at component-construction time
    # (``_build_component_attitude_tangent_log_var``, carried on the mixture as
    # ``_attitude_tangent_log_var_by_slot``) and scored by ``rotation_tangent_nll``
    # at the per-particle ``MixtureSameFamily``-component level
    # (``_apply_mixture_attitude_tangent_nll``). ``_temporal_mixture_distribution_at_i``
    # returns a ``_ManifoldMeanMixtureSameFamily`` whose ``.mean`` / ``.sample()``
    # canonicalise every attitude slot back onto S³. BIT-EXACT when the rep is OFF
    # (``quaternion``) — a plain ``MixtureSameFamily`` is returned. The Re-Particle
    # (raw ``mixture.sample()`` re-normalisation) and Distributional (doubled
    # ``[h_mean ‖ h_logvar]`` encoder input) variants also set this flag and add
    # their variant-specific AR-feedback seams.
    #
    # DEFERRED (Phase-2 / RLRP-738): the ensemble ``num_members > 1`` propagation
    # reshapes (tracked as P2.0). The numeric verdict is compute-gated (P2.4/P3.4).
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
        # RLRP-761 S3.4 / S3.5 -- see `AbstractFeatureWeightedMultiStepMLP`.
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
        # Each group is an optional dict (Hydra passes the nested YAML mapping). ``None`` -> the
        # group defaults in ``_LOSS_CFG_GROUP_DEFAULTS``. See that constant for the allowed keys.
        # NOTE (RLRP-711+, Stage C.1): ``projection.backprop_mode`` defaults to "mixing_only".
        # "mixing_only" detaches the mixture components so only the summary head + mixing function
        # train; "full_network" additionally trains the prediction body (validated working — it
        # shows proof of life when training); "head_only" trains only the summary head. All three
        # are first-class, selectable options.
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
        # Dedicated forecast-path teacher-forcing scheduler (RLRP-726), INDEPENDENT of the CP/deploy
        # `teacher_forcing_*` knobs nested in `compounded_prediction_deploy_loss`. Default-OFF.
        forecast_teacher_forcing: Optional[
            Union[Dict, omegaconf.DictConfig]
        ] = None,  # This the forecaster teacher forcing
        # Deploy-path training stabilization (RLRP-722): nested group with three independent,
        # OFF-by-default sub-blocks (``warmup`` / ``two_timescale_lr`` / ``ema_forecast_clone``).
        deploy_path_training: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        temporal_mixture_weights: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        ar_sample_clamp: Optional[float] = None,
        train_time_domain_randomization: Optional[
            Union[Dict, omegaconf.DictConfig]
        ] = None,
        receive_sequence_batch: bool = False,
        description: Optional[str] = None,
        feature_geometry=None,
        internal_orientation=None,
        orientation_singlestep_slots=None,
        # RLRP-830 Item D: route the per-u-forward unroll attitude decode of the
        # WIDTH-PRESERVING quaternion reps through the RLRP-747 gather/scatter
        # primitive (one stacked ``F.normalize`` instead of one per unroll slot).
        # Bit-exact with the generic ``_splice_slots`` path; ``False`` = legacy path
        # (kept in-tree for the parity tests).
        vectorized_unroll_orientation_decode: bool = True,
        # RLRP-830 Item C: score the U compounded-prediction (CP) deploy steps with ONE
        # stacked per-feature true-mixture NLL after the (sequential) unroll instead of
        # one NLL pass per step. The unroll itself and the ``gamma`` accumulation order
        # are unchanged; bit-exact with the per-step path. ``False`` = legacy path.
        # Known exception (Narval A100 sm_80 parity run 2026-09-19, FR4 option (a),
        # operator decision): ``manifold_aware_nll=True`` (so3-tangent NLL, OFF in the
        # paper configs) in fp64 on CUDA -> ONE trunk weight grad differs at ulp level
        # (fp64 reduction order of the stacked tangent NLL kernel); forward / deploy /
        # loss stay equal, fp32 CUDA and CPU fp64 are bit-exact. Kept fast-by-default.
        batched_cp_scoring: bool = True,
        # RLRP-830 Item B1 (OPT-IN, legacy-by-default per plan FR4): the ``true_mixture``
        # per-step MS NLL re-uses the ``MixtureSameFamily`` that
        # ``_default_forward_mixture_dist`` just built for the same horizon step instead
        # of building it a second time. Forward / deploy / loss are bit-exact, but the
        # gradient of the MIXER parameter (``temporal_mixture_weights.mixture_logits``)
        # is NOT: autograd then sums the two consumer gradients at the shared
        # normalised-logits node instead of at the parameter leaf (``J(g1+g2)`` vs
        # ``J g1 + J g2``), a ~1-ulp summation-order deviation (rel. 7e-8 fp32 / 7e-18
        # fp64 on the UAV paper fixture) that fails the RLRP-830 "bit-identical or
        # provably closer to exact" gate. Every other parameter gradient is bit-exact.
        reuse_step_mixture: bool = False,
        # RLRP-786 (CUDA-graph captured training step, plan
        # ``perf_RLRP-786_cuda_graph_train_step_plan_20260919.md``, FR6 / Key Decision 4): make
        # the forecast-path teacher-forcing splice graph-capturable. The per-horizon-step CPU
        # coin (``should_use_teacher_forcing()``, a Python ``if``) becomes a static ``(F,)`` bool
        # DEVICE mask refilled from the CPU scheduler once per training step
        # (:meth:`refill_forecast_tf_mask`, called from ``_probabilistic_loss`` on the eager
        # path and from the graph ``before_replay`` hook on the captured path) and consumed
        # with ``torch.where(mask[i], gt, pred)`` in the sampling-free ``state_history_update``
        # (selection, no arithmetic: ``torch.equal`` to the Python branch, same CPU RNG stream
        # -- F coins drawn in horizon order). ``False`` = today's Python branch (default).
        cuda_graph_capture_ready: bool = False,
    ):

        assert 1 <= unrol_len <= horizon_len
        self.unroll_len = unrol_len
        self.cuda_graph_capture_ready: bool = bool(cuda_graph_capture_ready)
        # RLRP-786: lazily allocated ``(horizon_len,)`` bool device buffer (see the kwarg doc).
        self._forecast_tf_mask: Optional[Tensor] = None
        self.vectorized_unroll_orientation_decode: bool = bool(
            vectorized_unroll_orientation_decode
        )
        self.batched_cp_scoring: bool = bool(batched_cp_scoring)
        self.reuse_step_mixture: bool = bool(reuse_step_mixture)
        # (horizon_index, obs_space_only, mixture) stash, valid within ONE forecast loop
        # iteration only (set by ``_default_forward_mixture_dist``, cleared in
        # ``_default_forward`` right after ``_accumulate_post_mix_losses``).
        self._current_step_mixture: Optional[Tuple[int, bool, MixtureSameFamily]] = None

        self.singlestep_obs_len = singlestep_obs_len
        self.singlestep_act_len = singlestep_act_len

        # .... Unpack the nested loss-term / projection / deploy config groups (RLRP-704) ..........
        # Each group dict is merged with its defaults and unpacked into the historical flat local
        # variables used by the rest of this constructor (keeps the body + ``self.*`` names stable).
        _ss = _resolve_loss_cfg_group(
            "single_step_loss",
            single_step_loss,
            _LOSS_CFG_GROUP_DEFAULTS["single_step_loss"],
        )
        enable_single_step_loss = _ss["enable"]
        ss_composite_loss_weight = _ss["loss_weight"]
        single_step_auto_weighting = _ss["auto_weighting"]

        _pre = _resolve_loss_cfg_group(
            "pre_mixture_u_loss",
            pre_mixture_u_loss,
            _LOSS_CFG_GROUP_DEFAULTS["pre_mixture_u_loss"],
        )
        enable_pre_mixture_u_loss = _pre["enable"]

        _postmix = _resolve_loss_cfg_group(
            "post_mixture_loss",
            post_mixture_loss,
            _LOSS_CFG_GROUP_DEFAULTS["post_mixture_loss"],
        )
        enable_post_mixture_loss = _postmix["enable"]
        post_mixture_backprop_mode = _postmix["backprop_mode"]
        post_mixture_loss_weight = _postmix["loss_weight"]

        _iproj = _resolve_loss_cfg_group(
            "info_projection_loss",
            info_projection_loss,
            _LOSS_CFG_GROUP_DEFAULTS["info_projection_loss"],
        )
        enable_info_projection_loss = _iproj["enable"]
        info_projection_num_samples = _iproj["num_samples"]
        info_projection_head_data_nll = _iproj["head_data_nll"]
        info_projection_loss_weight = _iproj["loss_weight"]

        _siw_mp = _resolve_loss_cfg_group(
            "siw_mp_loss",
            siw_mp_loss,
            _LOSS_CFG_GROUP_DEFAULTS["siw_mp_loss"],
        )
        enable_siw_mp_loss = _siw_mp["enable"]
        siw_mp_num_samples = _siw_mp["num_samples"]
        siw_mp_loss_weight = _siw_mp["loss_weight"]

        _gms = _resolve_loss_cfg_group(
            "gms_iwae_loss",
            gms_iwae_loss,
            _LOSS_CFG_GROUP_DEFAULTS["gms_iwae_loss"],
        )
        enable_gms_iwae_loss = _gms["enable"]
        gms_iwae_num_samples = _gms["num_samples"]
        gms_iwae_loss_weight = _gms["loss_weight"]
        # The decoder is itself a nested group; resolve it against its own defaults so partial
        # overrides (e.g. only `arch`) keep the remaining decoder keys.
        _gms_decoder = _resolve_loss_cfg_group(
            "gms_iwae_loss.decoder",
            _gms["decoder"],
            _LOSS_CFG_GROUP_DEFAULTS["gms_iwae_loss"]["decoder"],
        )

        _proj = _resolve_loss_cfg_group(
            "projection",
            projection,
            _LOSS_CFG_GROUP_DEFAULTS["projection"],
        )
        projection_backprop_mode = _proj["backprop_mode"]
        per_step_conditioning = _proj["per_step_conditioning"]
        # Shared auto-weighting toggle for the (mutually exclusive) IPROJ / SIW-MP / GMS-IWAE
        # projection objectives (formerly the per-objective siw_mp_loss.auto_weighting /
        # gms_iwae_loss.auto_weighting keys).
        projection_auto_weighting = _proj["auto_weighting"]
        projection_temporal_weights = _proj["temporal_weights"]
        projection_head_hidden = _proj["head_hidden"]
        projection_head_num_layers = _proj["head_num_layers"]
        projection_head_split_num_layers = _proj["head_split_num_layers"]
        projection_head_dropout = _proj["head_dropout"]
        # The projection head ``head_hidden`` lives on the shared ``projection`` group (the
        # IPROJ / SIW-MP / GMS-IWAE objectives are mutually exclusive and all train the SHARED
        # head q_theta, so the head width is a single shared setting).

        _tmw = _resolve_loss_cfg_group(
            "temporal_mixture_weights",
            temporal_mixture_weights,
            _LOSS_CFG_GROUP_DEFAULTS["temporal_mixture_weights"],
        )
        temporal_mixture_weights_kind = _tmw["kind"]
        temporal_mixture_weights_head_size = _tmw["head_size"]
        # RLRP-735: history-head trunk depth + block-owned dropout for the
        # ``index_and_history_dependent`` mixer (ignored by the other kinds).
        temporal_mixture_weights_num_layers = _tmw["num_layers"]
        temporal_mixture_weights_dropout = _tmw["dropout"]

        _deploy = _resolve_loss_cfg_group(
            "deploy_head",
            deploy_head,
            _LOSS_CFG_GROUP_DEFAULTS["deploy_head"],
        )
        deploy_head_mode = _deploy["mode"]

        _rc = _resolve_loss_cfg_group(
            "rollout_consistency_loss",
            rollout_consistency_loss,
            _LOSS_CFG_GROUP_DEFAULTS["rollout_consistency_loss"],
        )
        enable_rollout_consistency_loss = _rc["enable"]
        rollout_consistency_num_steps = _rc["num_steps"]
        rollout_consistency_backprop_mode = _rc["backprop_mode"]
        rollout_consistency_loss_weight = _rc["loss_weight"]
        # Dedicated RC auto-weighting toggle (independent from projection.auto_weighting).
        rollout_consistency_auto_weighting = _rc["auto_weighting"]

        _cp = _resolve_loss_cfg_group(
            "compounded_prediction_deploy_loss",
            compounded_prediction_deploy_loss,
            _LOSS_CFG_GROUP_DEFAULTS["compounded_prediction_deploy_loss"],
        )
        enable_compounded_prediction_deploy_loss = _cp["enable"]
        compounded_prediction_deploy_loss_weight = _cp["loss_weight"]
        compounded_prediction_auto_weighting = _cp["auto_weighting"]
        # Task 2c (RLRP-708): dedicated CP unroll horizon, decoupled from the MS forecast
        # ``horizon_len`` / ``unrol_len``. ``None`` -> fall back to the MS forecast ``horizon_len``
        # (preserves the historical behaviour). Passed to the iterator as its ``horizon_len`` so
        # the shared unroll helpers drive the CP unroll length off this value (the mixin U-loop is
        # bypassed for MTM-Pro, so this never affects the MS forecast).
        compounded_prediction_horizon_len = (
            _cp["horizon_len"] if _cp["horizon_len"] is not None else horizon_len
        )
        # The CP unroll scores the per-step obs target sliced from the MS forecast horizon
        # (``target_HOxDoa`` has ``horizon_len`` steps), so the CP horizon cannot exceed it.
        assert 1 <= compounded_prediction_horizon_len <= horizon_len, (
            f"compounded_prediction_deploy_loss.horizon_len must be in [1, horizon_len="
            f"{horizon_len}], got {compounded_prediction_horizon_len}"
        )
        # Task 2b (RLRP-708): second-AR-stage scheduler / teacher-forcing / temporal-discount
        # knobs are now nested under ``compounded_prediction_deploy_loss`` (formerly flat ``ar_*`` /
        # ``teacher_forcing_*`` kwargs). Unpack them into the historical flat locals so the
        # ``_setup_compounded_prediction_iterator(...)`` call below is unchanged.
        ar_temporal_weights = _cp["ar_temporal_weights"]
        ar_temporal_weights_start = _cp["ar_temporal_weights_start"]
        ar_temporal_weights_warmup = _cp["ar_temporal_weights_warmup"]
        ar_temporal_weights_ramp_stop = _cp["ar_temporal_weights_ramp_stop"]
        ar_unrol_len_probablity_decay_start = _cp["ar_unrol_len_probablity_decay_start"]
        ar_unrol_len_probablity_decay_stop = _cp["ar_unrol_len_probablity_decay_stop"]
        ar_unrol_len_decay_method = _cp["ar_unrol_len_decay_method"]
        ar_unrol_len_start_horizon_len = _cp["ar_unrol_len_start_horizon_len"]
        teacher_forcing_decay_stop = _cp["teacher_forcing_decay_stop"]
        teacher_forcing_decay_start = _cp["teacher_forcing_decay_start"]
        teacher_forcing_method = _cp["teacher_forcing_method"]
        # Deploy-history drift residual (DH) sub-term keys (RLRP-731): DH is part of the CP
        # logic/config group (operator decision, 2026-07-05) so it inherits the CP gate, the CP
        # sampled unroll length, the CP teacher-forcing schedule and the CP scheduler stepping.
        enable_history_drift_loss = _cp["enable_history_drift_loss"]
        history_drift_loss_weight = _cp["history_drift_loss_weight"]
        history_drift_auto_weighting = _cp["history_drift_auto_weighting"]
        history_drift_target_mode = _cp["history_drift_target_mode"]
        # Task 2a (RLRP-708): the MTM-Pro second AR stage no longer exposes separate
        # ``ar_enabled`` / ``ar_train_horizon_unroll`` knobs. Every MTM-Pro use of
        # ``ar_enabled and ar_train_horizon_unroll`` is CP-specific (deploy-splice stash,
        # deploy-mixture cache, the CP loss gate, the ``ar_memory`` deploy record), so both are
        # derived from the single ``compounded_prediction_deploy_loss.enable`` switch.
        ar_enabled = enable_compounded_prediction_deploy_loss
        ar_train_horizon_unroll = enable_compounded_prediction_deploy_loss

        # .... Forecast-path teacher-forcing scheduler config (RLRP-726) ..........................
        # DEDICATED scheduled-sampling scheduler for the MS forecast self-feed, INDEPENDENT of the
        # CP/deploy `teacher_forcing_*` knobs above. Default-OFF (always_off / decay_stop=0).
        _ftf = _resolve_loss_cfg_group(
            "forecast_teacher_forcing",
            forecast_teacher_forcing,
            _LOSS_CFG_GROUP_DEFAULTS["forecast_teacher_forcing"],
        )
        forecast_teacher_forcing_method = _ftf["method"]
        forecast_teacher_forcing_decay_start = _ftf["decay_start"]
        forecast_teacher_forcing_decay_stop = _ftf["decay_stop"]

        # .... Deploy-path training stabilization (RLRP-722) ......................................
        # Three INDEPENDENT, OFF-by-default mechanisms. The YAML group is nested into three
        # readable sub-blocks; each is resolved against its own ``deploy_path_training.<sub>``
        # defaults entry (top-level-key validation per sub-block).
        _dpt = dict(deploy_path_training) if deploy_path_training is not None else {}
        _dpt_unknown = set(_dpt) - {"warmup", "two_timescale_lr", "ema_forecast_clone"}
        if _dpt_unknown:
            raise ValueError(
                f"Unknown key(s) {sorted(_dpt_unknown)} in 'deploy_path_training' config group; "
                f"allowed keys: ['ema_forecast_clone', 'two_timescale_lr', 'warmup']"
            )
        # (1) Forecaster warmup.
        _dpt_warmup = _resolve_loss_cfg_group(
            "deploy_path_training.warmup",
            _dpt.get("warmup"),
            _LOSS_CFG_GROUP_DEFAULTS["deploy_path_training.warmup"],
        )
        self._deploy_warmup_enable = bool(_dpt_warmup["enable"])
        self._deploy_warmup_steps = int(_dpt_warmup["forecast_only_steps"])
        self._deploy_warmup_ramp_steps = int(_dpt_warmup["ramp_steps"])
        # (2) Two-timescale LR (consumed in optimizer_instantiation.change_optimizer).
        _dpt_tt = _resolve_loss_cfg_group(
            "deploy_path_training.two_timescale_lr",
            _dpt.get("two_timescale_lr"),
            _LOSS_CFG_GROUP_DEFAULTS["deploy_path_training.two_timescale_lr"],
        )
        self._two_timescale_enable = bool(_dpt_tt["enable"])
        self._two_timescale_body_lr_mult = float(_dpt_tt["body_lr_mult"])
        self._two_timescale_head_lr_mult = float(_dpt_tt["head_lr_mult"])
        self._two_timescale_body_wd_mult = float(_dpt_tt["body_wd_mult"])
        self._two_timescale_head_wd_mult = float(_dpt_tt["head_wd_mult"])
        # (3) EMA frozen-decaying MS-forecast clone.
        _dpt_ema = _resolve_loss_cfg_group(
            "deploy_path_training.ema_forecast_clone",
            _dpt.get("ema_forecast_clone"),
            _LOSS_CFG_GROUP_DEFAULTS["deploy_path_training.ema_forecast_clone"],
        )
        self._ema_forecast_enable = bool(_dpt_ema["enable"])
        # RLRP-722 phased rollout (plan §6): features (1) warmup + (2) two-timescale LR landed
        # first; this EMA frozen-decaying MS-forecast clone (3) is the gated follow-up. When
        # enabled, the deploy-path terms named in ``feed_to`` read a STABLE (slowly-varying)
        # forecast from a detached EMA copy of the forecast body+mixer instead of the live,
        # fast-moving body. OFF by default -> byte-for-byte identical to the prior behaviour.
        self._ema_forecast_momentum = float(_dpt_ema["momentum"])
        _ema_sched = _resolve_loss_cfg_group(
            "deploy_path_training.ema_forecast_clone.momentum_schedule",
            _dpt_ema["momentum_schedule"],
            _LOSS_CFG_GROUP_DEFAULTS["deploy_path_training.ema_forecast_clone"][
                "momentum_schedule"
            ],
        )
        self._ema_momentum_schedule_kind = str(_ema_sched["kind"])
        assert self._ema_momentum_schedule_kind in {"none", "linear", "cosine"}, (
            f"deploy_path_training.ema_forecast_clone.momentum_schedule.kind must be "
            f"'none', 'linear' or 'cosine', got {self._ema_momentum_schedule_kind}"
        )
        self._ema_momentum_schedule_start = float(_ema_sched["start"])
        self._ema_momentum_schedule_end = float(_ema_sched["end"])
        self._ema_momentum_schedule_horizon = int(_ema_sched["horizon_steps"])
        _ema_feed_to = list(_dpt_ema["feed_to"])
        _ema_feed_unknown = set(_ema_feed_to) - {"cp", "ss", "projection", "rc"}
        if _ema_feed_unknown:
            raise ValueError(
                f"Unknown deploy term(s) {sorted(_ema_feed_unknown)} in "
                f"'deploy_path_training.ema_forecast_clone.feed_to'; "
                f"allowed: ['cp', 'projection', 'rc', 'ss']"
            )
        self._ema_feed_to = set(_ema_feed_to)
        # The detached EMA parameter dict of the forecast sub-network; lazily initialised from the
        # live params on first use (so a fresh model and a checkpoint without the clone both start
        # ``theta_ema = theta_live``). Kept as a plain attribute (NOT an ``nn.Module`` / buffer) so
        # it never leaks into ``self.parameters()`` / the optimizer.
        self._ema_forecast: Optional[Dict[str, Tensor]] = None
        # One-shot guard so the (lazy) clone init / post-load device-dtype alignment runs once, not
        # on every ``_ema_forecast_weights`` swap + ``_update_ema_forecast`` call within a batch.
        self._ema_forecast_initialized: bool = False
        # Per-batch training-step counter (drives warmup gating + EMA momentum schedule). Persisted
        # across checkpoints (see ``_save_state_dict`` / ``_load_state_dict``).
        self._train_step_count: int = 0

        self.enable_pre_mixture_u_loss = enable_pre_mixture_u_loss

        # .... Post-mixing / projection loss options (Stage B/G/H/I) ..............................
        assert ms_probabilistic_loss_mode in {"true_mixture", "moment_matched"}, (
            f"ms_probabilistic_loss_mode must be 'true_mixture' or 'moment_matched', "
            f"got {ms_probabilistic_loss_mode}"
        )
        self.ms_probabilistic_loss_mode = ms_probabilistic_loss_mode

        assert post_mixture_backprop_mode in {"full_network", "mixing_only"}, (
            f"post_mixture_backprop_mode must be 'full_network' or 'mixing_only', "
            f"got {post_mixture_backprop_mode}"
        )
        self.enable_post_mixture_loss = enable_post_mixture_loss
        self.post_mixture_backprop_mode = post_mixture_backprop_mode
        # Static (config) weight applied to the MIX term BEFORE the automatic weighting. It scales
        # the term in BOTH modes (auto-weighting on AND off), so the auxiliary objectives keep a
        # usable magnitude lever even when `enable_auto_loss_weighting=False` (integration fix 2b).
        self.post_mixture_loss_weight = float(post_mixture_loss_weight)

        self.enable_info_projection_loss = enable_info_projection_loss
        self.info_projection_num_samples = int(info_projection_num_samples)
        self.projection_head_hidden = int(projection_head_hidden)
        # Deployable summary head q_theta (RLRP-704): the capacity formerly placed in the
        # (non-deployed) GMS decoder now lives here, on the object actually emitted at deploy. The
        # head is always the ensemble-aware residual `_ProjectionNetworkHead` (the legacy `mlp`
        # `head_arch` selector was removed by the PME Projection-Head Consolidation `.junie` plan,
        # `refactor_pme_projection_head_consolidation_ensemble_plan_20260622.md`).
        self.projection_head_num_layers = int(projection_head_num_layers)
        self.projection_head_split_num_layers = int(projection_head_split_num_layers)
        # SHARED dropout probability applied inside every residual block of BOTH the projection
        # head trunk AND the mean/logvar split heads (a single `projection.head_dropout` key drives
        # both regions, mirroring the block-owned dropout of `EnsembleResidualBlock`).
        self.projection_head_dropout = float(projection_head_dropout)
        self.info_projection_head_data_nll = info_projection_head_data_nll
        self.info_projection_loss_weight = float(info_projection_loss_weight)

        self.enable_siw_mp_loss = enable_siw_mp_loss
        self.siw_mp_num_samples = int(siw_mp_num_samples)
        self.siw_mp_loss_weight = float(siw_mp_loss_weight)

        # .... Generative Mode-Seeking IWAE (GMS-IWAE) options ....................................
        # GMS-IWAE samples from the head q_theta (proposal) and importance-weights against a
        # generative decoder p(y|z) * mixture; it is mode-seeking + generative (the diagonal
        # opposite of the mass-covering SIW-MP). See `_gms_iwae_loss`.
        self.enable_gms_iwae_loss = bool(enable_gms_iwae_loss)
        self.gms_iwae_num_samples = int(gms_iwae_num_samples)
        self.gms_iwae_loss_weight = float(gms_iwae_loss_weight)
        # Decoder p(y|z) sub-config (see `_build_gms_decoder` / `_make_decoder_dist`). The decoder
        # is intentionally a FIXED identity observation-noise model (mean = z): it is a training-time
        # generative anchor and is NOT deployed, so making it learned would only turn z into a
        # non-deployable latent. Deployable expressiveness lives on the projection head q_theta
        # (`projection.head_arch`). Only the per-obs-feature noise scale is configurable / learnable.
        self.gms_iwae_decoder_obs_noise = float(_gms_decoder["obs_noise"])
        self.gms_iwae_decoder_learnable_noise = bool(_gms_decoder["learnable_noise"])

        # Mutual-exclusion guard (plan §14.1): GMS-IWAE, SIW-MP and IPROJ all optimise the SAME
        # shared projection head q_theta with conflicting objectives (mode-seeking+generative vs
        # mass-covering vs mode-seeking reverse-KL), so co-enabling them makes the head an
        # unspecified compromise. Enable exactly one projection objective.
        if self.enable_gms_iwae_loss and (
            self.enable_siw_mp_loss or self.enable_info_projection_loss
        ):
            raise ValueError(
                "gms_iwae_loss is mutually exclusive with siw_mp_loss and info_projection_loss: "
                "GMS-IWAE is mode-seeking + generative (KL(q||p_mix) with a decoder), SIW-MP is "
                "mass-covering (KL(p_mix||q)), and IPROJ is mode-seeking reverse-KL on the SAME "
                "shared projection head; co-enabling them makes that head an unspecified "
                "compromise. Enable exactly one projection objective."
            )

        # Backprop-scope lever shared by the IPROJ and SIWAE objectives (integration fix 2a),
        # mirroring `post_mixture_backprop_mode`:
        #   full_network -> the learned obs-only mixture trains the whole prediction body (default,
        #                   current behavior);
        #   mixing_only  -> the mixture COMPONENTS are detached so only the dedicated summary head
        #                   q_theta and the mixing function receive gradient from these terms.
        #   head_only    -> ALSO detaches the mixer log-weights so ONLY the dedicated summary head
        #                   q_theta (and the GMS decoder params) train (distil a frozen forecast +
        #                   mixer into the deploy head). Strongest forecast-protection setting.
        assert projection_backprop_mode in {
            "full_network",
            "mixing_only",
            "head_only",
        }, (
            f"projection_backprop_mode must be 'full_network', 'mixing_only' or 'head_only', "
            f"got {projection_backprop_mode}"
        )
        self.projection_backprop_mode = projection_backprop_mode
        self._projection_detach_components = projection_backprop_mode in {
            "mixing_only",
            "head_only",
        }
        # head_only additionally detaches the categorical mixer log-weights of the per-step
        # mixtures (see `_build_obs_only_temporal_mixtures` / `_detach_mixture_weights`).
        self._projection_detach_mixing_weights = projection_backprop_mode == "head_only"

        # .... SIWAE deploy-compression & anti-compounding options (RLRP-711+) ....................
        # Stage A: make q_theta a true per-step mixture-compressor.
        self.per_step_conditioning = bool(per_step_conditioning)
        # Unified `projection.temporal_weights` lever: a single float gamma. The boolean
        # `temporal_weighting` toggle was removed — the default 1.0 means DISABLED (uniform / flat
        # mean), and any value != 1.0 enables the gamma^i horizon discount. We derive the internal
        # `siw_mp_temporal_weighting` flag from whether a non-trivial (non-1.0 scalar) discount was
        # requested (a per-step profile sequence always enables it).
        if isinstance(projection_temporal_weights, (int, float)):
            self.projection_temporal_weights = float(projection_temporal_weights) != 1.0
        else:
            self.projection_temporal_weights = True
        # Dedicated SIWAE/RC horizon-discount. This is intentionally SEPARATE from the main
        # `temporal_weights` (which discounts the multi-step *forecast* NLL): the SIWAE/RC reduction
        # discounts the per-step projection/rollout-consistency terms, which serve a different
        # purpose. It is indexed on the HORIZON / per-step axis (NOT the ensemble num_members axis):
        # a scalar gamma (applied as gamma^i) or a per-horizon-step profile of length `horizon_len`.
        # Stashed here; the tensor is built after `super().__init__` (needs horizon_len/device/dtype).
        self._projection_temporal_weights_cfg = projection_temporal_weights

        # Stage B: deployable compressed prediction. ``forecast`` (the moment-matched / true-mixture
        # forecast output, formerly "moment_matched") is the safe default; ``projection`` routes
        # through the single trained projection head. The ``deploy_head.source`` selector was
        # removed: the projection objectives (GMS-IWAE / SIW-MP / IPROJ) are mutually exclusive, so
        # there is never more than one trained head to choose from.
        assert deploy_head_mode in {
            "forecast",
            "projection",
        }, f"deploy_head_mode must be 'forecast' or 'projection', got {deploy_head_mode}"
        self.deploy_head_mode = deploy_head_mode

        # Stage C: shared optional auto-weighting of the (mutually exclusive) projection objectives
        # (IPROJ / SIW-MP / GMS-IWAE). When False they enter the composite sum at their static
        # `loss_weight`; when True the active term is routed through a dedicated auto-weight slot.
        self.projection_auto_weighting = bool(projection_auto_weighting)

        # Stage D: rollout self-consistency distillation.
        # `backprop_mode` mirrors the shared `projection.backprop_mode` lever so RC can distil into
        # the deploy head with the same forecast-protection scopes:
        #   full_network -> the re-rolled forward output trains the whole prediction body;
        #   mixing_only  -> the re-rolled forward output is detached so RC trains the head and the
        #                   mixing function (but not the prediction body);
        #   head_only    -> ALSO detaches the teacher mixtures' mixer log-weights so ONLY the
        #                   deploy head q_theta trains (distil a frozen forecast + mixer).
        self.enable_rollout_consistency_loss = bool(enable_rollout_consistency_loss)
        self.rollout_consistency_num_steps = int(rollout_consistency_num_steps)
        assert rollout_consistency_backprop_mode in {
            "full_network",
            "mixing_only",
            "head_only",
        }, (
            f"rollout_consistency_backprop_mode must be 'full_network', 'mixing_only' or "
            f"'head_only', got {rollout_consistency_backprop_mode}"
        )
        self.rollout_consistency_backprop_mode = rollout_consistency_backprop_mode
        self._rollout_consistency_detach_components = (
            rollout_consistency_backprop_mode in {"mixing_only", "head_only"}
        )
        self._rollout_consistency_detach_mixing_weights = (
            rollout_consistency_backprop_mode == "head_only"
        )
        # The RC divergence geometry is NOT a free knob: it is auto-derived from the (mutually
        # exclusive) active projection objective so RC never pulls the shared head q_theta against
        # the primary objective's geometry. GMS-IWAE -> generative mode-seeking; SIW-MP ->
        # mass-covering; IPROJ or no projection objective -> mode-seeking reverse-KL.
        if self.enable_gms_iwae_loss:
            self.rollout_consistency_divergence = "gms_iwae"
        elif self.enable_siw_mp_loss:
            self.rollout_consistency_divergence = "siw_mp"
        else:
            self.rollout_consistency_divergence = "kl"
        self.rollout_consistency_loss_weight = float(rollout_consistency_loss_weight)
        # Dedicated RC auto-weighting toggle (independent from projection.auto_weighting): RC may
        # run while no projection objective is enabled, so it gets its own knob + dedicated slot.
        self.rollout_consistency_auto_weighting = bool(
            rollout_consistency_auto_weighting
        )

        # .... Compounded-prediction deploy loss (CP) — second AR stage (RLRP-708) ................
        self.enable_compounded_prediction_deploy_loss = bool(
            enable_compounded_prediction_deploy_loss
        )
        self.compounded_prediction_deploy_loss_weight = float(
            compounded_prediction_deploy_loss_weight
        )
        self.compounded_prediction_auto_weighting = bool(
            compounded_prediction_auto_weighting
        )

        # .... Deploy-history drift residual (DH) — CP sub-term (RLRP-731) ........................
        # Terminal-step raw-space residual MSE scored inside the CP unroll (see
        # ``_compounded_deploy_nll``). Gated by ``_cp_active and enable_history_drift_loss``.
        self.enable_history_drift_loss = bool(enable_history_drift_loss)
        self.history_drift_loss_weight = float(history_drift_loss_weight)
        self.history_drift_auto_weighting = bool(history_drift_auto_weighting)
        assert history_drift_target_mode in {"inline_denorm", "raw_passthrough"}, (
            f"compounded_prediction_deploy_loss.history_drift_target_mode must be "
            f"'inline_denorm' or 'raw_passthrough', got {history_drift_target_mode}"
        )
        # RLRP-731 batch ``B4-raw``: ``raw_passthrough`` (Option 1) is now implemented — the
        # obs-block RAW target is threaded from ``OneDTransitionRewardModelV2`` down to
        # ``_compounded_deploy_nll`` (see ``loss`` / ``update`` overrides + ``_probabilistic_loss``
        # ``target_raw_HOxDoa`` plumbing). The narrower ``target_is_delta`` scope guard lives in
        # the wrapper's ``_process_batch`` (a normalized-space delta has no ``next_obs`` raw
        # counterpart). MTM-Pro-family-only support (non-family models never see the kwarg).
        self.history_drift_target_mode = history_drift_target_mode

        assert temporal_mixture_weights_kind in {
            "index",
            "input_dependent",
            "index_and_history_dependent",
            "additive_index_and_history_dependent",
            "input_and_index_dependent",
            "additive_index_input_and_history_dependent",
        }, (
            f"temporal_mixture_weights.kind must be 'index', 'input_dependent', "
            f"'index_and_history_dependent', 'additive_index_and_history_dependent', "
            f"'input_and_index_dependent' or 'additive_index_input_and_history_dependent', got "
            f"{temporal_mixture_weights_kind}"
        )
        self.temporal_mixture_weights_kind = temporal_mixture_weights_kind
        # Hidden width of the input_dependent / index_and_history_dependent mixer head (used only
        # with those kinds).
        self.temporal_mixture_weights_head_size = int(
            temporal_mixture_weights_head_size
        )
        # RLRP-735: history-head trunk depth + block-owned dropout, consumed ONLY by the
        # index_and_history_dependent mixer (index / input_dependent ignore them -> parity).
        self.temporal_mixture_weights_num_layers = int(
            temporal_mixture_weights_num_layers
        )
        self.temporal_mixture_weights_dropout = float(temporal_mixture_weights_dropout)
        # Input-dependent mixers (this variant + the transformer subclasses) derive their mixing
        # logits from the per-step component tokens, so they must receive the NON-detached
        # full-width tokens (see `_temporal_mixture_distribution_at_i`).
        self._mixer_is_input_dependent = (
            temporal_mixture_weights_kind == "input_dependent"
        )
        # RLRP-735: index + input-history conditioned mixer. Its logits come from an
        # ensemble-aware residual head over the frozen ``s_τ^H`` realisation (per-batch output).
        # Covers BOTH the pure-history head (``index_and_history_dependent``) and the P3 additive
        # index-base variant (``additive_index_and_history_dependent``): both need the SAME frozen
        # ``s_τ^H`` token + ψ capture / per-batch routing machinery (the additive variant only adds
        # a zero-init index-γ base on top of the identical history head).
        # Also covers the combined ``additive_index_input_and_history_dependent`` variant (RLRP-735
        # P3 combined): it reuses the SAME frozen ``s_τ^H`` token + ψ capture / per-batch routing
        # machinery, and additionally adds a per-step input head ``ρ`` (see
        # ``_mixer_uses_input_tokens``).
        self._mixer_is_index_and_history_dependent = temporal_mixture_weights_kind in {
            "index_and_history_dependent",
            "additive_index_and_history_dependent",
            "additive_index_input_and_history_dependent",
        }
        # RLRP-735: index + input conditioned mixer (pure step-indexed γ + per-step input head ρ,
        # NO frozen-history ψ), so it needs the per-step input tokens but none of the ``s_τ^H``
        # capture machinery.
        self._mixer_is_input_and_index_dependent = (
            temporal_mixture_weights_kind == "input_and_index_dependent"
        )
        # Helper flag: mixers that consume the per-step ``cat([mean, logvar])`` input tokens.
        # Covers the standalone input mixer, the index+input mixer, and the combined
        # index+input+history mixer. Drives the ``input_tokens`` build in
        # ``_temporal_mixture_distribution_at_i``.
        self._mixer_uses_input_tokens = (
            self._mixer_is_input_dependent
            or self._mixer_is_input_and_index_dependent
            or temporal_mixture_weights_kind
            == "additive_index_input_and_history_dependent"
        )
        # Helper flag: mixers whose logits are PER-BATCH (E x B x k) rather than the index
        # mixer's batch-agnostic (E x k). Drives BOTH conditionals in
        # ``_temporal_mixture_distribution_at_i`` (the mixer call AND the ``.expand`` guard) so a
        # per-batch mixer is never re-broadcast over B (review item 1 / R8).
        self._mixer_is_per_batch = (
            self._mixer_is_input_dependent
            or self._mixer_is_index_and_history_dependent
            or self._mixer_is_input_and_index_dependent
        )

        # Double-count guard: true_mixture primary loss and a full-network post-mixture (MIX)
        # term are the same signal on the primary path (see plan Stage G.4).
        if (
            self.ms_probabilistic_loss_mode == "true_mixture"
            and self.enable_post_mixture_loss
            and self.post_mixture_backprop_mode == "full_network"
        ):
            raise ValueError(
                "Refusing to double-count: ms_probabilistic_loss_mode='true_mixture' and "
                "enable_post_mixture_loss with post_mixture_backprop_mode='full_network' optimise "
                "the same true-mixture signal on the primary path. Use post_mixture_backprop_mode="
                "'mixing_only' or set ms_probabilistic_loss_mode='moment_matched'."
            )

        # Per-loss accumulators / targets (allocated/reset each `_probabilistic_loss` call).
        self._post_mixture_loss_accumulator: Tensor | None = None
        # Per-horizon-step list of per-feature (obs+action) true-mixture NLLs (each (E x B x O+A)).
        # Kept per step (instead of a pre-summed scalar) so the `true_mixture` primary loss can reuse
        # the same obs-feature-weight / temporal-discount / horizon-reduction pipeline as the legacy
        # `moment_matched` path.
        self._ms_true_mixture_loss_steps: Optional[list] = None
        self._post_mixture_loss_target: Tensor | None = None
        # RLRP-726: ground-truth (E x B x Ho x O+A) target stashed by `_probabilistic_loss` before
        # the forecast forward so `state_history_update` can splice obs+act on "teacher" steps.
        # Train-only side-channel (mirrors `_post_mixture_loss_target`): cleared at the loss tail
        # and nulled during the RC re-forwards so the rollout-consistency term stays free-running.
        self._forecast_tf_target: Tensor | None = None
        # RLRP-735: per-forward frozen input-history token ``s_τ^H`` and its once-computed head
        # output ``ψ`` for the index_and_history_dependent mixer (D4-a). Both are (re)built at
        # forecast entry in ``setup_state_history`` (so RC re-forwards naturally rebuild them
        # from the rolled window) and nulled at the loss tail. Only populated when
        # ``self._mixer_is_index_and_history_dependent`` (flag-gated -> other kinds stay
        # allocation-free; review item 3).
        self._forecast_history_token: Tensor | None = None
        self._forecast_history_psi: Tensor | None = None

        if unrol_len > 1:
            self.compose_obs_unroll_len = compute_multistep_model_out_size(
                singlestep_obs_len=self.singlestep_obs_len,
                singlestep_act_len=self.singlestep_act_len,
                multistep_len=self.unroll_len,
                learned_reward=False,
            )
        else:
            self.compose_obs_unroll_len = (
                self.singlestep_obs_len + self.singlestep_act_len
            )

        super().__init__(
            in_size,
            out_size,
            device,
            singlestep_obs_len=singlestep_obs_len,
            singlestep_act_len=singlestep_act_len,
            history_len=history_len,
            horizon_len=horizon_len,
            temporal_weights=temporal_weights,
            obs_feature_weights=obs_feature_weights,
            act_feature_weights=act_feature_weights,
            feature_weight_mode=feature_weight_mode,
            feature_weight_max_ratio=feature_weight_max_ratio,
            num_layers=num_layers,
            ensemble_size=ensemble_size,
            hid_size=hid_size,
            deterministic=False,
            propagation_method=propagation_method,
            learn_logvar_bounds=learn_logvar_bounds,
            logvar_bound_grad_clip=logvar_bound_grad_clip,
            auto_weighting_noise_model=auto_weighting_noise_model,
            auto_weighting_scheme=auto_weighting_scheme,
            activation_fn_cfg=activation_fn_cfg,
            distribution_name=distribution_name,
            mae_loss=mae_loss,
            model_use_double_precision=model_use_double_precision,
            dropout=dropout,
            ms_head_dropout=ms_head_dropout,
            ms_head_num_layers=ms_head_num_layers,
            residual_form=residual_form,
            layer_bloc=layer_bloc,
            ms_temporal_weighting_mode=ms_temporal_weighting_mode,
            enable_auto_loss_weighting=enable_auto_loss_weighting,
            ms_probabilities_reduction=ms_probabilities_reduction,
            ms_energy_beta=ms_energy_beta,
            ms_energy_axis=ms_energy_axis,
            train_time_domain_randomization=train_time_domain_randomization,
            description=description,
            feature_geometry=feature_geometry,
            internal_orientation=internal_orientation,
            orientation_singlestep_slots=orientation_singlestep_slots,
        )

        # AR-sample clamp ceiling (normalized space). ``None`` -> dtype-aware,
        # finfo-derived default (RLRP-750: ``0.1 * sqrt(finfo(dtype).max)``, i.e.
        # ~1.8e18 for float32 / ~1.3e153 for float64 -- an overflow-only guard, no
        # longer the historical tight ``1e2`` float32 / ``1e6`` float64 bias). A
        # non-None value overrides it (config knob). See
        # ``precision_bounds.ar_sample_clamp_ceil`` and ``_resolve_ar_sample_clamp``.
        if ar_sample_clamp is not None:
            self.ar_sample_clamp = ar_sample_clamp
        else:
            self.ar_sample_clamp = ar_sample_clamp_ceil(self.model_dtype)

        # A2 (RLRP-783): transient per-call stash of diagnostic ``meta`` scalars kept ON
        # DEVICE until a single flush. Introduced by action ``A2`` of the RLRC MTM-Pro models
        # code optimization `.junie` plan
        # (``perf_RLRP-783_mtm_pro_models_code_optimization_plan_20260827.md``). NOT an
        # ``nn.Module`` buffer -- it is transient, per-``_probabilistic_loss``-call state.
        self._pending_meta_scalars: Dict[str, Tensor] = {}

        # Build the dedicated SIWAE/RC horizon-discount tensor now that `super().__init__` has set
        # `self.horizon_len`, `self.device` and `self.model_dtype`. Kept separate from the main
        # forecast `self._temporal_weights` (different purpose; see `__init__` doc above).
        #
        # IMPORTANT: this discount lives on the HORIZON / per-step axis (the same axis along which
        # the composed-next-obs array packs its timesteps, see
        # `AbstractTemporalyWeightedMultiStepMLP._init_discount_factors_on_horizon_slice`), NOT on
        # the ensemble (`num_members`) axis. The SIWAE/RC reduction averages over a single shared
        # horizon profile, so it is indexed by horizon step:
        #   - a scalar gamma is applied as gamma^i over the `horizon_len` steps;
        #   - a sequence must therefore have length `horizon_len` (one weight per horizon step).
        _siwae_tw = self._projection_temporal_weights_cfg
        if isinstance(_siwae_tw, (float, int)):
            # Single discount factor gamma; expanded to gamma^i in `_horizon_step_temporal_weights`.
            _siwae_tw = (float(_siwae_tw),)
        else:
            assert isinstance(
                _siwae_tw, Sequence
            ), f"siwae_temporal_weights must be a float or a Sequence, got {type(_siwae_tw)}"
            assert len(_siwae_tw) == self.horizon_len, (
                f"siwae_temporal_weights is a per-horizon-step profile and must match horizon_len "
                f"(the composed-obs / per-step axis, NOT the ensemble size num_members), currently "
                f"{_siwae_tw=} (len {len(_siwae_tw)}) != {self.horizon_len=}"
            )
            _siwae_tw = tuple(float(g) for g in _siwae_tw)
        assert np.all(0.0 <= np.array(_siwae_tw)), (
            f"All gamma values passed to param `siwae_temporal_weights` "
            f"must be 0 <= gamma, currently {_siwae_tw=}"
        )
        self._projection_temporal_weights = torch.tensor(
            _siwae_tw, device=self.device, dtype=self.model_dtype
        )

        # Note: `TemporalMixtureWeighs` is instantiated AFTER `super().__init__(..., device, ...)`
        # which has already moved the parent's params to `self.device`. Its `nn.Parameter`
        # defaults to CPU unless explicitly moved, causing a CUDA↔CPU device mismatch in
        # `MixtureSameFamily.mean` during forward on CUDA devices. Move it to `self.device`.
        if self.temporal_mixture_weights_kind == "input_dependent":
            # Input-dependent mixer: an ensemble-aware residual MLP over per-step tokens
            # cat([mean, logvar]) of full O+A width (matches the transformer-mixer token
            # convention). RLRP-735 refactor: same `head_size` / `num_layers` / `dropout` head
            # pattern as the index+history mixer (zero-init out_proj -> uniform warm start), built
            # on CPU then moved to `self.device`.
            _mixture_inp_dim = 2 * (self.singlestep_obs_len + self.singlestep_act_len)
            self.temporal_mixture_weights = InputDependentTemporalMixtureWeighs(
                inputdim=_mixture_inp_dim,
                ensemble_size=self.num_members,
                head_size=self.temporal_mixture_weights_head_size,
                num_layers=self.temporal_mixture_weights_num_layers,
                dropout=self.temporal_mixture_weights_dropout,
                activation_factory=lambda: create_activation_(activation_fn_cfg),
                dtype=self.model_dtype,
            ).to(self.device)
        elif self._mixer_is_index_and_history_dependent:
            # RLRP-735: index + input-history conditioned mixer. Its ensemble-aware residual head
            # ingests the frozen ``s_τ^H`` realisation — the flat ``model_input`` (means-only per
            # D2), width ``D_h = (O+A) * history_len``. The head is built on CPU (its
            # `EnsembleLinearLayer` / `EnsembleResidualBlock` take NO device arg, unlike
            # `nn.Linear`) then moved to `self.device` (review item 4), exactly like the other
            # branches. P3 (``additive_index_and_history_dependent``) uses the subclass that adds a
            # zero-init index-γ base on top of the SAME history head (byte-for-byte init parity).
            _history_token_dim = (
                self.singlestep_obs_len + self.singlestep_act_len
            ) * self.history_len
            if (
                self.temporal_mixture_weights_kind
                == "additive_index_input_and_history_dependent"
            ):
                # RLRP-735 P3 combined: γ index base + ψ history head + ρ per-step input head.
                # Passes BOTH the history args AND the input-head args (input_token_dim matches
                # the input-dependent branch's cat([mean, logvar]) width 2*(O+A)); the shared
                # `head_size` / `num_layers` / `dropout` knobs drive BOTH heads (P-plan B.4).
                _mixture_inp_dim = 2 * (
                    self.singlestep_obs_len + self.singlestep_act_len
                )
                self.temporal_mixture_weights = (
                    AdditiveIndexInputAndHistoryDependentTemporalMixtureWeighs(
                        unrol_len=self.unroll_len,
                        horizon_len=self.horizon_len,
                        ensemble_size=self.num_members,
                        history_token_dim=_history_token_dim,
                        head_size=self.temporal_mixture_weights_head_size,
                        num_layers=self.temporal_mixture_weights_num_layers,
                        dropout=self.temporal_mixture_weights_dropout,
                        activation_factory=lambda: create_activation_(
                            activation_fn_cfg
                        ),
                        dtype=self.model_dtype,
                        input_token_dim=_mixture_inp_dim,
                        input_head_size=self.temporal_mixture_weights_head_size,
                        input_num_layers=self.temporal_mixture_weights_num_layers,
                        input_dropout=self.temporal_mixture_weights_dropout,
                    ).to(self.device)
                )
            else:
                _mixer_cls = (
                    AdditiveIndexAndHistoryDependentTemporalMixtureWeighs
                    if self.temporal_mixture_weights_kind
                    == "additive_index_and_history_dependent"
                    else IndexAndHistoryDependentTemporalMixtureWeighs
                )
                self.temporal_mixture_weights = _mixer_cls(
                    unrol_len=self.unroll_len,
                    horizon_len=self.horizon_len,
                    ensemble_size=self.num_members,
                    history_token_dim=_history_token_dim,
                    head_size=self.temporal_mixture_weights_head_size,
                    num_layers=self.temporal_mixture_weights_num_layers,
                    dropout=self.temporal_mixture_weights_dropout,
                    activation_factory=lambda: create_activation_(activation_fn_cfg),
                    dtype=self.model_dtype,
                ).to(self.device)
        elif self._mixer_is_input_and_index_dependent:
            # RLRP-735: index + input conditioned mixer (pure step-indexed γ + per-step input head
            # ρ, NO frozen history). input_token_dim matches the input-dependent branch's
            # cat([mean, logvar]) width 2*(O+A); the shared head knobs drive the input head.
            _mixture_inp_dim = 2 * (self.singlestep_obs_len + self.singlestep_act_len)
            self.temporal_mixture_weights = InputAndIndexDependentTemporalMixtureWeighs(
                unrol_len=self.unroll_len,
                horizon_len=self.horizon_len,
                ensemble_size=self.num_members,
                input_token_dim=_mixture_inp_dim,
                head_size=self.temporal_mixture_weights_head_size,
                num_layers=self.temporal_mixture_weights_num_layers,
                dropout=self.temporal_mixture_weights_dropout,
                activation_factory=lambda: create_activation_(activation_fn_cfg),
                dtype=self.model_dtype,
            ).to(self.device)
        else:
            self.temporal_mixture_weights = TemporalMixtureWeighs(
                unrol_len=self.unroll_len,
                horizon_len=self.horizon_len,
                ensemble_size=self.num_members,
                dtype=self.model_dtype,
            ).to(self.device)

        self.forecast_mean_accumulator: Tensor | None = None
        self.forecast_logvar_accumulator: Tensor | None = None
        self.ms_state_forecast_mean: list = []
        self.ms_state_forecast_logvar: list = []

        # .... loss related .......................................................................
        # Note: force cast to float
        self.ss_composite_loss_weight = float(ss_composite_loss_weight)
        # Whether the (enabled) single-step term is routed through the composite auto-weighting
        # ("SS" slot) or enters the sum at its static `ss_composite_loss_weight` only.
        self.single_step_auto_weighting = bool(single_step_auto_weighting)
        self.ms_composite_loss_weight = float(ms_composite_loss_weight)

        if self.enable_pre_mixture_u_loss:
            consol_msg_universal_one_liner(f"U composite loss term enabled")
            self._pre_mixture_loss_target = None
            self.composite_loss_automatic_weighting.extend({"U": 2})

        # .... Register post-mixing / projection auto-weighting slots ............................
        # Contiguous indices after U=2: MIX=3, IPROJ=4, SIWAE=5. Each slot is only registered
        # when its term is enabled (keeps the SS=0 slot always present, see plan B.2).
        # Each slot uses the next FREE index = current map length (NOT a hardcoded value), so it
        # stays correct even when an earlier optional term (e.g. U) is disabled.
        def _next_weight_index() -> int:
            return len(self.composite_loss_automatic_weighting.auto_loss_weights_map)

        if self.enable_post_mixture_loss:
            consol_msg_universal_one_liner(
                f"MIX (post-mixture) composite loss term enabled "
                f"[{self.post_mixture_backprop_mode}]"
            )
            self.composite_loss_automatic_weighting.extend(
                {"MS_MIX": _next_weight_index()}
            )

        if self.enable_info_projection_loss:
            consol_msg_universal_one_liner(
                f"IPROJ (information-projection) composite loss term enabled "
                f"[{self.projection_backprop_mode}"
                f"{', auto_weighting' if self.projection_auto_weighting else ''}]"
            )
            # IPROJ is a (clamped) KL divergence; like the other projection objectives it enters
            # the composite sum at its static weight by default. The SHARED projection auto-weighting
            # toggle (Stage C.2) registers a dedicated slot.
            if self.projection_auto_weighting:
                self.composite_loss_automatic_weighting.extend(
                    {"IPROJ": _next_weight_index()}
                )

        if self.enable_siw_mp_loss:
            consol_msg_universal_one_liner(
                f"SIW-MP composite loss term enabled "
                f"[{self.projection_backprop_mode}"
                f"{', auto_weighting' if self.projection_auto_weighting else ''}]"
            )
            # SIWAE is treated as a regularisation term and enters the composite sum at a static
            # weight by default. The SHARED projection auto-weighting toggle registers a slot.
            if self.projection_auto_weighting:
                self.composite_loss_automatic_weighting.extend(
                    {"MS_SIW_MP": _next_weight_index()}
                )

        if self.enable_rollout_consistency_loss:
            consol_msg_universal_one_liner(
                f"RC (rollout self-consistency) composite loss term enabled "
                f"[{self.rollout_consistency_backprop_mode}, "
                f"{self.rollout_consistency_divergence}"
                f"{', auto_weighting' if self.rollout_consistency_auto_weighting else ''}]"
            )
            # RC has its OWN dedicated auto-weighting toggle (independent from the projection
            # objectives), registering a dedicated `MS_RC` slot when enabled.
            if self.rollout_consistency_auto_weighting:
                self.composite_loss_automatic_weighting.extend(
                    {"MS_RC": _next_weight_index()}
                )

        if self.enable_gms_iwae_loss:
            consol_msg_universal_one_liner(
                f"GMS-IWAE composite loss term enabled "
                f"[{self.projection_backprop_mode}, decoder=identity, "
                f"head=network"
                f"{', auto_weighting' if self.projection_auto_weighting else ''}]"
            )
            # GMS-IWAE is a (negated) IWAE bound that can be negative -> regularisation-term
            # default (static weight); the SHARED projection auto-weighting toggle registers a slot.
            if self.projection_auto_weighting:
                self.composite_loss_automatic_weighting.extend(
                    {"MS_GMS_IWAE": _next_weight_index()}
                )

        if (
            self.enable_compounded_prediction_deploy_loss
            and self.compounded_prediction_auto_weighting
        ):
            consol_msg_universal_one_liner(
                f"CP (compounded-prediction deploy) composite loss term auto_weighting enabled"
            )
            # CP is the standalone deploy-path second-AR-stage NLL term (Option C2, RLRP-708). It
            # routes through the composite auto-weighting (`loss_id='CP'`) ONLY when its dedicated
            # `compounded_prediction_deploy_loss.auto_weighting` toggle is on, so it gets a
            # dedicated slot here (next FREE index, robust to disabled earlier optional terms).
            self.composite_loss_automatic_weighting.extend({"CP": _next_weight_index()})

        if (
            self.enable_compounded_prediction_deploy_loss
            and self.enable_history_drift_loss
        ):
            consol_msg_universal_one_liner(
                f"DH (deploy-history drift residual) CP sub-term enabled "
                f"[{self.history_drift_target_mode}"
                f"{', auto_weighting' if self.history_drift_auto_weighting else ''}]"
            )
            # RLRP-748 (audit §5 item 1): the DH residual is a RAW-space residual (L1 or L2 per
            # ``ms_model.mae_loss``), so obs features with large physical ranges can dominate the
            # DH gradient (intra-term feature imbalance, RLRP-731 assessment §1.4).
            # ``history_drift_auto_weighting`` is the intended inter-term mitigation; advise
            # loudly when DH runs WITHOUT it so an imbalanced raw-scale gradient is a conscious
            # choice, not a silent default.
            if not self.history_drift_auto_weighting:
                consol_msg_universal_one_liner(
                    f"DH advisory: 'history_drift_auto_weighting' is OFF while DH is enabled. "
                    f"DH is a raw-space residual; large-range obs features may dominate its gradient. "
                    f"Keep 'compounded_prediction_deploy_loss.history_drift_auto_weighting=True' "
                    f"unless the raw feature scales are already comparable."
                )
            # DH is the CP sub-term terminal-step raw-space residual (RLRP-731). It routes
            # through the composite auto-weighting (`loss_id='DH'`) ONLY when its dedicated
            # `history_drift_auto_weighting` toggle is on (dedicated slot, next FREE index).
            if self.history_drift_auto_weighting:
                self.composite_loss_automatic_weighting.extend(
                    {"DH": _next_weight_index()}
                )

        # RLRP-751 (task T6): register the per-feature geometry AUTO-weighting slots
        # LAST (after every other term's slot) so the ``FEAT_GEOM_SS/MS/CP`` indices
        # are the next-free ones and never collide with the pre-existing slots. The
        # helper is a no-op unless ``feature_geometry.auto_weighting`` is on AND the
        # ctor-known geometry gates (non-zero ``loss_weight`` + non-``None``
        # objective) hold. All three channels are registered up-front (idempotent);
        # per-call routing still degrades gracefully to STATIC on the channels a
        # given step does not exercise.
        if self._feature_geom_auto_slots_ctor_on():
            consol_msg_universal_one_liner(
                f"FEAT_GEOM (per-feature geometry) composite loss term auto_weighting enabled"
            )
            self._register_feature_geometry_auto_slots(("SS", "MS", "CP"))

        if self.ms_probabilistic_loss_mode == "true_mixture":
            consol_msg_universal_one_liner(
                f"Primary MS probabilistic loss mode: true_mixture"
            )

        # Single-step (SS) composite term: now an explicit on/off switch (RLRP-704) instead of
        # the legacy ``ss_composite_loss_weight == 0.0`` sentinel.
        self.multitask_loss_singlestep_term_enable = bool(enable_single_step_loss)
        if self.multitask_loss_singlestep_term_enable:
            consol_msg_universal_one_liner(
                f"Single-step composite loss term enabled"
                f"{', auto_weighting' if self.single_step_auto_weighting else ''}"
            )
        else:
            consol_msg_universal_one_liner(f"Single-step composite loss term disabled")

        # Keep the auto-weighting module alive whenever ANY composite term that ROUTES through it
        # is active (otherwise e.g. a MIX/IPROJ/SIWAE-only config, or an SS term with
        # auto_weighting disabled, could null it). The SS term only calls the module when its
        # dedicated `single_step_auto_weighting` toggle is on.
        if not (
            (
                self.multitask_loss_singlestep_term_enable
                and self.single_step_auto_weighting
            )
            or self.enable_pre_mixture_u_loss
            or self.enable_post_mixture_loss
            or (self.enable_info_projection_loss and self.projection_auto_weighting)
            or (self.enable_siw_mp_loss and self.projection_auto_weighting)
            or (
                self.enable_rollout_consistency_loss
                and self.rollout_consistency_auto_weighting
            )
            or (self.enable_gms_iwae_loss and self.projection_auto_weighting)
            or (
                self.enable_compounded_prediction_deploy_loss
                and self.compounded_prediction_auto_weighting
            )
            or (
                self.enable_compounded_prediction_deploy_loss
                and self.enable_history_drift_loss
                and self.history_drift_auto_weighting
            )
            # RLRP-751 (task T6): keep the module alive when the per-feature
            # geometry term is the ONLY AUTO-routed term.
            or self._feature_geom_auto_slots_ctor_on()
        ):
            self.composite_loss_automatic_weighting = None

        # .... Final build step ...................................................................
        self.build_network_post(learn_logvar_bounds)
        # RLRP-786 (Step 4 finding, Orin smoke): move the WHOLE module to ``self.device``, not only
        # the dtype. The layers are built with ``device=`` but the ``CompositeLossAutomaticWeighting``
        # submodule (``MultiStepMLP.__init__``) is not, so its ``auto_loss_log_vars`` parameter
        # stayed on the CPU in production (the tests always call ``model.to(device)`` and never saw
        # it): every step paid a hidden device->host copy of that parameter's gradient (its
        # ``AddBackward0`` was the op that broke the CUDA-graph capture) and a CPU-side Adam update.
        # Mirrors ``ExponentialFamilyMLP.build_network_post`` (``self.to(self.device)``).
        self.to(device=self.device, dtype=self.model_dtype)

        # Deploy-path training stabilization (RLRP-722): now that the projection head (if any) is
        # built and the deploy-head mode is known, hard-fail an incoherent EMA forecast clone
        # configuration (a fed deploy term with no trainable head distinct from the frozen body).
        self._validate_ema_forecast_feed_to()

        # .... Compounded-prediction (second) AR stage setup (RLRP-708) ...........................
        # Initialize the AR iteration state + schedulers provided by the
        # ``CompoundedPredictionMultiStepIterator`` mixin (no base ``__init__`` is run by it).
        # This applies the final dtype cast. Defaults keep the second AR stage OFF
        # (``ar_train_horizon_unroll=False``); ``_ar_memory_records_deploy_predictions=True``
        # repurposes ``ar_memory`` as a growing buffer of deploy-time single-step predictions
        # (R4) since MTM-Pro has no recurrent hidden state to store there.
        self._setup_compounded_prediction_iterator(
            # Task 2c (RLRP-708): drive the CP unroll off the dedicated, decoupled CP horizon.
            horizon_len=compounded_prediction_horizon_len,
            model_use_double_precision=model_use_double_precision,
            ar_enabled=ar_enabled,
            receive_sequence_batch=receive_sequence_batch,
            ar_train_horizon_unroll=ar_train_horizon_unroll,
            teacher_forcing_decay_stop=teacher_forcing_decay_stop,
            teacher_forcing_decay_start=teacher_forcing_decay_start,
            teacher_forcing_method=teacher_forcing_method,
            ar_temporal_weights=ar_temporal_weights,
            ar_temporal_weights_start=ar_temporal_weights_start,
            ar_temporal_weights_warmup=ar_temporal_weights_warmup,
            ar_temporal_weights_ramp_stop=ar_temporal_weights_ramp_stop,
            ar_unrol_len_probablity_decay_start=ar_unrol_len_probablity_decay_start,
            ar_unrol_len_probablity_decay_stop=ar_unrol_len_probablity_decay_stop,
            ar_unrol_len_decay_method=ar_unrol_len_decay_method,
            ar_unrol_len_start_horizon_len=ar_unrol_len_start_horizon_len,
        )
        self._ar_memory_records_deploy_predictions = True

        # .... Forecast-path teacher-forcing scheduler (RLRP-726) .................................
        # DEDICATED instance for the MS forecast self-feed (`state_history_update`), kept SEPARATE
        # from the CP/deploy `self._teacher_forcing_scheduler` (built by
        # `_setup_compounded_prediction_iterator` above). Default-OFF (always_off / decay_stop=0
        # -> `is_fully_disabled`) => `state_history_update` takes the exact free-running branch
        # (byte-for-byte parity). Stepped exactly once per `_probabilistic_loss` (never inside the
        # RC-re-entered forward), so all RC re-forwards + the eval forward see a frozen schedule.
        self._forecast_teacher_forcing_scheduler = TeacherForcingScheduler(
            decay_start=forecast_teacher_forcing_decay_start,
            decay_stop=forecast_teacher_forcing_decay_stop,
            method=forecast_teacher_forcing_method,
        )

        # Option C2 (RLRP-708): MTM-Pro computes the compounded-prediction objective itself
        # (a standalone deploy-path ``CP`` term inside ``_probabilistic_loss``), so the mixin
        # ``loss`` U-loop must be bypassed (it would slice the multistep target to an obs-only
        # single step and crash the host forecast loss). The mixin delegates to the host loss.
        self._compounded_prediction_handled_internally = True

    # (PRIORITY) inprogress: implement save/load method with model specialized fields

    def _save_state_dict(self) -> dict[str, list[int] | dict[str, Any] | Any]:
        # R4 (RLRP-708): persist the compounded-prediction ``ar_memory`` buffer the way
        # ``AbstractMS2SSAutoRegressive`` does. For MTM-Pro this buffer is a growing record
        # of deploy-time single-step predictions (not a recurrent hidden state); it is
        # ``None`` whenever the second AR stage has produced no deploy prediction yet.
        model_dict = super()._save_state_dict()
        # A4 (RLRP-783): ``ar_memory`` is now a narrow VIEW onto an over-allocated capacity
        # buffer (amortised growth), so persist a COMPACT copy sized to the logical length --
        # otherwise ``torch.save`` would serialise the whole (up to 2x) capacity storage. The
        # saved value is bit-identical to the pre-change checkpoint. See the RLRC MTM-Pro models
        # code optimization `.junie` plan
        # (``perf_RLRP-783_mtm_pro_models_code_optimization_plan_20260827.md``).
        model_dict["ar_memory"] = (
            self.ar_memory.contiguous().clone() if self.ar_memory is not None else None
        )
        # Deploy-path training stabilization (RLRP-722): persist the per-batch training-step
        # counter (drives warmup gating + EMA momentum schedule) and the EMA forecast clone so a
        # resumed run keeps a correct warmup phase + teacher network. Both default-guarded on load
        # so older checkpoints (missing these keys) still load.
        model_dict["train_step_count"] = int(self._train_step_count)
        model_dict["ema_forecast"] = self._ema_forecast
        return model_dict

    def _load_state_dict(
        self, model_dict: dict[str, list[int] | dict[str, Any] | Any]
    ) -> None:
        # .... Validate required keys before mutating any state ...................................
        if "ar_memory" not in model_dict:
            raise KeyError(
                f"{type(self).__name__}._load_state_dict: missing keys in "
                f"model_dict: ar_memory"
            )

        loaded_ar_memory = model_dict["ar_memory"]
        if loaded_ar_memory is not None and not isinstance(
            loaded_ar_memory, torch.Tensor
        ):
            raise TypeError(
                f"{type(self).__name__}._load_state_dict: 'ar_memory' must be "
                f"either None or a torch.Tensor, got {type(loaded_ar_memory).__name__}."
            )

        # Deploy-path training stabilization (RLRP-722): validate the optional new keys before
        # mutating state (missing-key guarded so pre-RLRP-722 checkpoints load).
        loaded_step_count = model_dict.get("train_step_count", 0)
        if not isinstance(loaded_step_count, int):
            raise TypeError(
                f"{type(self).__name__}._load_state_dict: 'train_step_count' must be an int, "
                f"got {type(loaded_step_count).__name__}."
            )
        loaded_ema_forecast = model_dict.get("ema_forecast", None)
        if loaded_ema_forecast is not None and not (
            isinstance(loaded_ema_forecast, dict)
            and all(isinstance(v, torch.Tensor) for v in loaded_ema_forecast.values())
        ):
            raise TypeError(
                f"{type(self).__name__}._load_state_dict: 'ema_forecast' must be either None or "
                f"a dict of torch.Tensor, got {type(loaded_ema_forecast).__name__}."
            )

        super()._load_state_dict(model_dict)

        self.ar_memory = loaded_ar_memory
        self._train_step_count = int(loaded_step_count)
        # ``None`` -> lazily re-init from the live net on first use (so older checkpoints and
        # newly-enabled clones both start ``theta_ema = theta_live``). Reset the one-shot guard so
        # the loaded dict is device/dtype-aligned (or re-seeded) on next use.
        self._ema_forecast = loaded_ema_forecast
        self._ema_forecast_initialized = False
        return None

    # .... Deploy-path training stabilization (RLRP-722): EMA frozen-decaying forecast clone ......
    # The forecast sub-network the deploy path consumes is the body (``hidden_layers`` ->
    # ``mean_and_logvar`` -> ``mean_layer`` / ``logvar_layer``) + the temporal mixer
    # (``temporal_mixture_weights``). The EMA clone is a detached parameter dict over exactly these
    # submodules; the deploy-path terms named in ``feed_to`` run their forward against the EMA
    # weights (a stable teacher) while the MS forecast term always uses the live, trainable body.
    _EMA_FORECAST_SUBMODULES: Tuple[str, ...] = (
        "hidden_layers",
        "mean_and_logvar",
        "mean_layer",
        "logvar_layer",
        "temporal_mixture_weights",
    )

    def _ema_forecast_named_params(self):
        """Yield ``(qualified_name, live_param)`` over the TRAINABLE forecast body + mixer params.

        ``qualified_name`` is relative to ``self`` (e.g. ``hidden_layers.0.1.weight``) so it can be
        used both as the EMA-dict key and to locate the owning leaf module for the param swap.
        Non-trainable params (e.g. the frozen logvar-bound constant) are skipped: they are constant
        in the live net, so the teacher need not track them and swapping them is a no-op (this also
        mirrors the ``requires_grad`` filter in ``build_mtm_pro_param_groups``).
        """
        for sub_name in self._EMA_FORECAST_SUBMODULES:
            sub = getattr(self, sub_name, None)
            if sub is None:
                continue
            for p_name, p in sub.named_parameters(recurse=True):
                if not p.requires_grad:
                    continue
                yield f"{sub_name}.{p_name}", p

    def _maybe_init_ema_forecast(self) -> None:
        """Lazily initialise (or, after a checkpoint load, align) the EMA forecast param dict.

        A fresh model and a checkpoint without the clone both start ``theta_ema = theta_live``.
        Loaded tensors are aligned to the live params' device/dtype and any missing key is
        re-seeded from the live net (so a clone enabled on an older checkpoint still works).
        """
        if self._ema_forecast_initialized:
            return
        live = dict(self._ema_forecast_named_params())
        if self._ema_forecast is None:
            self._ema_forecast = {
                name: p.detach().clone().requires_grad_(False)
                for name, p in live.items()
            }
        else:
            # Loaded from a checkpoint: align device/dtype + re-seed any missing key.
            self._ema_forecast = {
                name: (
                    self._ema_forecast[name]
                    .detach()
                    .to(device=p.device, dtype=p.dtype)
                    .requires_grad_(False)
                    if name in self._ema_forecast
                    else p.detach().clone().requires_grad_(False)
                )
                for name, p in live.items()
            }
        self._ema_forecast_initialized = True

    def _current_ema_momentum(self) -> float:
        """Resolve the EMA momentum ``d`` for the current step (constant, or scheduled).

        ``kind == none`` -> the constant ``momentum``. ``linear`` / ``cosine`` interpolate
        ``start -> end`` over ``horizon_steps`` using the per-batch ``_train_step_count``; both
        directions (slow-down ``0.99 -> 0.9999`` and speed-up ``0.9999 -> 0.99``) are supported via
        the ``start``/``end`` ordering.
        """
        if self._ema_momentum_schedule_kind == "none":
            return self._ema_forecast_momentum
        horizon = self._ema_momentum_schedule_horizon
        start = self._ema_momentum_schedule_start
        end = self._ema_momentum_schedule_end
        if horizon <= 0:
            return end
        frac = min(1.0, self._train_step_count / float(horizon))
        if self._ema_momentum_schedule_kind == "linear":
            return start + (end - start) * frac
        # cosine: frac=0 -> start, frac=1 -> end (smooth).
        return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * frac))

    @torch.no_grad()
    def _update_ema_forecast(self) -> float:
        """EMA update ``theta_ema <- d*theta_ema + (1-d)*theta_live`` over the forecast body+mixer.
        Returns the momentum ``d`` used (for the TensorBoard diagnostic).

        The update is performed OUT-OF-PLACE (each dict entry is rebound to a freshly-allocated
        tensor) rather than via ``mul_().add_()``. This is mandatory: the fed deploy terms (e.g. the
        CP AR rollout) read the swapped-in EMA tensors inside the live autograd graph this batch, so
        those tensors are *saved for backward*. An in-place mutation here (which runs after the
        forward but before Lightning's backward) would bump their version counter and raise
        ``RuntimeError: one of the variables needed for gradient computation has been modified by an
        inplace operation``. Rebinding leaves the graph-captured tensor untouched while the dict
        (consumed by the next batch's swap) points at the updated teacher.
        """
        self._maybe_init_ema_forecast()
        d = self._current_ema_momentum()
        for name, p in self._ema_forecast_named_params():
            self._ema_forecast[name] = (
                self._ema_forecast[name] * d + p.detach() * (1.0 - d)
            ).requires_grad_(False)
        return d

    @contextlib.contextmanager
    def _ema_forecast_weights(self):
        """Context manager that temporarily swaps the live forecast body+mixer parameters for the
        detached EMA tensors (stateless-style, mirroring ``torch.func.functional_call``).

        Because the substituted tensors have ``requires_grad=False``, any forward run inside the
        context treats the body+mixer as a CONSTANT teacher: no gradient flows into the live body
        params, while gradients to the deploy heads (applied downstream, outside the swapped set)
        are preserved -- exactly the existing ``head_only`` detach semantics, but against a stable
        forecast. The original ``nn.Parameter`` objects are restored on exit (even on error).
        """
        self._maybe_init_ema_forecast()
        saved = []
        try:
            for name, ema_tensor in self._ema_forecast.items():
                parent, _, leaf = name.rpartition(".")
                owner = self.get_submodule(parent) if parent else self
                saved.append((owner, leaf, owner._parameters[leaf]))
                owner._parameters[leaf] = ema_tensor
            yield
        finally:
            for owner, leaf, original in saved:
                owner._parameters[leaf] = original

    def _ema_term_ctx(self, term: str):
        """Return the EMA-forecast param-swap context for the deploy ``term`` (one of
        ``{cp, ss, projection, rc}``) when the EMA clone is enabled AND ``term`` is in ``feed_to``;
        otherwise a no-op context so the term keeps using the live forecast body (byte-for-byte).
        """
        if self._ema_forecast_enable and term in self._ema_feed_to:
            return self._ema_forecast_weights()
        return contextlib.nullcontext()

    def _validate_ema_forecast_feed_to(self) -> None:
        """RLRP-722: hard-fail an INCOHERENT EMA forecast clone configuration at construction time.

        The EMA frozen-decaying clone freezes the forecast body+mixer so a deploy head that is
        DISTINCT from that body can train against a stable teacher forecast (the ``head_only``
        detach semantics, but against a non-moving forecast). That only makes sense when such a
        distinct, trainable head actually exists for the fed term; otherwise the param-swap merely
        detaches every parameter that produces the term's prediction -> the term's gradient is
        identically zero (it IS the frozen forecast network), silently disabling it.

        Per-term coherence (the swap is only meaningful when the term routes through a trainable
        head separate from the swapped body):

        - ``cp`` / ``ss``: run their forward through :meth:`deploy`, which routes through the
          projection head ONLY when ``deploy_head_mode == 'projection'`` AND a projection head
          exists. Otherwise :meth:`deploy` falls back to the forecast/mixing output -> the prediction
          IS the (now-frozen) body -> self-referential. So they require an EFFECTIVE projection
          deploy head.
        - ``projection``: consumes the EMA teacher through the dedicated projection head, which is
          trained by a projection objective (IPROJ / SIW-MP / GMS-IWAE) independently of the deploy
          head mode. Requires one of those objectives to be enabled.
        - ``rc``: requires the rollout-consistency objective.

        A no-op when the clone is disabled or ``feed_to`` is empty (the OFF path is byte-for-byte
        unchanged).
        """
        if not self._ema_forecast_enable:
            return
        has_projection_head = self.projection_head is not None
        uses_projection_deploy = (
            self.deploy_head_mode == "projection" and has_projection_head
        )
        reasons: Dict[str, str] = {}
        for term in sorted(self._ema_feed_to):
            if term in ("cp", "ss"):
                if not uses_projection_deploy:
                    reasons[term] = (
                        "requires an effective projection deploy head "
                        "(deploy_head.mode='projection' AND a projection objective enabled so the "
                        "projection head is built); otherwise deploy() falls back to the forecast "
                        "body and the term trains the (now-frozen) forecast network itself"
                    )
            elif term == "projection":
                if not (
                    self.enable_info_projection_loss
                    or self.enable_siw_mp_loss
                    or self.enable_gms_iwae_loss
                ):
                    reasons[term] = (
                        "requires a projection objective (info_projection_loss / siw_mp_loss / "
                        "gms_iwae_loss) to be enabled so the projection head exists and is trained"
                    )
            elif term == "rc":
                if not self.enable_rollout_consistency_loss:
                    reasons[term] = "requires rollout_consistency_loss to be enabled"
        if reasons:
            detail = "; ".join(f"'{t}' {why}" for t, why in reasons.items())
            raise ValueError(
                "deploy_path_training.ema_forecast_clone is enabled but feed_to term(s) "
                f"{sorted(reasons)} have no trainable deploy head distinct from the frozen "
                f"forecast body, so freezing the forecast would silently zero their gradient: "
                f"{detail}. Fix: either disable the EMA clone "
                "(deploy_path_training.ema_forecast_clone.enable=false), remove the offending "
                "term(s) from feed_to, or enable the deploy head / objective they require "
                "(e.g. deploy_head.mode='projection' with a projection objective for 'cp'/'ss')."
            )

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

        MS_HEAD_HIDDEN_SIZE = hid_size

        create_activation = partial(create_activation_, activation_fn_cfg)

        create_linear_layer = partial(create_linear_layer_, ensemble_size)

        # .... Layer-bloc resolvers (RLRP-768: per-region type / layer_norm / dropout) ...........
        # Two independent resolvers so the encoder/hidden stack (region "encoder", uses `dropout`)
        # and the MS-head stack (region "ms_head", uses `ms_head_dropout`) are toggled separately.
        enc_res_kind, make_enc_res = self._layer_bloc_factory(
            "encoder", ensemble_size, create_activation, dropout
        )
        ms_res_kind, make_ms_res = self._layer_bloc_factory(
            "ms_head", ensemble_size, create_activation, self.ms_head_dropout
        )

        hidden_layers = [
            nn.Sequential(
                nn.Dropout(p=dropout),
                create_linear_layer(in_size, hid_size),
                create_activation(),
            )
        ]

        # Stack residual layer
        for i in range(num_layers - 1):
            hidden_layers.append(
                make_layer_bloc_seq(
                    enc_res_kind, make_enc_res, hid_size, create_activation, dropout
                )
            )

        self.hidden_layers = nn.Sequential(*hidden_layers)

        # .... Multi-Step head ....................................................................
        ms_mean_layers = []
        ms_logvar_layers = []

        # Note: Dropout is only on the first ms head layer on purposes and use a dedicated
        # parammeter 'ms_head_dropout' instead of the 'dropout' one which is used for
        # the encoder hiden layers.
        self.mean_and_logvar = nn.Sequential(
            nn.Dropout(p=self.ms_head_dropout),
            create_linear_layer(hid_size, 2 * MS_HEAD_HIDDEN_SIZE),
            create_activation(),
        )

        # Stack multi-step mean and logvar residual layers
        for i in range(self.ms_head_num_layers - 1):
            ms_mean_layers.append(
                make_layer_bloc_seq(
                    ms_res_kind,
                    make_ms_res,
                    MS_HEAD_HIDDEN_SIZE,
                    create_activation,
                    self.ms_head_dropout,
                )
            )

            ms_logvar_layers.append(
                make_layer_bloc_seq(
                    ms_res_kind,
                    make_ms_res,
                    MS_HEAD_HIDDEN_SIZE,
                    create_activation,
                    self.ms_head_dropout,
                )
            )

        # RLRP-736 §4A-B P1.2 (mean path): resolve the per-unroll-block attitude
        # bookkeeping for the shared MS/deploy MEAN head. The composed head output is
        # feature-major (all ``unroll_len`` obs blocks first, then the action blocks;
        # see ``revert_timestep_first_multistep_dim_unflaten_array``), so each obs-block
        # attitude slot ``s`` (``s < singlestep_obs_len``, tracked in
        # ``_ss_ori_out_slots``) repeats once per unroll step at stride
        # ``singlestep_obs_len``. When the rep is active the mean head EXPANDS by
        # ``internal_rep_out_width(rep, external_width) - external_width`` per such
        # slot and every slot is decoded back to a unit quaternion by construction;
        # the logvar head is left UNCHANGED (symmetric, external width per slot).
        # Neutral/empty + zero-extra when OFF => bit-exact.
        self._setup_unroll_orientation_output_bookkeeping()

        # Finale multi-step mean layer
        ms_mean_layers.append(
            nn.Sequential(
                create_linear_layer(
                    MS_HEAD_HIDDEN_SIZE,
                    self.compose_obs_unroll_len + self._unroll_mean_head_extra,
                ),  # <--
            )
        )
        self.mean_layer = nn.Sequential(*ms_mean_layers)

        # Finale multi-step logvar layer
        ms_logvar_layers.append(
            nn.Sequential(
                create_linear_layer(
                    MS_HEAD_HIDDEN_SIZE, self.compose_obs_unroll_len
                ),  # <--
            )
        )
        self.logvar_layer = nn.Sequential(*ms_logvar_layers)

        # .... Dedicated information-projection summary head (Stage H.3b) .........................
        self._build_projection_head(activation_fn_cfg)

        # .... GMS-IWAE generative decoder p(y|z) (built next to the projection head) .............
        self._build_gms_decoder(activation_fn_cfg)

        # .... Deploy head (i.e., single-step head) ...............................................
        self.deploy_head_mean_adapter = None
        self.deploy_head_logvar_adapter = None

        # self.deploy_head_mean_and_logvar = self.mean_and_logvar
        # self.deploy_head_mean = self.mean_layer

        # (Quick-hack) Only use for training the singlestep logvar bound
        self.deploy_head_logvar = nn.Sequential()

        self.apply(truncated_normal_init)
        # Re-zero the identity residual blocks' last linear AFTER the global
        # truncated-normal init so they start as an exact identity (y == x).
        zero_init_residual_blocks_(self)
        self.to(self.device)

        return None

    def _setup_unroll_orientation_output_bookkeeping(self) -> None:
        """Resolve the per-unroll-block attitude bookkeeping for the MEAN head (RLRP-736 §4A-B P1.2).

        The shared MS/deploy mean head (``_default_forward_u`` -> ``self.mean_layer``)
        emits ONE composed ``compose_obs_unroll_len`` window packed feature-major (all
        ``unroll_len`` single-step obs blocks first, each ``singlestep_obs_len`` wide,
        then the action blocks). Each obs-block attitude quaternion slot ``s``
        (``s < singlestep_obs_len``, resolved by the base ``_setup_orientation_rep`` into
        ``_ss_ori_out_slots``) therefore appears once per unroll step at stride
        ``singlestep_obs_len``. This method stores:

        - ``_unroll_mean_head_extra``: the total width the mean head grows by,
          ``n_slots * (raw_width - external_width)`` (0 when the rep is OFF =>
          bit-exact);
        - ``_unroll_ori_out_trunk_slots``: the start index of every attitude slot in the
          rep-EXPANDED head-output layout (external position shifted by the cumulative
          ``raw_width - external_width`` of the preceding slots), consumed by
          :meth:`_apply_unroll_orientation_output_decoding`.

        Neutral (empty slots, zero extra) unless the by-construction wiring is active AND
        an obs-block attitude output slot was tracked. The logvar head is intentionally
        left symmetric (external 4-D per slot); the 3-D so3 tangent (log-)variance head is
        the deferred follow-up (see the class-level opt-in note).
        """
        self._unroll_mean_head_extra: int = 0
        self._unroll_ori_out_trunk_slots: Tuple[int, ...] = ()
        # RLRP-744 item A (§A.3 step 2) + item C (rev.3b, always-on ``_unit_decode``):
        # the by-construction decode also fires on the plain-``quaternion`` path
        # (``_manifold_aware_nll`` or ``_unit_decode``), where
        # ``raw_width == external_width`` makes the bookkeeping purely positional
        # (extra == 0).
        if not (
            getattr(self, "_internal_orientation_rep_active", False)
            or getattr(self, "_manifold_aware_nll", False)
            or getattr(self, "_unit_decode", False)
        ) or not getattr(self, "_ss_ori_out_slots", ()):
            return None
        external_width = getattr(
            self, "_ori_external_width", DEFAULT_EXTERNAL_ORIENTATION_WIDTH
        )
        raw_width = internal_rep_out_width(self._orientation_rep, external_width)
        base = sorted(self._ss_ori_out_slots)
        # External positions of every attitude slot in the feature-major composed layout.
        flat_ext = sorted(
            u * self.singlestep_obs_len + s
            for u in range(self.unroll_len)
            for s in base
        )
        self._unroll_mean_head_extra = len(flat_ext) * (raw_width - external_width)
        # Map external -> rep-expanded (trunk) positions: the i-th slot start shifts by
        # the cumulative ``raw_width - external_width`` of the ``i`` preceding slots.
        self._unroll_ori_out_trunk_slots = tuple(
            start + i * (raw_width - external_width)
            for i, start in enumerate(flat_ext)
        )
        return None

    def _apply_unroll_orientation_output_decoding(self, mean: Tensor) -> Tensor:
        """Decode every per-unroll-block attitude slot of the MEAN head to a unit quaternion.

        RLRP-736 §4A-B P1.2. The mean head emits
        ``internal_rep_out_width(rep, external_width)`` raw values per attitude
        slot (in the rep-expanded composed layout resolved by
        :meth:`_setup_unroll_orientation_output_bookkeeping`); each is mapped onto ``SO(3)``
        and returned as a 4-D unit quaternion in the external composed layout, so both the
        multi-step forecast and the single-step deploy prediction are unit quaternions by
        construction (Geist et al. 2024; Zhou et al. 2019). No-op / bit-exact unless the
        by-construction rep is active and an obs-block attitude slot was tracked.
        """
        slots = getattr(self, "_unroll_ori_out_trunk_slots", ())
        if (
            not (
                getattr(self, "_internal_orientation_rep_active", False)
                or getattr(self, "_manifold_aware_nll", False)
                or getattr(self, "_unit_decode", False)
            )
            or not slots
        ):
            return mean
        external_width = getattr(
            self, "_ori_external_width", DEFAULT_EXTERNAL_ORIENTATION_WIDTH
        )
        raw_width = internal_rep_out_width(self._orientation_rep, external_width)
        if self._orientation_rep is InternalOrientationRep.S2_TANGENT:
            # RLRP-796 ``D-796-S1-6-A``: the per-step chart makes this a SEQUENTIAL
            # scan over the unroll horizon — block ``u`` is decoded at block
            # ``u-1``'s decoded direction, seeded from the last history gravity. The
            # closure is shared with the composed/SS decodes of the base seam, and
            # ``_splice_slots`` invokes it once per slot in ascending (i.e. unroll)
            # order, which is exactly the scan order the ruling asks for.
            transform = self._s2_tangent_decode_transform()
        else:

            def transform(raw: Tensor) -> Tensor:
                return decode_internal_rep_to_orientation_slot(
                    raw, self._orientation_rep
                )

            if (
                getattr(self, "vectorized_unroll_orientation_decode", False)
                and raw_width == external_width
            ):
                # RLRP-830 Item D (Step 1): the width-preserving reps (``QUATERNION``
                # is a trailing-axis ``F.normalize``, ``QUATERNIONLEGACY`` the identity)
                # make the splice a feature-axis permutation, so the ``U`` per-slot
                # transform calls + the ``2U+1``-segment ``cat`` collapse into ONE
                # gather -> ONE stacked transform -> ONE scatter
                # (:meth:`ExponentialFamilyMLP._splice_slots_width_preserving`,
                # RLRP-747, bit-exact with the generic route). This method runs once
                # per ``_default_forward_u`` call, i.e. ``F`` forecast u-forwards +
                # ``U`` CP deploys per training step, so it is the largest
                # launch-count lever of the MTM-Pro family at the paper config.
                return self._splice_slots_width_preserving(
                    mean, slots, raw_width, transform
                )

        return self._splice_slots(
            mean,
            slots,
            external_width=raw_width,
            transform=transform,
        )

    def _reinjected_attitude_canonicalization_active(self) -> bool:
        """Whether re-injected AR-sample attitude slots must be re-normalised.

        RLRP-736 §4A-B P1.2 / RLRP-738 (item b). The AR history-update rule re-injects
        RAW ``MixtureSameFamily.sample()`` draws back into the history buffer, which is
        then fed to the encoder input-encode ``q -> R`` (``_apply_ss_orientation_input_encoding``)
        that ASSUMES a valid rotation. A raw mixture draw's 4-D attitude sub-vector is a
        generic vector — neither unit-norm nor sign-canonical — so it must be projected
        back onto the unit-quaternion double-cover before re-injection. Gated on the
        per-family opt-in flag, an ACTIVE non-``quaternion`` rep, and tracked single-step
        attitude INPUT slot(s); bit-exact OFF (returns the raw sample untouched) otherwise.
        """
        return bool(
            getattr(self, "_supports_probabilistic_by_construction_orientation", False)
            and getattr(self, "_internal_orientation_rep_active", False)
            and getattr(self, "_ss_ori_in_slots", ())
        )

    def _canonicalize_reinjected_quaternion(self, sample: Tensor) -> Tensor:
        """Re-normalise the attitude sub-vector(s) of a raw AR sample.

        RLRP-736 §4A-B P1.2 / RLRP-738 (item b). ``sample`` is ONE re-injected single-step
        obs+act block ``(..., O+A)`` drawn from the forecast mixture in data space. For every
        tracked single-step attitude INPUT slot ``s`` (``self._ss_ori_in_slots``, 4-D external
        quaternion within the obs sub-block) the slot is L2-normalised to the unit sphere
        (validity only; the memoryless ``w >= 0`` hemisphere flip was removed by RLRP-744
        action B-4 — hemisphere continuity is owned by the reference-relative alignment),
        so the downstream input-encode ``q -> R`` receives a valid
        rotation by construction. No-op / bit-exact unless the by-construction wiring is active
        and an obs-block attitude input slot was tracked.
        """
        if not self._reinjected_attitude_canonicalization_active():
            return sample
        eps = variance_floor(self.model_dtype)

        def _renorm(q: Tensor) -> Tensor:
            # RLRP-744 action B-4: the memoryless ``w >= 0`` sign step was removed
            # (Quaternion manifold upgrade + memoryless ``w >= 0`` removal `.junie`
            # plan,
            # `rlrp-744-quaternion-manifold-upgrade-and-memoryless-wge0-removal-plan-20260720.md`);
            # only the L2-normalise (validity) is kept — hemisphere continuity is
            # owned by the reference-relative alignment applied last.
            return torch.nn.functional.normalize(q, dim=-1, eps=eps)

        return self._splice_slots(
            sample,
            sorted(self._ss_ori_in_slots),
            external_width=4,
            transform=_renorm,
        )

    def _canonicalize_exposed_mean_quaternion(self, mean: Tensor) -> Tensor:
        """Project every attitude slot of an EXPOSED mixture mean back onto S^3.

        RLRP-736 §4A-B P3.2 (RLRP-738; ★ compounded-error lever). The exposed forecast /
        deploy point estimate is ``ss_dist_mixture.mean`` — a moment-matched weighted
        average of the component quaternions and therefore generically NON-unit. It is
        re-injected as ``next_obs`` in the free-running CP unroll and stashed for the
        feature-geometry term, so a non-unit quaternion silently yields a slightly
        invalid rotation each step (``q -> R``) and the drift accumulates over the
        horizon. This canonicalises each tracked single-step attitude OUTPUT slot
        (``self._ss_ori_out_slots``) via a chordal-L2 projection (the memoryless
        ``w >= 0`` sign step was removed by RLRP-744 action B-3)
        (Forster et al. 2016; Geist et al. 2024), so the exposed / CP-fed mean lives on
        the manifold. Complements P3.1 (which canonicalises only the
        ``_ManifoldMeanMixtureSameFamily`` mean); this covers the BROADER
        ``_internal_orientation_rep_active`` gate — including the plain ``MixtureSameFamily`` and the
        transformer-subclass paths. Idempotent on an already-unit mean; bit-exact no-op
        when the rep is OFF or no attitude output slot is tracked.
        """
        if not getattr(self, "_internal_orientation_rep_active", False):
            return mean
        slots = getattr(self, "_ss_ori_out_slots", ())
        if not slots:
            return mean
        return _canonicalize_quaternion_slots(
            mean, sorted(slots), variance_floor(self.model_dtype)
        )

    def build_network_post(self, learn_logvar_bounds: bool) -> None:

        # .... Encoder logvar bound ...............................................................
        self.logvar_layer.add_module(  # Make the module fetchable by name
            "logvar_bound",
            create_logvar_bound_layer(
                self.compose_obs_unroll_len,
                learn_logvar_bounds,
                # bound_min=weighted_bound_min,
                # bound_max=weighted_bound_max,
                bound_min_init=torch.tensor(
                    self._LOGVAR_MIN_BOUND_INIT, device=self.device
                ),
                bound_max_init=torch.tensor(
                    self._LOGVAR_MAX_BOUND_INIT, device=self.device
                ),
                grad_clip=self.logvar_bound_grad_clip,
            ),
        )

        # .... Decoder logvar bound ...............................................................
        self.deploy_head_logvar.add_module(  # Make the module fetchable by name
            "logvar_bound",
            create_logvar_bound_layer(
                self.singlestep_obs_len,
                learn_logvar_bounds,
                bound_min_init=torch.tensor(
                    self._LOGVAR_MIN_BOUND_INIT, device=self.device
                ),
                bound_max_init=torch.tensor(
                    self._LOGVAR_MAX_BOUND_INIT, device=self.device
                ),
                grad_clip=self.logvar_bound_grad_clip,
            ),
        )

        # .... Dedicated 3-D so3-tangent (log-)variance HEAD (RLRP-736 §4A-B / RLRP-738)
        self._build_attitude_tangent_logvar_head()

        return None

    def _attitude_tangent_logvar_head_eligible(self) -> bool:
        """Whether the dedicated 3-D so3-tangent logvar HEAD should exist.

        Mirrors the :meth:`_mixture_orientation_nll_active` guard exactly (per-family
        opt-in flag + resolved ``_manifold_aware_nll`` lever (RLRP-744 §A.3 step 3)
        + tracked single-step attitude OUTPUT slot(s) + probabilistic model)
        EXCEPT it is evaluated at BUILD time,
        so it must be safe before the forward path runs. Bit-exact OFF: when this is
        ``False`` the head is never instantiated and the scoring path keeps the
        legacy vector-part placeholder branch (which itself is only reached under the
        same active guard), so no parameter is added and the OFF path is byte-for-byte
        identical.
        """
        return bool(
            getattr(self, "_supports_probabilistic_by_construction_orientation", False)
            and getattr(self, "_manifold_aware_nll", False)
            and getattr(self, "_ss_ori_out_slots", ())
            and not getattr(self, "_deterministic_rep", True)
        )

    def _build_attitude_tangent_logvar_head(self) -> None:
        """Build the dedicated, physically-narrowed 3-D so3-tangent logvar HEAD.

        RLRP-736 §4A-B P1.2 / RLRP-738 (structural increment, base Particle variant,
        ensemble ``num_members == 1``). This REPLACES the fixed
        ``σ_tan² = scale[..., 1:4]²`` placeholder used inside
        :meth:`_apply_mixture_attitude_tangent_nll` with a LEARNABLE per-attitude-slot
        map ``4 -> 3`` from the component's quaternion-space (log-)variance to a
        genuine 3-D so3-tangent log-variance (Forster et al. 2016; Geist et al. 2024).

        Design (bounded, structural-only; the 4-D temporal-mixture accumulator /
        ``_temporal_mixture_distribution_at_i`` pipeline is intentionally left intact
        so every NON-attitude Gaussian marginal stays bit-exact): one small
        ``nn.Linear(4, 3)`` per sorted single-step attitude output slot, shared across
        the ``(E, B, K)`` component axes (Phase 1 is ``num_members == 1``). Each head
        is INITIALISED to reproduce the landed vector-part placeholder EXACTLY — its
        weight selects the quaternion vector-part columns ``(x, y, z)`` (input columns
        ``1, 2, 3``) with unit gain and zero bias — so training STARTS at the previous
        behaviour and only departs from it as the objective learns, preserving
        continuity with the already-landed increment.

        Neutral / bit-exact OFF: when the by-construction mixture-orientation split is
        inactive the head is NOT created (``self._attitude_tangent_logvar_head`` stays
        ``None``) and the scoring path is never reached, so no parameter is added.
        """
        self._attitude_tangent_logvar_head: Optional[nn.ModuleList] = None
        if not self._attitude_tangent_logvar_head_eligible():
            return None

        slots = sorted(self._ss_ori_out_slots)
        heads = nn.ModuleList()
        for _ in slots:
            linear = nn.Linear(4, 3)
            with torch.no_grad():
                # Select the quaternion vector-part (x, y, z) => input cols 1, 2, 3.
                weight = torch.zeros(3, 4)
                weight[0, 1] = 1.0
                weight[1, 2] = 1.0
                weight[2, 3] = 1.0
                linear.weight.copy_(weight)
                linear.bias.zero_()
            heads.append(linear)
        heads = heads.to(device=self.device)
        if hasattr(self, "model_dtype"):
            heads.to(dtype=self.model_dtype)
        # Index the per-slot head by its sorted attitude-slot base index.
        self._attitude_tangent_logvar_head_slots: Tuple[int, ...] = tuple(slots)
        self._attitude_tangent_logvar_head = heads
        return None

    def _build_projection_head(self, activation_fn_cfg: omegaconf.DictConfig) -> None:
        """
        Builds a projection head when specific loss terms or objectives are enabled, ensuring it
        is trained properly for deployment and preventing the use of untrained or underperforming
        heads in the deploy path.

        :param activation_fn_cfg: Configuration for the activation function used in building the
            projection head.
        :return: None
        """
        if (
            self.enable_info_projection_loss
            or self.enable_siw_mp_loss
            or self.enable_gms_iwae_loss
            or self.enable_rollout_consistency_loss
        ):
            self.projection_head = self._make_projection_head_module(activation_fn_cfg)
        else:
            self.projection_head = None

        return None

    def _make_projection_head_module(
        self, activation_fn_cfg: omegaconf.DictConfig
    ) -> nn.Module:
        """Factory for the obs-only summary head q_theta (``2O -> ... -> {mean, logvar}``).

        The DEPLOYED prediction head (q_theta is the object emitted at deploy; the GMS decoder is a
        training-time anchor only — see RLRP-704) is the ensemble-aware residual
        :class:`_ProjectionNetworkHead` (a ``_build_network``-style residual trunk + split
        mean/logvar heads, deploy-quality so the single-step prediction is not bottlenecked). It is
        member-aware (``self.num_members`` = ensemble size) so it preserves the ensemble's
        per-member capacity and elite selection at deploy, exactly like the encoder MS stacks. The
        legacy ``head_arch="mlp"`` selector was removed by the PME Projection-Head Consolidation
        `.junie` plan (`refactor_pme_projection_head_consolidation_ensemble_plan_20260622.md`).
        """
        _o = self.singlestep_obs_len
        # RLRP-768 (B4b): resolve the `projection` region bloc builder (default
        # `dense`, LayerNorm/gated forbidden). The projection head builds its
        # blocs directly (for elite forwarding), so the `residual_legacy` bloc
        # (which needs external dropout + trailing activation wrapping) is not
        # supported here -- fail loud instead of silently mis-wiring.
        proj_kind, make_proj_bloc = self._layer_bloc_factory(
            "projection",
            self.num_members,
            lambda: create_activation_(activation_fn_cfg),
            self.projection_head_dropout,
        )
        if proj_kind != "block":
            raise ValueError(
                "The `projection` region does not support the 'residual_legacy' "
                "layer bloc; use 'dense' (default) or 'residual_v2' (RLRP-768)."
            )
        head: nn.Module = _ProjectionNetworkHead(
            num_members=self.num_members,
            obs_len=_o,
            hid=self.projection_head_hidden,
            num_layers=self.projection_head_num_layers,
            split_num_layers=self.projection_head_split_num_layers,
            dropout=self.projection_head_dropout,
            activation_factory=lambda: create_activation_(activation_fn_cfg),
            make_bloc=make_proj_bloc,
        )
        head = head.to(device=self.device)
        if hasattr(self, "model_dtype"):
            head.to(dtype=self.model_dtype)
        return head

    def _build_gms_decoder(self, activation_fn_cfg: omegaconf.DictConfig) -> None:
        """Build the GMS-IWAE generative decoder ``p(y|z)`` (obs-only) — IDENTITY only.

        The decoder is a FIXED identity observation-noise model ``p(y|z) = f(z, sigma_obs)`` (mean
        = z, scale from the per-obs-feature noise model). It is a training-time generative anchor
        and is NOT deployed, so a learned decoder would only turn z into a non-deployable latent and
        break the "deploy q_theta = obs prediction" assumption (RLRP-704). All deployable
        expressiveness lives on the projection head q_theta (``projection.head_arch``). Only the
        per-obs-feature noise scale ``log_sigma_obs`` is configurable / optionally learnable.
        Built ONLY when ``enable_gms_iwae_loss`` is True (same gating as the projection head).
        """
        if self.gms_iwae_decoder_learnable_noise:
            self.gms_iwae_log_sigma_obs = None
        if not self.enable_gms_iwae_loss:
            return None

        _o = self.singlestep_obs_len
        # `model_dtype` is set AFTER `_build_network` (the final `self.to(dtype=...)` casts every
        # parameter/buffer), so build on device only and let the post-build cast handle dtype.
        _dtype = getattr(self, "model_dtype", None)

        # Per-obs-feature observation-noise log-sigma (the identity decoder's only parameter).
        log_sigma_init = float(np.log(max(self.gms_iwae_decoder_obs_noise, 1e-6)))
        _log_sigma = torch.full((_o,), log_sigma_init, device=self.device)
        if _dtype is not None:
            _log_sigma = _log_sigma.to(dtype=_dtype)
        if self.gms_iwae_decoder_learnable_noise:
            self.gms_iwae_log_sigma_obs = nn.Parameter(_log_sigma)
        else:
            self.register_buffer("gms_iwae_log_sigma_obs", _log_sigma)
        return None

    def _make_decoder_dist(self, z: Tensor):
        """Build the obs-only generative decoder distribution ``p(y|z)`` for samples ``z``.

        Identity observation-noise model: mean = z, log-variance = ``2*log_sigma_obs`` (per-obs
        feature), floored at ``2*log(obs_noise)`` (keeps the emission appropriately dispersed for
        the planner) and capped at ``_MIX_LOGVAR_SAFE_MAX``. Returns an ``Independent`` (sum-over-obs)
        Normal/Laplace.
        """
        logvar_floor = 2.0 * float(np.log(max(self.gms_iwae_decoder_obs_noise, 1e-6)))
        mean = z
        logvar = (2.0 * self.gms_iwae_log_sigma_obs).expand_as(z)
        logvar = logvar.clamp(min=logvar_floor, max=self._MIX_LOGVAR_SAFE_MAX)
        return dist.Independent(
            self._to_distribution(mean, logvar, scale_are_log_variance=True), 1
        )

    def _get_deploy_logvar_bound_layer(self) -> Union[LogvarBoundLayer, nn.Module]:
        return self.deploy_head_logvar.get_submodule("logvar_bound")

    def _maybe_toggle_layers_use_only_elite(self, only_elite: bool) -> None:
        """Extend the base encoder/MS elite toggle to ALSO cover the projection head.

        Introduced by Item 1 of the PME Projection-Head Consolidation `.junie` plan
        (`refactor_pme_projection_head_consolidation_ensemble_plan_20260622.md`). The base
        implementation only toggles the encoder hidden / MS-head layers; the ensemble-aware
        projection head (Item 3) must follow the SAME elite selection so the deployed q_theta uses
        the elite members. Like the base, this is a no-op unless ``elite_models`` is set,
        ``num_members > 1`` and ``only_elite``; it is balanced (toggled on then off) by the
        bracketing in :meth:`_default_forward_u` / :meth:`_default_forward_projection_head`.
        """
        super()._maybe_toggle_layers_use_only_elite(only_elite)
        if self.elite_models is None:
            return
        if self.num_members > 1 and only_elite and self.projection_head is not None:
            self.projection_head.set_elite(self.elite_models)
            self.projection_head.toggle_use_only_elite()

    def _default_forward_projection_head(
        self, mean: Tensor, logvar: Tensor, only_elite: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """Single owner of the projection-head forward q_theta (obs-only summary head).

        Introduced by Item 1 of the PME Projection-Head Consolidation `.junie` plan
        (`refactor_pme_projection_head_consolidation_ensemble_plan_20260622.md`). Runs the summary
        head on ``(mean, logvar)`` and SYSTEMATICALLY applies the deploy logvar bound to the head's
        logvar output, so the bound is APPLIED iff a projection head exists (matching
        :meth:`_logvar_bound_penalty_specs`, keeping the *penalise-iff-applied* invariant exact).
        Callers must NOT re-apply the deploy bound downstream (e.g. :meth:`_make_head_dist` no
        longer does). The ``only_elite`` toggle brackets the head forward exactly as
        :meth:`_default_forward_u`, so the ensemble-aware head honours elite-member selection at
        deploy/eval (training call sites pass ``only_elite=False``, preserving full-ensemble
        behaviour).
        """
        self._maybe_toggle_layers_use_only_elite(only_elite)
        q_mean, q_logvar = self.projection_head(mean, logvar)
        q_logvar = self._get_deploy_logvar_bound_layer()(q_logvar)
        self._maybe_toggle_layers_use_only_elite(only_elite)
        # RLRP-736 §4A-B P3.3 (RLRP-738; ★ compounded-error lever). The projection head
        # (`_ProjectionNetworkHead`) is a generic obs-only summary MLP: it emits a learned
        # mean vector with NO `q -> R`/`R -> q` decode and NO renorm, so with
        # `deploy_head.mode="projection"` its attitude slot is generically NON-unit and,
        # since this output is re-injected as `next_obs` on the deploy/CP path, the invalid
        # rotation would silently drift over the horizon — precisely on the rollout-
        # consistency path enabled to MITIGATE compounded error. Canonicalise every tracked
        # single-step attitude output slot back onto S^3 (chordal-L2 projection only;
        # the memoryless `w >= 0` sign step was removed by RLRP-744 action B-3),
        # reusing the P3.2 helper. Bit-exact no-op when the rep is OFF/quaternion or no
        # attitude output slot is tracked (returns `q_mean` untouched).
        q_mean = self._canonicalize_exposed_mean_quaternion(q_mean)
        return q_mean, q_logvar

    def _logvar_bound_penalty_specs(self):
        """Two-bound spec (RLRP-718): the encoder/MS bound (penalised over its composed
        multistep slice ``x[..., -ho_out_size:]``, matching the old MS ``bound_losses``
        adapter) and the deploy/SS bound (identity adapter, matching the old SS call).

        Invariant *penalise iff applied* (RLRP-718 follow-up): the deploy bound is only
        APPLIED on the projection path — Stage A clamps ``q_logvar`` in :meth:`_make_head_dist`
        and the deploy projection branch clamps the head output in
        :meth:`_maybe_apply_deploy_projection_head` — both of which require a projection head to
        exist. It is therefore penalised iff ``self.projection_head is not None``. The SS
        *forecast* output is NOT bounded by the deploy bound (it is already bounded by the
        encoder/MS bound applied in :meth:`_default_forward_u`), so the deploy bound is NOT
        penalised in the SS-on / no-projection regime — this avoids the unanchored
        ``0.01*(max-min)`` runaway that has no NLL counterpart in that regime. Deduped by
        identity in :meth:`_logvar_bound_penalty`.
        """
        specs = []
        enc = self._get_logvar_bound_layer()
        if enc is not None:
            ho_out_size = compute_multistep_model_out_size(
                self.singlestep_obs_len, self.singlestep_act_len, self.horizon_len
            )
            specs.append((enc, lambda x: x[..., -ho_out_size:]))
        if not self.deterministic and self.projection_head is not None:
            # Deploy bound is only applied on the projection path (Stage A + deploy branch),
            # so penalise it iff a projection head exists.
            dep = self._get_deploy_logvar_bound_layer()
            if dep is not None:
                specs.append((dep, None))
        return specs

    def _maybe_apply_deploy_projection_head(
        self,
        ss_mean: Tensor,
        ss_logvar: Tensor,
        only_elite: bool = False,
        deploy_head_mode_override: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Stage B (RLRP-711+): optionally route the deploy output through the compressed summary
        head, gated on `deploy_head.mode` with a safe ``forecast`` fallback.

        The ``deploy_head.source`` selector was removed: the projection objectives (GMS-IWAE /
        SIW-MP / IPROJ) are mutually exclusive, so there is at most one trained projection head.

        ``deploy_head_mode_override`` (post-training deploy-only override) wins over the configured
        ``self.deploy_head_mode`` when not ``None`` — lets a deployment run force ``projection`` /
        ``forecast`` WITHOUT retraining. Validated against ``{forecast, projection}``.

        ``only_elite`` is forwarded to :meth:`_default_forward_projection_head` so the
        ensemble-aware head honours elite-member selection on the projection deploy branch (PME
        Projection-Head Consolidation `.junie` plan,
        `refactor_pme_projection_head_consolidation_ensemble_plan_20260622.md`).
        """
        mode = (
            deploy_head_mode_override
            if deploy_head_mode_override is not None
            else self.deploy_head_mode
        )
        assert mode in {
            "forecast",
            "projection",
        }, f"deploy head mode override must be 'forecast' or 'projection', got {mode}"
        head = None
        if mode == "projection":
            head = self.projection_head
        if head is None:
            if mode == "projection" and not getattr(
                self, "_warned_deploy_head_missing", False
            ):
                consol_msg_universal_one_liner(
                    "deploy_head.mode='projection' requested but no projection head is "
                    "available; falling back to the 'forecast' deploy output."
                )
                self._warned_deploy_head_missing = True
            # RLRP-718 (follow-up): the SS *forecast* (no projection head) path does NOT need the
            # deploy logvar bound. `ss_logvar` here originates from `_default_forward_u`, where the
            # encoder/MS bound (`_get_logvar_bound_layer()`) has ALREADY been applied, and it is
            # additionally guarded by the `variance_floor`/`_MIX_LOGVAR_SAFE_MAX` clamp in
            # `_default_deploy_head`. Re-applying the (separate) deploy bound here was therefore
            # redundant. It is now reserved for the projection branch below (the projection head
            # output is otherwise unbounded) and Stage-A `_make_head_dist`, keeping the
            # *penalise-iff-applied* invariant exact (see `_logvar_bound_penalty_specs`).
            return ss_mean, ss_logvar
        # Projection branch: delegate to the single owner of the projection-head forward, which
        # also SYSTEMATICALLY applies the deploy logvar bound (Stage B.2) to the head output.
        return self._default_forward_projection_head(
            ss_mean, ss_logvar, only_elite=only_elite
        )

    def _default_forward(
        self, x: Tensor, only_elite: bool = False, debug: bool = False, **_kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        debug = debug or self._GLOBAL_DEBUG

        remaining_state_history = None

        state_history, batch_size = self.setup_state_history(x)

        # .... Initialise container ...............................................................
        self.ms_state_forecast_mean = []
        self.ms_state_forecast_logvar = []

        accumulator_shape = (
            self.num_members,
            batch_size,
            self.horizon_len,
            self.horizon_len,
            self.singlestep_obs_len + self.singlestep_act_len,
        )
        self.forecast_mean_accumulator = torch.zeros(
            accumulator_shape, dtype=self.model_dtype, device=self.device
        )
        self.forecast_logvar_accumulator = torch.zeros(
            accumulator_shape, dtype=self.model_dtype, device=self.device
        )

        if self.horizon_len < self.history_len:
            remaining_state_history = self.state_history_remaining(state_history)

        # .... Begin auto-regressive multistep prediction unroll ..................................
        for each_i in np.arange(self.horizon_len):

            # .... Unrol model for U steps ........................................................
            u_ms_mean, u_ms_logvar = self._default_forward_u(state_history, only_elite)

            if (
                self.enable_pre_mixture_u_loss
                and self._pre_mixture_loss_target is not None
            ):
                self._pre_mixture_loss(
                    ms_mean=u_ms_mean,
                    ms_logvar=u_ms_logvar,
                    i=each_i,
                    ms_target=self._pre_mixture_loss_target,
                )

            # .... Update prediction forecast container ...........................................
            # (E x B x MS x O+A) or (E x MS x O+A)
            if self.unroll_len > 1:
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
            else:
                u_ms_mean = u_ms_mean.unsqueeze(-2)
                u_ms_logvar = u_ms_logvar.unsqueeze(-2)

            col_end = min(each_i + self.unroll_len, self.horizon_len)
            trim_len = col_end - each_i

            # 💎 Out-of-place accumulator update to avoid breaking autograd graph
            if debug:
                SENTINEL = float("nan")  # or a unique large value like 1e30
                forecast_mean_contribution = torch.full(
                    accumulator_shape,
                    SENTINEL,
                    dtype=self.model_dtype,
                    device=u_ms_mean.device,
                )
            else:
                forecast_mean_contribution = torch.zeros(
                    accumulator_shape,
                    dtype=self.model_dtype,
                    device=u_ms_mean.device,
                )

            # Only the valid slice gets overwritten with actual data
            forecast_mean_contribution[
                ..., each_i, each_i:col_end, : u_ms_mean.shape[-1]
            ] = u_ms_mean[..., :trim_len, :]

            # if debug: # ToDo: on task end >> UN-mute next bloc ↓↓
            #     assert not u_ms_mean[..., :trim_len, :].isnan().any()
            #     assert (
            #         not forecast_mean_contribution[
            #             ..., each_i, each_i:col_end, : u_ms_mean.shape[-1]
            #         ]
            #         .isnan()
            #         .any()
            #     )

            self.forecast_mean_accumulator = (
                self.forecast_mean_accumulator + forecast_mean_contribution
            )
            # if debug: # (Priority) ToDo: on task end >> UN-mute next bloc ↓↓
            #     assert not self.forecast_mean_accumulator.isnan().all()

            # 💎 Out-of-place accumulator update to avoid breaking autograd graph
            forecast_logvar_contribution = torch.zeros(
                accumulator_shape,
                dtype=self.model_dtype,
                device=u_ms_logvar.device,
            )
            forecast_logvar_contribution[
                ..., each_i, each_i:col_end, : u_ms_logvar.shape[-1]
            ] = u_ms_logvar[..., :trim_len, :]
            self.forecast_logvar_accumulator = (
                self.forecast_logvar_accumulator + forecast_logvar_contribution
            )

            # .... Update MS state forecast .......................................................
            ss_mean, ss_logvar = self._default_forward_mixture_dist(
                self.forecast_mean_accumulator,
                self.forecast_logvar_accumulator,
                each_i,
            )

            # (CRITICAL) ToDo: assess if we should add  `ss_logvar = self._get_deploy_logvar_bound_layer()(ss_logvar)` here since the mixture bound is different than the u-forward one (ref task RLRP-764)
            # (CRITICAL) ToDo: if it work, assess if we can get rid of the clampimg safeguard downstream (ref task RLRP-764)
            # raise NotImplementedError("(CRITICAL) ToDo: validate (ref task RLRP-764)")
            # ss_logvar = self._get_deploy_logvar_bound_layer()(ss_logvar)

            self.ms_state_forecast_mean.append(ss_mean)
            self.ms_state_forecast_logvar.append(ss_logvar)

            # .... Accumulate post-mixing / true-mixture per-step NLL (Stage B/G) .................
            # The per-step temporal mixture is only available inside this loop; accumulate the
            # post-mixing objective(s) here when a target was provided by `_probabilistic_loss`.
            if self._post_mixture_loss_target is not None and (
                self.enable_post_mixture_loss
                or self.ms_probabilistic_loss_mode == "true_mixture"
            ):
                self._accumulate_post_mix_losses(each_i)
            # RLRP-830 Item B1: the stash is scoped to this iteration only.
            self._current_step_mixture = None

            # .... Update MS state history ........................................................
            state_history = self.state_history_update(
                state_history,
                self.forecast_mean_accumulator,
                self.forecast_logvar_accumulator,
                each_i,
            )

        # .... Compute multi-step forecast ........................................................

        ms_mean = torch.stack(self.ms_state_forecast_mean, dim=-2)
        ms_logvar = torch.stack(self.ms_state_forecast_logvar, dim=-2)

        # (CRITICAL) ToDo: validate new output shape to HO instead of the legacy HI MS len (ref task RLRP-681)
        # if self.horizon_len < self.history_len:
        #     if remaining_state_history.dim() < ms_mean.dim() and not self.training:
        #         remaining_state_history = remaining_state_history.unsqueeze(0)
        #
        #     ms_mean = torch.dstack([remaining_state_history, ms_mean])
        #     # Use logvar=0 (unit variance) for the remaining history entries that
        #     # were NOT predicted by the model.  The previous value
        #     # (_LOGVAR_MIN_LIMIT ≈ -69) created near-zero scale in the Laplace
        #     # distribution, making NLL explode for any tiny target mismatch
        #     # (~1e15 per element) and completely dominating the training loss.
        #     ms_logvar = torch.dstack(
        #         [
        #             torch.zeros_like(remaining_state_history),
        #             ms_logvar,
        #         ]
        #     )

        if debug:
            assert (
                ms_mean.dim() >= 3
            ), f"ms_mean dimension should be >= 3, got {ms_mean.dim()} with {ms_mean.shape}"
            expected_shape = f"(..., {self.horizon_len}, {self.singlestep_obs_len + self.singlestep_act_len})"

            # (CRITICAL) ToDo: validate new output shape to HO instead of the legacy HI MS len (ref task RLRP-681)
            # assert (ms_mean.shape[-2] == self.history_len) and (
            #     ms_mean.shape[-1] == self.singlestep_obs_len + self.singlestep_act_len
            # ), f"ms_mean shape expected shape {expected_shape}, got {ms_mean.shape}"
            assert (ms_mean.shape[-2] == self.horizon_len) and (
                ms_mean.shape[-1] == self.singlestep_obs_len + self.singlestep_act_len
            ), f"ms_mean shape expected shape {expected_shape}, got {ms_mean.shape}"

        ms_mean = revert_timestep_first_multistep_dim_unflaten_array(
            ms_mean,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            remove_last_action_padding=True,
        )

        ms_logvar = revert_timestep_first_multistep_dim_unflaten_array(
            ms_logvar,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            remove_last_action_padding=True,
        )

        return ms_mean, ms_logvar

    def setup_state_history(self, x: Tensor) -> tuple[StateHistory, int]:
        state_history = StateHistory(h_mean=x)

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
                f"Support for {state_history.h_mean.dim()}D input and torch training {self.training} not implemented"
            )

        # RLRP-735: capture the FROZEN input-history realisation ``s_τ^H`` (D2/D4-a) for the
        # index_and_history_dependent mixer. Flag-gated so index / input_dependent runs stay
        # allocation-free (review item 3). ``x`` is the raw ``model_input`` (means-only): 2-D
        # (B x D_h) in eval/deploy, 3-D (E x B x D_h) in training; the mixer head broadcasts the
        # 2-D case over the ensemble axis. The token is detached (history is data, not a gradient
        # path into the prediction body) and ``ψ`` is computed once (constant across ``i``).
        self._maybe_capture_forecast_history_token(x, state_history)
        return state_history, batch_size

    def _maybe_capture_forecast_history_token(
        self, x: Tensor, state_history: StateHistory
    ) -> None:
        """Build the frozen ``s_τ^H`` token + cached ``ψ`` for the history-conditioned mixer.

        No-op unless ``self._mixer_is_index_and_history_dependent`` (flag-gated, review item 3),
        so the other kinds incur no per-forward ``detach().clone()`` allocation. Populates
        :py:attr:`StateHistory.h_obs_original` (short-circuiting if a subclass — e.g. the
        resampled-particle model — already snapshotted it) and stashes the built token on
        ``self._forecast_history_token`` and its once-computed head output on
        ``self._forecast_history_psi`` (D4-a per-forward attributes).
        """
        if not self._mixer_is_index_and_history_dependent:
            return
        # Reuse the RLRP-647 frozen snapshot if a subclass already captured it; otherwise take it.
        if state_history.h_obs_original is None:
            state_history.h_obs_original = x.detach().clone()
        token = state_history.h_obs_original.detach().to(
            device=self.device, dtype=self.model_dtype
        )
        self._forecast_history_token = token
        # ψ is constant across the i..F AR steps (s_τ^H frozen), so compute it ONCE here.
        self._forecast_history_psi = self.temporal_mixture_weights.compute_psi(token)
        return None

    def state_history_remaining(self, state_history: StateHistory) -> Tensor:
        # Reshape data to
        # (..., O[1:Do]_1 + ... + O[1:Do]_MS + A[1:Da]_1 + ... + A[1:Da]_MS ) --> (..., MS, O+A)
        _state_history: Tensor = timestep_first_multistep_dim_unflaten_array(
            state_history.h_mean,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            sequence_len=self.history_len,
            enable_last_action_padding=False,
        )

        remaining_len = self.history_len - self.horizon_len

        remaining_state_history = _state_history[..., -remaining_len:, :]

        return remaining_state_history

    @torch.compiler.disable
    def state_history_update(
        self,
        state_history: StateHistory,
        forecast_mean_accumulator: Tensor,
        forecast_logvar_accumulator: Tensor,
        horizon_index: int,
        debug: bool = False,
    ) -> StateHistory:
        debug = debug or self._GLOBAL_DEBUG

        # RLRP-619 resolution: the denormalize->shift->normalize cycle (step 1
        # below + step 2 further down) is STILL REQUIRED for the legacy
        # ``ZScoreNormalizer`` (``input_normalizer``) path because that path
        # is asymmetric: model **input** is normalized via
        # ``input_normalizer.normalize`` (see ``OneDTransitionRewardModelV2.
        # _get_model_input``), but the **target** is left in raw data space
        # (``_process_batch`` keeps ``target_obs = next_obs_t`` when
        # ``target_is_delta=False`` and ``_uses_block_facade=False``).
        # As a consequence, ``ss_dist_mixture.sample()`` below produces a
        # draw in raw data space, while ``state_history.h_mean`` arrives in
        # normalized space, so one coordinate change is unavoidable to
        # concatenate them coherently.
        #
        # For every **block-facade** normalizer regime — ``SoftWinsorizedNormalizer``,
        # ``QuantileNormalizer`` and the RLRP-684 (A1) symmetric Z-score path
        # ``normalizer_type="standard_symmetric"`` — the target IS normalized too
        # (``_process_batch`` calls ``_normalize_composed_obs(next_obs_t)``),
        # so mixture samples are in normalized space and the cycle becomes a
        # no-op. The ``if self.get_one_d_trj_model_input_normalizer():``
        # guard correctly short-circuits the round-trip in that case
        # (``input_normalizer`` is ``None`` for every block facade; only the
        # asymmetric ``"standard"`` Z-score path keeps it non-``None`` and
        # genuinely needs the round-trip). See also
        # ``.junie/ai_artifact/reports/report_rlrp530_denorm_norm_skip_rationale.md``.

        # Denormalize->normalize model input (step 1)
        if self.get_one_d_trj_model_input_normalizer():
            state_history.h_mean = (
                self.get_one_d_trj_model_input_normalizer().denormalize(
                    state_history.h_mean
                )
            )

        # Reshape data to
        # (..., O[1:Do]_1 + ... + O[1:Do]_MS + A[1:Da]_1 + ... + A[1:Da]_MS ) --> (..., MS, O+A)
        state_history.h_mean = timestep_first_multistep_dim_unflaten_array(
            state_history.h_mean,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            sequence_len=self.history_len,
            enable_last_action_padding=False,
        )

        state_history.h_mean = state_history.h_mean[..., 1:, :]

        # Note: Correspond to version step sampling mixture
        ss_dist_mixture = self._temporal_mixture_at_i(
            forecast_mean_accumulator,
            forecast_logvar_accumulator,
            horizon_index,
            obs_space_only=False,
        )
        next_obs = ss_dist_mixture.sample()  # (B x E x MS)

        # (CRITICAL) ToDo: validate ↓↓ (ref task RLRP-530)
        # Guard against extreme AR samples: when mixture components disagree,
        # the mixture variance is large and .sample() can produce values that
        # overflow float32 during the denormalize→normalize round-trip below
        # (denormalize computes std * sample + mean, which overflows if
        # |sample| > ~3.4e38 / std).
        # Clamp samples to a safe range in normalized space and replace any
        # remaining non-finite values with the mixture mean.
        next_obs = next_obs.clamp(-self.ar_sample_clamp, self.ar_sample_clamp)
        if not torch.isfinite(next_obs).all():
            next_obs = torch.where(
                torch.isfinite(next_obs), next_obs, ss_dist_mixture.mean
            )

        # Note: with robust normalizers, state_history and next_obs (from the
        # mixture sample) are both in normalized space — no denormalization
        # needed.  The denormalize→shift→renormalize cycle via input_normalizer
        # (below) only applies to the standard normalizer path.

        # .... Forecast-path teacher forcing (scheduled sampling, RLRP-726) .......................
        # DEFAULT-OFF: when the dedicated scheduler is disabled (or outside training, or no GT
        # target was stashed) this branch is skipped and the free-running mixture sample above is
        # used verbatim (byte-for-byte parity). On a "teacher" step, splice the ground-truth obs+act
        # `self._forecast_tf_target[..., horizon_index, :]` into the next window instead of the
        # mixture sample. The GT target is in the SAME space as the mixture sample at this point
        # (raw for the asymmetric `standard` normalizer, normalized for the block-facade regimes),
        # mirroring the CP splice, so it rides the SAME step-2 round-trip below with no extra
        # coordinate change. The per-step coin is drawn at the frozen probability (the scheduler is
        # stepped once per loss in `_probabilistic_loss`, never here).
        # RLRP-786: ``cuda_graph_capture_ready`` is INERT on this particle variant (the sampled
        # splice is not capturable; ``_CUDA_GRAPH_CAPTURABLE_VARIANT`` is False so the mask is
        # never refilled here) -> the Python branch stays the one and only coin consumer.
        if (
            self.training
            and self._forecast_tf_target is not None
            and self._forecast_teacher_forcing_scheduler is not None
            and self._forecast_teacher_forcing_scheduler.should_use_teacher_forcing()
        ):
            next_obs = self._forecast_tf_target[..., horizon_index, :].to(
                dtype=next_obs.dtype
            )

        # RLRP-736 Item 1: on the free-running MS forecast self-feed, sign-align the
        # re-injected attitude prediction to the LAST retained history frame
        # (``⟨q_pred, q_ref⟩ ≥ 0`` + L2-projection to S^3) so the history buffer
        # stays hemisphere-continuous. This REPLACES the memoryless ``w >= 0``
        # canonicaliser (``_canonicalize_reinjected_quaternion``) for the covered reps
        # (``quaternion`` / ``sixd`` / ``nine_d_svd``) and is applied strictly last;
        # ``quaternion_legacy`` (and the no-attitude cases) fall back to the
        # byte-for-byte pre-RLRP-736 canonicaliser.
        # RLRP-761 S4.4: ``next_obs`` (mixture sample or spliced GT) lives in
        # TARGET space, ``state_history.h_mean`` in INPUT space. Under
        # ``standard_symmetric_innovation`` those differ by the diagonal bridge
        # gain; for every other type the gain is unregistered and this is a
        # strict identity. Applied BEFORE the attitude continuity alignment,
        # which compares against ``state_history.h_mean`` and must therefore see
        # an input-space value (the gain is 1 on ``unit_norm`` dims by
        # construction, so the quaternion itself is untouched either way).
        # RLRP-761 S12.4/S12.9: ``next_obs`` is the full composed step (obs+act),
        # both PREDICTED and re-injected, so BOTH slices cross the bridge — the
        # obs via ``ar_bridge_gain`` (``S4.4``) and the act (commands / ``dt``)
        # via ``ar_bridge_gain_act`` (``S12.4``). Strict identity for every
        # non-decoupled type (``M5``).
        next_obs = self._ar_bridge_step_target_to_input(next_obs)

        if self._quaternion_ar_continuity_active():
            q_ref = state_history.h_mean[..., -1, :]
            next_obs = align_quaternion_slots_to_reference(
                next_obs, q_ref, self._orientation_singlestep_slots
            )
        else:
            next_obs = self._canonicalize_reinjected_quaternion(next_obs)

        next_obs = next_obs.unsqueeze(dim=-2)

        if not self.training and next_obs.dim() > state_history.h_mean.dim():
            # Quick-hack for casse where forward is run from deploy and we need to reduce
            # the ensembles dimensions. (NICE TO HAVE) ToDo: implement ensemble decision
            next_obs = next_obs.mean(dim=0)

        next_state_history_mean = torch.concatenate(
            [state_history.h_mean, next_obs], dim=-2
        )

        # Reshape data to
        # (..., MS, O+A) --> (..., O[1:Do]_1 + ... + O[1:Do]_MS + A[1:Da]_1 + ... + A[1:Da]_MS)
        next_state_history_mean = revert_timestep_first_multistep_dim_unflaten_array(
            next_state_history_mean,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            remove_last_action_padding=False,
        )

        # Denormalize->normalize model input (step 2)
        if self.get_one_d_trj_model_input_normalizer():
            next_state_history_mean = (
                self.get_one_d_trj_model_input_normalizer().normalize(
                    next_state_history_mean
                )
            )

        if debug:
            assert (
                next_state_history_mean.dim() >= 2
            ), f"next_state_history_mean dimension should be >= 2, got {next_state_history_mean.dim()} with {next_state_history_mean.shape}"

            _expected_compose_obs_len = compute_multistep_model_in_size(
                singlestep_obs_len=self.singlestep_obs_len,
                singlestep_act_len=self.singlestep_act_len,
                multistep_len=self.history_len,
            )
            expected_shape = f"(..., {_expected_compose_obs_len})"
            assert (
                next_state_history_mean.shape[-1] == _expected_compose_obs_len
            ), f"next_state_history_mean shape expected shape {expected_shape}, got (..., {next_state_history_mean.shape[-1]})"

            # # ToDo: on task end >> UN-mute next bloc ↓↓
            # assert not torch.isnan(
            #     next_state_history_mean
            # ).any(), f"next_state_history_mean contains 'nan'"

        return StateHistory(h_mean=next_state_history_mean)

    @torch.compiler.disable
    def _default_forward_mixture_dist(
        self,
        forecast_mean_accumulator: Tensor,
        forecast_logvar_accumulator: Tensor,
        horizon_index: int,
        obs_space_only: bool = False,
    ) -> tuple[Tensor, Tensor]:

        ss_dist_mixture = self._temporal_mixture_at_i(
            forecast_mean_accumulator,
            forecast_logvar_accumulator,
            horizon_index,
            obs_space_only=obs_space_only,
        )
        if getattr(self, "reuse_step_mixture", False):
            # RLRP-830 Item B1: expose this step's mixture to ``_accumulate_post_mix_losses``
            # (same iteration; cleared by ``_default_forward``).
            self._current_step_mixture = (horizon_index, obs_space_only, ss_dist_mixture)

        ss_mean = ss_dist_mixture.mean
        # RLRP-736 §4A-B P3.2 (RLRP-738): the moment-matched exposed mean is generically
        # NON-unit for the attitude slot(s); project it back onto S^3 so the value
        # re-injected as ``next_obs`` in the CP unroll is a valid rotation (no-op OFF).
        ss_mean = self._canonicalize_exposed_mean_quaternion(ss_mean)

        # (CRITICAL) ToDo: validate ↓↓ (ref task RLRP-530)
        # Clamp variance before log to prevent -inf (near-zero var) and extreme
        # positive logvar (component mean disagreement) that would overflow
        # float32 in the subsequent _to_distribution exp() call.
        # Dtype-aware variance floor (RLRC Gate C): float32 keeps 1e-10, float64
        # uses 1e-30 so double-precision multi-step rollouts retain dynamic range.
        ss_logvar = torch.log(
            ss_dist_mixture.variance.clamp(min=variance_floor(self.model_dtype))
        )
        ss_logvar = ss_logvar.clamp(max=self._MIX_LOGVAR_SAFE_MAX)

        # ToDo: validate droping logvar-bound since the _default_forward_u already apply a learned bound
        # ss_logvar = self._get_logvar_bound_layer()(ss_logvar)
        return ss_mean, ss_logvar

    @torch.compiler.disable
    def _temporal_mixture_at_i(
        self,
        forecast_mean_accumulator: Tensor,  # (E x B x F x F x O+A)
        forecast_logvar_accumulator: Tensor,  # (E x B x F x F x O+A)
        horizon_index: int,
        scale_are_log_variance: bool = True,
        obs_space_only: bool = False,
        validate_args: bool = False,
        debug: bool = False,
        detach_components: bool = False,
    ) -> Union[
        dist.Laplace,
        dist.Normal,
        dist.StudentT,
        dist.ExponentialFamily,
        dist.MixtureSameFamily,
    ]:
        debug = debug or self._GLOBAL_DEBUG

        U = self.unroll_len
        i = horizon_index

        # (E x B x F x F x O+A) -> (E x B x min(i,U) x O+A)
        pred_mean_i = forecast_mean_accumulator[..., max(i - U + 1, 0) : i + 1, i, :]
        pred_logvar_i = forecast_logvar_accumulator[
            ..., max(i - U + 1, 0) : i + 1, i, :
        ]

        # Tokens for an input-dependent mixer keep the FULL O+A width (component-vs-token split,
        # see plan Stage A/B); the index mixer ignores them.
        token_mean_i = pred_mean_i
        token_logvar_i = pred_logvar_i

        if obs_space_only:
            # (E x B x min(i,U) x O+A) -> # (E x B x min(i,U) x O)
            pred_mean_i = pred_mean_i[..., : self.singlestep_obs_len]
            pred_logvar_i = pred_logvar_i[..., : self.singlestep_obs_len]

        if debug:
            assert (
                pred_mean_i.dim() == 4
            ), f"Expected 4 dimensions i.e., (E x B x U x O+A), got {pred_mean_i.dim()}"

        ms_distribution_mixture = self._temporal_mixture_distribution_at_i(
            pred_mean_i,
            pred_logvar_i,
            i,
            scale_are_log_variance,
            validate_args,
            debug,
            detach_components=detach_components,
            token_mean_i=token_mean_i,
            token_logvar_i=token_logvar_i,
        )

        return ms_distribution_mixture

    @torch.compiler.disable
    def _temporal_mixture_distribution_at_i(
        self,
        pred_mean_i: Tensor,
        pred_logvar_i: Tensor,
        i: int,
        scale_are_log_variance: bool = True,
        validate_args: bool = False,
        debug: bool = False,
        nam_to_num: bool = False,
        detach_components: bool = False,
        token_mean_i: Optional[Tensor] = None,
        token_logvar_i: Optional[Tensor] = None,
    ) -> MixtureSameFamily:
        debug = debug or self._GLOBAL_DEBUG

        # `pred_*_i` are the mixture COMPONENT tensors (possibly obs-only width O); `token_*_i`
        # are the FULL O+A width tensors feeding an input-dependent mixing-weight network. When
        # not provided they fall back to the components (index-mixer / deploy call convention).
        if token_mean_i is None:
            token_mean_i = pred_mean_i
        if token_logvar_i is None:
            token_logvar_i = pred_logvar_i

        # Backprop-scope lever (Stage B.3 / option 2.1 'mixing_only'): detach BOTH the component
        # tensors and the mixing-weight token tensors. The mixing function's own parameters still
        # receive gradient (detach only stops the flow PAST the mixer input into the prediction
        # body), so this isolates training to the mixing function for input-dependent mixers (this
        # variant + transformer subclasses); the index mixer is unaffected either way.
        if detach_components:
            pred_mean_i = pred_mean_i.detach()
            pred_logvar_i = pred_logvar_i.detach()
            token_mean_i = token_mean_i.detach()
            token_logvar_i = token_logvar_i.detach()

        if debug:
            # # (Priority) ToDo: on task end >> UN-mute next bloc ↓↓
            # assert not torch.isnan(pred_mean_i).any(), f"pred_mean_i contains 'nan'"
            # assert not torch.isnan(pred_logvar_i).any(), f"pred_logvar_i contains 'nan'"
            is_finite = torch.isfinite(pred_mean_i).all().item()
            is_finite = torch.isfinite(pred_logvar_i).all().item() and is_finite

        if nam_to_num:
            pred_mean_i = torch.nan_to_num(pred_mean_i)
            pred_logvar_i = torch.nan_to_num(pred_logvar_i)

        ms_distribution = super()._to_distribution(
            pred_mean_i, pred_logvar_i, None, scale_are_log_variance, validate_args
        )
        ms_distribution = dist.Independent(
            ms_distribution, reinterpreted_batch_ndims=1, validate_args=validate_args
        )  # Make the features dim independent

        batch_size = pred_mean_i.shape[1]

        if debug:
            assert ms_distribution.batch_shape == torch.Size(
                (self.num_members, batch_size, pred_mean_i.shape[-2])
            ), f"Expected batch shape (E x B x min(i,U)), got {ms_distribution.batch_shape}"
            assert ms_distribution.event_shape == torch.Size(
                (pred_mean_i.shape[-1],)
            ), f"Expected event shape (O+A), got {ms_distribution.event_shape}"

        if self._mixer_is_index_and_history_dependent:
            # RLRP-735: index + input-history conditioned mixer. Per-step logits come from the
            # ensemble-aware residual head over the FROZEN ``s_τ^H`` token (means-only per D2),
            # built once at forecast entry (`setup_state_history`) with its head output ``ψ``
            # cached (constant across ``i``). Output already batched (E x B x min(i+1,U)).
            assert self._forecast_history_token is not None, (
                "index_and_history_dependent mixer requires the frozen s_τ^H token to be built "
                "by setup_state_history before the mixer call."
            )
            # The combined ``additive_index_input_and_history_dependent`` kind also consumes the
            # per-step tokens ``cat([mean, logvar])`` (already detached above iff
            # ``detach_components``); the other history kinds ignore ``input_tokens=None`` via the
            # base-class no-op kwarg, so a SINGLE branch serves all history kinds.
            input_tokens = None
            if self._mixer_uses_input_tokens:
                input_tokens = torch.cat([token_mean_i, token_logvar_i], dim=-1)
                # (E x B x min(i,U) x 2(O+A))
            mix_gamma_log_prob = self.temporal_mixture_weights(
                self._forecast_history_token,
                i,
                log_prob=True,
                psi=self._forecast_history_psi,
                input_tokens=input_tokens,
            )
        elif self._mixer_is_input_and_index_dependent:
            # RLRP-735: index + input conditioned mixer (pure step-indexed γ + per-step input head
            # ρ, NO frozen history). Per-step logits from the (detach-honouring) full-width tokens.
            input_tokens = torch.cat([token_mean_i, token_logvar_i], dim=-1)
            # (E x B x min(i,U) x 2(O+A))
            mix_gamma_log_prob = self.temporal_mixture_weights(
                input_tokens, i, log_prob=True
            )
            # Output already batched (E x B x min(i,U))
        elif self._mixer_is_input_dependent:
            # Input-dependent mixer: per-step logits from the NON-detached full-width tokens.
            input_tokens = torch.cat([token_mean_i, token_logvar_i], dim=-1)
            # (E x B x min(i,U) x 2(O+A))
            mix_gamma_log_prob = self.temporal_mixture_weights(
                input_tokens, log_prob=True
            )
            # Output already batched (E x B x min(i,U))
        else:
            mix_gamma_log_prob = self.temporal_mixture_weights(i, log_prob=True)
            # Output (E x min(i,U))
            # Add batch dim (E x B x min(i,U))
            mix_gamma_log_prob = mix_gamma_log_prob.unsqueeze(1)

        if debug:
            is_finite = torch.isfinite(mix_gamma_log_prob).all().item() and is_finite

        if nam_to_num:
            mix_gamma_log_prob = torch.nan_to_num(mix_gamma_log_prob)

        mix_distribution = dist.Categorical(
            logits=mix_gamma_log_prob, validate_args=validate_args
        )
        if not self._mixer_is_per_batch:
            # Index mixer logits are batch-agnostic (E x 1 x min(i,U)); broadcast over B. The
            # per-batch mixers (input_dependent + index_and_history_dependent) already produce
            # per-batch logits (E x B x min(i,U)), so they must NOT be re-expanded (review item 1).
            mix_distribution = mix_distribution.expand(
                batch_shape=torch.Size((self.num_members, batch_size))
            )

        if debug:
            if not is_finite:
                print(
                    f"\n{ConsoleFormat.MSG_ERROR_FORMAT}"
                    f"{consol_msg_universal_one_liner('Non finite value detected!', print_it=False)}"
                    f"{ConsoleFormat.MSG_END_FORMAT}\n"
                )
            assert mix_distribution.batch_shape == torch.Size(
                (self.num_members, batch_size)
            ), f"Expected batch shape (E x B), got {mix_distribution.batch_shape}"
            assert (
                mix_distribution.event_shape == torch.Size()
            ), f"Expected event shape (min(i,U)), got {mix_distribution.event_shape}"

        # RLRP-736 §4A-B P3.1 (RLRP-738, ★ compounded-error lever): make the TEMPORAL
        # MIXTURE manifold-native for the attitude slot(s). When the by-construction
        # PROBABILISTIC attitude split is active we (i) PRODUCE the per-slot 3-D
        # ``so3``-tangent log-variance HERE, at component-construction time, via the
        # dedicated learnable ``4 -> 3`` head (moved OUT of the scoring-only path), and
        # carry it on the mixture object, and (ii) return a mixture whose exposed
        # ``.mean`` canonicalises each attitude slot back onto the unit-quaternion
        # manifold — so the CP-re-injected mean stays a valid rotation and uncertainty
        # propagates manifold-consistently (Forster et al. 2016; Geist et al. 2024). The
        # NON-attitude component marginals + the whole OFF path are BIT-EXACT: the plain
        # ``dist.MixtureSameFamily`` is returned unchanged when the split is inactive.
        ms_distribution_mixture = self._finalize_temporal_mixture(
            mix_distribution, ms_distribution, validate_args
        )
        return ms_distribution_mixture

    def _finalize_temporal_mixture(
        self,
        mix_distribution: dist.Categorical,
        ms_distribution: dist.Independent,
        validate_args: bool = False,
    ) -> MixtureSameFamily:
        """Wrap the categorical + component distributions into the temporal mixture.

        RLRP-736 §4A-B P3.1 (RLRP-738, ★ compounded-error lever). Shared final step of
        ``_temporal_mixture_distribution_at_i`` used by BOTH the base method AND the
        transformer subclasses (``…_trsf_{original,v1,v2}``), so the manifold-native
        attitude path lands identically everywhere and cannot silently diverge between
        base and subclass. When the by-construction PROBABILISTIC attitude split is active
        (``_mixture_orientation_nll_active()``) it returns a ``_ManifoldMeanMixtureSameFamily``
        whose exposed ``.mean`` canonicalises each attitude slot back onto the unit-quaternion
        manifold and which CARRIES the per-slot 3-D ``so3``-tangent log-variance produced by
        the dedicated ``4 -> 3`` head at component-construction time (Forster et al. 2016;
        Geist et al. 2024). Otherwise it returns a plain ``dist.MixtureSameFamily`` — BIT-EXACT
        with the pre-P3.1 code when the split is inactive.
        """
        if self._mixture_orientation_nll_active():
            ms_distribution_mixture = _ManifoldMeanMixtureSameFamily(
                mix_distribution, ms_distribution, validate_args
            )
            ms_distribution_mixture._attitude_out_slots = tuple(
                sorted(self._ss_ori_out_slots)
            )
            ms_distribution_mixture._attitude_variance_eps = variance_floor(
                self.model_dtype
            )
            ms_distribution_mixture._attitude_tangent_log_var_by_slot = (
                self._build_component_attitude_tangent_log_var(ms_distribution)
            )
        else:
            ms_distribution_mixture = dist.MixtureSameFamily(
                mix_distribution, ms_distribution, validate_args
            )
        return ms_distribution_mixture

    def _build_component_attitude_tangent_log_var(
        self, component_distribution: dist.Independent
    ) -> Dict[int, Tensor]:
        """Build, per attitude slot, the 3-D ``so3``-tangent log-variance CARRIED by the
        mixture components.

        RLRP-736 §4A-B P3.1 (RLRP-738). This moves the ``4 -> 3`` tangent-logvar HEAD out
        of the scoring-only path and INTO component construction: for every single-step
        attitude output slot ``s`` (``_ss_ori_out_slots``) it maps the component's 4-D
        quaternion-space log-variance ``2 log sigma`` (per particle ``k``) to a genuine
        3-D ``so3``-tangent log-variance (Forster et al. 2016; Geist et al. 2024). The
        dedicated learnable head (``_build_attitude_tangent_logvar_head``, initialised to
        the quaternion vector-part) is preferred; the fixed vector-part placeholder is the
        fallback when the head is absent. Returned as a ``{slot: (E x B x K x 3)}`` dict so
        scoring consumes exactly the tangent the components were built with.
        """
        base_component = component_distribution.base_dist
        comp_scale = base_component.scale  # (E x B x K x D) component std
        eps = variance_floor(self.model_dtype)
        head = getattr(self, "_attitude_tangent_logvar_head", None)
        head_slots = getattr(self, "_attitude_tangent_logvar_head_slots", ())
        tangent_by_slot: Dict[int, Tensor] = {}
        width = comp_scale.shape[-1]
        for s in sorted(self._ss_ori_out_slots):
            if s + 4 > width:
                continue
            scale_k = comp_scale[..., s : s + 4]  # (E x B x K x 4)
            if head is not None and s in head_slots:
                comp_logvar_k = 2.0 * torch.log(scale_k.clamp_min(eps))
                tangent_log_var = head[head_slots.index(s)](comp_logvar_k)
            else:
                tangent_log_var = 2.0 * torch.log(scale_k[..., 1:4].clamp_min(eps))
            tangent_by_slot[s] = tangent_log_var
        return tangent_by_slot

    def _default_forward_u(
        self, state_history: StateHistory, only_elite: bool
    ) -> tuple[Tensor, Tensor]:
        self._maybe_toggle_layers_use_only_elite(only_elite)

        state_history.h_mean = self._maybe_cast_to_model_dtype(state_history.h_mean)

        # RLRP-736 §4A-B P1.2 (mean path): encode the attitude input slot(s) of the
        # (history-window) input to the internal rep BEFORE the trunk so the encoder
        # consumes the continuous, sign-invariant representation (``R(q)=R(-q)``; Geist
        # et al. 2024). Applied to a LOCAL view only (``state_history.h_mean`` is left in
        # the external layout so the history token / mixer + the AR ``state_history_update``
        # round-trip stay representation-agnostic). No-op / bit-exact when the rep is OFF.
        h_in = self._apply_orientation_input_encoding(state_history.h_mean)

        u_ms_head = self.hidden_layers(h_in)

        u_ms_mean_and_logvar = self.mean_and_logvar(u_ms_head)

        # Architecture 1
        u_ms_mean_split, u_ms_logvar_split = torch.chunk(
            u_ms_mean_and_logvar, chunks=2, dim=-1
        )
        u_ms_mean_head = self.mean_layer(u_ms_mean_split)
        # RLRP-736 §4A-B P1.2 (mean path): decode every per-unroll-block attitude slot of
        # the rep-expanded mean head back to a unit quaternion, so the composed forecast
        # AND the deploy single-step prediction (both funnel through this method) are unit
        # quaternions by construction and downstream code sees the external width. No-op /
        # bit-exact when the rep is OFF.
        u_ms_mean_head = self._apply_unroll_orientation_output_decoding(u_ms_mean_head)
        u_ms_logvar_head = self.logvar_layer(u_ms_logvar_split)
        u_ms_logvar_head = self._get_logvar_bound_layer()(u_ms_logvar_head)

        # Architecture 2
        # u_ms_mean_head = self.mean_layer(u_ms_mean_and_logvar)
        # u_ms_logvar_head = self.logvar_layer(u_ms_mean_and_logvar)

        self._maybe_toggle_layers_use_only_elite(only_elite)
        return u_ms_mean_head, u_ms_logvar_head

    def deploy(
        self,
        x: Tensor,
        only_elite: bool = True,
        deploy_head_mode_override: Optional[str] = None,
        **_kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Output the multi-step model deploy head (instead of the main head).

        :param x: Input tensor to be processed by the model.
        :param only_elite: If ``True``, only the elite subset of the deployment head
            is used. Defaults to ``True``.
        :param deploy_head_mode_override: An optional string to override the deployment
            head mode. If not specified, the default deployment mode is used.
        :param _kwargs: Additional unused keyword arguments, included for flexibility
            in function signature.
        :return: A tuple containing:
                  - The mean tensor of the predicted distribution.
                  - The log-variance tensor of the predicted distribution (or
                    ``None`` if not applicable).
        """
        x = self._maybe_cast_to_model_dtype(x)

        ss_mean, ss_logvar = self._default_deploy_head(
            x, only_elite, deploy_head_mode_override=deploy_head_mode_override
        )

        # R4 (RLRP-708): this in-class ``deploy`` (defined in the MTM-Pro class body) OVERRIDES
        # the mixin ``CompoundedPredictionMultiStepIterator.deploy`` (a directly-defined method
        # wins over an inherited one regardless of MRO order). So the deploy-prediction
        # recording / AR-stage stashing hooks the mixin provides must be replicated here.
        # At deploy time (eval rollout, not training) grow the ``ar_memory`` buffer with each
        # single-step prediction. Opt-in (``_ar_memory_records_deploy_predictions``) and gated
        # off during training so the OFF-path / AR-family behaviour stays unchanged.
        if (
            self._ar_memory_records_deploy_predictions
            and self.ar_enabled
            and not self.training
        ):
            self._append_ar_memory(ss_mean)

        if self.ar_enabled and self.ar_train_horizon_unroll:
            # Stash the predictions for the compounded-prediction free-running splice.
            # When NOT training (eval/inference deploy) the stash must be detached so an
            # eval-mode graph (e.g. a cuDNN RNN forward, which does not retain the reserve
            # space required for backward) cannot leak into a subsequent training
            # ``loss().backward()`` (surfaces on GPU as "cudnn RNN backward can only be
            # called in training mode"). Mirrors the mixin ``deploy`` guard.
            if self._DETACH_FORWARD_PRED or not self.training:
                self._last_pred_mean = ss_mean.detach()
                if ss_logvar is not None:
                    self._last_pred_logvar = ss_logvar.detach()
            else:
                self._last_pred_mean = ss_mean
                if ss_logvar is not None:
                    self._last_pred_logvar = ss_logvar
        return ss_mean, ss_logvar

    # @torch.compiler.disable
    def _default_deploy_head(
        self,
        x: Tensor,
        only_elite: bool = False,
        deploy_head_mode_override: Optional[str] = None,
        **_kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Provides a default implementation of the deploy head for processing tensors
        and producing mean and log-variance predictions, with optional configuration
        overrides and elite filtering.

        :param x: Input tensor containing the data to be processed.
        :param only_elite: Boolean flag indicating whether to process only elite
            components. Defaults to False.
        :param deploy_head_mode_override: Optional string to override the deployment
            head mode. If not provided, the default mode is used.
        :param _kwargs: Additional keyword arguments for internal use.
        :return: A tuple containing the single-step mean tensor and an optional log-variance tensor.
        """
        x = self._maybe_cast_to_model_dtype(x)

        V1 = True  # V1 is the fast version
        # (CRITICAL) ToDo: assess dropping V2 from the deploy head method (ref task RLRP-680)
        if V1:
            # .... V1 (fast) ......................................................................
            _deploy_state_history = StateHistory(h_mean=x)
            u_ms_mean, u_ms_logvar = self._default_forward_u(
                _deploy_state_history, only_elite
            )

            # RLRP-735: the fast deploy path builds ``StateHistory`` inline (it does NOT go
            # through ``setup_state_history``), so the frozen ``s_τ^H`` token for the
            # index_and_history_dependent mixer must be captured here too — otherwise the mixer
            # call below asserts on a missing token. Flag-gated -> no-op for the other kinds.
            self._maybe_capture_forecast_history_token(x, _deploy_state_history)

            if (self.unroll_len > 1 and u_ms_mean.dim() == 3) or u_ms_mean.dim() == 4:
                # Training casse (E x B x MS x O+A) or deploy casse (E x MS x O+A)
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
            # RLRP-736 §4A-B P3.2 (RLRP-738): canonicalise the exposed deploy attitude
            # mean onto S^3 (chordal-L2 + sign) before it is consumed downstream / fed
            # back through the CP unroll ``q -> R`` input encode (no-op OFF).
            ss_mean = self._canonicalize_exposed_mean_quaternion(ss_mean)

            # RLRP-736 Item 1 (SS ≡ CP(horizon=1); §2.2A): sign-align the exposed
            # deploy attitude mean to the LAST obs frame of the incoming history
            # ``x`` (``⟨q_pred, q_ref⟩ ≥ 0`` + L2-projection to S^3), applied STRICTLY
            # LAST so it overrides the memoryless ``w >= 0`` step above. This makes a
            # standalone SS forecast equal the horizon-1 CP forecast AND covers the
            # test-time rollout deployer (``sample_1d -> _forward_propagation ->
            # self.deploy(x)``) with NO change to ``OneDTransitionRewardModelV2``.
            # NOTE: this ``self.deploy(x)`` route is taken only for
            # ``propagation_method in {expectation, post_sampling_expectation, None}``
            # (the MTM-Pro default is ``expectation``); a ``forward``-based mode is a
            # documented out-of-scope coverage gap (§2.2A.1). No-op OFF / legacy.
            ss_mean = self._apply_deploy_attitude_continuity(ss_mean, x)

            # RLRP-530 (validated, retained): these clamps act on the MIXTURE variance, NOT on a
            # raw network logvar, so they are distinct from (and not redundant with) the per-
            # component `LogvarBoundLayer` already applied in `_default_forward_u`. The mixture
            # variance = within-component variance + between-component spread
            # `Sum_k alpha_k (mu_k - mu_bar)^2`; that spread term is NOT bounded by the per-
            # component bound and can exceed `exp(logvar_max)`, so `_MIX_LOGVAR_SAFE_MAX` is a real
            # upper guard. The lower `variance_floor` is mandatory numerical safety for the
            # `torch.log(...)` (avoids `log(0) -> -inf/NaN`, e.g. the i=0 "mixture of one" / very
            # negative learned `logvar_min`).
            ss_logvar = torch.log(
                ss_dist_mixture.variance.clamp(min=variance_floor(self.model_dtype))
            )
            ss_logvar = ss_logvar.clamp(max=self._MIX_LOGVAR_SAFE_MAX)
            # ---------------------------------------------------------------- RLRP-680 ---(end)---

            # Extract obs-only dimensions to match eval_score/multistep_to_singlestep_next_obs_adapter
            ss_mean = ss_mean[..., : self.singlestep_obs_len]
            ss_logvar = ss_logvar[..., : self.singlestep_obs_len]

            # Stage B (RLRP-711+): deploy uses the compressed summary head ONLY when explicitly
            # requested via `deploy_head_mode == "projection"` AND the selected head exists. This
            # makes "SIWAE/IPROJ as a deploy compressor" a deliberate, ablatable switch and keeps
            # legacy / regularizer-only configs (and checkpoints without the head) on the safe,
            # always-available moment-matched output.
            return self._maybe_apply_deploy_projection_head(
                ss_mean,
                ss_logvar,
                only_elite=only_elite,
                deploy_head_mode_override=deploy_head_mode_override,
            )

        else:
            # .... V2 (slow) ......................................................................
            self._maybe_toggle_layers_use_only_elite(only_elite)
            ms_pred_mean, ms_pred_logvar = self._default_forward(x)
            self._maybe_toggle_layers_use_only_elite(only_elite)

            ss_mean = self.multistep_to_singlestep_next_obs_adapter(ms_pred_mean)
            ss_logvar = self.multistep_to_singlestep_next_obs_adapter(ms_pred_logvar)
            return ss_mean, ss_logvar

    def loss(
        self,
        model_in: Tensor,
        target: Optional[Tensor] = None,
        target_raw_HOxDoa: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """MTM-Pro-family ``loss`` override threading the RLRP-731 raw obs target (batch ``B4-raw``).

        It adds the optional ``target_raw_HOxDoa`` kwarg so :meth:`OneDTransitionRewardModelV2.loss`
        can thread the obs-block RAW target for ``history_drift_target_mode='raw_passthrough'``
        (Option 1). CRUCIAL: it does NOT bypass the inherited ``loss`` chain — it STASHES the raw
        target on ``self`` and delegates to ``super().loss(model_in, target)`` so the
        ``CompoundedPredictionMultiStepIterator`` delegation AND the
        ``TrainTimeDomainRandomizationMultiStepMLP`` input-randomization top layer keep running
        exactly as before. :meth:`_probabilistic_loss` then reads the stash (DH is a
        probabilistic-path term). ``target_raw_HOxDoa is None`` (every ``inline_denorm`` / OFF /
        non-passthrough caller) reproduces the inherited behaviour byte-for-byte.
        """
        _prev_pending = getattr(self, "_dh_pending_target_raw_HOxDoa", None)
        self._dh_pending_target_raw_HOxDoa = target_raw_HOxDoa
        try:
            return super().loss(model_in, target)
        finally:
            self._dh_pending_target_raw_HOxDoa = _prev_pending

    @deprecated(
        reason=(
            "Model.update is deprecated and will be removed in a future version. "
            "Please use `training_step` or `pytorch_lightning.Trainer` instead."
        )
    )
    def update(
        self,
        model_in: Tensor,
        optimizer: torch.optim.Optimizer,
        target: Optional[Tensor] = None,
        target_raw_HOxDoa: Optional[Tensor] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """MTM-Pro-family ``update`` override threading the RLRP-731 raw obs target (batch ``B4-raw``).

        Replicates the mbrl :meth:`Model.update` body (``zero_grad`` -> ``loss`` -> ``backward``
        -> ``step``) because that base method calls ``self.loss(model_in, target)`` POSITIONALLY
        and therefore cannot forward the extra ``target_raw_HOxDoa`` kwarg. The call to
        ``self.loss`` routes through the stash + full inherited chain (domain randomization
        included). ``target_raw_HOxDoa is None`` reproduces the inherited numerics.

        RLRP-788 (Option B): the inherited A1 ``grad_norm`` diagnostic block was DROPPED from
        this override. ``update`` is ``@deprecated`` and never on the MTM-Pro live training path
        (which runs Lightning ``fit`` -> ``training_step``, not ``update``); the single consumer
        of ``meta["grad_norm"]`` in the repo is the PlaNet algorithm loop, which trains through
        the base :meth:`Model.update` (kept intact, A1 preserved) -- so the MTM-Pro override
        never needed the reduction. The live MTM-Pro TensorBoard gradient-norm signal is
        produced solely by ``setup_gradient_monitoring_callback``. See the RLRC MTM-Pro models
        code optimization `.junie` plan
        (``perf_RLRP-783_mtm_pro_models_code_optimization_plan_20260827.md``) and the RLRC
        meta-collection kill-switch `.junie` plan
        (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).

        RLRP-788 (follow-up): the A3 ``flush_non_finite_reports`` reporting hook was MOVED OUT of
        this ``@deprecated`` override into :meth:`training_step`. Because the live MTM-Pro path is
        Lightning ``fit`` -> ``training_step`` (never ``update``), the per-step NaN/inf flush now
        lives where it actually fires in production; ``update`` no longer flushes.
        """
        self.train()
        optimizer.zero_grad()
        loss, meta = self.loss(
            model_in, target=target, target_raw_HOxDoa=target_raw_HOxDoa
        )
        loss.backward()
        optimizer.step()
        return loss.item(), meta

    def training_step(self, batch: TransitionBatch, batch_idx: int):
        # A3 (RLRP-783): materialise the accumulated on-device non-finite flags ONCE per training
        # step. This is the reporting-interval hook that keeps the production NaN/inf warning firing
        # (V4) after the per-call synchronisation was removed from ``weigthing._check_finite``; it
        # is the third and last permitted device->host sync of the step (with ``loss.item()`` and
        # the stacked ``meta`` flush). RLRP-788 follow-up: this hook was MOVED here from the
        # ``@deprecated`` ``update`` because the live MTM-Pro training path is Lightning
        # ``fit`` -> ``training_step`` (never ``update``); flushing BEFORE delegating to the base
        # :meth:`Model.training_step` keeps it once-per-step on the path that actually runs.
        # RLRP-786 (``performance_mode``, FR8): development-only instrumentation -- under ``fast``
        # the guard kernels are never issued (``weigthing.set_non_finite_guard_enabled(False)``), so
        # there is nothing to flush and the per-step host sync is skipped.
        if self._performance_mode == "dev":
            flush_non_finite_reports()
        return super().training_step(batch, batch_idx)

    # ==== RLRP-786: ``performance_mode`` + CUDA-graph capture readiness =========================
    #: ``dev`` (default, today's behaviour) | ``fast`` (paper-run mode: dev/debug-only
    #: instrumentation OFF; precondition of the CUDA-graph captured step). Driven by
    #: ``pipeline.performance_mode`` through :meth:`set_performance_mode` (``setup.py``). FR8 /
    #: Key Decision 8 of ``perf_RLRP-786_cuda_graph_train_step_plan_20260919.md``.
    #: ``_performance_mode`` / ``PERFORMANCE_MODES`` / ``set_performance_mode`` are inherited from
    #: ``PerformanceModeMixin`` (through ``CompoundedPredictionMultiStepIterator``); this class
    #: only contributes its ``fast``-mode refusal (``_refuse_fast_mode_reasons``).
    #: Only the SAMPLING-FREE variant records soundly (Key Decision 6): the particle / re-sampled
    #: particle variants draw device samples in the forecast, and the philox offset stream of a
    #: replayed graph differs from eager -> not bit-identical by construction. Overridden to
    #: ``True`` by ``MS2MS2SSArTemporalMixtureSamplingFreePME``.
    _CUDA_GRAPH_CAPTURABLE_VARIANT: bool = False

    def _refuse_fast_mode_reasons(self) -> List[str]:
        """``fast`` refuses the class-level ``_GLOBAL_DEBUG`` sentinel path (it synchronises
        inside the loss); use ``dev`` for debugging. See ``PerformanceModeMixin``."""
        reasons = list(super()._refuse_fast_mode_reasons())
        if self._GLOBAL_DEBUG:
            reasons.append(
                "the class-level _GLOBAL_DEBUG sentinel path (it synchronises inside the loss); "
                "use 'dev' for debugging"
            )
        return reasons

    def _forecast_tf_coins_consumed(self) -> bool:
        """``True`` when the eager forecast loop would draw a teacher-forcing coin per horizon step."""
        return (
            self.training
            and self._forecast_tf_target is not None
            and self._forecast_teacher_forcing_scheduler is not None
        )

    def refill_forecast_tf_mask(self, assume_target: bool = False) -> Tensor:
        """Draw the ``horizon_len`` forecast teacher-forcing coins on the CPU and copy them into the
        static ``(F,)`` bool device mask (RLRP-786 FR6).

        Same consumer, same order as the eager Python branch of ``state_history_update`` (one
        ``should_use_teacher_forcing()`` call per horizon step ``i = 0..F-1``, at the frozen
        probability of the current step), so the CPU RNG stream is identical. When the eager
        branch would not draw (eval / no GT target / no scheduler) the mask is all-``False``
        and no coin is consumed. Called once per training step: from ``_probabilistic_loss``
        (eager) or from the CUDA-graph ``before_replay`` hook (captured path).

        :param assume_target: captured path only -- the GT target reference is dropped at the
            loss tail (``self._forecast_tf_target = None``) while the recorded graph keeps
            reading the static target memory, so the hook asserts "a target is present" itself.
        """
        F = self.horizon_len
        if self._forecast_tf_mask is None or self._forecast_tf_mask.shape[0] != F:
            self._forecast_tf_mask = torch.zeros(F, dtype=torch.bool, device=self.device)
        draw = (
            self.training and self._forecast_teacher_forcing_scheduler is not None
            if assume_target
            else self._forecast_tf_coins_consumed()
        )
        if draw:
            coins = [
                bool(self._forecast_teacher_forcing_scheduler.should_use_teacher_forcing())
                for _ in range(F)
            ]
        else:
            coins = [False] * F
        self._forecast_tf_mask.copy_(torch.tensor(coins, dtype=torch.bool))
        return self._forecast_tf_mask

    def advance_host_step_state(self) -> None:
        """Replay the HOST-side per-step bookkeeping of one ``_probabilistic_loss`` call (RLRP-786).

        On the CUDA-graph captured path the Python body of the loss does not run, so the
        CPU-side state it advances once per training step must be advanced by the graph
        ``before_replay`` hook instead: the training-step counter, the forecast teacher-forcing
        scheduler (stepped AFTER the coins are drawn, as in the eager loss) and the CP
        schedulers. Order = eager order: counter -> coins (:meth:`refill_forecast_tf_mask`) ->
        forecast scheduler ``step`` -> CP schedulers ``step``.
        """
        self._train_step_count += 1
        self.refill_forecast_tf_mask(assume_target=True)
        if self._forecast_teacher_forcing_scheduler is not None:
            self._forecast_teacher_forcing_scheduler.step()
        if self.enable_compounded_prediction_deploy_loss:
            self._step_ar_schedulers()

    def cuda_graph_capture_blockers(self) -> List[str]:
        """Reasons why recording this model's training step in a CUDA graph would be UNSOUND.

        Empty list = capturable. A graph replays the kernels recorded once, so every piece of
        HOST-side state that changes the computation from one step to the next must either be
        frozen (schedulers that are off / complete) or fed through a device buffer (the
        teacher-forcing mask). Anything else here is reported and the caller falls back to eager
        with a one-line notice (RLRP-786 FR5).
        """
        blockers: List[str] = []
        if not self._CUDA_GRAPH_CAPTURABLE_VARIANT:
            blockers.append(
                f"{type(self).__name__} is not the sampling-free variant (device RNG in the forecast)"
            )
        if not self.cuda_graph_capture_ready:
            blockers.append("ms_model.cuda_graph_capture_ready is False (TF splice is a Python branch)")
        if self._performance_mode != "fast":
            blockers.append("pipeline.performance_mode is not 'fast' (dev instrumentation syncs)")
        if self._GLOBAL_DEBUG:
            blockers.append("_GLOBAL_DEBUG sentinel path (syncs inside the loss)")
        if self._enable_meta_collection:
            blockers.append("enable_meta_collection is True (`_flush_meta_scalars` host sync inside the loss)")
        if self._deploy_warmup_enable:
            blockers.append("deploy_path_training.warmup is enabled (per-step Python gate on the loss terms)")
        if self._ema_forecast_enable:
            blockers.append("deploy_path_training.ema_forecast_clone is enabled (host-side weight swap per step)")
        if self.ms_probabilistic_loss_mode != "true_mixture":
            blockers.append(f"ms_probabilistic_loss_mode={self.ms_probabilistic_loss_mode!r} is not the paper path")
        if getattr(self, "_unroll_len_sampler", None) is not None:
            blockers.append("CP unroll-length curriculum is active (host-sampled unroll length)")
        if getattr(self, "_temporal_weight_scheduler", None) is not None:
            blockers.append("CP temporal-weight ramp is active (gamma changes per step)")
        _cp_tf = getattr(self, "_teacher_forcing_scheduler", None)
        if _cp_tf is not None and _cp_tf.method not in ("always_off", "always_on"):
            blockers.append("CP unroll teacher forcing is a scheduled Python coin (only always_off / always_on capture)")
        return blockers

    @torch.compiler.disable
    def _deterministic_loss(
        self, model_in: Tensor, target: Tensor, reduce=True
    ) -> Tuple[Tensor, Dict[str, Any]]:
        raise NotImplementedError("This a probabilistic only model")

    def _pre_mixture_loss(
        self,
        ms_mean: Tensor,
        ms_logvar: Tensor,
        i: int,
        ms_target: Tensor,
    ) -> None:
        if self.unroll_len != 1:
            ms_mean_mask, _ = self._pre_mixture_u_mask(i, ms_mean)
            ms_logvar_mask, _ = self._pre_mixture_u_mask(i, ms_logvar)
        else:
            # Quick-hack to setup the array shape of SS observations the same way as MS ones
            ms_mean_mask = ms_mean.unsqueeze(-2)
            ms_logvar_mask = ms_logvar.unsqueeze(-2)
            ms_mean_mask = revert_timestep_first_multistep_dim_unflaten_array(
                ms_mean_mask,
                singlestep_obs_len=self.singlestep_obs_len,
                singlestep_act_len=self.singlestep_act_len,
                remove_last_action_padding=True,
            )
            ms_logvar_mask = revert_timestep_first_multistep_dim_unflaten_array(
                ms_logvar_mask,
                singlestep_obs_len=self.singlestep_obs_len,
                singlestep_act_len=self.singlestep_act_len,
                remove_last_action_padding=True,
            )

        u_target, u_target_ms_len = self._pre_mixture_u_mask(
            i, ms_target, is_target=True
        )
        u_ms_distribution = self._to_distribution(ms_mean_mask, ms_logvar_mask)
        u_loss = -u_ms_distribution.log_prob(u_target)

        if self.unroll_len == 1 or (
            u_target.dim() != ms_target.dim() and self.horizon_len - i == 1
        ):
            # self._pre_mixture_loss_accumulator[..., : self.singlestep_obs_len] = self._pre_mixture_loss_accumulator[..., : self.singlestep_obs_len] + u_loss
            self._pre_mixture_loss_accumulator[..., : self.singlestep_obs_len] += u_loss
        else:
            u_loss = timestep_first_multistep_dim_unflaten_array(
                u_loss,
                self.singlestep_obs_len,
                self.singlestep_act_len,
                sequence_len=u_target_ms_len,
                enable_last_action_padding=True,
            )

            u_loss = self.reduce_multistep_losses_horizon(
                u_loss.swapaxes(-2, -1),
                probabilistic_losses=True,
                unflaten_composed_array_enabled=False,
            )

            # self._pre_mixture_loss_accumulator = self._pre_mixture_loss_accumulator + u_loss
            self._pre_mixture_loss_accumulator += u_loss
        return None

    def _pre_mixture_u_mask(
        self, i: int, ms: Tensor, is_target: bool = False
    ) -> Union[Tensor, int]:

        if not is_target:
            ms_unflatten = timestep_first_multistep_dim_unflaten_array(
                ms,
                self.singlestep_obs_len,
                self.singlestep_act_len,
                sequence_len=self.unroll_len,
                enable_last_action_padding=True,
            )
        else:
            assert (
                ms.dim() == 4
            ), f"Expected 4D tensor (E x B x MS x Doa), got {ms.dim()}D"
            ms_unflatten = ms

        # E, B, MS, Doa = ms_unflatten.shape

        if not is_target:
            ms_mask = ms_unflatten[
                ..., 0 : min(self.unroll_len, self.horizon_len - i), :
            ]
        else:
            ms_mask = ms_unflatten[
                ..., i : min(self.unroll_len + i, self.horizon_len), :
            ]

        output_ms_len = ms_mask.shape[-2]
        ms_flatten = revert_timestep_first_multistep_dim_unflaten_array(
            ms_mask,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            remove_last_action_padding=True,
        )
        return ms_flatten, output_ms_len

    @torch.compiler.disable
    def _true_mixture_per_feature_nll(
        self,
        joint_mixture: MixtureSameFamily,
        target_i: Tensor,
    ) -> Tensor:
        """Per-feature NLL of a temporal mixture -> (E x B x D) where D is the mixture event
        width (O+A for the `true_mixture` primary path's full obs+action mixture, O for an
        obs-only mixture). NOTE: D is the per-feature event width, NOT the forecast/horizon
        length F.

        The JOINT mixture built by `_temporal_mixture_at_i` (subclass-aware: handles the
        index, input-dependent and transformer mixers + `detach_components`) scores the whole
        feature vector at once, so its `log_prob` collapses the event to an (E x B) scalar. To
        let the `true_mixture` primary loss reuse the SAME logvar-bound / obs-feature-weight /
        temporal-discount / horizon-reduction pipeline as the legacy `moment_matched` path, we
        re-express it as the per-feature MARGINAL temporal mixtures: each feature dim becomes its
        own 1-D mixture over the K components, sharing the per-step categorical mixing weights.
        This preserves (per-feature) multi-modality while exposing the feature axis. (The
        true_mixture path passes the full obs+action mixture/target so the action-prediction
        forecast signal is retained, ref RLRP-530.)

        We reuse the already-built joint mixture's parameters (mixing logits + component
        loc/scale) rather than rebuilding the mixer, so all subclass mixer overrides and the
        gradient-scope `detach_components` choice are honoured automatically.

        EXPOSITION, NOT AN EXTRA ASSUMPTION (RLRP-704). The per-feature decomposition is an
        *exposition* of the joint mixture used solely to expose the feature axis for the weighting
        pipeline; it is NOT an additional independence assumption. In general the joint mixture NLL
        is NOT the sum of per-feature marginal NLLs. It is faithful here ONLY because of the
        specific structure actually used by this model:
          (i) each mixture COMPONENT is an `Independent(diagonal)` base (Normal/Laplace), i.e.
              the features are conditionally independent GIVEN the component index k, and
          (ii) the per-step categorical mixing weights are SHARED across all feature dims D
               (we broadcast the same `mix_logits` over D below).
        Under (i)+(ii) the per-feature marginals carry the full information of the joint EXCEPT the
        cross-feature coupling that the mixture induces THROUGH the shared categorical; that
        coupling is preserved per-feature (same weights) and the downstream obs-feature-weighted
        reduction recovers the intended objective. If either property changed (full-covariance
        components, or per-feature mixing weights) this decomposition would no longer be exact.
        """
        mix_logits, comp_loc, comp_scale, carried_tangent = (
            self._mixture_scoring_params(joint_mixture)
        )
        return self._per_feature_mixture_nll_from_params(
            mix_logits, comp_loc, comp_scale, carried_tangent, target_i
        )

    @staticmethod
    def _mixture_scoring_params(
        joint_mixture: MixtureSameFamily,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Dict[int, Tensor]]]:
        """Extract the tensors the per-feature true-mixture NLL is a pure function of.

        RLRP-830 Item C seam: ``(mix_logits (… x K), comp_loc (… x K x D), comp_scale
        (… x K x D), carried so3-tangent log-variance by slot or None)``. Shared by the
        single-mixture :meth:`_true_mixture_per_feature_nll` and the stacked CP scoring
        :meth:`_stacked_true_mixture_per_feature_nll`, so both paths run the SAME ops.
        """
        # Mixing logits over the K components -> (E x B x K).
        mix_logits = joint_mixture.mixture_distribution.logits
        # Component loc/scale per feature -> (E x B x K x D). The component is an Independent
        # wrapper around the base Normal/Laplace, so go through `.base_dist`.
        base_component = joint_mixture.component_distribution.base_dist
        comp_loc = base_component.mean  # (E x B x K x D)
        comp_scale = base_component.scale  # (E x B x K x D) (stddev / Laplace scale)
        # RLRP-736 §4A-B P3.1 (RLRP-738): manifold-native mixtures CARRY the 3-D so3-tangent
        # log-variance produced at component construction (consumed by the attitude NLL).
        carried_tangent = getattr(
            joint_mixture, "_attitude_tangent_log_var_by_slot", None
        )
        return mix_logits, comp_loc, comp_scale, carried_tangent

    def _per_feature_mixture_nll_from_params(
        self,
        mix_logits: Tensor,
        comp_loc: Tensor,
        comp_scale: Tensor,
        carried_tangent: Optional[Dict[int, Tensor]],
        target_i: Tensor,
    ) -> Tensor:
        """Per-feature true-mixture NLL from the mixture parameters -> ``(… x D)``.

        Body of :meth:`_true_mixture_per_feature_nll` (see its docstring for the maths);
        every op indexes from the RIGHT so an arbitrary leading batch shape is admitted
        (``(E x B)`` for one mixture, ``(U x E x B)`` for the RLRP-830 stacked CP scoring).
        """
        # Move the mixture-component axis K last: (E x B x K x D) -> (E x B x D x K) so each
        # feature becomes its own 1-D mixture over the K components.
        comp_loc_t = comp_loc.transpose(-2, -1)
        comp_scale_t = comp_scale.transpose(-2, -1)
        component_distribution = super()._to_distribution(
            comp_loc_t, comp_scale_t, None, False, False
        )  # batch (E x B x D x K), event () ; scale_are_log_variance=False (scale is std)

        # Broadcast the per-step mixing logits across the feature axis D -> (E x B x D x K).
        mix_logits_d = mix_logits.unsqueeze(-2).expand(comp_loc_t.shape)

        per_feature_mixture = dist.MixtureSameFamily(
            dist.Categorical(logits=mix_logits_d), component_distribution
        )  # batch (E x B x D)

        per_feature_nll = -per_feature_mixture.log_prob(target_i)  # (E x B x D)

        # RLRP-736 §4A-B P1.2 (option 1, structural): when the by-construction
        # PROBABILISTIC attitude head is active, RE-SCORE the attitude slot(s) with
        # the per-particle so3-tangent mixture NLL instead of the per-feature
        # quaternion-space Gaussian marginals (no-op / bit-exact OFF).
        return self._apply_mixture_attitude_tangent_nll_from_params(
            mix_logits, comp_loc, comp_scale, carried_tangent, target_i, per_feature_nll
        )

    def _stacked_true_mixture_per_feature_nll(
        self, mixtures: Sequence[MixtureSameFamily], targets: Sequence[Tensor]
    ) -> Tensor:
        """Per-feature true-mixture NLL of ``U`` same-shape mixtures in ONE pass -> ``(U x … x D)``.

        RLRP-830 Item C (Step 2). The CP unroll is sequential (each deploy feeds the next
        window) but SCORING is not: the ``U`` K=1 deploy mixtures share their shapes, so their
        parameters are stacked on a new leading axis and scored with the same ops as
        :meth:`_true_mixture_per_feature_nll` (element-wise / last-axis reductions ->
        bit-exact with the per-step path, ``nll[t] == _true_mixture_per_feature_nll(m_t, y_t)``).
        """
        params = [self._mixture_scoring_params(m) for m in mixtures]
        mix_logits = torch.stack([p[0] for p in params], dim=0)
        comp_loc = torch.stack([p[1] for p in params], dim=0)
        comp_scale = torch.stack([p[2] for p in params], dim=0)
        carried: Optional[Dict[int, Tensor]] = None
        if params[0][3] is not None:
            slots = tuple(params[0][3].keys())
            assert all(
                p[3] is not None and tuple(p[3].keys()) == slots for p in params
            ), "RLRP-830: every stacked CP mixture must carry the same tangent slots"
            carried = {
                s: torch.stack([p[3][s] for p in params], dim=0) for s in slots
            }
        target = torch.stack(list(targets), dim=0)
        # Guard against SILENT broadcasting (a rank mismatch between the mixture batch
        # shape and the target would still "work" and inflate the NLL tensor).
        assert target.shape[:-1] == mix_logits.shape[:-1], (
            "RLRP-830 stacked CP scoring: target batch shape "
            f"{tuple(target.shape[:-1])} != mixture batch shape "
            f"{tuple(mix_logits.shape[:-1])}"
        )
        return self._per_feature_mixture_nll_from_params(
            mix_logits, comp_loc, comp_scale, carried, target
        )

    def _mixture_orientation_nll_active(self) -> bool:
        """Whether the by-construction PROBABILISTIC attitude MIXTURE-NLL split is live.

        RLRP-736 §4A-B P1.2 (base Particle variant, ensemble ``num_members == 1``).
        Mirrors the base ``_orientation_nll_active`` guard: requires the per-family
        opt-in flag, the resolved ``_manifold_aware_nll`` lever (RLRP-744 §A.3
        step 3 — replaces the former ``_internal_orientation_rep_active`` gate so the plain
        ``quaternion`` rep can opt IN and the matrix reps can opt OUT; ``null``
        reproduces the pre-lever behaviour), tracked single-step attitude
        OUTPUT slot(s), and a probabilistic (non-deterministic) model. Bit-exact OFF
        (legacy per-feature Gaussian mixture marginal NLL) otherwise.
        """
        return bool(
            getattr(self, "_supports_probabilistic_by_construction_orientation", False)
            and getattr(self, "_manifold_aware_nll", False)
            and getattr(self, "_ss_ori_out_slots", ())
            and not getattr(self, "_deterministic_rep", True)
        )

    def _apply_mixture_attitude_tangent_nll(
        self,
        joint_mixture: MixtureSameFamily,
        target_i: Tensor,
        per_feature_nll: Tensor,
    ) -> Tensor:
        """Re-score the attitude slot(s) with a per-particle so3-tangent MIXTURE NLL.

        RLRP-736 §4A-B P1.2 (option 1, structural increment; base Particle variant,
        ensemble ``num_members == 1``). This is the probabilistic analogue of the base
        ``ExponentialFamilyMLP._orientation_split_nll`` for the ★ MTM-Pro TEMPORAL
        MIXTURE path: the objective scores via a ``MixtureSameFamily`` over ``K``
        temporal-particle components, NOT a plain Gaussian split, so the tangent NLL is
        integrated at the per-particle mixture-COMPONENT level (Geist et al. 2024;
        Forster et al. 2016 for the right-Jacobian sub-mode).

        For every single-step attitude output slot ``s`` (``self._ss_ori_out_slots``,
        4-D EXTERNAL quaternion, decoded to a unit quaternion by construction upstream
        in ``_default_forward_u``), the component quaternion means ``μ_k`` (4-D) + the
        per-step target quaternion + a 3-D so3-tangent log-variance derived from the
        component scale are scored by :func:`rotation_tangent_nll` per component ``k``,
        then combined across the ``K`` components with the SHARED per-step categorical
        mixing weights ``w_k`` as a proper mixture NLL

            ``-log Σ_k w_k · p_tangent,k = -logsumexp_k( log w_k − nll_tangent,k )`` .

        The result replaces the leading attitude column of ``per_feature_nll`` (the
        remaining 3 attitude columns are zeroed, so the attitude slot is scored ONCE and
        the downstream obs-feature-weighted / temporal-discount / horizon reduction is
        applied through that leading column); all non-attitude columns keep the exact
        legacy per-feature Gaussian mixture marginal NLL. No-op / bit-exact when the
        by-construction wiring is inactive.

        DEDICATED 3-D so3-tangent logvar HEAD (RLRP-736 §4A-B / RLRP-738, structural
        increment): when :meth:`_build_attitude_tangent_logvar_head` has instantiated
        the per-slot learnable ``nn.Linear(4, 3)`` head, the 3-D tangent log-variance is
        produced by that head from the component's 4-D quaternion-space log-variance
        ``2 log σ`` — a genuine, physically-narrowed head trained through this NLL. The
        head is INITIALISED to select the quaternion vector-part ``(x, y, z)`` so it
        STARTS EXACTLY at the previous placeholder (``σ_tan² = scale[..., 1:4]²``) and
        only departs as the objective learns; if the head is absent the placeholder
        branch is used verbatim. This is INTENTIONALLY bounded: the symmetric 4-D
        temporal-mixture accumulator (``forecast_{mean,logvar}_accumulator``) and
        ``_temporal_mixture_distribution_at_i`` are left intact (so every NON-attitude
        Gaussian marginal stays bit-exact); a full 3-D-native accumulator / joint-
        mixture rework and the NUMERIC correctness gate remain compute-gated (RLRP-736
        P2.4 / RLRP-738).
        """
        if not self._mixture_orientation_nll_active():
            # Bit-exact OFF / no-op contract: return the input untouched without
            # inspecting the mixture (callers may pass ``None`` when inactive).
            return per_feature_nll
        mix_logits, comp_loc, comp_scale, carried_tangent = (
            self._mixture_scoring_params(joint_mixture)
        )
        return self._apply_mixture_attitude_tangent_nll_from_params(
            mix_logits, comp_loc, comp_scale, carried_tangent, target_i, per_feature_nll
        )

    def _apply_mixture_attitude_tangent_nll_from_params(
        self,
        mix_logits: Tensor,
        comp_loc: Tensor,
        comp_scale: Tensor,
        carried_tangent: Optional[Dict[int, Tensor]],
        target_i: Tensor,
        per_feature_nll: Tensor,
    ) -> Tensor:
        """Parameter-based body of :meth:`_apply_mixture_attitude_tangent_nll`.

        RLRP-830 Item C seam: takes the mixture tensors (see :meth:`_mixture_scoring_params`)
        instead of the ``MixtureSameFamily`` so the stacked CP scoring can reuse it with an
        extra leading unroll axis; every op indexes from the right.
        """
        if not self._mixture_orientation_nll_active():
            return per_feature_nll
        if is_s2_orientation_rep(self._orientation_rep):
            # RLRP-796 ``D-793-4`` scope boundary, widened to EVERY `S^2`
            # representation (RLRP-799). This TRUE-MIXTURE per-feature path is
            # quaternion-specific by construction: the ``s + 4`` slicing, the
            # dedicated learnable ``4 -> 3`` so3 logvar head, its vector-part
            # ``scale[..., 1:4]`` placeholder initialisation and the manifold-native
            # carried tangent all assume the 4-D quaternion component. ``s2_tangent``
            # would crash on the shape mismatch, but ``s2_identity`` was previously
            # ADMITTED here and SILENTLY MIS-SCORED: on the standard 9-D gravity block
            # the orientation slot is ``s = 3``, so ``s + 4 = 7 <= 9`` and the
            # defensive width ``continue`` below cannot fire -- the slice reads
            # ``gravity.{x,y,z}`` PLUS ``angular_vels.x``, which both destroys that
            # feature's own Gaussian marginal and lets it move the orientation score
            # (probe-confirmed: NLL delta 4.30 for a 0.0 -> 0.9 perturbation).
            # TEMPORARY guard: fail loud rather than mis-score, until RLRP-799 gives
            # this path a real 3 -> 2 `S^2` head family.
            raise NotImplementedError(
                "The TRUE-MIXTURE per-feature orientation NLL is quaternion-block "
                "only (a dedicated 4 -> 3 so3 logvar head + quaternion vector-part "
                "placeholder), so it CANNOT score an S^2 direction slot: "
                f"representation='{self._orientation_rep.value}' is incompatible "
                "with ms_probabilistic_loss_mode='true_mixture'. "
                "WORKAROUND: set ms_probabilistic_loss_mode='moment_matched' for any "
                "S^2 run (the split-NLL seam is width-correct for both S^2 reps), or "
                "use internal_orientation.representation='quaternion' on a quaternion "
                "observation space to keep the true-mixture mode. "
                "This is a TEMPORARY guard: real S^2 support on the true-mixture path "
                "is tracked by YouTrack RLRP-799 (item 2) -- that task must be "
                "completed before this combination can be used "
                "(RLRP-796 'D-793-4')."
            )
        # Per-step categorical mixing weights over the K components -> (E x B x K).
        log_w = torch.log_softmax(mix_logits, dim=-1)
        # comp_loc: (E x B x K x D) decoded 4-D quaternion means; comp_scale: component std.
        eps = variance_floor(self.model_dtype)

        # Keep the non-attitude columns verbatim; zero the attitude columns so the
        # per-slot tangent NLL can be spliced back onto the leading attitude column.
        keep_mask = per_feature_nll.new_ones(per_feature_nll.shape[-1])
        add_term = per_feature_nll.new_zeros(per_feature_nll.shape)
        # RLRP-736 §4A-B / RLRP-738: prefer the DEDICATED learnable 3-D so3-tangent
        # logvar HEAD (``4 -> 3`` per slot) over the fixed vector-part placeholder.
        head = getattr(self, "_attitude_tangent_logvar_head", None)
        head_slots = getattr(self, "_attitude_tangent_logvar_head_slots", ())
        # RLRP-736 §4A-B P3.1 (RLRP-738): when the mixture was built manifold-native
        # (``_ManifoldMeanMixtureSameFamily``) the 3-D so3-tangent log-variance was
        # already PRODUCED at component-construction time and CARRIED on the mixture
        # (``carried_tangent``). Consume it directly so the head is applied ONCE
        # (component construction), not re-derived here — the component now lives in
        # the tangent chart it is scored in.
        for s in sorted(self._ss_ori_out_slots):
            if s + 4 > per_feature_nll.shape[-1]:
                # Defensive: attitude slot must fit inside the scored event width.
                continue
            keep_mask[s : s + 4] = 0.0
            q_pred_k = comp_loc[..., s : s + 4]  # (E x B x K x 4)
            scale_k = comp_scale[..., s : s + 4]  # (E x B x K x 4)
            if carried_tangent is not None and s in carried_tangent:
                # CARRIED tangent (P3.1): the component was BUILT with this 3-D so3
                # tangent log-variance; consume it verbatim (no double head application).
                tangent_log_var = carried_tangent[s]
            elif head is not None and s in head_slots:
                # DEDICATED HEAD: map the 4-D quaternion-space component log-variance
                # ``2 log σ`` to a genuine 3-D so3-tangent log-variance. Initialised to
                # select the vector-part (x, y, z), so it STARTS at the placeholder.
                comp_logvar_k = 2.0 * torch.log(
                    scale_k.clamp_min(eps)
                )  # (E x B x K x 4)
                tangent_log_var = head[head_slots.index(s)](comp_logvar_k)  # (...x3)
            else:
                # STRUCTURAL PLACEHOLDER (head inactive): 3-D so3-tangent log-variance
                # from the quaternion vector-part std.
                tangent_log_var = 2.0 * torch.log(scale_k[..., 1:4].clamp_min(eps))
            q_target = target_i[..., s : s + 4].unsqueeze(-2)  # (E x B x 1 x 4)
            q_target_k = q_target.expand_as(q_pred_k)  # (E x B x K x 4)
            # Per-component tangent NLL -> (E x B x K).
            nll_tangent_k = rotation_tangent_nll(
                q_pred_k,
                q_target_k,
                tangent_log_var,
                with_right_jacobian=getattr(self, "_tangent_nll_right_jac", False),
                reduce=False,
            )
            # Proper mixture NLL over the K particles -> (E x B).
            att_nll = -torch.logsumexp(log_w - nll_tangent_k, dim=-1)
            add_term[..., s] = att_nll
        return per_feature_nll * keep_mask + add_term

    @torch.compiler.disable
    def _accumulate_post_mix_losses(self, horizon_index: int) -> None:
        """
        Accumulate the per-horizon-step post-mixing negative log-likelihood of the *true*
        temporal mixture (Stage B 'MIX' and Stage G 'true_mixture').

        The post-mixture (MIX) term keeps the JOINT obs mixture `log_prob` (event dim = O) reduced
        to an (E x B) per-step scalar, accumulated into `_post_mixture_loss_accumulator`.

        The `true_mixture` primary term instead keeps PER-OBS-FEATURE granularity
        (`_true_mixture_per_feature_nll` -> (E x B x O+A)) for each horizon step, appended to
        `_ms_true_mixture_loss_steps`. This is what lets `_probabilistic_loss` apply the same
        logvar-bound / obs-feature-weight / temporal-discount / horizon-reduction pipeline as the
        legacy `moment_matched` path while still preserving (per-feature) multi-modality.
        """
        i = horizon_index

        # obs-only target for horizon step i  -> (E x B x O)
        target_i = self._post_mixture_loss_target[..., i, : self.singlestep_obs_len]

        # .... true-mixture primary objective (Stage G, full network) ...........................
        if self.ms_probabilistic_loss_mode == "true_mixture":
            # Build the FULL obs+action JOINT temporal mixture (subclass-aware), then re-express it
            # as per-feature marginal mixtures -> (E x B x O+A); kept per horizon step. We KEEP the
            # action dims (NOT obs-only) because the primary objective optimises the multi-step
            # FORECAST, which is obs+action at every step except the last one (ref task RLRP-530).
            # Dropping the action dims here would discard a large part of the forecast learning
            # signal. The last step's action slots are padding and are removed downstream by
            # `remove_last_action_padding=True`, exactly mirroring the legacy `moment_matched` path
            # which scores the whole composed obs+act array.
            target_i_full = self._post_mixture_loss_target[..., i, :]
            stash = getattr(self, "_current_step_mixture", None)
            if (
                getattr(self, "reuse_step_mixture", False)
                and stash is not None
                and stash[0] == i
                and stash[1] is False  # obs_space_only=False == the FULL obs+act mixture
            ):
                # RLRP-830 Item B1: ``_default_forward_mixture_dist`` built EXACTLY this
                # mixture (same accumulators, same ``i``, ``obs_space_only=False``,
                # ``detach_components=False``) a few lines earlier in the same loop
                # iteration -> re-use the object instead of rebuilding it (bit-exact:
                # same tensors; autograd accumulates the two uses).
                true_mixture = stash[2]
            else:
                true_mixture = self._temporal_mixture_at_i(
                    self.forecast_mean_accumulator,
                    self.forecast_logvar_accumulator,
                    i,
                    obs_space_only=False,
                    detach_components=False,
                )
            tm_loss_i = self._true_mixture_per_feature_nll(true_mixture, target_i_full)
            self._ms_true_mixture_loss_steps.append(tm_loss_i)

        # .... post-mixture (MIX) objective with selectable backprop scope (Stage B) ............
        if self.enable_post_mixture_loss:
            detach_components = self.post_mixture_backprop_mode == "mixing_only"
            mix = self._temporal_mixture_at_i(
                self.forecast_mean_accumulator,
                self.forecast_logvar_accumulator,
                i,
                obs_space_only=True,
                detach_components=detach_components,
            )
            mix_loss_i = -mix.log_prob(target_i)  # (E x B)
            self._post_mixture_loss_accumulator = (
                self._post_mixture_loss_accumulator + mix_loss_i
            )
        return None

    @torch.compiler.disable
    def _build_obs_only_temporal_mixtures(
        self, detach_components: bool, detach_mixing_weights: bool = False
    ) -> list:
        """Build the per-horizon-step learned obs-only temporal mixtures once.

        Shared by the IPROJ / SIW-MP / GMS-IWAE objectives so the (potentially expensive, esp. with
        the transformer mixers) per-step `MixtureSameFamily` builds happen a single time per
        `_probabilistic_loss` call even when both terms are enabled (integration fix 2d).

        `detach_components` follows `projection_backprop_mode` (`mixing_only`/`head_only` -> True):
        detaching the mixture COMPONENT tensors keeps the gradient out of the prediction body while
        the dedicated summary head `q_theta` and the mixing function still train.

        `detach_mixing_weights` follows `projection_backprop_mode == head_only`: it ALSO detaches
        the categorical mixer log-weights so ONLY `q_theta` (+ the GMS decoder) trains (distil a
        frozen forecast + mixer). Applied generically here (works for every mixer variant) instead
        of threading a flag through the subclass `_temporal_mixture_*` overrides.
        """
        mixtures = [
            self._temporal_mixture_at_i(
                self.forecast_mean_accumulator,
                self.forecast_logvar_accumulator,
                i,
                obs_space_only=True,
                detach_components=detach_components,
            )
            for i in range(self.horizon_len)
        ]
        if detach_mixing_weights:
            mixtures = [self._detach_mixture_weights(m) for m in mixtures]
        return mixtures

    @staticmethod
    def _detach_mixture_weights(mixture: MixtureSameFamily) -> MixtureSameFamily:
        """Return a copy of `mixture` whose categorical mixer log-weights are detached.

        Used by the `head_only` projection backprop scope: stops the gradient into the mixing
        function (and, for input-dependent / transformer mixers, the mixer network params) while
        leaving the component distribution untouched, so only the summary head `q_theta` trains.
        """
        detached_cat = dist.Categorical(
            logits=mixture.mixture_distribution.logits.detach()
        )
        return dist.MixtureSameFamily(detached_cat, mixture.component_distribution)

    @torch.compiler.disable
    def _info_projection_loss(
        self,
        ms_pred_mean: Tensor,
        ms_pred_logvar: Tensor,
        mixtures: Optional[Sequence[MixtureSameFamily]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # (Priority) inprogress: implement information-projection loss
        """
        Information-projection (I-projection) loss KL(q_theta || mixture) (Stage H, H.3b head).

        PME-faithful, horizon-based estimator (plan option "b"): instead of the history-indexed
        base-class `kl_divergence_dist_to_mixture` (which only supports `history_len == horizon_len`
        and uses the *static* gamma weights), we build the *learned* per-horizon-step obs-only
        temporal mixture via `_temporal_mixture_at_i` and Monte-Carlo estimate
        `KL(q_theta || mixture_i) = E_{x~q_theta}[log q_theta(x) - log mixture_i(x)]` for every
        step `i`, then average over the horizon. `q_theta` is the dedicated obs-only summary head.

        The per-step mixtures may be supplied via `mixtures` (shared with `_siw_mp_loss`); otherwise
        they are built here honouring `projection_backprop_mode`.

        Returns the (E x B x 1) KL term and the head stats `(q_mean, q_logvar)` (used by the
        optional obs-only data-NLL anchor in `_probabilistic_loss`).

        Stage A (RLRP-711+): when `per_step_conditioning` is set the head distribution is
        re-derived from the *step-i* mixture moments (a true mixture-compressor) instead of always
        the step-0 slice; when `siw_mp_temporal_weighting` is set the per-step terms are combined
        with a temporal-discount weighted average instead of a flat mean. IPROJ uses the single
        shared projection head q_theta.
        """
        if mixtures is None:
            mixtures = self._build_obs_only_temporal_mixtures(
                self._projection_detach_components,
                self._projection_detach_mixing_weights,
            )

        # Legacy / deploy-aligned step-0 head stats (also returned for the data-NLL anchor).
        # The step-0 projection-head forward is needed ONLY when per-step conditioning is OFF (the
        # per-step branch below re-derives the head from each step-i mixture) OR when the optional
        # obs-only data-NLL anchor (`info_projection_head_data_nll`) consumes the step-0 head. Skip
        # it otherwise to avoid wasted compute.
        q_mean0 = q_logvar0 = None
        if not self.per_step_conditioning or self.info_projection_head_data_nll:
            q_mean0, q_logvar0 = self._projection(ms_pred_mean, ms_pred_logvar)

        T = self.info_projection_num_samples
        kl_steps = []
        for i in range(self.horizon_len):
            mixture_i = mixtures[i]
            if self.per_step_conditioning:
                q_mean_i, q_logvar_i = self._projection_from_mixture(mixture_i)
            else:
                q_mean_i, q_logvar_i = q_mean0, q_logvar0
            q_dist = self._make_head_dist(q_mean_i, q_logvar_i)
            kl_steps.append(self._kl_dist_to_mixture_loss_at_i(q_dist, mixture_i, T))

        # Reduce over the horizon (flat or temporal-weighted) and add the feature dim -> (E x B x 1)
        kl_loss = self._reduce_horizon_steps(kl_steps).unsqueeze(-1)
        return kl_loss, (q_mean0, q_logvar0)

    def _make_head_dist(self, q_mean: Tensor, q_logvar: Tensor):
        """Wrap the summary head (mean, logvar) as an obs-only Independent distribution.

        The deploy logvar bound is NO LONGER applied here: it is now applied SYSTEMATICALLY at the
        single owner of the projection-head forward, :meth:`_default_forward_projection_head`, which
        produces every ``(q_mean, q_logvar)`` fed to this method (via :meth:`_projection` /
        :meth:`_projection_from_mixture`). Re-applying it here would double-bound the training path.
        This keeps training matched to deployment and the bound anchored in the NLL graph of
        whichever projection objective (IPROJ / SIW-MP / GMS-IWAE) is active, while the
        *penalise-iff-applied* invariant stays exact (PME Projection-Head Consolidation `.junie`
        plan, `refactor_pme_projection_head_consolidation_ensemble_plan_20260622.md`).
        """
        return dist.Independent(
            self._to_distribution(q_mean, q_logvar, scale_are_log_variance=True), 1
        )

    def _kl_dist_to_mixture_loss_at_i(
        self, q_dist, mixture_i: MixtureSameFamily, num_samples: int
    ) -> Tensor:
        """Reparameterised MC estimate of `KL(q_dist || mixture_i)` -> (E x B), clamped >= 0.

        A finite-sample MC estimate of a (non-negative) KL can come out slightly negative when
        `num_samples` is small (unbiased only in expectation). Clamp at 0 so the term stays a valid
        divergence and never pulls the composite sum negative / produces a spurious reward gradient.
        """
        samples = q_dist.rsample(torch.Size([num_samples]))  # (T x E x B x O)
        log_q = q_dist.log_prob(samples)  # (T x E x B)
        log_mixture = mixture_i.log_prob(samples)  # (T x E x B)
        return (log_q - log_mixture).mean(dim=0).clamp_min(0.0)  # (E x B)

    def _siw_mp_dist_to_mixture_loss_at_i(
        self, p_target_dist, q_proposal_i: MixtureSameFamily, num_samples: int
    ) -> Tensor:
        """Stratified-IWAE negated bound of the head `p_target_dist` under mixture-stratified
        samples from `q_proposal_i` -> (E x B). See `_siw_mp_loss` docstring for the math.
        """
        N = num_samples
        log_N = float(np.log(N))
        # Normalised categorical mixing log-weights log pi_k -> (E x B x K_i) -> (K_i x E x B).
        log_pi = q_proposal_i.mixture_distribution.logits.permute(2, 0, 1)
        # Stratified sampling: N samples per mixture component (the K_i unroll candidates).
        z = q_proposal_i.component_distribution.rsample(torch.Size([N]))
        z = z.permute(0, 3, 1, 2, 4)  # (T x K_i x E x B x O)
        log_p = p_target_dist.log_prob(z)  # (T x K_i x E x B)
        log_q = q_proposal_i.log_prob(z)  # (T x K_i x E x B)
        log_weights = log_pi + log_p - log_q  # (T x K_i x E x B)
        elbo_i = torch.logsumexp(log_weights, dim=(0, 1)) - log_N  # (E x B)
        return -elbo_i  # negate to minimise

    def _distribution_to_mixture_divergence_at_j(
        self,
        head_dist_j,
        mixture_j: MixtureSameFamily,
        kind: str,
        num_samples: int,
        target_j: Optional[Tensor] = None,
    ) -> Tensor:
        """Dispatch a single-distribution-to-mixture divergence (Stage D shared estimator).

        `kind="kl"`     -> mode-seeking `KL(head_dist || mixture_i)`.
        `kind="siw_mp"` -> mass-covering stratified-IWAE negated bound of `head_dist` under
                           `mixture_i`-stratified samples.
        `kind="gms_iwae"` -> generative mode-seeking IWAE negated bound (samples from `head_dist`,
                           importance-weights against `p(y|z) * mixture_i`). Requires `target_i`.
        """
        if kind == "kl":
            return self._kl_dist_to_mixture_loss_at_i(
                head_dist_j, mixture_j, num_samples
            )
        elif kind == "siw_mp":
            return self._siw_mp_dist_to_mixture_loss_at_i(
                head_dist_j, mixture_j, num_samples
            )
        elif kind == "gms_iwae":
            assert (
                target_j is not None
            ), "kind='gms_iwae' needs the obs-only target_i (the decoder reconstruction anchor)"
            return self._gms_iwae_dist_to_mixture_loss_at_i(
                head_dist_j, mixture_j, target_j, num_samples
            )
        raise ValueError(f"Unknown divergence kind: {kind}")

    def _gms_iwae_dist_to_mixture_loss_at_i(
        self,
        q_proposal_dist,
        mixture_i: MixtureSameFamily,
        target_i: Tensor,
        num_samples: int,
    ) -> Tensor:
        """Generative mode-seeking IWAE negated bound -> (E x B).

        Samples ``z_n ~ q_theta`` (the head, reparameterised) and importance-weights against the
        generative decoder ``p(y|z)`` times the temporal mixture ``p_mix``:

            L = -( logsumexp_n [ log p(y|z_n) + log p_mix(z_n) - log q_theta(z_n) ] - log N )

        Mode-seeking (sample from the head) + generative (the decoder anchors q_theta to the mode
        that best explains the observed target ``y == target_i``).
        """
        N = num_samples
        log_N = float(np.log(N))
        z = q_proposal_dist.rsample(torch.Size([N]))  # (N x E x B x O), reparameterised
        log_decoder = self._make_decoder_dist(z).log_prob(
            target_i
        )  # (N x E x B); p(y|z)
        log_mixture = mixture_i.log_prob(z)  # (N x E x B); p_mix
        log_q = q_proposal_dist.log_prob(z)  # (N x E x B); q_theta
        elbo_i = torch.logsumexp(log_decoder + log_mixture - log_q, dim=0) - log_N
        return -elbo_i  # negate to minimise

    def _horizon_step_temporal_weights(self) -> Tensor:
        """Per-horizon-step temporal-discount weights, normalised to sum to 1 -> shape (horizon_len,).

        Uses the DEDICATED `projection.temporal_weights` discount (stored as
        `self._projection_temporal_weights`), which is indexed on the HORIZON / per-step axis:
          - a single gamma is expanded as gamma^i over the `horizon_len` steps;
          - a per-step profile (length `horizon_len`) is used directly.
        This is intentionally distinct from the main forecast `temporal_weights` (which discounts
        the multi-step forecast NLL), since the projection (IPROJ/SIW-MP/GMS-IWAE)/RC per-step
        terms serve a different purpose. When the discount is a scalar 1.0 this reduces to a uniform
        1/H (the flat mean), so `siw_mp_temporal_weighting=True` with gamma=1 matches the flat
        reduction.
        """
        w = (
            self._projection_temporal_weights.detach()
        )  # Guarantee that we don't backprop through this.
        if w.numel() == 1:
            # Scalar discount factor gamma -> gamma^i over the horizon steps.
            gamma = w.reshape(()).clamp(min=1e-6)
            steps = torch.arange(
                self.horizon_len, device=self.device, dtype=self.model_dtype
            )
            w = torch.pow(gamma, steps)
        # Otherwise `w` is already a per-horizon-step profile of length `horizon_len`.
        return w / w.sum()

    def _reduce_horizon_steps(self, steps: Sequence[Tensor]) -> Tensor:
        """Reduce a list of `horizon_len` per-step (E x B) terms to a single (E x B) term.

        Flat mean by default; temporal-discount weighted average when `siw_mp_temporal_weighting`.
        """
        stacked = torch.stack(list(steps), dim=0)  # (H x E x B)
        if self.projection_temporal_weights:
            w = self._horizon_step_temporal_weights().view(-1, 1, 1)  # (H x 1 x 1)
            return (stacked * w).sum(dim=0)
        return stacked.mean(dim=0)

    def _projection(
        self, ms_pred_mean: Tensor, ms_pred_logvar: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Build the shared summary head distribution q_theta (obs-only) from mixture stats."""
        # Extract the obs-only moment-matched (mean, logvar) summary as the leading
        # `singlestep_obs_len` dimensions of the (horizon-length) forward output. This matches
        # the obs-only convention used by the deploy head (see `_default_deploy_head` V1) and the
        # `obs_space_only=True` mixture path. NOTE: we intentionally do NOT reuse
        # `multistep_to_singlestep_next_obs_adapter` here: its obs slice is indexed on the full
        # history-length composed array, so it only aligns with the horizon-length forward output
        # when `history_len == horizon_len` (it returns a 0-width slice otherwise).
        # When the projection backprop scope detaches the components (mixing_only / head_only), the
        # head's CONDITIONING INPUT (the moment-matched body output) must also be detached, else the
        # gradient would still reach the prediction body through the head input (the per-step
        # conditioning path is already protected because it reads the detached mixture moments).
        if self._projection_detach_components:
            ms_pred_mean = ms_pred_mean.detach()
            ms_pred_logvar = ms_pred_logvar.detach()
        ss_mean = ms_pred_mean[..., : self.singlestep_obs_len]
        ss_logvar = ms_pred_logvar[..., : self.singlestep_obs_len]
        # Single owner of the head forward (+ systematic deploy bound). `only_elite=False`: the
        # projection objectives train on the full ensemble (matching the prior behaviour).
        return self._default_forward_projection_head(
            ss_mean, ss_logvar, only_elite=False
        )

    def _projection_from_mixture(
        self, mixture_i: MixtureSameFamily
    ) -> Tuple[Tensor, Tensor]:
        """Per-step conditioning (Stage A.1): feed the step-i mixture's moment-matched obs-only
        stats to the summary head (vs always the step-0 slice).

        This makes `q_theta` a genuine learned *function* of moment-matched stats
        `g: (mixture moments) -> compressed dist`, trained on the full set of `(stats_i, mixture_i)`
        pairs instead of an implicit step-0-only horizon-average extrapolation.
        """
        # Reuse the exact moment-matched clamp convention from `_default_forward_mixture_dist`.
        ss_mean = mixture_i.mean
        ss_logvar = torch.log(
            mixture_i.variance.clamp(min=variance_floor(self.model_dtype))
        ).clamp(max=self._MIX_LOGVAR_SAFE_MAX)
        # Single owner of the head forward (+ systematic deploy bound). `only_elite=False`: the
        # projection objectives train on the full ensemble (matching the prior behaviour).
        return self._default_forward_projection_head(
            ss_mean, ss_logvar, only_elite=False
        )

    @torch.compiler.disable
    def _siw_mp_loss(
        self,
        ms_pred_mean: Tensor,
        ms_pred_logvar: Tensor,
        state_forecast: Optional[Sequence[MixtureSameFamily]] = None,
    ) -> Tensor:
        """
        SIWAE / stratified IWAE bound loss (Stage I).

        Two KL version:
            # ToDo: validate semantic
            Standard MC KL: Samples from p (SS) and computes KL(p || q).
            SIWAE: Samples from q (MS) and computes KL(q || p).

        These are different objectives:
            # ToDo: validate semantic
            KL(p || q) is mode-seeking for p. It forces the SS head to fit inside one mode of the MS mixture.
            KL(q || p) is mean-covering for p. It forces the SS head to stretch and cover all components of the MS mixture.


        ELBO convention. We bound the (log) evidence of the dedicated summary head, the
        *target* density ``p == info_projection head``, using the learned obs-only temporal mixture as the
        *proposal* ``q == prob_mix = Sum_k pi_k q_k`` (we sample from its components). For every
        horizon step ``i`` we draw ``T`` stratified reparameterised samples, one batch per
        component (the ``K_i`` unroll-window candidates), and form the stratified-IWAE bound

            ELBO_i = logsumexp_{t,k} [ log pi_k + log p(x_{t,k}) - log q(x_{t,k}) ] - log T

        where ``log pi_k`` are the (normalised) categorical mixing log-weights, ``p`` is the head
        and ``q`` is the FULL mixture density (the importance-sampling denominator). Including
        ``log pi_k`` in the stratified sum is what makes the bound respect the learned temporal
        mixture weights; with ``Sum_k pi_k = 1`` the correct normaliser is ``- log T`` (the number
        of samples per stratum), not ``- log(T * K_i)``. The per-step negated bounds are averaged
        over the horizon.

        PME-faithful, horizon-based estimator (plan option "b"): this replaces the history-indexed
        base-class `SIW_MP_temporal_mixture_to_dist_loss` (which only supports
        `history_len == horizon_len` and omitted the ``log pi_k`` weights). The per-step mixtures
        may be supplied via `mixtures` (shared with `_info_projection_loss`); otherwise they are
        built here honouring `projection_backprop_mode`.

        Stage A (RLRP-711+): when `per_step_conditioning` is set the head `p` is re-derived
        from the *step-i* mixture moments (a true compressor) instead of always the step-0 slice;
        when `siw_mp_temporal_weighting` is set the per-step negated bounds are combined with a
        temporal-discount weighted average instead of a flat mean.
        """
        if state_forecast is None:
            state_forecast = self._build_obs_only_temporal_mixtures(
                self._projection_detach_components,
                self._projection_detach_mixing_weights,
            )

        # Legacy / deploy-aligned step-0 head stats, used ONLY when per-step conditioning is OFF.
        # Skip the projection-head forward when ON (the per-step branch re-derives the head from
        # each step-i mixture) to avoid wasted compute.
        ss_proj_mean0 = ss_proj_logvar0 = None
        if not self.per_step_conditioning:
            ss_proj_mean0, ss_proj_logvar0 = self._projection(
                ms_pred_mean, ms_pred_logvar
            )

        N = self.siw_mp_num_samples
        siw_mp_steps = []
        for i in range(self.horizon_len):
            q_proposal_i = state_forecast[i]  # proposal mixture q = Sum_k pi_k q_k
            if self.per_step_conditioning:
                p_mean_i, p_logvar_i = self._projection_from_mixture(q_proposal_i)
            else:
                p_mean_i, p_logvar_i = ss_proj_mean0, ss_proj_logvar0
            # Target density p == projection head: batch (E x B), event (O).
            p_target_dist = self._make_head_dist(p_mean_i, p_logvar_i)
            siw_mp_steps.append(
                self._siw_mp_dist_to_mixture_loss_at_i(p_target_dist, q_proposal_i, N)
            )

        # Reduce the negated per-step bounds over the horizon -> (E x B x 1).
        return self._reduce_horizon_steps(siw_mp_steps).unsqueeze(-1)

    @torch.compiler.disable
    def _gms_iwae_loss(
        self,
        ms_pred_mean: Tensor,
        ms_pred_logvar: Tensor,
        target_HOxDoa: Tensor,
        mixtures: Optional[Sequence[MixtureSameFamily]] = None,
    ) -> Tensor:
        """Generative Mode-Seeking IWAE (GMS-IWAE) loss -> (E x B x 1).

        The diagonal opposite of SIW-MP: instead of stratified-sampling from the mixture and
        importance-weighting against the head, GMS-IWAE samples from the head `q_theta` (the
        proposal) and importance-weights against a generative decoder `p(y|z)` times the per-step
        temporal mixture `p_mix`. This is mode-seeking (sample from q_theta) AND generative (the
        decoder reconstruction anchors q_theta to the data-supported mode of the mixture).

        Mirrors `_siw_mp_loss` plumbing: the obs-only per-step mixtures may be supplied via
        `mixtures` (shared with the other projection terms) else built here honouring
        `projection_backprop_mode` (incl. `head_only`); the head is conditioned per-step when
        `per_step_conditioning`, and the per-step negated bounds are reduced over the
        horizon (flat or `siw_mp_temporal_weighting`).
        """
        if mixtures is None:
            mixtures = self._build_obs_only_temporal_mixtures(
                self._projection_detach_components,
                self._projection_detach_mixing_weights,
            )

        # Step-0 projection-head stats are used ONLY when per-step conditioning is OFF (the per-step
        # branch below re-derives the head from each step-i mixture); skip the forward when ON to
        # avoid wasted compute.
        q_mean0 = q_logvar0 = None
        if not self.per_step_conditioning:
            q_mean0, q_logvar0 = self._projection(ms_pred_mean, ms_pred_logvar)

        N = self.gms_iwae_num_samples
        gms_iwae_steps = []
        for i in range(self.horizon_len):
            mixture_i = mixtures[i]
            if self.per_step_conditioning:
                q_mean_i, q_logvar_i = self._projection_from_mixture(mixture_i)
            else:
                q_mean_i, q_logvar_i = q_mean0, q_logvar0
            q_dist = self._make_head_dist(q_mean_i, q_logvar_i)
            # obs-only single-step next-observation target y_i.
            target_i = target_HOxDoa[..., i, : self.singlestep_obs_len]
            gms_iwae_steps.append(
                self._gms_iwae_dist_to_mixture_loss_at_i(q_dist, mixture_i, target_i, N)
            )

        return self._reduce_horizon_steps(gms_iwae_steps).unsqueeze(-1)

    @torch.compiler.disable
    def _rollout_window_update(
        self, model_in: Tensor, next_obs: Tensor, act_gt: Tensor
    ) -> Tensor:
        """Port of the AR sliding-window splice from
        ``CompoundedPredictionMultiStepIterator.compouned_prediction_model_in_update`` (reference,
        not a dependency — the base classes differ).

        Drops the oldest history step and appends ``cat([next_obs, act_gt])`` as the newest step:
        ``reshape -> drop oldest -> cat new -> revert``.

        **Normalisation contract (RLRP-761 ``F-3``)**: both ``next_obs`` (a projection-head output)
        and ``act_gt`` (a slice of ``target_HOxDoa``) are **TARGET**-space quantities, whereas
        ``model_in`` is **INPUT**-space. The composed new step is therefore routed through
        :meth:`_ar_bridge_step_target_to_input` — the same single source of truth used by the AR
        self-feed (``state_history_update``) and by the CP deploy splice — before being spliced into
        the window. The bridge is a **strict identity** (tensor returned untouched, ``M5``
        bit-exact) for every shared-facade normalizer type, i.e. everything but
        ``standard_symmetric_innovation``; under the decoupled innovation facades it applies the
        diagonal ``s / sigma_state`` gain that this method previously omitted.

        The asymmetric ``standard`` normalizer's ``denormalize -> shift -> renormalize`` round-trip
        is intentionally not replicated here (this term is an opt-in experimental lever).
        """
        # (..., F) -> (..., MS, O+A)
        win = timestep_first_multistep_dim_unflaten_array(
            model_in,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            sequence_len=self.history_len,
            enable_last_action_padding=False,
        )
        win_next = win[..., 1:, :]  # drop oldest step
        # RLRP-761 F-3: TARGET -> INPUT before splicing into the INPUT-space window.
        # Strict identity (bit-exact) whenever no decoupled gain is registered.
        new_step = self._ar_bridge_step_target_to_input(
            torch.cat([next_obs, act_gt], dim=-1)
        ).unsqueeze(
            -2
        )  # (..., 1, O+A)
        win = torch.cat([win_next, new_step], dim=-2)
        # (..., MS, O+A) -> (..., F)
        return revert_timestep_first_multistep_dim_unflaten_array(
            win,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            remove_last_action_padding=False,
        )

    def _rc_bridge_provenance_notice(self) -> None:
        """One-shot provenance signal for the RC rollout window bridge (RLRP-761 ``F-3``).

        Emitted once per process when the rollout-consistency term runs **and** a decoupled
        target->input bridge gain is registered, so an operator comparing against pre-``F-3``
        logs knows the rollout window is now coordinate-corrected. Intentionally an ``info``-level
        one-liner, NOT a warning: post-``F-3`` the term is *correct*, so warning on every run of a
        correct code path would only desensitize operators to real warnings.
        """
        if getattr(self, "_rc_bridge_notice_emitted", False):
            return
        getter = getattr(self, "get_one_d_trj_model_ar_bridge_gain", None)
        act_getter = getattr(self, "get_one_d_trj_model_ar_bridge_gain_act", None)
        has_gain = (getter is not None and getter() is not None) or (
            act_getter is not None and act_getter() is not None
        )
        if not has_gain:
            return
        self._rc_bridge_notice_emitted = True
        consol_msg_universal_one_liner(
            "RLRP-761 F-3: rollout-consistency window splice is now bridged "
            "TARGET->INPUT (decoupled normalizer gain is registered).",
            caller_name="_rollout_consistency_loss",
        )

    @torch.compiler.disable
    def _rollout_consistency_loss(
        self,
        model_in: Tensor,
        target_HOxDoa: Tensor,
        mixtures_orig: Sequence[MixtureSameFamily],
    ) -> Tensor:
        """Rollout self-consistency distillation (Stage D, the core anti-compounding lever).

        Trains the deploy head so its *autoregressive composition* reproduces the trusted multi-step
        mixture forecast — penalising the compounding gap rather than only per-step marginals.

        For each rollout step ``j`` we (1) ``forward`` the current window, (2) build the head's
        step-0 compressed prediction ``q_theta``, (3) accumulate
        ``divergence(q_theta_j, model_k_step_mixture_j)`` against the ORIGINAL forward's step-``j``
        obs-only mixture (``mixtures_orig`` — detached per ``rollout_consistency_backprop_mode`` so
        the consistency term trains the head to follow the forecast, not the reverse), then (4)
        splice the head's predicted next-obs + the ground-truth action back into the window
        (free-running rollout, mirroring deploy's ``self._last_pred_mean`` convention).
        """
        self._rc_bridge_provenance_notice()
        RC = self.rollout_consistency_num_steps
        if RC <= 0:
            RC = min(3, self.horizon_len)
        RC = min(RC, self.horizon_len)

        num_samples = max(
            self.siw_mp_num_samples,
            self.info_projection_num_samples,
            self.gms_iwae_num_samples,
        )

        window = model_in
        rc_steps = []
        # Suppress the primary `true_mixture` / post-mixing (MIX) per-step accumulation during the
        # RC re-forwards: those accumulators were already consumed by `_probabilistic_loss` BEFORE
        # RC runs, so re-firing `_accumulate_post_mix_losses` here only appends discarded entries
        # (wasted compute/memory). Temporarily clear the target guard and restore it afterwards.
        _saved_post_mix_target = self._post_mixture_loss_target
        self._post_mixture_loss_target = None
        # RLRP-726: the RC re-forwards MUST stay free-running (no GT leakage into the compounded
        # metric), so null the forecast teacher-forcing target for the duration of the rollout and
        # restore it afterwards (mirrors the post-mix target guard above).
        _saved_forecast_tf_target = self._forecast_tf_target
        self._forecast_tf_target = None
        try:
            for j in range(RC):
                ms_mean_j, ms_logvar_j = self.forward(window, use_propagation=False)
                # mixing_only: detach the rolled-window forward output so the consistency term
                # trains ONLY the head (and downstream mixing fct via mixtures_orig), not the body.
                if self._rollout_consistency_detach_components:
                    ms_mean_j = ms_mean_j.detach()
                    ms_logvar_j = ms_logvar_j.detach()
                q_mean_j, q_logvar_j = self._projection(ms_mean_j, ms_logvar_j)
                head_dist_j = self._make_head_dist(q_mean_j, q_logvar_j)
                rc_steps.append(
                    self._distribution_to_mixture_divergence_at_j(
                        head_dist_j,
                        mixtures_orig[j],
                        self.rollout_consistency_divergence,
                        num_samples,
                        # obs-only step-j target (the GMS decoder reconstruction anchor; ignored by
                        # the kl / siw_mp kinds).
                        target_j=target_HOxDoa[..., j, : self.singlestep_obs_len],
                    )
                )
                if j < RC - 1:
                    act_gt = target_HOxDoa[..., j, -self.singlestep_act_len :]
                    window = self._rollout_window_update(window, q_mean_j, act_gt)
        finally:
            self._post_mixture_loss_target = _saved_post_mix_target
            self._forecast_tf_target = _saved_forecast_tf_target

        rc = torch.stack(rc_steps, dim=0).mean(dim=0).unsqueeze(-1)  # (E x B x 1)
        return rc

    def _denorm_obs_for_dh(self, obs: Tensor) -> Tensor:
        """Map a single-step obs tensor to RAW (denormalized) space for the DH residual (RLRP-731).

        Dispatch on the wrapper-propagated normalizer handles:

        - **Robust / block-facade normalizers** (``standard_symmetric`` / ``winsorized`` /
          ``quantile``): the wrapper propagates a dedicated obs-block DENORM handle (action
          ``B3-handle``); apply it (piecewise-linear / affine map, differentiable a.e.).
        - **Asymmetric ``standard`` normalizer:** targets (and thus deploy predictions) are
          already in RAW target space -> the handle is ``None`` and the map is the identity.
        - **Normalization disabled:** identity.
        """
        denorm_handle = getattr(self, "_one_d_trj_model_obs_denorm_handle", None)
        if denorm_handle is not None:
            return denorm_handle(obs)
        return obs

    def _compounded_deploy_nll(
        self,
        model_in: Tensor,
        target_HOxDoa: Tensor,
        meta: Optional[Dict[str, Any]] = None,
        target_raw_HOxDoa: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compounded-prediction deploy loss (CP) per-feature NLL accumulator (RLRP-708, C2).

        Free-running AR over the single-step DEPLOY head: at unroll step ``k`` score
        ``deploy(x_k)`` against the per-step obs target ``y_obs_k`` and accumulate the
        gamma-weighted, ``Z``-normalised per-feature NLL ``(E x B x O)``. The MTM-Pro
        multistep forecast loss is NOT involved; this trains ONLY the deploy path. Mirrors
        the step-0 SS-term NLL (so CP's ``k=0`` term equals the SS term it subsumes) but over
        the whole unroll. ``deploy`` honours ``deploy_head_mode`` internally, so CP trains the
        projection head when ``deploy_head.mode=projection`` and the forecast/mixing path
        otherwise. The returned per-feature tensor matches the SS-term shape so the caller can
        reuse the exact SS feature-weight / reduction / auto-weighting pipeline.

        RLRP-731 (action ``B2-loss`` of the Deploy History Drift Residual Loss `.junie` plan,
        ``feat_deploy_history_drift_residual_loss_plan_20260705.md``): when
        ``enable_history_drift_loss`` is on, ALSO returns the deploy-history drift residual
        (DH) scored on the TERMINAL unroll step, in RAW (denormalized) obs space:
            ``(denorm(pred_F) - denorm(target_F))**2``  (per-feature, ``E x B x O``).
        The DH residual is computed on the exact CP trajectory (same sampled horizon
        ``horizon_unroll_len``, same teacher-forcing splice — TF inheritance is the
        documented intent, plan §2), so the CP schedulers keep single ownership of horizon
        sampling / TF probability / stepping, and no extra forward pass is needed (one
        unroll, two objectives). Loop convention: the deploy head is applied ``F`` times
        (``range(F)``) and the terminal prediction is scored against target index ``F-1``
        (``F=1`` degenerates to a one-step raw MSE).

        :return: ``(cp_nll, dh_residual)`` where ``dh_residual`` is ``None`` when the DH
            sub-term is disabled.
        """
        # .... Sample the unroll length + temporal discount (shared with the mixin loss U-loop) .
        # These reuse the ``CompoundedPredictionMultiStepIterator`` building blocks so the
        # unroll-length policy / temporal discount stay defined in exactly ONE place.
        horizon_unroll_len = self._sample_ar_horizon_unroll_len()
        gamma = self._current_ar_temporal_gamma()

        # Register the second-AR-stage scheduler diagnostics for TensorBoard. The host CP path
        # bypasses the mixin ``loss`` U-loop (Option C2), so these keys must be emitted here too;
        # otherwise the "Compounded prediction" TensorBoard cards stay empty/flat (RLRP-708).
        # A7 (RLRP-788): pure diagnostic writes -> gated behind the meta-collection
        # kill-switch (the sampled ``horizon_unroll_len`` / ``gamma`` above are still computed
        # because the unroll below consumes them). RLRC meta-collection kill-switch `.junie`
        # plan (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
        if meta is not None and self._enable_meta_collection:
            meta["horizon_unroll_len"] = horizon_unroll_len
            meta["teacher_forcing_prob"] = (
                self._teacher_forcing_scheduler.current_probability
                if self._teacher_forcing_scheduler is not None
                else 1.0
            )
            meta["ar_temporal_weights"] = gamma

        # The compounded-prediction splice (``compouned_prediction_model_in_update`` ->
        # ``reshape_tensor_to_ms_f_dim``) is written for a 2D ``(B, in)`` model input (the AR
        # family contract; it ``raise``s for ``num_members > 1``). In ``_probabilistic_loss``
        # the model input is ``(E, B, in)`` with ``E == num_members``, so drop the singleton
        # ensemble axis before the unroll to keep the sliding-window splice shape-consistent.
        if model_in.dim() > 2 and model_in.shape[0] == self.num_members:
            model_in = model_in.squeeze(0)

        self.wipe_ar_memory()
        cp_nll: Optional[Tensor] = None
        # RLRP-751 (task T5): accumulate the per-step CP deploy points here; the
        # composite ``_probabilistic_loss`` stage composes the FEAT_GEOM_CP term
        # ONCE (SS-under-CP is intentionally SUPPRESSED — the ``k == 0`` unroll
        # step already covers the ``t+1`` deploy signal the SS term would score).
        if self._feature_loss_active():
            self._cp_feature_points = []
        # RLRP-830 Item C (Step 2): in ``true_mixture`` mode the per-step deploy mixtures
        # (K = 1, uniform shapes) are COLLECTED here and scored with ONE stacked per-feature
        # NLL pass after the loop (``_stacked_true_mixture_per_feature_nll``); the unroll
        # stays sequential and the ``gamma`` accumulation order below is unchanged.
        batched_true_mixture_scoring = (
            self.batched_cp_scoring
            and self.ms_probabilistic_loss_mode != "moment_matched"
        )
        deferred_mixtures: list = []
        deferred_targets: list = []
        for each_u_t in range(horizon_unroll_len):
            ss_pred_mean, ss_pred_logvar = self.deploy(model_in, only_elite=False)

            ss_target_t = target_HOxDoa[..., each_u_t, : self.singlestep_obs_len]
            if self.ms_probabilistic_loss_mode == "moment_matched":
                ss_distribution = self._to_distribution(ss_pred_mean, ss_pred_logvar)
                step_nll = -ss_distribution.log_prob(ss_target_t)
            elif batched_true_mixture_scoring:
                deferred_mixtures.append(self._latest_deploy_ss_dist_mixture)
                deferred_targets.append(target_HOxDoa[..., each_u_t, :])
                step_nll = None
            else:  # true_mixture (the per-feature NLL, mirroring the SS block)
                ss_obs_act_target = target_HOxDoa[..., each_u_t, :]
                step_nll = self._true_mixture_per_feature_nll(
                    self._latest_deploy_ss_dist_mixture, ss_obs_act_target
                )[..., : self.singlestep_obs_len]

            if step_nll is not None:
                cp_nll = self._accumulate_ar_temporal_weighted(
                    cp_nll, step_nll, each_u_t, gamma
                )

            # RLRP-751 (task T5) — compounded-prediction / free-running unroll
            # geometry supervision. Accumulate EVERY CP unroll step's deploy point
            # (deploy-head MEAN — NEVER a sample — in the single-step obs layout
            # ``(..., singlestep_obs_len)``) + matching per-step obs target, in
            # unroll order, so the composite stage composes the DEDICATED
            # ``FEAT_GEOM_CP`` term ONCE (mean over the horizon). This is the
            # priority MTM-Pro lever: attitude drift compounds along the
            # free-running unroll, so the geometry term must penalize the
            # orientation error at each unrolled step, not only ``t=1``. Grad flows
            # through the AR splice to every step. SS-under-CP is intentionally
            # SUPPRESSED (the ``k == 0`` step already covers the ``t+1`` deploy
            # signal). No-op / bit-neutral unless the term is active.
            if self._feature_loss_active():
                self._cp_feature_points.append(
                    (ss_pred_mean[..., : self.singlestep_obs_len], ss_target_t)
                )

            if each_u_t < horizon_unroll_len - 1:
                model_in = self.compouned_prediction_model_in_update(
                    each_u_t, model_in, target_HOxDoa, next_obs=ss_pred_mean
                )

        if deferred_mixtures:
            # RLRP-830 Item C: ONE scoring pass over the U stacked deploy mixtures ->
            # (U x E x B x O+A); then the SAME per-step ``gamma`` accumulation, in order.
            stacked_nll = self._stacked_true_mixture_per_feature_nll(
                deferred_mixtures, deferred_targets
            )[..., : self.singlestep_obs_len]
            for each_u_t in range(horizon_unroll_len):
                cp_nll = self._accumulate_ar_temporal_weighted(
                    cp_nll, stacked_nll[each_u_t], each_u_t, gamma
                )

        # .... DH terminal-step residual (RLRP-731) — same unroll, no extra forward ...........
        # ``ss_pred_mean`` is the step-``F-1`` deploy prediction left by the last loop
        # iteration (the graph still back-propagates through all F applications of the deploy
        # head via the AR splice). Both operands go through the SAME denorm operator, so any
        # denorm bias cancels in the residual (plan §4.2, Option 2).
        #
        # RLRP-748 (audit §5 item 3) WINDOW SEMANTICS — implementation vs whiteboard notation:
        # the RLRP-731 whiteboard recursion self-feeds only the HISTORY channel, but the CP
        # splice (``compouned_prediction_model_in_update`` above) self-feeds the WHOLE obs
        # window (only the ACTION columns stay ground-truth). DH is therefore intentionally a
        # FULLY FREE-RUNNING endpoint loss: it measures the terminal obs-window drift of an
        # uninterrupted deploy rollout, which is the deployment-faithful quantity we want to
        # penalize (and stricter than the history-only notation). This divergence is by design,
        # not a defect (RLRP-731 assessment §1.6).
        dh_residual: Optional[Tensor] = None
        if self.enable_history_drift_loss:
            # The PREDICTION is always denormalized (it is a model output in normalized space
            # under robust normalizers; a no-op under asymmetric ``standard``). Only the TARGET
            # sourcing differs between the two modes (plan §5.2 / §5.5):
            dh_pred_raw = self._denorm_obs_for_dh(
                ss_pred_mean[..., : self.singlestep_obs_len]
            )
            if self.history_drift_target_mode == "raw_passthrough":
                # RLRP-731 batch ``B4-raw`` (Option 1): consume the SEPARATE raw obs target
                # threaded from ``OneDTransitionRewardModelV2._process_batch`` (never reuse the
                # NORMALIZED ``target_HOxDoa`` — that is what the removed ``__init__`` guard
                # protected against). Same terminal index + obs-block slice as ``dh_pred_raw``;
                # match its ``compute_dtype`` to avoid a float32/float64 subtraction.
                assert target_raw_HOxDoa is not None, (
                    "history_drift_target_mode='raw_passthrough' requires the wrapper to "
                    "thread ``target_raw_HOxDoa`` (obs-block raw target); got None. This "
                    "usually means the model was called directly, bypassing "
                    "``OneDTransitionRewardModelV2.loss`` / ``update``."
                )
                dh_target_raw = target_raw_HOxDoa[
                    ..., horizon_unroll_len - 1, : self.singlestep_obs_len
                ].to(dh_pred_raw.dtype)
            else:  # inline_denorm (landed B2 default): denormalize the normalized target
                # RLRP-748 (audit §5 item 2) INVARIANT: in ``inline_denorm`` the DH
                # correctness — including the ``target_is_delta`` case — rests on BOTH operands
                # passing through the *identical* affine denorm map so the additive shift ``m``
                # cancels in the residual (leaving ``s·(pred − target)``; this holds for both the
                # L2 ``(pred − target)²`` and the L1 ``|pred − target|`` form selected by
                # ``mae_loss``; see §2 / the in-code note above at the DH block header). Route
                # the TARGET through the SAME
                # ``self._denorm_obs_for_dh`` handle used for ``dh_pred_raw`` — never a second /
                # divergent handle. ``dh_pred_raw`` (above) and ``dh_target_raw`` (here) MUST
                # therefore share one callable; ``test_dh_inline_denorm_uses_shared_denorm_handle``
                # guards this so a future change to a different handle fails loud, not silent.
                dh_target_norm = target_HOxDoa[
                    ..., horizon_unroll_len - 1, : self.singlestep_obs_len
                ]
                dh_target_raw = self._denorm_obs_for_dh(dh_target_norm)
            # RLRP-736 Item 2: the DH residual is a raw per-feature residual and is
            # therefore the one attitude term that is NOT sign-invariant — on a
            # ``w ≈ 0`` frame where the target-continuity and the prediction
            # hemisphere disagree it can incur the full antipodal error (``‖q − (−q)‖²
            # = 4‖q‖²`` under L2, ``‖q − (−q)‖₁ = 2‖q‖₁`` under L1) as spurious noise,
            # worst in the aggressive-rotation regime. Sign-align each attitude obs
            # slot of the prediction to the target (``⟨q_pred, q_tgt⟩ ≥ 0`` flip
            # only — no re-normalisation, so non-antipodal magnitudes are untouched)
            # before the residual, making the DH residual double-cover-invariant on
            # the attitude columns (matching the sign-invariant CP-NLL / geodesic /
            # chordal terms). The flip-only alignment is defined on the final operands
            # and is valid for BOTH the squared (L2) and absolute (L1) residual, so it
            # is applied regardless of the ``mae_loss`` toggle below. Non-attitude
            # columns keep the exact raw residual (bit-exact); no-op OFF /
            # ``quaternion_legacy``. Applies identically in both ``inline_denorm`` and
            # ``raw_passthrough`` modes.
            if self._quaternion_ar_continuity_active():
                # Reuse the shared Item 1 helper in its flip-ONLY mode
                # (``project_to_s3=False``): sign-align each attitude slot of the
                # prediction to the target WITHOUT re-normalising, so non-antipodal
                # magnitudes keep their exact raw-MSE contribution.
                dh_pred_raw = align_quaternion_slots_to_reference(
                    dh_pred_raw,
                    dh_target_raw,
                    self._orientation_singlestep_slots,
                    project_to_s3=False,
                )

            # RLRP-748: the DH residual honours the model-level loss-type toggle
            # ``ms_model.mae_loss`` (L1 vs L2), consistent with the primary forecast
            # fitting loss. The RLRP-731 maths were written using an MSE *example* but
            # are not constrained to L2: the denorm-bias/shift cancellation in
            # ``inline_denorm`` holds for both forms (the shared affine shift ``m``
            # cancels in ``|pred − target|`` just as in ``(pred − target)²``), and the
            # composite auto-weighting noise model (``ms_model.auto_weighting_noise_model``)
            # is orthogonal to the per-term residual form. The attitude columns are
            # already sign-aligned above, so they are scored consistently under either
            # loss (L1-safe per the RLRP-736 flip-only alignment).
            if self.mae_loss:
                dh_residual = (
                    dh_pred_raw - dh_target_raw
                ).abs()  # per-feature (E x B x O)
            else:
                dh_residual = (dh_pred_raw - dh_target_raw).pow(
                    2
                )  # per-feature (E x B x O)

            # A7 (RLRP-788): gate the diagnostic ``dh_horizon_len`` write behind the
            # meta-collection kill-switch (RLRC meta-collection kill-switch `.junie` plan,
            # ``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
            if meta is not None and self._enable_meta_collection:
                meta["dh_horizon_len"] = horizon_unroll_len

        # .... Normalise the horizon + step the schedulers (shared with the mixin loss) .......
        cp_nll = self._normalize_ar_horizon(cp_nll, gamma, horizon_unroll_len)
        # RLRP-786: not during a CUDA-graph recording pass (``advance_host_step_state`` owns it).
        if not _is_recording_cuda_graph():
            self._step_ar_schedulers()

        self.wipe_ar_memory()
        return cp_nll, dh_residual

    def _stash_meta_scalar(self, key: str, value: Tensor) -> Tensor:
        """Defer the device->host transfer of a diagnostic scalar destined for ``meta``.

        Permanent diagnostic plumbing. Introduced by action ``A2`` of the RLRC MTM-Pro
        models code optimization `.junie` plan
        (``perf_RLRP-783_mtm_pro_models_code_optimization_plan_20260827.md``).
        The 0-dim tensor is kept ON DEVICE and materialised by
        :meth:`_flush_meta_scalars` in a SINGLE transfer, replacing 11 per-batch
        ``.item()`` synchronisations. Returns the stashed tensor so intra-call
        consumers (e.g. ``ms_projection_vs_horizon_ratio``) can keep computing on device.

        A7 (RLRP-788): when the meta-collection kill-switch is OFF, return the raw value
        WITHOUT the ``detach().mean()`` reduction and WITHOUT stashing, so neither the
        reduction kernel nor the (deferred) device->host transfer happens. The only
        intra-call consumers of the return value (the ``ms_projection_vs_horizon_ratio``
        read-back blocks) are themselves gated behind the same flag, so the un-reduced
        value is never observed. Introduced by action ``A7`` of the RLRC meta-collection
        kill-switch `.junie` plan
        (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
        """
        if not self._enable_meta_collection:
            return value
        value = value.detach().mean()
        self._pending_meta_scalars[key] = value
        return value

    def _flush_meta_scalars(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Materialise every stashed diagnostic scalar with ONE device->host transfer.

        Permanent diagnostic plumbing. Introduced by action ``A2`` of the RLRC MTM-Pro
        models code optimization `.junie` plan
        (``perf_RLRP-783_mtm_pro_models_code_optimization_plan_20260827.md``).
        ``meta`` keeps its ``Dict[str, float]`` contract: the values and keys are
        IDENTICAL to the pre-change behaviour (bit-exact -- each 0-dim tensor is widened
        to float64 before the stack, which is a lossless widening of the float32 value the
        legacy per-scalar ``.item()`` returned), only the number of transfers changes
        (11 -> 1). Must be called BEFORE any in-call consumer that reads these keys back
        (the ``*_composite_share`` monitor and ``_compose_feature_geometry``).
        """
        pending = self._pending_meta_scalars
        if not pending:
            return meta
        keys = list(pending)
        # Widen to float64 so heterogeneous-dtype stashes (e.g. an energy beta) stack
        # cleanly; widening is exact, so ``.tolist()`` yields the same Python floats the
        # legacy per-scalar ``.item()`` calls produced.
        values = torch.stack([pending[k].double() for k in keys]).cpu().tolist()
        meta.update(dict(zip(keys, values)))
        pending.clear()
        return meta

    @torch.compiler.disable
    def _probabilistic_loss(
        self,
        model_in: Tensor,
        target: Tensor,
        reduce: bool = True,
        target_raw_HOxDoa: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Single head handles both MS and SS predictions.
        Optional double backward pass: one for ms loss, one for the ss loss.

        losses = ss_nll_losses + ms_nll_losses

        RLRP-731 batch ``B4-raw`` (Option 1, ``history_drift_target_mode='raw_passthrough'``):
        ``target_raw_HOxDoa`` is the obs-block RAW (non-normalized) target threaded from
        ``OneDTransitionRewardModelV2._process_batch`` (obs-block only, no reward). It is ``None``
        for every other path (``inline_denorm`` / OFF / non-family callers), so behaviour is
        byte-identical when the raw passthrough ablation is not selected.
        """
        # (CRITICAL) ToDo: assess single term loss ↓↓ (ref task TASK)

        meta = {}

        # A2 (RLRP-783): defensively drop any diagnostic scalar stashed by a prior aborted
        # loss call so a mid-loss exception cannot leak stale on-device tensors into this call.
        self._pending_meta_scalars.clear()

        # RLRP-751 (task T5): defensively reset the transient per-feature geometry
        # point handoffs at loss ENTRY so a partially-populated state from a prior
        # aborted forward (e.g. an exception raised mid-loss, before the composite
        # stage cleared them) can never leak stale points into this call. These are
        # a purely within-call handoff (populated below, consumed + cleared in the
        # composite stage under ``try/finally``).
        self._ss_feature_point = None
        self._ms_feature_head = None
        self._cp_feature_points = []

        # .... Deploy-path training stabilization (RLRP-722) ......................................
        # ``_probabilistic_loss`` is @torch.compiler.disable and runs exactly once per batch (the
        # single composite host entry; the CP schedulers also step here once per batch), so it is
        # the deterministic place to advance the per-batch training-step counter that drives the
        # forecaster-warmup gate and the EMA momentum schedule.
        # RLRP-786: not during a CUDA-graph recording pass (see ``_is_recording_cuda_graph`` /
        # ``advance_host_step_state``).
        _recording = _is_recording_cuda_graph()
        if not _recording:
            self._train_step_count += 1
        # Forecaster warmup: while active, suppress ONLY the deploy-path terms (SS / CP /
        # projection / RC); the MS forecast term is always left on so the body keeps learning.
        # Per-batch gate (does NOT mutate the persistent ``self.enable_*`` attributes).
        _warmup_active = self._deploy_warmup_enable and (
            self._train_step_count <= self._deploy_warmup_steps
        )
        # RLRP-751 (task T4/T5): the per-feature geometry SS / CP channels are
        # DEPLOY-PATH terms (scored on the SS / CP k=0 deploy head). During
        # forecaster warmup they are suppressed simply by the inline call-site
        # gate (``not _warmup_active`` on the SS channel; CP is inactive under
        # warmup), so NO explicit suppression signal is needed — the old
        # stash/drain fail-loud guard (and its ``_signal_feature_loss_suppressed``
        # hook) was removed with the stash seam.
        # Optional linear ramp of the deploy terms' static weight after warmup (avoids a magnitude
        # discontinuity). ``ramp_steps == 0`` -> hard switch (ramp factor == 1.0).
        if (
            self._deploy_warmup_enable
            and not _warmup_active
            and self._deploy_warmup_ramp_steps > 0
        ):
            _deploy_ramp = min(
                1.0,
                (self._train_step_count - self._deploy_warmup_steps)
                / float(self._deploy_warmup_ramp_steps),
            )
        else:
            _deploy_ramp = 1.0

        model_in, target = self._setup_loss_input(model_in, target)

        # .... Strip legacy target shape to horizon length ........................................
        # To (E x B x Hi x O+A)
        target = timestep_first_multistep_dim_unflaten_array(
            target,
            self.singlestep_obs_len,
            self.singlestep_act_len,
            sequence_len=self.history_len,
            enable_last_action_padding=True,
        )
        # To (E x B x Ho x O+A)
        target_HOxDoa = target[..., -self.horizon_len :, :]
        ss_target = target_HOxDoa[..., 0, : self.singlestep_obs_len]
        # To (..., O[1:Do]_1 + ... + O[1:Do]_Ho + A[1:Da]_1 + ... + A[1:Da]_Ho-1)
        target = revert_timestep_first_multistep_dim_unflaten_array(
            target_HOxDoa,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            remove_last_action_padding=True,
        )

        # .... RLRP-731 batch ``B4-raw``: resolve the threaded RAW obs target (Option 1) .........
        # ``target_raw_HOxDoa`` arrives from ``OneDTransitionRewardModelV2._process_batch`` in the
        # SAME composed obs layout as ``target`` (obs block only, no reward). Mirror the exact
        # ensemble expansion (``_setup_loss_input``) + unflatten + horizon slice applied to
        # ``target`` above so it lands in the (E x B x Ho x O+A) layout that
        # ``_compounded_deploy_nll`` slices with ``[..., F - 1, : singlestep_obs_len]``. ``None``
        # (inline_denorm / OFF path) is threaded through untouched.
        #
        # The raw target is normally delivered via the ``loss`` / ``update`` override STASH (so the
        # inherited ``super().loss`` chain — CP-iterator delegation + domain-randomization top
        # layer — keeps running unchanged); it can also be passed directly as a kwarg (used by the
        # test-suite direct-model-call path). The kwarg wins; otherwise fall back to the stash.
        if target_raw_HOxDoa is None:
            target_raw_HOxDoa = getattr(self, "_dh_pending_target_raw_HOxDoa", None)
        if target_raw_HOxDoa is not None:
            if (
                target_raw_HOxDoa.ndim == 2
            ):  # add ensemble dim (mirror ``_setup_loss_input``)
                target_raw_HOxDoa = target_raw_HOxDoa.unsqueeze(0)
            if target_raw_HOxDoa.shape[0] != self.num_members:
                target_raw_HOxDoa = target_raw_HOxDoa.repeat(self.num_members, 1, 1)
            target_raw_HOxDoa = timestep_first_multistep_dim_unflaten_array(
                target_raw_HOxDoa,
                self.singlestep_obs_len,
                self.singlestep_act_len,
                sequence_len=self.history_len,
                enable_last_action_padding=True,
            )
            target_raw_HOxDoa = target_raw_HOxDoa[..., -self.horizon_len :, :]

        # .... Accumulate gradient ................................................................
        E, B, MS, Doa = target_HOxDoa.shape
        if self.enable_pre_mixture_u_loss:
            self._pre_mixture_loss_accumulator = torch.zeros(
                (E, B, Doa), device=model_in.device, dtype=self.model_dtype
            )  # Reset accumulator
            self._pre_mixture_loss_target = target_HOxDoa

        # Post-mixing (MIX) and true-mixture (Stage G) per-step accumulators are filled in the
        # forward AR loop (the per-step temporal mixture is not retained afterwards).
        _need_post_mix = (
            self.enable_post_mixture_loss
            or self.ms_probabilistic_loss_mode == "true_mixture"
        )
        if _need_post_mix:
            self._post_mixture_loss_target = target_HOxDoa
            if self.enable_post_mixture_loss:
                self._post_mixture_loss_accumulator = torch.zeros(
                    (E, B), device=model_in.device, dtype=self.model_dtype
                )
            if self.ms_probabilistic_loss_mode == "true_mixture":
                # Per-horizon-step list of per-feature (obs+action) NLLs (filled in the forward
                # AR loop; the action signal is retained per RLRP-530).
                self._ms_true_mixture_loss_steps = []

        # .... Forecast-path teacher forcing (RLRP-726): stash GT target + freeze schedule ........
        # Stash the GT target so `state_history_update` can splice obs+act on "teacher" steps. The
        # probability is read/frozen here and the dedicated scheduler is NOT stepped inside the
        # forward, so every RC re-forward and the eval forward see a stable schedule. Train-only
        # (the splice is additionally guarded by `self.training`); cleared at the loss tail and
        # nulled during the RC re-forwards.
        if self.training:
            self._forecast_tf_target = target_HOxDoa
        # RLRP-786 FR6: on the capture-ready path the F per-step coins are drawn HERE (eager),
        # once, into the static device mask consumed by ``state_history_update`` -- same
        # scheduler calls in the same order as the Python branch, so the CPU RNG stream is
        # unchanged. Skipped during a graph recording pass (the ``before_replay`` hook owns it).
        if (
            self.cuda_graph_capture_ready
            and self._CUDA_GRAPH_CAPTURABLE_VARIANT
            and not _recording
        ):
            self.refill_forecast_tf_mask()
        # A7 (RLRP-788): pure diagnostic write -> gated behind the meta-collection
        # kill-switch (the ``_forecast_tf_target`` stash above is training state and stays
        # unconditional). RLRC meta-collection kill-switch `.junie` plan
        # (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
        if self._enable_meta_collection:
            meta["forecast_teacher_forcing_prob"] = (
                self._forecast_teacher_forcing_scheduler.current_probability
                if self._forecast_teacher_forcing_scheduler is not None
                else 0.0
            )

        ms_pred_mean, ms_pred_logvar = self.forward(model_in, use_propagation=False)

        # RLRP-751 (task T5): retain the MS (multi-step forecast) head point
        # estimate + composed target so the composite stage composes the
        # ``FEAT_GEOM_MS`` term ONCE, INLINE. ``ms_pred_mean`` is the forecast
        # POINT estimate (mixture / forecast mean, NEVER a sample) in the composed
        # multi-step next-obs layout; MTM-Pro emits its forecast in the FUTURE
        # composed layout — a horizon-length window (``ho_out_size = obs*Ho +
        # act*(Ho-1)``, last-action padding removed) — NOT the legacy
        # history-length window, so ``_compose_feature_geometry`` is invoked with
        # ``ms_head_legacy_composed_shape=False``. No-op / bit-neutral unless the
        # term is active AND ``horizon_len > 1``.
        self._ms_feature_head = (
            (ms_pred_mean, target) if self._feature_loss_active() else None
        )

        # .... Advance the forecast teacher-forcing scheduler exactly once per loss (RLRP-726) .....
        # Separate call from the CP `_step_ar_schedulers()` (stepped inside `_compounded_deploy_nll`)
        # so neither instance is double-stepped. Stepped AFTER the forward so the frozen probability
        # emitted above matches the per-step coins used during the unroll (mirrors the CP cadence).
        if self._forecast_teacher_forcing_scheduler is not None and not _recording:
            self._forecast_teacher_forcing_scheduler.step()

        # .... Horizon Nll Loss (Ms Head) .........................................................
        # Legacy moment-matched (M-projection) primary loss: scores the data under the single
        # Gaussian/Laplace re-built from the mixture moments. Only computed when selected; the
        # `true_mixture` mode (default) instead uses the exact per-step mixture NLL accumulated in
        # the forward loop (see `_accumulate_post_mix_losses`).
        if self.ms_probabilistic_loss_mode == "moment_matched":
            # (CRITICAL) ToDo: assess >> mixture weight are already computed (ref task RLRP-530)
            ms_distribution = self._to_distribution(ms_pred_mean, ms_pred_logvar)
            ms_nll_losses = -ms_distribution.log_prob(target)

            # .... logvar bound penalty: now added ONCE on the composite (RLRP-718) ...............
            ho_out_size = compute_multistep_model_out_size(
                self.singlestep_obs_len, self.singlestep_act_len, self.horizon_len
            )

            # .... Apply temporal weight ..........................................................
            ms_nll_losses = self.apply_next_obs_temporal_discount_factor_weights(
                ms_nll_losses,
                log_space=True,
                pre_process_weight=lambda x: x[..., -ho_out_size:],
                temporal_mode_aware=True,
            )

            # .... Apply feature weight ...........................................................
            ms_nll_losses = self.apply_next_obs_feature_weights(
                ms_nll_losses,
                log_space=True,
                apply_minus_log=False,
                pre_process_weight=lambda x: x[..., -ho_out_size:],
            )

            # .... Reduce multistep horizon .......................................................
            if self.horizon_len > 1:
                ms_nll_losses = timestep_first_multistep_dim_unflaten_array(
                    ms_nll_losses,
                    self.singlestep_obs_len,
                    self.singlestep_act_len,
                    sequence_len=self.horizon_len,
                    enable_last_action_padding=True,
                ).swapaxes(-2, -1)
                ms_nll_losses = self.reduce_multistep_losses_horizon(
                    ms_nll_losses,
                    probabilistic_losses=True,
                    unflaten_composed_array_enabled=False,
                    enable_feature_energy_axis=True,
                )
        elif self.ms_probabilistic_loss_mode == "true_mixture":
            # Per-step per-feature true-mixture NLL -> (E x B x MS x O+A). The forecast is obs+act
            # at every step except the last (obs-only); embed it in the composed horizon layout and
            # let `remove_last_action_padding=True` drop the last step's (padding) action slots, so
            # we reuse the EXACT `moment_matched` weighting pipeline (logvar bound, temporal
            # discount, feature weights, horizon reduction) over the full obs+act forecast signal.
            tm_steps = torch.stack(
                self._ms_true_mixture_loss_steps, dim=-2
            )  # (E x B x MS x O+A)
            # (E x B x MS x O+A) -> composed flat (E x B x ho_out_size)
            ms_nll_losses = revert_timestep_first_multistep_dim_unflaten_array(
                tm_steps,
                singlestep_obs_len=self.singlestep_obs_len,
                singlestep_act_len=self.singlestep_act_len,
                remove_last_action_padding=True,
            )

            ho_out_size = compute_multistep_model_out_size(
                self.singlestep_obs_len, self.singlestep_act_len, self.horizon_len
            )

            # .... logvar bound penalty: now added ONCE on the composite (RLRP-718) ...............

            # .... Apply temporal weight ..........................................................
            ms_nll_losses = self.apply_next_obs_temporal_discount_factor_weights(
                ms_nll_losses,
                log_space=True,
                pre_process_weight=lambda x: x[..., -ho_out_size:],
                temporal_mode_aware=True,
            )

            # .... Apply feature weight ...........................................................
            ms_nll_losses = self.apply_next_obs_feature_weights(
                ms_nll_losses,
                log_space=True,
                apply_minus_log=False,
                pre_process_weight=lambda x: x[..., -ho_out_size:],
            )

            # .... Reduce multistep horizon .......................................................
            if self.horizon_len > 1:
                ms_nll_losses = timestep_first_multistep_dim_unflaten_array(
                    ms_nll_losses,
                    self.singlestep_obs_len,
                    self.singlestep_act_len,
                    sequence_len=self.horizon_len,
                    enable_last_action_padding=True,
                ).swapaxes(-2, -1)
                ms_nll_losses = self.reduce_multistep_losses_horizon(
                    ms_nll_losses,
                    probabilistic_losses=True,
                    unflaten_composed_array_enabled=False,
                    enable_feature_energy_axis=True,
                )

        # .... Compounded-prediction deploy term (CP) gate (RLRP-708, Option C2) ..................
        # CP is gated by the second-AR-stage toggles + its own enable flag, INDEPENDENT of the SS
        # term. When active it SUBSUMES the step-0 SS term (CP's k=0 term == the SS NLL), so the
        # plain SS block below is skipped to avoid double-counting.
        #
        # IMPORTANT (TensorBoard visibility): because the SS block is skipped when `_cp_active`,
        # the SS diagnostics are INTENTIONALLY hidden when both `single_step_loss.enable` and
        # `compounded_prediction_deploy_loss.enable` are true. Specifically, `meta["singlestep_loss"]`
        # (the `Loss/train single-step loss` scalar) and the SS auto-weighting (`"SS"` slot ->
        # `Loss/train SS loss auto weighting`) are NOT emitted in this mode. This is expected: the
        # single-step deploy signal is fully carried by the CP term's k=0 unroll, so it is published
        # under the "CP" term (`ms_compounded_prediction_deploy_loss` / `Loss/train CP ...`) instead.
        # To get the standalone SS TensorBoard cards back, disable CP (then the SS term is a
        # first-class composite term again). See ask/answer (Option C, 2026-06-29).
        _cp_active = (
            self.enable_compounded_prediction_deploy_loss
            and self.ar_enabled
            and self.ar_train_horizon_unroll
            # RLRP-722 forecaster warmup: a deploy-path term, suppressed while warmup is active.
            and not _warmup_active
        )

        # .... Horizon at t=1 NLL loss (SS head)...................................................
        if (
            self.multitask_loss_singlestep_term_enable
            and not _cp_active
            and not _warmup_active  # RLRP-722 forecaster warmup gate (deploy-path term)
        ):
            if self.ms_probabilistic_loss_mode == "moment_matched":
                # Get the MS distribution mixture mean and log variance. RLRP-722: when the EMA
                # forecast clone feeds ``ss``, the deploy-body forward runs against the stable
                # teacher weights (no grad into the live body; only the deploy head trains).
                with self._ema_term_ctx("ss"):
                    ss_pred_mean, ss_pred_logvar = self.deploy(
                        model_in, only_elite=False
                    )

                ss_distribution = self._to_distribution(ss_pred_mean, ss_pred_logvar)
                ss_nll_losses = -ss_distribution.log_prob(ss_target)

            elif self.ms_probabilistic_loss_mode == "true_mixture":
                # (RLRP-711/RLRP-718 fix) Score the SS term as the PER-FEATURE true-mixture NLL
                # (mirroring the MS primary path via `_true_mixture_per_feature_nll`), NOT the JOINT
                # `MixtureSameFamily.log_prob`. The joint `log_prob` collapses the whole O+A event
                # into a single (E x B x 1) scalar, which (a) folds the UNPREDICTABLE action channels
                # (command/dt) into the SS NLL -> an irreducible ~constant floor that never trains
                # (observed: SS loss frozen ~16 while the MS term descends), and (b) makes the
                # downstream obs-only slice + obs-feature-weighting a no-op because the feature axis
                # was already gone. Keeping per-feature granularity (E x B x O+A) lets us restrict
                # to the obs dims and apply the obs feature weights exactly like the MS /
                # `moment_matched` paths.
                # RLRP-722: route the deploy-body forward through the EMA teacher when ``ss`` is
                # fed by the clone (``_latest_deploy_ss_dist_mixture`` is then built on it).
                with self._ema_term_ctx("ss"):
                    _, _ = self.deploy(model_in, only_elite=False)
                # (E x B x O+A) step-0 target (NOT unsqueezed): the per-feature mixture batches over
                # the feature axis, so its `log_prob` consumes the (E x B x O+A) target directly and
                # returns the per-feature NLL (E x B x O+A) — same shape contract as the
                # `moment_matched` branch (so the shared obs-feature-weight + `mean(2)` reduction and
                # the composite auto-weighting receive the expected 3D tensor).
                ss_obs_act_target = target_HOxDoa[..., 0, :]
                ss_nll_losses = self._true_mixture_per_feature_nll(
                    self._latest_deploy_ss_dist_mixture, ss_obs_act_target
                )
                # Extract obs-only dimensions to match eval_score/multistep_to_singlestep_next_obs_adapter
                ss_nll_losses = ss_nll_losses[..., : self.singlestep_obs_len]

            # .... logvar bound penalty: now added ONCE on the composite (RLRP-718) ...............

            # .... Apply feature weight ...........................................................
            ss_nll_losses = self.apply_next_obs_feature_weights(
                ss_nll_losses,
                log_space=True,
                pre_process_weight=lambda x: self.multistep_to_singlestep_next_obs_adapter(
                    x
                ),  # Required because obs feature weigths shape follow the HO=HI legacy setup
            )

            # RLRP-751 (task T5): retain the SS deploy point estimate + step-0
            # target so the composite stage composes the ``FEAT_GEOM_SS`` term ONCE,
            # INLINE. Point estimate = the mixture mean (deterministic decode, NOT a
            # sample): ``ss_pred_mean`` in ``moment_matched`` mode, else the
            # ``_latest_deploy_ss_dist_mixture`` mean. No-op / bit-neutral unless a
            # handler + non-zero weight is registered (``_feature_loss_active``).
            if self._feature_loss_active():
                if self.ms_probabilistic_loss_mode == "moment_matched":
                    _feat_ss_point = ss_pred_mean[..., : self.singlestep_obs_len]
                elif self._latest_deploy_ss_dist_mixture is not None:
                    # RLRP-736 §4A-B P3.2: canonicalise the attitude slot(s) of the
                    # stashed deploy mean so the feature-geometry point estimate is a
                    # valid rotation (no-op OFF).
                    _feat_ss_point = self._canonicalize_exposed_mean_quaternion(
                        self._latest_deploy_ss_dist_mixture.mean
                    )[..., : self.singlestep_obs_len]
                else:
                    _feat_ss_point = None
                if _feat_ss_point is not None:
                    self._ss_feature_point = (_feat_ss_point, ss_target)

        # ==== Composite loss =====================================================================
        # Reduce over feature dim
        # Sum of log over feature dim imply they are independent random variables
        # e.g., from a multi-variate distribution
        if (
            self.multitask_loss_singlestep_term_enable
            and not _cp_active
            and not _warmup_active  # RLRP-722 forecaster warmup gate (deploy-path term)
        ):
            ss_nll_losses = ss_nll_losses.mean(2, keepdim=True)
            # Static SS weight, applied in BOTH auto-weighting modes (consistent with the other
            # composite terms). The SS term is routed through the composite auto-weighting only
            # when its dedicated `single_step_auto_weighting` toggle is on; otherwise it enters
            # the composite sum at this static weight.
            ss_nll_losses = (
                _deploy_ramp * self.ss_composite_loss_weight
            ) * ss_nll_losses
            self._stash_meta_scalar("singlestep_loss", ss_nll_losses)
            if self.single_step_auto_weighting:
                ss_nll_losses, meta = self.composite_loss_automatic_weighting(
                    ss_nll_losses, "SS", meta, are_log_prob_losses=True
                )

        # .... Compounded-prediction deploy term (CP, RLRP-708 Option C2) .........................
        # Standalone deploy-path term: the U-step free-running deploy NLL. Reuses the EXACT SS
        # feature-weight / `mean(2)` / static-weight / auto-weighting pipeline so it slots into the
        # composite like the other terms. The MTM-Pro multistep forecast term is untouched.
        if _cp_active:
            # RLRP-722: when the exponential moving average (EMA) clone feeds ``cp``,
            # the free-running deploy unroll forwards the body against the stable teacher weights
            # (detached) instead of the live body. RLRP-731: the DH sub-term is scored INSIDE the
            # same unroll (one unroll, two objectives), so it shares the CP EMA-teacher context
            # (no separate "dh" EMA slot), the CP sampled horizon and the CP TF schedule.
            with self._ema_term_ctx("cp"):
                cp_nll_losses, dh_residual = self._compounded_deploy_nll(
                    model_in,
                    target_HOxDoa,
                    meta=meta,
                    target_raw_HOxDoa=target_raw_HOxDoa,
                )
            cp_nll_losses = self.apply_next_obs_feature_weights(
                cp_nll_losses,
                log_space=True,
                pre_process_weight=lambda x: self.multistep_to_singlestep_next_obs_adapter(
                    x
                ),
            )
            cp_nll_losses = cp_nll_losses.mean(2, keepdim=True)
            cp_nll_losses = (
                _deploy_ramp * self.compounded_prediction_deploy_loss_weight
            ) * cp_nll_losses
            self._stash_meta_scalar(
                "ms_compounded_prediction_deploy_loss", cp_nll_losses
            )
            if self.compounded_prediction_auto_weighting:
                cp_nll_losses, meta = self.composite_loss_automatic_weighting(
                    cp_nll_losses, "CP", meta, are_log_prob_losses=True
                )
        else:
            cp_nll_losses = 0.0
            dh_residual = None

        # .... Deploy-history drift residual sub-term (DH, RLRP-731) ..............................
        # Terminal-step raw-space residual MSE on the CP unroll trajectory (returned by
        # ``_compounded_deploy_nll`` above). Single gate: active iff ``_cp_active`` AND
        # ``enable_history_drift_loss`` (no DH-only mode; plan §5.3). Reuses the exact SS/CP
        # feature-weight / ``mean(2)`` / ramp*static-weight pipeline; being an MSE (not an NLL),
        # the feature weights apply LINEARLY (``log_space=False``) and the auto-weighting routes
        # through the non-log-prob path (``are_log_prob_losses=False``, mirroring IPROJ/RC).
        if dh_residual is not None:
            dh_losses = self.apply_next_obs_feature_weights(
                dh_residual,
                log_space=False,
                pre_process_weight=lambda x: self.multistep_to_singlestep_next_obs_adapter(
                    x
                ),
            )
            dh_losses = dh_losses.mean(2, keepdim=True)
            dh_losses = (_deploy_ramp * self.history_drift_loss_weight) * dh_losses
            self._stash_meta_scalar("ms_deploy_history_drift_loss", dh_losses)
            if self.history_drift_auto_weighting:
                dh_losses, meta = self.composite_loss_automatic_weighting(
                    dh_losses, "DH", meta, are_log_prob_losses=False
                )
        else:
            dh_losses = 0.0

        # .... Primary multi-step term: true-mixture (default) or moment-matched (legacy) .........
        if self.ms_probabilistic_loss_mode == "moment_matched":
            ms_nll_losses = ms_nll_losses.mean(2, keepdim=True)
        elif self.ms_probabilistic_loss_mode == "true_mixture":
            # Reduce over the (already horizon-reduced) feature dim, identical to the
            # `moment_matched` path: the obs+act forecast signal is retained (the last step's
            # padding action slots were already dropped via `remove_last_action_padding=True`).
            ms_nll_losses = ms_nll_losses.mean(2, keepdim=True)

        ms_nll_losses = self.ms_composite_loss_weight * ms_nll_losses
        # NOTE: `horizon_loss` is the POST-`ms_composite_loss_weight` primary term; the projection
        # `ms_projection_vs_horizon_ratio` diagnostic is therefore read against this weighted value.
        # A2 (RLRP-783): keep the stashed 0-dim tensor so the projection ratio below can be
        # computed on device without forcing a device->host sync.
        _horizon_loss_t = self._stash_meta_scalar("horizon_loss", ms_nll_losses)

        # Track the inverse-temperature beta of the 'energy-density' and 'soft-max-energy' horizon
        # reduction so its evolution can be monitored in tensorboard. Recorded for BOTH the
        # learnable case (anneals mean -> max) AND the fixed-float case (constant trace), so the
        # chart is always populated when the 'energy-density' and 'soft-max-energy' reduction is
        # active (the other reductions do not use beta).
        if self.ms_probabilities_reduction in ("energy-density", "soft-max-energy"):
            self._stash_meta_scalar("ms_energy_beta", self._resolve_ms_energy_beta())

        if self.enable_pre_mixture_u_loss:
            # Note: U-losses are already reduced on the MS axis.
            # Out-of-place division (Stage A.2): avoid mutating the accumulator via an alias.
            ms_u_loss = self._pre_mixture_loss_accumulator / self.horizon_len
            assert ms_u_loss.requires_grad, "Pre-mixture loss should require gradients"
            ms_u_loss = ms_u_loss.mean(2, keepdim=True)
            self._stash_meta_scalar("ms_pre_mixture_u_loss", ms_u_loss)
            ms_u_loss, meta = self.composite_loss_automatic_weighting(
                ms_u_loss, "U", meta, are_log_prob_losses=True
            )
        else:
            ms_u_loss = 0.0

        # .... Post-mixture (MIX) term with selectable backprop scope (Stage B) ...................
        if self.enable_post_mixture_loss:
            # Static config weight applied in BOTH auto-weighting modes (integration fix 2b).
            mix_loss = self.post_mixture_loss_weight * (
                self._post_mixture_loss_accumulator / self.horizon_len
            ).unsqueeze(-1)
            self._stash_meta_scalar("ms_post_mixture_loss", mix_loss)
            # MIX is a (post-mixing) NLL, so the log-prob/softplus stabilisation applies.
            mix_loss, meta = self.composite_loss_automatic_weighting(
                mix_loss, "MS_MIX", meta, are_log_prob_losses=True
            )
        else:
            mix_loss = 0.0

        # .... RLRP-722 EMA teacher forecast for the projection / RC consumers ....................
        # When the EMA clone feeds ``projection`` or ``rc``, those terms must read a STABLE forecast
        # (slowly-varying teacher) instead of the live, fast-moving body. Re-run the forecast
        # forward ONCE under the detached EMA body+mixer weights to obtain a teacher
        # ``(mean, logvar)`` + the per-step obs-only teacher mixtures (built while the mixer is
        # ALSO swapped, so the whole teacher mixture is consistent + detached). This runs AFTER the
        # MS / U / MIX accumulators above have been consumed, so transiently overwriting the
        # forward accumulators is harmless; the live accumulators are snapshotted and restored for
        # any live (non-fed) consumer. The forward graph is constant w.r.t. the body (no grad leaks
        # into it); only the dedicated projection head (applied downstream) still trains.
        _proj_feed_ema = (
            self._ema_forecast_enable
            and not _warmup_active
            and "projection" in self._ema_feed_to
        )
        _rc_feed_ema = (
            self._ema_forecast_enable
            and not _warmup_active
            and "rc" in self._ema_feed_to
        )
        _ema_proj_mean = _ema_proj_logvar = None
        _ema_proj_mixtures = None
        _ema_rc_mixtures = None
        if (
            _proj_feed_ema
            and (
                self.enable_info_projection_loss
                or self.enable_siw_mp_loss
                or self.enable_gms_iwae_loss
            )
        ) or (_rc_feed_ema and self.enable_rollout_consistency_loss):
            _live_mean_acc = self.forecast_mean_accumulator
            _live_logvar_acc = self.forecast_logvar_accumulator
            with self._ema_forecast_weights():
                _ema_proj_mean, _ema_proj_logvar = self.forward(
                    model_in, use_propagation=False
                )
                if _proj_feed_ema and (
                    self.enable_info_projection_loss
                    or self.enable_siw_mp_loss
                    or self.enable_gms_iwae_loss
                ):
                    _ema_proj_mixtures = self._build_obs_only_temporal_mixtures(
                        self._projection_detach_components,
                        self._projection_detach_mixing_weights,
                    )
                if _rc_feed_ema and self.enable_rollout_consistency_loss:
                    _ema_rc_mixtures = self._build_obs_only_temporal_mixtures(
                        self._rollout_consistency_detach_components,
                        self._rollout_consistency_detach_mixing_weights,
                    )
            # Restore the live forward accumulators for any live (non-fed) consumer + safety.
            self.forecast_mean_accumulator = _live_mean_acc
            self.forecast_logvar_accumulator = _live_logvar_acc

        # Forecast (mean, logvar) the projection terms consume: the EMA teacher when fed, else the
        # live body forecast (byte-for-byte unchanged when the clone is OFF / not feeding it).
        _proj_mean = _ema_proj_mean if _proj_feed_ema else ms_pred_mean
        _proj_logvar = _ema_proj_logvar if _proj_feed_ema else ms_pred_logvar

        # .... Shared per-step obs-only temporal mixtures for IPROJ + SIWAE (integration fix 2d) ..
        # Built once per loss call (honouring projection_backprop_mode) and reused by both terms
        # so the per-step MixtureSameFamily builds are not duplicated. When fed by the EMA clone,
        # reuse the teacher mixtures built above (under the EMA mixer weights).
        _proj_mixtures = None
        if not _warmup_active and (
            self.enable_info_projection_loss
            or self.enable_siw_mp_loss
            or self.enable_gms_iwae_loss
        ):
            _proj_mixtures = (
                _ema_proj_mixtures
                if _proj_feed_ema
                else self._build_obs_only_temporal_mixtures(
                    self._projection_detach_components,
                    self._projection_detach_mixing_weights,
                )
            )

        # .... Information-projection (IPROJ) term, dedicated head (Stage H) ......................
        if self.enable_info_projection_loss and not _warmup_active:
            iproj_loss, (q_mean, q_logvar) = self._info_projection_loss(
                _proj_mean, _proj_logvar, mixtures=_proj_mixtures
            )
            if self.info_projection_head_data_nll:
                # Optional anchor: obs-only data NLL of the summary head q_theta. Use the SAME
                # Independent (sum-over-obs) reduction as the KL term's q_theta so both head
                # usages are on a consistent scale (integration fix 2e).
                q_anchor_dist = dist.Independent(
                    self._to_distribution(
                        q_mean, q_logvar, scale_are_log_variance=True
                    ),
                    1,
                )
                q_nll = (-q_anchor_dist.log_prob(ss_target)).unsqueeze(-1)
                iproj_loss = iproj_loss + q_nll
            # Static config weight applied in BOTH auto-weighting modes (integration fix 2b).
            iproj_loss = (_deploy_ramp * self.info_projection_loss_weight) * iproj_loss
            self._stash_meta_scalar("ms_info_projection_loss", iproj_loss)
            # IPROJ is a (clamped) KL divergence, not an NLL -> use the plain linear weighting
            # path rather than the NLL softplus stabilisation (integration fix 2c). Gated by the
            # SHARED projection auto-weighting toggle; when off, IPROJ enters at its static weight.
            if self.projection_auto_weighting:
                iproj_loss, meta = self.composite_loss_automatic_weighting(
                    iproj_loss, "IPROJ", meta, are_log_prob_losses=False
                )
        else:
            iproj_loss = 0.0

        # .... SIWAE / stratified IWAE bound term (Stage I) .......................................
        if self.enable_siw_mp_loss and not _warmup_active:
            siw_mp_loss = self._siw_mp_loss(
                _proj_mean, _proj_logvar, state_forecast=_proj_mixtures
            )
            # Static config weight applied in BOTH auto-weighting modes (integration fix 2b).
            siw_mp_loss = (_deploy_ramp * self.siw_mp_loss_weight) * siw_mp_loss
            # A7 (RLRP-788): block-guard the whole SIW-MP diagnostic read-back region so
            # the on-device ratio computation (and its feeding stashes) is skipped wholesale
            # when the meta-collection kill-switch is OFF -- leaving it active with the
            # stashes suppressed would compute a ratio against an un-reduced ``_horizon_loss_t``.
            # RLRC meta-collection kill-switch `.junie` plan
            # (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
            if self._enable_meta_collection:
                _siw_mp_loss_t = self._stash_meta_scalar("ms_siw_mp_loss", siw_mp_loss)
                # Relative-magnitude log for calibration vs the primary horizon loss (Stage C.2).
                # Shared across the (mutually exclusive) projection objectives, so a single key
                # `ms_projection_vs_horizon_ratio` covers whichever one is active. A2 (RLRP-783):
                # computed ON DEVICE from the stashed 0-dim tensors, in float64 so it matches the
                # legacy Python-float arithmetic bit-for-bit, and stashed itself (no read-back sync).
                _horizon_mag = _horizon_loss_t.double().abs() + 1e-8
                # Note: This ratio is a unitless relative-magnitude indicator. It lets you watch,
                # how much weight the projection term is effectively pulling in the composite.
                self._stash_meta_scalar(
                    "ms_projection_vs_horizon_ratio",
                    _siw_mp_loss_t.double().abs() / _horizon_mag,
                )

            # Optional auto-weighting (Stage C.2). SIW-MP is a negated ELBO bound (can be negative)
            # -> plain linear weighting, not the NLL softplus path (integration fix 2c). When off,
            # SIW-MP enters the sum at its static weight (regularisation-term default).
            if self.projection_auto_weighting:
                siw_mp_loss, meta = self.composite_loss_automatic_weighting(
                    siw_mp_loss, "MS_SIW_MP", meta, are_log_prob_losses=False
                )
        else:
            siw_mp_loss = 0.0

        # .... Generative Mode-Seeking IWAE (GMS-IWAE) term .......................................
        if self.enable_gms_iwae_loss and not _warmup_active:
            gms_iwae_loss = self._gms_iwae_loss(
                _proj_mean, _proj_logvar, target_HOxDoa, mixtures=_proj_mixtures
            )
            # Static config weight applied in BOTH auto-weighting modes (integration fix 2b).
            gms_iwae_loss = (_deploy_ramp * self.gms_iwae_loss_weight) * gms_iwae_loss
            # A7 (RLRP-788): block-guard the whole GMS-IWAE diagnostic read-back region
            # (same rationale as the SIW-MP block above) so it is skipped wholesale when
            # the meta-collection kill-switch is OFF. RLRC meta-collection kill-switch
            # `.junie` plan (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
            if self._enable_meta_collection:
                _gms_iwae_loss_t = self._stash_meta_scalar(
                    "ms_gms_iwae_loss", gms_iwae_loss
                )
                # Shared projection calibration diagnostic (GMS / SIW-MP / IPROJ are mutually
                # exclusive, so the single key applies to whichever is active). A2 (RLRP-783):
                # computed ON DEVICE (float64) from the stashed tensors and stashed itself.
                _horizon_mag = _horizon_loss_t.double().abs() + 1e-8
                self._stash_meta_scalar(
                    "ms_projection_vs_horizon_ratio",
                    _gms_iwae_loss_t.double().abs() / _horizon_mag,
                )
            # GMS-IWAE is a negated IWAE bound (can be negative) -> plain linear weighting.
            if self.projection_auto_weighting:
                gms_iwae_loss, meta = self.composite_loss_automatic_weighting(
                    gms_iwae_loss, "MS_GMS_IWAE", meta, are_log_prob_losses=False
                )
        else:
            gms_iwae_loss = 0.0

        # .... Rollout self-consistency (RC) distillation term (Stage D) ..........................
        if self.enable_rollout_consistency_loss and not _warmup_active:
            if _rc_feed_ema:
                # RLRP-722: distil against the STABLE EMA teacher. The reference mixtures were
                # already built above under the EMA body+mixer; the rollout re-forward inside
                # ``_rollout_consistency_loss`` also runs against the detached teacher weights.
                _rc_mixtures = _ema_rc_mixtures
                with self._ema_forecast_weights():
                    rc_loss = self._rollout_consistency_loss(
                        model_in, target_HOxDoa, _rc_mixtures
                    )
            else:
                # Build the ORIGINAL forward's per-step obs-only mixtures (detach per RC mode)
                # BEFORE the rollout re-forwards (which reassigns the forecast accumulators; the
                # captured mixtures keep their original tensors).
                _rc_mixtures = self._build_obs_only_temporal_mixtures(
                    self._rollout_consistency_detach_components,
                    self._rollout_consistency_detach_mixing_weights,
                )
                rc_loss = self._rollout_consistency_loss(
                    model_in, target_HOxDoa, _rc_mixtures
                )
            rc_loss = (_deploy_ramp * self.rollout_consistency_loss_weight) * rc_loss
            self._stash_meta_scalar("ms_rollout_consistency_loss", rc_loss)
            # RC has its OWN dedicated auto-weighting toggle (independent from the projection
            # objectives); when off it enters the composite sum at its static weight.
            if self.rollout_consistency_auto_weighting:
                rc_loss, meta = self.composite_loss_automatic_weighting(
                    rc_loss, "MS_RC", meta, are_log_prob_losses=False
                )
        else:
            rc_loss = 0.0

        # RLRP-722 forecaster warmup: while active, the deploy-path auto-weighted terms (SS /
        # IPROJ / SIW-MP / GMS-IWAE / RC / CP) are suppressed, so they must NOT trip the "MS"
        # reshaping guard (otherwise the MS term would be reshaped against an inactive slot during
        # warmup). The non-deploy U / MIX terms are forecast-body terms and stay unaffected.
        if (
            (
                self.multitask_loss_singlestep_term_enable
                and self.single_step_auto_weighting
                and not _warmup_active
            )
            or self.enable_pre_mixture_u_loss
            or self.enable_post_mixture_loss
            or (
                self.enable_info_projection_loss
                and self.projection_auto_weighting
                and not _warmup_active
            )
            or (
                self.enable_siw_mp_loss
                and self.projection_auto_weighting
                and not _warmup_active
            )
            or (
                self.enable_rollout_consistency_loss
                and self.rollout_consistency_auto_weighting
                and not _warmup_active
            )
            or (
                self.enable_gms_iwae_loss
                and self.projection_auto_weighting
                and not _warmup_active
            )
            or (_cp_active and self.compounded_prediction_auto_weighting)
            or (
                _cp_active
                and self.enable_history_drift_loss
                and self.history_drift_auto_weighting
            )
        ):
            # NOTE: when the primary MS term is the ONLY active term (no SS / U / MIX / projection /
            # RC term), this guard is False and the "MS" auto-weight reshaping is intentionally
            # skipped: single-term uncertainty auto-weighting is a monotone constant reshaping that
            # does not change the optimum, so the raw (weighted) NLL is used directly.
            ms_nll_losses, meta = self.composite_loss_automatic_weighting(
                ms_nll_losses, "MS", meta, are_log_prob_losses=True
            )

        losses = (
            ms_nll_losses
            + ms_u_loss
            + mix_loss
            + iproj_loss
            + siw_mp_loss
            + gms_iwae_loss
            + rc_loss
            + cp_nll_losses
            + dh_losses
        )
        # SS-subsume (RLRP-708): when CP is active it carries the deploy-path NLL (its k=0 term
        # == the step-0 SS term), so the plain SS term is NOT added (it was skipped above).
        # RLRP-722 warmup also suppresses the SS term (``ss_nll_losses`` is never computed then).
        if (
            self.multitask_loss_singlestep_term_enable
            and not _cp_active
            and not _warmup_active
        ):
            losses = losses + ss_nll_losses

        # A2 (RLRP-783): single device->host flush of every stashed diagnostic scalar, placed
        # BEFORE the in-call read-back consumers (the feature-geometry composition and the
        # term-balance ``*_composite_share`` monitor below), so those keep reading real values.
        # See :meth:`_flush_meta_scalars`.
        meta = self._flush_meta_scalars(meta)

        # .... Per-feature geometry term(s) (RLRP-751, task T5) ...................................
        # Compose the SS / MS / CP feature-geometry channels INLINE into the
        # (E,B,1) composite accumulator, at the SAME reduction/auto-weighting stage
        # as every other MTM-Pro term. SS (deploy head) is composed ONLY when the
        # plain SS term is live (SUPPRESSED under CP / warmup — CP's k=0 unroll step
        # already covers the deploy signal); MS = the forecast head (FUTURE composed
        # layout, ``legacy_composed_shape=False``); CP = the free-running unroll
        # sequence accumulated in ``_compounded_deploy_nll``. STATIC scales by
        # ``loss_weight``; AUTO routes each channel through
        # ``composite_loss_automatic_weighting`` (FEAT_GEOM_SS/MS/CP slots).
        # No-op / bit-neutral unless the term is active.
        _ss_channel = (
            getattr(self, "_ss_feature_point", None)
            if (
                self.multitask_loss_singlestep_term_enable
                and not _cp_active
                and not _warmup_active
            )
            else None
        )
        try:
            losses = self._compose_feature_geometry(
                losses,
                meta,
                ss=_ss_channel,
                ms_head=getattr(self, "_ms_feature_head", None),
                ms_head_legacy_composed_shape=False,
                cp_sequence=(
                    getattr(self, "_cp_feature_points", None) if _cp_active else None
                ),
            )
        finally:
            # Always clear the transient point handoffs, even if composition
            # raises, so stale points can never leak into a subsequent loss call.
            self._ss_feature_point = None
            self._ms_feature_head = None
            self._cp_feature_points = []

        # .... Term-balance regression monitor (RLRP-720) .........................................
        # Share of each ACTIVE composite term in the composite total magnitude. Purpose: detect a
        # DWARFED term (share -> 0) or a DOMINATING term (share -> 1); the auto-weighting goal is
        # that every active term keeps a non-trivial share. Reuses the per-term magnitudes already
        # recorded in `meta` (cheap, no extra backward graph). Introduced by stage 4.6 of the
        # "improve CompositeLossAutomaticWeighting" .junie plan
        # (refactor_composite_loss_auto_weighting_and_shifted_softplus_plan_20260623.md).
        # A7 (RLRP-788): block-guard the term-balance read-back monitor. With the
        # meta-collection kill-switch OFF none of the per-term magnitude keys were
        # written, so leaving this active would compute shares over an empty dict; skip
        # it wholesale. RLRC meta-collection kill-switch `.junie` plan
        # (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
        if self._enable_meta_collection:
            _term_meta_keys = {
                "MS": "horizon_loss",
                "SS": "singlestep_loss",
                "U": "ms_pre_mixture_u_loss",
                "MIX": "ms_post_mixture_loss",
                "IPROJ": "ms_info_projection_loss",
                "SIW_MP": "ms_siw_mp_loss",
                "GMS_IWAE": "ms_gms_iwae_loss",
                "RC": "ms_rollout_consistency_loss",
                "CP": "ms_compounded_prediction_deploy_loss",
                "DH": "ms_deploy_history_drift_loss",
            }
            _active_terms = {
                term: abs(meta[k]) for term, k in _term_meta_keys.items() if k in meta
            }
            _total_mag = sum(_active_terms.values()) + 1e-8
            for term, mag in _active_terms.items():
                meta[f"{term}_composite_share"] = mag / _total_mag

        # .... Standalone fixed-coefficient logvar bound penalty (RLRP-718) .......................
        # Added once, AFTER all per-term weighting + auto-weighting, so its effective coefficient
        # is the intended fixed 0.01 (decoupled from the horizon reduction / auto-weight).
        losses = losses + self._logvar_bound_penalty()

        if reduce:
            losses = reduce_probabilistic_compose_loss(losses)

        # .... memory management ..................................................................
        # (CRITICAL) ToDo: validate grad ok
        del model_in, target
        del (
            ms_pred_mean,
            ms_pred_logvar,
        )
        if self.multitask_loss_singlestep_term_enable or _cp_active:
            del ss_target
            # del ss_pred_mean, ss_pred_logvar
            self._latest_deploy_ss_dist_mixture = (
                None  # (CRITICAL) ToDo: validate reseting here instead of in deploy
            )

        if self.enable_pre_mixture_u_loss:
            self._pre_mixture_loss_target = None
        self._post_mixture_loss_target = None
        # RLRP-726: drop the forecast teacher-forcing GT target reference (train-only side-channel).
        self._forecast_tf_target = None
        # RLRP-735: drop the per-forward frozen history token + cached ψ (rebuilt next forward by
        # setup_state_history; nulled here to free the reference, mirroring the TF target above).
        self._forecast_history_token = None
        self._forecast_history_psi = None

        # .... Deploy-path training stabilization diagnostics (RLRP-722) ..........................
        # A7 (RLRP-788): these are pure diagnostic writes -> gated behind the
        # meta-collection kill-switch (RLRC meta-collection kill-switch `.junie` plan,
        # ``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
        if self._enable_meta_collection and self._deploy_warmup_enable:
            meta["deploy_warmup_active"] = float(_warmup_active)
            meta["deploy_warmup_step"] = float(self._train_step_count)
            if not _warmup_active and self._deploy_warmup_ramp_steps > 0:
                meta["deploy_ramp"] = float(_deploy_ramp)

        # EMA frozen-decaying MS-forecast clone (3): advance the teacher ONCE per batch (after it
        # was read by the fed deploy terms above, so the teacher lags the live body by one update),
        # then publish the current momentum ``d`` for the TensorBoard card.
        # A7 (RLRP-788): ``_update_ema_forecast()`` advances TRAINING STATE (the EMA teacher)
        # and MUST run every batch -- only the diagnostic ``meta`` write is gated behind the
        # meta-collection kill-switch, never the state advance. RLRC meta-collection
        # kill-switch `.junie` plan
        # (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
        if self._ema_forecast_enable:
            _ema_forecast_momentum = float(self._update_ema_forecast())
            if self._enable_meta_collection:
                meta["ema_forecast_momentum"] = _ema_forecast_momentum

        return losses, meta
