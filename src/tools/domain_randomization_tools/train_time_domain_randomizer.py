# coding=utf-8
"""Train-time domain randomization for model inputs.

A common input-side data-augmentation / domain-randomization scheme applied during fitting
(BYOL/TD-MPC2 cited in RLRP-707 only as intuition; this is not a port of either). Domain
randomization / data augmentation at training time is a common, well-established scheme across
deep learning and (sim-to-real) RL; this module is a general, configurable instance of that
pattern tailored to the multistep dynamics models.

Permanent production module. Introduced by task RLRP-707 of the Training-Pipeline Domain
Randomization `.junie` plan (``feature_training_pipeline_domain_randomization_plan_20260610.md``).
Gated by ``cfg.<model>.train_time_domain_randomization.enable`` (default OFF, legacy-preserving).
"""
import warnings
from typing import Optional, Sequence, Tuple, Union

import omegaconf
import torch
from torch import nn

from tools.console_tools.message import consol_msg_universal_one_liner
from tools.domain_randomization_tools.noise_scale_scheduler import NoiseScaleScheduler
from tools.domain_randomization_tools.spec_containers import (
    NoiseScaleSchedulerSpec,
    TrainTimeDomainRandomizationSpec,
)
from tools.multistep_tools.utils import (
    compute_multistep_model_in_size,
    compute_multistep_model_out_size,
)
from tools.multistep_tools.multistep_model_util import (
    revert_timestep_first_multistep_dim_unflaten_array,
    timestep_first_multistep_dim_unflaten_array,
)

#: RLRP-761 S1.7 -- how ``per_feature_scale`` is interpreted. See
#: :class:`TrainTimeDomainRandomizationSpec` and
#: :meth:`TrainTimeDomainRandomizer.bind_normalizer_scale`.
_ABSOLUTE_SCALE_MODE = "absolute"
_RELATIVE_SCALE_MODE = "relative_to_normalized_std"
_PER_FEATURE_SCALE_MODES = (_ABSOLUTE_SCALE_MODE, _RELATIVE_SCALE_MODE)

_SUPPORTED_NOISE_KINDS = (
    "gaussian",
    "uniform",
    "time_correlated",
    "reverse_time_correlated",
    "time_correlated_hi_to_ho",
    "reverse_tc_in_tc_out",
)
# Kinds that apply AR(1) correlation along the multistep/time axis.
_TIME_CORRELATED_KINDS = (
    "time_correlated",
    "reverse_time_correlated",
    "time_correlated_hi_to_ho",
    "reverse_tc_in_tc_out",
)
# Kinds that additionally apply a per-timestep linear noise-magnitude weight along the time axis.
_TIME_WEIGHTED_KINDS = (
    "reverse_time_correlated",
    "time_correlated_hi_to_ho",
    "reverse_tc_in_tc_out",
)


class TrainTimeDomainRandomizer(nn.Module):
    """Train-time top layer that perturbs model inputs during fitting only.

    The module is a strict no-op when disabled or in eval mode, supports batching and mbrl-lib
    ensembles (``E x B x Id`` tensors with per-member independent noise), and exposes a
    configurable per-feature noise scheme modulated by a training-time scheduler multiplier.

    Supported ``noise_kind`` values (RLRP-707):
      * ``gaussian`` / ``uniform`` -> i.i.d. additive noise (no time structure).
      * ``time_correlated`` -> AR(1)-correlated noise along the multistep/time axis.
      * ``reverse_time_correlated`` -> AR(1) noise with a per-timestep weight decreasing from full
        randomness at the oldest input timestep ``t-history_len`` to zero at the most recent ``t``.
      * ``time_correlated_hi_to_ho`` -> AR(1) noise with a per-timestep weight increasing from
        ``t-history_len`` to fully random at the target sequence end ``t+1+horizon_len`` (spans the
        input and target; requires ``randomize_target: true``).
      * ``reverse_tc_in_tc_out`` -> ``reverse_time_correlated`` weighting on the model input and
        plain (unweighted) ``time_correlated`` noise on the model target (requires
        ``randomize_target: true``).
    """

    per_feature_scale: torch.Tensor

    def __init__(
        self,
        feature_dim: int,
        per_feature_scale: torch.Tensor,
        noise_kind: str = "gaussian",
        correlation: float = 0.0,
        enable: bool = False,
        scheduler: Optional[NoiseScaleScheduler] = None,
        generator: Optional[torch.Generator] = None,
        singlestep_obs_len: Optional[int] = None,
        singlestep_act_len: Optional[int] = None,
        sequence_len: Optional[int] = None,
        horizon_len: Optional[int] = None,
        per_feature_scale_mode: str = "absolute",
        per_feature_scale_calibrated_strategy: Optional[Sequence[str]] = None,
    ):
        """Initialize the randomizer.

        :param feature_dim: The size of the input feature axis (last dim of ``model_in``).
        :param per_feature_scale: A tensor of shape ``(feature_dim,)`` with the per-feature noise
            magnitude (already expanded to the flattened multistep ``feature_dim``).
        :param per_feature_scale_mode: ``"absolute"`` (default, legacy) or
            ``"relative_to_normalized_std"``. See
            :class:`TrainTimeDomainRandomizationSpec` and
            :meth:`bind_normalizer_scale` (RLRP-761 S1.7).
        :param per_feature_scale_calibrated_strategy: Optional record of the
            ``per_dim_strategy`` in effect when ``per_feature_scale`` was
            calibrated. Purely declarative here: it is read by the
            training-start feature-normalization diagnostic to raise the
            ``DR-STALE`` flag (RLRP-761 S2.8).
        :param noise_kind: One of ``gaussian``, ``uniform``, ``time_correlated``,
            ``reverse_time_correlated``, ``time_correlated_hi_to_ho`` or ``reverse_tc_in_tc_out``
            (see the class docstring for the per-kind behaviour, RLRP-707).
        :param correlation: AR(1) coefficient ``rho`` in ``[0, 1)`` controlling the temporal
            correlation of the noise along the multistep/time axis for the time-correlated kinds
            (``time_correlated``, ``reverse_time_correlated``, ``time_correlated_hi_to_ho``,
            ``reverse_tc_in_tc_out``), via ``eps_t = rho * eps_{t-1} + sqrt(1 - rho**2) * w_t``:
            ``0.0`` => i.i.d. noise (no temporal correlation, each timestep independent);
            ``0.5`` => moderate correlation between consecutive timesteps;
            ``-> 1.0`` => strongly persistent noise (nearly constant along the time axis).
        :param enable: Whether the randomizer is active during training. Disabled by default.
        :param scheduler: Optional scheduler driving the noise-scale multiplier per epoch.
        :param generator: Optional seeded ``torch.Generator`` for reproducible noise.
        :param singlestep_obs_len: Single-step observation feature length. Required (with
            ``singlestep_act_len`` and ``sequence_len``) to make the time-correlated kinds
            multistep-aware (RLRP-707): the flattened input axis is laid out as per-timestep blocks
            ``(O_1..O_MS, A_1..A_MS)``, not a single increasing sequence.
        :param singlestep_act_len: Single-step action feature length (see ``singlestep_obs_len``).
        :param sequence_len: Multistep length ``MS`` of the model input (``history_len``); used to
            reshape the flattened feature axis to ``(..., MS, O+A)`` before applying AR(1) along the
            true time axis.
        :param horizon_len: Prediction horizon length of the model output/target. Required by the
            ``time_correlated_hi_to_ho`` kind to size the full ``history + horizon`` sequence over
            which the per-timestep noise weight increases (RLRP-707).
        """
        super().__init__()
        if noise_kind not in _SUPPORTED_NOISE_KINDS:
            raise ValueError(
                f"noise_kind must be one of {_SUPPORTED_NOISE_KINDS}, got {noise_kind!r}"
            )
        if per_feature_scale.numel() != feature_dim:
            raise ValueError(
                f"per_feature_scale must have feature_dim={feature_dim} elements, "
                f"got {per_feature_scale.numel()}"
            )

        self.feature_dim = int(feature_dim)
        self.enable = bool(enable)
        self.noise_kind = noise_kind
        self.correlation = float(correlation)
        self.horizon_len = None if horizon_len is None else int(horizon_len)
        self._scheduler = scheduler
        self._generator = generator
        # RLRP-707 fix (scheduler "not always working"): seed the multiplier from the scheduler at
        # construction so the schedule governs the noise from the very first training epoch. The
        # epoch callback only updates the multiplier at the *end* of each epoch, so without this
        # the first epoch (and any run where the epoch callback path is not wired) would silently
        # use a fixed ``1.0`` multiplier and ignore the configured schedule.
        self._scale_multiplier: float = (
            float(self._scheduler.value_at(0)) if self._scheduler is not None else 1.0
        )

        # .... Multistep layout (RLRP-707) ........................................................
        # When all three are provided and consistent with ``feature_dim``, the flattened input
        # feature axis can be reshaped to ``(..., MS, O+A)`` so that time-correlated noise is
        # applied along the genuine multistep/time axis (not over contiguous flattened indices).
        self.singlestep_obs_len = (
            None if singlestep_obs_len is None else int(singlestep_obs_len)
        )
        self.singlestep_act_len = (
            None if singlestep_act_len is None else int(singlestep_act_len)
        )
        self.sequence_len = None if sequence_len is None else int(sequence_len)
        self._multistep_aware = self._resolve_multistep_aware()
        self._ar1_non_temporal_warned = False

        # Fail-fast (RLRP-704): a time-correlated/time-weighted noise_kind is meaningless without a
        # known multistep layout — the flattened feature axis is per-timestep blocks, not a time
        # sequence. Previously this degraded *silently* to i.i.d. noise at apply time (a one-time
        # warning), which is easy to miss during large-scale experimentation. When the randomizer is
        # ENABLED with such a kind but the layout is unknown/inconsistent, raise at construction so
        # the misconfiguration surfaces immediately instead of corrupting an entire run unnoticed.
        if (
            self.enable
            and noise_kind in _TIME_CORRELATED_KINDS
            and not self._multistep_aware
        ):
            raise ValueError(
                f"TrainTimeDomainRandomizer: noise_kind={noise_kind!r} requires a known multistep "
                f"layout (singlestep_obs_len/singlestep_act_len/sequence_len consistent with "
                f"feature_dim={self.feature_dim}), but it is missing or inconsistent. The flattened "
                f"feature axis is laid out as per-timestep blocks, not a time sequence, so AR(1) "
                f"correlation/time-weighting cannot be applied. Provide the multistep layout or use "
                f"an i.i.d. noise_kind (gaussian/uniform)."
            )

        if per_feature_scale_mode not in _PER_FEATURE_SCALE_MODES:
            raise ValueError(
                f"per_feature_scale_mode={per_feature_scale_mode!r} is unknown; "
                f"expected one of {sorted(_PER_FEATURE_SCALE_MODES)}."
            )
        self.per_feature_scale_mode = per_feature_scale_mode
        self.per_feature_scale_calibrated_strategy: Optional[Tuple[str, ...]] = (
            None
            if per_feature_scale_calibrated_strategy is None
            else tuple(str(s) for s in per_feature_scale_calibrated_strategy)
        )
        self._normalizer_scale_bound = False
        self.register_buffer("per_feature_scale", per_feature_scale.reshape(-1).clone())
        # RLRP-761 S1.7: keep the AS-CONFIGURED vector so a (re-)binding always
        # starts from the dimensionless value rather than compounding scales.
        self.register_buffer(
            "configured_per_feature_scale", per_feature_scale.reshape(-1).clone()
        )

        # .... Model-output (target) layout (RLRP-707) ............................................
        # The randomizer is built for the flattened model *input* layout ``(O+A)*MS`` (feature_dim
        # == in_size). The model *output*/target follows the codebase convention
        # ``out_size = O*MS + A*(MS-1) = in_size - singlestep_act_len`` (the last single-step action
        # block is dropped, see ``unflaten_multistep_composed_array`` / the ``enable_last_action_
        # padding`` flag). The previous ``loss`` guard compared ``target.shape[-1] == feature_dim``,
        # which is therefore *always* false. We pre-compute the target feature dim and its
        # per-feature scale (input obs blocks + input action blocks minus the last timestep) so the
        # ``"target"`` segment can be perturbed with the correct shape and last-action padding.
        self._target_feature_dim: Optional[int] = None
        if (
            self._multistep_aware
            and self._ar1_last_action_padding is False
            and self.singlestep_obs_len
            and self.singlestep_act_len
            and self.sequence_len
        ):
            self._target_feature_dim = (
                self.singlestep_obs_len * self.sequence_len
                + self.singlestep_act_len * (self.sequence_len - 1)
            )
        # RLRP-761 S4.10: the per-single-step obs gain mapping an INPUT-space
        # perturbation to the TARGET space. Empty = the two spaces coincide,
        # which is the case for every normalizer type but
        # ``standard_symmetric_innovation`` (see :meth:`bind_target_space_obs_gain`).
        self.register_buffer(
            "target_space_obs_gain",
            torch.zeros(0, dtype=self.per_feature_scale.dtype),
        )
        self.register_buffer(
            "target_per_feature_scale",
            torch.zeros(0, dtype=self.per_feature_scale.dtype),
        )
        self._rebuild_target_per_feature_scale()
        if (
            self.enable
            and self.per_feature_scale_mode == _RELATIVE_SCALE_MODE
        ):
            warnings.warn(
                "TrainTimeDomainRandomizer: per_feature_scale_mode="
                f"'{_RELATIVE_SCALE_MODE}' requires bind_normalizer_scale(...) to be "
                "called once the normalizer statistics are fitted (this happens in "
                "OneDTransitionRewardModelV2.update_normalizer). Until then the "
                "configured DIMENSIONLESS vector is used as-is.",
                RuntimeWarning,
                stacklevel=2,
            )

        state = "ENABLED" if self.enable else "DISABLED"
        if noise_kind in _TIME_CORRELATED_KINDS:
            consol_msg_universal_one_liner(
                f"TrainTimeDomainRandomizer {state} "
                f"(noise_kind={self.noise_kind}, correlation={self.correlation})",
                caller_name="TrainTimeDomainRandomizer",
            )
        else:
            consol_msg_universal_one_liner(
                f"TrainTimeDomainRandomizer {state} "
                f"(noise_kind={self.noise_kind})",
                caller_name="TrainTimeDomainRandomizer",
            )

    def _resolve_multistep_aware(self) -> bool:
        """Whether the multistep layout is known and consistent with ``feature_dim``.

        ``feature_dim`` can correspond to either the multistep model **input** layout
        ``(O+A)*MS`` or the multistep model **output** layout ``O*MS + A*(MS-1)`` (the latter has
        one fewer single-step action block, RLRP-707). The expected sizes are computed with the
        existing :func:`compute_multistep_model_in_size` / :func:`compute_multistep_model_out_size`
        helpers rather than re-deriving them here. When the output layout is detected, AR(1) noise
        reshaping must use last-action padding (see :attr:`_ar1_last_action_padding`).

        :return: ``True`` if the flattened feature axis can be reshaped to ``(..., MS, O+A)``.
        """
        self._ar1_last_action_padding = False
        if (
            self.singlestep_obs_len is None
            or self.singlestep_act_len is None
            or self.sequence_len is None
        ):
            return False

        in_size = compute_multistep_model_in_size(
            self.singlestep_obs_len, self.singlestep_act_len, self.sequence_len
        )
        if self.feature_dim == in_size:
            self._ar1_last_action_padding = False
            return True

        out_size = compute_multistep_model_out_size(
            self.singlestep_obs_len, self.singlestep_act_len, self.sequence_len
        )
        if self.feature_dim == out_size:
            self._ar1_last_action_padding = True
            return True

        return False

    def bind_normalizer_scale(self, sigma_norm: torch.Tensor) -> None:
        """Resolve the effective noise scale against the normalizer output std.

        RLRP-761 S1.7. The randomizer is constructed **before** any statistic is
        fitted, so ``relative_to_normalized_std`` requires this late binding.
        Re-binding on every ``update_normalizer`` call is correct and required
        (the statistics are refit as the replay buffer grows) — hence the
        configured vector is kept separately and this method is idempotent.

        No-op in ``absolute`` mode, so the legacy path stays byte-identical.

        :param sigma_norm: per-**single-step**-feature (``obs + act``) or already
            flattened std of the normalizer output.
        """
        if self.per_feature_scale_mode != _RELATIVE_SCALE_MODE:
            return None
        sigma = torch.as_tensor(
            sigma_norm,
            device=self.per_feature_scale.device,
            dtype=self.per_feature_scale.dtype,
        ).reshape(-1)
        if sigma.numel() != self.feature_dim:
            expanded = _expand_per_feature_scale_to_flattened(
                [float(v) for v in sigma],
                feature_dim=self.feature_dim,
                singlestep_obs_len=self.singlestep_obs_len,
                singlestep_act_len=self.singlestep_act_len,
                sequence_len=self.sequence_len,
            )
            sigma = torch.tensor(
                expanded,
                device=self.per_feature_scale.device,
                dtype=self.per_feature_scale.dtype,
            )
        # A non-finite / non-positive sigma would silently kill the augmentation;
        # fall back to 1.0 (i.e. the configured value taken as absolute) on those
        # dims rather than zeroing the noise.
        sigma = torch.where(
            torch.isfinite(sigma) & (sigma > 0.0), sigma, torch.ones_like(sigma)
        )
        self.per_feature_scale = self.configured_per_feature_scale * sigma
        self._rebuild_target_per_feature_scale()
        self._normalizer_scale_bound = True
        return None

    def bind_target_space_obs_gain(self, gain: Optional[torch.Tensor]) -> None:
        """Register the INPUT-space -> TARGET-space obs conversion (RLRP-761 ``S4.10``).

        DR perturbs tensors that are **already normalized**, so a configured
        value means "so many units of the space it is injected into". With
        ``standard_symmetric_innovation`` the input and target obs spaces are
        different (state std vs one-step innovation), so a single vector cannot
        mean the same PHYSICAL perturbation on both segments -- the invariant
        ``randomize_target: true`` implicitly relies on.

        The conversion reuses the **same fused diagonal affine as the AR splice**
        (:attr:`OneDTransitionRewardModel.ar_bridge_gain`), deliberately, so
        there is a single definition of the input<->target scale relation. A
        physical noise of ``sigma_phys`` is ``sigma_phys / sigma_state`` input
        units and ``sigma_phys / s`` target units, hence::

            target_scale = input_scale / gain,   gain = s / sigma_state

        :param gain: per-single-step obs gain ``(Do,)``, or ``None`` / empty for
            the identity (every legacy type -- strict no-op).
        """
        if gain is None:
            resolved = torch.zeros(0, dtype=self.per_feature_scale.dtype)
        else:
            resolved = torch.as_tensor(
                gain,
                device=self.per_feature_scale.device,
                dtype=self.per_feature_scale.dtype,
            ).reshape(-1)
            if self.singlestep_obs_len and resolved.numel() != self.singlestep_obs_len:
                raise ValueError(
                    f"bind_target_space_obs_gain expects a per-single-step obs gain of "
                    f"{self.singlestep_obs_len} entries, got {resolved.numel()}."
                )
            resolved = torch.where(
                torch.isfinite(resolved) & (resolved > 0.0),
                resolved,
                torch.ones_like(resolved),
            )
        self.target_space_obs_gain = resolved
        self._rebuild_target_per_feature_scale()
        return None

    def _rebuild_target_per_feature_scale(self) -> None:
        """Derive the ``"target"`` segment scale from the input one.

        The target layout drops the LAST single-step action block
        (``out_size = in_size - singlestep_act_len``); on top of that, the obs
        block is converted to target space when the two differ (``S4.10``).
        """
        if not self._target_feature_dim:
            self.target_per_feature_scale = torch.zeros(
                0,
                device=self.per_feature_scale.device,
                dtype=self.per_feature_scale.dtype,
            )
            return None
        obs_block = self.singlestep_obs_len * self.sequence_len
        act_block_target = self.singlestep_act_len * (self.sequence_len - 1)
        obs_scale = self.per_feature_scale[:obs_block]
        if self.target_space_obs_gain.numel():
            obs_scale = obs_scale / self.target_space_obs_gain.repeat(self.sequence_len)
        self.target_per_feature_scale = torch.cat(
            [obs_scale, self.per_feature_scale[obs_block : obs_block + act_block_target]]
        )
        return None

    #: Buffers whose size is NOT known at construction time: they are empty until a
    #: late binding seam sizes them (:meth:`bind_target_space_obs_gain` /
    #: :meth:`_rebuild_target_per_feature_scale`, both driven from
    #: ``OneDTransitionRewardModelV2.update_normalizer``).
    _LAZILY_SIZED_BUFFERS = ("target_space_obs_gain", "target_per_feature_scale")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Resize the lazily-sized buffers to the checkpoint before loading.

        RLRP-761 ``P5``. The late-binding seam lives in ``update_normalizer``, which
        only runs during **training**. A load-only consumer (deploy-only re-run,
        test-time rollout from a checkpoint, evaluation tooling) therefore holds a
        freshly constructed randomizer whose lazily-sized buffers are still empty,
        while the checkpoint carries the bound ones. ``load_state_dict`` raises on a
        size mismatch **even with** ``strict=False`` (``strict`` only tolerates
        missing / unexpected KEYS, never a shape mismatch), so the whole checkpoint
        would be unloadable.

        Symmetric by construction: it equally handles loading an OLD (unbound,
        empty) checkpoint into a model whose buffers are already sized.
        """
        for name in self._LAZILY_SIZED_BUFFERS:
            key = f"{prefix}{name}"
            if key not in state_dict:
                continue
            current = getattr(self, name, None)
            incoming = state_dict[key]
            if current is None or tuple(current.shape) == tuple(incoming.shape):
                continue
            setattr(
                self,
                name,
                torch.zeros(
                    incoming.shape, device=current.device, dtype=current.dtype
                ),
            )
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def normalizer_scale_bound(self) -> bool:
        """Whether :meth:`bind_normalizer_scale` has resolved the effective scale."""
        return self._normalizer_scale_bound

    @property
    def effective_noise_scale(self) -> torch.Tensor:
        """Current per-feature noise magnitude actually applied during training.

        Equal to ``per_feature_scale * scale_multiplier``, capturing both the configured per-feature
        magnitude and the scheduler's per-epoch modulation. Introduced by task RLRP-707 (§4ter) so
        the epoch callback can record the *current* noise to tensorboard without reaching into
        internal buffers.

        :return: A tensor of shape ``(feature_dim,)`` with the effective per-feature noise scale.
        """
        return self.per_feature_scale * self._scale_multiplier

    def set_scale_multiplier(self, value: float) -> None:
        """Set the scalar noise-scale multiplier (typically from the scheduler).

        :param value: The new scalar multiplier.
        :return: None
        """
        self._scale_multiplier = float(value)
        return None

    def update_scale_from_scheduler(self, epoch: int) -> float:
        """Update the scalar multiplier from the scheduler for the given epoch.

        :param epoch: The current training epoch.
        :return: The applied scalar multiplier.
        """
        if self._scheduler is not None:
            self._scale_multiplier = float(self._scheduler.value_at(epoch))
        return self._scale_multiplier

    def forward(self, x: torch.Tensor, segment: str = "input") -> torch.Tensor:
        """Apply the train-time perturbation to the input (no-op when disabled or in eval mode).

        :param x: The model input (or target) tensor with the feature axis as the last dimension.
        :param segment: Either ``"input"`` (model input / history portion) or ``"target"`` (model
            output / horizon portion). Used by the ``time_correlated_hi_to_ho`` kind to place the
            per-timestep weight along the full ``history + horizon`` sequence (RLRP-707).
        :return: The perturbed (training) or unchanged (eval/disabled) input tensor.
        """
        if (not self.enable) or (not self.training):
            return x
        params = self._resolve_segment_params(x, segment=segment)
        if params is None:
            # Unsupported segment/shape (e.g. a target whose feature dim matches neither the input
            # nor the model-output layout): strict no-op instead of crashing (RLRP-707).
            return x
        noise = self._sample_noise(x, segment=segment, params=params)
        return x + noise

    def _resolve_segment_params(self, x: torch.Tensor, segment: str):
        """Resolve the per-segment feature dim, per-feature scale and last-action padding flag.

        The randomizer is built for the flattened model *input* layout ``(O+A)*MS``. The model
        *target* (output) follows the codebase convention ``out_size = in_size -
        singlestep_act_len`` (the last single-step action block is dropped, hence reshaping the
        target to ``(..., MS, O+A)`` requires ``enable_last_action_padding=True``). This selects
        the correct layout for the requested ``segment`` (RLRP-707).

        :param x: The tensor to be perturbed.
        :param segment: ``"input"`` or ``"target"``.
        :return: ``(feature_dim, per_feature_scale, last_action_padding)`` or ``None`` when the
            requested target segment shape is unsupported.
        """
        if (
            segment == "target"
            and self._target_feature_dim is not None
            and x.shape[-1] == self._target_feature_dim
        ):
            return (
                self._target_feature_dim,
                self.target_per_feature_scale,
                True,
            )
        if segment == "target" and x.shape[-1] != self.feature_dim:
            # Target requested but its feature dim matches neither the model-output layout (handled
            # above) nor the model-input layout: cannot place the noise safely -> no-op.
            return None
        return (self.feature_dim, self.per_feature_scale, self._ar1_last_action_padding)

    def _sample_noise(
        self, x: torch.Tensor, segment: str = "input", params=None
    ) -> torch.Tensor:
        """Sample a noise tensor broadcastable over ``x`` and scaled per feature.

        :param x: The reference input tensor.
        :param segment: ``"input"`` or ``"target"`` (see :meth:`forward`).
        :param params: Optional pre-resolved ``(feature_dim, per_feature_scale,
            last_action_padding)`` tuple (see :meth:`_resolve_segment_params`); resolved on demand
            when ``None``.
        :return: The additive noise tensor matching ``x`` shape, dtype and device.
        """
        if params is None:
            params = self._resolve_segment_params(x, segment=segment)
            if params is None:
                return torch.zeros_like(x)
        feature_dim, per_feature_scale, last_action_padding = params

        scale = (
            per_feature_scale.to(device=x.device, dtype=x.dtype)
            * self._scale_multiplier
        )
        # Broadcast scale over all leading (ensemble/batch/...) dimensions.
        broadcast_shape = (1,) * (x.dim() - 1) + (feature_dim,)
        scale = scale.reshape(broadcast_shape)

        if self.noise_kind == "uniform":
            # Uniform in [-1, 1] then scaled (matches a +-scale magnitude).
            unit = (
                torch.rand(
                    x.shape, generator=self._generator, device=x.device, dtype=x.dtype
                )
                * 2.0
                - 1.0
            )
            return unit * scale

        unit = torch.randn(
            x.shape, generator=self._generator, device=x.device, dtype=x.dtype
        )

        if self.noise_kind in _TIME_CORRELATED_KINDS and self.correlation > 0.0:
            unit = self._apply_ar1_correlation(
                unit, last_action_padding=last_action_padding
            )

        # Per-timestep linear noise-magnitude weight along the time axis (RLRP-707):
        #   * reverse_time_correlated  -> full randomness at the oldest input timestep
        #     ``t-history_len`` decreasing to no randomness at the most recent timestep ``t``.
        #   * time_correlated_hi_to_ho -> randomness increasing from ``t-history_len`` to fully
        #     random at the target sequence end ``t+1+horizon_len`` (spans input and target).
        #   * reverse_tc_in_tc_out -> reverse_time_correlated weighting on the model input segment
        #     and plain (unweighted) time_correlated noise on the model target segment.
        if self.noise_kind in _TIME_WEIGHTED_KINDS and not (
            self.noise_kind == "reverse_tc_in_tc_out" and segment == "target"
        ):
            unit = self._apply_time_weight(
                unit, segment=segment, last_action_padding=last_action_padding
            )

        return unit * scale

    def _apply_ar1_correlation(
        self, white_noise: torch.Tensor, last_action_padding: bool = False
    ) -> torch.Tensor:
        """Turn white noise into AR(1)-correlated noise along the multistep/time axis.

        ``eps_t = rho * eps_{t-1} + sqrt(1 - rho**2) * w_t`` keeps unit variance, where ``t`` is the
        multistep timestep index ``MS`` (RLRP-707). The flattened model-input feature axis is laid
        out as per-timestep blocks ``(O_1..O_MS, A_1..A_MS)`` and is therefore **not** an increasing
        time sequence; this method reshapes it to ``(..., MS, O+A)`` so the recursion runs along the
        genuine time axis, then flattens back to the original layout.

        When the multistep layout is unknown (non-multistep caller), it falls back to an i.i.d. (no
        correlation) noise with a one-time warning, since correlating over the contiguous flattened
        feature axis would be semantically wrong.

        :param white_noise: A white-noise tensor with the flattened feature axis as the last dim.
        :return: The AR(1)-correlated noise tensor (same shape).
        """
        if not self._multistep_aware:
            if not self._ar1_non_temporal_warned:
                warnings.warn(
                    "TrainTimeDomainRandomizer: time_correlated noise requested without a known "
                    "multistep layout (singlestep_obs_len/singlestep_act_len/sequence_len); the "
                    "flattened feature axis is not a time sequence, so AR(1) correlation is skipped "
                    "(falling back to i.i.d. noise).",
                    RuntimeWarning,
                )
                self._ar1_non_temporal_warned = True
            return white_noise

        # (..., feature_dim) --> (..., MS, O+A) so the last-but-one axis is the genuine time axis.
        # The model-output layout has one fewer single-step action block, so the reshape pads the
        # last action timestep (RLRP-707).
        unflat = timestep_first_multistep_dim_unflaten_array(
            white_noise,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            sequence_len=self.sequence_len,
            enable_last_action_padding=last_action_padding,
        )

        rho = self.correlation
        innovation_scale = (1.0 - rho * rho) ** 0.5
        correlated = torch.empty_like(unflat)
        # Time axis is dim=-2 (MS); feature axis (O+A) is dim=-1 and stays independent per feature.
        correlated[..., 0, :] = unflat[..., 0, :]
        for timestep_idx in range(1, unflat.shape[-2]):
            correlated[..., timestep_idx, :] = (
                rho * correlated[..., timestep_idx - 1, :]
                + innovation_scale * unflat[..., timestep_idx, :]
            )

        # (..., MS, O+A) --> (..., feature_dim) restoring the original flattened block layout.
        return revert_timestep_first_multistep_dim_unflaten_array(
            correlated,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            remove_last_action_padding=last_action_padding,
        )

    def _time_weight_vector(
        self,
        num_timesteps: int,
        segment: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build the per-timestep linear noise-magnitude weight along the time axis.

        For ``reverse_time_correlated`` the weight decreases linearly from ``1`` at the oldest
        timestep (``t-history_len``, index 0) to ``0`` at the most recent timestep (``t``, last
        index). For ``time_correlated_hi_to_ho`` the weight increases linearly across the full
        ``history + horizon`` sequence, from ``t-history_len`` up to ``1`` at the target sequence
        end ``t+1+horizon_len``; the ``"target"`` segment is offset by ``history_len`` so the input
        and target portions form a single monotonically increasing ramp (RLRP-707).

        :param num_timesteps: The number of timesteps (``MS``) on the unflattened time axis.
        :param segment: ``"input"`` or ``"target"``.
        :param device: Target device for the weight vector.
        :param dtype: Target dtype for the weight vector.
        :return: A weight tensor of shape ``(num_timesteps,)``.
        """
        if num_timesteps <= 1:
            return torch.ones(num_timesteps, device=device, dtype=dtype)

        idx = torch.arange(num_timesteps, device=device, dtype=dtype)
        if self.noise_kind in ("reverse_time_correlated", "reverse_tc_in_tc_out"):
            # 1 at oldest (index 0) -> 0 at most recent (last index). For ``reverse_tc_in_tc_out``
            # this weight is only ever applied to the ``"input"`` segment (the ``"target"`` segment
            # keeps plain unweighted time_correlated noise).
            return 1.0 - idx / (num_timesteps - 1)

        # time_correlated_hi_to_ho: increasing ramp over the full history + horizon sequence.
        history_len = self.sequence_len if self.sequence_len else num_timesteps
        horizon_len = self.horizon_len if self.horizon_len else 0
        total_len = history_len + horizon_len + 1
        offset = 0 if segment == "input" else history_len
        denom = max(1, total_len - 1)
        weight = (offset + idx) / denom
        return torch.clamp(weight, 0.0, 1.0)

    def _apply_time_weight(
        self, noise: torch.Tensor, segment: str, last_action_padding: bool = False
    ) -> torch.Tensor:
        """Scale the noise per timestep along the genuine multistep/time axis.

        Reshapes the flattened feature axis to ``(..., MS, O+A)`` (same machinery as
        :meth:`_apply_ar1_correlation`), multiplies by the per-timestep weight vector, then reverts
        to the original flattened block layout. Falls back to a no-op (with a one-time warning) when
        the multistep layout is unknown (RLRP-707).

        :param noise: A noise tensor with the flattened feature axis as the last dim.
        :param segment: ``"input"`` or ``"target"``.
        :return: The per-timestep-weighted noise tensor (same shape).
        """
        if not self._multistep_aware:
            if not self._ar1_non_temporal_warned:
                warnings.warn(
                    "TrainTimeDomainRandomizer: per-timestep time-weighted noise requested without "
                    "a known multistep layout (singlestep_obs_len/singlestep_act_len/sequence_len); "
                    "the flattened feature axis is not a time sequence, so the per-timestep weight "
                    "is skipped (falling back to unweighted noise).",
                    RuntimeWarning,
                )
                self._ar1_non_temporal_warned = True
            return noise

        unflat = timestep_first_multistep_dim_unflaten_array(
            noise,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            sequence_len=self.sequence_len,
            enable_last_action_padding=last_action_padding,
        )
        num_timesteps = unflat.shape[-2]
        weight = self._time_weight_vector(
            num_timesteps, segment=segment, device=unflat.device, dtype=unflat.dtype
        )
        # Weight broadcasts over the (O+A) feature axis (dim=-1), scaling each timestep (dim=-2).
        unflat = unflat * weight.reshape(num_timesteps, 1)

        return revert_timestep_first_multistep_dim_unflaten_array(
            unflat,
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            remove_last_action_padding=last_action_padding,
        )


def _resolve_spec(
    cfg: Optional[Union[dict, omegaconf.DictConfig, TrainTimeDomainRandomizationSpec]],
) -> TrainTimeDomainRandomizationSpec:
    """Resolve the train-time domain randomization spec from heterogeneous config inputs.

    :param cfg: A spec instance, a (DictConfig) mapping, or ``None``.
    :return: A :class:`TrainTimeDomainRandomizationSpec` instance.
    """
    if cfg is None:
        return TrainTimeDomainRandomizationSpec()
    if isinstance(cfg, TrainTimeDomainRandomizationSpec):
        return cfg
    if isinstance(cfg, omegaconf.DictConfig):
        cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)

    scheduler_cfg = cfg.get("scheduler", None)
    scheduler_spec = None
    if scheduler_cfg is not None:
        scheduler_spec = NoiseScaleSchedulerSpec(
            kind=scheduler_cfg.get("kind", "constant"),
            start_scale=scheduler_cfg.get("start_scale", 1.0),
            end_scale=scheduler_cfg.get("end_scale", 1.0),
            warmup_epochs=scheduler_cfg.get("warmup_epochs", 0),
            total_epochs=scheduler_cfg.get("total_epochs", 0),
            cycle_length=scheduler_cfg.get("cycle_length", 0),
            cycle_length_growth=scheduler_cfg.get("cycle_length_growth", 0.0),
            trough_growth=scheduler_cfg.get("trough_growth", 0.0),
        )
    return TrainTimeDomainRandomizationSpec(
        enable=cfg.get("enable", False),
        noise_kind=cfg.get("noise_kind", "gaussian"),
        per_feature_scale=cfg.get("per_feature_scale", None),
        per_feature_scale_mode=cfg.get("per_feature_scale_mode", "absolute"),
        per_feature_scale_calibrated_strategy=cfg.get(
            "per_feature_scale_calibrated_strategy", None
        ),
        default_scale=cfg.get("default_scale", 0.0),
        correlation=cfg.get("correlation", 0.0),
        randomize_target=cfg.get("randomize_target", False),
        scheduler=scheduler_spec,
    )


def _expand_per_feature_scale_to_flattened(
    scale_values: list,
    feature_dim: int,
    singlestep_obs_len: Optional[int],
    singlestep_act_len: Optional[int],
    sequence_len: Optional[int],
) -> list:
    """Expand a per-single-step-feature scale to the flattened multistep ``feature_dim``.

    The flattened model-input feature axis is laid out as per-timestep blocks
    ``(O[1:Do]_1..O[1:Do]_MS, A[1:Da]_1..A[1:Da]_MS)`` (RLRP-707), so a per-single-step-feature
    scale of length ``singlestep_obs_len + singlestep_act_len`` must be tiled over the ``MS``
    timesteps independently for the observation and action blocks. A scale already given at the
    full flattened ``feature_dim`` length is accepted as-is (back-compat).

    :param scale_values: The configured per-feature scale list (single-step or flattened length).
    :param feature_dim: The flattened model input feature dimension.
    :param singlestep_obs_len: Single-step observation feature length (or ``None``).
    :param singlestep_act_len: Single-step action feature length (or ``None``).
    :param sequence_len: Multistep length ``MS`` of the model input (or ``None``).
    :return: A per-feature scale list of length ``feature_dim``.
    """
    multistep_known = (
        singlestep_obs_len is not None
        and singlestep_act_len is not None
        and sequence_len is not None
    )
    single_feature = (
        (singlestep_obs_len + singlestep_act_len) if multistep_known else None
    )

    if multistep_known and len(scale_values) == single_feature:
        obs_scale = scale_values[:singlestep_obs_len]
        act_scale = scale_values[singlestep_obs_len:single_feature]
        # Tile each single-step block across the MS timesteps: (O_1..O_MS, A_1..A_MS).
        expanded = obs_scale * sequence_len + act_scale * sequence_len
        if len(expanded) != feature_dim:
            raise ValueError(
                f"per_feature_scale per-single-step length ({len(scale_values)}) tiled over "
                f"sequence_len ({sequence_len}) gives {len(expanded)} != feature_dim "
                f"({feature_dim}); check the multistep layout."
            )
        return expanded

    if len(scale_values) == feature_dim:
        return list(scale_values)

    expected = (
        f"either the per-single-step length ({single_feature}) or the flattened "
        f"feature_dim ({feature_dim})"
        if multistep_known
        else f"the flattened feature_dim ({feature_dim})"
    )
    raise ValueError(
        f"per_feature_scale length ({len(scale_values)}) does not match {expected}"
    )


def train_time_domain_randomizer_factory(
    cfg: Optional[Union[dict, omegaconf.DictConfig, TrainTimeDomainRandomizationSpec]],
    feature_dim: int,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    singlestep_obs_len: Optional[int] = None,
    singlestep_act_len: Optional[int] = None,
    sequence_len: Optional[int] = None,
    horizon_len: Optional[int] = None,
) -> TrainTimeDomainRandomizer:
    """Build a :class:`TrainTimeDomainRandomizer` from a config (OFF by default).

    :param cfg: The train-time domain randomization config (spec, mapping or ``None``).
    :param feature_dim: The flattened model input feature dimension (``in_size``).
    :param device: The target device for the noise scale buffer.
    :param dtype: The target dtype for the noise scale buffer.
    :param generator: Optional seeded ``torch.Generator`` for reproducible noise.
    :param singlestep_obs_len: Single-step observation feature length. When provided (with
        ``singlestep_act_len`` and ``sequence_len``) the factory is multistep-aware (RLRP-707): a
        per-single-step ``per_feature_scale`` is tiled across the ``MS`` timesteps and the
        multistep layout is forwarded to the randomizer for time-axis AR(1) noise.
    :param singlestep_act_len: Single-step action feature length (see ``singlestep_obs_len``).
    :param sequence_len: Multistep length ``MS`` of the model input (``history_len``).
    :param horizon_len: Prediction horizon length; forwarded to the randomizer for the
        ``time_correlated_hi_to_ho`` kind (RLRP-707).
    :return: A configured :class:`TrainTimeDomainRandomizer` instance.
    """
    spec = _resolve_spec(cfg)

    # ``time_correlated_hi_to_ho`` weights randomness from the input history through to the target
    # sequence end, so it only makes sense with target randomization explicitly enabled (RLRP-707).
    if spec.noise_kind == "time_correlated_hi_to_ho" and not spec.randomize_target:
        raise ValueError(
            "noise_kind='time_correlated_hi_to_ho' requires 'randomize_target: true' "
            "(the noise ramp extends from the input history to the target sequence end)."
        )

    # ``reverse_tc_in_tc_out`` applies reverse_time_correlated noise on the model input and plain
    # time_correlated noise on the model target, so it requires target randomization to be enabled
    # (RLRP-707).
    if spec.noise_kind == "reverse_tc_in_tc_out" and not spec.randomize_target:
        raise ValueError(
            "noise_kind='reverse_tc_in_tc_out' requires 'randomize_target: true' "
            "(reverse_time_correlated noise on the input, time_correlated noise on the target)."
        )

    if spec.per_feature_scale is not None:
        scale_values = _expand_per_feature_scale_to_flattened(
            list(spec.per_feature_scale),
            feature_dim=feature_dim,
            singlestep_obs_len=singlestep_obs_len,
            singlestep_act_len=singlestep_act_len,
            sequence_len=sequence_len,
        )
        per_feature_scale = torch.tensor(scale_values, device=device, dtype=dtype)
    else:
        per_feature_scale = torch.full(
            (feature_dim,), float(spec.default_scale), device=device, dtype=dtype
        )

    scheduler = NoiseScaleScheduler.from_spec(spec.scheduler)

    return TrainTimeDomainRandomizer(
        feature_dim=feature_dim,
        per_feature_scale=per_feature_scale,
        per_feature_scale_mode=spec.per_feature_scale_mode,
        per_feature_scale_calibrated_strategy=spec.per_feature_scale_calibrated_strategy,
        noise_kind=spec.noise_kind,
        correlation=spec.correlation,
        enable=spec.enable,
        scheduler=scheduler,
        generator=generator,
        singlestep_obs_len=singlestep_obs_len,
        singlestep_act_len=singlestep_act_len,
        sequence_len=sequence_len,
        horizon_len=horizon_len,
    )
