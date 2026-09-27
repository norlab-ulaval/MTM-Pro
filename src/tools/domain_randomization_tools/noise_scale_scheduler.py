# coding=utf-8
"""Train-time domain randomization noise-scale scheduler.

Permanent production module. Introduced by task RLRP-707 of the Training-Pipeline Domain
Randomization `.junie` plan (``feature_training_pipeline_domain_randomization_plan_20260610.md``).
A small, framework-free helper returning a scalar multiplier as a function of the current training
epoch, used to increase/decrease the train-time domain randomization noise scale during fitting.

Supported kinds (RLRP-707):
  * ``constant``         -> holds ``start_scale`` for the whole run.
  * ``linear``           -> linear ramp from ``start_scale`` to ``end_scale`` over ``total_epochs``.
  * ``ease_in_out_ramp`` -> single smooth half-cosine S-curve ramp (slow start, slow finish) from
                            ``start_scale`` to ``end_scale`` (a *one-shot* ease-in/ease-out, NOT a
                            wave). Formerly named ``cosine``.
  * ``cyclic_cosine``    -> a repeating cosine *wave* whose peak grows/shrinks over the run via an
                            envelope ramping from ``start_scale`` to ``end_scale`` over
                            ``total_epochs``. The base period (wave length in epochs) is
                            ``cycle_length``. Two extra shape controls (RLRP-707 follow-up):
                            ``cycle_length_growth`` makes the period get wider (``>0``) or narrower
                            (``<0``) as epochs progress, and ``trough_growth`` lifts the wave's
                            lowest point per completed cycle (``0.0`` -> every trough returns to
                            ``0``; ``0.25`` -> the trough scales up by 25% of the envelope each
                            cycle).
"""
import math
from typing import Optional

from tools.domain_randomization_tools.spec_containers import NoiseScaleSchedulerSpec

_SUPPORTED_KINDS = ("constant", "linear", "ease_in_out_ramp", "cyclic_cosine")


class NoiseScaleScheduler:
    """Scalar noise-scale multiplier scheduler driven by the training epoch."""

    def __init__(
        self,
        kind: str = "constant",
        start_scale: float = 1.0,
        end_scale: float = 1.0,
        warmup_epochs: int = 0,
        total_epochs: int = 0,
        cycle_length: int = 0,
        cycle_length_growth: float = 0.0,
        trough_growth: float = 0.0,
    ):
        """Initialize the scheduler.

        :param kind: One of ``constant``, ``linear``, ``ease_in_out_ramp`` or ``cyclic_cosine``.
        :param start_scale: Multiplier value at (or before the end of) the warmup phase. For
            ``cyclic_cosine`` it is the envelope (wave peak) starting value; the wave's lowest point
            returns to ``0`` unless lifted by ``trough_growth``.
        :param end_scale: Multiplier value reached at ``total_epochs``. For the ramp kinds this is
            the ramp target; for ``cyclic_cosine`` it is the target of the growing/shrinking
            envelope that modulates the wave magnitude over the run.
        :param warmup_epochs: Number of initial epochs held at ``start_scale``.
        :param total_epochs: Epoch index at which ``end_scale`` is reached (and clamped after). For
            ``cyclic_cosine`` it is the horizon over which the wave envelope ramps from
            ``start_scale`` to ``end_scale``.
        :param cycle_length: ``cyclic_cosine`` base wave length (period) in epochs.
        :param cycle_length_growth: ``cyclic_cosine`` control of how the period grows/shrinks as
            epochs progress, **independent of** ``total_epochs``. ``0.0`` => constant period;
            ``> 0`` => the cycles get **wider** (longer period) over the run; ``< 0`` => the cycles
            get **narrower** (shorter period). It is the number of epochs added to the period per
            elapsed epoch, i.e. the instantaneous period is
            ``L(t) = cycle_length + cycle_length_growth * t`` (``t`` = epochs since warmup); the
            phase uses the closed-form integral of ``1 / L(t)`` so the period varies smoothly.
        :param trough_growth: ``cyclic_cosine`` control of the wave's lowest point as epochs
            progress. ``0.0`` => every cycle's trough returns to ``0``; ``0.25`` => the trough
            scales up by 25% of the current envelope per completed cycle (clamped at the envelope,
            i.e. the oscillation flattens once the trough reaches the peak).
        """
        if kind not in _SUPPORTED_KINDS:
            raise ValueError(
                f"NoiseScaleScheduler kind must be one of {_SUPPORTED_KINDS}, got {kind!r}"
            )
        self.kind = kind
        self.start_scale = float(start_scale)
        self.end_scale = float(end_scale)
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)
        self.cycle_length = int(cycle_length)
        self.cycle_length_growth = float(cycle_length_growth)
        self.trough_growth = float(trough_growth)

    @classmethod
    def from_spec(
        cls, spec: Optional[NoiseScaleSchedulerSpec]
    ) -> "NoiseScaleScheduler":
        """Build a scheduler from a :class:`NoiseScaleSchedulerSpec` (``None`` -> constant 1.0).

        :param spec: The scheduler specification or ``None``.
        :return: A configured :class:`NoiseScaleScheduler` instance.
        """
        if spec is None:
            return cls()
        return cls(
            kind=spec.kind,
            start_scale=spec.start_scale,
            end_scale=spec.end_scale,
            warmup_epochs=spec.warmup_epochs,
            total_epochs=spec.total_epochs,
            cycle_length=spec.cycle_length,
            cycle_length_growth=spec.cycle_length_growth,
            trough_growth=spec.trough_growth,
        )

    def value_at(self, epoch: int) -> float:
        """Return the scalar noise-scale multiplier for the given training epoch.

        :param epoch: The current training epoch (0-indexed).
        :return: The scalar multiplier value.
        """
        if epoch <= self.warmup_epochs:
            return self.start_scale

        if self.kind == "constant":
            return self.start_scale

        if self.kind == "cyclic_cosine":
            # Repeating cosine wave whose peak is the (growing/shrinking) envelope ramping
            # ``start_scale`` -> ``end_scale`` over ``total_epochs``. Unlike ``ease_in_out_ramp`` the
            # phase is NOT clamped, so the full ``2*pi`` cycle repeats indefinitely. Two shape
            # controls (RLRP-707 follow-up): ``cycle_length_growth`` widens/narrows the period over
            # the run and ``trough_growth`` lifts the per-cycle lowest point.
            if self.cycle_length <= 0:
                return self.start_scale
            elapsed = epoch - self.warmup_epochs
            # Growing/shrinking envelope (wave peak) driven by ``end_scale`` over ``total_epochs``.
            envelope_span = self.total_epochs - self.warmup_epochs
            if envelope_span <= 0:
                envelope_progress = 1.0
            else:
                envelope_progress = min(1.0, max(0.0, elapsed / envelope_span))
            envelope = self.start_scale + (
                self.end_scale - self.start_scale
            ) * envelope_progress
            # Phase (number of cycles since warmup). ``cycle_length_growth`` makes the period
            # grow/shrink directly with elapsed epochs (see below); ``g == 0`` is the constant
            # period ``t / cycle_length``.
            growth = self.cycle_length_growth
            if growth == 0.0:
                phase = elapsed / self.cycle_length
            else:
                # Instantaneous period grows/shrinks directly with elapsed epochs (independent of
                # ``total_epochs``): ``L(t) = cycle_length + growth * t`` (``t`` = epochs since
                # warmup). ``growth`` is the number of epochs added to the period per elapsed epoch
                # (``> 0`` widens, ``< 0`` narrows). The phase is the closed-form integral of
                # ``1 / L(t)``: ``(1 / growth) * ln(1 + growth * t / cycle_length)``.
                ratio = 1.0 + growth * elapsed / self.cycle_length
                ratio = max(ratio, 1e-6)  # guard against a non-positive instantaneous period
                phase = math.log(ratio) / growth
            wave_factor = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
            # Per-cycle lowest point: ``trough_growth`` lifts the trough by that fraction of the
            # current envelope for each completed cycle (clamped at the envelope peak).
            cycle_index = math.floor(phase)
            trough = min(self.trough_growth * cycle_index * envelope, envelope)
            return trough + (envelope - trough) * wave_factor

        span = self.total_epochs - self.warmup_epochs
        if span <= 0:
            return self.end_scale

        progress = (epoch - self.warmup_epochs) / span
        if progress >= 1.0:
            return self.end_scale

        if self.kind == "linear":
            return self.start_scale + (self.end_scale - self.start_scale) * progress

        # ease_in_out_ramp: single smooth half-cosine S-curve (slow start, slow finish). The cosine
        # argument only sweeps [0, pi] (half a period) so the multiplier transitions ONCE from
        # start_scale to end_scale and never oscillates.
        ease_factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.start_scale + (self.end_scale - self.start_scale) * ease_factor
