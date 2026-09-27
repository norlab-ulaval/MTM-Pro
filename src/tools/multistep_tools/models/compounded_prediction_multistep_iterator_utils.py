# coding=utf-8
import abc
import math

import torch
from torch import Tensor
from torch.distributions import Beta, Uniform, Exponential, Distribution

from tools.console_tools.message import consol_msg_universal_one_liner


class HorizonUnrollLengthDecaySampler:
    """
    Sampler for horizon unroll length during autoregressive training.

    Implements an *increasing* curriculum: the sampled unroll length grows from a short
    ``start_horizon_len`` (default 1) up to the full ``horizon_len``. The warmup phase
    therefore returns ``start_horizon_len`` (NOT the full horizon), and the progression
    phase grows the sampled length toward ``horizon_len``.

    ``decay_stop`` is an ABSOLUTE global-step index (NOT a duration): the curriculum
    spans ``[decay_start, decay_stop]`` and completes exactly at ``step == decay_stop``.

    Timeline:
    - step 0 to decay_start: Warmup phase, always returns start_horizon_len (short unroll)
    - step decay_start to (decay_start + transition_steps): Ease-in transition
      from start_horizon_len toward distribution sampling
    - step (decay_start + transition_steps) to decay_stop: Progression phase
      (distribution grows toward horizon_len)
    - step >= decay_stop: Always returns horizon_len (full unroll)

    Methods:
    - "beta": Use Beta distribution PDF to weight sampling probabilities
    - "exponential": Use exponential weights favoring higher indices
    - "linear": Two-stage sampling with decaying random probability
    """

    _step_count: int = 0
    method: str
    distrib_param: dict
    unroll_trajectory_index: Tensor

    # Warmup and transition parameters
    _decay_start: int = 0
    _transition_steps: int = 0  # Computed from transition_ratio * (decay_stop - decay_start)

    # Parameters for Beta distribution progression

    # # cfg 1
    # _alpha_start: float = 1.0
    # _beta_start: float = 3
    # _alpha_end: float = 20
    # _beta_end: float = 1.0

    # # cfg 2
    # _alpha_start: float = 0.25
    # _beta_start: float = 1.0
    # _alpha_end: float = 50
    # _beta_end: float = 1.25

    # # cfg 3
    # _alpha_start: float = 0.5
    # _beta_start: float = 4.0
    # _alpha_end: float = 20
    # _beta_end: float = 1.0

    # # cfg 4 ★
    # _alpha_start: float = 1.0
    # _beta_start: float = 6.0
    # _alpha_end: float = 20
    # _beta_end: float = 1.0

    # cfg 5 ★
    _alpha_start: float = 1.8
    _beta_start: float = 4.0
    _alpha_end: float = 16
    _beta_end: float = 1.0

    # Parameters for Exponential method
    _exp_rate_start: float = 0.0
    _exp_rate_end: float = 10.0

    # Parameters for Linear method
    _random_prob: float = 1.0

    _decay_stop: int
    _decay_duration: int = 0
    _progression_steps: int = 1

    def __init__(
        self,
        decay_stop: int,
        horizon_len: int,
        decay_start: int = 0,
        transition_ratio: float = 0.4,
        method: str = "beta",
        start_horizon_len: int = 1,
    ):
        """
        Args:
            decay_stop: ABSOLUTE global step count at which the curriculum completes
                        (the sampler returns the full horizon_len). The curriculum spans
                        ``[decay_start, decay_stop]``. Must be > decay_start when active.
            horizon_len: Maximum unroll length (full horizon) reached at the end of the curriculum.
            method: Sampling method. Options: "beta", "exponential", "linear".
            decay_start: Number of steps before progression begins (warmup phase).
                         During warmup, always returns start_horizon_len (short unroll).
            transition_ratio: Ratio of the decay window to use for ease-in transition after warmup.
                              E.g., 0.2 means 20% of the window is used for transition.
                              During transition, smoothly blends from returning start_horizon_len
                              to sampling from the distribution. Must be in [0, 1].
            start_horizon_len: Initial (short) unroll length the curriculum starts from
                               (returned during warmup). Defaults to 1; the sampled length
                               then grows toward horizon_len. Must be in [1, horizon_len].
        """
        super().__init__()

        assert 0 <= decay_stop
        assert 0 <= decay_start
        assert 0.0 <= transition_ratio <= 1.0
        assert isinstance(decay_stop, int)
        assert isinstance(decay_start, int)
        assert isinstance(start_horizon_len, int)
        assert (
            1 <= start_horizon_len <= horizon_len
        ), f"start_horizon_len must be in [1, horizon_len], got {start_horizon_len} (horizon_len={horizon_len})"
        # decay_stop is an ABSOLUTE global-step index (not a duration): the curriculum
        # completes exactly at step == decay_stop. 0 disables (handled by the caller gate).
        if decay_stop != 0 and decay_stop <= decay_start:
            raise ValueError(
                f"decay_stop ({decay_stop}) must be > decay_start ({decay_start}) for an "
                f"active horizon-unroll curriculum. decay_stop is an ABSOLUTE step index "
                f"(the curriculum completes at global step == decay_stop)."
            )

        self._decay_stop = decay_stop
        self._decay_start = decay_start
        self._transition_ratio = transition_ratio
        self._decay_duration = max(0, decay_stop - decay_start)
        self._transition_steps = int(self._decay_duration * transition_ratio)
        # Steps spent in the progression phase AFTER the ease-in transition; the sampler
        # reaches full horizon_len exactly at step == decay_stop.
        self._progression_steps = max(1, self._decay_duration - self._transition_steps)
        self.horizon_len = horizon_len
        self.start_horizon_len = start_horizon_len
        self.unroll_trajectory_index = torch.arange(start=1, end=horizon_len + 1)

        if method == "linear":
            consol_msg_universal_one_liner(
                f"AR unroll length: linear growth, warmup={decay_start} (returns start_H={start_horizon_len}), "
                f"transition={self._transition_steps} ({transition_ratio:.0%} of window), "
                f"grows {start_horizon_len} -> {horizon_len}, decay_stop={decay_stop} (abs step, "
                f"duration={self._decay_duration})"
            )
            self._random_prob = 1.0
            self._linear_decay_step = 1.0 / max(self._progression_steps, 1)
            self.distrib_param = {}
        elif method == "exponential":
            consol_msg_universal_one_liner(
                f"AR unroll length: exponential growth, warmup={decay_start} (returns start_H={start_horizon_len}), "
                f"transition={self._transition_steps} ({transition_ratio:.0%} of window), "
                f"grows {start_horizon_len} -> {horizon_len}, decay_stop={decay_stop} (abs step, "
                f"duration={self._decay_duration})"
            )
            self.distrib_param = {
                "rate": self._exp_rate_start,
            }
        elif method == "beta":
            consol_msg_universal_one_liner(
                f"AR unroll length: beta distribution growth, warmup={decay_start} (returns start_H={start_horizon_len}), "
                f"transition={self._transition_steps} ({transition_ratio:.0%} of window), "
                f"grows {start_horizon_len} -> {horizon_len}, decay_stop={decay_stop} (abs step, "
                f"duration={self._decay_duration})"
            )
            self.distrib_param = {
                "concentration1": self._alpha_start,
                "concentration0": self._beta_start,
            }
        else:
            raise ValueError(f"Invalid method: {method}")

        self.method = method

    @property
    def _effective_step_count(self) -> int:
        """Get effective step count for decay calculations (after warmup and transition)."""
        return max(0, self._step_count - self._decay_start - self._transition_steps)

    @property
    def _is_in_warmup_phase(self) -> bool:
        """Check if we're in the warmup phase (always return horizon_len)."""
        return self._step_count < self._decay_start

    @property
    def _is_in_transition_phase(self) -> bool:
        """Check if we're in the ease-out transition phase."""
        return (
            self._decay_start
            <= self._step_count
            < self._decay_start + self._transition_steps
        )

    @property
    def _transition_progress(self) -> float:
        """Get progress through transition phase [0, 1]."""
        if not self._is_in_transition_phase or self._transition_steps == 0:
            return 1.0
        steps_into_transition = self._step_count - self._decay_start
        return steps_into_transition / self._transition_steps

    @property
    def _is_decay_complete(self) -> bool:
        """Check if the curriculum is complete (reached the absolute decay_stop step)."""
        return self._step_count >= self._decay_stop

    def step(self) -> None:
        """Update scheduler state after each training step."""
        self._step_count += 1
        self._step_param()
        return None

    def _step_param(self) -> None:
        """Progressively update distribution parameters based on the method."""
        # During warmup or if decay is complete, no need to update
        if self._is_in_warmup_phase:
            return

        if self._is_decay_complete:
            return

        # Calculate progress ratio (clamped to [0, 1])
        progress = min(self._effective_step_count / max(self._progression_steps, 1), 1.0)

        if self.method == "beta":
            alpha = self._alpha_start + progress * (self._alpha_end - self._alpha_start)
            beta = self._beta_start + progress * (self._beta_end - self._beta_start)
            self.distrib_param["concentration1"] = alpha
            self.distrib_param["concentration0"] = beta

        elif self.method == "exponential":
            rate = self._exp_rate_start + progress * (
                self._exp_rate_end - self._exp_rate_start
            )
            self.distrib_param["rate"] = rate

        elif self.method == "linear":
            self._random_prob = max(0.0, 1.0 - progress)

    def _sample_unroll_len_probabilities(self) -> Tensor:
        """Sample probability weights for each possible horizon_unroll_len."""
        if self.method == "beta":
            return self._sample_beta_probabilities()
        elif self.method == "exponential":
            return self._sample_exponential_probabilities()
        elif self.method == "linear":
            return self._sample_linear_probabilities()
        else:
            return torch.ones(self.horizon_len) / self.horizon_len

    def _sample_beta_probabilities(self) -> Tensor:
        """Uses the Beta distribution PDF to weight sampling probabilities."""
        normalized_horizon = self.unroll_trajectory_index.float() / self.horizon_len
        normalized_horizon = torch.clamp(normalized_horizon, 1e-6, 1.0 - 1e-6)

        dist = Beta(**self.distrib_param)
        log_probs = dist.log_prob(normalized_horizon)
        weights = torch.exp(log_probs)
        weights = weights / weights.sum()

        return weights

    def _sample_exponential_probabilities(self) -> Tensor:
        """Computes exponentially increasing weights favoring higher indices."""
        rate = self.distrib_param["rate"]

        if rate <= 1e-6:
            weights = torch.ones(self.horizon_len)
        else:
            normalized_horizon = (
                self.unroll_trajectory_index.float() / self.horizon_len
            )
            weights = torch.exp(rate * normalized_horizon)

        weights = weights / weights.sum()
        return weights

    def _sample_linear_probabilities(self) -> Tensor:
        """Computes probability weights for linear decay method."""
        weights = torch.ones(self.horizon_len) * (self._random_prob / self.horizon_len)
        weights[-1] += 1.0 - self._random_prob
        return weights

    def _ease_out_cubic(self, t: float) -> float:
        """Cubic ease-out function: fast start, slow end."""
        return 1.0 - (1.0 - t) ** 3

    def _ease_in_out_sine(self, t: float) -> float:
        """Sine ease-in-out function: smooth S-curve."""
        return -(math.cos(math.pi * t) - 1) / 2

    def sample_unroll_len(self) -> int:
        """
        Sample a horizon_unroll_len value.

        Returns:
            int: A value in [1, horizon_len] with probability determined by
                 the current state and distribution parameters.

        Behavior (increasing curriculum start_horizon_len -> horizon_len):
        - Warmup phase: Always returns start_horizon_len (short unroll)
        - Transition phase: Ease-in from start_horizon_len toward sampled values
        - Progression phase: Samples from distribution (grows toward horizon_len)
        - Post-progression: Always returns horizon_len (full unroll)
        """
        # Warmup phase: always return the (short) start horizon length
        if self._is_in_warmup_phase:
            return self.start_horizon_len

        # Post-progression: always return the full horizon length
        if self._is_decay_complete:
            return self.horizon_len

        # Sample from the distribution
        weights = self._sample_unroll_len_probabilities()
        sample_idx = torch.multinomial(weights, num_samples=1)
        sampled_len = self.unroll_trajectory_index[sample_idx].item()

        # Transition phase: ease-in from start_horizon_len toward the sampled value
        if self._is_in_transition_phase:
            # Use ease-out curve for smooth transition out of the warmup value
            t = self._ease_out_cubic(self._transition_progress)

            # Stochastic blending: with probability (1-t), return start_horizon_len; else sampled
            if torch.rand(1).item() > t:
                return self.start_horizon_len
            else:
                # During early transition, bias toward lower values (closer to start_horizon_len)
                # by interpolating between sampled_len and start_horizon_len
                blended_len = sampled_len + (1.0 - t) * (
                    self.start_horizon_len - sampled_len
                )
                return max(1, min(self.horizon_len, int(round(blended_len))))

        return sampled_len


# =================================================================================================
class TeacherForcingScheduler:
    """
    Scheduler for teacher forcing probability during training.

    Controls the transition from teacher forcing (using ground truth) to
    free-running (using model predictions) during autoregressive training
    (scheduled sampling, Bengio et al., 2015).

    Methods:
    - "linear": Linearly decay teacher forcing probability from 1.0 to 0.0.
                The decay curve is genuinely linear in the decay progress
                ``base = 1 - progress`` (with the optional eased start below).
    - "exponential": Exponentially decay teacher forcing probability. The raw
                exponential lands at ``_exp_target_prob`` (e.g. 0.01) rather than
                0, so a smooth-to-zero gate is applied to terminate cleanly at 0.
    - "always_on": Always use teacher forcing (never disable)
    - "always_off": Never use teacher forcing (always free-running)

    An optional quadratic ease-in (``transition_ratio`` of the decay window)
    smooths the *start* of the decay so the probability leaves 1.0 with zero
    initial slope; past that window the named curve resumes. This shapes only the
    leading edge and does not change the 1.0 -> 0.0 endpoints.

    Units / cadence:
    - ``decay_start`` and ``decay_stop`` are expressed in **training steps
      (optimizer minibatches / gradient steps)**, NOT epochs. ``step()`` is called
      exactly once per training batch, so a ``decay_stop`` of e.g. 5000 means the
      decay completes at global step 5000 (i.e. ``5000 / steps_per_epoch`` epochs,
      dataset-size and batch-size dependent).

    ``decay_stop`` is an ABSOLUTE global-step index (NOT a duration): the decay spans
    ``[decay_start, decay_stop]`` and completes exactly at ``step == decay_stop``.

    Timeline (in training steps):
    - step 0 to decay_start: Teacher forcing probability = 1.0 (warmup phase)
    - step decay_start to decay_stop: Smooth decay with eased start
    - step >= decay_stop: Teacher forcing probability = 0.0
    """

    _step_count: int = 0
    method: str
    _teacher_forcing_prob: float = 1.0
    _decay_stop: int
    _decay_duration: int = 0
    _decay_start: int = 0
    _transition_ratio: float = 0.4

    # Parameters for exponential decay
    _exp_target_prob: float = 0.01

    def __init__(
        self,
        decay_start: int = 0,
        decay_stop: int = 0,
        method: str = "linear",
        transition_ratio: float = 0.2,
    ):
        """
        Args:
            decay_start: Number of steps before decay begins (warmup phase).
                         During this phase, teacher forcing probability = 1.0.
            decay_stop: ABSOLUTE global step count at which the decay completes
                        (teacher forcing probability reaches 0.0). The decay spans
                        ``[decay_start, decay_stop]``. Set to -1 for always_on, 0 for
                        always_off. Must be > decay_start for an active decay.
            method: Decay method. Options: "linear", "exponential", "always_on", "always_off"
            transition_ratio: Ratio of the decay window for ease-in at the start of decay.
                              Controls how gradually the decay begins. Must be in [0, 1].
        """
        super().__init__()
        self._decay_stop = decay_stop
        self._decay_start = max(0, decay_start)
        self._transition_ratio = max(0.0, min(1.0, transition_ratio))
        self.method = method

        # Handle special cases (absolute-step sentinels)
        if decay_stop == -1:
            self.method = "always_on"
        elif decay_stop == 0:
            self.method = "always_off"

        # Decay duration (span from decay_start to the ABSOLUTE decay_stop). Only
        # meaningful for the active linear/exponential methods.
        self._decay_duration = max(0, self._decay_stop - self._decay_start)

        if self.method == "always_on":
            consol_msg_universal_one_liner(
                "Teacher forcing: ALWAYS ON (never disabled)"
            )
            self._teacher_forcing_prob = 1.0
        elif self.method == "always_off":
            consol_msg_universal_one_liner(
                "Teacher forcing: ALWAYS OFF (free-running from start)"
            )
            self._teacher_forcing_prob = 0.0
        elif self.method == "linear":
            self._validate_active_decay_stop()
            consol_msg_universal_one_liner(
                f"Teacher forcing: linear decay, warmup={decay_start}, "
                f"ease_ratio={transition_ratio:.0%}, decay_stop={decay_stop} (abs step, "
                f"duration={self._decay_duration})"
            )
            self._teacher_forcing_prob = 1.0
        elif self.method == "exponential":
            self._validate_active_decay_stop()
            consol_msg_universal_one_liner(
                f"Teacher forcing: exponential decay, warmup={decay_start}, "
                f"ease_ratio={transition_ratio:.0%}, decay_stop={decay_stop} (abs step, "
                f"duration={self._decay_duration})"
            )
            self._teacher_forcing_prob = 1.0
            self._exp_decay_rate = self._exp_target_prob ** (
                1.0 / max(self._decay_duration, 1)
            )
        else:
            raise ValueError(
                f"Invalid teacher forcing method: {method}. "
                f"Options: 'linear', 'exponential', 'always_on', 'always_off'"
            )

    def _validate_active_decay_stop(self) -> None:
        """Fail-fast: an active decay needs an absolute stop strictly after the warmup."""
        if self._decay_stop <= self._decay_start:
            raise ValueError(
                f"decay_stop ({self._decay_stop}) must be > decay_start "
                f"({self._decay_start}) for an active '{self.method}' teacher-forcing decay. "
                f"decay_stop is an ABSOLUTE step index (not a duration): the decay completes "
                f"at global step == decay_stop."
            )

    @property
    def _effective_step_count(self) -> int:
        """Get effective step count for decay (0 during warmup, counts from decay_start)."""
        return max(0, self._step_count - self._decay_start)

    @property
    def _is_in_warmup_phase(self) -> bool:
        """Check if we're still in the warmup phase (before decay starts)."""
        return self._step_count < self._decay_start

    @property
    def _is_decay_complete(self) -> bool:
        """Check if the decay phase is complete (reached the absolute decay_stop step)."""
        return self._step_count >= self._decay_stop

    def _ease_in_quad(self, t: float) -> float:
        """Quadratic ease-in: slow start, accelerating. f(0)=0, f(1)=1, f'(0)=0."""
        return t * t

    def _apply_ease_in(self, linear_progress: float) -> float:
        """
        Apply ease-in curve to the early portion of progress.

        This creates a smooth transition from prob=1.0 by making the progress
        start slowly (derivative = 0 at start).

        Args:
            linear_progress: Raw progress through decay [0, 1]

        Returns:
            Eased progress value [0, 1] with smooth start
        """
        if self._transition_ratio <= 0.0 or linear_progress <= 0.0:
            return linear_progress

        if self._transition_ratio >= 1.0:
            # Entire decay is eased
            return self._ease_in_quad(linear_progress)

        # Blend eased and linear portions
        # Before transition_ratio: use ease-in curve scaled to meet linear at boundary
        # After transition_ratio: use linear progress

        if linear_progress <= self._transition_ratio:
            # In the ease-in region: map [0, transition_ratio] -> [0, transition_ratio]
            # but with ease-in curve
            normalized_t = linear_progress / self._transition_ratio
            eased_t = self._ease_in_quad(normalized_t)
            return eased_t * self._transition_ratio
        else:
            # Past the ease-in region: linear from here
            # This is already continuous because at linear_progress = transition_ratio:
            # eased value = ease_in_quad(1.0) * transition_ratio = 1.0 * transition_ratio = transition_ratio
            return linear_progress

    def step(self) -> None:
        """Update scheduler state after each training step."""
        self._step_count += 1
        self._step_param()
        return None

    def _compute_base_decay_probability(self, progress: float) -> float:
        """
        Compute decay probability from the base decay curve.

        The teacher-forcing probability must reach exactly ``0.0`` at
        ``progress == 1.0`` so that training transitions cleanly to free-running.
        How that zero-termination is reached depends on the method, and the
        smooth-to-zero ``end_gate = (1 - progress)`` is applied **only** to methods
        whose base curve does not already terminate at 0:

        - ``"linear"``: the base curve ``1 - progress`` already terminates at 0, so
          it is returned as-is. This keeps the decay *genuinely linear*, matching
          the method name. (Multiplying the gate here would turn it into a
          quadratic ``(1 - progress) ** 2`` curve -- the F-TF1 defect this fix
          removes.)
        - ``"exponential"``: the base curve ``target_prob ** progress`` lands at
          ``_exp_target_prob`` (e.g. 0.01), *not* 0. The ``end_gate`` is therefore
          multiplied in to remove the final-step discontinuity and drive the curve
          smoothly to 0 at ``progress == 1.0``.

        Args:
            progress: Value in [0, 1] representing progress through decay phase.

        Returns:
            Probability value in [0, 1].
        """
        progress = max(0.0, min(1.0, progress))

        if self.method == "linear":
            # Base already terminates at 0.0 at progress == 1.0 -> genuinely linear,
            # no end_gate (applying it would make the curve quadratic).
            base = 1.0 - progress

        elif self.method == "exponential":
            equivalent_step = progress * self._decay_duration
            base = self._exp_decay_rate ** equivalent_step
            # Exponential lands at _exp_target_prob (not 0); gate it smoothly to
            # zero so the schedule terminates cleanly at progress == 1.0.
            base = base * (1.0 - progress)

        else:
            base = 1.0

        return max(0.0, min(1.0, base))

    def _step_param(self) -> None:
        """Update teacher forcing probability based on the method."""
        if self.method in ["always_on", "always_off"]:
            return

        if self._is_in_warmup_phase:
            self._teacher_forcing_prob = 1.0
            return

        # Compute progress even at/after completion so the last value is the
        # smooth curve value at progress=1.0 (which will be exactly 0.0).
        linear_progress = self._effective_step_count / max(self._decay_duration, 1)
        linear_progress = min(1.0, linear_progress)

        # Apply ease-in to smooth the start of decay
        eased_progress = self._apply_ease_in(linear_progress)

        # Get decay probability (now guaranteed to hit 0 at the end)
        self._teacher_forcing_prob = self._compute_base_decay_probability(eased_progress)
        return None

    def should_use_teacher_forcing(self) -> bool:
        """
        Determine whether to use teacher forcing for the current step.

        Returns:
            bool: True if teacher forcing should be used, False for free-running.
        """
        if self.method == "always_on":
            return True
        elif self.method == "always_off":
            return False
        elif self._is_decay_complete:
            return False
        else:
            return torch.rand(1).item() < self._teacher_forcing_prob

    @property
    def current_probability(self) -> float:
        """Get the current teacher forcing probability."""
        if self.is_fully_disabled:
            return 0.0
        return self._teacher_forcing_prob

    @property
    def is_fully_disabled(self) -> bool:
        """Check if teacher forcing is fully disabled."""
        return self._teacher_forcing_prob <= 0.0 or self._is_decay_complete


class TemporalWeightScheduler:
    """
    Scheduler for autoregressive temporal discount weight (gamma).

    Monotonically transitions from start_weight to target_weight,
    supporting both ramp-up (start < target) and ramp-down (start > target).

    ``ramp_stop`` is an ABSOLUTE global-step index (NOT a duration): the ramp spans
    ``[warmup_steps, ramp_stop]`` and completes exactly at ``step == ramp_stop``.

    Timeline:
    - step 0 to warmup_steps: Returns start_weight (flat warmup)
    - step warmup_steps to ramp_stop: Smooth transition
    - step >= ramp_stop: Returns target_weight
    """

    _step_count: int = 0

    def __init__(
        self,
        target_weight: float,
        start_weight: float = 0.1,
        warmup_steps: int = 0,
        ramp_stop: int = 1000,
        method: str = "linear",
    ):
        """
        Args:
            target_weight: Final temporal weight (gamma) to reach.
            start_weight: Initial temporal weight at the start of transition.
            warmup_steps: Steps before transition begins (stays at start_weight).
            ramp_stop: ABSOLUTE global step count at which the ramp completes (reaches
                       target_weight). The ramp spans ``[warmup_steps, ramp_stop]``.
                       Must be > warmup_steps when active.
            method: Transition method. Options: "linear", "ease_in", "ease_out", "ease_in_out".
        """
        assert start_weight >= 0.0, f"start_weight must be >= 0, got {start_weight}"
        assert target_weight >= 0.0, f"target_weight must be >= 0, got {target_weight}"
        assert warmup_steps >= 0
        assert ramp_stop >= 0
        # ramp_stop is an ABSOLUTE global-step index (not a duration): the ramp completes
        # exactly at step == ramp_stop. 0 disables (handled by the caller gate).
        if ramp_stop != 0 and ramp_stop <= warmup_steps:
            raise ValueError(
                f"ramp_stop ({ramp_stop}) must be > warmup_steps ({warmup_steps}) for an "
                f"active temporal-weight ramp. ramp_stop is an ABSOLUTE step index "
                f"(the ramp completes at global step == ramp_stop)."
            )

        self._start_weight = start_weight
        self._target_weight = target_weight
        self._warmup_steps = warmup_steps
        self._ramp_stop = ramp_stop
        self._ramp_duration = max(1, ramp_stop - warmup_steps)
        self._method = method
        self._current_weight = start_weight

        # Determine direction for logging
        if start_weight < target_weight:
            direction = "ramp-up"
        elif start_weight > target_weight:
            direction = "ramp-down"
        else:
            direction = "constant"

        consol_msg_universal_one_liner(
            f"AR temporal weight scheduler ({direction}): {start_weight:.3f} → {target_weight:.3f}, "
            f"warmup={warmup_steps}, ramp_stop={ramp_stop} (abs step, "
            f"duration={self._ramp_duration}), method={method}"
        )

    @property
    def current_weight(self) -> float:
        """Get the current temporal weight (gamma)."""
        return self._current_weight

    @property
    def _is_in_warmup(self) -> bool:
        # Unified boundary convention: warmup active while step < warmup_steps.
        return self._step_count < self._warmup_steps

    @property
    def _is_ramp_complete(self) -> bool:
        # Completes at the ABSOLUTE ramp_stop step (unified >= convention).
        return self._step_count >= self._ramp_stop

    def _ease_in_quad(self, t: float) -> float:
        """Quadratic ease-in: slow start."""
        return t * t

    def _ease_out_quad(self, t: float) -> float:
        """Quadratic ease-out: slow end."""
        return 1.0 - (1.0 - t) ** 2

    def _ease_in_out_quad(self, t: float) -> float:
        """Quadratic ease-in-out: slow start and end."""
        if t < 0.5:
            return 2.0 * t * t
        else:
            return 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0

    def _apply_easing(self, t: float) -> float:
        """Apply easing function based on method."""
        t = max(0.0, min(1.0, t))
        if self._method == "linear":
            return t
        elif self._method == "ease_in":
            return self._ease_in_quad(t)
        elif self._method == "ease_out":
            return self._ease_out_quad(t)
        elif self._method == "ease_in_out":
            return self._ease_in_out_quad(t)
        else:
            return t

    def step(self) -> None:
        """Update scheduler state after each training step."""
        self._step_count += 1
        self._update_weight()

    def _update_weight(self) -> None:
        """Compute the current weight based on progress."""
        if self._is_in_warmup:
            self._current_weight = self._start_weight
            return

        if self._is_ramp_complete:
            self._current_weight = self._target_weight
            return

        # Compute linear progress through ramp phase [0, 1]
        # _step_count has already been incremented, so subtract warmup to get steps INTO ramp
        steps_into_ramp = self._step_count - self._warmup_steps
        linear_progress = steps_into_ramp / max(self._ramp_duration, 1)
        linear_progress = min(1.0, linear_progress)  # Clamp to [0, 1]

        # Apply easing (works for both directions since we interpolate)
        eased_progress = self._apply_easing(linear_progress)

        # Linear interpolation: works for both ramp-up and ramp-down
        self._current_weight = (
            self._start_weight
            + eased_progress * (self._target_weight - self._start_weight)
        )
