"""
Static (pre-flight) audit of the composed ERLL/UDER learning-rate law.

Motivation
----------
Under Law-U (RLRP-727) the realized learning rate of an ERLL/UDER run is the product

    lr(p, e) = base · g(p) · h(e)

where ``p`` is the ERLL/UDER pass index, ``e`` the epoch index *within* that pass, ``g`` the outer
per-pass envelope (``uder_lr_schedule.exp_decreassing``) and ``h`` the inner per-epoch shape
(``training.trainer_lr_schedule``) which RESTARTS at every pass.

Every term of that product is known **before the first gradient step**. When the inner decay is
sized for a horizon much shorter than the realized per-pass epoch budget, a large tail of every
pass runs at a learning rate so small that no parameter can move — compute is burned and
``patience``-based early stopping can only ever trigger inside that flat tail, where ``val/loss``
is flat by construction.

This module turns that into a cheap, deterministic **pre-flight check**: it evaluates the law on
the planned epoch grid and reports the fraction of the budget spent below a relative-LR threshold,
so the operator is warned at ``t = 0`` instead of discovering it on a TensorBoard plot hours later.

Scope / honesty about the approximation
---------------------------------------
The audit models the *intent* of ``setup_trainer_lr_scheduler_callback``, not its bit-exact torch
stepping order (off-by-one on the warmup boundary is possible). It is an advisory diagnostic, never
a training input: nothing here touches the optimizer. Inner shapes that cannot be evaluated
statically (``lambda_lr``, ``cyclic_lr``) disable the audit rather than produce a wrong verdict.
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import omegaconf

# Default: an epoch whose realized LR is below 1% of the run's peak realized LR is considered
# "effectively frozen" — for Adam with a typical `1e-3` peak this is `1e-5`, at which the per-epoch
# parameter displacement is a rounding error relative to the first epochs.
DEFAULT_RELATIVE_LR_THRESHOLD = 0.01

# Default: flag the configuration when more than a quarter of the total epoch budget is frozen.
DEFAULT_MAX_FLAT_RATIO = 0.25

#: Inner-schedule keys this module can evaluate statically.
SUPPORTED_INNER_SHAPE_KEYS = ("warmup", "exponential_lr", "cosine_annealing_lr")

#: Inner-schedule keys that make the audit undecidable (arbitrary python / non-monotonic).
UNSUPPORTED_INNER_SHAPE_KEYS = ("lambda_lr", "cyclic_lr")


@dataclass(frozen=True)
class LRSchedulePassAudit:
    """Per-ERLL-pass outcome of the LR law audit."""

    erll_pass: int
    num_epochs: int
    restart_lr: float
    flat_epochs: int
    first_flat_epoch: Optional[int]

    @property
    def flat_ratio(self) -> float:
        if self.num_epochs <= 0:
            return 0.0
        return self.flat_epochs / self.num_epochs


@dataclass(frozen=True)
class LRScheduleAudit:
    """Whole-run outcome of the LR law audit."""

    passes: Tuple[LRSchedulePassAudit, ...]
    peak_lr: float
    relative_lr_threshold: float
    supported: bool = True
    unsupported_reason: Optional[str] = None

    @property
    def total_epochs(self) -> int:
        return sum(each.num_epochs for each in self.passes)

    @property
    def total_flat_epochs(self) -> int:
        return sum(each.flat_epochs for each in self.passes)

    @property
    def flat_ratio(self) -> float:
        if self.total_epochs <= 0:
            return 0.0
        return self.total_flat_epochs / self.total_epochs

    @property
    def absolute_lr_threshold(self) -> float:
        return self.peak_lr * self.relative_lr_threshold

    def is_flagged(self, max_flat_ratio: float = DEFAULT_MAX_FLAT_RATIO) -> bool:
        """Whether the planned schedule wastes more than ``max_flat_ratio`` of its epoch budget."""
        if not self.supported:
            return False
        return self.flat_ratio > max_flat_ratio

    def format_report(self, max_flat_ratio: float = DEFAULT_MAX_FLAT_RATIO) -> str:
        """Human readable, console-friendly audit report."""
        if not self.supported:
            return (
                "LR schedule pre-flight audit SKIPPED: "
                f"{self.unsupported_reason}"
            )

        flagged = self.is_flagged(max_flat_ratio)
        header = (
            f"LR schedule pre-flight audit {'⚠️  FLAGGED' if flagged else '✅ ok'}\n"
            f"  peak realized LR       : {self.peak_lr:.3e}\n"
            f"  'frozen' LR threshold  : {self.absolute_lr_threshold:.3e} "
            f"({self.relative_lr_threshold:.1%} of peak)\n"
            f"  total epoch budget     : {self.total_epochs}\n"
            f"  epochs below threshold : {self.total_flat_epochs} "
            f"({self.flat_ratio:.1%}, budget alarm at {max_flat_ratio:.0%})\n"
        )
        rows = [
            "  per ERLL/UDER pass:",
            "    pass | epochs | restart LR | frozen from | frozen epochs",
        ]
        for each in self.passes:
            _from = "-" if each.first_flat_epoch is None else str(each.first_flat_epoch)
            rows.append(
                f"    {each.erll_pass:>4d} | {each.num_epochs:>6d} | "
                f"{each.restart_lr:.3e} | {_from:>11s} | "
                f"{each.flat_epochs:>3d} ({each.flat_ratio:.0%})"
            )
        footer = ""
        if flagged:
            footer = (
                "\n  → A large share of the compute budget runs at a learning rate that cannot "
                "move the parameters.\n"
                "    Consider: raising `training.trainer_lr_schedule.exponential_lr.gamma`, "
                "switching to\n"
                "    `cosine_annealing_lr` (whose `T_max` is sized per pass under Law-U), or "
                "shortening\n"
                "    `UDER.num_epochs_train_model` / `UDER.final_max_num_epochs_train_model`.\n"
                "    Note that `training.patience` can then only trigger inside the frozen tail, "
                "where\n"
                "    `val/loss` is flat by construction."
            )
        return header + "\n".join(rows) + footer


def make_outer_envelope(
    exp_decreassing: float, warmup_passes: int = 0, warmup_factor: float = 1.0
) -> Callable[[int], float]:
    """Build the Law-U outer per-pass envelope ``g(p)``.

    Mirrors ``AbstractExperienceReplayLearningLoop._law_u_outer_envelope``.

    :param exp_decreassing: geometric per-pass decay of the restart height.
    :param warmup_passes: number of leading passes held at ``warmup_factor``.
    :param warmup_factor: constant multiplier applied during the pass warmup.
    :return: ``g(p)``.
    """

    def _envelope(erll_pass: int) -> float:
        if erll_pass < warmup_passes:
            return float(warmup_factor)
        return float(exp_decreassing ** (erll_pass - warmup_passes))

    return _envelope


def make_inner_shape(
    cfg_lr_schedule: Optional[omegaconf.DictConfig],
) -> Tuple[Optional[Callable[[int, int], float]], Optional[str]]:
    """Build the inner per-epoch shape ``h(e)`` (a multiplier in ``]0, 1]``) from the config.

    Mirrors the *intent* of ``setup_trainer_lr_scheduler_callback._build_schedulers``: the
    post-warmup schedulers are chained, i.e. their multipliers multiply.

    :param cfg_lr_schedule: the ``training.trainer_lr_schedule`` node (may be ``None``).
    :return: a tuple ``(shape, unsupported_reason)``. ``shape`` is
        ``h(epoch_in_pass, pass_num_epochs) -> multiplier``, or ``None`` when the schedule cannot
        be evaluated statically, in which case ``unsupported_reason`` explains why.
    """
    if cfg_lr_schedule is None:
        # No inner schedule at all: constant LR within a pass. Nothing can ever be "frozen".
        return (lambda epoch_in_pass, pass_num_epochs: 1.0), None

    for each_key in UNSUPPORTED_INNER_SHAPE_KEYS:
        if omegaconf.OmegaConf.select(cfg_lr_schedule, each_key, default=None) is not None:
            return (
                None,
                f"`trainer_lr_schedule.{each_key}` cannot be evaluated statically",
            )

    cfg_warmup = omegaconf.OmegaConf.select(cfg_lr_schedule, "warmup", default=None)
    warmup_steps = 0
    warmup_factor = 1.0
    if cfg_warmup is not None and int(cfg_warmup.get("steps", 0) or 0) > 0:
        warmup_steps = int(cfg_warmup.get("steps", 0))
        warmup_factor = float(cfg_warmup.get("factor", 1.0))

    cfg_exponential = omegaconf.OmegaConf.select(
        cfg_lr_schedule, "exponential_lr", default=None
    )
    # RLRP-770 (option 3): the opt-in `exponential_lr.end_fraction` makes `gamma` PER-PASS
    # (`end_fraction ** (1/(pass_num_epochs-1))`), so it cannot be a single static scalar; defer it
    # to `_shape` where the pass budget is known. When `end_fraction` is unset the historical fixed
    # `gamma` is used.
    exp_end_fraction = (
        None if cfg_exponential is None else cfg_exponential.get("end_fraction", None)
    )
    gamma = (
        None
        if cfg_exponential is None or exp_end_fraction is not None
        else float(cfg_exponential.gamma)
    )

    cfg_cosine = omegaconf.OmegaConf.select(
        cfg_lr_schedule, "cosine_annealing_lr", default=None
    )
    cosine_t_max_cfg = (
        None if cfg_cosine is None else cfg_cosine.get("T_max", None)
    )

    def _shape(epoch_in_pass: int, pass_num_epochs: int) -> float:
        if epoch_in_pass < warmup_steps:
            return warmup_factor
        decay_epoch = epoch_in_pass - warmup_steps
        multiplier = 1.0
        if gamma is not None:
            multiplier *= gamma**decay_epoch
        elif exp_end_fraction is not None:
            # Auto-`gamma` sized to land on `end_fraction` at the last epoch of THIS pass.
            _n = int(pass_num_epochs or 0)
            _pass_gamma = (
                1.0
                if _n <= 1
                else float(exp_end_fraction) ** (1.0 / (_n - 1))
            )
            multiplier *= _pass_gamma**decay_epoch
        if cfg_cosine is not None:
            # Under Law-U `T_max` is sized from the realized per-pass budget (RLRP-727 B2).
            t_max = float(cosine_t_max_cfg or pass_num_epochs or 1)
            if t_max > 0:
                multiplier *= (1.0 + math.cos(math.pi * min(decay_epoch, t_max) / t_max)) / 2.0
        return multiplier

    return _shape, None


def audit_lr_schedule(
    pass_epoch_budgets: Sequence[int],
    base_lr: float,
    outer_envelope: Callable[[int], float],
    inner_shape: Optional[Callable[[int, int], float]],
    relative_lr_threshold: float = DEFAULT_RELATIVE_LR_THRESHOLD,
    unsupported_reason: Optional[str] = None,
) -> LRScheduleAudit:
    """Evaluate ``lr(p, e) = base · g(p) · h(e)`` on the planned epoch grid.

    :param pass_epoch_budgets: realized number of inner epochs for each ERLL/UDER pass.
    :param base_lr: the optimizer base learning rate.
    :param outer_envelope: ``g(p)``.
    :param inner_shape: ``h(e, pass_num_epochs)``; ``None`` marks the audit as unsupported.
    :param relative_lr_threshold: an epoch is "frozen" when its LR is below this fraction of the
        run's peak realized LR.
    :param unsupported_reason: why ``inner_shape`` is ``None``.
    :return: the audit outcome.
    """
    if inner_shape is None:
        return LRScheduleAudit(
            passes=tuple(),
            peak_lr=float(base_lr),
            relative_lr_threshold=relative_lr_threshold,
            supported=False,
            unsupported_reason=unsupported_reason or "inner LR shape is not statically evaluable",
        )

    realized: List[List[float]] = []
    for each_pass, each_budget in enumerate(pass_epoch_budgets):
        g_p = outer_envelope(each_pass)
        realized.append(
            [
                base_lr * g_p * inner_shape(each_epoch, each_budget)
                for each_epoch in range(max(int(each_budget), 0))
            ]
        )

    peak_lr = max((max(each) for each in realized if each), default=float(base_lr))
    threshold = peak_lr * relative_lr_threshold

    pass_audits: List[LRSchedulePassAudit] = []
    for each_pass, each_lr_seq in enumerate(realized):
        flat_indices = [i for i, lr in enumerate(each_lr_seq) if lr < threshold]
        pass_audits.append(
            LRSchedulePassAudit(
                erll_pass=each_pass,
                num_epochs=len(each_lr_seq),
                restart_lr=each_lr_seq[0] if each_lr_seq else 0.0,
                flat_epochs=len(flat_indices),
                first_flat_epoch=flat_indices[0] if flat_indices else None,
            )
        )

    return LRScheduleAudit(
        passes=tuple(pass_audits),
        peak_lr=peak_lr,
        relative_lr_threshold=relative_lr_threshold,
    )
