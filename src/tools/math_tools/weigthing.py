# coding=utf-8
import warnings
from typing import Callable, Optional, Union

import numpy as np
import torch
from deprecated import deprecated

from tools.math_tools.ndarray_tools.custom_msg import nan_infinity_console_warning

#: RLRP-761 S11.3 -- once-per-run latch on the NaN/inf reporting of
#: :func:`weight_values` / :func:`apply_weights`. Each ``torch.all(torch.isfinite(...))``
#: is a device->host SYNCHRONIZATION, and on the log-space loss path three of them run
#: per call.
#: We keep first-detection (the review forbids deleting the production warning), but
#: once a given tensor has been reported non-finite we stop re-checking it, so the
#: repeated per-call sync after a problem is first seen is removed. ``debug`` mode
#: (tests / dev) always checks and raises, bypassing the latch.
#: NOTE (RLRP-769): the checks only run for the additive log-space operators
#: (``'log-add'`` / ``'log-sub'``), the only ones that can turn finite inputs into
#: ``-inf``/``NaN`` through ``log(w)``. Call sites using a multiplicative operator on a
#: log-space quantity (e.g. the ``tempered`` tempering ``w_j * nll_j``) must call
#: :func:`check_finite` explicitly if they want the same guard on their inputs.
_NON_FINITE_REPORTED: set = set()

#: RLRP-783 ``A3`` -- per-name ON-DEVICE non-finite accumulator. Introduced by action ``A3`` of
#: the RLRC MTM-Pro models code optimization `.junie` plan
#: (``perf_RLRP-783_mtm_pro_models_code_optimization_plan_20260827.md``).
#: ``torch.isfinite(...).all()`` stays on the device (no ``if`` on a 0-dim tensor -> no
#: synchronisation); the accumulated flags are read with a SINGLE transfer at the reporting
#: interval by :func:`flush_non_finite_reports`. First-detection is preserved -- the warning is
#: DELAYED by at most one interval, never suppressed. ``debug`` mode is unchanged: it still
#: checks and raises immediately.
_NON_FINITE_FLAGS: dict = {}

#: RLRP-786 (``performance_mode: dev|fast``, FR8 of
#: ``perf_RLRP-786_cuda_graph_train_step_plan_20260919.md``) -- process-wide switch of the
#: NON-``debug`` guard above. ``True`` (default) = today's behaviour (``dev``): the on-device
#: ``isfinite().all()`` flag kernel runs at every guarded weighting call. ``False`` (``fast``, the
#: paper-run mode and the CUDA-graph capture precondition): no guard kernel is issued at all and
#: :func:`flush_non_finite_reports` has nothing to read. The ``debug=True`` path is NOT affected
#: (it still checks and raises). Pure diagnostics: the weighted values are identical either way.
_NON_FINITE_GUARD_ENABLED: bool = True


def set_non_finite_guard_enabled(enabled: bool) -> None:
    """Enable (``dev``) / disable (``fast``) the sync-free production non-finite guard (RLRP-786)."""
    global _NON_FINITE_GUARD_ENABLED
    _NON_FINITE_GUARD_ENABLED = bool(enabled)


def is_non_finite_guard_enabled() -> bool:
    return _NON_FINITE_GUARD_ENABLED


def check_finite(tensor: torch.Tensor, name: str, debug: bool = False) -> None:
    """Public entry point of the latched non-finite guard, for weighting call sites that do NOT
    use an additive log-space operator of :func:`weight_values` (RLRP-769).

    :param tensor: The tensor to check.
    :param name: Tensor name used in the console warning.
    :param debug: Raise instead of warning (tests / dev).
    """
    _check_finite(tensor, name, debug)


def _check_finite(tensor: torch.Tensor, name: str, debug: bool) -> None:
    """Report (and, in ``debug``, raise on) a non-finite tensor -- latched, sync-free.

    RLRP-783 ``A3``: the non-``debug`` production path no longer synchronises on every call.
    Instead of an ``if`` on ``torch.all(torch.isfinite(...))`` (which forces a device->host
    transfer of a 0-dim tensor), the finite flag is accumulated ON DEVICE per ``name`` and read
    once by :func:`flush_non_finite_reports`. The production warning is preserved (never
    deleted -- a prior review forbids it), only DELAYED by at most one reporting interval. The
    ``debug`` path is unchanged: it checks and raises immediately.
    """
    if debug:
        # Unchanged legacy path: immediate check + raise (tests / dev).
        if not torch.all(torch.isfinite(tensor)):
            nan_infinity_console_warning(name)
            raise AssertionError
        return
    if not _NON_FINITE_GUARD_ENABLED or name in _NON_FINITE_REPORTED:
        return
    flag = torch.isfinite(tensor).all()  # stays on device -- NO sync
    previous = _NON_FINITE_FLAGS.get(name)
    _NON_FINITE_FLAGS[name] = flag if previous is None else (previous & flag)


def flush_non_finite_reports() -> None:
    """Read the accumulated on-device non-finite flags and emit any pending warning.

    Permanent diagnostic plumbing. Introduced by action ``A3`` of the RLRC MTM-Pro models code
    optimization `.junie` plan
    (``perf_RLRP-783_mtm_pro_models_code_optimization_plan_20260827.md``).
    Call it once per logging interval / training step / epoch boundary: exactly ONE device->host
    transfer for all tracked tensors, replacing the 3 synchronisations per weighting call that
    the ``debug=False`` path used to incur. First-detection is preserved -- a ``name`` seen
    non-finite is reported (once) and added to :data:`_NON_FINITE_REPORTED`, matching the legacy
    latch semantics.
    """
    if not _NON_FINITE_FLAGS:
        return
    names = list(_NON_FINITE_FLAGS)
    all_finite = torch.stack([_NON_FINITE_FLAGS[n] for n in names]).cpu().tolist()
    _NON_FINITE_FLAGS.clear()
    for name, finite in zip(names, all_finite):
        if not finite and name not in _NON_FINITE_REPORTED:
            nan_infinity_console_warning(name)
            _NON_FINITE_REPORTED.add(name)


#: RLRP-769 -- the four weighting OPERATORS supported by :func:`weight_values`.
#: Each name states the ARITHMETIC that is performed, not an assumption about the data:
#:
#:  - ``"log-add"``      -> ``values + log(w)``. For LOG-space values (log-density, log-prob):
#:                          adding ``log(w)`` is equivalent to scaling the underlying density
#:                          by ``w``.
#:  - ``"log-sub"``      -> ``values - log(w)``. Same, with the inverse weight ``1/w``. This is
#:                          the operator to use on a NEGATIVE log-quantity (e.g. an NLL) when the
#:                          intent is to scale the underlying density by ``w``.
#:  - ``"scale"``        -> ``w * values``. The plain product. Use it whenever the desired result
#:                          is a proportional rescaling around the ORIGIN, whatever the nature of
#:                          ``values`` (this is also the TEMPERING operator on a log-quantity:
#:                          ``w * (-log p) = -log p**w``).
#:  - ``"shrink-down"``  -> ``values - (1 - w) * |values|``. A SIGNED downward shift by the
#:                          un-weighted fraction of the magnitude. Equals ``w * values`` for
#:                          positive values but AMPLIFIES negative ones (``(2 - w) * values``).
#:                          Legacy operator for quantities whose neutral reference is NOT the
#:                          origin, where "less weight" must always mean "lower value".
WEIGHTING_OPERATORS = ("log-add", "log-sub", "scale", "shrink-down")

#: Operators for which the non-finite guard runs inside :func:`weight_values` (a ``log(w)`` can
#: produce ``-inf``/``NaN`` silently). The purely multiplicative operators cannot introduce a
#: non-finite value out of finite inputs, so they are not gated -- call :func:`check_finite`
#: explicitly if the inputs themselves are suspect.
_LOG_OPERATORS = ("log-add", "log-sub")


def weight_values(
    values: torch.Tensor,
    values_weights: torch.Tensor,
    operator: str,
    debug: bool = False,
    pre_process_weight: Optional[Callable] = None,
) -> torch.Tensor:
    """Apply a weight tensor to a value tensor using an EXPLICITLY named operator.

    This is the primary entry point (RLRP-769). It supersedes :func:`apply_weights`, whose
    ``log_space`` / ``values_centred_around_zero`` boolean pair was misleading: those flags read
    as a DESCRIPTION OF THE DATA ("my values are in log space", "my values are centred around
    zero") while they actually SELECT AN ARITHMETIC OPERATOR. The two are not interchangeable --
    e.g. tempering a negative log-likelihood (a log-space quantity) requires the plain product,
    so the legacy call had to declare ``log_space=False`` on a log-space tensor to reach it.
    Here the caller names the operator it wants and nothing is implied about ``values``.

        >>> weight_values(torch.tensor([0.5]), torch.tensor([0.5]), operator="scale")
        tensor([0.2500])
        >>> weight_values(torch.tensor([-0.5]), torch.tensor([0.5]), operator="scale")
        tensor([-0.2500])
        >>> weight_values(torch.tensor([0.5]), torch.tensor([0.5]), operator="shrink-down")
        tensor([0.2500])
        >>> weight_values(torch.tensor([-0.5]), torch.tensor([0.5]), operator="shrink-down")
        tensor([-0.7500])

    Broadcasting: a ``(E, D)`` weight tensor against ``(E, B, D)`` values is reshaped to
    ``(E, 1, D)``; otherwise, when only the trailing dim matches, the weights are collapsed with
    ``mean(dim=0)`` (a ``RuntimeWarning`` is raised when that collapse discards a non-uniform
    per-row structure, cf. RLRP-761 ``S9.8`` blocker ``B4`` class).

    :param values: Input tensor.
    :param values_weights: Weight tensor.
    :param operator: One of `WEIGHTING_OPERATORS`, i.e. the arithmetic to apply:
        `'log-add'` -> `values + log(w)`, `'log-sub'` -> `values - log(w)`,
        `'scale'` -> `w * values`, `'shrink-down'` -> `values - (1 - w) * |values|`.
    :param debug: Raise (instead of warning to console) on NaN/infinity.
    :param pre_process_weight: A function applied to `values_weights` before weighting.
    :return: A weighted value tensor.
    """
    if operator not in WEIGHTING_OPERATORS:
        raise ValueError(
            f"Unknown weighting operator {operator!r}, expected one of {WEIGHTING_OPERATORS}"
        )
    return _apply_weighting_operator(
        values,
        values_weights=values_weights,
        operator=operator,
        debug=debug,
        pre_process_weight=pre_process_weight,
    )


def flags_to_operator(
    log_space: bool,
    values_centred_around_zero: bool = False,
    apply_minus_log: bool = False,
) -> str:
    """Map the legacy :func:`apply_weights` boolean-flag triple to a named operator (RLRP-769).

    Lets call sites migrate to :func:`weight_values` while forwarding the same
    ``(log_space, values_centred_around_zero, apply_minus_log)`` flags they already receive.

    :param log_space: Select an additive log-space operator (`'log-add'`/`'log-sub'`).
    :param values_centred_around_zero: With `log_space=False`, select `'scale'` (plain product)
        instead of `'shrink-down'`.
    :param apply_minus_log: With `log_space=True`, select `'log-sub'` instead of `'log-add'`.
    :return: One of `WEIGHTING_OPERATORS`.
    """
    if log_space:
        return "log-sub" if apply_minus_log else "log-add"
    return "scale" if values_centred_around_zero else "shrink-down"


@deprecated(
    reason=(
        "Use `weight_values` with an explicit `operator` instead (see `flags_to_operator` "
        "to translate the legacy flags) (RLRP-769)."
    )
)
def apply_weights(
    values: torch.Tensor,
    values_weights: torch.Tensor,
    log_space: bool,
    values_centred_around_zero: bool = False,
    apply_minus_log: bool = False,
    debug: bool = False,
    pre_process_weight: Optional[Callable] = None,
) -> torch.Tensor:
    """LEGACY boolean-flag front-end of :func:`weight_values` -- prefer the latter (RLRP-769).

    .. deprecated:: RLRP-769
        Use :func:`weight_values` with an explicit ``operator`` (see :func:`flags_to_operator`
        to translate the legacy flags). Kept as a thin, behaviour-preserving front-end and will
        be removed in a future release.

    The flags do NOT describe `values`, they SELECT the arithmetic operator:

    ===============================================  =================  ==============================
    flags                                            operator           arithmetic
    ===============================================  =================  ==============================
    `log_space=True, apply_minus_log=False`          `'log-add'`        `values + log(w)`
    `log_space=True, apply_minus_log=True`           `'log-sub'`        `values - log(w)`
    `log_space=False, centred_around_zero=True`      `'scale'`          `w * values`
    `log_space=False, centred_around_zero=False`     `'shrink-down'`    `values - (1 - w) * |values|`
    ===============================================  =================  ==============================

    Beware the last row: it is NOT a product. It matches `w * values` for positive values but
    AMPLIFIES negative ones, which is what the doctests below show:

        >>> apply_weights(values=torch.tensor([0.5]),values_weights=torch.tensor([0.5]),log_space=False,values_centred_around_zero=False)
        tensor([0.2500])
        >>> apply_weights(values=torch.tensor([-0.5]),values_weights=torch.tensor([0.5]),log_space=False,values_centred_around_zero=False)
        tensor([-0.7500])

    With `values_centred_around_zero=True` the operator is the plain product:

        >>> apply_weights(values=torch.tensor([0.5]),values_weights=torch.tensor([0.5]),log_space=False,values_centred_around_zero=True)
        tensor([0.2500])
        >>> apply_weights(values=torch.tensor([-0.5]),values_weights=torch.tensor([0.5]),log_space=False,values_centred_around_zero=True)
        tensor([-0.2500])

    :param values: Input tensor.
    :param values_weights: Weight tensor.
    :param log_space: Select an additive log-space operator (`'log-add'`/`'log-sub'`) instead of
        a normal-space one. It is a CHOICE OF OPERATOR, not a statement about `values`.
    :param values_centred_around_zero: Select the plain product `'scale'` instead of the signed
        `'shrink-down'` operator. Only used with 'log_space=False'.
    :param apply_minus_log: Select `'log-sub'` (`values - log w`) instead of `'log-add'`
        (`values + log w`). Only used with 'log_space=True'.
    :param debug: Print to console NaN/infinity encountered warning.
    :param pre_process_weight: A function that will be applied to values_weights before weighting.
    :return: A weighted value tensor.
    """
    return _apply_weighting_operator(
        values,
        values_weights=values_weights,
        operator=flags_to_operator(
            log_space=log_space,
            values_centred_around_zero=values_centred_around_zero,
            apply_minus_log=apply_minus_log,
        ),
        debug=debug,
        pre_process_weight=pre_process_weight,
    )


def _apply_weighting_operator(
    values: torch.Tensor,
    values_weights: torch.Tensor,
    operator: str,
    debug: bool,
    pre_process_weight: Optional[Callable],
) -> torch.Tensor:
    """Shared implementation of :func:`weight_values` and :func:`apply_weights`."""
    log_space = operator in _LOG_OPERATORS

    if isinstance(values, np.ndarray):
        warnings.warn(
            "[RLRC torch-first] 'values' is a numpy ndarray in the weighting helper "
            "(`weight_values`/`apply_weights`). Update the upstream caller to pass torch "
            "tensors directly.",
            DeprecationWarning,
            stacklevel=3,
        )
        values = torch.from_numpy(values).to(values_weights.device)

    if log_space:
        # # .... Check that values_weights is a probalility .......................................
        # assert values_weights.min() >= 0.0, (
        #     f"values_weights tensor is not normalized: " f"{values_weights.min()=}  ! >= 0.0"
        # )
        # assert values_weights.max() <= 1.0, (
        #     f"values_weights tensor is not normalized: " f"{values_weights.max()=}  ! <= 1.0"
        # )
        #
        # if values_weights.ndim > 1:
        #     values_weights_sum = values_weights.sum(-1).mean()
        #     assert np.allclose(values_weights_sum.item(), 1.0), (
        #         f"{values_weights=} tensor is not normalized over ensemble dimensions: "
        #         f"{values_weights.sum(-1)} != {torch.ones(values_weights_sum.shape)=}"
        #     )
        # else:
        #     assert np.allclose(values_weights.sum().item(), 1.0), (
        #         f"{values_weights=} tensor is not normalized: {values_weights.sum()}  != " f"1.0"
        #     )

        # .... Check that both tensors have finite values .........................................
        _check_finite(values, "values", debug)
        _check_finite(values_weights, "values_weights", debug)

    if pre_process_weight:
        values_weights = pre_process_weight(values_weights)

    if (
        values.ndim == 3
        and values_weights.ndim == 2
        and values.shape[0] == values_weights.shape[0]
    ):
        values_weights = values_weights.reshape(
            values_weights.shape[0], 1, values_weights.shape[1]
        )
    elif values.shape[-1] == values_weights.shape[-1]:
        # RLRP-761 S9.8 -- the `mean(dim=0)` collapse is the same silent-failure
        # class as blocker `B4`: a per-ROW weight structure (per-feature `[D]`
        # collapsed to a SCALAR, or per-ensemble-member `[E, D]` collapsed to
        # `[D]`) is discarded whenever it is not uniform along dim 0. It is a
        # legitimate reduction only when the rows already agree; otherwise the
        # caller believes it is weighting per-row and it is not. Warn (once) so
        # the collapse becomes observable instead of a mysterious near-uniform
        # result -- the behaviour itself is preserved.
        if values_weights.shape[0] > 1:
            row0 = values_weights[:1]
            if not torch.allclose(values_weights, row0.expand_as(values_weights)):
                warnings.warn(
                    "weight_values: a non-uniform weight tensor is being reduced "
                    "by `mean(dim=0)` before broadcasting, so its per-row "
                    "structure is DISCARDED (RLRP-761 S9.8, blocker `B4` class). "
                    "If per-row weighting was intended, reshape the weights to "
                    "broadcast against `values` instead of relying on this "
                    "collapse.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        values_weights = values_weights.mean(dim=0)

    if operator == "shrink-down":
        # `values - (1 - w) * |values|` (spelled out with sign/abs for numerical symmetry).
        values_sign = torch.sign(values)
        abs_values = torch.abs(values)
        weighted_abs_values = abs_values * values_weights
        abs_delta_values = abs_values - weighted_abs_values
        weighted_values = values_sign * (abs_values - values_sign * abs_delta_values)
    elif operator == "scale":
        weighted_values = values_weights * values
    else:
        if operator == "log-sub":
            weighted_values = values - torch.log(values_weights)
        else:
            weighted_values = values + torch.log(values_weights)

        _check_finite(weighted_values, "weighted_values", debug)

    return weighted_values
