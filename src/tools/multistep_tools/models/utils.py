# coding=utf-8
import torch


def _reduce_compose_loss(losses: torch.Tensor, feature_reduction: str) -> torch.Tensor:
    """
    Shared composite-loss reducer over the ``(E, B, F)`` train-mode loss tensor.

    Dimension convention (train mode): ``losses`` has shape ``(E, B, F)`` with
        - ``E``: the (unreduced) ensemble dimension (``model.num_members``),
        - ``B``: the (unreduced) batch dimension,
        - ``F``: the feature dimension (often already collapsed to ``1`` upstream, but NOT
          always -- some models do not reduce the feature axis in their own loss computation,
          which is why the reduction is (re)applied here).

    Reduction rationale (per dimension):
        - FEATURE (``F``): probabilistic -> ``mean``; deterministic -> ``sum``.
          A sum-of-log over independent feature dims is the joint log-likelihood of a
          (diagonal) multivariate distribution, so summing is the "correct" probabilistic
          aggregation. We nonetheless ``mean`` the feature axis in the probabilistic case to keep
          the NLL on a per-dimension scale commensurate with the fixed-coefficient regularisers it
          is combined with (e.g. the ``0.01 * (max_logvar - min_logvar)`` logvar-bound penalty and
          the composite auto-weighting ``0.5 * s`` log-normalisers): those added terms are
          dimension/batch-size-independent constants, so the data term must be too, otherwise their
          relative weight would silently drift with ``F``/``B``. The deterministic (pure MSE) loss
          carries no such fixed-coefficient companion term, so the legacy ``sum`` over features is
          retained.
        - BATCH (``B``): ALWAYS ``mean``. This is the standard Monte-Carlo estimate of the expected
          per-sample loss ``E_{x~data}[L]``, making the objective (and its gradient scale)
          INVARIANT to the minibatch size -- so learning-rate tuning transfers across batch sizes.
          Crucially, this is what makes the composite auto-weighting valid: the
          ``CompositeLossAutomaticWeighting`` ``0.5 * s`` regulariser is broadcast over the ``B``
          batch slots and is only recovered ONCE per ensemble member because the batch axis is
          averaged (a ``sum`` here would over-count it by ``B`` and require a ``1/batch_size``
          correction).
        - ENSEMBLE (``E``): ALWAYS ``sum``. Each ensemble member is an INDEPENDENT (bootstrap)
          model; summing gives every member a full-strength, independent gradient in a single
          backward pass (equivalent to training ``E`` separate models in parallel) instead of
          diluting each by ``1/E``. Both the data term and any regulariser are scaled by the same
          ``E``, so the ratio/optimum is preserved -- only the absolute loss magnitude is ``E x``.

    Reference (upstream): this mirrors the original ``mbrl-lib``
    ``GaussianMLP._nll_loss`` (probabilistic: ``gaussian_nll(...).mean((1, 2)).sum()``) and
    ``GaussianMLP._mse_loss`` (deterministic: ``mse_loss(...).sum((1, 2)).sum()``) reductions; see
    https://github.com/facebookresearch/mbrl-lib (``mbrl/models/gaussian_mlp.py``). NOTE: the only
    deviation from upstream is the BATCH axis of the deterministic path -- upstream ``mse_loss``
    SUMS over the batch, but we ``mean`` it (like the probabilistic path) so that the deterministic
    model can ALSO be combined with the batch-size-independent composite auto-weighting; for a
    single-term MSE this is a benign constant rescale (absorbable by the learning rate).

    :param losses: loss tensor of shape ``(E, B, F)``.
    :param feature_reduction: ``"mean"`` (probabilistic) or ``"sum"`` (deterministic), applied to
        the feature dimension.
    :return: scalar reduced loss.
    """
    if feature_reduction == "mean":
        losses = losses.mean(2)  # feature dim -> per-dimension average (probabilistic)
    elif feature_reduction == "sum":
        losses = losses.sum(2)  # feature dim -> joint log-likelihood sum (deterministic)
    else:
        raise ValueError(
            f"Unsupported feature_reduction {feature_reduction!r} "
            f"(expected 'mean' or 'sum')."
        )
    losses = losses.mean(1)  # batch dim -> batch-size-invariant Monte-Carlo mean
    losses = losses.sum()  # ensemble dim -> independent-member sum
    return losses


def reduce_deterministic_compose_loss(losses: torch.Tensor) -> torch.Tensor:
    """
    Reduce a DETERMINISTIC composite loss (e.g. MSE) over the ``(E, B, F)`` train-mode tensor:
    SUM over the feature dim, MEAN over the batch dim, SUM over the ensemble dim.

    The feature dim is summed (joint over output dims) while the batch dim is averaged so the loss
    stays batch-size-invariant -- which is what allows the deterministic model to be combined with
    the composite auto-weighting (its fixed-coefficient ``0.5 * s`` regulariser requires a
    batch-size-independent data term). See ``_reduce_compose_loss`` for the full per-dimension
    rationale and the ``mbrl-lib`` reference.

    :param losses: deterministic loss tensor of shape ``(E, B, F)``.
    :return: scalar reduced loss.
    """
    return _reduce_compose_loss(losses, feature_reduction="sum")


def reduce_probabilistic_compose_loss(losses: torch.Tensor) -> torch.Tensor:
    """
    Reduce a PROBABILISTIC composite loss (NLL) over the ``(E, B, F)`` train-mode tensor:
    MEAN over the feature dim, MEAN over the batch dim, SUM over the ensemble dim.

    The feature and batch dims are averaged so the NLL stays on a per-dimension, batch-size-
    invariant scale commensurate with the fixed-coefficient terms it is combined with (the
    ``0.01 * (max_logvar - min_logvar)`` logvar-bound penalty and the composite auto-weighting
    ``0.5 * s`` log-normalisers). The ensemble dim is summed so each independent member gets a
    full-strength gradient. See ``_reduce_compose_loss`` for the full per-dimension rationale and
    the ``mbrl-lib`` reference.

    :param losses: probabilistic (NLL) loss tensor of shape ``(E, B, F)``.
    :return: scalar reduced loss.
    """
    return _reduce_compose_loss(losses, feature_reduction="mean")
