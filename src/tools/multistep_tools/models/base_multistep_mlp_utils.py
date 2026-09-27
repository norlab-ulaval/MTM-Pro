# coding=utf-8
from typing import Tuple

import torch
from torch import nn as nn

from tools.torch_tools.functional import shifted_softplus


class CompositeLossAutomaticWeighting(nn.Module):
    """
    Automatic composite loss (aka multitask loss) weighting using task uncertainty.

    This class computes weighted composite losses so that heterogeneous-magnitude terms
    (e.g. ``L = MS_nll + SS_nll + GMS_IWAE``) each keep a meaningful gradient share and no term is
    dwarfed by another (a real risk: the MS horizon NLL is much larger than the SS NLL early in
    training, especially at long horizons). It implements task uncertainty-based weighting as
    described in "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and
    Semantics" (Kendall, Gal & Cipolla, 2018; https://arxiv.org/pdf/1705.07115). The per-task
    log-variances are learnable parameters optimized during training, balancing objectives
    dynamically.

    Parameter semantics (IMPORTANT):
        - ``auto_loss_log_vars[idx]`` is the per-task log task-noise variance ``s_i``. Under the
          ``noise_model="gaussian"`` weighting it means ``s = log sigma^2``; under
          ``noise_model="laplace"`` it means ``s = log b^2``. The same buffer is reused, so a
          given ``loss_id`` must always be weighted under the same ``noise_model``.
        - The ``0.5 * s_i`` term is the entropy/log-normaliser regulariser that prevents the
          trivial ``sigma_i -> inf`` (down-weight everything) solution. NOTE: the per-task weights
          ``0.5 * exp(-s_i)`` are inverse task-noise variances, NOT convex weights -- they are
          intentionally NOT sum-to-one constrained (the regulariser is the structural substitute
          for a constraint).

    ``noise_model`` (weighting task-noise model, set once at construction):
        - This is the noise model OF THE WEIGHTING SCHEME, conceptually INDEPENDENT from the
          dynamics model's predictive distribution (``dynamics_model.distribution_name``).

    ``weighting_scheme`` (how NLL terms are weighted, set once at construction):
        - ``"shifted_softplus_heuristic"``: ``L_i = 0.5 e^{-s} Phi(NLL_i) + 0.5 s`` where
          ``Phi = shifted_softplus`` lifts the (possibly negative) NLL to R_{>=0}. This is a
          well-motivated HEURISTIC generalisation of Kendall to log-likelihood terms, NOT a proper
          likelihood.
        - ``"tempered_likelihood"`` (default): ``L_i = e^{-s} NLL_i + 0.5 s`` (gaussian) /
          ``e^{-0.5 s} NLL_i + 0.5 s`` (laplace). This is the PROPER power-/tempered-likelihood
          objective (valid for negative NLL, no Phi lift); ``0.5 s`` is the exact tempered-density
          log-normaliser. See the .junie plan
          (refactor_composite_loss_auto_weighting_and_shifted_softplus_plan_20260623.md, RLRP-720).

    Introduced/refactored by the "improve CompositeLossAutomaticWeighting" .junie plan
    (refactor_composite_loss_auto_weighting_and_shifted_softplus_plan_20260623.md, ref RLRP-720).
    """

    # A7 (RLRP-788): diagnostic ``meta`` collection master switch. This module is the ONE
    # ``meta`` writer that is NOT a ``Model`` subclass, so it cannot inherit the flag; the
    # owning model's ``set_meta_collection_enabled`` setter propagates it here at setup time
    # (see ``mbrl.models.Model.set_meta_collection_enabled``). Default ``True`` -> today's
    # behaviour; ``forward`` is write-only w.r.t. ``meta`` (never reads it), so gating its
    # writes leaves the weighted-loss return bit-exact. Permanent diagnostic default.
    # Introduced by action ``A7`` of the RLRC meta-collection kill-switch `.junie` plan
    # (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
    _enable_meta_collection: bool = True

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        enable: bool = True,
        noise_model: str = "gaussian",
        weighting_scheme: str = "tempered_likelihood",
        ensemble_size: int = 1,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.enable = enable
        # Expected size of the leading (ensemble) dim of the train-mode `losses` tensor; set from
        # the owning model's `num_members`. `forward` always asserts the input's first dim matches
        # it (a structural guard). Defaults to 1 (single-member / non-ensembled).
        self.ensemble_size = ensemble_size

        assert noise_model in (
            "gaussian",
            "laplace",
        ), f"Unsupported noise_model {noise_model!r} (expected 'gaussian' or 'laplace')"
        assert weighting_scheme in (
            "shifted_softplus_heuristic",
            "tempered_likelihood",
        ), (
            f"Unsupported weighting_scheme {weighting_scheme!r} "
            f"(expected 'shifted_softplus_heuristic' or 'tempered_likelihood')"
        )
        self.noise_model = noise_model
        self.weighting_scheme = weighting_scheme

        # index 0: Single-step loss, index 1: Multi-step loss, index 2: Multi-step mixture loss
        self.auto_loss_weights_map = {
            "SS": 0,
            "MS": 1,
        }

        self.auto_loss_log_vars = nn.Parameter(
            torch.zeros(len(self.auto_loss_weights_map), dtype=dtype),
            requires_grad=True,
        )

    def extend(self, extend_loss_weighting_map: dict) -> None:
        """
        Initializes a module for managing weighted loss calculation.

        :param extend_loss_weighting_map:
            A dictionary for extending the predefined mapping of loss types.
            Additional loss types provided in this dictionary will be merged into the
            predefined map. Keys represent the loss type names, and values represent
            their respective indices (integers).
        """
        self.auto_loss_weights_map.update(extend_loss_weighting_map)

        self.auto_loss_log_vars = nn.Parameter(
            torch.zeros(
                len(self.auto_loss_weights_map),
                dtype=self.auto_loss_log_vars.dtype,
                device=self.auto_loss_log_vars.device,
            ),
            requires_grad=True,
        )
        return None

    def forward(
        self,
        losses: float | torch.Tensor,
        loss_id: str,
        meta: dict,
        are_log_prob_losses: bool = False,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate the uncertainty-based weighting for a single composite-loss term.
        Ref https://arxiv.org/pdf/1705.07115 (Kendall et al., 2018).

        The weighting ``noise_model`` and ``weighting_scheme`` are fixed properties of this module
        (set in ``__init__``); they are intentionally NOT per-call arguments so a given ``loss_id``
        is always weighted consistently (a per-call value could corrupt the shared
        ``auto_loss_log_vars`` semantics).

        :param losses: The per-task loss tensor, shape (E x B x ...) in train mode. For
            ``are_log_prob_losses=True`` this is the per-task NLL (may be negative).
        :param loss_id: e.g. 'SS'=Single-step, 'MS'=Multi-step, 'MS_MIX'=Multi-step mixture, ...
        :param meta: Metadata dictionary (mutated in place; receives the resolved weight).
        :param are_log_prob_losses: True if ``losses`` is a (log-prob) NLL term.
        :param debug: enable debug-only assertions.
        :return: (weighted_losses, meta).
        """

        if self.enable:
            idx = self.auto_loss_weights_map[loss_id]

            assert losses.dim() == 3, (
                f"Support for {losses.dim()}D input not implemented "
                f"(expected train-mode E x B x ...)."
            )
            # The leading dim MUST be the (unreduced) ensemble dim. The downstream
            # `reduce_probabilistic_compose_loss` SUMS over dim 0 as the ensemble axis, so a
            # mismatched first dim would silently corrupt the reduction.
            assert losses.shape[0] == self.ensemble_size, (
                f"Expected the leading (ensemble) dim to be ensemble_size="
                f"{self.ensemble_size}, got {losses.shape[0]} "
                f"(input shape {tuple(losses.shape)})."
            )
            # The feature dim MUST already be reduced to 1 upstream (each call site applies
            # `.mean(2, keepdim=True)` / `.unsqueeze(-1)` before this forward). This is what makes
            # the `0.5 * s` regulariser a per-task scalar (not a per-feature vector) and keeps the
            # documented reduction contract (see the weighted-term comment below) exact.
            assert losses.shape[-1] == 1, (
                f"Expected the feature dim to be already reduced to 1, got {losses.shape[-1]} "
                f"(input shape {tuple(losses.shape)}); call sites must feature-reduce "
                f"(`.mean(2, keepdim=True)`) before auto-weighting so the 0.5*s regulariser and "
                f"the 'no 1/batch_size factor' reduction contract hold."
            )

            s_i = self.auto_loss_log_vars[idx]

            if are_log_prob_losses and self.weighting_scheme == "tempered_likelihood":
                # Proper power-/tempered-likelihood (eq. 10): the per-task multiplier is the
                # INVERSE TEMPERATURE beta = exp(-s) applied DIRECTLY to the (possibly negative)
                # NLL -- NOT the Kendall residual-form 0.5*exp(-s). No Phi lift. The 0.5*s term is
                # the exact tempered-Gaussian log-normaliser (-0.5 log beta); for the laplace
                # noise model and the mixture/true_mixture SS term it is a documented approximation
                # (see plan sec 2.4f).
                auto_weights = torch.exp(-s_i)
                task_loss = losses
            else:
                # Kendall residual-form weight: gaussian 0.5*exp(-s) = 1/(2 sigma^2) (s=log sigma^2);
                # laplace exp(-0.5 s) = 1/b (s=log b^2).
                if self.noise_model == "gaussian":
                    auto_weights = 0.5 / torch.exp(s_i)
                else:  # laplace
                    auto_weights = 1 / torch.exp(0.5 * s_i)

                if are_log_prob_losses:
                    # Heuristic lift Phi(NLL) -> R_{>=0} (eq. 5). NOT a proper likelihood, but
                    # smooth/monotone; torch.exp(NLL) would be numerically unstable.
                    task_loss = shifted_softplus(losses)
                else:
                    # Already a residual-like, non-NLL term (projection / RC): weight linearly.
                    task_loss = losses

            # Weighted term: weight * task_loss + 0.5 * s (the 0.5 * s regulariser is the
            # log-normaliser; note 0.5 * log(sigma^2) = log(sigma)).
            #
            # Reduction contract (IMPORTANT -- this is what makes the 0.5 * s regulariser correct):
            #   - UPSTREAM: callers pass `losses` already FEATURE-reduced to shape (E, B, 1) (each
            #     call site applies `.mean(2, keepdim=True)` before this forward). The batch dim B
            #     is still UNREDUCED here.
            #   - `auto_weights` and `s_i` are per-task scalars, so `0.5 * s_i` is BROADCAST over
            #     all B batch slots (it appears B times along the batch axis).
            #   - DOWNSTREAM: `reduce_probabilistic_compose_loss` MEAN-reduces the batch dim (and
            #     feature dim) then SUMS the ensemble dim. Because the batch axis is AVERAGED (not
            #     summed), the B broadcast copies of `0.5 * s_i` collapse back to exactly ONE per
            #     ensemble member -- so NO `1/batch_size` factor is needed here.
            #   - This `0.5 * s_i` correctness therefore RELIES on the downstream batch reduction
            #     being a MEAN. If the batch axis were SUMMED downstream instead, the regulariser
            #     would be over-counted by B and a `1/batch_size` correction WOULD be required.
            #   - The ensemble dim is SUMMED downstream, so the regulariser (like the data term) is
            #     scaled by the ensemble size E -- ratio/optimum preserved, only absolute magnitude
            #     is E x.
            weighted_losses = auto_weights * task_loss + 0.5 * s_i

            # A7 (RLRP-788): gate the diagnostic write (and its ``.item()`` sync) behind the
            # meta-collection kill-switch; the weighted-loss return above is untouched
            # (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
            if self._enable_meta_collection:
                meta[f"{loss_id}_loss_auto_weighting"] = auto_weights.detach().item()
            if debug and are_log_prob_losses and self.weighting_scheme == (
                "shifted_softplus_heuristic"
            ):
                # The shifted-softplus-converted term must be non-negative.
                assert torch.all(
                    task_loss >= 0
                ), f"Converted losses should be non-negative, got {task_loss.min().item()}"

            return weighted_losses, meta
        else:
            # Disabled -> no weighting
            # A7 (RLRP-788): gate the diagnostic write behind the meta-collection kill-switch
            # (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
            if self._enable_meta_collection:
                meta[f"{loss_id}_loss_auto_weighting"] = 1.0
            return losses, meta
