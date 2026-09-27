# coding=utf-8
import torch
from torch.nn import functional as F


def shifted_softplus(
    x, beta: float = 0.05, offset_beta: float = 0.5, threshold: float = 80.0, origin=0.0
):
    """
    Applies a shifted softplus transformation to the input tensor.

    Smooth, monotone, NON-NEGATIVE map  Phi: R -> R_{>=0}  used to lift a (possibly negative)
    negative-log-likelihood into a non-negative, residual-like quantity for the Kendall
    uncertainty-weighting algebra (see CompositeLossAutomaticWeighting).

    Shift / offset math (IMPORTANT, the docstring previously mis-stated this):
        Phi(x) = max(0, softplus_beta(x + origin) - softplus_{offset_beta}(origin)),
        so at the origin  Phi(0) = ln(2) * (1/beta - 1/offset_beta).
        Therefore Phi(0) == 0 ONLY when offset_beta == beta. The defaults DELIBERATELY use
        DISTINCT betas (beta=0.05 < offset_beta=0.5 -> Phi(0) = 18*ln2 ~= 12.48 > 0). This is
        intentional and load-bearing: because Phi is clamped to >= 0, choosing offset_beta == beta
        (true zero anchor) would make Phi(x) <= 0 for EVERY x < 0 and thus clamp them to 0,
        ZEROING the gradient for all negative inputs. NLL inputs are routinely negative (sharp,
        well-fit densities), so a true zero anchor would silently kill their learning signal.
        The distinct-beta defaults push the clamp/zero-gradient region out to the deep-negative
        tail (~ x < -52.7 for the defaults), keeping a live gradient across the realistic NLL
        range while staying strictly positive there.

    Note:
        - Numerically stable and smooth everywhere; the final clamp(min=0) only engages on the
          deep-negative tail with the (intended) distinct-beta defaults.
        - Larger beta makes it closer to ReLU, smaller beta makes it smoother.

    :param x: Input tensor on which the shifted softplus operation will be applied.
    :param beta: The beta parameter for the softplus function, which controls the smoothness
        of the transition (smaller = smoother, larger -> ReLU). Defaults to 0.05.
    :param offset_beta: The beta parameter used for calculating the offset. Defaults to 0.5
        (> beta on purpose). Keep offset_beta > beta so the clamp/zero-gradient region stays in
        the deep-negative tail; setting offset_beta == beta gives a true Phi(0) = 0 anchor BUT
        zeroes the gradient for all negative inputs (see the shift/offset note above).
    :param threshold: The threshold value beyond which the softplus function switches to a
        linear approximation for numerical stability. Defaults to 80.0.
    :param origin: A constant value added to the input before applying the softplus
        transformation. Defaults to 0.0.
    :return: A tensor containing the shifted softplus transformed values, clamped to be
        non-negative.


    """
    offset = F.softplus(
        torch.zeros(1) + origin, beta=offset_beta, threshold=threshold
    ).item()

    ssp = F.softplus(x + origin, beta=beta, threshold=threshold) - offset

    # Clamp to guarantee non-negativity (handles numerical precision issues)
    ssp = torch.clamp(ssp, min=0.0)

    return ssp
