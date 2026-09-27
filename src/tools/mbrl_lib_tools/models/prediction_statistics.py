# coding=utf-8
"""Typed container for the deploy-time prediction-statistics channel (RLRP-761 ``P7``).

Structural cause of the RLRP-761 denormalization defect: ``model_state`` is an
untyped ``Dict[str, torch.Tensor]`` handed through five layers (model ->
:class:`OneDTransitionRewardModelV2` -> deployer -> adapter -> collector ->
metric dataclass), so **no layer owns the units of its contents**. Every
transform applied to ``next_obs`` was a fresh chance to forget its sibling
statistics — which is exactly how the defect arose.

:class:`PredictionStatistics` gives those tensors an owner: the unit ``space``,
the ``variance_approximation`` regime and the "no variance" ``variance_absent``
marker are set **at construction, by the producer**, and can be asserted at any
consumer boundary via :meth:`require_space`.

Back-compatibility (``P7.3``): ``model_state`` stays a plain mapping and the
container is exposed under an ADDITIONAL key (:data:`MODEL_STATE_KEY`) next to
the historical ``ensemble_means`` / ``ensemble_logvars`` entries, so no existing
consumer breaks.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

#: ``model_state`` key under which the container is exposed (``P7.3``).
MODEL_STATE_KEY: str = "prediction_statistics"

#: Canonical "no variance" logvar sentinel (RLRP-761 ``S10.4``).
#:
#: A deterministic / single-member model has no aleatoric spread to report, so its
#: logvar channel is filled with this fixed value as a MARKER, not a statistic
#: (``variance_absent=True``). It is a *semantic* bound (identical for
#: float32/float64), NOT a precision guard, and such a tensor must never be
#: rescaled by a denormalization Jacobian (risk ``Q-A``).
#:
#: This is the SINGLE source of truth: the four historical
#: ``_LOGVAR_MIN_LIMIT = math.log(1e-30)`` redefinitions across the model tree
#: (``ExponentialFamilyMLP``, ``GaussianMLPExtended``, ``OneDTransitionRewardModelV2``
#: and ``mbrl_lib_tools.models.utils``) now alias this constant, so the value can
#: no longer drift between the sites that EMIT the sentinel and the container that
#: INTERPRETS it.
SENTINEL_LOGVAR: float = math.log(1e-30)

#: The two unit spaces a statistics channel can be in, plus the legacy unknown.
SPACE_PHYSICAL: str = "physical"
SPACE_NORMALIZED: str = "normalized"
SPACE_UNKNOWN: str = "unknown"

#: Variance-transport regimes (RLRP-761 ``P1.1``).
VARIANCE_EXACT: str = "exact"
VARIANCE_LOCAL_LINEAR: str = "local_linear"


@dataclass(frozen=True)
class PredictionStatistics:
    """Deploy-time prediction statistics with an explicit unit contract.

    :ivar mean: per-member (or reduced) prediction mean.
    :ivar logvar: per-member prediction log-variance, or ``None``.
    :ivar space: ``physical`` | ``normalized`` | ``unknown`` — the unit space of
        :attr:`mean` / :attr:`logvar`. Set by the producer, never guessed.
    :ivar variance_approximation: ``exact`` when the denormalization is affine,
        ``local_linear`` when :attr:`logvar` was transported through a
        first-order linearization of a NON-affine denormalizer (``winsorized`` /
        ``quantile``, RLRP-761 ``P1.1`` regime B).
    :ivar variance_absent: ``True`` when :attr:`logvar` is the
        ``_LOGVAR_MIN_LIMIT`` **sentinel** (a "no variance" marker emitted for a
        deterministic / single-member model) rather than a statistic. Such a
        tensor must NEVER be rescaled (risk ``Q-A``).
    :ivar normalizer_type: the ``normalizer_type`` in effect, for traceability.
    """

    mean: torch.Tensor
    logvar: Optional[torch.Tensor] = None
    space: str = SPACE_UNKNOWN
    variance_approximation: str = VARIANCE_EXACT
    variance_absent: bool = False
    normalizer_type: Optional[str] = None

    @staticmethod
    def is_sentinel_logvar(logvar: Optional[torch.Tensor], atol: float = 1e-6) -> bool:
        """Whether ``logvar`` is the :data:`SENTINEL_LOGVAR` "no variance" marker.

        Returns ``True`` when every finite element is (approximately) the sentinel,
        i.e. the tensor is a deterministic / single-member fill rather than a
        statistic. ``None`` (no logvar at all) also counts as variance-absent.

        This is the one detector the producers should use to set
        :attr:`variance_absent`, so "what the sentinel is" and "how to recognise
        it" live in the same place (RLRP-761 ``S10.4``).
        """
        if logvar is None:
            return True
        finite = torch.isfinite(logvar)
        if not bool(finite.any()):
            return False
        return bool(torch.all(torch.abs(logvar[finite] - SENTINEL_LOGVAR) <= atol))

    def __post_init__(self) -> None:
        if self.space not in (SPACE_PHYSICAL, SPACE_NORMALIZED, SPACE_UNKNOWN):
            raise ValueError(
                f"PredictionStatistics.space must be one of "
                f"{(SPACE_PHYSICAL, SPACE_NORMALIZED, SPACE_UNKNOWN)}, got "
                f"{self.space!r}."
            )
        if self.variance_approximation not in (VARIANCE_EXACT, VARIANCE_LOCAL_LINEAR):
            raise ValueError(
                f"PredictionStatistics.variance_approximation must be one of "
                f"{(VARIANCE_EXACT, VARIANCE_LOCAL_LINEAR)}, got "
                f"{self.variance_approximation!r}."
            )

    # ---- Consumer-side contract (P7.2) --------------------------------------

    @property
    def is_physical(self) -> bool:
        """Whether the statistics are in physical (deployable) units."""
        return self.space == SPACE_PHYSICAL

    def require_space(self, expected: str, consumer: str) -> "PredictionStatistics":
        """Assert the unit space at a consumer boundary; return ``self``.

        Raises on a **known** mismatch and warns once on ``unknown`` (a legacy
        artifact that predates the contract), so a stale channel cannot silently
        enter a plot or a metric.
        """
        if self.space == expected:
            return self
        if self.space == SPACE_UNKNOWN:
            warnings.warn(
                f"[{consumer}] prediction statistics carry no recorded unit space "
                f"(legacy artifact); assuming '{expected}'. Regenerate the deploy "
                "phase to stamp it (RLRP-761 P4/P7).",
                RuntimeWarning,
                stacklevel=2,
            )
            return self
        raise ValueError(
            f"[{consumer}] prediction statistics are in '{self.space}' space but "
            f"'{expected}' is required. A NORMALIZED statistics channel compared "
            "against a PHYSICAL target is inflated by 1/sigma_target (1x under "
            "normalizer_type='standard', 3-40x under "
            "standard_symmetric_innovation). Re-run the deploy phase with a "
            "post-RLRP-761-P1 build (plan stage P5)."
        )

    # ---- Mapping interop (P7.3) ---------------------------------------------

    def attach_to(self, model_state: dict) -> dict:
        """Expose ``self`` under :data:`MODEL_STATE_KEY` without touching the rest."""
        model_state[MODEL_STATE_KEY] = self
        return model_state

    @staticmethod
    def from_model_state(
        model_state: Optional[Mapping[str, Any]]
    ) -> Optional["PredictionStatistics"]:
        """Read the container back, or ``None`` when absent (legacy mapping)."""
        if model_state is None:
            return None
        value = model_state.get(MODEL_STATE_KEY, None)
        return value if isinstance(value, PredictionStatistics) else None
