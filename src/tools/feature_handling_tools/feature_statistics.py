# coding=utf-8
"""Per-feature dataset scale estimators (RLRP-761 stage S3).

Permanent framework module.

The RLRP-761 investigation established that, under a per-feature NLL with a free
log-variance, *rescaling the target* is nearly inert while an explicit
**per-feature loss weight** is the intervention that actually changes the
optimization. This module provides the statistics that weight is derived from,
measured on the dataset instead of hand-authored in YAML.

Three scales per feature, all trajectory-aware (a difference is **never** taken
across an episode boundary):

- ``state_std`` — the plain state std ``sigma``. What ``standard_symmetric``
  normalizes by, and the reason the terrain-vibration channels take over the
  target budget (root cause H1).
- ``delta_std`` — the one-step innovation std ``sigma_Delta``: the magnitude of
  what the model is actually asked to predict.
- ``noise_floor`` — the IRREDUCIBLE high-frequency component ``sigma_floor``,
  estimated from the **second** difference, which annihilates any locally-linear
  signal:

  ``Delta2 x_t = x_{t+1} - 2 x_t + x_{t-1}``, and for white noise ``e``,
  ``Var(Delta2 e) = 6 Var(e)`` — hence
  ``sigma_floor = 1.4826 * MAD(Delta2 x) / sqrt(6)``.

  The **MAD** form (rather than the std) is load-bearing for this research
  program: a genuine adverse event (traction loss on a rock, weight transfer in
  an aggressive turn) is a RARE LARGE excursion. It must inflate ``delta_std``
  without inflating the floor it is compared against — otherwise the very events
  of interest would raise their own bar and be weighted down.

The derived quantity:

``predictability = (delta_std / noise_floor) ** 2``

i.e. how much of a channel's one-step movement is *reducible*. During nominal
driving a vibration channel sits at ~1 floor unit and its weight is low; during
a rock strike the residual is many floor units wide and the loss reacts
strongly. Neither raw scaling (which hides the channel: ``sigma_Delta = 0.0023``
for ``linear_vels.z``) nor state-``sigma`` scaling (which lets nominal vibration
dominate the target budget) has that property.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Consistency constant of the MAD as a std estimator for Gaussian data.
_MAD_TO_STD = 1.4826
#: ``Var(x[t+1] - 2 x[t] + x[t-1]) = 6 Var(x)`` for white noise.
_SECOND_DIFF_VARIANCE_GAIN = 6.0
#: Absolute floor keeping the ``delta_std / noise_floor`` ratio finite.
_FLOOR_EPS = 1e-12


@dataclass(frozen=True)
class FeatureScales:
    """Per-feature dataset scales, indexed consistently with :attr:`dim_names`."""

    dim_names: Tuple[str, ...]
    state_std: np.ndarray
    delta_std: np.ndarray
    noise_floor: np.ndarray
    predictability: np.ndarray
    n_samples: int

    def as_mapping(self, key: str = "predictability") -> Dict[str, float]:
        """Return ``{dim_name: value}`` for one of the arrays.

        :param key: One of ``state_std`` / ``delta_std`` / ``noise_floor`` /
            ``predictability``.
        """
        values = getattr(self, key)
        return {
            name: float(values[i]) for i, name in enumerate(self.dim_names)
        }


def _as_sequences(
    source, n_features: Optional[int] = None
) -> List[np.ndarray]:
    """Coerce *source* into a list of ``(T, D)`` float64 trajectory arrays.

    Accepted forms:

    - a single ``(T, D)`` array (treated as ONE trajectory);
    - a sequence of ``(T, D)`` arrays (one per trajectory / episode);
    - a sequence of objects exposing ``observations`` (the TCT / test-env form).
    """
    if source is None:
        return []
    if isinstance(source, np.ndarray):
        arrays = [source]
    else:
        arrays = []
        for item in source:
            if hasattr(item, "observations"):
                item = np.asarray(item.observations)
            arrays.append(np.asarray(item))
    out: List[np.ndarray] = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if n_features is not None:
            if arr.shape[-1] < n_features:
                continue
            arr = arr[:, :n_features]
        out.append(arr)
    return out


def estimate_feature_scales(
    source,
    dim_names: Sequence[str],
    weight_floor: float = 0.0,
) -> Optional[FeatureScales]:
    """Estimate the per-feature dataset scales of RLRP-761 ``S3.1``.

    Deterministic and seedless: every statistic is an exact reduction over the
    provided data, accumulated in ``float64``. Differences are computed **within**
    each trajectory only.

    :param source: A ``(T, D)`` array, a sequence of such arrays (one per
        trajectory) or a sequence of objects exposing ``observations``.
    :param dim_names: The feature names, in column order. Its length fixes ``D``.
    :param weight_floor: Optional lower bound applied to ``noise_floor`` (use it
        to bound the ``predictability`` of an essentially noiseless channel).
    :returns: the scales, or ``None`` when no trajectory is long enough (a
        second difference needs at least 3 samples).
    """
    names = tuple(str(n) for n in dim_names)
    n_features = len(names)
    if n_features == 0:
        return None
    sequences = _as_sequences(source, n_features)
    sequences = [s for s in sequences if s.shape[0] >= 3]
    if not sequences:
        return None

    pooled = np.concatenate(sequences, axis=0)
    state_std = pooled.std(axis=0)

    deltas = np.concatenate([s[1:] - s[:-1] for s in sequences], axis=0)
    delta_std = deltas.std(axis=0)

    second = np.concatenate(
        [s[2:] - 2.0 * s[1:-1] + s[:-2] for s in sequences], axis=0
    )
    median = np.median(second, axis=0, keepdims=True)
    mad = np.median(np.abs(second - median), axis=0)
    noise_floor = _MAD_TO_STD * mad / np.sqrt(_SECOND_DIFF_VARIANCE_GAIN)
    noise_floor = np.maximum(noise_floor, max(float(weight_floor), _FLOOR_EPS))

    predictability = (delta_std / noise_floor) ** 2

    return FeatureScales(
        dim_names=names,
        state_std=state_std,
        delta_std=delta_std,
        noise_floor=noise_floor,
        predictability=predictability,
        n_samples=int(pooled.shape[0]),
    )
