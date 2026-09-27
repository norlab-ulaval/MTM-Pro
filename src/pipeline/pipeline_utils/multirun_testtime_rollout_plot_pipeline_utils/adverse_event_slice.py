# coding=utf-8
"""RLRP-761 ``S6.2`` — the ``M2`` adverse-event metric slice.

Why this exists
---------------
The RLRP-761 A/B protocol scores the ``standard`` vs ``standard_symmetric``
question on RAW-space metrics. But its central claim — that the
terrain-vibration triplet ``(linear_vels.z, angular_vels.x, angular_vels.y)``
carries the *rare adverse events* the model must capture (traction loss on a
rock, weight transfer in an aggressive turn, an aggressive UAV manoeuvre) — is
NOT measurable by any existing metric: a whole-trajectory MAE is dominated by
the nominal regime, where those channels are near-pure vibration noise. A
configuration that gets the rare excursions right and the nominal vibration
slightly worse is INDISTINGUISHABLE from the opposite trade-off.

``M2`` closes that gap by scoring the error restricted to the windows where the
GROUND TRUTH actually moves on those channels: the top ``top_fraction`` of
timesteps ranked by ``‖Δ(vz, wx, wy)‖`` (first difference over time), dilated by
``window`` steps on each side so the model's *response* to the event — not only
its instant — is inside the slice.

Design notes
------------
- The slice is derived from the GROUND TRUTH only, so it is identical across
  every A/B cell and the comparison stays fair.
- The channels are resolved BY NAME against each recorded run's saved
  ``environment.obs_dims`` (never positionally). This is the same defect class
  RLRP-758 fixed in ``deploy_drift_eval``: positional slicing silently labels
  ``angular_vels.*`` as ``attitude.*`` whenever a non-suffix obs block is
  disabled.
- Everything here is read-only post-processing over already-recorded rollouts:
  no model, no GPU.
"""
from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import omegaconf

logger = logging.getLogger(__name__)

#: The UGV/UAV terrain-vibration + weight-transfer triplet (RLRP-761 report).
DEFAULT_ADVERSE_EVENT_DIMS: tuple[str, ...] = (
    "linear_vels.z",
    "angular_vels.x",
    "angular_vels.y",
)


@dataclass(frozen=True)
class AdverseEventSliceSpec:
    """Resolved ``show.adverse_event_slice`` configuration.

    :ivar feature_dims: obs-dim NAMES defining the event magnitude channel set.
    :ivar top_fraction: fraction of timesteps retained as "event" (``0 < f <= 1``).
    :ivar window: symmetric dilation (in timesteps) applied around every
        selected timestep, so the model's response to the event is scored too.
    :ivar metric_key: which recorded per-timestep error array to slice.
    :ivar min_slice_steps: a rollout contributing fewer than this many slice
        timesteps is skipped (a 3-step slice is not a measurement).
    """

    feature_dims: tuple[str, ...] = DEFAULT_ADVERSE_EVENT_DIMS
    top_fraction: float = 0.05
    window: int = 2
    metric_key: str = "mae"
    min_slice_steps: int = 5

    def __post_init__(self) -> None:
        if not self.feature_dims:
            raise ValueError(
                "`show.adverse_event_slice.feature_dims` is empty: the adverse-event "
                "slice has no channel to rank timesteps by."
            )
        if not (0.0 < float(self.top_fraction) <= 1.0):
            raise ValueError(
                f"`show.adverse_event_slice.top_fraction` must be in ]0, 1], got "
                f"{self.top_fraction!r}."
            )
        if int(self.window) < 0:
            raise ValueError(
                f"`show.adverse_event_slice.window` must be >= 0, got {self.window!r}."
            )
        if self.metric_key not in ("mae", "l2_norm"):
            raise ValueError(
                f"`show.adverse_event_slice.metric_key` must be 'mae' or 'l2_norm', "
                f"got {self.metric_key!r}."
            )


@dataclass
class AdverseEventGroupScore:
    """Per-group ``M2`` aggregate."""

    group_key: str
    label: str
    n_rollouts: int = 0
    n_skipped: int = 0
    slice_steps: int = 0
    total_steps: int = 0
    slice_error: float = float("nan")
    overall_error: float = float("nan")
    skipped_reasons: list[str] = field(default_factory=list)

    @property
    def amplification(self) -> float:
        """``slice_error / overall_error`` — how much harder the events are.

        A value of ``1`` means the adverse windows are no harder than the
        nominal regime; the RLRP-761 report predicts a value well above ``1``
        for every cell, and the A/B compares this ratio ACROSS cells.
        """
        if not np.isfinite(self.overall_error) or self.overall_error == 0.0:
            return float("nan")
        return float(self.slice_error / self.overall_error)


def resolve_adverse_event_slice_cfg(cfg: Any) -> Optional[AdverseEventSliceSpec]:
    """Read ``show.adverse_event_slice`` and return the resolved spec, or ``None``.

    Absent block or ``render: false`` -> ``None`` (strict no-op: the historical
    behaviour of every existing plot config is preserved untouched).
    """
    show = cfg.get("show", None) if hasattr(cfg, "get") else None
    if show is None:
        return None
    raw = show.get("adverse_event_slice", None)
    if raw is None:
        return None
    if not bool(raw.get("render", False)):
        return None

    feature_dims = raw.get("feature_dims", None)
    if feature_dims is None:
        feature_dims = DEFAULT_ADVERSE_EVENT_DIMS
    else:
        feature_dims = tuple(str(each) for each in feature_dims)

    return AdverseEventSliceSpec(
        feature_dims=tuple(feature_dims),
        top_fraction=float(raw.get("top_fraction", 0.05)),
        window=int(raw.get("window", 2)),
        metric_key=str(raw.get("metric_key", "mae")),
        min_slice_steps=int(raw.get("min_slice_steps", 5)),
    )


def load_recorded_obs_dims(experiment_base: str) -> Optional[list[str]]:
    """Read ``environment.obs_dims`` from a recorded run's saved Hydra config.

    :return: the ordered obs-dim names, or ``None`` when the run predates the
        feature contract / has no saved config (caller degrades gracefully).
    """
    hydra_cfg_path = os.path.join(str(experiment_base), ".hydra", "config.yaml")
    if not os.path.isfile(hydra_cfg_path):
        return None
    try:
        experiment_cfg = omegaconf.OmegaConf.load(hydra_cfg_path)
        obs_dims = experiment_cfg.environment.obs_dims
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Adverse-event slice: cannot read `environment.obs_dims` from %s (%s).",
            hydra_cfg_path,
            exc,
        )
        return None
    if obs_dims is None:
        return None
    return [str(each) for each in obs_dims]


def resolve_adverse_event_dim_indices(
    obs_dims: Sequence[str], feature_dims: Sequence[str]
) -> list[int]:
    """Map the configured event channel NAMES to column indices.

    :raises ValueError: when a configured channel is absent from the recorded
        layout. Fail fast rather than silently scoring a different channel set
        — a slice computed on the wrong columns answers a different question
        while looking perfectly plausible.
    """
    resolved = [str(each) for each in obs_dims]
    missing = [name for name in feature_dims if name not in resolved]
    if missing:
        raise ValueError(
            f"Adverse-event slice: channel(s) {missing} are not part of the recorded "
            f"`environment.obs_dims` {resolved}. Either the run was recorded with a "
            f"different feature layout, or `show.adverse_event_slice.feature_dims` is "
            f"stale."
        )
    return [resolved.index(name) for name in feature_dims]


def _as_time_major(target: Any, n_timesteps: int) -> Optional[np.ndarray]:
    """Return ``target`` as a ``(T, D)`` float array aligned with the error array.

    The recorded ``target`` orientation is not contractually pinned, so the time
    axis is identified by matching ``n_timesteps`` instead of being assumed.
    """
    if target is None:
        return None
    array = np.asarray(
        target.detach().cpu().numpy() if hasattr(target, "detach") else target
    )
    if array.ndim != 2:
        return None
    if array.shape[0] == n_timesteps:
        return array.astype(np.float64, copy=False)
    if array.shape[1] == n_timesteps:
        return array.T.astype(np.float64, copy=False)
    return None


def compute_adverse_event_mask(
    target_time_major: np.ndarray,
    dim_indices: Sequence[int],
    top_fraction: float,
    window: int,
) -> np.ndarray:
    """Boolean ``(T,)`` mask selecting the adverse-event timesteps.

    The magnitude is ``‖x[t] - x[t-1]‖₂`` over the selected channels (``0`` at
    ``t = 0``), each channel first divided by its own std over the trajectory so
    a single high-variance channel cannot monopolise the ranking (the triplet
    spans an order of magnitude: ``σ_vz ≈ 0.013`` vs ``σ_wy ≈ 0.034``).
    """
    columns = target_time_major[:, list(dim_indices)]
    scale = np.std(columns, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    delta = np.diff(columns / scale, axis=0)
    magnitude = np.zeros(columns.shape[0], dtype=np.float64)
    magnitude[1:] = np.linalg.norm(delta, axis=1)
    magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)

    n_selected = max(1, int(round(float(top_fraction) * magnitude.shape[0])))
    n_selected = min(n_selected, magnitude.shape[0])
    threshold_idx = np.argsort(magnitude)[-n_selected:]
    mask = np.zeros(magnitude.shape[0], dtype=bool)
    mask[threshold_idx] = True

    if window > 0:
        dilated = mask.copy()
        for shift in range(1, int(window) + 1):
            dilated[shift:] |= mask[:-shift]
            dilated[:-shift] |= mask[shift:]
        mask = dilated
    return mask


def _to_float_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(
        value.detach().cpu().numpy() if hasattr(value, "detach") else value
    ).astype(np.float64, copy=False)


def _error_array(metric: Any, metric_key: str) -> Optional[np.ndarray]:
    """The per-timestep error to slice, preferring the OBS-space error (RLRP-761 S8.16 A).

    On robotic-3D the recorded ``mae`` is the 3-D world-POSE error, which is not what
    the obs-space slice should measure. When the producer recorded an obs-space
    per-step error (``obs_mae``), use it for ``metric_key='mae'``; otherwise fall back
    to the named field (the math env, where state == obs, records it obs-space already).
    """
    if metric_key == "mae":
        obs_error = _to_float_array(getattr(metric, "obs_mae", None))
        if obs_error is not None and obs_error.size > 0:
            return obs_error.reshape(-1)
    array = _to_float_array(getattr(metric, metric_key, None))
    if array is None:
        return None
    return array.reshape(-1)


def _target_array(metric: Any) -> Any:
    """The obs-space GT trajectory used to rank timesteps (RLRP-761 S8.16 A).

    Prefer the obs-space GT (``obs_target``) recorded at rollout time; fall back to the
    legacy ``target`` (obs-space on the math env, 3-D pose on robotic-3D — the latter is
    rejected downstream by the ``len(obs_dims)`` width check, which is the intended
    fail-loud behaviour).
    """
    obs_target = getattr(metric, "obs_target", None)
    if obs_target is not None:
        return obs_target
    return getattr(metric, "target", None)


def compute_adverse_event_scores(
    groups: dict[str, Any], spec: AdverseEventSliceSpec
) -> list[AdverseEventGroupScore]:
    """Score every group on the ``M2`` adverse-event slice.

    :param groups: the per-group dict returned by ``compute_per_group_metric``.
    :param spec: the resolved slice configuration.
    :return: one :class:`AdverseEventGroupScore` per group, in group order.
    """
    scores: list[AdverseEventGroupScore] = []
    obs_dims_cache: dict[str, Optional[list[str]]] = {}

    for group_key, group in groups.items():
        score = AdverseEventGroupScore(
            group_key=str(group_key), label=str(group.get("label", group_key))
        )
        slice_sum = 0.0
        overall_sum = 0.0

        for each_metric in group.get("_original_metrics", []) or []:
            error = _error_array(each_metric, spec.metric_key)
            if error is None or error.size == 0:
                score.n_skipped += 1
                score.skipped_reasons.append("no recorded error array")
                continue

            experiment_base = getattr(each_metric, "_experiment_base", None)
            if experiment_base not in obs_dims_cache:
                obs_dims_cache[experiment_base] = (
                    load_recorded_obs_dims(experiment_base)
                    if experiment_base is not None
                    else None
                )
            obs_dims = obs_dims_cache[experiment_base]
            if obs_dims is None:
                score.n_skipped += 1
                score.skipped_reasons.append("no recorded `environment.obs_dims`")
                continue

            target = _as_time_major(_target_array(each_metric), error.size)
            if target is None:
                score.n_skipped += 1
                score.skipped_reasons.append(
                    "recorded `target` missing or not alignable with the error array"
                )
                continue
            if target.shape[1] != len(obs_dims):
                score.n_skipped += 1
                score.skipped_reasons.append(
                    f"recorded target width {target.shape[1]} != "
                    f"len(obs_dims)={len(obs_dims)}"
                )
                continue

            dim_indices = resolve_adverse_event_dim_indices(obs_dims, spec.feature_dims)
            mask = compute_adverse_event_mask(
                target, dim_indices, spec.top_fraction, spec.window
            )
            finite = np.isfinite(error)
            selected = mask & finite
            if int(selected.sum()) < spec.min_slice_steps:
                score.n_skipped += 1
                score.skipped_reasons.append(
                    f"slice too short ({int(selected.sum())} < {spec.min_slice_steps})"
                )
                continue

            score.n_rollouts += 1
            score.slice_steps += int(selected.sum())
            score.total_steps += int(finite.sum())
            slice_sum += float(error[selected].sum())
            overall_sum += float(error[finite].sum())

        if score.slice_steps > 0:
            score.slice_error = slice_sum / score.slice_steps
        if score.total_steps > 0:
            score.overall_error = overall_sum / score.total_steps
        scores.append(score)

    return scores


def render_adverse_event_table(
    scores: Sequence[AdverseEventGroupScore], spec: AdverseEventSliceSpec
) -> str:
    """Render the ``M2`` table (console / report artifact)."""
    header = (
        f"RLRP-761 M2 -- adverse-event slice "
        f"(top {spec.top_fraction:.1%} of |delta({', '.join(spec.feature_dims)})|, "
        f"+/-{spec.window} step dilation, metric={spec.metric_key})"
    )
    columns = (
        f"{'group':<28} {'n':>4} {'skip':>5} {'slice steps':>12} "
        f"{'slice err':>12} {'overall err':>12} {'amplif.':>9}"
    )
    lines = [header, "-" * len(columns), columns, "-" * len(columns)]
    for each in scores:
        coverage = (
            f"{each.slice_steps} ({each.slice_steps / each.total_steps:.1%})"
            if each.total_steps
            else f"{each.slice_steps}"
        )
        lines.append(
            f"{each.label[:28]:<28} {each.n_rollouts:>4} {each.n_skipped:>5} "
            f"{coverage:>12} {each.slice_error:>12.5g} "
            f"{each.overall_error:>12.5g} {each.amplification:>9.3f}"
        )
    lines.append("-" * len(columns))
    lines.append(
        "Note: the slice is derived from the GROUND TRUTH only, hence identical "
        "across A/B cells. Compare `slice err` across cells; `amplif.` states how "
        "much harder the adverse windows are than the nominal regime."
    )
    skipped = [each for each in scores if each.n_skipped]
    for each in skipped:
        reasons = sorted(set(each.skipped_reasons))
        lines.append(f"  ! {each.label}: {each.n_skipped} rollout(s) skipped: {reasons}")
    return "\n".join(lines)


def report_adverse_event_slice(
    cfg: Any, groups: dict[str, Any]
) -> Optional[list[AdverseEventGroupScore]]:
    """Resolve, compute and print the ``M2`` slice. ``None`` when disabled.

    A failure to score is NEVER allowed to take the plotting pipeline down —
    the slice is an added diagnostic, not a plotting prerequisite — EXCEPT for
    a channel-resolution failure, which is a configuration error the operator
    must see (see :func:`resolve_adverse_event_dim_indices`).
    """
    spec = resolve_adverse_event_slice_cfg(cfg)
    if spec is None:
        return None
    scores = compute_adverse_event_scores(groups, spec)
    table = render_adverse_event_table(scores, spec)
    logger.info("\n%s", table)
    print(table)
    _warn_if_slice_empty(scores)
    return scores


def _warn_if_slice_empty(scores: Sequence[AdverseEventGroupScore]) -> None:
    """Fail loud (warn) when NO rollout was scored across ALL groups (RLRP-761 S8.16).

    An all-skipped slice renders as a table of ``nan`` -- indistinguishable at a
    glance from a healthy result. The dominant cause on the robotic-3D env is a
    SCHEMA mismatch, not a config typo: the recorded metric stores the 3-D
    integrated world ``pose`` as ``target`` and the pose error as ``mae``, whereas
    the slice needs the OBS-space GT (``len == obs_dims``) and an obs-space
    per-timestep error. The slice therefore silently skips every rollout. Surface
    it explicitly rather than letting a ``nan`` table pass for a measurement.
    """
    if not scores:
        return
    if any(each.n_rollouts > 0 for each in scores):
        return
    if not any(each.n_skipped > 0 for each in scores):
        return  # nothing to score at all (empty groups) -- a different, benign case
    reasons = sorted(
        {reason for each in scores for reason in each.skipped_reasons}
    )
    warnings.warn(
        "RLRP-761 M2 adverse-event slice scored ZERO rollouts across every group: "
        "the slice produced no measurement (the rendered table is all `nan`). "
        f"Dominant skip reason(s): {reasons}. On the robotic-3D env this is the "
        "expected pose-vs-obs SCHEMA gap: the recorded `target`/`mae` are the 3-D "
        "integrated world pose, while the slice needs the obs-space GT trajectory "
        "(len == environment.obs_dims) and an obs-space per-timestep error. Runs "
        "recorded after RLRP-761 S8.16 (option A) carry these as `obs_target` / "
        "`obs_mae` and score natively; a run reaching this warning predates those "
        "fields (e.g. the pre-S8.16 S7 cells) and must be re-scored via a "
        "deploy-only re-run, as on the math env (state == obs) it is scorable as-is.",
        RuntimeWarning,
        stacklevel=2,
    )
