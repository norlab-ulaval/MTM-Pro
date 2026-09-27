"""Utilities for the new ``show`` options of the multirun testtime rollout
plot pipeline.

This module hosts the pure helpers behind the two new config keys:

- ``show.breakdown: [null | 'category' | 'trajectory']`` — replaces the legacy
  boolean ``show.category_breakdown``.
- ``show.general_length: [null | 'min' | 'max' | <int>]`` — replaces the legacy
  ``show.general_plot_right_xlim``. Gates/parameterizes the *general plot*
  stage and also drives the right xlim of that plot.
- ``show.selected_trajectories: [null | <list of trajectory_name>]`` (RLRP-841)
  — restricts the drift-rate plots to an explicitly named subset of the test
  trajectories (verbatim simulator-config spelling).

Legacy keys (``show.category_breakdown``, ``show.general_plot_right_xlim``)
are **not supported** — the codebase is unpublished so the migration is a
hard cutover. :func:`assert_no_legacy_show_keys` raises ``ValueError`` if
either legacy key is still present in a cfg.

All helpers here are pure: they don't touch matplotlib, Hydra, or any
filesystem. This keeps them trivially unit-testable outside of a live
pipeline run.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "LEGACY_SHOW_KEYS",
    "VALID_BREAKDOWN_MODES",
    "VALID_GENERAL_LENGTH_SPECS",
    "VALID_WARMUP_MODES",
    "assert_no_legacy_show_keys",
    "resolve_breakdown_mode",
    "resolve_general_length",
    "resolve_selected_trajectories",
    "resolve_warmup_mode",
    "select_trajectories_by_length",
]

#: Legacy ``show.*`` keys that have been removed. Presence of any of these in
#: a cfg is a hard error.
LEGACY_SHOW_KEYS: tuple[str, ...] = (
    "category_breakdown",
    "general_plot_right_xlim",
)

#: Valid values for ``show.breakdown``.
VALID_BREAKDOWN_MODES: tuple[Any, ...] = (None, "category", "trajectory")

#: Valid string values for ``show.general_length``. Integers are also valid
#: but not enumerated here.
VALID_GENERAL_LENGTH_SPECS: tuple[Any, ...] = (None, "min", "max")

#: Valid values for ``show.ground_truth_feed_warmup.mode``.
VALID_WARMUP_MODES: tuple[Any, ...] = (None, "discard", "reference_line")


def assert_no_legacy_show_keys(show_cfg: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if ``show_cfg`` contains any removed legacy key.

    The codebase is unpublished, so legacy keys ``category_breakdown`` /
    ``general_plot_right_xlim`` are removed outright rather than aliased.
    Any remaining reference (e.g. in a stale specialized cfg) is a loud
    cfg error.

    :param show_cfg: the ``cfg.show`` sub-mapping.
    :raises ValueError: if any legacy key is present. The message lists all
        offending keys so the operator can migrate them in one pass.
    """
    # Use ``in`` rather than ``.get(...) is not None`` so that e.g.
    # ``category_breakdown: null`` still trips the guard — any mention of the
    # stale key is an error.
    offenders = [k for k in LEGACY_SHOW_KEYS if k in show_cfg]
    if offenders:
        raise ValueError(
            "Removed legacy `show.*` cfg keys found: "
            f"{offenders}. These keys have been replaced — migrate to: "
            "`show.breakdown: [null|'category'|'trajectory']` (replaces "
            "`show.category_breakdown`) and "
            "`show.general_length: [null|'min'|'max'|<int>]` (replaces "
            "`show.general_plot_right_xlim`)."
        )


def resolve_breakdown_mode(show_cfg: Mapping[str, Any]) -> Any:
    """Return the validated ``show.breakdown`` value.

    :param show_cfg: the ``cfg.show`` sub-mapping.
    :returns: one of ``None``, ``'category'``, ``'trajectory'``.
    :raises ValueError: if the value is not one of the valid modes.
    """
    # Default: no breakdown.
    mode = show_cfg.get("breakdown", None) if hasattr(show_cfg, "get") else None
    if mode not in VALID_BREAKDOWN_MODES:
        raise ValueError(
            f"Invalid `show.breakdown` value: {mode!r}. "
            f"Must be one of {list(VALID_BREAKDOWN_MODES)}."
        )
    return mode


def resolve_warmup_mode(show_cfg: Mapping[str, Any]) -> Any:
    """Return the validated ``show.ground_truth_feed_warmup.mode`` value.

    The mode selects how the ground-truth-feed warm-up interval
    (``0 .. ground_truth_feed_warmup_steps``) is treated on every time-axis
    plot:

    - ``None`` (default) — current behavior, no treatment applied.
    - ``'discard'`` — clamp the left x-limit to the warm-up boundary so the
      warm-up region is not shown.
    - ``'reference_line'`` — keep all data but draw a vertical marker at the
      warm-up boundary.

    The warm-up step count itself is never taken from config: it is always
    auto-derived from each experiment's persisted rollout config.

    :param show_cfg: the ``cfg.show`` sub-mapping.
    :returns: one of ``None``, ``'discard'``, ``'reference_line'``.
    :raises ValueError: if the value is not one of the valid modes.
    """
    # Default: no warm-up treatment.
    warmup_cfg = (
        show_cfg.get("ground_truth_feed_warmup", None)
        if hasattr(show_cfg, "get")
        else None
    )
    if warmup_cfg is None:
        mode = None
    elif hasattr(warmup_cfg, "get"):
        mode = warmup_cfg.get("mode", None)
    else:
        raise ValueError(
            "Invalid `show.ground_truth_feed_warmup` value: "
            f"{warmup_cfg!r}. Expected a mapping with a `mode` key."
        )
    if mode not in VALID_WARMUP_MODES:
        raise ValueError(
            f"Invalid `show.ground_truth_feed_warmup.mode` value: {mode!r}. "
            f"Must be one of {list(VALID_WARMUP_MODES)}."
        )
    return mode


def resolve_selected_trajectories(show_cfg: Mapping[str, Any]) -> list[str] | None:
    """Return the validated ``show.selected_trajectories`` list (RLRP-841).

    The key restricts the drift-rate plots (``generate_drift_rate_plot`` and
    ``generate_drift_rate_plot_per_train_epoch``) to an explicitly named
    subset of the test trajectories. Names must be spelled exactly as the
    ``trajectory_name`` entries of the simulator config
    (``data.test_InD_trajectory`` / ``data.test_OOD_trajectory``), e.g.
    ``test_trajectories/ellipse_1``. Membership against that list is checked
    downstream (``select_ground_truth_entries``); this helper only validates
    the cfg shape.

    :param show_cfg: the ``cfg.show`` sub-mapping.
    :returns: ``None`` when the key is absent or ``null`` (no filtering),
        else a plain ``list[str]`` in the operator's order.
    :raises ValueError: if the value is not a non-empty sequence of unique,
        non-empty strings.
    """
    selected = (
        show_cfg.get("selected_trajectories", None)
        if hasattr(show_cfg, "get")
        else None
    )
    if selected is None:
        return None

    # Accept ``list``/``tuple``/OmegaConf ``ListConfig`` but reject a bare
    # string (iterable of characters) and any other scalar.
    if isinstance(selected, (str, bytes, Mapping)) or not hasattr(selected, "__iter__"):
        raise ValueError(
            f"Invalid `show.selected_trajectories` value: {selected!r} (type "
            f"{type(selected).__name__}). Must be null or a list of "
            "trajectory_name strings."
        )

    names = list(selected)
    if not names:
        raise ValueError(
            "Invalid `show.selected_trajectories` value: []. Use null to "
            "disable the selection or list at least one trajectory_name."
        )

    bad_items = [n for n in names if not isinstance(n, str) or not n.strip()]
    if bad_items:
        raise ValueError(
            f"Invalid `show.selected_trajectories` item(s): {bad_items!r}. "
            "Every item must be a non-empty trajectory_name string."
        )

    names = [str(n) for n in names]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"Invalid `show.selected_trajectories`: duplicate entries "
            f"{duplicates!r}. Each trajectory_name must appear once."
        )
    return names


def resolve_general_length(
    spec: Any,
    trajectory_lengths: Sequence[int] | Iterable[int],
) -> int | None:
    """Resolve the ``show.general_length`` spec into a concrete int cutoff.

    :param spec: one of ``None``, ``'min'``, ``'max'``, or a positive int.
    :param trajectory_lengths: iterable of ground-truth trajectory lengths
        (number of timesteps) across all test trajectories. Only consulted
        when ``spec`` is ``'min'`` or ``'max'``.
    :returns: ``None`` if ``spec`` is ``None`` (general plot disabled), else
        a positive int used as both the length cutoff (trajectories with
        ``length >= result`` are selected) and the right xlim of the general
        plot.
    :raises ValueError: for unsupported spec values, non-positive ints, or
        empty ``trajectory_lengths`` when a ``'min'`` / ``'max'`` resolution
        is requested.
    """
    if spec is None:
        return None

    if isinstance(spec, bool):
        # Guard against ``bool`` sneaking in via YAML since ``bool`` is a
        # subclass of ``int``.
        raise ValueError(
            f"Invalid `show.general_length` value: {spec!r}. "
            "Booleans are not accepted; use null, 'min', 'max', or a "
            "positive integer."
        )

    if isinstance(spec, int):
        if spec <= 0:
            raise ValueError(
                f"Invalid `show.general_length` integer: {spec}. "
                "Must be strictly positive."
            )
        return spec

    if isinstance(spec, str):
        if spec not in ("min", "max"):
            raise ValueError(
                f"Invalid `show.general_length` string: {spec!r}. "
                "Must be one of 'min', 'max', or a positive integer."
            )
        lengths = list(trajectory_lengths)
        if not lengths:
            raise ValueError(
                f"Cannot resolve `show.general_length`={spec!r}: "
                "no trajectory lengths available."
            )
        return min(lengths) if spec == "min" else max(lengths)

    raise ValueError(
        f"Invalid `show.general_length` value: {spec!r} (type "
        f"{type(spec).__name__}). Must be null, 'min', 'max', or a positive "
        "integer."
    )


def select_trajectories_by_length(
    trajectory_lengths: Mapping[str, int],
    length_cutoff: int,
) -> list[str]:
    """Return the list of trajectory names with ``length >= length_cutoff``.

    Ties (multiple trajectories at the same length) are all retained,
    matching the spec's "most likely be only one" phrasing for ``'max'``.

    :param trajectory_lengths: mapping ``{trajectory_name: length}``.
    :param length_cutoff: inclusive lower bound on trajectory length.
    :returns: list of trajectory names, sorted for determinism.
    """
    if length_cutoff <= 0:
        raise ValueError(
            f"`length_cutoff` must be strictly positive, got {length_cutoff}."
        )
    return sorted(
        name for name, length in trajectory_lengths.items()
        if length >= length_cutoff
    )
