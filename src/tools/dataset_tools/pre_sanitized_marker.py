# coding=utf-8
"""Tri-state consolidated ``<dirname>.pre_sanitized`` marker for the dataset
pre-sanitization pipeline.

The marker is the synchronization contract between the **operator-side**
sanitizer (run locally before ``rsync`` to HPC) and the **HPC-side** HPO
workers. Its presence and content guarantee that
:func:`pipeline.pipeline_utils.robotic_env_pipeline_utils.dataset_pre_processing.sanitize_csv_data`
can take the read-only fast path (no ``pd.read_csv`` / ``df.to_csv``),
which is required for safe parallel HPO under joblib (``n_jobs > 1``).

Layout
------
A **single consolidated marker file per directory** records the state
of every CSV in that directory:

.. code-block:: text

    valid/
        valid.pre_sanitized            ← consolidated marker
        merged_2021-02-18-13-44-23_seg_2.csv
        merged_2021-02-18-16-53-35_seg_2.csv
        ...

This replaces the previous "one ``<csv>.pre_sanitized`` file per CSV"
layout that polluted every dataset directory with as many marker files
as CSVs.

Marker file format
------------------
Plain-text, UTF-8, line-oriented:

.. code-block:: text

    # Consolidated tri-state marker for directory: valid
    # Emitted by tools.dataset_tools.pre_sanitized_marker.
    # Each non-comment line maps a CSV filename to one of:
    #     {false, inprogress, true}
    merged_2021-02-18-13-44-23_seg_2.csv = true
    merged_2021-02-18-16-53-35_seg_2.csv = true
    ...

Lines starting with ``#`` and blank lines are comments. Order is
preserved on rewrite to keep diffs readable.

States
------
- :attr:`MarkerState.FALSE` — the CSV needs sanitization.
- :attr:`MarkerState.INPROGRESS` — sanitization in progress (or a
  crashed local run that left the marker partially written). On HPC:
  fail-fast.
- :attr:`MarkerState.TRUE` — CSV is sanitized and immutable. HPC fast
  path: skip the read-modify-write entirely.

Back-compat
-----------
1. **Legacy per-CSV marker file** (``<csv>.pre_sanitized`` adjacent to
   the CSV): if the consolidated marker has no entry for this CSV but
   a legacy sibling marker exists, the CSV is treated as
   :attr:`MarkerState.TRUE` (legacy ``presence-implies-sanitized``
   contract). The next :func:`write_state` call upgrades the entry
   into the consolidated file; legacy sibling markers are not deleted
   automatically — that is the operator's responsibility (see
   ``.junie/active_plans/improve_hpo_capabilities_RLRP-624.md``).
2. **Empty / free-text body** in either marker form is also treated as
   :attr:`MarkerState.TRUE` for continuity with pre-tri-state markers.

Atomicity
---------
:func:`write_state` writes via ``tempfile`` + ``os.replace``: readers
always observe either the previous content or the new content, never a
partial write. Sanitization is a single-writer pipeline (operator runs
locally, single process), so there is no concurrent-writer contention
on the consolidated marker.

See ``.junie/active_plans/improve_hpo_capabilities_RLRP-624.md`` Phase
E.6 for the full design.
"""
from __future__ import annotations

import enum
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

#: Suffix of the consolidated per-directory marker file. The full name is
#: ``<dirname>.pre_sanitized`` (e.g. ``valid/valid.pre_sanitized``).
PRE_SANITIZED_MARKER_SUFFIX: str = ".pre_sanitized"

PathLike = Union[str, os.PathLike]


class MarkerState(enum.Enum):
    """Tri-state lifecycle for the per-directory ``.pre_sanitized`` marker."""

    FALSE = "false"
    INPROGRESS = "inprogress"
    TRUE = "true"

    @classmethod
    def from_token(cls, token: str) -> Optional["MarkerState"]:
        """Return the :class:`MarkerState` matching ``token`` (case-insensitive),
        or ``None`` if no exact match.
        """
        normalized = token.strip().lower()
        for state in cls:
            if state.value == normalized:
                return state
        return None


def marker_path_for(csv_path: PathLike) -> Path:
    """Return the consolidated marker path for the directory containing
    ``csv_path``.

    The marker file lives next to the CSV at
    ``<csv_path.parent>/<csv_path.parent.name>.pre_sanitized``.
    """
    csv = Path(csv_path)
    parent = csv.parent
    return parent / f"{parent.name}{PRE_SANITIZED_MARKER_SUFFIX}"


def _legacy_per_csv_marker_path(csv_path: PathLike) -> Path:
    """Return the legacy per-CSV marker path: ``<csv>.pre_sanitized``."""
    return Path(str(csv_path) + PRE_SANITIZED_MARKER_SUFFIX)


# ---------------------------------------------------------------------------
# Internal: parse / serialize the consolidated marker file
# ---------------------------------------------------------------------------


def _parse_marker_file(marker: Path) -> Tuple[Dict[str, MarkerState], List[str]]:
    """Parse ``marker`` and return ``(entries, raw_lines)``.

    ``entries`` maps CSV filename → :class:`MarkerState`. Lines that are
    comments, blank, or unparseable are preserved in ``raw_lines`` (for
    diff-friendly rewrites) but excluded from ``entries``.

    For back-compat, a marker file whose body has **no** ``key = value``
    entries (e.g. legacy free-text or empty body) is treated as a
    "presence-implies-sanitized" marker: callers that look up any CSV
    in this directory will receive :attr:`MarkerState.TRUE`.
    """
    entries: Dict[str, MarkerState] = {}
    raw_lines: List[str] = []
    try:
        text = marker.read_text(encoding="utf-8")
    except OSError:
        return entries, raw_lines
    for line in text.splitlines():
        raw_lines.append(line)
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        state = MarkerState.from_token(value)
        if not key or state is None:
            continue
        entries[key] = state
    return entries, raw_lines


def _is_legacy_presence_marker(marker: Path) -> bool:
    """Return ``True`` iff ``marker`` exists with a body that is empty or
    contains only comments / unparseable lines (legacy marker semantics).
    """
    if not marker.exists():
        return False
    entries, _ = _parse_marker_file(marker)
    if entries:
        return False
    # Marker exists but has no parseable ``key = value`` entries.
    return True


def _serialize_marker_file(
    marker: Path, entries: Dict[str, MarkerState]
) -> str:
    """Return the textual body for ``entries`` with the standard header.

    Order of ``entries`` is preserved (Python dict insertion order).
    """
    dirname = marker.parent.name
    lines: List[str] = [
        f"# Consolidated tri-state marker for directory: {dirname}",
        f"# Emitted by tools.dataset_tools.pre_sanitized_marker.",
        f"# Each non-comment line maps a CSV filename to one of:",
        f"#     {{{MarkerState.FALSE.value}, "
        f"{MarkerState.INPROGRESS.value}, {MarkerState.TRUE.value}}}",
        f"#",
    ]
    for csv_name, state in entries.items():
        lines.append(f"{csv_name} = {state.value}")
    lines.append("")  # trailing newline
    return "\n".join(lines)


def _atomic_write(marker: Path, body: str) -> None:
    """Atomically write ``body`` to ``marker`` via ``tempfile`` + ``os.replace``."""
    marker.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=marker.name + ".", suffix=".tmp", dir=str(marker.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.replace(tmp_path, marker)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public API (per-CSV; semantics unchanged from the previous layout)
# ---------------------------------------------------------------------------


def read_state(csv_path: PathLike) -> Optional[MarkerState]:
    """Return the marker state recorded for ``csv_path``.

    Resolution order:

    1. Consolidated marker (``<csv.parent>/<csv.parent.name>.pre_sanitized``):
       if it has an explicit entry for ``csv_path.name``, return that.
    2. Consolidated marker exists but has no explicit entries (legacy
       free-text / empty body): return :attr:`MarkerState.TRUE`.
    3. Legacy per-CSV marker (``<csv>.pre_sanitized``) exists: return
       :attr:`MarkerState.TRUE`.
    4. Otherwise: return ``None``.
    """
    csv = Path(csv_path)
    consolidated = marker_path_for(csv)
    if consolidated.exists():
        entries, _ = _parse_marker_file(consolidated)
        explicit = entries.get(csv.name)
        if explicit is not None:
            return explicit
        if not entries:
            # Legacy presence marker (no parseable entries) → TRUE.
            return MarkerState.TRUE
        # Consolidated marker exists with entries but none for this CSV →
        # treat as "no record" so callers can fall back to legacy lookup
        # below before giving up.
    legacy = _legacy_per_csv_marker_path(csv)
    if legacy.exists():
        return MarkerState.TRUE
    return None


def write_state(csv_path: PathLike, state: MarkerState) -> Path:
    """Atomically record ``state`` for ``csv_path`` in the consolidated marker.

    The consolidated marker file is read-modify-written via
    ``tempfile`` + ``os.replace`` so concurrent readers always observe
    a fully-written marker. The sanitization pipeline is single-writer
    (operator runs locally), so there is no concurrent-writer contention
    by design.

    :param csv_path: Path to the CSV file (NOT the marker file itself).
    :param state: New marker state to record.
    :returns: Path to the written consolidated marker file.
    """
    if not isinstance(state, MarkerState):
        raise TypeError(
            f"[pre_sanitized_marker.write_state] state must be a MarkerState, "
            f"got {type(state).__name__}"
        )
    csv = Path(csv_path)
    marker = marker_path_for(csv)

    if marker.exists():
        entries, _ = _parse_marker_file(marker)
    else:
        entries = {}
    entries[csv.name] = state

    body = _serialize_marker_file(marker, entries)
    _atomic_write(marker, body)
    return marker


def is_sanitized(csv_path: PathLike) -> bool:
    """Convenience check: return ``True`` iff the recorded state is
    :attr:`MarkerState.TRUE` (or the legacy back-compat case)."""
    return read_state(csv_path) == MarkerState.TRUE


def require_sanitized(csv_path: PathLike) -> None:
    """Raise :class:`RuntimeError` unless the CSV is marked as sanitized.

    Intended for the HPC-side fast path. The error message points the
    operator at the local prep pipeline.
    """
    state = read_state(csv_path)
    if state == MarkerState.TRUE:
        return
    if state is None:
        reason = "no marker entry recorded"
    else:
        reason = f"marker state is '{state.value}' (expected 'true')"
    raise RuntimeError(
        f"[pre_sanitized_marker.require_sanitized] {reason}: '{csv_path}'\n"
        f"Hint: run the dataset pre-sanitization pipeline locally\n"
        f"  python3 launcher/robotic_3d_env_sanitize_dataset.py\n"
        f"then rsync the resulting *{PRE_SANITIZED_MARKER_SUFFIX} markers to the HPC server.\n"
        f"See .junie/active_plans/improve_hpo_capabilities_RLRP-624.md (Phase E.6)."
    )
