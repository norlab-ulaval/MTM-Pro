# coding=utf-8
"""R2S2R-specific extensions of the `hydra-optuna-sweeper` plugin.

Two thin extensions, governed under RLRP-624 cross-cutting follow-ups:

* **X.5** — :class:`R2S2RTPESamplerConfig`: widens the upstream
  ``TPESamplerConfig`` schema (which only exposes a 9-field subset) to also
  accept the modern TPE keys ``group``, ``constant_liar`` and
  ``warn_independent_sampling`` natively understood by
  :class:`optuna.samplers.TPESampler`. Registered into
  :class:`hydra.core.config_store.ConfigStore` under
  ``hydra/sweeper/sampler/tpe_r2s2r``; selected via
  ``override hydra/sweeper/sampler: tpe_r2s2r`` in
  ``hparam_optimization_single_objective_base.yaml`` and
  ``hparam_optimization_multi_objective_base.yaml``.

* **X.6** — :class:`R2S2RJournalAwareOptunaSweeper`: subclass of
  :class:`hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper` that
  detects ``storage_kind=journal`` (signalled by the configured ``storage``
  resolving to a bare filesystem path) and replaces it with an
  :class:`optuna.storages.JournalStorage` object before forwarding to
  :class:`hydra_plugins.hydra_optuna_sweeper._impl.OptunaSweeperImpl`. This
  unblocks parallel-safe HPO on shared filesystems (Valeria, Mamba, CC) where
  the SQLite backend is fragile under joblib ``n_jobs > 1``.

Stability note (RLRP-624 X.5)
-----------------------------
``group`` and ``constant_liar`` have been *experimental* in Optuna since
v2.8.0 (June 2021). They are widely used and have been API-stable for ~5
years, but Optuna's contract is "the interface may change without notice".
This module therefore treats them as **soft, opt-in dependencies**:

* No Python branching on these flags — they are pure YAML keys that flow
  through to :class:`optuna.samplers.TPESampler` as-is.
* Failure mode if Optuna ever removes them: a ``TypeError`` at sampler
  construction caught by the cheap CI guard
  ``tests/tests_tools/tests_hydra_apps_tools/test_tpe_sampler_constructible.py``
  (plan task D.5).

Stability note (RLRP-624 X.6)
-----------------------------
:class:`optuna.storages.JournalStorage` and :class:`JournalFileStorage` are
public, non-experimental API in Optuna 4.x. The sweeper plugin
(:mod:`hydra_plugins.hydra_optuna_sweeper`) does not branch on the
``storage`` argument type — it forwards it directly to
:func:`optuna.create_study`, which accepts both strings and storage objects.
This subclass therefore needs to override only ``__init__`` to swap the
argument before forwarding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import optuna
from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_optuna_sweeper.optuna_sweeper import OptunaSweeper
from omegaconf import DictConfig
from optuna.storages import JournalStorage

# Optuna 4.0+ exposes the file backend as `JournalFileBackend` under the
# `optuna.storages.journal` submodule; the legacy `JournalFileStorage` name
# is deprecated (removal scheduled for v6.0.0). Prefer the modern class but
# fall back to the legacy alias so the module imports on older minor versions
# of Optuna 4.x.
try:
    from optuna.storages.journal import JournalFileBackend as _JournalFileBackend
except ImportError:  # pragma: no cover — fallback for older Optuna 4.x
    from optuna.storages import JournalFileStorage as _JournalFileBackend  # noqa: F401


# =============================================================================
# X.5 — Widened TPE sampler config
# =============================================================================
@dataclass
class R2S2RTPESamplerConfig:
    """TPE sampler config exposing the modern (experimental) TPE keys.

    Mirrors the upstream ``TPESamplerConfig`` and adds the three keys missing
    from the installed sweeper plugin schema:

    * ``group`` — joint sampling for conditional / structured params (e.g.
      dict-of-dict choices like ``M_tl={HI,HO,UL}``).
    * ``constant_liar`` — prevents duplicate suggestions when running
      multiple parallel trials (essential for ``n_jobs >= 2``).
    * ``warn_independent_sampling`` — emits a warning when TPE falls back to
      independent sampling for a conditional dimension (kept default
      ``True`` so misconfigurations stay visible).

    See plan task X.5 + the module docstring for the experimental-flag
    governance rules.
    """

    _target_: str = "optuna.samplers.TPESampler"
    seed: Optional[int] = None
    consider_prior: bool = True
    prior_weight: float = 1.0
    consider_magic_clip: bool = True
    consider_endpoints: bool = False
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    # ── Modern TPE keys (experimental since Optuna v2.8.0) ───────────────
    multivariate: bool = False
    group: bool = False
    constant_liar: bool = False
    warn_independent_sampling: bool = True


# =============================================================================
# X.6 — Journal-aware OptunaSweeper subclass
# =============================================================================
_SQLITE_MAGIC = b"SQLite format 3\x00"


def _assert_not_sqlite_file(file_path: str) -> None:
    """Fail-fast if `file_path` points at an existing SQLite database.

    JournalFileBackend reads the file as a text-based append-only log; opening
    a SQLite `.db` file produces a confusing `UnicodeDecodeError` at sweeper
    instantiation time. This helper raises a clear, actionable error instead.

    See RLRP-624 (Valeria smoke regression after switching `storage_kind`
    from `sqlite` to `journal` while keeping the same `db_path`).
    """
    import os

    if not os.path.isfile(file_path):
        return
    try:
        with open(file_path, "rb") as fh:
            header = fh.read(len(_SQLITE_MAGIC))
    except OSError:
        return
    if header == _SQLITE_MAGIC:
        raise RuntimeError(
            f"[R2S2RJournalAwareOptunaSweeper] Refusing to open a SQLite "
            f"database as a JournalStorage file: {file_path!r}.\n"
            f"This typically happens when `OPTUNA_STORAGE_KIND` was switched "
            f"from `sqlite` to `journal` but a stale `.db` file from a prior "
            f"run still sits at the journal target path. Either:\n"
            f"  (a) delete/rename the stale SQLite file, or\n"
            f"  (b) ensure `hparam_optimizer.db_name_ext` resolves to "
            f"`.journal` (default) so journal and sqlite paths cannot collide."
        )


def _build_journal_storage(file_path: str) -> JournalStorage:
    """Build a `JournalStorage` backed by the journal file backend.

    `file_path` is created if its parent directory exists; missing parent
    directories must be created by the SLURM template (`OPTUNA_DB_DIR`).
    Uses Optuna 4.0+ :class:`JournalFileBackend` when available, with a
    fallback to the deprecated `JournalFileStorage` alias.

    Fails fast if `file_path` is an existing SQLite database (see
    :func:`_assert_not_sqlite_file`).
    """
    _assert_not_sqlite_file(file_path)
    return JournalStorage(_JournalFileBackend(file_path))


def _resolve_journal_storage(storage: Any) -> Any:
    """Convert a journal-mode storage spec to a `JournalStorage` object.

    Accepts:
    * Already-built storage objects (returned untouched).
    * Strings prefixed with ``journal://<path>`` — stripped and passed to
      :class:`JournalFileStorage`.
    * Bare filesystem paths (e.g. ``/scratch/.../foo.db``) — returned
      untouched; this is the sqlite path and must be wrapped in
      ``sqlite:///`` upstream.

    The convention chosen here (``journal://`` URI scheme) keeps the
    YAML-side ``storage`` resolution pattern uniform with sqlite/postgres
    and avoids ambiguity with bare paths.
    """
    if not isinstance(storage, str):
        return storage
    prefix = "journal://"
    if storage.startswith(prefix):
        return _build_journal_storage(storage[len(prefix):])
    return storage


class R2S2RJournalAwareOptunaSweeper(OptunaSweeper):
    """`OptunaSweeper` that auto-converts ``journal://<path>`` to a
    :class:`JournalStorage` object.

    Selected by setting ``hydra.sweeper._target_`` to this class. All other
    behavior is delegated to the upstream
    :class:`hydra_plugins.hydra_optuna_sweeper._impl.OptunaSweeperImpl`.
    """

    def __init__(
        self,
        sampler: Any,
        direction: Any,
        storage: Optional[Any],
        study_name: Optional[str],
        n_trials: int,
        n_jobs: int,
        max_failure_rate: float,
        search_space: Optional[DictConfig],
        custom_search_space: Optional[str],
        params: Optional[DictConfig],
    ) -> None:
        resolved_storage = _resolve_journal_storage(storage)
        super().__init__(
            sampler=sampler,
            direction=direction,
            storage=resolved_storage,
            study_name=study_name,
            n_trials=n_trials,
            n_jobs=n_jobs,
            max_failure_rate=max_failure_rate,
            search_space=search_space,
            custom_search_space=custom_search_space,
            params=params,
        )


# =============================================================================
# ConfigStore registration
# =============================================================================
def register_r2s2r_optuna_extensions() -> None:
    """Register R2S2R extensions with Hydra's ConfigStore.

    Idempotent: safe to call multiple times. Invoked at import time of this
    module so any launcher that imports
    :mod:`tools.hydra_apps_tools.r2s2r_apps_utils` (which imports us
    transitively via the sweeper subclass) automatically picks the
    extensions up.
    """
    cs = ConfigStore.instance()
    cs.store(
        group="hydra/sweeper/sampler",
        name="tpe_r2s2r",
        node=R2S2RTPESamplerConfig,
        provider="r2s2r",
    )


register_r2s2r_optuna_extensions()
