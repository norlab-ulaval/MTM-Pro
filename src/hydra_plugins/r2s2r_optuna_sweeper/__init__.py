# coding=utf-8
"""Hydra-plugin entrypoint for the R2S2R journal-aware Optuna sweeper.

Hydra's plugin discovery (`hydra/core/plugins.py::Plugins._initialize`)
auto-imports `hydra_plugins.*` namespace packages and registers each
discovered class under the key ``f"{clazz.__module__}.{clazz.__name__}"``.
Hydra's `_instantiate` then enforces that any sweeper `_target_` starts
with ``hydra_plugins.`` AND is present in that registry.

Re-exporting via ``from tools... import R2S2R...`` is **not** sufficient —
the imported class keeps its original ``__module__`` (``tools.hydra_apps_tools.r2s2r_optuna_sweeper``),
so it is registered under the wrong key and the sweeper YAML target
``hydra_plugins.r2s2r_optuna_sweeper.R2S2RJournalAwareOptunaSweeper`` fails
with ``Unknown plugin class``.

The fix is to **define** the subclass directly inside this module so its
``__module__`` matches the plugin namespace path used in YAML. The
implementation logic still lives in `tools.hydra_apps_tools.r2s2r_optuna_sweeper`
(`_resolve_journal_storage`, `R2S2RTPESamplerConfig`,
`register_r2s2r_optuna_extensions`); this file only re-binds the sweeper
class under the correct module path.

Important: ``src/hydra_plugins/`` MUST remain a PEP 420 namespace package
(no top-level ``__init__.py``) so the installed
``hydra_plugins.hydra_optuna_sweeper`` from the upstream wheel stays
discoverable side-by-side with this one.
"""
from tools.hydra_apps_tools.r2s2r_optuna_sweeper import (  # noqa: F401
    R2S2RJournalAwareOptunaSweeper as _R2S2RJournalAwareOptunaSweeperImpl,
    R2S2RTPESamplerConfig,
    register_r2s2r_optuna_extensions,
)


class R2S2RJournalAwareOptunaSweeper(_R2S2RJournalAwareOptunaSweeperImpl):
    """Thin re-bind of the implementation class under the
    ``hydra_plugins.r2s2r_optuna_sweeper`` module so Hydra's plugin scanner
    registers it under the canonical key
    ``hydra_plugins.r2s2r_optuna_sweeper.R2S2RJournalAwareOptunaSweeper``.

    See module docstring for why a re-export (``from ... import ...``) is
    not sufficient.
    """

    pass


__all__ = [
    "R2S2RJournalAwareOptunaSweeper",
    "R2S2RTPESamplerConfig",
    "register_r2s2r_optuna_extensions",
]
