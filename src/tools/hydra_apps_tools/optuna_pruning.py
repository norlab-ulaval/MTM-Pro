# coding=utf-8
"""Optuna pruning glue for the hydra-optuna-sweeper.

The `hydra-optuna-sweeper` plugin (v1.x) does not pass the live
`optuna.Trial` object to the `@hydra.main` objective and does not wire a
pruner. This module rebuilds the trial by loading the study from the
configured storage and matching the RUNNING trial that owns the current
Hydra job, then exposes a small handle that the rest of the codebase
(notably ``R2S2RPipelineHydraApp`` and trainer epoch callbacks) can use
to call ``trial.report(...)`` / ``trial.should_prune()``.

Phase B of `.junie/active_plans/improve_hpo_capabilities_RLRP-624.md`.
[RLRP-624]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import hydra.utils
import optuna
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from optuna.trial import TrialState

from tools.console_tools.message import consol_msg_universal_one_liner


@dataclass
class OptunaTrialHandle:
    """Lightweight container for the active Optuna trial inside a Hydra job."""

    study: optuna.Study
    trial: optuna.Trial
    storage_url: str

    @property
    def trial_id(self) -> int:
        # Internal but stable across optuna 3.x; used by `set_trial_state_values`.
        return self.trial._trial_id


def _coerce(v: Any) -> Any:
    """Normalize values so Hydra-override strings compare to optuna params."""
    if isinstance(v, bool):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _params_match(hydra_overrides: dict, trial_params: dict) -> bool:
    """Best-effort match of a RUNNING trial's params against Hydra overrides."""
    for k, v in trial_params.items():
        if k not in hydra_overrides:
            return False
        if _coerce(hydra_overrides[k]) != _coerce(v):
            return False
    return True


def _build_storage_url(cfg: DictConfig) -> str:
    """Concatenate the (optionally env-overridden) `db_path_pre` with `db_path`.

    Mirrors the resolution in ``hparam_optimization_storage_launcher_base.yaml``
    (the shared storage/launcher partial) so the URL we
    use to load the study is identical to the one the sweeper passes via
    ``hydra.sweeper.storage``.
    """
    pre = cfg.hparam_optimizer.get("db_path_pre", "") or ""
    path = cfg.hparam_optimizer.get("db_path", "") or ""
    return f"{pre}{path}"


def acquire_optuna_trial(cfg: DictConfig) -> Optional[OptunaTrialHandle]:
    """Return the live ``optuna.Trial`` owned by the current hydra-multirun job.

    Returns ``None`` (pruning disabled) when:
      - ``hparam_optimizer.sweep_optimizer_dryrun`` is True,
      - ``cfg.hparam_optimizer.pruner`` is missing/null,
      - the run is not in MULTIRUN mode (single run, no sweeper),
      - the study cannot be loaded or the trial cannot be matched.

    All failure modes are logged but never raise — pruning is strictly opt-in
    and must never break the training pipeline.
    """
    hparam = cfg.get("hparam_optimizer", None)
    if hparam is None:
        return None
    if bool(hparam.get("sweep_optimizer_dryrun", False)):
        return None
    if not hparam.get("pruner"):
        return None

    try:
        hcfg = HydraConfig.get()
        if str(hcfg.mode).upper() != "MULTIRUN":
            return None
        study_name = hcfg.sweeper.study_name
    except Exception as e:
        consol_msg_universal_one_liner(
            f"[OptunaPruning] HydraConfig unavailable ({e}); pruning disabled."
        )
        return None

    storage_url = _build_storage_url(cfg)
    if not storage_url:
        consol_msg_universal_one_liner(
            "[OptunaPruning] empty storage URL; pruning disabled."
        )
        return None

    # Build {param_name: value} from this job's task-level overrides.
    overrides: dict = {}
    try:
        for ov in hcfg.overrides.task:
            if "=" in ov and not ov.startswith("+"):
                k, v = ov.split("=", 1)
                overrides[k.lstrip("~")] = v
    except Exception:
        pass

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception as e:
        consol_msg_universal_one_liner(
            f"[OptunaPruning] cannot load study '{study_name}' "
            f"from '{storage_url}': {e}. Pruning disabled."
        )
        return None

    running = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    candidate = None
    for t in running:
        if _params_match(overrides, t.params):
            candidate = t
            break
    if candidate is None and len(running) == 1:
        # Single concurrent worker → unambiguous.
        candidate = running[0]
    if candidate is None:
        consol_msg_universal_one_liner(
            "[OptunaPruning] could not locate RUNNING trial matching this "
            "hydra job; pruning disabled."
        )
        return None

    try:
        trial = optuna.Trial(study, candidate._trial_id)
    except Exception as e:
        consol_msg_universal_one_liner(
            f"[OptunaPruning] failed to rebuild Trial({candidate._trial_id}): {e}. "
            f"Pruning disabled."
        )
        return None

    # Attach the configured pruner to the study (idempotent across workers).
    try:
        pruner = hydra.utils.instantiate(cfg.hparam_optimizer.pruner)
        study.pruner = pruner
        consol_msg_universal_one_liner(
            f"[OptunaPruning] attached pruner={type(pruner).__name__} "
            f"on trial #{candidate.number}"
        )
    except Exception as e:
        consol_msg_universal_one_liner(
            f"[OptunaPruning] failed to instantiate pruner: {e}. Pruning disabled."
        )
        return None

    return OptunaTrialHandle(study=study, trial=trial, storage_url=storage_url)


def mark_trial_pruned(handle: OptunaTrialHandle) -> None:
    """Mark the trial PRUNED in storage so the dashboard reflects reality."""
    try:
        handle.study._storage.set_trial_state_values(
            handle.trial_id, state=TrialState.PRUNED
        )
    except Exception as e:
        consol_msg_universal_one_liner(
            f"[OptunaPruning] mark_trial_pruned failed: {e}"
        )
