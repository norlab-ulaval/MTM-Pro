# coding=utf-8
"""Static (cfg-only) ERLL per-pass epoch budget (RLRP-839, extracted from RLRP-773/RLRP-824).

Reproduces the per-pass epoch budget decision of
``AbstractExperienceReplayLearningLoop.execute()`` WITHOUT instantiating a loop (no data source,
no trainer, no torch), so the deploy-stage resume audit can decide whether a run's training is
complete from its config alone. The instance method
``AbstractExperienceReplayLearningLoop._planned_pass_epoch_budgets`` delegates here, which keeps
the two views of the budget in lock-step by construction.

Loop plan (mirrors ``_iter_erll_epochs``):

* ``UDER.loop_kind == "single_global_loop"`` => one fused pass ``[(0, True, True)]``;
* otherwise (``pber`` / ``uder``)             => ``range(uder_num_epochs + 1)``, epoch ``0`` is the
  initialization pass and epoch ``uder_num_epochs`` the final one.

Per-pass budget (mirrors ``execute()``):

* init pass with a truthy ``uder_num_epoch_init_override`` => that override;
* final pass                                              => ``final_max_num_epochs_train_model``;
* otherwise                                               => ``num_epochs_train_model // uder_num_epochs``
  (``None`` when ``num_epochs_train_model`` is ``null`` or the ``0`` single-global-loop sentinel).

Any falsy budget (``null`` / ``0`` => patience-driven) makes the whole plan unknown => ``None``.
"""
from typing import Any, List, Optional, Sequence, Tuple

import omegaconf

SINGLE_GLOBAL_LOOP_KIND = "single_global_loop"


def _select(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, (omegaconf.DictConfig, omegaconf.ListConfig)):
        return omegaconf.OmegaConf.select(cfg, key, default=default)
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def iter_erll_epochs(cfg: Any, erll_cfg_key: str = "UDER") -> List[Tuple[int, bool, bool]]:
    """Static twin of ``AbstractExperienceReplayLearningLoop._iter_erll_epochs``.

    :return: the ordered ``(erll_epoch, is_initialization, is_final)`` plan.
    """
    if _select(cfg, f"{erll_cfg_key}.loop_kind") == SINGLE_GLOBAL_LOOP_KIND:
        return [(0, True, True)]
    erll_max_epochs = int(_select(cfg, f"{erll_cfg_key}.uder_num_epochs"))
    return [
        (each, each == 0, each == erll_max_epochs) for each in range(erll_max_epochs + 1)
    ]


def experiment_planned_pass_epoch_budgets(
    cfg: Any,
    erll_cfg_key: str = "UDER",
    erll_epochs: Optional[Sequence[Tuple[int, bool, bool]]] = None,
) -> Optional[List[int]]:
    """Planned number of inner epochs per ERLL pass, or ``None`` when any pass is patience-driven.

    :param cfg: the consolidated Hydra cfg (``DictConfig`` or plain mapping).
    :param erll_cfg_key: the key the algorithm block is exposed under (always ``UDER``).
    :param erll_epochs: an explicit loop plan (the instance method passes its own
        ``_iter_erll_epochs()`` so subclass overrides are honoured); ``None`` => :func:`iter_erll_epochs`.
    """
    plan = list(erll_epochs) if erll_epochs is not None else iter_erll_epochs(cfg, erll_cfg_key)

    total_num_epochs = _select(cfg, f"{erll_cfg_key}.num_epochs_train_model")
    uder_num_epochs = _select(cfg, f"{erll_cfg_key}.uder_num_epochs")
    if total_num_epochs and uder_num_epochs:
        per_erll_epoch: Optional[int] = int(total_num_epochs) // int(uder_num_epochs)
    else:
        per_erll_epoch = None
    init_override = _select(cfg, f"{erll_cfg_key}.uder_num_epoch_init_override", default=False)
    final_budget = _select(cfg, f"{erll_cfg_key}.final_max_num_epochs_train_model")

    budgets: List[int] = []
    for _, is_init, is_final in plan:
        if is_init and init_override:
            budget = init_override
        elif is_final:
            budget = final_budget
        else:
            budget = per_erll_epoch
        if not budget:
            return None
        budgets.append(int(budget))
    return budgets


def experiment_planned_total_epochs(cfg: Any, erll_cfg_key: str = "UDER") -> Optional[int]:
    """Sum of :func:`experiment_planned_pass_epoch_budgets` (the global epoch ``E`` a complete run ends at).

    :return: the total, or ``None`` when the plan is unknown (patience-driven budget).
    """
    budgets = experiment_planned_pass_epoch_budgets(cfg, erll_cfg_key=erll_cfg_key)
    return sum(budgets) if budgets is not None else None
