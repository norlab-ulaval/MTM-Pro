# coding=utf-8
import os

import omegaconf
from pipeline.pipeline_utils.robotic_env_pipeline_utils.dataset_sanitization import (
    sanitize_dataset,
)


def execute(cfg: omegaconf.DictConfig, headless: bool = True) -> None:
    """Execute data pre-processing pipeline for robotic 3d environment

    Option: run with hydra flag ``--pipeline.sanitize_force=true`` to sanitize all CSVs
    even if it has already been marked as sanitized.

    :param cfg: Hydra configuration file
    :param headless: turn environment rendering off (param for commandline flag)
    :return: None
    """

    # .... Phase E.6 — dataset pre-sanitization stage .............................................
    # The pipeline crawls every CSV referenced by ``cfg.environment.data*`` and writes the
    # tri-state ``<csv>.pre_sanitized`` marker (FALSE → INPROGRESS → TRUE). This is the canonical
    # operator-side step run **locally** before ``rsync`` to HPC, where parallel HPO workers then
    # take the read-only fast path of ``sanitize_csv_data``.
    sanitize_dataset(
        cfg,
        force=cfg.pipeline.get("sanitize_force", False),
        verbose=cfg.debug_mode if hasattr(cfg, "debug_mode") else True,
    )
    return None
