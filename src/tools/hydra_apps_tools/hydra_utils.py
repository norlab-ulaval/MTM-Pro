# coding=utf-8
import os
from typing import AnyStr

import hydra
import omegaconf
from hydra.core.hydra_config import HydraConfig

from tools.console_tools.message import consol_msg_DEV_universal


def get_hydra_experiment_cwd(cfg):
    """Get the hydra output directory (i.e. the experiment dir) but work with both `hydra.compose`
    and `hydra.main`.
    - The `hydra.main` case return the experiment directory as specified by `cfg.hydra.run`
    - The `hydra.compose` case assume a unit-test run and fetch the manualy set `cfg.unittest_cwd`

    Note: assume main config is set to pre hydra 1.2 behaviour, i.e.:
        >>> hydra:
        >>>     job:
        >>>         chdir: true

    :param cfg: a hydra config
    :return: the curent working directopry
    """
    if cfg.experiment == "unit_test_with_hydra_compose":
        work_dir = cfg.unittest_cwd
    else:
        # work_dir = os.getcwd()
        work_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    return work_dir


def get_hydra_original_cwd(cfg):
    """Get the hydra run original working directory from where was called the hydra app but work
    with both `hydra.compose` and `hydra.main`.
    - The `hydra.main` case return the same value as calling `hydra.utils.get_original_cwd()`
    - The `hydra.compose` case assume a unit-test run and fetch the manualy set `cfg.orig_cwd`

    :param cfg: a hydra config
    :return: the curent working directopry
    """
    if cfg.experiment == "unit_test_with_hydra_compose":
        original_work_dir = cfg.orig_cwd
    else:
        original_work_dir = hydra.utils.get_original_cwd()
    return original_work_dir


def is_hydra_multirun() -> bool:
    """Checks if the current Hydra configuration mode is set to "MULTIRUN".
    Require the HydraConfig object to be initialized.

    :return: True if the Hydra configuration mode is "MULTIRUN", False otherwise.
    """
    hydra_conf_mode_name = None
    try:
        hydra_conf = HydraConfig.get()
        hydra_conf_mode_name = hydra_conf.mode.name
    except ValueError as e:
        # Quickhack for unit-test
        if str(e) == "HydraConfig was not set":
            pass

    return hydra_conf_mode_name == "MULTIRUN"

def is_hydra_optuna_run(config: omegaconf.DictConfig, debug_mode: bool = False) -> bool:
    """Checks if the current Hydra configuration mode is an optuna hyperparameter sweep multirun.
    Can be used in Hydra callback `on_multirun_start()` method as it does not require HydraConfig
     object to be initialised.

    :return: True if the Hydra configuration is an optuna sweep run, False otherwise.
    """
    try:
        hydra_sweeper_conf = config.hydra.get('sweeper')
    except omegaconf.errors.ConfigAttributeError as e:
        hydra_conf = HydraConfig.get()
        hydra_sweeper_conf = hydra_conf.sweeper

    if debug_mode:
        consol_msg_DEV_universal(omegaconf.OmegaConf.to_yaml(hydra_sweeper_conf, resolve=True))

    if hydra_sweeper_conf:
        hydra_sweeper_name = hydra_sweeper_conf._target_
        is_optuna_sweeper = 'optuna_sweeper' in hydra_sweeper_name
    else:
        is_optuna_sweeper = False

    return is_optuna_sweeper

def is_hydra_sweeper_multiprocess_run(config: omegaconf.DictConfig) -> bool:
    try:
        hydra_sweeper_conf = config.hydra.get('sweeper')
    except omegaconf.errors.ConfigAttributeError as e:
        hydra_conf = HydraConfig.get()
        hydra_sweeper_conf = hydra_conf.sweeper

    return hydra_sweeper_conf.get('n_jobs', 1) > 1


def fetch_project_root_path_via_hydra(
    cfg: omegaconf.DictConfig, lvl_up: int, expected_root_dir: str
) -> AnyStr:
    """Returns the absolute path to the project root directory using hydra original cwd logic.

    :param cfg: the hydra config dict
    :param lvl_up: the number of level up from the hydra original cwd
    :param expected_root_dir: the name of the expected project root directory (for sanity check)
    :return: the absolute path to the project root directory
    """
    hydra_original_cwd = get_hydra_original_cwd(cfg)
    project_root_path = os.path.realpath(os.path.join(hydra_original_cwd, "../" * lvl_up))

    assert os.path.basename(project_root_path) == expected_root_dir
    return project_root_path


def get_hydra_experiment_id() -> str:
    if is_hydra_multirun():
        sweep_cfg = HydraConfig.get().sweep
        experiment_date_dir = os.path.basename(os.path.dirname(sweep_cfg.dir))
        experiment_time = os.path.basename(sweep_cfg.dir)
        subdir = os.path.basename(sweep_cfg.subdir)
        exp_run = os.path.join(experiment_date_dir, experiment_time, subdir)
    else:
        run_cfg = HydraConfig.get().run
        experiment_date_dir = os.path.basename(os.path.dirname(run_cfg.dir))
        experiment_time = os.path.basename(run_cfg.dir)
        exp_run = os.path.join(experiment_date_dir, experiment_time)
    return exp_run


def get_hydra_experiment_dir_path_and_id_names(cfg):
    exp_cwd = get_hydra_experiment_cwd(cfg)

    if is_hydra_multirun():
        exp_multirun_run_id = os.path.basename(exp_cwd)
        exp_date = os.path.basename(os.path.dirname(os.path.dirname(exp_cwd)))
        exp_time = os.path.basename(os.path.dirname(exp_cwd))
        exp_id_name = f"{exp_date}.{exp_time}.{exp_multirun_run_id}"
    else:
        exp_date = os.path.basename(os.path.dirname(exp_cwd))
        exp_time = os.path.basename(exp_cwd)
        exp_id_name = f"{exp_date}.{exp_time}"
    return exp_id_name
