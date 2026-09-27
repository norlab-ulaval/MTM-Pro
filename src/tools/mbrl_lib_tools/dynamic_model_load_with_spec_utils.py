# coding=utf-8
import os
from typing import Tuple, Union

import omegaconf
from mbrl.models import OneDTransitionRewardModel
import mbrl.util.common
from omegaconf import DictConfig, ListConfig

from tools.console_tools.format import ConsoleFormat
from tools.console_tools.message import (
    consol_msg_universal_one_liner,
)
from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist
from tools.mbrl_lib_tools.common_tools import create_one_dim_tr_model_v2


def load_pretrained_dynamic_model_with_spec(
    cfg: omegaconf.DictConfig,
    dynamic_model_experiment_path: str,
    env_motion_model_spec: omegaconf.DictConfig,
) -> OneDTransitionRewardModel:
    # (Priority) ToDo: RLRP-242 refactor: rethink algorithm.env_motion_model_spec logic
    """Loads a pre-trained dynamic model.

    :param cfg: a hydra config dictionary
    :param dynamic_model_experiment_path: Path to the experiment dir containing the dynamic model.
    :param env_motion_model_spec: Specification object containing observation and action shapes.
    :return: An instance of OneDTransitionRewardModel.
    """
    consol_msg_universal_one_liner(
        f"Loading from experiment path: {dynamic_model_experiment_path}"
    )

    # .... Pre-condition ..........................................................................
    in_size = None
    out_size = None
    if cfg.dynamics_model.get("_target_") == "mbrl.models.BasicEnsemble":
        if is_cfg_key_exist(cfg.dynamics_model.member_cfg, "in_size") and is_cfg_key_exist(
            cfg.dynamics_model.member_cfg, "out_size"
        ):
            in_size = cfg.dynamics_model.member_cfg.in_size
            out_size = cfg.dynamics_model.member_cfg.out_size
    else:
        if is_cfg_key_exist(cfg.dynamics_model, "in_size") and is_cfg_key_exist(
            cfg.dynamics_model, "out_size"
        ):
            in_size = cfg.dynamics_model.in_size
            out_size = cfg.dynamics_model.out_size

    # .... Setup ..................................................................................
    saved_dynamic_model_abs_path = experiment_path_to_saved_dynamic_model_path(
        cfg, dynamic_model_experiment_path
    )

    target_env_obs_len = env_motion_model_spec.target_env_obs_shape
    if isinstance(target_env_obs_len, Tuple):
        if len(target_env_obs_len) == 1:
            target_env_obs_len = target_env_obs_len[0]
        else:
            raise NotImplementedError(
                f"Case {len(env_motion_model_spec.target_env_obs_shape)=} not suported yet"
            )

    target_env_act_len = env_motion_model_spec.target_env_act_shape
    if isinstance(target_env_act_len, Tuple):
        if len(target_env_act_len) == 1:
            target_env_act_len = target_env_act_len[0]
        else:
            raise NotImplementedError(
                f"Case {len(env_motion_model_spec.target_env_act_shape)=} not suported yet"
            )

    # .... Load model .............................................................................
    try:
        dynamics_model = create_one_dim_tr_model_v2(
            cfg,
            singlestep_obs_len=target_env_obs_len,
            singlestep_act_len=target_env_act_len,
            model_dir=saved_dynamic_model_abs_path,
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"{ConsoleFormat.MSG_ERROR_FORMAT}{e}\n\n"
            f"{ConsoleFormat.MSG_EMPH_FORMAT}"
            f"{ConsoleFormat.MSG_ERROR_FORMAT}env_motion_model_spec:"
            f"{ConsoleFormat.MSG_END_FORMAT}\n"
            f"{omegaconf.OmegaConf.to_yaml(env_motion_model_spec, resolve=False)}\n"
            f"{ConsoleFormat.MSG_EMPH_FORMAT}cfg.dynamics_model:"
            f"{ConsoleFormat.MSG_END_FORMAT}\n"
            f"{omegaconf.OmegaConf.to_yaml(cfg.dynamics_model, resolve=False)}\n"
        )

    if in_size is not None and out_size is not None:
        assert dynamics_model.model.in_size == in_size
        assert dynamics_model.model.out_size == out_size

    return dynamics_model


def load_saved_dynamic_model_spec_cfg(
    cfg: omegaconf.DictConfig, dynamic_model_experiment_path: str
) -> Union[DictConfig, ListConfig]:
    # (Priority) ToDo: RLRP-242 refactor: rethink algorithm.env_motion_model_spec logic
    # (NICE TO HAVE) ToDo: unit-test. Currently it's indirectly covered
    """
    Loads a saved dynamic model specification from a file and updates the hydra configuration.

    :param cfg: hydra configuration dictoinary.
    :param dynamic_model_experiment_path: The path to the dynamic model experiment directory.
    :return: The loaded environment motion model specification as a configuration dictionary.
    """

    dynamic_model_path = experiment_path_to_saved_dynamic_model_path(
        cfg, dynamic_model_experiment_path
    )

    with open(os.path.join(dynamic_model_path, "env_spec.yaml")) as file:
        env_motion_model_spec = omegaconf.OmegaConf.load(file)

    omegaconf.OmegaConf.update(
        cfg, "algorithm.env_motion_model_spec", env_motion_model_spec, merge=False
    )

    return env_motion_model_spec


def experiment_path_to_saved_dynamic_model_path(
    cfg: omegaconf.DictConfig, dynamic_model_experiment_path: str
) -> str:
    """
    Expand a dynamic model experiment path with the path to the saved dynamic model directory.

    :param cfg: hydra configuration dictoinary
    :param dynamic_model_experiment_path: The file path of the dynamic model experiment.
    :return: The relative path to the saved dynamic model.
    """
    project_root_abs_path = cfg.project_root_path
    saved_dynamic_model_abs_path = os.path.join(
        project_root_abs_path,
        dynamic_model_experiment_path,
        "mbrl_data/saved_dynamic_model",
    )
    saved_dynamic_model_abs_path = os.path.realpath(saved_dynamic_model_abs_path)
    assert os.path.exists(
        saved_dynamic_model_abs_path
    ), f"Path {saved_dynamic_model_abs_path} unreachable"
    return saved_dynamic_model_abs_path
