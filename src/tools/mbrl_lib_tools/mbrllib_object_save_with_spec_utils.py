# coding=utf-8
import os
from typing import Optional

import omegaconf
from mbrl import models
from mbrl.util import ReplayBuffer
import numpy as np

from tools.console_tools.message import consol_msg_motion_model_learner_one_line
from tools.hydra_apps_tools.hydra_utils import get_hydra_experiment_cwd, get_hydra_original_cwd


def cfg_based_save_replaybuffer_and_dynamic_model(
    cfg: omegaconf.DictConfig,
    dynamics_model: Optional[models.OneDTransitionRewardModel],
    replay_buffer: Optional[ReplayBuffer],
) -> str:
    """Save a mbrl-lib motion dynamic model and/or mbrl-lib replay buffer to disk base on hydra
    configuration object settings.

    Save flag (or not) provided via hydra configuration file using the following field:
    - `overrides.motion_model.save_motion_model`
    - `overrides.sampler_rollout.save_replay_buffer`

    Save path automatically generated via hydra mechanism based on experiment name, time/date
    and so on.
    Can be overiden vi hydra configuration file using the following field:
    - `overrides.motion_model.overwrite_default_save_dir`

    :param cfg: hydra configuration
    :param dynamics_model: (optional) a mbrl-lib motion dynamic model
    :param replay_buffer: (optional) a mbrl-lib replay buffer
    :return: saved_mbrl_data_dir_path
    """
    save_motion_model = False
    save_replay_buffer = False

    if dynamics_model is not None:
        save_motion_model = cfg.overrides.motion_model.save_motion_model

    if replay_buffer is not None:
        save_replay_buffer = cfg.overrides.sampler_rollout.save_replay_buffer

    if save_motion_model or save_replay_buffer:
        if save_motion_model:
            save_motion_model = dynamics_model
        else:
            save_motion_model = None

        if save_replay_buffer:
            save_replay_buffer = replay_buffer
        else:
            save_replay_buffer = None

        # Note: Return None if the key `overwrite_default_save_dir` doesn't exist in `cfg`
        save_to_dir = omegaconf.OmegaConf.select(
            cfg,
            "overrides.motion_model.overwrite_default_save_dir",
            throw_on_missing=False,
        )

        saved_mbrl_data_dir_path = save_mbrllib_objects_and_env_spec(
            cfg, save_motion_model, save_replay_buffer, save_to_dir
        )

        consol_msg_motion_model_learner_one_line(f"mbrl data saved to: {saved_mbrl_data_dir_path}")
    else:
        saved_mbrl_data_dir_path = None

    return saved_mbrl_data_dir_path


def save_mbrllib_objects_and_env_spec(
    cfg: omegaconf.DictConfig,
    dynamics_model: Optional[models.OneDTransitionRewardModel] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
    override_save_dir: Optional[str] = None,
) -> str:
    """Save learned mbrl model and replay buffer

    :param cfg: hydra configuration dictionary
    :param dynamics_model: an mbrl-lib model
    :param replay_buffer: an mbrl-lib replay buffer
    :param override_save_dir: Must be a relative. Use the hydra directory management if set to None
    :return: The path to the mbrl data root
    """

    #  Dev note: run/debug/test cwd for component using `hydra compose` instead of `hydra main`
    #  when developping in remote dev mode trough ssh in Dockerized-AnonLab container:
    #       cwd='/home/non-interactive-ros2/tmp/MTM-Pro/src'

    # .... Setup ..................................................................................
    hydra_experiment_cwd = get_hydra_experiment_cwd(cfg)
    hydra_orginal_cwd = get_hydra_original_cwd(cfg)

    data_dir = "mbrl_data"

    if override_save_dir:
        # Change working dir temporarily
        os.chdir(hydra_orginal_cwd)
        mbrl_data_root = os.path.join(override_save_dir, data_dir)
        mbrl_data_root = os.path.relpath(mbrl_data_root)
    else:
        mbrl_data_root = os.path.join(hydra_experiment_cwd, data_dir)

    # ... Save dynamic model and env_spec .........................................................
    if dynamics_model:
        model_dir = "saved_dynamic_model"
        model_dir_path = os.path.realpath(os.path.join(mbrl_data_root, model_dir))

        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)

        with open(f"{model_dir_path}/env_spec.yaml", "w") as file:
            omegaconf.OmegaConf.save(cfg.algorithm.env_motion_model_spec, file)

        dynamics_model.save(model_dir_path)

    # ... Save replay buffer ......................................................................
    if replay_buffer is not None:  # replay_buffer is a generator
        buffer_dir = "saved_replay_buffer"
        buffer_dir_path = os.path.realpath(os.path.join(mbrl_data_root, buffer_dir))

        if not os.path.exists(buffer_dir_path):
            os.makedirs(buffer_dir_path)

        replay_buffer_spec = omegaconf.OmegaConf.create(
            {
                "sampler_rollout": {
                    "max_trajectory_length": cfg.overrides.sampler_rollout.trajectory_max_length,
                },
                "motion_model": {
                    "velocity_only_observation": cfg.overrides.motion_model.velocity_only_observation,
                    "history_len": cfg.overrides.motion_model.history_len,
                    "horizon_len": cfg.overrides.motion_model.horizon_len,
                },
                "obs_shape": replay_buffer.obs_shape[0],
                "next_obs_shape": replay_buffer.obs_shape[0],
                "action_shape": replay_buffer.action_shape[0],
                "normalize_double_precision": cfg.algorithm.get(
                    "normalize_double_precision", None
                ),
                "capacity": replay_buffer.capacity,
                "num_stored": replay_buffer.num_stored,
                "stores_trajectories": replay_buffer.stores_trajectories,
                "exploration_policy": omegaconf.OmegaConf.select(
                    cfg.overrides, "exploration_policy", default=None
                ),
                "domain_randomisation": omegaconf.OmegaConf.select(
                    cfg.overrides, "domain_randomisation", default=None
                ),
            }
        )

        with open(f"{buffer_dir_path}/replaybuffer_spec.yaml", "w") as file:
            omegaconf.OmegaConf.save(replay_buffer_spec, file)

        replay_buffer.save(buffer_dir_path)

    # .... Teardown ...............................................................................
    if override_save_dir:
        mbrl_data_root = os.path.normpath(os.path.join(hydra_orginal_cwd, mbrl_data_root))

        # Return to the hydra working dir
        os.chdir(hydra_orginal_cwd)

    return mbrl_data_root
