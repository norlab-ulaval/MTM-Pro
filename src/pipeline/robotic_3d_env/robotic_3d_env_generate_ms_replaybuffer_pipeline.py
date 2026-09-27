# coding=utf-8
import os

import omegaconf
from mbrl.util import ReplayBuffer

from algorithm.utils import seed_me
from pipeline.pipeline_utils.general.setup import (
    setup_source_multi_step_replay_buffer,
    uder_cfg_validation,
)
from pipeline.pipeline_utils.robotic_env_pipeline_utils.setup_utils import (
    create_ss_trajectory_replaybuffer_from_csv,
)
from tools.console_tools.message import consol_msg_universal_one_liner

from tools.hydra_apps_tools.hydra_utils import (
    fetch_project_root_path_via_hydra, get_hydra_experiment_cwd,
)

import matplotlib.pyplot as plt

# .... Matplotlib configuration ...................................................................
import matplotlib as mpl

from tools.multistep_tools.ms_replaybuffer_save_load_utils import (
    save_multistep_replaybuffer_with_spec,
)

plt.style.use("classic")
mpl.rcParams["figure.facecolor"] = "white"

# mpl.rcParams["legend.markerscale"] = 5
# mpl.rcParams["legend.numpoints"] = 3
# mpl.rcParams["legend.scatterpoints"] = 5

mpl.rcParams["font.size"] = 14
mpl.rcParams["legend.loc"] = "best"
# mpl.rcParams['legend.loc'] = 'lower right'
mpl.rcParams["axes.prop_cycle"] = plt.cycler(color=["b", "g", "r", "y"])


def execute(cfg: omegaconf.DictConfig, headless: bool = False) -> ReplayBuffer:
    """Execute math toy environment Sim2Sim motion model learner feedback loop

    :param cfg: Hydra configuration file
    :param headless: turn environment rendering off (param for commandline flag)
    :return: a HyperparamObjectives object.
    """

    exp_dir_relative_path = get_hydra_experiment_cwd(cfg)
    seed, torch_rng = seed_me(cfg, output_torch_rdn_generator=True)

    # .... Configuration setting validation .......................................................
    uder_cfg_validation(cfg)

    # :::: Define environment dynamic :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # .... Create source single step replay buffer ................................................
    ss_full_time_space_replay_buffers = create_ss_trajectory_replaybuffer_from_csv(
        cfg, exp_dir_relative_path, headless
    )

    # .... Create multi-step replay buffer ........................................................
    _, ms_replay_buffer_explorable_region = setup_source_multi_step_replay_buffer(
        cfg, ss_full_time_space_replay_buffers
    )

    if cfg.pipeline.get('enable_ms_replaybuffer_saving_to_data_dir', True):
        ms_replaybuffer_path = save_multistep_replaybuffer_with_spec(
            cfg,
            ms_replay_buffer_explorable_region,
            override_save_dir=os.path.join(cfg.project_root_path, cfg.environment.data_path),
        )

        consol_msg_universal_one_liner(
            f"Multistep replay buffer saved to '{ms_replaybuffer_path}'"
        )

    # ==== Teardown ===============================================================================
    plt.close("all")

    return ms_replay_buffer_explorable_region
