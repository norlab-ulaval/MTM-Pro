# coding=utf-8
import os

import omegaconf

from algorithm.utils import seed_me
from pipeline.pipeline_utils.general.setup import get_model_description, setup_multistep_step_model, \
    setup_tensorboard_writer
from pipeline.pipeline_utils.math_env_pipeline_utils.setup_utils import (
    setup_test_target_rollouts,
)
from pipeline.pipeline_utils.general.train_and_deploy_utils import (
    execute_ms_model_test_time_rollouts,
    run_epoch_checkpoint_rollouts_if_enabled,
)

from tools.hydra_apps_tools.hydra_utils import (
    fetch_project_root_path_via_hydra,
    get_hydra_experiment_cwd,
)

from tools.dna_dev_tools.dn_CI_tools import show_plot_unless_CI_server_runned
from tools.dna_dev_tools.dn_pytest_tools import is_pytest_run

import matplotlib.pyplot as plt

# .... Matplotlib configuration ...................................................................
import matplotlib as mpl

plt.style.use("classic")
mpl.rcParams["figure.facecolor"] = "white"

# mpl.rcParams["legend.markerscale"] = 5
# mpl.rcParams["legend.numpoints"] = 3
# mpl.rcParams["legend.scatterpoints"] = 5

mpl.rcParams["font.size"] = 14
mpl.rcParams["legend.loc"] = "best"
# mpl.rcParams['legend.loc'] = 'lower right'
mpl.rcParams["axes.prop_cycle"] = plt.cycler(color=["b", "g", "r", "y"])


def execute(cfg: omegaconf.DictConfig, headless: bool = False) -> None:
    """Execute math toy environment Sim2Sim motion model learner feedback loop

    :param cfg: Hydra configuration file
    :param headless: turn environment rendering off (param for commandline flag)
    :return: a HyperparamObjectives object
    """
    exp_dir_relative_path = get_hydra_experiment_cwd(cfg)
    seed, torch_rng = seed_me(cfg, output_torch_rdn_generator=True)

    # .... Define environment dynamic .............................................................

    target_InD_rollouts, target_OOD_rollouts = setup_test_target_rollouts(
        cfg, exp_dir_relative_path, headless
    )

    # .... Setup model ............................................................................
    root_project_path = fetch_project_root_path_via_hydra(
        cfg, 1, "MTM-Pro"
    )

    load_pretrained_path = os.path.join(root_project_path, cfg.pretrained_model_path)
    motion_model_container = setup_multistep_step_model(cfg, load_pretrained_path=load_pretrained_path)

    ms_model_description = get_model_description(cfg.ms_model, motion_model_container)

    # :::: Deploy :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    if show_plot_unless_CI_server_runned(True) and not is_pytest_run():
        ms_tensorboard_writer = setup_tensorboard_writer(cfg, ms_model_description)
    else:
        ms_tensorboard_writer = None

    ms_deploy_result = execute_ms_model_test_time_rollouts(
        cfg,
        motion_model_container,
        ms_model_description,
        cfg.environment.data.label,
        target_InD_rollouts,
        target_OOD_rollouts,
        None,
        ms_tensorboard_writer,
        exp_dir_relative_path,
        headless,
        torch_rng,
    )

    # RLRP-773 (R6/G1.1): additionally run per-epoch-checkpoint rollouts when enabled (opt-in).
    # The `epoch_checkpoints/` tree lives beside the primary model dir, i.e. under the parent of
    # `load_pretrained_path` (which points at `<exp>/model_<ClassName>`); per-epoch rollouts are
    # written under the current deploy cwd. No-op when OFF or when the tree is absent.
    run_epoch_checkpoint_rollouts_if_enabled(
        cfg,
        motion_model_container,
        ms_model_description,
        cfg.environment.data.label,
        target_InD_rollouts,
        target_OOD_rollouts,
        None,
        ms_tensorboard_writer,
        exp_dir_relative_path,
        headless,
        torch_rng,
        checkpoints_source_dir=os.path.dirname(load_pretrained_path),
    )

    # ==== Post: Metric ===========================================================================

    # ==== Teardown ===============================================================================
    if show_plot_unless_CI_server_runned(True) and not is_pytest_run():
        ms_tensorboard_writer.close()

    plt.close("all")

    return None
