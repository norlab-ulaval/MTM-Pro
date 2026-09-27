# coding=utf-8
from typing import Optional

import omegaconf

from algorithm.utils import seed_me
from pipeline.pipeline_utils.math_env_pipeline_utils.setup_utils import setup_data
from pipeline.pipeline_utils.math_env_pipeline_utils.train_and_deploy_utils import (
    math_env_ms_model_train_and_deploy,
)
from pipeline.pipeline_utils.general.setup import (
    assert_ood_objective_has_ood_trajectories,
    uder_cfg_validation,
)

from tools.hydra_apps_tools.hparam_optimization import HyperparamObjectives
from tools.hydra_apps_tools.hydra_utils import (
    get_hydra_experiment_cwd,
)

from tools.dna_dev_tools.dn_CI_tools import show_plot_unless_CI_server_runned
from tools.dna_dev_tools.dn_pytest_tools import is_pytest_run

import matplotlib.pyplot as plt

# .... Matplotlib configuration ...................................................................
import matplotlib as mpl

from tools.hydra_apps_tools.r2s2r_apps_utils import R2S2RPipelineHydraApp

plt.style.use("classic")
mpl.rcParams["figure.facecolor"] = "white"

# mpl.rcParams["legend.markerscale"] = 5
# mpl.rcParams["legend.numpoints"] = 3
# mpl.rcParams["legend.scatterpoints"] = 5

mpl.rcParams["font.size"] = 14
mpl.rcParams["legend.loc"] = "best"
# mpl.rcParams['legend.loc'] = 'lower right'
mpl.rcParams["axes.prop_cycle"] = plt.cycler(color=["b", "g", "r", "y"])


def execute(
    cfg: omegaconf.DictConfig,
    headless: bool = False,
    pipeline_app: Optional[R2S2RPipelineHydraApp] = None,
) -> HyperparamObjectives:
    """Execute math toy environment Sim2Sim motion model learner feedback loop

    :param cfg: Hydra configuration file
    :param headless: turn environment rendering off (param for commandline flag)
    :param pipeline_app:
    :return: a HyperparamObjectives object
    """
    exp_dir_relative_path = get_hydra_experiment_cwd(cfg)
    seed, torch_rng = seed_me(cfg, output_torch_rdn_generator=True)

    # .... Configuration setting validation .......................................................
    uder_cfg_validation(cfg)

    # :::: Define environment dynamic :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    (
        state_space_label,
        ss_full_time_space_replay_buffers,
        val_env_trjs,
        target_InD_rollouts,
        target_OOD_rollouts,
    ) = setup_data(cfg, exp_dir_relative_path, headless)

    # Fail loudly/early when an HPO study optimises the OOD deploy metric but the
    # dataset declares zero OOD test trajectories (RLRP-778, plan D1).
    assert_ood_objective_has_ood_trajectories(cfg, len(target_OOD_rollouts))

    # :::: Learn models :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    (
        best_pred_mae_score,
        ms_deploy_result,
        ms_model_description,
        ms_tensorboard_writer,
        best_pred_val_loss,
    ) = math_env_ms_model_train_and_deploy(
        cfg,
        ss_full_time_space_replay_buffers,
        state_space_label,
        target_InD_rollouts,
        target_OOD_rollouts,
        val_env_trjs,
        exp_dir_relative_path,
        headless,
        torch_rng,
        pipeline_app=pipeline_app,
    )

    # ==== Post: Metric =================================================================================

    # ==== Teardown ===============================================================================
    if show_plot_unless_CI_server_runned(True) and not is_pytest_run():
        ms_tensorboard_writer.close()

    plt.close("all")

    # Always-recorded deploy-stage compounded-prediction-mode MAE on the InD/OOD target
    # rollout sets, available as selectable HPO objectives (see cfg.hparam_optimizer.objectives_name).
    deploy_compounded_pred_mae_target_InD = ms_deploy_result.compounded_pred_mae_target_InD
    deploy_compounded_pred_mae_target_OOD = ms_deploy_result.compounded_pred_mae_target_OOD

    hyperparam_objectives = HyperparamObjectives(cfg)
    hyperparam_objectives.record("best_pred_mae_score", best_pred_mae_score)
    hyperparam_objectives.record("best_pred_val_loss", best_pred_val_loss)
    hyperparam_objectives.record(
        "deploy_compounded_pred_mae_target_InD", deploy_compounded_pred_mae_target_InD
    )
    hyperparam_objectives.record(
        "deploy_compounded_pred_mae_target_OOD", deploy_compounded_pred_mae_target_OOD
    )
    return hyperparam_objectives.auto_record(locals())
