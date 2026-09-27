# coding=utf-8
from typing import List

import omegaconf
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
from tools.plot_tools.plot_management import (
    cfg_based_plot_save,
    plot_manager,
)
from tools.hydra_apps_tools.hydra_utils import get_hydra_experiment_dir_path_and_id_names


def plot_motion_model_train_validation_loss(
    cfg: omegaconf.DictConfig,
    headless: bool,
    train_losses: List[float],
    val_losses: List[float],
) -> None:
    """Plot train/validation loss

    Note: To close the plot window remotly, open an ssh session to the dockerised-AnonLab container
    and execute: $ wmctrl -c Figure 1

    :param cfg: an hydra configuration object
    :param headless: execute the env in headless mode
    :param train_losses:
    :param val_losses:
    """
    with plot_manager(cfg.show_plot, headless):
        fig, ax = plt.subplots(2, 1, figsize=(16, 8), dpi=50) # default dpi=100
        # ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("train loss (gaussian NLogL)")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("val loss (mse)")

        ax[0].plot(train_losses)
        ax[1].plot(val_losses)

        # Print exp config overrides setting on plot
        hydra_job_task_overrides = ""
        try:
            hydra_conf = HydraConfig.get()
            hydra_job_task_overrides = omegaconf.OmegaConf.to_yaml(hydra_conf.overrides.task)
        except ValueError as e:
            # Quickhack for unit-test
            if str(e) == "HydraConfig was not set":
                pass

        exp_id_name = get_hydra_experiment_dir_path_and_id_names(cfg)
        fig_title = f"Experiment '{cfg.experiment}', ID: {exp_id_name}\n{hydra_job_task_overrides}"
        fig.suptitle(fig_title)

        train_plot_name = f"exp{exp_id_name}_train_validation_loss.png"
        cfg_based_plot_save(cfg, train_plot_name)

        # (NICE TO HAVE) ToDo: assessment >> is this 'headless' condition still useful here?
        #                 Use 'show_plot' directly in the mean time.
        #
        # if headless:
        #     # # Note:
        #     # #     - PyCharm can fetch remote plot, but hydra break this capabily
        #     # #     - (NICE TO HAVE) ToDo: investigate??
        #     # #         - Check `os.environ` to see what env variable are available at runtime
        #     # #         - Check `HydraConfig.get().job.env_set`
        #     # #             and `HydraConfig.get().job.env_copy` to see what's
        #     # #             transfered inside the hydra run.
        #     # #     -  ToDo: unit-test the current workaround
        #     # #     - See this link for comment on non blocking plot display
        #     # #        https://matplotlib.org/2.0.2/api/pyplot_api.html#matplotlib.pyplot.show
        #     # if cfg.show_plot and cfg.IDE.pycharm:
        #     #     plt.show(block=False)
        #
        #     pass
        #
        # else:
        #     # (NICE TO HAVE) ToDo: assessment >> is 'IDE.ide_remote_run' still useful?
        #     # if cfg.show_plot and not cfg.IDE.ide_remote_run:
        #     if cfg.show_plot:
        #         consol_msg_motion_model_learner(
        #             "To close the plot window remotly, open an ssh session to the "
        #             "dockerised-AnonLab container "
        #             "and execute: \n"
        #             "$ wmctrl -c Figure 1"
        #         )
        #         plt.show()

        return None
