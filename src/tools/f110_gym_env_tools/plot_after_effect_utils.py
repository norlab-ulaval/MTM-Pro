# coding=utf-8
from matplotlib import pyplot as plt
from tools.plot_tools.plot_management import (
    cfg_based_plot_save,
    plot_manager,
)
from tools.hydra_apps_tools.hydra_utils import get_hydra_experiment_dir_path_and_id_names
from trajectory_container_tools.dataclasses.f110_gym_trajectory_dataclass import (
    F110MotionDynamicDataclass,
)


def deployed_env_trj_poses_obs_errors_aftereffect_plot(
    cfg,
    headless,
    obs_container_1: F110MotionDynamicDataclass,
    obs_container_2: F110MotionDynamicDataclass,
) -> None:
    # (NICE TO HAVE) ToDo: make a generalizable ploting function from the two plot fct
    """Plot velocity observation error between two different trajectory.

    Note: Plot used by After Effect project `exp_visualisation_tools/sim2sim_env_compare.aep`

    Usage note: To close the plot window remotly, open an ssh session to the dockerised-AnonLab
    container
    and execute: $ wmctrl -c Figure 1

    :param cfg: hydra configuration
    :param headless: will display plot if not in headless mode
    :param obs_container_1: trajectory samples number one
    :param obs_container_2: trajectory samples number two
    """
    with plot_manager(cfg.show_plot, headless):

        """(!) WARNING:
        This plot is used by After Effect project `exp_visualisation_tools/sim2sim_env_compare.aep`
        Be advise, that any modification to size and numbers of plot axes will result in breaking
        change in this AE project and require modification downstream.
        """
        fig, ax = plt.subplots(3, 1, figsize=(16, 12))

        ax[0].set_ylabel("pose_x error [m]")
        ax[1].set_ylabel("pose_y error [m]")
        ax[2].set_ylabel("pose_theta error [rad]")
        ax[2].set_xlabel("Timesteps")

        pose_x_error = obs_container_1.next_pose_x - obs_container_2.next_pose_x
        pose_y_error = obs_container_1.next_pose_y - obs_container_2.next_pose_y
        pose_theta_error = (
            obs_container_1.next_pose_theta - obs_container_2.next_pose_theta
        )

        ax[0].plot(pose_x_error)
        ax[1].plot(pose_y_error)

        ax[2].set_ylim((-0.5, 0.5))
        ax[2].plot(pose_theta_error)

        exp_id_name = get_hydra_experiment_dir_path_and_id_names(cfg)
        fig_title = (
            f"Experiment '{cfg.experiment}', ID: {exp_id_name}\n"
            f"Learned dynamic {obs_container_1.feature_name} vs (original)"
            f" {obs_container_2.feature_name}\n"
            "World pose error"
        )
        fig.suptitle(fig_title)

        deploy_plot_name = f"exp{exp_id_name}_deploy_rollout_vel_error.png"

        cfg_based_plot_save(cfg, deploy_plot_name)

        return None


def deployed_env_trj_velocity_obs_errors_aftereffect_plot(
    cfg,
    headless,
    obs_container_1: F110MotionDynamicDataclass,
    obs_container_2: F110MotionDynamicDataclass,
) -> None:
    # (NICE TO HAVE) ToDo: make a generalizable ploting function from the two plot fct
    """Plot pose observation error between two different trajectory.

    Note: Plot used by After Effect project `exp_visualisation_tools/sim2sim_env_compare.aep`

    Usage note: To close the plot window remotly, open an ssh session to the dockerised-AnonLab
    container
    and execute: $ wmctrl -c Figure 1

    :param cfg: hydra configuration
    :param headless: will display plot if not in headless mode
    :param obs_container_1: trajectory samples number one
    :param obs_container_2: trajectory samples number two
    """
    with plot_manager(cfg.show_plot, headless):

        """(!) WARNING:
        This plot is used by After Effect project `exp_visualisation_tools/sim2sim_env_compare.aep`
        Be advise, that any modification to size and numbers of plot axes will result in breaking
        change in this AE project and require modification downstream.
        """
        fig, ax = plt.subplots(3, 1, figsize=(16, 12))

        ax[0].set_ylabel("vel_long error [m/s]")
        ax[1].set_ylabel("vel_lat error [m/s]")
        ax[2].set_ylabel("vel_ang error [rad/s]")
        ax[2].set_xlabel("Timesteps")

        vel_x_error = abs(obs_container_1.next_vel_x - obs_container_2.next_vel_x)
        vel_y_error = abs(obs_container_1.next_vel_y - obs_container_2.next_vel_y)
        vel_ang_error = abs(obs_container_1.next_vel_ang - obs_container_2.next_vel_ang)

        # # (NICE TO HAVE) ToDo: implement vectorized version (ref task RLRP-182)
        # vel_ang_error_tmp = obs_container_1.vel_ang - obs_container_2.vel_ang
        # for idx, each_ang_vel in enumerate(vel_ang_error_tmp):
        #     # (NICE TO HAVE) ToDo: validate angle wrapper usage (ref task RLRP-182)
        #     vel_ang_error_tmp[idx] = normalize_world_orientation(each_ang_vel)
        #     # vel_ang_error_tmp[idx] = normalize_angular_velocity_hack(each_ang_vel)
        # vel_ang_error = abs(vel_ang_error_tmp)

        ax[0].plot(vel_x_error, linewidth=0.5)
        ax[1].plot(vel_y_error, linewidth=0.5)
        ax[2].plot(vel_ang_error, linewidth=0.5)

        ax[0].set_ylim((0.0, 0.15))
        ax[1].set_ylim((0.0, 0.15))
        ax[2].set_ylim((0.0, 0.15))

        exp_id_name = get_hydra_experiment_dir_path_and_id_names(cfg)
        fig_title = (
            f"Target environment robot frame velocity prediction error"
            f"\n"
            f"'{obs_container_2.feature_name}' vs '{obs_container_1.feature_name}'"
            f"\n"
            f"Experiment '{cfg.experiment}', ID: {exp_id_name}"
        )
        fig.suptitle(fig_title)

        deploy_plot_name = f"exp{exp_id_name}_deploy_rollout_vel_error.png"
        cfg_based_plot_save(cfg, deploy_plot_name)

        return None
