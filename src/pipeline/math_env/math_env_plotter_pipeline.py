# coding=utf-8
import csv
import os
import warnings
from typing import Union

import numpy as np
import omegaconf

import gymnasium as gym

from tqdm import tqdm

from algorithm.utils import seed_me
from pipeline.pipeline_utils.math_env_pipeline_utils.setup_utils import (
    setup_burst_noise_config,
    setup_noise_cfg_str,
    setup_test_target_rollouts,
)
from pipeline.pipeline_utils.math_env_pipeline_utils.math_env_selector_utils import (
    math_environment_selector,
)
from tools.hydra_apps_tools.hydra_utils import (
    get_hydra_experiment_cwd,
    get_hydra_experiment_id,
)

from math_gymnasium.tools.plot_3d_utils import (
    three_dimension_environment_space_plot,
)

from math_gymnasium.envs.arbitrary_dim_math_continuous import MathContinuousGymnasium
from tools.math_tools.ease_in_ease_out import easeInOutSine
from tools.plot_tools.plot_management import (
    manage_matplotlib_backend,
    manage_matplotlib_warnings,
    show_and_save_plot_helper,
)

import matplotlib.pyplot as plt

# .... Matplotlib configuration ...................................................................
import matplotlib as mpl

plt.style.use("classic")
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["font.size"] = 14

# mpl.rcParams["legend.markerscale"] = 1.0
# mpl.rcParams["legend.numpoints"] = 1
# mpl.rcParams["legend.scatterpoints"] = 1

mpl.rcParams["legend.loc"] = "best"
# mpl.rcParams['legend.loc'] = 'lower right'
mpl.rcParams["axes.prop_cycle"] = plt.cycler(color=["b", "g", "r", "y"])


def _save_trajectory_to_csv(
    exp_dir_path: str,
    file_name: str,
    time_space: np.ndarray,
    state_space_3d: np.ndarray,
    state_space_3d_with_noise: np.ndarray,
) -> None:
    """Save trajectory data passed to :func:`three_dimension_environment_space_plot`
    as a csv file with columns ``timestep, x, y, z, x_n, y_n, z_n``.

    :param exp_dir_path: Output directory (same as the corresponding plot file).
    :param file_name: Base file name (without extension); ``.csv`` will be appended.
    :param time_space: 1D array of timesteps, shape ``(N,)``.
    :param state_space_3d: Ground truth 3D positions, shape ``(N, 3)``.
    :param state_space_3d_with_noise: Noisy 3D positions, shape ``(N, 3)``.
    """
    os.makedirs(exp_dir_path, exist_ok=True)
    time_arr = np.asarray(time_space).reshape(-1)
    pose = np.asarray(state_space_3d).reshape(time_arr.size, -1)
    pose_n = np.asarray(state_space_3d_with_noise).reshape(time_arr.size, -1)
    csv_path = os.path.join(exp_dir_path, f"{file_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "x", "y", "z", "x_n", "y_n", "z_n"])
        for i in range(time_arr.size):
            writer.writerow(
                [
                    time_arr[i],
                    pose[i, 0],
                    pose[i, 1],
                    pose[i, 2],
                    pose_n[i, 0],
                    pose_n[i, 1],
                    pose_n[i, 2],
                ]
            )
    return None


def execute(cfg: omegaconf.DictConfig, headless: bool = False) -> None:
    """Execute math toy environment Sim2Sim motion model learner feedback loop

    :param cfg: Hydra configuration file
    :param headless: turn environment rendering off (param for commandline flag)
    """
    exp_dir_relative_path = get_hydra_experiment_cwd(cfg)
    seed, torch_rng = seed_me(cfg, output_torch_rdn_generator=True)

    # :::: Define environment dynamic :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    state_space_label = cfg.environment.data.label
    env_initial_conditions = cfg.environment.data.trajectories
    if cfg.pipeline.animation.execute:
        env_initial_conditions = [
            env_initial_conditions[cfg.pipeline.animation.select_initial_conditions]
        ]
        env_cfg_nb = 1
        granularity = cfg.environment.time_space.granularity
        frame_len = cfg.pipeline.animation.show_environment_1d_subplot_interval_len
        frame_in = 1
        frame_out = frame_len
        subplot_interval_len = [int(frame_in), int(frame_out)]
        total_frame = int(granularity / frame_len)
        progressbar = tqdm(
            desc="Environment animation generation", leave=False, total=total_frame
        )
    else:
        env_cfg_nb = len(env_initial_conditions)
        total_frame = 1
        subplot_interval_len = cfg.pipeline.plot.show_environment_1d_subplot_interval
        progressbar = tqdm(
            desc="Environment plot generation", leave=False, total=env_cfg_nb
        )

        explorable_size = 0
        for explo_interval in omegaconf.OmegaConf.to_object(
            cfg.environment.explorable_space
        ):
            explorable_size += explo_interval[1] - explo_interval[0]
        granularity = cfg.environment.time_space.granularity
        progressbar.write(
            f"\nSample size:\n"
            f"    global: {granularity * env_cfg_nb}\n"
            f"    explorable: {explorable_size * env_cfg_nb}\n"
        )

    if cfg.pipeline.plot.show_training_trj:

        with warnings.catch_warnings():
            manage_matplotlib_warnings()
            manage_matplotlib_backend(cfg.pipeline.plot.show_environment_plot, headless)
            for each in range(total_frame):
                for idx, env_init_cfg in enumerate(env_initial_conditions):

                    state_space_3d_fct = math_environment_selector(cfg, env_init_cfg)

                    measurement_noise_cfgs, selected_burst_noise_cfgs = (
                        setup_burst_noise_config(cfg)
                    )

                    # .... Create trajectory ..........................................................
                    learning_env: Union[MathContinuousGymnasium, gym.Env] = gym.make(
                        "math_gymnasium:math-continuous-gymnasium-v0",
                        math_function_callback=state_space_3d_fct,
                        math_function_label=state_space_label,
                        time_axis_cfg=cfg.environment.time_space,
                        explorable_regions_cfg=cfg.environment.explorable_space,
                        measurement_noise_cfg=measurement_noise_cfgs,
                        time_function_callback=None,
                        # ToDo: RLRP-294 experiment with dt env obs noise and lag
                        observed_state_are_dt_derivatives=cfg.environment.obs_are_dt_derivatives,
                        observed_time_is_delta_time=cfg.environment.obs_time_is_delta_time,
                    )

                    # .... Plot trajectory ............................................................
                    if subplot_interval_len:
                        subplot_interval_ = slice(
                            subplot_interval_len[0], subplot_interval_len[1]
                        )
                    else:
                        subplot_interval_ = None

                    env_fig, ax_3d, ax_z, ax_x, ax_y = (
                        three_dimension_environment_space_plot(
                            cfg,
                            time_space=learning_env.trj.time_axis.wall,
                            state_space_3d=learning_env.trj.state_axes.poses,
                            state_space_3d_with_noise=learning_env.trj.state_axes.poses_with_noise,
                            title=f"State space and explored space. Environment configuration "
                            f"{idx + 1}/"
                            f"{env_cfg_nb}",
                            state_space_label=state_space_label,
                            subplot_1d_interval=subplot_interval_,
                            show_samples=True,
                            show_3d_grid=cfg.pipeline.plot.get("show_3d_grid", True),
                            show_3d_axes=cfg.pipeline.plot.get("show_3d_axes", True),
                            figsize=cfg.pipeline.plot.figsize,
                            figdpi=cfg.pipeline.plot.figdpi,
                            extra_info_str=(
                                f"{setup_noise_cfg_str(selected_burst_noise_cfgs, cfg.environment.global_measurement_noise)}\n"
                                f"  Initiale coordinates: {env_init_cfg.initiale_coordinates}\n"
                                f"  System params: {env_init_cfg.param}\n"
                            ),
                            experiment_id=get_hydra_experiment_id(),
                        )
                    )

                    if cfg.pipeline.animation.execute:
                        file_id = each + 1
                    else:
                        file_id = f"{idx + 1}_of_{env_cfg_nb}"

                    plot_file_name = (
                        f"state_space_and_explorable_space_samples"
                        f"_env_cfg_{file_id}"
                    )
                    show_and_save_plot_helper(
                        env_fig,
                        exp_dir_relative_path,
                        plot_file_name,
                        headless,
                        cfg.pipeline.plot.show_environment_plot,
                        cfg.pipeline.plot.save_dpi,
                    )
                    _save_trajectory_to_csv(
                        exp_dir_path=exp_dir_relative_path,
                        file_name=plot_file_name,
                        time_space=learning_env.trj.time_axis.wall,
                        state_space_3d=learning_env.trj.state_axes.poses,
                        state_space_3d_with_noise=learning_env.trj.state_axes.poses_with_noise,
                    )

                    if cfg.pipeline.animation.execute:
                        # time_increment = each * total_frame

                        if each <= total_frame / 4:
                            frame_in = 0
                        else:
                            factor = easeInOutSine(each / 10)
                            frame_in += frame_len * factor

                        frame_out += frame_len

                        subplot_interval_len = [int(frame_in), int(frame_out)]

                    progressbar.update()

        # .... Train Plot Teardown ................................................................
        progressbar.close()
        learning_env.close()

    # :::: Test target trajectories plots ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Updated by phase 7 of the math_env multi test-trajectory `.junie` plan
    # (``feature_math_env_multi_test_trajectory_plan_20260517.md``): iterate
    # over ``setup_test_target_rollouts`` lists so every entry in
    # ``cfg.environment.data.test_{InD,OOD}_trajectory`` gets its own plot.
    if cfg.pipeline.plot.show_ind_tests_trj or cfg.pipeline.plot.show_ood_tests_trj:
        target_InD_rollouts, target_OOD_rollouts = setup_test_target_rollouts(
            cfg, exp_dir_relative_path, headless
        )
        test_groups = []
        if cfg.pipeline.plot.show_ind_tests_trj:
            test_groups.append(
                (
                    "InD",
                    target_InD_rollouts,
                    list(cfg.environment.data.test_InD_trajectory or []),
                    cfg.environment.measurement_noise,
                    cfg.environment.global_measurement_noise,
                )
            )
        if cfg.pipeline.plot.show_ood_tests_trj:
            test_groups.append(
                (
                    "OoD",
                    target_OOD_rollouts,
                    list(cfg.environment.data.test_OOD_trajectory or []),
                    cfg.environment.ood_test_measurement_noise,
                    cfg.environment.ood_test_global_measurement_noise,
                )
            )

        for group_label, entries, traj_cfgs, group_noise_cfg, group_global_noise_cfg in test_groups:
            total_n = len(entries)
            progressbar = tqdm(
                desc=f"{group_label} test environment plot generation",
                leave=False,
                total=total_n,
            )

            with warnings.catch_warnings():
                manage_matplotlib_warnings()
                manage_matplotlib_backend(
                    cfg.pipeline.plot.show_environment_plot, headless
                )

                for idx, (entry, traj_cfg) in enumerate(
                    zip(entries, traj_cfgs), start=1
                ):
                    env_initial_coord = traj_cfg.initiale_coordinates
                    param = traj_cfg.param

                    # .... Plot trajectory ....................................................
                    env_fig, ax_3d, ax_z, ax_x, ax_y = (
                        three_dimension_environment_space_plot(
                            cfg,
                            time_space=entry.env.timestamps,
                            state_space_3d=entry.env.pose_gt,
                            state_space_3d_with_noise=entry.env.pose,
                            title=(
                                f"State space and explored space. "
                                f"{group_label} tests environment configuration "
                                f"{idx}/{total_n}"
                            ),
                            state_space_label=state_space_label,
                            subplot_1d_interval=None,
                            show_samples=True,
                            show_3d_grid=cfg.pipeline.plot.get("show_3d_grid", True),
                            show_3d_axes=cfg.pipeline.plot.get("show_3d_axes", True),
                            figsize=cfg.pipeline.plot.figsize,
                            figdpi=cfg.pipeline.plot.figdpi,
                            extra_info_str=(
                                f"{setup_noise_cfg_str(group_noise_cfg, group_global_noise_cfg)}\n"
                                f"  Trajectory id: {entry.short_name}\n"
                                f"  Initiale coordinates: {env_initial_coord}\n"
                                f"  System params: {param}\n"
                            ),
                            experiment_id=get_hydra_experiment_id(),
                        )
                    )

                    plot_file_name = (
                        f"state_space_and_explorable_space_samples_{group_label}_test_{idx}_of_{total_n}"
                    )
                    show_and_save_plot_helper(
                        env_fig,
                        exp_dir_relative_path,
                        plot_file_name,
                        headless,
                        cfg.pipeline.plot.show_environment_plot,
                        cfg.pipeline.plot.save_dpi,
                    )
                    _save_trajectory_to_csv(
                        exp_dir_path=exp_dir_relative_path,
                        file_name=plot_file_name,
                        time_space=entry.env.timestamps,
                        state_space_3d=entry.env.pose_gt,
                        state_space_3d_with_noise=entry.env.pose,
                    )

                    progressbar.update()

            # .... Test Plot Teardown ........................................................
            progressbar.close()

    # ==== Teardown ===============================================================================
    plt.close("all")
    return None
