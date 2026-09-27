# coding=utf-8
import warnings

import omegaconf

from tqdm import tqdm
import numpy as np
from algorithm.utils import seed_me
from pipeline.pipeline_utils.robotic_env_pipeline_utils.setup_utils import (
    setup_trajectory_from_csv,
)
from pipeline.pipeline_utils.robotic_env_pipeline_utils.utils import (
    compute_position_from_velocity_and_attitude,
    resolve_max_angular_velocity_from_cfg,
)
from tools.console_tools.message import consol_msg_universal_one_liner
from tools.hydra_apps_tools.hydra_utils import (
    get_hydra_experiment_cwd,
    get_hydra_experiment_id,
)

from math_gymnasium.tools.plot_3d_utils import (
    three_dimension_environment_space_plot,
)

from tools.math_tools.ease_in_ease_out import easeInOutSine
from tools.math_tools.space_conversion_tools.coordinate_to_velocity import (
    convert_state_coordinate_to_state_derivatives,
    convert_state_derivatives_to_state_coordinate,
)
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


def execute(cfg: omegaconf.DictConfig, headless: bool = False) -> None:
    """Execute math toy environment Sim2Sim motion model learner feedback loop

    :param cfg: Hydra configuration file
    :param headless: turn environment rendering off (param for commandline flag)
    """
    exp_dir_relative_path = get_hydra_experiment_cwd(cfg)
    seed, torch_rng = seed_me(cfg, output_torch_rdn_generator=True)

    USE_DATASET_POSE = cfg.deploy.trajectory_computation.use_dataset_pose
    COMPUTE_VEL_FROM_WORLD_POSE = (
        cfg.deploy.trajectory_computation.compute_vel_from_world_pose
    )
    ATTITUDE_PROPAGATION = cfg.deploy.trajectory_computation.attitude_propagation
    # RLRP-761 S5.8: fail-fast (or at least WARN) instead of silently degenerating
    # when the attitude is propagated with no orientation feature in obs_dims.
    from tools.feature_handling_tools.env_handlers import (
        validate_orientation_propagation_source,
    )

    validate_orientation_propagation_source(cfg)
    QUATERNION_ANG_VEL_BLEND = (
        cfg.deploy.trajectory_computation.quaternion_ang_vel_blend
    )
    SRC_TIMESTAMPS = cfg.deploy.trajectory_computation.get("src_timestamps", True)

    ts_console_str = f"{'src' if SRC_TIMESTAMPS else 'recomputed'} timestamps"
    if not USE_DATASET_POSE:
        consol_msg_universal_one_liner(
            "Compute trajectory from dataset velocity "
            f"using linear={cfg.environment.data.linear_velocity_frame}/"
            f"angular={cfg.environment.data.angular_velocity_frame} frame velocities and {ts_console_str}"
        )
    elif USE_DATASET_POSE:
        if COMPUTE_VEL_FROM_WORLD_POSE:
            consol_msg_universal_one_liner(
                "Compute trajectory from world frame velocity re-computed from dataset poses and {ts_console_str}"
            )
        else:
            consol_msg_universal_one_liner(
                "Compute trajectory from dataset world frame pose"
            )

    # :::: Define environment dynamic :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    trajectories_cfg = cfg.environment.data.trajectories
    # Normalise ``null`` (YAML ``None``) test-trajectory lists to ``[]`` so a
    # simulator config with zero OOD (or InD) test trajectories iterates cleanly
    # (feature_empty_ood_test_trajectory_support_plan_20260821_RLRP-778.md).
    test_InD_trajectory_cfg = cfg.environment.data.test_InD_trajectory or []
    test_OOD_trajectory_cfg = cfg.environment.data.test_OOD_trajectory or []

    if cfg.pipeline.animation.execute:
        raise NotImplementedError(
            "(NICE TO HAVE) ToDo: implement animation plot support"
        )
        trajectories_cfg = [
            trajectories_cfg[cfg.pipeline.animation.select_initial_conditions]
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
        env_cfg_nb = len(trajectories_cfg)
        total_frame = 1
        subplot_interval_len = cfg.pipeline.plot.show_environment_1d_subplot_interval
        # progressbar = tqdm(
        #     desc="Environment plot generation", leave=False, total=env_cfg_nb
        # )

        # explorable_size = 0
        # for explo_interval in omegaconf.OmegaConf.to_object(
        #     cfg.environment.explorable_space
        # ):
        #     explorable_size += explo_interval[1] - explo_interval[0]
        # granularity = cfg.environment.time_space.granularity
        # progressbar.write(
        #     f"\nSample size:\n"
        #     f"    global: {granularity * env_cfg_nb}\n"
        #     f"    explorable: {explorable_size * env_cfg_nb}\n"
        # )

    with warnings.catch_warnings():
        manage_matplotlib_warnings()
        manage_matplotlib_backend(cfg.pipeline.plot.show_environment_plot, headless)

        for each in range(total_frame):
            plot_InD_trajectories = []
            plot_OoD_trajectories = []

            if not cfg.pipeline.plot.show_training_trj:
                trajectories_cfg = []

            if cfg.pipeline.plot.show_ind_tests_trj:
                for each_InD_cfg in test_InD_trajectory_cfg:
                    plot_InD_trajectories.append(
                        omegaconf.OmegaConf.create(
                            {"trajectory_name": each_InD_cfg.trajectory_name}
                        )
                    )
            else:
                test_InD_trajectory_cfg = []

            if cfg.pipeline.plot.show_ood_tests_trj:
                for each_OoD_cfg in test_OOD_trajectory_cfg:
                    plot_OoD_trajectories.append(
                        omegaconf.OmegaConf.create(
                            {"trajectory_name": each_OoD_cfg.trajectory_name}
                        )
                    )
            else:
                test_OOD_trajectory_cfg = []

            if (
                len(trajectories_cfg) == 0
                and len(plot_InD_trajectories) == 0
                and len(plot_OoD_trajectories) == 0
            ):
                consol_msg_universal_one_liner("No trajectory to plot")
                continue

            for k, trj_cfg in [
                ["train_val_trajectories", trajectories_cfg],
                ["test_InD_trajectory", plot_InD_trajectories],
                ["test_OOD_trajectory", plot_OoD_trajectories],
            ]:
                total_trj_len = []

                for idx, env_init_cfg in enumerate(trj_cfg):

                    # .... Create trajectory ..........................................................
                    trajectory = setup_trajectory_from_csv(cfg, env_init_cfg)
                    # print(trajectory)

                    # .... Plot trajectory ............................................................
                    if subplot_interval_len:
                        subplot_interval_ = slice(
                            subplot_interval_len[0], subplot_interval_len[1]
                        )
                    else:
                        subplot_interval_ = None

                    timestamps = trajectory.timestamps
                    if USE_DATASET_POSE:

                        if COMPUTE_VEL_FROM_WORLD_POSE:
                            consol_msg_universal_one_liner("Compute trajectory from world frame velocity re-computed from dataset poses to assess timestamps resolution in post-processing")
                            DELTA_TIME = True
                            if SRC_TIMESTAMPS:
                                if DELTA_TIME:
                                    ts_ = timestamps.delta_stamps
                                else:
                                    ts_ = timestamps
                            else:
                                raise DeprecationWarning(
                                    "TCT Timestamps now support float"
                                )
                                if DELTA_TIME:
                                    ts_ = trajectory.timestamps.delta_stamps
                                else:
                                    ts_ = trajectory.timestamps.stamps

                            vel_world = convert_state_coordinate_to_state_derivatives(
                                trajectory.poses.stack,
                                delta_time_space=timestamps.stamps,
                                time_space_is_delta_time=False,
                            )
                            poses = convert_state_derivatives_to_state_coordinate(
                                vel_world,
                                # delta_time_space=trajectory.bag_timestamps.delta_stamps,
                                delta_time_space=ts_,
                                # initiale_coordinates=env_init_cfg.initiale_coordinates,
                                initiale_coordinates=trajectory.poses.stack[0],
                                time_space_is_delta_time=DELTA_TIME,
                            )
                        else:
                            # Compute trajectory from dataset world frame pose
                            poses = trajectory.poses.stack

                    else:
                        # Compute trajectory from dataset velocity
                        if SRC_TIMESTAMPS:
                            ts_ = timestamps
                        else:
                            raise DeprecationWarning("TCT Timestamps now support float")
                            ts_ = trajectory.timestamps
                        poses, vel_world = compute_position_from_velocity_and_attitude(
                            linear_velocity=trajectory.linear_vels.stack,
                            angular_velocity=trajectory.angular_vels.stack,
                            quaternions=trajectory.attitude.stack, timestamps=ts_,
                            initial_position=trajectory.poses.stack[0],
                            # RLRP-791: state the start attitude explicitly (it was
                            # an implicit `quaternions[0]` read). Same value, but now
                            # symmetric with `initial_position` and self-documenting.
                            initial_orientation=trajectory.attitude.stack[0],
                            linear_velocity_frame=trajectory.linear_velocity_frame,
                            angular_velocity_frame=trajectory.angular_velocity_frame,
                            interpolate_quaternions=ATTITUDE_PROPAGATION,
                            quaternion_blend_weight=QUATERNION_ANG_VEL_BLEND,
                            max_angular_velocity=resolve_max_angular_velocity_from_cfg(cfg),
                            integration_scheme=cfg.deploy.trajectory_computation.get("integration_scheme", "forward_euler"),
                            debug=True)

                    with omegaconf.read_write(cfg):
                        omegaconf.OmegaConf.update(
                            cfg,
                            "environment.time_space.granularity",
                            len(trajectory),
                            merge=False,
                        )

                    ts_str = f"Δt from {'src timestamps' if SRC_TIMESTAMPS else 'recomputed timestamps'}\n"
                    if cfg.deploy.trajectory_computation.use_dataset_pose:
                        if (
                            not cfg.deploy.trajectory_computation.compute_vel_from_world_pose
                        ):
                            trj_compu_str = f"  Compute world pose from dataset pose\n"
                        else:
                            trj_compu_str = (
                                f"  Compute world pose from dataset pose\n"
                                f"  world pose → world velocity x dt → world pose\n"
                                f"  with {ts_str}\n"
                            )
                    else:
                        if cfg.deploy.trajectory_computation.attitude_propagation:
                            trj_compu_str = (
                                f"  Compute world pose from dataset linear={trajectory.linear_velocity_frame}/angular={trajectory.angular_velocity_frame} velocity (linear, angular) and attitude\n"
                                f"    Attitude propagation: {cfg.deploy.trajectory_computation.attitude_propagation}\n"
                                f"    Quaternion angular velocity blend: {cfg.deploy.trajectory_computation.quaternion_ang_vel_blend}\n"
                            )
                        else:
                            trj_compu_str = (
                                f"  Compute world pose from dataset linear={trajectory.linear_velocity_frame} velocity (linear) and attitude\n"
                                f"    Attitude propagation: {cfg.deploy.trajectory_computation.attitude_propagation}\n"
                            )

                        trj_compu_str += f"    {ts_str}\n"

                    env_fig, ax_3d, ax_z, ax_x, ax_y = (
                        three_dimension_environment_space_plot(
                            cfg,
                            time_space=timestamps.stamps,
                            state_space_3d=trajectory.poses.stack,
                            state_space_3d_with_noise=poses,
                            title=f"State space and explored space. Environment configuration "
                            f"{idx + 1}/"
                            f"{env_cfg_nb}",
                            state_space_label=cfg.environment.data.label,
                            subplot_1d_interval=subplot_interval_,
                            show_samples=True,
                            show_3d_grid=cfg.pipeline.plot.get("show_3d_grid", True),
                            show_3d_axes=cfg.pipeline.plot.get("show_3d_axes", True),
                            figsize=cfg.pipeline.plot.figsize,
                            figdpi=cfg.pipeline.plot.figdpi,
                            show_explorable_space=False,
                            extra_info_str=(
                                f"  Initiale coordinates: {poses[0]}\n"
                                f"  Trajectory recording: {env_init_cfg.trajectory_name}\n"
                                f"{trj_compu_str}"
                            ),
                            experiment_id=get_hydra_experiment_id(),
                        )
                    )

                    total_trj_len.append(len(timestamps.stamps))

                    if cfg.pipeline.animation.execute:
                        file_id = each + 1
                    else:
                        file_id = f"{idx + 1}_of_{env_cfg_nb}"

                    show_and_save_plot_helper(
                        env_fig,
                        exp_dir_relative_path,
                        f"state_space_and_explorable_space_samples"
                        f"_env_cfg_{file_id}",
                        headless,
                        cfg.pipeline.plot.show_environment_plot,
                        cfg.pipeline.plot.save_dpi,
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

                    # progressbar.update()

                total_trj_len = np.array(total_trj_len)
                if len(total_trj_len) > 0:
                    print(f"\n{k} length:")
                    for idx, each in enumerate(total_trj_len):
                        print(f"   {idx}: {each}")
                    print(f"\nTotal {k} trajectory length: {total_trj_len.sum()}\n")

            # progressbar.close()
    # ==== Teardown ===============================================================================

    plt.close("all")
    return None
