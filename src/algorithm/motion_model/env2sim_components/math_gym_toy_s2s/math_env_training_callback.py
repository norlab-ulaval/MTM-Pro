# coding=utf-8
import time
import warnings
from typing import Callable, Optional, Union
import os

import mbrl.models
import omegaconf
from mbrl import models as models
from mbrl.util import ReplayBuffer

from math_gymnasium.envs.arbitrary_dim_math_continuous import (
    MathContinuousGymnasium,
)
from math_gymnasium.tools.utils import \
    math_continuous_gymnasium_env_to_test_motion_trajectory_dataclass
from tools.f110_gym_env_tools.plot_tensorboard import OnlineTensorboardWritter
from algorithm.experience_replay_learning_loop.uder_plot_utils import three_dimension_partition_based_UDER_sampling_plot
from tools.multistep_tools.models import MultiStepMLP
from tools.plot_tools.plot_management import (
    manage_matplotlib_backend,
    manage_matplotlib_warnings,
    show_and_save_plot_helper,
)
from trajectory_container_tools.dataclasses import TestMotionTrajectoryDataclass
from trajectory_container_tools.dataclasses.math_gymnasium_trajectory_dataclass import (
    MathEnvTrajectoryDataclass,
)


def setup_math_env_uder_buffer_sampling_plotter_callback(
    cfg: omegaconf.DictConfig,
    cfg_training: omegaconf.DictConfig,
    model_trainer: models.ModelTrainer,
    tensorboard_writer: Optional[OnlineTensorboardWritter],
    test_env: Union[MathEnvTrajectoryDataclass, MathContinuousGymnasium, TestMotionTrajectoryDataclass],
    state_space_label: str,
    exp_dir_relative_path: str,
    headless: bool,
) -> Callable:
    """`uncertainty_driven_experience_replay` callback for plotting dataset sampling progression

    - Require the source `replay_buffer` be collected using the `collect_full_time_space_rollout`
      method with parameter `replace_rewards_with_timestep_index=True`.
    - Require `pipeline.plot.render_dataset_exploration_plot=true`

    """
    assert isinstance(test_env, (MathEnvTrajectoryDataclass, MathContinuousGymnasium, TestMotionTrajectoryDataclass))
    if not isinstance(test_env, TestMotionTrajectoryDataclass):
        trj = math_continuous_gymnasium_env_to_test_motion_trajectory_dataclass(test_env)

    uder_epoch_plot_dir_path = os.path.join(exp_dir_relative_path, "uder_epoch_plot")
    if not os.path.exists(uder_epoch_plot_dir_path):
        os.makedirs(uder_epoch_plot_dir_path)

    if isinstance(model_trainer.model, MultiStepMLP):
        horizon_len = model_trainer.model.horizon_len
    else:
        horizon_len = None

    start_time = time.time()

    def math_env_uder_buffer_sampling_plotter_callback(
        model: mbrl.models.Model,
        uder_epoch: int,
        global_epoch: int,
        replay_buffer: ReplayBuffer,
    ) -> None:
        """Called by `uncertainty_driven_experience_replay` method at the end of each dataset
        optimization epoch.
        Require `pipeline.plot.render_dataset_exploration_plot=true`
        """

        if cfg.pipeline.plot.render_dataset_exploration_plot:
            with warnings.catch_warnings():
                manage_matplotlib_warnings()
                manage_matplotlib_backend(
                    cfg.pipeline.plot.show_environment_plot, headless
                )

                training_time = time.time() - start_time

                env_fig, ax_3d, ax_z, ax_x, ax_y = (
                    three_dimension_partition_based_UDER_sampling_plot(
                        cfg,
                        replay_buffer,
                        uder_epoch,
                        global_epoch,
                        trj.actions,
                        trj.pose_gt,
                        trj.pose,
                        state_space_label,
                        training_time,
                        horizon_len,
                        model.model,
                    )
                )

                file_name = (
                    f"dataset_exploration_policy_selected_samples_epoch_"
                    f""
                    f"{uder_epoch}"
                )
                show_and_save_plot_helper(env_fig, uder_epoch_plot_dir_path, file_name, headless,
                                          cfg.pipeline.plot.show_dataset_exploration_plot,
                                          cfg.pipeline.plot.save_dpi)

                if tensorboard_writer is not None:
                    tensorboard_writer.writer.add_figure(
                        tag="Replay buffer optimization sampling",
                        figure=env_fig,
                        close=False,
                        global_step=uder_epoch,
                    )

        return None

    return math_env_uder_buffer_sampling_plotter_callback
