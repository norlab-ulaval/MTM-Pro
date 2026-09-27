# coding=utf-8
# coding=utf-8
from typing import Callable, Optional, Union

import omegaconf
import torch
from mbrl import models as models
from mbrl.util import ReplayBuffer

from tools.r2s_motion_model_container_tools.container import R2SMotionModelContainer
from algorithm.experience_replay_learning_loop import (
    AbstractExperienceReplayLearningLoop,
)
from tools.f110_gym_env_tools.plot_tensorboard import OnlineTensorboardWritter
from trajectory_container_tools.dataclasses.erll_trajectory_dataclass import (
    TestMotionTrajectoryDataclass,
    TestTrajectoryDataclass,
)

class ProgressiveBatchExperienceReplay(AbstractExperienceReplayLearningLoop):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        cfg_training: omegaconf.DictConfig,
        model_trainer: models.ModelTrainer,
        source_replay_buffer: ReplayBuffer,
        test_trajectories: Union[list[TestTrajectoryDataclass], list[TestMotionTrajectoryDataclass]],
        tensorboard_writer: Optional[OnlineTensorboardWritter],
        erll_epoch_pre_training_callback: Optional[Callable],
        model_trainer_epoch_callback: Optional[Callable],
        model_trainer_batch_callback: Optional[Callable],
        erll_epoch_post_training_callback: Optional[Callable],
        torch_rng: torch.Generator,
        motion_model_container: R2SMotionModelContainer = None,
        replay_buffer_learning_loop_cfg_key: str = "PBER",
    ):
        super().__init__(
            cfg,
            cfg_training,
            model_trainer,
            source_replay_buffer,
            test_trajectories,
            tensorboard_writer,
            erll_epoch_pre_training_callback,
            model_trainer_epoch_callback,
            model_trainer_batch_callback,
            erll_epoch_post_training_callback,
            torch_rng,
            motion_model_container,
            replay_buffer_learning_loop_cfg_key,
        )

    @property
    def cfg_algo_required_key(self) -> list[str]:
        """
        Extends the base required keys with the progressive batch-size schedule parameters
        (``batch_size.gamma`` / ``batch_size.limit``) owned by this class since RLRP-727. The
        partition-based UDER inherits this contract; the single-global-loop child does not.
        """
        return super().cfg_algo_required_key + [
            "batch_size.gamma",
            "batch_size.limit",
        ]

    def _batch_size_scheduling(self, epoch: int) -> int:
        """
        Geometric progressive batch-size schedule (RLRP-727 owner = PBER).

        The initialization epoch (``epoch == 0``) uses ``batch_size.init_value``; subsequent epochs
        grow (or shrink) geometrically as ``init_value * gamma**epoch``, clamped to
        ``batch_size.limit`` (upper clamp when ``gamma >= 1.0``, lower clamp otherwise).

        :param epoch: the current ERLL epoch index.
        :return: the integer batch size for this ERLL epoch.
        """
        bs_cfg = self._cfg_algo.batch_size
        if epoch > 0:
            new_batch_size = bs_cfg.init_value * (bs_cfg.gamma**epoch)
            if bs_cfg.gamma >= 1.0:
                if new_batch_size > bs_cfg.limit:
                    new_batch_size = bs_cfg.limit
            else:
                if new_batch_size < bs_cfg.limit:
                    new_batch_size = bs_cfg.limit
            new_batch_size = int(new_batch_size)
        else:
            new_batch_size = bs_cfg.init_value
        return new_batch_size

    def _pre_train_step(
        self,
        current_erll_epoch: int,
        erll_epoch_batch_size: int,
        is_erll_initialization_epoch: bool,
        is_final_erll_epoch: bool,
    ) -> None:
        # Nothing to do
        return None

    def _post_train_step(
        self,
        current_erll_epoch: int,
        erll_epoch_batch_size: int,
        is_erll_initialization_epoch: bool,
        is_final_erll_epoch: bool,
    ) -> None:
        # Nothing to do
        return None
