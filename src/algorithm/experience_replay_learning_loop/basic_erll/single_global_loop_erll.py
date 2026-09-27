# coding=utf-8
from typing import Callable, List, Optional, Tuple, Union

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


class SingleGlobalLoopERLL(AbstractExperienceReplayLearningLoop):
    """
    Single-global-loop ERLL variant (RLRP-727).

    A first-class way to run **one** global training pass with the ERLL machinery disabled — the
    clean control for ablation studies and for validating the optimizer/LR in isolation.

    Unlike the progressive PBER/UDER loops (which run one mandatory initialization pass followed by
    ``uder_num_epochs`` progressive passes), this loop collapses the initialization and final stages
    into a **single fused pass** where ``is_erll_initialization_epoch == is_final_erll_epoch ==
    True``. That fused pass takes the ``is_final`` branch of ``execute()`` and therefore draws its
    epoch budget from ``final_max_num_epochs_train_model`` (operator decision D2), while
    ``num_epochs_train_model`` is set to the sentinel ``0`` (see
    :attr:`_allows_zero_num_epochs_train_model`).

    Overrides (everything needed to recover the single-global-loop):

    - :meth:`_iter_erll_epochs` → exactly ``[(0, True, True)]`` (one fused pass);
    - :meth:`_batch_size_scheduling` → constant ``batch_size.init_value`` (no progressive growth);
    - :attr:`_allows_zero_num_epochs_train_model` → ``True`` (accepts ``num_epochs_train_model: 0``);
    - :meth:`_pre_train_step` / :meth:`_post_train_step` → no-op.
    """

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
        replay_buffer_learning_loop_cfg_key: str = "UDER",
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
    def _allows_zero_num_epochs_train_model(self) -> bool:
        # RLRP-727: the single fused pass uses `final_max_num_epochs_train_model`, so the
        # per-erll-epoch divisor derived from `num_epochs_train_model` is never needed; accept the
        # `num_epochs_train_model: 0` sentinel and skip the base divisibility asserts.
        return True

    def _iter_erll_epochs(self) -> List[Tuple[int, bool, bool]]:
        # One fused pass: initialization == final == True → single global training loop.
        return [(0, True, True)]

    def _batch_size_scheduling(self, epoch: int) -> int:
        # Constant batch size (no progressive growth) for the single global pass.
        return self._cfg_algo.batch_size.init_value

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
