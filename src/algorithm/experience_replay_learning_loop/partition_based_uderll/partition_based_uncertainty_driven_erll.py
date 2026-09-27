# coding=utf-8
from typing import Callable, Optional, Tuple, Union

import mbrl.models
import mbrl.util
import omegaconf
import torch
from mbrl import models as models
from mbrl.util import ReplayBuffer
from tqdm import tqdm

from algorithm.experience_replay_learning_loop.core.replay_buffer_utils import (
    explore_and_partition_replay_buffer,
)
from tools.r2s_motion_model_container_tools.container import R2SMotionModelContainer
from algorithm.experience_replay_learning_loop.basic_erll.progressive_batch_experience_replay import (
    ProgressiveBatchExperienceReplay,
)

from algorithm.policy.replay_buffer_exploration_policy import (
    BaseReplayBufferExplorationPolicy,
    PartitionBasedReplayBufferExplorationPolicy,
    RandomReplayBufferExplorationPolicy,
)
from tools.f110_gym_env_tools.plot_tensorboard import OnlineTensorboardWritter
from tools.multistep_tools.models import (
    AbstractFeatureWeightedMultiStepMLP,
    MultiStepMLP,
)
from trajectory_container_tools.dataclasses.erll_trajectory_dataclass import (
    TestMotionTrajectoryDataclass,
    TestTrajectoryDataclass,
)


class PartitionBasedUncertaintyDrivenERLL(ProgressiveBatchExperienceReplay):
    """
    Partition-based Uncertainty-Driven ERLL (UDER).

    RLRP-727: UDER *is* "PBER + uncertainty-driven partitioning", so it now inherits the geometric
    progressive batch-size schedule and the ``batch_size.gamma`` / ``batch_size.limit`` required-key
    contract from :class:`ProgressiveBatchExperienceReplay`, and adds the partition pre-train step.
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

        assert omegaconf.OmegaConf.is_dict(
            cfg.get(self._cfg_key_algo).replay_buffer_size
        )

        self.uder_scan_window_len = (
            self._cfg_algo.uder_exploration_policy.scan_window_len
        )
        assert (
            self.uder_scan_window_len >= self._buffer_iterator_sequence_len
        ), f"{self.uder_scan_window_len} !>= {self._buffer_iterator_sequence_len}"
        self.uder_replay_buffer_size: Union[None, int] = None
        self.uder_exploration_policy: Union[None, BaseReplayBufferExplorationPolicy] = (
            None
        )

    @property
    def cfg_algo_required_key(self):
        return super().cfg_algo_required_key + [
            "replay_buffer_size",
            "uder_exploration_policy.scan_window_len",
        ]

    def _check_data_source_supported(self, data_source) -> None:
        """RLRP-824: the partition surgery of :meth:`_pre_train_step` re-partitions a materialized
        ``ReplayBuffer`` with an exploration policy; a lazy window data source has no buffer to
        partition (deferred follow-up: a ``Sampler`` re-implementation)."""
        raise NotImplementedError(
            f"{self._console_acronymn} (UDER.loop_kind=uder) does not support a non-replay-buffer "
            f"data source ({type(data_source).__name__}, pipeline.data_manager=dataloader): the "
            "uncertainty-driven partition surgery needs a materialized mbrl ReplayBuffer. Use "
            "loop_kind=single_global_loop or pber (RLRP-824 deferred follow-up)."
        )

    def _pre_train_step(
        self,
        current_erll_epoch: int,
        erll_epoch_batch_size: int,
        is_erll_initialization_epoch: bool,
        is_final_erll_epoch: bool,
    ) -> None:
        """
        Execute an exploration and partitioning strategy on the replay buffer based on uncertainty,
         while optionally performing buffer normalization.

        This function interacts heavily with the `train_replay_buffer`, exploration policy,
         and model trainer to prepare for further training processes. It also incorporates
         configurable parameters to enable flexibility in behavior depending on the specific
         pre-training workflow.

        :param current_erll_epoch: The current epoch count during ERLL training.
        :param erll_epoch_batch_size: The size of the replay buffer batch used in ERLL training.
        :param is_erll_initialization_epoch: Flag indicating whether the training is on the
            initialization epoch for ERLL.
        :param is_final_erll_epoch: Flag indicating whether the training is on the final epoch
            for ERLL.
        :return: None
        """
        (
            self.train_replay_buffer,
            self.uder_replay_buffer_size,
            self.uder_exploration_policy,
        ) = _explore_and_partition_replay_buffer_base_on_uncertainty(
            self._cfg,
            self.iter0_train_replay_buffer,
            self.uder_exploration_policy,
            self.model_trainer,
            self.motion_model_container,
            self._buffer_max_trajectory_len,
            self.uder_replay_buffer_size,
            erll_epoch_batch_size,
            current_erll_epoch,
            self.uder_scan_window_len,
            is_erll_initialization_epoch,
            is_final_erll_epoch,
            self._progressbar_main,
            self.tensorboard_writer,
            self.torch_rng,
        )

        if self._cfg_algo.uder_exploration_policy.get(
            "uder_epoch_buffer_normalization", False
        ):
            self.model_trainer.model.update_normalizer(
                self.train_replay_buffer.get_all()
            )

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


def _explore_and_partition_replay_buffer_base_on_uncertainty(
    cfg: omegaconf.DictConfig,
    train_replay_buffer: ReplayBuffer,
    uder_exploration_policy: BaseReplayBufferExplorationPolicy,
    model_trainer: mbrl.models.ModelTrainer,
    motion_model_container: R2SMotionModelContainer,
    buffer_max_trajectory_len: int,
    erll_replay_buffer_size: int,
    erll_batch_size: int,
    erll_current_epoch: int,
    scan_window_len: int,
    is_erll_initialization_epoch: int,
    is_final_erll_epoch: int,
    progressbar_main: tqdm,
    tensorboard_writer: Optional[OnlineTensorboardWritter],
    torch_rng: torch.Generator,
) -> Tuple[ReplayBuffer, int, BaseReplayBufferExplorationPolicy]:
    if is_erll_initialization_epoch:
        uder_exploration_policy = RandomReplayBufferExplorationPolicy(
            cfg, train_replay_buffer
        )
        erll_replay_buffer_size = cfg.UDER.replay_buffer_size.init_value
    else:
        progressbar_main.display("Update exploration policy")

        # (CRITICAL) ToDo: assessment >> not sure if its still usefull to fetch from container
        if isinstance(model_trainer.model.model, MultiStepMLP):
            assert motion_model_container is not None, (
                f"MultiStepMLP model require a R2SMotionModelContainer object, "
                f"curently motion_model_container is set to None"
            )
            dynamic_model = motion_model_container
        else:
            dynamic_model = model_trainer.model

        if isinstance(
            uder_exploration_policy, PartitionBasedReplayBufferExplorationPolicy
        ):
            uder_exploration_policy.update_policy(train_replay_buffer, dynamic_model)
        else:
            # noinspection PyTypeChecker
            uder_exploration_policy = PartitionBasedReplayBufferExplorationPolicy(
                cfg, train_replay_buffer, dynamic_model, torch_rng
            )

        # .... Dataset size .......................................................................
        assert omegaconf.OmegaConf.is_dict(cfg.UDER.replay_buffer_size)
        rbs_cfg = cfg.UDER.replay_buffer_size
        assert rbs_cfg.gamma <= 1.0
        assert rbs_cfg.gamma_out >= 1.0
        if erll_current_epoch <= rbs_cfg.out_epoch:
            replay_buffer_size_epoch = erll_current_epoch
            erll_replay_buffer_size = rbs_cfg.init_value * (
                rbs_cfg.gamma**replay_buffer_size_epoch
            )
            erll_replay_buffer_size -= erll_replay_buffer_size % scan_window_len
            if erll_replay_buffer_size < rbs_cfg.floor:
                erll_replay_buffer_size = rbs_cfg.floor
        else:
            replay_buffer_size_epoch = erll_current_epoch - rbs_cfg.out_epoch
            erll_replay_buffer_size = erll_replay_buffer_size * (
                rbs_cfg.gamma_out**replay_buffer_size_epoch
            )
            erll_replay_buffer_size -= erll_replay_buffer_size % scan_window_len
            if erll_replay_buffer_size > rbs_cfg.out_value:
                erll_replay_buffer_size = rbs_cfg.out_value

        erll_replay_buffer_size = int(erll_replay_buffer_size)
        assert erll_replay_buffer_size <= train_replay_buffer.num_stored, (
            f"{erll_replay_buffer_size=} !<= "
            f"{train_replay_buffer.num_stored=}. "
            "Check train/validation dataset split configuration "
            f"{cfg.source_replay_buffer.source_size=} * "
            f"{cfg.source_replay_buffer.val_ratio=}"
        )
        assert (
            erll_batch_size <= erll_replay_buffer_size
        ), f"{erll_batch_size=} !<= {erll_replay_buffer_size=}"

    # .... Construct the optimized replay buffer ..................................................
    optimized_replay_buffer, _, _ = explore_and_partition_replay_buffer(
        uder_exploration_policy,
        erll_replay_buffer_size,
        replay_buffer=train_replay_buffer,
        buffer_max_trajectory_len=buffer_max_trajectory_len,
    )

    # .... Update MS model observation feature weights ............................................
    if isinstance(model_trainer.model.model, AbstractFeatureWeightedMultiStepMLP):
        if is_erll_initialization_epoch:
            new_feature_weight = model_trainer.model.model.obs_feature_weights
        else:
            # Torch-first: convert tensor to numpy at the numpy-only method boundary
            pred_epi_np = uder_exploration_policy.prediction_epistemic_uncertainty.cpu().numpy()
            new_feature_weight = model_trainer.model.model.compute_ideal_obs_feature_weights_update_from_prediction_std_trajectory(
                pred_epi_np,
                scale_min=cfg.UDER.feature_weight_scale_min,
                scale_max=cfg.UDER.feature_weight_scale_max,
            )

            model_trainer.model.model.set_observation_feature_weights(
                new_feature_weight
            )
            model_trainer.model.model.set_action_feature_weights(new_feature_weight)

        if tensorboard_writer is not None and len(new_feature_weight) >= 3:
            new_obs_feature_weight = new_feature_weight[:3]
            tensorboard_writer.add_scalar_per_epoch_monitoring(
                tag="Observation feature weight/X", value=new_obs_feature_weight[0]
            )
            tensorboard_writer.add_scalar_per_epoch_monitoring(
                tag="Observation feature weight/Y", value=new_obs_feature_weight[1]
            )
            tensorboard_writer.add_scalar_per_epoch_monitoring(
                tag="Observation feature weight/Z", value=new_obs_feature_weight[2]
            )

            if isinstance(
                uder_exploration_policy, PartitionBasedReplayBufferExplorationPolicy
            ):
                tensorboard_policy_sample_uncertainty_ratio = (
                    uder_exploration_policy.current_sample_uncertainty_ratio
                )
            else:
                tensorboard_policy_sample_uncertainty_ratio = 0.0

            tensorboard_writer.add_scalar_per_epoch_monitoring(
                tag="Replay buffer optimization scheduling/policy sample uncertainty ratio",
                value=tensorboard_policy_sample_uncertainty_ratio,
            )

    return (
        optimized_replay_buffer,
        erll_replay_buffer_size,
        uder_exploration_policy,
    )
