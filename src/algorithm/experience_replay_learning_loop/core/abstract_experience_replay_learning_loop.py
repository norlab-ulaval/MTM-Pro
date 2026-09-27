# coding=utf-8
import os
import time
import warnings
from copy import copy
from datetime import datetime
from typing import Callable, List, Optional, Tuple, Union
import abc
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import omegaconf
from omegaconf import DictConfig

import mbrl
import mbrl.util
from algorithm.experience_replay_learning_loop.uder_plot_utils import (
    plot_extra_info_str,
)
from mbrl.util import ReplayBuffer, TransitionIterator, common as common_utils
from mbrl import models as models
from mbrl.types import TransitionBatch

from algorithm.experience_replay_learning_loop.core.data_classes import (
    PostTrainScore,
    PredictionMetric,
)
from algorithm.experience_replay_learning_loop.core._stage_timers import (
    ERLLStageTimer,
)
from algorithm.experience_replay_learning_loop.core.erll_epoch_budget import (
    experiment_planned_pass_epoch_budgets,
)
from algorithm.experience_replay_learning_loop.core.lr_schedule_audit import (
    DEFAULT_MAX_FLAT_RATIO,
    DEFAULT_RELATIVE_LR_THRESHOLD,
    audit_lr_schedule,
    make_inner_shape,
    make_outer_envelope,
)
from algorithm.experience_replay_learning_loop.core.model_testing_utils import (
    model_testtime_rollout_and_compute_prediction_metric,
)

from algorithm.experience_replay_learning_loop.core.replay_buffer_utils import (
    split_replay_buffer,
)
from algorithm.experience_replay_learning_loop.core.tensorboard_utils import (
    _collect_gradient_information,
    plot_grad_flow,
    tensorboard_prediction_error,
)
from algorithm.experience_replay_learning_loop.core.training_callback import (
    fetch_cfg_pipeline_tensorboard_key_value,
    tensorboard_arbitrary_dimension_prediction_trajectory_writer,
)
from tools.mbrl_lib_tools.models.utils import is_model_ensemble
from tools.mbrl_lib_tools import persistent_checkpoint_utils
from tools.mbrl_lib_tools.setup_utils import setup_saved_model_dir
from tools.r2s_motion_model_container_tools.container import R2SMotionModelContainer
from tools.console_tools.format import ConsoleFormat
from tools.f110_gym_env_tools.plot_tensorboard import OnlineTensorboardWritter

from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist

from tools.console_tools.message import consol_msg, consol_msg_universal_one_liner
from tools.console_tools.progressbar_tools import init_progressbar
from tools.hydra_apps_tools.hydra_utils import get_hydra_experiment_cwd
from tools.mbrl_lib_tools.models.one_dim_tr_model_v2 import OneDTransitionRewardModelV2
from tools.multistep_tools.models import (
    AutoRegressiveSequenceIterator,
)
from tools.multistep_tools.multistep_model_util import (
    reshape_sequence_dim_into_batch_dim,
)
from tools.multistep_tools.window_dataset.erll_data_source import (
    ERLLDataSource,
    ReplayBufferDataSource,
)
from trajectory_container_tools.dataclasses.erll_trajectory_dataclass import (
    TestMotionTrajectoryDataclass,
    TestTrajectoryDataclass,
)


class AbstractExperienceReplayLearningLoop(abc.ABC):
    _console_acronymn: str = "ERLL"
    _iter_batch_size: int
    _initial_dataset_size_from_source: int

    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        cfg_training: omegaconf.DictConfig,
        model_trainer: models.ModelTrainer,
        source_replay_buffer: Union[ReplayBuffer, ERLLDataSource],
        test_trajectories: Union[
            list[TestTrajectoryDataclass], list[TestMotionTrajectoryDataclass]
        ],
        tensorboard_writer: Optional[OnlineTensorboardWritter],
        erll_epoch_pre_training_callback: Optional[Callable],
        model_trainer_epoch_callback: Optional[Callable],
        model_trainer_batch_callback: Optional[Callable],
        erll_epoch_post_training_callback: Optional[Callable],
        torch_rng: torch.Generator,
        motion_model_container: R2SMotionModelContainer = None,
        replay_buffer_learning_loop_cfg_key: str = "ERLL",
    ):
        """
        Uncertainty Driven Experience Replay (UDER) performs uncertainty-driven exploration of a
        replay buffer using an uncertainty measures based exploration policy and retraining the
         model iteratively.

        ``source_replay_buffer`` is either a materialized mbrl ``ReplayBuffer`` (legacy path,
        byte-identical) or an ``ERLLDataSource`` -- e.g. the lazy window
        ``WindowDataLoaderDataSource`` of ``pipeline.data_manager: dataloader`` (RLRP-824).

        Optional cfg:

          >>> UDER.uder_num_epoch_init_override
          >>>   # int, false => skip override, null => 'patience'. Default=False
          >>> UDER.max_sequence_batches_per_loop_train
          >>>   # Options: int, null, 'match_optimizer_num_epoch'. Default=null
          >>> UDER.uder_exploration_policy.uder_epoch_buffer_normalization # Default=false
          >>> pipeline.tensorboard.generate_model_graph # Default=false
          >>> pipeline.tensorboard.generate_gradient_figure # Default=false
          >>> pipeline.tensorboard.trajectory_record_step_size # Default=100
          >>> pipeline.tensorboard.obs_dim_label # Default=("X", "Y", "Z")
          >>> environment.obs_are_dt_derivatives # Default=true

        :param cfg: The configuration dictionary for the replay buffer exploration.
        :param cfg_training: The configuration for training parameters.
        :param model_trainer: The model trainer used to train the neural network models.
        :param source_replay_buffer: The source replay buffer containing pre-collected
        trajectories.
        :param test_trajectories: A list of test trajectories used to assess model improvement.
        :param tensorboard_writer: Optional object to handle Tensorboard logging.
        :param erll_epoch_pre_training_callback: Optional callback to be invoked before model
         training in each epoch.
        :param model_trainer_epoch_callback: Optional callback to be used by the model trainer
         during each epoch.
        :param model_trainer_batch_callback: Optional callback to be used by the model trainer
         during each call to update/eval_score.
        :param erll_epoch_post_training_callback: Optional callback to be invoked after model
         training in each epoch.
        :param torch_rng: Random number generator instance for PyTorch operations.
        :param motion_model_container: Optional container for motion models, required for certain
         model types.
        dataset.
        :param replay_buffer_learning_loop_cfg_key: The algorithm configuration key.
        :return: Returns training and validation losses lists per epoch, and the total duration of
         the exploration in seconds.
        """
        # .... setup configuration ................................................................
        self._cfg = cfg
        self._cfg_training = cfg_training
        self._cfg_key_algo = replay_buffer_learning_loop_cfg_key
        self._cfg_algo = cfg.get(self._cfg_key_algo)

        self._check_required_config_key(cfg, cfg_training)

        # .... setup variables ....................................................................
        self._debug_mode = cfg.get("debug_mode", False)
        # RLRP-824 (plan KD4): the source is either a materialized mbrl ``ReplayBuffer`` (legacy,
        # wrapped in a thin ``ReplayBufferDataSource`` so every historical statement below runs
        # unchanged on ``self._source_replay_buffer``) or an ``ERLLDataSource`` such as the lazy
        # window ``WindowDataLoaderDataSource`` (``pipeline.data_manager: dataloader``).
        if isinstance(source_replay_buffer, ERLLDataSource):
            self._data_source: ERLLDataSource = source_replay_buffer
        else:
            self._data_source = ReplayBufferDataSource(source_replay_buffer)
        self._source_replay_buffer = (
            self._data_source.replay_buffer
            if self._data_source.is_replay_buffer
            else self._data_source
        )
        self._test_trajectories = test_trajectories
        self._pre_train_step_callback = erll_epoch_pre_training_callback
        self._model_trainer_epoch_callback = model_trainer_epoch_callback
        self._model_trainer_batch_callback = model_trainer_batch_callback
        self._post_train_step_callback = erll_epoch_post_training_callback
        self.model_trainer = model_trainer
        if not self._data_source.is_replay_buffer:
            self._check_data_source_supported(self._data_source)
        self.motion_model_container = motion_model_container
        self.tensorboard_writer = tensorboard_writer
        self.torch_rng = torch_rng
        self._current_model_trainer_num_epoch_max: Optional[int] = None

        # .... Fetch configuration settings .......................................................
        self._buffer_max_trajectory_len = self._cfg_algo.buffer_max_trajectory_len
        self._buffer_iterator_sequence_len = self._cfg_algo.buffer_iterator_sequence_len
        self._val_buffer_iterator_sequence_len = (
            self._cfg_algo.val_buffer_iterator_sequence_len
        )
        self._erll_max_epochs = self._cfg_algo.uder_num_epochs

        # .... Variable setup .....................................................................
        model_trainer_total_num_epochs = self._cfg.UDER.num_epochs_train_model
        if model_trainer_total_num_epochs == 0:
            # RLRP-727 sentinel: `num_epochs_train_model: 0` is legal ONLY for the
            # single-global-loop child. Its single fused (init == final) pass draws its epoch
            # budget from `final_max_num_epochs_train_model`, so the per-erll-epoch divisor is
            # never needed. Treat it like `None` (no divisor, skip the divisibility asserts).
            if not self._allows_zero_num_epochs_train_model:
                raise ValueError(
                    f"{self._console_acronymn} `num_epochs_train_model: 0` is only legal under "
                    "`loop_kind: single_global_loop` (SingleGlobalLoopERLL). Set a positive "
                    "value for the progressive PBER/UDER loops."
                )
            self._model_trainer_num_epoch_per_erll_epoch = None
        elif model_trainer_total_num_epochs is not None:
            self._model_trainer_num_epoch_per_erll_epoch = (
                model_trainer_total_num_epochs // self._erll_max_epochs
            )
        else:
            self._model_trainer_num_epoch_per_erll_epoch = None

        # .... Pre-condition ......................................................................
        assert self._erll_max_epochs >= 1
        for each_tt in test_trajectories:
            assert isinstance(
                each_tt, (TestTrajectoryDataclass, TestMotionTrajectoryDataclass)
            )

        if self._model_trainer_num_epoch_per_erll_epoch is not None:
            assert self._erll_max_epochs <= model_trainer_total_num_epochs, (
                f"{self._console_acronymn} number of epochs need to be lower than the model "
                "trainer max total number of epoch.\n"
                f"Curently set to`{self._cfg.UDER.uder_num_epochs=} !< "
                f"{self._cfg.UDER.num_epochs_train_model=}`"
            )
            assert model_trainer_total_num_epochs % self._erll_max_epochs == 0, (
                f"{self._console_acronymn} number of epochs need to be a multiple of the "
                "model trainer max total number of epoch.\nCurently set to "
                f"`{self._cfg.UDER.num_epochs_train_model=} % "
                f"{self._cfg.UDER.uder_num_epochs=} != 0`"
            )

        if self._buffer_max_trajectory_len is not None:
            assert (
                self._buffer_max_trajectory_len >= self._buffer_iterator_sequence_len
            ), f"{self._buffer_max_trajectory_len} !>= {self._buffer_iterator_sequence_len}"
            assert (
                self._buffer_max_trajectory_len
                >= self._val_buffer_iterator_sequence_len
            ), f"{self._buffer_max_trajectory_len} !>= {self._val_buffer_iterator_sequence_len}"

        # RLRP-824 (FR7): optional resume-from-checkpoint. Resolved BEFORE the normalizer fit so
        # the recorded statistics / split of the interrupted run win over a fresh fit / split.
        self._resume_ckpt_dir: Optional[str] = None
        self._resume_run_root: Optional[str] = None
        self._resume_epoch_offset: int = 0
        _resume_from = omegaconf.OmegaConf.select(
            self._cfg, "training_common.resume_from_checkpoint", default=None
        )
        if _resume_from:
            (
                self._resume_ckpt_dir,
                self._resume_run_root,
                _resume_meta,
            ) = persistent_checkpoint_utils.resolve_resume_checkpoint(str(_resume_from))
            self._resume_epoch_offset = int(_resume_meta["epoch"])
            consol_msg(
                who_am_i=self._console_acronymn,
                msg=(
                    f"resume_from_checkpoint: `{self._resume_ckpt_dir}` (global epoch "
                    f"{self._resume_epoch_offset}, run root `{self._resume_run_root}`)"
                ),
                space_before=False,
                space_after=False,
            )

        if hasattr(self.model_trainer.model, "update_normalizer"):
            consol_msg_universal_one_liner(
                "Updates normalizer statistics using source dataset"
            )
            self.model_trainer.model.update_normalizer(
                self._source_replay_buffer.get_all()
            )
            # RLRP-824 (FR6/FR7): the fitted statistics are a restart artifact. On a fresh run,
            # snapshot them at the run root (`normalizers/`); on resume, REPLACE the fresh fit by
            # the interrupted run's snapshot (falls back to the statistics stored inside the
            # checkpoint when the run root has no snapshot, e.g. a pre-RLRP-824 run).
            _exp_cwd = get_hydra_experiment_cwd(self._cfg)
            if self._resume_run_root is not None:
                if persistent_checkpoint_utils.load_normalizers_snapshot(
                    self.model_trainer.model, self._resume_run_root
                ):
                    consol_msg_universal_one_liner(
                        "resume_from_checkpoint: normalizer statistics restored from the run-root "
                        "`normalizers/` snapshot"
                    )
                elif not persistent_checkpoint_utils.epoch_checkpoint_has_normalizers(
                    self._resume_ckpt_dir
                ):
                    raise FileNotFoundError(
                        f"{self._console_acronymn} resume_from_checkpoint: `{self._resume_ckpt_dir}` "
                        "was written with `save_normalizer_every_epoch: false` and the run root "
                        f"`{self._resume_run_root}` has no `normalizers/` snapshot to restore."
                    )
            if os.path.abspath(_exp_cwd) != os.path.abspath(str(self._resume_run_root or "")):
                persistent_checkpoint_utils.save_normalizers_snapshot(
                    self.model_trainer.model, _exp_cwd
                )
            if self._resume_ckpt_dir is not None:
                # RLRP-824 (FR7): restore the weights (+ optimizer state when persisted) of the
                # interrupted run. `include_normalizers=None` follows the checkpoint manifest, so a
                # weights-only epoch dir keeps the statistics restored above.
                persistent_checkpoint_utils.load_epoch_checkpoint_model(
                    self.model_trainer.model, self._resume_ckpt_dir
                )
                self.model_trainer.model.train()
                _optimizer_restored = persistent_checkpoint_utils.load_epoch_checkpoint_optimizer(
                    self.model_trainer.optimizer, self._resume_ckpt_dir
                ) or persistent_checkpoint_utils.load_running_latest_optimizer(
                    self.model_trainer.optimizer, self._resume_run_root
                )
                consol_msg_universal_one_liner(
                    "resume_from_checkpoint: model weights restored; optimizer state "
                    + ("restored" if _optimizer_restored else "NOT found (fresh optimizer moments)")
                )
            # RLRP-761 S2.4: secondary hook point -- same renderer, once, right
            # after the statistics are fitted. S3.3 resolves a named per-feature
            # loss-weight criterion first, so the table reports the weights the
            # run will use.
            from tools.feature_handling_tools.feature_loss_weights import (
                resolve_and_apply_feature_loss_weights,
            )
            from tools.feature_handling_tools.normalization_diagnostic import (
                log_feature_normalization_report,
            )

            resolve_and_apply_feature_loss_weights(
                self._cfg,
                self.model_trainer.model,
                self._source_replay_buffer,
            )
            log_feature_normalization_report(
                self._cfg,
                self.model_trainer.model,
                self._source_replay_buffer,
            )

        # .... Set components .....................................................................
        self._progressbar_main = tqdm(
            desc=f"{self._console_acronymn} epoch",
            total=self._erll_max_epochs,
            leave=False,
            position=0,
        )
        self._progressbar_inner: Optional[tqdm] = None

        # .... Data preparation ...................................................................

        if self._cfg.source_replay_buffer.source_size == "all":
            # Use all samples from offline sources
            self._initial_dataset_size_from_source = (
                self._source_replay_buffer.num_stored
            )
        else:
            self._initial_dataset_size_from_source = (
                self._cfg.source_replay_buffer.source_size
            )

        if self._data_source.is_replay_buffer:
            # NOTE (RLRP-775 action ``A19``): the train/val buffers built here — NOT
            # the source buffer — are the ones the epoch loop samples from. Resolve
            # their placement from the cfg (``mbrl_lib.keep_replay_buffer_on_device``
            # + the VRAM guard) so action ``A17`` is not silently cancelled by the
            # split. Returns ``None`` (legacy host storage) whenever the flag is off,
            # CUDA is unavailable or the guard fires.
            from tools.mbrl_lib_tools.setup_utils import resolve_replay_buffer_device

            _train_val_buffer_device = resolve_replay_buffer_device(
                self._cfg,
                capacity=int(self._initial_dataset_size_from_source),
                obs_shape=tuple(self._source_replay_buffer.obs_shape),
                action_shape=tuple(self._source_replay_buffer.action_shape),
                obs_type=self._source_replay_buffer.obs_type,
                action_type=self._source_replay_buffer.action_type,
                reward_type=self._source_replay_buffer.reward_type,
                max_trajectory_length=self._buffer_max_trajectory_len,
            )

            self.iter0_train_replay_buffer, self.val_replay_buffer = split_replay_buffer(
                self._source_replay_buffer,
                datasets_size=self._initial_dataset_size_from_source,
                val_ratio=self._cfg.source_replay_buffer.val_ratio,
                buffer_max_trajectory_len=self._buffer_max_trajectory_len,
                device=_train_val_buffer_device,
                # Seeded like the FR5 sample-level split of the dataloader path (same run seed
                # -> same train/val partition); ``None`` keeps the legacy fresh draw.
                seed=self._cfg.get("seed", None),
            )
        else:
            # RLRP-824: sample-level seeded split of the lazy window dataset, recorded as a
            # ``DataSplitRecord`` (FR5); the train / val "buffers" are light views exposing the
            # ``num_stored`` / ``stores_trajectories`` attributes the loop reads.
            _restored_split = False
            if self._resume_run_root is not None:
                _record_path = persistent_checkpoint_utils.split_record_path(
                    self._resume_run_root
                )
                if os.path.isfile(_record_path):
                    from tools.multistep_tools.window_dataset.split_record import (
                        DataSplitRecord,
                    )

                    self._data_source.restore_split(DataSplitRecord.from_json(_record_path))
                    _restored_split = True
                    consol_msg_universal_one_liner(
                        f"resume_from_checkpoint: train/val split restored from `{_record_path}`"
                    )
                else:
                    warnings.warn(
                        f"{self._console_acronymn} resume_from_checkpoint: no "
                        f"`{persistent_checkpoint_utils.SPLIT_RECORD_FNAME}` under "
                        f"`{self._resume_run_root}`; re-splitting under cfg.seed={self._cfg.seed} "
                        "(identical to the interrupted run only if it used the same seed / data).",
                        stacklevel=2,
                    )
            if not _restored_split:
                self._data_source.split(
                    datasets_size=(
                        None
                        if self._cfg.source_replay_buffer.source_size == "all"
                        else int(self._initial_dataset_size_from_source)
                    ),
                    val_ratio=self._cfg.source_replay_buffer.val_ratio,
                )
            # RLRP-824 (FR5): the sample-level split is a restart artifact of the run.
            _record = self._data_source.split_record()
            if _record is not None:
                _record.to_json(
                    persistent_checkpoint_utils.split_record_path(
                        get_hydra_experiment_cwd(self._cfg)
                    )
                )
            self.iter0_train_replay_buffer = self._data_source.train_view()
            self.val_replay_buffer = self._data_source.val_view()
            if self.val_replay_buffer is None:
                raise ValueError(
                    f"{self._console_acronymn} pipeline.data_manager=dataloader needs a non-empty "
                    f"validation set (source_replay_buffer.val_ratio="
                    f"{self._cfg.source_replay_buffer.val_ratio})"
                )

        consol_msg_universal_one_liner(
            f"Train replay buffer size: {self.iter0_train_replay_buffer.num_stored}"
        )
        consol_msg_universal_one_liner(
            f"Validation replay buffer size: {self.val_replay_buffer.num_stored}"
        )

        self.train_replay_buffer: mbrl.util.ReplayBuffer = (
            self.iter0_train_replay_buffer
        )

        assert (
            self.iter0_train_replay_buffer.num_stored > 0
        ), f"Replay buffer empty {self.iter0_train_replay_buffer.num_stored=}"
        assert (
            self.val_replay_buffer.num_stored > 0
        ), f"Replay buffer empty {self.val_replay_buffer.num_stored=}"

        # .... ERLL epoch learning rate ...........................................................
        cfg_uder_lr_warmup = omegaconf.OmegaConf.select(
            self._cfg_algo.uder_lr_schedule, "warmup", default=None
        )

        # `LambdaLR` requires one `lr_lambda` per optimizer param group. The two-timescale-LR
        # deploy-path stabilization feature (RLRP-722) splits the optimizer into >1 param groups
        # (slow body / fast heads), so the exponential-decay lambda must be broadcast to every
        # param group; otherwise torch raises `ValueError: Expected N lr_lambdas, but got 1`.
        num_param_groups = len(self.model_trainer.optimizer.param_groups)

        # RLRP-727 (B1) — Law-U opt-in. See `_apply_law_u_pass_base_lr` and Appendix A of the plan.
        #
        #   Law-U composition:  lr(p, e) = base_i · g(p) · h(e)
        #
        #     * `base_i`  : the per-param-group base LR captured HERE at construction (preserves the
        #                   RLRP-722 two-timescale body/head ratio);
        #     * `g(p)`    : the OUTER per-pass envelope (exponential decay of the per-pass *restart
        #                   height*, with an optional constant warmup over the first `w` passes);
        #     * `h(e)`    : the INNER per-epoch shape (`setup_trainer_lr_scheduler_callback`), which
        #                   RESTARTS each pass (h(0)=1) — an SGDR-like warm restart.
        #
        # This REPLACES the historical two-schedulers-on-one-optimizer pattern (an outer `LambdaLR`
        # + the inner `ChainedScheduler` both OVERWRITING `param_groups['lr']`, an order-dependent
        # clobber where the outer decay was effectively dead). When enabled, the loop sets
        # `param_groups[i]['lr'] = base_i · g(p)` at each pass start and asks the inner schedule to
        # restart from there; the outer torch `LambdaLR` below is NOT built (nothing clobbers).
        # Default (false) = byte-identical historical behaviour.
        self._law_u_enable = bool(
            omegaconf.OmegaConf.select(
                self._cfg_algo, "uder_lr_schedule.law_u_enable", default=False
            )
        )
        self._law_u_base_lrs = [
            group["lr"] for group in self.model_trainer.optimizer.param_groups
        ]
        _law_u_warmup_steps = 0
        _law_u_warmup_factor = 1.0
        if cfg_uder_lr_warmup is not None:
            _law_u_warmup_steps = cfg_uder_lr_warmup.get("uder_steps", 1)
            _law_u_warmup_factor = cfg_uder_lr_warmup.get("factor", 1.0)
        self._law_u_warmup_steps = _law_u_warmup_steps
        self._law_u_warmup_factor = _law_u_warmup_factor
        self._law_u_exp_decreassing = self._cfg_algo.uder_lr_schedule.exp_decreassing

        if self._law_u_enable:
            # Law-U owns the LR explicitly; do NOT build the clobbering outer torch scheduler.
            self._erll_lr_scheduler = None
            return

        schedulers = []
        if cfg_uder_lr_warmup is not None:
            cfg_warmup_uder_steps = cfg_uder_lr_warmup.get("uder_steps", 1)
            schedulers.append(
                torch.optim.lr_scheduler.ConstantLR(
                    self.model_trainer.optimizer,
                    factor=cfg_uder_lr_warmup.get("factor", 1.0),
                    total_iters=cfg_warmup_uder_steps,
                ),
            )
            schedulers.append(
                torch.optim.lr_scheduler.LambdaLR(
                    self.model_trainer.optimizer,
                    lr_lambda=[
                        lambda uder_epoch: self._cfg_algo.uder_lr_schedule.exp_decreassing
                        ** (uder_epoch - cfg_warmup_uder_steps)
                    ]
                    * num_param_groups,
                )
            )
            self._erll_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer=self.model_trainer.optimizer,
                schedulers=schedulers,
                milestones=[cfg_warmup_uder_steps],
            )
        else:
            self._erll_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.model_trainer.optimizer,
                lr_lambda=[
                    lambda uder_epoch: self._cfg_algo.uder_lr_schedule.exp_decreassing
                    ** uder_epoch
                ]
                * num_param_groups,
            )

    @property
    def cfg_training_required_key(self):
        return [
            "patience",
            "model_wd",
        ]

    @property
    def cfg_required_key(self):
        return [
            self._cfg_key_algo,
            "source_replay_buffer.source_size",
            "source_replay_buffer.val_ratio",
        ]

    @property
    def cfg_algo_required_key(self):
        return [
            "batch_size",
            "batch_size.init_value",
            "buffer_max_trajectory_len",
            "buffer_iterator_sequence_len",
            "val_buffer_iterator_sequence_len",
            "final_max_num_epochs_train_model",
            "uder_num_epochs",
            "num_epochs_train_model",
            "uder_lr_schedule.exp_decreassing",
        ]

    @property
    def _allows_zero_num_epochs_train_model(self) -> bool:
        """
        RLRP-727 opt-in seam. When ``True``, the base ``__init__`` accepts the sentinel
        ``num_epochs_train_model: 0`` (single-global-loop mode) and skips the per-erll-epoch
        divisor / divisibility asserts. Default ``False`` for the progressive PBER/UDER loops.
        """
        return False

    def _law_u_outer_envelope(self, erll_pass: int) -> float:
        """
        Law-U outer per-pass envelope ``g(p)`` (RLRP-727 B1).

        ``g(p)`` sets the *restart height* of each ERLL pass: a constant warmup ``factor`` over the
        first ``w`` passes, then geometric decay ``exp_decreassing**(p - w)``. For a single pass
        (``p = 0``, single-global-loop) with no warmup this is ``1.0``, so the realized LR reduces to
        the inner schedule ``base · h(e)`` — the clean ablation control.

        :param erll_pass: the ERLL pass index ``p``.
        :return: the outer envelope multiplier ``g(p)``.
        """
        if erll_pass < self._law_u_warmup_steps:
            return float(self._law_u_warmup_factor)
        return float(
            self._law_u_exp_decreassing ** (erll_pass - self._law_u_warmup_steps)
        )

    def _apply_law_u_pass_base_lr(self, erll_pass: int) -> None:
        """
        Law-U pass-start LR application (RLRP-727 B1).

        Sets ``param_groups[i]['lr'] = base_i · g(p)`` for every optimizer param group (preserving
        the RLRP-722 two-timescale body/head ratio), then asks the inner per-epoch schedule to
        RESTART from that value (``h(0)=1``) via the ``model_trainer._law_u_reset_inner_lr`` handle
        installed by ``setup_trainer_lr_scheduler_callback``. The realized LR within the pass is
        therefore ``base_i · g(p) · h(e)``. No-op when Law-U is disabled.

        :param erll_pass: the ERLL pass index ``p``.
        """
        if not self._law_u_enable:
            return
        g_p = self._law_u_outer_envelope(erll_pass)
        for group, base_lr in zip(
            self.model_trainer.optimizer.param_groups, self._law_u_base_lrs
        ):
            group["lr"] = base_lr * g_p

        # Stash the realized per-pass epoch budget so the inner cosine `T_max` (B2) is sized per
        # pass, then trigger the inner-schedule restart (re-snapshots `base_lrs` from the lr just
        # set above). The handle is absent when no inner trainer LR schedule is configured.
        self.model_trainer._law_u_pass_num_epochs = (
            self._current_model_trainer_num_epoch_max
        )
        _reset_inner_lr = getattr(self.model_trainer, "_law_u_reset_inner_lr", None)
        if callable(_reset_inner_lr):
            # RLRP-727: the inner `trainer_lr_schedule.warmup` is a one-time global optimizer
            # stabilization, so only include it on the ERLL initialization pass (``p == 0``); later
            # per-pass SGDR restarts skip the warmup plateau and anneal from the restart height.
            _reset_inner_lr(include_warmup=(erll_pass == 0))

    def _planned_pass_epoch_budgets(self) -> Optional[List[int]]:
        """Replicate ``execute()``'s per-pass epoch budget decision, statically.

        Mirrors the ``is_erll_initialization_epoch`` / ``is_final_erll_epoch`` / default branches of
        ``execute()``. Returns ``None`` when any pass budget is patience-driven (``null``), in which
        case the epoch grid is not known ahead of time and no static audit is possible.

        RLRP-839: delegates to the cfg-only :func:`erll_epoch_budget.experiment_planned_pass_epoch_budgets`
        (fed with this instance's ``_iter_erll_epochs()`` so subclass loop plans are honoured), the
        same function the deploy-stage resume audit uses without instantiating a loop.

        :return: the planned number of inner epochs per ERLL pass, or ``None``.
        """
        return experiment_planned_pass_epoch_budgets(
            self._cfg,
            erll_cfg_key=self._cfg_key_algo,
            erll_epochs=self._iter_erll_epochs(),
        )

    def _lr_schedule_preflight_audit(self) -> None:
        """Pre-flight (``t = 0``) audit of the composed LR law — advisory, never mutates training.

        Answers the "can we know up front that a large tail of the run will train at an
        effectively frozen learning rate?" question: under Law-U every term of
        ``lr(p, e) = base · g(p) · h(e)`` and the whole epoch grid are known before the first
        gradient step, so the wasted budget is computable without training anything.

        Configuration (all optional, legacy default = warn only)::

            diagnostics:
              lr_schedule_audit:
                enable: true                    # false -> fully skip
                relative_lr_threshold: 0.01     # "frozen" = below 1% of the peak realized LR
                max_flat_ratio: 0.25            # flag when >25% of the budget is frozen
                on_violation: warn              # warn | raise

        :raises ValueError: when the audit is flagged and ``on_violation: raise``.
        """
        cfg_audit = omegaconf.OmegaConf.select(
            self._cfg, "diagnostics.lr_schedule_audit", default=None
        )
        if not bool(
            omegaconf.OmegaConf.select(cfg_audit, "enable", default=True)
            if cfg_audit is not None
            else True
        ):
            return

        if not self._law_u_enable:
            # The legacy (non Law-U) path chains an outer `LambdaLR` and the inner schedule on the
            # same optimizer, an order-dependent clobber whose realized LR is not reproducible
            # statically. Skipping is the honest outcome (see `lr_schedule_audit` module docstring).
            return

        budgets = self._planned_pass_epoch_budgets()
        if budgets is None:
            consol_msg(
                who_am_i=self._console_acronymn,
                msg=(
                    "LR schedule pre-flight audit SKIPPED: at least one ERLL pass has a "
                    "patience-driven (null) epoch budget, so the epoch grid is unknown ahead of "
                    "time."
                ),
                space_before=False,
                space_after=False,
            )
            return

        inner_shape, unsupported_reason = make_inner_shape(
            omegaconf.OmegaConf.select(
                self._cfg_training, "trainer_lr_schedule", default=None
            )
        )
        audit = audit_lr_schedule(
            pass_epoch_budgets=budgets,
            base_lr=max(self._law_u_base_lrs) if self._law_u_base_lrs else 0.0,
            outer_envelope=make_outer_envelope(
                exp_decreassing=self._law_u_exp_decreassing,
                warmup_passes=self._law_u_warmup_steps,
                warmup_factor=self._law_u_warmup_factor,
            ),
            inner_shape=inner_shape,
            relative_lr_threshold=float(
                omegaconf.OmegaConf.select(
                    self._cfg,
                    "diagnostics.lr_schedule_audit.relative_lr_threshold",
                    default=DEFAULT_RELATIVE_LR_THRESHOLD,
                )
            ),
            unsupported_reason=unsupported_reason,
        )

        max_flat_ratio = float(
            omegaconf.OmegaConf.select(
                self._cfg,
                "diagnostics.lr_schedule_audit.max_flat_ratio",
                default=DEFAULT_MAX_FLAT_RATIO,
            )
        )
        consol_msg(
            who_am_i=self._console_acronymn,
            msg=audit.format_report(max_flat_ratio),
            space_before=True,
            space_after=True,
        )

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.writer.add_text(
                tag="Replay buffer optimization scheduling/LR schedule pre-flight audit",
                text_string=f"```\n{audit.format_report(max_flat_ratio)}\n```",
                global_step=0,
            )

        if audit.is_flagged(max_flat_ratio):
            on_violation = str(
                omegaconf.OmegaConf.select(
                    self._cfg,
                    "diagnostics.lr_schedule_audit.on_violation",
                    default="warn",
                )
            ).lower()
            if on_violation == "raise":
                raise ValueError(
                    f"{self._console_acronymn} LR schedule pre-flight audit failed: "
                    f"{audit.flat_ratio:.1%} of the epoch budget runs below "
                    f"{audit.absolute_lr_threshold:.3e} (alarm threshold "
                    f"{max_flat_ratio:.0%}). Set "
                    "`diagnostics.lr_schedule_audit.on_violation: warn` to downgrade this to a "
                    "warning."
                )
            warnings.warn(
                f"{self._console_acronymn} LR schedule pre-flight audit flagged: "
                f"{audit.flat_ratio:.1%} of the epoch budget runs at an effectively frozen "
                "learning rate.",
                stacklevel=2,
            )
        return None

    def _iter_erll_epochs(self) -> List[Tuple[int, bool, bool]]:
        """
        Base seam (RLRP-727) defining the ERLL outer-loop plan consumed by ``execute()``.

        Returns the ordered list of ``(erll_epoch, is_initialization, is_final)`` tuples the
        outer loop iterates over. The default reproduces the historical behaviour byte-for-byte:
        one mandatory initialization pass (``epoch 0``) followed by ``_erll_max_epochs`` progressive
        passes, the last of which is flagged ``is_final``.

        Subclasses override this to change the loop plan without copy-overriding the whole
        ``execute()`` skeleton. ``SingleGlobalLoopERLL`` returns a single fused
        ``[(0, True, True)]`` pass so the initialization and final stages collapse into one global
        training loop (ablation / ERLL disabled).

        :return: the list of ``(erll_epoch, is_erll_initialization_epoch, is_final_erll_epoch)``.
        """
        return [
            (
                each_erll_epoch,
                each_erll_epoch == 0,
                each_erll_epoch == self._erll_max_epochs,
            )
            for each_erll_epoch in range(self._erll_max_epochs + 1)
        ]

    def execute(self) -> Tuple[List[float], List[float], float, float, float]:
        """
        Executes the replay buffer learning loop for training, validation, and
        model checkpointing, adjusting parameters and configurations dynamically.

        :returns: A tuple containing the list of training losses per epoch, the list
                  of validation losses per epoch, the total training time, and the
                  best mean prediction absolute error (MAE) score.
        """
        param_accumulator = []
        start_time = time.time()
        post_train_score = PostTrainScore()

        # Pre-flight (t = 0) audit of the composed LR law. Advisory only: it reports how much of
        # the planned epoch budget will run at an effectively frozen learning rate BEFORE any
        # gradient step, instead of leaving it to be discovered on a TensorBoard plot hours later.
        self._lr_schedule_preflight_audit()

        model_data_path = os.path.join(
            get_hydra_experiment_cwd(self._cfg),
            f"model_{self.model_trainer.model.model.__class__.__name__}",
        )
        os.makedirs(model_data_path, exist_ok=True)

        # //// ERLL replay buffer iteration ///////////////////////////////////////////////////////
        # Cadence for the test-time rollout (Training Speed & Efficiency
        # plan, stage 1 Batch 1 — B1). Legacy default `1` preserves the
        # historical behaviour: rollout runs on every ERLL epoch except
        # the initialization one (and always on the final one). Raising
        # this value to N > 1 makes the rollout fire only every Nth
        # ERLL epoch, with the final ERLL epoch always forced on.
        testtime_rollout_cadence = int(
            omegaconf.OmegaConf.select(
                self._cfg,
                "UDER.testtime_rollout_every_n_erll_epochs",
                default=1,
            )
        )
        assert testtime_rollout_cadence >= 1, (
            "`cfg.UDER.testtime_rollout_every_n_erll_epochs` must be >= 1 "
            f"(got {testtime_rollout_cadence})"
        )

        # //// Intra-final-pass checkpoint cadence /////////////////////////////////////////////////
        # Follow-up to RLRP-727 (`single_global_loop`): the outer per-ERLL-pass rollout/checkpoint
        # gate (below) fires only ONCE for the single fused pass, collapsing the periodic best-model
        # persistence that multi-pass PBER/UDER get for free. This knob adds an *intra-pass*
        # checkpoint cadence that fires every `N` inner training epochs, but ONLY during the final
        # ERLL pass (`is_final_erll_epoch == True`) — precisely the long `train()` call where model
        # quality can peak-then-degrade with no intermediate save. Legacy default `0` = off (every
        # existing run stays byte-identical).
        checkpoint_every_n_final_inner_epochs = int(
            omegaconf.OmegaConf.select(
                self._cfg,
                "UDER.checkpoint_every_n_final_inner_epochs",
                default=0,
            )
        )
        assert checkpoint_every_n_final_inner_epochs >= 0, (
            "`cfg.UDER.checkpoint_every_n_final_inner_epochs` must be >= 0 "
            f"(got {checkpoint_every_n_final_inner_epochs})"
        )

        # //// Epoch-interval persistent checkpointing (RLRP-773) //////////////////////////////////
        # Optional feature: persist `(model, optimizer)` snapshots tagged with the global completed
        # inner-epoch count, at a user-configured interval, so the model's evolution from epoch 0 to
        # epoch E can be recovered post-training (e.g. for per-epoch deploy drift-rate plots).
        # `training_common.force_checkpoint_interval: null` (default) => feature OFF => legacy
        # behaviour, zero new code on the hot path. Independent of the best-metric checkpoint logic
        # (R5): writes to a SEPARATE `epoch_checkpoints/` tree.
        # RLRP-773 config aggregation: the cadence knob moved under the `training_common.checkpoint`
        # group (`training_common.checkpoint.force_interval`); the legacy flat
        # `training_common.force_checkpoint_interval` is still honoured as a fallback.
        _force_checkpoint_interval_raw = self._read_checkpoint_option(
            "force_interval", "force_checkpoint_interval", None
        )
        force_checkpoint_interval = (
            int(_force_checkpoint_interval_raw)
            if _force_checkpoint_interval_raw
            else None
        )
        if force_checkpoint_interval is not None:
            assert force_checkpoint_interval >= 1, (
                "`cfg.training_common.checkpoint.force_interval` must be >= 1 or null "
                f"(got {force_checkpoint_interval})"
            )
            # R4 (advisory divisor warning) + R10 (hard guard against a patience-driven budget).
            self._epoch_checkpoint_preflight(force_checkpoint_interval)
        epoch_checkpoint_exp_cwd = get_hydra_experiment_cwd(self._cfg)
        # RLRP-773 (§11): tri-state optimizer persistence in each epoch checkpoint.
        # `training_common.checkpoint.include_optimizer`:
        #   null (default; legacy `false`) => model-only snapshot (no optimizer);
        #   all  (legacy `true`)           => an `optimizer.pth` in EVERY epoch dir;
        #   latest                          => a single run-wide `optimizer_latest.pth` overwritten
        #                                      on each save (O(1) optimizer disk cost).
        epoch_checkpoint_include_optimizer = (
            persistent_checkpoint_utils.normalize_include_optimizer(
                self._read_checkpoint_option(
                    "include_optimizer", "include_optimizer", None
                )
            )
        )
        # RLRP-773 (§11): optional pass-boundary best-val re-snapshot. Default OFF (operator Q7);
        # set `training_common.checkpoint.include_pass_best: true` to also persist the rewound
        # best-val weights as `epoch_<E>_pass_best/` at interval-aligned pass boundaries.
        epoch_checkpoint_include_pass_best = bool(
            self._read_checkpoint_option(
                "include_pass_best", "include_pass_best", False
            )
        )
        # RLRP-824 (FR6): `training_common.checkpoint.save_normalizer_every_epoch` (default true =
        # legacy layout). `false` => weights-only epoch dirs; the statistics live in the run-root
        # `normalizers/` snapshot written in `__init__`.
        self._epoch_checkpoint_include_normalizers = bool(
            self._read_checkpoint_option(
                "save_normalizer_every_epoch", "save_normalizer_every_epoch", True
            )
        )

        # RLRP-824 (FR7): resume bookkeeping. `resume_offset` global inner epochs were completed by
        # the interrupted run; ERLL passes whose planned budget ends at or before the offset are
        # skipped, the pass containing it runs its remaining epochs, later passes run in full. The
        # global epoch counter (checkpoint names, callbacks) continues from the offset.
        resume_offset = int(self._resume_epoch_offset)
        planned_budgets = self._planned_pass_epoch_budgets() if resume_offset > 0 else None
        if resume_offset > 0 and planned_budgets is None:
            raise ValueError(
                f"{self._console_acronymn} `training_common.resume_from_checkpoint` requires every "
                "ERLL pass to have an explicit integer epoch budget (a patience-driven / null budget "
                "has no deterministic epoch grid to resume into)."
            )
        planned_before_pass = 0

        # //// Per-stage profile breakdown (F-E1-instr, stage-1 follow-up) ////
        # Opt-in per-stage timer for the ERLL outer loop. Legacy default
        # `cfg.diagnostics.profile_breakdown=false` short-circuits every
        # timer call to a zero-cost no-op (see `_stage_timers.py`). When
        # enabled, uses `torch.cuda.Event` on CUDA devices and
        # `time.perf_counter` on CPU/MPS. Five fused stages are wired
        # at this outer-loop level: `data`, `train_step`, `rollout`,
        # `save`, `tb`. A finer-grained split of `train_step` into
        # `forward` and `backward` requires a `legacy_batch_callback` hook in
        # the `utilities/mbrl-lib` fork and is tracked as a follow-up
        # refinement in the stage-1 follow-up plan.
        _profile_breakdown = bool(
            omegaconf.OmegaConf.select(
                self._cfg,
                "diagnostics.profile_breakdown",
                default=False,
            )
        )
        try:
            _timer_device = next(self.model_trainer.model.parameters()).device
        except (StopIteration, AttributeError):
            _timer_device = None
        self._stage_timer = ERLLStageTimer(
            enabled=_profile_breakdown, device=_timer_device
        )

        for (
            each_erll_epoch,
            is_erll_initialization_epoch,
            is_final_erll_epoch,
        ) in self._iter_erll_epochs():
            self._progressbar_main.moveto(1)

            # Expose the current ERLL epoch to the tensorboard writer so
            # cadence-gated heavy publishes (C5) line up with the ERLL
            # outer loop rather than the inner gradient-update counter.
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.set_current_erll_epoch(each_erll_epoch)

            """
            Optional override on first train run. If not false, either set an arbitrary nb
            of epoch or None i.e. use model trainer 'patience' threshold setting.
            """
            if is_erll_initialization_epoch and self._cfg.UDER.get(
                "uder_num_epoch_init_override", False
            ):
                self._current_model_trainer_num_epoch_max = (
                    self._cfg.UDER.uder_num_epoch_init_override
                )
                consol_msg(
                    who_am_i=self._console_acronymn,
                    msg=f"run initialisation epoch with num_epoch_init_override={self._cfg.UDER.uder_num_epoch_init_override}",
                    space_before=False,
                    space_after=False,
                )
            elif is_final_erll_epoch:
                # Final train run. Either set an arbitrary nb of epoch or None i.e. use model
                # trainer 'patience' setting.
                self._current_model_trainer_num_epoch_max = self._cfg_algo.get(
                    "final_max_num_epochs_train_model", None
                )
            else:
                self._current_model_trainer_num_epoch_max = (
                    self._model_trainer_num_epoch_per_erll_epoch
                )

            # RLRP-727 (B1) — Law-U: set this pass's per-group base LR to `base_i · g(p)` and
            # restart the inner per-epoch schedule from there (no-op when Law-U disabled). Done
            # after the per-pass epoch budget is known so the inner cosine `T_max` (B2) is sized
            # per pass, and before training so the pass anneals from the correct restart height.
            self._apply_law_u_pass_base_lr(each_erll_epoch)

            # RLRP-824 (FR7): skip / shorten passes already completed by the interrupted run.
            if resume_offset > 0:
                pass_budget = planned_budgets[each_erll_epoch]
                done_in_pass = min(max(resume_offset - planned_before_pass, 0), pass_budget)
                planned_before_pass += pass_budget
                if done_in_pass >= pass_budget:
                    consol_msg(
                        who_am_i=self._console_acronymn,
                        msg=(
                            f"resume_from_checkpoint: ERLL pass {each_erll_epoch} "
                            f"({pass_budget} epochs) already completed -- skipped"
                        ),
                        space_before=False,
                        space_after=False,
                    )
                    self._progressbar_main.update(1)
                    continue
                if done_in_pass > 0:
                    self._current_model_trainer_num_epoch_max = pass_budget - done_in_pass
                    consol_msg(
                        who_am_i=self._console_acronymn,
                        msg=(
                            f"resume_from_checkpoint: ERLL pass {each_erll_epoch} resumes at "
                            f"inner epoch {done_in_pass}/{pass_budget} "
                            f"({self._current_model_trainer_num_epoch_max} epochs left; the inner "
                            "LR schedule restarts from the pass base LR)"
                        ),
                        space_before=False,
                        space_after=False,
                    )

            # ==== ERLL epoch pre-training stage ==================================================
            erll_epoch_batch_size = self._batch_size_scheduling(each_erll_epoch)

            self._pre_train_step(
                each_erll_epoch,
                erll_epoch_batch_size,
                is_erll_initialization_epoch,
                is_final_erll_epoch,
            )

            with self._stage_timer.stage("data"):
                train_buffer_iterator, val_buffer_iterator = (
                    self._setup_replay_buffers_iterator(erll_epoch_batch_size)
                )

            if is_erll_initialization_epoch:
                print(plot_extra_info_str(self._cfg, self.model_trainer.model.model))

            self.model_trainer.print_model_summary_once()

            trainer_epoch_callback = self._set_model_trainer_progressbar(
                is_final_erll_epoch
            )

            # Intra-final-pass checkpointing (follow-up to RLRP-727): only on the final ERLL pass,
            # and only when the cadence knob is enabled, wrap the inner per-epoch callback so the
            # checkpoint routine also runs every `N` inner epochs (see the cadence read above).
            if is_final_erll_epoch and checkpoint_every_n_final_inner_epochs > 0:
                trainer_epoch_callback = (
                    self._wrap_trainer_callback_with_intra_pass_checkpoint(
                        trainer_epoch_callback,
                        post_train_score,
                        checkpoint_every_n_final_inner_epochs,
                    )
                )

            # Epoch-interval checkpointing (RLRP-773): on EVERY ERLL pass, wrap the inner callback
            # so a `(model, optimizer)` snapshot is written whenever the global completed inner-epoch
            # count is a multiple of the interval. The offset is captured at pass start and equals
            # the number of epochs completed in all previous passes (Q2).
            if force_checkpoint_interval is not None:
                trainer_epoch_callback = (
                    self._wrap_trainer_callback_with_epoch_checkpoint(
                        trainer_epoch_callback,
                        interval=force_checkpoint_interval,
                        global_epoch_offset=resume_offset + len(post_train_score.total_train_losses),
                        erll_pass=each_erll_epoch,
                        exp_cwd=epoch_checkpoint_exp_cwd,
                        include_optimizer_mode=epoch_checkpoint_include_optimizer,
                    )
                )

            with self._stage_timer.stage("tb"):
                self._pre_train_step_tensorboard_publish(
                    erll_epoch_batch_size,
                    is_erll_initialization_epoch,
                    train_buffer_iterator,
                )

            if self._pre_train_step_callback:
                self._pre_train_step_callback(
                    model=self.model_trainer.model,
                    uder_epoch=each_erll_epoch,
                    global_epoch=resume_offset + len(post_train_score.total_train_losses),
                    replay_buffer=self.train_replay_buffer,
                )

            # .... Pre-training sanity check ......................................................
            assert self.train_replay_buffer.num_stored > 0, (
                f"The train_replay_buffer is empty {self.train_replay_buffer.num_stored=}!"
                "Make sure the implementation of `self._pre_train_step` method is ok."
            )

            # ==== ERLL epoch training stage ======================================================
            # RLRP-736: defer early-stopping patience until the optimizer LR warmup phase ends.
            # The trainer LR schedule warmup (`trainer_lr_schedule.warmup.steps`) is a one-time
            # global optimizer stabilization stage; during it the LR is still ramping so the
            # `val/loss` is not representative. Passing the warmup length as `patience_warmup_epochs`
            # makes the per-call `EarlyStopping` skip patience counting (and best-score seeding)
            # until warmup completes, preventing premature stop on the post-warmup LR-jump transient.
            _patience_warmup_epochs = int(
                omegaconf.OmegaConf.select(
                    self._cfg_training,
                    "trainer_lr_schedule.warmup.steps",
                    default=0,
                )
                or 0
            )

            # RLRP-773 G2: epoch-interval checkpointing disables early-stopping for the run so the
            # full planned budget runs and the final epoch E lands deterministically on an interval
            # multiple. `patience=None` is the trainer's real "never early-stop" contract
            # (`model_trainer.py`: EarlyStopping is installed only when `patience is not None`), so a
            # None here removes the callback entirely rather than relying on a large sentinel.
            _effective_patience = (
                None
                if force_checkpoint_interval is not None
                else self._cfg_training.patience
            )

            with self._stage_timer.stage("train_step"), warnings.catch_warnings():
                # Note: PyTorch-Lightning (private `torch.utils._pytree` helper in
                # `pytorch_lightning/utilities/_pytree.py`) still constructs/uses
                # `LeafSpec`, which torch >= 2.12 deprecated in favour of
                # `TreeSpec.is_leaf()`. This is an upstream PL<->torch version-mismatch
                # deprecation (not our code) that fires once when PL first flattens a
                # training batch; silence it so it does not pollute the epoch progress bar.
                warnings.filterwarnings(
                    "ignore",
                    message=r".*`isinstance\(treespec, LeafSpec\)` is deprecated.*",
                )
                erll_epoch_train_losses, erll_epoch_val_losses = (
                    self.model_trainer.train(
                        train_buffer_iterator,
                        val_buffer_iterator,
                        num_epochs=self._current_model_trainer_num_epoch_max,
                        patience=_effective_patience,
                        patience_warmup_epochs=_patience_warmup_epochs,
                        improvement_threshold=(
                            omegaconf.OmegaConf.select(
                                self._cfg_training,
                                "improvement_threshold",
                                default=0.01,
                            )
                        ),
                        callback=trainer_epoch_callback,
                        batch_callback=self._model_trainer_batch_callback,
                        silent=True,
                        # evaluate=is_final_erll_epoch # (CRITICAL) ToDo: validate (ref task RLRP-773)
                    )
                )

            post_train_score.total_train_losses += erll_epoch_train_losses
            post_train_score.total_val_losses += erll_epoch_val_losses

            # RLRP-773 R11: `ModelTrainer.train()` rewinds the model to its best-val weights before
            # returning, so a cadence (`last_epoch`) snapshot at a pass boundary differs from these
            # best-val weights. When the pass-boundary global epoch is an interval multiple, also
            # snapshot the rewound best-val weights as `epoch_<E>_pass_best/` so both provenances are
            # inspectable from disk.
            if (
                force_checkpoint_interval is not None
                and epoch_checkpoint_include_pass_best
            ):
                self._maybe_save_pass_best_epoch_checkpoint(
                    interval=force_checkpoint_interval,
                    global_completed=resume_offset + len(post_train_score.total_train_losses),
                    erll_pass=each_erll_epoch,
                    exp_cwd=epoch_checkpoint_exp_cwd,
                    include_optimizer_mode=epoch_checkpoint_include_optimizer,
                )

            # ==== ERLL epoch post-training stage =================================================
            with warnings.catch_warnings():
                # Note: This warning is intenal to pytorch when calling SequentialLr step fct,
                # so its not our problem to fix.
                warnings.filterwarnings(
                    "ignore",
                    message="The epoch parameter in `scheduler\\.step\\(\\)` was not necessary and is being deprecated where possible.*",
                )
                # RLRP-727 (B1): under Law-U the loop owns the LR explicitly (per-pass envelope +
                # inner restart), so there is no outer torch scheduler to step (`None`).
                if self._erll_lr_scheduler is not None:
                    self._erll_lr_scheduler.step()

            self._post_train_step(
                each_erll_epoch,
                erll_epoch_batch_size,
                is_erll_initialization_epoch,
                is_final_erll_epoch,
            )

            # Note: is_final_erll_epoch is required for case where max erll epoch is 1
            # Cadence gate (B1) — legacy behaviour when
            # `testtime_rollout_cadence == 1`: rollout runs on every
            # non-initialization ERLL epoch and always on the final one.
            # With N > 1, non-final epochs additionally require
            # `each_erll_epoch % N == 0` so the expensive rollout +
            # per-trajectory metric computation can be amortised over
            # several ERLL epochs without ever skipping the final one.
            _rollout_on_cadence = (each_erll_epoch % testtime_rollout_cadence) == 0
            if (
                not is_erll_initialization_epoch and _rollout_on_cadence
            ) or is_final_erll_epoch:
                with self._stage_timer.stage("rollout"):
                    prediction_metric = model_testtime_rollout_and_compute_prediction_metric(
                        self._cfg,
                        self.model_trainer.model,
                        self.motion_model_container,
                        self.torch_rng,
                        self._test_trajectories,
                        compounded_predictions_score=self._cfg.deploy.target_experiment.compounded_predictions_score,
                        next_state_deterministic_selection=self._cfg.deploy.model_runtime.next_state_deterministic_selection,
                        next_state_sampling_size=self._cfg.deploy.model_runtime.next_state_sampling_size,
                        ground_truth_feed_warmup_steps=self._cfg.deploy.target_experiment.ground_truth_feed_warmup_steps,
                        tutor_and_release=self._cfg.deploy.target_experiment.get(
                            "tutor_and_release", False
                        ),
                        show_debug_info=self._cfg.get("debug_mode", False),
                    )

                with self._stage_timer.stage("save"):
                    # RLRP-723 F2/§2.5(2b): reduce the coordinate (feature) axis by ``mean``
                    # so ``best_pred_mae_score`` is a true per-step MAE in physical units
                    # (meters) — matching the deploy/test-time scalar (``_reduce_pred_mae_to_scalar``,
                    # which also feature-means). This is a constant-factor (=#coordinates)
                    # rescale that leaves checkpoint-selection ranking unchanged.
                    post_train_score = self._score_model_performance_and_save_checkpoint_on_improvement(
                        prediction_metric,
                        post_train_score,
                        feature_reduction="mean",
                    )

                with self._stage_timer.stage("tb"):
                    param_accumulator = self._post_train_step_tensorboard_publish(
                        each_erll_epoch,
                        param_accumulator,
                        prediction_metric,
                        post_train_score,
                    )

            if self._post_train_step_callback:
                # (StandBy) inprogress: RLRP-383 perf: optimize post-train step
                #   - ToDo: refactor callback signature to accept PostTrainScore dataclass
                #   - ToDo: refactor callback signature to accept PredictionMetric dataclass
                self._post_train_step_callback(
                    model=self.model_trainer.model,
                    train_iteration=each_erll_epoch,
                    epoch=None,
                    total_avg_loss=None,
                    eval_score=None,
                    best_val_score=None,
                )

            # ==== ERLL epoch teardown ============================================================
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.register_latest_train_run_epoch_count()

            if self._progressbar_inner is not None:
                self._progressbar_inner.close()

            self._progressbar_main.moveto(-1)
            self._progressbar_main.update(1)

            # F-E1-instr: flush per-stage timings once per ERLL epoch.
            # `report()` is a zero-cost no-op when
            # `cfg.diagnostics.profile_breakdown` is false and returns
            # an empty dict, which `format_report` renders as "" — the
            # `if _profile_line:` guard below then skips the log call
            # and preserves byte-identical console output vs. legacy.
            _profile_line = self._stage_timer.format_report(self._stage_timer.report())
            if _profile_line:
                consol_msg(
                    who_am_i=self._console_acronymn,
                    msg=f"erll_epoch={each_erll_epoch} {_profile_line}",
                    space_before=False,
                    space_after=False,
                )

        # //// ERLL replay buffer iteration teardown //////////////////////////////////////////////
        training_time = time.time() - start_time

        model_checkpoint_path = os.path.join(model_data_path, "saved_dynamic_model")
        if os.path.exists(os.path.join(model_checkpoint_path, "model.pth")):
            self.model_trainer.model.load(model_checkpoint_path)
            self.model_trainer.model.model.eval()
            if self.motion_model_container:
                self.motion_model_container.set_dynamics_model(self.model_trainer.model)
        else:
            consol_msg_universal_one_liner(
                f"{ConsoleFormat.MSG_ERROR_FORMAT}Be advised no model checkpoint where "
                "saved.\n"
                f"{os.listdir(model_checkpoint_path)=}"
                f"{ConsoleFormat.MSG_END_FORMAT}"
            )

        self._progressbar_main.close()

        return (
            post_train_score.total_train_losses,
            post_train_score.total_val_losses,
            training_time,
            post_train_score.best_pred_mae_score,
            post_train_score.best_pred_val_loss,
        )

    @abc.abstractmethod
    def _pre_train_step(
        self,
        current_erll_epoch: int,
        erll_epoch_batch_size: int,
        is_erll_initialization_epoch: bool,
        is_final_erll_epoch: bool,
    ) -> None:
        """
        Defines an abstract pre-training step, differentiates between initialization and final
        epochs, and
        is aware of the current epoch during training.

        Usefull for modifying the `train_replay_buffer` at each ERLL epochs.

        :param current_erll_epoch: The current epoch number in the ERLL training process.
            This value is used to determine the position within the training sequence.
        :param erll_epoch_batch_size: The size of the batch being processed within
            the current epoch. This defines the number of data samples handled during
            a single batch.
        :param is_erll_initialization_epoch: A flag indicating whether the current
            epoch is the initialization epoch for the ERLL process. This determines
            if any special initialization steps need to be performed.
        :param is_final_erll_epoch: A flag indicating whether the current epoch is
            the final epoch in the ERLL training sequence. This is used to decide if
            any finalization tasks are required.
        :return: None
        """
        ...

    @abc.abstractmethod
    def _post_train_step(
        self,
        current_erll_epoch: int,
        erll_epoch_batch_size: int,
        is_erll_initialization_epoch: bool,
        is_final_erll_epoch: bool,
    ) -> None:
        """
        Defines an abstract post-training step executed after ERLL model training step but
        before model rollout and scoring step. differentiates between initialization and final
        epochs, and is aware of the current epoch during training.

        Usefull for modifying the `train_replay_buffer` at each ERLL epochs.

        :param current_erll_epoch: The current epoch number in the ERLL training process.
            This value is used to determine the position within the training sequence.
        :param erll_epoch_batch_size: The size of the batch being processed within
            the current epoch. This defines the number of data samples handled during
            a single batch.
        :param is_erll_initialization_epoch: A flag indicating whether the current
            epoch is the initialization epoch for the ERLL process. This determines
            if any special initialization steps need to be performed.
        :param is_final_erll_epoch: A flag indicating whether the current epoch is
            the final epoch in the ERLL training sequence. This is used to decide if
            any finalization tasks are required.
        :return: None
        """
        ...

    def _pre_train_step_tensorboard_publish(
        self, erll_epoch_batch_size, is_erll_initialization_epoch, train_buffer_iterator
    ):
        if self.tensorboard_writer is not None:
            # RLRP-727: the "dataset opt" LR monitor tracks the OUTER per-pass envelope. Under
            # Law-U there is no outer torch scheduler (`None`); the envelope is applied directly to
            # `param_groups[0]['lr']` at pass start, so report that instead of `get_last_lr()[0]`.
            if self._erll_lr_scheduler is not None:
                _dataset_opt_lr = self._erll_lr_scheduler.get_last_lr()[0]
            else:
                _dataset_opt_lr = self.model_trainer.optimizer.param_groups[0]["lr"]
            self.tensorboard_writer.add_scalar_per_epoch_monitoring(
                tag="Replay buffer optimization scheduling/LR schedule (dataset opt)",
                value=_dataset_opt_lr,
            )

            self.tensorboard_writer.add_scalar_per_epoch_monitoring(
                tag="Replay buffer optimization scheduling/batch size",
                value=erll_epoch_batch_size,
            )
            self.tensorboard_writer.add_scalar_per_epoch_monitoring(
                tag="Replay buffer optimization scheduling/dataset size",
                value=self.train_replay_buffer.num_stored,
            )

            # .... generate TensorBoard graph .................................................
            if (
                is_erll_initialization_epoch
                and fetch_cfg_pipeline_tensorboard_key_value(
                    self._cfg, key="generate_model_graph", key_value_default=False
                )
            ):
                # ToDo: fix the graph generation logic to be compatible with Lightning
                trained_model: OneDTransitionRewardModelV2 = self.model_trainer.model
                save_propagation_method = trained_model.model.propagation_method
                trained_model.model.set_propagation_method(None)
                consol_msg_universal_one_liner(
                    "Generate TensorBoard graph for "
                    f"{trained_model.model.__class__.__name__}"
                    " with propagation method set to None"
                )
                if isinstance(train_buffer_iterator, DataLoader):
                    # RLRP-824: a ``DataLoader`` is re-iterable; peek its first batch.
                    mock_batch: TransitionBatch = next(iter(train_buffer_iterator))
                else:
                    mock_batch: TransitionBatch = copy(train_buffer_iterator).__next__()
                model_in, _ = trained_model._process_batch(mock_batch)
                if self.train_replay_buffer.stores_trajectories:
                    model_in, _ = reshape_sequence_dim_into_batch_dim(
                        model_in, trained_model.model.num_members
                    )
                self.tensorboard_writer.writer.add_graph(
                    trained_model,
                    input_to_model=model_in,
                    verbose=False,
                    use_strict_trace=False,
                )
                trained_model.model.set_propagation_method(save_propagation_method)

    def _post_train_step_tensorboard_publish(
        self,
        each_erll_epoch: int,
        param_accumulator: List[Tuple[int, List[dict]]],
        prediction_metric: PredictionMetric,
        post_train_score: PostTrainScore,
    ) -> List[Tuple[int, List[dict]]]:
        if self.tensorboard_writer is not None:

            if fetch_cfg_pipeline_tensorboard_key_value(
                self._cfg, key="generate_gradient_figure", key_value_default=False
            ):
                epoch_param_accumulator = _collect_gradient_information(
                    self.model_trainer.model
                )
                param_accumulator.append((each_erll_epoch, epoch_param_accumulator))

                self.tensorboard_writer.writer.add_figure(
                    tag=f"Model gradient flow (figure)/all (no min max)",
                    figure=plot_grad_flow(
                        param_accumulator,
                        alpha=0.4,
                        show_min_max=False,
                    ),
                    global_step=each_erll_epoch,
                )
                self.tensorboard_writer.writer.add_figure(
                    tag=f"Model gradient flow (figure)/all",
                    figure=plot_grad_flow(
                        param_accumulator,
                        alpha=0.4,
                        show_min_max=True,
                        min_max_alpha=0.05,
                    ),
                    global_step=each_erll_epoch,
                )

            tensorboard_arbitrary_dimension_prediction_trajectory_writer(
                self.tensorboard_writer,
                prediction_metric,
                each_erll_epoch,
                trajectory_record_step_size=fetch_cfg_pipeline_tensorboard_key_value(
                    self._cfg, key="trajectory_record_step_size", key_value_default=100
                ),
                is_model_ensemble=is_model_ensemble(self.model_trainer.model.model)
                and self.model_trainer.model.model.num_members > 1,
                label_pre=f"[{self._cfg_key_algo}] ",
                obs_dim_label=fetch_cfg_pipeline_tensorboard_key_value(
                    self._cfg, key="obs_dim_label", key_value_default=("X", "Y", "Z")
                ),
                rollout_dim_label=fetch_cfg_pipeline_tensorboard_key_value(
                    self._cfg,
                    key="rollout_dim_label",
                    key_value_default=("X", "Y", "Z"),
                ),
                show_prediction_distributions=fetch_cfg_pipeline_tensorboard_key_value(
                    self._cfg, "generate_prediction_distributions", False
                ),
                show_prediction_rollout_mae=fetch_cfg_pipeline_tensorboard_key_value(
                    self._cfg, "generate_prediction_rollout_mae", False
                ),
            )

            self.tensorboard_writer.add_scalar_per_epoch_monitoring(
                tag=f"[{self._cfg_key_algo}] Prediction/uncertainty (epistemic)",
                value=post_train_score.new_pred_std_epi_score,
            )

            self.tensorboard_writer.add_scalar_per_epoch_monitoring(
                tag=f"[{self._cfg_key_algo}] Prediction/uncertainty (aleatoric+epistemic)",
                value=post_train_score.new_pred_std_score,
            )

            tensorboard_prediction_error(
                self._cfg,
                self._cfg_key_algo,
                post_train_score.new_pred_mae_score,
                post_train_score.new_pred_l2_norm_score,
                self.tensorboard_writer,
                # RLRP-723: pass the rollout horizon (number of scored steps) so
                # the trajectory-length normalized ``avg-mae`` / ``avg-L2 norm``
                # variants can be logged alongside the cumulative ones.
                rollout_horizon=int(prediction_metric.mae.shape[0]),
            )

            self.tensorboard_writer.add_scalar_per_epoch_monitoring(
                tag=f"[{self._cfg_key_algo}] Model/saved model nb",
                value=post_train_score.nb_saved_model,
            )

        return param_accumulator

    def _set_model_trainer_progressbar(self, is_final_erll_epoch) -> Callable:
        """
        Sets up the progress bar for the model trainer epochs to monitor the training process.

        This method wrapp the provided `model_trainer_epoch_callback` with an initialized
        progress bar that will be updated at each model trainer training epoch. In the absence of
        a pre-defined number of model trainer epochs (i.e. `_model_trainer_num_epoch=None`),
        an alternative trainer setup is initialized for of training stages.

        :param is_final_erll_epoch: Indicates whether this is the final epoch in an
            extended range of limitless training.
        :return: A callable epoch callback function that either updates an inner
            progress bar or directly executes the user-provided callback function.
        """
        final_msg = ""
        if is_final_erll_epoch:
            final_msg = f"Final {self._console_acronymn} epoch"

        if (
            self._current_model_trainer_num_epoch_max
            and self._model_trainer_epoch_callback is not None
        ):

            if is_final_erll_epoch:
                self._progressbar_main.display(f"{final_msg}")

            self._progressbar_inner = init_progressbar(
                self._current_model_trainer_num_epoch_max, "Optimiser train epoch"
            )

            def _model_trainer_epoch_callback_with_pbar(
                model,
                train_iteration,
                epoch,
                total_avg_loss,
                eval_score,
                best_val_score,
            ) -> None:
                self._model_trainer_epoch_callback(
                    model,
                    train_iteration,
                    epoch,
                    total_avg_loss,
                    eval_score,
                    best_val_score,
                )
                self._progressbar_inner.update(1)
                return None

            trainer_epoch_callback = _model_trainer_epoch_callback_with_pbar
        else:

            self._progressbar_main.display(
                f"{final_msg} › Training model without num epoch cap (Started at "
                f"{datetime.now().isoformat(sep=' ', timespec='minutes')})"
            )

            trainer_epoch_callback = self._model_trainer_epoch_callback
        return trainer_epoch_callback

    def _wrap_trainer_callback_with_intra_pass_checkpoint(
        self,
        base_callback: Optional[Callable],
        post_train_score: PostTrainScore,
        checkpoint_every_n_inner_epochs: int,
    ) -> Callable:
        """
        Wrap the inner per-epoch trainer callback so that, in addition to its usual behaviour, the
        checkpoint routine runs every ``checkpoint_every_n_inner_epochs`` inner training epochs.

        Installed by ``execute()`` **only on the final ERLL pass** (follow-up to RLRP-727): the long
        final ``train()`` call is where the model can peak-then-degrade with no intermediate save, so
        this restores periodic best-model persistence there. The shared ``post_train_score`` is
        mutated in place, so mid-pass saves and the end-of-pass outer save keep a single
        "best across the run" state.

        :param base_callback: The original inner per-epoch callback (may be ``None``).
        :param post_train_score: The run-scoped accumulator shared with the outer gate.
        :param checkpoint_every_n_inner_epochs: Cadence ``N >= 1`` in inner epochs.
        :return: A callback matching the ``ModelTrainer`` per-epoch signature.
        """

        def _intra_pass_checkpoint_callback(
            model,
            train_iteration,
            epoch,
            total_avg_loss,
            eval_score,
            best_val_score,
        ) -> None:
            if base_callback is not None:
                base_callback(
                    model,
                    train_iteration,
                    epoch,
                    total_avg_loss,
                    eval_score,
                    best_val_score,
                )
            # `epoch > 0` so we never checkpoint on the very first inner epoch (nothing trained yet).
            if epoch > 0 and (epoch % checkpoint_every_n_inner_epochs) == 0:
                self._intra_pass_checkpoint(
                    post_train_score, current_val_loss=eval_score
                )
            return None

        return _intra_pass_checkpoint_callback

    def _intra_pass_checkpoint(
        self, post_train_score: PostTrainScore, current_val_loss: Optional[float]
    ) -> None:
        """
        Run the checkpoint routine mid-``train()`` for the intra-final-pass cadence.

        Two paths, selected by ``training_common.checkpoint_performance_measure``:

        - ``val_loss`` (all current configs incl. DR): **no rollout** — select purely on the live
          ``current_val_loss`` (the current inner epoch's ``eval_score``). Cheap.
        - rollout measures (``testtime_rollout_mae`` / ``testtime_rollout_l2norm``): run the deploy
          test-time rollout, wrapped so it cannot corrupt the in-progress fit (model ``training``
          mode saved/restored, executed under ``torch.no_grad()``).

        :param post_train_score: The run-scoped accumulator (mutated in place).
        :param current_val_loss: The current inner epoch's validation score.
        """
        measure = self._read_checkpoint_option(
            "performance_measure", "checkpoint_performance_measure", "val_loss"
        )

        if measure == "val_loss":
            # Fast path: no rollout needed, select on the live val loss.
            self._score_model_performance_and_save_checkpoint_on_improvement(
                prediction_metric=None,
                post_train_score=post_train_score,
                feature_reduction="mean",
                current_val_loss=current_val_loss,
            )
            self._intra_pass_checkpoint_tensorboard_publish(post_train_score)
            return None

        # Rollout-measure path: run the (expensive) deploy rollout, state-safely.
        inner_model = self.model_trainer.model.model
        was_training = inner_model.training
        try:
            inner_model.eval()
            with torch.no_grad():
                prediction_metric = model_testtime_rollout_and_compute_prediction_metric(
                    self._cfg,
                    self.model_trainer.model,
                    self.motion_model_container,
                    self.torch_rng,
                    self._test_trajectories,
                    compounded_predictions_score=self._cfg.deploy.target_experiment.compounded_predictions_score,
                    next_state_deterministic_selection=self._cfg.deploy.model_runtime.next_state_deterministic_selection,
                    next_state_sampling_size=self._cfg.deploy.model_runtime.next_state_sampling_size,
                    ground_truth_feed_warmup_steps=self._cfg.deploy.target_experiment.ground_truth_feed_warmup_steps,
                    tutor_and_release=self._cfg.deploy.target_experiment.get(
                        "tutor_and_release", False
                    ),
                    show_debug_info=self._cfg.get("debug_mode", False),
                )
        finally:
            # Restore the Lightning fit's training mode so the outer `train()` is undisturbed.
            if was_training:
                inner_model.train()

        self._score_model_performance_and_save_checkpoint_on_improvement(
            prediction_metric=prediction_metric,
            post_train_score=post_train_score,
            feature_reduction="mean",
            current_val_loss=current_val_loss,
        )
        self._intra_pass_checkpoint_tensorboard_publish(post_train_score)
        return None

    def _intra_pass_checkpoint_tensorboard_publish(
        self, post_train_score: PostTrainScore
    ) -> None:
        """
        Record the running saved-model count to TensorBoard for an intra-final-pass checkpoint.

        The outer per-ERLL-epoch gate publishes ``Model/saved model nb`` once per pass via
        ``_post_train_step_tensorboard_publish``. The intra-final-pass cadence fires *inside* the
        long final ``train()`` call, so without this the mid-pass saves bump
        ``post_train_score.nb_saved_model`` in memory but never appear on the graph until the
        end-of-pass publish. Logging it here (against the inner-epoch counter, matching the
        outer tag) gives per-save granularity during the final pass. No-op when no writer is set.

        :param post_train_score: The run-scoped accumulator holding ``nb_saved_model``.
        """
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar_per_epoch_monitoring(
                tag=f"[{self._cfg_key_algo}] Model/saved model nb",
                value=post_train_score.nb_saved_model,
            )
        return None

    def _read_checkpoint_option(self, new_key: str, legacy_key: str, default):
        """Read a ``training_common.checkpoint.<new_key>`` option with a legacy fallback (RLRP-773).

        The checkpoint knobs (``performance_measure``, ``force_interval``, ``include_optimizer``)
        were aggregated under a dedicated ``training_common.checkpoint`` group. Configs that still
        set the old flat ``training_common.<legacy_key>`` key keep working via this fallback
        (the new grouped key wins when both are present).

        :param new_key: the sub-key under ``training_common.checkpoint``.
        :param legacy_key: the legacy flat key under ``training_common``.
        :param default: value returned when neither key is set.
        :return: the resolved option value.
        """
        value = omegaconf.OmegaConf.select(
            self._cfg, f"training_common.checkpoint.{new_key}", default=None
        )
        if value is not None:
            return value
        return omegaconf.OmegaConf.select(
            self._cfg, f"training_common.{legacy_key}", default=default
        )

    def _epoch_checkpoint_preflight(self, interval: int) -> None:
        """Pre-flight for epoch-interval checkpointing (RLRP-773 R4/R10).

        Two responsibilities, both driven by the statically-planned per-pass epoch budgets from
        ``_planned_pass_epoch_budgets()``:

        * **Hard guard (R10)** — the feature's premise is a deterministic epoch grid ending at
          epoch ``E``. When ANY ERLL pass has a patience-driven (``null``) epoch budget,
          ``_planned_pass_epoch_budgets()`` returns ``None`` and, because the feature also disables
          early-stopping (G2), ``ModelTrainer.train`` would fall back to ``max_epochs=1000`` with no
          early stop — a silent 1000-epoch runaway with no divisor warning. Raise instead.
        * **Divisor warning (R4)** — advisory ``warnings.warn`` when ``interval`` does not evenly
          divide EVERY planned per-pass budget, i.e. the corresponding UDER global-epoch's last
          inner epoch will NOT land on an epoch checkpoint. Advisory only, never raises.

        :param interval: the configured ``force_checkpoint_interval`` (``>= 1``).
        :raises ValueError: when at least one ERLL pass has a patience-driven (``null``) budget.
        """
        budgets = self._planned_pass_epoch_budgets()
        if budgets is None:
            raise ValueError(
                f"{self._console_acronymn} `training_common.force_checkpoint_interval="
                f"{interval}` requires every ERLL pass to have an explicit integer epoch budget, "
                "but at least one pass budget is patience-driven (null) (e.g. "
                "`final_max_num_epochs_train_model: null` or `num_epochs_train_model: null`). "
                "Epoch-interval checkpointing disables early-stopping (G2), so a null budget would "
                "silently run a 1000-epoch pass with no early stop. Set explicit integer budget(s) "
                "or disable the feature (`training_common.force_checkpoint_interval: null`)."
            )

        non_dividing = [budget for budget in budgets if (budget % interval) != 0]
        if non_dividing:
            warnings.warn(
                f"{self._console_acronymn} `force_checkpoint_interval={interval}` does not evenly "
                f"divide every planned per-pass epoch budget ({budgets}); the last inner epoch of "
                f"the non-dividing pass budget(s) {non_dividing} will NOT receive an epoch "
                "checkpoint.",
                stacklevel=2,
            )
        return None

    def _save_epoch_checkpoint_for_mode(
        self,
        exp_cwd: str,
        epoch: int,
        include_optimizer_mode: Optional[str],
        extra_meta: dict,
        dir_suffix: Optional[str] = None,
    ) -> None:
        """Save one epoch checkpoint honouring the tri-state optimizer mode (RLRP-773 §11).

        Shared by the cadence wrapper and the pass-best save so the ``include_optimizer`` policy
        lives in ONE place:

        * ``None``   => model-only snapshot (no optimizer file);
        * ``"all"``  => an ``optimizer.pth`` inside THIS epoch dir;
        * ``"latest"`` => model-only epoch dir + overwrite the single run-wide
          ``optimizer_latest.pth`` so it always tracks the most recent save (Q8/Q9).

        :param exp_cwd: the experiment cwd holding the ``epoch_checkpoints/`` tree.
        :param epoch: the global completed inner-epoch count this checkpoint is tagged with.
        :param include_optimizer_mode: one of ``{None, "all", "latest"}``.
        :param extra_meta: plain-scalar manifest fields (must include ``weights_provenance``).
        :param dir_suffix: optional dir suffix (e.g. ``PASS_BEST_SUFFIX``).
        """
        persistent_checkpoint_utils.save_epoch_checkpoint(
            self.model_trainer.model,
            self.model_trainer.optimizer if include_optimizer_mode == "all" else None,
            exp_cwd,
            epoch=epoch,
            extra_meta=extra_meta,
            dir_suffix=dir_suffix,
            include_normalizers=getattr(
                self, "_epoch_checkpoint_include_normalizers", True
            ),
        )
        if include_optimizer_mode == "latest":
            persistent_checkpoint_utils.save_running_latest_optimizer(
                exp_cwd,
                self.model_trainer.optimizer,
                epoch=epoch,
                extra_meta={
                    "weights_provenance": extra_meta.get("weights_provenance"),
                    "erll_pass": extra_meta.get("erll_pass"),
                },
            )
        return None

    def _wrap_trainer_callback_with_epoch_checkpoint(
        self,
        base_callback: Optional[Callable],
        interval: int,
        global_epoch_offset: int,
        erll_pass: int,
        exp_cwd: str,
        include_optimizer_mode: Optional[str] = None,
    ) -> Callable:
        """Wrap the inner per-epoch callback to save an epoch checkpoint on cadence (RLRP-773 R1/R2).

        Mirrors ``_wrap_trainer_callback_with_intra_pass_checkpoint`` but is INDEPENDENT of
        ``post_train_score`` / best-metric gating (R5): it writes a ``(model, optimizer)`` snapshot
        unconditionally whenever the global completed inner-epoch count is a multiple of
        ``interval``. The global count is ``global_epoch_offset + epoch + 1`` because the inner
        callback reports a 0-based per-pass ``epoch`` and the trainer appends the epoch's train loss
        BEFORE invoking the callback (Q2), so at callback time exactly ``epoch + 1`` epochs of this
        pass have completed.

        :param base_callback: the original inner per-epoch callback (may be ``None``).
        :param interval: the checkpoint cadence in global completed inner epochs (``>= 1``).
        :param global_epoch_offset: epochs completed in all previous ERLL passes.
        :param erll_pass: the current ERLL pass index (stamped into the manifest).
        :param exp_cwd: the experiment cwd holding the ``epoch_checkpoints/`` tree.
        :param include_optimizer_mode: tri-state ``training_common.checkpoint.include_optimizer``
            (``None``/``"all"``/``"latest"``) — see :meth:`_save_epoch_checkpoint_for_mode`.
        :return: a callback matching the ``ModelTrainer`` per-epoch signature.
        """

        def _epoch_checkpoint_callback(
            model,
            train_iteration,
            epoch,
            total_avg_loss,
            eval_score,
            best_val_score,
        ) -> None:
            if base_callback is not None:
                base_callback(
                    model,
                    train_iteration,
                    epoch,
                    total_avg_loss,
                    eval_score,
                    best_val_score,
                )
            global_completed = global_epoch_offset + epoch + 1
            if (global_completed % interval) == 0:
                self._save_epoch_checkpoint_for_mode(
                    exp_cwd,
                    epoch=global_completed,
                    include_optimizer_mode=include_optimizer_mode,
                    extra_meta={
                        "erll_pass": int(erll_pass),
                        "pass_epoch": int(epoch),
                        "interval": int(interval),
                        "weights_provenance": "last_epoch",
                    },
                )
            return None

        return _epoch_checkpoint_callback

    def _maybe_save_pass_best_epoch_checkpoint(
        self,
        interval: int,
        global_completed: int,
        erll_pass: int,
        exp_cwd: str,
        include_optimizer_mode: Optional[str] = None,
    ) -> None:
        """Optionally snapshot the pass-boundary best-val weights (RLRP-773 R11).

        ``ModelTrainer.train()`` rewinds the model to its best-val weights before returning, so a
        cadence (``last_epoch``) snapshot taken at a pass boundary and these rewound best-val
        weights generally differ. When the pass-boundary global epoch is an interval multiple, also
        persist the rewound (best-val) weights as ``epoch_<E>_pass_best/`` so both provenances are
        inspectable from disk. Discovery ignores ``*_pass_best`` dirs unless explicitly requested.

        :param interval: the checkpoint cadence in global completed inner epochs (``>= 1``).
        :param global_completed: the global completed inner-epoch count at the pass boundary.
        :param erll_pass: the just-finished ERLL pass index.
        :param exp_cwd: the experiment cwd holding the ``epoch_checkpoints/`` tree.
        :param include_optimizer_mode: tri-state ``training_common.checkpoint.include_optimizer``
            (``None``/``"all"``/``"latest"``) — see :meth:`_save_epoch_checkpoint_for_mode`.
        """
        if (global_completed % interval) != 0:
            return None
        self._save_epoch_checkpoint_for_mode(
            exp_cwd,
            epoch=global_completed,
            include_optimizer_mode=include_optimizer_mode,
            extra_meta={
                "erll_pass": int(erll_pass),
                "interval": int(interval),
                "weights_provenance": "pass_best_val",
            },
            dir_suffix=persistent_checkpoint_utils.PASS_BEST_SUFFIX,
        )
        return None

    def _check_data_source_supported(self, data_source: ERLLDataSource) -> None:
        """Fail fast on unsupported combinations of a NON-replay-buffer data source (RLRP-824).

        The base loop accepts any ``ERLLDataSource`` for a single-member model; sub-classes that
        perform replay-buffer surgery (``PartitionBasedUncertaintyDrivenERLL``) override this
        hook to raise ``NotImplementedError``.
        """
        num_members = int(getattr(self.model_trainer.model.model, "num_members", 1) or 1)
        if num_members > 1:
            raise NotImplementedError(
                f"{self._console_acronymn} pipeline.data_manager=dataloader does not support model "
                f"ensembles (num_members={num_members}); the lazy window DataLoader path yields "
                "un-bootstrapped single-member batches (RLRP-824, out of scope)."
            )
        return None

    def _setup_replay_buffers_iterator(
        self, batch_size: int
    ) -> Tuple[TransitionIterator, TransitionIterator]:
        """
        Initializes and sets up replay buffer iterators for training and validation.

        This function configures replay buffer iterators based on whether the model being used
        requires sequence batch data. For models that process sequential data, it validates
        the logic around the maximum number of gradient updates per loop, ensures sanity checks
        for trajectory indices in buffers, and creates sequence-based iterators for both training
        and validation datasets. For non-sequential models, simpler iterators are set up using
        basic replay buffer iterators. Sanity checks are performed on both iterators to ensure
        proper setup.

        :param batch_size: Integer batch size that specifies the number of samples to be
                           processed per iteration.
        :return: A tuple containing the training buffer iterator and the validation buffer
                 iterator.
        """
        if not self._data_source.is_replay_buffer:
            # RLRP-824 (plan KD4 / FR12): the lazy window source owns ONE ``DataLoader`` pair for
            # the whole run (constant batch size; ``persistent_workers`` when ``num_workers > 0``)
            # that ``ModelTrainer.train`` consumes as-is.
            train_loader = self._data_source.train_iterable(batch_size)
            val_loader = self._data_source.val_iterable(
                self._cfg_algo.batch_size.init_value
            )
            assert len(train_loader) > 0, f"{len(train_loader)=} !> 0"
            assert val_loader is not None and len(val_loader) > 0, "empty validation loader"
            return train_loader, val_loader

        if (
            isinstance(self.model_trainer.model.model, AutoRegressiveSequenceIterator)
            and self.model_trainer.model.model.receive_sequence_batch
        ):
            # .... Sequence data management  ..................................................
            num_grad_updates = self._set_number_of_gradiend_updates()

            train_buffer_iterator, _ = common_utils.get_sequence_buffer_iterator(
                self.train_replay_buffer,
                batch_size=batch_size,
                val_ratio=0,
                sequence_length=self._buffer_iterator_sequence_len,
                ensemble_size=self.model_trainer.model.model.num_members,
                shuffle_each_epoch=True,
                use_simple_sampler=False,
                max_batches_per_loop_train=num_grad_updates,
            )

            val_buffer_iterator, _ = common_utils.get_sequence_buffer_iterator(
                self.val_replay_buffer,
                batch_size=batch_size,
                val_ratio=0,
                sequence_length=self._val_buffer_iterator_sequence_len,
                ensemble_size=self.model_trainer.model.model.num_members,
                shuffle_each_epoch=True,
                use_simple_sampler=False,
                max_batches_per_loop_train=num_grad_updates,
            )

            # .... Sequence data related sanity check .........................................
            assert (
                len(self.train_replay_buffer.trajectory_indices) > 0
            ), f"{len(self.train_replay_buffer.trajectory_indices)} !> 0"

            assert (
                len(self.val_replay_buffer.trajectory_indices) > 0
            ), f"{len(self.val_replay_buffer.trajectory_indices)} !> 0"

        else:
            train_buffer_iterator, _ = common_utils.get_basic_buffer_iterators(
                self.train_replay_buffer,
                batch_size,
                val_ratio=0,
                ensemble_size=self.model_trainer.model.model.num_members,
                shuffle_each_epoch=True,
            )

            val_buffer_iterator, _ = common_utils.get_basic_buffer_iterators(
                self.val_replay_buffer,
                self._cfg_algo.batch_size.init_value,
                val_ratio=0,
                ensemble_size=self.model_trainer.model.model.num_members,
                shuffle_each_epoch=True,
            )

        # .... Buffer iterator related sanity check ...........................................
        assert (
            train_buffer_iterator.num_stored > 0
        ), f"{train_buffer_iterator.num_stored=} !> 0"
        assert (
            val_buffer_iterator.num_stored > 0
        ), f"{val_buffer_iterator.num_stored=} !> 0"

        return train_buffer_iterator, val_buffer_iterator

    def _set_number_of_gradiend_updates(self):
        # (CRITICAL) ToDo: validate `num_grad_updates` logic (ref task RLRP-220)
        max_sequence_batches_per_loop_train: Optional[Union[int, str]] = (
            self._cfg_algo.get("max_sequence_batches_per_loop_train", None)
        )
        if max_sequence_batches_per_loop_train == "match_optimizer_num_epoch":
            num_grad_updates: Optional[int] = (
                self._current_model_trainer_num_epoch_max
                if self._current_model_trainer_num_epoch_max is not None
                else 1
            )
        else:
            num_grad_updates = max_sequence_batches_per_loop_train
        return num_grad_updates

    def _check_required_config_key(
        self, cfg: DictConfig, cfg_training: DictConfig
    ) -> None:
        for each_key in self.cfg_required_key:
            is_cfg_key_exist(cfg, each_key, raise_error=True)

        for each_key in self.cfg_algo_required_key:
            is_cfg_key_exist(cfg, f"{self._cfg_key_algo}.{each_key}", raise_error=True)

        assert omegaconf.OmegaConf.is_dict(cfg.get(self._cfg_key_algo).batch_size)

        for each_key in self.cfg_training_required_key:
            is_cfg_key_exist(cfg_training, each_key, raise_error=True)

        return None

    @abc.abstractmethod
    def _batch_size_scheduling(self, epoch: int) -> int:
        """
        Batch-size policy seam (RLRP-727) invoked once per ERLL epoch by ``execute()``.

        Moved out of the abstract base so the batch-size *policy* is owned by the concrete loop
        (clean separation of concern):

        - ``ProgressiveBatchExperienceReplay`` (and, by inheritance, the partition-based UDER)
          implements the geometric progressive schedule.
        - ``SingleGlobalLoopERLL`` returns the constant ``batch_size.init_value`` (no growth).

        :param epoch: the current ERLL epoch index.
        :return: the batch size to use for this ERLL epoch.
        """
        raise NotImplementedError

    def _score_model_performance_and_save_checkpoint_on_improvement(
        self,
        prediction_metric: Optional[PredictionMetric],
        post_train_score: PostTrainScore,
        feature_reduction: str = "mean",
        current_val_loss: Optional[float] = None,
    ) -> PostTrainScore:
        """
        Compute dimension wide prediction uncertainty (ale+epi), prediction epistemic
        uncertainty and prediction MAE score then checkpoint the model if the prediction MAE (or latest val_loss)
        score improve.

        :param prediction_metric: The test-time rollout metrics. May be ``None`` on the
            intra-final-pass ``val_loss`` fast path (no rollout is run), in which case the
            last-known rollout scores are reused and only the ``val_loss`` measure is eligible
            for checkpointing (asserted below).
        :param current_val_loss: Optional live validation loss to select on for the ``val_loss``
            measure. When ``None`` (default, legacy path) the last value of
            ``post_train_score.total_val_losses`` is used. The intra-final-pass cadence passes the
            current inner epoch's ``eval_score`` here because ``total_val_losses`` is only extended
            *after* ``train()`` returns, so mid-pass it would otherwise hold a stale value.
        """

        # .... val loss selection value (live override wins) ......................................
        if current_val_loss is not None:
            new_val_losses = float(current_val_loss)
        else:
            new_val_losses = float(post_train_score.total_val_losses[-1])

        # .... Scoring logic ......................................................................
        if prediction_metric is not None:
            # sum over trajectory steps
            pred_std_score = torch.sum(prediction_metric.std, dim=0)
            pred_epi_score = torch.sum(prediction_metric.std_epi, dim=0)
            pred_mae_score = torch.sum(prediction_metric.mae, dim=0)
            pred_l2_norm_score = torch.sum(prediction_metric.l2_norm, dim=0)

            # Training Speed & Efficiency plan, stage 2 Batch S2-2 — action B2.
            # Coalesce three per-tensor device syncs (one per `float(torch.*)` call)
            # into a single `.cpu().tolist()` round-trip. Bit-exact vs. the legacy
            # three-call path on CPU and CUDA (dtype-preserving reduction + identical
            # scalar cast semantics); the only observable difference is one
            # `cudaStreamSynchronize` on CUDA instead of three.
            if feature_reduction == "mean":
                reduced = torch.stack(
                    [
                        torch.mean(pred_std_score),
                        torch.mean(pred_epi_score),
                        torch.mean(pred_mae_score),
                        torch.mean(pred_l2_norm_score),
                    ]
                )
            elif feature_reduction == "sum":
                reduced = torch.stack(
                    [
                        torch.sum(pred_std_score),
                        torch.sum(pred_epi_score),
                        torch.sum(pred_mae_score),
                        torch.sum(pred_l2_norm_score),
                    ]
                )
            else:
                raise NotImplementedError(f"{feature_reduction=} not implemented")

            (
                new_pred_std_score,
                new_pred_std_epi_score,
                new_pred_mae_score,
                new_l2_norm_score,
            ) = reduced.cpu().tolist()

            post_train_score.new_pred_std_score = new_pred_std_score
            post_train_score.new_pred_std_epi_score = new_pred_std_epi_score
            post_train_score.new_pred_mae_score = new_pred_mae_score
            post_train_score.new_pred_l2_norm_score = new_l2_norm_score
        else:
            # Intra-final-pass `val_loss` fast path: no rollout was run, so reuse the last-known
            # rollout scores (recorded by the most recent rollout-based call). Only the `val_loss`
            # measure may drive checkpointing here (guarded below).
            new_pred_std_score = post_train_score.new_pred_std_score
            new_pred_std_epi_score = post_train_score.new_pred_std_epi_score
            new_pred_mae_score = post_train_score.new_pred_mae_score
            new_l2_norm_score = post_train_score.new_pred_l2_norm_score

        # .... Checkpointing logic ................................................................
        model_checkpoint_path = os.path.join(
            get_hydra_experiment_cwd(self._cfg),
            setup_saved_model_dir(self.model_trainer.model),
        )

        if not os.path.exists(model_checkpoint_path):
            os.makedirs(model_checkpoint_path)

        cfg_checkpoint_performance_measure = self._read_checkpoint_option(
            "performance_measure", "checkpoint_performance_measure", "val_loss"
        )
        assert not (
            prediction_metric is None
            and cfg_checkpoint_performance_measure != "val_loss"
        ), (
            "Intra-final-pass checkpointing without a rollout (`prediction_metric=None`) is only "
            f"valid for the `val_loss` measure, got `{cfg_checkpoint_performance_measure}`."
        )
        if cfg_checkpoint_performance_measure == "testtime_rollout_mae":
            if new_pred_mae_score <= post_train_score.best_pred_mae_score:
                post_train_score.nb_saved_model += 1
                self.model_trainer.model.save(model_checkpoint_path)
        elif cfg_checkpoint_performance_measure == "val_loss":
            if new_val_losses <= post_train_score.best_pred_val_loss:
                post_train_score.nb_saved_model += 1
                self.model_trainer.model.save(model_checkpoint_path)
        elif cfg_checkpoint_performance_measure == "testtime_rollout_l2norm":
            if new_l2_norm_score <= post_train_score.best_pred_l2_norm_score:
                post_train_score.nb_saved_model += 1
                self.model_trainer.model.save(model_checkpoint_path)
        else:
            raise NotImplementedError(
                f"Config training_common.checkpoint_performance_measure: {cfg_checkpoint_performance_measure} is not supported!"
            )

        # .... Record stats .......................................................................
        if new_pred_mae_score <= post_train_score.best_pred_mae_score:
            post_train_score.best_pred_mae_score = new_pred_mae_score

        if new_val_losses <= post_train_score.best_pred_val_loss:
            post_train_score.best_pred_val_loss = new_val_losses

        if new_l2_norm_score <= post_train_score.best_pred_l2_norm_score:
            post_train_score.best_pred_l2_norm_score = new_l2_norm_score

        return post_train_score
