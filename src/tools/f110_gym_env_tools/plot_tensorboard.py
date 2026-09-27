# coding=utf-8
import json
import os
import shutil
import warnings
from typing import Any, Optional, Sequence, Union
import omegaconf

import torch
import numpy as np

import mbrl.models
from hydra.core.hydra_config import HydraConfig

from tools.console_tools.format import ConsoleFormat
from tools.console_tools.message import (
    consol_msg_motion_model_learner_one_line,
    )
from tools.general_utils import sanitize_dirname
from tools.hydra_apps_tools.hydra_utils import (
    get_hydra_experiment_cwd,
    is_hydra_multirun,
)
from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist


class OnlineTensorboardWritter:
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        reset_tmp_dir: bool = False,
        clear_tmp_directory_run_artifact_on_teardown=False,
        comment: Optional[str] = None,
    ) -> None:
        """TensorBoard support either as a callback or by using the
        `add_sequential_scalar_values` method.

        Usage:
            1. Use a `OnlineTensorboardWritter` instance as a callback function by adding values
               that you want to monitor at runtime to the __call__ method;
            2. or manualy use `add_sequential_scalar_values` method to add already collected data.

                >>> tensorboard_plotter = OnlineTensorboardWritter(cfg)
                >>> tensorboard_plotter.add_sequential_scalar_values(
                >>>     tag='Deploy › pose x error', scalar_value_array=(0.1, 0.5, 0.2, 0.4,)
                >>> )

        :param cfg: an hydra configuration object
        :param reset_tmp_dir: Set to True to reset the `artifact/tensorboard_tmp` directory
        :param clear_tmp_directory_run_artifact_on_teardown: Set to true to move the run artifact
        instead of copying it.
        """
        self.experiment = cfg.experiment
        self._exp_data_dir_realpath = get_hydra_experiment_cwd(cfg)
        self.comment = comment or ""
        dir_name = sanitize_dirname(self.comment)

        self.clear_tmp_directory_run_artifact_on_teardown = (
            clear_tmp_directory_run_artifact_on_teardown
        )

        self.train_run_latest_epoch = 0
        self.train_iteration = 0
        self.train_epochs_per_iteration = []
        self.total_epoch = 0

        self.rng = np.random.default_rng()

        # Lossless, legacy-default throttling knobs (Training Speed &
        # Efficiency plan, stage 1 Batch 1 — A5 + C5). Values live under
        # `cfg.pipeline.tensorboard.*` and fall back to
        # legacy defaults when absent so existing experiment cfgs keep
        # working bit-exact.
        #   - flush_secs: passed to torch SummaryWriter; default 120 s
        #     matches the upstream default (no behaviour change).
        #   - publish_every_n_erll_epochs: heavy publishes (e.g. the
        #     sequential-probability-distribution histogram path which
        #     does a .cpu().numpy() + RNG sampling + per-step
        #     add_histogram) are skipped when the current ERLL epoch
        #     does not match the cadence. Default 1 == publish on every
        #     ERLL epoch (legacy).
        _cfg_tb = omegaconf.OmegaConf.select(cfg.pipeline, "tensorboard", default=None)
        self._flush_secs: int = int(
            omegaconf.OmegaConf.select(_cfg_tb, "flush_secs", default=120)
            if _cfg_tb is not None
            else 120
        )
        self._publish_every_n_erll_epochs: int = int(
            omegaconf.OmegaConf.select(
                _cfg_tb, "publish_every_n_erll_epochs", default=1
            )
            if _cfg_tb is not None
            else 1
        )
        assert self._publish_every_n_erll_epochs >= 1, (
            "`cfg.pipeline.tensorboard.publish_every_n_erll_epochs` must be >= 1 "
            f"(got {self._publish_every_n_erll_epochs})"
        )
        # ERLL-epoch cursor; defaults to 0 so the first ERLL epoch
        # always publishes heavy content (cadence gate evaluates True
        # for epoch 0 regardless of N).
        self._current_erll_epoch: int = 0

        if is_hydra_multirun():
            exp_multirun_run_id = os.path.basename(self._exp_data_dir_realpath)
            self.exp_date = os.path.basename(
                os.path.dirname(os.path.dirname(self._exp_data_dir_realpath))
            )
            self.exp_time = os.path.basename(
                os.path.dirname(self._exp_data_dir_realpath)
            )

            if is_cfg_key_exist(cfg, "overrides.experiment") and is_cfg_key_exist(cfg, "experiment_super"):
                # Multirun experiment give a distinctive name to the last exp dir
                dir_name = os.path.basename(cfg.overrides.experiment)

            self.exp_base_dir = os.path.join(
                self.exp_date, self.exp_time, exp_multirun_run_id, dir_name
            )
        else:
            self.exp_date = os.path.basename(
                os.path.dirname(self._exp_data_dir_realpath)
            )
            self.exp_time = os.path.basename(self._exp_data_dir_realpath)
            self.exp_base_dir = os.path.join(self.exp_date, self.exp_time, dir_name)

        artifact_dir_abs_path = cfg.project_experiment_root_path

        # Sanity check
        assert os.path.exists(artifact_dir_abs_path)

        self.tensorboard_tmp_dir = os.path.realpath(
            os.path.join(artifact_dir_abs_path, "tensorboard_tmp")
        )

        if self.experiment == "DEBUG" or cfg.get("experiment_super", "") == "DEBUG":
            self.tensorboard_tmp_dir = os.path.join(self.tensorboard_tmp_dir, "DEBUG")

        if not os.path.exists(self.tensorboard_tmp_dir):
            os.makedirs(self.tensorboard_tmp_dir)

        if reset_tmp_dir:
            self._reset_tensorboard_tmp_directory()

        self._init_tensorboard_writer()

        consol_msg_motion_model_learner_one_line(
            "From a DN container termninal, open tensorboard using the following "
            "command:\n\n"
            f"  {ConsoleFormat.MSG_DIMMED_FORMAT}$ tensorboard --bind_all --logdir="
            f"{self.tensorboard_tmp_dir}/{self.exp_date}{ConsoleFormat.MSG_END_FORMAT}\n\n"
            "and then open a web browser using the explicit host ip adress "
            f"e.g.: (Griffintown wlan) http://10.0.0.86:6006\n"
            "The recorded tensorboard file will be copied to the experiment directory on exit."
        )

    def clear_tensorboard_tmp_directory_content(self) -> None:
        """Reset the content of `artifact/tensorboard_tmp/`"""
        self._reset_tensorboard_tmp_directory()
        self._init_tensorboard_writer()
        return None

    def __call__(
        self,
        model: mbrl.models.Model,
        train_iteration: int,
        epoch: int,
        total_avg_loss: float,
        eval_score: float,
        best_val_score: float,
    ) -> None:
        """Use as a callback to monitor values at runtime."""
        self._increment_total_epoch()
        self.train_run_latest_epoch = epoch
        self.train_iteration = train_iteration

        self.writer.add_scalar(
            tag="Loss/train avg loss",
            scalar_value=total_avg_loss,
            global_step=self.total_epoch,
            new_style=True,
        )
        self.writer.add_scalar(
            tag="Loss/val avg loss",
            scalar_value=eval_score.mean().item(),
            global_step=self.total_epoch,
            new_style=True,
        )
        return None

    def add_scalar_per_epoch_monitoring(
        self, tag: str, value: Union[int, float], epoch: Optional[int] = None
    ) -> None:
        """Record arbitrary value.

        :param tag: the value name displayed in TensorBoard (Use `<GROUP>/<NAME>` to group tag)
        :param value: value to monitor.
        :param epoch: manualy set the epoch if not used in conjunction with `__call__`.
        """
        if epoch is None:
            epoch = self.total_epoch

        self.writer.add_scalar(
            tag=tag,
            scalar_value=value,
            global_step=epoch,
            new_style=True,
        )
        return None

    @property
    def current_global_epoch(self) -> int:
        """The monotonic *global* epoch counter shared by every scalar publish (RLRP-289).

        The `mbrl.models.ModelTrainer.train(...)` callback `epoch` argument is the *inner*
        Lightning `trainer.current_epoch`, which restarts at 0 at every `train(...)` call (i.e. at
        every ERLL/UDER pass). Publishing histograms with that inner value collides with the step
        axis already written by the previous pass, which makes TensorBoard purge/overwrite the tag
        and renders the parameter distribution as if it were frozen from the second pass onward.

        Use this property as `global_step` for any per-epoch publish so histograms share the same
        axis as `add_scalar_per_epoch_monitoring` and `__call__`.
        """
        return self.total_epoch

    def register_latest_train_run_epoch_count(self) -> None:
        """Note: Execute after executing `mbrl.models.ModelTrainer` method `train(...)`"""
        self.train_epochs_per_iteration.append(self.train_run_latest_epoch)
        return None

    def set_current_erll_epoch(self, erll_epoch: int) -> None:
        """Record the current ERLL epoch for cadence-based publish gating.

        Introduced by the Training Speed & Efficiency plan (stage 1,
        Batch 1 — C5). The ERLL loop calls this at the top of each
        ERLL epoch so that heavy publish methods (histograms, etc.)
        can be skipped when the epoch index does not match
        `cfg.pipeline.tensorboard.publish_every_n_erll_epochs`.

        The default cadence is `1` (every ERLL epoch), which preserves
        legacy behaviour bit-exact.
        """
        self._current_erll_epoch = int(erll_epoch)
        return None

    def should_publish_heavy(self, force: bool = False) -> bool:
        """Return True when the current ERLL epoch matches the publish
        cadence set by `cfg.pipeline.tensorboard.publish_every_n_erll_epochs`.

        :param force: when True, bypass the cadence gate (use for the
            final ERLL epoch or when the operator explicitly asks for a
            publish regardless of the cadence setting).
        """
        if force:
            return True
        n = self._publish_every_n_erll_epochs
        if n <= 1:
            return True
        # Epoch 0 always publishes so runs with ``uder_num_epochs == 0``
        # or a single ERLL epoch keep producing histograms.
        return (self._current_erll_epoch % n) == 0

    def _increment_total_epoch(self) -> None:
        self.total_epoch += 1
        return None

    def add_sequential_scalar_values(
        self,
        tag: str,
        scalar_value_array: Union[np.ndarray, Sequence[Union[int, float]]],
        step_size=1,
    ) -> None:
        """Add an array of sequentialy collected data i.e. a trajectory.

        :param tag: The labbel that will be shown on TensorBoard
        :param scalar_value_array: the array
        :param step_size:
        :return: None
        """
        trj_len = len(scalar_value_array)
        for idx in np.arange(0, trj_len, step_size):
            self.writer.add_scalar(
                tag, scalar_value=scalar_value_array[idx], global_step=idx
            )
        return None

    def add_sequential_probability_distribution(
        self,
        tag: str,
        trajectory_means: Union[torch.Tensor, np.ndarray],
        trajectory_stds: Union[torch.Tensor, np.ndarray],
        step_size: int = 1,
    ) -> None:
        # (Critical) ToDo: this fct can't represent multi-modal distribution, it should take a torch distribution object instead of "trajectory_means" and "trajectory_stds" ⚠️
        """Add an array of sequentialy collected probability distribution i.e. a trajectory.

        :param tag: The labbel that will be shown on TensorBoard
        :param trajectory_means: a sequence of probability distribution means
        :param trajectory_stds: a sequence of probability distribution std
        :param step_size: timestep jump forward between each record i.e., (trj_len/step_size)=nb of histogram displayed per trajectory
        :return: None
        """
        # Cadence gate (Training Speed & Efficiency plan — C5). Skips the
        # per-call `.cpu().numpy()` + RNG sampling + per-step
        # `add_histogram` work when the current ERLL epoch is not on
        # cadence. Legacy default is publish on every ERLL epoch
        # (`cfg.pipeline.tensorboard.publish_every_n_erll_epochs == 1`).
        if not self.should_publish_heavy():
            return None

        DISTRIBUTION_SIZE = 100

        try:
            # Torch-first: convert to numpy once for TensorBoard API
            if isinstance(trajectory_means, torch.Tensor):
                trajectory_means = trajectory_means.detach().cpu().numpy()
            if isinstance(trajectory_stds, torch.Tensor):
                trajectory_stds = trajectory_stds.detach().cpu().numpy()

                trajectory_means = trajectory_means.squeeze()
                trajectory_stds = trajectory_stds.squeeze()
                assert trajectory_means.ndim == 1
                assert trajectory_stds.ndim == 1
                assert trajectory_means.size == trajectory_stds.size
                trj_len = trajectory_means.size
                assert trj_len >= step_size

                # Note: Tensorboard add_histogram can't handle large number
                # (NICE TO HAVE) ToDo: find the highest cliping value that prevent bug
                VIS_CLIP_VALUE = 1e10
                trajectory_stds = np.clip(
                    trajectory_stds, a_min=-VIS_CLIP_VALUE, a_max=VIS_CLIP_VALUE
                )

            # Pre-generate all sampled distributions in a single batched operation
            step_indices = np.arange(0, trj_len, step_size)
            means_batch = trajectory_means[step_indices]
            stds_batch = trajectory_stds[step_indices]
            all_distributions = self.rng.normal(
                means_batch[:, np.newaxis],
                stds_batch[:, np.newaxis],
                (len(step_indices), DISTRIBUTION_SIZE),
            )
            all_distributions = np.nan_to_num(all_distributions)

            for i, idx in enumerate(step_indices):
                try:
                    self.writer.add_histogram(
                        tag,
                        values=all_distributions[i],
                        bins="auto",
                        global_step=idx,
                    )
                except ValueError as e:
                    print(f"{stds_batch[i].dtype}")
                    warnings.warn(str(e))
                    print(f"\n\n{trajectory_means=}\n\n{trajectory_stds=}\n\n")
                    print(f"{idx=}, mean={means_batch[i]}, std={stds_batch[i]}")
                    print(f"{idx=}", all_distributions[i])
        except AssertionError as e:
            warnings.warn(str(e))
        return None

    def close(self) -> None:
        """Close the summary writer and copy data to experiment path"""
        self.writer.close()

        ts_tmp_path = os.path.join(self.tensorboard_tmp_dir, self.exp_base_dir)
        tf_events_file_names = os.listdir(ts_tmp_path)
        exp_data_tensorboard_path = self._exp_data_dir_realpath
        for each_events_file in tf_events_file_names:
            # Implemented in two step (move+rm or copy) to prevent error case where memory quota
            # is almost reached on hpc
            if self.clear_tmp_directory_run_artifact_on_teardown:
                shutil.move(
                    os.path.join(ts_tmp_path, each_events_file),
                    exp_data_tensorboard_path,
                )
            else:
                shutil.copy(
                    os.path.join(ts_tmp_path, each_events_file),
                    exp_data_tensorboard_path,
                )

        if self.clear_tmp_directory_run_artifact_on_teardown:
            shutil.rmtree(ts_tmp_path)

        consol_msg_motion_model_learner_one_line(
            "Recorded tensorboard file have been consolidated to the experiment directory. "
            "From a DN container termninal, open tensorboard using the following "
            "command:\n\n"
            f"  {ConsoleFormat.MSG_DIMMED_FORMAT}$ tensorboard --bind_all --logdir="
            f"{exp_data_tensorboard_path}{ConsoleFormat.MSG_END_FORMAT}\n\n"
            "and then open a web browser using the explicit host ip adress "
            "e.g.: (Griffintown wlan) http://10.0.0.86:6006"
        )
        return None

    def _init_tensorboard_writer(self) -> None:
        """Utility for TensorBoard summary writer initialization"""

        # Note: Quick hack for dealing with protobuf related error
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(
            log_dir=os.path.join(self.tensorboard_tmp_dir, self.exp_base_dir),
            filename_suffix="_tensorboard",
            flush_secs=self._flush_secs,
        )

        return None

    def add_hydra_override_text(self):
        hydra_conf = HydraConfig.get()
        hydra_job_task_overrides = omegaconf.OmegaConf.to_yaml(
            hydra_conf.overrides.task
        )

        self.writer.add_text(
            tag="Experiment",
            text_string=(
                f"Experiment '{self.experiment}' at {self.exp_base_dir}\n"
                f"Hydra overrides.task:\n{hydra_job_task_overrides}"
            ),
        )
        return None

    @staticmethod
    def pretty_json(hp):
        """Credit https://www.tensorflow.org/tensorboard/text_summaries"""
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))

    def add_hydra_cfg_text(self, cfg_string: str = ""):
        self.writer.add_text(
            tag="Hydra configuration selected hparam",
            text_string=cfg_string,
        )
        return None

    def add_hydra_cfg_selected_hparam(self, cfg_object: Any):
        self.writer.add_hparams(
            hparam_dict=omegaconf.OmegaConf.to_container(
                omegaconf.OmegaConf.create(cfg_object)
            ),
            metric_dict={},
            run_name=os.path.join(
                self.tensorboard_tmp_dir, self.exp_base_dir, "hparam"
            ),
        )
        return None

    def _reset_tensorboard_tmp_directory(self) -> None:
        tmp_copy_path = os.path.join(
            os.path.dirname(self.tensorboard_tmp_dir), "tmp_copy"
        )
        os.makedirs(tmp_copy_path)
        shutil.move(
            os.path.join(os.path.dirname(self.tensorboard_tmp_dir), "reference_exp"),
            tmp_copy_path,
        )
        shutil.rmtree(path=self.tensorboard_tmp_dir, ignore_errors=True)
        os.makedirs(self.tensorboard_tmp_dir)
        shutil.move(
            os.path.join(tmp_copy_path, "reference_exp"), self.tensorboard_tmp_dir
        )
        shutil.rmtree(path=tmp_copy_path, ignore_errors=True)

        return None
