# coding=utf-8

from typing import Optional, Tuple

import mbrl.models
import numpy as np
import omegaconf
from matplotlib import pyplot as plt
from mbrl.types import TransitionBatch
from mbrl.util import ReplayBuffer

from tools.console_tools.progressbar_tools import init_progressbar
from math_gymnasium.tools.plot_3d_utils import (
    LEGEND_BBOX_TO_ANCHOR,
    three_dimension_environment_space_plot,
)
from tools.hydra_apps_tools.hydra_utils import get_hydra_experiment_id
from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist
from tools.mbrl_lib_tools.models.utils import is_model_ensemble, is_probabilistic_model
from tools.multistep_tools.models import (
    AbstractMS2SSAutoRegressive,
    AbstractTemporalyWeightedMultiStepMLP,
    CompoundedPredictionMultiStepIterator,
    AutoRegressiveSequenceIterator,
    MS2MS2SSArTemporalMixturePME,
    MultiStepMLP,
)


def three_dimension_partition_based_UDER_sampling_plot(
    cfg: omegaconf.DictConfig,
    replay_buffer: ReplayBuffer,
    uder_epoch: int,
    global_epoch: int,
    time_space: np.ndarray,
    state_space_3d: np.ndarray,
    state_space_3d_with_noise: np.ndarray,
    state_space_label: str,
    training_time: float,
    horizon_len: Optional[int],
    model: mbrl.models.Model,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes, plt.Axes]:
    all_samples: TransitionBatch = replay_buffer.get_all(shuffle=False)

    progressbar = init_progressbar(
        len(all_samples), description="Dataset opt epoch sampling plotter"
    )

    explo_min_idx = +np.inf
    explo_max_idx = -np.inf

    if is_cfg_key_exist(cfg, "environment.explorable_space"):
        explorable_space_interval = [*cfg.environment.explorable_space]
    else:
        explorable_space_interval = [[0, len(time_space)]]

    for each_slice in explorable_space_interval:
        explo_min_idx = (
            each_slice[0] if each_slice[0] < explo_min_idx else explo_min_idx
        )
        explo_max_idx = (
            each_slice[1] if each_slice[1] > explo_max_idx else explo_max_idx
        )

    time_space_bound = cfg.environment.time_space.bound
    sample_plot_len = (time_space_bound[1] - time_space_bound[0]) / (
        explo_max_idx - explo_min_idx
    )

    # Note: ``training_time`` is a *duration* in seconds (not an epoch timestamp).
    # Using ``time.localtime`` here would treat it as seconds-since-epoch and a
    # ``"%Mm %Ss"`` format would silently drop the hour/day components, leading to
    # grossly under-reported wall-clock times (e.g. a ~7h training rendered as
    # "03m 39s"). Compute h/m/s manually so the formatting also handles durations
    # longer than 24 hours correctly.
    _total_seconds = int(round(float(training_time)))
    _hours, _remainder = divmod(_total_seconds, 3600)
    _minutes, _seconds = divmod(_remainder, 60)
    training_time = f"{_hours:02d}h {_minutes:02d}m {_seconds:02d}s"
    env_fig, ax_3d, ax_z, ax_x, ax_y = three_dimension_environment_space_plot(
        cfg,
        time_space=time_space,
        state_space_3d=state_space_3d,
        state_space_3d_with_noise=state_space_3d_with_noise,
        title=(
            "Replay buffer exploration policy selected samples "
            f"(replay buffer optimization epoch {uder_epoch}, "
            f"global epoch {global_epoch}, {training_time})"
        ),
        state_space_label=state_space_label,
        subplot_1d_interval=slice(explo_min_idx, explo_max_idx),
        show_samples=True,
        show_3d_grid=cfg.pipeline.plot.get("show_3d_grid", True),
        show_3d_axes=cfg.pipeline.plot.get("show_3d_axes", True),
        figsize=cfg.pipeline.plot.figsize,
        show_explorable_space=False,
        extra_info_str=plot_extra_info_str(cfg, model),
        experiment_id=get_hydra_experiment_id(),
    )

    show_explorable_label_once = "Selected samples"
    for idx in np.arange(len(all_samples)):
        each_sample = all_samples[idx]
        timestep_idx = int(each_sample.rewards)

        if not horizon_len:
            horizon_len = 1

        for each_axis in [ax_z, ax_x, ax_y]:
            each_axis.axvspan(
                xmin=time_space[timestep_idx],
                xmax=time_space[timestep_idx] + sample_plot_len * horizon_len,
                linewidth=0,
                color="yellowgreen",
                alpha=0.35,
                label=show_explorable_label_once,
            )
            show_explorable_label_once = ""

        progressbar.update(1)

    ax_z.legend(
        loc="upper right",
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        bbox_transform=env_fig.transFigure,
    )

    progressbar.close()
    return env_fig, ax_3d, ax_z, ax_x, ax_y


def plot_extra_info_str(
    cfg: omegaconf.DictConfig, model: mbrl.models.Model = None
) -> str:
    cfg_source_dataset = cfg.get("source_replay_buffer", None)
    cfg_uder = cfg.UDER
    cfg_uder_py = cfg_uder.get("uder_exploration_policy", None)
    sample_score_evaluation_method = None
    if cfg_uder_py:
        if cfg_uder_py.sample_score_evaluation_method == "NLPD":
            sample_score_evaluation_method = "NLPD"
        elif cfg_uder_py.sample_score_evaluation_method == "MaxEpi":
            sample_score_evaluation_method = "Max epistemic uncertainty"
        elif cfg_uder_py.sample_score_evaluation_method == "both":
            sample_score_evaluation_method = "NLPD + MaxEpi"
        else:
            raise NotImplementedError(
                f"Sample score evaluation method "
                f"'{cfg_uder_py.sample_score_evaluation_method}' not implemented!"
            )

    history_len = 1
    horizon_len = 1
    if model is not None and isinstance(model, MultiStepMLP):
        # Multi-step model
        history_len = model.history_len
        if cfg.ms_model.get("enable_multistep_head", True):
            horizon_len = model.horizon_len
        else:
            horizon_len = 1

        if isinstance(horizon_len, float):
            horizon_len = int(history_len * horizon_len)

        model_head_spec = []
        if hasattr(model, "decoder_num_layers") and hasattr(model, "decoder_hid_size"):
            model_head_spec.append(
                f"L{model.decoder_num_layers} x H{model.decoder_hid_size} → "
            )

        model_deploy_head_spec = []
        if hasattr(model, "ss_head_num_layers") or hasattr(model, "ms_head_num_layers"):
            if hasattr(model, "ms_head_num_layers"):
                model_deploy_head_spec.append(f"MS{model.ms_head_num_layers}")

            if hasattr(model, "ss_head_num_layers"):
                model_deploy_head_spec.append(f"SS{model.ss_head_num_layers}")

        if len(model_head_spec) > 0 or len(model_deploy_head_spec) > 0:
            model_head_spec = "".join(model_head_spec)
            model_head_spec = f" → De({model_head_spec}"

            if len(model_deploy_head_spec) > 0:
                if len(model_deploy_head_spec) > 1:
                    model_deploy_head_spec = "|".join(model_deploy_head_spec)
                else:
                    model_deploy_head_spec = "".join(model_deploy_head_spec)
                model_head_spec = f"{model_head_spec}{model_deploy_head_spec}"

            model_head_spec = f"{model_head_spec}) "
        else:
            model_head_spec = " "

        input_output_len = []

        if hasattr(model, "unroll_len"):
            input_output_len.append(f"U{model.unroll_len} ")

        input_output_len.append(f"[{history_len}←t→")

        if model is not None and isinstance(model, AbstractMS2SSAutoRegressive):
            input_output_len.append(f"1x")

        input_output_len.append(f"{horizon_len}")

        if model is not None and isinstance(model, MS2MS2SSArTemporalMixturePME):
            if hasattr(model, "enable_compounded_prediction_deploy_loss") and model.enable_compounded_prediction_deploy_loss:
                input_output_len.append(f"→1x{model.ar_horizon_len}")
            else:
                input_output_len.append(f"→1")

        input_output_len.append(f"]")
        input_output_len = "".join(input_output_len)

        model_spec = (
            f"E{cfg.ms_model.ensemble_size} x "
            f"En(L{cfg.ms_model.num_layers} x H{cfg.ms_model.hid_size}){model_head_spec}"
            f"{input_output_len}"
        )

    else:
        # Single-step model
        model_spec = (
            f"E{cfg.ss_model.ensemble_size} x "
            f"L{cfg.ss_model.num_layers} x H{cfg.ss_model.hid_size} "
            f"[{history_len}←t→{horizon_len}]"
        )

    model_str = ""
    if model is not None:

        obs_str = "MS-Idim"
        if history_len == 1:
            obs_str = "Id"
        if cfg.ms_model.get("receive_sequence_batch", False):
            model_str += f"  Input E x S x B x {obs_str}\n"
        else:
            model_str += f"  Input E x B x {obs_str}\n"

        if isinstance(
            model,
            (AutoRegressiveSequenceIterator, CompoundedPredictionMultiStepIterator),
        ):
            # Read the AR flags from the MODEL object (not the hydra config): MTM-Pro consolidated
            # its ``ar_enabled`` / ``ar_train_horizon_unroll`` under the ``compounded_prediction_deploy_loss``
            # group (RLRP-708 Task 2), so the top-level config keys no longer exist for that family.
            # The iterator setup always sets these attributes on the model for every AR family.
            if model.ar_enabled is True:
                model_str += f"  Auto-regressive compounded pred: enabled"
                if isinstance(model, CompoundedPredictionMultiStepIterator):
                    model_str += (
                        f", batch horizon unroll train: {model.ar_train_horizon_unroll}"
                    )
                elif isinstance(model, AutoRegressiveSequenceIterator):
                    model_str += (
                        f", sequence batch unroll train: {model.receive_sequence_batch}"
                    )
                model_str += "\n"

        if is_model_ensemble(model):
            model_str += f"  Propagation method: {model.propagation_method}\n"

        if (
            isinstance(model, AbstractTemporalyWeightedMultiStepMLP)
            and horizon_len > 1
            and is_cfg_key_exist(cfg.ms_model, "temporal_weights")
        ):
            model_str += (
                f"  MS pred temporal weight(s): " f"{cfg.ms_model.temporal_weights}\n"
            )

        if (
            hasattr(model, "dual_head_shared_input_layer")
            and model.dual_head_shared_input_layer is True
        ):
            model_str += f"  Dual head shared input layer: {model.dual_head_shared_input_layer}\n"

        if (
            hasattr(model, "model_use_double_precision")
            and model.model_use_double_precision is True
        ):
            model_str += (
                f"  Model use double precision: {model.model_use_double_precision}\n"
            )

        if (
            hasattr(model, "double_backward_pass")
            and model.double_backward_pass is True
        ):
            model_str += f"  Double backward pass: {model.double_backward_pass}\n"

        if hasattr(model, "MC_KL_sample_size"):
            model_str += f"  MC-KL sample size: {model.MC_KL_sample_size}\n"

        if hasattr(model, "dropout") and model.dropout > 0.0:
            model_str += f"  Dropout: {model.dropout}\n"

        if hasattr(model, "ms_head_dropout") and model.ms_head_dropout > 0:
            model_str += f"  MS head dropout: {model.ms_head_dropout}\n"

        if (
            is_cfg_key_exist(cfg, "one_dim_transition_model.normalize")
            and cfg.one_dim_transition_model.normalize == False
        ):
            model_str += f"  Normalizer: disabled\n"
        elif (
            is_cfg_key_exist(cfg, "one_dim_transition_model.normalizer_type")
            and is_cfg_key_exist(cfg, "one_dim_transition_model.normalize")
            and cfg.one_dim_transition_model.normalize == True
        ):
            model_str += (
                f"  Normalizer: {cfg.one_dim_transition_model.normalizer_type}\n"
            )

        if hasattr(model, "_forecast_teacher_forcing_scheduler"):

            _ms_tf_scheduler = model._forecast_teacher_forcing_scheduler
            if (
                _ms_tf_scheduler.method not in ["always_off", "always_on"]
                and _ms_tf_scheduler._decay_stop > 0
            ):
                model_str += (
                    f"  MS teacher forcing decay: "
                    f"start={_ms_tf_scheduler._decay_start} "
                    f"stop={_ms_tf_scheduler._decay_stop} "
                    f"method={_ms_tf_scheduler.method}\n"
                )
            else:
                model_str += f"  MS teacher forcing decay: method={_ms_tf_scheduler.method}\n"

        if hasattr(model, "enable_compounded_prediction_deploy_loss") and model.enable_compounded_prediction_deploy_loss:
            if hasattr(model, "_teacher_forcing_scheduler"):

                _cp_tf_scheduler = model._teacher_forcing_scheduler
                if (
                        _cp_tf_scheduler is not None
                        and _cp_tf_scheduler.method not in ["always_off", "always_on"]
                        and _cp_tf_scheduler._decay_stop > 0
                ):
                    model_str += (
                            f"  CP teacher forcing decay: "
                            f"start={_cp_tf_scheduler._decay_start} "
                            f"stop={_cp_tf_scheduler._decay_stop} "
                            f"method={_cp_tf_scheduler.method}\n"
                    )
                else:
                    model_str += f"  CP teacher forcing decay: method={_cp_tf_scheduler.method if _cp_tf_scheduler is not None else 'always_off'}\n"

            if hasattr(model, "_temporal_weight_scheduler"):
                if model.ar_temporal_weights_ramp_stop == 0:
                    model_str += (
                        f"  CP temporal weights: "
                        f"{model.ar_temporal_weights} (scheduling disabled)"
                        f"\n"
                    )
                else:
                    model_str += (
                        f"  CP temporal weights: "
                        f"start={model.ar_temporal_weights_start} "
                        f"target={model.ar_temporal_weights} "
                        f"warmup={model.ar_temporal_weights_warmup} "
                        f"ramp_stop={model.ar_temporal_weights_ramp_stop} "
                        f"\n"
                    )

            if (
                hasattr(model, "_unroll_len_sampler")
                and model.ar_unrol_len_probablity_decay_stop > 0
            ):
                model_str += (
                    f"  CP unroll len probablity decay: "
                    f"start={model.ar_unrol_len_probablity_decay_start} "
                    f"stop={model.ar_unrol_len_probablity_decay_stop} "
                    f"method={model.ar_unrol_len_decay_method} "
                    f"start_horizon_len={model.ar_unrol_len_start_horizon_len}\n"
                )

        # RLRP-736: report the model-internal orientation representation, the feature-geometry
        # loss objective, and the tangent-NLL right-Jacobian flag. Fetched from the MODEL object
        # (not the hydra config) so the panel reflects what the model was actually constructed with
        # (auto-inferred slots / gating can differ from the raw config).
        if hasattr(model, "_orientation_rep") and getattr(
            model, "_internal_orientation_rep_active", False
        ):
            orientation_rep = model._orientation_rep
            orientation_rep_str = getattr(
                orientation_rep, "value", str(orientation_rep)
            )
            model_str += f"  Internal orientation rep: {orientation_rep_str}"
            if getattr(model, "_tangent_nll_right_jac", False):
                model_str += f" (active; tangent-NLL right-Jacobian: True"
            model_str += "\n"

        if (
            hasattr(model, "_feature_geometry_loss_objective")
            and getattr(model, "_feature_geometry_loss_weight", 0.0) != 0.0
            and model._feature_geometry_loss_objective is not None
        ):
            model_str += (
                f"  Feature geometry loss: "
                f"objective={model._feature_geometry_loss_objective} "
                f"weight={model._feature_geometry_loss_weight}\n"
            )

        if isinstance(model, MS2MS2SSArTemporalMixturePME):
            model_str += (
                f"  Mixer kind: "
                f"{model.temporal_mixture_weights_kind}"
                f"\n"
            )

        # RLRP-768/RLRP-773: ``layer_bloc`` defaults to ``None`` (no per-region override), and
        # ``OmegaConf.to_yaml(None)`` raises ``ValueError: Invalid input``. Only dump the raw
        # selector when it is an actual dict/DictConfig; otherwise skip (legacy panel behaviour).
        _layer_bloc = getattr(model, "layer_bloc", None)
        if _layer_bloc is not None:
            model_str += (
                f"{omegaconf.OmegaConf.to_yaml(_layer_bloc)}"
                f"\n"
            )

    # RLRP-727: the optimized-dataset / exploration-policy block is UDER-specific; gate on the
    # `UDER.loop_kind` selector (replaces the legacy `replay_buffer_exploration_policy_enable`).
    if omegaconf.OmegaConf.select(cfg, "UDER.loop_kind", default=None) == "uder":
        replay_buffer_exploration_policy_str = (
            f"Optimized dataset:\n"
            f"  Size: {cfg_uder.replay_buffer_size.init_value} → "
            f"{cfg_uder.replay_buffer_size.floor} → {cfg_uder.replay_buffer_size.out_value}\n"
            f"  Batch: {cfg_uder.batch_size.init_value} → "
            f"{cfg_uder.batch_size.limit}\n"
            f"Replay buffer exploration policy:\n"
            f"  Sample score evaluation: {sample_score_evaluation_method}\n"
        )
    else:
        replay_buffer_exploration_policy_str = (
            # f"Batch: {cfg_uder.batch_size.init_value}\n"
            f"  Batch: {cfg_uder.batch_size.init_value} → "
            f"{cfg_uder.batch_size.limit}\n"
        )

    if cfg_source_dataset and cfg_source_dataset.get("original_dataset_size", False):
        _dataset_size = cfg_source_dataset.original_dataset_size
        dataset_str = (
            f"Source dataset: original {_dataset_size} → "
            f"train/val {int(_dataset_size * (1 - cfg_source_dataset.val_ratio))}"
            f"/{int(_dataset_size * cfg_source_dataset.val_ratio)}\n"
        )
    else:
        dataset_str = ""

    if is_cfg_key_exist(cfg, "environment.ood_test_global_measurement_noise"):
        noise_str = f"  Global noise: interval={cfg.environment.ood_test_global_measurement_noise.interval} magnitude={cfg.environment.ood_test_global_measurement_noise.magnitude}\n"
    else:
        noise_str = ""

    return (
        f"{noise_str}"
        f"{dataset_str}"
        f"{replay_buffer_exploration_policy_str}"
        f"Model {model_spec}\n"
        f"{model_str}"
    )
