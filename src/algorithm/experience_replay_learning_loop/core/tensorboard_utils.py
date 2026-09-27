# coding=utf-8
from typing import Dict, List, Optional, Tuple

from omegaconf import DictConfig

import mbrl.models
import numpy as np
from matplotlib import pyplot as plt

from tools.f110_gym_env_tools.plot_tensorboard import OnlineTensorboardWritter


def _collect_gradient_information(model: mbrl.models.Model) -> List[Dict]:
    epoch_param_accumulator = []
    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name:
            try:
                param_grad = param.grad.clone().cpu()
                epoch_param_accumulator.append(
                    {
                        "layer_name": name,
                        "grad_min": param_grad.min(),
                        "grad_mean": param_grad.abs().mean(),
                        "grad_max": param_grad.max(),
                    }
                )
            except AttributeError as e:
                if str(e) == "'NoneType' object has no attribute 'clone'":
                    pass
                else:
                    raise e
    return epoch_param_accumulator


def _epoch_alpha(gamma: float, acc_len: int, epoch: int):
    acc_len += 1
    return ((1.0 - gamma) ** (acc_len - epoch)) / acc_len


def plot_grad_flow(
    param_accumulator_: List[Tuple[int, List[Dict]]],
    show_min_max: bool,
    alpha,
    min_max_alpha=0.1,
    fetch_name: Optional[str] = None,
):
    acc_len = len(param_accumulator_)
    fig, ax = plt.subplots(figsize=(16, 12))
    for idx, (epoch, epoch_param) in enumerate(param_accumulator_):
        layers_name = []
        epoch_min_grads = []
        epoch_mean_grads = []
        epoch_max_grads = []

        # Example of layer name list
        # >>> epoch_param = [
        # >>>   'model.hidden_layers.0.0.weight', 'model.hidden_layers.1.0.weight',
        # >>>   'model.logvar_layer.0.weight', 'model.logvar_min', 'model.logvar_max',
        # >>>   'model.mean_and_logvar.weight', 'model.mean_layer.0.weight', ]

        # Sort such that ordering be as follows:
        # - hiden layer(s),
        # - mean_and_logvar layer,
        # - mean layer(s),
        # - logvar layer(s),
        # - logvar_[min|max]
        def check_name(x: Tuple[int, Dict], check: str) -> bool:
            return check in x["layer_name"]

        epoch_param.sort(
            key=lambda x: 100
            if (check_name(x, "logvar_min") or check_name(x, "logvar_max"))
            else (
                10
                if check_name(x, "mean_and_logvar")
                else (30 if check_name(x, "logvar") else (20 if check_name(x, "mean") else 0))
            )
        )

        for each in epoch_param:
            name = each["layer_name"]
            if fetch_name is None or fetch_name in name:
                layers_name.append(name)

                epoch_mean_grads.append(each["grad_mean"])
                if show_min_max:
                    epoch_min_grads.append(each["grad_min"])
                    epoch_max_grads.append(each["grad_max"])

        layer_space = np.arange(len(layers_name))
        ax.plot(layer_space, epoch_mean_grads, alpha=alpha, color="b")
        fill_cfg = {"color": "b", "alpha": _epoch_alpha(min_max_alpha, acc_len, epoch)}
        if show_min_max:
            ax.fill_between(layer_space, epoch_mean_grads, epoch_max_grads, **fill_cfg)
            ax.fill_between(layer_space, epoch_min_grads, epoch_mean_grads, **fill_cfg)

    ax.hlines(0, 0, len(layers_name) + 1, linewidth=1, color="k")
    ax.set_xticks(range(0, len(layers_name), 1), layers_name, rotation="vertical")
    ax.set_xlim(xmin=0, xmax=len(layers_name) - 1)
    ax.set_xlabel("Layers")
    ax.set_ylabel("average gradient")
    show_min_max_str = " with min max" if show_min_max else ""
    fetch_name_ = f" ({fetch_name})" if fetch_name else ""
    ax.set_title(f"Gradient flow{show_min_max_str}{fetch_name_}")
    ax.grid(True)
    fig.tight_layout(pad=2)

    return fig


def tensorboard_prediction_error(
    cfg: DictConfig,
    cfg_key_algo: str,
    pred_mae_score: float,
    pred_l2_norm_score: float,
    tensorboard_writer: OnlineTensorboardWritter,
    rollout_horizon: Optional[int] = None,
):
    # RLRP-723: in the compounded case the prediction MAE/L2 scalars are
    # cumulative quantities (sum over the rollout horizon) whose magnitude grows
    # with the trajectory length, so two runs are only directly comparable when
    # they share the same horizon. We therefore log:
    #   - ``Cum-MAE`` / ``Cum-L2 norm``   : the cumulative (length-dependent) score;
    #   - ``Avg-MAE`` / ``Avg-L2 norm``: the cumulative score normalized by the
    #     rollout horizon (number of scored steps), a trajectory-length
    #     normalized metric that is comparable across heterogeneous horizons.
    # The averaged variant is only emitted when ``rollout_horizon`` is provided.
    compounded = cfg.deploy.target_experiment.compounded_predictions_score

    pred_mae_type = "Compounding prediction Cum-MAE" if compounded else "Prediction Cum-MAE"
    tensorboard_writer.add_scalar_per_epoch_monitoring(
        tag=f"[{cfg_key_algo}] Prediction/{pred_mae_type}",
        value=pred_mae_score,
    )

    pred_l2_norm_type = (
        "Compounding prediction Cum-L2 norm" if compounded else "Prediction Cum-L2 norm"
    )
    tensorboard_writer.add_scalar_per_epoch_monitoring(
        tag=f"[{cfg_key_algo}] Prediction/{pred_l2_norm_type}",
        value=pred_l2_norm_score,
    )

    if compounded and rollout_horizon is not None and int(rollout_horizon) > 0:
        horizon = int(rollout_horizon)
        tensorboard_writer.add_scalar_per_epoch_monitoring(
            tag=f"[{cfg_key_algo}] Prediction/Compounding prediction Avg-MAE",
            value=pred_mae_score / horizon,
        )
        tensorboard_writer.add_scalar_per_epoch_monitoring(
            tag=f"[{cfg_key_algo}] Prediction/Compounding prediction Avg-L2 norm",
            value=pred_l2_norm_score / horizon,
        )
