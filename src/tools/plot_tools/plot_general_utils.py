# coding=utf-8
from typing import List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

from tools.math_tools.ndarray_tools.ndarray_checker import (
    check_ndarray_list_composante_shapes_match,
)
from tools.plot_tools.style import AXIS_LABEL_STYLE


def arbitrary_dimension_array_plot(
    data: Union[np.ndarray, Tuple[np.ndarray, ...]],
    plot_label: Union[str, Tuple[str, ...]] = "",
    y_label: Tuple[str, ...] = ("y",),
    y_lim: Optional[List[Tuple[float, float]]] = None,
    title: Optional[str] = None,
    figsize: tuple = (20, 8),
    figdpi: int = 50,
    plot_style: Optional[dict] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    if plot_style is None:
        plot_style = {"linewidth": 0.85, "alpha": 0.7}
    legend_cfg = {
        "loc": "upper right",
    }

    if not isinstance(data, Tuple):
        data = [data]
        plot_label = [plot_label]
    else:
        assert isinstance(plot_label, Tuple), (
            f"Param `plot_label` argument need to be a sequence of "
            f"string if param `pred_std` is a sequence of ndarray"
        )
        data = list(data)
        plot_label = list(plot_label)
        check_ndarray_list_composante_shapes_match(data)

    if data[0].ndim > 1:
        pred_obs_dim = data[0].shape[-1]
        assert pred_obs_dim == len(y_label), (
            f"Param `data` arg and `y_labels` need to match ({data[0].shape[-1]} != "
            f"{len(y_label)=})."
        )
        legend_cfg.update({"bbox_to_anchor": (1.0, 1.2)})
    else:
        pred_obs_dim = 1
        assert pred_obs_dim == len(y_label), (
            f"Param `data` arg and `y_labels` need to match ({data[0].ndim} != "
            f"{len(y_label)=})."
        )
        legend_cfg.update({"bbox_to_anchor": (1.03, 1.2)})

    fig, ax = plt.subplots(
        pred_obs_dim, 1, figsize=figsize, dpi=figdpi
    )  # default dpi=100

    if y_lim is None:
        y_lim = [(None, None) for _ in range(3)]
    else:
        assert isinstance(y_lim, List)
        assert len(y_lim) == pred_obs_dim, f"{len(y_lim)=} != {pred_obs_dim=}"
        for each_lim in y_lim:
            assert isinstance(each_lim, tuple), f"{type(each_lim)=} != tuple"
            assert isinstance(each_lim[0], (int, float)) and isinstance(
                each_lim[1], (int, float)
            ), f"({type(each_lim[0])=},{type(each_lim[1])=}) != (float, float)"

    for idx in range(len(data)):
        if pred_obs_dim > 1:
            each_dim = None
            for each_dim in np.arange(pred_obs_dim):
                ax[each_dim].plot(
                    data[idx][..., each_dim], **plot_style, label=plot_label[idx]
                )
                plot_label[idx] = ""
                ax[each_dim].set_ylabel(y_label[each_dim], **AXIS_LABEL_STYLE)
                if each_dim != pred_obs_dim - 1:
                    ax[each_dim].set_xticks([])
                if y_lim[each_dim][0] is not None:
                    ax[each_dim].set_ylim(bottom=y_lim[each_dim][0])
                if y_lim[each_dim][1] is not None:
                    ax[each_dim].set_ylim(top=y_lim[each_dim][1])
            else:
                ax[each_dim or 0].set_xlabel("t", **AXIS_LABEL_STYLE)
                ax[0].legend(**legend_cfg)
        else:
            ax.plot(data[idx], **plot_style, label=plot_label[idx])
            ax.set_ylabel(y_label[0], **AXIS_LABEL_STYLE)
            if y_lim[0][0] is not None:
                ax.set_ylim(bottom=y_lim[0][0])
            if y_lim[0][1] is not None:
                ax.set_ylim(top=y_lim[0][1])
            ax.set_xlabel("t", **AXIS_LABEL_STYLE)
            ax.legend(**legend_cfg)

    if title is not None:
        fig.suptitle(title, size="large", weight="bold")

    if pred_obs_dim > 1:
        fig.tight_layout(pad=2)
        plt.subplots_adjust(hspace=0.1, top=0.925)
    return fig, ax
