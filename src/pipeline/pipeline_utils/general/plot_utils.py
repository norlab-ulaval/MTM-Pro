# coding=utf-8
import warnings
from typing import Optional

import numpy as np
import omegaconf
from matplotlib.ticker import MaxNLocator
from mbrl.models import Model
from numpy import ndarray
from omegaconf import DictConfig

import trajectory_container_tools as tct
from algorithm.experience_replay_learning_loop.uder_plot_utils import (
    plot_extra_info_str,
)
from math_gymnasium.tools.plot_3d_utils import (
    LEGEND_BBOX_TO_ANCHOR,
    _setup_1d_axis_prediction_subplot,
    three_dimension_prediction_plot,
)
from tools.f110_gym_env_tools.plot_tensorboard import OnlineTensorboardWritter
from tools.general_utils import sanitize_dirname
from tools.hydra_apps_tools.hydra_utils import get_hydra_experiment_id
from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist
from tools.mbrl_lib_tools.models.utils import (
    is_model_ensemble,
    is_probabilistic_model,
)
from tools.plot_tools.plot_management import (
    manage_matplotlib_backend,
    manage_matplotlib_warnings,
    show_and_save_plot_helper,
)
from trajectory_container_tools.dataclasses import TestMotionTrajectoryDataclass

# RLRP-741: 3D world-coordinate axis order used by `three_dimension_prediction_plot`
# (mirrors `math_gymnasium.tools.plot_3d_utils.DIM_X/DIM_Y/DIM_Z`).
_COORD_AXES = ("x", "y", "z")


def _aggregate_coordinate_uncertainty(
    feature_std: ndarray,
    obs_dim_names,
) -> Optional[ndarray]:
    """Aggregate per-feature obs-space std into a per-coordinate (x/y/z) std.

    Permanent pipeline plotting helper. Introduced by the RLRP-741 arbitrary
    feature-dim rollout plot `.junie` plan
    (`feat_arbitrary_feature_dim_rollout_plot_plan_RLRP-741_20260716.md`).

    For each world-coordinate axis ``c in {x, y, z}`` the aggregated std is::

        coord_std[c] = mean( std[linear_vels.c],
                             std[angular_vels.c],
                             mean(std[attitude.*]) )

    i.e. the mean of the matching linear-velocity std, the matching
    angular-velocity std, and the mean std over the attitude components. This
    replaces the legacy behaviour where the x/y/z subplots indexed the raw
    obs-space std at columns 0/1/2 (``linear_vels.{x,y,z}`` only), which was a
    misleading uncertainty band for a *world-pose* mean/target.

    :param feature_std: obs-space per-feature std of shape ``(T, obs_dim)``.
    :param obs_dim_names: ordered observation-dimension names
        (``environment.obs_dims``) used to resolve the aggregation columns.
    :return: ``(T, 3)`` aggregated std ordered ``[x, y, z]``, or ``None`` when
        the required names are not all present (e.g. non robotic-3D env), in
        which case the caller falls back to the legacy raw std passthrough.
    """
    if obs_dim_names is None:
        return None
    names = list(obs_dim_names)
    attitude_idx = [i for i, n in enumerate(names) if str(n).startswith("attitude.")]
    if not attitude_idx:
        return None

    feature_std = np.asarray(feature_std)
    attitude_mean_std = feature_std[..., attitude_idx].mean(axis=-1)

    coord_cols = []
    for coord in _COORD_AXES:
        lin_name = f"linear_vels.{coord}"
        ang_name = f"angular_vels.{coord}"
        if lin_name not in names or ang_name not in names:
            return None
        lin_std = feature_std[..., names.index(lin_name)]
        ang_std = feature_std[..., names.index(ang_name)]
        coord_cols.append((lin_std + ang_std + attitude_mean_std) / 3.0)

    return np.stack(coord_cols, axis=-1)


def _add_ground_truth_warmup_shading(
    cfg: DictConfig,
    fig,
    axes,
    time_space: ndarray,
    compounded_predictions_score: bool,
) -> bool:
    """Overlay the ``Ground truth feed warm-up`` shaded ``axvspan`` on ``axes``.

    Permanent pipeline plotting helper. Introduced by the RLRP-741 arbitrary
    feature-dim rollout plot `.junie` plan
    (`feat_arbitrary_feature_dim_rollout_plot_plan_RLRP-741_20260716.md`).

    Factored out of ``_plot_3d_rollout`` so the shading (and the subsequent
    legend refresh) can be applied identically to the always-present x/y/z
    coordinate subplots *and* to the arbitrary feature-dim subplots of the
    dedicated second figure. The label is only attached to the first axis so the
    legend shows a single ``Ground truth feed warm-up`` entry.

    :return: ``True`` when the shading was applied, ``False`` otherwise (i.e. not
        a compounded-prediction rollout, or the warm-up-steps key is absent).
    """
    if not (
        compounded_predictions_score
        and is_cfg_key_exist(
            cfg, "deploy.target_experiment.ground_truth_feed_warmup_steps"
        )
    ):
        return False

    gt_warmup_len = cfg.deploy.target_experiment.ground_truth_feed_warmup_steps
    show_ground_truth_warmup_label_once = "Ground truth feed warm-up"
    for each_ax in axes:
        each_ax.axvspan(
            xmin=time_space[0],
            xmax=(
                (
                    (time_space[-1] - time_space[0])
                    * (gt_warmup_len / len(time_space))
                )
                + time_space[0]
            ),
            linestyle="--",
            linewidth=2,
            color="lightgray",
            alpha=0.5,
            zorder=0,  # Put in back
            label=show_ground_truth_warmup_label_once,
        )
        show_ground_truth_warmup_label_once = ""

    # Update legend i.e. rerun last step of three_dimension_prediction_plot() fct
    axes[0].legend(
        loc="upper right",
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        bbox_transform=fig.transFigure,
        numpoints=3,
        markerscale=11.0,
    )
    return True


def _plot_arbitrary_feature_dim_predictions(
    cfg: DictConfig,
    fig,
    time_space: ndarray,
    observations,
    pred_feature_mean: ndarray,
    pred_feature_ale_epi_std: ndarray,
    pred_feature_epi_std: ndarray,
    selected_dim_names,
    obs_dim_names,
    subplot_1d_interval: Optional[slice],
    show_ale_uncertainty: bool,
    ale_uncertainty_scaling: float,
    show_epi_uncertainty: bool,
    epi_uncertainty_scaling: float,
    replace_axes=None,
) -> list:
    """Draw the arbitrary feature-dim 1D prediction subplots into ``fig``.

    Permanent pipeline plotting helper. Introduced by the RLRP-741 arbitrary
    feature-dim rollout plot `.junie` plan
    (`feat_arbitrary_feature_dim_rollout_plot_plan_RLRP-741_20260716.md`).

    Adds one 1D subplot per obs dimension named in
    ``cfg.pipeline.plot.show_prediction_feature_dims``, resolved by name against
    ``environment.obs_dims``. Each new subplot plots ``pred_feature_mean[:, idx]``
    against ``observations[:, idx]`` with the per-feature aleatoric/epistemic
    bands, fully obs-space consistent, reusing (not re-implementing)
    ``_setup_1d_axis_prediction_subplot`` so the style matches the x/y/z
    coordinate subplots exactly.

    The subplots occupy the **right-column band** of the figure. When
    ``replace_axes`` is provided (the dedicated second figure use case), those
    placeholder x/y/z axes define the band and are removed first, so the new
    subplots inherit the *identical width and spacing* as the original x/y/z
    coordinate subplots while the left ``ax_3d`` keeps its original aspect ratio.
    When ``replace_axes`` is ``None`` a default right band is used.

    Requested names that are absent from ``environment.obs_dims`` (e.g.
    ``angular_vels.*`` when angular velocities are disabled) are skipped with a
    ``warnings.warn`` instead of raising, so the plot still renders for the
    remaining available dims.

    :return: the list of newly created ``matplotlib`` axes (``[]`` when the
        selection is empty or resolves to no available dims).
    """
    obs_dim_names = list(obs_dim_names) if obs_dim_names is not None else []

    # Resolve each requested name against `environment.obs_dims`. A name that is
    # absent (e.g. `angular_vels.*` requested while angular velocities are
    # disabled in `environment.obs_dims`) is SKIPPED with a warning rather than
    # aborting the run, so the plot still renders for the available dims.
    selected_indices = []
    resolved_dim_names = []
    missing_dim_names = []
    for name in selected_dim_names:
        try:
            selected_indices.append(obs_dim_names.index(name))
        except ValueError:
            missing_dim_names.append(name)
            continue
        resolved_dim_names.append(name)
    if missing_dim_names:
        warnings.warn(
            f"Skipping unknown feature-dim name(s) {missing_dim_names} in "
            f"`pipeline.plot.show_prediction_feature_dims`. Valid names "
            f"(from `environment.obs_dims`): {obs_dim_names}",
            stacklevel=2,
        )
    selected_dim_names = resolved_dim_names

    k = len(selected_indices)
    if k == 0:
        return []

    # Reverse the internal order within each consecutive same-group run so that,
    # e.g., `linear_vels.{x,y,z}` is displayed top-to-bottom as z, y, x, matching
    # the 3D coordinate plot ordering.
    def _group_prefix(name):
        name_str = str(name)
        return name_str.rpartition(".")[0] if "." in name_str else name_str

    selected_dim_names = list(selected_dim_names)
    reordered_positions = []
    run_start = 0
    while run_start < len(selected_dim_names):
        run_end = run_start
        while (
            run_end + 1 < len(selected_dim_names)
            and _group_prefix(selected_dim_names[run_end + 1])
            == _group_prefix(selected_dim_names[run_start])
        ):
            run_end += 1
        reordered_positions.extend(range(run_end, run_start - 1, -1))
        run_start = run_end + 1
    selected_indices = [selected_indices[pos] for pos in reordered_positions]
    selected_dim_names = [selected_dim_names[pos] for pos in reordered_positions]

    observations = np.asarray(
        observations.detach().cpu().numpy()
        if hasattr(observations, "detach")
        else observations
    )
    pred_feature_mean = np.asarray(pred_feature_mean)
    pred_feature_ale_epi_std = np.asarray(pred_feature_ale_epi_std)
    pred_feature_epi_std = np.asarray(pred_feature_epi_std)

    if subplot_1d_interval is None:
        subplot_1d_interval = slice(0, len(time_space) - 1)

    # .... Resolve the right-column band .........................................................
    # Reuse the exact placement of the x/y/z coordinate subplots so the new
    # subplots share their width and spacing, and the left `ax_3d` is untouched.
    if replace_axes:
        positions = [each_ax.get_position() for each_ax in replace_axes]
        band_left = min(pos.x0 for pos in positions)
        band_right = max(pos.x1 for pos in positions)
        band_top = max(pos.y1 for pos in positions)
        band_bottom = min(pos.y0 for pos in positions)
        for each_ax in replace_axes:
            each_ax.remove()
    else:
        band_left, band_right, band_top, band_bottom = 0.40, 0.95, 0.92, 0.08

    new_gridspec = fig.add_gridspec(
        k,
        1,
        left=band_left,
        right=band_right,
        top=band_top,
        bottom=band_bottom,
        hspace=0.05,
    )

    # Split each selected name (e.g. `angular_vels.x`) into a group prefix
    # (`angular_vels`) and a per-axis sub-label (`x`). Names without a `.` keep
    # the full name as both group and sub-label.
    group_prefixes = []
    sub_labels = []
    for name in selected_dim_names:
        name_str = str(name)
        if "." in name_str:
            prefix, _, suffix = name_str.rpartition(".")
        else:
            prefix, suffix = name_str, name_str
        group_prefixes.append(prefix)
        sub_labels.append(suffix)

    new_axes = []
    shared_axis = None
    for row, dim_idx in enumerate(selected_indices):
        axis = fig.add_subplot(new_gridspec[row, 0], sharex=shared_axis)
        if shared_axis is None:
            shared_axis = axis
        # Mirror the exact next-step slicing used by `three_dimension_prediction_plot`
        # before delegating to the shared subplot helper.
        _setup_1d_axis_prediction_subplot(
            axis,
            cfg,
            time_space=time_space[1:],
            state_space_3d_target=observations[1:, ...],
            pred_mean_3d=pred_feature_mean[:-1, ...],
            pred_std_3d=pred_feature_ale_epi_std[:-1, ...],
            pred_epi_std_3d=pred_feature_epi_std[:-1, ...],
            subplot_1d_interval=subplot_1d_interval,
            selected_dimension=dim_idx,
            show_ale_uncertainty=show_ale_uncertainty,
            ale_uncertainty_scaling=ale_uncertainty_scaling,
            show_epi_uncertainty=show_epi_uncertainty,
            epi_uncertainty_scaling=epi_uncertainty_scaling,
            observations_fmt="--",
        )
        # Only show the per-axis sub-label (e.g. `x`) on each subplot; the shared
        # group label (e.g. `angular_vels`) is drawn once per group below.
        axis.set_ylabel(str(sub_labels[row]), fontsize=14, fontweight="bold")
        # Show at least three y ticks (top / middle / bottom) per subplot and a
        # background grid for readability.
        axis.yaxis.set_major_locator(MaxNLocator(nbins=2))
        axis.grid(True, alpha=0.3)
        # All subplots share the x (time) axis: only the bottom one keeps its
        # tick labels and the `time` axis label.
        if row != len(selected_indices) - 1:
            axis.tick_params(axis="x", labelbottom=False)
        new_axes.append(axis)

    new_axes[-1].set_xlabel("time", fontsize=16, fontweight="bold")

    # With the minimal inter-subplot spacing the top/bottom y-tick numbers of a
    # subplot would collide with the bottom/top numbers of the adjacent subplot
    # (each edge label is centered on the axis edge and spills into the
    # neighbour). Align the extreme tick labels *inward* (top label anchored at
    # its top, bottom label anchored at its bottom) so both are drawn inside the
    # subplot bounds and never overlap the neighbouring subplot. A `draw` pass is
    # required so the tick labels exist and are positioned before re-aligning.
    fig.canvas.draw()
    for axis in new_axes:
        tick_labels = axis.get_yticklabels()
        if not tick_labels:
            continue
        tick_labels[0].set_verticalalignment("bottom")
        tick_labels[-1].set_verticalalignment("top")

    # Draw a single shared group label (e.g. `angular_vels`) to the left of each
    # run of consecutive subplots sharing the same prefix.
    row = 0
    while row < len(group_prefixes):
        start = row
        while (
            row + 1 < len(group_prefixes)
            and group_prefixes[row + 1] == group_prefixes[start]
        ):
            row += 1
        top_pos = new_axes[start].get_position()
        bottom_pos = new_axes[row].get_position()
        y_center = (top_pos.y1 + bottom_pos.y0) / 2.0
        fig.text(
            max(band_left - 0.03, 0.005),
            y_center,
            str(group_prefixes[start]),
            fontsize=16,
            fontweight="bold",
            rotation="vertical",
            va="center",
            ha="center",
        )
        row += 1

    return new_axes


def _build_arbitrary_feature_dim_figure(
    cfg: DictConfig,
    model: Model,
    target_env: TestMotionTrajectoryDataclass,
    time_space: ndarray,
    fig_title: str,
    state_space_label: str,
    pred_3d_mean_coord: ndarray,
    std_3d_for_plot: ndarray,
    epi_std_3d_for_plot: ndarray,
    pred_feature_mean: ndarray,
    pred_feature_ale_epi_std: ndarray,
    pred_feature_epi_std: ndarray,
    selected_dim_names,
    obs_dim_names,
    show_ale: bool,
    show_epi: bool,
    show_interval_subplot: Optional[slice],
    test_env_is_OoD: bool,
    compounded_predictions_score: bool,
):
    """Build the dedicated *second* rollout figure for arbitrary feature dims.

    Permanent pipeline plotting helper. Introduced by the RLRP-741 arbitrary
    feature-dim rollout plot `.junie` plan
    (`feat_arbitrary_feature_dim_rollout_plot_plan_RLRP-741_20260716.md`).

    Per the AI-operator decision, the arbitrary feature-dim predictions no longer
    share the main figure (which would shrink ``ax_3d`` and crowd the x/y/z
    subplots). Instead this builds a *separate* figure that mirrors the main
    layout — the 3D world-pose plot on the left and 1D prediction-vs-target
    subplots on the right — but the right column shows *only* the selected
    observation dimensions (no x/y/z). It reuses ``three_dimension_prediction_plot``
    to draw the left 3D plot (and to place the right-column x/y/z placeholders,
    which are then swapped for the arbitrary-dim subplots via
    ``_plot_arbitrary_feature_dim_predictions``), so ``ax_3d`` keeps its original
    aspect ratio and the new subplots inherit the exact width/spacing/style.

    :return: the built ``matplotlib`` figure, or ``None`` when the selection is
        empty.
    """
    if not selected_dim_names:
        return None

    feat_fig, ax_3d, ax_z, ax_x, ax_y = three_dimension_prediction_plot(
        cfg,
        time_space=time_space,
        state_space_3d_base=target_env.pose_gt,
        state_space_3d_target=target_env.pose,
        pred_mean_3d=pred_3d_mean_coord.copy(),
        pred_std_3d=std_3d_for_plot.copy(),
        pred_epi_std_3d=epi_std_3d_for_plot.copy(),
        title=f"{fig_title}\n[arbitrary feature-dim predictions]",
        state_space_label=state_space_label,
        show_ale_uncertainty=show_ale,
        ale_uncertainty_scaling=cfg.pipeline.plot.ale_uncertainty_scaling,
        show_epi_uncertainty=show_epi,
        epi_uncertainty_scaling=cfg.pipeline.plot.epi_uncertainty_scaling,
        show_explorable_space=not test_env_is_OoD,
        subplot_1d_interval=show_interval_subplot,
        figsize=cfg.pipeline.plot.figsize,
        figdpi=cfg.pipeline.plot.figdpi,
        extra_info_str=plot_extra_info_str(cfg, model),
        experiment_id=get_hydra_experiment_id(),
    )

    # Swap the x/y/z placeholder subplots for the arbitrary feature-dim subplots,
    # reusing their exact right-column band (width/spacing) and leaving `ax_3d`
    # untouched.
    new_axes = _plot_arbitrary_feature_dim_predictions(
        cfg,
        feat_fig,
        time_space,
        observations=target_env.observations,
        pred_feature_mean=pred_feature_mean,
        pred_feature_ale_epi_std=pred_feature_ale_epi_std,
        pred_feature_epi_std=pred_feature_epi_std,
        selected_dim_names=selected_dim_names,
        obs_dim_names=obs_dim_names,
        subplot_1d_interval=show_interval_subplot,
        show_ale_uncertainty=show_ale,
        ale_uncertainty_scaling=cfg.pipeline.plot.ale_uncertainty_scaling,
        show_epi_uncertainty=show_epi,
        epi_uncertainty_scaling=cfg.pipeline.plot.epi_uncertainty_scaling,
        replace_axes=[ax_z, ax_x, ax_y],
    )

    # Same ground-truth warm-up shaded overlay as the x/y/z coordinate subplots.
    shaded = _add_ground_truth_warmup_shading(
        cfg, feat_fig, new_axes, time_space, compounded_predictions_score
    )
    if not shaded:
        # Refresh the legend on the first new subplot (the placeholder axis that
        # originally hosted the legend was removed above).
        new_axes[0].legend(
            loc="upper right",
            bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
            bbox_transform=feat_fig.transFigure,
            numpoints=3,
            markerscale=11.0,
        )

    return feat_fig


def _log_and_save_deploy_figure(
    cfg: DictConfig,
    fig,
    fig_title: str,
    save_name: str,
    plot_save_dir: str,
    tensorboard_writer: Optional[OnlineTensorboardWritter],
    headless: bool,
) -> None:
    """Log a rollout figure to TensorBoard and save it to disk.

    Permanent pipeline plotting helper. Introduced by the RLRP-741 arbitrary
    feature-dim rollout plot `.junie` plan
    (`feat_arbitrary_feature_dim_rollout_plot_plan_RLRP-741_20260716.md`).

    Factored out of ``_plot_3d_rollout`` so both the main rollout figure and the
    dedicated arbitrary feature-dim figure share the same logging/saving path.
    """
    try:
        if tensorboard_writer is not None:
            tensorboard_writer.writer.add_figure(
                tag=f"Deployment roll-out/{fig_title}",
                figure=fig,
                close=False,
            )

        show_and_save_plot_helper(
            fig,
            plot_save_dir,
            save_name,
            headless,
            cfg.pipeline.plot.show_prediction_plot,
            cfg.pipeline.plot.save_dpi,
        )
    except (RuntimeError, ValueError, Exception) as e:
        # Capture LinAlgError (often wrapped by matplotlib/torch)
        print(f"Skipping figure logging due to: {e}")


def _plot_3d_rollout(
    cfg: DictConfig,
    model: Model,
    target_env: TestMotionTrajectoryDataclass,
    pred_3d_ale_epi_std: ndarray,
    pred_3d_epi_std: ndarray,
    pred_3d_mean_coord: ndarray,
    tensorboard_writer: Optional[OnlineTensorboardWritter],
    exp_dir_relative_path: str,
    state_space_label: str,
    show_interval_subplot: Optional[slice],
    probabilistic_rollout: bool,
    compounded_predictions_score: bool,
    test_env_is_OoD: bool,
    headless: bool,
    trajectory_subdir: Optional[str] = None,
    pred_feature_mean: Optional[ndarray] = None,
    epoch_rollout_label: Optional[str] = None,
) -> None:
    # RLRP-741: `pred_feature_mean` is the obs-space per-feature prediction mean
    # of shape `(T, obs_dim)` (previously not forwarded). It powers the optional
    # *dedicated second figure* of arbitrary feature-dim subplots (3D plot left,
    # selected obs-dim subplots right) that leaves the main figure untouched.
    # `pred_3d_ale_epi_std` / `pred_3d_epi_std` are the obs-space per-feature std
    # arrays `(T, obs_dim)`, reused both for the §11 aggregated x/y/z bands and
    # the per-feature subplots.
    test_env_type = "OoD test env" if test_env_is_OoD else "InD test env"

    if hasattr(model, "description"):
        model_name = model.description
    else:
        model_name = model.__class__.__name__

    if probabilistic_rollout:
        rollout_type_str = f"probabilistic rollout"
    else:
        rollout_type_str = f"deterministic rollout"

    # RLRP-727: annotate with the ERLL loop kind (replaces the legacy
    # `replay_buffer_exploration_policy_enable` boolean).
    loop_kind = omegaconf.OmegaConf.select(cfg, "UDER.loop_kind", default=None)
    loop_kind_str = f"loop_kind={loop_kind}, " if loop_kind is not None else ""

    # RLRP-773: on per-epoch-checkpoint rollouts, surface the checkpoint's real training epoch in
    # the title (e.g. "... Model Roll-Out (epoch 80/560)") so each rendered figure is identifiable.
    epoch_title_str = (
        f" (epoch {epoch_rollout_label})" if epoch_rollout_label else ""
    )
    fig_title = (
        f"{model_name} Model Roll-Out{epoch_title_str}\n"
        f"({test_env_type}, compounded_predictions_score={compounded_predictions_score}, "
        f"{loop_kind_str}{rollout_type_str})"
    )

    with warnings.catch_warnings():
        manage_matplotlib_warnings()
        manage_matplotlib_backend(cfg.pipeline.plot.show_prediction_plot, headless)

        time_space = target_env.timestamps
        if isinstance(time_space, tct.temporal.Timestamps):
            time_space = target_env.timestamps.stamps

        # RLRP-741: compute the uncertainty flags once and reuse them for both the
        # shared x/y/z subplots and the arbitrary feature-dim subplots.
        show_ale = is_probabilistic_model(model)
        show_epi = is_model_ensemble(model)

        # RLRP-741 (§11): resolve obs-dim names and aggregate the per-feature
        # obs-space std into per-coordinate (x/y/z) bands. Falls back to the legacy
        # raw obs-space std when the required names are absent (non robotic-3D env).
        obs_dim_names = omegaconf.OmegaConf.select(
            cfg, "environment.obs_dims", default=None
        )
        agg_ale_epi_std_3d = _aggregate_coordinate_uncertainty(
            pred_3d_ale_epi_std, obs_dim_names
        )
        agg_epi_std_3d = _aggregate_coordinate_uncertainty(
            pred_3d_epi_std, obs_dim_names
        )
        if agg_ale_epi_std_3d is not None and agg_epi_std_3d is not None:
            std_3d_for_plot = agg_ale_epi_std_3d
            epi_std_3d_for_plot = agg_epi_std_3d
        else:
            std_3d_for_plot = pred_3d_ale_epi_std
            epi_std_3d_for_plot = pred_3d_epi_std

        try:
            deploy_fig, ax_3d, ax_z, ax_x, ax_y = three_dimension_prediction_plot(
                cfg,
                time_space=time_space,
                state_space_3d_base=target_env.pose_gt,
                state_space_3d_target=target_env.pose,
                pred_mean_3d=pred_3d_mean_coord.copy(),
                pred_std_3d=std_3d_for_plot.copy(),
                pred_epi_std_3d=epi_std_3d_for_plot.copy(),
                title=fig_title,
                state_space_label=state_space_label,
                show_ale_uncertainty=show_ale,
                ale_uncertainty_scaling=cfg.pipeline.plot.ale_uncertainty_scaling,
                show_epi_uncertainty=show_epi,
                epi_uncertainty_scaling=cfg.pipeline.plot.epi_uncertainty_scaling,
                show_explorable_space=not test_env_is_OoD,
                subplot_1d_interval=show_interval_subplot,
                figsize=cfg.pipeline.plot.figsize,
                figdpi=cfg.pipeline.plot.figdpi,
                extra_info_str=plot_extra_info_str(cfg, model),
                experiment_id=get_hydra_experiment_id(),
            )

            # RLRP-741: the always-present x/y/z coordinate subplots keep the
            # original layout (3D plot left, x/y/z subplots right) untouched.
            _add_ground_truth_warmup_shading(
                cfg,
                deploy_fig,
                [ax_z, ax_x, ax_y],
                time_space,
                compounded_predictions_score,
            )
        except (RuntimeError, ValueError, Exception) as e:
            # Capture LinAlgError (often wrapped by matplotlib/torch)
            print(f"Skipping figure logging due to: {e}")
            return None

        # RLRP-741: precompute the shared save path + name pieces (keep the
        # `compounded_predictions_score` boolean intact for the second figure).
        compounded_suffix = (
            "_compounded_predictions_score" if compounded_predictions_score else ""
        )
        test_env_save_type = "ood_test_env" if test_env_is_OoD else "InD_test_env"
        plot_save_dir = exp_dir_relative_path
        if trajectory_subdir is not None:
            import os

            plot_save_dir = os.path.join(
                exp_dir_relative_path, "testtime_rollouts", trajectory_subdir
            )

        _log_and_save_deploy_figure(
            cfg,
            deploy_fig,
            fig_title,
            f"{model.__class__.__name__}_ensemble_model{compounded_suffix}"
            f"_{test_env_save_type}_{sanitize_dirname(rollout_type_str)}",
            plot_save_dir,
            tensorboard_writer,
            headless,
        )

        # RLRP-741 (§10): the arbitrary feature-dim figure is a *dedicated second
        # figure* (3D plot on the left, only the selected obs-dim subplots on the
        # right) so the main figure and its `ax_3d` aspect ratio stay untouched.
        # Built OUTSIDE the broad exception guard above so a genuine wiring/shape
        # bug surfaces instead of being silently swallowed.
        selected_feature_dims = list(
            cfg.pipeline.plot.get("show_prediction_feature_dims", []) or []
        )
        if selected_feature_dims and pred_feature_mean is not None:
            feature_fig = _build_arbitrary_feature_dim_figure(
                cfg,
                model,
                target_env,
                time_space,
                fig_title,
                state_space_label,
                pred_3d_mean_coord,
                std_3d_for_plot,
                epi_std_3d_for_plot,
                pred_feature_mean=pred_feature_mean,
                pred_feature_ale_epi_std=pred_3d_ale_epi_std,
                pred_feature_epi_std=pred_3d_epi_std,
                selected_dim_names=selected_feature_dims,
                obs_dim_names=obs_dim_names,
                show_ale=show_ale,
                show_epi=show_epi,
                show_interval_subplot=show_interval_subplot,
                test_env_is_OoD=test_env_is_OoD,
                compounded_predictions_score=compounded_predictions_score,
            )
            if feature_fig is not None:
                _log_and_save_deploy_figure(
                    cfg,
                    feature_fig,
                    f"{fig_title} [feature-dims]",
                    f"{model.__class__.__name__}_ensemble_model{compounded_suffix}"
                    f"_{test_env_save_type}_{sanitize_dirname(rollout_type_str)}"
                    f"_feature_dims",
                    plot_save_dir,
                    tensorboard_writer,
                    headless,
                )

    return None
