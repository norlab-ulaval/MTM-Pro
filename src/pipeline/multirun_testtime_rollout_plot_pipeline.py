# coding=utf-8
import warnings
from copy import deepcopy
from typing import Any
from deprecated import deprecated

import numpy as np
import omegaconf

from algorithm.utils import seed_me


from pipeline.pipeline_utils.multirun_testtime_rollout_plot_pipeline_utils.generate_fct import (
    generate_arbitrary_len_general_plot,
    generate_drift_rate_plot,
    generate_drift_rate_plot_per_train_epoch,
    generate_final_general_plot,
    generate_per_category_plots,
    generate_per_trajectory_plots,
    generate_rollout_wall_clock_time_plot,
    generate_rollout_inference_cost_breakdown_plot,
    generate_rollout_inference_speed_plot,
    generate_training_wall_clock_time_plot,
    regenerate_rollouts,
    get_plot_render_type_cfg,
    is_latex_includegraphics_render_type,
)
from pipeline.pipeline_utils.multirun_testtime_rollout_plot_pipeline_utils.adverse_event_slice import (
    report_adverse_event_slice,
)
from pipeline.pipeline_utils.multirun_testtime_rollout_plot_pipeline_utils.main_utils import (
    _load_ground_truth_entries,
    category_breakdown,
    compute_per_group_metric,
    manage_torch_future_warnings,
    resolve_consistent_warmup_steps,
    trajectory_breakdown,
)
from pipeline.pipeline_utils.multirun_testtime_rollout_plot_pipeline_utils.show_options_utils import (
    assert_no_legacy_show_keys,
    resolve_breakdown_mode,
    resolve_selected_trajectories,
    resolve_warmup_mode,
)
from tools.hydra_apps_tools.hydra_utils import (
    fetch_project_root_path_via_hydra,
    get_hydra_experiment_cwd,
)
from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist

import matplotlib.pyplot as plt

# .... Matplotlib configuration ...................................................................
import matplotlib as mpl

# This set the default matplotlib style and figure background color
# Remark: Those default might be overriten by multirun_testtime_rollout_plot_pipeline.yaml style key
# (NICE TO HAVE) ToDo: RLRP-801 refactor: consolidate matplotlib related cfg pipeline file
plt.style.use("classic")
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["legend.markerscale"] = 5
mpl.rcParams["legend.numpoints"] = 3
mpl.rcParams["font.size"] = 18
mpl.rcParams["legend.loc"] = "best"
# mpl.rcParams['legend.loc'] = 'lower right'
# mpl.rcParams["axes.prop_cycle"] = plt.cycler(color=["b", "g", "r", "y"])


@deprecated(
    reason=(
        "With the many group metric and original/process variation, this is an error-prone"
        " approach that is very hard to debug"
    )
)
def _reset_cumulative_error_at_warmup(
    groups: dict[str, Any], warmup_steps: int | None
) -> None:
    """Reset the cumulative error to zero at the warm-up boundary (RLRP-736).

    In ``show.ground_truth_feed_warmup.mode: 'discard'`` the warm-up interval
    ``0 .. ground_truth_feed_warmup_steps`` is trimmed off every time-axis
    plot. For cumulative metrics (C-MAE) the error accumulated during the
    warm-up interval — where the model is fed ground truth and its error is
    artificially near-zero — would otherwise leak into the post-warm-up curve
    (the cumulative sum keeps that head-start). Zeroing the first
    ``warmup_steps`` timesteps of every recorded per-rollout error array
    (``mae`` / ``l2_norm``) makes the downstream cumulative sum restart from
    zero on the warm-up boundary ``t == warmup_steps``.

    The operation mutates the recorded ``_original_metrics`` entries in-place
    and is a no-op when ``warmup_steps`` is ``None`` or ``<= 0`` (a resolved
    ``0`` means "no warm-up", nothing to reset).

    :param groups: Per-group metric dict as returned by ``compute_per_group_metric``.
    :param warmup_steps: Auto-derived warm-up step count, or ``None``.
    """
    if not warmup_steps or warmup_steps <= 0:
        return
    for group in groups.values():
        for ogm in group.get("_original_metrics", []) or []:
            for _attr in ("mae", "l2_norm"):
                arr = getattr(ogm, _attr, None)
                if arr is None or not hasattr(arr, "shape") or arr.size == 0:
                    continue
                n = min(int(warmup_steps), int(arr.shape[0]))
                if n <= 0:
                    continue
                arr = np.array(arr, dtype=float, copy=True)
                arr[:n] = 0.0
                setattr(ogm, _attr, arr)


def execute(cfg: omegaconf.DictConfig, headless: bool = False) -> None:
    """Execute math toy environment Sim2Sim motion model learner feedback loop

    :param cfg: Hydra configuration file
    :param headless: turn environment rendering off (param for commandline flag)
    """
    exp_dir_relative_path = get_hydra_experiment_cwd(cfg)
    seed, torch_rng = seed_me(cfg, output_torch_rdn_generator=True)

    root_project_path = fetch_project_root_path_via_hydra(
        cfg, 1, "MTM-Pro"
    )
    # root_project_path = os.path.join(get_hydra_original_cwd(cfg), "..",)

    # .... `show.*` option resolution (Phase 2 — plan §4.8) ......................................
    # Reject any removed legacy keys (`show.category_breakdown`,
    # `show.general_plot_right_xlim`). The codebase is unpublished so migration
    # is a hard cutover (plan §3 R3 / §4.3).
    assert_no_legacy_show_keys(cfg.show)

    do_average_across_trajectories = cfg.show.get("average_across_trajectories", False)
    breakdown_mode = resolve_breakdown_mode(cfg.show)
    do_category_breakdown = breakdown_mode == "category"
    do_trajectory_breakdown = breakdown_mode == "trajectory"  # wired in a later phase
    general_length_spec = cfg.show.get("general_length", None)
    do_summary_table = cfg.show.get("summary_table", False)
    do_regenerate_rollouts = cfg.get("regenerate_rollouts", False)
    do_cleanup_legacy_rollouts = cfg.get("cleanup_legacy_rollouts", False)
    # .... Ground-truth-feed warm-up treatment (YouTrack RLRP-733) ...............................
    # ``show.ground_truth_feed_warmup.mode`` selects how the warm-up interval
    # (``0 .. ground_truth_feed_warmup_steps``) is treated on every time-axis
    # plot: ``None`` (no treatment, current behavior), ``'discard'`` (clamp the
    # left x-limit to the warm-up boundary), or ``'reference_line'`` (draw a
    # vertical marker at the boundary). The warm-up step count itself is always
    # auto-derived from the recorded rollouts (never from config).
    warmup_mode = resolve_warmup_mode(cfg.show)

    # ``show.selected_trajectories: [null | <list of trajectory_name>]`` (RLRP-841)
    # restricts the drift-rate plots to an explicitly named subset of the test
    # trajectories (verbatim simulator-config spelling). ``null`` = no filtering.
    # Membership is validated (fail fast) inside the drift-rate functions
    # against the ground-truth entries resolved from ``simulator_config``.
    selected_trajectories = resolve_selected_trajectories(cfg.show)

    with warnings.catch_warnings():
        manage_torch_future_warnings()
        # .... Regenerate rollouts (Phase 6.4) ........................................................
        if do_regenerate_rollouts:
            regenerate_rollouts(
                cfg, root_project_path, do_cleanup_legacy_rollouts, torch_rng
            )

        # .... Compute pre-group metrics ..............................................................
        # Compute per-group metrics for test-time rollout predictions based on the specified
        # averaging, category breakdown rules and provided configuration e.g., compounded
        # predictions vs non-compounded predictions, InD vs OoD environment, model type, model
        # h-param
        # When a warm-up treatment is active, additionally collect the
        # per-experiment warm-up step count so the fail-fast consistency guard
        # can run. When the feature is off, the historical single-dict return
        # and behavior are preserved untouched.
        warmup_steps: int | None = None
        if warmup_mode is not None:
            groups, warmup_steps_by_experiment_base = compute_per_group_metric(
                cfg,
                root_project_path,
                do_average_across_trajectories,
                do_category_breakdown,
                collect_ground_truth_feed_warmup_steps=True,
            )
            # Fail fast if the contributing experiments disagree; ``None`` when
            # nothing resolved (graceful skip handled downstream).
            warmup_steps = resolve_consistent_warmup_steps(
                warmup_steps_by_experiment_base
            )
            # RLRP-736: in ``discard`` mode the warm-up interval is trimmed off
            # every plot. Reset the cumulative error to zero at the warm-up
            # boundary so the artificially near-zero warm-up error (ground truth
            # is fed to the model there) does not leak into the post-warm-up
            # cumulative (C-MAE) curve.
            # (Priority) ToDo: deprecate whitening the warmup part of the rollout at this stage.
            #  Its hard to track, error prone and should be perform closer to where the values are used.
            # if warmup_mode == "discard":
            #     _reset_cumulative_error_at_warmup(groups, warmup_steps)
        else:
            groups = compute_per_group_metric(
                cfg,
                root_project_path,
                do_average_across_trajectories,
                do_category_breakdown,
            )

        # .... M2 adverse-event slice (RLRP-761 S6.2) .................................................
        # Score the error restricted to the rare high-dynamics windows of the
        # GROUND TRUTH (top-`top_fraction` of `‖Δ(vz, wx, wy)‖`). No existing
        # metric isolates them: a whole-trajectory MAE is dominated by the
        # nominal regime. Strict no-op when `show.adverse_event_slice` is
        # absent or `render: false`.
        report_adverse_event_slice(cfg, groups)

        # .... Category breakdown (Phase 6.3) .........................................................
        category_breakdown_data = category_breakdown(
            cfg, deepcopy(groups), do_average_across_trajectories, do_category_breakdown
        )

        # .... Trajectory breakdown (Phase 4 — new show.breakdown='trajectory') ......................
        trajectory_breakdown_data = trajectory_breakdown(
            cfg,
            deepcopy(groups),
            do_average_across_trajectories,
            do_trajectory_breakdown,
        )

    # .... Setup figure ...........................................................................
    if cfg.show.cumulative_metric:
        metric_label = f"Cumulative Mean Absolute Error"
        metric_acro_label = f"C-MAE"
    else:
        metric_label = "Mean Absolute Error"
        metric_acro_label = f"MAE"

    if cfg.show.variation_type == "group_mae_std":
        variation_type_label = f"{metric_acro_label} standard deviation"
    elif cfg.show.variation_type == "group_avg_uncertainty":
        variation_type_label = f"average aleatoric+epistemic uncertainty"
    elif cfg.show.variation_type == "group_avg_epi":
        variation_type_label = f"average epistemic uncertainty"
    elif cfg.show.variation_type == "group_avg_ale":
        variation_type_label = f"average aleatoric uncertainty"
    else:
        raise ValueError(f"Unsupported {cfg.show.variation_type=}")

    if cfg.show.compounded_predictions_score:
        compounded_pred_label = "Compounded-predictions rollout"
    else:
        compounded_pred_label = "Ground-truth rollout"

    if cfg.show.target_is_ood:
        if is_cfg_key_exist(cfg, "label_override.OoD"):
            target_type_label = cfg.label_override.OoD
        else:
            target_type_label = "Out-of-Distribution environment (OoD)"
    else:
        if is_cfg_key_exist(cfg, "label_override.InD"):
            target_type_label = cfg.label_override.InD
        else:
            target_type_label = "In-Distribution environment (InD)"

    if cfg.show.variation_scale != 1:
        variation_type_label = (
            f"{variation_type_label} (scaled {cfg.show.variation_scale}X)"
        )

    # Allow user to override the word "trials" in plot titles/legends via
    # ``label_override.trials`` (default: ``"trials"``). Mirrors the
    # ``label_override.InD`` / ``label_override.OoD`` mechanism above.
    if is_cfg_key_exist(cfg, "label_override.trials"):
        trials_label = cfg.label_override.trials
    else:
        trials_label = "trials"
    # ``show.y_axis_in_logscale`` (default: ``False``). When ``True``, the
    # General plot, per-category plots, and per-trajectory plots render the
    # y-axis in log scale (matplotlib ``ax.set_yscale('log')``).
    if is_cfg_key_exist(cfg, "show.y_axis_in_logscale"):
        y_axis_in_logscale = bool(cfg.show.y_axis_in_logscale)
    else:
        y_axis_in_logscale = False
    if is_cfg_key_exist(cfg, "show.x_axis_in_logscale"):
        x_axis_in_logscale = bool(cfg.show.x_axis_in_logscale)
    else:
        x_axis_in_logscale = False

    if cfg.show.variation_type == "group_mae_std":
        metric_and_variation_type = ""
    else:
        metric_and_variation_type = (
            "Variation type: {trials_label} {variation_type_label}. "
        )

    title = f"{cfg.env_name} test-time rollout {metric_label} ({metric_acro_label}).\n{metric_and_variation_type}{compounded_pred_label}. {target_type_label}."

    plot_linewidth = get_plot_render_type_cfg(cfg).style.plot_linewidth
    fill_between_alpha = get_plot_render_type_cfg(cfg).style.fill_between_alpha

    # .... Per-category plots (Phase 6.3 — Task 2) ................................................
    generate_per_category_plots(
        cfg,
        category_breakdown_data,
        compounded_pred_label,
        do_average_across_trajectories,
        do_category_breakdown,
        do_summary_table,
        exp_dir_relative_path,
        fill_between_alpha,
        plot_linewidth,
        metric_acro_label,
        metric_label,
        target_type_label,
        headless,
        trials_label=trials_label,
        y_axis_in_logscale=y_axis_in_logscale,
        warmup_mode=warmup_mode,
        warmup_steps=warmup_steps,
    )

    # .... Per-trajectory plots (Phase 4 — show.breakdown='trajectory') ...........................
    # Phase 6: load ground-truth ``TestTrajectoryEntry`` list so the 3D snapshot
    # inset on each per-trajectory plot renders real data. Only attempted when
    # ``do_trajectory_breakdown`` is active; any failure yields ``None`` and
    # the snapshot is silently skipped (plan R2 graceful-skip).
    _ground_truth_entries: Any = None
    if do_trajectory_breakdown:
        _ground_truth_entries = _load_ground_truth_entries(
            cfg, root_project_path, bool(cfg.show.target_is_ood)
        )
    generate_per_trajectory_plots(
        cfg,
        trajectory_breakdown_data,
        compounded_pred_label,
        do_trajectory_breakdown,
        exp_dir_relative_path,
        fill_between_alpha,
        plot_linewidth,
        metric_acro_label,
        metric_label,
        target_type_label,
        headless,
        ground_truth_entries=_ground_truth_entries,
        trials_label=trials_label,
        y_axis_in_logscale=y_axis_in_logscale,
        warmup_mode=warmup_mode,
        warmup_steps=warmup_steps,
    )

    # .... General plot (Phase 5 — `show.general_length`) ........................................
    generate_arbitrary_len_general_plot(
        cfg,
        deepcopy(groups),
        category_breakdown_data,
        general_length_spec,
        do_average_across_trajectories,
        do_category_breakdown,
        title,
        exp_dir_relative_path,
        fill_between_alpha,
        plot_linewidth,
        metric_acro_label,
        metric_label,
        headless,
        trials_label=trials_label,
        y_axis_in_logscale=y_axis_in_logscale,
        warmup_mode=warmup_mode,
        warmup_steps=warmup_steps,
    )

    # .... Final general plot (all-trajectory aggregation) ........................................
    if cfg.show.get("generate_final_general_plot", {}).get("render", True):
        generate_final_general_plot(
            cfg,
            deepcopy(groups),
            do_average_across_trajectories,
            title,
            exp_dir_relative_path,
            fill_between_alpha,
            plot_linewidth,
            metric_acro_label,
            metric_label,
            headless,
            trials_label=trials_label,
            y_axis_in_logscale=y_axis_in_logscale,
            sem_alpha=get_plot_render_type_cfg(cfg).style.get("sem_alpha", 0.35),
            skip_nan_contaminated_mae=cfg.show.generate_final_general_plot.get(
                "skip_nan_contaminated_mae", False
            ),
            skip_nan_contaminated_std_uncertainty=cfg.show.generate_final_general_plot.get(
                "skip_nan_contaminated_std_uncertainty", False
            ),
            group_stack_cumsum=cfg.show.generate_final_general_plot.get(
                "group_stack_cumsum", "post"
            ),
            final_length_spec=cfg.show.generate_final_general_plot.get(
                "final_length_spec", None
            ),
            warmup_mode=warmup_mode,
            warmup_steps=warmup_steps,
        )

    # .... C-MAE drift-rate plot (per-rollout vs matching GT trajectory) ..........................
    _drift_rate_cfg = cfg.show.get("generate_drift_rate_plot", None)
    if _drift_rate_cfg is not None and _drift_rate_cfg.get("render", False):
        _drift_rate_gt_entries = _load_ground_truth_entries(
            cfg, root_project_path, bool(cfg.show.target_is_ood)
        )
        generate_drift_rate_plot(
            cfg,
            deepcopy(groups),
            _drift_rate_gt_entries,
            title,
            exp_dir_relative_path,
            fill_between_alpha,
            plot_linewidth,
            headless,
            root_project_path=root_project_path,
            train_epochs=_drift_rate_cfg.get("train_epochs", None),
            length_cutoff=_drift_rate_cfg.get("length_cutoff", None),
            trials_label=trials_label,
            y_axis_in_logscale=y_axis_in_logscale,
            x_axis_in_logscale=x_axis_in_logscale,
            warmup_mode=warmup_mode,
            warmup_steps=warmup_steps,
            dispersion_type=_drift_rate_cfg.get("dispersion_type", "std-err"),
            generate_diaporama=_drift_rate_cfg.get("generate_diaporama", False),
            diaporama_fps=_drift_rate_cfg.get("diaporama_fps", 1.0),
            selected_trajectories=selected_trajectories,
        )

    # .... Per-train-epoch C-MAE drift-rate plot (RLRP-773 W9) ....................................
    # Opt-in NEW plot: one point per discovered ``epoch_<E>`` training snapshot
    # under ``<experiment_base>/epoch_checkpoints_rollouts/``. Gated on the
    # nested ``per_train_epoch.render`` sub-key (default OFF => the existing
    # drift plot above runs UNCHANGED). ``show.*`` has no structured schema so
    # every sub-key is read defensively with safe defaults. Legacy-safe: a no-op
    # when no per-epoch rollout tree exists.
    _drift_rate_per_epoch_cfg = omegaconf.OmegaConf.select(
        cfg, "show.generate_drift_rate_plot.per_train_epoch", default=None
    )
    if _drift_rate_per_epoch_cfg is not None and _drift_rate_per_epoch_cfg.get(
        "render", False
    ):
        _drift_rate_parent_cfg = cfg.show.get("generate_drift_rate_plot", None)
        _parent_length_cutoff = (
            _drift_rate_parent_cfg.get("length_cutoff", None)
            if _drift_rate_parent_cfg is not None
            else None
        )
        _parent_drift_type = (
            _drift_rate_parent_cfg.get("drift_type", "timesteps")
            if _drift_rate_parent_cfg is not None
            else "timesteps"
        )
        _drift_rate_per_epoch_gt_entries = _load_ground_truth_entries(
            cfg, root_project_path, bool(cfg.show.target_is_ood)
        )
        generate_drift_rate_plot_per_train_epoch(
            cfg,
            deepcopy(groups),
            _drift_rate_per_epoch_gt_entries,
            title,
            exp_dir_relative_path,
            fill_between_alpha,
            plot_linewidth,
            headless,
            root_project_path,
            length_cutoff=_drift_rate_per_epoch_cfg.get(
                "length_cutoff", _parent_length_cutoff
            ),
            trials_label=trials_label,
            y_axis_in_logscale=_drift_rate_per_epoch_cfg.get(
                "y_axis_in_logscale", y_axis_in_logscale
            ),
            x_axis_in_logscale=_drift_rate_per_epoch_cfg.get(
                "x_axis_in_logscale", x_axis_in_logscale
            ),
            drift_type=_drift_rate_per_epoch_cfg.get("drift_type", _parent_drift_type),
            warmup_mode=warmup_mode,
            warmup_steps=warmup_steps,
            dispersion_type=_drift_rate_per_epoch_cfg.get(
                "dispersion_type",
                (
                    _drift_rate_parent_cfg.get("dispersion_type", "std-err")
                    if _drift_rate_parent_cfg is not None
                    else "std-err"
                ),
            ),
            generate_diaporama=_drift_rate_per_epoch_cfg.get(
                "generate_diaporama",
                (
                    _drift_rate_parent_cfg.get("generate_diaporama", False)
                    if _drift_rate_parent_cfg is not None
                    else False
                ),
            ),
            diaporama_fps=_drift_rate_per_epoch_cfg.get(
                "diaporama_fps",
                (
                    _drift_rate_parent_cfg.get("diaporama_fps", 1.0)
                    if _drift_rate_parent_cfg is not None
                    else 1.0
                ),
            ),
            selected_trajectories=selected_trajectories,
        )

    # .... Training wall-clock time box plot (average per model type) .............................
    if cfg.show.generate_training_wall_clock_time_plot.render:
        generate_training_wall_clock_time_plot(
            cfg,
            deepcopy(groups),
            exp_dir_relative_path,
            headless,
            # y_axis_in_logscale=y_axis_in_logscale, # <-- Show in normal space instead
        )

    # .... Rollout wall-clock time box plot (average per model type) ..............................
    if cfg.show.generate_rollout_wall_clock_time_plot.render:
        generate_rollout_wall_clock_time_plot(
            cfg,
            deepcopy(groups),
            exp_dir_relative_path,
            headless,
            # y_axis_in_logscale=y_axis_in_logscale, # <-- Show in normal space instead
        )

    # .... Rollout inference speed box plot (Hz/fps) ..............................................
    _inf_speed_cfg = cfg.show.get("generate_rollout_inference_speed_plot", {})
    if _inf_speed_cfg.get("render", False):
        generate_rollout_inference_speed_plot(
            cfg,
            deepcopy(groups),
            exp_dir_relative_path,
            headless,
            unit=_inf_speed_cfg.get("unit", "hz"),
            levels=_inf_speed_cfg.get("levels", ("control_loop_step", "model_call")),
            source=_inf_speed_cfg.get("source", "measured"),
            # y_axis_in_logscale=y_axis_in_logscale, # <-- Show in normal space instead
        )

    # .... Rollout inference cost breakdown bar plot (ms) .........................................
    _cost_breakdown_cfg = cfg.show.get(
        "generate_rollout_inference_cost_breakdown_plot", {}
    )
    if _cost_breakdown_cfg.get("render", False):
        generate_rollout_inference_cost_breakdown_plot(
            cfg,
            deepcopy(groups),
            exp_dir_relative_path,
            headless,
            min_components=_cost_breakdown_cfg.get("min_components", 2),
        )

    # ==== Teardown ===============================================================================
    # Always close every remaining figure on the way out of ``execute`` so
    # repeated invocations do not leak figures across runs (D12 oracle).
    plt.close("all")
    return None
