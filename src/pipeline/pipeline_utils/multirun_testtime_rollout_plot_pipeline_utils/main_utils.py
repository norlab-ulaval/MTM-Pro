# coding=utf-8
import logging
import os
import warnings
from collections import defaultdict
from copy import copy
from typing import Any, List

import numpy as np
import omegaconf
from matplotlib.figure import Figure
from omegaconf import DictConfig

from algorithm.experience_replay_learning_loop.core.data_classes import (
    TestTimeRolloutPredictionMetric,
    get_metric_sub_dir,
)

from pipeline.pipeline_utils.multirun_testtime_rollout_plot_pipeline_utils.category_utils import (
    _get_category_max_length,
    _resolve_category_lookup,
    _truncate_mae_to_category_length,
)
from pipeline.pipeline_utils.multirun_testtime_rollout_plot_pipeline_utils.trajectory_processing_utils import (
    _resolve_ground_truth_feed_warmup_steps,
    _resolve_test_trajectories_split,
    apply_simulator_config_override,
)
from pipeline.pipeline_utils.general.testtime_trajectory_dataclass_util import (
    TestTrajectoryEntry,
)
from tools.console_tools.message import consol_msg_universal_one_liner


def assert_homogeneous_stats_space(metrics: List[Any], group_label: str) -> None:
    """Refuse to aggregate prediction statistics across incompatible unit spaces.

    RLRP-761 ``P4.4``. ``mean`` / ``std`` / ``std_epi`` are only comparable across
    cells that agree on ``stats_space``: a pre-``P1`` block-facade artifact holds
    them in the NORMALIZED target space while a post-``P1`` one (or any
    ``standard`` run) holds them in physical units, and the two differ by
    ``1/sigma_target`` — up to ``40x`` on the reference UGV layout. Averaging them
    on one axis produces a plausible-looking, meaningless curve.

    ``mae`` / ``l2_norm`` are scored from the pose channel, are always physical
    and are deliberately NOT restricted by this guard.

    ``"unknown"`` is tolerated (with a warning): it marks a legacy artifact whose
    space could not be resolved, and rejecting it would make every pre-``P4``
    archive unplottable — a cure worse than the disease.

    Args:
        metrics: the metrics contributing to one aggregation group.
        group_label: group name, for the error message.

    Raises:
        ValueError: when two contributing cells declare DIFFERENT known spaces.
    """
    spaces: dict = {}
    for metric in metrics:
        space = getattr(metric, "stats_space", None)
        if space is None or space == "unknown":
            continue
        spaces.setdefault(space, []).append(
            getattr(metric, "_trial_key", getattr(metric, "model_name", "?"))
        )
    if len(spaces) > 1:
        detail = "; ".join(
            f"{space}: {sorted(set(map(str, keys)))}" for space, keys in spaces.items()
        )
        raise ValueError(
            f"[{group_label}] refusing to aggregate prediction statistics across "
            f"incompatible unit spaces ({detail}). `mean`/`std`/`std_epi` of a "
            f"pre-RLRP-761 block-facade run are in the NORMALIZED target space and "
            f"are not comparable with physical ones. Re-run the deploy phase of the "
            f"offending cells (no retraining needed), or restrict the plot to one "
            f"space. Note `mae`/`l2_norm` are unaffected and remain comparable."
        )
    if not spaces and metrics:
        warnings.warn(
            f"[{group_label}] no `stats_space` recorded on any contributing metric "
            "— these are legacy (pre-RLRP-761 P4) artifacts, so the unit space of "
            "`mean`/`std`/`std_epi` cannot be verified. Treat the uncertainty "
            "surfaces of this group with caution.",
            RuntimeWarning,
            stacklevel=2,
        )
    return None


def _pad_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad a 1-D array with NaN to reach target_len.

    If the array is already >= target_len, return as-is.
    """
    if arr.shape[0] >= target_len:
        return arr
    pad = np.full(target_len - arr.shape[0], np.nan)
    return np.concatenate([arr, pad])


def _element_at(seq: list, index: int) -> Any:
    """Return ``seq[index]`` or ``None`` when the list is shorter than expected.

    RLRP-785 A3 — the per-trial ``rollout_steps`` / ``benchmark`` lists are
    appended in lockstep with the other per-trial keys, but legacy artifacts may
    have contributed neither, so the reduction stays bounds-safe.
    """
    return seq[index] if seq is not None and index < len(seq) else None


def _build_trajectory_length_map(groups: dict) -> dict[str, int]:
    """Derive ``{trajectory_name: length}`` from all groups' ``_original_metrics``.

    Length of a trajectory is the maximum ``mae`` timestep count observed for
    that ``_trajectory_name`` across every group/trial that reported it. This
    is the authoritative ground-truth horizon used to resolve
    ``show.general_length`` per plan §0 / Q2.
    """
    length_map: dict[str, int] = {}
    for grp in groups.values():
        for ogm in grp.get("_original_metrics", []) or []:
            tname = getattr(ogm, "_trajectory_name", None) or "unknown"
            mae_arr = getattr(ogm, "mae", None)
            if mae_arr is None or not hasattr(mae_arr, "shape") or mae_arr.size == 0:
                continue
            tlen = int(mae_arr.shape[0])
            if tlen > length_map.get(tname, 0):
                length_map[tname] = tlen
    return length_map


def _aggregate_group_over_selected_trajectories(
    group: dict,
    selected_tnames: set[str],
    cutoff_len: int,
    use_cumulative: bool,
    average_across_trajectories: bool = False,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Aggregate a group's MAE across a trajectory-name subset, truncated to ``cutoff_len``.

    Returns ``(avg_mae, std_mae, n_units)`` or ``None`` if the group has no
    matching rollouts. Each matching ``_original_metrics`` entry is reduced
    (mean over trailing dim + optional cumsum) and truncated to the minimum
    of its natural length and ``cutoff_len``; curves shorter than ``cutoff_len``
    are NaN-padded so they can be stacked.

    Two aggregation semantics, selected by ``average_across_trajectories``:

    - ``False`` (default, legacy behavior): pool every ``(trial, trajectory)``
      rollout into a single stack; mean/std are taken across that pool. The
      reported ``n_units`` is the number of rollouts. This surfaces
      inter-rollout / inter-trial dispersion mixed with inter-trajectory-
      pattern dispersion.
    - ``True``: first reduce each trajectory to its mean curve across trials
      (removing intra-trajectory trial dispersion), then take mean/std across
      those per-trajectory means. ``n_units`` is the number of distinct
      trajectories. This surfaces inter-trajectory-pattern variation.
    """
    per_rollout: list[tuple[str, np.ndarray]] = []
    for ogm in group.get("_original_metrics", []) or []:
        tname = getattr(ogm, "_trajectory_name", None) or "unknown"
        if tname not in selected_tnames:
            continue
        mae_arr = getattr(ogm, "mae", None)
        if mae_arr is None or not hasattr(mae_arr, "shape") or mae_arr.size == 0:
            continue
        reduced = np.mean(mae_arr, axis=-1) if mae_arr.ndim > 1 else mae_arr
        if use_cumulative:
            reduced = np.cumsum(reduced)
        reduced = np.asarray(reduced, dtype=float)[:cutoff_len]
        per_rollout.append((tname, _pad_to_length(reduced, cutoff_len)))

    if not per_rollout:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if not average_across_trajectories:
            # Legacy semantics: inter-rollout dispersion over the pooled set.
            stack = np.stack([c for _, c in per_rollout])
            avg = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
            return avg, std, len(per_rollout)
        # Inter-trajectory-pattern semantics: collapse trials per trajectory
        # first, then mean/std across the per-trajectory means.
        by_traj: dict[str, list[np.ndarray]] = {}
        for tname, c in per_rollout:
            by_traj.setdefault(tname, []).append(c)
        traj_means = np.stack(
            [np.nanmean(np.stack(cs), axis=0) for cs in by_traj.values()]
        )
        avg = np.nanmean(traj_means, axis=0)
        std = np.nanstd(traj_means, axis=0)
        return avg, std, int(traj_means.shape[0])


def _aggregate_group_over_all_trajectories(
    group: dict,
    max_len: int,
    use_cumulative: bool,
    average_across_trajectories: bool = False,
    skip_nan_contaminated_mae: bool = False,
    skip_nan_contaminated_std_uncertainty: bool = False,
    error_type: str = "elementwise_mae",
    group_stack_cumsum="post",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """
    Aggregates group data over all trajectories with options for cumulative
    calculation, averaging, and handling NaN-contaminated metrics.


    :param group: A dictionary containing metrics data grouped by trajectory. The key
        "_original_metrics" is expected, containing objects with attributes like
        'mae' for metrics, 'std' for standard deviation, and a trajectory label
        accessible through '_trajectory_name'.
    :param max_len: Maximum length up to which trajectories will be padded or truncated.
    :param use_cumulative: If True, computes cumulative sums for the reduced metrics.
    :param average_across_trajectories: If True, averages metrics across different trajectories,
        prioritizing inter-trajectory pattern variation over inter-trial variability.
    :param skip_nan_contaminated_mae: If True, skips trajectories where the 'metric.mae' attribute
        contains NaN values.
    :param skip_nan_contaminated_std_uncertainty: If True, skips trajectories where the 'metric.std' attribute
        contains NaN values. Note: Probabilistic model rollout with nan-contaminated std metric  is a red flag
    :param group_stack_cumsum: group stack cumulative sum reduction ordering, either 'pre' or 'post'
    :return: A tuple with the following components:
        - A numpy array representing the averaged metrics.
        - A numpy array representing the standard deviation of the metrics.
        - A numpy array representing the number of valid data points per timestep.
        - An integer indicating the number of trajectories processed.
        Returns None if no valid trajectories remain after filtering.
    """

    per_rollout: list[tuple[str, np.ndarray]] = []
    for ogm in group.get("_original_metrics", []) or []:

        if error_type == "l2_norm":
            _error_type_label = "L2 (Euclidean) norm"
            error_arr = getattr(ogm, "l2_norm", None)
        elif error_type == "elementwise_mae":
            _error_type_label = "per feature dimension MAE"
            error_arr = getattr(ogm, "mae", None)
        else:
            raise NotImplementedError(f"Unsupported error type: {error_type}")

        if error_arr is None or not hasattr(error_arr, "shape") or error_arr.size == 0:
            continue

        if skip_nan_contaminated_mae:
            if np.isnan(error_arr).any():
                continue
        # else:
        #     error_arr = np.nan_to_num(error_arr)

        if skip_nan_contaminated_std_uncertainty:
            # This only relevant for probabilistic model
            if np.isnan(getattr(ogm, "std", None)).any():
                continue

        gname = getattr(ogm, "model_name", None) or "unknown"
        tname = getattr(ogm, "_trajectory_name", None) or "unknown"
        trialkey = getattr(ogm, "_trial_key", None) or "unknown"
        _full_ref_label = f"Name: {gname} | {trialkey[0]} {trialkey[1]} | {tname}"

        # Reduce feature dimensions e.g., coord 3D mae -> mean coord mae
        if error_type == "elementwise_mae":
            error_arr: np.ndarray
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "error", category=RuntimeWarning
                    )  # turn into exception
                    feature_reduced = (
                        np.nanmean(error_arr, axis=-1)
                        if error_arr.ndim > 1
                        else error_arr
                    )
            except RuntimeWarning as e:
                # Show which trajectory record have a timestep slice with xyz NaN when the
                #  following warning is raised: "RuntimeWarning: Mean of empty slice
                #   np.nanmean(error_arr, axis=-1) if error_arr.ndim > 1 else error_arr"
                warnings.warn(
                    f"{e} (meaning a trajectory timestep slice with xyz features where all NaN) {_full_ref_label}"
                )
                feature_reduced = (
                    np.nanmean(error_arr, axis=-1) if error_arr.ndim > 1 else error_arr
                )

        # Note: post-feature reduction cumsum ie., 3d mae -> 1d mae -> 1d c-mae -> 1d c-mae g-mean
        if use_cumulative and group_stack_cumsum == "pre":
            feature_reduced = np.nancumsum(feature_reduced, axis=-1)

        feature_reduced = np.asarray(feature_reduced, dtype=float)[:max_len]
        per_rollout.append((tname, _pad_to_length(feature_reduced, max_len)))

    if not per_rollout:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if not average_across_trajectories:
            # Shows the inter-trial variation
            stack = np.stack([c for _, c in per_rollout])
            avg = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)

            # Note: post-group stack cumsum ie., 3d mae -> 1d mae -> 1d mae g-mean -> 1d c-mae g-mean
            if use_cumulative and group_stack_cumsum == "post":
                avg = np.nancumsum(avg, axis=-1)
                std = np.nancumsum(std, axis=-1)

            n_valid_per_t = np.sum(~np.isnan(stack), axis=0)
            return avg, std, n_valid_per_t, len(per_rollout)
        else:
            # shows inter-trajectory-pattern variation rather than inter-trial variation.
            by_traj: dict[str, list[np.ndarray]] = {}
            for tname, c in per_rollout:
                by_traj.setdefault(tname, []).append(c)

            traj_means = np.stack(
                [np.nanmean(np.stack(cs), axis=0) for cs in by_traj.values()]
            )
            avg = np.nanmean(traj_means, axis=0)
            std = np.nanstd(traj_means, axis=0)
            n_valid_per_t = np.sum(~np.isnan(traj_means), axis=0)
            return avg, std, n_valid_per_t, int(traj_means.shape[0])


def _format_grp_count_label(
    grp_name: str,
    *,
    n_trials: int | None,
    n_trajectories: int | float | None,
    n_total_rollouts: int | None,
    trials_label: str,
    do_average_across_trajectories: bool,
    show_acronym: bool = True,
) -> str:
    """Format ``"<grp_name> (<n_trials> <trials_label> \u00d7 <n_traj> trajectories \u2026)"``.

    Shared between the General-plot legend and the per-category plot label so
    both surfaces follow the same convention. Branches:

    - Multi-trajectory + averaging ON: ``(<n_tr> <trials> \u00d7 <n_tj> trajectories, mean within)``
    - Multi-trajectory + averaging OFF: ``(<n_tr> <trials> \u00d7 <n_tj> trajectories = <r> rollouts)``
      (``<r>`` defaults to ``n_tr * n_tj`` if ``n_total_rollouts`` is not given).
    - Single-trajectory: ``(<n_tr> <trials>)`` — collapses to the legacy form.
    - Fallback (no ``n_trials``): ``(n=<n_tj> trajectories, <r> rollouts)`` or
      ``(n=<n_tj> rollouts)`` to keep older callers/back-compat readable.
    """
    if show_acronym:
        trajectories_word = "trj"
        rollouts_word = "roll"
    else:
        trajectories_word = "trajectories" if n_trajectories > 1 else "trajectory"
        rollouts_word = "rollouts" if n_total_rollouts > 1 else "rollout"

    n_tr = n_trials or 0
    n_tj = n_trajectories or 0
    if n_tr and n_tj > 1:
        if do_average_across_trajectories:
            return f"{grp_name} ({n_tr} {trials_label} \u00d7 {n_tj} {trajectories_word}, mean within)"
        total = n_total_rollouts if n_total_rollouts is not None else n_tr * n_tj
        return (
            f"{grp_name} ({n_tr} {trials_label} \u00d7 {n_tj:.0f} {trajectories_word}"
            f" = {total} {rollouts_word})"
        )
    if n_tr:
        return f"{grp_name} ({n_tr} {trials_label})"
    if n_total_rollouts is not None and n_tj:
        return f"{grp_name} (n={n_tj} {trajectories_word}, {n_total_rollouts} {rollouts_word})"
    return f"{grp_name} (n={n_tj} {rollouts_word})"


def manage_torch_future_warnings() -> None:
    """Note: use inside a context manager.
    Example:
        >>> with warnings.catch_warnings():
        >>>     manage_torch_future_warnings()
        >>>     ...
    """
    warnings.filterwarnings(
        "ignore",
        message=(
            ".*You are using `torch.load` with `weights_only=False` \\(the current default value\\), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling \\(See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details\\). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature."
        ),
        category=FutureWarning,
    )
    return None


def display_experiment_name(cfg: DictConfig, cat_fig: Figure):
    try:
        experiment_name = cfg.experiment
    except (omegaconf.errors.ConfigAttributeError, AttributeError, KeyError):
        return
    text_style = {
        "color": "#888888",
        "ha": "left",
        # "fontsize": 18
        "fontsize": 17,
    }
    cat_fig.text(
        x=0.005,
        y=0.005,
        s=f"Experiment name: {experiment_name}",
        va="bottom",
        **text_style,
    )


def category_breakdown(
    cfg: DictConfig,
    groups: dict[Any, Any],
    do_average_across_trajectories: bool,
    do_category_breakdown: bool,
) -> dict[str, dict[str, dict]]:
    """
    Generates and computes category breakdown statistics for metrics grouped
    by given keys and categories.

    :param cfg: The configuration object containing settings such as display options.
    :param groups: A dictionary where keys identify groups and values contain their respective data.
    :param do_average_across_trajectories: Flag indicating whether metrics should be averaged across trajectory patterns.
    :param do_category_breakdown: Flag indicating whether to compute category breakdown statistics.
    :return: A dictionary containing per-group, per-category reduced mean absolute error (MAE) arrays and relevant statistics.
    """
    # category_breakdown_data: stores per-group, per-category reduced MAE arrays for plotting
    category_breakdown_data: dict[str, dict[str, dict]] = {}
    if do_category_breakdown:
        category_max_lengths = _get_category_max_length(cfg)
        consol_msg_universal_one_liner("Compute category breakdown stats:")
        for each_group_key, each_group in groups.items():
            grp_name = each_group["grp_name"]
            # Use original (pre-averaging) metrics so all trajectories are counted
            grp_org_metric = each_group.get("_original_metrics", each_group["metrics"])
            if not grp_org_metric:
                continue

            # Partition metrics by _category attribute; fallback to length-based heuristic
            by_category: dict[str, list] = {"S": [], "M": [], "L": [], "unknown": []}
            for each_ogm in grp_org_metric:
                cat = getattr(each_ogm, "_category", None)
                if cat is not None and cat in ("S", "M", "L"):
                    by_category[cat].append(each_ogm)
                else:
                    # Fallback: length-based heuristic
                    traj_len = each_ogm.mae.shape[0]
                    assigned = False
                    for heur_cat, max_len in sorted(
                        category_max_lengths.items(), key=lambda x: x[1]
                    ):
                        if traj_len <= max_len:
                            by_category[heur_cat].append(each_ogm)
                            assigned = True
                            break
                    if not assigned:
                        # ToDo: assess >> why assigne to cat L as a fallback when we actualy dont know its lenght.
                        #  Maybe we should warn and pass instead.
                        by_category["L"].append(each_ogm)

            category_breakdown_data[each_group_key] = {}
            print(f"  Group: {grp_name}")
            for cat in ["S", "M", "L"]:
                cat_metrics_list = by_category.get(cat, [])
                if not cat_metrics_list:
                    continue
                cat_max_len = category_max_lengths.get(cat)

                # Reduce and truncate MAE arrays to category max length
                def _reduce_and_truncate(ogm):
                    mae_r = np.mean(ogm.mae, axis=-1) if ogm.mae.ndim > 1 else ogm.mae
                    if cat_max_len is not None and mae_r.shape[0] > cat_max_len:
                        mae_r = mae_r[:cat_max_len]
                    if cfg.show.cumulative_metric:
                        mae_r = np.cumsum(mae_r)
                    return mae_r

                # Trial / trajectory counts for this category (shared by both branches)
                cat_n_trials = len(
                    {
                        getattr(m, "_trial_key", None)
                        for m in cat_metrics_list
                        if getattr(m, "_trial_key", None) is not None
                    }
                )
                cat_n_trajectories = len(
                    {
                        getattr(m, "_trajectory_name", None) or "unknown"
                        for m in cat_metrics_list
                    }
                )
                cat_n_total = len(cat_metrics_list)

                if do_average_across_trajectories:
                    # --- Averaging mode ---
                    # Group metrics by trajectory name, average within each
                    # trajectory (across trials), then compute mean/std across
                    # the per-trajectory averages. This shows inter-trajectory
                    # variation rather than inter-trial variation.
                    by_pattern: dict[str, list] = defaultdict(list)
                    for each_ogm in cat_metrics_list:
                        pattern = (
                            getattr(each_ogm, "_trajectory_name", None) or "unknown"
                        )
                        by_pattern[pattern].append(_reduce_and_truncate(each_ogm))

                    # Average within each trajectory across trials
                    pattern_averages = []
                    for pattern_name, pattern_maes in by_pattern.items():
                        p_min_len = min(m.shape[0] for m in pattern_maes)
                        pattern_maes = [m[:p_min_len] for m in pattern_maes]
                        pattern_averages.append(np.mean(np.stack(pattern_maes), axis=0))

                    # Truncate all per-trajectory averages to the shortest
                    actual_len = min(m.shape[0] for m in pattern_averages)
                    pattern_averages = [m[:actual_len] for m in pattern_averages]
                    cat_terminal_maes = [float(m[-1]) for m in pattern_averages]
                    print(
                        f"    Category {cat} ({cat_n_trials} trials \u00d7 "
                        f"{cat_n_trajectories} trajectories = {cat_n_total} rollouts, "
                        f"max_len={category_max_lengths.get(cat, '?')}): "
                        f"mean MAE={np.mean(cat_terminal_maes):.4f}, "
                        f"std MAE={np.std(cat_terminal_maes):.4f}"
                    )

                    cat_maes_stack = np.stack(pattern_averages)
                    category_breakdown_data[each_group_key][cat] = {
                        "grp_name": grp_name,
                        "line_style": each_group["line_style"],
                        "grp_avg_mae": np.mean(cat_maes_stack, axis=0),
                        "grp_std_mae": np.std(cat_maes_stack, axis=0),
                        "n_trials": cat_n_trials,
                        "n_trajectories": cat_n_trajectories,
                        "n_total_rollouts": cat_n_total,
                        "mean_terminal_mae": float(np.mean(cat_terminal_maes)),
                        "std_terminal_mae": float(np.std(cat_terminal_maes)),
                    }
                else:
                    # --- Non-averaging mode ---
                    # Each trial \u00d7 trajectory is a separate data point; std
                    # reflects both inter-trial AND inter-trajectory variation.
                    cat_maes_truncated = [
                        _reduce_and_truncate(m) for m in cat_metrics_list
                    ]

                    actual_len = min(m.shape[0] for m in cat_maes_truncated)
                    cat_maes_truncated = [m[:actual_len] for m in cat_maes_truncated]
                    cat_terminal_maes = [float(m[-1]) for m in cat_maes_truncated]
                    print(
                        f"    Category {cat} ({cat_n_trials} trials \u00d7 "
                        f"{cat_n_trajectories} trajectories = {cat_n_total} rollouts, "
                        f"max_len={category_max_lengths.get(cat, '?')}): "
                        f"mean MAE={np.mean(cat_terminal_maes):.4f}, "
                        f"std MAE={np.std(cat_terminal_maes):.4f}"
                    )

                    cat_maes_stack = np.stack(cat_maes_truncated)
                    category_breakdown_data[each_group_key][cat] = {
                        "grp_name": grp_name,
                        "line_style": each_group["line_style"],
                        "grp_avg_mae": np.mean(cat_maes_stack, axis=0),
                        "grp_std_mae": np.std(cat_maes_stack, axis=0),
                        "n_trials": cat_n_trials,
                        "n_trajectories": cat_n_trajectories,
                        "n_total_rollouts": cat_n_total,
                        "mean_terminal_mae": float(np.mean(cat_terminal_maes)),
                        "std_terminal_mae": float(np.std(cat_terminal_maes)),
                    }
    return category_breakdown_data


def compute_per_group_metric(
    cfg: DictConfig,
    root_project_path: str | bytes,
    do_average_across_trajectories: bool,
    do_category_breakdown: bool,
    collect_ground_truth_feed_warmup_steps: bool = False,
) -> dict[Any, Any] | tuple[dict[Any, Any], dict[str, int]]:
    """
    Compute per-group metrics for test-time rollout predictions based on the specified averaging,
    category breakdown rules and provided configuration e.g., compounded predictions vs
    non-compounded predictions, InD vs OoD environment, model type, model h-param

    :param cfg: The configuration object containing group details and parameters for
        processing metrics.
    :param root_project_path: The root path of the project-directory structure
        to locate relevant files and directories for metrics computation.
    :param do_average_across_trajectories: Boolean flag indicating whether metrics should
        be averaged across trajectories within a group or not.
    :param do_category_breakdown: Boolean flag indicating whether metrics should be
        broken down by specific categories (e.g., S, M, L) during averaging.
    :param collect_ground_truth_feed_warmup_steps: When ``True``, additionally
        collect the per-``experiment_base`` ground-truth-feed warm-up step
        count (auto-derived from each experiment's saved Hydra config) and
        return it as a second value. Defaults to ``False`` so existing callers
        keep the historical single-dict return type. The mapping omits
        experiments whose value is unresolvable (``None``); a resolved ``0``
        is preserved.
    :return: A dictionary mapping group names to their computed metrics,
        aggregated data, and relevant metadata. When
        ``collect_ground_truth_feed_warmup_steps`` is ``True``, returns the
        tuple ``(groups, warmup_steps_by_experiment_base)`` where the second
        element maps each ``experiment_base`` path to its resolved warm-up step
        count.
    """
    groups = {}
    group_nb = 0
    # Per-``experiment_base`` warm-up steps mapping. A single group can span
    # multiple ``experiment_base`` (several ``experiments`` x
    # ``multirun_paths``), so a per-experiment mapping is used rather than a
    # per-group scalar. Only populated when explicitly requested.
    warmup_steps_by_experiment_base: dict[str, int] = {}
    _category_lookup_cache: dict[str, dict[str, str]] = {}
    each_grp: dict
    for each_grp in omegaconf.OmegaConf.to_container(cfg.groups):

        if each_grp["grp_name"] not in omegaconf.OmegaConf.to_container(
            cfg.show.grp_names
        ):
            continue

        groups[f"group_{group_nb}"] = {
            "grp_name": each_grp["grp_name"],
            "gpr_short_name": each_grp["gpr_short_name"],
            "line_style": each_grp["line_style"],
            "metrics": [], # Type TestTimeRolloutPredictionMetric
            "_original_metrics": [], # Type TestTimeRolloutPredictionMetric
            "mean": [],
            "std": [],
            "std_epi": [],
            "mae": [],
            "target": [],
            "training_wall_clock_time": [],
            "rollout_wall_clock_time": [],
            # RLRP-785 A3 — the per-rollout step count (divisor for a rate) and the benchmark
            # metric set, so the reporting chain can finally express Hz/fps instead of only
            # total seconds (defect D3). Legacy artifacts predate both -> filled by fallback.
            "rollout_steps": [],
            "benchmark": [],
            "grp_avg_mae": None,
            "grp_std_mae": None,
            "gpr_avg_uncertainty_std": None,
            "gpr_avg_uncertainty_std_epi": None,
            "grp_size": None,
        }
        for each_experiments in each_grp["experiments"]:
            # Treat both ``None`` and empty list as "no multirun sweep" and look
            # for artifacts directly under ``experiment_path``. The regenerate
            # path uses the same idiom; without this, ``multirun_paths: []``
            # silently skips the experiment.
            multirun_path = each_experiments["multirun_paths"] or ["."]
            for each_multirun_path in multirun_path:
                experiment_base = os.path.realpath(
                    os.path.join(
                        root_project_path,
                        str(each_experiments["experiment_path"]),
                        str(each_multirun_path),
                    )
                )

                # Record the auto-derived ground-truth-feed warm-up step count
                # for this experiment (keyed by ``experiment_base``) when
                # requested. Unresolvable experiments (``None``) are omitted;
                # a resolved ``0`` is a valid warm-up length and is preserved.
                if collect_ground_truth_feed_warmup_steps:
                    _warmup_steps = _resolve_ground_truth_feed_warmup_steps(
                        experiment_base
                    )
                    if _warmup_steps is not None:
                        warmup_steps_by_experiment_base[experiment_base] = (
                            _warmup_steps
                        )

                # Discover trajectories: new layout under testtime_rollouts/<name>/
                testtime_rollouts_dir = os.path.join(
                    experiment_base, "testtime_rollouts"
                )
                if os.path.isdir(testtime_rollouts_dir):
                    trajectory_names = sorted(
                        d
                        for d in os.listdir(testtime_rollouts_dir)
                        if os.path.isdir(os.path.join(testtime_rollouts_dir, d))
                    )
                    ttrpm_paths = [
                        os.path.join(
                            testtime_rollouts_dir, tname, str(each_grp["ttrpm_dir"])
                        )
                        for tname in trajectory_names
                    ]
                else:
                    # Fallback: old flat layout
                    warnings.warn(
                        f"'testtime_rollouts/' not found at '{experiment_base}'. "
                        f"Falling back to the old flat layout. "
                        f"Be advise, the trajectory record is either missing or is in the legacy format and need to be converted by running with regenerate_rollouts=true to migrate to the new directory structure.",
                        stacklevel=2,
                    )
                    ttrpm_paths = [
                        os.path.join(experiment_base, str(each_grp["ttrpm_dir"]))
                    ]
                    trajectory_names = [None]

                for ttrpm_path, traj_name in zip(ttrpm_paths, trajectory_names):
                    if not os.path.isdir(ttrpm_path):
                        warnings.warn(
                            f"TTRPM directory not found, skipping: '{ttrpm_path}'",
                            stacklevel=2,
                        )
                        continue

                    # Skip trajectories that don't match the requested target_is_ood
                    # (InD trajectories only have InD_compounded_* subdirs, OOD only OOD_*)
                    expected_metric_subdir = get_metric_sub_dir(
                        cfg.show.compounded_predictions_score, cfg.show.target_is_ood
                    )
                    if not os.path.isdir(
                        os.path.join(ttrpm_path, expected_metric_subdir)
                    ):
                        continue

                    metric = TestTimeRolloutPredictionMetric.load(
                        ttrpm_path,
                        compounded_predictions_score=cfg.show.compounded_predictions_score,
                        target_is_ood=cfg.show.target_is_ood,
                    )

                    # ToDo: Add _trajectory_name and _trial_key to TestTimeRolloutPredictionMetric dataclass
                    # Attach trajectory metadata for downstream averaging/category logic
                    metric._trajectory_name = traj_name
                    # Attach trial provenance (experiment_path + multirun_path)
                    # so the legend label can distinguish ``n_trials`` from
                    # ``n_rollouts = n_trials × n_trajectories``.
                    metric._trial_key = (
                        str(each_experiments["experiment_path"]),
                        str(each_multirun_path),
                    )
                    # Attach the resolved experiment base directory so
                    # downstream consumers (e.g.
                    # :func:`_collect_wall_clock_samples`'s training-time
                    # fallback) can locate the on-disk artifacts of the
                    # trial (``console.log``, ``testtime_rollouts/``, …)
                    # without having to re-resolve paths from
                    # ``_trial_key`` + ``root_project_path``.
                    metric._experiment_base = experiment_base

                    # Resolve category from saved Hydra config
                    if traj_name is not None:
                        if experiment_base not in _category_lookup_cache:
                            _sim_cfg_name = cfg.get("simulator_config", None)
                            _proj_cfg_root = getattr(
                                cfg, "project_config_root_path", None
                            )
                            _category_lookup_cache[experiment_base] = (
                                _resolve_category_lookup(
                                    experiment_base,
                                    simulator_config_name=_sim_cfg_name,
                                    project_config_root=_proj_cfg_root,
                                )
                            )
                        metric._category = _category_lookup_cache[experiment_base].get(
                            traj_name, "unknown"
                        )
                    else:
                        metric._category = "unknown"

                    groups[f"group_{group_nb}"]["metrics"].append(copy(metric))
                    groups[f"group_{group_nb}"]["mean"].append(np.copy(metric.mean))
                    groups[f"group_{group_nb}"]["std"].append(np.copy(metric.std))
                    groups[f"group_{group_nb}"]["std_epi"].append(
                        np.copy(metric.std_epi)
                    )

                    # Options: l2_norm, elementwise_mae. Default to elementwise_mae for
                    # backward compatibility when callers/fixtures omit ``show.error_type``.
                    _error_type = omegaconf.OmegaConf.select(
                        cfg, "show.error_type", default="elementwise_mae"
                    )

                    try:
                        groups[f"group_{group_nb}"]["l2_norm"].append(
                            np.copy(metric.l2_norm)
                        )
                    except KeyError:
                        if _error_type == "l2_norm":
                            raise KeyError(
                                f"You requested 'show.l2_norm', but it is not available in {metric._trial_key} TestTimeRolloutPredictionMetric object! Maybe re-run deployment rollout to re-generate the object records."
                            )
                        else:
                            pass

                    groups[f"group_{group_nb}"]["mae"].append(np.copy(metric.mae))
                    groups[f"group_{group_nb}"]["target"].append(np.copy(metric.target))
                    groups[f"group_{group_nb}"]["training_wall_clock_time"].append(
                        np.copy(metric.training_wall_clock_time)
                    )
                    groups[f"group_{group_nb}"]["rollout_wall_clock_time"].append(
                        np.copy(metric.rollout_wall_clock_time)
                    )
                    # RLRP-785 A3 — carry the step count so a per-trial *rate* (Hz == fps) can be
                    # derived downstream. Legacy artifacts predate ``rollout_steps`` (resolves to
                    # ``None`` via the dataclass default) -> fall back to the MAE length, which is
                    # the per-rollout step count.
                    _rollout_steps = getattr(metric, "rollout_steps", None)
                    if _rollout_steps is None and metric.mae is not None:
                        try:
                            _rollout_steps = int(len(metric.mae))
                        except TypeError:
                            _rollout_steps = None
                    groups[f"group_{group_nb}"]["rollout_steps"].append(_rollout_steps)
                    # RLRP-785 A3 — forward the per-rollout ``BenchmarkMetricSet`` (``None`` unless
                    # the producing run had instrumentation ON); the plot layer reads it per level.
                    groups[f"group_{group_nb}"]["benchmark"].append(
                        getattr(metric, "benchmark", None)
                    )

        # .... Save original metrics before averaging for category breakdown ......................
        groups[f"group_{group_nb}"]["_original_metrics"] = list(
            groups[f"group_{group_nb}"]["metrics"]
        )

        # RLRP-761 P4.4 — refuse to aggregate statistics living in different unit
        # spaces. ``mean`` / ``std`` / ``std_epi`` are only comparable across cells
        # that agree on ``stats_space``; ``mae`` / ``l2_norm`` are always physical
        # and stay unrestricted.
        assert_homogeneous_stats_space(
            groups[f"group_{group_nb}"]["_original_metrics"], f"group_{group_nb}"
        )

        # Count distinct trials and trajectories that contributed to this group
        # (computed on the pre-averaging snapshot so the legend can report
        # accurate counts regardless of ``do_average_across_trajectories``).
        _trial_keys = {
            getattr(m, "_trial_key", None)
            for m in groups[f"group_{group_nb}"]["_original_metrics"]
        }
        _trial_keys.discard(None)
        _traj_names = {
            getattr(m, "_trajectory_name", None)
            for m in groups[f"group_{group_nb}"]["_original_metrics"]
        }
        _traj_names.discard(None)
        groups[f"group_{group_nb}"]["n_trials"] = len(_trial_keys)
        groups[f"group_{group_nb}"]["n_trajectories"] = len(_traj_names)

        # .... Average across trajectories (Phase 6.2) ............................................
        #
        # KNOWN LIMITATION (RLRP-803 item 7.1) -- DOCUMENTED ON PURPOSE, NOT A BUG TO FIX HERE.
        # When ``do_average_across_trajectories`` is ON, the curve-shaped keys (``mae``, ``std``,
        # ``std_epi``, ``mean``) ARE averaged over the N trials, but the non-averageable ones --
        # ``rollout_steps`` and ``benchmark`` (a ``BenchmarkMetricSet``) -- are reduced by KEEPING
        # THE REPRESENTATIVE TRIAL only (``cat_indices[0]`` / index ``0``) and DROPPING the other
        # N-1. Consequence: any inference-rate figure fed by this path (the boxplot of
        # ``generate_rollout_inference_speed_plot``) shows the dispersion of ONE trial, not of the
        # N averaged ones -- the box is narrower/differently shaped than the true across-trial
        # spread, even though the legend counts N trials.
        # Fixing it is deliberately OUT OF SCOPE: a rate is not linearly averageable with the
        # curve keys, so it would require a repo-wide averaging-policy change (every consumer of
        # these per-trial lists agrees on the "one representative entry" contract, see the
        # positional walk in ``_collect_inference_rate_samples``).
        # ROUTE TO EXACT DISPERSION: run the NON-averaged path
        # (``do_average_across_trajectories: false``) with ``benchmark.keep_raw_samples: true``;
        # every trial then keeps its own population and the boxplot draws TRUE quartiles.
        if (
            do_average_across_trajectories
            and len(groups[f"group_{group_nb}"]["metrics"]) > 1
        ):
            category_max_lengths = _get_category_max_length(cfg)

            if do_category_breakdown:
                # Average within each category separately, then replace group data
                all_avg_mae, all_avg_std, all_avg_std_epi = [], [], []
                all_avg_mean, all_avg_target = [], []
                all_avg_train_wc, all_avg_rollout_wc, all_avg_metrics = [], [], []
                # RLRP-785 A3 — reduced in step with the keys above, otherwise those two
                # stay at their pre-averaging length N and the downstream positional walk
                # (``_collect_inference_rate_samples``) reads a misaligned trial.
                all_avg_rollout_steps, all_avg_benchmark = [], []

                for cat in ["S", "M", "L"]:
                    cat_indices = [
                        i
                        for i, m in enumerate(groups[f"group_{group_nb}"]["metrics"])
                        if getattr(m, "_category", "unknown") == cat
                    ]
                    if not cat_indices:
                        continue

                    cat_max_len = category_max_lengths.get(cat)
                    cat_maes = [
                        groups[f"group_{group_nb}"]["mae"][i] for i in cat_indices
                    ]
                    cat_stds = [
                        groups[f"group_{group_nb}"]["std"][i] for i in cat_indices
                    ]
                    cat_stds_epi = [
                        groups[f"group_{group_nb}"]["std_epi"][i] for i in cat_indices
                    ]
                    cat_means = [
                        groups[f"group_{group_nb}"]["mean"][i] for i in cat_indices
                    ]

                    if cat_max_len is not None:
                        cat_maes = [
                            _truncate_mae_to_category_length(m, cat_max_len)
                            for m in cat_maes
                        ]
                        cat_stds = [
                            _truncate_mae_to_category_length(s, cat_max_len)
                            for s in cat_stds
                        ]
                        cat_stds_epi = [
                            _truncate_mae_to_category_length(s, cat_max_len)
                            for s in cat_stds_epi
                        ]
                        cat_means = [
                            _truncate_mae_to_category_length(m, cat_max_len)
                            for m in cat_means
                        ]

                    trunc_len = min(m.shape[0] for m in cat_maes)
                    all_avg_mae.append(
                        np.mean(np.stack([m[:trunc_len] for m in cat_maes]), axis=0)
                    )
                    all_avg_std.append(
                        np.mean(np.stack([s[:trunc_len] for s in cat_stds]), axis=0)
                    )
                    all_avg_std_epi.append(
                        np.mean(np.stack([s[:trunc_len] for s in cat_stds_epi]), axis=0)
                    )
                    all_avg_mean.append(
                        np.mean(np.stack([m[:trunc_len] for m in cat_means]), axis=0)
                    )
                    all_avg_target.append(
                        groups[f"group_{group_nb}"]["target"][cat_indices[0]][
                            :trunc_len
                        ]
                    )
                    all_avg_train_wc.append(
                        groups[f"group_{group_nb}"]["training_wall_clock_time"][
                            cat_indices[0]
                        ]
                    )
                    all_avg_rollout_wc.append(
                        groups[f"group_{group_nb}"]["rollout_wall_clock_time"][
                            cat_indices[0]
                        ]
                    )
                    all_avg_metrics.append(
                        groups[f"group_{group_nb}"]["metrics"][cat_indices[0]]
                    )
                    # Same representative selection as ``target`` / wall-clock / ``metrics``
                    # above: keep the retained trial's entry (a step count and a
                    # ``BenchmarkMetricSet`` are not averageable). Bounds-checked since legacy
                    # artifacts may not have contributed either key.
                    # RLRP-803 item 7.1: this is THE site where the other trials' inference
                    # rates are dropped -- see the block comment on the averaging branch above
                    # for the boxplot-dispersion consequence and the supported workaround.
                    all_avg_rollout_steps.append(
                        _element_at(
                            groups[f"group_{group_nb}"]["rollout_steps"], cat_indices[0]
                        )
                    )
                    all_avg_benchmark.append(
                        _element_at(
                            groups[f"group_{group_nb}"]["benchmark"], cat_indices[0]
                        )
                    )

                if all_avg_mae:
                    groups[f"group_{group_nb}"]["mae"] = all_avg_mae
                    groups[f"group_{group_nb}"]["std"] = all_avg_std
                    groups[f"group_{group_nb}"]["std_epi"] = all_avg_std_epi
                    groups[f"group_{group_nb}"]["mean"] = all_avg_mean
                    groups[f"group_{group_nb}"]["target"] = all_avg_target
                    groups[f"group_{group_nb}"][
                        "training_wall_clock_time"
                    ] = all_avg_train_wc
                    groups[f"group_{group_nb}"][
                        "rollout_wall_clock_time"
                    ] = all_avg_rollout_wc
                    groups[f"group_{group_nb}"]["metrics"] = all_avg_metrics
                    groups[f"group_{group_nb}"]["rollout_steps"] = all_avg_rollout_steps
                    groups[f"group_{group_nb}"]["benchmark"] = all_avg_benchmark
            else:
                # Current behavior: average all trajectories together using min_len
                min_len = min(m.shape[0] for m in groups[f"group_{group_nb}"]["mae"])
                avg_mae = np.mean(
                    np.stack([m[:min_len] for m in groups[f"group_{group_nb}"]["mae"]]),
                    axis=0,
                )
                avg_std = np.mean(
                    np.stack([s[:min_len] for s in groups[f"group_{group_nb}"]["std"]]),
                    axis=0,
                )
                avg_std_epi = np.mean(
                    np.stack(
                        [s[:min_len] for s in groups[f"group_{group_nb}"]["std_epi"]]
                    ),
                    axis=0,
                )
                groups[f"group_{group_nb}"]["mae"] = [avg_mae]
                groups[f"group_{group_nb}"]["std"] = [avg_std]
                groups[f"group_{group_nb}"]["std_epi"] = [avg_std_epi]
                groups[f"group_{group_nb}"]["mean"] = [
                    np.mean(
                        np.stack(
                            [m[:min_len] for m in groups[f"group_{group_nb}"]["mean"]]
                        ),
                        axis=0,
                    )
                ]
                groups[f"group_{group_nb}"]["target"] = [
                    groups[f"group_{group_nb}"]["target"][0][:min_len]
                ]
                groups[f"group_{group_nb}"]["training_wall_clock_time"] = [
                    groups[f"group_{group_nb}"]["training_wall_clock_time"][0]
                ]
                groups[f"group_{group_nb}"]["rollout_wall_clock_time"] = [
                    groups[f"group_{group_nb}"]["rollout_wall_clock_time"][0]
                ]
                groups[f"group_{group_nb}"]["metrics"] = [
                    groups[f"group_{group_nb}"]["metrics"][0]
                ]
                # RLRP-785 A3 — keep the trial-0 representative here too so every per-trial
                # list stays the same length; leaving these at length N desynchronizes the
                # positional walk in ``_collect_inference_rate_samples``.
                # RLRP-803 item 7.1: as above, trial 0's ``BenchmarkMetricSet`` is the ONLY one
                # kept, so the inference-rate boxplot on this path reflects one trial's
                # dispersion. Intentional -- see the block comment on the averaging branch.
                groups[f"group_{group_nb}"]["rollout_steps"] = [
                    _element_at(groups[f"group_{group_nb}"]["rollout_steps"], 0)
                ]
                groups[f"group_{group_nb}"]["benchmark"] = [
                    _element_at(groups[f"group_{group_nb}"]["benchmark"], 0)
                ]

        # .... Merge group trials .................................................................
        groups[f"group_{group_nb}"]["grp_size"] = len(
            groups[f"group_{group_nb}"]["metrics"]
        )
        grp_mae = groups[f"group_{group_nb}"]["mae"]
        grp_uncertainty_std = groups[f"group_{group_nb}"]["std"]
        grp_uncertainty_std_epi = groups[f"group_{group_nb}"]["std_epi"]
        grp_training_wall_clock_time = groups[f"group_{group_nb}"][
            "training_wall_clock_time"
        ]
        grp_rollout_wall_clock_time = groups[f"group_{group_nb}"][
            "rollout_wall_clock_time"
        ]

        # Feature reduction (mean or sum)
        feature_reduction = cfg.pre_processing.feature_reduction
        for idx, each in enumerate(grp_mae):
            if feature_reduction == "mean":
                grp_mae[idx] = np.mean(each, axis=-1)
            elif feature_reduction == "sum":
                grp_mae[idx] = np.sum(each, axis=-1)
            else:
                raise NotImplementedError(f"{feature_reduction=} not supported.")

        for idx, each_std in enumerate(grp_uncertainty_std):
            if feature_reduction == "mean":
                grp_uncertainty_std[idx] = np.mean(each_std, axis=-1)
            elif feature_reduction == "sum":
                grp_uncertainty_std[idx] = np.sum(each_std, axis=-1)
            else:
                raise NotImplementedError(f"{feature_reduction=} not supported.")

        for idx, each_std_epi in enumerate(grp_uncertainty_std_epi):
            if feature_reduction == "mean":
                grp_uncertainty_std_epi[idx] = np.mean(each_std_epi, axis=-1)
            elif feature_reduction == "sum":
                grp_uncertainty_std_epi[idx] = np.sum(each_std_epi, axis=-1)
            else:
                raise NotImplementedError(f"{feature_reduction=} not supported.")

        for idx, each_train_wall_clock in enumerate(grp_training_wall_clock_time):
            # (CRITICAL) ToDo: assess -> not sure about that logic
            if None not in each_train_wall_clock:
                if feature_reduction == "mean":
                    grp_training_wall_clock_time[idx] = np.mean(each_train_wall_clock)
                elif feature_reduction == "sum":
                    grp_training_wall_clock_time[idx] = np.sum(each_train_wall_clock)
                else:
                    raise NotImplementedError(f"{feature_reduction=} not supported.")
            else:
                grp_training_wall_clock_time[idx] = 0.0

        for idx, each_rollout_wall_clock in enumerate(grp_rollout_wall_clock_time):
            # (CRITICAL) ToDo: assess -> not sure about that logic
            if feature_reduction == "mean":
                grp_rollout_wall_clock_time[idx] = np.mean(each_rollout_wall_clock)
            elif feature_reduction == "sum":
                grp_rollout_wall_clock_time[idx] = np.sum(each_rollout_wall_clock)
            else:
                raise NotImplementedError(f"{feature_reduction=} not supported.")

        # Guard: if all metrics for this group were filtered out (e.g. by the
        # InD/OoD expected_metric_subdir filter), yield an empty-metrics group
        # gracefully instead of crashing on `min([])` (closes BUG-P3-01). The
        # group entry is preserved so downstream consumers can inspect
        # ``_original_metrics == []``; ``execute`` skips groups without
        # ``grp_avg_mae`` via its own ``not fig_init`` guard.
        if len(grp_mae) == 0:
            groups[f"group_{group_nb}"]["_original_metrics"] = []
            groups[f"group_{group_nb}"]["metrics"] = []
            group_nb += 1
            continue

        # (PRIORITY) ToDo: assess should it be just padded? >> next bloc ↓↓ (RLRP-654)
        # Truncate all arrays to the minimum length across the group so that
        # np.vstack succeeds when trajectories have heterogeneous timestep counts
        _min_ts = min(a.shape[0] for a in grp_mae)
        grp_mae = [a[:_min_ts] for a in grp_mae]
        grp_uncertainty_std = [a[:_min_ts] for a in grp_uncertainty_std]
        grp_uncertainty_std_epi = [a[:_min_ts] for a in grp_uncertainty_std_epi]

        mae = np.vstack(grp_mae).T
        uncertainty_std = np.vstack(grp_uncertainty_std).T
        uncertainty_std_epi = np.vstack(grp_uncertainty_std_epi).T

        # (CRITICAL) ToDo: assess >> these lines ↓ why is `rollout_wall_clock_time` commented out?
        # Those are the total wall clock time (not the per step one)
        training_wall_clock_time = np.vstack(grp_training_wall_clock_time).T
        # rollout_wall_clock_time = np.vstack(grp_rollout_wall_clock_time).T #

        if cfg.show.cumulative_metric:
            mae = np.cumsum(mae, axis=0)
            uncertainty_std = np.cumsum(uncertainty_std, axis=0)
            uncertainty_std_epi = np.cumsum(uncertainty_std_epi, axis=0)

        # Fail-soft NaN handling (Decision 2b): when an input rollout contains
        # NaN, emit a user-visible warning naming the offending group/trial so
        # the operator can investigate, but let the group curve still render by
        # reducing via nanmean/nanstd/nanmedian over the remaining finite
        # trials. A timestep with ALL-NaN trials will naturally become NaN and
        # the fill_between band will render nothing at that sample.
        if cfg.show.get("show_NaN_warning", True):
            _grp_name_for_log = groups[f"group_{group_nb}"]["grp_name"]
            for _arr_name, _arr in (
                ("mae", mae),
                ("uncertainty_std", uncertainty_std),
                ("uncertainty_std_epi", uncertainty_std_epi),
            ):
                if np.isnan(_arr).any():
                    # Trial axis is the last axis (shape (T, n_trials) post-vstack.T)
                    _trial_has_nan = np.isnan(_arr).any(
                        axis=tuple(range(_arr.ndim - 1))
                    )
                    for _trial_idx in np.flatnonzero(_trial_has_nan):
                        _gpr_metric = groups[f"group_{group_nb}"]["metrics"]
                        _offending_record_path = os.path.join(
                            _gpr_metric[int(_trial_idx)]._trial_key[0],
                            _gpr_metric[int(_trial_idx)]._trial_key[1],
                        )
                        logger.warning(
                            (
                                "NaN detected in %s for group=%r, trial_index=%d; "
                                "NaN-contaminated samples will be ignored from group avg "
                                "via nan-aware reduction (fail-soft)."
                                f" Path: {_offending_record_path}",
                            ),
                            _arr_name,
                            _grp_name_for_log,
                            int(_trial_idx),
                        )

        # Group reduction (nan-aware: fail-soft under NaN, identical to
        # mean/median/std on NaN-free inputs).
        gpr_reduction = cfg.pre_processing.gpr_reduction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if gpr_reduction == "median":
                grp_avg_mae = np.nanmedian(mae, -1)
                grp_avg_uncertainty_std = np.nanmedian(uncertainty_std, -1)
                grp_avg_uncertainty_std_epi = np.nanmedian(uncertainty_std_epi, -1)
                grp_avg_training_wall_clock_time = np.nanmedian(
                    training_wall_clock_time, -1
                )
            elif gpr_reduction == "mean":
                grp_avg_mae = np.nanmean(mae, -1)
                grp_avg_uncertainty_std = np.nanmean(uncertainty_std, -1)
                grp_avg_uncertainty_std_epi = np.nanmean(uncertainty_std_epi, -1)
                grp_avg_training_wall_clock_time = np.nanmean(
                    training_wall_clock_time, -1
                )
            else:
                raise NotImplementedError(f"{gpr_reduction=} not supported.")

            grp_std_mae = np.nanstd(mae, -1)

        groups[f"group_{group_nb}"]["grp_avg_mae"] = grp_avg_mae
        groups[f"group_{group_nb}"]["grp_std_mae"] = grp_std_mae
        # Return full-length uncertainty arrays aligned with grp_avg_mae so the
        # general plot's fill_between(mae ± scale × variation) broadcasts correctly
        # for every variation_type (closes BUG-P2-01..04).
        groups[f"group_{group_nb}"]["gpr_avg_uncertainty_std"] = grp_avg_uncertainty_std
        groups[f"group_{group_nb}"][
            "gpr_avg_uncertainty_std_epi"
        ] = grp_avg_uncertainty_std_epi
        groups[f"group_{group_nb}"][
            "grp_avg_training_wall_clock_time"
        ] = grp_avg_training_wall_clock_time

        # .... Group teardown .....................................................................
        group_nb += 1
    if collect_ground_truth_feed_warmup_steps:
        return groups, warmup_steps_by_experiment_base
    return groups


def resolve_consistent_warmup_steps(
    warmup_steps_by_experiment_base: dict[str, int],
) -> int | None:
    """Return the single common ground-truth-feed warm-up value, or fail fast.

    Fail-fast consistency guard for the ``show.ground_truth_feed_warmup``
    feature. The warm-up treatment (``'discard'`` / ``'reference_line'``) only
    makes sense when every contributing recorded trajectory was rolled out
    with the *same* ``ground_truth_feed_warmup_steps`` — curves recorded with
    different warm-up lengths behave differently and cannot be compared on a
    shared axis.

    :param warmup_steps_by_experiment_base: mapping of ``experiment_base`` path
        to its resolved (non-``None``) warm-up step count, as produced by
        :func:`compute_per_group_metric` with
        ``collect_ground_truth_feed_warmup_steps=True``.
    :returns: ``None`` when the mapping is empty (nothing resolved → graceful
        skip upstream); otherwise the single warm-up value shared by every
        contributing experiment. A resolved value of ``0`` is valid and is
        returned as ``0`` (never coerced to ``None``).
    :raises ValueError: if two or more experiments disagree on their warm-up
        value. The message enumerates the distinct values together with the
        ``experiment_base`` path(s) each originated from. Note a mix of ``0``
        and a non-zero value is a disagreement and triggers this error.
    """
    if not warmup_steps_by_experiment_base:
        return None

    distinct_values = set(warmup_steps_by_experiment_base.values())
    if len(distinct_values) == 1:
        return next(iter(distinct_values))

    # Disagreement — build an informative message listing each distinct value
    # and the experiment_base path(s) it came from.
    by_value: dict[int, list[str]] = defaultdict(list)
    for base, value in warmup_steps_by_experiment_base.items():
        by_value[value].append(base)
    details = "\n".join(
        f"  - ground_truth_feed_warmup_steps={value}: {sorted(bases)}"
        for value, bases in sorted(by_value.items())
    )
    raise ValueError(
        "Inconsistent `ground_truth_feed_warmup_steps` across the contributing "
        "recorded rollouts. The `show.ground_truth_feed_warmup` treatment "
        "requires every experiment to have been recorded with the same warm-up "
        "length, because curves recorded with different warm-up lengths cannot "
        "be compared on a shared time axis. Found the following conflicting "
        f"values:\n{details}"
    )


def trajectory_breakdown(
    cfg: DictConfig,
    groups: dict[Any, Any],
    do_average_across_trajectories: bool,
    do_trajectory_breakdown: bool,
) -> dict[str, dict[str, dict]]:
    """Compute per-group, per-``trajectory_name`` breakdown statistics.

    Mirrors :func:`category_breakdown` but keys metrics by ``_trajectory_name``
    instead of the S/M/L category. No truncation cap is applied — each
    trajectory's natural length is preserved so the per-trajectory plot spans
    exactly the ground-truth horizon of that trajectory.

    :param cfg: Hydra configuration object (uses ``show.cumulative_metric``).
    :param groups: output of :func:`compute_per_group_metric`.
    :param do_average_across_trajectories: flag kept for signature parity with
        :func:`category_breakdown`; it is not meaningful here because trajectory
        breakdown is already keyed by trajectory_name (averaging across
        trajectories would collapse the breakdown). Accepted and ignored.
    :param do_trajectory_breakdown: when ``False``, an empty dict is returned.
    :return: ``{group_key: {trajectory_name: {grp_name, line_style, grp_avg_mae,
        grp_std_mae, n_rollouts, short_name}}}``.
    """
    trajectory_breakdown_data: dict[str, dict[str, dict]] = {}
    if not do_trajectory_breakdown:
        return trajectory_breakdown_data

    consol_msg_universal_one_liner("Compute trajectory breakdown stats:")
    for each_group_key, each_group in groups.items():
        grp_name = each_group["grp_name"]
        grp_org_metric = each_group.get("_original_metrics", each_group["metrics"])
        if not grp_org_metric:
            continue

        # Partition by trajectory_name
        by_trajectory: dict[str, list] = defaultdict(list)
        for each_ogm in grp_org_metric:
            tname = getattr(each_ogm, "_trajectory_name", None) or "unknown"
            by_trajectory[tname].append(each_ogm)

        trajectory_breakdown_data[each_group_key] = {}
        print(f"  Group: {grp_name}")
        for tname in sorted(by_trajectory.keys()):
            tname_metrics = by_trajectory[tname]

            def _reduce(ogm):
                mae_r = np.mean(ogm.mae, axis=-1) if ogm.mae.ndim > 1 else ogm.mae
                if cfg.show.cumulative_metric:
                    mae_r = np.cumsum(mae_r)
                return mae_r

            maes = [_reduce(m) for m in tname_metrics]
            actual_len = min(m.shape[0] for m in maes)
            maes = [m[:actual_len] for m in maes]
            maes_stack = np.stack(maes)
            terminal_maes = [float(m[-1]) for m in maes]
            print(
                f"    Trajectory {tname} ({len(tname_metrics)} rollouts, "
                f"len={actual_len}): mean MAE={np.mean(terminal_maes):.4f}, "
                f"std MAE={np.std(terminal_maes):.4f}"
            )
            trajectory_breakdown_data[each_group_key][tname] = {
                "grp_name": grp_name,
                "line_style": each_group["line_style"],
                "grp_avg_mae": np.mean(maes_stack, axis=0),
                "grp_std_mae": np.std(maes_stack, axis=0),
                "n_rollouts": len(tname_metrics),
                "short_name": os.path.basename(tname),
                "length": int(actual_len),
            }
    return trajectory_breakdown_data


def _load_ground_truth_entries(
    cfg: DictConfig,
    root_project_path: str | bytes,
    target_is_ood: bool,
) -> list | None:
    """Load ground-truth ``TestTrajectoryEntry`` list for plotting (Phase 6).

    Lightweight, read-only counterpart to the entry-building logic inside
    :func:`regenerate_rollouts` — loads only what is needed by
    :func:`generate_per_trajectory_plots` to render its 1/8-scaled 3D
    snapshot inset. No model is loaded.

    Iterates the unique ``experiment_base`` paths across ``cfg.groups``
    looking for the first one that has a saved Hydra config
    (``.hydra/config.yaml``); uses it as the experiment reference to build
    the entries. Returns ``None`` on any failure (missing cfg, build error,
    empty groups) — the caller treats this as "no snapshot inset", which is
    the documented graceful behavior from Phase 4 R2.

    :param cfg: Top-level multirun plot configuration.
    :param root_project_path: Absolute path used to resolve each
        ``experiment_path`` + ``multirun_paths`` entry under ``cfg.groups``.
    :param target_is_ood: Whether the caller wants OOD entries (``True``)
        or InD entries (``False``). Matches ``cfg.show.target_is_ood``.
    :return: list of ``TestTrajectoryEntry`` or ``None`` on failure.
    """
    import logging as _logging

    _log = _logging.getLogger(__name__)

    try:
        groups_container = omegaconf.OmegaConf.to_container(cfg.groups)
    except Exception as _exc:
        _log.warning("Snapshot inset disabled: cannot read cfg.groups (%s).", _exc)
        return None
    if not groups_container:
        _log.warning("Snapshot inset disabled: cfg.groups is empty.")
        return None

    _seen: set[str] = set()
    _last_error: Exception | None = None
    for each_grp in groups_container:
        if "experiments" not in each_grp or each_grp["experiments"] is None:
            continue
        for each_experiments in each_grp["experiments"]:
            multirun_paths = each_experiments.get("multirun_paths") or ["."]
            for each_multirun_path in multirun_paths:
                experiment_base = os.path.realpath(
                    os.path.join(
                        root_project_path,
                        str(each_experiments["experiment_path"]),
                        str(each_multirun_path),
                    )
                )
                if experiment_base in _seen:
                    continue
                _seen.add(experiment_base)

                hydra_cfg_path = os.path.join(experiment_base, ".hydra", "config.yaml")
                if not os.path.isfile(hydra_cfg_path):
                    continue
                try:
                    experiment_cfg = omegaconf.OmegaConf.load(hydra_cfg_path)
                    # Refresh stale data-source keys (e.g. renamed dataset folder)
                    # from the top-level ``simulator_config`` so CSV resolution
                    # uses the *current* on-disk layout, not the path captured at
                    # training time.
                    experiment_cfg = apply_simulator_config_override(
                        cfg, experiment_cfg
                    )
                    ind_entries, ood_entries = _resolve_test_trajectories_split(
                        cfg, experiment_base
                    )
                    traj_dicts = ood_entries if target_is_ood else ind_entries
                    return _build_entries(
                        traj_dicts, experiment_cfg, target_is_ood=target_is_ood
                    )
                except Exception as _exc:
                    _last_error = _exc
                    _log.warning(
                        "Snapshot inset: failed to build ground-truth entries from %s: %s",
                        experiment_base,
                        _exc,
                    )
                    continue
    if _last_error is None:
        _log.warning(
            "Snapshot inset disabled: no experiment directory provided a valid "
            ".hydra/config.yaml for ground-truth entry loading."
        )
    return None


def _is_math_env_traj_dict(traj_dict: dict) -> bool:
    """Detect whether a trajectory dict targets a math env (vs robotic).

    Math env entries carry ``initiale_coordinates`` (and typically ``param``) —
    the analogue of the robotic ``trajectory_name``/``category`` pair. Robotic
    entries carry ``trajectory_name``.
    """
    if traj_dict is None:
        return False
    if hasattr(traj_dict, "get"):
        if "initiale_coordinates" in traj_dict:
            return True
        if "trajectory_name" in traj_dict:
            return False
    return False


def _build_entries(
    traj_dicts: list[dict[str, str]],
    experiment_cfg_ref: DictConfig,
    target_is_ood: bool = False,
) -> List["TestTrajectoryEntry"]:
    """Build a list of ``TestTrajectoryEntry`` from a list of trajectory-dicts.

    Dispatches per environment type:

    - **Robotic env** (entries with ``trajectory_name``): loads the trajectory
      CSV via ``setup_trajectory_from_csv``.
    - **Math env** (entries with ``initiale_coordinates`` + ``param``): builds
      a fresh ``MathContinuousGymnasium`` per entry through
      :func:`build_math_test_trajectory_entries`. The ``target_is_ood`` flag
      selects the InD vs OoD measurement-noise configuration.

    Shared across all groups and all experiments in ``regenerate_rollouts``: because
    the goal of the multirun test-time rollout plot pipeline is to compare test-time
    rollouts from models trained on the *same* data, the ground-truth trajectory
    entries (simulator-side) are identical for every model group — so this helper is
    invoked exactly once per rollout-regeneration run (once for InD, once for OOD).
    """
    if len(traj_dicts) > 0 and _is_math_env_traj_dict(traj_dicts[0]):
        from pipeline.pipeline_utils.math_env_pipeline_utils.setup_utils import (
            build_math_test_trajectory_entries,
        )

        return build_math_test_trajectory_entries(
            experiment_cfg_ref, traj_dicts, target_is_ood=target_is_ood
        )

    # Deferred imports to avoid circular dependency at module level.
    from pipeline.pipeline_utils.robotic_env_pipeline_utils.setup_utils import (
        setup_trajectory_from_csv,
        robotic_data_to_test_motion_trajectory_dataclass,
    )
    from pipeline.pipeline_utils.general.testtime_trajectory_dataclass_util import (
        TestTrajectoryEntry,
    )

    entries = []
    for traj_dict in traj_dicts:
        traj_dict_cfg = omegaconf.OmegaConf.create(traj_dict)
        env = setup_trajectory_from_csv(experiment_cfg_ref, traj_dict_cfg)
        test_motion = robotic_data_to_test_motion_trajectory_dataclass(env)
        entries.append(
            TestTrajectoryEntry(
                env=test_motion,
                trajectory_name=traj_dict["trajectory_name"],
                category=traj_dict.get("category", "unknown"),
                short_name=os.path.basename(traj_dict["trajectory_name"]),
            )
        )
    return entries


logger = logging.getLogger(__name__)
