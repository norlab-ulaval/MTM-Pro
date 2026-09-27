# coding=utf-8
import math
import random
import warnings
from typing import Any, Union, List, TYPE_CHECKING

import gymnasium as gym
import numpy as np
import omegaconf
from gymnasium import Env
from mbrl.util import ReplayBuffer
from omegaconf import DictConfig
from tqdm import tqdm

from algorithm.motion_model.env2sim_components.math_gym_toy_s2s.env_dynamic_sampling import (
    collect_full_time_space_rollout,
)
from math_gymnasium.envs.arbitrary_dim_math_continuous import MathContinuousGymnasium
from math_gymnasium.tools.plot_3d_utils import three_dimension_environment_space_plot
from math_gymnasium.tools.utils import (
    math_continuous_gymnasium_env_to_test_motion_trajectory_dataclass,
)
from pipeline.pipeline_utils.math_env_pipeline_utils.math_env_selector_utils import (
    math_environment_selector,
)
from tools.console_tools.message import consol_msg_universal_one_liner
from tools.hydra_apps_tools.hydra_utils import get_hydra_experiment_id
from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist
from tools.plot_tools.plot_management import (
    manage_matplotlib_backend,
    manage_matplotlib_warnings,
    show_and_save_plot_helper,
)
from trajectory_container_tools.dataclasses import TestMotionTrajectoryDataclass


def _validate_math_test_trajectory_config(
    cfg_key_name: str, cfg_value, allow_empty: bool = False
) -> list:
    """Validate and normalise a math-env ``test_*_trajectory`` cfg value.

    Permanent helper. Introduced by phase 1 of the math_env multi
    test-trajectory `.junie` plan
    (``feature_math_env_multi_test_trajectory_plan_20260517.md``).

    Empty/``null`` support (``feature_empty_ood_test_trajectory_support_plan_
    20260821_RLRP-778.md``): a ``None`` (YAML ``null``) value is normalised to
    an empty list, and an empty list is accepted **only** when
    ``allow_empty=True`` (used for ``test_OOD_trajectory`` so a simulator config
    can legitimately declare zero OOD test trajectories). ``test_InD_trajectory``
    keeps ``allow_empty=False`` and still fails loud on empty/``null``.

    Strict math mirror of
    :func:`pipeline.pipeline_utils.robotic_env_pipeline_utils.setup_utils.
    _validate_test_trajectory_config`: raises :class:`ValueError` with an
    actionable upgrade hint when a legacy single-dict shape
    (``{initiale_coordinates: ..., param: ...}``) is detected. Returns the
    value as a plain :class:`list` of trajectory dicts otherwise.

    :param cfg_key_name: Name of the cfg key being validated (used in the
        error message), e.g. ``"test_InD_trajectory"``.
    :param cfg_value: Value read from ``cfg.environment.data.<key>``.
    :param allow_empty: When ``True``, a ``None``/empty list is accepted and
        normalised to ``[]`` (used for ``test_OOD_trajectory``). When ``False``
        (default), an empty/``null`` value raises :class:`ValueError`.
    :return: ``list`` of trajectory dicts, ready for
        :func:`build_math_test_trajectory_entries`.
    :raises ValueError: When ``cfg_value`` is a single trajectory dict
        (legacy format), or when it is empty/``null`` and ``allow_empty`` is
        ``False``.
    """
    if cfg_value is None:
        cfg_value = []
    if isinstance(cfg_value, (dict, omegaconf.DictConfig)) and (
        "initiale_coordinates" in cfg_value
    ):
        try:
            init_coords = list(cfg_value["initiale_coordinates"])
        except Exception:
            init_coords = cfg_value["initiale_coordinates"]
        raise ValueError(
            f"Legacy single-dict format detected for "
            f"'cfg.environment.data.{cfg_key_name}'.\n"
            f"  Please update your simulator config to the new list-of-dicts "
            f"format:\n"
            f"    {cfg_key_name}:\n"
            f"      - initiale_coordinates: {init_coords}\n"
            f"        param:\n"
            f"          ...\n"
            f"        category: L  # or S, M\n"
            f"  (see lorenz_v6_3XinitC_OoD=burst_noise.yaml for reference)."
        )
    if not isinstance(cfg_value, (list, omegaconf.ListConfig)):
        raise ValueError(
            f"Expected 'cfg.environment.data.{cfg_key_name}' to be a list of "
            f"trajectory dicts, got {type(cfg_value).__name__}."
        )
    if len(cfg_value) == 0 and not allow_empty:
        raise ValueError(
            f"'cfg.environment.data.{cfg_key_name}' is empty; expected at "
            f"least one entry with 'initiale_coordinates' and 'param'."
        )
    return list(cfg_value)


if TYPE_CHECKING:  # this block runs only under mypy/Pyright
    from pipeline.pipeline_utils.general.testtime_trajectory_dataclass_util import \
        TestTrajectoryEntry


def build_math_test_trajectory_entries(
    cfg: DictConfig,
    trajectory_dicts: list,
    target_is_ood: bool,
) -> List["TestTrajectoryEntry"]:
    """Build a ``list[TestTrajectoryEntry]`` from math-env trajectory dicts.

    Counterpart to the robotic env entry builder used by the multirun test-time
    rollout plot pipeline. Each entry pairs a freshly-built
    :class:`MathContinuousGymnasium` (wrapped as a
    :class:`TestMotionTrajectoryDataclass`) with the metadata expected by the
    downstream rollout/plot code (``trajectory_name``, ``category``, ``short_name``).

    :param cfg: Experiment Hydra config (must expose ``cfg.environment.*`` keys
        used by :func:`setup_test_target_rollouts` and
        :func:`math_environment_selector`).
    :param trajectory_dicts: List of math env trajectory dicts, each with keys
        ``initiale_coordinates``, ``param`` and optional ``category``.
    :param target_is_ood: When ``True``, build envs with the OoD measurement-noise
        configuration; otherwise use the InD configuration.
    :return: List of ``TestTrajectoryEntry`` ready for
        ``execute_ms_model_test_time_rollouts``.
    """
    # Deferred import to avoid circular dependency at module load.
    from pipeline.pipeline_utils.general.testtime_trajectory_dataclass_util import (
        TestTrajectoryEntry,
    )

    if target_is_ood:
        if is_cfg_key_exist(cfg.environment, "ood_test_measurement_noise"):
            measurement_noise_cfgs = (
                cfg.environment.ood_test_global_measurement_noise,
                *cfg.environment.ood_test_measurement_noise,
            )
        else:
            measurement_noise_cfgs = cfg.environment.ood_test_global_measurement_noise
    else:
        if is_cfg_key_exist(cfg.environment, "measurement_noise"):
            measurement_noise_cfgs = (
                cfg.environment.global_measurement_noise,
                *cfg.environment.measurement_noise,
            )
        else:
            measurement_noise_cfgs = cfg.environment.measurement_noise

    state_space_label = cfg.environment.data.label

    entries = []
    # Defensive dedup bookkeeping: even with the increased numeric precision
    # in :func:`_math_trajectory_synthetic_name`, two literally identical
    # trajectory configs (or values that collide only after the
    # :func:`sanitize_math_short_name` character replacements) must NOT
    # silently overwrite each other's test-rollout records on disk.
    seen_short_names: dict = {}
    seen_trajectory_names: dict = {}
    for idx, traj_dict in enumerate(trajectory_dicts):
        env_init_cfg = omegaconf.OmegaConf.create(traj_dict)
        state_space_3d_fct = math_environment_selector(cfg, env_init_cfg)
        target_env: Union[MathContinuousGymnasium, gym.Env] = gym.make(
            "math_gymnasium:math-continuous-gymnasium-v0",
            math_function_callback=state_space_3d_fct,
            math_function_label=state_space_label,
            time_axis_cfg=cfg.environment.time_space,
            explorable_regions_cfg=cfg.environment.explorable_space,
            measurement_noise_cfg=measurement_noise_cfgs,
            time_function_callback=None,
            observed_state_are_dt_derivatives=cfg.environment.obs_are_dt_derivatives,
            observed_time_is_delta_time=cfg.environment.obs_time_is_delta_time,
        )
        test_motion = math_continuous_gymnasium_env_to_test_motion_trajectory_dataclass(
            target_env
        )
        trajectory_name = _math_trajectory_synthetic_name(traj_dict, target_is_ood)
        short_name = sanitize_math_short_name(trajectory_name)

        # ---- Defensive dedup -----------------------------------------------------------------
        # If a collision is detected, append a stable positional suffix
        # ``__dupK`` (K = 1-based occurrence index) to both the trajectory
        # name and the short name so artifacts are written to distinct
        # directories. We also emit a warning so the operator can fix the
        # underlying simulator config when the collision was unintended.
        if short_name in seen_short_names or trajectory_name in seen_trajectory_names:
            dup_count = seen_short_names.get(short_name, 0) + 1
            suffix = f"__dup{dup_count}"
            warnings.warn(
                (
                    "build_math_test_trajectory_entries: detected duplicate "
                    f"test-trajectory identifier (trajectory_name={trajectory_name!r}, "
                    f"short_name={short_name!r}) at index {idx}. Appending "
                    f"suffix {suffix!r} to preserve experiment records. "
                    "Check your simulator config for unintended duplicates."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            trajectory_name = f"{trajectory_name}{suffix}"
            short_name = f"{short_name}{suffix}"
            seen_short_names[short_name.rsplit(suffix, 1)[0]] = dup_count

        seen_short_names.setdefault(short_name, 0)
        seen_trajectory_names.setdefault(trajectory_name, 0)
        # --------------------------------------------------------------------------------------

        category = (
            traj_dict.get("category", "unknown")
            if hasattr(traj_dict, "get")
            else "unknown"
        )
        entries.append(
            TestTrajectoryEntry(
                env=test_motion,
                trajectory_name=trajectory_name,
                category=category,
                short_name=short_name,
            )
        )
    return entries


def _math_trajectory_synthetic_name(traj_dict, target_is_ood: bool) -> str:
    """Synthesize a stable trajectory identifier from a math-env init dict.

    The robotic env uses an explicit ``trajectory_name`` key — math env entries
    have ``initiale_coordinates`` + ``param`` instead. We mimic the robotic
    ``train/...`` vs ``test/...`` prefix using the InD/OOD flag, then encode
    initial coordinates + (alphabetically sorted) params, so the resulting
    identifier is unique per math entry while remaining filesystem-friendly.
    """
    if hasattr(traj_dict, "get"):
        init_coords = traj_dict.get("initiale_coordinates", [])
        param = traj_dict.get("param", {})
    else:
        init_coords = traj_dict["initiale_coordinates"]
        param = traj_dict.get("param", {}) if hasattr(traj_dict, "get") else {}
    try:
        init_coords = list(
            omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.create(init_coords))
        )
    except Exception:
        init_coords = list(init_coords)
    try:
        param = dict(
            omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.create(dict(param)))
        )
    except Exception:
        param = dict(param)
    # R5 fix (math_env multi test-trajectory `.junie` plan,
    # ``feature_math_env_multi_test_trajectory_plan_20260517.md``):
    # both InD and OOD entries produced by this helper are *test*
    # trajectories — distinguish them via the InD/OOD axis only.
    prefix = "test_OOD" if target_is_ood else "test_InD"
    # Use 12 significant digits so near-identical numeric values (e.g.
    # ``r=28.00000325`` vs ``r=28.000009``) yield *distinct* identifiers.
    # The previous default ``:g`` (6 sig. digits) collapsed such entries
    # to the same name, causing test-rollout record overwrites
    # (see issue RLRP-XXX / artifact ``2026.05.23/100522/testtime_rollouts``).
    coords_str = "_".join(f"{c:.12g}" for c in init_coords)
    if param:
        params_str = "_".join(f"{k}{param[k]:.12g}" for k in sorted(param.keys()))
        return f"{prefix}/coord_{coords_str}__{params_str}"
    return f"{prefix}/coord_{coords_str}"


def sanitize_math_short_name(trajectory_name: str) -> str:
    """Derive a filesystem-friendly short name from a synthetic math traj name."""
    base = trajectory_name.rsplit("/", 1)[-1]
    return base.replace(".", "p").replace("-", "n")


def setup_test_target_rollouts(
    cfg: DictConfig, exp_dir_relative_path: str | Any, headless: bool
) -> tuple[list, list]:
    """Build the math-env InD and OoD test-target rollouts.

    Permanent setup helper. Introduced by phase 1 of the math_env multi
    test-trajectory `.junie` plan
    (``feature_math_env_multi_test_trajectory_plan_20260517.md``).

    1-to-1 math mirror of
    :func:`pipeline.pipeline_utils.robotic_env_pipeline_utils.setup_utils.
    setup_test_target_rollouts`. The ``exp_dir_relative_path`` and
    ``headless`` parameters are accepted for signature parity with the
    robotic counterpart even if unused for now.

    :return: Pair ``(target_InD_rollouts, target_OOD_rollouts)`` of
        ``list[TestTrajectoryEntry]`` ready for
        :func:`pipeline.pipeline_utils.general.train_and_deploy_utils
        .execute_ms_model_test_time_rollouts`.
    """
    ind_cfg = _validate_math_test_trajectory_config(
        "test_InD_trajectory", cfg.environment.data.test_InD_trajectory
    )
    ood_cfg = _validate_math_test_trajectory_config(
        "test_OOD_trajectory",
        cfg.environment.data.test_OOD_trajectory,
        allow_empty=True,
    )
    return (
        build_math_test_trajectory_entries(cfg, ind_cfg, target_is_ood=False),
        build_math_test_trajectory_entries(cfg, ood_cfg, target_is_ood=True),
    )


def setup_data(
    cfg: DictConfig, exp_dir_relative_path: str | Any, headless: bool
) -> tuple[
    Any,
    list[ReplayBuffer],
    list[TestMotionTrajectoryDataclass],
    list,
    list,
]:
    """Bundle train/val source data with the InD/OoD test-target rollouts.

    Updated by phase 2 of the math_env multi test-trajectory `.junie`
    plan (``feature_math_env_multi_test_trajectory_plan_20260517.md``)
    to return ``list[TestTrajectoryEntry]`` instead of single envs.
    """

    _, ss_full_time_space_replay_buffers, val_env_trjs = (
        setup_train_and_val_source_data(cfg, exp_dir_relative_path, headless)
    )

    target_InD_rollouts, target_OOD_rollouts = setup_test_target_rollouts(
        cfg, exp_dir_relative_path, headless
    )

    state_space_label = cfg.environment.data.label

    return (
        state_space_label,
        ss_full_time_space_replay_buffers,
        val_env_trjs,
        target_InD_rollouts,
        target_OOD_rollouts,
    )


def setup_train_and_val_source_data(
    cfg: DictConfig, exp_dir_relative_path: str | Any, headless: bool
) -> tuple[str, list[ReplayBuffer], list[TestMotionTrajectoryDataclass]]:
    ss_full_time_space_replay_buffers = []
    state_space_label = cfg.environment.data.label
    env_initial_conditions = cfg.environment.data.trajectories
    total_env_rollout = cfg.environment.get("env_rnd_instances", 1) * len(
        env_initial_conditions
    )
    for idx, env_init_cfg in enumerate(env_initial_conditions):
        state_space_3d_fct = math_environment_selector(cfg, env_init_cfg)

        measurement_noise_cfgs, selected_burst_noise_cfgs = setup_burst_noise_config(
            cfg
        )

        learning_env: Union[MathContinuousGymnasium, gym.Env] = gym.make(
            "math_gymnasium:math-continuous-gymnasium-v0",
            math_function_callback=state_space_3d_fct,
            math_function_label=state_space_label,
            time_axis_cfg=cfg.environment.time_space,
            explorable_regions_cfg=cfg.environment.explorable_space,
            measurement_noise_cfg=measurement_noise_cfgs,
            time_function_callback=None,
            # ToDo: RLRP-294 experiment with dt env obs noise and lag
            observed_state_are_dt_derivatives=cfg.environment.obs_are_dt_derivatives,
            observed_time_is_delta_time=cfg.environment.obs_time_is_delta_time,
        )

        # .... Setup source replay buffer .........................................................
        progressbar = tqdm(
            desc="Dataset generation", leave=False, total=total_env_rollout
        )
        # RLRP-824 (dtype hygiene): on the ``pipeline.data_manager: dataloader`` path the
        # single-step rollouts are collected directly in the SOURCE dtype (the normalizer dtype,
        # ``one_dim_transition_model.normalize_double_precision`` -> model dtype when null)
        # instead of the environment's ``float64``; the legacy ``replay-buffer`` path keeps
        # ``None`` (environment dtype, bit-exact).
        from tools.multistep_tools.window_dataset.pipeline_utils import (
            resolve_source_buffer_double_precision,
        )

        source_buffer_double_precision = resolve_source_buffer_double_precision(cfg)
        for each_env_rnd in np.arange(cfg.environment.get("env_rnd_instances", 1)):
            # Note: New noise is generated on environment reset
            ss_full_time_space_replay_buffers.append(
                collect_full_time_space_rollout(
                    learning_env,
                    replace_rewards_with_timestep_index=True,
                    # max_trajectory_length=cfg.UDER
                    # .buffer_max_trajectory_len
                    max_trajectory_length=None,
                    double_precision=source_buffer_double_precision,
                )
            )
            progressbar.update()
        progressbar.close()

        consol_msg_universal_one_liner(
            f"Replay buffer: obs dtype={ss_full_time_space_replay_buffers[0].obs_type}, "
            f"act dtype={ss_full_time_space_replay_buffers[0].action_type} "
            f"(double_precision={source_buffer_double_precision!r}: "
            f"{'source/normalizer dtype, data_manager=dataloader' if source_buffer_double_precision is not None else 'environment dtype, legacy'})"
        )

        if (
            cfg.pipeline.plot.show_environment_plot
            or cfg.pipeline.plot.save_environment_plot
        ):
            with warnings.catch_warnings():
                manage_matplotlib_warnings()
                manage_matplotlib_backend(
                    cfg.pipeline.plot.show_environment_plot, headless
                )

                env_fig, ax_3d, ax_z, ax_x, ax_y = (
                    three_dimension_environment_space_plot(
                        cfg,
                        time_space=learning_env.trj.time_axis.wall,
                        state_space_3d=learning_env.trj.state_axes.poses,
                        state_space_3d_with_noise=learning_env.trj.state_axes.poses_with_noise,
                        title=(
                            f"State space and explored space. Environment configuration {idx + 1}/"
                            f"{len(env_initial_conditions)}"
                        ),
                        state_space_label=state_space_label,
                        subplot_1d_interval=cfg.pipeline.plot.show_environment_1d_subplot_interval,
                        show_samples=True,
                        show_3d_grid=cfg.pipeline.plot.get("show_3d_grid", True),
                        show_3d_axes=cfg.pipeline.plot.get("show_3d_axes", True),
                        figsize=cfg.pipeline.plot.figsize,
                        extra_info_str=(
                            f"{setup_noise_cfg_str(selected_burst_noise_cfgs, cfg.environment.global_measurement_noise)}\n"
                            f"  Initiale coordinates: {env_init_cfg.initiale_coordinates}\n"
                            f"  System params: {env_init_cfg.param}\n"
                        ),
                        experiment_id=get_hydra_experiment_id(),
                    )
                )

                show_and_save_plot_helper(
                    env_fig,
                    exp_dir_relative_path,
                    "state_space_and_explorable_space_samples"
                    f"_env_cfg_{idx + 1}_of_{len(env_initial_conditions)}",
                    headless,
                    cfg.pipeline.plot.show_environment_plot,
                    cfg.pipeline.plot.save_dpi,
                    save=cfg.pipeline.plot.save_environment_plot,
                )
        learning_env.close()

    # .... Create validation environment ..........................................................
    rng = np.random.default_rng(cfg.seed)

    val_env_nb = int(
        math.ceil(total_env_rollout * cfg.deploy.val_rollout.replay_buffer_ratio)
    )
    env_init_cfg_indexes = rng.choice(
        len(env_initial_conditions), size=val_env_nb
    ).tolist()
    val_env_trjs = []
    for each_init_idx in env_init_cfg_indexes:
        env_init_cfg = env_initial_conditions[each_init_idx]
        val_state_space_3d_fct = math_environment_selector(cfg, env_init_cfg)
        val_env: Union[MathContinuousGymnasium, gym.Env] = gym.make(
            "math_gymnasium:math-continuous-gymnasium-v0",
            math_function_callback=val_state_space_3d_fct,
            math_function_label=cfg.environment.data.label,
            time_axis_cfg=cfg.environment.time_space,
            explorable_regions_cfg=cfg.environment.explorable_space,
            measurement_noise_cfg=(
                cfg.environment.global_measurement_noise,
                *cfg.environment.measurement_noise,
            ),
            time_function_callback=None,
            # ToDo: RLRP-294 experiment with dt env obs noise and lag
            observed_state_are_dt_derivatives=cfg.environment.obs_are_dt_derivatives,
            observed_time_is_delta_time=cfg.environment.obs_time_is_delta_time,
        )
        val_rollout = math_continuous_gymnasium_env_to_test_motion_trajectory_dataclass(
            val_env
        )
        val_rollout_trj_length = int(
            val_rollout.trajectory_len * cfg.deploy.val_rollout.trajectory_length_ratio
        )
        val_env_trjs.append(val_rollout[:val_rollout_trj_length])
        val_env.close()

    consol_msg_universal_one_liner(
        f"Validation trajectory rollout: {val_env_nb} X {val_env_trjs[0].trajectory_len} (size X trj len)"
    )
    return state_space_label, ss_full_time_space_replay_buffers, val_env_trjs


def setup_noise_cfg_str(
    selected_burst_noise_cfgs: omegaconf.DictConfig,
    global_noise_cfg: omegaconf.DictConfig,
) -> str:
    noise_str = f"  Global noise: interval={global_noise_cfg.interval} magnitude={global_noise_cfg.magnitude}"
    if selected_burst_noise_cfgs is not None:
        noise_str += f"\n  Burst noise:\n"
        noise_str += "\n".join(
            f"    interval: {item['interval']}, magnitude: {item['magnitude']}"
            for item in selected_burst_noise_cfgs
        )
    return noise_str


def setup_burst_noise_config(
    cfg: omegaconf.DictConfig,
) -> tuple[omegaconf.DictConfig, omegaconf.DictConfig]:
    burst_noise_cfgs = [*cfg.environment.measurement_noise]
    selected_burst_noise_cfgs = None
    burst_noise_cfg_len = len(burst_noise_cfgs)
    if burst_noise_cfg_len >= 1:
        burst_noise_cfg_rnd_pick = random.randint(1, burst_noise_cfg_len)
        selected_burst_noise_cfgs = random.choices(
            burst_noise_cfgs, k=burst_noise_cfg_rnd_pick
        )
        measurement_noise_cfgs = (
            cfg.environment.global_measurement_noise,
            *selected_burst_noise_cfgs,
        )
    else:
        measurement_noise_cfgs = cfg.environment.global_measurement_noise
    return measurement_noise_cfgs, selected_burst_noise_cfgs
