# coding=utf-8
import inspect
import os
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import hydra.utils
from pipeline.pipeline_utils.general.testtime_trajectory_dataclass_util import (
    TestTrajectoryEntry,
)

import mbrl.models

import numpy as np
import omegaconf
import torch
from gymnasium import Env

from mbrl.models import ModelTrainer, OneDTransitionRewardModel
from mbrl.util import ReplayBuffer
from omegaconf import DictConfig
from tqdm import tqdm

from algorithm.experience_replay_learning_loop.core.training_callback import (
    fetch_cfg_pipeline_tensorboard_key_value,
    setup_batch_callback_aggregator,
    setup_compounded_pred_unroll_len_batch_callback,
    setup_erll_epoch_pre_training_callback_aggregator,
    setup_gradient_monitoring_callback,
    setup_multistep_loss_batch_callback,
    setup_tensorboard_arbitrary_dimension_trajectory_prediction_monitor_callback,
    setup_train_time_domain_randomization_scheduler_callback,
    setup_trainer_epoch_callback_aggregator,
    setup_trainer_lr_scheduler_callback,
)
from algorithm.motion_model.env2sim_components.uder_tools import (
    restrict_replay_buffer_to_explorable_region,
)
from algorithm.motion_model.env2sim_components.math_gym_toy_s2s.math_env_training_callback import (
    setup_math_env_uder_buffer_sampling_plotter_callback,
)
from algorithm.motion_model.utils.inference_optimization_utils import (
    optimize_model_speed,
)
from math_gymnasium.envs.arbitrary_dim_math_continuous import MathContinuousGymnasium
from pipeline.pipeline_utils.robotic_env_pipeline_utils.robotic_trajectory_dataclass import (
    obs_2_vector,
)
from pipeline.pipeline_utils.robotic_env_pipeline_utils.utils import (
    compute_position_from_velocity_and_attitude,
)
from tools.console_tools.message import consol_msg_universal_one_liner
from tools.console_tools.progressbar_tools import init_progressbar
from tools.dna_dev_tools.dn_CI_tools import show_plot_unless_CI_server_runned
from tools.dna_dev_tools.dn_pytest_tools import is_pytest_run
from tools.f110_gym_env_tools.plot_tensorboard import OnlineTensorboardWritter
from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist
from tools.hydra_apps_tools.r2s2r_apps_utils import R2S2RPipelineHydraApp

from tools.mbrl_lib_tools.models.one_dim_tr_model_v2 import OneDTransitionRewardModelV2
from tools.mbrl_lib_tools.replaybuffer_tools import aggregate_replay_buffer
from tools.mbrl_lib_tools.setup_utils import (
    resolve_mbrl_trainer_opt_in_kwargs_from_cfg,
)
from tools.multistep_tools.data_buffer_processor.multistep_data_buffer_processor_arbitrary_dim import (
    MultistepDataBufferProcessorArbitraryDimension,
)
from tools.multistep_tools.models import MultiStepMLP
from tools.multistep_tools.replaybuffer_singlestep_to_multistep_converter import (
    convert_singlestep_replaybuffer_to_multistep,
)
from tools.r2s_motion_model_container_tools.container import R2SMotionModelContainer
from tools.r2s_motion_model_container_tools.factory import (
    R2SMotionModelFactorySpec,
    r2s_motion_model_container_factory,
)
from tools.torch_tools.optimizer_instantiation import change_optimizer
from trajectory_container_tools.dataclasses import TestMotionTrajectoryDataclass


def get_model_description(
    model_cfg: DictConfig,
    motion_model: Union[OneDTransitionRewardModel, R2SMotionModelContainer],
) -> str:
    if isinstance(motion_model, R2SMotionModelContainer):
        motion_model = motion_model.dynamics_model

    return model_cfg.get(
        "description",
        motion_model.model.__class__.__name__,
    )


def setup_source_single_step_replay_buffer(
    cfg: omegaconf.DictConfig, ss_full_time_space_replay_buffers: List[ReplayBuffer]
) -> ReplayBuffer:
    # RLRP-736 shape-key removal (2026-07-17): ``environment.obs_shape`` /
    # ``act_shape`` are no longer hand-maintained config keys. The single-step
    # (SS/legacy) pipeline instantiates ``cfg.ss_model`` — whose ``in_size`` /
    # ``out_size`` still interpolate ``${environment.obs_shape[0]}`` /
    # ``${environment.act_shape[0]}`` — BEFORE the multi-step buffer setup (which
    # is where the shapes get stamped from data). Stamp them here too, from the
    # loaded DATA (math source of truth), so the SS-model interpolation resolves.
    from tools.feature_handling_tools.env_handlers import (
        resolve_act_shape,
        resolve_obs_shape,
    )

    ss_obs_shape = ss_full_time_space_replay_buffers[0].obs_shape[-1]
    ss_act_shape = ss_full_time_space_replay_buffers[0].action_shape[-1]
    resolved_obs_shape = resolve_obs_shape(cfg, data_shape=[ss_obs_shape])
    resolved_act_shape = resolve_act_shape(cfg, data_shape=[ss_act_shape])
    with omegaconf.read_write(cfg), omegaconf.open_dict(cfg):
        omegaconf.OmegaConf.update(
            cfg, "environment.obs_shape", resolved_obs_shape, merge=False
        )
        omegaconf.OmegaConf.update(
            cfg, "environment.act_shape", resolved_act_shape, merge=False
        )

    progressbar = tqdm(
        np.arange(len(ss_full_time_space_replay_buffers)),
        desc="Restrict SS replay buffer to explorable region",
        leave=False,
    )
    ss_explorable_region_replay_buffers = []
    for each in ss_full_time_space_replay_buffers:

        if cfg.environment.get("explorable_space", None):
            ss_explorable_region_replay_buffers.append(
                restrict_replay_buffer_to_explorable_region(cfg, each)
            )
        else:
            ss_explorable_region_replay_buffers.append(each)
        progressbar.update()

    progressbar.close()

    ss_explorable_region_replay_buffers = aggregate_replay_buffer(
        ss_explorable_region_replay_buffers
    )

    consol_msg_universal_one_liner(
        f"Original dataset size: {ss_explorable_region_replay_buffers.num_stored}"
    )
    with omegaconf.read_write(cfg):
        omegaconf.OmegaConf.update(
            cfg,
            "source_replay_buffer.original_dataset_size",
            ss_explorable_region_replay_buffers.num_stored,
            merge=False,
        )

    return ss_explorable_region_replay_buffers


def setup_source_multi_step_replay_buffer(
    cfg: omegaconf.DictConfig, ss_full_time_space_replay_buffers: List[ReplayBuffer]
) -> Union[MultistepDataBufferProcessorArbitraryDimension, ReplayBuffer]:

    from tools.feature_handling_tools.env_handlers import (
        resolve_act_shape,
        resolve_obs_shape,
    )

    ss_obs_shape = ss_full_time_space_replay_buffers[0].obs_shape[-1]
    ss_act_shape = ss_full_time_space_replay_buffers[0].action_shape[-1]
    # RLRP-736 shape-key removal (2026-07-17): ``environment.obs_shape`` /
    # ``act_shape`` are no longer hand-maintained config keys. The DATA is the
    # source of truth here; ``resolve_{obs,act}_shape`` returns ``[len(dims)]`` for
    # robotic envs (and fail-loud-checks agreement with the data) or, for math envs
    # (no dims), the data-derived shape passed in. We then STAMP the resolved shape
    # back into ``cfg.environment`` so all downstream consumers keep working with a
    # value now DERIVED from data (or dims) rather than a hand-copied literal.
    resolved_obs_shape = resolve_obs_shape(cfg, data_shape=[ss_obs_shape])
    resolved_act_shape = resolve_act_shape(cfg, data_shape=[ss_act_shape])
    # ``open_dict`` so the key can be (re-)ADDED when absent (the config no longer
    # declares it) even under OmegaConf struct mode.
    with omegaconf.read_write(cfg), omegaconf.open_dict(cfg):
        omegaconf.OmegaConf.update(
            cfg, "environment.obs_shape", resolved_obs_shape, merge=False
        )
        omegaconf.OmegaConf.update(
            cfg, "environment.act_shape", resolved_act_shape, merge=False
        )

    ms_data_buffer_processor_3D = MultistepDataBufferProcessorArbitraryDimension(
        history_len=cfg.ms_model.history_len,
        obs_dim=ss_obs_shape,
        act_dim=ss_act_shape,
        obs_dtype=ss_full_time_space_replay_buffers[0].obs_type,
        act_dtype=ss_full_time_space_replay_buffers[0].action_type,
        horizon_len=cfg.ms_model.horizon_len,
        consol_log=False,
    )

    progressbar = init_progressbar(
        len(ss_full_time_space_replay_buffers),
        "Convert to MS replay buffer and restrict to explorable region",
    )

    ms_explorable_region_replay_buffers = []
    for each in ss_full_time_space_replay_buffers:
        each_ms_full_time_space_replay_buffer = (
            convert_singlestep_replaybuffer_to_multistep(
                each,
                ms_data_buffer_processor_3D,
                max_trajectory_length=None,
            )
        )

        if cfg.environment.get("explorable_space", None):
            ms_explorable_region_replay_buffers.append(
                restrict_replay_buffer_to_explorable_region(
                    cfg, each_ms_full_time_space_replay_buffer
                )
            )
        else:
            ms_explorable_region_replay_buffers.append(
                each_ms_full_time_space_replay_buffer
            )

        progressbar.update()

    progressbar.close()

    ms_explorable_region_replay_buffers = aggregate_replay_buffer(
        ms_explorable_region_replay_buffers
    )

    consol_msg_universal_one_liner(
        f"DONE converting SS to MS replay buffer. Multistep replay buffer size: {ms_explorable_region_replay_buffers.num_stored}"
    )

    with omegaconf.read_write(cfg):
        omegaconf.OmegaConf.update(
            cfg,
            "source_replay_buffer.original_dataset_size",
            ms_explorable_region_replay_buffers.num_stored,
            merge=False,
        )

    return ms_data_buffer_processor_3D, ms_explorable_region_replay_buffers


def setup_tensorboard_writer(cfg, model_name: str):
    if show_plot_unless_CI_server_runned(True) and not is_pytest_run():
        tensorboard_plot_callback = OnlineTensorboardWritter(
            cfg,
            reset_tmp_dir=cfg.get("reset_tensorboard_dir", False),
            clear_tmp_directory_run_artifact_on_teardown=fetch_cfg_pipeline_tensorboard_key_value(
                cfg, "clear_tmp_directory_run_artifact_on_teardown", False
            ),
            comment=model_name,
        )
        tensorboard_plot_callback.add_hydra_override_text()
    else:
        tensorboard_plot_callback = None
    return tensorboard_plot_callback


def _ctor_accepts_kwarg(cls: type, name: str) -> bool:
    """``True`` when ``cls.__init__`` (or its MRO) declares ``name`` or a ``**kwargs`` catch-all."""
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # builtins / C extensions: assume the kwarg is forwarded
        return True
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _apply_meta_collection_flag(cfg: DictConfig, model) -> None:
    """Attach the diagnostic ``meta`` collection master switch post-construction.

    Permanent model-setup seam. Introduced by action ``A7`` of the RLRC
    meta-collection kill-switch `.junie` plan
    (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``). Reads
    ``pipeline.tensorboard.enable_meta_collection`` (absent -> ``True``, today's
    bit-exact behaviour) and forwards it through the model's
    ``set_meta_collection_enabled`` setter (mirrors the ``set_training_frame`` and
    ``set_feature_handler`` seams). Fully defensive: a model family that does not
    expose the setter is a no-op, so an unedited config can never break.
    """
    from algorithm.experience_replay_learning_loop.core.training_callback import (
        fetch_cfg_pipeline_tensorboard_key_value,
    )

    # RLRP-786 (FR8 / Key Decision 8, ``perf_RLRP-786_cuda_graph_train_step_plan_20260919.md``):
    # ``pipeline.performance_mode: dev|fast`` owns the dev/debug-only instrumentation of the
    # training step. Resolved ONCE here and forwarded through the model's
    # ``set_performance_mode`` setter (``PerformanceModeMixin``: MTM-Pro + AR MS2SS families;
    # other families: no-op). Under ``fast`` the
    # diagnostic ``meta`` collection DEFAULTS to off -- an explicit
    # ``tensorboard.enable_meta_collection`` stays authoritative in both modes.
    performance_mode = resolve_performance_mode(cfg)
    _mode_setter = getattr(model, "set_performance_mode", None)
    if _mode_setter is not None:
        _mode_setter(performance_mode)

    _meta_setter = getattr(model, "set_meta_collection_enabled", None)
    if _meta_setter is not None:
        _meta_setter(
            fetch_cfg_pipeline_tensorboard_key_value(
                cfg,
                key="enable_meta_collection",
                key_value_default=(performance_mode == "dev"),
            )
        )
    return None


PERFORMANCE_MODES = ("dev", "fast")


def resolve_performance_mode(cfg: DictConfig) -> str:
    """Read ``pipeline.performance_mode`` (absent -> ``dev``, today's behaviour) and validate it.

    RLRP-786 FR8 (``perf_RLRP-786_cuda_graph_train_step_plan_20260919.md``). ``dev`` keeps every
    development / debug diagnostic of the training step; ``fast`` is the paper-run mode and the
    precondition of the CUDA-graph captured step. Both are ``torch.equal`` on loss / grads / Adam
    state (the switch only removes instrumentation, never arithmetic).
    """
    mode = omegaconf.OmegaConf.select(cfg, "pipeline.performance_mode", default=None)
    mode = "dev" if mode is None else str(mode).lower()
    if mode not in PERFORMANCE_MODES:
        raise ValueError(
            f"pipeline.performance_mode={mode!r} is not one of {PERFORMANCE_MODES}"
        )
    return mode


def setup_single_step_model(
    cfg: omegaconf.DictConfig,
    rng: Optional[torch.Generator],
    disable_compile: bool = False,
) -> Tuple[OneDTransitionRewardModelV2, mbrl.models.ModelTrainer]:
    single_step_model_ensemble: mbrl.models.Ensemble = hydra.utils.instantiate(
        cfg.ss_model, _recursive_=False
    )

    # A7 (RLRP-788): attach the diagnostic ``meta`` collection master switch
    # post-construction (same seam as the multi-step site). Absent config key ->
    # ``True`` (today's behaviour, bit-exact). Introduced by action ``A7`` of the
    # RLRC meta-collection kill-switch `.junie` plan
    # (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
    _apply_meta_collection_flag(cfg, single_step_model_ensemble)

    # RLRP-736 S3.11 (§14C): make the handler's normalizer-level (A) + model-input
    # (B) handling live on the TRAINING path. The wrapper is built via
    # ``hydra.utils.instantiate`` (NOT ``create_one_dim_tr_model_v2``, which is
    # only reached from the model-load/deploy path), so the S3.7/S3.8 merge must
    # be applied here as instantiate overrides. Empty (bit-exact) for the neutral
    # scalar / math handler.
    from tools.feature_handling_tools.env_handlers import (
        apply_feature_handler_to_transition_model,
        resolve_feature_handler_transition_model_overrides,
    )

    _ss_feature_overrides = resolve_feature_handler_transition_model_overrides(
        cfg, cfg.one_dim_transition_model
    )
    ss_1D_transition_model: OneDTransitionRewardModelV2 = hydra.utils.instantiate(
        cfg.one_dim_transition_model,
        model=single_step_model_ensemble,
        learned_rewards=False,
        _recursive_=False,
        **_ss_feature_overrides,
    )

    # RLRP-736 S3.12 (§14C): single post-construction feature-handler seam
    # (loss-handler registration + completeness marker), shared with the
    # multi-step site and the load-path builder. Previously the single-step path
    # never registered the handler (the "1 of 3 sites" gap); registration is a
    # no-op for today's hookless single-step ensembles (GaussianMLP), so this is
    # a consistency fix, not a behavioural change.
    apply_feature_handler_to_transition_model(
        cfg, ss_1D_transition_model, single_step_model_ensemble
    )

    if not disable_compile and (
        is_cfg_key_exist(cfg, "UDER.batch_size.limit")
        and is_cfg_key_exist(cfg, "UDER.batch_size.init_value")
        and cfg.UDER.batch_size.limit == cfg.UDER.batch_size.init_value
    ):
        ss_1D_transition_model = optimize_model_speed(ss_1D_transition_model)

    ss_trainer = mbrl.models.ModelTrainer(
        ss_1D_transition_model,
        optim_lr=cfg.ss_training.model_lr,
        weight_decay=cfg.ss_training.model_wd,
        # Training Speed & Efficiency plan (B0-bis + F-C2) — opt-in
        # kwargs resolved from `cfg.mbrl_lib.*`. All flags default to
        # legacy behaviour so current experiments stay bit-exact with
        # the pre-patch code path.
        **resolve_mbrl_trainer_opt_in_kwargs_from_cfg(cfg),
    )

    ss_trainer = change_optimizer(cfg.ss_training, ss_trainer)

    return ss_1D_transition_model, ss_trainer


def _register_feature_handler_on_model(cfg: DictConfig, model) -> None:
    """Instantiate the per-environment feature handler and register it on ``model``.

    RLRP-736 S1.3 — the single choke-point wiring of the centralized
    per-environment feature-handling contract onto the model's opt-in
    geometry-loss hook (:meth:`ExponentialFamilyMLP.set_feature_handler`).

    Selection mirrors the existing Hydra ``_target_`` hooks: when
    ``cfg.environment.feature_handler`` is present it is instantiated via
    ``hydra.utils.instantiate``; otherwise we fall back to the neutral
    scalar-only handler (``build_default_scalar_handler``).

    Geometry-loss weight resolution: the weight is **only** a model constructor
    parameter (``ms_model.feature_geometry.loss_weight``, surfaced on
    :class:`FeatureGeometryLossMixin` via ``MultiStepMLP``). There is no
    ``ms_training`` config-key override — the model keeps the value it was
    constructed with (default ``0.0`` -> OFF / bit-exact).

    Fully defensive: if the model does not expose ``set_feature_handler`` (e.g.
    a non-``ExponentialFamilyMLP`` variant) this is a no-op.
    """
    setter = getattr(model, "set_feature_handler", None)
    if setter is None:
        return None

    # RLRP-736 S1.5: single centralized selection point (shared with the deploy
    # seam) — instantiates ``cfg.environment.feature_handler`` or falls back to
    # the neutral scalar-only handler.
    #
    # RLRP-736 S3.12 (§14C): this remains a thin, registration-only shim kept for
    # back-compat. The canonical construction seam is now
    # ``apply_feature_handler_to_transition_model`` (which also stamps the
    # completeness marker and is called at every construction site).
    from tools.feature_handling_tools.env_handlers import (
        instantiate_feature_handler,
    )

    handler = instantiate_feature_handler(cfg)

    setter(handler)
    return None


#: Both ``quaternion`` (normalized) and ``quaternion_legacy`` (raw, pre-RLRP-736
#: un-normalized contract) are NEUTRAL 4-D passthroughs: no continuous-rep switch,
#: no orientation-slot derivation, no activation assertion.
_NEUTRAL_ORIENTATION_REPS = ("quaternion", "quaternion_legacy")


def _read_requested_orientation_rep(cfg: DictConfig) -> str:
    """Return ``ms_model.internal_orientation.representation`` (default neutral)."""
    if is_cfg_key_exist(cfg, "ms_model.internal_orientation.representation"):
        return str(cfg.ms_model.internal_orientation.representation)
    return "quaternion"


def resolve_active_orientation_slots(cfg: DictConfig) -> Optional[tuple]:
    """Resolve the orientation slot(s) to thread into the leaf model ctor.

    RLRP-736 orientation-slots auto-inference (2026-07-17): the by-construction
    internal-orientation head's attitude slot is not a user-facing config key.
    When ``ms_model.internal_orientation.representation`` requests an ACTIVE
    representation, the single-step base slot(s) are derived from
    ``environment.obs_dims`` (the single source of truth, honoring user-disabled
    dims) and validated against ``ms_model.singlestep_obs_len``.

    RLRP-793 ``D-793-5`` (2026-08-30): extracted from the body of
    :func:`setup_multistep_step_model` so the fail-loud branches are reachable
    from a test, and the "no slots" message now names the orientation block that
    is ACTUALLY present. A 9-D gravity observation space (RLRP-753) declares a
    3-D ``gravity.*`` block instead of the 4-D quaternion, and
    ``_assert_single_orientation_block`` REJECTS declaring ``attitude.*``
    alongside it — so the legacy "declare the attitude block" advice pointed at a
    configuration that cannot exist.

    RLRP-796 Stage 1 (``D-796-S1-5-A``): the resolver is orientation-group-driven,
    so a gravity observation space now RESOLVES a layout (3-D external slot,
    ``GRAVITY_DIRECTION``) instead of returning nothing. The former blanket "no
    ``InternalOrientationRep`` can consume a 3-vector" refusal is therefore
    replaced by a declared rep x block compatibility check, which is symmetric:
    it also catches an ``S²`` rep requested on an attitude observation space.

    :param cfg: The resolved experiment configuration.
    :return: ``None`` for a NEUTRAL representation (nothing is threaded → the
        head stays bit-exact OFF), else the validated
        :class:`~tools.feature_handling_tools.env_handlers.OrientationSlotLayout`
        (a ``tuple`` subclass of the base slot indices).
    :raise ValueError: when an active representation is requested but no
        orientation block is present, when the requested representation cannot
        consume the block that IS present, or when the derived slot does not fit
        the declared single-step observation length.
    """
    from tools.feature_handling_tools.env_handlers import (
        resolve_orientation_singlestep_slots,
    )
    from tools.feature_handling_tools.feature_spec import (
        InternalOrientationRep,
        orientation_rep_supported_kinds,
    )

    requested_rep = _read_requested_orientation_rep(cfg)
    if requested_rep in _NEUTRAL_ORIENTATION_REPS:
        return None

    obs_dims = (
        list(cfg.environment.obs_dims)
        if is_cfg_key_exist(cfg, "environment.obs_dims")
        else []
    )
    orientation_slots = resolve_orientation_singlestep_slots(cfg)
    if not orientation_slots:
        raise ValueError(
            f"ms_model.internal_orientation.representation={requested_rep!r} "
            f"requests an active by-construction orientation head, but no "
            f"orientation block is present in environment.obs_dims={obs_dims}. "
            f"Either declare an orientation block — attitude "
            f"('attitude.w/x/y/z', S^3) or gravity ('gravity.x/y/z', S^2) — or use "
            f"representation='quaternion'/'quaternion_legacy' "
            f"(RLRP-736 orientation-"
            f"slots auto-inference)."
        )
    # RLRP-796 ``D-796-S1-5-A``: rep x block compatibility, from the declared
    # table rather than an ad-hoc gravity branch. This replaces the former blanket
    # refusal, which claimed no representation could ever consume a 3-vector —
    # true before Stage 1, false now — and which could only ever catch ONE of the
    # two possible mismatches. The check is symmetric: a quaternion-family rep on
    # a gravity observation space AND an S^2 rep on an attitude observation space
    # both fail loud here, naming the requested rep, the block actually present
    # and the reps compatible with it.
    supported = orientation_rep_supported_kinds(requested_rep)
    if orientation_slots.kind not in supported:
        compatible = sorted(
            rep.value
            for rep in InternalOrientationRep
            if orientation_slots.kind in orientation_rep_supported_kinds(rep)
        )
        raise ValueError(
            f"ms_model.internal_orientation.representation={requested_rep!r} "
            f"cannot consume the orientation block present in "
            f"environment.obs_dims={obs_dims}: that block is "
            f"{orientation_slots.kind.value!r} "
            f"({orientation_slots.external_width}-D external slot), while "
            f"{requested_rep!r} requires one of "
            f"{sorted(k.value for k in supported)}. Declaring both "
            f"'attitude.w/x/y/z' and 'gravity.x/y/z' is rejected by the "
            f"orientation-block exclusivity guard, so switching blocks is NOT "
            f"the fix. Representations compatible with this block: "
            f"{compatible} (RLRP-796 'D-796-S1-5-A', RLRP-793 'D-793-5')."
        )
    ss_obs_len = int(cfg.ms_model.singlestep_obs_len)
    if obs_dims and ss_obs_len != len(obs_dims):
        raise ValueError(
            f"ms_model.singlestep_obs_len={ss_obs_len} disagrees with "
            f"len(environment.obs_dims)={len(obs_dims)}; the orientation slot is "
            f"derived from obs_dims and would be mis-placed (RLRP-736)."
        )
    if int(orientation_slots[0]) >= ss_obs_len:
        raise ValueError(
            f"Derived orientation base slot {orientation_slots[0]} does not fit "
            f"within ms_model.singlestep_obs_len={ss_obs_len} (RLRP-736)."
        )
    return orientation_slots


def setup_multistep_step_model(
    cfg: DictConfig,
    load_pretrained_path: Optional[str],
    cfg_model_key: str = None,
    disable_compile: bool = False,
) -> R2SMotionModelContainer:

    if load_pretrained_path:
        assert os.path.exists(
            load_pretrained_path
        ), f"Pretrained path {load_pretrained_path} does not exist"

        if cfg_model_key is None:
            for each_key in ["ms_model", "ss_model", "model"]:
                if is_cfg_key_exist(cfg, each_key):
                    cfg_model_key = each_key
                    break

        experiment_path = os.path.dirname(load_pretrained_path)
        experiment_hydra_cfg_path = os.path.join(experiment_path, ".hydra/config.yaml")
        experiment_hydra_cfg = omegaconf.OmegaConf.load(experiment_hydra_cfg_path)
        omegaconf.OmegaConf.update(
            cfg, cfg_model_key, experiment_hydra_cfg.get(cfg_model_key), merge=False
        )

    # RLRP-736 shape-key removal: resolve the single-step obs/act shape from the
    # single source of truth (robotic: ``obs_dims`` / ``act_dims``; math: the value
    # already stamped from the loaded data by ``setup_source_multi_step_replay_buffer``,
    # falling back to a still-present explicit key). Never index the raw config key.
    from tools.feature_handling_tools.env_handlers import (
        resolve_act_shape,
        resolve_obs_shape,
    )

    _target_obs_shape = resolve_obs_shape(cfg)
    _target_act_shape = resolve_act_shape(cfg)
    # RLRP-736 shape-key removal (2026-07-17): ``environment.obs_shape`` /
    # ``act_shape`` are no longer declared in the simulator configs, but several
    # ms_model config keys still interpolate them (e.g.
    # ``ms_model.singlestep_obs_len: ${environment.obs_shape[0]}``). Robotic
    # deploy / full-pipeline runs that LOAD a pre-saved MS replay buffer bypass
    # ``setup_source_multi_step_replay_buffer`` (which stamps the shapes from
    # data), so the key would still be missing here. Stamp the resolved shape
    # (robotic: from ``obs_dims``/``act_dims``; math: from the value already
    # stamped from data) so every downstream interpolation resolves. Idempotent:
    # re-stamping the same value is a no-op and ``resolve_*`` fail-loud-checks any
    # drift with a still-present explicit key.
    with omegaconf.read_write(cfg), omegaconf.open_dict(cfg):
        omegaconf.OmegaConf.update(
            cfg, "environment.obs_shape", _target_obs_shape, merge=False
        )
        omegaconf.OmegaConf.update(
            cfg, "environment.act_shape", _target_act_shape, merge=False
        )
    # RLRP-824: the composed OUTPUT window ``W`` is a property of the model FAMILY
    # (``history_len`` for every legacy family, ``max(H, F)`` for the MS->MS forecast family),
    # resolved from the Hydra ``_target_`` class BEFORE instantiation so the container sizes
    # ``ms_model.out_size`` as ``Do*W + Da*(W-1)``. ``W == H`` for every ``F <= H`` config.
    _ms_model_cls = hydra.utils._locate(cfg.ms_model._target_)
    _output_window_len = (
        _ms_model_cls.resolve_output_window_len(
            cfg.ms_model.history_len, cfg.ms_model.horizon_len
        )
        if issubclass(_ms_model_cls, MultiStepMLP)
        else None
    )
    motion_model_container = r2s_motion_model_container_factory(
        R2SMotionModelFactorySpec(
            singlestep_obs_len=cfg.ms_model.singlestep_obs_len,
            singlestep_act_len=cfg.ms_model.singlestep_act_len,
            multistep_len=cfg.ms_model.history_len,
            target_env_obs_shape=_target_obs_shape,
            target_env_next_obs_shape=_target_obs_shape,
            target_env_act_shape=_target_act_shape,
            learned_rewards=False,
            output_window_len=_output_window_len,
        ),
        target_env_to_model_ss_in_obs_adapter=lambda x: x,
        model_ss_out_to_target_env_next_obs_adapter=lambda next_obs, info, last_env_state: (
            next_obs,
            info,
        ),
    )

    with omegaconf.read_write(cfg):
        # Update cfg with model [in|out]_size
        omegaconf.OmegaConf.update(
            cfg,
            "ms_model.in_size",
            motion_model_container.motion_model_ms_in_size,
            merge=False,
        )
        omegaconf.OmegaConf.update(
            cfg,
            "ms_model.out_size",
            motion_model_container.motion_model_ms_out_size,
            merge=False,
        )
    # RLRP-736 orientation-slots auto-inference (2026-07-17): the by-construction
    # internal-orientation head's orientation slot is no longer a user-facing config
    # key. When the ms_model config requests an active representation, derive the
    # orientation block's single-step base slot(s) from ``environment.obs_dims``
    # (single source of truth, honoring user-disabled dims) for the orientation block
    # (quaternion or gravity) and pass it into the leaf ctor via the dedicated
    # internal ``orientation_singlestep_slots`` argument. Group-less /
    # ``quaternion`` configs pass nothing -> neutral / bit-exact OFF.
    _instantiate_kwargs = {}
    _orientation_slots = resolve_active_orientation_slots(cfg)
    if _orientation_slots is not None:
        _instantiate_kwargs["orientation_singlestep_slots"] = _orientation_slots

    # RLRP-758 (T6/T8b): the velocity ``training_frame`` is a model-internal fact
    # (``ms_model.training_frame``: world legacy | body new default;
    # gravity_aligned = heading frame, RLRP-792). It is NOT a constructor kwarg of the
    # model families, so strip it from the config node handed to
    # ``hydra.utils.instantiate`` (which would otherwise forward it as an unknown
    # ctor kwarg) and instead attach it POST-construction via ``set_training_frame``
    # (mirrors the ``set_feature_handler`` seam below). Absent -> resolver default
    # (``body``); the same key drives the ingestion-time frame conversion.
    from pipeline.pipeline_utils.robotic_env_pipeline_utils.utils import (
        resolve_training_frame_from_cfg,
    )

    _training_frame = resolve_training_frame_from_cfg(cfg)
    _ms_model_cfg = cfg.ms_model
    if is_cfg_key_exist(cfg, "ms_model.training_frame"):
        _ms_model_cfg = _ms_model_cfg.copy()
        with omegaconf.open_dict(_ms_model_cfg):
            del _ms_model_cfg["training_frame"]

    # ....(DEV) this is a prototyping quick-hack for benchmarking ↓↓...............................
    # RLRP-824 FR9 follow-up (precision parity, 2026-09-17): ``ms_model.mixed_precision`` (autocast
    # over training_step / validation_step) is read by the mbrl ``Model`` base from the constructed
    # ensemble (``resolve_autocast_dtype``), but only the MS->MS forecast family exposes it as a
    # constructor kwarg. For every other family (AR-TCN / AR-GRU / AR-LSTM / AR-MLP / MTM-Pro) strip
    # it from the node handed to ``hydra.utils.instantiate`` and attach it POST-construction, so the
    # same Hydra key drives bf16 on every baseline (``null`` / absent -> bit-exact fp32, unchanged).
    _mixed_precision_post = None
    if is_cfg_key_exist(cfg, "ms_model.mixed_precision") and not _ctor_accepts_kwarg(
        _ms_model_cls, "mixed_precision"
    ):
        _mixed_precision_post = cfg.ms_model.mixed_precision
        if _ms_model_cfg is cfg.ms_model:
            _ms_model_cfg = _ms_model_cfg.copy()
        with omegaconf.open_dict(_ms_model_cfg):
            del _ms_model_cfg["mixed_precision"]

    multistep_step_model_ensemble: MultiStepMLP = hydra.utils.instantiate(
        _ms_model_cfg, _recursive_=False, **_instantiate_kwargs
    )
    if _mixed_precision_post is not None:
        # Fail fast on an unsupported value (same accepted set as the MS->MS ``_coerce_mixed_precision``).
        multistep_step_model_ensemble.mixed_precision = _mixed_precision_post
        multistep_step_model_ensemble.resolve_autocast_dtype()
    # .....................................................................................(DEV)...

    # RLRP-758 (T8b): attach the resolved training frame post-construction so it is
    # persisted with the checkpoint and guarded on load. For model families that do
    # NOT expose ``set_training_frame`` the frame can be neither stamped nor guarded;
    # that is tolerable ONLY for the default (``body``), but an EXPLICIT non-default
    # ``ms_model.training_frame`` on such a family would be silently dropped — fail
    # loud instead (RLRP-758 merit review, concern #3).
    _frame_setter = getattr(multistep_step_model_ensemble, "set_training_frame", None)
    if _frame_setter is not None:
        _frame_setter(_training_frame)
    elif is_cfg_key_exist(cfg, "ms_model.training_frame"):
        raise RuntimeError(
            f"ms_model.training_frame={_training_frame!r} was set explicitly but the "
            f"constructed model {type(multistep_step_model_ensemble).__name__} does "
            f"not expose `set_training_frame` (no TrainingFrameMixin in its MRO), so "
            f"the frame cannot be persisted/guarded with the checkpoint. Use a model "
            f"family that supports it, or drop the override (RLRP-758)."
        )

    # A7 (RLRP-788): attach the diagnostic ``meta`` collection master switch
    # post-construction (mirrors the ``set_training_frame`` seam above), so every
    # ``meta[...]`` writer in the model MRO is reached by inheritance rather than
    # through 30+ constructor kwargs. Absent config key -> ``True`` (today's
    # behaviour, bit-exact). Defensive: models that predate the setter are a no-op.
    # Introduced by action ``A7`` of the RLRC meta-collection kill-switch `.junie`
    # plan (``perf_RLRP-788_meta_collection_killswitch_plan_20260827.md``).
    _apply_meta_collection_flag(cfg, multistep_step_model_ensemble)

    # RLRP-736 fail-loud activation assertion (anti-leak): because
    # ``orientation_singlestep_slots`` is threaded explicitly through the whole model
    # MRO, a single missed forward would silently leave the head inactive (an inert
    # ``sixd`` head — the exact bug this removes). When an active rep was requested,
    # assert the constructed model actually reports ``_internal_orientation_rep_active`` and that
    # its tracked slots are non-empty; otherwise fail construction loudly.
    # NOTE: ``_internal_orientation_rep_active is False`` has two distinct causes and
    # gates only the internal-representation lift (encode/decode + trunk-width
    # expansion), NOT the presence of orientation in the state: (1) a passthrough rep
    # (``quaternion`` / ``quaternion_legacy``, external 4-D == internal 4-D), or
    # (2) an unresolved orientation slot (e.g. a gravity block). Only a requested
    # ACTIVE rep is asserted here.
    _requested_rep = _read_requested_orientation_rep(cfg)
    if _requested_rep not in _NEUTRAL_ORIENTATION_REPS:
        _built = getattr(multistep_step_model_ensemble, "model", None) or (
            multistep_step_model_ensemble
        )
        _active = getattr(_built, "_internal_orientation_rep_active", None)
        _in_slots = getattr(_built, "_ori_in_slots", ())
        _out_slots = getattr(_built, "_ori_out_slots", ())
        if not (_active and (_in_slots or _out_slots)):
            raise RuntimeError(
                f"Orientation head requested (representation={_requested_rep!r}) but "
                f"the constructed model {type(multistep_step_model_ensemble).__name__} "
                f"reports _internal_orientation_rep_active={_active!r}, _ori_in_slots={_in_slots!r}, "
                f"_ori_out_slots={_out_slots!r}. A model class in the MRO likely "
                f"failed to forward 'orientation_singlestep_slots' (RLRP-736 anti-leak "
                f"activation assertion)."
            )

    # RLRP-736 S3.11 (§14C): make the handler's normalizer-level (A) + model-input
    # (B) handling live on the TRAINING path. See ``setup_single_step_model`` for
    # the rationale (instantiate bypasses ``create_one_dim_tr_model_v2``). Empty
    # (bit-exact) for the neutral scalar / math handler.
    from tools.feature_handling_tools.env_handlers import (
        resolve_feature_handler_transition_model_overrides,
    )

    _ms_feature_overrides = resolve_feature_handler_transition_model_overrides(
        cfg, cfg.one_dim_transition_model
    )
    ms_1D_transition_model: OneDTransitionRewardModelV2 = hydra.utils.instantiate(
        cfg.one_dim_transition_model,
        model=multistep_step_model_ensemble,
        learned_rewards=False,
        _recursive_=False,
        **_ms_feature_overrides,
    )

    # RLRP-736 S1.3 / S3.12 (§14C): single post-construction feature-handler seam
    # (register the per-environment handler on the model's opt-in geometry-loss
    # hook + stamp the completeness marker), shared with the single-step site and
    # the load-path builder so no construction path can silently bypass it.
    # Neutral by default: absent config -> scalar-only handler, and
    # ``feature_geometry_loss_weight`` defaults to 0.0, so the training loss stays
    # bit-exact with the legacy path.
    from tools.feature_handling_tools.env_handlers import (
        apply_feature_handler_to_transition_model,
    )

    apply_feature_handler_to_transition_model(
        cfg, ms_1D_transition_model, multistep_step_model_ensemble
    )

    if load_pretrained_path:
        ms_1D_transition_model.load(
            os.path.join(load_pretrained_path, "saved_dynamic_model")
        )

    if not disable_compile and (
        is_cfg_key_exist(cfg, "UDER.batch_size.limit")
        and is_cfg_key_exist(cfg, "UDER.batch_size.init_value")
        and cfg.UDER.batch_size.limit == cfg.UDER.batch_size.init_value
    ):
        ms_1D_transition_model = optimize_model_speed(ms_1D_transition_model)

    motion_model_container.set_dynamics_model(ms_1D_transition_model)
    return motion_model_container


def setup_multistep_step_model_and_trainer(
    cfg: omegaconf.DictConfig, load_pretrained_path: Optional[str] = None
) -> Tuple[R2SMotionModelContainer, mbrl.models.ModelTrainer]:

    # NOTE: process-global `torch.backends` knobs (TF32 / cudnn.benchmark) are owned by
    # `cfg.torch_backend.*` and applied ONCE in `R2S2RPipelineHydraApp.setup()` via
    # `apply_torch_backend_cfg`; do not write `torch.backends` here (RLRP-824 consolidation).
    motion_model_container = setup_multistep_step_model(cfg, load_pretrained_path)

    # RLRP-786 (FR5): the opt-in CUDA-graph captured training step needs a ``capturable`` Adam
    # (device-side step counter) -> request it BEFORE the optimizer is built. Read once here.
    cuda_graph_training_step = bool(
        omegaconf.OmegaConf.select(cfg, "pipeline.cuda_graph_training_step", default=False)
    )
    if cuda_graph_training_step:
        with omegaconf.open_dict(cfg.ms_training.optimizer):
            cfg.ms_training.optimizer.adam_capturable = True

    ms_trainer = mbrl.models.ModelTrainer(
        motion_model_container.dynamics_model,
        optim_lr=cfg.ms_training.model_lr,
        weight_decay=cfg.ms_training.model_wd,
        # Training Speed & Efficiency plan (B0-bis + F-C2) — opt-in
        # kwargs resolved from `cfg.mbrl_lib.*`. All flags default to
        # legacy behaviour so current experiments stay bit-exact with
        # the pre-patch code path.
        **resolve_mbrl_trainer_opt_in_kwargs_from_cfg(cfg),
    )
    ms_trainer = change_optimizer(cfg.ms_training, ms_trainer)

    if cuda_graph_training_step:
        maybe_enable_cuda_graph_training_step(cfg, ms_trainer)

    return motion_model_container, ms_trainer


def maybe_enable_cuda_graph_training_step(
    cfg: omegaconf.DictConfig, trainer: mbrl.models.ModelTrainer
) -> bool:
    """``pipeline.cuda_graph_training_step: true`` -> route the Lightning ``training_step`` of the
    wrapper through the RLRP-786 ``CudaGraphTrainStep`` (FR5 of
    ``perf_RLRP-786_cuda_graph_train_step_plan_20260919.md``).

    Requires ``pipeline.performance_mode: fast`` (the graph precondition, applied by
    :func:`_apply_meta_collection_flag`) and ``ms_model.cuda_graph_capture_ready: true``; every
    blocker is reported by the wrapper as a one-line notice and the run stays eager (never
    raises). Returns ``True`` when the captured step is active.
    """
    if not bool(
        omegaconf.OmegaConf.select(cfg, "pipeline.cuda_graph_training_step", default=False)
    ):
        return False
    wrapper = trainer.model
    enable = getattr(wrapper, "enable_cuda_graph_training_step", None)
    if enable is None:
        consol_msg_universal_one_liner(
            "RLRP-786 pipeline.cuda_graph_training_step requested but the Lightning module "
            f"{type(wrapper).__name__} has no enable_cuda_graph_training_step seam -> eager"
        )
        return False
    warmup_iters = int(
        omegaconf.OmegaConf.select(cfg, "pipeline.cuda_graph_warmup_steps", default=3)
    )
    # AR-baseline extension (FR4 of ``perf_RLRP-786_ar_tcn_cuda_graph_fp32_baseline_plan_20260919.md``):
    # ``pipeline.cuda_graph_allow_autocast: true`` opts into the EXPLORATORY capture of a
    # ``mixed_precision`` step (``torch.autocast(..., cache_enabled=False)``, allclose parity only).
    # Default ``false`` -> an autocast step keeps the "fp32 only" blocker and stays eager.
    allow_autocast = bool(
        omegaconf.OmegaConf.select(cfg, "pipeline.cuda_graph_allow_autocast", default=False)
    )
    return bool(
        enable(trainer.optimizer, warmup_iters=warmup_iters, allow_autocast=allow_autocast)
    )


def _make_optuna_pruner_epoch_callback_factory(
    pipeline_app: R2S2RPipelineHydraApp,
) -> Callable:
    """RLRP-624 Phase B — factory matching the
    ``setup_trainer_epoch_callback_aggregator`` registration signature
    ``(cfg, cfg_training, model_trainer, tensorboard_writer) -> callback``.

    Returned callback reports ``eval_score`` (validation loss) at each epoch
    and raises ``optuna.TrialPruned`` when ``study.pruner.should_prune()``
    asks to terminate. Cumulative ``step`` counter is preserved across UDER
    outer loops because the closure outlives single trainer-loops.
    """

    def _factory(cfg, cfg_training, model_trainer, tensorboard_writer):
        state = {"step": 0}

        def _cb(
            model, train_iteration, epoch, total_avg_loss, eval_score, best_val_score
        ):
            # Report only when we have a usable scalar.
            if eval_score is None:
                state["step"] += 1
                return None
            try:
                val = float(eval_score)
            except (TypeError, ValueError):
                state["step"] += 1
                return None
            # `raise_if_should_prune` is a no-op when no trial handle is
            # attached on the pipeline_app.
            pipeline_app.raise_if_should_prune(val, state["step"])
            state["step"] += 1
            return None

        return _cb

    return _factory


def setup_ms_model_train_callback(
    cfg: DictConfig,
    ms_tensorboard_writer: OnlineTensorboardWritter | None | Any,
    motion_model_container: R2SMotionModelContainer,
    ms_trainer: ModelTrainer,
    val_env_trjs: list[TestMotionTrajectoryDataclass],
    test_InD_trjs: Optional[list[TestTrajectoryEntry]],
    test_OoD_trjs: Optional[list[TestTrajectoryEntry]],
    state_space_label: list[Any],
    exp_dir_relative_path: str | Any,
    headless: bool,
    pipeline_app: Optional[R2S2RPipelineHydraApp] = None,
) -> tuple[
    Callable[..., Any],
    Callable[..., Any],
    Optional[Callable[..., Any]],
    Callable[..., Any] | Any,
]:
    """
    Setup callbacks for multi-step model training and evaluation.

    This function initializes and aggregates various callbacks required for multi-step
    model training and evaluation pipelines, including epoch callbacks, batch callbacks,
    and pre/post training callbacks. It also configures trajectory monitoring and logging
    functionality, such as TensorBoard integration, simulation visualization, and gradient
    monitoring.

    :param test_OoD_trjs:
    :param test_InD_trjs:
    :param cfg: Configuration object for training parameters.
    :param ms_tensorboard_writer: TensorBoard writer for logging data during training.
    :param motion_model_container: Container for motion model components and configurations.
    :param ms_trainer: Trainer object managing the multi-step training process.
    :param val_env_trjs: List of validation environment trajectories.
    :param state_space_label: Labels for state space dimensions.
    :param exp_dir_relative_path: Relative path to the experiment directory.
    :param headless: Flag to specify if visualization should run in headless mode.
    :param pipeline_app:
    :return: A tuple containing pre-training, post-training, batch, and epoch callback functions.
    """
    # ....Pre-train-epoch callback.................................................................
    ms_erll_epoch_pre_training_callback = setup_erll_epoch_pre_training_callback_aggregator(
        cfg,
        cfg.ms_training,
        ms_trainer,
        ms_tensorboard_writer,
        register_callbacks=[
            # Require `pipeline.plot.render_dataset_exploration_plot=true`
            partial(
                setup_math_env_uder_buffer_sampling_plotter_callback,
                **{
                    "test_env": val_env_trjs[0],
                    # Pick one env
                    "state_space_label": state_space_label,
                    "exp_dir_relative_path": exp_dir_relative_path,
                    "headless": headless,
                },
            )
        ],
    )

    # ....Train-epoch callback.....................................................................
    def _setup_partial_tsb_trajectory_callback(
        _cfg: DictConfig,
        _motion_model_container: R2SMotionModelContainer,
        _env_trjs: list[TestMotionTrajectoryDataclass],
        _label: str,
    ) -> partial[Callable[..., Any]]:
        return partial(
            setup_tensorboard_arbitrary_dimension_trajectory_prediction_monitor_callback,
            **{
                "trajectories": _env_trjs,
                "trajectory_record_step_size": fetch_cfg_pipeline_tensorboard_key_value(
                    _cfg,
                    key="trajectory_record_step_size",
                    key_value_default=100,
                ),
                "execute_every_n_epoch": fetch_cfg_pipeline_tensorboard_key_value(
                    _cfg,
                    key="record_trajectory_pred_every_n_train_epoch",
                    key_value_default=None,
                ),
                "motion_model_container": _motion_model_container,
                "obs_dim_label": fetch_cfg_pipeline_tensorboard_key_value(
                    _cfg,
                    key="obs_dim_label",
                    key_value_default=("X", "Y", "Z"),
                ),
                "rollout_dim_label": fetch_cfg_pipeline_tensorboard_key_value(
                    _cfg,
                    key="rollout_dim_label",
                    key_value_default=("X", "Y", "Z"),
                ),
                "label": _label,
            },
        )

    epoch_register: list = [
        setup_trainer_lr_scheduler_callback,
        setup_gradient_monitoring_callback,
        # RLRP-707 (§4ter) — step the train-time domain randomization noise-scale scheduler and
        # record the current noise scale to tensorboard once per model training epoch. No-op when
        # DR is disabled (default) or for non-DR models.
        setup_train_time_domain_randomization_scheduler_callback,
        _setup_partial_tsb_trajectory_callback(
            cfg, motion_model_container, val_env_trjs,
            # "Train/val"
            "Train"
        ),
    ]

    # Guard on truthiness (not just ``is not None``) so a ``null``/empty
    # test-trajectory list skips callback registration instead of tripping
    # ``min([])`` (feature_empty_ood_test_trajectory_support_plan_20260821_RLRP-778.md).
    if test_InD_trjs:
        test_InD_trjs: list[TestMotionTrajectoryDataclass] = [
            each.env for each in test_InD_trjs
        ]
        shortest_test_trj_len = min([len(each) for each in test_InD_trjs])
        test_InD_trjs = [each[:shortest_test_trj_len] for each in test_InD_trjs]
        epoch_register.append(
            _setup_partial_tsb_trajectory_callback(
                cfg,
                motion_model_container,
                test_InD_trjs,
                "Test InD",
            ),
        )

    if test_OoD_trjs:
        test_OoD_trjs: list[TestMotionTrajectoryDataclass] = [
            each.env for each in test_OoD_trjs
        ]
        shortest_test_trj_len = min([len(each) for each in test_OoD_trjs])
        test_OoD_trjs = [each[:shortest_test_trj_len] for each in test_OoD_trjs]
        epoch_register.append(
            _setup_partial_tsb_trajectory_callback(
                cfg,
                motion_model_container,
                test_OoD_trjs,
                "Test OoD",
            ),
        )

    # RLRP-624 Phase B — register Optuna pruner epoch callback when an
    # active trial handle is attached to the pipeline_app.
    if (
        pipeline_app is not None
        and getattr(pipeline_app, "optuna_trial_handle", None) is not None
    ):
        epoch_register.append(_make_optuna_pruner_epoch_callback_factory(pipeline_app))

    ms_trainer_epoch_callback = setup_trainer_epoch_callback_aggregator(
        cfg,
        cfg.ms_training,
        ms_trainer,
        ms_tensorboard_writer,
        register_callbacks=epoch_register,
    )

    # ....Batch callback...........................................................................
    if cfg.pipeline.tensorboard.batch_callback_execute_every_n is not None:
        if cfg.pipeline.tensorboard.batch_callback_execute_every_n >= 0:
            consol_msg_universal_one_liner(
                f"Set batch callback to run every {cfg.pipeline.tensorboard.batch_callback_execute_every_n} batches"
            )

        ms_trainer_batch_callback = setup_batch_callback_aggregator(
            cfg,
            cfg.ms_training,
            ms_trainer,
            ms_tensorboard_writer,
            register_callbacks=[
                setup_multistep_loss_batch_callback,
                setup_compounded_pred_unroll_len_batch_callback,
            ],
        )
    else:
        consol_msg_universal_one_liner(f"Skip batch callback")
        ms_trainer_batch_callback = None

    # ....Post-train-epoch callback................................................................
    ms_erll_epoch_post_training_callback = setup_trainer_epoch_callback_aggregator(
        cfg,
        cfg.ms_training,
        ms_trainer,
        ms_tensorboard_writer,
        register_callbacks=[
            # setup_erll_tensorboard_manual_lr_monitor_callback,
        ],
        enable_tensorboard_call_method=False,
    )
    return (
        ms_erll_epoch_post_training_callback,
        ms_erll_epoch_pre_training_callback,
        ms_trainer_batch_callback,
        ms_trainer_epoch_callback,
    )


def assert_ood_objective_has_ood_trajectories(cfg: DictConfig, n_ood: int) -> None:
    """Fail loudly and early when an HPO study optimises the OOD deploy metric
    against a dataset that declares **zero** OOD test trajectories.

    HPO runs are long and resource-expensive, so a mis-configured objective must
    trip at setup time (before any model training) rather than silently
    poisoning the Optuna study with the ``_SENTINEL`` (``float64.max``) value
    that an empty OOD rollout set produces
    (feature_empty_ood_test_trajectory_support_plan_20260821_RLRP-778.md, D1).

    The guard only fires for a **live Optuna sweep** whose
    ``cfg.hparam_optimizer.objectives_name`` references
    ``deploy_compounded_pred_mae_target_OOD``. Ordinary (non-sweep) train /
    deploy / plot runs of a ``null``-OOD dataset are unaffected, even though they
    still *record* the metric.

    :param cfg: Hydra experiment config.
    :param n_ood: Number of resolved OOD test-target rollouts.
    :raises ValueError: When a live OOD-objective sweep meets ``n_ood == 0``.
    """
    from tools.hydra_apps_tools.hparam_optimization import HyperparamObjectives

    if n_ood > 0:
        return

    _OOD_OBJECTIVE = "deploy_compounded_pred_mae_target_OOD"
    hyperparam_objectives = HyperparamObjectives(cfg)
    if not hyperparam_objectives.is_hydra_hyperparameter_optimization_run:
        return
    if _OOD_OBJECTIVE not in hyperparam_objectives.get_objectives_name():
        return

    raise ValueError(
        f"HPO objective '{_OOD_OBJECTIVE}' is configured in "
        f"`cfg.hparam_optimizer.objectives_name`, but the resolved dataset "
        f"declares ZERO OOD test trajectories "
        f"(`cfg.environment.data.test_OOD_trajectory` is empty/null). "
        f"Optimising this objective would poison the study with the worst-case "
        f"sentinel value on every trial and waste expensive HPO compute. "
        f"Either provide at least one OOD test trajectory or remove "
        f"'{_OOD_OBJECTIVE}' from the study objectives."
    )


def uder_cfg_validation(cfg: DictConfig):
    env_explorable_size = 0
    if (
        is_cfg_key_exist(cfg, "environment.explorable_space")
        and is_cfg_key_exist(cfg, "UDER.buffer_max_trajectory_len")
        and cfg.UDER.buffer_max_trajectory_len is not None
    ):
        for explo_interval in omegaconf.OmegaConf.to_object(
            cfg.environment.explorable_space
        ):
            env_explorable_size += explo_interval[1] - explo_interval[0]
        assert env_explorable_size >= cfg.UDER.buffer_max_trajectory_len, (
            f"The sum of cfg.environment.explorable_space={env_explorable_size} !>= "
            f"{cfg.UDER.buffer_max_trajectory_len=}"
        )

    # .... Validate `original_timestamps_unit` (mandatory) ........................................
    # The robotic pipelines build a 100 Hz resampling grid in SECONDS in
    # `online_pre_processing`. A wrong / missing unit causes a huge
    # `np.arange` allocation and an external SIGKILL with no Python
    # traceback (see refactoring around RLRP issue: ns timestamps SIGKILL).
    # Hard-fail at config-validation time rather than at runtime.
    if is_cfg_key_exist(cfg, "environment.data.original_timestamps_label"):
        allowed_units = {"s", "ms", "us", "ns"}
        if not is_cfg_key_exist(cfg, "environment.data.original_timestamps_unit"):
            raise omegaconf.errors.ConfigAttributeError(
                "Missing mandatory config key "
                "`environment.data.original_timestamps_unit`. "
                f"Must be one of {sorted(allowed_units)}. "
                "Add it to your simulator general config "
                "(e.g. `src/launcher/configs/simulator/<robot>_general.yaml`)."
            )
        unit = cfg.environment.data.original_timestamps_unit
        if unit not in allowed_units:
            raise ValueError(
                f"`environment.data.original_timestamps_unit={unit!r}` not in "
                f"{sorted(allowed_units)}."
            )

        # .... Validate `subscaling_kind` (optional) ..............................................
        # Dataset frame-rate sub-scaling method used by `online_pre_processing`
        # (RLRP-742). When absent it defaults to `decimate`. Hard-fail here on an
        # unknown value rather than at trajectory-extraction time.
        if is_cfg_key_exist(cfg, "environment.data.subscaling_kind"):
            allowed_subscaling_kinds = {"interpolate", "decimate"}
            subscaling_kind = cfg.environment.data.subscaling_kind
            if subscaling_kind not in allowed_subscaling_kinds:
                raise ValueError(
                    f"`environment.data.subscaling_kind={subscaling_kind!r}` not in "
                    f"{sorted(allowed_subscaling_kinds)}."
                )
