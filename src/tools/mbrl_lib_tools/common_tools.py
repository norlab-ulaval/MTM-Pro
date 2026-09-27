# coding=utf-8
from typing import Callable, List, Optional, Tuple, Union
import pathlib
import hydra

import gymnasium as gym
import mbrl.models
import numpy as np
import omegaconf
from mbrl import models
from mbrl.models import OneDTransitionRewardModel
from mbrl.util import ReplayBuffer, common as common_util
import mbrl.util.common

from algorithm.motion_model.utils.inference_optimization_utils import (
    optimize_model_speed,
)
from custom_types.custom_types_all_gym_wrappers import ALL_CUSTOM_GYM_WRAPPER_TYPES
from custom_types.custom_types_learnable_f110_gym_env import LEARNABLE_F110ENV_TYPES

from tools.console_tools.message import (
    consol_msg_motion_model_learner_one_line,
    consol_msg_universal_one_liner,
)
from tools.feature_handling_tools.feature_loss_weights import (
    resolve_and_apply_feature_loss_weights,
)
from tools.feature_handling_tools.normalization_diagnostic import (
    log_feature_normalization_report,
)
from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist
from tools.mbrl_lib_tools.models.one_dim_tr_model_v2 import OneDTransitionRewardModelV2
from tools.multistep_tools.models import MultiStepMLP
from tools.multistep_tools.utils import (
    compute_multistep_model_in_size,
    compute_multistep_model_out_size,
)


def train_model_mbrllib_v2(
    dynamics_model: models.OneDTransitionRewardModel,
    dynamics_model_trainer: mbrl.models.ModelTrainer,
    cfg: omegaconf.DictConfig,
    replay_buffer: ReplayBuffer,
    callback: Optional[Callable] = None,
    batch_callback: Optional[Callable] = None,
) -> Tuple[models.OneDTransitionRewardModel, List[float], List[float]]:
    # (CRITICAL) ToDo: RLRP-361 feat: add mbrl-lib get_sequence_buffer_iterator support
    """This is a re-implementation of mbrl-lib `common.train_model_and_save_model_and_data()`
    (without the save model and data part) wich return the train metric train_losses and val_losses

    :param dynamics_model: a mbrl-lib dynamic model
    :param dynamics_model_trainer: the mbrl-lib model trainer
    :param cfg: hydra config
    :param replay_buffer: a mbrl-lib replay buffer
    :param callback: (optional) this function will be called after every training epoch.
            See `mbrl.models.ModelTrainer` for signature.
    :param batch_callback: (optional) this function will be called for every batch
            See `mbrl.models.ModelTrainer` for signature.
    :return: a trained dynamic model plus the history of training losses and validation losses
    """
    train_dataset, val_dataset = mbrl.util.common.get_basic_buffer_iterators(
        replay_buffer=replay_buffer,
        batch_size=cfg.overrides.get("model_batch_size", 256),
        val_ratio=cfg.overrides.get("validation_ratio", 0.2),
        ensemble_size=cfg.dynamics_model.get("ensemble_size", 1),
        shuffle_each_epoch=cfg.overrides.get("shuffle_each_epoch", True),
        bootstrap_permutes=cfg.overrides.get("bootstrap_permutes", False),
    )

    if hasattr(dynamics_model, "update_normalizer"):
        consol_msg_motion_model_learner_one_line(
            "Updates normalizer statistics using dataset"
        )
        dynamics_model.update_normalizer(replay_buffer.get_all())
        # RLRP-761 S3.3: resolve a NAMED per-feature loss-weight criterion
        # ('predictability' / 'raw_equivalent') against the dataset. Must run
        # BEFORE the diagnostic so the rendered `w` column shows the weights the
        # run will actually use. A strict no-op for a numeric/absent spec.
        resolve_and_apply_feature_loss_weights(cfg, dynamics_model, replay_buffer)
        # RLRP-761 S2.4: render the resolved per-feature normalization contract
        # ONCE, after the statistics are fitted and before the first gradient
        # step. Read-only and failure-tolerant by contract.
        log_feature_normalization_report(cfg, dynamics_model, replay_buffer)

    consol_msg_motion_model_learner_one_line(
        f"Begin {dynamics_model.__class__.__name__}/"
        f"{dynamics_model.model.__class__.__name__} model training"
    )

    train_losses, val_losses = dynamics_model_trainer.train(
        train_dataset,
        val_dataset,
        num_epochs=cfg.overrides.get("num_epochs_train_model", None),
        patience=cfg.overrides.get("patience", None),
        improvement_threshold=cfg.overrides.get("improvement_threshold", 0.01),
        callback=callback,
        batch_callback=batch_callback,
        silent=False,
    )
    return dynamics_model, train_losses, val_losses


def create_one_dim_tr_model_v2_explicit_size(
    cfg: omegaconf.DictConfig, in_size: int, out_size: int
) -> OneDTransitionRewardModelV2:
    # (NICE TO HAVE) ToDo: refactor >> directly instanciate mbrl.models.OneDTransitionRewardModel
    #  instead of relying on mbrl-lib implementation
    """Creates a dynamic model based on hydra cfg except in/out size given explicitly.

    :param cfg: Configuration dictionary containing model specifications.
    :param in_size: The input size of the dynamic model.
    :param out_size: The output size of the dynamic model.
    :return: a dynamic model ready to train.
    """
    # .... dynamic model cfg sanity check .........................................................
    propagation_method = cfg.dynamics_model.get("propagation_method", None)
    ensemble_size = cfg.dynamics_model.get("ensemble_size", 1)
    if propagation_method is not None and ensemble_size == 1:
        raise AttributeError(
            "cfg.dynamics_model.propagation_method should be set to null when "
            "cfg.dynamics_model.ensemble_size == 1 as propagation method logic is only used "
            "by mbrl-lib ensemble model. Currently, propagation_method="
            f"{propagation_method} and ensemble_size={ensemble_size}"
        )

    deterministic_model = cfg.dynamics_model.get("deterministic", False)
    learn_logvar_bounds = cfg.dynamics_model.get("learn_logvar_bounds", False)
    if deterministic_model is True and learn_logvar_bounds is True:
        raise AttributeError(
            "cfg.dynamics_model.deterministic should be set to false when "
            "cfg.dynamics_model.learn_logvar_bounds == true as learn logvar bounds logic is "
            "only used by mbrl-lib probabilistic model. Curently deterministic="
            f"{deterministic_model} and learn_logvar_bounds={learn_logvar_bounds}"
        )

    # ....Setup dynamic model......................................................................
    model_cfg_key = "dynamics_model"

    # noinspection PyProtectedMember
    if cfg.dynamics_model._target_ == "mbrl.models.BasicEnsemble":
        # Set the BasicEnsemble member model in/out size
        model_cfg_key = "dynamics_model.member_cfg"

    # Note: param 'obs_shape' and 'act_shape' are computed from cfg field 'in_size' and 'out_size'
    omegaconf.OmegaConf.update(cfg, f"{model_cfg_key}.in_size", in_size, merge=False)
    omegaconf.OmegaConf.update(cfg, f"{model_cfg_key}.out_size", out_size, merge=False)

    dynamics_model = create_one_dim_tr_model_v2(cfg)

    # Note: temporary sanity check
    # assert dynamics_model.target_is_delta == cfg.algorithm.get("target_is_delta", None)
    # assert dynamics_model.target_is_delta is False # ToDo: experiment <--

    return dynamics_model


def create_one_dim_tr_model_v2(
    cfg: omegaconf.DictConfig,
    singlestep_obs_len: Optional[int] = None,
    singlestep_act_len: Optional[int] = None,
    model_dir: Optional[Union[str, pathlib.Path]] = None,
    disable_compile: bool = False,
) -> OneDTransitionRewardModelV2:
    # (NICE TO HAVE) ToDo: implement test case (indirectly tested)
    """Modifyed version of `mbrl.util.common.create_one_dim_tr_model` that instanciate class
    `tools.mbrl_lib_tools.models.one_dim_tr_model_v2.OneDTransitionRewardModelV2` instead of
     class `mbrl.models.OneDTransitionRewardModel`

    Note: Both `obs_shape` and `act_shape` take singlestep shape. Subclass of `MultiStepMLP` model
    will be automaticly setup from cfg object.

    """
    # This first part takes care of the case where model is BasicEnsemble and in/out sizes
    # are handled by member_cfg
    model_cfg = cfg.dynamics_model
    if issubclass(hydra.utils._locate(model_cfg._target_), mbrl.models.BasicEnsemble):
        model_cfg = model_cfg.member_cfg

    if issubclass(hydra.utils._locate(model_cfg._target_), MultiStepMLP):
        if model_cfg.get("singlestep_obs_len", None) and singlestep_obs_len is None:
            singlestep_obs_len = cfg.dynamics_model.singlestep_obs_len

        if model_cfg.get("singlestep_act_len", None) and singlestep_act_len is None:
            singlestep_act_len = cfg.dynamics_model.singlestep_act_len

    if model_cfg.get("in_size", None) is None:
        assert singlestep_obs_len is not None
        assert singlestep_act_len is not None
        if issubclass(hydra.utils._locate(model_cfg._target_), MultiStepMLP):
            model_cfg.in_size = compute_multistep_model_in_size(
                singlestep_obs_len=singlestep_obs_len,
                singlestep_act_len=singlestep_act_len if singlestep_act_len else 1,
                multistep_len=model_cfg.history_len,
            )
        else:
            model_cfg.in_size = singlestep_obs_len + (
                singlestep_act_len if singlestep_act_len else 1
            )

    if model_cfg.get("out_size", None) is None:
        assert singlestep_obs_len is not None
        assert singlestep_act_len is not None
        _model_cls = hydra.utils._locate(model_cfg._target_)
        if issubclass(_model_cls, MultiStepMLP):
            # RLRP-824: the composed OUTPUT window is ``W = resolve_output_window_len(H, F)``
            # (``history_len`` for every legacy family; ``max(H, F)`` for the MS->MS forecast
            # family), resolved from the ``_target_`` class BEFORE instantiation.
            model_cfg.out_size = compute_multistep_model_out_size(
                singlestep_obs_len=singlestep_obs_len,
                singlestep_act_len=singlestep_act_len if singlestep_act_len else 1,
                multistep_len=_model_cls.resolve_output_window_len(
                    model_cfg.history_len, model_cfg.horizon_len
                ),
                learned_reward=cfg.algorithm.get("learned_rewards", False),
            )
        else:
            model_cfg.out_size = singlestep_obs_len + int(
                cfg.algorithm.get("learned_rewards", False)
            )

    # Now instantiate the model
    model = hydra.utils.instantiate(cfg.dynamics_model, _recursive_=False)

    name_obs_process_fn = cfg.overrides.get("obs_process_fn", None)
    if name_obs_process_fn:
        obs_process_fn = hydra.utils.get_method(cfg.overrides.obs_process_fn) # (NICE TO HAVE) ToDo: cfg should be related to model or environment, not 'override'
    else:
        obs_process_fn = None
    normalizer_kwargs = cfg.algorithm.get("normalizer_kwargs", None)
    if normalizer_kwargs is not None:
        normalizer_kwargs = omegaconf.OmegaConf.to_container(
            normalizer_kwargs, resolve=True
        )

    # RLRP-736 S3.7 (§14C): inject the per-environment feature handler's
    # per-block normalizer strategies (quaternion UNIT_NORM, dt -> IDENTITY,
    # per-dim normalize_dims mask) into ``normalizer_kwargs`` so the inherited
    # fork ``_build_normalizers`` / ``StrategyAwareNormalizer`` routing actually
    # applies at training time. No-op / bit-exact when the handler is neutral
    # (no handler / scalar / math -> every dim INHERIT), so math stays bit-exact.
    from tools.feature_handling_tools.env_handlers import (
        merge_feature_handler_normalizer_kwargs,
        resolve_feature_handler_no_delta_list,
    )

    normalizer_kwargs = merge_feature_handler_normalizer_kwargs(
        cfg, normalizer_kwargs
    )

    # RLRP-736 S3.8 (§14C): thread the handler's model-input handling (B) into
    # the production model. Today only ``no_delta_list`` has an observable effect
    # (the handler's ``obs_process_fn`` is identity under the symmetric
    # external-quaternion contract): for robotic-3D the quaternion attitude block
    # is flagged absolute-target so it is never regressed as a Euclidean delta.
    # No-op / bit-exact when the handler is neutral (empty ``no_delta_list``).
    _resolved_no_delta_list = resolve_feature_handler_no_delta_list(
        cfg, cfg.overrides.get("no_delta_list", None)
    )

    # For robust normalizers, obs_dim / act_dim must match the model's
    # single-step dimensions (not the env's raw obs space which may differ
    # when obs_process_fn remaps features).  Prefer the model's own
    # singlestep_obs_len / singlestep_act_len when available.
    _normalizer_obs_dim = getattr(model, "singlestep_obs_len", None) or singlestep_obs_len
    _normalizer_act_dim = getattr(model, "singlestep_act_len", None) or singlestep_act_len

    dynamics_model = OneDTransitionRewardModelV2(
        model,
        target_is_delta=cfg.algorithm.get("target_is_delta", False),
        normalize=cfg.algorithm.get("normalize", False),
        normalize_double_precision=cfg.algorithm.get(
            "normalize_double_precision", None
        ),
        learned_rewards=cfg.algorithm.get("learned_rewards", False),
        obs_process_fn=obs_process_fn,
        no_delta_list=_resolved_no_delta_list,
        num_elites=cfg.overrides.get("num_elites", None),
        normalizer_type=cfg.algorithm.get("normalizer_type", "winsorized"),
        obs_dim=_normalizer_obs_dim,
        act_dim=_normalizer_act_dim,
        normalizer_kwargs=normalizer_kwargs,
    )
    # RLRP-736 S3.12 (§14C): single post-construction feature-handler seam
    # (loss-handler registration + completeness marker), shared with the two
    # training-path sites in ``setup.py`` so no construction path can silently
    # bypass the handler wiring. No-op / bit-exact for the neutral scalar/math
    # handler and for hookless ensembles (e.g. GaussianMLP).
    from tools.feature_handling_tools.env_handlers import (
        apply_feature_handler_to_transition_model,
    )

    apply_feature_handler_to_transition_model(cfg, dynamics_model, model)

    if model_dir:
        dynamics_model.load(model_dir)

    if not disable_compile:
        if is_cfg_key_exist(cfg, "UDER.batch_size"):
            if (
                is_cfg_key_exist(cfg, "UDER.batch_size.limit")
                and is_cfg_key_exist(cfg, "UDER.batch_size.init_value")
                and cfg.UDER.batch_size.limit == cfg.UDER.batch_size.init_value
            ):
                dynamics_model = optimize_model_speed(dynamics_model)
        else:
            dynamics_model = optimize_model_speed(dynamics_model)

    return dynamics_model


def create_sampling_replay_buffer_from_env(
    cfg: omegaconf.DictConfig,
    sampling_env: Union[LEARNABLE_F110ENV_TYPES, ALL_CUSTOM_GYM_WRAPPER_TYPES],
    seed: Union[int, None] = None,
) -> ReplayBuffer:
    """Creates an empty sampling replay buffer for learning a dynamic model.

    :param cfg: A hydra config file.
    :param sampling_env: A gym/gymnasium environment.
    :param seed: (optional) Random number generator seed for the replay buffer.
    :return: An empty replay buffer for storing and sampling experiences.
    """
    assert isinstance(sampling_env.observation_space, gym.spaces.Box), (
        "The `env` observation space need to be a single dimension gym Box observation space "
        f"instead of {type(sampling_env.observation_space)}"
    )

    # ....Setup replay buffer......................................................................
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", None)
    dtype = np.double if use_double_dtype else np.float32

    dataset_size_ = cfg.algorithm.get("dataset_size", None)
    if dataset_size_:
        consol_msg_universal_one_liner(
            f"hydra config `algorithm.dataset_size` is set to {dataset_size_}"
        )
    else:
        num_step_ = cfg.overrides.get("num_steps", None)
        if num_step_:
            consol_msg_universal_one_liner(
                "hydra config `algorithm.dataset_size` was not set. Initial replay buffer "
                f"size will be set using `overrides.num_step_`={num_step_}."
            )
        else:
            consol_msg_universal_one_liner(
                f"Neither cfg `algorithm.dataset_size` or `overrides.num_step_` where set (!)"
            )

    sampling_obs_shape = sampling_env.observation_space.shape
    sampling_act_shape = sampling_env.action_space.shape

    empty_replay_buffer = common_util.create_replay_buffer(
        cfg,
        sampling_obs_shape,
        sampling_act_shape,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
        rng=np.random.default_rng(seed=seed),
        collect_trajectories=False,
    )

    consol_msg_universal_one_liner(
        "Initialized replay buffer:"
        f"  obs_shape={sampling_obs_shape}\n"
        f"  act_shape={sampling_act_shape}\n"
        f"  capacity={empty_replay_buffer.capacity}\n"
        f"  collect_trajectories={empty_replay_buffer.stores_trajectories}\n"
        f"  dtype={dtype}\n"
    )
    return empty_replay_buffer
