# coding=utf-8
import os
from typing import Optional, Union

import numpy as np
import omegaconf
import torch
from mbrl.util import ReplayBuffer

from tools.console_tools.message import consol_msg_universal_one_liner
from tools.hydra_apps_tools.hydra_utils import (
    get_hydra_experiment_cwd,
    get_hydra_original_cwd,
)

_DTYPE_TO_SPEC: dict = {
    torch.float64: "torch.float64",
    torch.float32: "torch.float32",
    np.float64: "numpy.float64",
    np.float32: "numpy.float32",
}

_SPEC_TO_DTYPE: dict = {v: k for k, v in _DTYPE_TO_SPEC.items()}
# Backward-compatible aliases for spec files saved before the framework prefix was added
_SPEC_TO_DTYPE["float64"] = np.float64
_SPEC_TO_DTYPE["float32"] = np.float32


def _resolve_ms_replaybuffer_feature_dim_tag(cfg: omegaconf.DictConfig) -> str:
    """Return the ``obsD=<N>_actD=<M>`` feature-dimension identification tag.

    Permanent helper. Introduced by action ``A1`` of the RLRP-798 multistep
    replay-buffer feature-dim identification ``.junie`` plan
    (``rlrp-798-ms-replaybuffer-feature-dim-identification-plan-20260830.md``).

    The counts come from the hydra cfg via the RLRP-736 single source of truth
    (``resolve_obs_shape`` / ``resolve_act_shape``) and NOT from the
    replay-buffer object, because the load path resolves this tag before any
    buffer exists and before ``environment.obs_shape`` is stamped by
    ``setup_multistep_step_model_and_trainer``. Save and load therefore always
    agree on the same directory.

    Fails loud (``ValueError`` naming the unresolvable key) for a cfg that
    declares neither ``environment.obs_dims`` / ``act_dims`` nor an explicit
    ``environment.obs_shape`` / ``act_shape`` — intended, this module is
    robotic-only.
    """
    from tools.feature_handling_tools.env_handlers import (
        resolve_act_shape,
        resolve_obs_shape,
    )

    obs_dim_count = int(resolve_obs_shape(cfg)[-1])
    act_dim_count = int(resolve_act_shape(cfg)[-1])
    return f"obsD={obs_dim_count}_actD={act_dim_count}"


def _validate_ms_replaybuffer_spec_feature_dims(
    cfg: omegaconf.DictConfig,
    rb_spec: omegaconf.DictConfig,
    spec_path: Union[str, os.PathLike],
) -> None:
    """Fail loud when a spec's recorded feature dims disagree with the cfg.

    Permanent guard. Introduced by action ``A4`` of the RLRP-798 multistep
    replay-buffer feature-dim identification ``.junie`` plan
    (``rlrp-798-ms-replaybuffer-feature-dim-identification-plan-20260830.md``).

    This is DEFENCE IN DEPTH, not redundancy with the directory tag: a dim
    rename that preserves the width (e.g. swapping an ``attitude.*`` dim for a
    ``gravity.*`` dim of equal width) resolves to the SAME
    ``obsD=<N>_actD=<M>`` directory, so only this name-level comparison can
    separate those two artifacts.

    A spec missing the keys is a hard error (no legacy support): such a file can
    only be a pre-RLRP-798 artifact that was manually moved into a new-style
    directory.
    """
    from tools.feature_handling_tools.env_handlers import _read_dims

    cfg_obs_dims, cfg_act_dims = _read_dims(cfg)

    for spec_key, cfg_dims in (
        ("obs_dims", cfg_obs_dims),
        ("act_dims", cfg_act_dims),
    ):
        if spec_key not in rb_spec:
            raise ValueError(
                f"Replay buffer spec {str(spec_path)!r} does not declare "
                f"{spec_key!r}. It predates RLRP-798 feature-dimension "
                f"identification and cannot be validated — delete the "
                f"directory and let the pipeline regenerate the buffer."
            )
        spec_dims = list(rb_spec[spec_key])
        if spec_dims != list(cfg_dims):
            raise ValueError(
                f"Replay buffer spec {str(spec_path)!r} {spec_key} mismatch: "
                f"spec={spec_dims} vs cfg environment.{spec_key}="
                f"{list(cfg_dims)}. The cached buffer was generated with a "
                f"different feature-dimension set (RLRP-798); delete the "
                f"directory and let the pipeline regenerate the buffer."
            )


def _resolve_ms_replaybuffer_buffer_dir_path(
    cfg: omegaconf.DictConfig, mbrl_data_root: Union[str, os.PathLike]
) -> str:
    """Build the canonical multistep-replay-buffer directory path.

    Layout (RLRP-544, extended by RLRP-798):
        ``<mbrl_data_root>/<environment.name>/source_size=<value>_obsD=<N>_actD=<M>/HI<h>_HO<ho>/``

    where:
      - ``<environment.name>`` is ``cfg.environment.name`` (the active simulator
        cfg name; mounted via ``simulator@environment`` in hydra defaults).
      - ``<value>`` is ``cfg.source_replay_buffer.source_size`` rendered as-is
        via ``str(...)`` so ``"all"`` stays ``all`` and integers stay numeric
        (e.g. ``100000``).
      - ``obsD=<N>_actD=<M>`` is the RLRP-798 feature-dimension identification
        tag, resolved from the cfg by
        :func:`_resolve_ms_replaybuffer_feature_dim_tag` (i.e. by
        ``resolve_obs_shape`` / ``resolve_act_shape``, never from a buffer
        object) so the save and the load path always agree. It keeps two
        simulator configs sharing an ``environment.name`` lineage but declaring
        different ``obs_dims`` / ``act_dims`` from colliding on one directory.
      - ``HI<h>_HO<ho>`` reflects ``cfg.ms_model.history_len`` and
        ``cfg.ms_model.horizon_len``.

    Missing keys raise the underlying ``omegaconf`` / ``KeyError`` exception
    naming the offending key, so misconfigurations fail fast.
    """
    environment_name = cfg.environment.name
    source_size = cfg.source_replay_buffer.source_size
    history_len = cfg.ms_model.history_len
    horizon_len = cfg.ms_model.horizon_len
    # RLRP-798: the middle segment carries the feature-dimension identification
    # tag so that two simulator configs sharing an ``environment.name`` lineage
    # but declaring different ``obs_dims`` / ``act_dims`` cannot collide.
    feature_dim_tag = _resolve_ms_replaybuffer_feature_dim_tag(cfg)
    return os.path.realpath(
        os.path.join(
            mbrl_data_root,
            str(environment_name),
            f"source_size={source_size}_{feature_dim_tag}",
            f"HI{history_len}_HO{horizon_len}",
        )
    )


def _dtype_to_spec_string(dtype) -> str:
    """Convert a torch or numpy dtype to its spec string representation.

    Handles both numpy type classes (e.g., ``np.float64``) and numpy dtype
    instances (e.g., ``np.dtype('float64')``).
    """
    # Normalize numpy dtype instances to their corresponding type class
    if isinstance(dtype, np.dtype):
        dtype = dtype.type
    try:
        return _DTYPE_TO_SPEC[dtype]
    except KeyError:
        raise ValueError(
            f"Unsupported dtype {dtype!r}. "
            f"Supported dtypes: {list(_DTYPE_TO_SPEC.keys())}"
        )


def _spec_string_to_dtype(spec_string: str):
    """Convert a spec string back to the corresponding torch or numpy dtype."""
    try:
        return _SPEC_TO_DTYPE[spec_string]
    except KeyError:
        raise ValueError(
            f"Unsupported spec string {spec_string!r}. "
            f"Supported spec strings: {list(_SPEC_TO_DTYPE.keys())}"
        )


def save_multistep_replaybuffer_with_spec(
    cfg: omegaconf.DictConfig,
    replay_buffer: ReplayBuffer,
    override_save_dir: Optional[Union[str, os.PathLike]] = None,
) -> Union[str, os.PathLike]:
    """
    Saves a multi-step replay buffer and its specifications.

    This function saves the provided replay buffer and its associated specifications
    to a directory following the layout (RLRP-544, extended by RLRP-798)::

        <save_dir>/<cfg.environment.name>/source_size=<cfg.source_replay_buffer.source_size>_obsD=<N>_actD=<M>/HI<history-len>_HO<horizon-len>/

    The parent ``<save_dir>`` can be configured via hydra ``cfg.pipeline.ms_replaybuffer_save_dir``.
    The ``<environment.name>``, ``source_size`` and ``obsD``/``actD`` segments key
    the saved buffer by the active simulator cfg, the source-dataset sub-sampling
    size and the enabled feature dimensions so that runs from different simulator
    configs / sub-sampling sizes / feature-dimension sets are reproducibly
    isolated.

    RLRP-798: the written ``replaybuffer_spec.yaml`` also records the
    ``obs_dims`` / ``act_dims`` name lists and their counts, making the artifact
    self-describing and validatable by name on load.

    It supports configurable saving paths and save to the hydra current experiment directory
    otherwise. The replay buffer's specifications are serialized into a hydra YAML file.

    :param cfg: Configuration object containing parameters for saving the replay buffer.
    :param replay_buffer: An instance of the ReplayBuffer containing the data to be saved.
    :param override_save_dir: Optional override directory for saving the replay buffer data.
        Accepts both absolute and relative paths. When relative, it is interpreted relative
        to the hydra original cwd (i.e., the directory from which the hydra app was launched).
    :return: The absolute path to the buffer-specific directory where the replay buffer was saved.
    """
    # Capture cwd at entry so we can restore it on exit regardless of outcome
    entry_cwd = os.getcwd()

    #  Dev note: run/debug/test cwd for component using `hydra compose` instead of `hydra main`
    #  when developping in remote dev mode trough ssh in Dockerized-AnonLab container:
    #       cwd='/home/non-interactive-ros2/tmp/MTM-Pro/src'

    # .... Setup ..................................................................................
    hydra_experiment_cwd = get_hydra_experiment_cwd(cfg)
    hydra_orginal_cwd = get_hydra_original_cwd(cfg)

    data_dir = cfg.pipeline.ms_replaybuffer_save_dir

    if override_save_dir:
        # Resolve override_save_dir relative to hydra original cwd when it is not absolute,
        # so the resulting path is deterministic regardless of the caller's cwd.
        override_save_dir_abs = (
            override_save_dir
            if os.path.isabs(override_save_dir)
            else os.path.realpath(os.path.join(hydra_orginal_cwd, override_save_dir))
        )
        mbrl_data_root = os.path.join(override_save_dir_abs, data_dir)
    else:
        mbrl_data_root = os.path.join(hydra_experiment_cwd, data_dir)

    # ... Save replay buffer ......................................................................
    buffer_dir_path = _resolve_ms_replaybuffer_buffer_dir_path(cfg, mbrl_data_root)

    if cfg.debug_mode:
        consol_msg_universal_one_liner(
            f"\n- entry_cwd={entry_cwd!r}\n"
            f"- hydra_experiment_cwd={hydra_experiment_cwd!r}\n"
            f"- hydra_original_cwd={hydra_orginal_cwd!r}\n"
            f"- override_save_dir={override_save_dir!r}\n"
            f"- data_dir={data_dir!r}\n"
            f"- environment.name={cfg.environment.name!r}\n"
            f"- source_replay_buffer.source_size={cfg.source_replay_buffer.source_size!r}\n"
            f"- feature_dim_tag={_resolve_ms_replaybuffer_feature_dim_tag(cfg)!r}\n"
            f"- buffer_dir_path={buffer_dir_path!r}"
        )

    if not os.path.exists(buffer_dir_path):
        consol_msg_universal_one_liner(
            f"Creating directory {buffer_dir_path!r}"
        )
        os.makedirs(buffer_dir_path)

    obs_type = _dtype_to_spec_string(replay_buffer.obs_type)
    action_type = _dtype_to_spec_string(replay_buffer.action_type)
    reward_type = _dtype_to_spec_string(replay_buffer.reward_type)

    # RLRP-798: record the feature-dimension identity of the artifact so the
    # spec is self-describing and the load path can validate it by NAME, not
    # only by width (an equal-width dim rename is invisible in the dir name).
    from tools.feature_handling_tools.env_handlers import _read_dims

    obs_dims, act_dims = _read_dims(cfg)

    replay_buffer_spec = omegaconf.OmegaConf.create(
        {
            "capacity": replay_buffer.capacity,
            "obs_shape": replay_buffer.obs_shape,
            "action_shape": replay_buffer.action_shape,
            "obs_type": obs_type,
            "action_type": action_type,
            "reward_type": reward_type,
            "max_trajectory_length": replay_buffer.max_trajectory_length,
            "stores_trajectories": replay_buffer.stores_trajectories,
            "num_stored": replay_buffer.num_stored,
            "obs_dims": list(obs_dims),
            "act_dims": list(act_dims),
            "obs_dim_count": len(obs_dims),
            "act_dim_count": len(act_dims),
            "motion_model": {
                "history_len": cfg.ms_model.history_len,
                "horizon_len": cfg.ms_model.horizon_len,
            },
        }
    )

    spec_path = os.path.join(buffer_dir_path, "replaybuffer_spec.yaml")
    consol_msg_universal_one_liner(
        f"Writing spec to {spec_path!r}"
    )
    with open(spec_path, "w") as file:
        omegaconf.OmegaConf.save(replay_buffer_spec, file)

    replay_buffer.save(buffer_dir_path)

    # .... Teardown ...............................................................................
    # Restore cwd to the state it was in at function entry (defensive: never leave cwd changed)
    if os.getcwd() != entry_cwd:
        consol_msg_universal_one_liner(
            f"Restoring cwd from {os.getcwd()!r} to {entry_cwd!r}"
        )
        os.chdir(entry_cwd)

    consol_msg_universal_one_liner(
        f"Saved to {buffer_dir_path!r}"
    )
    return buffer_dir_path


def load_multistep_replaybuffer_with_spec(
    cfg: omegaconf.DictConfig,
    override_load_dir: Optional[Union[str, os.PathLike]] = None,
) -> ReplayBuffer | None:
    """
    Loads a multistep replay buffer created from saved configuration.

    This function attempts to load a replay buffer based on the current
    ``ms_model`` history/horizon length, the active simulator cfg name, and the
    source-replay-buffer ``source_size``. It checks for the existence of a
    replay buffer directory following the layout (RLRP-544, extended by
    RLRP-798)::

        <save_dir>/<cfg.environment.name>/source_size=<cfg.source_replay_buffer.source_size>_obsD=<N>_actD=<M>/HI<history-len>_HO<horizon-len>/

    The parent ``<save_dir>`` can be configured via hydra
    ``cfg.pipeline.ms_replaybuffer_save_dir``. It reads the replay buffer saved
    specifications, and initializes the buffer with its parameters.

    RLRP-798: the recorded ``obs_dims`` / ``act_dims`` are validated against the
    active cfg; a name drift, or a spec that does not declare them at all,
    raises a ``ValueError`` instead of silently loading a mismatched buffer. A
    plain cache miss (no such directory) still returns ``None``.

    :param cfg: A dictionary configuration object containing experiment settings and paths.
    :param override_load_dir: An optional override directory for loading the replay buffer.
        Accepts both absolute and relative paths. When relative, it is interpreted relative
        to the hydra original cwd (i.e., the directory from which the hydra app was launched).
    :return: The loaded multistep replay buffer instance, or None if no matching directory exists.
    """
    # Capture cwd at entry so we can restore it on exit regardless of outcome
    entry_cwd = os.getcwd()

    # .... Setup ..................................................................................
    hydra_experiment_cwd = get_hydra_experiment_cwd(cfg)
    hydra_orginal_cwd = get_hydra_original_cwd(cfg)

    data_dir = cfg.pipeline.ms_replaybuffer_save_dir

    if override_load_dir:
        # Resolve override_load_dir relative to hydra original cwd when it is not absolute,
        # so the resulting path is deterministic regardless of the caller's cwd.
        override_load_dir_abs = (
            override_load_dir
            if os.path.isabs(override_load_dir)
            else os.path.realpath(os.path.join(hydra_orginal_cwd, override_load_dir))
        )
        mbrl_data_root = os.path.join(override_load_dir_abs, data_dir)
    else:
        mbrl_data_root = os.path.join(hydra_experiment_cwd, data_dir)

    # ... Load replay buffer ......................................................................
    buffer_dir_path = _resolve_ms_replaybuffer_buffer_dir_path(cfg, mbrl_data_root)

    if cfg.debug_mode:
        consol_msg_universal_one_liner(
            f"- entry_cwd={entry_cwd!r}\n"
            f"- hydra_experiment_cwd={hydra_experiment_cwd!r}\n"
            f"- hydra_original_cwd={hydra_orginal_cwd!r}\n"
            f"- override_load_dir={override_load_dir!r}\n"
            f"- data_dir={data_dir!r}\n"
            f"- environment.name={cfg.environment.name!r}\n"
            f"- source_replay_buffer.source_size={cfg.source_replay_buffer.source_size!r}\n"
            f"- feature_dim_tag={_resolve_ms_replaybuffer_feature_dim_tag(cfg)!r}\n"
            f"- buffer_dir_path={buffer_dir_path!r}"
        )

    if not os.path.exists(buffer_dir_path):
        consol_msg_universal_one_liner(
            f"No multistep replay buffer directory with matching "
            f"model configuration '{buffer_dir_path}', "
            f"skipping multistep replay buffer loading."
        )
        ms_replay_buffer = None
    else:
        spec_path = os.path.join(buffer_dir_path, "replaybuffer_spec.yaml")

        if not os.path.exists(spec_path):
            raise FileNotFoundError(
                f"Replay buffer directory exists "
                f"but spec file is missing {spec_path!r}. "
                f"The directory may be incomplete or corrupted."
            )

        consol_msg_universal_one_liner(
            f"Loading spec from {spec_path!r}"
        )
        with open(spec_path) as file:
            rb_spec = omegaconf.OmegaConf.load(file)

        _validate_ms_replaybuffer_spec_feature_dims(cfg, rb_spec, spec_path)

        obs_type = _spec_string_to_dtype(rb_spec.obs_type)
        action_type = _spec_string_to_dtype(rb_spec.action_type)
        reward_type = _spec_string_to_dtype(rb_spec.reward_type)

        # NOTE (C1 + F-C1b): resolve optional CUDA device + VRAM guard
        # from cfg. When ``cfg.mbrl_lib.keep_replay_buffer_on_device``
        # is false (legacy default), ``resolved_device`` is ``None``
        # and ``ReplayBuffer`` falls through to the pre-patch CPU path
        # — bit-exact with earlier behaviour. Introduced by actions
        # ``C1`` of the RLRC Training Speed & Efficiency ``.junie`` plan
        # and ``F-C1b`` of the stage-1 follow-up ``.junie`` plan.
        from tools.mbrl_lib_tools.setup_utils import resolve_replay_buffer_device

        resolved_device = resolve_replay_buffer_device(
            cfg,
            capacity=rb_spec.capacity,
            obs_shape=tuple(rb_spec.obs_shape),
            action_shape=tuple(rb_spec.action_shape),
            obs_type=obs_type,
            action_type=action_type,
            reward_type=reward_type,
            max_trajectory_length=rb_spec.max_trajectory_length,
        )

        ms_replay_buffer = ReplayBuffer(
            capacity=rb_spec.capacity,
            obs_shape=rb_spec.obs_shape,
            action_shape=rb_spec.action_shape,
            obs_type=obs_type,
            action_type=action_type,
            reward_type=reward_type,
            max_trajectory_length=rb_spec.max_trajectory_length,
            device=resolved_device,
        )

        ms_replay_buffer.load(buffer_dir_path)

        consol_msg_universal_one_liner(
            f"Loaded replay buffer: \n"
            f"- num_stored={ms_replay_buffer.num_stored}\n"
            f"- capacity={ms_replay_buffer.capacity}\n"
            f"- collect_trajectories={ms_replay_buffer.stores_trajectories}\n"
            f"- obs_type={ms_replay_buffer.obs_type}\n"
            f"- action_type={ms_replay_buffer.action_type}\n"
            f"- reward_type={ms_replay_buffer.reward_type}\n"
            f"- max_trajectory_length={ms_replay_buffer.max_trajectory_length}"
        )

    # .... Teardown ...............................................................................
    # Restore cwd to the state it was in at function entry (defensive: never leave cwd changed)
    if os.getcwd() != entry_cwd:
        consol_msg_universal_one_liner(
            f"[load_multistep_replaybuffer_with_spec] "
            f"Restoring cwd from {os.getcwd()!r} to entry_cwd={entry_cwd!r}"
        )
        os.chdir(entry_cwd)

    return ms_replay_buffer
