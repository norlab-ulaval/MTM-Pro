# coding=utf-8
"""Resolve a multirun-plot-style ``groups:`` config into fresh-model bench jobs.

Action ``A9`` of the RLRC test-time rollout deployer benchmarking `.junie` plan
(``perf_RLRP-785_testtime_rollout_benchmarking_plan_20260903.md``, rev. 5). Reads each
group's model configuration (``ms_model``) from the saved
``<experiment_path>/<multirun_path>/.hydra/config.yaml`` and hands a FRESH (non-trained)
model to the ``A5`` bench. Weights are intentionally never loaded -- inference latency is
set by architecture, not by trained parameters (operator ``Q1``/design decision).

Design invariants (plan section ``A9`` / §3.6 / §3.6.1):
  * **Read-only.** The driver only *reads* ``.hydra/config.yaml``. Rev. 5 (``R16``/``Q4``)
    dropped the ``bench_models.yaml`` exporter, so this module has no write path at all.
  * **Verify, then trust (``Q1``).** A group is assumed to share ONE architecture across every
    ``experiments[*]`` x ``multirun_paths[*]`` (trials differ only by seed). That assumption is
    *checked*, not trusted: the ``ms_model`` node of every listed run is hashed and any
    disagreement fails fast, naming the offending ``grp_name`` and the two disagreeing run dirs
    (gate ``G16``, test ``T25``). Only then is the first resolvable run dir used as the
    architecture source.
  * **Actionable miss (``Q4``).** An absent experiment directory raises a ``FileNotFoundError``
    that names the ``grp_name`` and the missing path, states that only ``.hydra/config.yaml`` is
    needed (no weights, no dataset), and prints the exact copy-pasteable ``rsync`` command
    (``missing_run_dir_error``, test ``T26``).
  * **Robotic-3D by default (``Q3``).** Dataset-free obs/act shape resolution works out of the
    box only for robotic envs (``environment.obs_dims`` / ``act_dims``). A non-robotic (MATH)
    ``environment`` node -- whose single-step widths are normally stamped from the LOADED
    dataset, which the dataset-free bench never loads -- is supported ONLY when the operator
    declares those widths EXPLICITLY through mode (d)'s
    ``benchmark.source_experiment_cfg.data_shape`` (carried on
    :attr:`BenchModelGroup.data_shape`); nothing is ever inferred. Without it the build raises
    an explicit ``NotImplementedError`` naming that key (§3.6.1).
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import omegaconf


@dataclasses.dataclass
class BenchModelGroup:
    """One resolved bench job: a group's architecture + the env it lives in.

    ``ms_model_cfg`` and ``environment_cfg`` are the two nodes that
    :func:`build_fresh_model` injects into the bench cfg before building; ``environment_cfg``
    is MANDATORY because ``resolve_obs_shape``/``resolve_act_shape`` read
    ``obs_dims``/``act_dims`` from it (rev. 4, ``R10``).
    """

    grp_name: str
    ms_model_cfg: "omegaconf.DictConfig"
    environment_cfg: "omegaconf.DictConfig"
    source_run_dir: str
    source_config_digest: str
    # Rev. 6 (mode (d) / A11): extra top-level nodes to inject into the bench cfg alongside
    # ``ms_model`` + ``environment`` before building (e.g. the operator-supplied sweep key
    # ``A_trj_len`` whose interpolations the composed ``ms_model`` still carries).
    #
    # RLRP-803 (gate ``V8.b``, on-hardware): mode (c) needs this too. A saved multirun
    # ``.hydra/config.yaml`` is NOT fully resolved -- its ``ms_model`` keeps sweep-key
    # interpolations such as ``history_len: ${A_trj_len.HI}``. The root nodes those
    # interpolations reference are therefore captured from the SAME saved config (see
    # :func:`_unresolved_root_keys`) and injected alongside ``ms_model``; without them the
    # fresh build died with ``InterpolationKeyError: Interpolation key 'A_trj_len.HI' not
    # found``. ``None`` -> nothing extra to inject.
    extra_cfg: Optional[dict] = None
    # Rev. 6 (mode (d) / A11): the ``one_dim_transition_model`` node
    # ``setup_multistep_step_model`` reads (``cfg.one_dim_transition_model``) to wrap the built
    # ensemble. In mode (c) it comes from the top-level bench defaults; in mode (d) the bench
    # config no longer composes a main experiment, so the node is extracted from the composed
    # launcher config (which DOES compose it) and injected by :func:`build_fresh_model`.
    # ``None`` -> rely on whatever the bench cfg already carries (mode (c)).
    one_dim_transition_model_cfg: Optional["omegaconf.DictConfig"] = None
    # Mode (d) math-env support: the operator-declared single-step obs/act WIDTHS, as
    # ``{"obs": [<w>], "act": [<w>]}`` (validated by
    # ``source_experiment_cfg._coerce_data_shape``). A MATH env (Lorenz & co) declares no
    # ``obs_dims``/``act_dims``; its shapes are normally stamped into
    # ``environment.obs_shape``/``act_shape`` from the loaded replay buffer by
    # ``setup_source_multi_step_replay_buffer``, a path the dataset-free bench never walks --
    # so ``ms_model.singlestep_obs_len: ${environment.obs_shape[0]}`` had nothing to resolve
    # against. ``build_fresh_model`` stamps these widths instead. NOTHING is inferred: absent
    # (``None``) on a non-robotic env keeps the explicit refusal. Supplying it on a ROBOTIC env
    # is allowed but must AGREE with ``obs_dims``/``act_dims`` (fail-loud on drift).
    data_shape: Optional[dict] = None


#: Root key of every ``${root.sub.sub}`` interpolation, e.g. ``A_trj_len`` in
#: ``${A_trj_len.HI}``. Used to carry a saved run's sweep keys along with its ``ms_model``.
_INTERPOLATION_ROOT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)")


def _unresolved_root_keys(*nodes: Any) -> List[str]:
    """Root keys referenced by the interpolations still present in ``nodes``.

    Introduced by RLRP-803 gate ``V8.b`` (mode (c) on the Jetson AGX Orin). A saved multirun
    ``.hydra/config.yaml`` carries UNRESOLVED sweep-key interpolations inside ``ms_model``
    (``history_len: ${A_trj_len.HI}``), so injecting ``ms_model`` alone into the bench cfg
    cannot resolve. The referenced ROOT nodes are copied from the same saved config, which
    keeps the fresh model bit-identical to the architecture the run actually used.

    :param nodes: config nodes to scan (``None`` entries are ignored).
    :return: the referenced root key names, de-duplicated, in first-seen order.
    """
    found: List[str] = []
    for node in nodes:
        if node is None:
            continue
        text = omegaconf.OmegaConf.to_yaml(node, resolve=False)
        for root_key in _INTERPOLATION_ROOT_RE.findall(text):
            if root_key not in found:
                found.append(root_key)
    return found


# =========================================================================================
# Path / config resolution (mirrors ``main_utils.compute_per_group_metric:606-645``).
# =========================================================================================
def _default_root_project_path() -> str:
    """Best-effort repo root when the caller does not supply one.

    The bench runs from ``<repo>/src`` (per repo rules), and group ``experiment_path`` values
    are repo-root-relative (``artifact/ICRA2026/...``), so the parent of a ``src`` cwd is the
    root. Callers that have a Hydra cfg should pass the value of
    ``fetch_project_root_path_via_hydra(cfg, 1, "MTM-Pro")`` explicitly.
    """
    cwd = os.getcwd()
    if os.path.basename(cwd) == "src":
        return os.path.dirname(cwd)
    return cwd


def _resolve_group_config_path(
    group_config_name_or_path: str, root_project_path: str
) -> str:
    """Resolve a ``model_group_config`` value to an on-disk YAML file.

    Accepts either a direct path (absolute or relative to cwd) or a Hydra-style config name
    such as ``multirun_testtime_rollout_plot/icra2026_pi_tcn_RLRP-757-B1a`` living under
    ``launcher/configs/``. Fails fast with the candidates tried when nothing resolves.
    """
    name = str(group_config_name_or_path)
    if os.path.isfile(name):
        return os.path.realpath(name)

    stem = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    candidates = [
        stem,
        os.path.join("launcher", "configs", stem),
        os.path.join(root_project_path, "src", "launcher", "configs", stem),
        os.path.join(root_project_path, "launcher", "configs", stem),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.realpath(candidate)
    raise FileNotFoundError(
        "Could not resolve benchmark.model_group_config="
        f"'{group_config_name_or_path}' to a file. Tried: "
        + ", ".join(repr(c) for c in candidates)
    )


def _resolve_run_dir(
    root_project_path: str, experiment_path: str, multirun_path: str
) -> str:
    """``realpath(root_project_path / experiment_path / multirun_path)`` (same as the plotter)."""
    return os.path.realpath(
        os.path.join(root_project_path, str(experiment_path), str(multirun_path))
    )


def _iter_group_run_dirs(each_grp: dict, root_project_path: str) -> List[str]:
    """Every ``experiments[*]`` x ``multirun_paths[*]`` run dir for a group, config order."""
    run_dirs: List[str] = []
    for each_experiments in each_grp["experiments"]:
        # Treat both ``None`` and empty list as "no multirun sweep" (mirrors the plotter idiom).
        multirun_paths = each_experiments.get("multirun_paths") or ["."]
        for each_multirun_path in multirun_paths:
            run_dirs.append(
                _resolve_run_dir(
                    root_project_path,
                    each_experiments["experiment_path"],
                    each_multirun_path,
                )
            )
    return run_dirs


def _hydra_config_path(run_dir: str) -> str:
    return os.path.join(run_dir, ".hydra", "config.yaml")


# =========================================================================================
# Errors (rev. 5).
# =========================================================================================
def missing_run_dir_error(grp_name: str, missing_path: str) -> FileNotFoundError:
    """An absent experiment directory must be ACTIONABLE, not cryptic (rev. 5, ``R16``/``Q4``).

    Only ``.hydra/config.yaml`` is needed -- no weights, no dataset -- and the message says so,
    then hands over the exact ``rsync`` command. The offending ``grp_name`` and path are always
    named (test ``T26``).
    """
    # The run dir is ``<repo>/<experiment_path>/<multirun_path>``; copy the whole experiment
    # subtree's configs (the parent that holds the run dirs) so a single command covers a group.
    top = missing_path
    message = (
        f"Missing experiment directory for group '{grp_name}':\n"
        f"  {top}\n"
        f"Only '.hydra/config.yaml' is required (no weights, no dataset).\n"
        f"Copy it to this host with:\n"
        f"  rsync -avz --include='*/' --include='.hydra/config.yaml' --exclude='*' \\\n"
        f"      <host>:<repo>/{top} \\\n"
        f"      {top}"
    )
    return FileNotFoundError(message)


# =========================================================================================
# Consistency assertion (rev. 5, ``R13``/``Q1``, gate ``G16``, test ``T25``).
# =========================================================================================
def _ms_model_digest(ms_model_cfg: Any) -> str:
    """Stable digest of an ``ms_model`` node (order-insensitive via sorted YAML)."""
    yaml_text = omegaconf.OmegaConf.to_yaml(ms_model_cfg, sort_keys=True)
    return hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()


def _load_ms_model_node(grp_name: str, run_dir: str) -> "omegaconf.DictConfig":
    """Load ``<run_dir>/.hydra/config.yaml`` and return its ``ms_model`` node (fail fast)."""
    hydra_cfg_path = _hydra_config_path(run_dir)
    if not os.path.isfile(hydra_cfg_path):
        raise missing_run_dir_error(grp_name, run_dir)
    saved = omegaconf.OmegaConf.load(hydra_cfg_path)
    ms_model = saved.get("ms_model", None)
    if ms_model is None:
        raise KeyError(
            f"group '{grp_name}': the saved config at '{hydra_cfg_path}' has no 'ms_model' "
            f"node -- cannot resolve the architecture for a fresh-model bench."
        )
    return ms_model


def assert_group_architecture_consistency(
    grp_name: str,
    run_dirs: Sequence[str],
) -> str:
    """VERIFY the seed-only assumption instead of trusting it (rev. 5, ``R13``/``Q1``).

    Hash ``OmegaConf.to_yaml(cfg.ms_model)`` of EVERY listed run and compare. Cheap (a YAML load
    per run dir, no model built) and fails fast, naming ``grp_name`` and the two disagreeing run
    dirs (gate ``G16``, test ``T25``). Unresolvable run dirs are reported through
    :func:`missing_run_dir_error`.

    :return: the FIRST run dir (the architecture source) once all listed runs agree.
    """
    if not run_dirs:
        raise ValueError(
            f"group '{grp_name}' lists no experiments/multirun_paths -- nothing to bench."
        )

    reference_digest: Optional[str] = None
    reference_dir: Optional[str] = None
    for run_dir in run_dirs:
        digest = _ms_model_digest(_load_ms_model_node(grp_name, run_dir))
        if reference_digest is None:
            reference_digest = digest
            reference_dir = run_dir
        elif digest != reference_digest:
            raise ValueError(
                f"group '{grp_name}': ms_model architecture disagreement across listed runs "
                f"(operator Q1 assumes a group shares ONE architecture, trials differ only by "
                f"seed). The following two run dirs carry different 'ms_model' nodes:\n"
                f"  {reference_dir}\n"
                f"  {run_dir}\n"
                f"Split them into separate groups or fix the config."
            )
    return reference_dir  # type: ignore[return-value]


# =========================================================================================
# Public entry points.
# =========================================================================================
def _selected_group_names(
    all_group_names: Sequence[str],
    show_grp_names: Optional[Sequence[str]],
    groups_select: Optional[Sequence[str]],
) -> List[str]:
    """Apply the source config's own ``show.grp_names`` first, then narrow by ``groups_select``.

    Rev. 4 (``R9``): ``groups_select`` is an ADDITIONAL, narrower filter -- it never widens the
    ``show.grp_names`` selection. Both are validated against the config's actual group names, so
    an unknown selected ``grp_name`` fails fast rather than silently benching nothing.
    """
    known = list(all_group_names)

    if show_grp_names is not None:
        for name in show_grp_names:
            if name not in known:
                raise KeyError(
                    f"show.grp_names lists '{name}', absent from the group config "
                    f"(known: {known})."
                )
        selected = [name for name in known if name in set(show_grp_names)]
    else:
        selected = list(known)

    if groups_select is not None:
        for name in groups_select:
            if name not in known:
                raise KeyError(
                    f"groups_select lists '{name}', absent from the group config "
                    f"(known: {known})."
                )
        narrowed = set(groups_select)
        selected = [name for name in selected if name in narrowed]

    if not selected:
        raise ValueError(
            "no groups selected after applying show.grp_names / groups_select "
            f"(known groups: {known})."
        )
    return selected


def load_bench_groups(
    group_config_name_or_path: str,
    groups_select: Optional[Sequence[str]] = None,
    root_project_path: Optional[str] = None,
) -> List[BenchModelGroup]:
    """Resolve a multirun-plot-style ``groups:`` config into a list of fresh-model bench jobs.

    For each selected ``groups[i]`` the architecture source is the FIRST resolvable run dir, but
    only after :func:`assert_group_architecture_consistency` confirms every listed run agrees
    (``Q1``). Both ``ms_model`` and the ``environment`` node are extracted from that run's saved
    ``.hydra/config.yaml`` (rev. 4, ``R10``). Paths resolve as
    ``realpath(root_project_path / experiment_path / multirun_path)`` -- the same rule the plot
    pipeline uses. The source config's own ``show.grp_names`` filter is honoured first;
    ``groups_select`` narrows further.

    Fails fast (never silently skips a model) with an informative error when a run dir, its
    ``.hydra/config.yaml``, the ``ms_model`` node or the ``environment`` node is missing, or when
    a selected ``grp_name`` is absent.
    """
    root = root_project_path or _default_root_project_path()
    config_path = _resolve_group_config_path(group_config_name_or_path, root)
    source_cfg = omegaconf.OmegaConf.load(config_path)

    groups = source_cfg.get("groups", None)
    if groups is None:
        raise KeyError(
            f"group config '{config_path}' has no 'groups:' section -- not a "
            f"multirun-plot-style model-group config."
        )
    group_list = list(omegaconf.OmegaConf.to_container(groups, resolve=True))

    all_group_names = [str(g["grp_name"]) for g in group_list]
    show = source_cfg.get("show", None)
    show_grp_names = None
    if show is not None:
        _raw = show.get("grp_names", None)
        if _raw is not None:
            show_grp_names = [str(n) for n in _raw]

    selected_names = _selected_group_names(
        all_group_names,
        show_grp_names,
        [str(n) for n in groups_select] if groups_select is not None else None,
    )

    by_name = {str(g["grp_name"]): g for g in group_list}
    resolved: List[BenchModelGroup] = []
    for grp_name in selected_names:
        each_grp = by_name[grp_name]
        run_dirs = _iter_group_run_dirs(each_grp, root)
        # Q1: verify every listed run shares one architecture, then take the first as the source.
        architecture_dir = assert_group_architecture_consistency(grp_name, run_dirs)

        hydra_cfg_path = _hydra_config_path(architecture_dir)
        saved = omegaconf.OmegaConf.load(hydra_cfg_path)
        ms_model_cfg = saved.get("ms_model", None)
        if ms_model_cfg is None:
            raise KeyError(
                f"group '{grp_name}': '{hydra_cfg_path}' has no 'ms_model' node."
            )
        environment_cfg = saved.get("environment", None)
        if environment_cfg is None:
            raise KeyError(
                f"group '{grp_name}': '{hydra_cfg_path}' has no 'environment' node -- it is "
                f"MANDATORY (obs_dims/act_dims live there and drive the single-step shapes)."
            )

        # rev. 12 (mode (c) parity with mode (d)): the bench launcher config no longer composes a
        # main experiment, so the ``one_dim_transition_model`` node that
        # ``setup_multistep_step_model`` wraps the fresh ensemble with is absent from the bench
        # cfg. The run's own ``.hydra/config.yaml`` carries it, so capture it here (OPTIONAL --
        # older runs without the node keep working) and let ``build_fresh_model`` inject it,
        # exactly as the A11 driver does; otherwise mode (c) fails with
        # ``Key 'one_dim_transition_model' is not in struct``.
        one_dim_transition_model_cfg = saved.get("one_dim_transition_model", None)

        # RLRP-803 (gate ``V8.b``): carry the sweep-key root nodes the saved ``ms_model``
        # interpolations still reference (e.g. ``A_trj_len``) -- see
        # :func:`_unresolved_root_keys`. Only keys the saved config itself defines are
        # carried; anything else (``${device}`` etc.) resolves against the bench scaffolding.
        extra_cfg: Dict[str, Any] = {}
        for root_key in _unresolved_root_keys(
            ms_model_cfg, one_dim_transition_model_cfg
        ):
            node = saved.get(root_key, None)
            if node is not None:
                extra_cfg[root_key] = node

        resolved.append(
            BenchModelGroup(
                grp_name=grp_name,
                ms_model_cfg=ms_model_cfg,
                environment_cfg=environment_cfg,
                source_run_dir=architecture_dir,
                source_config_digest=_ms_model_digest(ms_model_cfg),
                one_dim_transition_model_cfg=one_dim_transition_model_cfg,
                extra_cfg=extra_cfg or None,
            )
        )
    return resolved


def _environment_is_robotic(environment_cfg: Any) -> bool:
    """A robotic env declares ``obs_dims``/``act_dims`` (dataset-free shape resolution)."""
    obs_dims = None
    act_dims = None
    if environment_cfg is not None:
        try:
            obs_dims = environment_cfg.get("obs_dims", None)
            act_dims = environment_cfg.get("act_dims", None)
        except Exception:  # pragma: no cover - defensive, non-DictConfig node
            return False
    return bool(obs_dims) and bool(act_dims)


def _stamp_explicit_data_shape(working_cfg, group: BenchModelGroup) -> None:
    """Stamp the operator-declared single-step widths into ``environment.obs_shape``/``act_shape``.

    Mode (d) math-env support. A MATH ``environment`` node declares no ``obs_dims``/``act_dims``
    and the bench loads no dataset, so the widths ``ms_model.singlestep_obs_len:
    ${environment.obs_shape[0]}`` interpolates cannot be resolved from the config alone. The
    operator therefore declares them EXPLICITLY (``benchmark.source_experiment_cfg.data_shape``)
    and they are stamped here -- exactly what ``setup_source_multi_step_replay_buffer`` does from
    the loaded replay buffer during a real run.

    The declared width is first run through ``resolve_obs_shape``/``resolve_act_shape``, so a
    ROBOTIC env keeps its ``obs_dims``/``act_dims`` as the source of truth and any drift between
    the two fails loud (same contract as ``env_handlers._resolve_shape``) instead of the operator
    value silently winning.

    :param working_cfg: the bench cfg copy both saved nodes were already injected into.
    :param group: the bench job carrying the validated ``data_shape``.
    :raise ValueError: when the declared width disagrees with the declared dimension names.
    """
    from tools.feature_handling_tools.env_handlers import (
        resolve_act_shape,
        resolve_obs_shape,
    )

    declared = {
        "obs_shape": (list(group.data_shape["obs"]), resolve_obs_shape),
        "act_shape": (list(group.data_shape["act"]), resolve_act_shape),
    }
    for key, (declared_shape, resolver) in declared.items():
        resolved = resolver(working_cfg, data_shape=declared_shape)
        if [int(s) for s in resolved] != [int(s) for s in declared_shape]:
            raise ValueError(
                f"group '{group.grp_name}': the operator-declared "
                f"benchmark.source_experiment_cfg.data_shape yields environment.{key}="
                f"{declared_shape}, which disagrees with the authoritative "
                f"{resolved} resolved from the environment's declared dimension names. "
                f"A robotic env resolves its single-step widths from "
                f"'obs_dims'/'act_dims' -- drop the 'data_shape' block or align it "
                f"(RLRP-736 shape-key removal contract)."
            )
        with omegaconf.read_write(working_cfg), omegaconf.open_dict(working_cfg):
            omegaconf.OmegaConf.update(
                working_cfg, f"environment.{key}", resolved, merge=False
            )
    return None


def build_fresh_model(group: BenchModelGroup, bench_cfg, device: str):
    """Build a FRESH (weightless) model for one group (rev. 4, ``R10``).

    This is NOT simply ``setup_multistep_step_model(cfg, None)``: with ``load_pretrained_path=
    None`` that function builds from ``cfg.ms_model`` of the CURRENT run; the ``.hydra``
    reconstruction (``setup.py:540-545``) lives only on the checkpoint branch. So the driver
    first injects BOTH saved nodes into the bench cfg, then builds -- weights are never loaded.

    Shape resolution (``Q3``/``R15``). A ROBOTIC ``environment`` node resolves its single-step
    obs/act widths dataset-free from ``obs_dims``/``act_dims``. A MATH env declares neither (the
    widths are normally stamped from the loaded dataset, which this bench never loads), so it is
    supported ONLY through mode (d)'s explicit
    ``benchmark.source_experiment_cfg.data_shape`` (:attr:`BenchModelGroup.data_shape`), stamped
    by :func:`_stamp_explicit_data_shape`. Nothing is inferred: a non-robotic env with no
    declared ``data_shape`` raises an explicit ``NotImplementedError`` naming that key, stated up
    front rather than discovered on the Orin. Mode (c) (a saved run's ``.hydra/config.yaml``) has
    no such key, so it stays robotic-only.
    """
    if group.data_shape is None and not _environment_is_robotic(group.environment_cfg):
        raise NotImplementedError(
            f"group '{group.grp_name}': the 'environment' node declares no "
            f"'obs_dims'/'act_dims' (a MATH env), so the single-step obs/act shapes cannot be "
            f"resolved -- they are normally stamped from the loaded dataset, which the "
            f"dataset-free bench never loads. In mode (d) declare them explicitly, e.g.\n"
            f"    benchmark.source_experiment_cfg.data_shape:\n"
            f"      obs: [3]   # single-step observation width (Lorenz: 3 state axes)\n"
            f"      act: [1]   # single-step action width (math env: the time axis)\n"
            f"Nothing is inferred (operator Q3). Mode (c) (model_group_config) has no such key "
            f"and remains robotic-3D only."
        )

    working_cfg = bench_cfg.copy()
    with omegaconf.read_write(working_cfg), omegaconf.open_dict(working_cfg):
        omegaconf.OmegaConf.update(
            working_cfg, "ms_model", group.ms_model_cfg, merge=False
        )
        omegaconf.OmegaConf.update(
            working_cfg, "environment", group.environment_cfg, merge=False
        )
        # Rev. 6 (mode (d)): inject the operator-supplied sweep keys the composed ``ms_model``
        # interpolations still reference (e.g. ``A_trj_len``) so they resolve during setup; the
        # rest of the interpolations (``${device}`` etc.) resolve against the bench scaffolding.
        if group.extra_cfg:
            for extra_key, extra_value in group.extra_cfg.items():
                omegaconf.OmegaConf.update(
                    working_cfg, str(extra_key), extra_value, merge=False
                )
        # Rev. 6 (mode (d)): the bench config no longer composes a main experiment, so the
        # ``one_dim_transition_model`` node ``setup_multistep_step_model`` wraps the ensemble
        # with is absent from the top-level cfg. Inject the one extracted from the composed
        # launcher config (interpolations intact -- resolved during the fresh build).
        if group.one_dim_transition_model_cfg is not None:
            omegaconf.OmegaConf.update(
                working_cfg,
                "one_dim_transition_model",
                group.one_dim_transition_model_cfg,
                merge=False,
            )

    # Mode (d) math-env support: stamp the operator-declared single-step widths AFTER the
    # ``environment`` node was injected (it is the node they belong to), so the
    # ``ms_model.singlestep_{obs,act}_len: ${environment.{obs,act}_shape[0]}`` interpolations
    # every math-env launcher config carries resolve during the fresh build.
    if group.data_shape is not None:
        _stamp_explicit_data_shape(working_cfg, group)

    from pipeline.pipeline_utils.general.setup import setup_multistep_step_model

    container = setup_multistep_step_model(
        working_cfg, load_pretrained_path=None, disable_compile=True
    )
    return container, working_cfg
