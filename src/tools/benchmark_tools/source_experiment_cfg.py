# coding=utf-8
"""Compose-from-repo benchmark model source -- mode (d) ``benchmark.source_experiment_cfg``.

Action ``A11`` of the RLRC test-time rollout deployer benchmarking `.junie` plan
(``perf_RLRP-785_testtime_rollout_benchmarking_plan_20260903.md``, rev. 6, ``R18``-``R20``).

Motivation (the driving finding, plan §3.13). Mode (c) (``model_group_config``, ``A9``) benches one
fresh model per group by reading each run's *already-resolved* ``<run_dir>/.hydra/config.yaml``,
which requires the whole experiment tree on the target -- an rsync that takes ≈ 2 h and copies
weights and datasets the bench never touches (only ``ms_model`` + ``environment`` are consumed).
Yet the *same* architecture is already described, in the repo, by the raw multirun **launcher**
entrypoint configs (e.g. ``.../RLRP-757-PI-TCN-B1/multirun-gru.yaml``), whose ``defaults:`` compose
the ``dynamics_model@ms_model`` group and the environment. Mode (d) **Hydra-composes those
in-process** and benches a fresh model directly -- no run tree, no rsync, no checkpoint.

Design invariants (plan §3.13, the three risks made explicit):
  * **Search-root rooting (risk 2).** The raw configs reference *absolute* default groups
    (``/global_config``, ``/main_experiments/...``, ``/dynamics_model@ms_model``), so the compose
    search root MUST be ``src/launcher/configs`` (NOT ``source_experiment_cfg.path``). The compose
    ``config_name`` is the selected config's path **relative to that root**.
  * **Nested-Hydra clear/re-init (risk 1).** The driver runs *inside* the already-running
    ``@hydra.main`` app, so a second ``compose`` would raise. It therefore
    ``GlobalHydra.instance().clear()``, ``initialize_config_dir(...)``, ``compose(...)``, then
    clears again -- one clean compose per selected model, leaving the outer app clean.
  * **Unresolved mandatory key (risk 3, ``QA1``).** ``ms_model`` carries interpolations to the
    mandatory ``A_trj_len`` (``history_len: ${A_trj_len.HI}`` / ``horizon_len: ${A_trj_len.HO}``).
    Resolving the ``ms_model`` / ``environment`` subnodes to literals fails fast (``ValueError``
    naming the offending config and the missing key) if ``overrides`` omitted a ``???`` key -- the
    sweeper's first choice is NEVER auto-picked (test ``T29``).

The resolved ``ms_model`` + ``environment`` nodes are wrapped in a :class:`BenchModelGroup` (keyed
by config name) and handed to the existing :func:`build_fresh_model` of ``model_group_config.py``,
so the robotic-3D precondition (``R10``/``Q3``) and the weightless build path are **shared with
``A9``** -- mode (d) is a new *front-end*, not a second builder.

Math environments (``data_shape``). A MATH env (Lorenz & co) declares no
``environment.obs_dims``/``act_dims``: its single-step widths are stamped into
``environment.obs_shape``/``act_shape`` from the LOADED dataset during a real run, a path this
dataset-free bench never walks -- so ``ms_model.singlestep_obs_len:
${environment.obs_shape[0]}`` had nothing to resolve against and the build refused. Mode (d)
lifts that restriction through ONE optional operator-declared block,
``source_experiment_cfg.data_shape`` (``obs`` / ``act``, validated by
:func:`_coerce_data_shape` and carried on :attr:`BenchModelGroup.data_shape`). It is
**explicit by design**: nothing is inferred from the environment definition, and declaring it on
a robotic env is fail-loud-checked against ``obs_dims``/``act_dims``. Mode (c) has no such key
and stays robotic-3D only.
"""
from __future__ import annotations

import collections.abc
import os
from typing import Any, List, Mapping, Optional, Sequence, Union

import omegaconf

from tools.benchmark_tools.model_group_config import (
    BenchModelGroup,
    _default_root_project_path,
    _ms_model_digest,
)


# =========================================================================================
# Selection coercion (``QA2``/``R19``: a single config name OR a list share one entry point).
# =========================================================================================
def _coerce_selected_models(
    benchmark_selected_models: Union[str, Sequence[str]],
) -> List[str]:
    """A lone string is coerced to a one-item list (``QA2``); a list is validated non-empty."""
    if benchmark_selected_models is None:
        raise ValueError(
            "benchmark.source_experiment_cfg.benchmark_selected_models is required (a single "
            "config name or a list of names); got None."
        )
    if isinstance(benchmark_selected_models, str):
        names = [benchmark_selected_models]
    else:
        names = [str(name) for name in benchmark_selected_models]
    if not names:
        raise ValueError(
            "benchmark.source_experiment_cfg.benchmark_selected_models is empty -- name at "
            "least one launcher config to compose and bench."
        )
    return names


# =========================================================================================
# Per-model overrides (issue RLRP-785): different model groups need different sweep keys.
# =========================================================================================
#
# The base ``overrides`` list is passed verbatim to every composed model, but some models need a
# DIFFERENT mandatory sweep key than the rest -- e.g. the MTM-Pro group requires
# ``A_trj_len={HI:20,HO:10,U:10}`` (an extra ``U``) while the others need ``A_trj_len={HI:20,HO:10}``.
# Re-passing the SAME key twice to :func:`hydra.compose` raises ("Multiple values for ..."), so a
# per-model list cannot simply be appended. Instead the optional ``model_overrides`` mapping
# (config name -> list of override strings) is MERGED over the base list BY KEY: a per-model
# override whose key (the text before the first ``=``, stripped of the ``+``/``~`` compose prefixes)
# matches a base one REPLACES it; a new key is appended. So a shared base + a per-model exception
# stays DRY and never triggers Hydra's duplicate-key error.
def _override_key(override: str) -> str:
    """The config key a Hydra override string targets (``A_trj_len={...}`` -> ``A_trj_len``)."""
    return str(override).split("=", 1)[0].lstrip("+~").strip()


def _merge_overrides(
    base: Sequence[str], extra: Optional[Sequence[str]]
) -> List[str]:
    """Merge ``extra`` over ``base`` BY KEY (extra wins), preserving order (base first).

    A per-model override REPLACES the base override that targets the same key (avoiding Hydra's
    "Multiple values for <key>" duplicate error); a new key is appended after the base ones.
    """
    merged: List[str] = [str(o) for o in base]
    if not extra:
        return merged
    key_to_index = {_override_key(o): i for i, o in enumerate(merged)}
    for override in extra:
        override = str(override)
        key = _override_key(override)
        if key in key_to_index:
            merged[key_to_index[key]] = override
        else:
            key_to_index[key] = len(merged)
            merged.append(override)
    return merged


# =========================================================================================
# Search-root resolution (risk 2): compose MUST root at ``launcher/configs``, not at ``path``.
# =========================================================================================
def _resolve_source_dir(path: str, root_project_path: str) -> str:
    """Resolve ``source_experiment_cfg.path`` to an on-disk directory (fail fast)."""
    candidates = [str(path), os.path.join(root_project_path, str(path))]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.realpath(candidate)
    raise FileNotFoundError(
        "Could not resolve benchmark.source_experiment_cfg.path="
        f"'{path}' to a directory. Tried: " + ", ".join(repr(c) for c in candidates)
    )


def _split_launcher_configs_root(source_dir: str) -> "tuple[str, str]":
    """Split an absolute ``source_dir`` into ``(launcher/configs root, relative sub-dir)``.

    Walks up from ``source_dir`` until the directory *is* ``launcher/configs``. The compose
    search root MUST be that ``launcher/configs`` directory so the absolute default groups the
    raw configs reference (``/global_config``, ``/main_experiments/...``,
    ``/dynamics_model@ms_model``) resolve; the returned relative sub-dir prefixes the compose
    ``config_name`` (POSIX separators, as Hydra expects).
    """
    current = os.path.realpath(source_dir)
    rel_parts: List[str] = []
    while not (
        os.path.basename(current) == "configs"
        and os.path.basename(os.path.dirname(current)) == "launcher"
    ):
        parent = os.path.dirname(current)
        if parent == current:  # reached filesystem root without finding launcher/configs
            raise ValueError(
                f"benchmark.source_experiment_cfg.path='{source_dir}' is not under a "
                f"'launcher/configs' directory. The compose search root must be "
                f"'launcher/configs' so the raw configs' absolute default groups "
                f"(/global_config, /main_experiments/..., /dynamics_model@ms_model) resolve."
            )
        rel_parts.append(os.path.basename(current))
        current = parent
    rel_parts.reverse()
    return current, "/".join(rel_parts)


# =========================================================================================
# QA1 fail-fast + node extraction.
# =========================================================================================
#
# The raw launcher ``ms_model`` node is *not* resolved here: it carries interpolations that only
# the running app / the fresh build can satisfy -- ``${device}`` (the app's global device),
# ``${environment.obs_shape[0]}`` (computed by ``setup_multistep_step_model``), and ``???``
# placeholders such as ``in_size`` (also computed by setup). It is therefore extracted with
# interpolations intact and injected verbatim into the bench cfg by :func:`build_fresh_model`,
# exactly as the running app would compose it. The ONE thing the bench scaffolding cannot supply
# is the mandatory *sweep* key (e.g. ``A_trj_len``) the sweeper normally sets; that is what
# ``overrides`` provides (``QA1``). So the sweep keys are read from the config's own
# ``hydra.sweeper.params``: any that is still ``???`` after ``overrides`` fails fast (naming the
# config and the key), and the resolved ones are captured into ``extra_cfg`` so the composed
# ``ms_model`` interpolations (``history_len: ${A_trj_len.HI}``) resolve during the fresh build.
def _top_key(param_name: str) -> str:
    """The top-level cfg key a Hydra sweeper-param name targets (``ms_model.x`` -> ``ms_model``)."""
    return str(param_name).split(".")[0].split("[")[0].split("@")[0]


def _is_missing_safe(cfg, key: str) -> bool:
    try:
        return omegaconf.OmegaConf.is_missing(cfg, key)
    except Exception:  # pragma: no cover - key absent / not a mandatory slot
        return False


def _sweep_top_keys(raw_config_path: str) -> List[str]:
    """Top-level keys the config's ``hydra.sweeper.params`` sweeps (the mandatory sweep keys)."""
    raw = omegaconf.OmegaConf.load(raw_config_path)
    params = omegaconf.OmegaConf.select(raw, "hydra.sweeper.params", default=None)
    if params is None:
        return []
    seen: List[str] = []
    for param_name in params.keys():
        top = _top_key(param_name)
        if top and top not in seen:
            seen.append(top)
    return seen


def _extract_node(
    composed: "omegaconf.DictConfig", key: str, name: str, config_name: str
) -> "omegaconf.DictConfig":
    """Extract one subnode with interpolations INTACT (resolved later by the fresh build)."""
    node = composed.get(key, None)
    if node is None:
        raise KeyError(
            f"benchmark.source_experiment_cfg: composing model '{name}' from config "
            f"'{config_name}' produced no '{key}' node -- cannot build a fresh model without it."
        )
    container = omegaconf.OmegaConf.to_container(node, resolve=False)
    return omegaconf.OmegaConf.create(container)


def _extract_optional_node(
    composed: "omegaconf.DictConfig", key: str
) -> "Optional[omegaconf.DictConfig]":
    """Extract one subnode with interpolations INTACT, or ``None`` when it is absent.

    Unlike :func:`_extract_node`, a missing node is NOT an error here: it is returned as ``None``
    so :func:`build_fresh_model` simply falls back to whatever the bench cfg already carries.
    """
    node = composed.get(key, None)
    if node is None:
        return None
    container = omegaconf.OmegaConf.to_container(node, resolve=False)
    return omegaconf.OmegaConf.create(container)


def _resolve_sweep_keys(
    composed: "omegaconf.DictConfig",
    sweep_top_keys: Sequence[str],
    name: str,
    config_name: str,
) -> dict:
    """Fail fast (``QA1``) on an unset mandatory sweep key; return the resolved ones as a dict.

    A sweep key still ``???`` after ``overrides`` means the operator omitted it -- raise naming the
    config and the missing key(s), never auto-pick the sweeper's first choice. The resolved keys
    are returned so :func:`build_fresh_model` can inject them (the composed ``ms_model``
    interpolations reference them).
    """
    missing: List[str] = []
    extra_cfg: dict = {}
    for top in sweep_top_keys:
        if _is_missing_safe(composed, top):
            missing.append(top)
            continue
        value = omegaconf.OmegaConf.select(composed, top, throw_on_missing=False, default=None)
        if value is None:
            continue
        if isinstance(value, (omegaconf.DictConfig, omegaconf.ListConfig)):
            extra_cfg[top] = omegaconf.OmegaConf.to_container(value, resolve=True)
        else:
            extra_cfg[top] = value
    if missing:
        raise ValueError(
            f"benchmark.source_experiment_cfg: composing model '{name}' from config "
            f"'{config_name}' left mandatory sweep key(s) {missing} unresolved (still '???'). "
            f"Supply them explicitly via source_experiment_cfg.overrides "
            f"(e.g. \"A_trj_len={{HI:20,HO:10}}\") -- the sweeper's first choice is NOT "
            f"auto-picked (QA1)."
        )
    return extra_cfg


# =========================================================================================
# Math-env single-step widths (explicit, operator-declared -- never inferred).
# =========================================================================================
#: The two (and only two) keys of the ``source_experiment_cfg.data_shape`` block.
_DATA_SHAPE_KEYS = ("obs", "act")


def _coerce_data_shape_width(key: str, value) -> List[int]:
    """Validate ONE declared single-step width and return it as a one-element ``[width]`` list.

    Accepts the ergonomic scalar spelling (``obs: 3``) and the explicit one-element list
    (``obs: [3]``); anything else is an operator error and raises rather than being coerced.
    """
    prefix = f"benchmark.source_experiment_cfg.data_shape.{key}"
    if isinstance(value, omegaconf.ListConfig):
        value = omegaconf.OmegaConf.to_container(value, resolve=True)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                f"{prefix}={list(value)} must be a one-element shape: it declares the "
                f"SINGLE-STEP feature width (a 1-D quantity), not a multi-axis shape."
            )
        width = value[0]
    else:
        width = value
    if isinstance(width, bool) or not isinstance(width, int):
        raise ValueError(
            f"{prefix}={width!r} must be an int (the single-step feature count), "
            f"got {type(width).__name__}."
        )
    if width <= 0:
        raise ValueError(f"{prefix}={width} must be a positive int.")
    return [int(width)]


def _coerce_data_shape(data_shape) -> Optional[dict]:
    """Validate the operator-declared math-env single-step widths (fail-loud, never inferred).

    ``None`` (the default) passes through untouched, leaving the shared
    :func:`build_fresh_model` robotic-only precondition in force. Otherwise BOTH ``obs`` and
    ``act`` are mandatory and an unknown key raises instead of being silently dropped (a typo'd
    key would otherwise look like a successful declaration).

    :param data_shape: the raw ``source_experiment_cfg.data_shape`` node (``DictConfig`` when it
        comes from Hydra, ``dict`` from a direct caller), or ``None``.
    :return: ``{"obs": [<width>], "act": [<width>]}`` as plain ``int`` lists, or ``None``.
    :raise ValueError: on a missing key, an unknown key, or a non-positive / non-int /
        multi-element width.
    """
    if data_shape is None:
        return None
    if isinstance(data_shape, omegaconf.DictConfig):
        data_shape = omegaconf.OmegaConf.to_container(data_shape, resolve=True)
    if not isinstance(data_shape, collections.abc.Mapping):
        raise ValueError(
            f"benchmark.source_experiment_cfg.data_shape must be a mapping with the keys "
            f"{list(_DATA_SHAPE_KEYS)} (the single-step obs/act widths), got "
            f"{type(data_shape).__name__}."
        )
    unknown = [str(k) for k in data_shape.keys() if str(k) not in _DATA_SHAPE_KEYS]
    if unknown:
        raise ValueError(
            f"benchmark.source_experiment_cfg.data_shape carries unknown key(s) {unknown}; "
            f"only {list(_DATA_SHAPE_KEYS)} are supported (they are stamped into "
            f"environment.obs_shape / environment.act_shape)."
        )
    missing = [key for key in _DATA_SHAPE_KEYS if key not in data_shape]
    if missing:
        raise ValueError(
            f"benchmark.source_experiment_cfg.data_shape is missing the mandatory key(s) "
            f"{missing}: BOTH the single-step 'obs' and 'act' widths are needed to resolve a "
            f"math env's ms_model shapes (e.g. Lorenz: obs: [3], act: [1])."
        )
    return {
        key: _coerce_data_shape_width(key, data_shape[key]) for key in _DATA_SHAPE_KEYS
    }


# =========================================================================================
# Public entry point.
# =========================================================================================
def load_selected_source_models(
    path: str,
    benchmark_selected_models: Union[str, Sequence[str]],
    overrides: Optional[Sequence[str]] = None,
    root_project_path: Optional[str] = None,
    model_overrides: Optional[Mapping[str, Sequence[str]]] = None,
    data_shape: Optional[Mapping[str, Any]] = None,
) -> List[BenchModelGroup]:
    """Hydra-compose the selected in-repo launcher configs into fresh-model bench jobs (``A11``).

    For each selected model (``QA2``: a bare string is coerced to a one-item list) the driver
    composes the raw launcher config with the search root pinned to ``launcher/configs`` and
    ``overrides`` passed **verbatim** to :func:`hydra.compose` (``QA1``). When ``model_overrides``
    carries an entry for the model (config name -> list of override strings), those are merged
    over the base ``overrides`` BY KEY (per-model wins) so a model needing a different mandatory
    sweep key -- e.g. the MTM-Pro group's ``A_trj_len={HI:20,HO:10,U:10}`` -- gets it without
    re-passing the same key twice to Hydra. Nested composition is
    isolated with :func:`GlobalHydra.clear` + :func:`initialize_config_dir` per model, leaving
    ``GlobalHydra`` clean afterwards (risk 1, test ``T27``). The ``ms_model`` + ``environment``
    nodes are extracted with interpolations intact; the config's mandatory ``hydra.sweeper.params``
    keys are checked (``QA1`` fail-fast) and the resolved ones captured as ``extra_cfg`` so the
    shared :func:`build_fresh_model` can inject them and resolve the composed ``ms_model``
    interpolations during the weightless build.

    ``data_shape`` (optional, math envs only) is the operator-declared single-step ``obs``/``act``
    width block: validated ONCE by :func:`_coerce_data_shape` and attached to every returned job,
    it is what lets :func:`build_fresh_model` stamp ``environment.obs_shape``/``act_shape`` for an
    env that declares no ``obs_dims``/``act_dims``. Absent -> unchanged robotic-only behaviour.

    :return: one :class:`BenchModelGroup` per selected config, config order preserved.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    names = _coerce_selected_models(benchmark_selected_models)
    base_overrides = [str(o) for o in overrides] if overrides is not None else []
    # Validated ONCE (fail fast before any compose), then shared by every selected model.
    resolved_data_shape = _coerce_data_shape(data_shape)

    root = root_project_path or _default_root_project_path()
    source_dir = _resolve_source_dir(path, root)
    search_root, rel_dir = _split_launcher_configs_root(source_dir)

    resolved: List[BenchModelGroup] = []
    for name in names:
        config_name = f"{rel_dir}/{name}" if rel_dir else str(name)
        raw_config_path = os.path.join(source_dir, f"{name}.yaml")
        per_model = model_overrides.get(name) if model_overrides else None
        overrides_list = _merge_overrides(base_overrides, per_model)

        # Nested Hydra composition (risk 1): one clean compose per model; the outer @hydra.main
        # app is left in a clean state whether compose succeeds or raises.
        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=search_root, version_base=None)
        try:
            composed = compose(config_name=config_name, overrides=overrides_list)
            extra_cfg = _resolve_sweep_keys(
                composed, _sweep_top_keys(raw_config_path), name, config_name
            )
            ms_model_cfg = _extract_node(composed, "ms_model", name, config_name)
            environment_cfg = _extract_node(composed, "environment", name, config_name)
            # The bench config no longer composes a main experiment, so the
            # ``one_dim_transition_model`` node ``setup_multistep_step_model`` wraps the built
            # ensemble with must come from the composed launcher config (whose main-experiment
            # default DOES compose it). Extract with interpolations intact (resolved by the
            # fresh build), exactly like ``ms_model``.
            one_dim_transition_model_cfg = _extract_optional_node(
                composed, "one_dim_transition_model"
            )
        finally:
            GlobalHydra.instance().clear()

        resolved.append(
            BenchModelGroup(
                grp_name=str(name),
                ms_model_cfg=ms_model_cfg,
                environment_cfg=environment_cfg,
                source_run_dir=raw_config_path,
                source_config_digest=_ms_model_digest(ms_model_cfg),
                extra_cfg=extra_cfg or None,
                one_dim_transition_model_cfg=one_dim_transition_model_cfg,
                data_shape=(
                    dict(resolved_data_shape)
                    if resolved_data_shape is not None
                    else None
                ),
            )
        )
    return resolved
