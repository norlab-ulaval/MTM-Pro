# coding=utf-8
"""Resume-from-checkpoint scan / legality / progress-audit utilities (RLRP-839).

Extends the RLRP-824 training-stage resume (``training_common.resume_from_checkpoint``) so an
experiment can also be resumed at the *test-time rollout deployment stage*. The configured path
may be:

* an ``epoch_<E>`` checkpoint dir or a run root (job dir)  => exactly ONE candidate
  (RLRP-824 semantics preserved);
* a multirun dir or ANY ancestor (date dir / experiment dir) => a recursive **scan** for job
  dirs (a dir holding ``.hydra/overrides.yaml``).

Every discovered candidate is validated against the *current* launch (non-``trial_nb`` task
overrides + ``hydra.runtime.choices`` must be equal — fail fast otherwise) and audited on disk
into a :class:`ResumeExperiment` whose ``stage`` is one of ``train`` | ``deploy`` | ``complete`` |
``fresh``.

Kept dependency-light (paths + IO only, no torch / ERLL / pipeline imports) like
:mod:`persistent_checkpoint_utils`, so a dry-run over a whole experiment tree takes seconds.
Non-interactive by design (SLURM-safe): nothing here prompts; every failure raises with the
offending path and the corrective override in the message.
"""
import datetime
import fnmatch
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from tools.mbrl_lib_tools.persistent_checkpoint_utils import (
    EPOCH_CHECKPOINT_ROLLOUTS_ROOT_NAME,
    EPOCH_CHECKPOINTS_ROOT_NAME,
    NORMALIZERS_ROOT_NAME,
    PathLike,
    SAVED_MODEL_LEAF,
    discover_epoch_checkpoints,
    epoch_checkpoint_rollouts_dir,
    epoch_checkpoint_saved_model_dir,
    select_epoch_checkpoints,
)

# Naming ------------------------------------------------------------------------------
HYDRA_DIRNAME = ".hydra"
OVERRIDES_FNAME = "overrides.yaml"
HYDRA_YAML_FNAME = "hydra.yaml"
TESTTIME_ROLLOUTS_ROOT_NAME = "testtime_rollouts"
UDER_EPOCH_PLOT_DIRNAME = "uder_epoch_plot"
PRIMARY_MODEL_DIR_PREFIX = "model_"
RESUME_LOG_FNAME = "resume_log.json"

#: Stages a candidate can be classified into (see :func:`build_resume_experiment`).
STAGE_TRAIN = "train"
STAGE_DEPLOY = "deploy"
STAGE_COMPLETE = "complete"
STAGE_FRESH = "fresh"
STAGES = (STAGE_TRAIN, STAGE_DEPLOY, STAGE_COMPLETE, STAGE_FRESH)

#: Sub-dirs never descended into during a scan (they can hold thousands of files and never
#: contain a job dir).
SCAN_PRUNED_DIRNAMES = frozenset(
    {
        EPOCH_CHECKPOINTS_ROOT_NAME,
        EPOCH_CHECKPOINT_ROLLOUTS_ROOT_NAME,
        TESTTIME_ROLLOUTS_ROOT_NAME,
        UDER_EPOCH_PLOT_DIRNAME,
        HYDRA_DIRNAME,
        NORMALIZERS_ROOT_NAME,
    }
)

#: Legality defaults: ``trial_nb`` is a seed identifier (never used for matching) and the
#: ``training_common.resume*`` keys are what the resumed launch itself adds.
DEFAULT_IGNORED_OVERRIDE_KEYS: Tuple[str, ...] = ("trial_nb",)
DEFAULT_IGNORED_OVERRIDE_PREFIXES: Tuple[str, ...] = ("training_common.resume",)
#: Config groups whose choice is launcher-specific (``hydra/launcher: joblib`` vs ``basic``...).
DEFAULT_IGNORED_CHOICE_PREFIXES: Tuple[str, ...] = ("hydra/",)


class ResumeLegalityError(RuntimeError):
    """Raised when at least one scanned candidate does not match the current launch."""


@dataclass(frozen=True)
class ResumeCandidate:
    """A job dir discovered under ``training_common.resume_from_checkpoint``.

    :param run_root: absolute job dir (the Hydra cwd of the original run).
    :param multirun_dir: ``dirname(run_root)`` — the unique ``HHMMSSffffff`` multirun dir.
    :param override_dirname: ``basename(run_root)`` — Hydra's ``job.override_dirname``.
    :param task_overrides: ``.hydra/overrides.yaml`` parsed as ``{key: raw_value}``.
    :param choices: ``hydra.runtime.choices`` from ``.hydra/hydra.yaml``; ``None`` when the file
        or the section is absent (legality then fails explicitly).
    :param latest_ckpt_epoch: epoch of the latest cadence checkpoint, ``None`` when there is none.
    :param primary_model_present: ``<run_root>/model_<ClassName>/saved_dynamic_model/`` exists.
    """

    run_root: str
    multirun_dir: str
    override_dirname: str
    task_overrides: Dict[str, str] = field(default_factory=dict)
    choices: Optional[Dict[str, Any]] = None
    latest_ckpt_epoch: Optional[int] = None
    primary_model_present: bool = False

    @property
    def sort_key(self) -> Tuple[str, str]:
        return self.multirun_dir, self.override_dirname


@dataclass(frozen=True)
class ResumeExperiment:
    """The audited decision for one candidate.

    :param stage: ``train`` (ERLL resume, new cwd) | ``deploy`` (in-place completion) |
        ``complete`` (skip) | ``fresh`` (nothing on disk, skip with a warning).
    :param experiment_planned_total_epochs: the static training budget the audit compared against.
    :param main_deploy_complete: every expected ``testtime_rollouts/...`` metric leaf exists.
    :param missing_epoch_rollouts: epochs whose ``epoch_checkpoints_rollouts/epoch_<E>`` tree is
        missing or partial (to (re)run), ascending.
    :param partial_epoch_rollout_dirs: existing-but-incomplete epoch rollout dirs to wipe before
        re-running.
    :param missing_main_deploy_leaves: the missing main-deploy metric leaves (diagnostics only).
    """

    candidate: ResumeCandidate
    stage: str
    experiment_planned_total_epochs: Optional[int]
    main_deploy_complete: bool
    missing_epoch_rollouts: Tuple[int, ...] = ()
    partial_epoch_rollout_dirs: Tuple[str, ...] = ()
    missing_main_deploy_leaves: Tuple[str, ...] = ()

    @property
    def run_root(self) -> str:
        return self.candidate.run_root

    @property
    def needs_work(self) -> bool:
        return self.stage in (STAGE_TRAIN, STAGE_DEPLOY)


# Job dir discovery -------------------------------------------------------------------
def overrides_path(job_dir: PathLike) -> str:
    return os.path.join(str(job_dir), HYDRA_DIRNAME, OVERRIDES_FNAME)


def hydra_yaml_path(job_dir: PathLike) -> str:
    return os.path.join(str(job_dir), HYDRA_DIRNAME, HYDRA_YAML_FNAME)


def is_job_dir(path: PathLike) -> bool:
    """``True`` when *path* is a Hydra job dir, i.e. holds ``.hydra/overrides.yaml``."""
    return os.path.isfile(overrides_path(path))


def is_epoch_checkpoint_dir(path: PathLike) -> bool:
    """``True`` when *path* is an ``epoch_<E>`` dir with a ``saved_dynamic_model/`` leaf."""
    return os.path.isdir(epoch_checkpoint_saved_model_dir(path))


def parse_overrides(lines: Iterable[Any]) -> Dict[str, str]:
    """Parse Hydra override strings (``k=v``) into ``{key: raw_value}``.

    Leading ``+``/``++`` (append) and ``~`` (delete) markers are stripped from the key so the
    same override written with or without them compares equal. A line without ``=`` maps to
    ``""``.
    """
    parsed: Dict[str, str] = {}
    for raw in lines or ():
        line = str(raw).strip()
        if not line:
            continue
        key, sep, value = line.partition("=")
        key = key.lstrip("+~").strip()
        parsed[key] = value.strip() if sep else ""
    return parsed


def read_job_overrides(job_dir: PathLike) -> Dict[str, str]:
    """Read + parse ``<job_dir>/.hydra/overrides.yaml``."""
    with open(overrides_path(job_dir), "r", encoding="utf-8") as handle:
        lines = yaml.safe_load(handle) or []
    return parse_overrides(lines)


def read_job_choices(job_dir: PathLike) -> Optional[Dict[str, Any]]:
    """Read ``hydra.runtime.choices`` from ``<job_dir>/.hydra/hydra.yaml`` (``None`` if absent)."""
    path = hydra_yaml_path(job_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    choices = (((content.get("hydra") or {}).get("runtime") or {}).get("choices"))
    if not isinstance(choices, dict):
        return None
    return dict(choices)


def primary_model_dir(run_root: PathLike, primary_model_dirname: Optional[str] = None) -> Optional[str]:
    """Return the ``<run_root>/model_<ClassName>/`` dir holding a ``saved_dynamic_model/`` leaf.

    :param primary_model_dirname: exact dir name (e.g. ``model_MS2MSEndToEndTCN``); ``None``
        picks the first ``model_*`` dir carrying the leaf (sorted for determinism).
    :return: the absolute dir or ``None`` when absent.
    """
    root = str(run_root)
    if primary_model_dirname is not None:
        candidate = os.path.join(root, primary_model_dirname)
        return candidate if os.path.isdir(os.path.join(candidate, SAVED_MODEL_LEAF)) else None
    if not os.path.isdir(root):
        return None
    for name in sorted(os.listdir(root)):
        if not name.startswith(PRIMARY_MODEL_DIR_PREFIX):
            continue
        candidate = os.path.join(root, name)
        if os.path.isdir(os.path.join(candidate, SAVED_MODEL_LEAF)):
            return candidate
    return None


def make_candidate(run_root: PathLike, primary_model_dirname: Optional[str] = None) -> ResumeCandidate:
    """Build a :class:`ResumeCandidate` from a job dir (reads ``.hydra`` + checkpoints)."""
    root = os.path.abspath(os.fspath(run_root))
    discovered = discover_epoch_checkpoints(root)
    return ResumeCandidate(
        run_root=root,
        multirun_dir=os.path.dirname(root),
        override_dirname=os.path.basename(root),
        task_overrides=read_job_overrides(root) if is_job_dir(root) else {},
        choices=read_job_choices(root),
        latest_ckpt_epoch=discovered[-1][0] if discovered else None,
        primary_model_present=primary_model_dir(root, primary_model_dirname) is not None,
    )


def _is_under(path: str, roots: Sequence[str]) -> bool:
    for root in roots:
        try:
            if os.path.commonpath([path, root]) == root:
                return True
        except ValueError:  # different drives (windows) — cannot be under
            continue
    return False


def scan_resume_candidates(
    resume_from: PathLike,
    exclude_roots: Sequence[PathLike] = (),
    primary_model_dirname: Optional[str] = None,
) -> List[ResumeCandidate]:
    """Resolve ``training_common.resume_from_checkpoint`` into a sorted candidate list.

    * ``epoch_<E>`` dir  => the single run root ``dirname(dirname(path))`` (RLRP-824);
    * job dir            => that single candidate;
    * any other existing dir => recursive scan for job dirs, pruning
      :data:`SCAN_PRUNED_DIRNAMES`, ``model_*`` and anything under *exclude_roots* (the current
      launch's own ``hydra.sweep.dir`` / output dir).

    :return: candidates sorted by ``(multirun_dir, override_dirname)``.
    :raises FileNotFoundError: when the path does not exist or the scan finds no job dir.
    """
    path = os.path.abspath(os.fspath(resume_from))
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"training_common.resume_from_checkpoint={str(resume_from)!r}: directory does not "
            f"exist (resolved to `{path}`)."
        )

    if is_epoch_checkpoint_dir(path):
        return [make_candidate(os.path.dirname(os.path.dirname(path)), primary_model_dirname)]
    if is_job_dir(path) or discover_epoch_checkpoints(path):
        return [make_candidate(path, primary_model_dirname)]

    excluded = [os.path.abspath(os.fspath(each)) for each in exclude_roots if each]
    job_dirs: List[str] = []
    for dirpath, dirnames, _ in os.walk(path):
        if _is_under(dirpath, excluded):
            dirnames[:] = []
            continue
        if is_job_dir(dirpath):
            job_dirs.append(dirpath)
            dirnames[:] = []  # a job dir never nests another job dir
            continue
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in SCAN_PRUNED_DIRNAMES and not name.startswith(PRIMARY_MODEL_DIR_PREFIX)
        )

    if not job_dirs:
        raise FileNotFoundError(
            f"training_common.resume_from_checkpoint={str(resume_from)!r}: scanned `{path}` and "
            f"found no job dir (a dir holding `{HYDRA_DIRNAME}/{OVERRIDES_FNAME}`), no "
            f"`{EPOCH_CHECKPOINTS_ROOT_NAME}/epoch_<E>/` checkpoint and no epoch checkpoint dir."
        )

    candidates = [make_candidate(each, primary_model_dirname) for each in job_dirs]
    candidates.sort(key=lambda candidate: candidate.sort_key)
    return candidates


# Legality ------------------------------------------------------------------------------
def _filter_overrides(
    overrides: Dict[str, Any], ignore_keys: Sequence[str], ignore_prefixes: Sequence[str]
) -> Dict[str, str]:
    kept: Dict[str, str] = {}
    for key, value in overrides.items():
        if key in ignore_keys or any(key.startswith(prefix) for prefix in ignore_prefixes):
            continue
        kept[key] = str(value)
    return kept


def _filter_choices(choices: Dict[str, Any], ignore_prefixes: Sequence[str]) -> Dict[str, str]:
    return {
        str(group): str(choice)
        for group, choice in choices.items()
        if not any(str(group).startswith(prefix) for prefix in ignore_prefixes)
    }


def _diff(left: Dict[str, str], right: Dict[str, str]) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """``{key: (candidate_value, current_value)}`` for every differing key."""
    diff: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            diff[key] = (left.get(key), right.get(key))
    return diff


def candidate_legality_diff(
    candidate: ResumeCandidate,
    current_task_overrides: Iterable[Any],
    current_choices: Optional[Dict[str, Any]],
    ignore_keys: Sequence[str] = DEFAULT_IGNORED_OVERRIDE_KEYS,
    ignore_prefixes: Sequence[str] = DEFAULT_IGNORED_OVERRIDE_PREFIXES,
    ignore_choice_prefixes: Sequence[str] = DEFAULT_IGNORED_CHOICE_PREFIXES,
) -> Dict[str, Any]:
    """Return the legality diff of *candidate* vs the current launch (empty dict ⇔ legal).

    Keys: ``overrides`` (``{key: (candidate, current)}``), ``choices`` (``{group: (candidate,
    current)}``) and/or ``error`` (a message when the candidate metadata is unusable).
    """
    diff: Dict[str, Any] = {}
    current_overrides = _filter_overrides(
        parse_overrides(current_task_overrides), ignore_keys, ignore_prefixes
    )
    candidate_overrides = _filter_overrides(candidate.task_overrides, ignore_keys, ignore_prefixes)
    override_diff = _diff(candidate_overrides, current_overrides)
    if override_diff:
        diff["overrides"] = override_diff

    if candidate.choices is None:
        diff["error"] = (
            f"`{hydra_yaml_path(candidate.run_root)}` is missing or has no "
            f"`hydra.runtime.choices` section (very old run?) — cannot prove legality."
        )
    elif current_choices is None:
        diff["error"] = "current launch exposes no `hydra.runtime.choices` (not under Hydra?)."
    else:
        choice_diff = _diff(
            _filter_choices(candidate.choices, ignore_choice_prefixes),
            _filter_choices(dict(current_choices), ignore_choice_prefixes),
        )
        if choice_diff:
            diff["choices"] = choice_diff
    return diff


def format_legality_diff(run_root: str, diff: Dict[str, Any]) -> str:
    lines = [f"- {run_root}"]
    if "error" in diff:
        lines.append(f"    error: {diff['error']}")
    for section in ("overrides", "choices"):
        if section in diff:
            lines.append(f"    {section}:")
            for key, (candidate_value, current_value) in diff[section].items():
                lines.append(f"      {key}: candidate={candidate_value!r}  current={current_value!r}")
    return "\n".join(lines)


def validate_candidates_legality(
    candidates: Sequence[ResumeCandidate],
    current_task_overrides: Iterable[Any],
    current_choices: Optional[Dict[str, Any]],
    ignore_keys: Sequence[str] = DEFAULT_IGNORED_OVERRIDE_KEYS,
    ignore_prefixes: Sequence[str] = DEFAULT_IGNORED_OVERRIDE_PREFIXES,
    ignore_choice_prefixes: Sequence[str] = DEFAULT_IGNORED_CHOICE_PREFIXES,
) -> None:
    """Fail fast when ANY candidate is illegal w.r.t. the current launch (RLRP-839 FR2).

    :param current_task_overrides: ``HydraConfig.get().overrides.task`` (``k=v`` strings).
    :param current_choices: ``HydraConfig.get().runtime.choices``.
    :raises ResumeLegalityError: listing every offending candidate with its key-by-key diff.
    """
    blocks: List[str] = []
    for candidate in candidates:
        diff = candidate_legality_diff(
            candidate,
            current_task_overrides,
            current_choices,
            ignore_keys=ignore_keys,
            ignore_prefixes=ignore_prefixes,
            ignore_choice_prefixes=ignore_choice_prefixes,
        )
        if diff:
            blocks.append(format_legality_diff(candidate.run_root, diff))
    if blocks:
        raise ResumeLegalityError(
            f"{len(blocks)}/{len(candidates)} resume candidate(s) are NOT legal w.r.t. the current "
            "launch (non-`trial_nb` task overrides and `hydra.runtime.choices` must be equal). "
            "Launch with the SAME multirun config that generated the artifact, or point "
            "`training_common.resume_from_checkpoint` at a narrower path / use "
            "`training_common.resume.candidate_filter`.\n" + "\n".join(blocks)
        )


# Progress audit ------------------------------------------------------------------------
def testtime_rollouts_complete(
    base_dir: PathLike, expected_relative_metric_paths: Sequence[str]
) -> Tuple[bool, List[str]]:
    """Check that every expected metric leaf exists under *base_dir*.

    :param base_dir: a run root (main deploy) or an ``epoch_checkpoints_rollouts/epoch_<E>/`` dir.
    :param expected_relative_metric_paths: leaves such as
        ``testtime_rollouts/<traj>/TTRPM_<desc>_PR_rollout<pr>/InD_compounded_True/
        TestTimeRolloutPredictionMetric.pkl`` (relative to *base_dir*).
    :return: ``(complete, missing_relative_paths)``; an empty expectation is never complete.
    """
    root = str(base_dir)
    missing = [
        relpath
        for relpath in expected_relative_metric_paths
        if not os.path.isfile(os.path.join(root, relpath))
    ]
    complete = bool(expected_relative_metric_paths) and not missing
    return complete, missing


def build_resume_experiment(
    candidate: ResumeCandidate,
    experiment_planned_total_epochs: Optional[int],
    expected_relative_metric_paths: Sequence[str],
    epoch_rollouts_enabled: bool = True,
    epoch_stride: Optional[int] = None,
    epochs: Optional[Sequence[int]] = None,
) -> ResumeExperiment:
    """Audit *candidate* on disk and decide its stage (RLRP-839 FR3/FR4).

    * ``fresh``    — no cadence checkpoint AND no primary model;
    * ``train``    — ``latest_ckpt_epoch < experiment_planned_total_epochs`` or primary model absent;
    * ``deploy``   — main deploy incomplete, or (when *epoch_rollouts_enabled*) any selected epoch
      rollout tree missing / partial;
    * ``complete`` — otherwise.

    :param experiment_planned_total_epochs: the static ERLL budget; ``None`` raises (a patience-driven
      budget cannot be audited — same requirement as RLRP-824).
    :raises ValueError: when *experiment_planned_total_epochs* is ``None`` for a non-``fresh`` candidate.
    """
    root = candidate.run_root
    if candidate.latest_ckpt_epoch is None and not candidate.primary_model_present:
        return ResumeExperiment(
            candidate=candidate,
            stage=STAGE_FRESH,
            experiment_planned_total_epochs=experiment_planned_total_epochs,
            main_deploy_complete=False,
        )

    if experiment_planned_total_epochs is None:
        raise ValueError(
            f"resume: cannot audit `{root}` — the planned total number of epochs is unknown "
            "(a patience-driven / null epoch budget). Set explicit `num_epochs_train_model` / "
            "`final_max_num_epochs_train_model` budgets (RLRP-824 requirement) or resume that run "
            "explicitly with an `epoch_<E>` dir."
        )

    main_complete, missing_main = testtime_rollouts_complete(root, expected_relative_metric_paths)

    missing_epochs: List[int] = []
    partial_dirs: List[str] = []
    if epoch_rollouts_enabled:
        selected = select_epoch_checkpoints(
            discover_epoch_checkpoints(root),
            stride=epoch_stride,
            epochs=list(epochs) if epochs is not None else None,
        )
        for epoch, _ in selected:
            rollout_dir = epoch_checkpoint_rollouts_dir(root, epoch)
            complete, _ = testtime_rollouts_complete(rollout_dir, expected_relative_metric_paths)
            if complete:
                continue
            missing_epochs.append(epoch)
            if os.path.isdir(rollout_dir):
                partial_dirs.append(rollout_dir)

    training_done = (
        candidate.primary_model_present
        and candidate.latest_ckpt_epoch is not None
        and candidate.latest_ckpt_epoch >= int(experiment_planned_total_epochs)
    )
    if not training_done:
        stage = STAGE_TRAIN
    elif not main_complete or missing_epochs:
        stage = STAGE_DEPLOY
    else:
        stage = STAGE_COMPLETE

    return ResumeExperiment(
        candidate=candidate,
        stage=stage,
        experiment_planned_total_epochs=int(experiment_planned_total_epochs),
        main_deploy_complete=main_complete,
        missing_epoch_rollouts=tuple(missing_epochs),
        partial_epoch_rollout_dirs=tuple(partial_dirs),
        missing_main_deploy_leaves=tuple(missing_main),
    )


# Driver helpers ------------------------------------------------------------------------
def filter_candidates(
    candidates: Sequence[ResumeCandidate], candidate_filter: Optional[str]
) -> List[ResumeCandidate]:
    """Narrow *candidates* with ``training_common.resume.candidate_filter``.

    The filter is matched against the run root either as a plain substring or, when it contains
    glob meta-characters (``*``, ``?``, ``[``), as an ``fnmatch`` pattern against the whole path.
    ``None`` / ``""`` keeps everything.
    """
    if candidate_filter is None or str(candidate_filter).strip() == "":
        return list(candidates)
    pattern = str(candidate_filter).strip()
    if any(token in pattern for token in "*?["):
        glob = pattern if pattern.startswith("*") else f"*{pattern}"
        glob = glob if glob.endswith("*") else f"{glob}*"
        return [each for each in candidates if fnmatch.fnmatch(each.run_root, glob)]
    return [each for each in candidates if pattern in each.run_root]


def _fmt_epochs(epochs: Sequence[int]) -> str:
    return ",".join(str(each) for each in epochs) if epochs else "-"


def format_candidate_table(experiments: Sequence[ResumeExperiment]) -> str:
    """Render the audit table printed before anything runs (RLRP-839 FR8)."""
    header = ("#", "stage", "latest_ckpt", "planned", "model", "main_deploy", "missing_epoch_rollouts", "run_root")
    rows: List[Tuple[str, ...]] = []
    for idx, experiment in enumerate(experiments, start=1):
        candidate = experiment.candidate
        rows.append(
            (
                str(idx),
                experiment.stage,
                "-" if candidate.latest_ckpt_epoch is None else str(candidate.latest_ckpt_epoch),
                "-" if experiment.experiment_planned_total_epochs is None else str(experiment.experiment_planned_total_epochs),
                "yes" if candidate.primary_model_present else "no",
                "done" if experiment.main_deploy_complete else "TODO",
                _fmt_epochs(experiment.missing_epoch_rollouts),
                candidate.run_root,
            )
        )
    widths = [max(len(row[col]) for row in (header, *rows)) for col in range(len(header))]

    def render(row: Tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)).rstrip()

    counts = {stage: sum(1 for experiment in experiments if experiment.stage == stage) for stage in STAGES}
    summary = "  ".join(f"{stage}={count}" for stage, count in counts.items())
    lines = [render(header), render(tuple("-" * width for width in widths))]
    lines.extend(render(row) for row in rows)
    lines.append(f"{len(experiments)} candidate(s): {summary}")
    return "\n".join(lines)


def resume_log_path(run_root: PathLike) -> str:
    return os.path.join(str(run_root), RESUME_LOG_FNAME)


def read_resume_log(run_root: PathLike) -> List[Dict[str, Any]]:
    path = resume_log_path(run_root)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        content = json.load(handle)
    return list(content) if isinstance(content, list) else [content]


def append_resume_log(run_root: PathLike, entry: Dict[str, Any]) -> str:
    """Append *entry* (stamped with a UTC ``timestamp``) to ``<run_root>/resume_log.json``.

    The file is a JSON list; non-serialisable values are stringified.
    :return: the log path.
    """
    entries = read_resume_log(run_root)
    stamped = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    stamped.update(entry)
    entries.append(stamped)
    path = resume_log_path(run_root)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2, sort_keys=True, default=str)
    return path
