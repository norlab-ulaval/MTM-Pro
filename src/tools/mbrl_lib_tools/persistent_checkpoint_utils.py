# coding=utf-8
"""Persistent (epoch-interval) model+optimizer checkpoint utilities (RLRP-773).

Single source of truth for the *epoch checkpoint* feature: the on-disk layout,
the save/discover/select/load IO, and the small JSON manifest. Kept dependency-light
(paths + IO only, no ERLL/pipeline imports) so both the save side (ERLL training loop)
and the deploy/regenerate side can consume it without a circular import.

On-disk layout (sibling to the best-metric ``model_<ClassName>/`` dir, so the
best-metric checkpoint logic stays structurally independent — RLRP-773 R5)::

    <exp_cwd>/
      model_<ClassName>/                 # UNCHANGED best-metric checkpoint
      epoch_checkpoints/                 # this module owns this tree
        epoch_000100/
          saved_dynamic_model/           # OneDTransitionRewardModelV2.save(...) output
          optimizer.pth                  # torch.save(optimizer.state_dict())
          checkpoint_meta.json           # plain-scalar manifest (greppable/diffable)
        epoch_000200/
          ...

The ``checkpoint_meta.json`` manifest holds only plain scalars, e.g.::

    {"epoch": 100, "erll_pass": 0, "pass_epoch": 99,
     "weights_provenance": "last_epoch", "interval": 20, "schema_version": 1}

``weights_provenance`` is one of ``{"last_epoch", "pass_best_val"}`` (see RLRP-773 R11):
cadence snapshots hold *last-epoch* weights, whereas the optional ``*_pass_best`` snapshot
taken right after a ``ModelTrainer.train(...)`` return holds the best-val weights the pass
was rewound to.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# Directory / file naming --------------------------------------------------------------
EPOCH_CHECKPOINTS_ROOT_NAME = "epoch_checkpoints"
EPOCH_CHECKPOINT_ROLLOUTS_ROOT_NAME = "epoch_checkpoints_rollouts"
SAVED_MODEL_LEAF = "saved_dynamic_model"
OPTIMIZER_FNAME = "optimizer.pth"
META_FNAME = "checkpoint_meta.json"
#: RLRP-773 (§11): single run-wide "running latest" optimizer file + its manifest, written at the
#: ROOT of ``epoch_checkpoints/`` when ``include_optimizer == "latest"`` (overwritten on each save).
OPTIMIZER_LATEST_FNAME = "optimizer_latest.pth"
OPTIMIZER_LATEST_META_FNAME = "optimizer_latest_meta.json"

#: RLRP-773 (§11): allowed values of the tri-state ``training_common.checkpoint.include_optimizer``.
#: ``None`` => no optimizer persisted; ``"all"`` => an ``optimizer.pth`` in every epoch dir;
#: ``"latest"`` => a single run-wide :data:`OPTIMIZER_LATEST_FNAME`.
INCLUDE_OPTIMIZER_MODES = (None, "all", "latest")

#: RLRP-824 (FR5/FR6): run-root restart artifacts, siblings of ``epoch_checkpoints/``.
#: ``normalizers/`` holds the fitted normalizer statistics written right after
#: ``update_normalizer`` (``input_normalizer/`` + ``output_normalizer/``, the layout
#: ``OneDTransitionRewardModelV2.save`` nests under ``saved_dynamic_model/``);
#: ``data_split_record.json`` is the sample-level train/val ``DataSplitRecord`` of the lazy
#: window dataset (``pipeline.data_manager: dataloader``).
NORMALIZERS_ROOT_NAME = "normalizers"
SPLIT_RECORD_FNAME = "data_split_record.json"

#: Zero-padding width of the ``epoch_<E>`` directory name (``epoch_000100``).
EPOCH_DIR_PAD = 6
#: Suffix used for the optional pass-boundary best-val re-snapshot (RLRP-773 R11).
PASS_BEST_SUFFIX = "pass_best"
#: Schema version stamped into every manifest so future readers can branch on it.
CHECKPOINT_META_SCHEMA_VERSION = 1

#: Matches ONLY cadence dirs ``epoch_<digits>`` (NOT ``epoch_<digits>_pass_best``), so
#: discovery ignores the optional pass-boundary snapshots unless asked explicitly.
_EPOCH_DIR_RE = re.compile(r"^epoch_(\d+)$")

PathLike = Union[str, "os.PathLike[str]"]


def epoch_checkpoints_root(exp_cwd: PathLike) -> str:
    """Return the ``epoch_checkpoints/`` root under *exp_cwd*."""
    return os.path.join(str(exp_cwd), EPOCH_CHECKPOINTS_ROOT_NAME)


def epoch_checkpoint_rollouts_root(base_dir: PathLike) -> str:
    """Return the per-epoch rollout output root ``epoch_checkpoints_rollouts/`` under *base_dir*."""
    return os.path.join(str(base_dir), EPOCH_CHECKPOINT_ROLLOUTS_ROOT_NAME)


def epoch_checkpoint_dir(
    exp_cwd: PathLike, epoch: int, dir_suffix: Optional[str] = None
) -> str:
    """Return the (zero-padded) checkpoint dir for *epoch* under *exp_cwd*.

    :param exp_cwd: the experiment cwd holding the ``epoch_checkpoints/`` tree.
    :param epoch: the global completed inner-epoch count this checkpoint is tagged with.
    :param dir_suffix: optional suffix (e.g. :data:`PASS_BEST_SUFFIX`) appended after the
        epoch number, yielding ``epoch_<E>_<suffix>``.
    :return: the absolute checkpoint directory path.
    """
    name = f"epoch_{int(epoch):0{EPOCH_DIR_PAD}d}"
    if dir_suffix:
        name = f"{name}_{dir_suffix}"
    return os.path.join(epoch_checkpoints_root(exp_cwd), name)


def epoch_checkpoint_saved_model_dir(ckpt_dir: PathLike) -> str:
    """Return the nested ``saved_dynamic_model/`` dir inside a checkpoint dir.

    This mirrors the best-metric ``model_<ClassName>/saved_dynamic_model`` layout so the
    SAME :class:`OneDTransitionRewardModelV2.load(...)` restores an epoch checkpoint.
    """
    return os.path.join(str(ckpt_dir), SAVED_MODEL_LEAF)


def epoch_checkpoint_rollouts_dir(base_dir: PathLike, epoch: int) -> str:
    """Return the per-epoch rollout output dir ``epoch_checkpoints_rollouts/epoch_<E>/``."""
    return os.path.join(
        epoch_checkpoint_rollouts_root(base_dir),
        f"epoch_{int(epoch):0{EPOCH_DIR_PAD}d}",
    )


def normalizers_snapshot_dir(exp_cwd: PathLike) -> str:
    """Return the run-root ``normalizers/`` snapshot dir under *exp_cwd* (RLRP-824 FR6)."""
    return os.path.join(str(exp_cwd), NORMALIZERS_ROOT_NAME)


def split_record_path(exp_cwd: PathLike) -> str:
    """Return the run-root ``data_split_record.json`` path under *exp_cwd* (RLRP-824 FR5)."""
    return os.path.join(str(exp_cwd), SPLIT_RECORD_FNAME)


def save_normalizers_snapshot(model: Any, exp_cwd: PathLike) -> str:
    """Write the run-root ``normalizers/`` snapshot (RLRP-824 FR6).

    :param model: a :class:`OneDTransitionRewardModelV2` (or compatible) exposing
        ``save_normalizers(dir)``.
    :param exp_cwd: the experiment cwd.
    :return: the absolute snapshot directory that was written.
    """
    snapshot_dir = normalizers_snapshot_dir(exp_cwd)
    os.makedirs(snapshot_dir, exist_ok=True)
    model.save_normalizers(snapshot_dir)
    return snapshot_dir


def load_normalizers_snapshot(model: Any, exp_cwd: PathLike) -> bool:
    """Restore the run-root ``normalizers/`` snapshot in place (RLRP-824 FR7).

    :return: ``True`` if a snapshot dir was found and loaded, ``False`` otherwise.
    """
    snapshot_dir = normalizers_snapshot_dir(exp_cwd)
    if not os.path.isdir(snapshot_dir):
        return False
    model.load_normalizers(snapshot_dir)
    return True


def save_epoch_checkpoint(
    model: Any,
    optimizer: Optional[torch.optim.Optimizer],
    exp_cwd: PathLike,
    epoch: int,
    extra_meta: Optional[Dict[str, Any]] = None,
    dir_suffix: Optional[str] = None,
    include_normalizers: bool = True,
) -> str:
    """Persist a ``(model, optimizer)`` snapshot tagged with *epoch* (RLRP-773 R1/R2).

    Writes ``saved_dynamic_model/`` (weights + normalizers, via ``model.save``),
    ``optimizer.pth`` (``optimizer.state_dict()``) and a ``checkpoint_meta.json`` manifest.

    :param model: a :class:`OneDTransitionRewardModelV2` (or compatible) exposing ``save(dir)``.
    :param optimizer: the training optimizer (a single ``Adam`` over all params, even for
        ensembles); ``None`` skips the optimizer file.
    :param exp_cwd: the experiment cwd holding the ``epoch_checkpoints/`` tree.
    :param epoch: the global completed inner-epoch count this checkpoint is tagged with.
    :param extra_meta: extra plain-scalar fields merged into the manifest (e.g. ``erll_pass``,
        ``pass_epoch``, ``weights_provenance``, ``interval``).
    :param dir_suffix: optional dir suffix (e.g. :data:`PASS_BEST_SUFFIX`).
    :param include_normalizers: RLRP-824 (FR6, ``training_common.checkpoint.
        save_normalizer_every_epoch``): ``False`` writes the weights only (the statistics live in
        the run-root ``normalizers/`` snapshot); recorded in the manifest as
        ``normalizers_included``. Default ``True`` = legacy layout.
    :return: the absolute checkpoint directory that was written.
    """
    ckpt_dir = epoch_checkpoint_dir(exp_cwd, epoch, dir_suffix=dir_suffix)
    saved_model_dir = epoch_checkpoint_saved_model_dir(ckpt_dir)
    os.makedirs(saved_model_dir, exist_ok=True)

    if include_normalizers:
        model.save(saved_model_dir)
    else:
        model.save(saved_model_dir, include_normalizers=False)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, OPTIMIZER_FNAME))

    meta: Dict[str, Any] = {
        "epoch": int(epoch),
        "schema_version": CHECKPOINT_META_SCHEMA_VERSION,
        "normalizers_included": bool(include_normalizers),
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(ckpt_dir, META_FNAME), "w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file, indent=2, sort_keys=True)

    return ckpt_dir


def discover_epoch_checkpoints(exp_cwd: PathLike) -> List[Tuple[int, str]]:
    """Discover cadence epoch checkpoints under *exp_cwd*, epoch-sorted.

    Only ``epoch_<digits>`` dirs that contain a ``saved_dynamic_model/`` sub-dir are
    returned; malformed dirs and the optional ``*_pass_best`` snapshots are skipped.

    :param exp_cwd: the experiment cwd holding the ``epoch_checkpoints/`` tree.
    :return: an ascending list of ``(epoch, checkpoint_dir)`` tuples (empty when the tree is
        absent — the legacy-safe no-op case).
    """
    root = epoch_checkpoints_root(exp_cwd)
    if not os.path.isdir(root):
        return []

    discovered: List[Tuple[int, str]] = []
    for name in sorted(os.listdir(root)):
        match = _EPOCH_DIR_RE.match(name)
        if match is None:
            continue
        ckpt_dir = os.path.join(root, name)
        if not os.path.isdir(epoch_checkpoint_saved_model_dir(ckpt_dir)):
            continue
        discovered.append((int(match.group(1)), ckpt_dir))

    discovered.sort(key=lambda item: item[0])
    return discovered


def select_epoch_checkpoints(
    discovered: List[Tuple[int, str]],
    stride: Optional[int] = None,
    epochs: Optional[List[int]] = None,
) -> List[Tuple[int, str]]:
    """Narrow a discovered checkpoint list (RLRP-773 R12 subset selection).

    ``epochs`` (an explicit list) wins over ``stride`` (keep every *stride*-th discovered
    checkpoint). The LAST epoch is ALWAYS retained so the drift curve reaches ``E``.

    :param discovered: the ascending ``(epoch, dir)`` list from :func:`discover_epoch_checkpoints`.
    :param stride: keep every *stride*-th checkpoint (``None``/``<1`` => keep all).
    :param epochs: explicit epoch subset (``None`` => no explicit subset).
    :return: the selected ``(epoch, dir)`` list, ascending and de-duplicated.
    """
    if not discovered:
        return []

    if epochs is not None:
        wanted = {int(each) for each in epochs}
        selected = [item for item in discovered if item[0] in wanted]
    elif stride is not None and int(stride) >= 1:
        selected = [item for idx, item in enumerate(discovered) if idx % int(stride) == 0]
    else:
        selected = list(discovered)

    # Always keep the last (final) epoch so the curve reaches E.
    by_epoch: Dict[int, str] = {epoch: ckpt_dir for epoch, ckpt_dir in selected}
    last_epoch, last_dir = discovered[-1]
    by_epoch.setdefault(last_epoch, last_dir)

    return [(epoch, by_epoch[epoch]) for epoch in sorted(by_epoch)]


def _resolve_map_location():
    """Resolve the torch ``map_location`` consistently with the model loader."""
    try:
        from mbrl.util.common import resolve_load_map_location

        return resolve_load_map_location()
    except Exception:  # pragma: no cover - defensive fallback
        return "cuda" if torch.cuda.is_available() else "cpu"


def epoch_checkpoint_has_normalizers(ckpt_dir: PathLike) -> bool:
    """``True`` when the epoch checkpoint carries its own normalizer statistics (RLRP-824 FR6).

    Reads the ``normalizers_included`` manifest flag; checkpoints written before the flag existed
    (RLRP-773 layout) always include them.
    """
    return bool(read_epoch_checkpoint_meta(ckpt_dir).get("normalizers_included", True))


def load_epoch_checkpoint_model(
    model: Any, ckpt_dir: PathLike, include_normalizers: Optional[bool] = None
) -> None:
    """Load an epoch checkpoint's weights IN PLACE into *model* (RLRP-773 R6).

    :param model: a :class:`OneDTransitionRewardModelV2` (or compatible) exposing ``load(dir)``.
    :param ckpt_dir: an epoch checkpoint dir (from :func:`discover_epoch_checkpoints`).
    :param include_normalizers: also restore the normalizer statistics stored in the checkpoint;
        ``None`` (default) follows the manifest (:func:`epoch_checkpoint_has_normalizers`), so a
        weights-only checkpoint keeps the statistics currently held by *model* (RLRP-824 FR7).
    """
    if include_normalizers is None:
        include_normalizers = epoch_checkpoint_has_normalizers(ckpt_dir)
    if include_normalizers:
        model.load(epoch_checkpoint_saved_model_dir(ckpt_dir))
    else:
        model.load(epoch_checkpoint_saved_model_dir(ckpt_dir), include_normalizers=False)
    return None


def resolve_resume_checkpoint(resume_from: PathLike) -> Tuple[str, str, Dict[str, Any]]:
    """Resolve ``training_common.resume_from_checkpoint`` (RLRP-824 FR7).

    *resume_from* is either an ``epoch_<E>`` checkpoint dir or a run root (an experiment cwd
    holding ``epoch_checkpoints/``), in which case the LATEST cadence checkpoint is selected.

    :return: ``(ckpt_dir, run_root, meta)`` where ``run_root`` is the experiment cwd the
        checkpoint belongs to (holding the ``normalizers/`` snapshot and the split record) and
        ``meta`` the checkpoint manifest (``meta["epoch"]`` is the resume epoch offset).
    :raises FileNotFoundError: when no usable checkpoint is found.
    """
    path = os.path.abspath(os.fspath(resume_from))
    if os.path.isdir(epoch_checkpoint_saved_model_dir(path)):
        ckpt_dir = path
        run_root = os.path.dirname(os.path.dirname(path))
    else:
        discovered = discover_epoch_checkpoints(path)
        if not discovered:
            raise FileNotFoundError(
                f"training_common.resume_from_checkpoint={resume_from!r}: neither an epoch "
                f"checkpoint dir (with a `{SAVED_MODEL_LEAF}/` leaf) nor a run root holding "
                f"`{EPOCH_CHECKPOINTS_ROOT_NAME}/epoch_<E>/` checkpoints."
            )
        _, ckpt_dir = discovered[-1]
        run_root = path
    meta = read_epoch_checkpoint_meta(ckpt_dir)
    if "epoch" not in meta:
        raise FileNotFoundError(
            f"resume_from_checkpoint: `{ckpt_dir}` has no `{META_FNAME}` manifest (epoch unknown)."
        )
    return ckpt_dir, run_root, meta


def load_epoch_checkpoint_optimizer(
    optimizer: torch.optim.Optimizer, ckpt_dir: PathLike
) -> bool:
    """Restore an optimizer ``state_dict`` from an epoch checkpoint (RLRP-773 Q4).

    Implemented for future training-resume use; resume itself is out of scope for RLRP-773.

    :param optimizer: the optimizer to restore in place.
    :param ckpt_dir: an epoch checkpoint dir.
    :return: ``True`` if an ``optimizer.pth`` was found and loaded, ``False`` otherwise.
    """
    optimizer_path = os.path.join(str(ckpt_dir), OPTIMIZER_FNAME)
    if not os.path.isfile(optimizer_path):
        return False
    state_dict = torch.load(
        optimizer_path, map_location=_resolve_map_location(), weights_only=False
    )
    optimizer.load_state_dict(state_dict)
    return True


def read_epoch_checkpoint_meta(ckpt_dir: PathLike) -> Dict[str, Any]:
    """Read the ``checkpoint_meta.json`` manifest of an epoch checkpoint dir.

    :param ckpt_dir: an epoch checkpoint dir.
    :return: the manifest dict (empty when the manifest file is absent).
    """
    meta_path = os.path.join(str(ckpt_dir), META_FNAME)
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as meta_file:
        return json.load(meta_file)


def normalize_include_optimizer(value: Any) -> Optional[str]:
    """Normalize a ``training_common.checkpoint.include_optimizer`` value (RLRP-773 §11).

    Accepts the new tri-state (``null``/``"all"``/``"latest"``) AND the legacy bool so
    un-migrated configs keep working:

    * ``None``, ``False``, ``"none"``, ``"null"``, ``""`` => ``None`` (no optimizer persisted);
    * ``True``, ``"all"``                                => ``"all"`` (per-epoch ``optimizer.pth``);
    * ``"latest"``                                        => ``"latest"`` (single run-wide file).

    Matching is case-insensitive for strings.

    :param value: the raw config value.
    :return: one of :data:`INCLUDE_OPTIMIZER_MODES` (``None``, ``"all"`` or ``"latest"``).
    :raises ValueError: on any other value (typo guard).
    """
    if value is None or value is False:
        return None
    if value is True:
        return "all"
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("", "none", "null", "false"):
            return None
        if token in ("all", "true"):
            return "all"
        if token == "latest":
            return "latest"
    raise ValueError(
        "`training_common.checkpoint.include_optimizer` must be one of "
        "{null, all, latest} (legacy bool accepted); "
        f"got {value!r}."
    )


def running_latest_optimizer_path(exp_cwd: PathLike) -> str:
    """Return the single run-wide running-latest optimizer file path (RLRP-773 §11).

    :param exp_cwd: the experiment cwd holding the ``epoch_checkpoints/`` tree.
    :return: ``<exp_cwd>/epoch_checkpoints/optimizer_latest.pth``.
    """
    return os.path.join(epoch_checkpoints_root(exp_cwd), OPTIMIZER_LATEST_FNAME)


def save_running_latest_optimizer(
    exp_cwd: PathLike,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Overwrite the single run-wide running-latest optimizer file (RLRP-773 §11, Q8/Q9).

    Used when ``include_optimizer == "latest"``: instead of an ``optimizer.pth`` in every epoch
    dir, a single :data:`OPTIMIZER_LATEST_FNAME` at the ``epoch_checkpoints/`` root is overwritten
    on EACH save (cadence and, when enabled, pass-best — whichever is most recent), keeping the
    optimizer disk cost O(1). A sibling :data:`OPTIMIZER_LATEST_META_FNAME` records which epoch /
    provenance the file corresponds to.

    :param exp_cwd: the experiment cwd holding the ``epoch_checkpoints/`` tree.
    :param optimizer: the training optimizer whose ``state_dict`` is persisted.
    :param epoch: the global completed inner-epoch count this optimizer state corresponds to.
    :param extra_meta: extra plain-scalar fields merged into the manifest (e.g.
        ``weights_provenance``, ``erll_pass``).
    :return: the absolute path of the written optimizer file.
    """
    root = epoch_checkpoints_root(exp_cwd)
    os.makedirs(root, exist_ok=True)

    optimizer_path = os.path.join(root, OPTIMIZER_LATEST_FNAME)
    torch.save(optimizer.state_dict(), optimizer_path)

    meta: Dict[str, Any] = {
        "epoch": int(epoch),
        "schema_version": CHECKPOINT_META_SCHEMA_VERSION,
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(
        os.path.join(root, OPTIMIZER_LATEST_META_FNAME), "w", encoding="utf-8"
    ) as meta_file:
        json.dump(meta, meta_file, indent=2, sort_keys=True)

    return optimizer_path


def load_running_latest_optimizer(
    optimizer: torch.optim.Optimizer, exp_cwd: PathLike
) -> bool:
    """Restore the run-wide running-latest optimizer ``state_dict`` (RLRP-773 §11, Q4).

    Parity with :func:`load_epoch_checkpoint_optimizer`; provided for future training-resume use
    (resume itself is out of scope).

    :param optimizer: the optimizer to restore in place.
    :param exp_cwd: the experiment cwd holding the ``epoch_checkpoints/`` tree.
    :return: ``True`` if the running-latest optimizer file was found and loaded, ``False`` otherwise.
    """
    optimizer_path = running_latest_optimizer_path(exp_cwd)
    if not os.path.isfile(optimizer_path):
        return False
    state_dict = torch.load(
        optimizer_path, map_location=_resolve_map_location(), weights_only=False
    )
    optimizer.load_state_dict(state_dict)
    return True


def read_running_latest_optimizer_meta(exp_cwd: PathLike) -> Dict[str, Any]:
    """Read the running-latest optimizer manifest (RLRP-773 §11).

    :param exp_cwd: the experiment cwd holding the ``epoch_checkpoints/`` tree.
    :return: the manifest dict (empty when the manifest file is absent).
    """
    meta_path = os.path.join(
        epoch_checkpoints_root(exp_cwd), OPTIMIZER_LATEST_META_FNAME
    )
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as meta_file:
        return json.load(meta_file)
