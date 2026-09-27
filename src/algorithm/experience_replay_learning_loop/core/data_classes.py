# coding=utf-8
import os
import pickle
from dataclasses import MISSING, dataclass, field, fields
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
import torch

from tools.multistep_tools.models import ExponentialFamilyMLP

if TYPE_CHECKING:
    # RLRP-785 A3 — type-only import. ``BenchmarkMetricSet`` is a plain, torch-free,
    # JSON-safe dataclass (``tools.benchmark_tools.benchmark_metric``); referencing it via a
    # forward-ref string keeps ``data_classes`` free of any runtime import of the benchmark
    # package, so the default (OFF) deploy path never imports it.
    from tools.benchmark_tools.benchmark_metric import BenchmarkMetricSet

# Use the highest float64 value as a sentinel instead of np.inf since it causse problem with
# Optuna databse.
_SENTINEL = np.finfo(np.float64).max

@dataclass
class PostTrainScore:
    """
    Represents the scores and metrics of a post-training evaluation.

    This class is used to store and manage metrics and statistics relevant to post-training
    performance evaluation for a machine learning model. It contains attributes for various
    losses, prediction scores, and the number of saved models during training.

    :ivar best_pred_mae_score: The best Mean Absolute Error (MAE) score achieved during prediction.
    :type best_pred_mae_score: float
    :ivar best_pred_l2_norm_score: The best L2 (Euclidean) norm score achieved during prediction.
    :type best_pred_l2_norm_score: float
    :ivar best_pred_val_loss: The lowest validation loss observed during training.
    :type best_pred_val_loss: float
    :ivar nb_saved_model: The number of models that were saved during the training process.
    :type nb_saved_model: int
    :ivar new_pred_std_score: The standard deviation of the new predictions' scores.
    :type new_pred_std_score: float
    :ivar new_pred_std_epi_score: The standard deviation of the epistemic uncertainty in
        the new predictions.
    :type new_pred_std_epi_score: float
    :ivar new_pred_mae_score: The Mean Absolute Error (MAE) score of new predictions.
    :type new_pred_mae_score: float
    :ivar new_pred_l2_norm_score: The L2 (Euclidean) norm score of new predictions.
    :type new_pred_l2_norm_score: float
    :ivar total_train_losses: A list of total training losses recorded during the training process.
    :type total_train_losses: list
    :ivar total_val_losses: A list of total validation losses recorded during the training process.
    :type total_val_losses: list
    """
    best_pred_mae_score: float = _SENTINEL
    best_pred_l2_norm_score: float = _SENTINEL
    best_pred_val_loss: float = _SENTINEL
    nb_saved_model: int = 0
    new_pred_std_score: float = _SENTINEL
    new_pred_std_epi_score: float = _SENTINEL
    new_pred_mae_score: float = _SENTINEL
    new_pred_l2_norm_score: float = _SENTINEL
    total_train_losses: list = field(default_factory=lambda: [])
    total_val_losses: list = field(default_factory=lambda: [])


@dataclass
class MsTestTimeDeployResult:
    """
    Result of a multistep model test-time deployment rollout sweep.

    Carries the unchanged per-rollout statistics sequence (``run_stats``) plus two
    aggregated, compounded-prediction-mode MAE scalars — one for the InD target
    rollout set and one for the OOD target rollout set. Each scalar is the mean,
    over that set's target trajectories, of the per-trajectory MAE reduced with the
    same reduction used for ``PostTrainScore.best_pred_mae_score`` (RLRP-723 F2:
    cumulative — sum over steps — then ``mean`` over the coordinate/feature axis, so
    the scalar is a true MAE in meters). Buckets with no eligible (compounded) rollout fall back to
    ``_SENTINEL`` so HPO runs never crash on a missing objective record.

    :ivar run_stats: The unchanged per-rollout statistics tuples.
    :type run_stats: tuple
    :ivar compounded_pred_mae_target_InD: Mean compounded-mode MAE over the InD target trajectories.
    :type compounded_pred_mae_target_InD: float
    :ivar compounded_pred_mae_target_OOD: Mean compounded-mode MAE over the OOD target trajectories.
    :type compounded_pred_mae_target_OOD: float
    """
    run_stats: tuple = field(default_factory=tuple)
    compounded_pred_mae_target_InD: float = _SENTINEL
    compounded_pred_mae_target_OOD: float = _SENTINEL


@dataclass
class PredictionMetric:
    """
    Represents various metrics related to predictions and their statistical
    properties.

    This class is used to encapsulate information about predictions,
    including the observed values, mean, standard deviation, epistemic
    uncertainty, and mean absolute error (MAE).

    :ivar pred_obs: The observed prediction values.
    :type pred_obs: Union[np.ndarray, torch.Tensor]
    :ivar mean: The mean of the predictions.
    :type mean: Union[np.ndarray, torch.Tensor]
    :ivar std: The standard deviation of the predictions.
    :type std: Union[np.ndarray, torch.Tensor]
    :ivar std_epi: The epistemic uncertainty in the predictions.
    :type std_epi: Union[np.ndarray, torch.Tensor]
    :ivar mae: The mean absolute error of the predictions.
    :type mae: Union[np.ndarray, torch.Tensor]
    :ivar l2_norm: The L2 (Euclidean) vector norm of the predictions over the coordinate axis.
    :type l2_norm: Union[np.ndarray, torch.Tensor]
    """
    pred_obs: Union[np.ndarray, torch.Tensor]
    mean: Union[np.ndarray, torch.Tensor]
    std: Union[np.ndarray, torch.Tensor]
    std_epi: Union[np.ndarray, torch.Tensor]
    mae: Union[np.ndarray, torch.Tensor]
    l2_norm: Union[np.ndarray, torch.Tensor]


@dataclass
class TestTimeRolloutPredictionMetric:
    """
    Represents metrics collected during test-time rollout prediction evaluations.

    This class is designed to store, save, and load metrics related to the evaluation
    of prediction models during a test-time rollout. It includes attributes for storing
    model-related information, metric values, and runtime details.

    :ivar model_name: The name of the model used in the prediction.
    :type model_name: Optional[str]
    :ivar model_description: A brief description of the model.
    :type model_description: Optional[str]
    :ivar comment: Any additional comments or notes.
    :type comment: Optional[str]
    :ivar mean: The mean values of collected metrics.
    :type mean: Optional[Union[np.ndarray, torch.Tensor]]
    :ivar std: The standard deviations of collected metrics.
    :type std: Optional[Union[np.ndarray, torch.Tensor]]
    :ivar std_epi: The epistemic uncertainty (standard deviation) of collected metrics.
    :type std_epi: Optional[Union[np.ndarray, torch.Tensor]]
    :ivar mae: The mean absolute error (MAE) of predictions.
    :type mae: Optional[Union[np.ndarray, torch.Tensor]]
    :ivar l2_norm: The L2 (Euclidean) vector norm of predictions over the coordinate axis.
    :type l2_norm: Optional[Union[np.ndarray, torch.Tensor]]
    :ivar target: The target or ground truth values used for evaluation.
    :type target: Optional[Union[np.ndarray, torch.Tensor]]
    :ivar target_is_ood: Indicates whether the target is out-of-distribution (OOD) or in-distribution (InD).
    :type target_is_ood: Optional[bool]
    :ivar compounded_predictions_score: Indicates if metrics are from compounded predictions.
    :type compounded_predictions_score: Optional[bool]
    :ivar training_wall_clock_time: Total wall-clock time taken during model training.
    :type training_wall_clock_time: float
    :ivar rollout_wall_clock_time: Total wall-clock time taken for rollout prediction.
    :type rollout_wall_clock_time: float
    :ivar normalizer_type: The ``normalizer_type`` the producing run was trained
        with (RLRP-761 ``P4``). Recorded so a downstream aggregation can tell
        whether two cells are comparable.
    :type normalizer_type: Optional[str]
    :ivar stats_space: The unit space of ``mean`` / ``std`` / ``std_epi`` —
        ``"physical"``, ``"normalized"`` or ``"unknown"`` (RLRP-761 ``P4``).
        ``mae`` / ``l2_norm`` / ``target`` are ALWAYS physical and are not
        governed by this field.
    :type stats_space: Optional[str]
    :ivar obs_target: The OBS-space ground-truth trajectory ``(T, obs_dim)``
        recorded at rollout time (RLRP-761 ``S8.16`` option A). Distinct from
        ``target`` (the 3-D integrated world POSE): the adverse-event ``M2``
        slice ranks timesteps by the terrain-vibration / weight-transfer triplet
        magnitude, which lives in observation space, not pose space. ``None`` on
        the legacy single-step path and on runs recorded before this field.
    :type obs_target: Optional[Union[np.ndarray, torch.Tensor]]
    :ivar obs_mae: The OBS-space per-timestep (feature-mean) absolute error
        ``(T,)`` between the physical prediction mean and ``obs_target`` (RLRP-761
        ``S8.16`` option A). This is the obs-space error the ``M2`` slice slices;
        ``mae`` remains the POSE error. ``None`` when ``obs_target`` is ``None``.
    :type obs_mae: Optional[Union[np.ndarray, torch.Tensor]]
    """
    model_name: Optional[str] = None
    model_description: Optional[str] = None
    comment: Optional[str] = None
    mean: Optional[Union[np.ndarray, torch.Tensor]] = None
    std: Optional[Union[np.ndarray, torch.Tensor]] = None
    std_epi: Optional[Union[np.ndarray, torch.Tensor]] = None
    mae: Optional[Union[np.ndarray, torch.Tensor]] = None
    l2_norm: Optional[Union[np.ndarray, torch.Tensor]] = None
    target: Optional[Union[np.ndarray, torch.Tensor]] = None
    target_is_ood: Optional[bool] = None
    compounded_predictions_score: Optional[bool] = None
    training_wall_clock_time: float = None
    rollout_wall_clock_time: float = None
    # RLRP-761 P4.1 — the unit-space contract of the STATISTICS fields.
    normalizer_type: Optional[str] = None
    stats_space: Optional[str] = None
    # RLRP-761 S8.16 (option A) — the OBS-space GT trajectory + obs-space per-step
    # error, recorded so the adverse-event ``M2`` slice is scorable on robotic-3D
    # (where ``target`` / ``mae`` are the 3-D world pose, not obs-space).
    obs_target: Optional[Union[np.ndarray, torch.Tensor]] = None
    obs_mae: Optional[Union[np.ndarray, torch.Tensor]] = None
    # RLRP-785 A3 — carry the rollout step count and the benchmark metric set through
    # persistence so the reporting chain can finally express a *rate* (Hz == fps) instead of a
    # total-seconds stopwatch (defect D3). ``rollout_steps`` is the divisor for a harness-level
    # rate; ``benchmark`` is the ``A7`` carrier holding every measured level/regime/timing-pass.
    # Both MUST be plain immutable ``= None`` defaults and NOT ``field(default_factory=...)``:
    # unpickling a legacy artifact restores ``__dict__`` without calling ``__init__``, and only
    # immutable defaults become class attributes that ``__setstate__`` (and the fallback plot
    # path) can resolve on a pre-existing ``.pkl`` (plan constraint, verified by ``T6``).
    rollout_steps: Optional[int] = None
    benchmark: Optional["BenchmarkMetricSet"] = None
    # probabilistic_rollout: Optional[bool] = None

    #: RLRP-761 P4.3 — legacy runs whose ``normalizer_type`` says the target was
    #: raw are genuinely PHYSICAL (``standard`` has no output normalizer), so
    #: defaulting every legacy artifact to ``"normalized"`` would make the P4.4
    #: guard reject cells that are in fact comparable.
    _RAW_TARGET_NORMALIZER_TYPES = frozenset({"standard", None})

    def __setstate__(self, state: dict) -> None:
        """Restore a pickled instance, filling fields absent from legacy payloads.

        RLRP-761 ``P4.1b``. :meth:`save` uses ``pickle.dump(self, f)`` and
        unpickling a dataclass restores ``__dict__`` **without calling**
        ``__init__`` — so a legacy ``TestTimeRolloutPredictionMetric.pkl`` written
        before ``stats_space`` existed would yield an instance with **no such
        attribute** and raise ``AttributeError`` on read, *not* the field default.
        Defaulted fields alone are therefore NOT sufficient for back-compat.
        """
        self.__dict__.update(state)
        for field_ in fields(self):
            if field_.name in self.__dict__:
                continue
            default = field_.default if field_.default is not MISSING else None
            setattr(self, field_.name, default)
        if state.get("stats_space", None) is None:
            self.stats_space = self._resolve_legacy_stats_space(
                state.get("normalizer_type", None)
            )
        return None

    @classmethod
    def _resolve_legacy_stats_space(cls, normalizer_type: Optional[str]) -> str:
        """Infer the space of a legacy artifact from its recorded normalizer type."""
        if normalizer_type is None:
            # Neither field recorded: refuse to guess (P4.3).
            return "unknown"
        if normalizer_type in cls._RAW_TARGET_NORMALIZER_TYPES:
            return "physical"
        # Pre-P1 artifact of a block-facade run: the statistics were left in the
        # normalized target space.
        return "normalized"

    def set_name_and_description_from_model(
        self, model: Union[ExponentialFamilyMLP, Any]
    ) -> None:
        """Sets the name and description of the object based on the provided model.

        :param model: The model used to set the name and description.
        :return: None.
        """
        self.model_name = model.__class__.__name__
        if hasattr(model, "description"):
            self.model_description = model.description
        return None

    def save(self, path: str) -> None:
        """Saves the current instance of the object to a file.

        The method serializes the object and writes it to a file named
         "TestTimeRolloutPredictionMetric_MODEL_NAME" in the specified directory path.

        :param path: The directory path where the file will be saved.
        :return: None
        """
        if (
            self.compounded_predictions_score is not None
            and self.target_is_ood is not None
        ):
            path = os.path.join(
                path,
                get_metric_sub_dir(
                    self.compounded_predictions_score, self.target_is_ood
                ),
            )

        path = os.path.realpath(path)
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(
            path,
            f"TestTimeRolloutPredictionMetric.pkl",
        )
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

        assert os.path.exists(file_path), (
            f"Something went wrong! Unable to find saved "
            f"TestTimeRolloutPredictionMetric object at {file_path=}!"
        )
        return None

    @staticmethod
    def load(
        path: str,
        compounded_predictions_score: Optional[bool] = None,
        target_is_ood: Optional[bool] = None,
    ) -> "TestTimeRolloutPredictionMetric":
        """Load a TestTimeRolloutPredictionMetric from a file.

        Usage:

        >>> loaded_metric = TestTimeRolloutPredictionMetric.load("/path/to/dir")

        :param path: Directory path or full file path
        :param compounded_predictions_score: Indicates if the collected metric are from compounded
            predictions or not.
        :param target_is_ood: Indicates whether the target is out-of-distribution (OOD)
            or in-distribution (InD).
        :return: A TestTimeRolloutPredictionMetric instance
        """
        if compounded_predictions_score is not None and target_is_ood is not None:
            path = os.path.join(
                path,
                get_metric_sub_dir(compounded_predictions_score, target_is_ood),
            )

        path = os.path.realpath(path)
        assert os.path.isdir(path), f"{path=} is not a directory!"
        file_path = os.path.join(
            path,
            f"TestTimeRolloutPredictionMetric.pkl",
        )
        assert os.path.exists(file_path), f"{file_path=} does not exist!"

        with open(file_path, "rb") as f:
            if not torch.cuda.is_available():
                # The pickled object may contain torch tensors saved on a CUDA device.
                # When loading on a CPU-only machine, torch.storage._load_from_bytes
                # invokes torch.load() internally, which fails without map_location.
                # Temporarily patch torch.load to map storages to CPU.
                import functools as _functools

                _original_torch_load = torch.load
                torch.load = _functools.partial(
                    _original_torch_load, map_location=torch.device("cpu")
                )
                try:
                    return pickle.load(f)
                finally:
                    torch.load = _original_torch_load
            return pickle.load(f)


def get_metric_sub_dir(compounded_predictions_score: bool, target_is_ood: bool) -> str:
    """Determines and returns a saved TestTimeRolloutPredictionMetric object sub-directory name
    based on compounded-prediction-score and target type (in-distribution or out-of-distribution).

    :param compounded_predictions_score: Indicates if the collected metric are from compounded
        predictions or not.
    :param target_is_ood: Indicates whether the target is out-of-distribution (OOD)
        or in-distribution (InD).
    :return: A string representing the saved object sub-directory name, composed of
        the target distribution identifier and the compounded predictions status.
    """
    metric_sub_dir = "InD"
    if target_is_ood:
        metric_sub_dir = "OOD"
    metric_sub_dir = f"{metric_sub_dir}_compounded_{compounded_predictions_score}"
    return metric_sub_dir
