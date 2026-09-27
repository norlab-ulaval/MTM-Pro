# coding=utf-8
"""``performance_mode: dev | fast`` seam shared by the MTM-Pro and the AR MS2SS model families.

RLRP-786 FR8 / Key Decision 8 (``perf_RLRP-786_cuda_graph_train_step_plan_20260919.md``): the
development / debug-only instrumentation met on the training step is resolved ONCE from the
single pipeline key ``pipeline.performance_mode`` into a model-level attribute, and each hook
tests that attribute. ``dev`` keeps every diagnostic (default, bit-exact with the historical
behaviour); ``fast`` is the paper-run mode and the precondition of the CUDA-graph captured step.

Extracted from ``MS2MS2SSArTemporalMixturePME`` by the AR-baseline extension
(``perf_RLRP-786_ar_tcn_cuda_graph_fp32_baseline_plan_20260919.md``, Step 1) so that
``CompoundedPredictionMultiStepIterator`` (hence ``MS2SSProbabilisticTCN`` / GRU / LSTM / MLP-AR)
shares the very same seam -- ``setup.py`` keeps calling ``set_performance_mode`` unchanged.

The switch removes instrumentation only, never arithmetic: ``loss``, every parameter gradient
and the Adam state are ``torch.equal`` between the two modes.
"""
from typing import List, Tuple

from tools.math_tools.weigthing import set_non_finite_guard_enabled
from mbrl.util.normalization import (
    set_finite_guard_enabled as set_normalizer_finite_guard_enabled,
)


class PerformanceModeMixin:
    """Post-construction ``performance_mode`` seam (mirrors ``set_meta_collection_enabled``).

    Host requirement: an ``nn.Module`` (``self.modules()`` is used to propagate the mode to
    owned submodules declaring ``_performance_mode``).
    """

    #: ``dev`` (default, today's behaviour) | ``fast`` (paper-run mode: dev/debug-only
    #: instrumentation OFF; precondition of the CUDA-graph captured step). Driven by
    #: ``pipeline.performance_mode`` through :meth:`set_performance_mode` (``setup.py``).
    _performance_mode: str = "dev"
    PERFORMANCE_MODES: Tuple[str, ...] = ("dev", "fast")

    @property
    def performance_mode(self) -> str:
        return self._performance_mode

    def _refuse_fast_mode_reasons(self) -> List[str]:
        """Host hook: reasons why ``fast`` must be REFUSED (raises in :meth:`set_performance_mode`).

        Empty by default. ``MS2MS2SSArTemporalMixturePME`` reports its class-level
        ``_GLOBAL_DEBUG`` sentinel path here (it synchronises inside the loss).
        """
        return []

    def set_performance_mode(self, mode: str) -> None:
        """Resolve ``pipeline.performance_mode`` into the model (RLRP-786 FR8).

        ``dev``: every development / debug diagnostic of the training step stays on (the
        sync-free non-finite guard kernels of ``weight_values`` / ``check_finite`` and their
        once-per-step ``flush_non_finite_reports()`` host read-back, the synchronising
        ``isfinite().all()`` guard of the mbrl-lib normalizers). ``fast``: those are OFF.
        Propagated to owned submodules declaring ``_performance_mode``.

        :raises ValueError: unknown mode, or ``fast`` refused by :meth:`_refuse_fast_mode_reasons`.
        """
        mode = str(mode).lower()
        if mode not in self.PERFORMANCE_MODES:
            raise ValueError(
                f"performance_mode={mode!r} is not one of {self.PERFORMANCE_MODES}"
            )
        if mode == "fast":
            reasons = self._refuse_fast_mode_reasons()
            if reasons:
                raise ValueError(
                    "performance_mode='fast' is incompatible with " + "; ".join(reasons)
                )
        self._performance_mode = mode
        for _submodule in self.modules():
            if _submodule is not self and hasattr(_submodule, "_performance_mode"):
                _submodule._performance_mode = mode
        # The weighting guard is module-level state (shared by every weighting call site), and so is
        # the mbrl-lib normalizer guard (operator decision 2026-09-19: the fork may be modified; the
        # chordal geometry loss denormalises INSIDE the loss, so the guard's host sync would
        # otherwise block the CUDA-graph capture of the UAV paper path).
        set_non_finite_guard_enabled(mode == "dev")
        set_normalizer_finite_guard_enabled(mode == "dev")
