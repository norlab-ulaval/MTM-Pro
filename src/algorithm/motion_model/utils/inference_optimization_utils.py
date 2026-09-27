# coding=utf-8
import sys
from typing import Any

import torch
import time

from tools.console_tools.message import consol_msg_universal_one_liner
from tools.dna_dev_tools.dn_container_tools import (
    is_run_in_DN_arm64_jetson_architecture,
)
from tools.dna_dev_tools.dn_pytest_tools import is_debugging_run, is_pytest_run

# .... Monkey-patch BackendCompilerFailed for pickling support ....................................
# BackendCompilerFailed.__init__ requires positional args (backend_fn, inner_exception) that
# Python's default exception unpickling cannot provide.  When torch.compile raises this exception
# inside a joblib/loky worker (e.g. Hydra-Optuna multirun), the result fails to serialize back
# to the main process with:
#   TypeError: BackendCompilerFailed.__init__() missing 1 required positional argument: 'inner_exception'
# Adding __reduce__ lets pickle reconstruct the object with the correct arguments.
try:
    from torch._dynamo.exc import BackendCompilerFailed as _BCF

    if "__reduce__" not in _BCF.__dict__:

        def _backend_compiler_failed_reduce(self):
            return (
                _unpickle_backend_compiler_failed,
                (self.backend_name, self.inner_exception, str(self)),
            )

        _BCF.__reduce__ = _backend_compiler_failed_reduce

        class _BackendFnStub:
            """Lightweight stand-in whose ``__name__`` is the original backend name."""

            def __init__(self, name: str):
                self.__name__ = name

        def _unpickle_backend_compiler_failed(backend_name, inner_exception, msg):
            try:
                return _BCF(_BackendFnStub(backend_name), inner_exception)
            except Exception:
                return RuntimeError(msg)

except ImportError:
    pass


def optimize_model_speed(model: torch.nn.Module, disable: bool = True) -> Any:
    # inprogress: validate performance gain
    # (PRIORITY) todo: implement tensor-RT version

    disable_model_compiling = disable or is_pytest_run() or is_debugging_run()
    if not disable_model_compiling:

        # This is a workaround for `meta["horizon_loss"] = ms_nll_losses.detach().mean().item()`.
        # `item()` is a data-dependant operation that cause dynamo to capture scalar outputs in
        # the graph. Ref: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html#data-dependent-operations
        torch._dynamo.config.capture_scalar_outputs = True

        compile_start_time = time.time()
        if is_run_in_DN_arm64_jetson_architecture():
            # Skip torch.compile on Jetson ARM64 during training:
            # - The default inductor backend requires Triton (unavailable on ARM64).
            # - The aot_eager fallback provides negligible performance gains
            #   (no kernel fusion, no memory planning) while adding compile
            #   overhead and triggering dynamo recompilation cache exhaustion
            #   (cache_size_limit warnings) on functions with dynamic Python
            #   control flow (e.g. normalize, compounded_prediction).
            # - Future: re-enable with TensorRT backend for inference-only stage.
            # import torch_tensorrt  # future: switch to TensorRT backend
            consol_msg_universal_one_liner(
                "Torch model compile optimization skipped on Jetson ARM64 (aot_eager gains are negligible)."
            )
            return model
        else:
            optimized_model = torch.compile(
                model,
                # mode="default",
                mode='max-autotune-no-cudagraphs',
            )
        # Warmup: trigger inductor's C++ compilation (and its one-time CUDA
        # runtime detection message) now, before training progress bars start.
        _trigger_inductor_cuda_detection()

        compile_time = time.time() - compile_start_time
        consol_msg_universal_one_liner(f"Torch model compile optimitization DONE in {compile_time:>2.3f}.")
        return optimized_model
    else:
        consol_msg_universal_one_liner(f"Torch model compile optimitization disabled.")
        return model


def _trigger_inductor_cuda_detection() -> None:
    """Force a trivial inductor compilation so that the one-time diagnostic
    message ``No CUDA runtime is found, using CUDA_HOME=...`` is emitted
    *before* any training progress bar is displayed."""
    try:
        _warmup = torch.compile(lambda x: x + 1)
        _warmup(torch.tensor(1.0))
    except Exception:
        pass
    sys.stderr.flush()
    sys.stdout.flush()
