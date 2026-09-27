# coding=utf-8
"""Failure isolation for the experiment data consolidation (RLRP-818).

The consolidation is an AUXILIARY artifact: an export failure must never
break the figure generation it piggybacks on. Every export call site is
therefore wrapped in :func:`consolidation_guard`, which downgrades any
exception to a console warning.
"""
import traceback
from contextlib import contextmanager

from tools.console_tools.message import consol_msg_universal_one_liner


@contextmanager
def consolidation_guard(context: str):
    """Swallow any exception raised by a consolidation export.

    :param context: Human-readable identification of the guarded export (used
        in the warning message), e.g. ``"drift_rate"``.
    """
    try:
        yield
    except Exception as e:  # noqa: BLE001 -- auxiliary artifact, never fatal
        consol_msg_universal_one_liner(
            f"experiment_data_consolidation ({context}): the data consolidation "
            f"export FAILED and was skipped -- the plot generation is unaffected. "
            f"Cause: {type(e).__name__}: {e}\n"
            f"{traceback.format_exc()}"
        )
