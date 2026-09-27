# coding=utf-8
import sys
from os import getenv


def is_debugging_run() -> bool:
    """Utility function to check if code is likely in a debugging session.

    Note:
    - PyCharm IDE use pydevd for debugging
    - sys.gettrace() is usualy active by debugger

    :return: True if in a debugging session; otherwise, False.
    """
    return "pydevd" in sys.modules or sys.gettrace() is not None


def is_pytest_run() -> bool:
    """Utility function to check if code is executed in a pytest run.

    :return: True if run in pytest
    """
    pytest_is_running = (
        getenv("PYTEST_RUN_CONFIG") is not None
        or getenv("DNA_PYTEST_CI_RUN") is not None
    )
    return pytest_is_running


def has_pytest_marked_automated_test() -> bool:
    """
    Determines whether there is a pytest marker for automated tests in the current module.

    This function checks if a global pytest marker named "automated_test" is
    defined. If such a marker is present, it returns True; otherwise, it returns
    False.

    :return: A boolean indicating the presence of a "automated_test" pytest marker.
    """
    automated_test = False
    try:
        globals_pytestmark_ = globals()["pytestmark"]
        if globals_pytestmark_.markname == "automated_test":
            automated_test = True
    except KeyError:
        pass
    return automated_test
