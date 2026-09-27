# coding=utf-8
from dataclasses import dataclass


@dataclass()
class ConsoleFormat:
    """String formating definition

    Usage example:

    >>> from tools.console_tools.format import ConsoleFormat
    >>> print(f"[{ConsoleFormat.MSG_DONE_FORMAT}Done{ ConsoleFormat.MSG_END_FORMAT}]")

    """

    MSG_EMPH_FORMAT: str = "\033[1;97m"
    MSG_DIMMED_FORMAT: str = "\033[1;2m"
    MSG_BASE_FORMAT: str = "\033[1m"
    MSG_ERROR_FORMAT: str = "\033[1;31m"
    MSG_DONE_FORMAT: str = "\033[1;32m"
    MSG_WARNING_FORMAT: str = "\033[1;33m"
    MSG_END_FORMAT: str = "\033[0m"
