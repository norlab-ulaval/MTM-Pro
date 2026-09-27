# coding=utf-8
from tools.console_tools.format import ConsoleFormat
from tools.console_tools.message import consol_msg_universal_one_liner


def nan_infinity_console_warning(variable_name: str) -> None:
    print()
    consol_msg_universal_one_liner(
        f"{ConsoleFormat.MSG_WARNING_FORMAT}"
        f"Be advised, NaN or infinity values encountered in '{variable_name}'"
        f"{ConsoleFormat.MSG_END_FORMAT}"
    )
    return None
