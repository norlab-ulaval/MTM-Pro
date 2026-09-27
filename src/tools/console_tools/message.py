# coding=utf-8

from typing import Optional, Union
import inspect
from shutil import get_terminal_size
from tools.console_tools.format import ConsoleFormat
from tools.console_tools.message_prefix import MSG_PREFIX,MSG_OPEN_CHAR,MSG_CLOSE_CHAR, MSG_ID, MSG_SEP


def consol_msg(
    who_am_i: str, msg: str, space_before=True, space_after=True, print_it=True
) -> Union[None, str]:
    """Format a console message

    :param who_am_i: The name of the caller
    :param msg: The message to print to console
    :param space_before:
    :param space_after:
    :param print_it: wheter to print to consol or return the formated string
    :return: The formated string if kwarg 'print_it=False', nothing otherwise
    """
    the_str = f"{MSG_OPEN_CHAR}{who_am_i}{MSG_CLOSE_CHAR} {msg}"

    if space_before:
        the_str = "\n" + the_str

    if space_after:
        the_str += "\n"

    if print_it:
        # Note: ensure that sys.stdout buffer is flushed at every call to prevent logging problem
        # when executing huge execution loop (ref task RLRP-166)
        print(the_str, flush=True)
        return None
    else:
        return the_str


def consol_msg_universal(
    msg: str, space_before=True, space_after=True, print_it=True, caller_name: Optional[str] = None
) -> Union[None, str]:
    """Consol message function that automaticaly fetch the caller name.

    :param msg: The message to print to console
    :param space_before:
    :param space_after:
    :param print_it: wheter to print to consol or return the formated string
    :param caller_name:
    :return: The formated string if kwarg 'print_it=False', nothing otherwise
    """

    if not caller_name:
        # Fetch name of caller function
        caller_name = inspect.currentframe().f_back.f_code.co_name
        if caller_name == "__init__":
            # Fetch name of caller class otherwise
            caller_name = inspect.currentframe().f_back.f_locals["self"].__class__.__name__

    return consol_msg(
        f"{MSG_PREFIX}{caller_name}",
        msg,
        space_before=space_before,
        space_after=space_after,
        print_it=print_it,
    )


def consol_msg_universal_one_liner(msg: str, print_it=True, caller_name: Optional[str] = None) -> Union[None, str]:
    """Consol message function (one line version) that automaticaly fetch the caller name.

    :param msg: The message to print to console
    :param print_it: wheter to print to consol or return the formated string
    :param caller_name:
    :return: The formated string if kwarg 'print_it=False', nothing otherwise
    """

    if not caller_name:
        # Fetch name of caller function
        caller_name = inspect.currentframe().f_back.f_code.co_name
        if caller_name == "__init__":
            # Fetch name of caller class otherwise
            caller_name = inspect.currentframe().f_back.f_locals["self"].__class__.__name__

    return consol_msg(
        f"{MSG_PREFIX}{caller_name}",
        msg,
        space_before=False,
        space_after=False,
        print_it=print_it,
    )


def consol_msg_DEV_universal(
    msg: str = "<--!!", space_before=True, space_after=True, print_it=True
) -> Union[None, str]:
    """Consol message function (development version) that automaticaly fetch the caller name.

    :param msg: The message to print to console
    :param space_before:
    :param space_after:
    :param print_it: wheter to print to consol or return the formated string
    :return: The formated string if kwarg 'print_it=False', nothing otherwise
    """

    # Fetch name of caller function
    _caller_name = inspect.currentframe().f_back.f_code.co_name
    if _caller_name == "__init__":
        # Fetch name of caller class otherwise
        _caller_name = inspect.currentframe().f_back.f_locals["self"].__class__.__name__

    caller_str = (
        f"{ConsoleFormat.MSG_EMPH_FORMAT}{ConsoleFormat.MSG_ERROR_FORMAT}{MSG_PREFIX} "
        f"{_caller_name}"
    )
    return consol_msg(
        caller_str,
        f"{msg}{ConsoleFormat.MSG_END_FORMAT}",
        space_before=space_before,
        space_after=space_after,
        print_it=print_it,
    )


def consol_msg_motion_model_learner(msg: str, *args, **kwargs) -> Union[None, str]:
    return consol_msg(f"{MSG_PREFIX} motion model learner", msg, **kwargs)


def consol_msg_motion_model_learner_one_line(msg: str, *args, **kwargs) -> Union[None, str]:
    return consol_msg(
        f"{MSG_PREFIX} motion model learner", msg, space_before=False, space_after=False, **kwargs
    )


def consol_msg_itmpc_single_agent(msg: str, *args, **kwargs) -> Union[None, str]:
    return consol_msg(f"{MSG_PREFIX} IT-MPC", msg, **kwargs)


def consol_msg_itmpc_single_agent_one_line(msg: str, *args, **kwargs) -> Union[None, str]:
    return consol_msg(f"{MSG_PREFIX} IT-MPC", msg, space_before=False, space_after=False, **kwargs)


def consol_msg_f1tenth_vaul_main(msg: str, *args, **kwargs) -> Union[None, str]:
    return consol_msg(f"{MSG_PREFIX} f110_s2s_main", msg, **kwargs)


def consol_msg_main(msg: str, *args, **kwargs) -> Union[None, str]:
    return consol_msg(f"{MSG_PREFIX} main", msg, **kwargs)


def consol_msg_f1tenth_vaul_main_one_liner(msg: str, *args, **kwargs) -> Union[None, str]:
    return consol_msg(
        f"{MSG_PREFIX} f110_s2s_main", msg, space_before=False, space_after=False, **kwargs
    )


def consol_msg_f1Tenth_gym_example(msg: str, *args, **kwargs) -> Union[None, str]:
    return consol_msg("F1TENTH-gym example", msg, **kwargs)


def consol_msg_follow_raceline(msg: str, *args, **kwargs) -> Union[None, str]:
    return consol_msg("FOLLOW_RACELINE", msg, **kwargs)


def consol_msg_draw_terminal_wide_line(
    char="=", space_before=False, space_after=False, print_it=True
) -> Union[None, str]:
    """Draw a line terminal wide

    :param char: the character used to draw the line
    :param space_before:
    :param space_after:
    :param print_it: wheter to print to consol or return the formated string
    :return: The formated string if kwarg 'print_it=False', nothing otherwise
    """
    the_str = char * get_terminal_size()[0]

    if space_before:
        the_str = "\n" + the_str

    if space_after:
        the_str = the_str + "\n"

    if print_it:
        print(the_str, flush=True)
        return None
    else:
        return the_str


def console_msg_pipeline_footer(msg: str, char: str = "=") -> None:
    """Prints a formatted message and draws a terminal-wide line beneath it.

    :param msg: The message to be displayed.
    :param char: The character to be used for drawing the terminal-wide line.
    :return: None
    """
    consol_msg_main(
        f"{ConsoleFormat.MSG_DONE_FORMAT}{msg}{ConsoleFormat.MSG_END_FORMAT}",
        space_before=True,
        space_after=False,
    )
    consol_msg_draw_terminal_wide_line(char=char, space_before=False, space_after=True)
    return None


def console_msg_pipeline_header(msg: str, char: str = "=") -> None:
    """Draws a terminal-wide line and header message under it.

    :param msg: The message to display as the header.
    :param char: The character to use for the terminal-wide line.
    :return: None
    """
    consol_msg_draw_terminal_wide_line(char=char, space_before=True, space_after=False)
    consol_msg_main(
        f"{ConsoleFormat.MSG_EMPH_FORMAT}{msg}{ConsoleFormat.MSG_END_FORMAT}",
        space_before=False,
        space_after=True,
    )
    return None
