# coding=utf-8
from os import getenv

from tools.console_tools.message import consol_msg_universal_one_liner


def is_run_on_a_teamcity_continuous_integration_server() -> bool:
    """
    Check if python is executed in the continuous integration server.

    Note: The check is specific to TeamCity server
    """
    try:
        import teamcity as tc

        # Note: if it return 'LOCAL' then it is not running on a TeamCity server
        tc_version = getenv("TEAMCITY_VERSION")

        if tc_version != "LOCAL":
            consol_msg_universal_one_liner(
                f"is running under teamcity TEAMCITY_VERSION={tc_version}"
            )
            return True
        else:
            consol_msg_universal_one_liner(
                f"TEAMCITY_VERSION={tc_version} ››› run not executed on CI server"
            )
            return False
    except ImportError:
        consol_msg_universal_one_liner("python is not executed on CI server")
        return False


def show_plot_unless_CI_server_runned(show_plot: bool, silent: bool = True) -> bool:
    """
    Required to switch off matplotlib `plt.show()` on TeamCity continuous intergation server.

    Note: On `plt.show()`, python open up a window waiting to be close by the user which will
    stale the TeamCity server build queue.

    :param show_plot: pass True or False as you would normaly set a `show_plot` argument.
    :param silent: Set to False to console print msg.
    :return: 'show_plot' argument if python is executed locally, False otherwise.
    """
    try:
        import teamcity as tc

        # Note: if `getenv(...)` it return 'LOCAL' then it is not running on a TeamCity server
        tc_version = getenv("TEAMCITY_VERSION")

        if tc_version != "LOCAL":
            if not silent:
                consol_msg_universal_one_liner(
                    f"is running under teamcity TEAMCITY_VERSION={tc_version}"
                    " ››› switching `show_plot` to False\n"
                )
            return False
        else:
            if not silent:
                consol_msg_universal_one_liner(
                    f"TEAMCITY_VERSION={tc_version} ››› run not executed on CI server"
                    f" ››› use user argument {show_plot=}\n"
                )
            return show_plot
    except ImportError:
        # Run not executed on CI server
        if not silent:
            consol_msg_universal_one_liner(f"{show_plot=}\n")
        return show_plot


