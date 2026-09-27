# coding=utf-8
import os
import sys
import warnings
from contextlib import contextmanager

import matplotlib
import omegaconf
from matplotlib import pyplot as plt

from tools.dna_dev_tools.dn_CI_tools import show_plot_unless_CI_server_runned
from tools.dna_dev_tools.dn_pytest_tools import is_pytest_run
from tools.hydra_apps_tools.hydra_utils import get_hydra_experiment_cwd


@contextmanager
def plot_manager(show_plot: bool, headless: bool):
    """
    Context manager for setting up and tearing down a matplotlib plotting environment.

    :param show_plot: Boolean flag to determine if the plot should be displayed.
    :param headless: Boolean flag to determine if the plotting should run without a display
     (useful for server environments).
    :return: None
    """
    with warnings.catch_warnings():
        manage_matplotlib_warnings()
        manage_matplotlib_backend(show_plot, headless)

        yield

        if show_plot:
            _safe_show()

        plt.close()
        return None


def manage_matplotlib_backend(show_plot: bool, headless: bool) -> None:
    """
    Standard matplotlib backend:
      - interactive backends: GTK3Agg, GTK3Cairo, GTK4Agg, GTK4Cairo, MacOSX,
        nbAgg, QtAgg, QtCairo, TkAgg, TkCairo, WebAgg, WX, WXAgg, WXCairo,
        Qt5Agg, Qt5Cairo
      - non-interactive backends: agg, cairo, pdf, pgf, ps, svg
    """
    if headless:
        if not show_plot:
            matplotlib.use("agg")
    else:
        if show_plot:
            _try_set_native_interactive_backend()
    return None


def _is_pycharm_hosted() -> bool:
    """Return True when running inside PyCharm (terminal, run config, or
    remote interpreter).  PyCharm sets ``PYCHARM_HOSTED=1`` in all those
    contexts."""
    return bool(os.environ.get("PYCHARM_HOSTED"))


def _try_set_native_interactive_backend() -> None:
    """Select an appropriate matplotlib backend for interactive display.

    PyCharm injects its own ``backend_interagg`` backend which works
    everywhere (Mac, Linux, Docker).  Outside PyCharm, Qt platform
    plugins may produce 'This plugin does not support
    propagateSizeHints()/raise()' errors that block the pipeline, so we
    prefer the native macOS backend on darwin and fall back to 'agg'
    on Linux when no display server is available."""
    import sys

    if _is_pycharm_hosted():
        # PyCharm handles the backend; nothing to do.
        return None

    if sys.platform == "darwin":
        try:
            matplotlib.use("MacOSX")
        except Exception:
            pass
    elif sys.platform == "linux":
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            matplotlib.use("agg")
    return None


def _safe_show() -> None:
    """Call ``plt.show()`` only when the display environment is known to
    work reliably.

    Inside PyCharm (``PYCHARM_HOSTED=1``) the IDE's ``backend_interagg``
    handles display correctly on every platform.  Outside PyCharm, Qt
    platform plugins in Docker-on-Mac and some terminals produce
    blocking errors ('This plugin does not support
    propagateSizeHints()/raise()') or hangs, so we skip ``plt.show()``
    entirely — plots are already saved to disk by the caller."""
    if not _is_pycharm_hosted():
        # Outside PyCharm: do not attempt interactive display to avoid
        # Qt plugin errors/hangs. Plots are persisted to disk already.
        return None

    with warnings.catch_warnings():
        manage_matplotlib_warnings()
        try:
            plt.show()
        except Exception:
            pass


def manage_matplotlib_warnings() -> None:
    """Note: use inside a context manager.
    Example:
        >>> with warnings.catch_warnings():
        >>>     manage_matplotlib_warnings()
        >>>     ...
    """
    warnings.filterwarnings(
        "ignore",
        message=(
            "Matplotlib is curently using 'agg', a non-GUI backend, so cannot show the figure."
        ),
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
        category=UserWarning,
    )
    # It's a fix on a recent released version with a special fix for PyCharm IDE
    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
    return None


def cfg_based_plot_save(cfg: omegaconf.DictConfig, plot_name: str) -> None:
    exp_data_dir_realpath = get_hydra_experiment_cwd(cfg)
    exp_plot_dir_path = os.path.realpath(os.path.join(exp_data_dir_realpath, "plot"))
    if not os.path.exists(exp_plot_dir_path):
        os.makedirs(exp_plot_dir_path)
    exp_plot_path = os.path.join(exp_plot_dir_path, plot_name)
    plt.savefig(exp_plot_path)
    return None


def save_plot_special(
    fig: plt.Figure, exp_dir_path: str, file_name: str, dpi: int = 50
) -> None:
    """
    Save plot to disk without croping components that have an offset eg legend outside the plot
    despite having tight layout set.
    Note: close the figure after saving

    :param fig: The figure object to be saved.
    :param exp_dir_path: The directory path where the figure will be saved.
    :param file_name: The name of the file to save the figure as.
    :param dpi: The saved resolution in dot per inch (matplotlib default is dpi=100)
    :return: None
    """
    os.makedirs(exp_dir_path, exist_ok=True)
    fig.savefig(os.path.join(exp_dir_path, file_name), bbox_inches="tight", dpi=dpi)
    return None


def show_and_save_plot_helper(
    fig: plt.Figure,
    exp_dir_path: str,
    file_name: str,
    headless: bool,
    show: bool,
    saved_dpi: int | None,
    save: bool = True,
) -> None:
    """
    Displays and optionally saves a matplotlib figure under specified conditions.

    This function provides utility to manage the display and saving of a
    matplotlib figure (`fig`). It supports conditional behaviors based
    on the provided arguments, such as handling headless environments,
    display toggling, and the ability to save with a specified resolution.

    :param fig: The matplotlib figure instance to be displayed or saved.
    :param exp_dir_path: The directory path where the plot file will be saved.
    :param file_name: The name of the file in which the plot will be saved.
    :param headless: A flag indicating if the environment is headless or not.
    :param show: A flag indicating whether to display the plot.
    :param saved_dpi: The dots-per-inch (DPI) resolution for saving the plot (set to None to diable saving).
    :param save: Indicates whether to save the plot. Defaults to ``True``.
    :return: This function does not return any value.
    """
    if save and saved_dpi is not None:
        save_plot_special(fig, exp_dir_path, file_name, saved_dpi)
    if show_plot_unless_CI_server_runned(show and not headless) and not is_pytest_run():
        _safe_show()
    else:
        plt.close(fig)
    return None
