# coding=utf-8
import os
import shutil
import subprocess
from typing import List, Optional, Tuple

from tools.dna_dev_tools.dn_pytest_tools import is_pytest_run


def get_git_info() -> Tuple[str, str, str]:
    """Get git repository information.

    Should return something similar to the following:

        >>> print(get_git_info())
        "MTM-Pro",
        "https://github.com/anonymous/MTM-Pro",
        "/path/to/MTM-Pro/src/tools/hydra_apps_tools/"

    :return: Tuple containing (git_remote_url, git_name, project_path)
    """
    # .... Get the git remote URL .................................................................
    git_remote_url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            text=True
            ).strip()

    # .... Get the repository name ................................................................
    git_name = os.path.basename(git_remote_url)
    if git_name.endswith('.git'):
        git_name = git_name[:-4]

    # .... Get the project path ...................................................................
    project_path = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True
            ).strip()

    return git_remote_url, git_name, project_path


def echo_centering_str(the_str, the_style="\033[1;37m", the_pad_cha="\033[0m·", fill_left="",
                       fill_right=""):
    """Dynamic printf centering tool. Centering based on the terminal screen width at runtime.

    Python implementation of n2st::echo_centering_str from AnonLab-shell-script-tools.

    :param the_str: The string to center
    :param the_style: The style appended at the beginning of the line
    :param the_pad_cha: The padding character to use
    :param fill_left: Optional fill string for left side
    :param fill_right: Optional fill string for right side
    :return: The centered string
    """
    the_style_off = "\033[0m"

    # .... Get terminal width .....................................................................
    terminal_width = shutil.get_terminal_size().columns

    # .... Calculate padding ......................................................................
    str_len = len(the_str)
    total_padding_len = terminal_width - str_len
    single_side_padding_len = total_padding_len // 2

    # .... Create padding .........................................................................
    pad = the_pad_cha * single_side_padding_len

    # .... Return centered string .................................................................
    return f"{the_style}{fill_left}{pad}{the_str}{pad}{fill_right}{the_style_off}"


def AnonLab_splash(title='AnonLab', main_url: str = "https://anonymous.example.org",
                  optional_repo_url: Optional[str] = 'https://github.com/anonymous',
                  splash_type: str = 'negative'):
    """AnonLab Terminal splash screen

    Python implementation of n2st::AnonLab_splash from AnonLab-shell-script-tools.

    :param title: The title printed in the center of the splash screen
    :param main_url:
    :param optional_repo_url: The url printed at the bottom of the splash screen
    :param splash_type: The style of presentation: small, big or negative (default)
    :return: List of formatted lines for the splash screen
    """

    # .... Formatting .............................................................................
    title_formatting = "\033[1m"
    snow_formatting = "\033[2m"
    url_formatting = "\033[2m"

    # .... Add ASCII art based on splash type .....................................................
    if splash_type == 'small':
        ascii_art_up = [
                "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣤⣄⠀⠀⠀⣼⣿⣿⣿⣿⠀⠀⠀⢀⣤⠀⠀⠀⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⡿⢛⣩⣭⣶⣶⠒⣶⣦⣭⣉⠛⢿⣿⣿⣿⠂⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⣠⠟⣭⣾⣿⣿⣿⡿⠙⢿⠀⠿⠉⣿⣿⣿⣿⣶⣍⠻⣄⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⣴⣿⣿⠟⣶⣿⣿⣿⣿⣿⣿⠉⠻⣶⠀⣶⠛⠉⠟⣛⣛⣛⣛⠻⢦⠻⣿⣿⣦⠀⠀",
                "⠀⠀⠘⣿⣿⢡⡋⠙⠿⠀⣿⡀⢸⡟⠙⢷⣤⠀⣤⠞⠾⢿⡇⢰⡟⠀⠟⠉⢻⡟⣾⣿⠗⠀",
                "⠀⠀⠀⢰⢁⣿⣿⠛⠀⣀⠈⠀⢸⡇⢸⣶⣄⠀⣤⠘⡀⢸⠃⠈⠀⣀⠀⠛⣿⣼⠸⡄⠀⠀",
                "⠀⣶⣾⣿⣾⣿⣿⣿⠉⢀⣤⡶⠒⠀⠈⠛⢿⠀⠛⠚⠀⠀⠒⣶⣤⡀⢙⡿⣴⣿⣧⣿⣦⣤",
                ]

        ascii_art_down = [
                "⠀⣿⣿⣏⣿⣿⣿⣿⡿⣿⣄⠀⠛⣿⡿⠛⠀⠀⠀⠒⣶⡮⠛⠢⠿⣛⣭⣿⣿⣿⣿⣿⣿⣿",
                "⠀⠈⠉⣿⢹⣿⣿⠛⢶⣤⡀⠉⠉⠀⢰⣿⣿⠀⣿⣷⡀⠀⠉⠉⣀⣤⡾⠛⣿⣿⡏⡿⠛⠉",
                "⠀⠀⠀⣨⣆⣿⣿⠟⠀⠀⣠⠀⢸⡇⠘⠉⣀⠀⡀⠉⠀⢸⡆⢰⡄⠀⠐⠿⣿⣿⣼⠀⠀⠀",
                "⠀⠀⠺⣿⣿⣆⢷⣾⣿⣀⣿⣀⣾⣷⡾⠋⢀⠀⠈⠛⣶⣿⣇⣸⣿⣀⣿⣶⡿⣴⣿⣿⠆⠀",
                "⠀⠀⠀⠙⡿⠿⢷⡙⣿⣿⣿⣿⣿⣿⣶⠟⠉⠀⠉⠿⣶⣿⣿⣿⣿⣿⣿⢫⣿⣿⣿⠃⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⠈⣿⣬⡛⣿⣿⣿⣿⣶⣿⠀⣿⣶⣿⣿⣿⡿⢛⣵⣿⠀⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⠘⠿⣿⣿⣿⠓⠶⣭⣭⣉⣉⣉⣭⣭⠶⢾⣿⣿⣿⠿⠀⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⣿⣿⣿⣿⠃⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀"
                ]
    elif splash_type == 'big':
        ascii_art_up = [
                "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⣿⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣴⣾⣷⣤⡀⠀⠀⣀⣰⣿⣿⣿⣿⣿⣿⣿⣀⠀⠀⠀⢀⣴⣷⣦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣿⣿⣿⣿⣿⣿⠿⠛⣋⣉⣩⣤⣤⠤⠤⣤⣬⣍⣉⡙⠛⠶⣿⣿⣿⣿⣿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⠿⢛⣡⣴⣾⣿⣿⣿⣿⣿⣿⠀⠀⣿⣿⣿⣿⣿⣿⣶⣦⣌⡛⠿⣿⣿⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠛⣡⣶⣿⣿⣿⣿⣿⣿⣟⠁⠉⠻⠀⠀⠟⠉⠈⣻⣿⣿⣿⣿⣿⣿⣶⣌⠛⢧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⠀⣴⣿⣷⣶⣶⠋⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⠻⢷⣦⡀⠀⠀⢀⣴⡾⠟⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠙⣦⣤⣴⣶⣦⠀⠀⠀⠀",
                "⠀⠀⢀⣼⣿⣿⣿⡟⢡⣾⣿⣿⠿⠿⣿⣿⣿⣿⣿⣧⣀⠀⠉⠻⠀⠀⠟⠉⠀⡀⢈⣥⡴⣶⣶⣶⠶⠤⣭⣍⣓⡈⠻⠿⠿⢿⣧⡀⠀⠀",
                "⠀⠀⠺⣿⣿⣿⠟⣰⠏⠙⠻⣿⡀⠀⣿⡇⠀⢸⣿⠛⠻⢷⣦⣀⠀⠀⣀⣴⠎⣴⣿⣿⡇⠀⢸⣿⠀⢀⡿⠟⠋⠹⣿⡟⣰⣿⣿⡷⠀⠀",
                "⠀⠀⠀⠈⢻⡟⢰⣿⣶⣤⣀⠀⠁⠀⢹⡇⠀⢸⣿⠀⢀⡀⠈⠙⠀⠀⠋⠁⢸⠀⠀⣿⡇⠀⢸⡇⠀⠈⠀⣀⣤⣶⡿⢀⢻⣿⠟⠀⠀⠀",
                "⠀⠀⠀⠀⡿⢀⣿⣿⣟⠛⠉⠀⣀⣀⡀⠀⠀⠸⣿⠀⢸⣿⣶⣤⠀⠀⣤⣶⢸⡇⠀⣿⠇⠀⠀⢀⣀⣀⠀⠉⠛⣿⢃⣿⡄⢿⠀⠀⠀⠀",
                "⢀⣀⣤⣼⡇⣸⣿⣿⣿⣦⣶⠿⠛⠉⠀⢀⣠⡀⠀⠀⠘⢿⣿⣿⠀⠀⣿⠏⡸⠃⠀⠀⢀⣄⡀⠀⠉⠛⠿⣶⣴⠇⣼⣿⣇⢸⣇⡀⠀⠀",
                "⣿⣿⣿⣿⠁⣿⣿⣿⣿⣿⣿⣄⣠⣴⡾⠟⠉⠀⢀⣤⣄⠀⠈⠙⠀⠀⠁⠀⠀⣠⣤⡀⠀⠉⠻⢷⣦⣄⣠⡿⢃⣾⣿⣿⣿⠀⣿⣿⣿⣶",
                ]

        ascii_art_down = [
                "⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣏⡀⠀⠐⠾⣿⣿⣿⡿⠒⠀⠀⠀⠀⠠⢴⣶⣦⣍⠳⠦⣄⣼⡿⠟⣋⣴⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿",
                "⠿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⠁⠉⠛⠿⣶⣤⡀⠀⠉⠁⠀⣠⣴⠀⠀⣦⣀⠀⠈⠉⠀⢀⣤⣶⠶⠒⠉⠈⣿⣿⣿⣿⣿⣿⢀⣿⣿⣿⣿",
                "⠀⠀⠈⢹⡇⢹⣿⣿⣿⠛⠛⠿⣶⣤⡀⠀⠉⠀⢀⠀⢰⣿⣿⣿⠀⠀⣿⣿⣿⠀⠀⡀⠀⠉⠀⢀⣤⣶⠿⠛⠻⣿⣿⣿⡏⢸⡟⠛⠉⠁",
                "⠀⠀⠀⠀⣷⠘⣿⣿⣷⣦⣄⠀⠀⠁⠀⡀⠀⢸⣿⠀⢸⡿⠟⠋⠀⠀⠙⠻⢿⠀⠀⣿⡇⠀⢀⠀⠈⠀⠀⣠⣴⣿⣿⣿⠁⣾⠀⠀⠀⠀",
                "⠀⠀⠀⣴⣿⣧⠹⣿⠟⠉⠀⢀⠀⠀⣼⡇⠀⢸⣿⠀⠀⠀⣠⣴⠀⠀⣦⣀⠀⠀⠀⣿⡇⠀⢸⡇⠀⢀⡀⠀⠙⠻⣿⠇⣼⣧⡀⠀⠀⠀",
                "⠀⠀⢾⣿⣿⣿⣦⠹⣧⣴⣾⣿⠀⠀⣿⡇⠀⢸⣿⣦⣶⠿⠋⠁⠀⠀⠈⠛⠿⣶⣴⣿⡇⠀⢸⣿⠀⠈⣿⣷⣤⣼⠏⣴⣿⣿⣿⡦⠀⠀",
                "⠀⠀⠈⢻⣿⣿⣿⣧⡘⢿⣿⣿⣿⣶⣿⣿⣿⣿⣿⡏⠁⢀⣠⣶⠀⠀⣶⣄⡀⠈⢻⣿⣿⣿⣿⣿⣶⣿⣿⣿⡿⢃⣼⣿⣿⣿⡟⠁⠀⠀",
                "⠀⠀⠀⠀⠻⠿⠟⠛⠻⣄⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣶⠿⠋⠀⠀⠀⠀⠙⠿⣶⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⣠⠿⠿⢿⣿⠟⠀⠀⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢳⣤⡙⠿⣿⣿⣿⣿⣿⣿⣷⣄⣠⣾⠀⠀⣷⣄⣠⣾⣿⣿⣿⣿⣿⣿⠿⢋⣤⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⣿⣿⣶⣬⡙⠻⠿⣿⣿⣿⣿⣿⣿⠀⠀⣿⣿⣿⣿⣿⣿⡿⠟⢋⣥⣶⣿⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣿⣿⣿⣿⣿⣿⠶⣤⣌⣉⣙⡛⠛⠛⠛⠛⠛⣋⣉⣩⣤⣶⣿⣿⣿⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⢿⠟⠁⠀⠀⠀⠉⣿⣿⣿⣿⣿⣿⣿⠏⠉⠀⠀⠈⠛⢿⡿⠟⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣿⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
                "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀"
                ]
    elif splash_type == 'negative':
        ascii_art_up = [
                "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⢿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠉⠀⠀⠀⠀⠈⢉⣡⣤⠤⠶⠶⠶⠶⢤⣤⣈⡉⠋⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⣠⠶⠋⠉⠀⠀⠀⠀⠀⣿⣿⠀⠀⠀⠀⠀⠉⠛⢶⣄⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣿⣿⣿⣿⠛⠿⢿⡟⢁⡶⠉⠀⠀⠀⠀⠀⠀⠘⣿⣷⣿⣿⣾⡿⠃⠀⠀⠀⠀⠀⠀⠉⢶⡈⢿⣿⣿⡿⣿⣿⣿⣿⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣿⣿⣿⠁⠀⠀⢀⡾⠁⠀⠀⠀⠀⠀⠀⠀⠶⣿⣦⡀⣿⣿⣀⣴⣿⣦⠴⠒⠒⠒⠒⠦⢤⣌⣶⠀⠀⠀⠈⣿⣿⣿⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣿⣿⣄⠀⠀⢠⠋⣶⣤⡀⢻⣿⠀⣿⡇⠀⣤⣄⠉⠻⣿⣿⠛⢁⠋⠀⠀⣶⣶⠀⣿⡆⢀⣤⠀⠀⣰⠁⠀⠀⣿⣿⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣿⣿⣿⡿⢠⠃⠀⠀⠉⣻⣿⣿⣤⣿⣿⠀⣿⠛⠿⣿⣿⣿⣿⣿⢸⣿⠀⣿⣟⣤⣿⣿⣟⠉⠀⢠⠻⡀⣴⣿⣿⣿⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣿⣿⡿⠀⡟⠀⠀⠈⠟⠋⠁⣤⣿⣿⣿⣶⣿⠂⠀⠀⣿⣿⠀⣿⢸⣿⣶⣿⣿⣿⣄⠈⠙⠁⢀⠋⠀⣿⠈⣿⣿⣿⣿⣿⣿⣿",
                "⣿⣿⣿⣿⠀⠀⠀⢰⠁⠀⠀⠀⠀⠹⠟⠉⣀⣴⣾⡿⠛⠿⣿⣦⣿⣿⣿⣿⠟⠛⢿⣷⣤⡀⠉⠋⠀⣠⠋⠀⠀⢸⠀⠀⠀⠉⣿⣿⣿⣿",
                ]
        ascii_art_down = [
                "⣿⣿⣿⣿⠀⠀⠀⢸⠀⠀⠀⠀⠀⢀⡀⠈⠻⣿⣶⣄⠀⣀⣶⣿⣿⣿⣿⣦⣀⠀⣹⣦⣭⣥⣤⢖⠋⠀⠀⠀⠀⢸⠀⠀⠀⢸⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣷⣶⣤⠀⡇⠀⠀⠀⡀⠉⠛⣿⣷⣦⣴⣿⣿⠛⠁⠀⣿⣿⠀⠉⢻⣿⣿⣤⣴⣾⡿⠛⠉⣀⠀⠀⠀⢸⠀⢀⣠⣼⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣿⣿⣿⡆⢻⠀⠀⠈⠛⢿⣿⣾⣿⣿⣿⠀⣿⠀⣀⣴⣿⣿⣦⡀⢸⣿⠀⣿⡿⣿⣶⣿⠿⠛⠀⠀⠀⡟⣸⣿⣿⣿⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣿⣿⠟⠀⠀⢷⠀⣶⣿⠟⣻⣿⠀⣿⡏⠀⣿⡿⠛⢁⣿⣿⠉⠛⣿⣿⠀⣿⣿⠀⣿⡟⠿⣿⣶⠀⡿⠀⠙⣿⣿⣿⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣿⣿⣄⠀⠀⠀⠻⡀⠀⠀⠛⠛⠀⠉⠁⠀⣀⣶⣿⠟⣿⣿⠿⣿⣦⡀⠀⠉⠉⠀⠛⠃⠀⠀⢠⠟⠀⠀⠀⣰⣿⣿⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣿⣿⣿⣦⣀⣀⣤⡈⢷⡀⠀⠀⠀⠀⠀⠀⠀⠁⣠⣾⣿⣿⣶⣄⠉⠀⠀⠀⠀⠀⠀⠀⢀⡾⢁⡀⠀⠀⣾⣿⣿⣿⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⠈⠓⣤⠀⠀⠀⠀⠀⠈⠋⠀⣿⣿⠀⠛⠀⠀⠀⠀⠀⢀⣤⠛⢠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠀⠀⠀⠀⠉⠓⢦⣤⣀⡀⠀⠉⠉⠀⢀⣀⣤⡴⠚⠉⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⣀⠀⣴⣿⣿⣶⣆⠀⠀⠀⠀⠀⠀⣶⣶⣿⣶⡀⠀⣠⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣄⣀⣀⣀⣀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
                "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
                ]
    else:
        raise AttributeError(f"Parameter {splash_type=} value is not supported.")

    # .... Use different padding character for negative splash type ...............................
    padding_char = "⣿" if splash_type == 'negative' else "⠀"

    result = []
    top_botom_pad_len = 2
    while len(result) < top_botom_pad_len:
        result.append(" ")

    for line in ascii_art_up:
        result.append(echo_centering_str(line, snow_formatting, padding_char))

    # .... Title line with formatting .............................................................
    title_line = f"···•· {title} ··•••"
    result.append(echo_centering_str(title_line, title_formatting, "·"))

    for line in ascii_art_down:
        result.append(echo_centering_str(line, snow_formatting, padding_char))

    # .... Add URLs ...............................................................................
    result.append(" ")
    result.append(echo_centering_str(main_url, url_formatting, " "))
    if optional_repo_url is not None:
        result.append(echo_centering_str(optional_repo_url, url_formatting, " "))

    for each_bottom_row in range(top_botom_pad_len):
        result.append(" ")

    # .... Print splash screen ....................................................................
    if not is_pytest_run():
        for line in result:
            print(line)

    return result


def show_splash_screen_in_console(splash_type: str = 'negative') -> List[str]:
    """Display a splash screen in the console.

    Python implementation that replaces the shell script call to n2st::AnonLab_splash.

    :param splash_type: either 'small', 'big' or 'negative' (defaut)
    :return: the splash screen line by lines aggregated in a list
    """
    git_remote_url, git_name, project_path = get_git_info()

    return AnonLab_splash(title=git_name, optional_repo_url=git_remote_url, splash_type=splash_type)


if __name__ == '__main__':

    # Show off splash
    for splash_type in ['small', 'big', 'negative']:
        AnonLab_splash(title="MTM-Pro",
                      optional_repo_url=(
                              "https://github.com/anonymous/MTM-Pro"),
                      splash_type=splash_type)
