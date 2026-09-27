# coding=utf-8
import os
from os import getenv
from typing import AnyStr


def is_dockerized_AnonLab_env_var_set() -> bool:
    return os.getenv("DN_PROJECT_PATH") is not None


# (Priority) ToDo: unit-test >>> is indirectly tested for now
def fetch_dna_container_root_path_via_env_var(expected_root_dir: str) -> AnyStr:
    """Returns the absolute path to the dockerized-project root directory using
    environment variable DN_PROJECT_PATH

    :param expected_root_dir: the name of the expected project root directory (for sanity check)
    :return: the absolute path to the DN-project root directory
    """
    if is_dockerized_AnonLab_env_var_set():
        dn_project_root_path = os.getenv("DN_PROJECT_PATH")
        dn_project_root_name = str(os.path.basename(dn_project_root_path))
        assert dn_project_root_name == expected_root_dir, f"DN_PROJECT_PATH basename={dn_project_root_name} != {expected_root_dir=}"
        return dn_project_root_path
    else:
        raise NameError("Environment variable DN_PROJECT_PATH not found")

# ==== Host Architecture related ==================================================================
# (Priority) inprogress: NMO-669 feat: improve in container host architecture detection
# (Priority) ToDo: refactor to DNA
# (Priority) ToDo: unit-test

def is_run_on_DN_arm64_darwin_host() -> bool:
    """ Check if code is executed inside a Dockerized-Lab container for darwin/arm64 """
    architecture = _check_DN_container_host_architecture()
    if architecture is None:
        return False
    else:
        return architecture == "darwin/arm64"


def is_run_in_DN_arm64_jetson_architecture() -> bool:
    """ Check if code is executed inside a Dockerized-Lab container for Jetson """
    architecture = _check_DN_container_host_architecture()
    if architecture is None:
        return False
    else:
        return architecture == "l4t/arm64"


def is_run_on_DN_x86_linux_host() -> bool:
    """ Check if code is executed inside a Dockerized-Lab container for linux/x86 """
    architecture = _check_DN_container_host_architecture()
    if architecture is None:
        return False
    else:
        return architecture == "linux/x86"

def is_run_on_DN_arm64_linux_host() -> bool:
    """ Check if code is executed inside a Dockerized-Lab container for linux/x86 """
    architecture = _check_DN_container_host_architecture()
    if architecture is None:
        return False
    else:
        # Note: linux/arm64 will most likely be required for multiarch testing via dna ci-tests services
        # (CRITICAL) ToDo: implement support for linux/arm64 on DNA/DN (ref task NMO-669)
        return architecture == "linux/arm64"


def _check_DN_container_host_architecture() -> str | None:
    """ Check Dockerized-Lab container for host OS and architecture """

    if is_dockerized_AnonLab_env_var_set():
        if getenv("DN_HOST") is None:
            # NMO-669 note: DNA project-ci-tests service on arm64 is not reliably set yet (it fail on
            # teamcity multiarch build). Quickhack -> dont raise an error in the mean time.
            # raise OSError("Environment variable 'DN_HOST' is not set") # ToDo: on task NMO-669 end >> UN-mute this line ←
            return "unknown"
        else:
            return getenv("DN_HOST")
    else:
        return None
