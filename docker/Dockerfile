# syntax=docker/dockerfile:1
#
# MTM-Pro standalone image (CPU + NVIDIA GPU capable).
#
# The image mirrors the python environment of the research container the paper experiments were run in
# (Ubuntu 22.04, Python 3.10, PyTorch, hydra) without any lab specific tooling. GPU support comes from the
# CUDA libraries bundled in the PyTorch wheels + the NVIDIA container toolkit on the host
# (`docker compose --profile gpu ...`). The three norlab/RedLeader962 python packages (mbrl-lib fork,
# math-gymnasium, trajectory-container-tools) are installed from their git repositories, at the
# revision declared in `pyproject.toml`.
#
# Build from the repository root:
#   $ docker compose -f docker/docker-compose.yaml build
#
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MTM_PRO_PATH=/opt/MTM-Pro

ENV DN_PROJECT_PATH=${MTM_PRO_PATH} \
    DN_PROJECT_USER=mtm-pro \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    MPLBACKEND=Agg \
    TZ=UTC

# ....System packages..............................................................................
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-venv \
        git ca-certificates build-essential \
        libgl1 libglib2.0-0 libhdf5-dev \
        tree vim less \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && python -m pip install --upgrade pip

# ....PyTorch stack................................................................................
# Note: the PyPI wheels bundle the CUDA runtime on linux/amd64 (CPU only wheels on linux/arm64).
# `torchrl`/`tensordict` (mbrl-lib replay buffer backend) are resolved together with torch so that pip
# picks a mutually compatible set.
RUN python -m pip install "numpy<2" torch torchvision torchrl tensordict \
    && python -c 'import torch, torchrl, tensordict; print("torch", torch.__version__, "cuda build:", torch.version.cuda)'

# ....Third party python requirements..............................................................
COPY docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

# ....ROS2 `rclpy` import shim.....................................................................
# `trajectory_container_tools` imports `rclpy.time.Time` at import time (rosbag extraction path). No
# ROS2 distribution is installed here: the experiments only use its stamped csv extraction path.
COPY docker/rclpy_shim /tmp/rclpy_shim
RUN python -m pip install --no-deps /tmp/rclpy_shim && rm -rf /tmp/rclpy_shim

# ....Norlab python packages (from git)............................................................
# Same requirements as `pyproject.toml` (their third party dependencies are already installed above).
RUN python -m pip install --no-deps \
        "mbrl @ git+https://github.com/RedLeader962/mbrl-lib.git" \
        "math-gymnasium @ git+https://github.com/norlab-ulaval/math-gymnasium.git" \
        "trajectory-container-tools @ git+https://github.com/norlab-ulaval/trajectory-container-tools.git"

# ....Project (source)............................................................................
WORKDIR ${MTM_PRO_PATH}
COPY . ${MTM_PRO_PATH}/
RUN python -m pip install --no-deps --editable . --config-settings editable_mode=compat

# src-layout package: make the `src/` directory importable independently of the editable install
ENV PYTHONPATH=${MTM_PRO_PATH}/src

# ....Sanity check.................................................................................
RUN python -c 'import torch, hydra, omegaconf, mbrl, math_gymnasium, pypose, h5py' \
    && python -c 'import trajectory_container_tools as tct; tct.extractor.from_stamped_csv' \
    && cd src && python -c 'import launcher.main_utils; import tools.multistep_tools.models' \
    && echo "MTM-Pro import sanity check OK"

WORKDIR ${MTM_PRO_PATH}/src
CMD ["bash"]
