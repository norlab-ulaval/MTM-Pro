# coding=utf-8
from typing import Union

import gymnasium

# Standalone release: the f1tenth gym environments (`F110Env`, `F110EnvNoSteerBuffer`) are not
# shipped. The alias names are kept so that the shared training code signatures stay unchanged.
LEARNABLE_F110ENV_TYPES = Union[gymnasium.Env]

LEARNABLE_F110ENV_GEN_TYPES = Union[LEARNABLE_F110ENV_TYPES, gymnasium.Wrapper]
