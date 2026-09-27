# coding=utf-8
from typing import Union

import gymnasium

# Standalone release: the legacy `gym` (f1tenth) environments are not shipped, only gymnasium ones.
GYMNASIUM_ENV_TYPES = Union[gymnasium.Env]

GYMNASIUM_WRAPPERS_TYPES = Union[gymnasium.Wrapper]
