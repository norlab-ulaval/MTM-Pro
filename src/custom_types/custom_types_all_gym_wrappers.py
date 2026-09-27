# coding=utf-8
from typing import Union

import gymnasium

# Standalone release: the custom f1tenth/gym wrappers are not shipped. The alias name is kept so
# that the shared training code signatures stay unchanged.
ALL_CUSTOM_GYM_WRAPPER_TYPES = Union[gymnasium.Wrapper]
