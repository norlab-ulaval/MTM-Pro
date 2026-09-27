# coding=utf-8
"""PyPose ``SO(3)`` convention boundary (Variant B) for RLRP-736.

Introduced by action **S0 / §3.1 (convention shim)** of the Per-Environment
Feature Handling ``.junie`` plan — Variant B (PyPose-based):
``rlrp-736-per-environment-feature-handling-pypose-based-plan-20260711.md``
(YouTrack RLRP-736).

**Why this module exists (rationale).** The codebase stores attitude
quaternions in **``[w, x, y, z]`` (scalar-first / Hamilton, ``wxyz``)** order,
whereas PyPose's :class:`pypose.SO3` ``LieTensor`` stores them
**``[x, y, z, w]`` (scalar-last, ``xyzw``)**. Mixing the two silently produces a
*different rotation* (risk R-B1 in the plan). To keep that hazard contained,
**every** conversion between the two conventions must go through this single,
tested boundary — no ad-hoc slicing elsewhere.

The functions are thin, fully differentiable, and shape-preserving on the
leading (batch/…): only the last dimension (width 4) is reordered.

References
----------
- PyPose: Wang et al., "PyPose: A Library for Robot Learning with Physics-based
  Optimization", CVPR 2023 (``pypose.SO3`` uses ``xyzw`` storage).
"""
from __future__ import annotations

import torch

# ``pypose`` is a hard dependency (declared in pyproject.toml). Since RLRP-746
# it is imported (and validated) once at module load, so the SO(3) conversion
# helpers and rotation losses no longer pay a per-call availability check on the
# training hot path.
try:
    import pypose as pp
except ImportError as _exc:  # pragma: no cover - defensive import-time guard
    raise ImportError(
        "The 'pypose' package is required for the Variant-B SO(3) rotation "
        "handling (RLRP-736 §14A / S2.2-B, RLRP-746). It is declared in "
        "pyproject.toml; rebuild the DNA image (or `pip install pypose`) to make "
        "it available."
    ) from _exc


def wxyz_to_pp_SO3(q_wxyz: torch.Tensor) -> "pp.SO3":
    """Convert a scalar-first ``[..., 4]`` quaternion to a :class:`pypose.SO3`.

    :param q_wxyz: Attitude quaternion(s) in codebase ``[w, x, y, z]`` order.
    :return: A ``pypose.SO3`` ``LieTensor`` (internally ``[x, y, z, w]``).
    """
    w = q_wxyz[..., 0:1]
    xyz = q_wxyz[..., 1:4]
    xyzw = torch.cat([xyz, w], dim=-1)  # pypose expects x, y, z, w
    return pp.SO3(xyzw)


def pp_SO3_to_wxyz(rotation: "pp.SO3") -> torch.Tensor:
    """Convert a :class:`pypose.SO3` back to a scalar-first ``[..., 4]`` quaternion.

    :param rotation: A ``pypose.SO3`` ``LieTensor`` (internally ``[x, y, z, w]``).
    :return: Attitude quaternion(s) in codebase ``[w, x, y, z]`` order.
    """
    xyzw = rotation.tensor()
    return torch.cat([xyzw[..., 3:4], xyzw[..., 0:3]], dim=-1)
