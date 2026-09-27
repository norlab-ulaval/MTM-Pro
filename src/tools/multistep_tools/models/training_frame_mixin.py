# coding=utf-8
"""RLRP-758 — persist the model's velocity *training frame* with the checkpoint.

Plan provenance:
  `.junie/ai_artifact/plans/feat_extend_velocity_frame_logic_plan_RLRP-758_20260726.md`
  (task T8b, YouTrack RLRP-758).

The velocity ``training_frame`` (``world`` legacy | ``body`` new default;
``gravity_aligned`` = the heading frame, RLRP-792) is a MODEL-INTERNAL fact: it selects
the coordinate frame the dataset velocities were re-expressed into at ingestion
(see ``robotic_env_pipeline_utils.utils.resolve_training_frame_from_cfg`` +
``convert_velocity_channels_to_training_frame``). A model trained on ``body``
velocities can NEVER be safely deployed / reconstructed under a ``world``
integrator (and vice-versa) — the position/attitude roll-out would silently
double- or never-convert (RLRP-755 report §4 hazard).

To make that impossible this concern is isolated as a focused mixin, mirroring
:class:`tools.multistep_tools.models.feature_geometry_loss_mixin.FeatureGeometryLossMixin`:

* the frame is set post-construction at the setup seam
  (``set_training_frame`` <- ``resolve_training_frame_from_cfg(cfg)``), exactly
  like ``set_feature_handler`` — so NO new constructor kwarg has to be threaded
  through the whole model MRO;
* it is written into the host model's own ``model_config`` checkpoint entry and
  recovered + asserted on load (see
  :class:`ExponentialFamilyMLP._save_state_dict` / ``_load_state_dict``), so a
  frame-less legacy checkpoint fails loud by construction (per RLRP-758 D1:
  every pre-existing saved model is wrong w.r.t. the new default).

Neutral by default: with no explicit ``set_training_frame`` call the frame is the
new default ``body``; a save always records a valid frame.

NOTE (scope): ``training_frame`` is metadata used ONLY for checkpoint persistence
and the save/load + deploy-boundary guards — it is deliberately NOT read inside
the model ``forward``. The frame is physically realized at INGESTION (velocities
are re-expressed into it before training); the model itself is frame-agnostic. Do
not add ``forward``-time logic keyed off this attribute.
"""
from __future__ import annotations

# Valid model training frames. ``gravity_aligned`` (the heading frame: the yaw
# twist of the attitude about the world gravity axis) was a RESERVED slot until
# RLRP-792 implemented it in the ingestion converter and the deploy integrator.
_VALID_TRAINING_FRAMES = ("world", "body", "gravity_aligned")

# The neutral default frame (RLRP-758 D1: the new deliberate research default).
_DEFAULT_TRAINING_FRAME = "body"


def _validate_training_frame(frame: str) -> str:
    """Validate a training-frame label (RLRP-792 opened ``gravity_aligned``)."""
    if frame not in _VALID_TRAINING_FRAMES:
        raise ValueError(
            f"bad training_frame={frame!r} (expected one of {_VALID_TRAINING_FRAMES})"
        )
    return frame


class TrainingFrameMixin:
    """Persist + recover the model-internal velocity training frame (RLRP-758 T8b)."""

    _training_frame: str

    def _init_training_frame(self, training_frame: str = _DEFAULT_TRAINING_FRAME) -> None:
        """Initialise the training-frame state.

        Call once from the host model ``__init__`` (after ``nn.Module`` init).
        Defaults to the new ``body`` frame; the run-selected frame is attached
        post-construction via :meth:`set_training_frame` at the setup seam.
        """
        self._training_frame = _validate_training_frame(str(training_frame))
        return None

    def get_training_frame(self) -> str:
        """Return the model's velocity training frame (``world`` | ``body``).

        Fail loud if the frame was never initialised (host forgot to call
        :meth:`_init_training_frame`) — a silent ``body`` fallback here could let a
        genuinely-misconfigured model slip past the load-time stored==active guard
        (RLRP-758 merit review). ``ExponentialFamilyMLP.__init__`` always inits it,
        so this only trips on a mis-wired host.
        """
        try:
            return self._training_frame
        except AttributeError as e:
            raise AttributeError(
                f"{type(self).__name__}: training frame not initialised — the host "
                f"model must call `_init_training_frame()` in its `__init__` "
                f"(RLRP-758 TrainingFrameMixin)."
            ) from e

    def set_training_frame(self, training_frame: str) -> None:
        """Set the model's velocity training frame.

        Wired at ``setup_multistep_step_model`` from
        ``resolve_training_frame_from_cfg(cfg)`` (``ms_model.training_frame``),
        mirroring :meth:`FeatureGeometryLossMixin.set_feature_handler`. Validated
        immediately so a config typo surfaces at build time.
        """
        self._training_frame = _validate_training_frame(str(training_frame))
        return None
