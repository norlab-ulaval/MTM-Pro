# coding=utf-8
from dataclasses import dataclass

from trajectory_container_tools.dataclasses import TestMotionTrajectoryDataclass


@dataclass
class TestTrajectoryEntry:
    """Wrapper that pairs a test-time rollout environment with its trajectory metadata.

    Used to pass trajectory information through the multirun test-time rollout pipeline
    without modifying ``TestMotionTrajectoryDataclass`` itself.

    :ivar env: The test motion trajectory dataclass for rollout execution.
    :ivar trajectory_name: Full trajectory name as in the config, e.g. ``"test/race_track_2"``.
    :ivar category: Trajectory length category — ``"S"`` (short), ``"M"`` (medium), or ``"L"`` (long).
    :ivar short_name: Trajectory basename used for directory naming, e.g. ``"race_track_2"``.
    """

    env: TestMotionTrajectoryDataclass
    trajectory_name: str
    category: str
    short_name: str
