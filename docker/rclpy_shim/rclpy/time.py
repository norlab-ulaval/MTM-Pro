# coding=utf-8
"""See `rclpy/__init__.py`."""


class Time:
    """Stand-in for `rclpy.time.Time`. Instantiating it means a ROS2-only code path was reached."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "rclpy is not available in the MTM-Pro standalone image (ROS2 rosbag extraction is not "
            "supported, only stamped csv datasets)."
        )
