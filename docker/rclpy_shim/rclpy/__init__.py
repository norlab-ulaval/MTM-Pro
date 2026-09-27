# coding=utf-8
"""Import-only stand-in for the ROS2 `rclpy` package.

`trajectory_container_tools` (git submodule) imports `rclpy.time.Time` at module import time to
expose its rosbag extraction path. The MTM-Pro standalone image has no ROS2 distribution: the paper
experiments only use the stamped-csv extraction path (`tct.extractor.from_stamped_csv`). This shim
lets the package import; any actual ROS functionality raises a clear error.
"""
