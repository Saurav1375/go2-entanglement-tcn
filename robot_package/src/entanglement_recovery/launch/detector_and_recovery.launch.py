"""Launch the detector AND the recovery node together (full pipeline).

  ros2 launch entanglement_recovery detector_and_recovery.launch.py            # dry-run
  ros2 launch entanglement_recovery detector_and_recovery.launch.py enable_actuation:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    det_share = get_package_share_directory("entanglement_detector")
    rec_share = get_package_share_directory("entanglement_recovery")
    det_cfg = os.path.join(det_share, "config", "config.yaml")
    rec_cfg = os.path.join(rec_share, "config", "recovery.yaml")
    enable = LaunchConfiguration("enable_actuation")
    return LaunchDescription([
        DeclareLaunchArgument("enable_actuation", default_value="false"),
        Node(package="entanglement_detector", executable="entanglement_node",
             name="entanglement_detector", output="screen", parameters=[det_cfg]),
        Node(package="entanglement_recovery", executable="recovery_node",
             name="entanglement_recovery", output="screen",
             parameters=[rec_cfg, {"enable_actuation": enable}]),
    ])
