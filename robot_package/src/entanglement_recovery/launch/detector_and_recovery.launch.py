"""Launch the detector AND the recovery node together (full pipeline).

  ros2 launch entanglement_recovery detector_and_recovery.launch.py
  ros2 launch entanglement_recovery detector_and_recovery.launch.py network_interface:=eth0

WARNING: on an entanglement alarm the robot runs the sequence
stop -> move back -> stop -> front jump -> stop (once). Ensure flat ground, clear space
behind and ahead of the robot, the robot standing, and adequate battery.
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
    iface = LaunchConfiguration("network_interface")
    return LaunchDescription([
        DeclareLaunchArgument(
            "network_interface", default_value="eth0",
            description="Interface for the Unitree SDK2 DDS (Go2 internal ethernet = eth0)"),
        Node(package="entanglement_detector", executable="entanglement_node",
             name="entanglement_detector", output="screen", parameters=[det_cfg]),
        Node(package="entanglement_recovery", executable="recovery_node",
             name="entanglement_recovery", output="screen",
             parameters=[rec_cfg, {"network_interface": iface}]),
    ])
