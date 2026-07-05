"""Launch the front-jump recovery node alone (assumes the detector is already running).

    ros2 launch entanglement_recovery recovery.launch.py

Optionally override the network interface used by the Unitree SDK2:
    ros2 launch entanglement_recovery recovery.launch.py network_interface:=eth0

WARNING: this node actuates the robot (a real front jump) on a sustained
entanglement alarm. Ensure flat ground, clearance ahead, and adequate battery.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("entanglement_recovery")
    config = os.path.join(share, "config", "recovery.yaml")
    iface = LaunchConfiguration("network_interface")
    return LaunchDescription([
        DeclareLaunchArgument(
            "network_interface", default_value="eth0",
            description="Interface for the Unitree SDK2 DDS (Go2 internal ethernet = eth0)"),
        Node(
            package="entanglement_recovery",
            executable="recovery_node",
            name="entanglement_recovery",
            output="screen",
            parameters=[config, {"network_interface": iface}],
        ),
    ])
