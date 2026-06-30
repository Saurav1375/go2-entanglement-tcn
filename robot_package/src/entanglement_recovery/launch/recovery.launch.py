"""Launch the recovery node alone (assumes the detector is already running).

Override the safety gate at launch:  ros2 launch entanglement_recovery recovery.launch.py enable_actuation:=true
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
    enable = LaunchConfiguration("enable_actuation")
    return LaunchDescription([
        DeclareLaunchArgument("enable_actuation", default_value="false",
                              description="true actuates the robot; false = dry-run"),
        Node(
            package="entanglement_recovery",
            executable="recovery_node",
            name="entanglement_recovery",
            output="screen",
            parameters=[config, {"enable_actuation": enable}],
        ),
    ])
