"""Launch the entanglement detector with config.yaml parameters."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("entanglement_detector")
    config = os.path.join(share, "config", "config.yaml")
    return LaunchDescription([
        Node(
            package="entanglement_detector",
            executable="entanglement_node",
            name="entanglement_detector",
            output="screen",
            parameters=[config],
        ),
    ])
