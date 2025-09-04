import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    parameters_file = os.path.join(
        get_package_share_directory("livekit_ros2"),
        "config",
        "ros_parameters.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="livekit_ros2",
                executable="livekit_whip_node",
                name="livekit_whip_node",
                parameters=[parameters_file],
                output="screen",
                emulate_tty=True,
            )
        ]
    )
