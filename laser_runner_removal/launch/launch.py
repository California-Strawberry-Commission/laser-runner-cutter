from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="laser_runner_removal", executable="realsense", name="realsense"
            ),
            Node(
                package="laser_runner_removal", executable="main_node", name="main_node"
            ),
        ]
    )
