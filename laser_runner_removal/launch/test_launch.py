from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="laser_control",
                executable="laser_control_node",
                name="laser_control_node",
            ),
            Node(
                package="laser_runner_removal", executable="realsense", name="realsense"
            ),
            Node(
                package="laser_runner_removal", executable="test_node", name="test_node"
            ),
        ]
    )
