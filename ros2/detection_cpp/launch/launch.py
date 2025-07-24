from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="detection_cpp",
                executable="detection_node",
                name="detection_node",
                output="screen",
                emulate_tty=True,
            ),
        ]
    )
