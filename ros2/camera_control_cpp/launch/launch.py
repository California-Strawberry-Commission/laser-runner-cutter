from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="camera_control_cpp",
                executable="camera_control_node",
                name="camera_control_node",
                output="screen",
                emulate_tty=True,
            ),
        ]
    )
