from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="laser_control_cpp",
                executable="laser_control_node",
                name="laser_control_node",
                output="screen",
                emulate_tty=True,
            ),
        ]
    )
