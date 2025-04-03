from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="runner_cutter_control_cpp",
                executable="runner_cutter_control_node",
                name="runner_cutter_control_node",
                output="screen",
                emulate_tty=True,
            ),
        ]
    )
