from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="web_video_server",
                executable="web_video_server",
                respawn=True,
                respawn_delay=2.0,
            ),
        ]
    )  # type: ignore
