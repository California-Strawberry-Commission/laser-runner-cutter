from launch import LaunchDescription

from aioros2 import LaunchNode
from webrtc_ros2_py import webrtc_node


def generate_launch_description():
    return LaunchDescription(
        [
            LaunchNode(
                webrtc_node,
                name="webrtc",
                respawn=True,
                respawn_delay=2.0,
            )
        ]
    )
