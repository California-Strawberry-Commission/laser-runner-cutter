from launch import LaunchDescription

from aioros2 import LaunchNode
from gstreamer_webrtc import gstreamer_node


def generate_launch_description():
    return LaunchDescription(
        [
            LaunchNode(
                gstreamer_node,
                name="gstreamer",
                respawn=True,
                respawn_delay=2.0,
            )
        ]
    )
