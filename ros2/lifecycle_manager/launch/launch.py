from launch import LaunchDescription

from aioros2 import LaunchNode
from lifecycle_manager import lifecycle_manager_node


def generate_launch_description():
    return LaunchDescription(
        [
            LaunchNode(
                lifecycle_manager_node,
                name="lifecycle_manager",
                respawn=True,
                respawn_delay=2.0,
            )
        ]
    )
