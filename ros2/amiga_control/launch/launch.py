import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription

from aioros2 import LaunchNode
from amiga_control import amiga_control_node

def generate_launch_description():
    parameters_file = os.path.join(
        get_package_share_directory("amiga_control"),
        "config",
        "params.yaml",
    )

    return LaunchDescription(
        [
            LaunchNode(
                amiga_control_node, name="amiga0", parameters=[parameters_file]
            )
        ]
    )  # type: ignore
