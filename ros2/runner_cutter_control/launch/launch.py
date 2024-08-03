import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription

from aioros2 import LaunchNode
from camera_control import camera_control_node
from laser_control import laser_control_node
from runner_cutter_control import runner_cutter_control_node


def generate_launch_description():
    parameters_file = os.path.join(
        get_package_share_directory("runner_cutter_control"),
        "config",
        "parameters.yaml",
    )

    camera_node = LaunchNode(
        camera_control_node,
        name="camera0",
        parameters=[parameters_file],
        respawn=True,
        respawn_delay=2.0,
    )

    laser_node = LaunchNode(
        laser_control_node,
        name="laser0",
        parameters=[parameters_file],
        respawn=True,
        respawn_delay=2.0,
    )

    runner_cutter_node = LaunchNode(
        runner_cutter_control_node,
        name="control0",
        parameters=[parameters_file],
        respawn=True,
        respawn_delay=2.0,
    )

    # Link nodes
    runner_cutter_node.camera_node.link(camera_node)
    runner_cutter_node.laser_node.link(laser_node)

    return LaunchDescription([camera_node, laser_node, runner_cutter_node])  # type: ignore
