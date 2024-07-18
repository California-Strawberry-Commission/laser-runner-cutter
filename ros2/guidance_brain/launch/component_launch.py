from aioros2.launch_driver import LaunchNode
from amiga_control import amiga_control_node
from furrow_perceiver import realsense_stub
from furrow_perceiver import furrow_perceiver_node
from guidance_brain import guidance_brain_node

import os
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
    FrontendLaunchDescriptionSource,
)


def generate_launch_description():
    parameters_file = os.path.join(
        get_package_share_directory("guidance_brain"),
        "config",
        "params.yaml",
    )

    rs_node0 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("realsense2_camera"), "launch"
                ),
                "/rs_launch.py",
            ]
        ),
        launch_arguments={
            "camera": "camera_1",
            "camera_name": "cam0",
            "filters": "decimation,spatial,temporal,hole_filling",
        }.items(),
    )

    furrow_perc0 = LaunchNode(
        furrow_perceiver_node,
        name="furrow0",
        parameters=[parameters_file],
    )

    amiga = LaunchNode(
        amiga_control_node,
        name="amiga0",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
    )

    brain = LaunchNode(
        guidance_brain_node,
        name="brain0",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
    )

    # Link nodes
    brain.perceiver.link(furrow_perc0)
    brain.amiga.link(amiga)

    return LaunchDescription(
        [
            amiga,
            brain,
            rs_node0,
            furrow_perc0,
        ]
    )
