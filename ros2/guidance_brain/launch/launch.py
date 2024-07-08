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

    rs_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("realsense2_camera"), "launch"
                ),
                "/rs_launch.py",
            ]
        ),
        launch_arguments={
            "camera_name": "cam0",
            "filters": "decimation,spatial,temporal,hole_filling",
        }.items(),
    )

    rosbridge = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource(
            [
                get_package_share_directory("rosbridge_server"),
                "/launch/rosbridge_websocket_launch.xml",
            ]
        ),
    )

    amiga = LaunchNode(amiga_control_node, name="amiga0", parameters=[parameters_file])

    furrow_perc = LaunchNode(
        furrow_perceiver_node,
        name="furrow0",
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
    brain.perceiver.link(furrow_perc)

    return LaunchDescription(
        [
            # rosbridge,
            amiga,
            furrow_perc,
            brain,
            rs_node
        ]
    )  # type: ignore rs_node
