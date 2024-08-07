import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from camera_control import camera_control_node
from laser_control import laser_control_node
from runner_cutter_control import runner_cutter_control_node
   
from aioros2 import LaunchNode

from amiga_control import amiga_control_node
from furrow_perceiver import realsense_stub
from furrow_perceiver import furrow_perceiver_node
from guidance_brain import guidance_brain_node

import importlib.util

import launch_ros.actions
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
    FrontendLaunchDescriptionSource,
)

# 819312072040 - forward
# 017322073371 - backward

def generate_launch_description():
    parameters_file = os.path.join(
        get_package_share_directory("runner_cutter_control"),
        "config",
        "parameters.yaml",
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
            "serial_no": "'819312072040'",
            "camera_name": "cam0",
            "filters": "decimation,spatial,temporal,hole_filling",
        }.items(),
    )

    rs_node1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("realsense2_camera"), "launch"
                ),
                "/rs_launch.py",
            ]
        ),
        launch_arguments={
            "camera": "camera_2",
            "serial_no": "'017322073371'",
            "camera_name": "cam1",
            "filters": "decimation,spatial,temporal,hole_filling",
        }.items(),
    )

    furrow_perc0 = LaunchNode(
        furrow_perceiver_node,
        name="furrow0",
        parameters=[parameters_file],
    )

    furrow_perc1 = LaunchNode(
        furrow_perceiver_node,
        name="furrow1",
        parameters=[parameters_file],
    )

    rosbridge = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource(
            [
                get_package_share_directory("rosbridge_server"),
                "/launch/rosbridge_websocket_launch.xml",
            ]
        ),
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

    video_server = launch_ros.actions.Node(
        package="web_video_server", executable="web_video_server", name="wvs"
    )
    
    # brain.perceiver_forward.link(furrow_perc0)
    
    # brain.perceiver_backward.link(furrow_perc1)
    brain.amiga.link(amiga)
    
    launchables = [
        amiga,
        brain,
        rosbridge,
        video_server,
        rs_node0,
        rs_node1,
        furrow_perc0,
        furrow_perc1,
    ]
    
    if importlib.util.find_spec("arena_api"):
 
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
        print(runner_cutter_node)
        runner_cutter_node.camera_node.link(camera_node)
        runner_cutter_node.laser_node.link(laser_node)

        launchables.append(camera_node)
        launchables.append(laser_node)
        launchables.append(runner_cutter_node)
        


    return LaunchDescription(launchables)  # type: ignore
