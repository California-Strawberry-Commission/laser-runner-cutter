import os

import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import (
    FrontendLaunchDescriptionSource,
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration, PythonExpression

from aioros2 import LaunchNode
from amiga_control import amiga_control_node
from camera_control import camera_control_node
from furrow_perceiver import furrow_perceiver_node, realsense_stub
from guidance_brain import guidance_brain_node
from laser_control import laser_control_node
from runner_cutter_control import runner_cutter_control_node


def generate_launch_description():
    parameters_file = os.path.join(
        get_package_share_directory("runner_cutter_control"),
        "config",
        "parameters.yaml",
    )

    # region Launch args

    launch_nav_nodes_launch_arg = DeclareLaunchArgument(
        "launch_nav_nodes", default_value="True", description="Launch nav-related nodes"
    )
    launch_nav_nodes = LaunchConfiguration("launch_nav_nodes")
    launch_cutter_nodes_launch_arg = DeclareLaunchArgument(
        "launch_cutter_nodes",
        default_value="True",
        description="Launch cutter-related nodes",
    )
    launch_cutter_nodes = LaunchConfiguration("launch_cutter_nodes")
    use_single_process_launch_arg = DeclareLaunchArgument(
        "use_single_process",
        default_value="False",
        description="Start nodes in a single process",
    )
    use_single_process = LaunchConfiguration("use_single_process")

    # endregion

    # region Nav nodes

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
            "camera_namespace": "/",
            "decimation_filter.enable": "true",
            "temporal_filter.enable": "true",
            "spatial_filter.enable": "true",
            "hole_filling_filter.enable": "true",
        }.items(),
        condition=IfCondition(launch_nav_nodes),
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
            "camera_namespace": "/",
            "decimation_filter.enable": "true",
            "temporal_filter.enable": "true",
            "spatial_filter.enable": "true",
            "hole_filling_filter.enable": "true",
        }.items(),
        condition=IfCondition(launch_nav_nodes),
    )

    furrow_perc0 = LaunchNode(
        furrow_perceiver_node,
        name="furrow0",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
        condition=IfCondition(launch_nav_nodes),
    )

    furrow_perc1 = LaunchNode(
        furrow_perceiver_node,
        name="furrow1",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
        condition=IfCondition(launch_nav_nodes),
    )

    amiga = LaunchNode(
        amiga_control_node,
        name="amiga0",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
        condition=IfCondition(launch_nav_nodes),
    )

    brain = LaunchNode(
        guidance_brain_node,
        name="brain0",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
        condition=IfCondition(launch_nav_nodes),
    )

    # Link node dependencies
    brain.perceiver_forward.link(furrow_perc0)
    brain.perceiver_backward.link(furrow_perc1)
    brain.amiga.link(amiga)

    # endregion

    # region Cutter nodes

    camera_node = LaunchNode(
        camera_control_node,
        name="camera0",
        parameters=[parameters_file],
        respawn=True,
        respawn_delay=2.0,
        output="screen",
        emulate_tty=True,
        condition=IfCondition(
            PythonExpression([launch_cutter_nodes, " and not ", use_single_process])
        ),
    )

    laser_node = LaunchNode(
        laser_control_node,
        name="laser0",
        parameters=[parameters_file],
        respawn=True,
        respawn_delay=2.0,
        output="screen",
        emulate_tty=True,
        condition=IfCondition(
            PythonExpression([launch_cutter_nodes, " and not ", use_single_process])
        ),
    )

    runner_cutter_node = LaunchNode(
        runner_cutter_control_node,
        name="control0",
        parameters=[parameters_file],
        respawn=True,
        respawn_delay=2.0,
        output="screen",
        emulate_tty=True,
        condition=IfCondition(
            PythonExpression([launch_cutter_nodes, " and not ", use_single_process])
        ),
    )

    # Link node dependencies
    runner_cutter_node.camera_node.link(camera_node)
    runner_cutter_node.laser_node.link(laser_node)

    # The following is for if we want to run runner cutter related nodes in a single process.
    # Composable nodes are not supported for Python nodes yet, so we use an executable that spins
    # multiple nodes.
    single_process_runner = launch_ros.actions.Node(
        package="runner_cutter_control",
        executable="single_process_runner",
        parameters=[parameters_file],
        respawn=True,
        respawn_delay=2.0,
        output="screen",
        emulate_tty=True,
        condition=IfCondition(
            PythonExpression([launch_cutter_nodes, " and ", use_single_process])
        ),
    )

    # endregion

    return LaunchDescription(
        [
            launch_nav_nodes_launch_arg,
            launch_cutter_nodes_launch_arg,
            use_single_process_launch_arg,
            rs_node0,
            rs_node1,
            furrow_perc0,
            furrow_perc1,
            amiga,
            brain,
            camera_node,
            laser_node,
            runner_cutter_node,
            single_process_runner,
        ]
    )  # type: ignore
