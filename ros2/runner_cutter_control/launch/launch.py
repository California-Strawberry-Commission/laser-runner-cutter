import os

import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression

from aioros2 import LaunchNode
from amiga_control import amiga_control_node
from camera_control import camera_control_node
from furrow_perceiver import furrow_perceiver_node
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

    realsense_forward_launch_node = IncludeLaunchDescription(
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
            # Note: the furrow perceiver node depends on the RealSense node topics, but since
            # the RealSense topics are not using aioros2, we use a stub node def. This name
            # must match the one in parameters.yaml.
            "camera_name": "cam_forward",
            "camera_namespace": "/",
            # Hole filling and temporal filters significantly improve depth map quality,
            # but the other filters do not.
            "temporal_filter.enable": "true",
            "hole_filling_filter.enable": "true",
        }.items(),
        condition=IfCondition(launch_nav_nodes),
    )

    realsense_backward_launch_node = IncludeLaunchDescription(
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
            # Note: the furrow perceiver node depends on the RealSense node topics, but since
            # the RealSense topics are not using aioros2, we use a stub node def. This name
            # must match the one in parameters.yaml.
            "camera_name": "cam_backward",
            "camera_namespace": "/",
            # Hole filling and temporal filters significantly improve depth map quality,
            # but the other filters do not.
            "temporal_filter.enable": "true",
            "hole_filling_filter.enable": "true",
        }.items(),
        condition=IfCondition(launch_nav_nodes),
    )

    furrow_perceiver_forward_launch_node = LaunchNode(
        furrow_perceiver_node,
        name="furrow_perceiver_forward",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
        condition=IfCondition(launch_nav_nodes),
    )

    furrow_perceiver_backward_launch_node = LaunchNode(
        furrow_perceiver_node,
        name="furrow_perceiver_backward",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
        condition=IfCondition(launch_nav_nodes),
    )

    amiga_launch_node = LaunchNode(
        amiga_control_node,
        name="amiga",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
        condition=IfCondition(launch_nav_nodes),
    )

    guidance_brain_launch_node = LaunchNode(
        guidance_brain_node,
        name="guidance_brain",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
        condition=IfCondition(launch_nav_nodes),
    )

    # Link node dependencies
    guidance_brain_launch_node.furrow_perceiver_forward_node.link(
        furrow_perceiver_forward_launch_node
    )
    guidance_brain_launch_node.furrow_perceiver_backward_node.link(
        furrow_perceiver_backward_launch_node
    )
    guidance_brain_launch_node.amiga_node.link(amiga_launch_node)

    # endregion

    # region Cutter nodes

    camera_control_launch_node = LaunchNode(
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

    laser_control_launch_node = LaunchNode(
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

    runner_cutter_control_launch_node = LaunchNode(
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
    runner_cutter_control_launch_node.camera_node.link(camera_control_launch_node)
    runner_cutter_control_launch_node.laser_node.link(laser_control_launch_node)

    # The following is for if we want to run runner cutter related nodes in a single process.
    # Composable nodes are not supported for Python nodes yet, so we use an executable that spins
    # multiple nodes.
    single_process_runner_launch_node = launch_ros.actions.Node(
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
            realsense_forward_launch_node,
            realsense_backward_launch_node,
            furrow_perceiver_forward_launch_node,
            furrow_perceiver_backward_launch_node,
            amiga_launch_node,
            guidance_brain_launch_node,
            camera_control_launch_node,
            laser_control_launch_node,
            runner_cutter_control_launch_node,
            single_process_runner_launch_node,
        ]
    )  # type: ignore
