import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from aioros2.launch import launch
from amiga_control import amiga_control_node
from camera_control import camera_control_node
from furrow_perceiver import furrow_perceiver_node
from guidance_brain import guidance_brain_node


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

    furrow_perceiver_forward_launch_node = launch(
        furrow_perceiver_node,
        name="furrow_perceiver_forward",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
        condition=IfCondition(launch_nav_nodes),
    )

    furrow_perceiver_backward_launch_node = launch(
        furrow_perceiver_node,
        name="furrow_perceiver_backward",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
        condition=IfCondition(launch_nav_nodes),
    )

    amiga_launch_node = launch(
        amiga_control_node,
        name="amiga",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
        condition=IfCondition(launch_nav_nodes),
    )

    guidance_brain_launch_node = launch(
        guidance_brain_node,
        name="guidance_brain",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
        condition=IfCondition(launch_nav_nodes),
    )

    # Link node dependencies
    guidance_brain_launch_node.furrow_perceiver_forward_node = (
        furrow_perceiver_forward_launch_node
    )
    guidance_brain_launch_node.furrow_perceiver_backward_node = (
        furrow_perceiver_backward_launch_node
    )
    guidance_brain_launch_node.amiga_node = amiga_launch_node

    # endregion

    # region Cutter nodes

    camera_control_launch_node = launch(
        camera_control_node,
        name="camera0",
        parameters=[parameters_file],
        respawn=True,
        respawn_delay=2.0,
        output="screen",
        emulate_tty=True,
        condition=IfCondition(launch_cutter_nodes),
    )

    # camera_control_cpp_launch_node = Node(
    #     package="camera_control_cpp",
    #     executable="camera_control_node",
    #     name="camera0",
    #     parameters=[parameters_file],
    #     respawn=True,
    #     respawn_delay=2.0,
    #     output="screen",
    #     emulate_tty=True,
    #     condition=IfCondition(launch_cutter_nodes),
    # )

    # detection_cpp_launch_node = Node(
    #     package="detection_cpp",
    #     executable="detection_node",
    #     name="detection_node",
    #     parameters=[parameters_file],
    #     respawn=True,
    #     respawn_delay=2.0,
    #     output="screen",
    #     emulate_tty=True,
    #     condition=IfCondition(launch_cutter_nodes),
    # )

    camera_detection_container = ComposableNodeContainer(
        name="camera_detection_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[
            ComposableNode(
                package="camera_control_cpp",
                plugin="CameraControlNode",
                name="camera0",
                parameters=[parameters_file],
                extra_arguments=[{"use_intra_process_comms": True}]
            ),
            ComposableNode(
                package="detection_cpp",
                plugin="DetectionNode",
                name="detection_node",
                parameters=[parameters_file],
                extra_arguments=[{"use_intra_process_comms": True}]
            )
        ],
        respawn=True,
        respawn_delay=2.0,
        output="screen",
        emulate_tty=True,
        condition=IfCondition(launch_cutter_nodes)
    )

    laser_control_launch_node = Node(
        package="laser_control",
        executable="laser_control_node",
        name="laser0",
        parameters=[parameters_file],
        respawn=True,
        respawn_delay=2.0,
        output="screen",
        emulate_tty=True,
        condition=IfCondition(launch_cutter_nodes),
    )

    runner_cutter_control_launch_node = Node(
        package="runner_cutter_control",
        executable="runner_cutter_control_node",
        name="control0",
        parameters=[parameters_file],
        respawn=True,
        respawn_delay=2.0,
        output="screen",
        emulate_tty=True,
        condition=IfCondition(launch_cutter_nodes),
    )

    # endregion

    return LaunchDescription(
        [
            launch_nav_nodes_launch_arg,
            launch_cutter_nodes_launch_arg,
            realsense_forward_launch_node,
            realsense_backward_launch_node,
            furrow_perceiver_forward_launch_node,
            furrow_perceiver_backward_launch_node,
            amiga_launch_node,
            guidance_brain_launch_node,
            laser_control_launch_node,
            # camera_control_launch_node,
            # camera_control_cpp_launch_node,
            # detection_cpp_launch_node,
            camera_detection_container,
            runner_cutter_control_launch_node,
        ]
    )  # type: ignore
