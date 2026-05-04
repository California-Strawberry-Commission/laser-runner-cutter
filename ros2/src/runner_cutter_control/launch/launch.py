import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    parameters_file = os.path.join(
        get_package_share_directory("runner_cutter_control"),
        "config",
        "parameters.yaml",
    )

    camera_detection_launch_node = ComposableNodeContainer(
        name="camera_detection_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",  # multithreaded executor
        respawn=True,
        respawn_delay=2.0,
        output="screen",
        emulate_tty=True,
        composable_node_descriptions=[
            ComposableNode(
                package="camera_control",
                plugin="CameraControlNode",
                name="camera0",
                parameters=[parameters_file],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            ComposableNode(
                package="detection",
                plugin="DetectionNode",
                name="detection0",
                parameters=[parameters_file],
                extra_arguments=[{"use_intra_process_comms": True}],
                remappings=[
                    (
                        "color/image_raw",
                        "/camera0/color/image_raw",
                    ),  # sub, raw color camera image
                    (
                        "color/camera_info",
                        "/camera0/color/camera_info",
                    ),  # sub, color camera info
                    (
                        "depth/xyz",
                        "/camera0/depth/xyz",
                    ),  # sub, depth xyz data
                    (
                        "depth/camera_info",
                        "/camera0/depth/camera_info",
                    ),  # sub, depth camera info
                    (
                        "debug/image",
                        "/detection0/debug/image",
                    ),  # pub
                ],
            ),
        ],
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
    )

    return LaunchDescription(
        [
            laser_control_launch_node,
            camera_detection_launch_node,
            runner_cutter_control_launch_node,
        ]
    )  # type: ignore
