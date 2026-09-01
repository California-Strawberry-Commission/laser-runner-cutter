"""
Replays a camera rosbag recorded by CameraControlNode through a live
DetectionNode and records /detection0/detections into an output bag.

Usage:
    ros2 launch runner_cutter_control replay_detections.launch.py \
        source_bag:=/path/to/bag_20260830120000123 \
        output_bag:=/tmp/detections_bag \
        runner_model:=RunnerSegYoloV8l.engine

The launch shuts itself down a few seconds after `ros2 bag play` exits.

Requirements:
  - Needs a CUDA GPU and a TensorRT engine matching `runner_model` under the
    `detection` package's models directory.
  - `output_bag` must not already exist.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def launch_setup(context, *args, **kwargs):
    source_bag = LaunchConfiguration("source_bag").perform(context)
    output_bag = LaunchConfiguration("output_bag").perform(context)
    if not output_bag:
        output_bag = source_bag.rstrip("/") + "_detections"
    runner_model = LaunchConfiguration("runner_model").perform(context)
    start_delay = float(LaunchConfiguration("start_delay").perform(context))

    qos_overrides = os.path.join(
        get_package_share_directory("runner_cutter_control"),
        "config",
        "rosbag_replay_qos_overrides.yaml",
    )

    # DetectionNode, with its input topics remapped onto the names recorded by
    # CameraControlNode so the bag can be replayed unmodified.
    container = ComposableNodeContainer(
        name="offline_detection_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        output="screen",
        emulate_tty=True,
        composable_node_descriptions=[
            ComposableNode(
                package="detection",
                plugin="DetectionNode",
                name="detection0",
                parameters=[
                    {
                        "runner_model": runner_model,
                    }
                ],
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
                ],
            ),
        ],
    )

    start_record = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "-o",
            output_bag,
            "--storage",
            "mcap",
            "/detection0/detections",
            "/tf_static",
        ],
        output="screen",
    )

    # Enable RUNNER detection once the container has had time to load the
    # composable node / TensorRT engine.
    start_detection_proc = ExecuteProcess(
        cmd=[
            "ros2",
            "service",
            "call",
            "/detection0/start_detection",
            "detection_interfaces/srv/StartDetection",
            "{detection_type: 1}",
        ],
        output="screen",
    )
    start_detection = TimerAction(
        period=start_delay,
        actions=[start_detection_proc],
    )

    # Start playback with a delay after the start_detection service call
    # returns, so no replayed frames arrive before detection is enabled.
    play_proc = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            source_bag,
            "--qos-profile-overrides-path",
            qos_overrides,
        ],
        output="screen",
    )
    start_playback = RegisterEventHandler(
        OnProcessExit(
            target_action=start_detection_proc,
            on_exit=[TimerAction(period=2.0, actions=[play_proc])],
        )
    )

    shutdown_after_play = RegisterEventHandler(
        OnProcessExit(
            target_action=play_proc,
            on_exit=[
                TimerAction(
                    period=3.0,
                    actions=[EmitEvent(event=Shutdown(reason="playback complete"))],
                )
            ],
        )
    )

    return [
        container,
        start_record,
        start_detection,
        start_playback,
        shutdown_after_play,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "source_bag",
                description="Path to the camera bag",
            ),
            DeclareLaunchArgument(
                "output_bag",
                default_value="",
                description="Output bag directory (default: <source_bag>_detections)",
            ),
            DeclareLaunchArgument(
                "runner_model",
                default_value="RunnerSegYoloV8l.engine",
                description="TensorRT engine name under the detection package's "
                "models directory",
            ),
            DeclareLaunchArgument(
                "start_delay",
                default_value="5.0",
                description="Seconds to wait before starting detection",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
