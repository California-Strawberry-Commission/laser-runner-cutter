import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from aioros2.launch import launch
from amiga_control import amiga_control_node
from furrow_perceiver import furrow_perceiver_node
from guidance_brain import guidance_brain_node


def generate_launch_description():
    parameters_file = os.path.join(
        get_package_share_directory("guidance_brain"),
        "config",
        "parameters.yaml",
    )

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
    )

    furrow_perceiver_forward_launch_node = launch(
        furrow_perceiver_node,
        name="furrow_perceiver_forward",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
    )

    furrow_perceiver_backward_launch_node = launch(
        furrow_perceiver_node,
        name="furrow_perceiver_backward",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
    )

    amiga_launch_node = launch(
        amiga_control_node,
        name="amiga",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
    )

    guidance_brain_launch_node = launch(
        guidance_brain_node,
        name="guidance_brain",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
    )

    # Link node dependencies
    guidance_brain_launch_node.furrow_perceiver_forward_node = (
        furrow_perceiver_forward_launch_node
    )
    guidance_brain_launch_node.furrow_perceiver_backward_node = (
        furrow_perceiver_backward_launch_node
    )
    guidance_brain_launch_node.amiga_node = amiga_launch_node

    return LaunchDescription(
        [
            realsense_forward_launch_node,
            realsense_backward_launch_node,
            furrow_perceiver_forward_launch_node,
            furrow_perceiver_backward_launch_node,
            amiga_launch_node,
            guidance_brain_launch_node,
        ]
    )  # type: ignore
