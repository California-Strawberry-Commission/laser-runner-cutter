import launch
import os
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    parameters_file = os.path.join(
        get_package_share_directory("control_node"),
        "config",
        "parameters.yaml",
    )

    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="camera_control",
                executable="camera_node",
                name="camera_node",
                arguments=["--ros-args", "--log-level", "info"],
                parameters=[parameters_file],
            ),
            launch_ros.actions.Node(
                package="laser_control",
                executable="laser_control_node",
                name="laser_node",
                arguments=["--ros-args", "--log-level", "info"],
                parameters=[parameters_file],
            ),
            launch_ros.actions.Node(
                package="control_node",
                executable="control_node",
                name="control_node",
                arguments=["--ros-args", "--log-level", "info"],
                parameters=[parameters_file],
            ),
        ]
    )
