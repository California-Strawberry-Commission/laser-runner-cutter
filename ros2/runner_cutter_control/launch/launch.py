import launch
import os
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    parameters_file = os.path.join(
        get_package_share_directory("runner_cutter_control"),
        "config",
        "parameters.yaml",
    )

    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="camera_control",
                executable="camera_control_node",
                name="camera0",
                arguments=["--ros-args", "--log-level", "info"],
                parameters=[parameters_file],
            ),
            launch_ros.actions.Node(
                package="laser_control",
                executable="laser_control_node",
                name="laser0",
                arguments=["--ros-args", "--log-level", "info"],
                parameters=[parameters_file],
            ),
            launch_ros.actions.Node(
                package="runner_cutter_control",
                executable="runner_cutter_control_node",
                name="control0",
                arguments=["--ros-args", "--log-level", "info"],
                parameters=[parameters_file],
            ),
        ]
    )
