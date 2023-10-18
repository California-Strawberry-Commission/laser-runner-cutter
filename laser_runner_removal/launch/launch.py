import launch
from ament_index_python.packages import get_package_share_directory
import os
import launch_ros.actions


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("laser_runner_removal"),
        "config",
        "base_configs.yaml",
    )

    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="laser_runner_removal",
                executable="realsense",
                name="realsense",
                arguments=["--ros-args", "--log-level", "info"],
            ),
            launch_ros.actions.Node(
                package="laser_runner_removal",
                executable="main_node",
                name="main_node",
                arguments=["--ros-args", "--log-level", "info"],
                parameters=[config],
            ),
        ]
    )
