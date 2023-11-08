import launch
import os
import launch_ros.actions


def generate_launch_description():
    config = os.path.join(
        "config",
        "base_configs.yaml",
    )

    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="camera_control",
                executable="camera_node",
                name="camera_node",
                arguments=["--ros-args", "--log-level", "info"],
                parameters=[config],
            ),
            launch_ros.actions.Node(
                package="laser_control",
                executable="laser_control_node",
                name="laser_node",
                arguments=["--ros-args", "--log-level", "info"],
                parameters=[config],
            ),
            launch_ros.actions.Node(
                package="control_node",
                executable="control_node",
                name="control_node",
                arguments=["--ros-args", "--log-level", "info"],
                parameters=[config],
            ),
        ]
    )
