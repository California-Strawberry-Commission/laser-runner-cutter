import launch
import launch_ros.actions


def generate_launch_description():
    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="laser_runner_removal", executable="realsense", name="realsense"
            ),
            launch_ros.actions.Node(
                package="laser_runner_removal", executable="main_node", name="main_node"
            ),
        ]
    )
