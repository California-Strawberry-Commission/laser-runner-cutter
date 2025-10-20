from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    return LaunchDescription(
        [
            ComposableNodeContainer(
                name="detection_container",
                namespace="",
                package="rclcpp_components",
                executable="component_container_mt",
                output="screen",
                emulate_tty=True,
                composable_node_descriptions=[
                    ComposableNode(
                        package="detection",
                        plugin="DetectionNode",
                        name="detection_node",
                        extra_arguments=[{"use_intra_process_comms": True}],
                    ),
                ],
            )
        ]
    )
