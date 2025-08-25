from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="gstreamer_webrtc",
                executable="gstreamer_node",
                name="gstreamer_ros_webrtc_node",
                output="screen",
                emulate_tty=True,
            )
        ]
    )
