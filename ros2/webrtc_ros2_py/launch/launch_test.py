from launch import LaunchDescription
from launch_ros.actions import Node

from aioros2.launch import launch
from webrtc_ros2_py import webrtc_node


def generate_launch_description():

    '''
    test_video_frame_publisher_launch_node = Node(
        package="webrtc_ros2_py",
        executable="test_video_frame_publisher_node",
        name="test_video_publisher",
        output="screen",
        emulate_tty=True,
    )
    '''
    webrtc_launch_node = launch(
        webrtc_node,
        name="webrtc",
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            #test_video_frame_publisher_launch_node,
            webrtc_launch_node,
        ]
    )
