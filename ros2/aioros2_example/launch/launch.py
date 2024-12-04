import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription

from aioros2 import LaunchNode
from aioros2_example import another_node, circular_node, main_node


def generate_launch_description():
    parameters_file = os.path.join(
        get_package_share_directory("aioros2_example"),
        "config",
        "parameters.yaml",
    )

    main_node_launch_node = LaunchNode(
        main_node,
        name="main0",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
    )
    another_node_launch_node = LaunchNode(
        another_node,
        name="another0",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
    )
    circular_node_launch_node = LaunchNode(
        circular_node,
        name="circular0",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
    )

    # Link node dependencies
    main_node_launch_node.another_node.link(another_node_launch_node)
    main_node_launch_node.circular_node.link(circular_node_launch_node)
    circular_node_launch_node.main_node.link(main_node_launch_node)

    return LaunchDescription(
        [
            main_node_launch_node,
            another_node_launch_node,
            circular_node_launch_node,
        ]
    )  # type: ignore
