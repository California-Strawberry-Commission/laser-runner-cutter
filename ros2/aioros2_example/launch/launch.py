import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription

import aioros2_example.another_node as another_node
import aioros2_example.circular_node as circular_node
import aioros2_example.main_node as main_node
from aioros2.launch import launch


def generate_launch_description():
    parameters_file = os.path.join(
        get_package_share_directory("aioros2_example"),
        "config",
        "parameters.yaml",
    )

    main_node_launch_node = launch(
        main_node,
        name="main0",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
    )
    another_node_launch_node = launch(
        another_node,
        name="another0",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
    )
    yet_another_node_launch_node = launch(
        another_node,
        name="another1",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
    )
    circular_node_launch_node = launch(
        circular_node,
        name="circular0",
        parameters=[parameters_file],
        output="screen",
        emulate_tty=True,
    )

    # Link node dependencies
    main_node_launch_node.another_node_ref = another_node_launch_node
    main_node_launch_node.yet_another_node_ref = yet_another_node_launch_node
    main_node_launch_node.circular_node_ref = circular_node_launch_node
    circular_node_launch_node.main_node_ref = main_node_launch_node

    return LaunchDescription(
        [
            main_node_launch_node,
            another_node_launch_node,
            yet_another_node_launch_node,
            circular_node_launch_node,
        ]
    )  # type: ignore
