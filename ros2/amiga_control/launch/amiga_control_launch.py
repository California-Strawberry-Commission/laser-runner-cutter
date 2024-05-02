import os
from launch import LaunchDescription
from aioros2 import LaunchNode

from amiga_control import amiga_control_node, circular_node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get parameter config file
    config = os.path.join(
        get_package_share_directory("amiga_control"), "config", "params.yaml"
    )

    # First, define all nodes. Name and namespace should be specified!
    control_node: amiga_control_node.AmigaControlNode = LaunchNode(
        amiga_control_node,
        name="acn",
        namespace="/ns1",
        # Works
        parameters=[{"amiga_params.host": "test_host"}, config],
    )

    circ_node: circular_node.CircularNode = LaunchNode(
        circular_node, name="circ", namespace="/ns2"
    )

    # Define relations between nodes. Every import on every node should have `link` call
    control_node.dependant_node_1.link(circ_node)

    circ_node.dependant_node_1.link(control_node)

    return LaunchDescription(
        [
            control_node,
            circ_node,
            # Node(
            #     package='turtlesim',
            #     executable='mimic',
            #     name='mimic',
            #     remappings=[
            #         ('/input/pose', '/turtlesim1/turtle1/pose'),
            #         ('/output/cmd_vel', '/turtlesim2/turtle1/cmd_vel'),
            #     ]
            # )
        ]
    )
