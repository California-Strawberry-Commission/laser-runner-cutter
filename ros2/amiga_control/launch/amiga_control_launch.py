from launch import LaunchDescription
from aioros2 import LaunchNode

from amiga_control import amiga_control_node, circular_node


def generate_launch_description():
    # First, define all nodes. Name and namespace should be specified!
    control_node: amiga_control_node.AmigaControlNode = LaunchNode(
        amiga_control_node, 
        name="acn", 
        namespace="/ns1"
    )

    circ_node: circular_node.CircularNode = LaunchNode(
        circular_node,
        name="circ",
        namespace="/ns2"
    )

    # Define relations between nodes. Every import on every node should have `link` call
    control_node.dependant_node_1.link(circ_node)
    
    circ_node.dependant_node_1.link(control_node)
    
    return LaunchDescription([
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
    ])