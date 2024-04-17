from launch import LaunchDescription
from launch_ros.actions import Node

from amiga_control import amiga_control_node, circular_node

class AIOLaunchNode(Node):
    def __init__(package, name, namespace):
        pass

def generate_launch_description():
    control_node: amiga_control_node.AmigaControlNode = AIOLaunchNode(
        amiga_control_node, 
        name="acn", 
        namespace="ns1"
    )
    
    circ_node: circular_node.CircularNode = AIOLaunchNode(
        circular_node,
        name="circ",
        namespace="ns2"
    )

    # Define relations
    control_node.dependant_node_1.link(circ_node)
    
    circ_node.dependant_node_1.link(control_node)
    
    return LaunchDescription([
        control_node,
        Node(
            package='amiga_control',
            executable='circular_node',
            namespace='ns2',
            name='circnode'
        ),
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