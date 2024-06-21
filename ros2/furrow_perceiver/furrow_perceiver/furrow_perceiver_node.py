import asyncio
from typing import AsyncGenerator
from amiga_control_interfaces.srv import SetTwist
from amiga_control_interfaces.action import Run
from dataclasses import dataclass
from aioros2 import (
    timer,
    service,
    action,
    serve_nodes,
    result,
    feedback,
    subscribe,
    topic,
    import_node,
    params,
    node,
    subscribe_param,
    param,
    start,
)
from std_msgs.msg import String
from common_interfaces.msg import Vector2

@dataclass
class PerceiverNodeParams:
    depth_topic: str = "camera/image"

# Executable to call to launch this node (defined in `setup.py`)
@node("amiga_control_node")
class FurrowPerceiverNode:
    amiga_params = params(PerceiverNodeParams)
    
    # TODO: Allow these annotations using parameters
    # @subscribe(amiga_params.depth_topic)
    # @subscribe(amiga_params.rs_name + "/image")


# Boilerplate below here.
def main():
    serve_nodes(FurrowPerceiverNode())

if __name__ == "__main__":
    main()