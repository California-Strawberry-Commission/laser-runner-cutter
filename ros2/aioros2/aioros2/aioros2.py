import asyncio
from typing import Optional

import rclpy
import rclpy.node

from .server_driver import ServerDriver

# pip install -e laser-runner-cutter/ros2/aioros2/ --config-settings editable_mode=strict

# https://stackoverflow.com/questions/338101/python-function-attributes-uses-and-abuses
# https://robotics.stackexchange.com/questions/106026/ros2-multi-nodes-each-on-a-thread-in-same-process


async def _ros_spin_nodes(
    nodes,
    num_threads: Optional[int] = None,
):
    print("Ros starting up...")

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=num_threads)

    servers = [ServerDriver(n) for n in nodes]

    for node in servers:
        executor.add_node(node)

    print("Ros event loop running!")
    while rclpy.ok():
        executor.spin_once()
        await asyncio.sleep(0)


def serve_nodes(*nodes, num_threads: Optional[int] = None):
    from .decorators import deferrable_accessor

    # Notify deferrables that load has fully completed
    deferrable_accessor.deferrables_frozen = True

    rclpy.init()

    asyncio.run(_ros_spin_nodes(nodes, num_threads=num_threads))
