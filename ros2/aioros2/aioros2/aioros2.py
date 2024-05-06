import asyncio
import threading
from typing import Optional

import rclpy
from rclpy.executors import Executor, MultiThreadedExecutor
from rclpy.node import Node

from .server_driver import ServerDriver

# pip install -e laser-runner-cutter/ros2/aioros2/ --config-settings editable_mode=strict

# https://stackoverflow.com/questions/338101/python-function-attributes-uses-and-abuses
# https://robotics.stackexchange.com/questions/106026/ros2-multi-nodes-each-on-a-thread-in-same-process


async def _spin(node: Node, executor: Optional[Executor] = None):
    # From https://github.com/mavlink/MAVSDK-Python/issues/419
    cancel = node.create_guard_condition(lambda: None)

    def spin_inner(
        node: Node, future: asyncio.Future, event_loop: asyncio.AbstractEventLoop
    ):
        while not future.cancelled():
            rclpy.spin_once(node, executor=executor)
        if not future.cancelled():
            event_loop.call_soon_threadsafe(future.set_result, None)

    event_loop = asyncio.get_event_loop()
    spin_task = event_loop.create_future()
    spin_thread = threading.Thread(
        target=spin_inner, args=(node, spin_task, event_loop)
    )
    spin_thread.start()
    try:
        await spin_task
    except asyncio.CancelledError:
        cancel.trigger()
    spin_thread.join()
    node.destroy_guard_condition(cancel)


async def _ros_spin_nodes(
    nodes,
    num_threads: Optional[int] = None,
):
    executor = MultiThreadedExecutor(num_threads=num_threads)
    servers = [ServerDriver(node) for node in nodes]
    spin_tasks = [
        asyncio.get_event_loop().create_task(_spin(server, executor))
        for server in servers
    ]
    await asyncio.wait(spin_tasks, return_when=asyncio.FIRST_COMPLETED)

    # Cancel tasks
    for task in spin_tasks:
        if task.cancel():
            await task


def serve_nodes(*nodes, num_threads: Optional[int] = None):
    from .decorators import deferrable_accessor

    # Notify deferrables that load has fully completed
    deferrable_accessor.deferrables_frozen = True

    rclpy.init()

    asyncio.run(_ros_spin_nodes(nodes, num_threads=num_threads))

    rclpy.shutdown()
