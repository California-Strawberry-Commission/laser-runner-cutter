import asyncio
import threading
from typing import List, Optional

import rclpy
from rclpy.executors import Executor, MultiThreadedExecutor
from rclpy.node import Node

from .server_driver import ServerDriver

# pip install -e laser-runner-cutter/ros2/aioros2/ --config-settings editable_mode=strict

# https://stackoverflow.com/questions/338101/python-function-attributes-uses-and-abuses
# https://robotics.stackexchange.com/questions/106026/ros2-multi-nodes-each-on-a-thread-in-same-process


async def _spin(nodes, num_threads: Optional[int] = None):
    # From https://github.com/mavlink/MAVSDK-Python/issues/419

    servers = [ServerDriver(n) for n in nodes]
    executor = MultiThreadedExecutor(num_threads=num_threads)
    for node in servers:
        executor.add_node(node)

    cancels = [node.create_guard_condition(lambda: None) for node in servers]

    def spin_inner(
        executor: Executor,
        future: asyncio.Future,
        event_loop: asyncio.AbstractEventLoop,
    ):
        while not future.cancelled():
            executor.spin_once()
        if not future.cancelled():
            event_loop.call_soon_threadsafe(future.set_result, None)

    event_loop = asyncio.get_event_loop()
    spin_task = event_loop.create_future()
    spin_thread = threading.Thread(
        target=spin_inner, args=(executor, spin_task, event_loop)
    )
    spin_thread.start()
    try:
        await spin_task
    except asyncio.CancelledError:
        for cancel in cancels:
            cancel.trigger()
    spin_thread.join()

    for idx, node in enumerate(servers):
        node.destroy_guard_condition(cancels[idx])


def serve_nodes(*nodes, num_threads: Optional[int] = None):
    from .decorators import deferrable_accessor

    # Notify deferrables that load has fully completed
    deferrable_accessor.deferrables_frozen = True

    rclpy.init()

    asyncio.run(_spin(nodes, num_threads=num_threads))

    rclpy.shutdown()
