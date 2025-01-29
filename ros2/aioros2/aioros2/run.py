import asyncio
import inspect
import signal
import threading
from types import ModuleType
from typing import List, Optional

import rclpy
from rclpy.executors import Executor, MultiThreadedExecutor
from rclpy.node import Node

from aioros2.directives.directive import NodeInfo
from aioros2.exception import AioRos2Exception
from aioros2.util import get_module_ros_directives, snake_to_camel_case

# pip install -e laser-runner-cutter/ros2/aioros2/ --config-settings editable_mode=strict

# https://stackoverflow.com/questions/338101/python-function-attributes-uses-and-abuses
# https://robotics.stackexchange.com/questions/106026/ros2-multi-nodes-each-on-a-thread-in-same-process


def run(num_threads: Optional[int] = None):
    """
    Starts the ROS node contained within the calling file.
    Should only be called from within if __name__ == "__main__".
    """
    # Access caller module dict to find directives
    module_name = inspect.getmodule(inspect.stack()[1].frame).__name__.split(".").pop()
    module_dict = inspect.stack()[1].frame.f_globals

    directives = get_module_ros_directives(module_dict)

    if len(directives) <= 0:
        raise AioRos2Exception(
            f"Initialized module {module_dict.__name__} does not have any ROS directives!"
        )

    rclpy.init()

    loop = asyncio.get_event_loop()
    node = Node(snake_to_camel_case(module_name))

    name = node.get_name()
    namespace = node.get_namespace()

    for directive in directives:
        directive.server_impl(node, NodeInfo(namespace, name), loop)

    loop.create_task(_spin([node], num_threads))

    loop.run_forever()

    rclpy.shutdown()


async def _spin(nodes: List[Node], num_threads: Optional[int] = None):
    # From https://github.com/mavlink/MAVSDK-Python/issues/419

    executor = MultiThreadedExecutor(num_threads=num_threads)
    for node in nodes:
        executor.add_node(node)

    cancels = [node.create_guard_condition(lambda: None) for node in nodes]

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

    async def shutdown(event_loop: asyncio.AbstractEventLoop):
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        event_loop.stop()

    # Add signal handlers for SIGINT and SIGTERM
    event_loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        event_loop.add_signal_handler(
            sig, lambda: asyncio.create_task(shutdown(event_loop))
        )

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

    for idx, node in enumerate(nodes):
        node.destroy_guard_condition(cancels[idx])


def get_module_ros_imports(d):
    """
    Returns all found imports which are aioros nodes.

    Any Python file which contains exported aioros directives is treated
    as an aioros node.
    """
    if isinstance(d, ModuleType):
        d = d.__dict__

    return [
        d[k]
        for k in d
        if not k.startswith("__")
        and isinstance(d[k], ModuleType)
        and len(get_module_ros_directives(d[k])) > 0
    ]
