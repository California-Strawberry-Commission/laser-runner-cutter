import asyncio
import threading
from typing import Optional

import rclpy
import rclpy.node

from .server_driver import ServerDriver

# pip install -e laser-runner-cutter/ros2/aioros2/ --config-settings editable_mode=strict

# https://stackoverflow.com/questions/338101/python-function-attributes-uses-and-abuses
# https://robotics.stackexchange.com/questions/106026/ros2-multi-nodes-each-on-a-thread-in-same-process
# https://github.com/mavlink/MAVSDK-Python/issues/419#issuecomment-1383903908


async def _ros_spin_nodes(
    nodes, 
    num_threads: Optional[int] = None, 
    monitor_performance: bool = False
):
    print("Ros starting up...")

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=num_threads)

    servers = [ServerDriver(n, monitor_performance) for n in nodes]

    for node in servers:
        executor.add_node(node)

    # Derived from https://github.com/mavlink/MAVSDK-Python/issues/419#issuecomment-1383903908
    def _spin(event_loop: asyncio.AbstractEventLoop):
        try:
            while event_loop.is_running():
                executor.spin_once()
        except Exception:
            return

    event_loop = asyncio.get_running_loop()
    spin_thread = threading.Thread(target=_spin, args=(event_loop,), daemon=True)

    spin_thread.start()
    print("Ros event loop running!")
    spin_thread.join()


def serve_nodes(*nodes, num_threads: Optional[int] = None, monitor_performance: bool = False):
    from .decorators import deferrable_accessor

    # Notify deferrables that load has fully completed
    deferrable_accessor.deferrables_frozen = True

    rclpy.init()

    asyncio.run(_ros_spin_nodes(
        nodes, 
        num_threads=num_threads, 
        monitor_performance=monitor_performance
    ))
