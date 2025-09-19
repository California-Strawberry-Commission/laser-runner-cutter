import asyncio
from inspect import iscoroutinefunction, isfunction

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node

from aioros2.directives.directive import NodeInfo, RosDirective
from aioros2.exception import AioRos2Exception
from aioros2.util import catch


class RosTimer(RosDirective):
    """
    Uses a ROS timer to execute the decorated function when the node is run as a server
    """

    def __init__(self, interval_secs: float, allow_concurrent_execution: bool, fn):
        if not iscoroutinefunction(fn):
            raise TypeError("Timer functions must be async.")

        self._fn = fn
        self._interval_secs = interval_secs
        self._allow_concurrent_execution = allow_concurrent_execution

        self._client_mode = False

    async def __call__(self, *args: any, **kwargs: any) -> any:
        if self._client_mode:
            raise AioRos2Exception("Cannot call another node's timer function.")

        return await self._fn(*args, **kwargs)

    def server_impl(self, node: Node, nodeinfo: NodeInfo, loop: asyncio.BaseEventLoop):
        # Use a callback group to allow or prevent concurrent execution of callbacks
        callback_group = (
            ReentrantCallbackGroup()
            if self._allow_concurrent_execution
            else MutuallyExclusiveCallbackGroup()
        )

        @catch(node.get_logger().log)
        def callback():
            # Call handler function. This callback is called from another thread, so we need to
            # use run_coroutine_threadsafe
            asyncio.run_coroutine_threadsafe(self._fn(node), loop).result()

        node.create_timer(self._interval_secs, callback, callback_group=callback_group)

    def client_impl(self, node: Node, nodeinfo: NodeInfo, loop: asyncio.BaseEventLoop):
        self._client_mode = True
        return


def timer(interval_secs: float, allow_concurrent_execution: bool = True):
    """
    A function decorator for functions that will run at regular intervals.

    Args:
        interval_secs (float): Interval between function calls, in seconds.
        allow_concurrent_execution (bool): If false, the next call will occur concurrently even if the previous call has not completed yet. If true, the next call will be skipped if the previous call has not completed yet.
    Raises:
        TypeError: If the decorated object is not a function.
    """

    def _timer(fn) -> RosTimer:
        if not isfunction(fn):
            raise TypeError("This decorator can only be applied to functions.")
        return RosTimer(interval_secs, allow_concurrent_execution, fn)

    return _timer
