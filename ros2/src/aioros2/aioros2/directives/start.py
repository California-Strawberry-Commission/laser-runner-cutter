import asyncio
from inspect import iscoroutinefunction, isfunction

from rclpy.node import Node

from aioros2.directives.directive import NodeInfo, RosDirective
from aioros2.exception import AioRos2Exception


class RosStart(RosDirective):
    """
    Starts the decorated function when the node is run as a server
    """

    def __init__(self, fn):
        if not iscoroutinefunction(fn):
            raise TypeError("Start functions must be async.")

        self._fn = fn
        self._client_mode = False

    async def __call__(self, *args: any, **kwargs: any) -> any:
        if self._client_mode:
            raise AioRos2Exception("Cannot call another node's start function.")

        return await self._fn(*args, **kwargs)

    def server_impl(self, node: Node, nodeinfo: NodeInfo, loop: asyncio.BaseEventLoop):
        loop.create_task(self._fn(node))

    def client_impl(self, node: Node, nodeinfo: NodeInfo, loop: asyncio.BaseEventLoop):
        self._client_mode = True
        return


def start(fn) -> RosStart:
    """
    A function decorator for functions that will run immediately on node start.

    Raises:
        TypeError: If the decorated object is not a function.
    """

    if not isfunction(fn):
        raise TypeError("This decorator can only be applied to functions.")

    return RosStart(fn)
