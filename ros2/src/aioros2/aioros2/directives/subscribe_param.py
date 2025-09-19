import asyncio
from inspect import iscoroutinefunction, isfunction
from typing import Any

from rclpy.node import Node

from aioros2.deferrable import Deferrable
from aioros2.directives.directive import NodeInfo, RosDirective
from aioros2.exception import AioRos2Exception


class RosParamSubscription(RosDirective):
    def __init__(
        self,
        fn,
        param_deferrable: Deferrable,
    ):
        if not iscoroutinefunction(fn):
            raise TypeError("Subscription handlers must be async.")

        self._fn = fn
        self._param_deferrable = param_deferrable

        self._client_mode = False

    async def __call__(self, *args: any, **kwargs: any) -> any:
        if self._client_mode:
            raise AioRos2Exception("Cannot call another node's subscription function.")

        return await self._fn(*args, **kwargs)

    def server_impl(self, node: Node, nodeinfo: NodeInfo, loop: asyncio.BaseEventLoop):
        # TODO
        # Get the RosParam and the field name
        # param = self._param_deferrable.resolve()
        # Create a listener that calls self._fn and add it to RosParams
        return

    def client_impl(self, node: Node, nodeinfo: NodeInfo, loop: asyncio.BaseEventLoop):
        self._client_mode = True
        return


def subscribe_param(param: Any):
    """
    A function decorator for a function that will be run whenever any of the specified params change.

    Args:
        param: Param field for which changes will trigger the function call.
    Raises:
        TypeError: If the decorated object is not a function.
    """

    param_deferrable = Deferrable(param, frame=2)

    def _subscribe_param(fn) -> RosParamSubscription:
        if not isfunction(fn):
            raise TypeError("This decorator can only be applied to functions.")
        return RosParamSubscription(fn, param_deferrable)

    return _subscribe_param
