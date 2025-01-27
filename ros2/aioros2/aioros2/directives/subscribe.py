import asyncio
import functools
from inspect import iscoroutinefunction, isfunction
from typing import Any, Optional, Union

from rclpy.node import Node
from rclpy.qos import QoSProfile

from aioros2.deferrable import Deferrable
from aioros2.directives.directive import NodeInfo, RosDirective
from aioros2.directives.topic import RosTopic
from aioros2.exception import AioRos2Exception
from aioros2.util import catch, idl_to_kwargs


class RosSubscription(RosDirective):
    def __init__(
        self,
        fn,
        topic: Deferrable,
        idl: Optional[Any] = None,
        qos: Union[QoSProfile, int] = 10,
    ):
        if not iscoroutinefunction(fn):
            raise TypeError("Subscription handlers must be async.")

        self._topic = topic
        self._idl = idl
        self._qos = qos
        self._fn = fn

        self._client_mode = False

    async def __call__(self, *args: any, **kwargs: any) -> any:
        if self._client_mode:
            raise AioRos2Exception("Cannot call another node's subscription function.")

        return await self._fn(*args, **kwargs)

    def server_impl(self, node: Node, nodeinfo: NodeInfo, loop: asyncio.BaseEventLoop):
        topic = self._topic.resolve()
        idl = self._idl
        qos = self._qos

        # Get IDL, topic, and qos either explicitly or from external topic
        if type(topic) == str:
            if not idl:
                raise AioRos2Exception(
                    "An IDL must be provided for string-based subscriptions"
                )
        elif isinstance(topic, RosTopic):
            idl = topic.idl
            qos = topic.qos
            topic = topic.get_fully_qualified_path()
        else:
            raise TypeError("Not a topic or string")

        @catch(node.get_logger().log)
        def callback(data):
            kwargs = idl_to_kwargs(data)

            if iscoroutinefunction(self._fn):
                loop.create_task(self._fn(node, **kwargs))
            else:
                loop.run_in_executor(None, functools.partial(self._fn, node, **kwargs))

        node.create_subscription(idl, topic, callback, qos)

    def client_impl(self, node: Node, nodeinfo: NodeInfo, loop: asyncio.BaseEventLoop):
        self._client_mode = True
        return


def subscribe(
    topic: Union[RosTopic, str],
    idl: Optional[Any] = None,
    qos: Union[QoSProfile, int] = 10,
):
    """
    A function decorator for a function that will be run when a message is received on the
    specified topic. If a reference to another node's topic is provided, its IDL and QoS profiles
    are used. If a string topic name is provided, an IDL and QoS profile associated with the
    topic must be provided.

    Args:
        topic (Union[RosTopic, str]): Either a reference to a topic, an imported node's topic, or a string topic name.
        idl (Optional[Any]): ROS 2 message type associated with the topic. Must be provided if a string topic name is provided.
        qos (Union[QoSProfile, int]): Quality of Service policy profile, or an int representing the queue depth. Must be provided if a string topic name is provided.
    Raises:
        TypeError: If the decorated object is not a function.
    """

    topic = Deferrable(topic)

    def _subscribe(fn) -> RosSubscription:
        if not isfunction(fn):
            raise TypeError("This decorator can only be applied to functions.")
        return RosSubscription(fn, topic, idl, qos)

    return _subscribe
