import asyncio
from typing import Any, Optional, Union

from rclpy.expand_topic_name import expand_topic_name
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile
from rclpy.publisher import Publisher

from aioros2.directives.directive import NodeInfo, RosDirective
from aioros2.util import marshal_to_idl

QOS_LATCHED = QoSProfile(
    depth=1,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class RosTopic(RosDirective):
    def __init__(self, path: str, idl: Any, qos: Union[QoSProfile, int]):
        self._path = path
        self._idl = idl
        self._qos = qos

        self._loop: Optional[asyncio.BaseEventLoop] = None
        self._node: Optional[Node] = None
        self._publisher: Optional[Publisher] = None
        self._nodeInfo = NodeInfo(None, None)

    @property
    def idl(self) -> Any:
        return self._idl

    @property
    def qos(self) -> Union[QoSProfile, int]:
        return self._qos

    def publish(self, *args, **kwargs):
        """Publishes to this topic without blocking."""
        self._loop.create_task(self.publish_and_wait(*args, **kwargs))

    async def publish_and_wait(self, *args, **kwargs):
        """Allows caller to wait"""
        publisher = self._get_publisher()
        idl = marshal_to_idl(self._idl, *args, **kwargs)
        await self._loop.run_in_executor(None, publisher.publish, idl)

    def get_fully_qualified_path(self) -> str:
        return expand_topic_name(
            self._path, self._nodeInfo.name, self._nodeInfo.namespace
        )

    def server_impl(self, node: Node, nodeinfo, loop: asyncio.BaseEventLoop):
        self._node = node
        self._nodeInfo = nodeinfo
        self._loop = loop
        # For servers, create the publisher up front to ensure the topic is discoverable
        self._get_publisher()

    def client_impl(self, node: Node, nodeinfo, loop: asyncio.BaseEventLoop):
        self._node = node
        self._nodeInfo = nodeinfo
        self._loop = loop

    def _get_publisher(self) -> Publisher:
        if not self._publisher:
            self._publisher = self._node.create_publisher(
                self._idl, self.get_fully_qualified_path(), self._qos
            )

        return self._publisher


def topic(name: str, idl: Any, qos: Union[QoSProfile, int] = 10) -> RosTopic:
    """
    Defines a ROS 2 topic that can be published to by the node.

    Args:
        name (str): Topic name. Relative and private names are accepted and will be resolved appropriately.
        idl (Any): ROS 2 message type associated with the topic.
        qos (Union[QoSProfile, int]): Quality of Service policy profile, or an int representing the queue depth.
    """
    return RosTopic(name, idl, qos)
