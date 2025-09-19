from abc import ABC, abstractmethod
from asyncio import BaseEventLoop
from collections import namedtuple

from rclpy.node import Node

NodeInfo = namedtuple("NodeInfo", ["namespace", "name"])


class RosDirective(ABC):
    """
    Common base class for all aioros2 decorations. Used to discover aioros2
    implementations when instantiating drivers
    """

    @abstractmethod
    def server_impl(
        self,
        node: Node,
        nodeinfo: NodeInfo,
        loop: BaseEventLoop,
    ):
        """
        Called when loading an aioros2 node definition as a server.
        """
        pass

    @abstractmethod
    def client_impl(
        self,
        node: Node,
        nodeinfo: NodeInfo,
        loop: BaseEventLoop,
    ):
        """
        Called when loading a aioros2 node definition as a client via a `use` directive
        """
        pass
