import inspect
from typing import Any, Optional, Union

from rclpy.expand_topic_name import expand_topic_name
from rclpy.qos import QoSProfile

from .topic import RosTopic


class RosSubscription:
    def raw_topic(namespace, idl, qos, func):
        return RosSubscription(RosTopic(namespace, idl, qos), func)

    def __init__(self, topic: RosTopic, func):
        self.topic = topic
        self.handler = func

    def get_fqt(self) -> RosTopic:
        """Returns a fully-qualified topic name for this topic's path under the passed node."""
        if not self.topic.node:
            if self.topic.path.startswith("/"):
                return RosTopic(self.topic.path, self.topic.idl, self.topic.qos)
            else:
                raise RuntimeError(f"Node for topic >{self.topic.path}< was never set!")

        fully_qual = expand_topic_name(
            self.topic.path, self.topic.node._node_name, self.topic.node._node_namespace
        )
        return RosTopic(fully_qual, self.topic.idl, self.topic.qos)


def subscribe(
    topic: Union[Any, str],
    idl: Optional[Any] = None,
    qos: Optional[Union[QoSProfile, int]] = 10,
):
    """
    A function decorator for a function that will be run when a message is received on the
    specified topic. If a reference to another node's topic is provided, its IDL and QoS profiles
    are used. If a string topic name is provided, an IDL and QoS profile associated with the
    topic must be provided.

    Args:
        topic (Union[Any, str]): Either a reference to another node's topic, or a string topic name.
        idl (Optional[Any]): ROS 2 message type associated with the topic. Must be provided if a string topic name is provided.
        qos (Optional[Union[QoSProfile, int]]): Quality of Service policy profile, or an int representing the queue depth. Must be provided if a string topic name is provided.
    Raises:
        TypeError: If the decorated object is not a function.
    """

    def _subscribe(func) -> RosSubscription:
        if not inspect.isfunction(func):
            raise TypeError("This decorator can only be applied to functions.")

        if type(topic) == str:
            # Do arg checks
            if idl is None:
                raise ValueError("An IDL must be provided for a string-based topic")

            return RosSubscription.raw_topic(topic, idl, qos, func)
        else:
            return RosSubscription(topic, func)

    return _subscribe
