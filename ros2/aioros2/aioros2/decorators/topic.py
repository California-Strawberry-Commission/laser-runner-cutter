from typing import Any, Union

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile

# QoS profile for a latched topic (the last message published is saved and sent to any late subscribers)
QOS_LATCHED = QoSProfile(
    depth=1,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class RosTopic:
    def __init__(self, name: str, idl: Any, qos: Union[QoSProfile, int]):
        self.path = name
        self.idl = idl
        self.qos: QoSProfile = qos
        self.node = None


def topic(name: str, idl: Any, qos: Union[QoSProfile, int] = 10) -> RosTopic:
    """
    Defines a ROS 2 topic that can be published to by the node.

    Args:
        name (str): Topic name. Relative and private names are accepted and will be resolved appropriately.
        idl (Any): ROS 2 message type associated with the topic.
        qos (Union[QoSProfile, int]): Quality of Service policy profile, or an int representing the queue depth.
    """
    return RosTopic(name, idl, qos)
