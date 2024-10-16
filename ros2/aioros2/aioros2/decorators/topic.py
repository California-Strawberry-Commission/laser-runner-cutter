from typing import Any, Union
from ._decorators import RosDefinition
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)

QOS_LATCHED = QoSProfile(
    depth=1,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class RosTopic(RosDefinition):
    def __init__(
        self, namespace: str, msg_idl: Any, qos: Union[QoSProfile, int]
    ) -> None:
        self.path = namespace
        self.idl = msg_idl
        self.qos: QoSProfile = qos
        self.node = None


def topic(namespace: str, idl: Any, qos: Union[QoSProfile, int] = 10):
    return RosTopic(namespace, idl, qos)
