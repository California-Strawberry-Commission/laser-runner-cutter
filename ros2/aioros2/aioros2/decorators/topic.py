from typing import Any, Union
from ._decorators import RosDefinition
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    qos_profile_system_default,
)

QOS_LATCHED = QoSProfile(
    depth=1,
    history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
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

def latched_topic(namespace: str, idl: Any):
    return RosTopic(namespace, idl, qos=QOS_LATCHED)

