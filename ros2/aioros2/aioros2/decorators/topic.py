from typing import Any
from ._decorators import RosDefinition

class RosTopic(RosDefinition):
    def __init__(self, namespace, msg_idl, qos) -> None:
        self.path = namespace
        self.idl = msg_idl
        self.qos = qos
        self.node = None


def topic(namespace: str, idl: Any, queue: int = 10):
    return RosTopic(namespace, idl, queue)
    