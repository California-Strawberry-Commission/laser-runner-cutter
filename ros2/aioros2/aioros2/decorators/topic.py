from typing import Any
from ._decorators import RosDefinition
from .deferrable_accessor import DeferrableAccessor

class RosTopic(RosDefinition):
    def __init__(self, namespace, msg_idl, qos) -> None:
        self.namespace = namespace
        self.idl = msg_idl
        self.qos = qos


def topic(namespace: str, idl: Any, qos: int):
    return RosTopic(namespace, idl, qos)
    