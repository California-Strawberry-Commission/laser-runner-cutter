from enum import Enum
from typing import Callable, Any, AnyStr, TypeVar

class ros_type_e(Enum):
    NODE="node"
    SERVICE="service"
    ACTION="action"
    TIMER="timer"
    PARAM="param"
    TOPIC_SUBSCRIBER="topic_subscriber"
    TOPIC="topic"

class handler_t(Callable):
    _ros_type: ros_type_e
    _ros_namespace: AnyStr
    _ros_idl: Any
    _ros_qos: int


def decorate_handler(handler: Callable, type: ros_type_e, namespace=None, idl=None, qos=10) -> handler_t:
    handler._ros_type = type
    handler._ros_namespace = namespace
    handler._ros_idl = idl
    handler._ros_qos = qos
    
    return handler

T = TypeVar("T")

def decorate_node(node: T, param_class) -> T:
    class _node(node):
        node._ros_type = ros_type_e.NODE
        node._param_class = param_class
        
    return _node