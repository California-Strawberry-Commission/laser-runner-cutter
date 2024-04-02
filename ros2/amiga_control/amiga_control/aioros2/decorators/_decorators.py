from enum import Enum
from typing import Callable, Any, AnyStr, TypeVar

class ros_type_e(Enum):
    NODE="node"
    SERVICE="service"
    ACTION="action"
    TIMER="timer"
    PARAM="param"
    TOPIC_SUBSCRIBER="topic_subscriber"

class handler_t(Callable):
    _ros_type: ros_type_e
    _ros_namespace: AnyStr
    _ros_idl: Any



def decorate_handler(handler: Callable, ros_type: ros_type_e, ros_namespace=None, ros_idl=None) -> handler_t:
    handler._ros_type = ros_type
    handler._ros_namespace = ros_namespace
    handler._ros_idl = ros_idl
    
    return handler

T = TypeVar("T")

def decorate_node(node: T, param_class) -> T:
    class _node(node):
        node._ros_type = ros_type_e.NODE
        node._param_class = param_class
        
    return _node