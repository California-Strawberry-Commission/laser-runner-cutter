import inspect
from typing import Any, Union
from ._decorators import decorate_handler, ros_type_e
from .self import Self


def _decorate_str_topic(fn, topic: str, idl: Any):
    if idl is None:
        raise RuntimeError("An IDL must be passed for string-based subscriptions")
    
    return decorate_handler(fn, ros_type_e.TOPIC_SUBSCRIBER, idl=idl, namespace=topic)


def _decorate_self_topic(fn, topic):
    # TODO: remap namespace to reference node name.
    return decorate_handler(fn, ros_type_e.TOPIC_SUBSCRIBER, idl=topic._ros_idl, namespace=topic._ros_namespace)


def subscribe(topic: Union[Self, str], idl: Union[Any, None] = None):
    def _subscribe(fn):
        if isinstance(topic, Self):
            return _decorate_self_topic(fn, topic)
        else:
            return _decorate_str_topic(fn, topic, idl)

    return _subscribe
