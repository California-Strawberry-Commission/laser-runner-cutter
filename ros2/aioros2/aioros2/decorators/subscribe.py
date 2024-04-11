import inspect
from typing import Any, Union
from .self import Self
from ._decorators import RosDefinition
from .topic import RosTopic

class RosSubscription(RosDefinition):
    def raw_topic(namespace, idl, qos, server_handler):
        return RosSubscription(RosTopic(namespace, idl, qos), server_handler)

    def __init__(self, topic: RosTopic, server_handler):
        self.topic = topic
        self.server_handler = server_handler

    def get_topic(self, class_self) -> RosTopic:
        if isinstance(self.topic, Self):
            return self.topic.resolve(class_self)
        
        return self.topic



def subscribe(topic: Union[Self, str], idl: Union[Any, None] = None, qos=10):
    def _subscribe(fn):
        if isinstance(topic, Self):
            return RosSubscription(topic, fn)
        else:
            return RosSubscription.raw_topic(topic, idl, qos, fn)

    return _subscribe
