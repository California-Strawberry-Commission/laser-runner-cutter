from typing import Any, Union
from ._decorators import RosDefinition
from .topic import RosTopic
from rclpy.expand_topic_name import expand_topic_name


class RosSubscription(RosDefinition):
    def raw_topic(namespace, idl, qos_queue, server_handler):
        return RosSubscription(RosTopic(namespace, idl, qos_queue), server_handler)

    def __init__(self, topic: RosTopic, server_handler):
        self.topic = topic
        self.handler = server_handler

    def get_fqt(self) -> RosTopic:
        """Returns a fully-qualified topic name for this topic's path under the passed node."""
        if not self.topic.node:
            if self.topic.path.startswith("/"):
                return RosTopic(self.topic.path, self.topic.idl, self.topic.qos)
            else:
                raise RuntimeError(f"Node for topic >{self.topic.path}< was never set!a")
        
        fully_qual = expand_topic_name(self.topic.path, self.topic.node._node_name, self.topic.node._node_namespace)
        return RosTopic(fully_qual, self.topic.idl, self.topic.qos)


def subscribe(topic: Union[Any, str], idl: Union[Any, None] = None, qos_queue=10):
    def _subscribe(fn):
        if type(topic) == str:
            # Do arg checks
            if idl is None:
                raise ValueError("An IDL must be provided for a string-based topic")
            
            return RosSubscription.raw_topic(topic, idl, qos_queue, fn)
        else:
            return RosSubscription(topic, fn)

    return _subscribe
