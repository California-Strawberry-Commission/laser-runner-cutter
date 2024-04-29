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


    def get_fqt(self, node_name, node_ns) -> RosTopic:
        """Returns a fully-qualified topic name for this topic's path under the passed node."""
        fully_qual = expand_topic_name(self.topic.path, node_name, node_ns)
        return RosTopic(fully_qual, self.topic.idl, self.topic.qos)
    



def subscribe(topic: Union[Any, str], idl: Union[Any, None] = None, qos_queue=10):
    def _subscribe(fn):
        if type(topic) == str:
            return RosSubscription.raw_topic(topic, idl, qos_queue, fn)
        else:
            return RosSubscription(topic, fn)


    return _subscribe
