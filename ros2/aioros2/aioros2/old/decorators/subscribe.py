import inspect
from abc import ABC, abstractmethod
from inspect import getmembers
from typing import Any, Optional, Union

from rclpy.expand_topic_name import expand_topic_name
from rclpy.qos import QoSProfile

from aioros2.decorators.import_node import RosImport
from aioros2.decorators.topic import RosTopic
from aioros2.lazy_accessor import LazyAccessor


# RosSubscription can be made with name/idl/qos, RosTopic, or a LazyAccessor.
# Subscriptions on a LazyAccessor will create a LazyAccessorRosSubscription, while
# otherwise we will create a TopicRosSubscription. In either case, to get the fully
# qualified (resolved) topic name, we need to know the node name and namespace of
# the node where the topic is defined. For LazyAccessorRosSubscription, resolving
# the topic name is quite complex because we need to first resolve the accessor.
class RosSubscription(ABC):
    def __init__(self, func):
        self.handler = func

    @abstractmethod
    def get_fully_qualified_topic(self, async_driver) -> RosTopic:
        pass


class TopicRosSubscription(RosSubscription):
    def __init__(self, topic: RosTopic, func):
        super().__init__(func)
        self._topic = topic

    def get_fully_qualified_topic(self, async_driver) -> RosTopic:
        fqn = expand_topic_name(
            self._topic.path, async_driver.node_name, async_driver.node_namespace
        )
        return RosTopic(fqn, self._topic.idl, self._topic.qos)


class LazyAccessorRosSubscription(RosSubscription):
    def __init__(self, topic_accessor: LazyAccessor, func):
        super().__init__(func)
        self._topic_accessor = topic_accessor

    def get_fully_qualified_topic(self, async_driver) -> str:
        # The root accessor should always be a RosImport
        ros_import = self._topic_accessor.root_accessor
        if not isinstance(ros_import, RosImport):
            raise TypeError(
                "Subscription on an imported node's topic is not referenced properly."
            )

        # Search the driver's node def for the attr that has the reference to this particular RosImport
        node_def = async_driver.node_def
        matches = getmembers(node_def, lambda v: v is ros_import)
        ros_import_attr = matches[0][0] if matches else None
        if ros_import_attr is None:
            raise AttributeError(
                f"Import node attribute could not be found on the node def of {type(node_def)}."
            )

        # Traverse async_driver using the path to get the RosTopic and its AsyncDriver.
        # Traversing the path fully should resolve into a RosTopic (and specifically a CachedPublisher
        # in the case where the import traversal circles back to the ServerDriver), and travering
        # the path up to the penultimate element (if available) should resolve into its AsyncDriver.
        imported_node_driver = getattr(async_driver, ros_import_attr)
        self._topic_accessor.set_target_obj(imported_node_driver)
        ros_topic = self._topic_accessor.resolve()
        if not isinstance(ros_topic, RosTopic):
            raise TypeError(
                "Attempting to subscribe to a reference that is not a topic."
            )
        if len(self._topic_accessor.path) > 1:
            ros_topic_driver = self._topic_accessor.resolve(depth=-1)
        else:
            ros_topic_driver = imported_node_driver

        ros_topic_fqn = ros_topic.get_fully_qualified_name(
            ros_topic_driver.node_name, ros_topic_driver.node_namespace
        )

        return RosTopic(ros_topic_fqn, ros_topic.idl, ros_topic.qos)


def subscribe(
    topic: Union[LazyAccessor, RosTopic, str],
    idl: Optional[Any] = None,
    qos: Union[QoSProfile, int] = 10,
):
    """
    A function decorator for a function that will be run when a message is received on the
    specified topic. If a reference to another node's topic is provided, its IDL and QoS profiles
    are used. If a string topic name is provided, an IDL and QoS profile associated with the
    topic must be provided.

    Args:
        topic (Union[LazyAccessor, RosTopic, str]): Either a reference to a topic, an imported node's topic, or a string topic name.
        idl (Optional[Any]): ROS 2 message type associated with the topic. Must be provided if a string topic name is provided.
        qos (Union[QoSProfile, int]): Quality of Service policy profile, or an int representing the queue depth. Must be provided if a string topic name is provided.
    Raises:
        TypeError: If the decorated object is not a function.
    """

    def _subscribe(func) -> RosSubscription:
        if not inspect.isfunction(func):
            raise TypeError("This decorator can only be applied to functions.")

        if type(topic) == str:
            if idl is None:
                raise ValueError("An IDL must be provided for a string-based topic")

            return TopicRosSubscription(RosTopic(topic, idl, qos), func)
        elif type(topic) == RosTopic:
            return TopicRosSubscription(topic, func)
        else:
            return LazyAccessorRosSubscription(topic, func)

    return _subscribe
