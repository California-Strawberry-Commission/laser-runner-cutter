import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator
from std_msgs.msg import String
from std_srvs.srv import Trigger

from aioros2 import (
    QOS_LATCHED,
    action,
    feedback,
    import_node,
    node,
    params,
    result,
    serve_nodes,
    service,
    start,
    subscribe,
    subscribe_param,
    timer,
    topic,
)
from . import another_node, circular_node


# NOTE: dataclass REQUIRES type annotations to work
# The annotation should be the underlying type of the parameter.
# Only ros parameter types are supported.
# Look at `aioros2.async_driver.dataclass_ros_enum_map` for a full list of python annotations that work here
@dataclass
class SomeParams:
    my_string: str = "Default"

    # TODO: support parameterdescriptors
    # read_me: str = param("A test value", description="test", read_only=True)


@dataclass
class SomeOtherParams:
    my_string: str = "Default"


# Node definition, with the name of the executable (as defined in entry_points in setup.py)
# that will get called when the module containing this node is referred to from a launch file.
@node("main_node")
class MainNode:
    # Parameter definitions. Multiple instances of same dataclass are supported.
    some_params = params(SomeParams)
    some_other_params = params(SomeOtherParams)

    # Defines a topic accessible internally (to publish to) and externally (to subscribe to).
    my_topic = topic("~/my_topic", String, QOS_LATCHED)

    # Defines dependencies.
    another_node = import_node(another_node)
    circular_node = import_node(circular_node)

    # Defines a function that will run immediately on node start.
    @start
    async def start(self):
        print("start called")
        print(f"    Param some_params.my_string: {self.some_params.my_string}")
        print(
            f"    Param some_other_params.my_string: {self.some_other_params.my_string}"
        )

    # Defines a function that will run at regular intervals.
    @timer(1.0, allow_concurrent_execution=False)
    async def timer(self):
        print("timer called")
        # Publish to topics by calling the topic directly.
        asyncio.create_task(self.my_topic(data="Hello from MainNode"))

    # Defines a function that will run as a result of a ROS 2 service call.
    @service("~/my_service", Trigger)
    async def my_service(self):
        print("my_service called")
        # Services must return a result()
        return result(success=True)

    # Defines a function that will run when a message is received on a topic.
    @subscribe(another_node.my_topic)
    async def on_another_node_my_topic(self, data):
        print(f"message from another_node.my_topic received: {data}")

    @subscribe(circular_node.my_topic)
    async def on_circular_node_my_topic(self, data):
        print(f"message from circular_node.my_topic received: {data}")

    # Defines a function that will run whenever the specified parameters change.
    @subscribe_param(some_params.my_string, some_other_params.my_string)
    async def on_param_change(self):
        print("on_param_change called")
        print(f"    Param some_params.my_string: {self.some_params.my_string}")
        print(
            f"    Param some_other_params.my_string: {self.some_other_params.my_string}"
        )

    """
    @action("~/my_action", Run)
    async def my_action(self, fast) -> AsyncGenerator[int, None]:
        for i in range(10):
            yield feedback(progress=float(i))
            await asyncio.sleep(0.1 if fast else 1)
        # LAST YIELD MUST BE RESPONSE!!
        yield result(success=True)
    """


def main():
    serve_nodes(MainNode())


if __name__ == "__main__":
    main()
