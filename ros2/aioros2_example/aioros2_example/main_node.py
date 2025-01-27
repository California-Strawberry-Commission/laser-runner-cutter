from dataclasses import dataclass

from std_msgs.msg import String
from std_srvs.srv import Trigger

import aioros2

from . import another_node


# NOTE: dataclass requires type annotations to work. The annotation should be the underlying type
# of the parameter. Only ROS 2 parameter types are supported.
# See `aioros2.mappings.dataclass_ros_enum_map` for a full list of python annotations that work here
@dataclass
class SomeParams:
    my_string: str = "Default"


@dataclass
class SomeOtherParams:
    my_string: str = "Default"


# Declares params
some_params = aioros2.params(SomeParams)
some_other_params = aioros2.params(SomeOtherParams)
# Declares topics
my_topic = aioros2.topic("~/my_topic", String, aioros2.QOS_LATCHED)
# Declares dependencies
another_node_ref = aioros2.use(another_node)
yet_another_node_ref = aioros2.use(another_node)


# Defines a function that will run immediately on node start.
@aioros2.start
async def start(node):
    print("start called")
    print(f"    some_params.my_string = {some_params.my_string}")
    print(f"    some_other_params.my_string = {some_other_params.my_string}")


# Defines a function that will run at regular intervals.
@aioros2.timer(1.0, allow_concurrent_execution=False)
async def timer(node):
    print("timer called")


# Defines a function that will run as a result of a ROS 2 service call. Make sure the params
# match the input params of the service.
@aioros2.service("~/my_service", Trigger)
async def my_service(node):
    print("my_service called")

    # Publishes to this node's topic (async)
    await my_topic.publish_async(data="Hello from MainNode")

    # Publishes to this node's topic (fire and forget)
    my_topic.publish(data="Hello again from MainNode")

    # Publishes to another node's topic
    another_node_ref.my_topic.publish(
        data="Hello from MainNode via another_node_ref.my_topic"
    )

    # Calls another node's service. Make sure to pass in kwargs that match the input params of the
    # service.
    await another_node_ref.my_service()

    return {"success": True}


# Defines a function that will run when a message is received on a topic.
@aioros2.subscribe(another_node_ref.my_topic)
async def on_another_node_my_topic(node, data):
    print(f"message from another_node_ref.my_topic received: {data}")


@aioros2.subscribe(yet_another_node_ref.my_topic)
async def on_yet_another_node_my_topic(node, data):
    print(f"message from yet_another_node_ref.my_topic received: {data}")


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
    aioros2.run()


if __name__ == "__main__":
    main()
