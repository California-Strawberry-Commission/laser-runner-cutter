import asyncio
import importlib
from typing import AsyncGenerator, Callable, TypeVar
from amiga_control_interfaces.srv import SetTwist
from amiga_control_interfaces.action import Run
from dataclasses import dataclass
from rcl_interfaces.msg import ParameterDescriptor, ParameterEvent
from aioros2 import timer, service, action, serve_nodes, result, feedback, subscribe, topic, import_node, params, node, subscribe_param, param
from std_msgs.msg import String

from aioros2.decorators.subscribe import RosSubscription

from . import circular_node

# Remapping/linking
# Global topics - don't remap
# NEED - fully qualified node path for parameter, topic linking

# Future note: dataclass requires type annotations to work
@dataclass
class AmigaParams:
    host: str = "127.0.0.1"
    port_canbus: int = 6001

    # ParameterDescriptor(description="test", read_only=True)
    read_me: str = param("A test value", description="test", read_only=True)

@dataclass
class GenericParams:
    idk: str = "test"
    a_test: str = "t"
    # read_me: str = (
    #     "A test value",
    #     ParameterDescriptor(description="test", read_only=True),
    # )

# NOTE: In drivers, definitions must be treated as immutable
@node("amiga_control_node")
class AmigaControlNode:
    amiga_params = params(AmigaParams)
    generic_params = params(GenericParams)

    dependant_node_1: "circular_node.CircularNode" = import_node(circular_node)
    
    my_topic = topic("~/atopic", String, 10)

    @subscribe(dependant_node_1.a_topic)
    def sub_another_topic(self, data):
        print(data)

    @subscribe("/test/topic", String)
    async def test_topic(self, data):
        print(data)
    
    @subscribe(my_topic)
    async def on_my_topic(self, data):
        self.log.info(f"MYTOPIC {data}")

    @subscribe_param(amiga_params.host, amiga_params.port_canbus)
    async def on_change(self):
        print("CHANGE HANDLER", self.amiga_params.host, self.amiga_params.port_canbus)
        # Do something else...

    @timer(2) # Server only
    async def task(self):
        print(self.amiga_params, self.amiga_params.port_canbus)
        # if self.amiga_params.host != "TESTHOST":
        #     print("Param is different, resetting!")
        #     await self.amiga_params.set(host="TESTHOST")

    @service("~/set_twist", SetTwist)
    async def set_twist(self, twist) -> bool:
        await self.dependant_node_1.on_global2(data="test")
        await self.my_topic(data="TEST to my topic dude")

        return result(success=True)

    @subscribe("~/set_host", String, 10)
    async def set_host(self, data):
        await self.amiga_params.set(
            host = data
        )

        print(self.amiga_params.host)
        print("SET PARAM SUCCESSFUL")
        
    @action("~/test", Run)
    async def act(self, fast) -> AsyncGenerator[int, None]:
        for i in range(10):
            yield feedback(progress=float(i))
            await asyncio.sleep(0.1 if fast else 1)
        # LAST YIELD MUST BE RESPONSE!!
        yield result(success=True)
        


def main():
    serve_nodes(AmigaControlNode())

if __name__ == "__main__":
    main()

# CLIENT BELOW HERE
# n = AmigaControlNode().client()

# @n.on("topic")
# async def handler():
#     print("Got message on topic")

# async def main():
#     await n.set_twist({x: 1, y: 1})
    
#     # Problem: async generators don't have return values (...). Important?
#     # There are workarounds but not super pretty.
#     async for progress in n.act({x: 1, y: 1}):
#         print(f"Got progress {progress}")
