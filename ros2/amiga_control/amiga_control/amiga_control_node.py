import asyncio
import importlib
from typing import AsyncGenerator, Callable, TypeVar
from amiga_control_interfaces.srv import SetTwist
from amiga_control_interfaces.action import Run
from dataclasses import dataclass
from rcl_interfaces.msg import ParameterDescriptor
from aioros2 import node, param, timer, service, action, serve_nodes, result, feedback, subscribe, topic, self, import_node
from std_msgs.msg import String

from aioros2.decorators.subscribe import RosSubscription

from . import circular_node

# Future note: dataclass requires type annotations to work
@dataclass
class AmigaParams:
    host: str = "127.0.0.1"
    port_canbus: int = 6001
    # read_me: str = (
    #     "A test value",
    #     ParameterDescriptor(description="test", read_only=True),
    # )


@node(AmigaParams)
class AmigaControlNode:
    dependant_node_1 = import_node(lambda: circular_node.CircularNode("node_name"))
    
    my_topic = topic("~/atopic", String, 10)

    # @imports
    # def process_imports(self):

    # @subscribe(self.my_topic)
    # def own_topic(self, data):
    #     print(data)
        
    @subscribe(self.dependant_node_1.a_topic)
    def sub_another_topic(self, data):
        print(data)

    @subscribe("/test/topic", String)
    async def test_topic(self, data):
        print(data)
        
    @param(AmigaParams.host)
    async def set_host_param(self, host):
        self.params.host = host
        # Do something else...

    @timer(1) # Server only
    async def task(self):
        await self.publish()

    @service("~/set_twist", SetTwist)
    async def set_twist(self, twist) -> bool:
        await self.dependant_node_1.on_global(data="test")

        return result(success=True)

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
