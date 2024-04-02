import asyncio
from typing import AsyncGenerator
from amiga_control_interfaces.srv import SetTwist
from amiga_control_interfaces.action import Run
from dataclasses import dataclass
from rcl_interfaces.msg import ParameterDescriptor
from .aioros2 import node, param, timer, service, action, serve_nodes, result, feedback, on


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
    @on("test")
    def on_topic(self, val):
        print(val)
        
    @param(AmigaParams.host)
    async def set_host_param(self, host):
        self.params.host = host
        # Do something else...

    @timer(1) # Server only
    async def task(self):
        await self.publish()

    @service("~/set_twist", SetTwist)
    async def set_twist(self, twist) -> bool:
        print(twist)
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
