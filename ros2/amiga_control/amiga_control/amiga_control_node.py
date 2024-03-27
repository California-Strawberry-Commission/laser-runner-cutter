import asyncio
from typing import AsyncGenerator
from amiga_control_interfaces.srv import SetTwist
from amiga_control_interfaces.action import Run
from dataclasses import dataclass
from .ros_asyncio import AsyncNode, action, serve_nodes, service, timer, param, ros_params, rosnode
from rcl_interfaces.msg import ParameterDescriptor


# Future note: dataclass requires type annotations to work
@dataclass
class AmigaParams:
    host: str = "127.0.0.1"
    port_canbus: int = 6001
    # read_me: str = (
    #     "A test value",
    #     ParameterDescriptor(description="test", read_only=True),
    # )

@rosnode(AmigaParams)
class AmigaControlNode(AsyncNode):
    @param(AmigaParams.host)
    async def set_host_param(self, host):
        self.params.host = host
        # Do something else...

    @timer(1) # Server only
    async def task(self):
        await self.publish()

    @service("~/set_twist", SetTwist)
    async def set_twist(self, respond, twist) -> bool:
        print(twist)
        return respond(success=True)

    @action("~/test", Run)
    async def act(self, respond, feedback, fast) -> AsyncGenerator[int, None]:
        for i in range(10):
            yield feedback(progress=i)
            await asyncio.sleep(1)
        
        # LAST YIELD MUST BE RESPONSE!!
        yield respond(success=True)
        


def main():
    print(dir(Run))
    serve_nodes(a := AmigaControlNode())

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
