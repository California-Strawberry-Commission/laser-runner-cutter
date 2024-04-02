from .amiga_control_node import AmigaControlNode
import asyncio
from common_interfaces.msg import Vector2



async def _main():
    n = AmigaControlNode().client()
    print("Starting amiga control client")
    
    print("Calling service")
    res = await n.set_twist(twist=Vector2(x=1., y=1.))
    print("Service Result", res)
    
    print("Slow action")
    async for progress in (action := n.act(fast=False)):
        print(f"Got progress {progress}")

    print("Fast action")
    async for feedback in (action := n.act(fast=True)):
        print(f"Progress {feedback}")
    print("Result", action.result)
    
def main():
    asyncio.run(_main())

if __name__ == "__main__":
    main()
