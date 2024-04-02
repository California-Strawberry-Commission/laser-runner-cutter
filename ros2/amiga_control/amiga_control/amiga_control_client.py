from .amiga_control_node import AmigaControlNode
import asyncio
from common_interfaces.msg import Vector2



async def _main():
    n = AmigaControlNode().client()
    print("Starting amiga control client")
    res = await n.set_twist(twist=Vector2(x=1., y=1.))
    print("Result", res)

def main():
    asyncio.run(_main())

if __name__ == "__main__":
    main()
