from aioros2 import (
    QOS_LATCHED,
    node,
    serve_nodes,
    timer,
    topic,
)
from std_msgs.msg import String
import asyncio


@node("another_node")
class AnotherNode:
    my_topic = topic("~/my_topic", String, QOS_LATCHED)

    @timer(2.0)
    async def timer(self):
        asyncio.create_task(self.my_topic(data=f"Hello from AnotherNode"))


def main():
    serve_nodes(AnotherNode())


if __name__ == "__main__":
    main()
