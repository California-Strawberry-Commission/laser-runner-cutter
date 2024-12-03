import asyncio

from std_msgs.msg import String

from aioros2 import QOS_LATCHED, import_node, node, serve_nodes, timer, topic

from . import main_node


@node("circular_node")
class CircularNode:
    my_topic = topic("~/my_topic", String, QOS_LATCHED)

    main_node = import_node(main_node)

    @timer(3.0)
    async def timer(self):
        asyncio.create_task(self.my_topic(data=f"Hello from CircularNode"))


def main():
    serve_nodes(CircularNode())


if __name__ == "__main__":
    main()
