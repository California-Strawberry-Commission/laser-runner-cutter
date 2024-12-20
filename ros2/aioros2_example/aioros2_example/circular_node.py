import asyncio

from std_msgs.msg import String

from aioros2 import QOS_LATCHED, import_node, node, serve_nodes, subscribe, timer, topic

from . import main_node


@node("circular_node")
class CircularNode:
    my_topic = topic("~/my_topic", String, QOS_LATCHED)

    main_node = import_node(main_node)

    @timer(3.0)
    async def timer(self):
        asyncio.create_task(self.my_topic(data=f"Hello from CircularNode"))

    @subscribe(main_node.my_topic)
    async def on_main_node_my_topic(self, data):
        print(f"message from main_node.my_topic received: {data}")


def main():
    serve_nodes(CircularNode())


if __name__ == "__main__":
    main()
