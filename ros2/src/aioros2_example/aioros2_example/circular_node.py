from std_msgs.msg import String

import aioros2
import aioros2_example.main_node as main_node

my_topic = aioros2.topic("~/my_topic", String, aioros2.QOS_LATCHED)
main_node_ref = aioros2.use(main_node)


@aioros2.timer(3.0)
async def timer(node):
    my_topic.publish(data=f"Hello from CircularNode")


@aioros2.subscribe(main_node_ref.my_topic)
async def on_main_node_my_topic(node, data):
    print(f"Message from main_node.my_topic received: {data}")


def main():
    aioros2.run()


if __name__ == "__main__":
    main()
