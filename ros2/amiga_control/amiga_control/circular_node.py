from . import amiga_control_node
from aioros2 import node, param, timer, service, action, serve_nodes, result, feedback, subscribe, topic, import_node, self
from std_msgs.msg import String

@node()
class CircularNode:
    a_topic = topic("/atopic", String, 10)
    dependant_node_1 = import_node(lambda: amiga_control_node.AmigaControlNode("node_name1"))

    # dependant_node_1 = import_node(amiga_control_node, "node_name")
    
    @subscribe(self.dependant_node_1.my_topic)
    async def on_global(self, data):
        print("/global/topic", data)

    @subscribe("/global/topic", String)
    async def on_global2(self, data):
        print("/global/topic", data)
 


def main():
    serve_nodes(CircularNode())

if __name__ == "__main__":
    main()