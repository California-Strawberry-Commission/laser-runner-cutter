from . import amiga_control_node
from aioros2 import param, timer, service, action, serve_nodes, result, feedback, subscribe, topic, import_node, RosNode, params
from std_msgs.msg import String
from dataclasses import dataclass

@dataclass
class CircularParams:
    s: str = "A setting"

class CircularNode(RosNode):
    params = params(CircularParams)
    a_topic = topic("/atopic", String, 10)
    dependant_node_1: "amiga_control_node.AmigaControlNode" = import_node(amiga_control_node)
    
    @subscribe(dependant_node_1.my_topic)
    async def on_global(self, data):
        print("On node topic", data)
        await self.params.set(
            s="test"
        )

    @subscribe("localtopic", String)
    async def on_gdls(self, data):
        print("mklfksdnkf")

    @subscribe("/global/topic", String)
    async def on_global2(self, data):
        print("/global/topic", data)

    @subscribe("~/set_host", String)
    async def set_other(self, data):
        print("Call set param!!")
        await self.dependant_node_1.amiga_params.set(
            host = data
        )
 

def main():
    serve_nodes(CircularNode())

if __name__ == "__main__":
    main()