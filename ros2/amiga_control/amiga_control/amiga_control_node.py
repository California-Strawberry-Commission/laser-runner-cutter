import asyncio
from typing import AsyncGenerator
from amiga_control_interfaces.srv import SetTwist
from amiga_control_interfaces.action import Run
from dataclasses import dataclass
from aioros2 import timer, service, action, serve_nodes, result, feedback, subscribe, topic, import_node, params, node, subscribe_param, param
from std_msgs.msg import String

from aioros2.decorators.subscribe import RosSubscription

from . import circular_node

# Some test commands
# For a launch file example, look to ../launch/amiga_control_launch

# ros2 run amiga_control amiga_control_node --ros-args --remap __node:=acn --remap __ns:=/ns1 -p "dependant_node_1.name:=circ" -p "dependant_node_1.ns:=/ns2" --log-level DEBUG
# ros2 run amiga_control circular_node --ros-args --remap __node:=circ --remap __ns:=/ns2 -p "dependant_node_1.name:=acn" -p "dependant_node_1.ns:=/ns1" --log-level DEBUG

# ros2 action send_goal /ns1/acn/test amiga_control_interfaces/action/Run "{fast: 1}"

# ros2 topic pub /ns1/acn/set_host std_msgs/msg/String "{data: 'hello'}"

# ros2 param set /ns1/acn amiga_params.host "127.0.0.10"
# ros2 param get /ns1/acn amiga_params.host

# ros2 service call /ns1/acn/set_twist amiga_control_interfaces/srv/SetTwist

# ros2 topic pub /ns1/acn/set_host std_msgs/msg/String "{data: 'test'}"


# NOTE: dataclass REQUIRES type annotations to work
# The annotation should be the underlying type of the parameter.
# Only ros parameter types are supported. 
# Look at `aioros2.async_driver.dataclass_ros_enum_map` for a full list of python annotations that work here
@dataclass
class AmigaParams:
    host: str = "127.0.0.1"
    port_canbus: int = 6001

    # ParameterDescriptor(description="test", read_only=True)
    # TODO: support parameterdescriptors
    read_me: str = param("A test value", description="test", read_only=True)

@dataclass
class GenericParams:
    generic_one: str = "test1"
    a_test: str = "test2"

# Executable to call to launch this node (defined in `setup.py`)
@node("amiga_control_node")
class AmigaControlNode:
    # Parameter definitions. Multiple instances of same dataclass are supported.
    # Actual parameter name is `attr.dataclass_attr`
    # IE `host` in AmigaParams is fully qualified as `amiga_params.host` 
    amiga_params = params(AmigaParams)
    generic_params = params(GenericParams)

    # IMPORTANT: avoid circular imports here
    # If typing for intellisense, use quotations around the type
    # Import the module itself, NOT the specific node class
    dependant_node_1: "circular_node.CircularNode" = import_node(circular_node)
    
    # Defines a topic accessible internally and externally.
    my_topic = topic("~/atopic", String, 10)

    # Called whenever something is published to the specified topic
    @subscribe(my_topic)
    async def on_my_topic(self, data):
        self.log(f"MYTOPIC {data}")

    # Raw topics can be subscribed by passing a path and IDL
    @subscribe("/test/topic", String)
    async def test_topic(self, data):
        print(data)
    
    # Can subscribe to topics within other nodes. 
    # NOTE: MORE THAN ONE IMPORT LEVEL IS BROKEN FOR NON-GLOBAL TOPICS!
    # A 2-level import will link to the wrong topic. 
    @subscribe(dependant_node_1.a_topic)
    def sub_another_topic(self, data):
        print(data)

    # This function is called whenever the specified parameters change.
    @subscribe_param(amiga_params.host, amiga_params.port_canbus)
    async def on_change(self):
        print("CHANGE HANDLER: ", self.amiga_params.host, self.amiga_params.port_canbus)

    # Runs every x seconds.
    @timer(2) 
    async def task(self):
        print(self.amiga_params, self.amiga_params.port_canbus)
        self.print_host()

    # Service implementation.
    @service("~/set_twist", SetTwist)
    async def set_twist(self, twist) -> bool:

        # Directly call subscribed functions of imported nodes
        await self.dependant_node_1.on_global2(data="test")

        # Publish to own topics by directly calling the topic as you would an idl.Request
        await self.my_topic(data="Test to own topic")

        # raise Exception("Exception!")

        # Services must return result()
        return result(success=True)

    @subscribe("~/set_host", String, 10)
    async def set_host(self, data):
        # Sets own parameter. Valid keys are any that are in
        # the dataclass
        await self.amiga_params.set(
            host = data
        )

        # New value should be available after awaiting
        print("New host name: ", self.amiga_params.host)
        
    @action("~/test", Run)
    async def act(self, fast) -> AsyncGenerator[int, None]:
        for i in range(10):
            yield feedback(progress=float(i)) 
            await asyncio.sleep(0.1 if fast else 1)
        # LAST YIELD MUST BE RESPONSE!!
        yield result(success=True)


    a_value = 10
    def print_host(self):
        print(f"Host param is {self.amiga_params.host}")
        print(f"A value is {self.a_value}")
        self.a_static_method()

    @staticmethod
    def a_static_method():
        print("Called static method")

# Boilerplate below here.
def main():
    serve_nodes(AmigaControlNode())

if __name__ == "__main__":
    main()