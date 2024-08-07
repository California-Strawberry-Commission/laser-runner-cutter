import asyncio
from typing import AsyncGenerator
from amiga_control_interfaces.srv import SetTwist
from amiga_control_interfaces.action import Run
from dataclasses import dataclass
from aioros2 import (
    timer,
    service,
    action,
    serve_nodes,
    subscribe,
    topic,
    import_node,
    params,
    node,
    subscribe_param,
    param,
    start,
    QOS_LATCHED,
)
from farm_ng.canbus.canbus_pb2 import Twist2d
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig

from std_msgs.msg import Bool


# Some test commands
# For a launch file example, look to ../launch/amiga_control_launch

# ros2 launch amiga_control/launch_amiga_control_launch.py

# ros2 run amiga_control amiga_control_node --ros-args --remap __node:=acn --remap __ns:=/ns1 -p "dependant_node_1.name:=circ" -p "dependant_node_1.ns:=/ns2" --log-level DEBUG
# ros2 run amiga_control circular_node --ros-args --remap __node:=circ --remap __ns:=/ns2 -p "dependant_node_1.name:=acn" -p "dependant_node_1.ns:=/ns1" --log-level DEBUG

# ros2 action send_goal /ns1/acn/test amiga_control_interfaces/action/Run "{fast: 1}"

# ros2 topic pub /ns1/acn/set_host std_msgs/msg/String "{data: 'hello'}"

# ros2 param set /ns1/acn amiga_params.host "127.0.0.10"
# ros2 param get /ns1/acn amiga_params.host

# ros2 service call /ns1/acn/set_twist amiga_control_interfaces/srv/SetTwist

# ros2 topic pub /ns1/acn/set_host std_msgs/msg/String "{data: 'test'}"


#########################


@dataclass
class AmigaParams:
    amiga_host: str = "10.95.76.1"

    canbus_service_port: int = 6001


# Executable to call to launch this node (defined in `setup.py`)
@node("amiga_control_node")
class AmigaControlNode:
    amiga = None

    amiga_params = params(AmigaParams)

    amiga_available = topic("~/available", Bool, QOS_LATCHED)

    @start
    async def comm_amiga(self):
        await self.amiga_available(data=False)
        
        self.cli_canbus = EventClient(
            config=EventServiceConfig(
                name="canbus",
                port=self.amiga_params.canbus_service_port,
                host=self.amiga_params.amiga_host,
            )
        )

        while not await self.cli_canbus._try_connect():
            print("No amiga connection...")
            asyncio.sleep(1)

        self.log("Got amiga connection!")

        await self.amiga_available(data=True)

    # ros2 service call /set_twist amiga_control_interfaces/srv/SetTwist "{twist: {x: 1.0, y: 1.0}}"
    @service("~/set_twist", SetTwist)
    async def set_twist(self, twist) -> bool:
        if self.amiga_available.value.data:
            t = Twist2d(angular_velocity=twist.x, linear_velocity_x=twist.y)
            await self.cli_canbus.request_reply("/twist", t)
            return {"success": True}

        return {"success": False}


# Boilerplate below here.
def main():
    serve_nodes(AmigaControlNode())


if __name__ == "__main__":
    main()
