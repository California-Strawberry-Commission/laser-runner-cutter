import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from farm_ng.canbus.canbus_pb2 import Twist2d
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from std_msgs.msg import Bool

import aioros2
from amiga_control_interfaces.srv import SetTwist


@dataclass
class AmigaControlParams:
    amiga_host: str = "10.95.76.1"
    canbus_service_port: int = 6001


amiga_control_params = aioros2.params(AmigaControlParams)
amiga_available_topic = aioros2.topic("~/available", Bool, aioros2.QOS_LATCHED)


class SharedState:
    logger: Optional[logging.Logger] = None
    cli_canbus: Optional[EventClient] = None
    amiga_available = False


shared_state = SharedState()


@aioros2.start
async def start(node):
    shared_state.logger = node.get_logger()

    shared_state.amiga_available = False
    await amiga_available_topic.publish_and_wait(data=shared_state.amiga_available)

    shared_state.cli_canbus = EventClient(
        config=EventServiceConfig(
            name="canbus",
            port=amiga_control_params.canbus_service_port,
            host=amiga_control_params.amiga_host,
        )
    )

    while not await shared_state.cli_canbus._try_connect():
        shared_state.logger.warning("No amiga connection...")
        await asyncio.sleep(2)

    shared_state.logger.info("Got amiga connection!")

    shared_state.amiga_available = True
    await amiga_available_topic.publish_and_wait(data=shared_state.amiga_available)


@aioros2.service("~/set_twist", SetTwist)
async def set_twist(node, twist) -> bool:
    if not shared_state.amiga_available:
        return {"success": False}

    twist_msg = Twist2d(angular_velocity=twist.x, linear_velocity_x=twist.y)
    await shared_state.cli_canbus.request_reply("/twist", twist_msg)
    return {"success": True}


def main():
    aioros2.run()


if __name__ == "__main__":
    main()
