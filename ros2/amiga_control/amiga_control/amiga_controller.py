from farm_ng.canbus.canbus_pb2 import Twist2d
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig


class AmigaController:
    def __init__(self, host, canbus_port) -> None:
        self.host = host
        self.canbus_port = canbus_port

        self.cli_canbus = EventClient(
            config=EventServiceConfig(name="canbus", port=canbus_port, host=host)
        )
    
    async def wait_for_clients(self):
        await self.cli_canbus._try_connect()

    async def set_twist(self, lin_vel, ang_vel):
        
        if await self.cli_canbus._try_connect():
            t = Twist2d(angular_velocity=lin_vel, linear_velocity_x=ang_vel)
            await self.cli_canbus.request_reply("/twist", t)
            return True
        else:
            return False
