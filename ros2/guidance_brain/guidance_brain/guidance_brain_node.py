import asyncio
import time
from enum import IntEnum

from std_srvs.srv import Trigger

import amiga_control.amiga_control_node as amiga_control_node
import furrow_perceiver.furrow_perceiver_node as furrow_perceiver_node
from aioros2 import (
    QOS_LATCHED,
    import_node,
    node,
    serve_nodes,
    service,
    subscribe,
    timer,
    topic,
)
from common_interfaces.msg import PID, Vector2
from common_interfaces.srv import SetFloat32
from furrow_perceiver_interfaces.msg import PositionResult
from guidance_brain_interfaces.msg import State


FEET_PER_MIN_TO_METERS_PER_SEC = 0.00508
# Arbitrary scaling factor for P
P_SCALING = 1.0 / 5000.0


class GoDirection(IntEnum):
    FORWARD = 0
    BACKWARD = 1


# Executable to call to launch this node (defined in `setup.py`)
@node("guidance_brain_node")
class GuidanceBrainNode:
    perceiver_forward = import_node(furrow_perceiver_node)
    # TODO: aioros2 currently has a bug when importing nodes twice
    # perceiver_backward = import_node(furrow_perceiver_node)
    amiga = import_node(amiga_control_node)

    # Emits state.
    state_topic = topic("~/state", State, QOS_LATCHED)

    # State Vars
    state = State(
        guidance_active=False,
        amiga_connected=False,
        speed=20.0,  # Default - 10ft/min
        follower_pid=PID(p=50.0),
        # Keeps selected (forward/backward) perceiver result
        perceiver_valid=False,
        error=0.0,
        command=0.0,
        go_direction=GoDirection.FORWARD,
        go_last_valid_time=0.0,
    )

    @timer(0.05, False)
    async def s(self):
        self._publish_state()

        if self.state.perceiver_valid:
            self.state.go_last_valid_time = time.time()

        # 1 second has passed since furrow perciever was valid - kill following
        if self.state.guidance_active and (
            time.time() - self.state.go_last_valid_time > 1
        ):
            self.state.guidance_active = False

        # Short circuit if perceiver isn't valid.
        if not (self.state.guidance_active and self.state.perceiver_valid):
            self.state.command = 0.0
            await self.amiga.set_twist(twist=Vector2(x=0.0, y=0.0))
            return

        # Run PID
        self.state.command = self.state.follower_pid.p * self.state.error * P_SCALING
        speed_ms = self.state.speed * FEET_PER_MIN_TO_METERS_PER_SEC

        if self.state.go_direction == GoDirection.BACKWARD:
            speed_ms = -speed_ms

        await self.amiga.set_twist(twist=Vector2(x=self.state.command, y=speed_ms))

    @subscribe(amiga.amiga_available)
    async def on_amiga_available(self, data):
        self.state.amiga_connected = data

    # TODO: aioros2 currently has a bug when importing nodes twice, so we hardcode the topic
    # path here
    @subscribe("/furrow1/tracker_result", PositionResult)
    async def on_fp_back_result(self, linear_deviation, heading, is_valid):
        if self.state.go_direction == GoDirection.BACKWARD:
            self.state.perceiver_valid = is_valid
            self.state.error = linear_deviation

    @subscribe(perceiver_forward.tracker_result_topic)
    async def on_fp_forw_result(self, linear_deviation, heading, is_valid):
        if self.state.go_direction != GoDirection.BACKWARD:
            self.state.perceiver_valid = is_valid
            self.state.error = linear_deviation

    @service("~/set_p", SetFloat32)
    async def set_p(self, data: float):
        self.state.follower_pid.p = data
        return {"success": True}

    @service("~/set_i", SetFloat32)
    async def set_i(self, data: float):
        self.state.follower_pid.i = data
        return {"success": True}

    @service("~/set_d", SetFloat32)
    async def set_d(self, data: float):
        self.state.follower_pid.d = data
        return {"success": True}

    @service("~/set_speed", SetFloat32)
    async def set_speed(self, data: float):
        self.state.speed = data
        return {"success": True}

    @service("~/go_forward", Trigger)
    async def go_forward(self):
        self.state.go_last_valid_time = time.time()
        self.state.guidance_active = True
        self.state.go_direction = GoDirection.FORWARD

        return {"success": True}

    @service("~/go_backward", Trigger)
    async def go_backward(self):
        self.state.go_last_valid_time = time.time()
        self.state.guidance_active = True
        self.state.go_direction = GoDirection.BACKWARD
        return {"success": True}

    @service("~/stop", Trigger)
    async def stop(self):
        self.state.guidance_active = False
        return {"success": True}

    def _publish_state(self):
        asyncio.create_task(self.state_topic(self.state))


def main():
    serve_nodes(GuidanceBrainNode())


if __name__ == "__main__":
    main()


# 30 ft/min
# 50 pid

# FORWARD
# -12 offset
