import asyncio
import time
from enum import IntEnum
from typing import Awaitable, Optional

from std_srvs.srv import Trigger

import amiga_control.amiga_control_node as amiga_control_node
import furrow_perceiver.furrow_perceiver_node as furrow_perceiver_node
from aioros2 import (
    QOS_LATCHED,
    import_node,
    node,
    serve_nodes,
    service,
    start,
    subscribe,
    timer,
    topic,
)
from common_interfaces.msg import PID, Vector2
from common_interfaces.srv import SetFloat32
from guidance_brain_interfaces.msg import State

FEET_PER_MIN_TO_METERS_PER_SEC = 0.00508
# Arbitrary scaling factor for P
P_SCALING = 1.0 / 5000.0


class GoDirection(IntEnum):
    FORWARD = 0
    BACKWARD = 1


# Last used settings:
# - Speed: 30 ft/min
# - P gain: 50
# - Forward offset: -12


@node("guidance_brain_node")
class GuidanceBrainNode:
    amiga_node = import_node(amiga_control_node)
    furrow_perceiver_forward_node = import_node(furrow_perceiver_node)
    furrow_perceiver_backward_node = import_node(furrow_perceiver_node)

    state_topic = topic("~/state", State, QOS_LATCHED)

    @start
    async def start(self):
        self._state = State(
            guidance_active=False,
            amiga_connected=False,
            speed=20.0,
            follower_pid=PID(p=50.0),
            # Keeps selected (forward/backward) perceiver result
            perceiver_valid=False,
            error=0.0,
            command=0.0,
            go_direction=GoDirection.FORWARD,
            go_last_valid_time=0.0,
        )
        self._current_task: Optional[asyncio.Task] = None

        self._publish_state()

    @subscribe(amiga_node.amiga_available)
    async def on_amiga_available(self, data):
        self._state.amiga_connected = data

    @subscribe(furrow_perceiver_forward_node.tracker_result_topic)
    async def on_fp_forw_result(self, linear_deviation, heading, is_valid):
        if self._state.go_direction == GoDirection.FORWARD:
            self._state.perceiver_valid = is_valid
            self._state.error = linear_deviation

    @subscribe(furrow_perceiver_backward_node.tracker_result_topic)
    async def on_fp_back_result(self, linear_deviation, heading, is_valid):
        if self._state.go_direction == GoDirection.BACKWARD:
            self._state.perceiver_valid = is_valid
            self._state.error = linear_deviation

    @service("~/set_p", SetFloat32)
    async def set_p(self, data: float):
        self._state.follower_pid.p = data
        return {"success": True}

    @service("~/set_i", SetFloat32)
    async def set_i(self, data: float):
        self._state.follower_pid.i = data
        return {"success": True}

    @service("~/set_d", SetFloat32)
    async def set_d(self, data: float):
        self._state.follower_pid.d = data
        return {"success": True}

    @service("~/set_speed", SetFloat32)
    async def set_speed(self, data: float):
        self._state.speed = data
        return {"success": True}

    @service("~/go_forward", Trigger)
    async def go_forward(self):
        success = self._start_task(self._guidance_task(GoDirection.FORWARD))
        return {"success": success}

    @service("~/go_backward", Trigger)
    async def go_backward(self):
        success = self._start_task(self._guidance_task(GoDirection.BACKWARD))
        return {"success": success}

    @service("~/stop", Trigger)
    async def stop(self):
        success = await self._stop_current_task()
        return {"success": success}

    # region Task management

    def _start_task(self, coro: Awaitable, name: Optional[str] = None) -> bool:
        if self._current_task is not None and not self._current_task.done():
            return False

        async def coro_wrapper(coro: Awaitable):
            await self._reset_to_idle()
            await coro

        self._current_task = asyncio.create_task(coro_wrapper(coro), name=name)

        async def done_callback(task: asyncio.Task):
            await self._reset_to_idle()
            self._current_task = None
            self._publish_state()

        def done_callback_wrapper(task: asyncio.Task):
            asyncio.create_task(done_callback(task))

        self._current_task.add_done_callback(done_callback_wrapper)
        self._publish_state()
        return True

    async def _stop_current_task(self) -> bool:
        if self._current_task is None or self._current_task.done():
            return False

        self._current_task.cancel()
        try:
            await self._current_task
        except asyncio.CancelledError:
            pass

        return True

    async def _reset_to_idle(self):
        await self.amiga_node.set_twist(twist=Vector2(x=0.0, y=0.0))

    # endregion

    # region Task definitions

    async def _guidance_task(self, direction: GoDirection):
        self._state.go_last_valid_time = time.time()
        self._state.guidance_active = True
        self._state.go_direction = direction
        try:
            while True:
                print("guidance task loop")
                if self._state.perceiver_valid:
                    self._state.go_last_valid_time = time.time()

                # If more than 1 second has passed since furrow perciever was valid,
                # kill guidance
                if time.time() - self._state.go_last_valid_time > 1.0:
                    break

                # If perceiver is valid, run PID. Otherwise, stop Amiga
                if self._state.perceiver_valid:
                    self._state.command = (
                        self._state.follower_pid.p * self._state.error * P_SCALING
                    )
                    speed_ms = self._state.speed * FEET_PER_MIN_TO_METERS_PER_SEC
                    if self._state.go_direction == GoDirection.BACKWARD:
                        speed_ms = -speed_ms

                    await self.amiga_node.set_twist(
                        twist=Vector2(x=self._state.command, y=speed_ms)
                    )
                else:
                    self._state.command = 0.0
                    await self.amiga_node.set_twist(twist=Vector2(x=0.0, y=0.0))

                self._publish_state()
                await asyncio.sleep(0.05)
        finally:
            self._state.guidance_active = False

    # endregion

    def _publish_state(self):
        asyncio.create_task(self.state_topic(self._state))


def main():
    serve_nodes(GuidanceBrainNode())


if __name__ == "__main__":
    main()
