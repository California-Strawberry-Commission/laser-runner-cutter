import asyncio
import logging
import time
from enum import IntEnum
from typing import Awaitable, Optional

from std_srvs.srv import Trigger

import aioros2
import amiga_control.amiga_control_node as amiga_control_node
import furrow_perceiver.furrow_perceiver_node as furrow_perceiver_node
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


amiga_node = aioros2.use(amiga_control_node)
furrow_perceiver_forward_node = aioros2.use(furrow_perceiver_node)
furrow_perceiver_backward_node = aioros2.use(furrow_perceiver_node)
state_topic = aioros2.topic("~/state", State, aioros2.QOS_LATCHED)


class SharedState:
    logger: Optional[logging.Logger] = None
    current_task: Optional[asyncio.Task] = None
    guidance_active = False
    amiga_connected = False
    speed = 20.0
    follower_pid = PID(p=50.0)
    perceiver_valid = False
    error = 0.0
    command = 0.0
    go_direction = GoDirection.FORWARD
    go_last_valid_time = 0.0


shared_state = SharedState()


@aioros2.start
async def start(node):
    shared_state.logger = node.get_logger()
    _publish_state()


@aioros2.subscribe(amiga_node.amiga_available_topic)
async def on_amiga_available(node, data):
    shared_state.amiga_connected = data


@aioros2.subscribe(furrow_perceiver_forward_node.tracker_result_topic)
async def on_fp_forw_result(node, linear_deviation, heading, is_valid):
    if shared_state.go_direction == GoDirection.FORWARD:
        shared_state.perceiver_valid = is_valid
        shared_state.error = linear_deviation


@aioros2.subscribe(furrow_perceiver_backward_node.tracker_result_topic)
async def on_fp_back_result(node, linear_deviation, heading, is_valid):
    if shared_state.go_direction == GoDirection.BACKWARD:
        shared_state.perceiver_valid = is_valid
        shared_state.error = linear_deviation


@aioros2.service("~/set_p", SetFloat32)
async def set_p(node, data: float):
    shared_state.follower_pid.p = data
    return {"success": True}


@aioros2.service("~/set_i", SetFloat32)
async def set_i(node, data: float):
    shared_state.follower_pid.i = data
    return {"success": True}


@aioros2.service("~/set_d", SetFloat32)
async def set_d(node, data: float):
    shared_state.follower_pid.d = data
    return {"success": True}


@aioros2.service("~/set_speed", SetFloat32)
async def set_speed(node, data: float):
    shared_state.speed = data
    return {"success": True}


@aioros2.service("~/go_forward", Trigger)
async def go_forward(node):
    success = _start_task(_guidance_task(GoDirection.FORWARD))
    return {"success": success}


@aioros2.service("~/go_backward", Trigger)
async def go_backward(node):
    success = _start_task(_guidance_task(GoDirection.BACKWARD))
    return {"success": success}


@aioros2.service("~/stop", Trigger)
async def stop(node):
    success = await _stop_current_task()
    return {"success": success}


# region Task management


def _start_task(coro: Awaitable, name: Optional[str] = None) -> bool:
    if shared_state.current_task is not None and not shared_state.current_task.done():
        return False

    async def coro_wrapper(coro: Awaitable):
        await _reset_to_idle()
        await coro

    shared_state.current_task = asyncio.create_task(coro_wrapper(coro), name=name)

    async def done_callback(task: asyncio.Task):
        await _reset_to_idle()
        shared_state.current_task = None
        _publish_state()

    def done_callback_wrapper(task: asyncio.Task):
        asyncio.create_task(done_callback(task))

    shared_state.current_task.add_done_callback(done_callback_wrapper)
    _publish_state()
    return True


async def _stop_current_task() -> bool:
    if shared_state.current_task is None or shared_state.current_task.done():
        return False

    shared_state.current_task.cancel()
    try:
        await shared_state.current_task
    except asyncio.CancelledError:
        pass

    return True


async def _reset_to_idle():
    await amiga_node.set_twist(twist=Vector2(x=0.0, y=0.0))


# endregion

# region Task definitions


async def _guidance_task(direction: GoDirection):
    shared_state.go_last_valid_time = time.time()
    shared_state.guidance_active = True
    shared_state.go_direction = direction
    try:
        while True:
            if shared_state.perceiver_valid:
                shared_state.go_last_valid_time = time.time()

            # If more than 1 second has passed since furrow perciever was valid,
            # kill guidance
            if time.time() - shared_state.go_last_valid_time > 1.0:
                break

            # If perceiver is valid, run PID. Otherwise, stop Amiga
            if shared_state.perceiver_valid:
                shared_state.command = (
                    shared_state.follower_pid.p * shared_state.error * P_SCALING
                )
                speed_ms = shared_state.speed * FEET_PER_MIN_TO_METERS_PER_SEC
                if shared_state.go_direction == GoDirection.BACKWARD:
                    speed_ms = -speed_ms

                await amiga_node.set_twist(
                    twist=Vector2(x=shared_state.command, y=speed_ms)
                )
            else:
                shared_state.command = 0.0
                await amiga_node.set_twist(twist=Vector2(x=0.0, y=0.0))

            _publish_state()
            await asyncio.sleep(0.05)
    finally:
        shared_state.guidance_active = False


# endregion


def _get_state() -> State:
    return State(
        guidance_active=shared_state.guidance_active,
        amiga_connected=shared_state.amiga_connected,
        speed=shared_state.speed,
        follower_pid=shared_state.follower_pid,
        perceiver_valid=shared_state.perceiver_valid,
        error=shared_state.error,
        command=shared_state.command,
        go_direction=shared_state.go_direction,
        go_last_valid_time=shared_state.go_last_valid_time,
    )


def _publish_state():
    state_topic.publish(_get_state())


def main():
    aioros2.run()


if __name__ == "__main__":
    main()
