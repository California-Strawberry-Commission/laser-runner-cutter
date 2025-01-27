import asyncio
import functools
import logging
import os
import platform
from dataclasses import dataclass
from typing import Optional

from ament_index_python.packages import get_package_share_directory
from std_srvs.srv import Trigger

import aioros2
from laser_control.laser_dac import EtherDreamDAC, HeliosDAC, LaserDAC
from laser_control_interfaces.msg import DeviceState, State
from laser_control_interfaces.srv import (
    AddPoint,
    GetState,
    SetColor,
    SetPlaybackParams,
    SetPoints,
)


@dataclass
class LaserControlParams:
    dac_type: str = "helios"  # "helios" or "ether_dream"
    dac_index: int = 0
    fps: int = 30
    pps: int = 30000
    transition_duration_ms: float = 0.5


laser_control_params = aioros2.params(LaserControlParams)
state_topic = aioros2.topic("~/state", State, qos=aioros2.QOS_LATCHED)


class SharedState:
    logger: Optional[logging.Logger] = None
    dac: Optional[LaserDAC] = None
    connecting = False


shared_state = SharedState()


@aioros2.start
async def start(node):
    shared_state.logger = node.get_logger()
    include_dir = os.path.join(
        get_package_share_directory("laser_control"),
        "include",
        platform.machine(),
    )
    if laser_control_params.dac_type == "helios":
        shared_state.dac = HeliosDAC(
            os.path.join(include_dir, "libHeliosDacAPI.so"),
            logger=shared_state.logger,
        )
    elif laser_control_params.dac_type == "ether_dream":
        shared_state.dac = EtherDreamDAC(
            os.path.join(include_dir, "libEtherDream.so"), logger=shared_state.logger
        )
    else:
        raise Exception(f"Unknown dac_type: {laser_control_params.dac_type}")
    shared_state.connecting = False

    # Publish initial state
    _publish_state()


@aioros2.service("~/start_device", Trigger)
async def start_device(node):
    if shared_state.dac is None:
        return {"success": False}

    shared_state.connecting = True
    _publish_state()

    await asyncio.get_running_loop().run_in_executor(None, shared_state.dac.initialize)
    await asyncio.get_running_loop().run_in_executor(
        None,
        functools.partial(shared_state.dac.connect, laser_control_params.dac_index),
    )

    shared_state.connecting = False
    _publish_state()

    return {"success": True}


@aioros2.service("~/close_device", Trigger)
async def close_device(node):
    if shared_state.dac is None:
        return {"success": False}

    await asyncio.get_running_loop().run_in_executor(None, shared_state.dac.close)
    _publish_state()
    return {"success": True}


@aioros2.service("~/set_color", SetColor)
async def set_color(node, r, g, b, i):
    if shared_state.dac is None:
        return {"success": False}

    shared_state.dac.set_color(r, g, b, i)
    return {"success": True}


@aioros2.service("~/add_point", AddPoint)
async def add_point(node, point):
    if shared_state.dac is None:
        return {"success": False}

    shared_state.dac.add_point(point.x, point.y)
    return {"success": True}


@aioros2.service("~/set_points", SetPoints)
async def set_points(node, points):
    if shared_state.dac is None:
        return {"success": False}

    shared_state.dac.clear_points()
    for point in points:
        shared_state.dac.add_point(point.x, point.y)
    return {"success": True}


@aioros2.service("~/remove_point", Trigger)
async def remove_point(node):
    if shared_state.dac is None:
        return {"success": False}

    shared_state.dac.remove_point()
    return {"success": True}


@aioros2.service("~/clear_points", Trigger)
async def clear_points(node):
    if shared_state.dac is None:
        return {"success": False}

    shared_state.dac.clear_points()
    return {"success": True}


@aioros2.service("~/set_playback_params", SetPlaybackParams)
async def set_playback_params(node, fps, pps, transition_duration_ms):
    await laser_control_params.set(
        fps=fps,
        pps=pps,
        transition_duration_ms=transition_duration_ms,
    )
    return {"success": True}


@aioros2.service("~/play", Trigger)
async def play(node):
    if shared_state.dac is None:
        return {"success": False}

    shared_state.dac.play(
        laser_control_params.fps,
        laser_control_params.pps,
        laser_control_params.transition_duration_ms,
    )
    _publish_state()
    return {"success": True}


@aioros2.service("~/stop", Trigger)
async def stop(node):
    if shared_state.dac is None:
        return {"success": False}

    shared_state.dac.stop()
    _publish_state()
    return {"success": True}


@aioros2.service("~/get_state", GetState)
async def get_state(node):
    return {"state": _get_state()}


def _get_device_state() -> DeviceState:
    if shared_state.connecting:
        return DeviceState.CONNECTING
    elif shared_state.dac is None or not shared_state.dac.is_connected:
        return DeviceState.DISCONNECTED
    elif shared_state.dac.playing:
        return DeviceState.PLAYING
    else:
        return DeviceState.STOPPED


def _get_state() -> State:
    return State(device_state=_get_device_state())


def _publish_state():
    state_topic.publish(_get_state())


def main():
    aioros2.run()


if __name__ == "__main__":
    main()
