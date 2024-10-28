import asyncio
import functools
import os
import platform
from dataclasses import dataclass

from ament_index_python.packages import get_package_share_directory
from std_srvs.srv import Trigger

from aioros2 import (
    QOS_LATCHED,
    node,
    params,
    result,
    serve_nodes,
    service,
    start,
    topic,
)
from laser_control.laser_dac import EtherDreamDAC, HeliosDAC
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


@node("laser_control_node")
class LaserControlNode:
    laser_control_params = params(LaserControlParams)
    state_topic = topic("~/state", State, qos=QOS_LATCHED)

    @start
    async def start(self):
        include_dir = os.path.join(
            get_package_share_directory("laser_control"),
            "include",
            platform.machine(),
        )
        self._dac = None
        if self.laser_control_params.dac_type == "helios":
            self._dac = HeliosDAC(
                os.path.join(include_dir, "libHeliosDacAPI.so"),
                logger=self.get_logger(),
            )
        elif self.laser_control_params.dac_type == "ether_dream":
            self._dac = EtherDreamDAC(
                os.path.join(include_dir, "libEtherDream.so"), logger=self.get_logger()
            )
        else:
            raise Exception(f"Unknown dac_type: {self.laser_control_params.dac_type}")
        self._connecting = False

        # Publish initial state
        self._publish_state()

    @service("~/start_device", Trigger)
    async def start_device(self):
        if self._dac is None:
            return result(success=False)

        self._connecting = True
        self._publish_state()

        await asyncio.get_running_loop().run_in_executor(None, self._dac.initialize)
        await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(self._dac.connect, self.laser_control_params.dac_index),
        )

        self._connecting = False
        self._publish_state()

        return result(success=True)

    @service("~/close_device", Trigger)
    async def close_device(self):
        if self._dac is None:
            return result(success=False)

        await asyncio.get_running_loop().run_in_executor(None, self._dac.close)
        self._publish_state()
        return result(success=True)

    @service("~/set_color", SetColor)
    async def set_color(self, r, g, b, i):
        if self._dac is None:
            return result(success=False)

        self._dac.set_color(r, g, b, i)
        return result(success=True)

    @service("~/add_point", AddPoint)
    async def add_point(self, point):
        if self._dac is None:
            return result(success=False)

        self._dac.add_point(point.x, point.y)
        return result(success=True)

    @service("~/set_points", SetPoints)
    async def set_points(self, points):
        if self._dac is None:
            return result(success=False)

        self._dac.clear_points()
        for point in points:
            self._dac.add_point(point.x, point.y)
        return result(success=True)

    @service("~/remove_point", Trigger)
    async def remove_point(self):
        if self._dac is None:
            return result(success=False)

        self._dac.remove_point()
        return result(success=True)

    @service("~/clear_points", Trigger)
    async def clear_points(self):
        if self._dac is None:
            return result(success=False)

        self._dac.clear_points()
        return result(success=True)

    @service("~/set_playback_params", SetPlaybackParams)
    async def set_playback_params(self, fps, pps, transition_duration_ms):
        await self.laser_control_params.set(
            fps=fps,
            pps=pps,
            transition_duration_ms=transition_duration_ms,
        )
        return result(success=True)

    @service("~/play", Trigger)
    async def play(self):
        if self._dac is None:
            return result(success=False)

        self._dac.play(
            self.laser_control_params.fps,
            self.laser_control_params.pps,
            self.laser_control_params.transition_duration_ms,
        )
        self._publish_state()
        return result(success=True)

    @service("~/stop", Trigger)
    async def stop(self):
        if self._dac is None:
            return result(success=False)

        self._dac.stop()
        self._publish_state()
        return result(success=True)

    @service("~/get_state", GetState)
    async def get_state(self):
        return result(state=self._get_state())

    def _get_device_state(self) -> DeviceState:
        if self._connecting:
            return DeviceState.CONNECTING
        elif self._dac is None or not self._dac.is_connected:
            return DeviceState.DISCONNECTED
        elif self._dac.playing:
            return DeviceState.PLAYING
        else:
            return DeviceState.STOPPED

    def _get_state(self) -> State:
        return State(device_state=self._get_device_state())

    def _publish_state(self):
        state = self._get_state()
        asyncio.create_task(self.state_topic(device_state=state.device_state))


def main():
    serve_nodes(LaserControlNode())


if __name__ == "__main__":
    main()
