import asyncio
import os
from dataclasses import dataclass

from ament_index_python.packages import get_package_share_directory
from std_srvs.srv import Trigger

from aioros2 import node, params, result, serve_nodes, service, start, topic
from laser_control.laser_dac import EtherDreamDAC, HeliosDAC
from laser_control_interfaces.msg import State
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
    state_topic = topic("~/state", State, 5)

    @start
    async def start(self):
        # Initialize DAC
        include_dir = os.path.join(
            get_package_share_directory("laser_control"), "include"
        )
        self.dac = None
        if self.laser_control_params.dac_type == "helios":
            self.dac = HeliosDAC(os.path.join(include_dir, "libHeliosDacAPI.so"))
        elif self.laser_control_params.dac_type == "ether_dream":
            self.dac = EtherDreamDAC(os.path.join(include_dir, "libEtherDream.so"))
        else:
            raise Exception(f"Unknown dac_type: {self.laser_control_params.dac_type}")

        num_dacs = self.dac.initialize()
        self.log(f"{num_dacs} DACs of type {self.laser_control_params.dac_type} found")
        self.dac.connect(self.laser_control_params.dac_index)

    @service("~/set_color", SetColor)
    async def set_color(self, r, g, b, i):
        if self.dac is None:
            return result(success=False)

        self.dac.set_color(r, g, b, i)
        return result(success=True)

    @service("~/add_point", AddPoint)
    async def add_point(self, point):
        if self.dac is None:
            return result(success=False)

        self.dac.add_point(point.x, point.y)
        return result(success=True)

    @service("~/set_points", SetPoints)
    async def set_points(self, points):
        if self.dac is None:
            return result(success=False)

        self.dac.clear_points()
        for point in points:
            self.dac.add_point(point.x, point.y)
        return result(success=True)

    @service("~/remove_point", Trigger)
    async def remove_point(self):
        if self.dac is None:
            return result(success=False)

        self.dac.remove_point()
        return result(success=True)

    @service("~/clear_points", Trigger)
    async def clear_points(self):
        if self.dac is None:
            return result(success=False)

        self.dac.clear_points()
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
        if self.dac is None:
            return result(success=False)

        self.dac.play(
            self.laser_control_params.fps,
            self.laser_control_params.pps,
            self.laser_control_params.transition_duration_ms,
        )
        self._publish_state()
        return result(success=True)

    @service("~/stop", Trigger)
    async def stop(self):
        if self.dac is None:
            return result(success=False)

        self.dac.stop()
        self._publish_state()
        return result(success=True)

    @service("~/get_state", GetState)
    async def get_state(self):
        return result(
            dac_type=self.laser_control_params.dac_type,
            dac_index=self.laser_control_params.dac_index,
            state=State(data=self._get_state()),
        )

    def _get_state(self) -> State:
        if self.dac is None:
            return State.DISCONNECTED
        elif self.dac.playing:
            return State.PLAYING
        else:
            return State.STOPPED

    def _publish_state(self):
        asyncio.create_task(self.state_topic(data=self._get_state()))


def main():
    serve_nodes(LaserControlNode())


if __name__ == "__main__":
    main()
