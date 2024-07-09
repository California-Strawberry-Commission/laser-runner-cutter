"""File: runner_cutter_control_node.py

Main ROS2 control node for the Laser Runner Cutter. This 
node uses a state machine to control the general objective of the 
system. States are things like calibrating the camera laser system,
finding a specific runner to burn, and burning said runner.  
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from std_srvs.srv import Trigger
from transitions.extensions.asyncio import AsyncMachine

from aioros2 import (
    ClientDriver,
    node,
    params,
    result,
    serve_nodes,
    service,
    start,
    topic,
)
from camera_control.camera_control_node import CameraControlNode
from common_interfaces.msg import Vector2
from laser_control.laser_control_node import LaserControlNode
from runner_cutter_control.calibration import Calibration
from runner_cutter_control.camera_context import CameraContext
from runner_cutter_control.tracker import Track, Tracker, TrackState
from runner_cutter_control_interfaces.msg import State
from runner_cutter_control_interfaces.srv import (
    AddCalibrationPoints,
    GetState,
    ManualTargetAimLaser,
)


@dataclass
class RunnerCutterControlParams:
    laser_node_name: str = "laser0"
    camera_node_name: str = "camera0"
    tracking_laser_color: List[float] = field(default_factory=lambda: [0.15, 0.0, 0.0])
    burn_laser_color: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    burn_time_secs: float = 5.0


@node("runner_cutter_control_node")
class RunnerCutterControlNode:
    runner_cutter_control_params = params(RunnerCutterControlParams)
    state_topic = topic("~/state", State, 5)

    @start
    async def start(self):
        self.laser_node = ClientDriver(
            LaserControlNode(), self, self.runner_cutter_control_params.laser_node_name
        )
        self.camera_node = ClientDriver(
            CameraControlNode(),
            self,
            self.runner_cutter_control_params.camera_node_name,
        )
        self.calibration = Calibration(
            self.laser_node,
            self.camera_node,
            self.runner_cutter_control_params.tracking_laser_color,
            self.get_logger(),
        )
        self.runner_tracker = Tracker(self.get_logger())
        self.state_machine = StateMachine(
            self,
            self.laser_node,
            self.camera_node,
            self.calibration,
            self.runner_tracker,
            self.runner_cutter_control_params.tracking_laser_color,
            self.runner_cutter_control_params.burn_laser_color,
            self.runner_cutter_control_params.burn_time_secs,
            self.get_logger(),
        )

    # TODO: use action instead once there's a new release of roslib. Currently
    # roslib does not support actions with ROS2
    @service("~/calibrate", Trigger)
    async def calibrate(self):
        if self.state_machine.state != "idle":
            return result(success=False)

        asyncio.create_task(self.state_machine.run_calibration())
        return result(success=True)

    @service("~/add_calibration_points", AddCalibrationPoints)
    async def add_calibration_points(self, normalized_pixel_coords):
        if self.state_machine.state != "idle":
            return result(success=False)

        asyncio.create_task(
            self.state_machine.run_add_calibration_points(
                [
                    (normalized_pixel_coord.x, normalized_pixel_coord.y)
                    for normalized_pixel_coord in normalized_pixel_coords
                ]
            )
        )
        return result(success=True)

    @service("~/manual_target_aim_laser", ManualTargetAimLaser)
    async def manual_target_aim_laser(self, normalized_pixel_coord):
        if self.state_machine.state != "idle":
            return result(success=False)

        asyncio.create_task(
            self.state_machine.run_manual_target_aim_laser(
                (normalized_pixel_coord.x, normalized_pixel_coord.y)
            )
        )
        return result(success=True)

    @service("~/start_runner_cutter", Trigger)
    async def start_runner_cutter(self):
        if self.state_machine.state != "idle":
            return result(success=False)

        asyncio.create_task(self.state_machine.run_runner_cutter())
        return result(success=True)

    @service("~/stop", Trigger)
    async def stop(self):
        asyncio.create_task(self.state_machine.stop())
        return result(success=True)

    @service("~/get_state", GetState)
    async def get_state(self):
        return result(state=self._get_state())

    def publish_state(self):
        state = self._get_state()
        asyncio.create_task(
            self.state_topic(calibrated=state.calibrated, state=state.state)
        )

    def _get_state(self) -> State:
        return State(
            calibrated=self.state_machine.is_calibrated, state=self.state_machine.state
        )


class StateMachine:

    states = [
        "idle",
        "calibration",
        "add_calibration_points",
        "manual_target_aim_laser",
        "acquire_target",
        "aim_laser",
        "burn_target",
    ]

    def __init__(
        self,
        node: RunnerCutterControlNode,
        laser_node: LaserControlNode,
        camera_node: CameraControlNode,
        calibration: Calibration,
        runner_tracker: Tracker,
        tracking_laser_color: Tuple[float, float, float],
        burn_laser_color: Tuple[float, float, float],
        burn_time_secs: float,
        logger: Optional[logging.Logger] = None,
    ):
        self._node = node
        self._laser_node = laser_node
        self._camera_node = camera_node
        self._camera_context = CameraContext(camera_node)
        self._calibration = calibration
        self._runner_tracker = runner_tracker
        self._tracking_laser_color = tracking_laser_color
        self._burn_laser_color = burn_laser_color
        self._burn_time_secs = burn_time_secs
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)

        self.machine = AsyncMachine(
            model=self, states=StateMachine.states, initial="idle", queued=False
        )
        self.machine.add_transition("run_calibration", "idle", "calibration")
        self.machine.add_transition("calibration_complete", "calibration", "idle")
        self.machine.add_transition(
            "run_add_calibration_points",
            "idle",
            "add_calibration_points",
            conditions=["is_calibrated"],
        )
        self.machine.add_transition(
            "add_calibration_points_complete", "add_calibration_points", "idle"
        )
        self.machine.add_transition(
            "run_manual_target_aim_laser",
            "idle",
            "manual_target_aim_laser",
            conditions=["is_calibrated"],
        )
        self.machine.add_transition(
            "manual_target_aim_laser_complete", "manual_target_aim_laser", "idle"
        )
        self.machine.add_transition(
            "run_runner_cutter", "idle", "acquire_target", conditions=["is_calibrated"]
        )
        self.machine.add_transition("target_acquired", "acquire_target", "aim_laser")
        self.machine.add_transition(
            "no_target_found", "acquire_target", "acquire_target"
        )
        self.machine.add_transition("aim_successful", "aim_laser", "burn_target")
        self.machine.add_transition("aim_failed", "aim_laser", "acquire_target")
        self.machine.add_transition("burn_complete", "burn_target", "acquire_target")
        self.machine.add_transition("stop", "*", "idle")

    @property
    def is_calibrated(self):
        return self._calibration.is_calibrated

    async def on_enter_idle(self):
        self._node.publish_state()
        await self._laser_node.stop()
        await self._laser_node.clear_points()
        self._runner_tracker.clear()

    async def on_enter_calibration(self):
        self._node.publish_state()
        await self._calibration.calibrate()
        await self.calibration_complete()

    async def on_enter_add_calibration_points(
        self, normalized_pixel_coords: List[Tuple[float, float]]
    ):
        self._node.publish_state()
        # For each camera pixel, find the 3D position wrt the camera
        result = await self._camera_node.get_positions(
            normalized_pixel_coords=[
                Vector2(
                    x=normalized_pixel_coord[0],
                    y=normalized_pixel_coord[1],
                )
                for normalized_pixel_coord in normalized_pixel_coords
            ]
        )
        positions = [
            (position.x, position.y, position.z) for position in result.positions
        ]
        # Filter out any invalid positions
        positions = [p for p in positions if not all(x < 0 for x in p)]
        # Convert camera positions to laser pixels
        laser_coords = [
            self._calibration.camera_point_to_laser_coord(position)
            for position in positions
        ]
        await self._calibration.add_calibration_points(
            laser_coords, update_transform=True
        )
        await self.add_calibration_points_complete()

    async def on_enter_manual_target_aim_laser(
        self, normalized_pixel_coord: Tuple[float, float]
    ):
        self._node.publish_state()
        # Find the 3D position wrt the camera
        result = await self._camera_node.get_positions(
            normalized_pixel_coords=[
                Vector2(x=normalized_pixel_coord[0], y=normalized_pixel_coord[1])
            ]
        )
        positions = [
            (position.x, position.y, position.z) for position in result.positions
        ]
        target_position = positions[0]

        corrected_laser_coord = await self._aim(target_position, camera_pixel)
        if corrected_laser_coord is not None:
            self._logger.info("Aim laser successful.")
        else:
            self._logger.info("Failed to aim laser.")

        await self.manual_target_aim_laser_complete()

    async def on_enter_acquire_target(self):
        self._node.publish_state()

        # Do nothing if there is already an active target
        if self._runner_tracker.has_active_tracks:
            self._logger.info("Active target already exists.")
            await self.target_acquired(self._runner_tracker.active_tracks[0])
            return

        # If there are no pending tracks, check for new runners
        if not self._runner_tracker.has_pending_tracks:
            result = await self._camera_node.get_runner_detection()
            detection_result = result.result
            for instance in detection_result.instances:
                pixel = (int(instance.point.x), int(instance.point.y))
                position = (
                    instance.position.x,
                    instance.position.y,
                    instance.position.z,
                )
                self._runner_tracker.add_track(
                    pixel,
                    position,
                    track_id=instance.track_id,
                )

        if self._runner_tracker.has_pending_tracks:
            # There are pending tracks. Set one as active.
            self._logger.info("Found a pending track. Marking as active.")
            track = self._runner_tracker.pending_tracks[0]
            track.state = TrackState.ACTIVE
            await self.target_acquired(track)
        else:
            # There are no pending targets.
            self._logger.info("No pending targets found.")
            await self.no_target_found()

    async def on_enter_aim_laser(self, target: Track):
        self._node.publish_state()

        corrected_laser_coord = await self._aim(target.position, target.pixel)
        if corrected_laser_coord is not None:
            await self.aim_successful(target, corrected_laser_coord)
        else:
            self._logger.info("Failed to aim laser.")
            target.state = TrackState.FAILED
            await self.aim_failed()

    async def _aim(
        self, target_position: Tuple[float, float, float], target_pixel: Tuple[int, int]
    ) -> Optional[Tuple[float, float]]:
        # TODO: set exposure/gain automatically when detecting laser
        async with self._camera_context.laser_detection_settings():
            initial_laser_coord = self._calibration.camera_point_to_laser_coord(
                target_position
            )
            await self._laser_node.set_points(
                points=[Vector2(x=initial_laser_coord[0], y=initial_laser_coord[1])]
            )
            await self._laser_node.set_color(
                r=self._tracking_laser_color[0],
                g=self._tracking_laser_color[1],
                b=self._tracking_laser_color[2],
                i=0.0,
            )
            try:
                await self._laser_node.play()
                corrected_laser_coord = await self._correct_laser(
                    target_pixel, initial_laser_coord
                )
            finally:
                await self._laser_node.stop()

        return corrected_laser_coord

    async def _get_laser_pixel_and_pos(
        self, max_attempts: int = 3
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float, float]]]:
        attempt = 0
        while attempt < max_attempts:
            result = await self._camera_node.get_laser_detection()
            detection_result = result.result
            instances = detection_result.instances
            if instances:
                if len(instances) > 1:
                    self._logger.info("Found more than 1 laser during correction")
                # TODO: better handle case where more than 1 laser detected
                instance = instances[0]
                return (instance.point.x, instance.point.y), (
                    instance.position.x,
                    instance.position.y,
                    instance.position.z,
                )
            # No lasers detected. Try again.
            # TODO: optimize the frame callback time and reduce this
            await asyncio.sleep(0.5)
            attempt += 1
        return None, None

    async def _correct_laser(
        self,
        target_pixel: Tuple[int, int],
        original_laser_coord: Tuple[float, float],
        dist_threshold: float = 2.5,
    ) -> Optional[Tuple[float, float]]:
        current_laser_coord = original_laser_coord

        while True:
            await self._laser_node.set_points(
                points=[Vector2(x=current_laser_coord[0], y=current_laser_coord[1])]
            )
            # Wait for galvo to settle and for camera frame capture
            # TODO: optimize the frame callback time and reduce this
            await asyncio.sleep(0.5)
            laser_pixel, laser_pos = await self._get_laser_pixel_and_pos()
            if laser_pixel is None or laser_pos is None:
                self._logger.info("Could not detect laser.")
                return None

            # Calculate camera pixel distance
            laser_pixel = np.array(laser_pixel)
            target_pixel = np.array(target_pixel)
            dist = np.linalg.norm(laser_pixel - target_pixel)
            self._logger.info(
                f"Aiming laser. Target camera pixel = {target_pixel}, laser detected at = {laser_pixel}, dist = {dist}"
            )
            if dist <= dist_threshold:
                return current_laser_coord
            else:
                # Use this opportunity to add to calibration points since we have the laser
                # coord and associated position in camera space
                await self._calibration.add_point_correspondence(
                    current_laser_coord, laser_pos, update_transform=True
                )

                # Scale correction by the camera frame size
                correction = (target_pixel - laser_pixel) / np.array(
                    self._calibration.camera_frame_size
                )
                # Invert Y axis as laser coord Y is flipped from camera frame Y
                correction[1] *= -1
                new_laser_coord = current_laser_coord + correction
                self._logger.info(
                    f"Correcting laser. Dist = {dist}, correction = {correction}, current coord = {current_laser_coord}, new coord = {new_laser_coord}"
                )

                if np.any(
                    new_laser_coord[0] > 1.0
                    or new_laser_coord[1] > 1.0
                    or new_laser_coord[0] < 0.0
                    or new_laser_coord[1] < 0.0
                ):
                    self._logger.info("Laser coord is outside of renderable area.")
                    return None

                current_laser_coord = new_laser_coord

    async def on_enter_burn_target(
        self, target: Track, laser_coord: Tuple[float, float]
    ):
        self._node.publish_state()

        await self._laser_node.set_points(
            points=[Vector2(x=laser_coord[0], y=laser_coord[1])]
        )
        await self._laser_node.set_color(
            r=self._burn_laser_color[0],
            g=self._burn_laser_color[1],
            b=self._burn_laser_color[2],
            i=0.0,
        )
        try:
            await self._laser_node.play()
            await asyncio.sleep(self._burn_time_secs)
        finally:
            await self._laser_node.stop()
        target.state = TrackState.COMPLETED
        await self.burn_complete()


def main():
    serve_nodes(RunnerCutterControlNode())


if __name__ == "__main__":
    main()
