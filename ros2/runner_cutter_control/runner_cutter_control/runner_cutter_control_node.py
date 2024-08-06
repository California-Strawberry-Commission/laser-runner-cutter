"""File: runner_cutter_control_node.py

Main ROS2 control node for the Laser Runner Cutter. This 
node uses a state machine to control the general objective of the 
system. States are things like calibrating the camera laser system,
finding a specific runner to burn, and burning said runner.  
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_srvs.srv import Trigger
from transitions.extensions.asyncio import AsyncMachine

import camera_control.camera_control_node as camera_control_node
import laser_control.laser_control_node as laser_control_node
from aioros2 import (
    import_node,
    node,
    params,
    result,
    serve_nodes,
    service,
    start,
    topic,
)
from common_interfaces.msg import Vector2, Vector4
from runner_cutter_control.calibration import Calibration
from runner_cutter_control.camera_context import CameraContext
from runner_cutter_control.tracker import Track, Tracker, TrackState
from runner_cutter_control_interfaces.msg import State
from runner_cutter_control_interfaces.msg import Track as TrackMsg
from runner_cutter_control_interfaces.srv import (
    AddCalibrationPoints,
    GetState,
    ManualTargetAimLaser,
)


@dataclass
class RunnerCutterControlParams:
    tracking_laser_color: List[float] = field(default_factory=lambda: [0.15, 0.0, 0.0])
    burn_laser_color: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    burn_time_secs: float = 5.0
    enable_aiming: bool = True


@node("runner_cutter_control_node")
class RunnerCutterControlNode:
    runner_cutter_control_params = params(RunnerCutterControlParams)
    state_topic = topic(
        "~/state",
        State,
        qos=QoSProfile(
            depth=1,
            # Setting durability to Transient Local will persist samples for late joiners
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
        ),
    )

    laser_node: laser_control_node.LaserControlNode = import_node(laser_control_node)
    camera_node: camera_control_node.CameraControlNode = import_node(
        camera_control_node
    )

    @start
    async def start(self):
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
            self.runner_cutter_control_params.enable_aiming,
            self.get_logger(),
        )

        # Publish initial state
        self.publish_state()

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
        if self.state_machine.state == "idle":
            return result(success=False)

        asyncio.create_task(self.state_machine.stop())
        return result(success=True)

    @service("~/get_state", GetState)
    async def get_state(self):
        return result(state=self._get_state())

    def publish_state(self):
        state = self._get_state()
        asyncio.create_task(
            self.state_topic(
                calibrated=state.calibrated,
                state=state.state,
                tracks=state.tracks,
                normalized_laser_bounds=state.normalized_laser_bounds,
            )
        )

    def _get_state(self) -> State:
        frame_size = self.calibration.camera_frame_size

        state_msg = State(
            calibrated=self.state_machine.is_calibrated,
            state=self.state_machine.state,
            normalized_laser_bounds=Vector4(
                w=(
                    self.calibration.laser_bounds[0] / frame_size[0]
                    if frame_size[0] > 0
                    else 0.0
                ),
                x=(
                    self.calibration.laser_bounds[1] / frame_size[1]
                    if frame_size[1] > 0
                    else 0.0
                ),
                y=(
                    self.calibration.laser_bounds[2] / frame_size[0]
                    if frame_size[0] > 0
                    else 0.0
                ),
                z=(
                    self.calibration.laser_bounds[3] / frame_size[1]
                    if frame_size[1] > 0
                    else 0.0
                ),
            ),
        )

        for track_id in self.state_machine.detected_track_ids:
            track = self.runner_tracker.get_track(track_id)
            if track is None:
                continue

            track_msg = TrackMsg()
            track_msg.id = track.id
            track_msg.normalized_pixel_coords = Vector2(
                x=(track.pixel[0] / frame_size[0] if frame_size[0] > 0 else -1.0),
                y=(track.pixel[1] / frame_size[1] if frame_size[1] > 0 else -1.0),
            )
            if track.state == TrackState.PENDING:
                track_msg.state = TrackMsg.PENDING
            elif track.state == TrackState.ACTIVE:
                track_msg.state = TrackMsg.ACTIVE
            elif track.state == TrackState.COMPLETED:
                track_msg.state = TrackMsg.COMPLETED
            elif track.state == TrackState.FAILED:
                track_msg.state = TrackMsg.FAILED
            state_msg.tracks.append(track_msg)

        return state_msg


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
        laser_node: laser_control_node.LaserControlNode,
        camera_node: camera_control_node.CameraControlNode,
        calibration: Calibration,
        runner_tracker: Tracker,
        tracking_laser_color: Tuple[float, float, float],
        burn_laser_color: Tuple[float, float, float],
        burn_time_secs: float,
        enable_aiming: bool,
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
        self._enable_aiming = enable_aiming
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)
        self.detected_track_ids: Set[int] = set()

        self.machine = AsyncMachine(
            model=self,
            states=StateMachine.states,
            initial="idle",
            ignore_invalid_triggers=True,
        )
        # Calibration states
        self.machine.add_transition("run_calibration", "idle", "calibration")
        self.machine.add_transition("calibration_complete", "calibration", "idle")
        # Add Calibration Points states
        self.machine.add_transition(
            "run_add_calibration_points",
            "idle",
            "add_calibration_points",
            conditions=["is_calibrated"],
        )
        self.machine.add_transition(
            "add_calibration_points_complete", "add_calibration_points", "idle"
        )
        # Manual Target Aim Laser states
        self.machine.add_transition(
            "run_manual_target_aim_laser",
            "idle",
            "manual_target_aim_laser",
            conditions=["is_calibrated"],
        )
        self.machine.add_transition(
            "manual_target_aim_laser_complete", "manual_target_aim_laser", "idle"
        )
        # Runner Cutter states
        self.machine.add_transition(
            "run_runner_cutter", "idle", "acquire_target", conditions=["is_calibrated"]
        )
        if self._enable_aiming:
            self.machine.add_transition(
                "target_acquired", "acquire_target", "aim_laser"
            )
            self.machine.add_transition("aim_successful", "aim_laser", "burn_target")
            self.machine.add_transition("aim_failed", "aim_laser", "acquire_target")
        else:
            self.machine.add_transition(
                "target_acquired", "acquire_target", "burn_target"
            )
        self.machine.add_transition(
            "no_target_found", "acquire_target", "acquire_target"
        )
        self.machine.add_transition("burn_complete", "burn_target", "acquire_target")
        # Emergency Stop
        self.machine.add_transition("stop", "*", "idle")

        self._current_task: Optional[asyncio.Task] = None

    @property
    def is_calibrated(self):
        return self._calibration.is_calibrated

    async def on_enter_idle(self):
        self._logger.info(f"Entered state <idle>")

        # Cancel currently running task
        if self._current_task:
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
            self._current_task = None

        await self._laser_node.stop()
        await self._laser_node.clear_points()
        self._runner_tracker.clear()
        self._node.publish_state()

    async def on_enter_calibration(self):
        self._logger.info(f"Entered state <calibration>")
        self._node.publish_state()

        self._current_task = asyncio.create_task(self._calibration_task())

    async def _calibration_task(self):
        await self._calibration.calibrate()
        await self.calibration_complete()

    async def on_enter_add_calibration_points(
        self, normalized_pixel_coords: List[Tuple[float, float]]
    ):
        self._logger.info(f"Entered state <add_calibration_points>")
        self._node.publish_state()

        self._current_task = asyncio.create_task(
            self._add_calibration_points_task(normalized_pixel_coords)
        )

    async def _add_calibration_points_task(
        self, normalized_pixel_coords: List[Tuple[float, float]]
    ):
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
        self._logger.info(f"Entered state <manual_target_aim_laser>")
        self._node.publish_state()

        self._current_task = asyncio.create_task(
            self._manual_target_aim_laser_task(normalized_pixel_coord)
        )

    async def _manual_target_aim_laser_task(
        self, normalized_pixel_coord: Tuple[float, float]
    ):
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

        camera_pixel = (
            round(self._calibration.camera_frame_size[0] * normalized_pixel_coord[0]),
            round(self._calibration.camera_frame_size[1] * normalized_pixel_coord[1]),
        )
        corrected_laser_coord = await self._aim(target_position, camera_pixel)
        if corrected_laser_coord is not None:
            self._logger.info("Aim laser successful.")
        else:
            self._logger.info("Failed to aim laser.")

        await self.manual_target_aim_laser_complete()

    async def on_enter_acquire_target(self):
        self._logger.info(f"Entered state <acquire_target>")
        self._node.publish_state()

        self._current_task = asyncio.create_task(self._acquire_target_task())

    async def _acquire_target_task(self):
        # Detect runners and create/update tracks
        await self._detect_runners()

        target_track = None

        # If there is already an active track, just use that track
        active_tracks = self._runner_tracker.get_tracks_with_state(TrackState.ACTIVE)
        if len(active_tracks) > 0:
            target_track = active_tracks[0]
            self._logger.info(
                f"Active track with ID {target_track.id} already exists. Setting it as target."
            )
        else:
            # Find a track to target
            while True:
                track = self._runner_tracker.get_next_pending_track()
                if track is None:
                    break

                self._logger.info(f"Processing pending track {track.id}...")

                # Check whether the laser coordinates are out of bounds
                laser_coord = self._calibration.camera_point_to_laser_coord(
                    track.position
                )
                if (
                    laser_coord[0] < 0.0
                    or laser_coord[0] > 1.0
                    or laser_coord[1] < 0.0
                    or laser_coord[1] > 1.0
                ):
                    self._logger.info(
                        f"Track {track.id} is out of laser bounds. Marking as failed."
                    )
                    self._runner_tracker.process_track(track.id, TrackState.FAILED)
                    continue
                else:
                    self._logger.info(f"Setting track {track.id} as target.")
                    target_track = track
                    break

        if target_track is None:
            self._logger.info("No target found.")
            await self.no_target_found()
        else:
            if self._enable_aiming:
                await self.target_acquired(target_track)
            else:
                laser_coord = self._calibration.camera_point_to_laser_coord(
                    target_track.position
                )
                await self.target_acquired(target_track, laser_coord)

    async def _detect_runners(self):
        self._logger.info("Detecting runners...")
        result = await self._camera_node.get_runner_detection()
        detection_result = result.result

        prev_detected_track_ids = set(self.detected_track_ids)
        self.detected_track_ids.clear()
        for instance in detection_result.instances:
            pixel = (round(instance.point.x), round(instance.point.y))
            position = (
                instance.position.x,
                instance.position.y,
                instance.position.z,
            )
            track = self._runner_tracker.add_track(
                instance.track_id,
                pixel,
                position,
            )
            if track is None:
                continue

            self.detected_track_ids.add(instance.track_id)

            # Put detected tracks that are marked as failed back into the pending queue, since we want
            # to reattempt to burn them as they could now potentially be in bounds
            if track.state == TrackState.FAILED:
                self._runner_tracker.process_track(track.id, TrackState.PENDING)

        self._logger.info(
            f"Detected {len(self.detected_track_ids)} tracks with IDs {self.detected_track_ids}."
        )

        # Mark any tracks that were previously detected but are no longer detected as out of frame
        out_of_frame_track_ids = prev_detected_track_ids - self.detected_track_ids
        for track_id in out_of_frame_track_ids:
            self._logger.info(
                f"Track {track_id} was detected in the previous frame but is no longer detected."
            )
            track = self._runner_tracker.get_track(track_id)
            if track is None:
                continue

            track.pixel = (-1, -1)
            track.position = (-1.0, -1.0, -1.0)
            if track.state == TrackState.PENDING:
                self._runner_tracker.process_track(track_id, TrackState.FAILED)

    async def on_enter_aim_laser(self, target: Track):
        self._logger.info(f"Entered state <aim_laser>")
        self._node.publish_state()

        self._current_task = asyncio.create_task(self._aim_laser_task(target))

    async def _aim_laser_task(self, target: Track):
        self._logger.info(f"Attempting to aim laser at target track {target.id}...")
        corrected_laser_coord = await self._aim(target.position, target.pixel)
        if corrected_laser_coord is not None:
            self._logger.info(f"Aim at track {target.id} successful.")
            await self.aim_successful(target, corrected_laser_coord)
        else:
            self._logger.info(
                f"Failed to aim laser at track {target.id}. Marking track as failed."
            )
            self._runner_tracker.process_track(target.id, TrackState.FAILED)
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
            laser_pixel, laser_pos = await self._get_laser_pixel_and_pos(
                attempt_interval_s=0.25
            )
            if laser_pixel is None or laser_pos is None:
                self._logger.info("Could not detect laser.")
                return None

            # Calculate camera pixel distance
            dist = np.linalg.norm(
                np.array(laser_pixel).astype(float)
                - np.array(target_pixel).astype(float)
            )
            self._logger.info(
                f"Aiming laser. Target camera pixel = {target_pixel}, laser detected at = {laser_pixel}, dist = {dist}"
            )
            if dist <= dist_threshold:
                return current_laser_coord
            else:
                # Use this opportunity to add to calibration points since we have the laser
                # coord and associated position in camera space
                await self._calibration.add_point_correspondence(
                    current_laser_coord, laser_pixel, laser_pos, update_transform=True
                )

                # Scale correction by the camera frame size
                correction = (
                    np.array(target_pixel) - np.array(laser_pixel)
                ) / np.array(self._calibration.camera_frame_size)
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

    async def _get_laser_pixel_and_pos(
        self, max_attempts: int = 3, attempt_interval_s: float = 0.2
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[float, float, float]]]:
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
                return (round(instance.point.x), round(instance.point.y)), (
                    instance.position.x,
                    instance.position.y,
                    instance.position.z,
                )
            # No lasers detected. Try again.
            await asyncio.sleep(attempt_interval_s)
            attempt += 1
        return None, None

    async def on_enter_burn_target(
        self, target: Track, laser_coord: Tuple[float, float]
    ):
        self._logger.info(f"Entered state <burn_target>")
        self._node.publish_state()

        self._current_task = asyncio.create_task(
            self._burn_target_task(target, laser_coord)
        )

    async def _burn_target_task(self, target: Track, laser_coord: Tuple[float, float]):
        self._logger.info(f"Burning track {target.id}...")
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

        self._logger.info(
            f"Burn complete on track {target.id}. Marking track as completed."
        )
        self._runner_tracker.process_track(target.id, TrackState.COMPLETED)
        await self.burn_complete()


def main():
    serve_nodes(RunnerCutterControlNode())


if __name__ == "__main__":
    main()
