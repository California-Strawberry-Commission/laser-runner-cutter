"""File: runner_cutter_control_node.py

Main ROS2 control node for the Laser Runner Cutter. This 
node uses a state machine to control the general objective of the 
system. States are things like calibrating the camera laser system,
finding a specific runner to burn, and burning said runner.  
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Coroutine, List, Optional, Set, Tuple

import numpy as np
from rcl_interfaces.msg import Log
from std_srvs.srv import Trigger

import camera_control.camera_control_node as camera_control_node
import laser_control.laser_control_node as laser_control_node
from aioros2 import (
    QOS_LATCHED,
    import_node,
    node,
    params,
    result,
    serve_nodes,
    service,
    start,
    subscribe,
    topic,
)
from camera_control_interfaces.msg import DetectionType
from common_interfaces.msg import Vector2, Vector4
from runner_cutter_control.calibration import Calibration
from runner_cutter_control.camera_context import CameraContext
from runner_cutter_control.tracker import Track, Tracker, TrackState
from runner_cutter_control_interfaces.msg import State
from runner_cutter_control_interfaces.msg import Track as TrackMsg
from runner_cutter_control_interfaces.msg import Tracks as TracksMsg
from runner_cutter_control_interfaces.msg import TrackState as TrackStateMsg
from runner_cutter_control_interfaces.srv import (
    AddCalibrationPoints,
    GetState,
    ManualTargetAimLaser,
)


def milliseconds_to_ros_time(milliseconds):
    # ROS timestamps consist of two integers, one for seconds and one for nanoseconds
    seconds, remainder_ms = divmod(milliseconds, 1000)
    nanoseconds = remainder_ms * 1e6
    return int(seconds), int(nanoseconds)


@dataclass
class RunnerCutterControlParams:
    tracking_laser_color: List[float] = field(default_factory=lambda: [0.15, 0.0, 0.0])
    burn_laser_color: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    burn_time_secs: float = 5.0
    enable_aiming: bool = True
    save_dir: str = "~/runner-cutter-output"


@node("runner_cutter_control_node")
class RunnerCutterControlNode:
    runner_cutter_control_params = params(RunnerCutterControlParams)
    state_topic = topic("~/state", State, qos=QOS_LATCHED)
    notifications_topic = topic("/notifications", Log, qos=1)
    tracks_topic = topic("~/tracks", TracksMsg, qos=5)

    laser_node: laser_control_node.LaserControlNode = import_node(laser_control_node)
    camera_node: camera_control_node.CameraControlNode = import_node(
        camera_control_node
    )

    @start
    async def start(self):
        self._calibration = Calibration(
            self.laser_node,
            self.camera_node,
            self.runner_cutter_control_params.tracking_laser_color,
            self.get_logger(),
        )
        self._camera_context = CameraContext(self.camera_node)
        self._current_task: Optional[asyncio.Task] = None

        self._runner_tracker = Tracker()
        self._runner_tracker_lock = asyncio.Lock()
        self._last_detected_track_ids: Set[int] = set()
        self._runner_detection_event = asyncio.Event()

        # Publish initial state
        self._publish_state()

    @subscribe(camera_node.detections_topic)
    async def on_detection(self, detection_type, timestamp, instances, invalid_points):
        if (
            detection_type == DetectionType.RUNNER
            or detection_type == DetectionType.CIRCLE
        ):
            async with self._runner_tracker_lock:
                # For new tracks, add to tracker and set as pending. For tracks that are detected
                # again, update the track pixel and position; for FAILED tracks, set them as PENDING
                # since they may have moved since the last detection.
                prev_pending_tracks = set(
                    [
                        track.id
                        for track in self._runner_tracker.get_tracks_with_state(
                            TrackState.PENDING
                        )
                    ]
                )
                prev_detected_track_ids = set(self._last_detected_track_ids)
                self._last_detected_track_ids.clear()
                for instance in instances:
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

                    self._last_detected_track_ids.add(instance.track_id)

                    # Put detected tracks that are marked as failed back into the pending queue, since
                    # we want to reattempt to burn them as they could now potentially be in bounds.
                    if track.state == TrackState.FAILED:
                        self._runner_tracker.process_track(track.id, TrackState.PENDING)

                # Mark any tracks that were previously detected and are PENDING but are no longer
                # detected as FAILED.
                out_of_frame_track_ids = (
                    prev_detected_track_ids - self._last_detected_track_ids
                )
                for track_id in out_of_frame_track_ids:
                    track = self._runner_tracker.get_track(track_id)
                    if track is None:
                        continue

                    track.pixel = (-1, -1)
                    track.position = (-1.0, -1.0, -1.0)
                    if track.state == TrackState.PENDING:
                        self._runner_tracker.process_track(track_id, TrackState.FAILED)

                # Notify when the pending tracks have changed
                pending_tracks = set(
                    [
                        track.id
                        for track in self._runner_tracker.get_tracks_with_state(
                            TrackState.PENDING
                        )
                    ]
                )
                if prev_pending_tracks != pending_tracks:
                    self._runner_detection_event.set()

    # TODO: use action instead once there's a new release of roslib. Currently
    # roslib does not support actions with ROS2
    @service("~/calibrate", Trigger)
    async def calibrate(self):
        success = self._start_task(self._calibration.calibrate(), "calibration")
        return result(success=success)

    @service("~/save_calibration", Trigger)
    async def save_calibration(self):
        filepath = self._calibration.save(self.runner_cutter_control_params.save_dir)
        if filepath is not None:
            self._publish_notification(f"Calibration saved: {filepath}")
            return result(success=True)
        else:
            self._publish_notification(
                "Calibration could not be saved", level=logging.WARNING
            )
            return result(success=False)

    @service("~/load_calibration", Trigger)
    async def load_calibration(self):
        filepath = self._calibration.load(self.runner_cutter_control_params.save_dir)
        if filepath is not None:
            self._publish_notification(f"Calibration loaded: {filepath}")
            self._publish_state()
            return result(success=True)
        else:
            self._publish_notification(
                "Calibration could not be loaded", level=logging.WARNING
            )
            return result(success=False)

    @service("~/add_calibration_points", AddCalibrationPoints)
    async def add_calibration_points(self, normalized_pixel_coords):
        success = self._start_task(
            self._add_calibration_points_task(
                [
                    (normalized_pixel_coord.x, normalized_pixel_coord.y)
                    for normalized_pixel_coord in normalized_pixel_coords
                ]
            ),
            "add_calibration_points",
        )
        return result(success=success)

    @service("~/manual_target_aim_laser", ManualTargetAimLaser)
    async def manual_target_aim_laser(self, normalized_pixel_coord):
        success = self._start_task(
            self._manual_target_aim_laser_task(
                (normalized_pixel_coord.x, normalized_pixel_coord.y)
            ),
            "manual_target_aim_laser",
        )
        return result(success=success)

    @service("~/start_runner_cutter", Trigger)
    async def start_runner_cutter(self):
        success = self._start_task(self._runner_cutter_task(), "runner_cutter")
        return result(success=success)

    @service("~/start_circle_follower", Trigger)
    async def start_circle_follower(self):
        success = self._start_task(
            self._runner_cutter_task(
                detection_type=DetectionType.CIRCLE, enable_detection_during_burn=False
            ),
            "circle_follower",
        )
        return result(success=success)

    @service("~/stop", Trigger)
    async def stop(self):
        success = await self._stop_current_task()
        return result(success=success)

    @service("~/get_state", GetState)
    async def get_state(self):
        return result(state=self._get_state())

    # region Task management

    def _start_task(self, coro: Coroutine, name: str | None = None) -> bool:
        if self._current_task is not None and not self._current_task.done():
            return False

        async def coro_wrapper(coro: Coroutine):
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
        await self.laser_node.stop()
        await self.laser_node.clear_points()
        await self.camera_node.stop_all_detections()
        self._runner_tracker.clear()
        self._last_detected_track_ids.clear()

    # endregion

    # region Task definitions

    async def _add_calibration_points_task(
        self, normalized_pixel_coords: List[Tuple[float, float]]
    ):
        # For each camera pixel, find the 3D position wrt the camera
        result = await self.camera_node.get_positions(
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
        # Filter out laser coords that are out of bounds
        laser_coords = [
            (x, y) for (x, y) in laser_coords if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        ]
        await self._calibration.add_calibration_points(
            laser_coords, update_transform=True
        )

    async def _manual_target_aim_laser_task(
        self, normalized_pixel_coord: Tuple[float, float]
    ):
        # Find the 3D position wrt the camera
        result = await self.camera_node.get_positions(
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
            self.log("Aim laser successful.")
        else:
            self.log("Failed to aim laser.")

    async def _aim(
        self, target_position: Tuple[float, float, float], target_pixel: Tuple[int, int]
    ) -> Optional[Tuple[float, float]]:
        # TODO: set exposure/gain automatically when detecting laser
        async with self._camera_context.laser_detection_settings():
            initial_laser_coord = self._calibration.camera_point_to_laser_coord(
                target_position
            )
            await self.laser_node.set_points(
                points=[Vector2(x=initial_laser_coord[0], y=initial_laser_coord[1])]
            )
            tracking_laser_color = (
                self.runner_cutter_control_params.tracking_laser_color
            )
            await self.laser_node.set_color(
                r=tracking_laser_color[0],
                g=tracking_laser_color[1],
                b=tracking_laser_color[2],
                i=0.0,
            )
            try:
                await self.laser_node.play()
                corrected_laser_coord = await self._correct_laser(
                    target_pixel, initial_laser_coord
                )
            finally:
                await self.laser_node.stop()

        return corrected_laser_coord

    async def _correct_laser(
        self,
        target_pixel: Tuple[int, int],
        original_laser_coord: Tuple[float, float],
        dist_threshold: float = 2.5,
    ) -> Optional[Tuple[float, float]]:
        current_laser_coord = original_laser_coord

        while True:
            await self.laser_node.set_points(
                points=[Vector2(x=current_laser_coord[0], y=current_laser_coord[1])]
            )
            laser_pixel, laser_pos = await self._get_laser_pixel_and_pos()
            if laser_pixel is None or laser_pos is None:
                self.log_warn("Could not detect laser.")
                return None

            # Calculate camera pixel distance
            dist = np.linalg.norm(
                np.array(laser_pixel).astype(float)
                - np.array(target_pixel).astype(float)
            )
            self.log(
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
                self.log(
                    f"Correcting laser. Dist = {dist}, correction = {correction}, current coord = {current_laser_coord}, new coord = {new_laser_coord}"
                )

                if np.any(
                    new_laser_coord[0] > 1.0
                    or new_laser_coord[1] > 1.0
                    or new_laser_coord[0] < 0.0
                    or new_laser_coord[1] < 0.0
                ):
                    self.log("Laser coord is outside of renderable area.")
                    return None

                current_laser_coord = new_laser_coord

    async def _get_laser_pixel_and_pos(
        self, max_attempts: int = 3
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[float, float, float]]]:
        attempt = 0
        while attempt < max_attempts:
            result = await self.camera_node.get_detection(
                detection_type=DetectionType.LASER, wait_for_next_frame=True
            )
            detection_result = result.result
            instances = detection_result.instances
            if instances:
                if len(instances) > 1:
                    self.log("Found more than 1 laser during correction")
                # TODO: better handle case where more than 1 laser detected
                instance = instances[0]
                return (round(instance.point.x), round(instance.point.y)), (
                    instance.position.x,
                    instance.position.y,
                    instance.position.z,
                )
            # No lasers detected. Try again.
            attempt += 1
        return None, None

    async def _runner_cutter_task(
        self, detection_type=DetectionType.RUNNER, enable_detection_during_burn=False
    ):
        await self.camera_node.start_detection(detection_type=detection_type)
        while True:
            # Acquire target. If there are no valid targets, wait for another detection event.
            target = await self._acquire_next_target()
            if target is None:
                self.log("No targets found. Waiting for detection.")
                await self._runner_detection_event.wait()
                self._runner_detection_event.clear()
                continue

            try:
                if not enable_detection_during_burn:
                    # Temporarily disable runner detection during aim/burn
                    await self.camera_node.stop_detection(detection_type=detection_type)

                # Aim
                if self.runner_cutter_control_params.enable_aiming:
                    laser_coord = await self._aim(target.position, target.pixel)
                    if laser_coord is None:
                        self.log(f"Failed to aim laser at track {target.id}.")
                        self._runner_tracker.process_track(target.id, TrackState.FAILED)
                        continue
                else:
                    laser_coord = self._calibration.camera_point_to_laser_coord(
                        target.position
                    )

                # Burn
                await self._burn_target(target, laser_coord)
            finally:
                if not enable_detection_during_burn:
                    await self.camera_node.start_detection(
                        detection_type=detection_type
                    )

    async def _acquire_next_target(self) -> Optional[Track]:
        # If there is already an active track, just use that track. Otherwise, check pending
        # tracks until we find a suitable target.
        async with self._runner_tracker_lock:
            active_tracks = self._runner_tracker.get_tracks_with_state(
                TrackState.ACTIVE
            )
            if len(active_tracks) > 0:
                self.log(
                    f"Active track with ID {active_tracks[0].id} already exists. Setting it as target."
                )
                return active_tracks[0]

            self.log(
                f"Selecting target among {len(self._runner_tracker.get_tracks_with_state(TrackState.PENDING))} pending tracks."
            )
            while True:
                track = self._runner_tracker.get_next_pending_track()
                if track is None:
                    return None

                self.log(f"Processing pending track {track.id}...")
                # If the laser coordinates of the track are in bounds, use the track as the
                # target. Otherwise, mark the track as FAILED.
                laser_coord = self._calibration.camera_point_to_laser_coord(
                    track.position
                )
                if 0.0 <= laser_coord[0] <= 1.0 and 0.0 <= laser_coord[1] <= 1.0:
                    self.log(f"Setting track {track.id} as target.")
                    return track
                else:
                    self.log(
                        f"Track {track.id} is out of laser bounds. Marking as failed."
                    )
                    self._runner_tracker.process_track(track.id, TrackState.FAILED)

    async def _burn_target(self, target: Track, laser_coord: Tuple[float, float]):
        self.log(f"Burning track {target.id}...")
        burn_laser_color = self.runner_cutter_control_params.burn_laser_color
        await self.laser_node.set_color(
            r=burn_laser_color[0],
            g=burn_laser_color[1],
            b=burn_laser_color[2],
            i=0.0,
        )
        await self.laser_node.set_points(
            points=[Vector2(x=laser_coord[0], y=laser_coord[1])]
        )
        try:
            await self.laser_node.play()
            await asyncio.sleep(self.runner_cutter_control_params.burn_time_secs)
        finally:
            await self.laser_node.stop()
        self._runner_tracker.process_track(target.id, TrackState.COMPLETED)
        self.log(f"Burn complete on track {target.id}.")

    # endregion

    # region State and notifs publishing

    def _publish_state(self):
        state = self._get_state()
        asyncio.create_task(self.state_topic(state))

    def _publish_notification(self, msg: str, level: int = logging.INFO):
        timestamp_millis = int(time.time() * 1000)
        sec, nanosec = milliseconds_to_ros_time(timestamp_millis)
        log_message = Log()
        log_message.stamp.sec = sec
        log_message.stamp.nanosec = nanosec
        log_message.level = level
        log_message.msg = msg
        self.get_logger().log(msg, level)
        asyncio.create_task(self.notifications_topic(log_message))

    def _publish_tracks(self):
        tracks_msg = self._get_tracks_msg()
        asyncio.create_task(self.tracks_topic(tracks_msg))

    def _get_state(self) -> State:
        normalized_laser_bounds = self._calibration.normalized_laser_bounds
        return State(
            calibrated=self._calibration.is_calibrated,
            state=(
                "idle"
                if self._current_task is None or self._current_task.done()
                else self._current_task.get_name()
            ),
            normalized_laser_bounds=Vector4(
                w=normalized_laser_bounds[0],
                x=normalized_laser_bounds[1],
                y=normalized_laser_bounds[2],
                z=normalized_laser_bounds[3],
            ),
        )

    def _get_tracks_msg(self) -> TracksMsg:
        tracks_msg = TracksMsg()
        frame_size = self._calibration.camera_frame_size
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
                track_msg.state = TrackStateMsg.PENDING
            elif track.state == TrackState.ACTIVE:
                track_msg.state = TrackStateMsg.ACTIVE
            elif track.state == TrackState.COMPLETED:
                track_msg.state = TrackStateMsg.COMPLETED
            elif track.state == TrackState.FAILED:
                track_msg.state = TrackStateMsg.FAILED
            tracks_msg.tracks.append(track_msg)

        return tracks_msg

    # endregion


def main():
    serve_nodes(RunnerCutterControlNode())


if __name__ == "__main__":
    main()
