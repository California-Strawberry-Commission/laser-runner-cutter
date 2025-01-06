import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
from cv_bridge import CvBridge
from rcl_interfaces.msg import Log
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
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
from camera_control.camera.lucid_camera import create_lucid_rgbd_camera
from camera_control.camera.realsense_camera import RealSenseCamera
from camera_control.camera.rgbd_camera import State as RgbdCameraState
from camera_control.camera.rgbd_frame import RgbdFrame
from camera_control.detector.circle_detector import CircleDetector
from camera_control.detector.laser_detector import LaserDetector
from camera_control.detector.runner_detector import RunnerDetector
from camera_control_interfaces.msg import (
    DetectionResult,
    DetectionType,
    DeviceState,
    ObjectInstance,
    State,
)
from camera_control_interfaces.srv import (
    GetDetectionResult,
    GetFrame,
    GetPositions,
    GetState,
    SetExposure,
    SetGain,
    SetSaveDirectory,
    StartDetection,
    StartIntervalCapture,
    StopDetection,
)
from common_interfaces.msg import Vector2, Vector3
from common_interfaces.srv import GetBool


def milliseconds_to_ros_time(milliseconds):
    # ROS timestamps consist of two integers, one for seconds and one for nanoseconds
    seconds, remainder_ms = divmod(milliseconds, 1000)
    nanoseconds = remainder_ms * 1e6
    return int(seconds), int(nanoseconds)


@dataclass
class CameraControlParams:
    camera_type: str = "lucid"  # "realsense" or "lucid"
    camera_index: int = 0
    exposure_us: float = -1.0
    gain_db: float = -1.0
    save_dir: str = "~/runner-cutter-output"
    debug_frame_width: int = 640
    debug_video_fps: float = 30.0
    image_capture_interval_secs: float = 5.0


@node("camera_control_node")
class CameraControlNode:
    camera_control_params = params(CameraControlParams)
    state_topic = topic("~/state", State, qos=QOS_LATCHED)
    debug_frame_topic = topic("~/debug_frame", Image, qos=qos_profile_sensor_data)
    detections_topic = topic("~/detections", DetectionResult, qos=5)
    notifications_topic = topic("/notifications", Log, qos=1)

    @start
    async def start(self):
        self.enabled_detection_types = set()
        self._record_video_task_ref = None
        self._video_writer = None
        self._debug_frame = None
        self._interval_capture_task_ref = None
        # For converting numpy array to image msg
        self._cv_bridge = CvBridge()

        # Camera

        # After starting a camera device, when a new frame is available, a callback is called by
        # the camera interface. When a new frame is received, we queue a detection task. In order
        # to prevent multiple detection tasks running concurrently, we use a task queue of size 1
        # and a queue processor task. One additional consideration is that the camera device can
        # be closed at any time, and therefore we need to make sure that the detection task does
        # not use a frame after the device has been closed.

        # We don't need locks for these since read/write only happens on main event loop
        self.current_frame = None
        self._detection_task_queue = asyncio.Queue(1)
        self._detection_completed_event = (
            asyncio.Event()
        )  # to notify when a detection task has completed, used to avoid destroying frame buffer(s) while detection is happening
        self._detection_completed_event.set()
        self._camera_started = False  # flag to prevent frame callback or detection task from executing after device is closed
        self._frame_event = asyncio.Event()  # to notify when a new frame is available

        async def process_detection_task_queue():
            while True:
                task_func = await self._detection_task_queue.get()
                await task_func()
                self._detection_task_queue.task_done()

        loop = asyncio.get_running_loop()
        loop.create_task(process_detection_task_queue())

        def state_change_callback(state: RgbdCameraState):
            loop.call_soon_threadsafe(self._publish_state)

        if self.camera_control_params.camera_type == "realsense":
            self.camera = RealSenseCamera(
                camera_index=self.camera_control_params.camera_index,
                state_change_callback=state_change_callback,
                logger=self.get_logger(),
            )
        elif self.camera_control_params.camera_type == "lucid":
            self.camera = create_lucid_rgbd_camera(
                state_change_callback=state_change_callback,
                logger=self.get_logger(),
            )
        else:
            raise Exception(
                f"Unknown camera_type: {self.camera_control_params.camera_type}"
            )

        # Detectors
        self.runner_detector = RunnerDetector(logger=self.get_logger())
        self.laser_detector = LaserDetector(logger=self.get_logger())
        self.circle_detector = CircleDetector(logger=self.get_logger())

        # Publish initial state
        self._publish_state()

    @service("~/start_device", Trigger)
    async def start_device(self):
        loop = asyncio.get_running_loop()

        def frame_callback(frame: RgbdFrame):
            asyncio.run_coroutine_threadsafe(self._frame_callback(frame), loop)

        self.camera.start(
            exposure_us=self.camera_control_params.exposure_us,
            gain_db=self.camera_control_params.gain_db,
            frame_callback=frame_callback,
        )
        self._camera_started = True

        return result(success=True)

    @service("~/close_device", Trigger)
    async def close_device(self):
        self._camera_started = False
        # Wait until detection completes
        await self._detection_completed_event.wait()
        self.camera.stop()
        self.current_frame = None

        return result(success=True)

    @service("~/has_frames", GetBool)
    async def has_frames(self):
        frame = self.current_frame
        return result(data=(frame is not None))

    @service("~/get_frame", GetFrame)
    async def get_frame(self):
        frame = self.current_frame
        if frame is None:
            return result()

        return result(
            color_frame=self._get_color_frame_msg(
                frame.color_frame, frame.timestamp_millis
            ),
            depth_frame=self._get_depth_frame_msg(
                frame.depth_frame, frame.timestamp_millis
            ),
        )

    @service("~/set_exposure", SetExposure)
    async def set_exposure(self, exposure_us):
        await self.camera_control_params.set(exposure_us=exposure_us)
        self.camera.exposure_us = exposure_us
        self._publish_state()
        return result(success=True)

    @service("~/auto_exposure", Trigger)
    async def auto_exposure(self):
        await self.camera_control_params.set(exposure_us=-1.0)
        self.camera.exposure_us = -1.0
        self._publish_state()
        return result(success=True)

    @service("~/set_gain", SetGain)
    async def set_gain(self, gain_db):
        await self.camera_control_params.set(gain_db=gain_db)
        self.camera.gain_db = gain_db
        self._publish_state()
        return result(success=True)

    @service("~/auto_gain", Trigger)
    async def auto_gain(self):
        await self.camera_control_params.set(gain_db=-1.0)
        self.camera.gain_db = -1.0
        self._publish_state()
        return result(success=True)

    @service("~/get_detection", GetDetectionResult)
    async def get_detection(self, detection_type, wait_for_next_frame):
        if wait_for_next_frame:
            # Wait for two frames, as the next frame to arrive may have already been in progress
            frame_count = 0
            while frame_count < 2:
                self._frame_event.clear()
                await self._frame_event.wait()
                frame_count += 1

        frame = self.current_frame

        if frame is None:
            return result()

        if detection_type == DetectionType.LASER:
            laser_points, conf = await self.laser_detector.detect(frame.color_frame)
            return result(
                result=self._create_detection_result_msg(
                    detection_type, laser_points, frame
                )
            )
        elif detection_type == DetectionType.RUNNER:
            runner_masks, runner_centers, confs, track_ids = (
                await self.runner_detector.detect(frame.color_frame)
            )
            # runner_centers may contain None elements, so filter them out and also remove
            # the corresponding elements from track_ids
            filtered = [
                (center, track_id)
                for center, track_id in zip(runner_centers, track_ids)
                if center is not None
            ]
            if filtered:
                runner_centers, track_ids = zip(*filtered)
                runner_centers = list(runner_centers)
                track_ids = list(track_ids)
            else:
                runner_centers = []
                track_ids = []
            return result(
                result=self._create_detection_result_msg(
                    detection_type, runner_centers, frame, track_ids
                )
            )
        elif detection_type == DetectionType.CIRCLE:
            circle_centers = await self.circle_detector.detect(frame.color_frame)
            return result(
                result=self._create_detection_result_msg(
                    detection_type,
                    circle_centers,
                    frame,
                    [i + 1 for i in range(len(circle_centers))],
                )
            )
        else:
            return result()

    @service("~/start_detection", StartDetection)
    async def start_detection(self, detection_type):
        if detection_type not in self.enabled_detection_types:
            self.enabled_detection_types.add(detection_type)
            self._publish_state()
            return result(success=True)

        return result(success=False)

    @service("~/stop_detection", StopDetection)
    async def stop_detection(self, detection_type):
        if detection_type in self.enabled_detection_types:
            self.enabled_detection_types.discard(detection_type)
            self._publish_state()
            return result(success=True)

        return result(success=False)

    @service("~/stop_all_detections", Trigger)
    async def stop_all_detections(self):
        if len(self.enabled_detection_types) > 0:
            self.enabled_detection_types.clear()
            self._publish_state()
            return result(success=True)

        return result(success=False)

    @service("~/start_recording_video", Trigger)
    async def start_recording_video(self):
        if self._record_video_task_ref is not None:
            self._record_video_task_ref.cancel()
            self._record_video_task_ref = None

        self._record_video_task_ref = asyncio.create_task(
            self._record_video_task(self.camera_control_params.debug_video_fps)
        )
        self._publish_state()
        return result(success=True)

    @service("~/stop_recording_video", Trigger)
    async def stop_recording_video(self):
        if self._record_video_task_ref is None:
            return result(success=False)

        self._record_video_task_ref.cancel()
        self._record_video_task_ref = None
        self._publish_state()
        self._publish_notification("Stopped recording video")
        return result(success=True)

    @service("~/save_image", Trigger)
    async def save_image(self):
        if self._save_image() is None:
            return result(success=False)

        return result(success=True)

    @service("~/start_interval_capture", StartIntervalCapture)
    async def start_interval_capture(self, interval_secs):
        if self._interval_capture_task_ref is not None:
            self._interval_capture_task_ref.cancel()
            self._interval_capture_task_ref = None

        await self.camera_control_params.set(image_capture_interval_secs=interval_secs)
        self._interval_capture_task_ref = asyncio.create_task(
            self._interval_capture_task(interval_secs)
        )
        self._publish_state()
        self._publish_notification(
            f"Started interval capture with {interval_secs}s interval"
        )
        return result(success=True)

    @service("~/stop_interval_capture", Trigger)
    async def stop_interval_capture(self):
        if self._interval_capture_task_ref is None:
            return result(success=False)

        self._interval_capture_task_ref.cancel()
        self._interval_capture_task_ref = None
        self._publish_state()
        self._publish_notification("Stopped interval capture")
        return result(success=True)

    @service("~/set_save_directory", SetSaveDirectory)
    async def set_save_directory(self, save_directory):
        await self.camera_control_params.set(save_dir=save_directory)
        self._publish_state()
        return result(success=True)

    @service("~/get_state", GetState)
    async def get_state(self):
        return result(state=self._get_state())

    @service("~/get_positions", GetPositions)
    async def get_positions(self, normalized_pixel_coords):
        frame = self.current_frame
        if frame is None:
            return result()

        h, w, _ = frame.color_frame.shape

        positions = []
        for normalized_pixel_coord in normalized_pixel_coords:
            x = round(min(max(0.0, normalized_pixel_coord.x), 1.0) * w)
            y = round(min(max(0.0, normalized_pixel_coord.y), 1.0) * h)
            position = frame.get_position((x, y))
            positions.append(
                Vector3(x=position[0], y=position[1], z=position[2])
                if position is not None
                else Vector3(x=-1.0, y=-1.0, z=-1.0)
            )
        return result(positions=positions)

    # region Task definitions

    async def _frame_callback(self, frame: RgbdFrame):
        if not self._camera_started or self.camera.state != RgbdCameraState.STREAMING:
            return

        self.current_frame = frame
        self._frame_event.set()

        if self._detection_task_queue.empty():
            await self._detection_task_queue.put(self._detection_task)

    async def _detection_task(self):
        if not self._camera_started or self.camera.state != RgbdCameraState.STREAMING:
            return

        try:
            self._detection_completed_event.clear()

            frame = self.current_frame
            if frame is None:
                return

            debug_frame = np.copy(frame.color_frame)

            if DetectionType.LASER in self.enabled_detection_types:
                laser_points, confs = await self.laser_detector.detect(
                    frame.color_frame
                )
                debug_frame = self._debug_draw_lasers(debug_frame, laser_points, confs)
                msg = self._create_detection_result_msg(
                    DetectionType.LASER, laser_points, frame
                )
                asyncio.create_task(self.detections_topic(msg))

            if DetectionType.RUNNER in self.enabled_detection_types:
                runner_masks, runner_centers, confs, track_ids = (
                    await self.runner_detector.detect(frame.color_frame)
                )
                debug_frame = self._debug_draw_runners(
                    debug_frame, runner_masks, runner_centers, confs, track_ids
                )
                # runner_centers may contain None elements, so filter them out and also remove
                # the corresponding elements from track_ids
                filtered = [
                    (center, track_id)
                    for center, track_id in zip(runner_centers, track_ids)
                    if center is not None
                ]
                if filtered:
                    runner_centers, track_ids = zip(*filtered)
                    runner_centers = list(runner_centers)
                    track_ids = list(track_ids)
                else:
                    runner_centers = []
                    track_ids = []
                msg = self._create_detection_result_msg(
                    DetectionType.RUNNER, runner_centers, frame, track_ids
                )
                asyncio.create_task(self.detections_topic(msg))

            if DetectionType.CIRCLE in self.enabled_detection_types:
                circle_centers = await self.circle_detector.detect(frame.color_frame)
                debug_frame = self._debug_draw_circles(debug_frame, circle_centers)
                msg = self._create_detection_result_msg(
                    DetectionType.CIRCLE,
                    circle_centers,
                    frame,
                    [i + 1 for i in range(len(circle_centers))],
                )
                asyncio.create_task(self.detections_topic(msg))

            debug_frame = self._debug_draw_timestamp(
                debug_frame, frame.timestamp_millis
            )

            # Downscale debug_frame using INTER_NEAREST for best performance
            h, w, _ = debug_frame.shape
            aspect_ratio = h / w
            new_width = self.camera_control_params.debug_frame_width
            new_height = int(new_width * aspect_ratio)
            debug_frame = cv2.resize(
                debug_frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST
            )

            self._debug_frame = debug_frame

            msg = self._get_color_frame_msg(debug_frame, frame.timestamp_millis)
            asyncio.create_task(self.debug_frame_topic(msg))
        finally:
            self._detection_completed_event.set()

    async def _interval_capture_task(self, interval_secs: float):
        while True:
            self._save_image()
            await asyncio.sleep(interval_secs)

    def _save_image(self) -> Optional[str]:
        frame = self.current_frame
        if frame is None:
            return None

        save_dir = os.path.expanduser(self.camera_control_params.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        ts = time.time()
        datetime_obj = datetime.fromtimestamp(ts)
        datetime_string = datetime_obj.strftime("%Y%m%d%H%M%S")
        image_name = f"{datetime_string}.png"
        image_path = os.path.join(save_dir, image_name)
        cv2.imwrite(
            image_path,
            cv2.cvtColor(frame.color_frame, cv2.COLOR_RGB2BGR),
        )
        self._publish_notification(f"Saved image: {image_path}")
        return image_path

    async def _record_video_task(self, fps: float):
        try:
            while True:
                self._write_video_frame()
                await asyncio.sleep(1 / fps)
        finally:
            self._video_writer = None

    def _write_video_frame(self):
        if self._video_writer is None and self._debug_frame is not None:
            save_dir = os.path.expanduser(self.camera_control_params.save_dir)
            os.makedirs(save_dir, exist_ok=True)
            ts = time.time()
            datetime_obj = datetime.fromtimestamp(ts)
            datetime_string = datetime_obj.strftime("%Y%m%d%H%M%S")
            video_name = f"{datetime_string}.avi"
            video_path = os.path.join(save_dir, video_name)
            h, w, _ = self._debug_frame.shape
            self._video_writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"XVID"),
                self.camera_control_params.debug_video_fps,
                (w, h),
            )
            self._publish_notification(f"Started recording video: {video_path}")

        if self._video_writer is not None:
            self._video_writer.write(cv2.cvtColor(self._debug_frame, cv2.COLOR_RGB2BGR))

    # endregion

    # region State and notifs publishing

    def _get_device_state(self) -> DeviceState:
        if self.camera is None:
            return DeviceState.DISCONNECTED

        if self.camera.state == RgbdCameraState.CONNECTING:
            return DeviceState.CONNECTING
        elif self.camera.state == RgbdCameraState.STREAMING:
            return DeviceState.STREAMING
        else:
            return DeviceState.DISCONNECTED

    def _get_state(self) -> State:
        state = State()
        state.device_state = self._get_device_state()
        state.enabled_detection_types = list(self.enabled_detection_types)
        state.recording_video = self._record_video_task_ref is not None
        state.interval_capture_active = self._interval_capture_task_ref is not None
        state.exposure_us = self.camera.exposure_us
        exposure_us_range = self.camera.get_exposure_us_range()
        state.exposure_us_range = Vector2(
            x=exposure_us_range[0], y=exposure_us_range[1]
        )
        state.gain_db = self.camera.gain_db
        gain_db_range = self.camera.get_gain_db_range()
        state.gain_db_range = Vector2(x=gain_db_range[0], y=gain_db_range[1])
        state.save_directory = self.camera_control_params.save_dir
        state.image_capture_interval_secs = (
            self.camera_control_params.image_capture_interval_secs
        )
        return state

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

    # endregion

    # region Message builders

    def _create_detection_result_msg(
        self,
        detection_type: int,
        points: List[Tuple[int, int]],
        frame: RgbdFrame,
        track_ids: Optional[List[int]] = None,
    ) -> DetectionResult:
        msg = DetectionResult(
            detection_type=detection_type, timestamp=(frame.timestamp_millis / 1000)
        )
        for idx, point in enumerate(points):
            point_msg = Vector2(x=float(point[0]), y=float(point[1]))
            position = frame.get_position(point)
            if position is not None:
                object_instance = ObjectInstance()
                object_instance.track_id = (
                    track_ids[idx]
                    if track_ids is not None and idx < len(track_ids)
                    else -1
                )
                object_instance.position = Vector3(
                    x=position[0], y=position[1], z=position[2]
                )
                object_instance.point = point_msg
                msg.instances.append(object_instance)
            else:
                msg.invalid_points.append(point_msg)
        self.log_debug(
            f"{len(msg.instances)} instances had valid positions, out of {len(points)} total detected"
        )
        return msg

    def _get_color_frame_msg(
        self, color_frame: np.ndarray, timestamp_millis: float
    ) -> Image:
        msg = self._cv_bridge.cv2_to_imgmsg(color_frame, encoding="rgb8")
        sec, nanosec = milliseconds_to_ros_time(timestamp_millis)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        return msg

    def _get_depth_frame_msg(
        self, depth_frame: np.ndarray, timestamp_millis: float
    ) -> Image:
        msg = self._cv_bridge.cv2_to_imgmsg(depth_frame, encoding="mono16")
        sec, nanosec = milliseconds_to_ros_time(timestamp_millis)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        return msg

    def _get_color_frame_compressed_msg(
        self, color_frame: np.ndarray, timestamp_millis: float
    ) -> CompressedImage:
        sec, nanosec = milliseconds_to_ros_time(timestamp_millis)
        _, jpeg_data = cv2.imencode(
            ".jpg", cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
        )
        msg = CompressedImage()
        msg.format = "jpeg"
        msg.data = jpeg_data.tobytes()
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        return msg

    # endregion

    # region Debug frame drawing

    def _debug_draw_lasers(
        self, debug_frame, laser_points, confs, color=(255, 0, 255), draw_conf=True
    ):
        for laser_point, conf in zip(laser_points, confs):
            pos = [int(laser_point[0]), int(laser_point[1])]
            debug_frame = cv2.drawMarker(
                debug_frame,
                pos,
                color,
                cv2.MARKER_STAR,
                thickness=1,
                markerSize=20,
            )
            if draw_conf:
                pos = [int(laser_point[0]) - 15, int(laser_point[1]) - 15]
                font = cv2.FONT_HERSHEY_SIMPLEX
                debug_frame = cv2.putText(
                    debug_frame, f"{conf:.2f}", pos, font, 0.5, color
                )
        return debug_frame

    def _debug_draw_runners(
        self,
        debug_frame,
        runner_masks,
        runner_centers,
        confs,
        track_ids,
        mask_color=(255, 255, 255),
        center_color=(255, 64, 255),
        draw_conf=True,
        draw_track_id=True,
    ):
        for runner_mask, runner_center, conf, track_id in zip(
            runner_masks, runner_centers, confs, track_ids
        ):
            debug_frame = cv2.fillPoly(
                debug_frame,
                pts=[np.array(runner_mask, dtype=np.int32)],
                color=mask_color,
            )
            if runner_center is not None:
                pos = [int(runner_center[0]), int(runner_center[1])]
                debug_frame = cv2.drawMarker(
                    debug_frame,
                    pos,
                    center_color,
                    cv2.MARKER_TILTED_CROSS,
                    thickness=3,
                    markerSize=20,
                )
            if draw_conf:
                pos = [int(runner_center[0]) + 15, int(runner_center[1]) - 5]
                font = cv2.FONT_HERSHEY_SIMPLEX
                debug_frame = cv2.putText(
                    debug_frame, f"{conf:.2f}", pos, font, 1, center_color, 2
                )
            if draw_track_id and track_id > 0:
                pos = [int(runner_center[0]) + 15, int(runner_center[1]) + 20]
                font = cv2.FONT_HERSHEY_SIMPLEX
                debug_frame = cv2.putText(
                    debug_frame, f"{track_id}", pos, font, 1, center_color, 2
                )
        return debug_frame

    def _debug_draw_circles(self, debug_frame, circle_centers, color=(255, 0, 255)):
        for circle_center in circle_centers:
            pos = [int(circle_center[0]), int(circle_center[1])]
            debug_frame = cv2.drawMarker(
                debug_frame,
                pos,
                color,
                cv2.MARKER_STAR,
                thickness=1,
                markerSize=20,
            )
        return debug_frame

    def _debug_draw_timestamp(self, debug_frame, timestamp, color=(255, 255, 255)):
        h, w, _ = debug_frame.shape
        debug_frame = cv2.putText(
            debug_frame,
            f"{int(timestamp)}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
        return debug_frame

    # endregion


def main():
    serve_nodes(CameraControlNode())


if __name__ == "__main__":
    main()
