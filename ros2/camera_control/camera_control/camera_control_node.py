import asyncio
import functools
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from ml_utils.mask_center import contour_center
from runner_segmentation.yolo import Yolo
from sensor_msgs.msg import CompressedImage, Image
from std_srvs.srv import Trigger

from aioros2 import node, params, result, serve_nodes, service, start, timer, topic
from camera_control.camera.realsense import RealSense
from camera_control.camera.lucid import create_lucid_rgbd_camera
from camera_control.camera.rgbd_frame import RgbdFrame
from camera_control_interfaces.msg import (
    DetectionResult,
    DeviceState,
    ObjectInstance,
    State,
)
from camera_control_interfaces.srv import (
    GetDetectionResult,
    GetFrame,
    GetPositionsForPixels,
    GetState,
    SetExposure,
    SetSaveDirectory,
    StartIntervalCapture,
)
from common_interfaces.msg import Vector2, Vector3
from common_interfaces.srv import GetBool
from rcl_interfaces.msg import Log


@dataclass
class CameraControlParams:
    camera_type: str = "lucid"  # "realsense" or "lucid"
    camera_index: int = 0
    fps: int = 30
    rgb_size: List[int] = field(default_factory=lambda: [1280, 720])
    depth_size: List[int] = field(default_factory=lambda: [1280, 720])
    save_dir: str = "~/Pictures/runner-cutter-app"
    debug_frame_width: int = 640


def milliseconds_to_ros_time(milliseconds):
    # ROS timestamps consist of two integers, one for seconds and one for nanoseconds
    seconds, remainder_ms = divmod(milliseconds, 1000)
    nanoseconds = remainder_ms * 1e6
    return int(seconds), int(nanoseconds)


@node("camera_control_node")
class CameraControlNode:
    camera_control_params = params(CameraControlParams)
    state_topic = topic("~/state", State, 5)
    # Increasing queue size for Image topics seems to help prevent web_video_server's subscription
    # from stalling
    color_frame_topic = topic("~/color_frame", Image, 5)
    debug_frame_topic = topic("~/debug_frame", Image, 5)
    laser_detections_topic = topic("~/laser_detections", DetectionResult, 5)
    runner_detections_topic = topic("~/runner_detections", DetectionResult, 5)
    # ROS publishes logs on /rosout, but as it contains logs from all nodes and also contains
    # every single log message, we create a node-specific topic here for logs that would
    # potentially be displayed on UI
    log_topic = topic("~/log", Log, 10)

    async def get_current_frame(self):
        async with self._frame_lock:
            return self.__current_frame

    async def set_current_frame(self, frame):
        async with self._frame_lock:
            self.__current_frame = frame

    @start
    async def start(self):
        self.laser_detection_enabled = False
        self.runner_detection_enabled = False
        self.video_writer = None
        self.interval_capture_task = None
        # For converting numpy array to image msg
        self.cv_bridge = CvBridge()

        # Camera
        self._frame_lock = asyncio.Lock()
        self.__current_frame = None
        self._camera_lock = asyncio.Lock()
        if self.camera_control_params.camera_type == "realsense":
            self.camera = RealSense(
                self.camera_control_params.rgb_size,
                self.camera_control_params.depth_size,
                fps=self.camera_control_params.fps,
                camera_index=self.camera_control_params.camera_index,
                logger=self.get_logger(),
            )
        elif self.camera_control_params.camera_type == "lucid":
            self.camera = create_lucid_rgbd_camera(logger=self.get_logger())
        else:
            raise Exception(
                f"Unknown camera_type: {self.camera_control_params.camera_type}"
            )
        self.connecting = False

        # ML models
        package_share_directory = get_package_share_directory("camera_control")
        runner_weights_path = os.path.join(
            package_share_directory, "models", "RunnerSegYoloV8m.pt"
        )
        self.runner_seg_model = Yolo(runner_weights_path)
        self.runner_seg_size = (1024, 768)
        laser_weights_path = os.path.join(
            package_share_directory, "models", "LaserDetectionYoloV8n.pt"
        )
        self.laser_detection_model = Yolo(laser_weights_path)
        self.laser_detection_size = (640, 480)

    @service("~/start_device", Trigger)
    async def start_device(self):
        self.connecting = True
        self._publish_state()

        async with self._camera_lock:
            await asyncio.get_running_loop().run_in_executor(
                None, self.camera.initialize
            )

        self.connecting = False
        self._publish_state()

        return result(success=True)

    @service("~/close_device", Trigger)
    async def close_device(self):
        async with self._camera_lock:
            await asyncio.get_running_loop().run_in_executor(None, self.camera.close)

        self._publish_state()
        return result(success=True)

    @service("~/has_frames", GetBool)
    async def has_frames(self):
        frame = await self.get_current_frame()
        return result(data=(frame is not None))

    @service("~/get_frame", GetFrame)
    async def get_frame(self):
        frame = await self.get_current_frame()
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
        self.camera.exposure_us = exposure_us
        return result(success=True)

    @service("~/auto_exposure", Trigger)
    async def auto_exposure(self):
        self.camera.exposure_us = -1.0
        return result(success=True)

    @service("~/get_laser_detection", GetDetectionResult)
    async def get_laser_detection(self):
        frame = await self.get_current_frame()
        if frame is None:
            return result()

        laser_points, conf = await self._get_laser_points(frame.color_frame)
        return result(result=self._create_detection_result_msg(laser_points, frame))

    @service("~/get_runner_detection", GetDetectionResult)
    async def get_runner_detection(self):
        frame = await self.get_current_frame()
        if frame is None:
            return result()

        runner_masks, confs, track_ids = await self._get_runner_masks(frame.color_frame)
        runner_centers = await self._get_runner_centers(runner_masks)
        runner_centers = [center for center in runner_centers if center is not None]
        return result(
            result=self._create_detection_result_msg(runner_centers, frame, track_ids)
        )

    @service("~/start_laser_detection", Trigger)
    async def start_laser_detection(self):
        self.laser_detection_enabled = True
        self._publish_state()
        return result(success=True)

    @service("~/stop_laser_detection", Trigger)
    async def stop_laser_detection(self):
        self.laser_detection_enabled = False
        self._publish_state()
        return result(success=True)

    @service("~/start_runner_detection", Trigger)
    async def start_runner_detection(self):
        self.runner_detection_enabled = True
        self._publish_state()
        return result(success=True)

    @service("~/stop_runner_detection", Trigger)
    async def stop_runner_detection(self):
        self.runner_detection_enabled = False
        self._publish_state()
        return result(success=True)

    @service("~/start_recording_video", Trigger)
    async def start_recording_video(self):
        save_dir = os.path.expanduser(self.camera_control_params.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        ts = time.time()
        datetime_obj = datetime.fromtimestamp(ts)
        datetime_string = datetime_obj.strftime("%Y%m%d%H%M%S")
        video_name = f"{datetime_string}.avi"
        video_path = os.path.join(save_dir, video_name)
        self.video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"XVID"),
            self.camera_control_params.fps,
            (
                self.camera_control_params.rgb_size[0],
                self.camera_control_params.rgb_size[1],
            ),
        )
        self._publish_state()
        self._publish_log_message(f"Started recording video: {video_path}")
        return result(success=True)

    @service("~/stop_recording_video", Trigger)
    async def stop_recording_video(self):
        self.video_writer = None
        self._publish_state()
        self._publish_log_message("Stopped recording video")
        return result(success=True)

    @service("~/save_image", Trigger)
    async def save_image(self):
        if (await self._save_image()) is None:
            return result(success=False)

        return result(success=True)

    @service("~/start_interval_capture", StartIntervalCapture)
    async def start_interval_capture(self, interval_secs):
        if self.interval_capture_task is not None:
            self.interval_capture_task.cancel()
            self.interval_capture_task = None

        self.interval_capture_task = asyncio.create_task(
            self._interval_capture_task(interval_secs)
        )
        self._publish_state()
        self._publish_log_message(
            f"Started interval capture. Interval: {interval_secs}s"
        )
        return result(success=True)

    @service("~/stop_interval_capture", Trigger)
    async def stop_interval_capture(self):
        if self.interval_capture_task is None:
            return result(success=False)

        self.interval_capture_task.cancel()
        self.interval_capture_task = None
        self._publish_state()
        self._publish_log_message("Stopped interval capture")
        return result(success=True)

    @service("~/set_save_directory", SetSaveDirectory)
    async def set_save_directory(self, save_directory):
        await self.camera_control_params.set(save_dir=save_directory)
        self._publish_state()
        return result(success=True)

    async def _interval_capture_task(self, interval_secs: float):
        while True:
            await self._save_image()
            await asyncio.sleep(interval_secs)

    async def _save_image(self) -> Optional[str]:
        frame = await self.get_current_frame()
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
        self._publish_log_message(f"Saved image: {image_path}")
        return image_path

    @service("~/get_state", GetState)
    async def get_state(self):
        return result(state=self._get_state())

    @service("~/get_positions_for_pixels", GetPositionsForPixels)
    async def get_positions_for_pixels(self, pixels):
        frame = await self.get_current_frame()
        if frame is None:
            return result()

        positions = []
        for pixel in pixels:
            position = frame.get_position((pixel.x, pixel.y))
            positions.append(
                Vector3(x=position[0], y=position[1], z=position[2])
                if position is not None
                else Vector3(x=-1.0, y=-1.0, z=-1.0)
            )
        return result(positions=positions)

    # TODO: Define interval using param
    @timer(1.0 / 30, allow_concurrent_execution=False)
    async def frame_callback(self):
        async with self._camera_lock:
            # If the camera is not connected, set the current frame to None to prevent
            # further processing on frames.
            if not self.camera.is_connected:
                await self.set_current_frame(None)
                return

            frame = await asyncio.get_running_loop().run_in_executor(
                None,
                self.camera.get_frame,
            )
            # The camera may not have a new frame available yet. In that case, do nothing.
            if frame is None:
                return

            await self.set_current_frame(frame)

        # We're not currently consuming the color_frame topic anywhere. Disable it for now
        # to reduce CPU overhead
        """
        msg = self._get_color_frame_msg(frame.color_frame, frame.timestamp_millis)
        asyncio.create_task(
            self.color_frame_topic(
                header=msg.header,
                height=msg.height,
                width=msg.width,
                encoding=msg.encoding,
                is_bigendian=msg.is_bigendian,
                step=msg.step,
                data=msg.data,
            )
        )
        """

    # TODO: Define interval using param
    @timer(1.0 / 30, allow_concurrent_execution=False)
    async def detection_callback(self):
        frame = await self.get_current_frame()
        if frame is None:
            return

        debug_frame = np.copy(frame.color_frame)

        if self.laser_detection_enabled:
            laser_points, confs = await self._get_laser_points(frame.color_frame)
            debug_frame = self._debug_draw_lasers(debug_frame, laser_points, confs)
            msg = self._create_detection_result_msg(laser_points, frame)
            asyncio.create_task(
                self.laser_detections_topic(
                    timestamp=msg.timestamp,
                    instances=msg.instances,
                    invalid_points=msg.invalid_points,
                )
            )

        if self.runner_detection_enabled:
            runner_masks, confs, track_ids = await self._get_runner_masks(
                frame.color_frame
            )
            runner_centers = await self._get_runner_centers(runner_masks)
            debug_frame = self._debug_draw_runners(
                debug_frame, runner_masks, runner_centers, confs, track_ids
            )
            msg = self._create_detection_result_msg(runner_centers, frame, track_ids)
            asyncio.create_task(
                self.runner_detections_topic(
                    timestamp=msg.timestamp,
                    instances=msg.instances,
                    invalid_points=msg.invalid_points,
                )
            )

        # Downscale debug_frame using INTER_NEAREST for best performance
        h, w, _ = debug_frame.shape
        aspect_ratio = h / w
        new_width = self.camera_control_params.debug_frame_width
        new_height = int(new_width * aspect_ratio)
        debug_frame = cv2.resize(
            debug_frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        msg = self._get_color_frame_msg(debug_frame, frame.timestamp_millis)
        asyncio.create_task(
            self.debug_frame_topic(
                header=msg.header,
                height=msg.height,
                width=msg.width,
                encoding=msg.encoding,
                is_bigendian=msg.is_bigendian,
                step=msg.step,
                data=msg.data,
            )
        )

        if self.video_writer is not None:
            self.video_writer.write(cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))

    def _get_device_state(self) -> DeviceState:
        if self.connecting:
            return DeviceState.CONNECTING
        elif self.camera is not None and self.camera.is_connected:
            return DeviceState.STREAMING
        else:
            return DeviceState.DISCONNECTED

    def _get_state(self) -> State:
        state = State()
        device_state = DeviceState()
        device_state.data = self._get_device_state()
        state.device_state = device_state
        state.laser_detection_enabled = self.laser_detection_enabled
        state.runner_detection_enabled = self.runner_detection_enabled
        state.recording_video = self.video_writer is not None
        state.interval_capture_active = self.interval_capture_task is not None
        state.exposure_us = self.camera.exposure_us
        state.save_directory = self.camera_control_params.save_dir
        return state

    def _publish_state(self):
        state = self._get_state()
        asyncio.create_task(
            self.state_topic(
                device_state=state.device_state,
                laser_detection_enabled=state.laser_detection_enabled,
                runner_detection_enabled=state.runner_detection_enabled,
                recording_video=state.recording_video,
                interval_capture_active=state.interval_capture_active,
                exposure_us=state.exposure_us,
                save_directory=state.save_directory,
            )
        )

    def _publish_log_message(self, msg: str):
        # TODO: Support log levels
        timestamp_millis = int(time.time() * 1000)
        sec, nanosec = milliseconds_to_ros_time(timestamp_millis)
        log_message = Log()
        log_message.stamp.sec = sec
        log_message.stamp.nanosec = nanosec
        log_message.msg = msg
        asyncio.create_task(
            self.log_topic(stamp=log_message.stamp, msg=log_message.msg)
        )

    async def _get_laser_points(
        self, color_frame: np.ndarray, conf_threshold: float = 0.0
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        # Scale image before prediction to improve accuracy
        frame_width = color_frame.shape[1]
        frame_height = color_frame.shape[0]
        result_width = self.laser_detection_size[0]
        result_height = self.laser_detection_size[1]
        color_frame = cv2.resize(
            color_frame, self.laser_detection_size, interpolation=cv2.INTER_LINEAR
        )

        result = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                self.laser_detection_model.predict,
                color_frame,
            ),
        )
        result_conf = result["conf"]
        self.log(f"Laser point prediction found {result_conf.size} objects.")

        laser_points = []
        confs = []
        for idx in range(result["conf"].size):
            conf = result["conf"][idx]
            if conf >= conf_threshold:
                # bbox is in xyxy format
                bbox = result["bboxes"][idx]
                # Scale the result coords to frame coords
                laser_points.append(
                    (
                        round((bbox[0] + bbox[2]) * 0.5 * frame_width / result_width),
                        round((bbox[1] + bbox[3]) * 0.5 * frame_height / result_height),
                    )
                )
                confs.append(conf)
            else:
                self.log(
                    f"Laser point prediction ignored due to low confidence: {conf} < {conf_threshold}"
                )
        return laser_points, confs

    async def _get_runner_masks(
        self, color_frame: np.ndarray, conf_threshold: float = 0.0
    ) -> Tuple[List[np.ndarray], List[float], List[int]]:
        # Scale image before prediction to improve accuracy
        frame_width = color_frame.shape[1]
        frame_height = color_frame.shape[0]
        result_width = self.runner_seg_size[0]
        result_height = self.runner_seg_size[1]
        color_frame = cv2.resize(
            color_frame, self.runner_seg_size, interpolation=cv2.INTER_LINEAR
        )
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                self.runner_seg_model.track,
                color_frame,
            ),
        )
        result_conf = result["conf"]
        self.log(f"Runner mask prediction found {result_conf.size} objects.")

        runner_masks = []
        confs = []
        track_ids = []
        for idx in range(result_conf.size):
            conf = result_conf[idx]
            if conf >= conf_threshold:
                mask = result["masks"][idx]
                # Scale the result coords to frame coords
                mask[:, 0] *= frame_width / result_width
                mask[:, 1] *= frame_height / result_height
                runner_masks.append(mask)
                confs.append(conf)
                track_ids.append(
                    result["track_ids"][idx] if idx < len(result["track_ids"]) else -1
                )
            else:
                self.log(
                    f"Runner mask prediction ignored due to low confidence: {conf} < {conf_threshold}"
                )
        return runner_masks, confs, track_ids

    async def _get_runner_centers(
        self, runner_masks: List[np.ndarray]
    ) -> List[Tuple[int, int]]:
        runner_centers = []
        for mask in runner_masks:
            runner_center = await asyncio.get_running_loop().run_in_executor(
                None,
                functools.partial(
                    contour_center,
                    mask,
                ),
            )
            runner_centers.append(
                (runner_center[0], runner_center[1]) if runner_center else None
            )
        return runner_centers

    ## region Message builders

    def _create_detection_result_msg(
        self,
        points: List[Tuple[int, int]],
        frame: RgbdFrame,
        track_ids: Optional[List[int]] = None,
    ) -> DetectionResult:
        msg = DetectionResult()
        msg.timestamp = frame.timestamp_millis / 1000
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
        return msg

    def _get_color_frame_msg(
        self, color_frame: np.ndarray, timestamp_millis: float
    ) -> Image:
        msg = self.cv_bridge.cv2_to_imgmsg(color_frame, encoding="rgb8")
        sec, nanosec = milliseconds_to_ros_time(timestamp_millis)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        return msg

    def _get_depth_frame_msg(
        self, depth_frame: np.ndarray, timestamp_millis: float
    ) -> Image:
        msg = self.cv_bridge.cv2_to_imgmsg(depth_frame, encoding="mono16")
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

    ## endregion

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
                    thickness=1,
                    markerSize=20,
                )
            if draw_conf:
                pos = [int(runner_center[0]) + 15, int(runner_center[1]) - 5]
                font = cv2.FONT_HERSHEY_SIMPLEX
                debug_frame = cv2.putText(
                    debug_frame, f"{conf:.2f}", pos, font, 0.5, mask_color
                )
            if draw_track_id and track_id > 0:
                pos = [int(runner_center[0]) + 15, int(runner_center[1]) + 10]
                font = cv2.FONT_HERSHEY_SIMPLEX
                debug_frame = cv2.putText(
                    debug_frame, f"{track_id}", pos, font, 0.5, mask_color
                )
        return debug_frame


def main():
    serve_nodes(CameraControlNode())


if __name__ == "__main__":
    main()
