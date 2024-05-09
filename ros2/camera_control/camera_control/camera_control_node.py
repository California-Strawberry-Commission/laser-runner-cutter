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
from camera_control.camera.rgbd_frame import RgbdFrame
from camera_control_interfaces.msg import DetectionResult, ObjectInstance, State
from camera_control_interfaces.srv import (
    GetDetectionResult,
    GetFrame,
    GetPositionsForPixels,
    GetState,
    SetExposure,
)
from common_interfaces.msg import Vector2, Vector3
from common_interfaces.srv import GetBool


@dataclass
class CameraControlParams:
    camera_index: int = 0
    fps: int = 30
    rgb_size: List[int] = field(default_factory=lambda: [1280, 720])
    depth_size: List[int] = field(default_factory=lambda: [1280, 720])
    video_dir: str = "~/Videos/runner-cutter-app"
    image_dir: str = "~/Pictures/runner-cutter-app"


def milliseconds_to_ros_time(milliseconds):
    # ROS timestamps consist of two integers, one for seconds and one for nanoseconds
    seconds, remainder_ms = divmod(milliseconds, 1000)
    nanoseconds = remainder_ms * 1e6
    return int(seconds), int(nanoseconds)


@node("camera_control_node")
class CameraControlNode:
    camera_control_params = params(CameraControlParams)
    state_topic = topic("~/state", State, 5)
    color_frame_topic = topic("~/color_frame", Image, 1)
    debug_frame_topic = topic("~/debug_frame", Image, 1)
    laser_detections_topic = topic("~/laser_detections", DetectionResult, 5)
    runner_detections_topic = topic("~/runner_detections", DetectionResult, 5)

    @start
    async def start(self):
        self.laser_detection_enabled = False
        self.runner_detection_enabled = False
        self.video_writer = None
        # For converting numpy array to image msg
        self.cv_bridge = CvBridge()

        # Camera
        self.curr_frame = None
        self.camera = RealSense(
            self.camera_control_params.rgb_size,
            self.camera_control_params.depth_size,
            fps=self.camera_control_params.fps,
            camera_index=self.camera_control_params.camera_index,
            logger=self.get_logger(),
        )
        self.camera.initialize()

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

    @service("~/has_frames", GetBool)
    async def has_frames(self):
        return result(data=(self.curr_frame is not None))

    @service("~/get_frame", GetFrame)
    async def get_frame(self):
        if self.curr_frame is None:
            return result()

        return result(
            color_frame=self._get_color_frame_msg(
                self.curr_frame.color_frame, self.curr_frame.timestamp_millis
            ),
            depth_frame=self._get_depth_frame_msg(
                self.curr_frame.depth_frame, self.curr_frame.timestamp_millis
            ),
        )

    @service("~/set_exposure", SetExposure)
    async def set_exposure(self, exposure_us):
        self.camera.set_exposure(exposure_us)
        return result(success=True)

    @service("~/auto_exposure", Trigger)
    async def auto_exposure(self):
        self.camera.set_exposure(-1.0)
        return result(success=True)

    @service("~/get_laser_detection", GetDetectionResult)
    async def get_laser_detection(self):
        if self.curr_frame is None:
            return result()

        laser_points, conf = await self._get_laser_points(self.curr_frame.color_frame)
        return result(
            result=self._create_detection_result_msg(laser_points, self.curr_frame)
        )

    @service("~/get_runner_detection", GetDetectionResult)
    async def get_runner_detection(self):
        if self.curr_frame is None:
            return result()

        runner_masks, confs, track_ids = await self._get_runner_masks(
            self.curr_frame.color_frame
        )
        runner_centers = await self._get_runner_centers(runner_masks)
        return result(
            result=self._create_detection_result_msg(
                runner_centers, self.curr_frame, track_ids
            )
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
        os.makedirs(self.camera_control_params.video_dir, exist_ok=True)
        ts = time.time()
        datetime_obj = datetime.fromtimestamp(ts)
        datetime_string = datetime_obj.strftime("%Y%m%d%H%M%S")
        video_name = f"{datetime_string}.avi"
        self.video_writer = cv2.VideoWriter(
            os.path.join(self.camera_control_params.video_dir, video_name),
            cv2.VideoWriter_fourcc(*"XVID"),
            self.camera_control_params.fps,
            (
                self.camera_control_params.rgb_size[0],
                self.camera_control_params.rgb_size[1],
            ),
        )
        self._publish_state()
        return result(success=True)

    @service("~/stop_recording_video", Trigger)
    async def stop_recording_video(self):
        self.video_writer = None
        self._publish_state()
        return result(success=True)

    @service("~/save_image", Trigger)
    async def save_image(self):
        if self.curr_frame is None:
            return result()

        os.makedirs(self.camera_control_params.image_dir, exist_ok=True)
        ts = time.time()
        datetime_obj = datetime.fromtimestamp(ts)
        datetime_string = datetime_obj.strftime("%Y%m%d%H%M%S")
        image_name = f"{datetime_string}.png"
        cv2.imwrite(
            os.path.join(self.camera_control_params.image_dir, image_name),
            cv2.cvtColor(self.curr_frame.color_frame, cv2.COLOR_RGB2BGR),
        )
        return result(success=True)

    @service("~/get_state", GetState)
    async def get_state(self):
        return result(state=self._get_state())

    @service("~/get_positions_for_pixels", GetPositionsForPixels)
    async def get_positions_for_pixels(self, pixels):
        if self.curr_frame is None:
            return result()

        positions = []
        if self.curr_frame is not None:
            for pixel in pixels:
                position = self.curr_frame.get_position((pixel.x, pixel.y))
                positions.append(
                    Vector3(x=position[0], y=position[1], z=position[2])
                    if position is not None
                    else Vector3(x=-1.0, y=-1.0, z=-1.0)
                )
        return result(positions=positions)

    # TODO: Define interval using param
    @timer(1.0 / 30, allow_concurrent_execution=False)
    async def frame_callback(self):
        frame = self.camera.get_frame()
        if not frame:
            return

        self.curr_frame = frame

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

    # TODO: Define interval using param
    @timer(1.0 / 30, allow_concurrent_execution=False)
    async def detection_callback(self):
        if not self.curr_frame:
            return

        frame = self.curr_frame
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

    def _get_state(self) -> State:
        state = State()
        state.connected = self.camera is not None
        state.laser_detection_enabled = self.laser_detection_enabled
        state.runner_detection_enabled = self.runner_detection_enabled
        state.recording_video = self.video_writer is not None
        return state

    def _publish_state(self):
        state = self._get_state()
        asyncio.create_task(
            self.state_topic(
                connected=state.connected,
                laser_detection_enabled=state.laser_detection_enabled,
                runner_detection_enabled=state.runner_detection_enabled,
                recording_video=state.recording_video,
            )
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
            runner_centers.append((runner_center[0], runner_center[1]))
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
