import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from cv_bridge import CvBridge
from rcl_interfaces.msg import Log
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_srvs.srv import Trigger

import aioros2
from camera_control.camera.lucid_camera import CaptureMode as LucidCaptureMode
from camera_control.camera.lucid_camera import LucidRgbdCamera, create_lucid_rgbd_camera
from camera_control.camera.realsense_camera import RealSenseCamera
from camera_control.camera.rgbd_camera import RgbdCamera
from camera_control.camera.rgbd_camera import State as RgbdCameraState
from camera_control.camera.rgbd_frame import RgbdFrame
from camera_control.detector.circle_detector import CircleDetector
from camera_control.detector.laser_detector import LaserDetector
from camera_control.detector.runner_detector import RunnerDetector
from camera_control_interfaces.msg import (
    CaptureMode,
    DetectionResult,
    DetectionType,
    DeviceState,
    ObjectInstance,
    State,
)
from camera_control_interfaces.srv import (
    AcquireSingleFrame,
    GetDetectionResult,
    GetFrame,
    GetPositions,
    GetState,
    SetExposure,
    SetGain,
    SetSaveDirectory,
    StartDetection,
    StartDevice,
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
    save_dir: str = "~/runner_cutter/camera"
    debug_frame_width: int = 640
    debug_video_fps: float = 30.0
    image_capture_interval_secs: float = 5.0


camera_control_params = aioros2.params(CameraControlParams)
state_topic = aioros2.topic("~/state", State, qos=aioros2.QOS_LATCHED)
debug_frame_topic = aioros2.topic("~/debug_frame", Image, qos=qos_profile_sensor_data)
detections_topic = aioros2.topic(
    "~/detections", DetectionResult, qos=qos_profile_sensor_data
)
notifications_topic = aioros2.topic("/notifications", Log, qos=1)


class SharedState:
    logger: Optional[logging.Logger] = None
    # DetectionType -> normalized rect bounds (min x, min y, width, height)
    enabled_detections: Dict[int, Tuple[float, float, float, float]] = {}
    record_video_task: Optional[asyncio.Task] = None
    video_writer: Optional[cv2.VideoWriter] = None
    debug_frame: Optional[np.ndarray] = None
    interval_capture_task: Optional[asyncio.Task] = None
    # For converting numpy array to image msg
    cv_bridge = CvBridge()

    # Camera

    camera: Optional[RgbdCamera] = None

    # After starting a camera device, when a new frame is available, a callback is called by
    # the camera interface. When a new frame is received, we queue a detection task. In order
    # to prevent multiple detection tasks running concurrently, we use a task queue of size 1
    # and a queue processor task. One additional consideration is that the camera device can
    # be closed at any time, and therefore we need to make sure that the detection task does
    # not use a frame after the device has been closed.

    # We don't need locks for these since read/write only happens on main event loop
    current_frame: Optional[RgbdFrame] = None
    detection_task_queue = asyncio.Queue(1)
    # Used to notify when a detection task has completed, used to avoid destroying frame buffer(s)
    # while detection is happening
    detection_completed_event = asyncio.Event()
    detection_completed_event.set()
    # Flag to prevent frame callback or detection task from executing after device is closed
    camera_started = False
    # Used to notify when a new frame is available
    frame_event = asyncio.Event()

    # Detectors

    runner_detector: Optional[RunnerDetector] = None
    laser_detector: Optional[LaserDetector] = None
    circle_detector: Optional[CircleDetector] = None


shared_state = SharedState()


@aioros2.start
async def start(node):
    shared_state.logger = node.get_logger()

    async def process_detection_task_queue():
        while True:
            task_func = await shared_state.detection_task_queue.get()
            await task_func()
            shared_state.detection_task_queue.task_done()

    loop = asyncio.get_running_loop()
    loop.create_task(process_detection_task_queue())

    def state_change_callback(state: RgbdCameraState):
        # This callback is called from another thread, so we need to use call_soon_threadsafe
        loop.call_soon_threadsafe(_publish_state)

    if camera_control_params.camera_type == "realsense":
        shared_state.camera = RealSenseCamera(
            camera_index=camera_control_params.camera_index,
            state_change_callback=state_change_callback,
            logger=shared_state.logger,
        )
    elif camera_control_params.camera_type == "lucid":
        shared_state.camera = create_lucid_rgbd_camera(
            state_change_callback=state_change_callback,
            logger=shared_state.logger,
        )
    else:
        raise Exception(f"Unknown camera_type: {camera_control_params.camera_type}")

    # Detectors
    shared_state.runner_detector = RunnerDetector(logger=shared_state.logger)
    shared_state.laser_detector = LaserDetector(logger=shared_state.logger)
    shared_state.circle_detector = CircleDetector(logger=shared_state.logger)

    # Publish initial state
    _publish_state()


@aioros2.service("~/start_device", StartDevice)
async def start_device(node, capture_mode):
    loop = asyncio.get_running_loop()

    def frame_callback(frame: RgbdFrame):
        # This callback is called from another thread, so we need to use run_coroutine_threadsafe
        asyncio.run_coroutine_threadsafe(_frame_callback(frame), loop)

    is_lucid = isinstance(shared_state.camera, LucidRgbdCamera)
    capture_mode_arg = (
        None
        if not is_lucid
        else (
            LucidCaptureMode.SINGLE_FRAME
            if capture_mode == CaptureMode.SINGLE_FRAME
            else LucidCaptureMode.CONTINUOUS
        )
    )
    shared_state.camera.start(
        exposure_us=camera_control_params.exposure_us,
        gain_db=camera_control_params.gain_db,
        frame_callback=frame_callback,
        **({"capture_mode": capture_mode_arg} if capture_mode_arg is not None else {}),
    )

    shared_state.camera_started = True

    return {"success": True}


@aioros2.service("~/close_device", Trigger)
async def close_device(node):
    shared_state.camera_started = False
    _publish_state()

    # Wait until detection completes
    await shared_state.detection_completed_event.wait()
    shared_state.camera.stop()
    shared_state.current_frame = None

    return {"success": True}


@aioros2.service("~/has_frames", GetBool)
async def has_frames(node):
    frame = shared_state.current_frame
    return {"data": (frame is not None)}


@aioros2.service("~/get_frame", GetFrame)
async def get_frame(node):
    frame = shared_state.current_frame
    if frame is None:
        return {}

    return {
        "color_frame": _get_color_frame_msg(frame.color_frame, frame.timestamp_millis),
        "depth_frame": _get_depth_frame_msg(frame.depth_frame, frame.timestamp_millis),
    }


@aioros2.service("~/acquire_single_frame", AcquireSingleFrame)
async def acquire_single_frame(node):
    frame = await asyncio.get_running_loop().run_in_executor(
        None, shared_state.camera.get_frame
    )
    if frame is None:
        _publish_notification("Failed to acquire frame", level=logging.ERROR)
        return {}

    await _frame_callback(frame)
    _publish_notification("Successfully acquired frame")
    return {
        "preview_image": _get_color_frame_compressed_msg(
            frame.color_frame, frame.timestamp_millis
        )
    }


# TODO: Use param instead of service for setting exposure, gain, and save dir
@aioros2.service("~/set_exposure", SetExposure)
async def set_exposure(node, exposure_us):
    camera_control_params.exposure_us = exposure_us
    shared_state.camera.exposure_us = exposure_us
    _publish_state()
    return {"success": True}


@aioros2.service("~/auto_exposure", Trigger)
async def auto_exposure(node):
    camera_control_params.exposure_us = -1.0
    shared_state.camera.exposure_us = -1.0
    _publish_state()
    return {"success": True}


@aioros2.service("~/set_gain", SetGain)
async def set_gain(node, gain_db):
    camera_control_params.gain_db = gain_db
    shared_state.camera.gain_db = gain_db
    _publish_state()
    return {"success": True}


@aioros2.service("~/auto_gain", Trigger)
async def auto_gain(node):
    camera_control_params.gain_db = -1.0
    shared_state.camera.gain_db = -1.0
    _publish_state()
    return {"success": True}


@aioros2.service("~/get_detection", GetDetectionResult)
async def get_detection(node, detection_type, wait_for_next_frame):
    if wait_for_next_frame:
        # Wait for two frames, as the next frame to arrive may have already been in progress
        frame_count = 0
        while frame_count < 2:
            shared_state.frame_event.clear()
            await shared_state.frame_event.wait()
            frame_count += 1

    frame = shared_state.current_frame

    if frame is None:
        return {}

    if detection_type == DetectionType.LASER:
        laser_points, confs = await shared_state.laser_detector.detect(
            frame.color_frame
        )
        return {
            "result": _create_detection_result_msg(
                detection_type, laser_points, confs, frame
            )
        }
    elif detection_type == DetectionType.RUNNER:
        runner_masks, runner_centers, confs, track_ids = (
            await shared_state.runner_detector.detect(frame.color_frame)
        )
        # runner_centers may contain None elements, so filter them out and also remove
        # the corresponding elements from track_ids
        filtered = [
            (center, conf, track_id)
            for center, conf, track_id in zip(runner_centers, confs, track_ids)
            if center is not None
        ]
        if filtered:
            runner_centers, confs, track_ids = zip(*filtered)
            runner_centers = list(runner_centers)
            confs = list(confs)
            track_ids = list(track_ids)
        else:
            runner_centers = []
            confs = []
            track_ids = []
        return {
            "result": _create_detection_result_msg(
                detection_type, runner_centers, confs, frame, track_ids
            )
        }
    elif detection_type == DetectionType.CIRCLE:
        circle_centers = await shared_state.circle_detector.detect(frame.color_frame)
        confs = [1.0 for _ in circle_centers]
        return {
            "result": _create_detection_result_msg(
                detection_type,
                circle_centers,
                confs,
                frame,
                [i + 1 for i in range(len(circle_centers))],
            )
        }
    else:
        return {}


@aioros2.service("~/start_detection", StartDetection)
async def start_detection(node, detection_type, normalized_bounds):
    if detection_type not in shared_state.enabled_detections:
        # If normalized bounds are not defined, set to full bounds (0, 0, 1, 1)
        shared_state.enabled_detections[detection_type] = (
            (0.0, 0.0, 1.0, 1.0)
            if (
                normalized_bounds.w == 0.0
                and normalized_bounds.x == 0.0
                and normalized_bounds.y == 0.0
                and normalized_bounds.z == 0.0
            )
            else (
                normalized_bounds.w,
                normalized_bounds.x,
                normalized_bounds.y,
                normalized_bounds.z,
            )
        )
        _publish_state()
        return {"success": True}

    return {"success": False}


@aioros2.service("~/stop_detection", StopDetection)
async def stop_detection(node, detection_type):
    if detection_type in shared_state.enabled_detections:
        shared_state.enabled_detections.pop(detection_type, None)
        _publish_state()
        return {"success": True}

    return {"success": False}


@aioros2.service("~/stop_all_detections", Trigger)
async def stop_all_detections(node):
    if len(shared_state.enabled_detections) > 0:
        shared_state.enabled_detections.clear()
        _publish_state()
        return {"success": True}

    return {"success": False}


@aioros2.service("~/start_recording_video", Trigger)
async def start_recording_video(node):
    if shared_state.record_video_task is not None:
        shared_state.record_video_task.cancel()
        shared_state.record_video_task = None

    shared_state.record_video_task = asyncio.create_task(
        _record_video_task(camera_control_params.debug_video_fps)
    )
    _publish_state()
    return {"success": True}


@aioros2.service("~/stop_recording_video", Trigger)
async def stop_recording_video(node):
    if shared_state.record_video_task is None:
        return {"success": False}

    shared_state.record_video_task.cancel()
    shared_state.record_video_task = None
    _publish_state()
    _publish_notification("Stopped recording video")
    return {"success": True}


@aioros2.service("~/save_image", Trigger)
async def save_image(node):
    if _save_image() is None:
        return {"success": False}

    return {"success": True}


@aioros2.service("~/start_interval_capture", StartIntervalCapture)
async def start_interval_capture(node, interval_secs):
    if shared_state.interval_capture_task is not None:
        shared_state.interval_capture_task.cancel()
        shared_state.interval_capture_task = None

    camera_control_params.image_capture_interval_secs = interval_secs
    shared_state.interval_capture_task = asyncio.create_task(
        _interval_capture_task(interval_secs)
    )
    _publish_state()
    _publish_notification(f"Started interval capture with {interval_secs}s interval")
    return {"success": True}


@aioros2.service("~/stop_interval_capture", Trigger)
async def stop_interval_capture(node):
    if shared_state.interval_capture_task is None:
        return {"success": False}

    shared_state.interval_capture_task.cancel()
    shared_state.interval_capture_task = None
    _publish_state()
    _publish_notification("Stopped interval capture")
    return {"success": True}


@aioros2.service("~/set_save_directory", SetSaveDirectory)
async def set_save_directory(node, save_directory):
    camera_control_params.save_dir = save_directory
    _publish_state()
    return {"success": True}


@aioros2.service("~/get_state", GetState)
async def get_state(node):
    return {"state": _get_state()}


@aioros2.service("~/get_positions", GetPositions)
async def get_positions(node, normalized_pixel_coords):
    frame = shared_state.current_frame
    if frame is None:
        return {}

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
    return {"positions": positions}


# region Task definitions


async def _frame_callback(frame: RgbdFrame):
    if (
        not shared_state.camera_started
        or shared_state.camera.state != RgbdCameraState.STREAMING
    ):
        return

    shared_state.current_frame = frame
    shared_state.frame_event.set()

    if shared_state.detection_task_queue.empty():
        await shared_state.detection_task_queue.put(_detection_task)


async def _detection_task():
    if (
        not shared_state.camera_started
        or shared_state.camera.state != RgbdCameraState.STREAMING
    ):
        return

    try:
        shared_state.detection_completed_event.clear()

        frame = shared_state.current_frame
        if frame is None:
            return

        debug_frame = np.copy(frame.color_frame)

        if DetectionType.LASER in shared_state.enabled_detections:
            laser_points, confs = await shared_state.laser_detector.detect(
                frame.color_frame
            )
            debug_frame = _debug_draw_lasers(debug_frame, laser_points, confs)
            msg = _create_detection_result_msg(
                DetectionType.LASER, laser_points, confs, frame
            )
            detections_topic.publish(msg)

        if DetectionType.RUNNER in shared_state.enabled_detections:
            # Note: if bounds is defined, the runners' representative points are calculated to be
            # the point on the runner closest to the centroid of only the portion of the mask that
            # lies within the bounds. Thus, if a runner lies completely outside the bounds, its
            # representative point will be None.
            normalized_bounds = shared_state.enabled_detections.get(
                DetectionType.RUNNER, (0.0, 0.0, 1.0, 1.0)
            )
            # Denormalize bounds to frame size and detect runners
            width = frame.color_frame.shape[1]
            height = frame.color_frame.shape[0]
            runner_masks, runner_representative_points, confs, track_ids = (
                await shared_state.runner_detector.detect(
                    frame.color_frame,
                    bounds=(
                        math.ceil(normalized_bounds[0] * width),
                        math.ceil(normalized_bounds[1] * height),
                        math.floor(normalized_bounds[2] * width),
                        math.floor(normalized_bounds[3] * height),
                    ),
                )
            )
            debug_frame = _debug_draw_runners(
                debug_frame,
                runner_masks,
                runner_representative_points,
                confs,
                track_ids,
            )
            # runner_representative_points may contain None elements, so filter them out and also
            # remove the corresponding elements from track_ids
            filtered = [
                (center, conf, track_id)
                for center, conf, track_id in zip(
                    runner_representative_points, confs, track_ids
                )
                if center is not None
            ]
            if filtered:
                runner_representative_points, confs, track_ids = zip(*filtered)
                runner_representative_points = list(runner_representative_points)
                confs = list(confs)
                track_ids = list(track_ids)
            else:
                runner_representative_points = []
                confs = []
                track_ids = []
            msg = _create_detection_result_msg(
                DetectionType.RUNNER,
                runner_representative_points,
                confs,
                frame,
                track_ids,
            )
            detections_topic.publish(msg)

        if DetectionType.CIRCLE in shared_state.enabled_detections:
            circle_centers = await shared_state.circle_detector.detect(
                frame.color_frame
            )
            confs = [1.0 for _ in circle_centers]
            debug_frame = _debug_draw_circles(debug_frame, circle_centers)
            msg = _create_detection_result_msg(
                DetectionType.CIRCLE,
                circle_centers,
                confs,
                frame,
                [i + 1 for i in range(len(circle_centers))],
            )
            detections_topic.publish(msg)

        debug_frame = _debug_draw_timestamp(debug_frame, frame.timestamp_millis)

        # Downscale debug_frame using INTER_NEAREST for best performance
        h, w, _ = debug_frame.shape
        aspect_ratio = h / w
        new_width = camera_control_params.debug_frame_width
        new_height = int(new_width * aspect_ratio)
        debug_frame = cv2.resize(
            debug_frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )

        shared_state.debug_frame = debug_frame

        msg = _get_color_frame_msg(debug_frame, frame.timestamp_millis)
        debug_frame_topic.publish(msg)
    finally:
        shared_state.detection_completed_event.set()


async def _interval_capture_task(interval_secs: float):
    while True:
        _save_image()
        await asyncio.sleep(interval_secs)


def _save_image() -> Optional[str]:
    frame = shared_state.current_frame
    if frame is None:
        return None

    save_dir = os.path.expanduser(camera_control_params.save_dir)
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
    _publish_notification(f"Saved image: {image_path}")
    return image_path


async def _record_video_task(fps: float):
    try:
        while True:
            _write_video_frame()
            await asyncio.sleep(1 / fps)
    finally:
        shared_state.video_writer = None


def _write_video_frame():
    if shared_state.video_writer is not None:
        shared_state.video_writer.write(
            cv2.cvtColor(shared_state.debug_frame, cv2.COLOR_RGB2BGR)
        )
    elif shared_state.debug_frame is not None:
        save_dir = os.path.expanduser(camera_control_params.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        ts = time.time()
        datetime_obj = datetime.fromtimestamp(ts)
        datetime_string = datetime_obj.strftime("%Y%m%d%H%M%S")
        video_name = f"{datetime_string}.avi"
        video_path = os.path.join(save_dir, video_name)
        h, w, _ = shared_state.debug_frame.shape
        shared_state.video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"XVID"),
            camera_control_params.debug_video_fps,
            (w, h),
        )
        _publish_notification(f"Started recording video: {video_path}")


# endregion

# region State and notifs publishing


def _get_device_state() -> DeviceState:
    if shared_state.camera is None:
        return DeviceState.DISCONNECTED

    if shared_state.camera.state == RgbdCameraState.CONNECTING:
        return DeviceState.CONNECTING
    elif shared_state.camera.state == RgbdCameraState.STREAMING:
        if shared_state.camera_started:
            return DeviceState.STREAMING
        else:
            return DeviceState.DISCONNECTING
    else:
        return DeviceState.DISCONNECTED


def _get_state() -> State:
    state = State()
    state.device_state = _get_device_state()
    state.enabled_detection_types = list(shared_state.enabled_detections.keys())
    state.recording_video = shared_state.record_video_task is not None
    state.interval_capture_active = shared_state.interval_capture_task is not None
    state.exposure_us = shared_state.camera.exposure_us
    exposure_us_range = shared_state.camera.get_exposure_us_range()
    state.exposure_us_range = Vector2(x=exposure_us_range[0], y=exposure_us_range[1])
    state.gain_db = shared_state.camera.gain_db
    gain_db_range = shared_state.camera.get_gain_db_range()
    state.gain_db_range = Vector2(x=gain_db_range[0], y=gain_db_range[1])
    state.save_directory = camera_control_params.save_dir
    state.image_capture_interval_secs = (
        camera_control_params.image_capture_interval_secs
    )
    return state


def _publish_state():
    state_topic.publish(_get_state())


def _publish_notification(msg: str, level: int = logging.INFO):
    timestamp_millis = int(time.time() * 1000)
    sec, nanosec = milliseconds_to_ros_time(timestamp_millis)
    log_message = Log()
    log_message.stamp.sec = sec
    log_message.stamp.nanosec = nanosec
    log_message.level = level
    log_message.msg = msg
    shared_state.logger.log(msg, level)
    notifications_topic.publish(log_message)


# endregion

# region Message builders


def _create_detection_result_msg(
    detection_type: int,
    points: List[Tuple[int, int]],
    confs: List[float],
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
                if track_ids is not None and idx < len(track_ids) and track_ids[idx] > 0
                else 0
            )
            object_instance.position = Vector3(
                x=position[0], y=position[1], z=position[2]
            )
            object_instance.point = point_msg
            object_instance.confidence = confs[idx]
            msg.instances.append(object_instance)
        else:
            msg.invalid_points.append(point_msg)
    shared_state.logger.debug(
        f"{len(msg.instances)} instances had valid positions, out of {len(points)} total detected"
    )
    return msg


def _get_color_frame_msg(color_frame: np.ndarray, timestamp_millis: float) -> Image:
    msg = shared_state.cv_bridge.cv2_to_imgmsg(color_frame, encoding="rgb8")
    sec, nanosec = milliseconds_to_ros_time(timestamp_millis)
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    return msg


def _get_depth_frame_msg(depth_frame: np.ndarray, timestamp_millis: float) -> Image:
    msg = shared_state.cv_bridge.cv2_to_imgmsg(depth_frame, encoding="mono16")
    sec, nanosec = milliseconds_to_ros_time(timestamp_millis)
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    return msg


def _get_color_frame_compressed_msg(
    color_frame: np.ndarray, timestamp_millis: float
) -> CompressedImage:
    sec, nanosec = milliseconds_to_ros_time(timestamp_millis)
    _, jpeg_data = cv2.imencode(".jpg", cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR))
    msg = CompressedImage()
    msg.format = "jpeg"
    msg.data = jpeg_data.tobytes()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    return msg


# endregion

# region Debug frame drawing


def _debug_draw_lasers(
    debug_frame, laser_points, confs, color=(255, 0, 255), draw_conf=True
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
            debug_frame = cv2.putText(debug_frame, f"{conf:.2f}", pos, font, 0.5, color)
    return debug_frame


def _debug_draw_runners(
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

        if runner_center is None:
            continue

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


def _debug_draw_circles(debug_frame, circle_centers, color=(255, 0, 255)):
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


def _debug_draw_timestamp(debug_frame, timestamp, color=(255, 255, 255)):
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
    aioros2.run()


if __name__ == "__main__":
    main()
