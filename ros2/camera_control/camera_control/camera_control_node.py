import os
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from runner_segmentation.yolo import Yolo
from sensor_msgs.msg import CompressedImage
from shapely import Polygon
from shapely.ops import nearest_points
from std_srvs.srv import Trigger

from camera_control.camera.realsense import RealSense
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


def milliseconds_to_ros_time(milliseconds):
    # ROS timestamps consist of two integers, one for seconds and one for nanoseconds
    seconds, remainder_ms = divmod(milliseconds, 1000)
    nanoseconds = remainder_ms * 1e6
    return int(seconds), int(nanoseconds)


class CameraControlNode(Node):
    def __init__(self):
        super().__init__("camera_control_node")
        self.logger = self.get_logger()

        # Parameters

        self.declare_parameters(
            namespace="",
            parameters=[
                ("camera_index", 0),
                ("fps", 30),
                ("rgb_size", [1280, 720]),
                ("depth_size", [1280, 720]),
                ("video_dir", "~/Videos/runner-cutter-app"),
                ("image_dir", "~/Pictures/runner-cutter-app"),
            ],
        )
        self.camera_index = (
            self.get_parameter("camera_index").get_parameter_value().integer_value
        )
        self.fps = self.get_parameter("fps").get_parameter_value().integer_value
        self.rgb_size = (
            self.get_parameter("rgb_size").get_parameter_value().integer_array_value
        )
        self.depth_size = (
            self.get_parameter("depth_size").get_parameter_value().integer_array_value
        )
        self.video_dir = os.path.expanduser(
            self.get_parameter("video_dir").get_parameter_value().string_value
        )
        self.image_dir = os.path.expanduser(
            self.get_parameter("image_dir").get_parameter_value().string_value
        )

        # Pub/sub

        self.state_pub = self.create_publisher(State, "~/state", 5)
        self.color_frame_pub = self.create_publisher(
            CompressedImage, "~/color_frame", 1
        )
        self.debug_frame_pub = self.create_publisher(
            CompressedImage, "~/debug_frame", 1
        )
        self.laser_detections_pub = self.create_publisher(
            DetectionResult, "~/laser_detections", 5
        )
        self.runner_detections_pub = self.create_publisher(
            DetectionResult, "~/runner_detections", 5
        )

        # Services

        self.has_frames_srv = self.create_service(
            GetBool, "~/has_frames", self._has_frames_callback
        )
        self.get_frame_srv = self.create_service(
            GetFrame, "~/get_frame", self._get_frame_callback
        )
        self.set_exposure_srv = self.create_service(
            SetExposure, "~/set_exposure", self._set_exposure_callback
        )
        self.get_laser_detection_srv = self.create_service(
            GetDetectionResult,
            "~/get_laser_detection",
            self._single_laser_detection_callback,
        )
        self.get_runner_detection_srv = self.create_service(
            GetDetectionResult,
            "~/get_runner_detection",
            self._single_runner_detection_callback,
        )
        self.laser_detection_enabled = False
        self.start_laser_detection_srv = self.create_service(
            Trigger, "~/start_laser_detection", self._start_laser_detection_callback
        )
        self.stop_laser_detection_srv = self.create_service(
            Trigger, "~/stop_laser_detection", self._stop_laser_detection_callback
        )
        self.runner_detection_enabled = False
        self.start_runner_detection_srv = self.create_service(
            Trigger, "~/start_runner_detection", self._start_runner_detection_callback
        )
        self.stop_runner_detection_srv = self.create_service(
            Trigger, "~/stop_runner_detection", self._stop_runner_detection_callback
        )
        self.start_recording_video_srv = self.create_service(
            Trigger, "~/start_recording_video", self._start_recording_video_callback
        )
        self.stop_recording_video_srv = self.create_service(
            Trigger, "~/stop_recording_video", self._stop_recording_video_callback
        )
        self.stop_recording_srv = self.create_service(
            Trigger, "~/save_image", self._save_image_callback
        )
        self.get_state_srv = self.create_service(
            GetState, "~/get_state", self._get_state_callback
        )
        self.get_positions_for_pixels_srv = self.create_service(
            GetPositionsForPixels,
            "~/get_positions_for_pixels",
            self._get_positions_for_pixels_callback,
        )

        # Frame processing loop
        self.frame_call = self.create_timer(1.0 / self.fps, self._frame_callback)
        self.video_writer = None

        # For converting numpy array to image msg
        self.cv_bridge = CvBridge()

        # Camera

        self.curr_frames = None
        self.camera = RealSense(
            self.rgb_size,
            self.depth_size,
            fps=self.fps,
            camera_index=self.camera_index,
            logger=self.logger,
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

    def get_state(self):
        state = State()
        state.connected = self.camera is not None
        state.laser_detection_enabled = self.laser_detection_enabled
        state.runner_detection_enabled = self.runner_detection_enabled
        state.recording_video = self.video_writer is not None
        return state

    def _frame_callback(self):
        frames = self.camera.get_frames()
        if not frames:
            return

        self.curr_frames = frames
        color_frame = np.asanyarray(frames["color"].get_data())
        debug_frame = np.copy(color_frame)
        depth_frame = np.asanyarray(frames["depth"].get_data())
        timestamp_millis = frames["timestamp"]

        self.color_frame_pub.publish(
            self._get_color_frame_compressed_msg(color_frame, timestamp_millis)
        )

        if self.laser_detection_enabled:
            laser_points, confs = self._get_laser_points(color_frame)
            debug_frame = self._debug_draw_lasers(debug_frame, laser_points, confs)
            self.laser_detections_pub.publish(
                self._create_detection_result_msg(laser_points, frames)
            )

        if self.runner_detection_enabled:
            runner_masks, confs, track_ids = self._get_runner_masks(color_frame)
            runner_centroids = self._get_runner_centroids(runner_masks)
            debug_frame = self._debug_draw_runners(
                debug_frame, runner_masks, runner_centroids, confs, track_ids
            )
            self.runner_detections_pub.publish(
                self._create_detection_result_msg(runner_centroids, frames, track_ids)
            )

        self.debug_frame_pub.publish(
            self._get_debug_frame_compressed_msg(debug_frame, timestamp_millis)
        )

        if self.video_writer is not None:
            self.video_writer.write(cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))

    def _get_laser_points(self, color_frame, conf_threshold=0.0):
        # Scale image before prediction to improve accuracy
        frame_width = color_frame.shape[1]
        frame_height = color_frame.shape[0]
        result_width = self.laser_detection_size[0]
        result_height = self.laser_detection_size[1]
        color_frame = cv2.resize(
            color_frame, self.laser_detection_size, interpolation=cv2.INTER_LINEAR
        )
        result = self.laser_detection_model.predict(color_frame)
        result_conf = result["conf"]
        self.logger.info(f"Laser point prediction found {result_conf.size} objects.")

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
                self.logger.info(
                    f"Laser point prediction ignored due to low confidence: {conf} < {conf_threshold}"
                )
        return laser_points, confs

    def _get_runner_masks(self, color_frame, conf_threshold=0.0):
        # Scale image before prediction to improve accuracy
        frame_width = color_frame.shape[1]
        frame_height = color_frame.shape[0]
        result_width = self.runner_seg_size[0]
        result_height = self.runner_seg_size[1]
        color_frame = cv2.resize(
            color_frame, self.runner_seg_size, interpolation=cv2.INTER_LINEAR
        )
        result = self.runner_seg_model.track(color_frame)
        result_conf = result["conf"]
        self.logger.info(f"Runner mask prediction found {result_conf.size} objects.")

        self.logger.info(str(result["track_ids"]))

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
                self.logger.info(
                    f"Runner mask prediction ignored due to low confidence: {conf} < {conf_threshold}"
                )
        return runner_masks, confs, track_ids

    def _get_runner_centroids(self, runner_masks):
        centroids = []
        for mask in runner_masks:
            polygon = Polygon(mask)
            closest_polygon_point, closest_point = nearest_points(
                polygon, polygon.centroid
            )
            centroids.append((closest_polygon_point.x, closest_polygon_point.y))
        return centroids

    def _publish_state(self):
        self.state_pub.publish(self.get_state())

    ## region Service calls

    def _has_frames_callback(self, request, response):
        response.data = self.curr_frames is not None
        return response

    def _get_frame_callback(self, request, response):
        if self.curr_frames is not None:
            color_frame = np.asanyarray(self.curr_frames["color"].get_data())
            depth_frame = np.asanyarray(self.curr_frames["depth"].get_data())
            timestamp_millis = self.curr_frames["timestamp"]
            response.color_frame = self._get_color_frame_msg(
                color_frame, timestamp_millis
            )
            response.depth_frame = self._get_depth_frame_msg(
                depth_frame, timestamp_millis
            )
        return response

    def _set_exposure_callback(self, request, response):
        self.camera.set_exposure(request.exposure_ms)
        response.success = True
        return response

    def _single_laser_detection_callback(self, request, response):
        if self.curr_frames is not None:
            color_frame = np.asanyarray(self.curr_frames["color"].get_data())
            laser_points, conf = self._get_laser_points(color_frame)
            response.result = self._create_detection_result_msg(
                laser_points, self.curr_frames
            )
        return response

    def _single_runner_detection_callback(self, request, response):
        if self.curr_frames is not None:
            color_frame = np.asanyarray(self.curr_frames["color"].get_data())
            runner_masks, confs, track_ids = self._get_runner_masks(color_frame)
            runner_centroids = self._get_runner_centroids(runner_masks)
            response.result = self._create_detection_result_msg(
                runner_centroids, self.curr_frames, track_ids
            )
        return response

    def _start_laser_detection_callback(self, request, response):
        self.laser_detection_enabled = True
        self._publish_state()
        response.success = True
        return response

    def _stop_laser_detection_callback(self, request, response):
        self.laser_detection_enabled = False
        self._publish_state()
        response.success = True
        return response

    def _start_runner_detection_callback(self, request, response):
        self.runner_detection_enabled = True
        self._publish_state()
        response.success = True
        return response

    def _stop_runner_detection_callback(self, request, response):
        self.runner_detection_enabled = False
        self._publish_state()
        response.success = True
        return response

    def _start_recording_video_callback(self, request, response):
        os.makedirs(self.video_dir, exist_ok=True)
        ts = time.time()
        datetime_obj = datetime.fromtimestamp(ts)
        datetime_string = datetime_obj.strftime("%Y%m%d%H%M%S")
        video_name = f"{datetime_string}.avi"
        self.video_writer = cv2.VideoWriter(
            os.path.join(self.video_dir, video_name),
            cv2.VideoWriter_fourcc(*"XVID"),
            self.fps,
            (self.rgb_size[0], self.rgb_size[1]),
        )
        self._publish_state()
        response.success = True
        return response

    def _stop_recording_video_callback(self, request, response):
        self.video_writer = None
        self._publish_state()
        response.success = True
        return response

    def _save_image_callback(self, request, response):
        if self.curr_frames is not None:
            os.makedirs(self.image_dir, exist_ok=True)
            ts = time.time()
            datetime_obj = datetime.fromtimestamp(ts)
            datetime_string = datetime_obj.strftime("%Y%m%d%H%M%S")
            image_name = f"{datetime_string}.png"
            color_frame = np.asanyarray(self.curr_frames["color"].get_data())
            cv2.imwrite(
                os.path.join(self.image_dir, image_name),
                cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR),
            )
            response.success = True

        return response

    def _get_state_callback(self, request, response):
        response.state = self.get_state()
        return response

    def _get_positions_for_pixels_callback(self, request, response):
        response.positions = []
        for pixel in request.pixels:
            pos = self.camera.get_pos((pixel.x, pixel.y), self.curr_frames["depth"])
            response.positions.append(
                Vector3(x=pos[0], y=pos[1], z=pos[2])
                if pos is not None
                else Vector3(x=-1.0, y=-1.0, z=-1.0)
            )
        return response

    ## endregion

    ## region Message builders

    def _create_detection_result_msg(self, points, frames, track_ids=None):
        msg = DetectionResult()
        msg.timestamp = frames["timestamp"] / 1000
        for idx, point in enumerate(points):
            point_msg = Vector2(x=float(point[0]), y=float(point[1]))
            pos = self.camera.get_pos((point[0], point[1]), frames["depth"])
            if pos is not None:
                object_instance = ObjectInstance()
                object_instance.track_id = (
                    track_ids[idx]
                    if track_ids is not None and idx < len(track_ids)
                    else -1
                )
                object_instance.position = Vector3(x=pos[0], y=pos[1], z=pos[2])
                object_instance.point = point_msg
                msg.instances.append(object_instance)
            else:
                msg.invalid_points.append(point_msg)
        return msg

    def _get_color_frame_msg(self, color_frame, timestamp_millis):
        msg = self.cv_bridge.cv2_to_imgmsg(color_frame, encoding="rgb8")
        sec, nanosec = milliseconds_to_ros_time(timestamp_millis)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        return msg

    def _get_depth_frame_msg(self, depth_frame, timestamp_millis):
        msg = self.cv_bridge.cv2_to_imgmsg(depth_frame, encoding="mono16")
        sec, nanosec = milliseconds_to_ros_time(timestamp_millis)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        return msg

    def _get_color_frame_compressed_msg(self, color_frame, timestamp_millis):
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

    def _get_debug_frame_compressed_msg(self, debug_frame, timestamp_millis):
        sec, nanosec = milliseconds_to_ros_time(timestamp_millis)
        _, jpeg_data = cv2.imencode(
            ".jpg", cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
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
        runner_centroids,
        confs,
        track_ids,
        mask_color=(255, 255, 255),
        centroid_color=(255, 64, 255),
        draw_conf=True,
        draw_track_id=True,
    ):
        for runner_mask, runner_centroid, conf, track_id in zip(
            runner_masks, runner_centroids, confs, track_ids
        ):
            debug_frame = cv2.fillPoly(
                debug_frame,
                pts=[np.array(runner_mask, dtype=np.int32)],
                color=mask_color,
            )
            pos = [int(runner_centroid[0]), int(runner_centroid[1])]
            debug_frame = cv2.drawMarker(
                debug_frame,
                pos,
                centroid_color,
                cv2.MARKER_TILTED_CROSS,
                thickness=1,
                markerSize=20,
            )
            if draw_conf:
                pos = [int(runner_centroid[0]) + 15, int(runner_centroid[1]) - 5]
                font = cv2.FONT_HERSHEY_SIMPLEX
                debug_frame = cv2.putText(
                    debug_frame, f"{conf:.2f}", pos, font, 0.5, mask_color
                )
            if draw_track_id and track_id > 0:
                pos = [int(runner_centroid[0]) + 15, int(runner_centroid[1]) + 10]
                font = cv2.FONT_HERSHEY_SIMPLEX
                debug_frame = cv2.putText(
                    debug_frame, f"{track_id}", pos, font, 0.5, mask_color
                )
        return debug_frame


def main(args=None):
    rclpy.init(args=args)
    node = CameraControlNode()
    rclpy.spin(node)
    rclpy.shutdown()
    node.destroy_node()


if __name__ == "__main__":
    main()
