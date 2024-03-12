import os
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from common_interfaces.msg import Vector2, Vector3
from camera_control_interfaces.msg import PosData, State
from camera_control_interfaces.srv import (
    GetBool,
    GetFrame,
    GetState,
    GetPosData,
    SetExposure,
)
from cv_bridge import CvBridge
from rclpy.node import Node
from runner_segmentation.yolo import Yolo
from sensor_msgs.msg import CompressedImage
from shapely import Polygon
from shapely.ops import nearest_points
from std_srvs.srv import Empty

from camera_control.camera.realsense import RealSense


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
                ("fps", 10),
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
        self.laser_pos_pub = self.create_publisher(PosData, "~/laser_pos_data", 5)
        self.runner_pos_pub = self.create_publisher(PosData, "~/runner_pos_data", 5)

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
            GetPosData, "~/get_laser_detection", self._single_laser_detection_callback
        )
        self.get_runner_detection_srv = self.create_service(
            GetPosData, "~/get_runner_detection", self._single_runner_detection_callback
        )
        self.laser_detection_enabled = False
        self.start_laser_detection_srv = self.create_service(
            Empty, "~/start_laser_detection", self._start_laser_detection_callback
        )
        self.stop_laser_detection_srv = self.create_service(
            Empty, "~/stop_laser_detection", self._stop_laser_detection_callback
        )
        self.runner_detection_enabled = False
        self.start_runner_detection_srv = self.create_service(
            Empty, "~/start_runner_detection", self._start_runner_detection_callback
        )
        self.stop_runner_detection_srv = self.create_service(
            Empty, "~/stop_runner_detection", self._stop_runner_detection_callback
        )
        self.start_recording_video_srv = self.create_service(
            Empty, "~/start_recording_video", self._start_recording_video_callback
        )
        self.stop_recording_video_srv = self.create_service(
            Empty, "~/stop_recording_video", self._stop_recording_video_callback
        )
        self.stop_recording_srv = self.create_service(
            Empty, "~/save_image", self._save_image_callback
        )
        self.state_srv = self.create_service(
            GetState, "~/get_state", self._get_state_callback
        )

        # Frame processing loop
        self.frame_call = self.create_timer(1.0 / self.fps, self._frame_callback)
        self.video_writer = None

        # For converting numpy array to image msg
        self.cv_bridge = CvBridge()

        # Camera

        self.curr_frames = None
        self.camera = RealSense(
            self.logger, self.rgb_size, self.depth_size, camera_index=self.camera_index
        )
        self.camera.initialize()

        # ML models

        package_share_directory = get_package_share_directory("camera_control")
        runner_weights_path = os.path.join(
            package_share_directory, "models", "RunnerSegYoloV8m.pt"
        )
        self.runner_seg_model = Yolo(runner_weights_path)
        laser_weights_path = os.path.join(
            package_share_directory, "models", "LaserDetectionYoloV8n.pt"
        )
        self.laser_detection_model = Yolo(laser_weights_path)

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
            self.laser_pos_pub.publish(self._create_pos_data_msg(laser_points, frames))

        if self.runner_detection_enabled:
            runner_masks, confs = self._get_runner_masks(color_frame)
            runner_centroids = self._get_runner_centroids(runner_masks)
            debug_frame = self._debug_draw_runners(
                debug_frame, runner_masks, runner_centroids, confs
            )
            self.runner_pos_pub.publish(
                self._create_pos_data_msg(runner_centroids, frames)
            )

        self.debug_frame_pub.publish(
            self._get_debug_frame_compressed_msg(debug_frame, timestamp_millis)
        )

        if self.video_writer is not None:
            self.video_writer.write(cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))

    def _get_laser_points(self, color_frame, conf_threshold=0.4):
        result = self.laser_detection_model.predict(color_frame)
        laser_points = []
        confs = []
        for idx in range(result["conf"].size):
            conf = result["conf"][idx]
            if conf >= conf_threshold:
                # bbox is in xyxy format
                bbox = result["bboxes"][idx]
                laser_points.append(
                    (round((bbox[0] + bbox[2]) * 0.5), round((bbox[1] + bbox[3]) * 0.5))
                )
                confs.append(conf)
        return laser_points, confs

    def _get_runner_masks(self, color_frame, conf_threshold=0.4):
        # TODO: resolution should match what the model was trained on (1024x768)
        # for the best performance. We could resize the image before prediction, then
        # map the prediction points back to the original image.
        result = self.runner_seg_model.predict(color_frame)
        runner_masks = []
        confs = []
        for idx in range(result["conf"].size):
            conf = result["conf"][idx]
            if conf >= conf_threshold:
                mask = result["masks"][idx]
                runner_masks.append(mask)
                confs.append(conf)
        return runner_masks, confs

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
        return response

    def _single_laser_detection_callback(self, request, response):
        if self.curr_frames is not None:
            color_frame = np.asanyarray(self.curr_frames["color"].get_data())
            laser_points, conf = self._get_laser_points(color_frame)
            response.pos_data = self._create_pos_data_msg(
                laser_points, self.curr_frames
            )
        return response

    def _single_runner_detection_callback(self, request, response):
        if self.curr_frames is not None:
            color_frame = np.asanyarray(self.curr_frames["color"].get_data())
            runner_masks, confs = self._get_runner_masks(color_frame)
            runner_centroids = self._get_runner_centroids(runner_masks)
            response.pos_data = self._create_pos_data_msg(
                runner_centroids, self.curr_frames
            )
        return response

    def _start_laser_detection_callback(self, request, response):
        self.laser_detection_enabled = True
        self._publish_state()
        return response

    def _stop_laser_detection_callback(self, request, response):
        self.laser_detection_enabled = False
        self._publish_state()
        return response

    def _start_runner_detection_callback(self, request, response):
        self.runner_detection_enabled = True
        self._publish_state()
        return response

    def _stop_runner_detection_callback(self, request, response):
        self.runner_detection_enabled = False
        self._publish_state()
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
        return response

    def _stop_recording_video_callback(self, request, response):
        self.video_writer = None
        self._publish_state()
        return response

    def _save_image_callback(self, request, response):
        os.makedirs(self.image_dir, exist_ok=True)
        ts = time.time()
        datetime_obj = datetime.fromtimestamp(ts)
        datetime_string = datetime_obj.strftime("%Y%m%d%H%M%S")
        image_name = f"{datetime_string}.png"

        if self.curr_frames is not None:
            color_frame = np.asanyarray(self.curr_frames["color"].get_data())
            cv2.imwrite(
                os.path.join(self.image_dir, image_name),
                cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR),
            )

        return response

    def _get_state_callback(self, request, response):
        response.state = self.get_state()
        return response

    ## endregion

    ## region Message builders

    def _create_pos_data_msg(self, point_list, frames):
        msg = PosData()
        msg.pos_list = []
        msg.point_list = []
        msg.invalid_point_list = []
        msg.timestamp = frames["timestamp"] / 1000
        for point in point_list:
            point_msg = Vector2(x=float(point[0]), y=float(point[1]))
            pos = self.camera.get_pos((point[0], point[1]), frames["depth"])
            if pos is not None:
                pos_msg = Vector3(x=pos[0], y=pos[1], z=pos[2])
                msg.pos_list.append(pos_msg)
                msg.point_list.append(point_msg)
            else:
                msg.invalid_point_list.append(point_msg)
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
        mask_color=(255, 255, 255),
        centroid_color=(255, 0, 255),
        draw_conf=True,
    ):
        for runner_mask, runner_centroid, conf in zip(
            runner_masks, runner_centroids, confs
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
                pos = [int(runner_centroid[0]) + 15, int(runner_centroid[1]) - 15]
                font = cv2.FONT_HERSHEY_SIMPLEX
                debug_frame = cv2.putText(
                    debug_frame, f"{conf:.2f}", pos, font, 0.5, centroid_color
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
