import os
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from camera_control_interfaces.msg import Point, Pos, PosData
from camera_control_interfaces.srv import GetBool, GetFrame, GetPosData, SetExposure
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
                ("video_dir", "/opt/video_stream"),
                ("debug_video_dir", "/opt/debug_video_stream"),
                ("save_video", False),
                ("save_debug", False),
                ("camera_index", 0),
                ("frame_period", 0.1),
                ("rgb_size", [848, 480]),
                ("depth_size", [848, 480]),
            ],
        )

        self.video_dir = (
            self.get_parameter("video_dir").get_parameter_value().string_value
        )
        self.debug_video_dir = (
            self.get_parameter("debug_video_dir").get_parameter_value().string_value
        )
        self.rec_video_frame = (
            self.get_parameter("save_video").get_parameter_value().bool_value
        )
        self.rec_debug_frame = (
            self.get_parameter("save_debug").get_parameter_value().bool_value
        )
        self.camera_index = (
            self.get_parameter("camera_index").get_parameter_value().integer_value
        )
        self.frame_period = (
            self.get_parameter("frame_period").get_parameter_value().double_value
        )
        self.rgb_size = (
            self.get_parameter("rgb_size").get_parameter_value().integer_array_value
        )
        self.depth_size = (
            self.get_parameter("depth_size").get_parameter_value().integer_array_value
        )

        # Pub/sub

        self.color_frame_pub = self.create_publisher(
            CompressedImage, "~/color_frame", 1
        )
        self.debug_frame_pub = self.create_publisher(
            CompressedImage, "~/debug_frame", 1
        )
        self.laser_pos_pub = self.create_publisher(PosData, "~/laser_pos_data", 5)
        self.runner_pos_pub = self.create_publisher(PosData, "~/runner_pos_data", 5)

        # Services

        self.get_frame_srv = self.create_service(
            GetFrame, "~/get_frame", self._get_frame
        )
        self.get_runner_detection_srv = self.create_service(
            GetPosData,
            "~/get_runner_detection",
            self._single_runner_detection,
        )
        self.get_laser_detection_srv = self.create_service(
            GetPosData,
            "~/get_laser_detection",
            self._single_laser_detection,
        )
        self.has_frames_srv = self.create_service(
            GetBool, "~/has_frames", self._has_frames
        )
        self.set_exposure_srv = self.create_service(
            SetExposure, "~/set_exposure", self._set_exposure
        )
        self.laser_detection_enabled = False
        self.start_laser_detection_srv = self.create_service(
            Empty,
            "~/start_laser_detection",
            self._start_laser_detection,
        )
        self.stop_laser_detection_srv = self.create_service(
            Empty,
            "~/stop_laser_detection",
            self._stop_laser_detection,
        )
        self.runner_detection_enabled = False
        self.start_runner_detection_srv = self.create_service(
            Empty,
            "~/start_runner_detection",
            self._start_runner_detection,
        )
        self.stop_runner_detection_srv = self.create_service(
            Empty,
            "~/stop_runner_detection",
            self._stop_runner_detection,
        )

        # Frame processing loop
        self.frame_call = self.create_timer(self.frame_period, self._frame_callback)

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

        # Video recording

        self._initialize_recording()

    def _initialize_recording(self):
        # This is currently not functioning correctly because of permission errors
        if not os.path.isdir(self.video_dir) and self.rec_video_frame:
            os.makedirs(self.video_dir)
        if not os.path.isdir(self.debug_video_dir) and self.rec_debug_frame:
            os.makedirs(self.debug_video_dir)

        # handle default log location
        ts = time.time()
        datetime_obj = datetime.fromtimestamp(ts)
        datetime_string = datetime_obj.strftime("%Y%m%d%H%M%S")
        if self.rec_video_frame:
            video_name = f"{datetime_string}.avi"
            rec_name = os.path.join(self.video_dir, video_name)
            self.rec = cv2.VideoWriter(
                rec_name, 0, 1 / self.frame_period, (self.rgb_size[0], self.rgb_size[1])
            )
        if self.rec_debug_frame:
            debug_video_name = f"{datetime_string}_debug.avi"
            debug_rec_name = os.path.join(self.debug_video_dir, debug_video_name)
            self.rec_debug = cv2.VideoWriter(
                debug_rec_name,
                0,
                1 / self.frame_period,
                (self.rgb_size[0], self.rgb_size[1]),
            )

    def _frame_callback(self):
        frames = self.camera.get_frames()
        if not frames:
            return

        self.curr_frames = frames
        self.color_frame_pub.publish(self._get_color_frame_compressed_msg(frames))

        curr_image = np.asanyarray(frames["color"].get_data())
        debug_image = np.copy(curr_image)

        if self.laser_detection_enabled:
            laser_points, confs = self._get_laser_points(self.curr_frames)
            debug_image = self._debug_draw_lasers(debug_image, laser_points, confs)
            self.laser_pos_pub.publish(self._create_pos_data_msg(laser_points, frames))

        if self.runner_detection_enabled:
            runner_masks, confs = self._get_runner_masks(self.curr_frames)
            runner_centroids = self._get_runner_centroids(runner_masks)
            debug_image = self._debug_draw_runners(
                debug_image, runner_masks, runner_centroids, confs
            )
            self.runner_pos_pub.publish(
                self._create_pos_data_msg(runner_centroids, frames)
            )

        self.debug_frame_pub.publish(
            self._get_debug_frame_compressed_msg(frames, debug_image)
        )

        if self.rec_video_frame:
            self.rec.write(cv2.cvtColor(curr_image, cv2.COLOR_RGB2BGR))

        if self.rec_debug_frame:
            self.rec_debug.write(cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))

    def _get_laser_points(self, frames, conf_threshold=0.4):
        image = np.asanyarray(frames["color"].get_data())
        result = self.laser_detection_model.predict(image)
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

    def _get_runner_masks(self, frames, conf_threshold=0.4):
        image = np.asanyarray(frames["color"].get_data())
        result = self.runner_seg_model.predict(image)
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

    ## region Service calls

    def _get_frame(self, request, response):
        if self.curr_frames is None:
            return response

        response.color_frame = self._get_color_frame_msg(self.curr_frames)
        response.depth_frame = self._get_depth_frame_msg(self.curr_frames)
        return response

    def _has_frames(self, request, response):
        response.data = self.curr_frames is not None
        return response

    def _set_exposure(self, request, response):
        self.camera.set_exposure(request.exposure_ms)
        return response

    def _single_laser_detection(self, request, response):
        laser_points, conf = self._get_laser_points(self.curr_frames)
        response.pos_data = self._create_pos_data_msg(laser_points, self.curr_frames)
        return response

    def _single_runner_detection(self, request, response):
        runner_masks, confs = self._get_runner_masks(self.curr_frames)
        runner_centroids = self._get_runner_centroids(runner_masks)
        response.pos_data = self._create_pos_data_msg(
            runner_centroids, self.curr_frames
        )
        return response

    def _start_laser_detection(self, request, response):
        self.laser_detection_enabled = True
        return response

    def _stop_laser_detection(self, request, response):
        self.laser_detection_enabled = False
        return response

    def _start_runner_detection(self, request, response):
        self.runner_detection_enabled = True
        return response

    def _stop_runner_detection(self, request, response):
        self.runner_detection_enabled = False
        return response

    ## endregion

    ## region Message builders

    def _create_pos_data_msg(self, point_list, frames, timestamp=None):
        if timestamp is None:
            timestamp = frames["color"].get_timestamp() / 1000
        msg = PosData()
        msg.pos_list = []
        msg.point_list = []
        msg.invalid_point_list = []
        msg.timestamp = timestamp
        for point in point_list:
            point_msg = Point()
            point_msg.x = point[0]
            point_msg.y = point[1]
            pos = self.camera.get_pos_location(point[0], point[1], frames)
            if pos is not None:
                pos_msg = Pos()
                pos_msg.x = pos[0]
                pos_msg.y = pos[1]
                pos_msg.z = pos[2]
                msg.pos_list.append(pos_msg)
                msg.point_list.append(point_msg)
            else:
                msg.invalid_point_list.append(point_msg)
        return msg

    def _get_color_frame_msg(self, frames):
        if frames is None:
            return None

        color_frame = frames["color"]
        color_array = np.asanyarray(color_frame.get_data())
        msg = self.cv_bridge.cv2_to_imgmsg(color_array, encoding="rgb8")
        sec, nanosec = milliseconds_to_ros_time(color_frame.get_timestamp())
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        return msg

    def _get_depth_frame_msg(self, frames):
        if frames is None:
            return None

        depth_frame = frames["depth"]
        depth_array = np.asanyarray(depth_frame.get_data())
        msg = self.cv_bridge.cv2_to_imgmsg(depth_array, encoding="mono16")
        sec, nanosec = milliseconds_to_ros_time(depth_frame.get_timestamp())
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        return msg

    def _get_color_frame_compressed_msg(self, frames):
        if frames is None:
            return None

        color_frame = frames["color"]
        color_array = np.asanyarray(color_frame.get_data())
        sec, nanosec = milliseconds_to_ros_time(color_frame.get_timestamp())
        _, jpeg_data = cv2.imencode(
            ".jpg", cv2.cvtColor(color_array, cv2.COLOR_RGB2BGR)
        )
        msg = CompressedImage()
        msg.format = "jpeg"
        msg.data = jpeg_data.tobytes()
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        return msg

    def _get_debug_frame_compressed_msg(self, frames, debug_image):
        if frames is None:
            return None

        color_frame = frames["color"]
        sec, nanosec = milliseconds_to_ros_time(color_frame.get_timestamp())
        _, jpeg_data = cv2.imencode(
            ".jpg", cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
        )
        msg = CompressedImage()
        msg.format = "jpeg"
        msg.data = jpeg_data.tobytes()
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        return msg

    ## endregion

    def _debug_draw_lasers(
        self, debug_image, laser_points, confs, color=(255, 0, 255), draw_conf=True
    ):
        for laser_point, conf in zip(laser_points, confs):
            pos = [int(laser_point[0]), int(laser_point[1])]
            debug_image = cv2.drawMarker(
                debug_image,
                pos,
                color,
                cv2.MARKER_STAR,
                thickness=1,
                markerSize=20,
            )
            if draw_conf:
                pos = [int(laser_point[0]) - 15, int(laser_point[1]) - 15]
                font = cv2.FONT_HERSHEY_SIMPLEX
                debug_image = cv2.putText(
                    debug_image, f"{conf:.2f}", pos, font, 0.5, color
                )
        return debug_image

    def _debug_draw_runners(
        self,
        debug_image,
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
            debug_image = cv2.fillPoly(
                debug_image,
                pts=[np.array(runner_mask, dtype=np.int32)],
                color=mask_color,
            )
            pos = [int(runner_centroid[0]), int(runner_centroid[1])]
            debug_image = cv2.drawMarker(
                debug_image,
                pos,
                centroid_color,
                cv2.MARKER_TILTED_CROSS,
                thickness=1,
                markerSize=20,
            )
            if draw_conf:
                pos = [int(runner_centroid[0]) + 15, int(runner_centroid[1]) - 15]
                font = cv2.FONT_HERSHEY_SIMPLEX
                debug_image = cv2.putText(
                    debug_image, f"{conf:.2f}", pos, font, 0.5, centroid_color
                )
        return debug_image


def main(args=None):
    rclpy.init(args=args)
    node = CameraControlNode()
    rclpy.spin(node)
    rclpy.shutdown()
    node.destroy_node()


if __name__ == "__main__":
    main()
