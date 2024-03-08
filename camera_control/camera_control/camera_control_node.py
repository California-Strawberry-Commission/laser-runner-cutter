import os
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from camera_control_interfaces.msg import Point, Pos, PosData
from camera_control_interfaces.srv import (
    GetBool,
    GetFrame,
    GetPosData,
    SendEnable,
    SetExposure,
)
from cv_bridge import CvBridge
from rclpy.node import Node
from runner_segmentation.yolo import Yolo
from sensor_msgs.msg import Image

import camera_control.utils.cv_utils as cv_utils
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

        # For converting numpy array to image msg
        self.cv_bridge = CvBridge()

        # Pub/sub

        self.color_frame_pub = self.create_publisher(Image, "~/color_frame", 1)
        self.depth_frame_pub = self.create_publisher(Image, "~/depth_frame", 1)
        self.laser_pos_pub = self.create_publisher(PosData, "~/laser_pos_data", 5)
        self.runner_pos_pub = self.create_publisher(PosData, "~/runner_pos_data", 5)
        self.runner_point_sub = self.create_subscription(
            Point, "~/runner_point", self.runner_point_cb, 1
        )
        self.runner_point = None

        self.frame_call = self.create_timer(self.frame_period, self._frame_callback)

        # Services

        self.frame_srv = self.create_service(GetFrame, "~/get_frame", self.get_frame)
        self.single_runner_srv = self.create_service(
            GetPosData,
            "~/get_runner_detection",
            self.single_runner_detection,
        )
        self.single_laser_srv = self.create_service(
            GetPosData,
            "~/get_laser_detection",
            self.single_laser_detection,
        )
        self.has_frames_srv = self.create_service(
            GetBool, "~/has_frames", self.has_frames
        )
        self.set_exposure_srv = self.create_service(
            SetExposure, "~/set_exposure", self.set_exposure
        )

        self.control_laser_pub_srv = self.create_service(
            SendEnable,
            "~/control_laser_pub",
            self.control_laser_pub,
        )
        self.laser_pub_control = False

        self.control_runner_pub_srv = self.create_service(
            SendEnable,
            "~/control_runner_pub",
            self.control_runner_pub,
        )
        self.runner_pub_control = False

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

        self._rpc_runner_point_list = []
        self._rpc_runner_score_list = []
        self._rpc_laser_point_list = []
        self._rpc_laser_score_list = []

        # Video recording

        self.initialize_recording()

    def initialize_recording(self):
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

    def runner_point_cb(self, msg):
        # -1, -1 represents no point, if send instead set to None
        rec_point = [int(msg.x), int(msg.y)]
        if rec_point == [-1, -1]:
            rec_point = None
        self.runner_point = rec_point

    def _frame_callback(self):
        frames = self.camera.get_frames()
        if not frames:
            return

        self.curr_frames = frames

        self.color_frame_pub.publish(self._get_color_frame_msg())
        self.depth_frame_pub.publish(self._get_depth_frame_msg())

        # This is still using a realsense frame concept, for better multicamera
        # probability, all realsense based frame controls should move into the
        # realsense module.
        frame_ts = frames["color"].get_timestamp()
        # convert from ms to seconds
        frame_ts = frame_ts / 1000
        self.logger.debug(
            f"Publishing frame ts: {frame_ts}, current time:{time.time()}"
        )

        curr_image = np.asanyarray(frames["color"].get_data())

        if self.rec_video_frame:
            bgr_img = cv2.cvtColor(curr_image, cv2.COLOR_RGB2BGR)
            self.rec.write(bgr_img)

        laser_point_list = []
        laser_score_list = []
        runner_point_list = []
        runner_score_list = []

        if self.laser_pub_control:
            image = np.asanyarray(self.curr_frames["color"].get_data())
            laser_scores, laser_point_list = self.laser_detection_model.get_centroids(
                image
            )
            laser_msg = self.create_pos_data_msg(laser_point_list, frames)
            self.laser_pos_pub.publish(laser_msg)
        else:
            laser_point_list = self._rpc_laser_point_list
            laser_score_list = self._rpc_laser_score_list

        if self.runner_pub_control:
            image = np.asanyarray(self.curr_frames["color"].get_data())
            runner_scores, runner_point_list = self.runner_seg_model.get_centroids(
                image
            )
            runner_msg = self.create_pos_data_msg(runner_point_list, frames)
            self.runner_pos_pub.publish(runner_msg)
        else:
            runner_point_list = self._rpc_runner_point_list
            runner_score_list = self._rpc_runner_score_list

        if self.rec_debug_frame:
            debug_frame = np.copy(curr_image)
            debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
            debug_frame = cv_utils.draw_laser(
                debug_frame, laser_point_list, laser_score_list
            )
            debug_frame = cv_utils.draw_runners(
                debug_frame, runner_point_list, runner_score_list
            )
            if self.runner_point is not None:
                debug_frame = cv2.drawMarker(
                    debug_frame,
                    self.runner_point,
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    thickness=5,
                    markerSize=20,
                )
            self.rec_debug.write(debug_frame)

    def create_pos_data_msg(self, point_list, frames, timestamp=None):
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

    ###Service Calls
    def get_frame(self, request, response):
        if self.curr_frames is None:
            return response

        response.color_frame = self._get_color_frame_msg()
        response.depth_frame = self._get_depth_frame_msg()
        return response

    def has_frames(self, request, response):
        response.data = self.curr_frames is not None
        return response

    def set_exposure(self, request, response):
        self.camera.set_exposure(request.exposure_ms)
        return response

    def single_runner_detection(self, request, response):
        image = np.asanyarray(self.curr_frames["color"].get_data())
        runner_scores, runner_point_list = self.runner_seg_model.get_centroids(image)
        response.pos_data = self.create_pos_data_msg(
            runner_point_list, self.curr_frames
        )
        self.logger.debug(f"Camera runner msg:{response.pos_data}")
        self._rpc_runner_point_list = runner_point_list
        self._rpc_runner_score_list = runner_scores
        return response

    def single_laser_detection(self, request, response):
        image = np.asanyarray(self.curr_frames["color"].get_data())
        laser_scores, laser_point_list = self.laser_detection_model.get_centroids(image)
        response.pos_data = self.create_pos_data_msg(laser_point_list, self.curr_frames)
        self.logger.debug(f"Camera laser msg:{response.pos_data}")
        self._rpc_laser_point_list = laser_point_list
        self._rpc_laser_score_list = laser_scores
        return response

    def control_laser_pub(self, request, response):
        self.laser_pub_control = request.enable

    def control_runner_pub(self, request, response):
        self.runner_pub_control = request.enable

    def _get_color_frame_msg(self):
        if self.curr_frames is None:
            return None

        color_frame = self.curr_frames["color"]
        color_array = np.asanyarray(color_frame.get_data())
        color_frame_msg = self.cv_bridge.cv2_to_imgmsg(color_array, encoding="rgb8")
        sec, nanosec = milliseconds_to_ros_time(color_frame.get_timestamp())
        color_frame_msg.header.stamp.sec = sec
        color_frame_msg.header.stamp.nanosec = nanosec
        return color_frame_msg

    def _get_depth_frame_msg(self):
        if self.curr_frames is None:
            return None

        depth_frame = self.curr_frames["depth"]
        depth_array = np.asanyarray(depth_frame.get_data())
        depth_frame_msg = self.cv_bridge.cv2_to_imgmsg(depth_array, encoding="mono16")
        sec, nanosec = milliseconds_to_ros_time(depth_frame.get_timestamp())
        depth_frame_msg.header.stamp.sec = sec
        depth_frame_msg.header.stamp.nanosec = nanosec
        return depth_frame_msg


def main(args=None):
    rclpy.init(args=args)
    node = CameraControlNode()
    rclpy.spin(node)
    rclpy.shutdown()
    node.destroy_node()


if __name__ == "__main__":
    main()
