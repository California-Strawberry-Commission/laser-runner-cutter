""" File realsense.py

Class for using a realsense depth camera. 
"""

import cv2
import rclpy
import time
from datetime import datetime
from rclpy.node import Node

from camera_control.camera.realsense import RealSense
from camera_control_interfaces.msg import (
    PosData,
    Pos,
    Point,
    UInt8NumpyArray,
    Int16NumpyArray,
)
from camera_control_interfaces.srv import (
    GetBool,
    GetFrame,
    GetPosData,
    SendEnable,
    SetExposure,
)

import camera_control.utils.cv_utils as cv_utils
from ultralytics import YOLO

import numpy as np
import os

from ament_index_python.packages import get_package_share_directory


class CameraControlNode(Node):
    def __init__(self):
        super().__init__("camera_control_node")
        self.logger = self.get_logger()

        # declare parameters from a ros config file, if no parameter is found, the default is used
        self.declare_parameters(
            namespace="",
            parameters=[
                ("video_dir", "/opt/video_stream"),
                ("debug_video_dir", "/opt/debug_video_stream"),
                ("save_video", False),
                ("save_debug", False),
                ("frame_period", 0.1),
                ("rgb_size", [848, 480]),
                ("depth_size", [848, 480]),
            ],
        )

        # get class attributes from passed in parameters
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
        self.frame_period = (
            self.get_parameter("frame_period").get_parameter_value().double_value
        )
        self.rgb_size = (
            self.get_parameter("rgb_size").get_parameter_value().integer_array_value
        )
        self.depth_size = (
            self.get_parameter("depth_size").get_parameter_value().integer_array_value
        )

        # This is currently not functioning correctly because of permission errors
        if not os.path.isdir(self.video_dir) and self.rec_video_frame:
            os.makedirs(self.video_dir)
        if not os.path.isdir(self.debug_video_dir) and self.rec_debug_frame:
            os.makedirs(self.debug_video_dir)

        # Create publishers and subscribers
        self.laser_pos_publisher = self.create_publisher(PosData, "laser_pos_data", 5)
        self.runner_pos_publisher = self.create_publisher(PosData, "runner_pos_data", 5)
        self.runner_point_sub = self.create_subscription(
            Point, "runner_point", self.runner_point_cb, 1
        )
        self.runner_point = None

        self.frame_call = self.create_timer(self.frame_period, self.frame_callback)

        # Create services
        self.frame_srv = self.create_service(
            GetFrame, "camera_control/get_frame", self.get_frame
        )
        self.single_runner_srv = self.create_service(
            GetPosData,
            "camera_control/get_runner_detection",
            self.single_runner_detection,
        )
        self.single_laser_srv = self.create_service(
            GetPosData,
            "camera_control/get_laser_detection",
            self.single_laser_detection,
        )
        self.has_frames_srv = self.create_service(
            GetBool, "camera_control/has_frames", self.has_frames
        )
        self.set_exposure_srv = self.create_service(
            SetExposure, "camera_control/set_exposure", self.set_exposure
        )

        self.control_laser_pub_srv = self.create_service(
            SendEnable,
            "camera_control/control_laser_pub",
            self.control_laser_pub,
        )
        self.laser_pub_control = False

        self.control_runner_pub_srv = self.create_service(
            SendEnable,
            "camera_control/control_runner_pub",
            self.control_runner_pub,
        )
        self.runner_pub_control = False

        self.curr_frames = None
        self.camera = RealSense(self.logger, self.rgb_size, self.depth_size)
        self.background_image = None
        self.initialize()

    def initialize(self):
        # Setup yolo model
        include_dir = os.path.join(
            get_package_share_directory("camera_control"), "include"
        )
        self.runner_seg_model = YOLO(os.path.join(include_dir, "RunnerSegModel.pt"))
        self.laser_detection_model = YOLO(
            os.path.join(include_dir, "LaserDetectionModel.pt")
        )

        self.camera.initialize()
        self.initialize_recording()

    def initialize_recording(self):
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
        self.runner_point = [int(msg.x), int(msg.y)]

    def frame_callback(self):
        frames = self.camera.get_frames()
        if not frames:
            return

        self.curr_frames = frames

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
            self.rec.write(curr_image)

        if self.background_image is None:
            self.background_image = curr_image

        laser_point_list = []
        runner_point_list = []

        if self.laser_pub_control:
            laser_point_list = cv_utils.detect_lasers(
                self.laser_detection_model, frames
            )
            laser_msg = self.create_pos_data_msg(laser_point_list, frames)
            self.laser_pos_publisher.publish(laser_msg)

        if self.runner_pub_control:
            runner_point_list = cv_utils.detect_runners(self.runner_seg_model, frames)
            runner_msg = self.create_pos_data_msg(runner_point_list, frames)
            self.runner_pos_publisher.publish(runner_msg)

        if self.rec_debug_frame:
            debug_frame = np.copy(curr_image)
            debug_frame = cv_utils.draw_laser(debug_frame, laser_point_list)
            debug_frame = cv_utils.draw_runners(debug_frame, runner_point_list)
            if self.laser_on and self.runner_point is not None:
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

        response.timestamp = self.curr_frames["color"].get_timestamp()
        color_array = np.asanyarray(self.curr_frames["color"].get_data())
        depth_array = np.asanyarray(self.curr_frames["depth"].get_data())
        color_data = list(color_array.flatten())
        depth_data = list(depth_array.flatten())
        response.color_frame = UInt8NumpyArray(data=color_data, shape=color_array.shape)
        response.depth_frame = Int16NumpyArray(data=depth_data, shape=depth_array.shape)
        return response

    def has_frames(self, request, response):
        response.data = self.curr_frames is not None
        return response

    def set_exposure(self, request, response):
        self.camera.set_exposure(request.exposure_ms)
        return response

    def single_runner_detection(self, request, response):
        runner_point_list = cv_utils.detect_runners(
            self.runner_seg_model, self.curr_frames
        )
        response.pos_data = self.create_pos_data_msg(
            runner_point_list, self.curr_frames
        )
        return response

    def single_laser_detection(self, request, response):
        laser_point_list = cv_utils.detect_lasers(
            self.laser_detection_model, self.curr_frames
        )
        response.pos_data = self.create_pos_data_msg(laser_point_list, self.curr_frames)
        return response

    def control_laser_pub(self, request, response):
        self.laser_pub_control = request.enable

    def control_runner_pub(self, request, response):
        self.runner_pub_control = request.enable


def main(args=None):
    rclpy.init(args=args)
    # find way to pass Emu into args
    camera_node = CameraControlNode()
    rclpy.spin(camera_node)
    rclpy.shutdown()
    camera_node.destroy_node()


if __name__ == "__main__":
    main()
