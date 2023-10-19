""" File realsense.py

Class for using a realsense depth camera. 
"""

import cv2
import rclpy
import time
from datetime import datetime
from rclpy.node import Node

from shapely import Polygon
from lrr_interfaces.msg import FrameData, LaserOn, PosData, Pos, Point
from lrr_interfaces.srv import RetrieveFrame

from laser_runner_removal.cv_utils import find_laser_point
from ultralytics import YOLO

import pyrealsense2 as rs
import numpy as np
import os

from ament_index_python.packages import get_package_share_directory


class RealSense(Node):
    def __init__(self):
        Node.__init__(self, "RealsensePublisher")
        self.logger = self.get_logger()

        # declare parameters from a ros config file, if no parameter is found, the default is used
        self.declare_parameters(
            namespace="",
            parameters=[
                ("video_dir", "/opt/video_stream"),
                ("debug_video_dir", "/opt/debug_video_stream"),
                ("save_video", True),
                ("save_debug", False),
                ("frame_period", 0.1),
                ("rgb_size", [1920, 1080]),
                ("depth_size", [1280, 720]),
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

        self.ts_publisher = self.create_publisher(FrameData, "frame_data", 5)
        self.laser_pos_publisher = self.create_publisher(PosData, "laser_pos_data", 5)
        self.runner_pos_publisher = self.create_publisher(PosData, "runner_pos_data", 5)

        self.laser_on_sub = self.create_subscription(
            LaserOn, "laser_on", self.laser_on_cb, 1
        )
        self.runner_point = None
        self.runner_point_sub = self.create_subscription(
            Point, "runner_point", self.runner_point_cb, 1
        )
        self.laser_on = False
        self.frame_callback = self.create_timer(self.frame_period, self.frame_callback)

        self.frame_srv = self.create_service(
            RetrieveFrame, "retrieve_frame", self.retrieve_frame
        )

        self.background_frame = None
        self.initialize()

    def runner_point_cb(self, msg):
        self.logger.info(f"Runner Point: [{msg.x}, {msg.y}]")
        self.runner_point = [int(msg.x), int(msg.y)]

    def laser_on_cb(self, msg):
        self.logger.info(f"Laser State: {msg.laser_state}")
        self.laser_on = msg.laser_state

    def initialize(self):
        # Setup yolo model
        include_dir = os.path.join(
            get_package_share_directory("laser_runner_removal"), "include"
        )
        self.model = YOLO(os.path.join(include_dir, "RunnerSegModel.pt"))

        # Setup code based on https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.depth, self.depth_size[0], self.depth_size[1], rs.format.z16, 30
        )
        self.config.enable_stream(
            rs.stream.color, self.rgb_size[0], self.rgb_size[1], rs.format.bgr8, 30
        )
        self.profile = self.pipeline.start(self.config)
        self.setup_inten_extrins()

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self.depthMinMeters = 0.1
        self.depthMaxMeters = 10

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

    def frame_callback(self):
        frames = self.get_frames()
        if not frames:
            return

        frame_ts = frames["color"].get_timestamp()
        # convert from ms to seconds
        frame_ts = frame_ts / 1000

        ts_msg = FrameData()
        ts_msg.timestamp = frame_ts
        self.logger.debug(
            f"Publishing frame ts: {frame_ts}, current time:{time.time()}"
        )
        self.ts_publisher.publish(ts_msg)

        curr_image = np.asanyarray(frames["color"].get_data())
        self.rec.write(curr_image)

        laser_point_list = []
        runner_point_list = []

        if self.laser_on:
            laser_point_list = self.detect_laser(frames)
            laser_msg = self.create_pos_data_msg(frame_ts, laser_point_list, frames)
            self.laser_pos_publisher.publish(laser_msg)
        else:
            self.background_image = curr_image
            runner_point_list = self.detect_runners(frames)
            runner_msg = self.create_pos_data_msg(frame_ts, runner_point_list, frames)
            self.runner_pos_publisher.publish(runner_msg)

        if self.rec_debug_frame:
            debug_frame = np.copy(curr_image)
            debug_frame = self.draw_laser(debug_frame, laser_point_list)
            debug_frame = self.draw_runners(debug_frame, runner_point_list)
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

    def draw_laser(self, debug_frame, laser_list):
        for laser in laser_list:
            pos = [int(laser[0]), int(laser[1])]
            debug_frame = cv2.drawMarker(
                debug_frame,
                pos,
                (0, 0, 0),
                cv2.MARKER_CROSS,
                thickness=1,
                markerSize=20,
            )
        return debug_frame

    def draw_runners(self, debug_frame, runner_list):
        for runner in runner_list:
            pos = [int(runner[0]), int(runner[1])]
            debug_frame = cv2.drawMarker(
                debug_frame,
                pos,
                (255, 0, 0),
                cv2.MARKER_STAR,
                thickness=1,
                markerSize=20,
            )
        return debug_frame

    def create_pos_data_msg(self, timestamp, point_list, frames):
        msg = PosData()
        msg.pos_list = []
        msg.point_list = []
        msg.invalid_point_list = []
        msg.timestamp = timestamp
        for point in point_list:
            point_msg = Point()
            point_msg.x = point[0]
            point_msg.y = point[1]
            pos = self.get_pos_location(point[0], point[1], frames)
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

    def detect_runners(self, frames):
        image = np.asanyarray(frames["color"].get_data())
        res = self.model(image)

        point_list = []
        if res[0].masks:
            for cords in res[0].masks.xy:
                polygon = Polygon(cords)
                point_list.append((polygon.centroid.x, polygon.centroid.y))
        return point_list

    def detect_laser(self, frames):
        curr_image = np.asanyarray(frames["color"].get_data())
        image = cv2.absdiff(curr_image, self.background_image)
        found_point_list = find_laser_point(image)
        found_pos_list = []
        for point in found_point_list:
            self.logger.debug(f"Found Laser Point: {point}")
            found_pos_list.append(self.get_pos_location(point[0], point[1], frames))
        return found_point_list

    def get_clipping_distance(self):
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 1  # 1 meter
        self.clipping_distance = clipping_distance_in_meters / depth_scale

    def get_frames(self):
        frames = self.pipeline.poll_for_frames()
        if frames:
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                return None

            return {"color": color_frame, "depth": depth_frame}

    def setup_inten_extrins(self):
        color_prof = self.profile.get_stream(rs.stream.color)
        depth_prof = self.profile.get_stream(rs.stream.depth)

        self.depth_intrins = depth_prof.as_video_stream_profile().get_intrinsics()
        self.color_intrins = color_prof.as_video_stream_profile().get_intrinsics()
        self.logger.info(f"Color Frame intrinsics:{self.color_intrins}")
        self.depth_to_color_extrinsic = depth_prof.get_extrinsics_to(color_prof)
        self.color_to_depth_extrinsic = color_prof.get_extrinsics_to(depth_prof)

    def color_pixel_to_depth(self, x, y, frame):
        """Given the location of a x-y point in the color frame, return the corresponding x-y point in the depth frame."""
        # Based of a number of realsense github issues including
        # https://github.com/IntelRealSense/librealsense/issues/5440#issuecomment-566593866
        depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(
            frame["depth"].get_data(),
            self.depth_scale,
            self.depthMinMeters,
            self.depthMaxMeters,
            self.depth_intrins,
            self.color_intrins,
            self.depth_to_color_extrinsic,
            self.color_to_depth_extrinsic,
            (x, y),
        )
        self.logger.debug(
            f"Color point:[{x} {y}] corresponding Depth point{depth_pixel}"
        )
        if depth_pixel[0] < 0 or depth_pixel[1] < 0:
            return None
        return depth_pixel

    def get_pos_location(self, x, y, frame):
        """Given an x-y point in the color frame, return the x-y-z point with respect to the camera"""
        depth_point = self.color_pixel_to_depth(x, y, frame)
        if not depth_point:
            return None
        depth = frame["depth"].get_distance(
            round(depth_point[0]), round(depth_point[1])
        )
        pos_wrt_color = rs.rs2_deproject_pixel_to_point(
            self.color_intrins, [x, y], depth
        )
        pos_wrt_color = np.array(pos_wrt_color) * 1000
        self.logger.debug(
            f"color point:[{x}, {y}] corresponding color_pos: {pos_wrt_color}"
        )
        return pos_wrt_color

    def retrieve_frame(self, timestamp):
        raise NotImplementedError


"""
This should become EmuCam instead
"""


class EmuRealSense(RealSense):
    class EmuPipeline:
        def wait_for_frames(cam):
            return {
                "ts": time.time(),
                "color": None,
                "depth": None,
            }

    def initialize(self):
        self.pipeline = EmuRealSense.EmuPipeline()

    def get_frames(self):
        return None


def main(args=None):
    rclpy.init(args=args)
    # find way to pass Emu into args
    realsense_node = RealSense()
    rclpy.spin(realsense_node)
    rclpy.shutdown()
    realsense_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
