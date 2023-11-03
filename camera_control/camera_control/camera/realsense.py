import pyrealsense2 as rs
import numpy as np


class RealSense:
    def __init__(self, logger, rgb_size, depth_size):
        self.logger = logger
        self.rgb_size = rgb_size
        self.depth_size = depth_size

    def initialize(self):
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
        self._setup_inten_extrins()

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # General min and max possible depths pulled from realsense examples
        self.depthMinMeters = 0.1
        self.depthMaxMeters = 10

    def _setup_inten_extrins(self):
        color_prof = self.profile.get_stream(rs.stream.color)
        depth_prof = self.profile.get_stream(rs.stream.depth)

        self.depth_intrins = depth_prof.as_video_stream_profile().get_intrinsics()
        self.color_intrins = color_prof.as_video_stream_profile().get_intrinsics()
        self.logger.info(f"Color Frame intrinsics:{self.color_intrins}")
        self.depth_to_color_extrinsic = depth_prof.get_extrinsics_to(color_prof)
        self.color_to_depth_extrinsic = color_prof.get_extrinsics_to(depth_prof)

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

    def get_pos_location(self, x, y, frame):
        """Given an x-y point in the color frame, return the x-y-z position with respect to the camera"""
        depth_point = self._color_pixel_to_depth(x, y, frame)
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

    def _color_pixel_to_depth(self, x, y, frame):
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
