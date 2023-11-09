from .camera import Camera
import pyrealsense2 as rs
import numpy as np


class RealSense(Camera):
    def __init__(self, logger, color_frame_size, depth_frame_size):
        self.logger = logger
        self.color_frame_size = color_frame_size
        self.depth_frame_size = depth_frame_size

    def initialize(self):
        # Setup code based on https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.depth,
            self.depth_frame_size[0],
            self.depth_frame_size[1],
            rs.format.z16,
            30,
        )
        self.config.enable_stream(
            rs.stream.color,
            self.color_frame_size[0],
            self.color_frame_size[1],
            rs.format.rgb8,
            30,
        )
        self.profile = self.pipeline.start(self.config)
        self._setup_intrinsics_extrinsics()

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # General min and max possible depths pulled from realsense examples
        self.depth_min_meters = 0.1
        self.depth_max_meters = 10

        # Post-processing filters
        self.temporal_filter = rs.temporal_filter()
        # self.spatial_filter = rs.spatial_filter() # Doesn't seem to help much. Disabling for now.
        self.hole_filling_filter = rs.hole_filling_filter()

    def set_exposure(self, exposure_ms):
        color_sensor = self.profile.get_device().first_color_sensor()
        if exposure_ms < 0:
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        else:
            # D435 has a minimum exposure time of 1us
            exposure_us = max(1, round(exposure_ms * 1000))
            color_sensor.set_option(rs.option.exposure, exposure_us)

    def get_frames(self):
        frames = self.pipeline.poll_for_frames()
        if not frames:
            return None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not depth_frame or not color_frame:
            return None

        # Apply post-processing filters
        depth_frame = self.temporal_filter.process(depth_frame)
        depth_frame = self.hole_filling_filter.process(depth_frame)

        # The various post processing functions return a generic frame, so we need
        # to cast to depth_frame
        depth_frame = depth_frame.as_depth_frame()

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
            self.color_intrinsics, [x, y], depth
        )
        pos_wrt_color = np.array(pos_wrt_color) * 1000
        self.logger.debug(
            f"color point:[{x}, {y}] corresponding color_pos: {pos_wrt_color}"
        )
        return pos_wrt_color

    def _setup_intrinsics_extrinsics(self):
        color_prof = self.profile.get_stream(rs.stream.color)
        depth_prof = self.profile.get_stream(rs.stream.depth)

        self.depth_intrinsics = depth_prof.as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = color_prof.as_video_stream_profile().get_intrinsics()
        self.logger.info(f"Color Frame intrinsics:{self.color_intrinsics}")
        self.depth_to_color_extrinsics = depth_prof.get_extrinsics_to(color_prof)
        self.color_to_depth_extrinsics = color_prof.get_extrinsics_to(depth_prof)

    def _color_pixel_to_depth(self, x, y, frame):
        """Given the location of a x-y point in the color frame, return the corresponding x-y point in the depth frame."""
        # Based of a number of realsense github issues including
        # https://github.com/IntelRealSense/librealsense/issues/5440#issuecomment-566593866
        depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(
            frame["depth"].get_data(),
            self.depth_scale,
            self.depth_min_meters,
            self.depth_max_meters,
            self.depth_intrinsics,
            self.color_intrinsics,
            self.depth_to_color_extrinsics,
            self.color_to_depth_extrinsics,
            (x, y),
        )
        self.logger.debug(
            f"Color point:[{x} {y}] corresponding Depth point{depth_pixel}"
        )
        if depth_pixel[0] < 0 or depth_pixel[1] < 0:
            return None
        return depth_pixel
