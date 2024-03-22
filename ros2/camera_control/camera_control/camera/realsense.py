import logging

import numpy as np
import pyrealsense2 as rs
import time
import threading


class RealSense:
    def __init__(
        self,
        color_frame_size,
        depth_frame_size,
        fps=30,
        align_depth_to_color_frame=True,
        camera_index=0,
        logger=None,
    ):
        # Note: when connected via USB2, RealSense cameras are limited to:
        # - 1280x720 @ 6fps
        # - 640x480 @ 30fps
        # - 480x270 @ 60fps
        # See https://www.intelrealsense.com/usb2-support-for-intel-realsense-technology/
        self.color_frame_size = color_frame_size
        self.depth_frame_size = depth_frame_size
        self.fps = fps
        self.align_depth_to_color_frame = align_depth_to_color_frame
        self.camera_index = camera_index
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
        self.check_connection = False
        self.check_connection_thread = None
        self.connected_devices = []

    def initialize(self):
        # Setup code based on https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
        self.config = rs.config()

        # Connect to specific camera
        self.connected_devices = [
            device.get_info(rs.camera_info.serial_number)
            for device in rs.context().query_devices()
        ]
        if self.camera_index < 0 or self.camera_index >= len(self.connected_devices):
            raise Exception("camera_index is out of bounds")

        self.serial_number = self.connected_devices[self.camera_index]
        self.config.enable_device(self.serial_number)

        # Configure streams
        self.config.enable_stream(
            rs.stream.depth,
            self.depth_frame_size[0],
            self.depth_frame_size[1],
            rs.format.z16,
            int(self.fps),
        )
        self.config.enable_stream(
            rs.stream.color,
            self.color_frame_size[0],
            self.color_frame_size[1],
            rs.format.rgb8,
            int(self.fps),
        )

        # Start pipeline
        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(self.config)

        # Get camera intrinsics and extrinsics
        color_prof = self.profile.get_stream(rs.stream.color)
        depth_prof = self.profile.get_stream(rs.stream.depth)
        self.depth_intrinsics = depth_prof.as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = color_prof.as_video_stream_profile().get_intrinsics()
        self.depth_to_color_extrinsics = depth_prof.get_extrinsics_to(color_prof)
        self.color_to_depth_extrinsics = color_prof.get_extrinsics_to(depth_prof)

        # Get depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # General min and max possible depths pulled from realsense examples
        self.depth_min_meters = 0.1
        self.depth_max_meters = 10

        # Post-processing
        self.align = (
            rs.align(rs.stream.color) if self.align_depth_to_color_frame else None
        )
        self.temporal_filter = rs.temporal_filter()
        # self.spatial_filter = rs.spatial_filter()  # Doesn't seem to help much. Disabling for now.
        self.hole_filling_filter = rs.hole_filling_filter()

        # Exposure setting persists on device, so reset it to auto-exposure
        self.set_exposure(-1)

        def check_connection_thread():
            while self.check_connection:
                prev_devices = set(self.connected_devices)
                self.connected_devices = [
                    device.get_info(rs.camera_info.serial_number)
                    for device in rs.context().query_devices()
                ]
                curr_devices = set(self.connected_devices)
                disconnected_devices = prev_devices - curr_devices
                connected_devices = curr_devices - prev_devices

                if self.serial_number in connected_devices:
                    self.logger.info(f"Camera connected")
                    self.initialize()
                elif self.serial_number in disconnected_devices:
                    self.logger.info(f"Camera disconnected")
                    self.pipeline.stop()

                time.sleep(1)

        if self.check_connection_thread is None:
            self.check_connection = True
            self.check_connection_thread = threading.Thread(
                target=check_connection_thread, daemon=True
            )
            self.check_connection_thread.start()

    def set_exposure(self, exposure_ms):
        color_sensor = self.profile.get_device().first_color_sensor()
        if exposure_ms < 0:
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        else:
            # D435 has a minimum exposure time of 1us
            exposure_us = max(1, round(exposure_ms * 1000))
            color_sensor.set_option(rs.option.exposure, exposure_us)

    def get_frames(self):
        if not self._is_connected():
            return None

        try:
            frames = self.pipeline.poll_for_frames()
        except:
            return None

        if not frames:
            return None

        # Align depth frame to color frame if needed
        if self.align:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not depth_frame or not color_frame:
            return None

        # Apply post-processing filters
        depth_frame = self.temporal_filter.process(depth_frame)
        depth_frame = self.hole_filling_filter.process(depth_frame)

        # The various post processing functions return a generic frame, so we need
        # to cast back to depth_frame
        depth_frame = depth_frame.as_depth_frame()

        return {
            "color": color_frame,
            "depth": depth_frame,
            "timestamp": color_frame.get_timestamp(),
        }

    def get_pos(self, color_pixel, depth_frame):
        """Given an x-y point in the color frame, return the x-y-z position with respect to the camera"""
        depth_pixel = self._color_pixel_to_depth_pixel(color_pixel, depth_frame)
        if not depth_pixel:
            self.logger.info("No depth pixel")
            return None

        if np.isnan(depth_pixel[0]) or np.isnan(depth_pixel[1]):
            self.logger.info("Nan depth returned")
            return None

        depth = depth_frame.get_distance(round(depth_pixel[0]), round(depth_pixel[1]))
        if depth < 0:
            self.logger.info("Calculated negative depth")
            return None

        return rs.rs2_deproject_pixel_to_point(
            self.color_intrinsics, color_pixel, depth
        )

    def _color_pixel_to_depth_pixel(self, pixel, depth_frame):
        """Given the location of a x-y point in the color frame, return the corresponding x-y point in the depth frame."""
        if self.align:
            return pixel
        else:
            # Based of a number of realsense github issues including
            # https://github.com/IntelRealSense/librealsense/issues/5440#issuecomment-566593866
            depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(
                depth_frame.get_data(),
                self.depth_scale,
                self.depth_min_meters,
                self.depth_max_meters,
                self.depth_intrinsics,
                self.color_intrinsics,
                self.depth_to_color_extrinsics,
                self.color_to_depth_extrinsics,
                pixel,
            )

            return None if depth_pixel[0] < 0 or depth_pixel[1] < 0 else depth_pixel

    def _is_connected(self):
        return self.serial_number in self.connected_devices
