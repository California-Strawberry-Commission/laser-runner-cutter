import logging
import threading
import time
from typing import List, Optional, Sequence

import numpy as np
import pyrealsense2 as rs

from .camera import Camera
from .rgbd_frame import RGBDFrame

# D435 has a minimum exposure time of 1us
MIN_EXPOSURE_US = 1
# General min and max possible depths pulled from RealSense examples
DEPTH_MIN_METERS = 0.1
DEPTH_MAX_METERS = 10


class RealSenseFrame(RGBDFrame):
    def __init__(
        self,
        color_frame: rs.frame,
        depth_frame: rs.frame,
        timestamp_millis: float,
        color_depth_aligned: bool = False,
        depth_scale: Optional[float] = None,
        depth_intrinsics: Optional[rs.intrinsics] = None,
        color_intrinsics: Optional[rs.intrinsics] = None,
        depth_to_color_extrinsics: Optional[rs.extrinsics] = None,
        color_to_depth_extrinsics: Optional[rs.extrinsics] = None,
    ):
        """
        Args:
            color_frame (rs.frame): The color frame.
            depth_frame (rs.frame): The depth frame.
            timestamp_millis (float): The timestamp of the frame, in milliseconds since the device was started.
            color_depth_aligned (bool): Whether the color and depth frames are aligned.
            depth_scale (Optional[float]): Must be defined if color_depth_aligned is True.
            depth_intrinsics (Optional[rs.intrinsics]): Must be defined if color_depth_aligned is True.
            color_intrinsics (Optional[rs.intrinsics]): Must be defined if color_depth_aligned is True.
            depth_to_color_extrinsics (Optional[rs.extrinsics]): Must be defined if color_depth_aligned is True.
            color_to_depth_extrinsics (Optional[rs.extrinsics]): Must be defined if color_depth_aligned is True.
        """
        self.color_frame = np.asanyarray(color_frame.get_data())
        self.depth_frame = np.asanyarray(depth_frame.get_data())
        self._rs_depth_frame = depth_frame
        self.timestamp_millis = timestamp_millis
        self.color_depth_aligned = color_depth_aligned
        self._depth_scale = depth_scale
        self._depth_intrinsics = depth_intrinsics
        self._color_intrinsics = color_intrinsics
        self._depth_to_color_extrinsics = depth_to_color_extrinsics
        self._color_to_depth_extrinsics = color_to_depth_extrinsics

    def get_position(self, color_pixel: Sequence[int]) -> Optional[List[int]]:
        """
        Given an x-y coordinate in the color frame, return the x-y-z position with respect to the camera.

        Args:
            color_pixel (Sequence[int]): [x, y] coordinate in the color frame.

        Returns:
            Optional[List[int]]: [x, y, z] position with respect to the camera, or None if the position could not be determined
        """
        depth_pixel = self._color_pixel_to_depth_pixel(color_pixel)
        if depth_pixel is None or np.isnan(depth_pixel[0]) or np.isnan(depth_pixel[1]):
            return None

        depth = self._rs_depth_frame.get_distance(
            round(depth_pixel[0]), round(depth_pixel[1])
        )
        if depth < 0:
            return None

        return rs.rs2_deproject_pixel_to_point(
            self._color_intrinsics, color_pixel, depth
        )

    def _color_pixel_to_depth_pixel(
        self, color_pixel: Sequence[int]
    ) -> Optional[List[int]]:
        """
        Given an x-y coordinate in the color frame, return the corresponding x-y coordinate in the depth frame.

        Args:
            color_pixel (Sequence[int]): [x, y] coordinate in the color frame.

        Returns:
            Optional[List[int]]: [x, y] coordinate in the depth frame, or None if the depth is negative
        """
        if self.color_depth_aligned:
            return [color_pixel[0], color_pixel[1]]
        else:
            # Based of a number of RealSense Github issues including
            # https://github.com/IntelRealSense/librealsense/issues/5440#issuecomment-566593866
            depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(
                self._rs_depth_frame.get_data(),
                self._depth_scale,
                DEPTH_MIN_METERS,
                DEPTH_MAX_METERS,
                self._depth_intrinsics,
                self._color_intrinsics,
                self._depth_to_color_extrinsics,
                self._color_to_depth_extrinsics,
                color_pixel,
            )

            return (
                None
                if depth_pixel[0] < 0 or depth_pixel[1] < 0
                else [round(depth_pixel[0]), round(depth_pixel[1])]
            )


class RealSense(Camera):
    color_frame_size: Sequence[int]
    depth_frame_size: Sequence[int]
    fps: int
    serial_number: Optional[str]

    def __init__(
        self,
        color_frame_size: Sequence[int],
        depth_frame_size: Sequence[int],
        fps: int = 30,
        align_depth_to_color_frame: bool = True,
        serial_number: Optional[str] = None,
        camera_index: int = 0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Note: when connected via USB2, RealSense cameras are limited to:
        - 1280x720 @ 6fps
        - 640x480 @ 30fps
        - 480x270 @ 60fps
        See https://www.intelrealsense.com/usb2-support-for-intel-realsense-technology/

        Args:
            color_frame_size (Sequence[int]): [width, height] of the color frame.
            depth_frame_size (Sequence[int]): [width, height] of the depth frame.
            fps (int): Number of frames per second that the camera should capture.
            align_depth_to_color_frame (bool): Whether the color and depth frames should be aligned.
            serial_number (Optional[str]): Serial number of device to connect to. If None, camera_index will be used.
            camera_index (int): Index of detected camera to connect to. Will only be used if serial_number is None.
            logger (Optional[logging.Logger])
        """
        self.color_frame_size = color_frame_size
        self.depth_frame_size = depth_frame_size
        self.fps = fps
        self._align_depth_to_color_frame = align_depth_to_color_frame
        self.serial_number = serial_number
        self._camera_index = camera_index
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)

        self._check_connection = False
        self._check_connection_thread = None
        self._pipeline = None

    @property
    def is_connected(self) -> bool:
        """
        Returns:
            bool: Whether the camera is connected.
        """
        return self._pipeline is not None

    def initialize(self):
        """
        Set up the camera. Finds the serial number if needed, enables the device, configures the
        color and depth streams, starts the streams, and sets up any post processing steps.
        """
        # Setup code based on https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py

        connected_devices = [
            device.get_info(rs.camera_info.serial_number)
            for device in rs.context().query_devices()
        ]

        # If we don't have a serial number of a device, find one using camera_index
        if self.serial_number is None:
            if self._camera_index < 0 or self._camera_index >= len(connected_devices):
                raise Exception(
                    f"camera_index {self._camera_index} is out of bounds: {len(connected_devices)} devices found."
                )
            self.serial_number = connected_devices[self._camera_index]

        # Check if device is connected
        if self.serial_number not in connected_devices:
            raise Exception(f"No device with serial number {self.serial_number} found.")

        self._logger.info(f"Device {self.serial_number} is connected")

        # Enable device
        config = rs.config()
        config.enable_device(self.serial_number)

        # Configure streams
        config.enable_stream(
            rs.stream.depth,
            self.depth_frame_size[0],
            self.depth_frame_size[1],
            rs.format.z16,
            int(self.fps),
        )
        config.enable_stream(
            rs.stream.color,
            self.color_frame_size[0],
            self.color_frame_size[1],
            rs.format.rgb8,
            int(self.fps),
        )

        # Start pipeline
        self._pipeline = rs.pipeline()
        self._profile = self._pipeline.start(config)

        # Get camera intrinsics and extrinsics
        color_prof = self._profile.get_stream(rs.stream.color)
        depth_prof = self._profile.get_stream(rs.stream.depth)
        self._depth_intrinsics = depth_prof.as_video_stream_profile().get_intrinsics()
        self._color_intrinsics = color_prof.as_video_stream_profile().get_intrinsics()
        self._depth_to_color_extrinsics = depth_prof.get_extrinsics_to(color_prof)
        self._color_to_depth_extrinsics = color_prof.get_extrinsics_to(depth_prof)

        # Get depth scale
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        # General min and max possible depths pulled from realsense examples
        self._depth_min_meters = 0.1
        self._depth_max_meters = 10

        # Post-processing
        self._align = (
            rs.align(rs.stream.color) if self._align_depth_to_color_frame else None
        )
        self.temporal_filter = rs.temporal_filter()
        # self.spatial_filter = rs.spatial_filter()  # Doesn't seem to help much. Disabling for now.
        self.hole_filling_filter = rs.hole_filling_filter()

        # Exposure setting persists on device, so reset it to auto-exposure
        self.set_exposure(-1)

        def check_connection_thread():
            while self._check_connection:
                connected_devices = [
                    device.get_info(rs.camera_info.serial_number)
                    for device in rs.context().query_devices()
                ]
                connected = self.serial_number in connected_devices
                if self.is_connected != connected:
                    if connected:
                        self._logger.info(f"Device {self.serial_number} connected")
                        self.initialize()
                    else:
                        self._logger.info(f"Device {self.serial_number} disconnected")
                        self._pipeline.stop()
                        self._pipeline = None

                time.sleep(1)

        if self._check_connection_thread is None:
            self._check_connection = True
            self._check_connection_thread = threading.Thread(
                target=check_connection_thread, daemon=True
            )
            self._check_connection_thread.start()

        self._logger.info(f"Device {self.serial_number} is initialized")

    def set_exposure(self, exposure_us: float):
        """
        Set the exposure time of the camera.

        Args:
            exposure_us (float): Exposure time in microseconds.
        """
        if not self.is_connected:
            return

        color_sensor = self._profile.get_device().first_color_sensor()
        if exposure_us < 0:
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        else:
            # D435 has a minimum exposure time of 1us
            exposure_us = max(MIN_EXPOSURE_US, round(exposure_us))
            color_sensor.set_option(rs.option.exposure, exposure_us)

    def get_frame(self) -> Optional[RGBDFrame]:
        """
        Get the latest available color and depth frames from the camera.

        Returns:
            Optional[RGBDFrame]: The color and depth frames, or None if not available.
        """
        if not self.is_connected:
            return None

        try:
            frames = self._pipeline.poll_for_frames()
        except:
            return None

        if not frames:
            return None

        # Align depth frame to color frame if needed
        if self._align_depth_to_color_frame:
            frames = self._align.process(frames)

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

        return RealSenseFrame(
            color_frame,
            depth_frame,
            color_frame.get_timestamp(),
            self._align_depth_to_color_frame,
            self._depth_scale,
            self._depth_intrinsics,
            self._color_intrinsics,
            self._depth_to_color_extrinsics,
            self._color_to_depth_extrinsics,
        )
