from typing import Optional, Tuple

import numpy as np
import pyrealsense2 as rs

from .rgbd_frame import RgbdFrame

# General min and max possible depths pulled from RealSense examples
DEPTH_MIN_METERS = 0.1
DEPTH_MAX_METERS = 10


class RealSenseFrame(RgbdFrame):
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
        self._color_frame = np.asanyarray(color_frame.get_data())
        self._depth_frame = np.asanyarray(depth_frame.get_data())
        self._rs_depth_frame = depth_frame
        self._timestamp_millis = timestamp_millis
        self._color_depth_aligned = color_depth_aligned
        self._depth_scale = depth_scale
        self._depth_intrinsics = depth_intrinsics
        self._color_intrinsics = color_intrinsics
        self._depth_to_color_extrinsics = depth_to_color_extrinsics
        self._color_to_depth_extrinsics = color_to_depth_extrinsics

    @property
    def color_frame(self) -> np.ndarray:
        return self._color_frame

    @property
    def depth_frame(self) -> np.ndarray:
        return self._depth_frame

    @property
    def timestamp_millis(self) -> float:
        return self._timestamp_millis

    @property
    def color_depth_aligned(self) -> bool:
        return self._color_depth_aligned

    def get_position(
        self, color_pixel: Tuple[int, int]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Given an (x, y) coordinate in the color frame, return the (x, y, z) position with respect to the camera.

        Args:
            color_pixel (Sequence[int]): (x, y) coordinate in the color frame.

        Returns:
            Optional[Tuple[float, float, float]]: (x, y, z) position with respect to the camera, or None if the position could not be determined.
        """
        depth_pixel = self._color_pixel_to_depth_pixel(color_pixel)
        if depth_pixel is None or np.isnan(depth_pixel[0]) or np.isnan(depth_pixel[1]):
            return None

        depth = self._rs_depth_frame.get_distance(
            round(depth_pixel[0]), round(depth_pixel[1])
        )
        if depth < 0:
            return None

        position = rs.rs2_deproject_pixel_to_point(
            self._color_intrinsics, color_pixel, depth
        )
        return (position[0], position[1], position[2])

    def _color_pixel_to_depth_pixel(
        self, color_pixel: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """
        Given an x-y coordinate in the color frame, return the corresponding x-y coordinate in the depth frame.

        Args:
            color_pixel (Tuple[int, int]): [x, y] coordinate in the color frame.

        Returns:
            Optional[Tuple[int, int]]: [x, y] coordinate in the depth frame, or None if the depth is negative
        """
        if self.color_depth_aligned:
            return color_pixel
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
                else (round(depth_pixel[0]), round(depth_pixel[1]))
            )
