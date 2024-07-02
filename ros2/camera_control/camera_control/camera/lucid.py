import argparse
import concurrent.futures
import ctypes
import logging
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
from arena_api.buffer import BufferFactory
from arena_api.enums import PixelFormat
from arena_api.system import system

from .calibration import (
    construct_extrinsic_matrix,
    create_blob_detector,
    distort_pixel_coords,
    invert_extrinsic_matrix,
)
from .rgbd_camera import RgbdCamera
from .rgbd_frame import RgbdFrame

COLOR_CAMERA_MODEL_PREFIXES = ["ATL", "ATX", "PHX", "TRI", "TRT"]
DEPTH_CAMERA_MODEL_PREFIXES = ["HTP", "HLT", "HTR", "HTW"]
# General min and max possible depths
DEPTH_MIN_MM = 500
DEPTH_MAX_MM = 10000


def scale_grayscale_image(mono_image: np.ndarray) -> np.ndarray:
    """
    Scale a grayscale image so that it uses the full 8-bit range.

    Args:
        mono_image: Grayscale image to scale.

    Returns:
        np.ndarray: Scaled uint8 grayscale image.
    """
    # Convert to float to avoid overflow or underflow issues
    mono_image = np.array(mono_image, dtype=np.float32)

    # Find the minimum and maximum values in the image
    min_val = np.min(mono_image)
    max_val = np.max(mono_image)

    # Normalize the image to the range 0 to 1
    if max_val > min_val:
        mono_image = (mono_image - min_val) / (max_val - min_val)
    else:
        mono_image = mono_image - min_val

    # Scale to 0-255 and convert to uint8
    mono_image = (mono_image * 255).astype(np.uint8)

    return mono_image


def get_extrinsics(
    triton_mono_image: np.ndarray,
    helios_intensity_image: np.ndarray,
    helios_xyz_image: np.ndarray,
    triton_intrinsic_matrix: np.ndarray,
    triton_distortion_coeffs: np.ndarray,
    grid_size: Tuple[int, int],
    grid_type: int = cv2.CALIB_CB_SYMMETRIC_GRID,
    blob_detector=None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Finds the rotation and translation vectors that describe the conversion from Helios 3D coordinates
    to the Triton camera's coordinate system.

    Args:
        triton_mono_image (np.ndarray): Grayscale image from a Triton camera containing the calibration pattern.
        helios_intensity_image (np.ndarray): Grayscale image from a Helios camera containing the calibration pattern.
        helios_xyz_image (np.ndarray): 3D data from a Helios camera corresponding to helios_intensity_image
        triton_intrinsic_matrix (np.ndarray): Intrinsic matrix of the Triton camera.
        triton_distortion_coeffs: (np.ndarray): Distortion coefficients of the Triton camera.
        grid_size (Tuple[int, int]): (# cols, # rows) of the calibration pattern.
        grid_type (int): One of the following:
            cv2.CALIB_CB_SYMMETRIC_GRID uses symmetric pattern of circles.
            cv2.CALIB_CB_ASYMMETRIC_GRID uses asymmetric pattern of circles.
            cv2.CALIB_CB_CLUSTERING uses a special algorithm for grid detection. It is more robust to perspective distortions but much more sensitive to background clutter.
        blobDetector: Feature detector that finds blobs, like dark circles on light background. If None then a default implementation is used.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: A tuple of (rotation vector, translation vector), or (None, None) if unsuccessful.
    """
    # Based on https://support.thinklucid.com/app-note-helios-3d-point-cloud-with-rgb-color/

    # Scale images so that they each use the full 8-bit range
    triton_mono_image = scale_grayscale_image(triton_mono_image)
    helios_intensity_image = scale_grayscale_image(helios_intensity_image)

    # Find calibration circle centers on both cameras
    retval, triton_circle_centers = cv2.findCirclesGrid(
        triton_mono_image, grid_size, flags=grid_type, blobDetector=blob_detector
    )
    if not retval:
        print("Could not get circle centers from Triton mono image.")
        return None, None
    triton_circle_centers = np.squeeze(np.round(triton_circle_centers))

    retval, helios_circle_centers = cv2.findCirclesGrid(
        helios_intensity_image, grid_size, flags=grid_type, blobDetector=blob_detector
    )
    if not retval:
        print("Could not get circle centers from Helios intensity image.")
        return None, None
    helios_circle_centers = np.squeeze(np.round(helios_circle_centers).astype(np.int32))

    # Get XYZ values from the Helios 3D image that match 2D locations on the Triton (corresponding points)
    helios_circle_positions = helios_xyz_image[
        helios_circle_centers[:, 1], helios_circle_centers[:, 0]
    ]

    retval, rvec, tvec = cv2.solvePnP(
        helios_circle_positions,
        triton_circle_centers,
        triton_intrinsic_matrix,
        triton_distortion_coeffs,
    )
    if retval:
        return rvec, tvec
    else:
        return None, None


class LucidFrame(RgbdFrame):
    color_frame: np.ndarray
    depth_frame: np.ndarray
    timestamp_millis: float

    def __init__(
        self,
        color_frame: np.ndarray,
        depth_frame_xyz: np.ndarray,
        timestamp_millis: float,
        color_camera_intrinsic_matrix: np.ndarray,
        color_camera_distortion_coeffs: np.ndarray,
        depth_camera_intrinsic_matrix: np.ndarray,
        depth_camera_distortion_coeffs: np.ndarray,
        xyz_to_color_camera_extrinsic_matrix: np.ndarray,
        xyz_to_depth_camera_extrinsic_matrix: np.ndarray,
    ):
        """
        Args:
            color_frame (np.ndarray): The color frame in RGB8 format.
            depth_frame_xyz (np.ndarray): The depth frame in Coord3D_ABC16 format.
            timestamp_millis (float): The timestamp of the frame, in milliseconds since the device was started.
            color_camera_intrinsic_matrix (np.ndarray): Intrinsic matrix of the color camera.
            color_camera_distortion_coeffs (np.ndarray): Distortion coefficients of the color camera.
            depth_camera_intrinsic_matrix (np.ndarray): Intrinsic matrix of the depth camera.
            depth_camera_distortion_coeffs (np.ndarray): Distortion coefficients of the depth camera.
            xyz_to_color_camera_extrinsic_matrix (np.ndarray): Extrinsic matrix from depth camera's XYZ positions to the color camera.
            xyz_to_color_camera_extrinsic_matrix (np.ndarray): Extrinsic matrix from depth camera's XYZ positions to the depth camera.
        """
        self.color_frame = color_frame
        self._depth_frame_xyz = depth_frame_xyz
        self.depth_frame = np.sqrt(np.sum(np.square(depth_frame_xyz), axis=-1)).astype(
            np.uint16
        )  # Represent the depth frame as the L2 norm, and convert to mono16

        self.timestamp_millis = timestamp_millis
        self._color_camera_intrinsic_matrix = color_camera_intrinsic_matrix
        self._color_camera_distortion_coeffs = color_camera_distortion_coeffs
        self._depth_camera_intrinsic_matrix = depth_camera_intrinsic_matrix
        self._depth_camera_distortion_coeffs = depth_camera_distortion_coeffs
        self._xyz_to_color_camera_extrinsic_matrix = (
            xyz_to_color_camera_extrinsic_matrix
        )
        self._xyz_to_depth_camera_extrinsic_matrix = (
            xyz_to_depth_camera_extrinsic_matrix
        )
        self._color_to_depth_extrinsic_matrix = (
            xyz_to_depth_camera_extrinsic_matrix
            @ invert_extrinsic_matrix(xyz_to_color_camera_extrinsic_matrix)
        )

    def get_corresponding_depth_pixel_deprecated(
        self, color_pixel: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Given an (x, y) coordinate in the color frame, return the corresponding (x, y) coordinate in the depth frame.

        Args:
            color_pixel (Sequence[int]): (x, y) coordinate in the color frame.

        Returns:
            Tuple[int, int]: (x, y) coordinate in the depth frame.
        """

        # Undistort the pixel coordinate in color camera
        distorted_color_pixel = np.array([[color_pixel]], dtype=np.float32)
        undistorted_color_pixel = cv2.undistortPoints(
            distorted_color_pixel,
            self._color_camera_intrinsic_matrix,
            self._color_camera_distortion_coeffs,
            P=self._color_camera_intrinsic_matrix,
        ).reshape(-1)

        # Convert to normalized camera coordinates
        normalized_color_pixel = np.linalg.inv(
            self._color_camera_intrinsic_matrix
        ) @ np.append(undistorted_color_pixel, 1)
        print(f"normalized_color_pixel = {normalized_color_pixel}")

        # Transform the normalized coordinates to depth camera using the extrinsic matrix
        normalized_color_pixel *= DEPTH_MIN_MM
        transformed_homogeneous_depth_pixel = (
            self._color_to_depth_extrinsic_matrix @ np.append(normalized_color_pixel, 1)
        )
        print(
            f"transformed_homogeneous_depth_pixel = {transformed_homogeneous_depth_pixel}"
        )
        normalized_depth_pixel = (
            transformed_homogeneous_depth_pixel[:3]
            / transformed_homogeneous_depth_pixel[3]
        )
        print(f"normalized_depth_pixel = {normalized_depth_pixel}")

        # Convert to pixel coordinates in depth camera
        undistorted_depth_pixel = (
            self._depth_camera_intrinsic_matrix @ normalized_depth_pixel[:3]
        )
        undistorted_depth_pixel = (
            undistorted_depth_pixel[:2] / undistorted_depth_pixel[2]
        )
        print(f"undistorted_depth_pixel = {undistorted_depth_pixel}")

        # Apply distortion to the pixel coordinates in depth camera
        distorted_depth_pixel = distort_pixel_coords(
            undistorted_depth_pixel,
            self._depth_camera_intrinsic_matrix,
            self._depth_camera_distortion_coeffs,
        )
        print(f"distorted_depth_pixel = {distorted_depth_pixel}")

        return (
            int(round(distorted_depth_pixel[0])),
            int(round(distorted_depth_pixel[1])),
        )

    def get_corresponding_depth_pixel(
        self, color_pixel: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Given an (x, y) coordinate in the color frame, return the corresponding (x, y) coordinate in the depth frame.

        The general approach is as follows:
            1. Deproject the color image pixel coordinate to two positions in the color camera-space: one that corresponds to the
               position at the minimum depth, and one at the maximum depth.
            2. Transform the two positions from color camera-space to depth camera-space.
            3. Project the two positions to their respective depth image pixel coordinates.
            4. The target lies somewhere along the line formed by the two pixel coordinates found in the previous step. We
               iteratively move pixel by pixel along this line. For each depth image pixel, we grab the xyz data at the pixel,
               project it onto the color image plane, and see how far it is from the original color pixel coordinate. We find
               and return the closest match.

        Note that in order to achieve the above, we require two extrinsic matrices - one for projecting the xyz positions to
        the color camera image plane, and one for projecting the xyz positions to the depth camera image plane.

        Args:
            color_pixel (Sequence[int]): (x, y) coordinate in the color frame.

        Returns:
            Tuple[int, int]: (x, y) coordinate in the depth frame.
        """

        def deproject_pixel(pixel, depth, camera_matrix, distortion_coeffs):
            # Normalized, undistorted pixel coord
            pixel = np.array([[[pixel[0], pixel[1]]]], dtype=np.float32)
            pixel_undistorted = cv2.undistortPoints(
                pixel,
                camera_matrix,
                distortion_coeffs,
            )
            return np.array(
                [
                    pixel_undistorted[0][0][0] * depth,
                    pixel_undistorted[0][0][1] * depth,
                    depth,
                ]
            )

        def transform_position(position, extrinsic_matrix):
            position = np.array(position)
            homogeneous_position = np.append(position, 1)
            return np.dot(extrinsic_matrix, homogeneous_position)[:3]

        def project_position(
            position, camera_matrix, distortion_coeffs, extrinsic_matrix=None
        ):
            R = extrinsic_matrix[:3, :3] if extrinsic_matrix is not None else np.eye(3)
            t = extrinsic_matrix[:3, 3] if extrinsic_matrix is not None else np.zeros(3)
            pixels, _ = cv2.projectPoints(
                np.array([[position]]),
                R,
                t,
                camera_matrix,
                distortion_coeffs,
            )
            pixel = pixels[0].flatten()
            return (int(round(pixel[0])), int(round(pixel[1])))

        def adjust_pixel_to_bounds(pixel, width, height):
            x = max(0, min(int(round(pixel[0])), width - 1))
            y = max(0, min(int(round(pixel[1])), height - 1))
            return (x, y)

        def next_pixel_in_line(curr, start, end):
            # Move one pixel from curr to end
            curr = np.array(curr)
            end = np.array(end)
            direction = end - curr
            direction = direction / np.linalg.norm(direction)
            next = curr + direction
            return (int(round(next[0])), int(round(next[1])))

        def is_pixel_in_line(curr, start, end):
            min_x = min(start[0], end[0])
            max_x = max(start[0], end[0])
            min_y = min(start[1], end[1])
            max_y = max(start[1], end[1])

            return min_x <= curr[0] <= max_x and min_y <= curr[1] <= max_y

        # Min-depth and max-depth positions in color camera-space
        min_depth_position_color_space = deproject_pixel(
            color_pixel,
            DEPTH_MIN_MM,
            self._color_camera_intrinsic_matrix,
            self._color_camera_distortion_coeffs,
        )
        max_depth_position_color_space = deproject_pixel(
            color_pixel,
            DEPTH_MAX_MM,
            self._color_camera_intrinsic_matrix,
            self._color_camera_distortion_coeffs,
        )

        # Min-depth and max-depth positions in depth camera-space
        min_depth_position_depth_space = transform_position(
            min_depth_position_color_space, self._color_to_depth_extrinsic_matrix
        )
        max_depth_position_depth_space = transform_position(
            max_depth_position_color_space, self._color_to_depth_extrinsic_matrix
        )

        # Project depth camera-space positions to depth pixels
        min_depth_pixel = project_position(
            min_depth_position_depth_space,
            self._depth_camera_intrinsic_matrix,
            self._depth_camera_distortion_coeffs,
        )
        max_depth_pixel = project_position(
            max_depth_position_depth_space,
            self._depth_camera_intrinsic_matrix,
            self._depth_camera_distortion_coeffs,
        )

        # Make sure pixel coords are in boundary
        depth_frame_height, depth_frame_width, depth_channels = (
            self._depth_frame_xyz.shape
        )
        min_depth_pixel = adjust_pixel_to_bounds(
            min_depth_pixel, depth_frame_width, depth_frame_height
        )
        max_depth_pixel = adjust_pixel_to_bounds(
            max_depth_pixel, depth_frame_width, depth_frame_height
        )

        # Search along the line for the depth pixel for which its corresponding projected color pixel is the closest
        # to the target color pixel
        min_dist = -1
        closest_depth_pixel = min_depth_pixel
        curr_depth_pixel = min_depth_pixel
        while True:
            xyz_mm = self._depth_frame_xyz[curr_depth_pixel[1]][curr_depth_pixel[0]]
            curr_color_pixel = project_position(
                xyz_mm,
                self._color_camera_intrinsic_matrix,
                self._color_camera_distortion_coeffs,
                self._xyz_to_color_camera_extrinsic_matrix,
            )
            distance = np.linalg.norm(
                np.array(curr_color_pixel) - np.array(color_pixel)
            )
            if distance < min_dist or min_dist < 0:
                min_dist = distance
                closest_depth_pixel = curr_depth_pixel

            # Stop if we've processed the max_depth_pixel
            if (
                curr_depth_pixel[0] == max_depth_pixel[0]
                and curr_depth_pixel[1] == max_depth_pixel[1]
            ):
                break

            # Otherwise, find the next pixel along the line we should try
            curr_depth_pixel = next_pixel_in_line(
                curr_depth_pixel, min_depth_pixel, max_depth_pixel
            )
            if not is_pixel_in_line(curr_depth_pixel, min_depth_pixel, max_depth_pixel):
                break

        return closest_depth_pixel

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
        depth_pixel = self.get_corresponding_depth_pixel(color_pixel)
        position = self._depth_frame_xyz[depth_pixel[1]][depth_pixel[0]]
        # Negative depth indicates an invalid position
        if position[2] < 0.0:
            return None

        return (float(position[0]), float(position[1]), float(position[2]))


class LucidRgbd(RgbdCamera):
    """
    Combined interface for a LUCID color camera (such as the Triton) and depth camera (such as the
    Helios2). Unfortunately, we cannot modularize the LUCID cameras into individual instances, as
    calling system.device_infos is time consuming, and calling system.create_device more than once
    results in an error.
    """

    def __init__(
        self,
        color_camera_intrinsic_matrix: np.ndarray,
        color_camera_distortion_coeffs: np.ndarray,
        depth_camera_intrinsic_matrix: np.ndarray,
        depth_camera_distortion_coeffs: np.ndarray,
        xyz_to_color_camera_extrinsic_matrix: np.ndarray,
        xyz_to_depth_camera_extrinsic_matrix: np.ndarray,
        color_camera_serial_number: Optional[str] = None,
        depth_camera_serial_number: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            color_camera_intrinsic_matrix (np.ndarray): Intrinsic matrix of the color camera.
            color_camera_distortion_coeffs (np.ndarray): Distortion coefficients of the color camera.
            depth_camera_intrinsic_matrix (np.ndarray): Intrinsic matrix of the depth camera.
            depth_camera_distortion_coeffs (np.ndarray): Distortion coefficients of the depth camera.
            xyz_to_color_camera_extrinsic_matrix (np.ndarray): Extrinsic matrix from depth camera's XYZ positions to the color camera.
            xyz_to_color_camera_extrinsic_matrix (np.ndarray): Extrinsic matrix from depth camera's XYZ positions to the depth camera.
            color_camera_serial_number (Optional[str]): Serial number of color camera to connect to. If None, the first available color camera will be used.
            depth_camera_serial_number (Optional[str]): Serial number of depth camera to connect to. If None, the first available depth camera will be used.
            logger (Optional[logging.Logger]): Logger
        """
        self.color_camera_serial_number = color_camera_serial_number
        self._color_camera_intrinsic_matrix = color_camera_intrinsic_matrix
        self._color_camera_distortion_coeffs = color_camera_distortion_coeffs
        self.depth_camera_serial_number = depth_camera_serial_number
        self._depth_camera_intrinsic_matrix = depth_camera_intrinsic_matrix
        self._depth_camera_distortion_coeffs = depth_camera_distortion_coeffs
        self._xyz_to_color_camera_extrinsic_matrix = (
            xyz_to_color_camera_extrinsic_matrix
        )
        self._xyz_to_depth_camera_extrinsic_matrix = (
            xyz_to_depth_camera_extrinsic_matrix
        )
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)
        self._check_connection = False
        self._check_connection_thread = None
        self._color_device = None
        self._depth_device = None
        self._exposure_us = 0.0
        self._gain_db = 0.0

    @property
    def is_connected(self) -> bool:
        """
        Returns:
            bool: Whether the camera is connected.
        """
        return self._color_device is not None and self._depth_device is not None

    def initialize(self):
        # TODO: Implement device disconnection and reconnection logic
        device_infos = system.device_infos

        # If we don't have a serial number of a device, attempt to find one among connected devices
        if self.color_camera_serial_number is None:
            self.color_camera_serial_number = next(
                (
                    device_info["serial"]
                    for device_info in device_infos
                    if any(
                        device_info["model"].startswith(prefix)
                        for prefix in COLOR_CAMERA_MODEL_PREFIXES
                    )
                ),
                None,
            )
        if self.depth_camera_serial_number is None:
            self.depth_camera_serial_number = next(
                (
                    device_info["serial"]
                    for device_info in device_infos
                    if any(
                        device_info["model"].startswith(prefix)
                        for prefix in DEPTH_CAMERA_MODEL_PREFIXES
                    )
                ),
                None,
            )

        # Set up devices
        color_device_info = next(
            (
                device_info
                for device_info in device_infos
                if device_info["serial"] == self.color_camera_serial_number
            ),
            None,
        )
        depth_device_info = next(
            (
                device_info
                for device_info in device_infos
                if device_info["serial"] == self.depth_camera_serial_number
            ),
            None,
        )
        devices = system.create_device([color_device_info, depth_device_info])
        self._color_device = devices[0]
        self._depth_device = devices[1]

        # Configure color nodemap
        nodemap = self._color_device.nodemap
        stream_nodemap = self._color_device.tl_stream_nodemap
        nodemap["AcquisitionMode"].value = "Continuous"
        # Setting the buffer handling mode to "NewestOnly" ensures the most recent image
        # is delivered, even if it means skipping frames
        stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
        # Enable stream auto negotiate packet size, which instructs the camera to receive
        # the largest packet size that the system will allow
        stream_nodemap["StreamAutoNegotiatePacketSize"].value = True
        # Enable stream packet resend. If a packet is missed while receiving an image, a
        # packet resend is requested which retrieves and redelivers the missing packet
        # in the correct order.
        stream_nodemap["StreamPacketResendEnable"].value = True
        # Set frame size and pixel format
        nodemap["Width"].value = nodemap["Width"].max
        nodemap["Height"].value = nodemap["Height"].max
        color_frame_width = nodemap["Width"].value
        color_frame_height = nodemap["Height"].value
        nodemap["PixelFormat"].value = PixelFormat.RGB8
        # Set the following when Persistent IP is set on the camera
        nodemap["GevPersistentARPConflictDetectionEnable"].value = False

        # Configure depth nodemap
        nodemap = self._depth_device.nodemap
        stream_nodemap = self._depth_device.tl_stream_nodemap
        nodemap["AcquisitionMode"].value = "Continuous"
        # Setting the buffer handling mode to "NewestOnly" ensures the most recent image
        # is delivered, even if it means skipping frames
        stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
        # Enable stream auto negotiate packet size, which instructs the camera to receive
        # the largest packet size that the system will allow
        stream_nodemap["StreamAutoNegotiatePacketSize"].value = True
        # Enable stream packet resend. If a packet is missed while receiving an image, a
        # packet resend is requested which retrieves and redelivers the missing packet
        # in the correct order.
        stream_nodemap["StreamPacketResendEnable"].value = True
        # Set pixel format
        depth_frame_width = nodemap["Width"].value
        depth_frame_height = nodemap["Height"].value
        nodemap["PixelFormat"].value = PixelFormat.Coord3D_ABCY16
        # Set Scan 3D node values
        nodemap["Scan3dOperatingMode"].value = "Distance3000mmSingleFreq"
        nodemap["ExposureTimeSelector"].value = "Exp350Us"
        self._xyz_scale = nodemap["Scan3dCoordinateScale"].value
        nodemap["Scan3dCoordinateSelector"].value = "CoordinateA"
        x_offset = nodemap["Scan3dCoordinateOffset"].value
        nodemap["Scan3dCoordinateSelector"].value = "CoordinateB"
        y_offset = nodemap["Scan3dCoordinateOffset"].value
        nodemap["Scan3dCoordinateSelector"].value = "CoordinateC"
        z_offset = nodemap["Scan3dCoordinateOffset"].value
        self._xyz_offset = (x_offset, y_offset, z_offset)
        # Set confidence threshold
        nodemap["Scan3dConfidenceThresholdEnable"].value = True
        nodemap["Scan3dConfidenceThresholdMin"].value = 500
        # Set the following when Persistent IP is set on the camera
        nodemap["GevPersistentARPConflictDetectionEnable"].value = False

        # Set auto exposure and auto gain
        self.exposure_us = -1.0
        self.gain_db = -1.0

        # Start streams
        self._color_device.start_stream(10)
        self._logger.info(
            f"Device (color) {self.color_camera_serial_number} is now streaming at {color_frame_width}x{color_frame_height}"
        )
        self._depth_device.start_stream(10)
        self._logger.info(
            f"Device (depth) {self.depth_camera_serial_number} is now streaming at {depth_frame_width}x{depth_frame_height}"
        )

    @property
    def exposure_us(self) -> float:
        """
        Returns:
            float: Exposure time of the color camera in microseconds.
        """
        if not self.is_connected:
            return 0.0

        return self._exposure_us

    @exposure_us.setter
    def exposure_us(self, exposure_us: float):
        """
        Set the exposure time of the color camera. A negative value sets auto exposure.

        Args:
            exposure_us (float): Exposure time in microseconds. A negative value sets auto exposure.
        """
        if not self.is_connected:
            return

        nodemap = self._color_device.nodemap
        exposure_auto_node = nodemap["ExposureAuto"]
        exposure_time_node = nodemap["ExposureTime"]
        if exposure_us < 0.0:
            self._exposure_us = -1.0
            exposure_auto_node.value = "Continuous"
            self._logger.info(f"Auto exposure set")
        elif exposure_time_node is not None:
            exposure_auto_node.value = "Off"
            self._exposure_us = max(
                exposure_time_node.min, min(exposure_us, exposure_time_node.max)
            )
            exposure_time_node.value = self._exposure_us
            self._logger.info(f"Exposure set to {self._exposure_us}us")

    def get_exposure_us_range(self) -> Tuple[float, float]:
        """
        Returns:
            Tuple[float, float]: (min, max) exposure times of the color camera in microseconds.
        """
        if not self.is_connected:
            return (0.0, 0.0)

        nodemap = self._color_device.nodemap
        return (nodemap["ExposureTime"].min, nodemap["ExposureTime"].max)

    @property
    def gain_db(self) -> float:
        """
        Returns:
            float: Gain level of the color camera in dB.
        """
        if not self.is_connected:
            return 0.0

        return self._gain_db

    @gain_db.setter
    def gain_db(self, gain_db: float):
        """
        Set the gain level of the color camera.

        Args:
            gain_db (float): Gain level in dB.
        """
        if not self.is_connected:
            return

        nodemap = self._color_device.nodemap
        gain_auto_node = nodemap["GainAuto"]
        gain_node = nodemap["Gain"]
        if gain_db < 0.0:
            self._gain_db = -1.0
            gain_auto_node.value = "Continuous"
            self._logger.info(f"Auto gain set")
        elif gain_node is not None:
            gain_auto_node.value = "Off"
            self._gain_db = max(gain_node.min, min(gain_db, gain_node.max))
            gain_node.value = self._gain_db
            self._logger.info(f"Gain set to {self._gain_db} dB")

    def get_gain_db_range(self) -> Tuple[float, float]:
        """
        Returns:
            Tuple[float, float]: (min, max) gain levels of the color camera in dB.
        """
        if not self.is_connected:
            return (0.0, 0.0)

        nodemap = self._color_device.nodemap
        return (nodemap["Gain"].min, nodemap["Gain"].max)

    def get_color_frame(self) -> Optional[np.ndarray]:
        if self._color_device is None:
            return None

        # get_buffer must be called after start_stream and before stop_stream (or
        # system.destroy_device), and buffers must be requeued
        buffer = self._color_device.get_buffer()

        # Convert to numpy array
        # buffer is a list of (buffer.width * buffer.height * num_channels) uint8s
        num_channels = 3
        np_array = np.ndarray(
            buffer=(
                ctypes.c_ubyte * num_channels * buffer.width * buffer.height
            ).from_address(ctypes.addressof(buffer.pbytes)),
            dtype=np.uint8,
            shape=(buffer.height, buffer.width, num_channels),
        )

        self._color_device.requeue_buffer(buffer)

        return np_array

    def get_depth_frame(self) -> Optional[np.ndarray]:
        if self._depth_device is None:
            return None

        # get_buffer must be called after start_stream and before stop_stream (or
        # system.destroy_device), and buffers must be requeued
        buffer = self._depth_device.get_buffer()

        # Convert to numpy structured array
        # buffer is a list of (buffer.width * buffer.height * 8) 1-byte values. The 8 bytes per
        # pixel represent 4 channels, 16 bits each:
        #   - x position
        #   - y postion
        #   - z postion
        #   - intensity
        # Buffer.pdata is a (uint8, ctypes.c_ubyte) pointer. It is easier to deal with Buffer.pdata
        # if it is cast to 16 bits so each channel value is read/accessed easily.
        # PixelFormat.Coord3D_ABCY16 is unsigned, so we cast to a ctypes.c_uint16 pointer.
        pdata_16bit = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_uint16))
        num_pixels = buffer.width * buffer.height
        num_channels = 4
        np_array = np.frombuffer(
            (ctypes.c_int16 * num_pixels * num_channels).from_address(
                ctypes.addressof(pdata_16bit.contents)
            ),
            dtype=np.dtype(
                [
                    ("x", np.uint16),
                    ("y", np.uint16),
                    ("z", np.uint16),
                    ("i", np.uint16),
                ]
            ),
        )
        np_array = np_array.astype(
            np.dtype(
                [
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                    ("i", np.uint16),
                ]
            )
        )

        # Apply scale and offsets to convert (x, y, z) to mm
        np_array["x"] = np_array["x"] * self._xyz_scale + self._xyz_offset[0]
        np_array["y"] = np_array["y"] * self._xyz_scale + self._xyz_offset[1]
        np_array["z"] = np_array["z"] * self._xyz_scale + self._xyz_offset[2]

        # In unsigned pixel formats (such as ABCY16), values below the confidence threshold will have
        # their x, y, z, and intensity values set to 0xFFFF (denoting invalid). For these invalid pixels,
        # set (x, y, z) to (-1, -1, -1).
        invalid_pixels = np_array["i"] == 65535
        np_array["x"][invalid_pixels] = -1.0
        np_array["y"][invalid_pixels] = -1.0
        np_array["z"][invalid_pixels] = -1.0

        np_array = np_array.reshape(buffer.height, buffer.width)

        self._depth_device.requeue_buffer(buffer)

        return np_array

    def get_frame(self) -> Optional[LucidFrame]:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_color_frame = executor.submit(self.get_color_frame)
            future_depth_frame = executor.submit(self.get_depth_frame)
            color_frame = future_color_frame.result()
            depth_frame = future_depth_frame.result()

        if color_frame is None or depth_frame is None:
            return None

        # depth_frame is a numpy structured array containing both xyz and intensity data
        # depth_frame_intensity = (depth_frame["i"] / 256).astype(np.uint8)
        depth_frame_xyz = np.stack(
            [depth_frame["x"], depth_frame["y"], depth_frame["z"]], axis=-1
        )
        frame = LucidFrame(
            color_frame,
            depth_frame_xyz,  # type: ignore
            time.time() * 1000,
            self._color_camera_intrinsic_matrix,
            self._color_camera_distortion_coeffs,
            self._depth_camera_intrinsic_matrix,
            self._depth_camera_distortion_coeffs,
            self._xyz_to_color_camera_extrinsic_matrix,
            self._xyz_to_depth_camera_extrinsic_matrix,
        )
        return frame

    def close(self):
        # Destroy all created devices. Note that this will automatically call stop_stream() for each device
        self._color_device = None
        self._depth_device = None
        system.destroy_device()


def create_lucid_rgbd_camera(
    color_camera_serial_number: Optional[str] = None,
    depth_camera_serial_number: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> LucidRgbd:
    """
    Helper to create an instance of LucidRgbd using predefined calibration params.
    """
    calibration_params_dir = _get_calibration_params_dir()
    triton_intrinsic_matrix = np.load(
        os.path.join(calibration_params_dir, "triton_intrinsic_matrix.npy")
    )
    triton_distortion_coeffs = np.load(
        os.path.join(calibration_params_dir, "triton_distortion_coeffs.npy")
    )
    helios_intrinsic_matrix = np.load(
        os.path.join(calibration_params_dir, "helios_intrinsic_matrix.npy")
    )
    helios_distortion_coeffs = np.load(
        os.path.join(calibration_params_dir, "helios_distortion_coeffs.npy")
    )
    xyz_to_triton_extrinsic_matrix = np.load(
        os.path.join(calibration_params_dir, "xyz_to_triton_extrinsic_matrix.npy")
    )
    xyz_to_helios_extrinsic_matrix = np.load(
        os.path.join(calibration_params_dir, "xyz_to_helios_extrinsic_matrix.npy")
    )
    return LucidRgbd(
        triton_intrinsic_matrix,  # type: ignore
        triton_distortion_coeffs,  # type: ignore
        helios_intrinsic_matrix,  # type: ignore
        helios_distortion_coeffs,  # type: ignore
        xyz_to_triton_extrinsic_matrix,  # type: ignore
        xyz_to_helios_extrinsic_matrix,  # type: ignore
        color_camera_serial_number,
        depth_camera_serial_number,
        logger,
    )


def _get_calibration_params_dir():
    package_share_directory = get_package_share_directory("camera_control")
    return os.path.join(package_share_directory, "calibration_params")


def _get_frame(output_dir):
    output_dir = os.path.expanduser(output_dir)

    camera = create_lucid_rgbd_camera(
        color_camera_serial_number="241300039",
        depth_camera_serial_number="241400544",
    )
    camera.initialize()
    time.sleep(1)
    # camera.get_frame()

    color_frame = camera.get_color_frame()
    depth_frame = camera.get_depth_frame()

    if color_frame is not None and depth_frame is not None:
        triton_mono_image = cv2.cvtColor(color_frame, cv2.COLOR_RGB2GRAY)
        helios_intensity_image = (depth_frame["i"] / 256).astype(np.uint8)
        helios_xyz = np.stack(
            [depth_frame["x"], depth_frame["y"], depth_frame["z"]], axis=-1
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(
            os.path.join(output_dir, "triton_color_image.npy"),
            color_frame,
        )
        cv2.imwrite(
            os.path.join(output_dir, "triton_color_image.png"),
            cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR),
        )
        np.save(
            os.path.join(output_dir, "triton_mono_image.npy"),
            triton_mono_image,
        )
        cv2.imwrite(
            os.path.join(output_dir, "triton_mono_image.png"),
            triton_mono_image,
        )
        np.save(
            os.path.join(output_dir, "helios_intensity_image.npy"),
            helios_intensity_image,
        )
        cv2.imwrite(
            os.path.join(output_dir, "helios_intensity_image.png"),
            helios_intensity_image,
        )
        np.save(os.path.join(output_dir, "helios_xyz.npy"), helios_xyz)


def _get_position(
    color_pixel, triton_mono_image_path, helios_intensity_image, helios_xyz_path
):
    calibration_params_dir = _get_calibration_params_dir()
    triton_intrinsic_matrix = np.load(
        os.path.join(calibration_params_dir, "triton_intrinsic_matrix.npy")
    )
    triton_distortion_coeffs = np.load(
        os.path.join(calibration_params_dir, "triton_distortion_coeffs.npy")
    )
    helios_intrinsic_matrix = np.load(
        os.path.join(calibration_params_dir, "helios_intrinsic_matrix.npy")
    )
    helios_distortion_coeffs = np.load(
        os.path.join(calibration_params_dir, "helios_distortion_coeffs.npy")
    )
    xyz_to_triton_extrinsic_matrix = np.load(
        os.path.join(calibration_params_dir, "xyz_to_triton_extrinsic_matrix.npy")
    )
    xyz_to_helios_extrinsic_matrix = np.load(
        os.path.join(calibration_params_dir, "xyz_to_helios_extrinsic_matrix.npy")
    )

    triton_mono_image = np.load(os.path.expanduser(triton_mono_image_path))
    helios_intensity_image = np.load(os.path.expanduser(helios_intensity_image))
    helios_xyz = np.load(os.path.expanduser(helios_xyz_path))

    triton_image = cv2.cvtColor(triton_mono_image, cv2.COLOR_GRAY2RGB)

    frame = LucidFrame(
        triton_image,
        helios_xyz,  # type: ignore
        time.time() * 1000,
        triton_intrinsic_matrix,  # type: ignore
        triton_distortion_coeffs,  # type: ignore
        helios_intrinsic_matrix,  # type: ignore
        helios_distortion_coeffs,  # type: ignore
        xyz_to_triton_extrinsic_matrix,  # type: ignore
        xyz_to_helios_extrinsic_matrix,  # type: ignore
    )

    depth_pixel = frame.get_corresponding_depth_pixel(color_pixel)

    cv2.drawMarker(
        triton_image,
        color_pixel,
        (0, 0, 255),
        markerType=cv2.MARKER_STAR,
        markerSize=40,
        thickness=2,
    )
    h, w, c = triton_image.shape
    triton_image = cv2.resize(triton_image, (int(w / 2), int(h / 2)))
    cv2.imshow("Color camera", triton_image)

    helios_intensity_image = cv2.cvtColor(helios_intensity_image, cv2.COLOR_GRAY2RGB)
    cv2.drawMarker(
        helios_intensity_image,
        depth_pixel,
        (0, 0, 255),
        markerType=cv2.MARKER_STAR,
        markerSize=40,
        thickness=2,
    )
    cv2.imshow("Depth camera", helios_intensity_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _get_extrinsic(
    triton_mono_image_path, helios_intensity_image_path, helios_xyz_path, output_path
):
    output_path = os.path.expanduser(output_path)
    calibration_params_dir = _get_calibration_params_dir()
    triton_intrinsic_matrix = np.load(
        os.path.join(calibration_params_dir, "triton_intrinsic_matrix.npy")
    )
    triton_distortion_coeffs = np.load(
        os.path.join(calibration_params_dir, "triton_distortion_coeffs.npy")
    )
    triton_mono_image = np.load(os.path.expanduser(triton_mono_image_path))
    helios_intensity_image = np.load(os.path.expanduser(helios_intensity_image_path))
    helios_xyz = np.load(os.path.expanduser(helios_xyz_path))

    rvec, tvec = get_extrinsics(
        triton_mono_image,  # type: ignore
        helios_intensity_image,  # type: ignore
        helios_xyz,  # type: ignore
        triton_intrinsic_matrix,  # type: ignore
        triton_distortion_coeffs,  # type: ignore
        (5, 4),
        grid_type=cv2.CALIB_CB_SYMMETRIC_GRID,
        blob_detector=create_blob_detector(),
    )

    rvec = rvec.flatten()
    tvec = tvec.flatten()
    helios3d_to_triton_extrinsic_matrix = construct_extrinsic_matrix(rvec, tvec)

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(
        output_path,
        helios3d_to_triton_extrinsic_matrix,
    )


def _test_project(triton_mono_image_path, helios_intensity_image_path, helios_xyz_path):
    calibration_params_dir = _get_calibration_params_dir()
    triton_intrinsic_matrix = np.load(
        os.path.join(calibration_params_dir, "triton_intrinsic_matrix.npy")
    )
    triton_distortion_coeffs = np.load(
        os.path.join(calibration_params_dir, "triton_distortion_coeffs.npy")
    )
    helios3d_to_triton_extrinsic_matrix = np.load(
        os.path.join(calibration_params_dir, "helios3d_to_triton_extrinsic_matrix.npy")
    )
    triton_mono_image = np.load(os.path.expanduser(triton_mono_image_path))
    helios_xyz = np.load(os.path.expanduser(helios_xyz_path))
    helios_intensity_image = np.load(os.path.expanduser(helios_intensity_image_path))
    h, w, c = helios_xyz.shape
    object_points = helios_xyz.reshape(h * w, c)
    depths = object_points[:, 2]
    average_depth = np.mean(depths)
    print(f"average_depth: {average_depth}")
    helios_intensity_image = helios_intensity_image.flatten()
    R = helios3d_to_triton_extrinsic_matrix[:3, :3]
    t = helios3d_to_triton_extrinsic_matrix[:3, 3]
    start = time.perf_counter()
    image_points, jacobian = cv2.projectPoints(
        object_points, R, t, triton_intrinsic_matrix, triton_distortion_coeffs
    )  # (x, y), or (col, row)
    print(f"projectPoints took {time.perf_counter()-start} s")
    image_points = image_points.reshape(-1, 2)
    triton_height = 1536
    triton_width = 2048
    projected_image = np.zeros((triton_height, triton_width, 3), dtype=np.uint8)
    projected_image[:, :, 2] = (triton_mono_image < 100) * 128
    start = time.perf_counter()
    for point_idx in range(h * w):
        point = image_points[point_idx]
        col = round(point[0])
        row = round(point[1])
        if 0 <= col and col < triton_width and 0 <= row and row < triton_height:
            intensity = helios_intensity_image[point_idx]
            thresh = 1 if intensity < 30 else 0
            projected_image[row][col][1] = thresh * 255
    print(f"populate image took {time.perf_counter()-start} s")
    projected_image = cv2.resize(
        projected_image, (int(triton_width / 2), int(triton_height / 2))
    )
    cv2.imshow("projected", projected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _test_transform_depth_pixel_to_color_pixel(
    depth_pixel, triton_mono_image_path, helios_intensity_image_path, helios_xyz_path
):
    calibration_params_dir = _get_calibration_params_dir()
    triton_intrinsic_matrix = np.load(
        os.path.join(calibration_params_dir, "triton_intrinsic_matrix.npy")
    )
    triton_distortion_coeffs = np.load(
        os.path.join(calibration_params_dir, "triton_distortion_coeffs.npy")
    )
    helios3d_to_triton_extrinsic_matrix = np.load(
        os.path.join(calibration_params_dir, "helios3d_to_triton_extrinsic_matrix.npy")
    )
    triton_mono_image = np.load(os.path.expanduser(triton_mono_image_path))
    helios_xyz = np.load(os.path.expanduser(helios_xyz_path))
    helios_intensity_image = np.load(os.path.expanduser(helios_intensity_image_path))
    xyz_mm = helios_xyz[depth_pixel[1]][depth_pixel[0]]

    triton_image = cv2.cvtColor(triton_mono_image, cv2.COLOR_GRAY2RGB)
    helios_intensity_image = cv2.cvtColor(helios_intensity_image, cv2.COLOR_GRAY2RGB)

    def project_position(position, camera_matrix, distortion_coeffs, extrinsic_matrix):
        R = extrinsic_matrix[:3, :3]
        t = extrinsic_matrix[:3, 3]
        pixels, _ = cv2.projectPoints(
            np.array([[position]]),
            R,
            t,
            camera_matrix,
            distortion_coeffs,
        )
        pixel = pixels[0].flatten()
        return (int(round(pixel[0])), int(round(pixel[1])))

    color_pixel = project_position(
        xyz_mm,
        triton_intrinsic_matrix,
        triton_distortion_coeffs,
        helios3d_to_triton_extrinsic_matrix,
    )
    cv2.drawMarker(
        triton_image,
        color_pixel,
        (0, 0, 255),
        markerType=cv2.MARKER_STAR,
        markerSize=40,
        thickness=2,
    )
    h, w, c = triton_image.shape
    triton_image = cv2.resize(triton_image, (int(w / 2), int(h / 2)))
    cv2.imshow("Color camera", triton_image)

    cv2.drawMarker(
        helios_intensity_image,
        depth_pixel,
        (0, 0, 255),
        markerType=cv2.MARKER_STAR,
        markerSize=40,
        thickness=2,
    )
    cv2.imshow("Depth camera", helios_intensity_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def tuple_type(arg_string):
    try:
        # Parse the input string as a tuple
        parsed_tuple = tuple(map(int, arg_string.strip("()").split(",")))
        return parsed_tuple
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid tuple value: {arg_string}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="Available Commands", dest="command")

    get_frame_parser = subparsers.add_parser(
        "get_frame",
    )
    get_frame_parser.add_argument("--output_dir", type=str, default=None, required=True)

    get_extrinsic_parser = subparsers.add_parser(
        "get_extrinsic",
        help="Calculate extrinsic matrix between Helios2 to Triton cameras",
    )
    get_extrinsic_parser.add_argument(
        "--triton_mono_image", type=str, default=None, required=True
    )
    get_extrinsic_parser.add_argument(
        "--helios_intensity_image", type=str, default=None, required=True
    )
    get_extrinsic_parser.add_argument(
        "--helios_xyz", type=str, default=None, required=True
    )
    get_extrinsic_parser.add_argument("--output", type=str, default=None, required=True)

    get_position_parser = subparsers.add_parser(
        "get_position",
        help="Given pixel coordinates in the Triton image, calculate the 3D position",
    )
    get_position_parser.add_argument("--pixel", type=tuple_type, default=f"(1024, 768)")
    get_position_parser.add_argument(
        "--triton_mono_image", type=str, default=None, required=True
    )
    get_position_parser.add_argument(
        "--helios_intensity_image", type=str, default=None, required=True
    )
    get_position_parser.add_argument(
        "--helios_xyz", type=str, default=None, required=True
    )

    test_project_xyz_to_color_parser = subparsers.add_parser(
        "test_project_xyz_to_color",
        help="Verify calibration between Triton and Helios by projecting Helios point cloud onto Triton image",
    )
    test_project_xyz_to_color_parser.add_argument(
        "--triton_mono_image", type=str, default=None, required=True
    )
    test_project_xyz_to_color_parser.add_argument(
        "--helios_intensity_image", type=str, default=None, required=True
    )
    test_project_xyz_to_color_parser.add_argument(
        "--helios_xyz", type=str, default=None, required=True
    )

    test_transform_depth_pixel_to_color_pixel_parser = subparsers.add_parser(
        "test_transform_depth_pixel_to_color_pixel",
        help="Verify calibration between Triton and Helios by transforming a depth pixel to a color pixel",
    )
    test_transform_depth_pixel_to_color_pixel_parser.add_argument(
        "--pixel", type=tuple_type, default=f"(320, 240)"
    )
    test_transform_depth_pixel_to_color_pixel_parser.add_argument(
        "--triton_mono_image", type=str, default=None, required=True
    )
    test_transform_depth_pixel_to_color_pixel_parser.add_argument(
        "--helios_intensity_image", type=str, default=None, required=True
    )
    test_transform_depth_pixel_to_color_pixel_parser.add_argument(
        "--helios_xyz", type=str, default=None, required=True
    )

    args = parser.parse_args()

    if args.command == "get_frame":
        _get_frame(args.output_dir)
    elif args.command == "get_extrinsic":
        _get_extrinsic(
            args.triton_mono_image,
            args.helios_intensity_image,
            args.helios_xyz,
            args.output,
        )
    elif args.command == "get_position":
        _get_position(
            args.pixel,
            args.triton_mono_image,
            args.helios_intensity_image,
            args.helios_xyz,
        )
    elif args.command == "test_project_xyz_to_color":
        _test_project(
            args.triton_mono_image,
            args.helios_intensity_image,
            args.helios_xyz,
        )
    elif args.command == "test_transform_depth_pixel_to_color_pixel":
        _test_transform_depth_pixel_to_color_pixel(
            args.pixel,
            args.triton_mono_image,
            args.helios_intensity_image,
            args.helios_xyz,
        )
    else:
        print("Invalid command.")
