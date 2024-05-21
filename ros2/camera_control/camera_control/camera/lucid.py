import ctypes
import logging
import os
import time
from typing import Optional, Tuple
import argparse
import sys

import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
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

TRITON_FRAME_SIZE = (2048, 1536)
# General min and max possible depths pulled from RealSense examples
DEPTH_MIN_METERS = 0.1
DEPTH_MAX_METERS = 10


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
        depth_frame: np.ndarray,
        timestamp_millis: float,
        color_camera_intrinsic_matrix: np.ndarray,
        color_camera_distortion_coeffs: np.ndarray,
        depth_camera_intrinsic_matrix: np.ndarray,
        depth_camera_distortion_coeffs: np.ndarray,
        color_to_depth_extrinsic_matrix: np.ndarray,
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
        """
        self.color_frame = color_frame
        self.depth_frame = depth_frame
        self.timestamp_millis = timestamp_millis
        self._color_camera_intrinsic_matrix = color_camera_intrinsic_matrix
        self._color_camera_distortion_coeffs = color_camera_distortion_coeffs
        self._depth_camera_intrinsic_matrix = depth_camera_intrinsic_matrix
        self._depth_camera_distortion_coeffs = depth_camera_distortion_coeffs
        self._color_to_depth_extrinsic_matrix = color_to_depth_extrinsic_matrix
        self._depth_to_color_extrinsic_matrix = invert_extrinsic_matrix(
            color_to_depth_extrinsic_matrix
        )

    def get_corresponding_depth_pixel(self, color_pixel: Tuple[int, int]):
        # Undistort the pixel coordinate in color camera
        distorted_color_pixel = np.array([[color_pixel]], dtype=np.float32)
        print(f"distorted_color_pixel = {distorted_color_pixel}")
        undistorted_color_pixel = cv2.undistortPoints(
            distorted_color_pixel,
            self._color_camera_intrinsic_matrix,
            self._color_camera_distortion_coeffs,
            P=self._color_camera_intrinsic_matrix,
        ).reshape(-1)

        print(f"undistorted_color_pixel = {undistorted_color_pixel}")
        redistorted_color_pixel = distort_pixel_coords(
            undistorted_color_pixel,
            self._color_camera_intrinsic_matrix,
            self._color_camera_distortion_coeffs,
        )
        print(f"redistorted_color_pixel = {redistorted_color_pixel}")

        # Convert to normalized camera coordinates
        normalized_color_pixel = np.linalg.inv(
            self._color_camera_intrinsic_matrix
        ) @ np.append(undistorted_color_pixel, 1)
        print(f"normalized_color_pixel = {normalized_color_pixel}")

        # Transform the normalized coordinates to depth camera using the extrinsic matrix
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

        return (round(distorted_depth_pixel[0]), round(distorted_depth_pixel[1]))

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
        # TODO: See how RS does it: https://github.com/IntelRealSense/librealsense/blob/ff8a9fb213ec1227394de4060743b0ed61171985/src/rs.cpp#L4124

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

        def project_position(position, camera_matrix, distortion_coeffs):
            pixels, _ = cv2.projectPoints(
                np.array([[position]]),
                np.eye(3),
                np.zeros(3),
                camera_matrix,
                distortion_coeffs,
            )
            return pixels[0]

        def adjust_pixel_to_bounds(pixel, width, height):
            pixel = np.array(pixel)
            pixel[0] = max(0, min(pixel[0], width))
            pixel[1] = max(0, min(pixel[1], height))
            return pixel

        # Min-depth and max-depth positions in color camera-space
        min_depth_color_space_position = deproject_pixel(
            color_pixel,
            DEPTH_MIN_METERS,
            self._color_camera_intrinsic_matrix,
            self._color_camera_distortion_coeffs,
        )
        max_depth_color_space_position = deproject_pixel(
            color_pixel,
            DEPTH_MAX_METERS,
            self._color_camera_intrinsic_matrix,
            self._color_camera_distortion_coeffs,
        )

        # Min-depth and max-depth positions in depth camera-space
        min_depth_depth_space_position = transform_position(
            min_depth_color_space_position, self._color_to_depth_extrinsic_matrix
        )
        max_depth_depth_space_position = transform_position(
            max_depth_color_space_position, self._color_to_depth_extrinsic_matrix
        )

        # Project depth camera-space positions to depth pixels
        min_depth_pixel = project_position(
            min_depth_depth_space_position,
            self._depth_camera_intrinsic_matrix,
            self._depth_camera_distortion_coeffs,
        ).flatten()
        max_depth_pixel = project_position(
            max_depth_depth_space_position,
            self._depth_camera_intrinsic_matrix,
            self._depth_camera_distortion_coeffs,
        ).flatten()

        print(f"min_depth_pixel = {min_depth_pixel.flatten()}")
        print(f"max_depth_pixel = {max_depth_pixel.flatten()}")

        # Make sure pixel coords are in boundary
        depth_frame_height, depth_frame_width, depth_channels = self.depth_frame.shape
        min_depth_pixel = adjust_pixel_to_bounds(
            min_depth_pixel, depth_frame_width, depth_frame_height
        )
        max_depth_pixel = adjust_pixel_to_bounds(
            max_depth_pixel, depth_frame_width, depth_frame_height
        )

        """
        # Search along the line for the depth pixel for which its projected pixel is the closest
        # to the input pixel
        min_dist = -1
        curr_pixel = min_depth_pixel
        while is_pixel_in_line(curr_pixel, min_depth_pixel, max_depth_pixel):
            depth_mm = self.depth_frame[curr_pixel[1]][curr_pixel[0]][2]

            curr_pixel = next_pixel_in_line(
                curr_pixel, min_depth_pixel, max_depth_pixel
            )
        """


class LucidRgbd(RgbdCamera):
    """
    Combined interface for a LUCID color camera (such as the Triton) and depth camera (such as the
    Helios2). Unfortunately, we cannot modularize the LUCID cameras into individual instances, as
    calling system.device_infos is time consuming, and calling system.create_device more than once
    results in an error.
    """

    def __init__(
        self,
        frame_size: Tuple[int, int],
        color_camera_serial_number: str,
        color_camera_intrinsic_matrix: np.ndarray,
        color_camera_distortion_coeffs: np.ndarray,
        depth_camera_serial_number: str,
        depth_camera_intrinsic_matrix: np.ndarray,
        depth_camera_distortion_coeffs: np.ndarray,
        color_to_depth_extrinsic_matrix: np.ndarray,
        logger: Optional[logging.Logger] = None,
    ):
        self.frame_size = frame_size
        self.color_camera_serial_number = color_camera_serial_number
        self._color_camera_intrinsic_matrix = color_camera_intrinsic_matrix
        self._color_camera_distortion_coeffs = color_camera_distortion_coeffs
        self.depth_camera_serial_number = depth_camera_serial_number
        self._depth_camera_intrinsic_matrix = depth_camera_intrinsic_matrix
        self._depth_camera_distortion_coeffs = depth_camera_distortion_coeffs
        self._color_to_depth_extrinsic_matrix = color_to_depth_extrinsic_matrix
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)
        self._check_connection = False
        self._check_connection_thread = None
        self._color_device = None
        self._depth_device = None

    @property
    def is_connected(self) -> bool:
        """
        Returns:
            bool: Whether the camera is connected.
        """
        return self._color_device is not None and self._depth_device is not None

    def initialize(self):
        # TODO: implement device disconnection and reconnection logic
        device_infos = system.device_infos

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
        nodemap["Width"].value = self.frame_size[0]
        nodemap["Height"].value = self.frame_size[1]
        nodemap["PixelFormat"].value = PixelFormat.RGB8
        # Set the following when Persistent IP is set on the camera
        nodemap["GevPersistentARPConflictDetectionEnable"].value = False

        # Configure depth nodemap
        nodemap = self._depth_device.nodemap
        stream_nodemap = self._depth_device.tl_stream_nodemap
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
        nodemap["PixelFormat"].value = PixelFormat.Coord3D_ABCY16
        # Set depth camera specific node values
        nodemap["Scan3dOperatingMode"].value = "Distance3000mmSingleFreq"
        self._xyz_scale = nodemap["Scan3dCoordinateScale"].value
        nodemap["Scan3dCoordinateSelector"].value = "CoordinateA"
        x_offset = nodemap["Scan3dCoordinateOffset"].value
        nodemap["Scan3dCoordinateSelector"].value = "CoordinateB"
        y_offset = nodemap["Scan3dCoordinateOffset"].value
        nodemap["Scan3dCoordinateSelector"].value = "CoordinateC"
        z_offset = nodemap["Scan3dCoordinateOffset"].value
        self._xyz_offset = (x_offset, y_offset, z_offset)
        # Set the following when Persistent IP is set on the camera
        nodemap["GevPersistentARPConflictDetectionEnable"].value = False

        # Set auto exposure
        self.set_exposure(-1)

        # Start streams
        self._color_device.start_stream(1)
        self._logger.info(f"Device {self.color_camera_serial_number} is now streaming")
        self._depth_device.start_stream(1)
        self._logger.info(f"Device {self.depth_camera_serial_number} is now streaming")

    def set_exposure(self, exposure_us: float):
        """
        Set the exposure time of the camera.

        Args:
            exposure_us (float): Exposure time in microseconds.
        """
        if not self.is_connected:
            return

        nodemap = self._color_device.nodemap
        exposureAutoNode = nodemap["ExposureAuto"]
        exposureTimeNode = nodemap["ExposureTime"]
        if exposure_us < 0:
            exposureAutoNode.value = "Continuous"
        elif exposureTimeNode is not None and exposureTimeNode.is_writable:
            exposureAutoNode.value = "Off"
            if exposure_us > exposureTimeNode.max:
                exposureTimeNode.value = exposureTimeNode.max
            elif exposure_us < exposureTimeNode.min:
                exposureTimeNode.value = exposureTimeNode.min
            else:
                exposureTimeNode.value = exposure_us

    def get_color_frame(self) -> Optional[np.ndarray]:
        if self._color_device is None:
            return None

        # get_buffer must be called after start_stream and before stop_stream (or
        # system.destroy_device), and buffers must be requeued
        buffer = self._color_device.get_buffer()

        # Convert to numpy array
        # buffer is a list of (buffer.width * buffer.height * 3) uint8s
        buffer_bytes_per_pixel = int(len(buffer.data) / (buffer.width * buffer.height))
        np_array = np.asarray(buffer.data, dtype=np.uint8)
        np_array = np_array.reshape(buffer.height, buffer.width, buffer_bytes_per_pixel)

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

        np_array = np_array.reshape(buffer.height, buffer.width)

        self._depth_device.requeue_buffer(buffer)
        return np_array

    def get_frame(self) -> Optional[LucidFrame]:
        color_frame = self.get_color_frame()
        depth_frame = self.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None

        # depth_frame is a numpy structured array containing both xyz and intensity data
        # depth_frame_intensity = (depth_frame["i"] / 256).astype(np.uint8)
        depth_frame_xyz = np.stack(
            [depth_frame["x"], depth_frame["y"], depth_frame["z"]], axis=-1
        )
        return LucidFrame(
            color_frame,
            depth_frame_xyz,  # type: ignore
            time.time() * 1000,
            self._color_camera_intrinsic_matrix,
            self._color_camera_distortion_coeffs,
            self._depth_camera_intrinsic_matrix,
            self._depth_camera_distortion_coeffs,
            self._color_to_depth_extrinsic_matrix,
        )


def _get_frame(output_dir):
    output_dir = os.path.expanduser(output_dir)
    package_share_directory = get_package_share_directory("camera_control")
    calibration_params_dir = os.path.join(package_share_directory, "calibration_params")
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
    helios_to_triton_extrinsic_matrix = np.load(
        os.path.join(calibration_params_dir, "helios_to_triton_extrinsic_matrix.npy")
    )
    triton_to_helios_extrinsic_matrix = invert_extrinsic_matrix(
        helios_to_triton_extrinsic_matrix  # type: ignore
    )

    camera = LucidRgbd(
        TRITON_FRAME_SIZE,
        "241300039",
        triton_intrinsic_matrix,  # type: ignore
        triton_distortion_coeffs,  # type: ignore
        "241400544",
        helios_intrinsic_matrix,  # type: ignore
        helios_distortion_coeffs,  # type: ignore
        triton_to_helios_extrinsic_matrix,
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
    package_share_directory = get_package_share_directory("camera_control")
    calibration_params_dir = os.path.join(package_share_directory, "calibration_params")
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
    helios_to_triton_extrinsic_matrix = np.load(
        os.path.join(calibration_params_dir, "helios_to_triton_extrinsic_matrix.npy")
    )
    triton_to_helios_extrinsic_matrix = invert_extrinsic_matrix(
        helios_to_triton_extrinsic_matrix  # type: ignore
    )
    triton_mono_image = np.load(os.path.expanduser(triton_mono_image_path))
    triton_image = cv2.cvtColor(triton_mono_image, cv2.COLOR_GRAY2RGB)
    helios_xyz = np.load(os.path.expanduser(helios_xyz_path))

    frame = LucidFrame(
        triton_image,
        helios_xyz,  # type: ignore
        time.time() * 1000,
        triton_intrinsic_matrix,  # type: ignore
        triton_distortion_coeffs,  # type: ignore
        helios_intrinsic_matrix,  # type: ignore
        helios_distortion_coeffs,  # type: ignore
        triton_to_helios_extrinsic_matrix,  # type: ignore
    )

    depth_pixel = frame.get_corresponding_depth_pixel(color_pixel)
    frame.get_position(color_pixel)

    helios_intensity_image = np.load(os.path.expanduser(helios_intensity_image))
    helios_intensity_image = cv2.cvtColor(helios_intensity_image, cv2.COLOR_GRAY2RGB)

    cv2.drawMarker(
        triton_image,
        color_pixel,
        (0, 0, 255),
        markerType=cv2.MARKER_STAR,
        markerSize=40,
        thickness=2,
    )
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


def _get_extrinsic(
    triton_mono_image_path, helios_intensity_image_path, helios_xyz_path, output_path
):
    output_path = os.path.expanduser(output_path)
    package_share_directory = get_package_share_directory("camera_control")
    calibration_params_dir = os.path.join(package_share_directory, "calibration_params")
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
    helios_to_triton_extrinsic_matrix = construct_extrinsic_matrix(rvec, tvec)

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(
        output_path,
        helios_to_triton_extrinsic_matrix,
    )


def tuple_type(arg_string):
    try:
        # Parse the input string as a tuple
        parsed_tuple = tuple(map(int, arg_string.strip("()").split(",")))
        return parsed_tuple
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid tuple value: {arg_string}")


if __name__ == "__main__":
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
    get_position_parser.add_argument("--pixel", type=tuple_type, default=f"(100, 200)")
    get_position_parser.add_argument(
        "--triton_mono_image", type=str, default=None, required=True
    )
    get_position_parser.add_argument(
        "--helios_intensity_image", type=str, default=None, required=True
    )
    get_position_parser.add_argument(
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
    else:
        print("Invalid command.")
