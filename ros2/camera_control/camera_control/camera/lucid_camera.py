import argparse
import concurrent.futures
import ctypes
import logging
import os
import sys
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
from arena_api.enums import PixelFormat
from arena_api.system import system

from .calibration import (
    construct_extrinsic_matrix,
    create_blob_detector,
)
from .rgbd_camera import RgbdCamera, State
from .rgbd_frame import RgbdFrame
from .lucid_frame import LucidFrame
import threading

COLOR_CAMERA_MODEL_PREFIXES = ["ATL", "ATX", "PHX", "TRI", "TRT"]
DEPTH_CAMERA_MODEL_PREFIXES = ["HTP", "HLT", "HTR", "HTW"]


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


class LucidRgbdCamera(RgbdCamera):
    """
    Combined interface for a LUCID color camera (such as the Triton) and depth camera (such as the
    Helios2). Unfortunately, we cannot modularize the LUCID cameras into individual instances, as
    calling system.device_infos is time consuming, and calling system.create_device more than once
    results in an error.
    """

    color_camera_serial_number: Optional[str]
    depth_camera_serial_number: Optional[str]
    color_frame_size: Tuple[int, int]
    depth_frame_size: Tuple[int, int]

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
        state_change_callback: Optional[Callable[[State], None]] = None,
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
            state_change_callback (Optional[Callable[[State], None]]): Callback that gets called when the camera device state changes.
            logger (Optional[logging.Logger]): Logger
        """
        self.color_camera_serial_number = color_camera_serial_number
        self.color_frame_size = (0, 0)
        self._color_camera_intrinsic_matrix = color_camera_intrinsic_matrix
        self._color_camera_distortion_coeffs = color_camera_distortion_coeffs
        self.depth_camera_serial_number = depth_camera_serial_number
        self.depth_frame_size = (0, 0)
        self._depth_camera_intrinsic_matrix = depth_camera_intrinsic_matrix
        self._depth_camera_distortion_coeffs = depth_camera_distortion_coeffs
        self._xyz_to_color_camera_extrinsic_matrix = (
            xyz_to_color_camera_extrinsic_matrix
        )
        self._xyz_to_depth_camera_extrinsic_matrix = (
            xyz_to_depth_camera_extrinsic_matrix
        )
        self._state_change_callback = state_change_callback
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)

        self._color_device = None
        self._depth_device = None
        self._exposure_us = 0.0
        self._gain_db = 0.0

        self._cv = threading.Condition()
        self._is_running = False
        self._connection_thread = None
        self._acquisition_thread = None

    @property
    def state(self) -> State:
        """
        Returns:
            State: Current state of the camera device.
        """
        if self._color_device is not None and self._depth_device is not None:
            return State.STREAMING
        else:
            if self._is_running:
                return State.CONNECTING
            else:
                return State.DISCONNECTED

    def start(self, frame_callback: Optional[Callable[[RgbdFrame], None]] = None):
        """
        Connects device and starts streaming.

        Args:
            frame_callback (Optional[Callable[[RgbdFrame], None]]): Callback that gets called when a new frame is available.
        """
        if self._is_running:
            return

        self._is_running = True
        self._call_state_change_callback()

        # Start connection thread and acquisition thread
        self._connection_thread = threading.Thread(
            target=self._connection_thread_fn, daemon=True
        )
        self._connection_thread.start()
        self._acquisition_thread = threading.Thread(
            target=self._acquisition_thread_fn, args=(frame_callback,), daemon=True
        )
        self._acquisition_thread.start()

    def stop(self):
        """
        Stops streaming and disconnects device.
        """
        if not self._is_running:
            return

        self._is_running = False
        self._call_state_change_callback()

        with self._cv:
            self._cv.notify_all()  # Notify all waiting threads to wake up

        # Join the threads to ensure they have finished
        if self._connection_thread is not None:
            self._connection_thread.join()
        if self._acquisition_thread is not None:
            self._acquisition_thread.join()

    def _connection_thread_fn(self):
        found_devices = False

        with self._cv:
            while self._is_running:
                if found_devices:
                    self._logger.info(f"Devices found. Signaling acquisition thread")
                    self._cv.notify()
                    self._cv.wait()

                # Clean up existing connection if needed
                self._stop_stream()

                # Create new connection
                if self._is_running:
                    found_devices = False

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

                    # If the devices are connected, set up and start streaming
                    if color_device_info is not None and depth_device_info is not None:
                        found_devices = True
                        self._logger.info(
                            f"Device (color) {self.color_camera_serial_number} and device (depth) {self.depth_camera_serial_number} found"
                        )
                        self._start_stream(color_device_info, depth_device_info)
                    else:
                        self._logger.warn(
                            f"Either device (color) {self.color_camera_serial_number} or device (depth) {self.depth_camera_serial_number} was not found"
                        )
                        time.sleep(5)

            # Clean up existing connection
            self._stop_stream()

        self._logger.info(f"Terminating connection thread")

    def _acquisition_thread_fn(
        self, frame_callback: Optional[Callable[[RgbdFrame], None]] = None
    ):
        with self._cv:
            while self._is_running:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_color_frame = executor.submit(self._get_color_frame)
                    future_depth_frame = executor.submit(self._get_depth_frame)

                    try:
                        color_frame = future_color_frame.result()
                        depth_frame = future_depth_frame.result()
                    except:
                        self._logger.warn(
                            f"There was an issue with the camera. Signaling connection thread"
                        )
                        self._cv.notify()
                        self._cv.wait()
                        continue

                if color_frame is None or depth_frame is None:
                    self._logger.warn(
                        f"No frame available. Signaling connection thread"
                    )
                    self._cv.notify()
                    self._cv.wait()
                    continue

                # depth_frame is a numpy structured array containing both xyz and intensity data
                # depth_frame_intensity = (depth_frame["i"] / 256).astype(np.uint8)
                depth_frame_xyz = np.stack(
                    [depth_frame["x"], depth_frame["y"], depth_frame["z"]], axis=-1
                )

                if frame_callback is not None:
                    frame_callback(
                        LucidFrame(
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
                    )

        self._logger.info(f"Terminating acquisition thread")

    def _start_stream(self, color_device_info, depth_device_info):
        if self._color_device is not None or self._depth_device is not None:
            return

        devices = system.create_device([color_device_info, depth_device_info])
        self._color_device = devices[0]
        self._depth_device = devices[1]
        self._call_state_change_callback()

        # Configure color device nodemap
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
        self.color_frame_size = (nodemap["Width"].value, nodemap["Height"].value)
        nodemap["PixelFormat"].value = PixelFormat.RGB8
        # Set the following when Persistent IP is set on the camera
        nodemap["GevPersistentARPConflictDetectionEnable"].value = False

        # Configure depth device nodemap
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
        self.depth_frame_size = (nodemap["Width"].value, nodemap["Height"].value)
        nodemap["PixelFormat"].value = PixelFormat.Coord3D_ABCY16
        # Set Scan 3D node values
        nodemap["Scan3dOperatingMode"].value = "Distance3000mmSingleFreq"
        nodemap["ExposureTimeSelector"].value = "Exp88Us"
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
            f"Device (color) {self.color_camera_serial_number} is now streaming at {self.color_frame_size[0]}x{self.color_frame_size[1]}"
        )
        self._depth_device.start_stream(10)
        self._logger.info(
            f"Device (depth) {self.depth_camera_serial_number} is now streaming at {self.depth_frame_size[0]}x{self.depth_frame_size[1]}"
        )

    def _stop_stream(self):
        if self._color_device is None and self._depth_device is None:
            return

        self._color_device = None
        self._depth_device = None
        # Destroy all created devices. Note that this will automatically call stop_stream() for each device
        system.destroy_device()
        self._call_state_change_callback()

    def _call_state_change_callback(self):
        if self._state_change_callback is not None:
            self._state_change_callback(self.state)

    @property
    def exposure_us(self) -> float:
        """
        Returns:
            float: Exposure time of the color camera in microseconds.
        """
        if self.state != State.STREAMING:
            return 0.0

        return self._exposure_us

    @exposure_us.setter
    def exposure_us(self, exposure_us: float):
        """
        Set the exposure time of the color camera. A negative value sets auto exposure.

        Args:
            exposure_us (float): Exposure time in microseconds. A negative value sets auto exposure.
        """
        if self.state != State.STREAMING:
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
        if self.state != State.STREAMING:
            return (0.0, 0.0)

        nodemap = self._color_device.nodemap
        return (nodemap["ExposureTime"].min, nodemap["ExposureTime"].max)

    @property
    def gain_db(self) -> float:
        """
        Returns:
            float: Gain level of the color camera in dB.
        """
        if self.state != State.STREAMING:
            return 0.0

        return self._gain_db

    @gain_db.setter
    def gain_db(self, gain_db: float):
        """
        Set the gain level of the color camera.

        Args:
            gain_db (float): Gain level in dB.
        """
        if self.state != State.STREAMING:
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
        if self.state != State.STREAMING:
            return (0.0, 0.0)

        nodemap = self._color_device.nodemap
        return (nodemap["Gain"].min, nodemap["Gain"].max)

    def _get_color_frame(self) -> Optional[np.ndarray]:
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

    def _get_depth_frame(self) -> Optional[np.ndarray]:
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


def create_lucid_rgbd_camera(
    color_camera_serial_number: Optional[str] = None,
    depth_camera_serial_number: Optional[str] = None,
    state_change_callback: Optional[Callable[[State], None]] = None,
    logger: Optional[logging.Logger] = None,
) -> LucidRgbdCamera:
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
    return LucidRgbdCamera(
        triton_intrinsic_matrix,  # type: ignore
        triton_distortion_coeffs,  # type: ignore
        helios_intrinsic_matrix,  # type: ignore
        helios_distortion_coeffs,  # type: ignore
        xyz_to_triton_extrinsic_matrix,  # type: ignore
        xyz_to_helios_extrinsic_matrix,  # type: ignore
        color_camera_serial_number=color_camera_serial_number,
        depth_camera_serial_number=depth_camera_serial_number,
        state_change_callback=state_change_callback,
        logger=logger,
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
    camera.start()
    time.sleep(1)

    color_frame = camera._get_color_frame()
    depth_frame = camera._get_depth_frame()

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
