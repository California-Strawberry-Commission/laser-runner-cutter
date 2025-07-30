import argparse
import concurrent.futures
import ctypes
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from enum import Enum, auto
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
from arena_api.enums import PixelFormat
from arena_api.system import system

from .lucid_frame import LucidFrame
from .rgbd_camera import RgbdCamera, State
from .rgbd_frame import RgbdFrame

COLOR_CAMERA_MODEL_PREFIXES = ["ATL", "ATX", "PHX", "TRI", "TRT"]
DEPTH_CAMERA_MODEL_PREFIXES = ["HTP", "HLT", "HTR", "HTW"]


class CaptureMode(Enum):
    CONTINUOUS = (
        auto()
    )  # Continuous capture, with cameras synced using the color camera as the trigger signal
    SINGLE_FRAME = (
        auto()
    )  # Single frame capture, with cameras synced using the color camera as the trigger signal
    CONTINUOUS_PTP = (
        auto()
    )  # Continuous capture, with cameras synced using Precision Time Protocol


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
        color_frame_size: Tuple[int, int] = (2048, 1536),
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
            color_frame_size (Tuple[int, int]): Desired frame size of the color camera.
            state_change_callback (Optional[Callable[[State], None]]): Callback that gets called when the camera device state changes.
            logger (Optional[logging.Logger]): Logger
        """
        self.color_camera_serial_number = color_camera_serial_number
        self.color_frame_size = color_frame_size
        self._color_frame_offset = (0, 0)
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

        self._cv = (
            threading.Condition()
        )  # For syncing acquisition and connection threads
        self._setup_cv = (
            threading.Condition()
        )  # For syncing setup wait to the connection thread
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

    def start(
        self,
        exposure_us: float = -1.0,
        gain_db: float = -1.0,
        frame_callback: Optional[Callable[[RgbdFrame], None]] = None,
        **kwargs,
    ):
        """
        Connects device and starts streaming.

        Args:
            exposure_us (float): Exposure time in microseconds. A negative value sets auto exposure.
            gain_db (float): Gain level in dB. A negative value sets auto gain.
            frame_callback (Optional[Callable[[RgbdFrame], None]]): Callback that gets called when a new frame is available.
        """
        if self._is_running:
            return

        self._is_running = True
        self._call_state_change_callback()

        # Start connection thread and acquisition thread
        capture_mode = kwargs.get("capture_mode", CaptureMode.CONTINUOUS)
        self._connection_thread = threading.Thread(
            target=self._connection_thread_fn,
            args=(
                exposure_us,
                gain_db,
                capture_mode,
            ),
            daemon=True,
        )
        self._connection_thread.start()
        # We don't need an acquisition loop thread when doing single frame captures
        if capture_mode != CaptureMode.SINGLE_FRAME:
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

    def _connection_thread_fn(
        self,
        exposure_us: float = -1.0,
        gain_db: float = -1.0,
        capture_mode: CaptureMode = CaptureMode.CONTINUOUS,
    ):
        device_connected = False
        device_was_ever_connected = False

        with self._cv:
            while self._is_running:
                if device_connected:
                    self._logger.info(f"Devices found. Signaling acquisition thread")
                    self._cv.notify()
                    self._cv.wait()

                # Clean up existing connection if needed
                self._stop_stream()

                # Create new connection
                if self._is_running:
                    device_connected = False

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
                        self._logger.info(
                            f"Device (color, model={color_device_info['model']}, serial={color_device_info['serial']}, firmware_ver={color_device_info['version']}) and device (depth, model={depth_device_info['model']}, serial={depth_device_info['serial']}, firmware_ver={depth_device_info['version']}) found"
                        )
                        # Only set exposure/gain if this is the first time the device is connected
                        self._start_stream(
                            color_device_info,
                            depth_device_info,
                            **(
                                {
                                    "capture_mode": capture_mode,
                                }
                                if device_was_ever_connected
                                else {
                                    "exposure_us": exposure_us,
                                    "gain_db": gain_db,
                                    "capture_mode": capture_mode,
                                }
                            ),
                        )
                        device_was_ever_connected = True
                        device_connected = True
                        with self._setup_cv:
                            self._setup_cv.notify_all()
                    else:
                        self._logger.warning(
                            f"Either device (color, serial={self.color_camera_serial_number}) or device (depth, serial={self.depth_camera_serial_number}) was not found"
                        )
                        time.sleep(5)

            # Clean up existing connection
            self._stop_stream()

        self._logger.info(f"Terminating connection thread")

    def get_frame(self) -> Optional[LucidFrame]:
        # When in SingleFrame mode, we manually fire AcquisitionStart and AcquisitionStop
        color_nodemap = self._color_device.nodemap
        depth_nodemap = self._depth_device.nodemap
        is_single_frame_mode = color_nodemap["AcquisitionMode"].value == "SingleFrame"
        if is_single_frame_mode:
            color_nodemap["AcquisitionStart"].execute()
            depth_nodemap["AcquisitionStart"].execute()
            # Add a delay to ensure signal fires before attempting to read buffer
            time.sleep(0.1)

        try:
            frame = self._get_rgbd_frame()
        except Exception as e:
            self._logger.error(
                f"There was an issue with the camera: {e}. Signaling connection thread"
            )
            self._cv.notify()
            return None

        if frame is None:
            self._logger.error(f"No frame available. Signaling connection thread")
            self._cv.notify()
            return None

        if is_single_frame_mode:
            color_nodemap["AcquisitionStop"].execute()
            depth_nodemap["AcquisitionStop"].execute()

        return frame

    def _acquisition_thread_fn(
        self, frame_callback: Optional[Callable[[RgbdFrame], None]] = None
    ):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            with self._cv:
                while self._is_running:
                    try:
                        frame = self._get_rgbd_frame(executor)
                    except Exception as e:
                        self._logger.error(
                            f"There was an issue with the camera: {e}. Signaling connection thread"
                        )
                        self._cv.notify()
                        self._cv.wait()
                        continue

                    if frame is None:
                        self._logger.error(
                            f"No frame available. Signaling connection thread"
                        )
                        self._cv.notify()
                        self._cv.wait()
                        continue

                    if frame_callback is not None:
                        frame_callback(frame)

        self._logger.info(f"Terminating acquisition thread")

    def _start_stream(
        self,
        color_device_info,
        depth_device_info,
        exposure_us: Optional[float] = None,
        gain_db: Optional[float] = None,
        capture_mode: CaptureMode = CaptureMode.CONTINUOUS,
    ):
        if self._color_device is not None or self._depth_device is not None:
            return

        devices = system.create_device([color_device_info, depth_device_info])
        self._color_device = devices[0]
        self._depth_device = devices[1]

        def set_network_settings(device):
            device_nodemap = device.nodemap
            stream_nodemap = device.tl_stream_nodemap
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
            # Set the following when Persistent IP is set on the camera
            device_nodemap["GevPersistentARPConflictDetectionEnable"].value = False

        # Configure color device nodemap
        set_network_settings(self._color_device)
        color_nodemap = self._color_device.nodemap
        # Set frame size and pixel format
        # Use BayerRG (RGGB pattern) to achieve streaming at 30 FPS at max resolution. We will
        # demosaic to RGB on the host device.
        color_nodemap["PixelFormat"].value = PixelFormat.BayerRG8
        # Reset ROI (Region of Interest) offset, as it persists on the device.
        color_nodemap["OffsetX"].value = 0
        color_nodemap["OffsetY"].value = 0
        max_width = color_nodemap["Width"].max
        max_height = color_nodemap["Height"].max
        # Check that the desired color frame size is valid before attempting to set
        if (
            self.color_frame_size[0] <= 0
            or max_width < self.color_frame_size[0]
            or self.color_frame_size[1] <= 0
            or max_height < self.color_frame_size[1]
        ):
            raise Exception(
                f"Invalid color frame size specified: {self.color_frame_size[0]}x{self.color_frame_size[1]}. Max size is {max_width}x{max_height}."
            )
        color_nodemap["Width"].value = self.color_frame_size[0]
        color_nodemap["Height"].value = self.color_frame_size[1]
        # Set the ROI offset to be the center of the full frame
        self._color_frame_offset = (
            (max_width - self.color_frame_size[0]) // 2,
            (max_height - self.color_frame_size[1]) // 2,
        )
        color_nodemap["OffsetX"].value = self._color_frame_offset[0]
        color_nodemap["OffsetY"].value = self._color_frame_offset[1]

        # Configure depth device nodemap
        set_network_settings(self._depth_device)
        depth_nodemap = self._depth_device.nodemap
        # Set pixel format
        self.depth_frame_size = (
            depth_nodemap["Width"].value,
            depth_nodemap["Height"].value,
        )
        depth_nodemap["PixelFormat"].value = PixelFormat.Coord3D_ABCY16
        # Set Scan 3D node values
        depth_nodemap["Scan3dOperatingMode"].value = "Distance3000mmSingleFreq"
        depth_nodemap["ExposureTimeSelector"].value = "Exp88Us"
        self._xyz_scale = depth_nodemap["Scan3dCoordinateScale"].value
        depth_nodemap["Scan3dCoordinateSelector"].value = "CoordinateA"
        x_offset = depth_nodemap["Scan3dCoordinateOffset"].value
        depth_nodemap["Scan3dCoordinateSelector"].value = "CoordinateB"
        y_offset = depth_nodemap["Scan3dCoordinateOffset"].value
        depth_nodemap["Scan3dCoordinateSelector"].value = "CoordinateC"
        z_offset = depth_nodemap["Scan3dCoordinateOffset"].value
        self._xyz_offset = (x_offset, y_offset, z_offset)
        # Set confidence threshold
        depth_nodemap["Scan3dConfidenceThresholdEnable"].value = True
        depth_nodemap["Scan3dConfidenceThresholdMin"].value = 500

        if capture_mode == CaptureMode.CONTINUOUS_PTP:
            # Enable PTP Sync
            # See https://support.thinklucid.com/app-note-multi-camera-synchronization-using-ptp-and-scheduled-action-commands/
            color_nodemap["AcquisitionMode"].value = "Continuous"
            depth_nodemap["AcquisitionMode"].value = "Continuous"
            color_nodemap["PtpEnable"].value = True
            depth_nodemap["PtpEnable"].value = True
            color_nodemap["PtpSlaveOnly"].value = False
            depth_nodemap["PtpSlaveOnly"].value = True
            color_nodemap["AcquisitionStartMode"].value = "PTPSync"
            depth_nodemap["AcquisitionStartMode"].value = "PTPSync"

            # Set frame rate
            # When PTPSync mode is turned on, the AcquisitionFrameRate node will not be controlling
            # the actual frame rate but it might cap the max achievable frame rate. Instead, PTPSyncFrameRate
            # is used to change the frame rate. However, in order to avoid capping the frame rate, we need to
            # set AcquisitionFrameRate to its maximum value.
            color_fps = color_nodemap["AcquisitionFrameRate"].max
            color_nodemap["AcquisitionFrameRate"].value = color_fps
            depth_fps = depth_nodemap["AcquisitionFrameRate"].max
            depth_nodemap["AcquisitionFrameRate"].value = depth_fps
            # PTPSyncFrameRate needs to always be less than AcquisitionFrameRate
            # Use the min frame rate across both devices
            fps = min(color_fps, depth_fps)
            color_nodemap["PTPSyncFrameRate"].value = fps
            depth_nodemap["PTPSyncFrameRate"].value = fps

            # Set packet delay and frame transmission delay
            # See https://support.thinklucid.com/app-note-bandwidth-sharing-in-multi-camera-systems/
            # With packet size of 9000 bytes on a 1 Gbps link, the packet delay is:
            # (9000 bytes * 8 ns/byte) * 1.1 buffer = 79200 ns
            color_nodemap["GevSCFTD"].value = 0
            depth_nodemap["GevSCFTD"].value = 79200
            color_nodemap["GevSCPD"].value = 79200
            depth_nodemap["GevSCPD"].value = 79200

            # Wait for PTP status to be set
            self._logger.info(
                f"Waiting for device (color, serial={self.color_camera_serial_number}) to become Master. Current state: {color_nodemap['PtpStatus'].value}"
            )
            while color_nodemap["PtpStatus"].value != "Master":
                time.sleep(2)
            self._logger.info(
                f"Waiting for device (depth, serial={self.depth_camera_serial_number}) to become Slave. Current state: {depth_nodemap['PtpStatus'].value}"
            )
            while depth_nodemap["PtpStatus"].value != "Slave":
                time.sleep(2)
        else:
            color_nodemap["PtpEnable"].value = False
            depth_nodemap["PtpEnable"].value = False
            color_nodemap["AcquisitionStartMode"].value = "Normal"
            depth_nodemap["AcquisitionStartMode"].value = "Normal"

            color_nodemap["GevSCFTD"].value = 0
            depth_nodemap["GevSCFTD"].value = 0
            color_nodemap["GevSCPD"].value = 80
            depth_nodemap["GevSCPD"].value = 80

            # Select GPIO line to output strobe signal on color camera
            # See https://support.thinklucid.com/app-note-using-gpio-on-lucid-cameras/
            color_nodemap["LineSelector"].value = (
                "Line3"  # Non-isolated bi-directional GPIO channel
            )
            color_nodemap["LineMode"].value = "Output"
            color_nodemap["LineSource"].value = "ExposureActive"
            color_nodemap["LineSelector"].value = "Line1"  # Opto-isolated output
            color_nodemap["LineMode"].value = "Output"
            color_nodemap["LineSource"].value = "ExposureActive"
            # TODO: Enable trigger mode on depth camera. See https://support.thinklucid.com/app-note-using-gpio-on-lucid-cameras/#config

            if capture_mode == CaptureMode.CONTINUOUS:
                color_nodemap["AcquisitionMode"].value = "Continuous"
                depth_nodemap["AcquisitionMode"].value = "Continuous"
            else:
                color_nodemap["AcquisitionMode"].value = "SingleFrame"
                depth_nodemap["AcquisitionMode"].value = "SingleFrame"

        # Set exposure and gain
        if exposure_us is not None:
            self.exposure_us = exposure_us
        if gain_db is not None:
            self.gain_db = gain_db

        # Start streams
        num_buffers = 1 if capture_mode == CaptureMode.SINGLE_FRAME else 10
        self._color_device.start_stream(num_buffers)
        self._logger.info(
            f"Device (color, serial={self.color_camera_serial_number}) is now streaming with resolution ({self.color_frame_size[0]}, {self.color_frame_size[1]})"
        )
        self._depth_device.start_stream(num_buffers)
        self._logger.info(
            f"Device (depth, serial={self.depth_camera_serial_number}) is now streaming with resolution ({self.depth_frame_size[0]}, {self.depth_frame_size[1]})"
        )
        self._call_state_change_callback()

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

    def get_color_device_temperature(self) -> float:
        if self.state != State.STREAMING:
            return 0.0

        return self._color_device.nodemap["DeviceTemperature"].value

    def get_depth_device_temperature(self) -> float:
        if self.state != State.STREAMING:
            return 0.0

        return self._depth_device.nodemap["DeviceTemperature"].value

    def _get_rgbd_frame(
        self, executor: Optional[concurrent.futures.Executor] = None
    ) -> Optional[LucidFrame]:
        if executor is not None:
            future_color_frame = executor.submit(self._get_color_frame)
            future_depth_frame = executor.submit(self._get_depth_frame)

            color_frame = future_color_frame.result()
            depth_frame = future_depth_frame.result()
        else:
            color_frame = self._get_color_frame()
            depth_frame = self._get_depth_frame()

        if color_frame is None or depth_frame is None:
            return None

        # Convert BayerRG8 to RGB8
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BayerRGGB2RGB)

        # depth_frame is a numpy structured array containing both xyz and intensity data
        # depth_frame_intensity = (depth_frame["i"] / 256).astype(np.uint8)
        depth_frame_xyz = np.stack(
            [depth_frame["x"], depth_frame["y"], depth_frame["z"]], axis=-1
        )

        return LucidFrame(
            color_frame,
            depth_frame_xyz,
            time.time() * 1000,
            self._color_camera_intrinsic_matrix,
            self._color_camera_distortion_coeffs,
            self._depth_camera_intrinsic_matrix,
            self._depth_camera_distortion_coeffs,
            self._xyz_to_color_camera_extrinsic_matrix,
            self._xyz_to_depth_camera_extrinsic_matrix,
            color_frame_offset=self._color_frame_offset,
        )

    def _get_color_frame(
        self, timeout_ms: Optional[int] = None
    ) -> Optional[np.ndarray]:
        if self._color_device is None:
            return None

        # get_buffer must be called after start_stream and before stop_stream (or
        # system.destroy_device), and buffers must be requeued
        buffer = self._color_device.get_buffer(timeout=timeout_ms)

        # Convert to numpy array
        # buffer is a list of (buffer.width * buffer.height * num_channels) uint8s
        num_channels = 1  # one channel per pixel for BayerRG8
        np_array = np.ndarray(
            buffer=(
                ctypes.c_ubyte * num_channels * buffer.width * buffer.height
            ).from_address(ctypes.addressof(buffer.pbytes)),
            dtype=np.uint8,
            shape=(buffer.height, buffer.width, num_channels),
        )

        self._color_device.requeue_buffer(buffer)

        return np_array

    def _get_depth_frame(
        self, timeout_ms: Optional[int] = None
    ) -> Optional[np.ndarray]:
        if self._depth_device is None:
            return None

        # get_buffer must be called after start_stream and before stop_stream (or
        # system.destroy_device), and buffers must be requeued
        buffer = self._depth_device.get_buffer(timeout=timeout_ms)

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

    def _wait_for_setup(self):
        # Wait for the camera to start streaming
        with self._setup_cv:
            if not self.state == State.STREAMING:
                self._setup_cv.wait(timeout=10)  # Wait 10 seconds for camera
            if not self.state == State.STREAMING:
                raise RuntimeError("Camera failed to start streaming")

    def _wait_for_frame(self):
        # Check color AcquisitionMode for SingleFrame
        if self._color_device.nodemap["AcquisitionMode"].value == "SingleFrame":
            self._color_device.nodemap["AcquisitionStart"].execute()
            # Give 5 seconds to wait for first frame capture
            timeout = time.time() + 5.0
            while not self._color_device.has_buffer():
                if time.time() > timeout:
                    raise RuntimeError("Camera failed to capture first frame")
                time.sleep(0.01)
            self._color_device.nodemap["AcquisitionStop"].execute()

        # Check depth AcquisitionMode for SingleFrame
        if self._depth_device.nodemap["AcquisitionMode"].value == "SingleFrame":
            self._depth_device.nodemap["AcquisitionStart"].execute()
            # Give 5 seconds to wait for first frame capture
            timeout = time.time() + 5.0
            while not self._color_device.has_buffer():
                if time.time() > timeout:
                    raise RuntimeError("Camera failed to capture first frame")
                time.sleep(0.01)
            self._depth_device.nodemap["AcquisitionStop"].execute()


def create_lucid_rgbd_camera(
    color_camera_serial_number: Optional[str] = None,
    depth_camera_serial_number: Optional[str] = None,
    color_frame_size: Tuple[int, int] = (2048, 1536),
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
        color_frame_size=color_frame_size,
        state_change_callback=state_change_callback,
        logger=logger,
    )


def _get_calibration_params_dir():
    package_share_directory = get_package_share_directory("camera_control")
    return os.path.join(package_share_directory, "calibration_params")


def _get_frame(output_dir):
    output_dir = os.path.expanduser(output_dir)

    camera = create_lucid_rgbd_camera()
    camera.start()

    camera._wait_for_setup()
    camera._wait_for_frame()

    color_frame = camera._get_color_frame()
    depth_frame = camera._get_depth_frame()

    if color_frame is not None and depth_frame is not None:
        # Convert BayerRG8 to RGB8 first
        color_frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BayerRGGB2RGB)
        triton_mono_image = cv2.cvtColor(color_frame_rgb, cv2.COLOR_RGB2GRAY)
        helios_intensity_image = (depth_frame["i"] / 256).astype(np.uint8)
        helios_xyz = np.stack(
            [depth_frame["x"], depth_frame["y"], depth_frame["z"]], axis=-1
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(
            os.path.join(output_dir, "triton_color_image.npy"),
            color_frame_rgb,
        )
        cv2.imwrite(
            os.path.join(output_dir, "triton_color_image.png"),
            cv2.cvtColor(color_frame_rgb, cv2.COLOR_RGB2BGR),
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


def _get_heatmap_frame(output_dir):
    output_dir = os.path.expanduser(output_dir)

    camera = create_lucid_rgbd_camera()
    camera.start()

    camera._wait_for_setup()
    camera._wait_for_frame()

    depth_frame = camera._get_depth_frame()

    if depth_frame is not None:
        # Extract depth values (z-component) from the structured array
        depth_values = depth_frame["z"]

        # Create a mask for valid depth values (not -1)
        valid_mask = depth_values != -1.0

        # Create heatmap visualization
        if np.any(valid_mask):
            # Normalize depth values for visualization
            valid_depths = depth_values[valid_mask]
            min_depth = np.min(valid_depths)
            max_depth = np.max(valid_depths)

            # Create normalized depth image
            normalized_depth = np.zeros_like(depth_values, dtype=np.float32)
            if max_depth > min_depth:
                normalized_depth[valid_mask] = (
                    depth_values[valid_mask] - min_depth
                ) / (max_depth - min_depth)

            # Convert to 8-bit for colormap application
            depth_8bit = (normalized_depth * 255).astype(np.uint8)

            # Apply colormap (COLORMAP_JET gives a nice heatmap: blue=far, red=close)
            heatmap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

            # Set invalid pixels to black
            heatmap[~valid_mask] = [0, 0, 0]
        else:
            # If no valid depth data, create a black image
            heatmap = np.zeros(
                (depth_values.shape[0], depth_values.shape[1], 3), dtype=np.uint8
            )

        # Save all outputs
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save depth values as grayscale
        depth_grayscale = (normalized_depth * 255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(output_dir, "depth_grayscale.png"),
            depth_grayscale,
        )
        # Save heatmap
        cv2.imwrite(
            os.path.join(output_dir, "depth_heatmap.png"),
            heatmap,
        )

        print(f"Heatmap saved successfully!")
        print(f"Depth range: {min_depth:.1f}mm to {max_depth:.1f}mm")
        print(f"Valid pixels: {np.sum(valid_mask)} / {valid_mask.size}")

    camera.stop()


def _stream_heatmap_to_terminal():
    if shutil.which("chafa") is None:
        print(
            "chafa is not installed or not in PATH. Please install chafa to use this feature."
        )
        return

    camera = create_lucid_rgbd_camera()
    camera.start()
    try:
        camera._wait_for_setup()
        print("Press Ctrl+C to stop streaming.")
        # Use a persistent temporary file for chafa to avoid flicker
        with tempfile.NamedTemporaryFile(suffix=".png") as tmpfile:
            while True:
                depth_frame = camera._get_depth_frame()
                if depth_frame is not None:
                    depth_values = depth_frame["z"]
                    valid_mask = depth_values != -1.0
                    if np.any(valid_mask):
                        valid_depths = depth_values[valid_mask]
                        min_depth = np.min(valid_depths)
                        max_depth = np.max(valid_depths)
                        normalized_depth = np.zeros_like(depth_values, dtype=np.float32)
                        if max_depth > min_depth:
                            normalized_depth[valid_mask] = (
                                depth_values[valid_mask] - min_depth
                            ) / (max_depth - min_depth)
                        depth_8bit = (normalized_depth * 255).astype(np.uint8)
                        heatmap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
                        heatmap[~valid_mask] = [0, 0, 0]
                    else:
                        heatmap = np.zeros(
                            (depth_values.shape[0], depth_values.shape[1], 3),
                            dtype=np.uint8,
                        )

                    # Write heatmap to the persistent temporary PNG file
                    cv2.imwrite(tmpfile.name, heatmap)
                    # Use chafa to render the image in the terminal, overwrite previous output
                    subprocess.run(
                        ["chafa", "--size=80x40", tmpfile.name],
                        stdout=sys.stdout,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    print("No depth frame available.")
                # Use a short sleep to control frame rate (adjust as needed)
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopped streaming.")
    finally:
        camera.stop()


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
    triton_image = cv2.resize(triton_image, (round(w / 2), round(h / 2)))
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
    xyz_to_triton_extrinsic_matrix = np.load(
        os.path.join(calibration_params_dir, "xyz_to_triton_extrinsic_matrix.npy")
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
        return (round(pixel[0]), round(pixel[1]))

    color_pixel = project_position(
        xyz_mm,
        triton_intrinsic_matrix,
        triton_distortion_coeffs,
        xyz_to_triton_extrinsic_matrix,
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
    triton_image = cv2.resize(triton_image, (round(w / 2), round(h / 2)))
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

    get_heatmap_parser = subparsers.add_parser(
        "get_heatmap",
        help="Capture frames and generate a heatmapped depth visualization",
    )
    get_heatmap_parser.add_argument(
        "--output_dir", type=str, default=None, required=True
    )

    stream_heatmap_parser = subparsers.add_parser(
        "stream_heatmap",
        help="Stream depth heatmap visualization to terminal using chafa",
    )

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
    elif args.command == "get_heatmap":
        _get_heatmap_frame(args.output_dir)
    elif args.command == "stream_heatmap":
        _stream_heatmap_to_terminal()
    elif args.command == "get_position":
        _get_position(
            args.pixel,
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
