import logging
import threading
import time
from typing import List, Optional, Sequence, Tuple
from abc import ABC, abstractmethod
import cv2
import numpy as np
from arena_api.enums import PixelFormat
from arena_api.system import system
import ctypes
from .calibration import calibrate_camera
from .rgbd_camera import RgbdCamera
from .rgbd_frame import RgbdFrame

CALIBRATION_GRID_SIZE = (4, 5)
CALIBRATION_GRID_TYPE = cv2.CALIB_CB_SYMMETRIC_GRID
# General min and max possible depths pulled from RealSense examples
DEPTH_MIN_METERS = 0.1
DEPTH_MAX_METERS = 10


def _scale_grayscale_image(mono_image: np.ndarray) -> np.ndarray:
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
    triton_mono_image = _scale_grayscale_image(triton_mono_image)
    helios_intensity_image = _scale_grayscale_image(helios_intensity_image)

    # Find calibration circle centers on both cameras
    retval, triton_circle_centers = cv2.findCirclesGrid(
        triton_mono_image, grid_size, flags=grid_type, blobDetector=blob_detector
    )
    if not retval:
        print("Could not get circle centers from Triton mono image.")
        return None, None
    retval, helios_circle_centers = cv2.findCirclesGrid(
        helios_intensity_image, grid_size, flags=grid_type, blobDetector=blob_detector
    )
    if not retval:
        print("Could not get circle centers from Helios intensity image.")
        return None, None

    # Get XYZ values from the Helios 3D image that match 2D locations on the Triton (corresponding points)
    helios_circle_positions = helios_xyz_image[
        helios_circle_centers[:, 0], helios_circle_centers[:, 1]
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


class LucidCamera(ABC):
    """
    Represents a single LUCID camera.
    """

    frame_size: Tuple[int, int]
    serial_number: Optional[str]

    def __init__(
        self,
        frame_size: Tuple[int, int],
        serial_number: Optional[str] = None,
        model_prefix: Optional[str] = None,
        camera_index: int = 0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            frame_size (Tuple[int, int]): (width, height) of the frame.
            serial_number (Optional[str]): Serial number of device to connect to. If None, camera_index will be used.
            model_prefix (Optional[str]): Prefix of model name to match camera_index against. Will only be used if serial_number is None.
            camera_index (int): Index of detected camera to connect to. Will only be used if serial_number is None.
            logger (Optional[logging.Logger]): Logger
        """
        self.frame_size = frame_size
        self.serial_number = serial_number
        self._model_prefix = model_prefix
        self._camera_index = camera_index
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)
        self._check_connection = False
        self._check_connection_thread = None
        self._device = None

    @property
    def is_connected(self) -> bool:
        """
        Returns:
            bool: Whether the camera is connected.
        """
        return self._device is not None

    @property
    @abstractmethod
    def is_depth_camera(self) -> bool:
        pass

    def initialize(self):
        """
        Set up the camera. Finds the serial number if needed, enables the device, configures the
        stream, and starts the stream.
        """
        device_infos = system.device_infos

        # If we don't have a serial number of a device, attempt to find one among connected devices using camera_index
        if self.serial_number is None:
            filtered_device_infos = [
                device_info
                for device_info in device_infos
                if self._model_prefix is None
                or device_info["model"].startswith(self._model_prefix)
            ]
            if self._camera_index < 0 or self._camera_index >= len(
                filtered_device_infos
            ):
                raise Exception(
                    f"camera_index {self._camera_index} is out of bounds: {len(filtered_device_infos)} devices found with model prefix {self._model_prefix}"
                )
            self.serial_number = filtered_device_infos[self._camera_index]["serial"]

        # If the device is connected, set up and start streaming
        device_info = next(
            (
                device_info
                for device_info in device_infos
                if device_info["serial"] == self.serial_number
            ),
            None,
        )
        if device_info is not None:
            self._logger.info(f"Device {self.serial_number} found")
            self._start_stream(device_info)
        else:
            self._logger.warn(f"Device {self.serial_number} was not found")

        # TODO: use device disconnected callback
        def check_connection_thread():
            while self._check_connection:
                connected_devices = [
                    device_info["serial"] for device_info in system.device_infos
                ]
                connected = self.serial_number in connected_devices
                if self.is_connected != connected:
                    if connected:
                        self._logger.info(f"Device {self.serial_number} connected")
                        self.initialize()
                    else:
                        self._logger.info(f"Device {self.serial_number} disconnected")
                        system.destroy_device()
                        self._device = None

                time.sleep(1)

        if self._check_connection_thread is None:
            self._check_connection = True
            self._check_connection_thread = threading.Thread(
                target=check_connection_thread, daemon=True
            )
            self._check_connection_thread.start()

    def _start_stream(self, device_info):
        self._device = system.create_device(device_info)[0]
        self._configure_nodemap(self._device)
        self._device.start_stream(1)
        self._logger.info(f"Device {self.serial_number} is now streaming")

    @abstractmethod
    def _configure_nodemap(self, device):
        pass

    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        pass


class TritonCamera(LucidCamera):
    def __init__(
        self,
        frame_size: Tuple[int, int],
        serial_number: Optional[str] = None,
        camera_index: int = 0,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(
            frame_size,
            serial_number=serial_number,
            model_prefix="TRI",
            camera_index=camera_index,
            logger=logger,
        )

    @property
    def is_depth_camera(self) -> bool:
        return False

    def _configure_nodemap(self, device):
        # Configure stream
        nodemap = device.nodemap
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
        # Set frame size and pixel format
        nodemap["Width"].value = self.frame_size[0]
        nodemap["Height"].value = self.frame_size[1]
        nodemap["PixelFormat"].value = PixelFormat.RGB8
        # Set the following when Persistent IP is set on the camera
        nodemap["GevPersistentARPConflictDetectionEnable"].value = False

        # Set auto exposure
        self.set_exposure(-1)

    def set_exposure(self, exposure_us: float):
        """
        Set the exposure time of the camera.

        Args:
            exposure_us (float): Exposure time in microseconds.
        """
        if not self.is_connected:
            return

        nodemap = self._device.nodemap
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

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.is_connected:
            return None

        # get_buffer must be called after start_stream and before stop_stream (or
        # system.destroy_device), and buffers must be requeued
        buffer = self._device.get_buffer()

        # Convert to numpy array
        # buffer is a list of (buffer.width * buffer.height * 3) uint8s
        buffer_bytes_per_pixel = int(len(buffer.data) / (buffer.width * buffer.height))
        np_array = np.asarray(buffer.data, dtype=np.uint8)
        np_array = np_array.reshape(buffer.height, buffer.width, buffer_bytes_per_pixel)

        self._device.requeue_buffer(buffer)
        return np_array


class HeliosRayCamera(LucidCamera):
    def __init__(
        self,
        serial_number: Optional[str] = None,
        camera_index: int = 0,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(
            (640, 480),
            serial_number=serial_number,
            model_prefix="HTR",
            camera_index=camera_index,
            logger=logger,
        )

    @property
    def is_depth_camera(self) -> bool:
        return True

    def _configure_nodemap(self, device):
        # Configure stream
        nodemap = device.nodemap
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

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.is_connected:
            return None

        # get_buffer must be called after start_stream and before stop_stream (or
        # system.destroy_device), and buffers must be requeued
        buffer = self._device.get_buffer()

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
                    ("x", "f"),
                    ("y", "f"),
                    ("z", "f"),
                    ("i", np.uint16),
                ]
            )
        )

        # Apply scale and offsets to convert (x, y, z) to mm
        np_array["x"] = np_array["x"] * self._xyz_scale + self._xyz_offset[0]
        np_array["y"] = np_array["y"] * self._xyz_scale + self._xyz_offset[1]
        np_array["z"] = np_array["z"] * self._xyz_scale + self._xyz_offset[2]

        np_array = np_array.reshape(buffer.height, buffer.width)

        self._device.requeue_buffer(buffer)
        return np_array


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
        depth_to_color_extrinsic_matrix: np.ndarray,
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
        self._depth_to_color_extrinsic_matrix = depth_to_color_extrinsic_matrix

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
                    pixel_undistorted[0][0][0] * DEPTH_MIN_METERS,
                    pixel_undistorted[0][0][1] * DEPTH_MIN_METERS,
                    DEPTH_MIN_METERS,
                ]
            )

        def transform_position(position, extrinsic_matrix):
            position = np.array(position)
            homogeneous_position = np.append(position, 1)
            return np.dot(extrinsic_matrix, homogeneous_position)[:3]

        def project_position(position, camera_matrix, distortion_coeffs):
            pixels, _ = cv2.projectPoints(
                [position],
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
        )
        max_depth_pixel = project_position(
            max_depth_depth_space_position,
            self._depth_camera_intrinsic_matrix,
            self._depth_camera_distortion_coeffs,
        )

        # Make sure pixel coords are in boundary
        depth_frame_height, depth_frame_width, depth_channels = self.depth_frame.shape
        min_depth_pixel = adjust_pixel_to_bounds(
            min_depth_pixel, depth_frame_width, depth_frame_height
        )
        max_depth_pixel = adjust_pixel_to_bounds(
            max_depth_pixel, depth_frame_width, depth_frame_height
        )

        # Search along the line for the depth pixel for which its projected pixel is the closest
        # to the input pixel
        min_dist = -1
        curr_pixel = min_depth_pixel
        while is_pixel_in_line(curr_pixel, min_depth_pixel, max_depth_pixel):
            depth_mm = self.depth_frame[curr_pixel[1]][curr_pixel[0]][2]

            curr_pixel = next_pixel_in_line(
                curr_pixel, min_depth_pixel, max_depth_pixel
            )

        # TODO: WIP
        pass


class LucidRgbd(RgbdCamera):
    def __init__(
        self,
        color_camera: LucidCamera,
        color_camera_intrinsic_matrix: np.ndarray,
        color_camera_distortion_coeffs: np.ndarray,
        depth_camera: LucidCamera,
        depth_camera_intrinsic_matrix: np.ndarray,
        depth_camera_distortion_coeffs: np.ndarray,
        color_to_depth_extrinsic_matrix: np.ndarray,
        logger: Optional[logging.Logger] = None,
    ):
        self.color_camera = color_camera
        self.depth_camera = depth_camera
        self._color_camera_intrinsic_matrix = color_camera_intrinsic_matrix
        self._color_camera_distortion_coeffs = color_camera_distortion_coeffs
        self._depth_camera_intrinsic_matrix = depth_camera_intrinsic_matrix
        self._depth_camera_distortion_coeffs = depth_camera_distortion_coeffs
        self._color_to_depth_extrinsic_matrix = color_to_depth_extrinsic_matrix

        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)

    @property
    def is_connected(self) -> bool:
        return self.color_camera.is_connected and self.depth_camera.is_connected

    def initialize(self):
        self.color_camera.initialize()
        self.depth_camera.initialize()

    def set_exposure(self, exposure_us: float):
        pass

    def get_frame(self):
        color_frame = self.color_camera.get_frame()
        depth_frame = self.depth_camera.get_frame()
        if color_frame is None or depth_frame is None:
            return None

        depth_frame_xyz = np.stack(
            [depth_frame["x"], depth_frame["y"], depth_frame["z"]], axis=-1
        )
        timestamp_millis = time.time() * 1000

        return LucidFrame(
            color_frame,
            depth_frame_xyz,
            timestamp_millis,
            self._color_camera_intrinsic_matrix,
            self._color_camera_distortion_coeffs,
            self._depth_camera_intrinsic_matrix,
            self._depth_camera_distortion_coeffs,
            self._color_to_depth_extrinsic_matrix,
            self._color_to_depth_extrinsic_matrix,
        )


if __name__ == "__main__":
    import sys
    import os

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    triton_camera = TritonCamera((1920, 1080), serial_number="241300039")
    helios_camera = HeliosRayCamera(serial_number="241400544")
    lucid_rgbd = LucidRgbd(
        triton_camera,
        np.zeros(0),
        np.zeros(0),
        helios_camera,
        np.zeros(0),
        np.zeros(0),
        np.zeros(0),
        None,
    )
    lucid_rgbd.initialize()
    time.sleep(1)

    depth_frame = helios_camera.get_frame()
    if depth_frame is not None:
        print(depth_frame["i"].shape)

    """
    frame = lucid_rgbd.get_frame()
    if frame is not None:
        print(
            f"Got frame. Color frame: {frame.color_frame.shape} / Depth frame: {frame.depth_frame.shape}"
        )
    else:
        print(f"No frame")
    """
    """
    cv2.imwrite(
        os.path.expanduser("~/Pictures/triton_image.png"),
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
    )
    """
