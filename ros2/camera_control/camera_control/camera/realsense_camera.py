import logging
import threading
import time
from typing import Callable, Optional, Tuple

import pyrealsense2 as rs

from .realsense_frame import RealSenseFrame
from .rgbd_camera import RgbdCamera, State
from .rgbd_frame import RgbdFrame


class RealSenseCamera(RgbdCamera):
    color_frame_size: Tuple[int, int]
    depth_frame_size: Tuple[int, int]
    fps: int
    serial_number: Optional[str]

    def __init__(
        self,
        color_frame_size: Tuple[int, int] = (1280, 720),
        depth_frame_size: Tuple[int, int] = (1280, 720),
        fps: int = 30,
        align_depth_to_color_frame: bool = True,
        serial_number: Optional[str] = None,
        camera_index: int = 0,
        state_change_callback: Optional[Callable[[State], None]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Note: when connected via USB2, RealSense cameras are limited to:
        - 1280x720 @ 6fps
        - 640x480 @ 30fps
        - 480x270 @ 60fps
        See https://www.intelrealsense.com/usb2-support-for-intel-realsense-technology/

        Args:
            color_frame_size (Tuple[int, int]): (width, height) of the color frame.
            depth_frame_size (Tuple[int, int]): (width, height) of the depth frame.
            fps (int): Number of frames per second that the camera should capture.
            align_depth_to_color_frame (bool): Whether the color and depth frames should be aligned.
            serial_number (Optional[str]): Serial number of device to connect to. If None, camera_index will be used.
            camera_index (int): Index of detected camera to connect to. Will only be used if serial_number is None.
            state_change_callback (Optional[Callable[[State], None]]): Callback that gets called when the camera device state changes.
            logger (Optional[logging.Logger]): Logger
        """
        self.color_frame_size = color_frame_size
        self.depth_frame_size = depth_frame_size
        self.fps = fps
        self._align_depth_to_color_frame = align_depth_to_color_frame
        self.serial_number = serial_number
        self._camera_index = camera_index
        self._state_change_callback = state_change_callback
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)

        self._pipeline = None
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
        if self._pipeline is not None:
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
        found_device = False

        with self._cv:
            while self._is_running:
                if found_device:
                    self._logger.info(f"Device found. Signaling acquisition thread")
                    self._cv.notify()
                    self._cv.wait()

                # Clean up existing connection if needed
                self._stop_stream()

                # Create new connection
                if self._is_running:
                    found_device = False

                    connected_devices = [
                        device.get_info(rs.camera_info.serial_number)
                        for device in rs.context().query_devices()
                    ]

                    # If we don't have a serial number of a device, attempt to find one among connected devices using camera_index
                    if self.serial_number is None:
                        if self._camera_index < 0 or self._camera_index >= len(
                            connected_devices
                        ):
                            raise Exception(
                                f"camera_index {self._camera_index} is out of bounds: {len(connected_devices)} devices found."
                            )
                        self.serial_number = connected_devices[self._camera_index]

                    # If the device is connected, set up and start streaming
                    if self.serial_number in connected_devices:
                        found_device = True
                        self._logger.info(f"Device {self.serial_number} found")
                        self._start_stream()
                    else:
                        self._logger.warn(f"Device {self.serial_number} was not found")
                        time.sleep(5)

            # Clean up existing connection
            self._stop_stream()

        self._logger.info(f"Terminating connection thread")

    def _acquisition_thread_fn(
        self, frame_callback: Optional[Callable[[RgbdFrame], None]] = None
    ):
        with self._cv:
            while self._is_running:
                try:
                    frames = self._pipeline.wait_for_frames()
                except:
                    self._logger.warn(
                        f"There was an issue with the camera. Signaling connection thread"
                    )
                    self._cv.notify()
                    self._cv.wait()
                    continue

                if not frames:
                    self._logger.warn(
                        f"No frame available. Signaling connection thread"
                    )
                    self._cv.notify()
                    self._cv.wait()
                    continue

                # Align depth frame to color frame if needed
                if self._align_depth_to_color_frame:
                    frames = self._align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not depth_frame or not color_frame:
                    continue

                # Apply post-processing filters
                depth_frame = self.temporal_filter.process(depth_frame)
                depth_frame = self.hole_filling_filter.process(depth_frame)

                # The various post processing functions return a generic frame, so we need
                # to cast back to depth_frame
                depth_frame = depth_frame.as_depth_frame()

                if frame_callback is not None:
                    frame_callback(
                        RealSenseFrame(
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
                    )

        self._logger.info(f"Terminating acquisition thread")

    def _start_stream(self):
        if self._pipeline is not None:
            return

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
        self._call_state_change_callback()

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

        # Post-processing
        self._align = (
            rs.align(rs.stream.color) if self._align_depth_to_color_frame else None
        )
        self.temporal_filter = rs.temporal_filter()
        # self.spatial_filter = rs.spatial_filter()  # Doesn't seem to help much. Disabling for now.
        self.hole_filling_filter = rs.hole_filling_filter()

        # Exposure setting persists on device, so reset it to auto-exposure
        self.exposure_us = -1.0

        self._logger.info(f"Device {self.serial_number} is now streaming")

    def _stop_stream(self):
        if self._pipeline is None:
            return

        self._pipeline.stop()
        self._pipeline = None
        self._call_state_change_callback()

    def _call_state_change_callback(self):
        if self._state_change_callback is not None:
            self._state_change_callback(self.state)

    @property
    def exposure_us(self) -> float:
        """
        Returns:
            float: Exposure time in microseconds.
        """
        if self.state != State.STREAMING:
            return 0.0

        return self._exposure_us

    @exposure_us.setter
    def exposure_us(self, exposure_us: float):
        """
        Set the exposure time of the camera. A negative value sets auto exposure.

        Args:
            exposure_us (float): Exposure time in microseconds. A negative value sets auto exposure.
        """
        if self.state != State.STREAMING:
            return

        color_sensor = self._profile.get_device().first_color_sensor()
        if exposure_us < 0:
            self._exposure_us = -1.0
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            self._logger.info(f"Auto exposure set")
            # Setting auto exposure also means auto gain
            self._gain_db = -1.0
        else:
            exposure_us_range = self.get_exposure_us_range()
            self._exposure_us = max(
                exposure_us_range[0], min(exposure_us, exposure_us_range[1])
            )
            color_sensor.set_option(rs.option.exposure, self._exposure_us)
            self._logger.info(f"Exposure set to {self._exposure_us}us")
            # Setting manual exposure also automatically turns on manual gain
            self._gain_db = color_sensor.get_option(rs.option.gain)

    def get_exposure_us_range(self) -> Tuple[float, float]:
        """
        Returns:
            Tuple[float, float]: (min, max) exposure times in microseconds.
        """
        if self.state != State.STREAMING:
            return (0.0, 0.0)

        color_sensor = self._profile.get_device().first_color_sensor()
        exposure_us_range = color_sensor.get_option_range(rs.option.exposure)
        return (exposure_us_range.min, exposure_us_range.max)

    @property
    def gain_db(self) -> float:
        """
        Returns:
            float: Gain level in dB.
        """
        if self.state != State.STREAMING:
            return 0.0

        return self._gain_db

    @gain_db.setter
    def gain_db(self, gain_db: float):
        """
        Set the gain level of the camera.

        Args:
            gain_db (float): Gain level in dB.
        """
        if self.state != State.STREAMING:
            return

        if gain_db < 0:
            # Auto gain just sets auto exposure mode
            self.exposure_us = -1.0
        else:
            gain_range = self.get_gain_db_range()
            self._gain_db = max(gain_range[0], min(gain_db, gain_range[1]))
            color_sensor = self._profile.get_device().first_color_sensor()
            color_sensor.set_option(rs.option.gain, self._gain_db)
            self._logger.info(f"Gain set to {self._gain_db} dB")
            # Setting manual gain also automatically turns on manual exposure
            self._exposure_us = color_sensor.get_option(rs.option.exposure)

    def get_gain_db_range(self) -> Tuple[float, float]:
        """
        Returns:
            Tuple[float, float]: (min, max) gain levels in dB.
        """
        if self.state != State.STREAMING:
            return (0.0, 0.0)

        color_sensor = self._profile.get_device().first_color_sensor()
        gain_db_range = color_sensor.get_option_range(rs.option.gain)
        return (gain_db_range.min, gain_db_range.max)
