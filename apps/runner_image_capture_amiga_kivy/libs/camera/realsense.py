import pyrealsense2 as rs


class RealSense:
    def __init__(self, frame_size=(1920, 1080), fps=30, camera_index=0):
        # Note: when connected via USB2, RealSense cameras are limited to:
        # - 1280x720 @ 6fps
        # - 640x480 @ 30fps
        # - 480x270 @ 60fps
        # See https://www.intelrealsense.com/usb2-support-for-intel-realsense-technology/

        self.frame_size = frame_size
        self.camera_index = camera_index
        self.fps = fps
        self.pipeline = None

    @classmethod
    def get_devices(cls):
        serial_numbers = [
            device.get_info(rs.camera_info.serial_number)
            for device in rs.context().query_devices()
        ]
        return serial_numbers

    def initialize(self):
        # Setup code based on https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
        self.config = rs.config()

        # Connect to specific camera
        context = rs.context()
        devices = context.query_devices()
        if self.camera_index < 0 or self.camera_index >= len(devices):
            raise Exception("camera_index is out of bounds")

        serial_number = devices[self.camera_index].get_info(
            rs.camera_info.serial_number
        )
        self.config.enable_device(serial_number)

        # Configure stream
        self.config.enable_stream(
            rs.stream.color,
            self.frame_size[0],
            self.frame_size[1],
            rs.format.rgb8,
            self.fps,
        )

        # Start pipeline
        self.pipeline = rs.pipeline()
        try:
            self.profile = self.pipeline.start(self.config)
        except Exception as e:
            print(f"Error starting RealSense pipeline: {e}")
            # When connected via USB2, limit to 1280x720 @ 6fps
            self.frame_size = (1280, 720)
            self.fps = 6
            self.config.enable_stream(
                rs.stream.color,
                self.frame_size[0],
                self.frame_size[1],
                rs.format.rgb8,
                self.fps,
            )
            self.profile = self.pipeline.start(self.config)

    def stop(self):
        if self.pipeline is not None:
            self.pipeline.stop()

    def set_exposure(self, exposure_ms):
        color_sensor = self.profile.get_device().first_color_sensor()
        if exposure_ms < 0:
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        else:
            # D435 has a minimum exposure time of 1us
            exposure_us = max(1, round(exposure_ms * 1000))
            color_sensor.set_option(rs.option.exposure, exposure_us)

    def get_frame(self):
        if self.pipeline is None:
            return None

        frames = self.pipeline.wait_for_frames()
        if not frames:
            return None

        color_frame = frames.get_color_frame()
        if not color_frame:
            return None

        return color_frame
