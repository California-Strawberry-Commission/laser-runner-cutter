import os
import numpy as np
from ament_index_python.packages import get_package_share_directory
from rqt_gui_py.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget
from python_qt_binding.QtGui import QColor, QImage, QPixmap
from std_msgs.msg import String
from camera_control_interfaces.srv import GetFrame

CAMERA_FRAME_DISPLAY_FPS = 2.0


class RqtLaserRunnerRemoval(Plugin):
    def __init__(self, context):
        super().__init__(context)

        self.context = context
        self.widget = QWidget()
        self.setObjectName("Laser Runner Removal")

        # Get path to Qt Designer UI file and load the file
        resource_dir = os.path.join(
            get_package_share_directory("rqt_laser_runner_removal"), "resource"
        )
        ui_file = os.path.join(
            resource_dir,
            "rqt_laser_runner_removal.ui",
        )
        loadUi(ui_file, self.widget)

        self.widget.setObjectName("Laser Runner Removal UI")

        # Show _widget.windowTitle on left-top of each plugin (when
        # it's set in _widget). This is useful when you open multiple
        # plugins at once. Also if you open multiple instances of your
        # plugin at once, these lines add number to make it easy to
        # tell from pane to pane.
        if self.context.serial_number() > 1:
            self.widget.setWindowTitle(
                self.widget.windowTitle() + (" (%d)" % self.context.serial_number())
            )

        # Add widget to the user interface
        self.context.add_widget(self.widget)

        self.state_subscriber = self.context.node.create_subscription(
            String, "control_node/state", self.state_callback, 5
        )
        self.get_frame_client = self.context.node.create_client(
            GetFrame, "camera_control/get_frame"
        )
        # Used to ensure only one pending response at a time
        self.get_frame_response = None
        self.get_frame_timer = self.context.node.create_timer(
            1.0 / CAMERA_FRAME_DISPLAY_FPS, self.get_frame
        )

        # Placeholder for camera frame displays
        placeholder_image = QImage(self.widget.colorFrame.size(), QImage.Format_RGB888)
        placeholder_image.fill(QColor(200, 200, 200))
        self.widget.colorFrame.setPixmap(QPixmap.fromImage(placeholder_image))
        placeholder_image = QImage(self.widget.depthFrame.size(), QImage.Format_RGB888)
        placeholder_image.fill(QColor(200, 200, 200))
        self.widget.depthFrame.setPixmap(QPixmap.fromImage(placeholder_image))

    def state_callback(self, msg):
        self.widget.stateText.setText(msg.data)

    def get_frame(self):
        if (
            not self.get_frame_client.service_is_ready()
            or self.get_frame_response is not None
        ):
            return

        request = GetFrame.Request()
        self.get_frame_response = self.get_frame_client.call_async(request)
        # Note: async calls hang when nested inside timer (https://github.com/ros2/rclpy/issues/1018)
        self.get_frame_response.add_done_callback(self.frame_callback)

    def frame_callback(self, future):
        self.get_frame_response = None
        result = future.result()
        timestamp = result.timestamp
        if timestamp <= 0:
            return

        # Process and render color frame data
        data = result.color_frame.data
        shape = result.color_frame.shape
        frame = np.array(data).reshape(shape).astype(np.uint8)
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        pixmap = QPixmap.fromImage(
            QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        )
        pixmap = pixmap.scaled(self.widget.colorFrame.size(), aspectRatioMode=1)
        self.widget.colorFrame.setPixmap(pixmap)

        # Process and render depth frame data
        data = result.depth_frame.data
        shape = result.depth_frame.shape
        frame = np.array(data).reshape(shape).astype(np.int16)
        # Normalize 16-bit image to 8-bit for display
        frame = (
            (frame.astype(np.float32) - np.min(frame))
            / (np.max(frame) - np.min(frame))
            * 255
        ).astype(np.uint8)
        height, width = frame.shape
        bytes_per_line = width
        pixmap = QPixmap.fromImage(
            QImage(frame.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        )
        pixmap = pixmap.scaled(self.widget.depthFrame.size(), aspectRatioMode=1)
        self.widget.depthFrame.setPixmap(pixmap)

    def shutdown_plugin(self):
        self.context.node.destroy_subscription(self.state_subscriber)
        self.context.node.destroy_timer(self.get_frame_timer)

    def save_settings(self, plugin_settings, instance_settings):
        """Save settings."""

    def restore_settings(self, plugin_settings, instance_settings):
        """Restore settings."""
