import os
import numpy as np
import rclpy
import time
from ament_index_python.packages import get_package_share_directory
from rqt_gui_py.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget
from python_qt_binding.QtGui import QColor, QImage, QPixmap
from std_msgs.msg import Bool, String
from camera_control_interfaces.srv import GetFrame
from ros2node.api import get_node_names
from cv_bridge import CvBridge

CAMERA_FRAME_DISPLAY_FPS = 10.0
GET_FRAME_TIMEOUT_SECS = 0.5


class RqtLaserRunnerRemoval(Plugin):
    def __init__(self, context):
        super().__init__(context)

        self.context = context
        self.widget = QWidget()
        self.setObjectName("Laser Runner Removal")

        # For converting image msg to numpy array
        self.cv_bridge = CvBridge()

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
        self.show_placeholder_frames()

        self.state_subscriber = self.context.node.create_subscription(
            String, "control_node/state", self.state_callback, 5
        )
        self.laser_playing_subscriber = self.context.node.create_subscription(
            Bool, "laser_control/playing", self.laser_playing_callback, 5
        )
        self.get_frame_client = self.context.node.create_client(
            GetFrame, "camera_control/get_frame"
        )
        # Used to ensure only one pending response at a time
        self.pending_get_frame_response = None
        self.last_get_frame_request_time = 0.0
        self.get_frame_timer = self.context.node.create_timer(
            1.0 / CAMERA_FRAME_DISPLAY_FPS, self.get_frame
        )
        self.control_node_availability_timer = self.context.node.create_timer(
            1.0, self.check_control_node_availability
        )

    def state_callback(self, msg):
        self.widget.stateText.setText(msg.data)

    def laser_playing_callback(self, msg):
        self.widget.laserText.setText("On" if msg.data else "Off")

    def check_control_node_availability(self):
        available_nodes = [
            node_name.name
            for node_name in get_node_names(
                node=self.context.node, include_hidden_nodes=False
            )
        ]
        is_control_node_available = rclpy.ok() and "control_node" in available_nodes
        if not is_control_node_available:
            self.widget.stateText.setText("Not running")

    def get_frame(self):
        if not self.get_frame_client.service_is_ready():
            self.show_placeholder_frames()
            return

        if self.pending_get_frame_response is not None:
            if (time.time() - self.get_frame_request_time) > GET_FRAME_TIMEOUT_SECS:
                self.pending_get_frame_response.cancel()
                self.pending_get_frame_response = None
            return

        request = GetFrame.Request()
        self.get_frame_request_time = time.time()
        self.pending_get_frame_response = self.get_frame_client.call_async(request)
        # Note: async calls hang when nested inside timer (https://github.com/ros2/rclpy/issues/1018)
        self.pending_get_frame_response.add_done_callback(self.frame_callback)

    def frame_callback(self, future):
        self.pending_get_frame_response = None
        result = future.result()
        if (
            result is None
            or result.color_frame.header.stamp.sec <= 0
            or result.depth_frame.header.stamp.sec <= 0
        ):
            return

        # Process and render color frame data
        color_data = self.cv_bridge.imgmsg_to_cv2(result.color_frame)
        height, width, channels = color_data.shape
        bytes_per_line = channels * width
        pixmap = QPixmap.fromImage(
            QImage(color_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
        )
        pixmap = pixmap.scaled(self.widget.colorFrame.size(), aspectRatioMode=1)
        self.widget.colorFrame.setPixmap(pixmap)

        # Process and render depth frame data
        depth_data = self.cv_bridge.imgmsg_to_cv2(result.depth_frame)
        # Normalize 16-bit image to 8-bit for display
        depth_data = (
            (depth_data.astype(np.float32) - np.min(depth_data))
            / (np.max(depth_data) - np.min(depth_data))
            * 255
        ).astype(np.uint8)
        height, width = depth_data.shape
        bytes_per_line = width
        pixmap = QPixmap.fromImage(
            QImage(
                depth_data.data, width, height, bytes_per_line, QImage.Format_Grayscale8
            )
        )
        pixmap = pixmap.scaled(self.widget.depthFrame.size(), aspectRatioMode=1)
        self.widget.depthFrame.setPixmap(pixmap)

    def show_placeholder_frames(self):
        placeholder_image = QImage(self.widget.colorFrame.size(), QImage.Format_RGB888)
        placeholder_image.fill(QColor(200, 200, 200))
        self.widget.colorFrame.setPixmap(QPixmap.fromImage(placeholder_image))
        placeholder_image = QImage(self.widget.depthFrame.size(), QImage.Format_RGB888)
        placeholder_image.fill(QColor(200, 200, 200))
        self.widget.depthFrame.setPixmap(QPixmap.fromImage(placeholder_image))

    def shutdown_plugin(self):
        self.context.node.destroy_subscription(self.state_subscriber)
        self.context.node.destroy_timer(self.get_frame_timer)

    def save_settings(self, plugin_settings, instance_settings):
        """Save settings."""

    def restore_settings(self, plugin_settings, instance_settings):
        """Restore settings."""
