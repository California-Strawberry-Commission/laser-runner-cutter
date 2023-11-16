import os
from ament_index_python.packages import get_package_share_directory
from rqt_gui_py.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget
from std_msgs.msg import String


class RqtLaserRunnerRemoval(Plugin):
    def __init__(self, context):
        super().__init__(context)

        self._context = context
        self._widget = QWidget()
        self.setObjectName("Laser Runner Removal")

        # Get path to Qt Designer UI file and load the file
        resource_dir = os.path.join(
            get_package_share_directory("rqt_laser_runner_removal"), "resource"
        )
        ui_file = os.path.join(
            resource_dir,
            "rqt_laser_runner_removal.ui",
        )
        loadUi(ui_file, self._widget)

        self._widget.setObjectName("Laser Runner Removal UI")

        # Show _widget.windowTitle on left-top of each plugin (when
        # it's set in _widget). This is useful when you open multiple
        # plugins at once. Also if you open multiple instances of your
        # plugin at once, these lines add number to make it easy to
        # tell from pane to pane.
        if context.serial_number() > 1:
            self._widget.setWindowTitle(
                self._widget.windowTitle() + (" (%d)" % context.serial_number())
            )

        # Add widget to the user interface
        context.add_widget(self._widget)

        self.state_subscriber = self._context.node.create_subscription(
            String, "control_node/state", self.state_callback, 1
        )

    def state_callback(self, msg):
        self._widget.stateText.setText(msg.data)

    def shutdown_plugin(self):
        """Shutdown plugin."""

    def save_settings(self, plugin_settings, instance_settings):
        """Save settings."""

    def restore_settings(self, plugin_settings, instance_settings):
        """Restore settings."""
