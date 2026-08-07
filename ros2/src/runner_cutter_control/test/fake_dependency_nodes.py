"""
Minimal fake ROS 2 nodes standing in for laser_control_node, camera_control_node, and
detection_node in runner_cutter_control_node's integration tests.
"""

from camera_control_interfaces.msg import State as CameraState
from camera_control_interfaces.srv import AcquireSingleFrame
from camera_control_interfaces.srv import GetState as CameraGetState
from camera_control_interfaces.srv import StartDevice, StartIntervalCapture
from detection_interfaces.msg import DetectionResult
from detection_interfaces.msg import State as DetectionState
from detection_interfaces.srv import GetDetectionResult
from detection_interfaces.srv import GetPositions
from detection_interfaces.srv import GetState as DetectionGetState
from detection_interfaces.srv import StartDetection, StopDetection
from laser_control_interfaces.msg import State as LaserState
from laser_control_interfaces.srv import GetState as LaserGetState
from laser_control_interfaces.srv import RemovePath
from rclpy.node import Node
from std_srvs.srv import Trigger


def _add_trigger_service(node, service_name):
    def callback(request, response):
        response.success = True
        response.message = ""
        return response

    node.create_service(Trigger, service_name, callback)


class FakeLaserControlNode(Node):
    def __init__(self, name="laser0"):
        super().__init__(name, allow_undeclared_parameters=True)
        for service_name in (
            "~/start_device",
            "~/close_device",
            "~/clear_paths",
            "~/play",
            "~/stop",
        ):
            _add_trigger_service(self, service_name)
        self.create_service(RemovePath, "~/remove_path", self._on_remove_path)
        self.create_service(LaserGetState, "~/get_state", self._on_get_state)

    def _on_remove_path(self, request, response):
        response.success = True
        return response

    def _on_get_state(self, request, response):
        response.state = LaserState()
        return response


class FakeCameraControlNode(Node):
    def __init__(self, name="camera0"):
        super().__init__(name, allow_undeclared_parameters=True)
        self.create_service(StartDevice, "~/start_device", self._on_start_device)
        _add_trigger_service(self, "~/close_device")
        self.create_service(
            AcquireSingleFrame,
            "~/acquire_single_frame",
            self._on_acquire_single_frame,
        )
        _add_trigger_service(self, "~/save_image")
        self.create_service(
            StartIntervalCapture,
            "~/start_interval_capture",
            self._on_start_interval_capture,
        )
        _add_trigger_service(self, "~/stop_interval_capture")
        self.create_service(CameraGetState, "~/get_state", self._on_get_state)

    def _on_start_device(self, request, response):
        response.success = True
        return response

    def _on_acquire_single_frame(self, request, response):
        return response

    def _on_start_interval_capture(self, request, response):
        response.success = True
        return response

    def _on_get_state(self, request, response):
        response.state = CameraState()
        return response


class FakeDetectionNode(Node):
    def __init__(self, name="detection0"):
        super().__init__(name, allow_undeclared_parameters=True)
        self.create_service(
            GetDetectionResult, "~/get_detection", self._on_get_detection
        )
        self.create_service(
            StartDetection, "~/start_detection", self._on_start_detection
        )
        self.create_service(StopDetection, "~/stop_detection", self._on_stop_detection)
        for service_name in (
            "~/stop_all_detections",
            "~/start_recording_video",
            "~/stop_recording_video",
        ):
            _add_trigger_service(self, service_name)
        self.create_service(DetectionGetState, "~/get_state", self._on_get_state)
        self.create_service(GetPositions, "~/get_positions", self._on_get_positions)

    def _on_get_detection(self, request, response):
        response.result = DetectionResult()
        response.result.detection_type = request.detection_type
        return response

    def _on_start_detection(self, request, response):
        response.success = True
        return response

    def _on_stop_detection(self, request, response):
        response.success = True
        return response

    def _on_get_state(self, request, response):
        response.state = DetectionState()
        return response

    def _on_get_positions(self, request, response):
        # No positions available by default -- runner_cutter_control_node treats an empty result
        # as "no positions found" rather than an error.
        response.positions = []
        return response
