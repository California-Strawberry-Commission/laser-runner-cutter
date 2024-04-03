import logging

from rclpy.callback_groups import ReentrantCallbackGroup

from camera_control_interfaces.srv import (
    GetDetectionResult,
    GetPositionsForPixels,
    SetExposure,
)
from common_interfaces.msg import Vector2
from common_interfaces.srv import GetBool


# Could make a mixin if desired
class CameraControlClient:
    def __init__(self, node, camera_node_name, logger=None):
        self.node = node
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)

        callback_group = ReentrantCallbackGroup()
        node.camera_has_frames = node.create_client(
            GetBool, f"/{camera_node_name}/has_frames", callback_group=callback_group
        )
        node.camera_set_exposure = node.create_client(
            SetExposure,
            f"/{camera_node_name}/set_exposure",
            callback_group=callback_group,
        )
        node.camera_get_lasers = node.create_client(
            GetDetectionResult,
            f"/{camera_node_name}/get_laser_detection",
            callback_group=callback_group,
        )
        node.camera_get_runners = node.create_client(
            GetDetectionResult,
            f"/{camera_node_name}/get_runner_detection",
            callback_group=callback_group,
        )
        node.camera_get_positions_for_pixels = node.create_client(
            GetPositionsForPixels,
            f"/{camera_node_name}/get_positions_for_pixels",
            callback_group=callback_group,
        )

    def wait_active(self):
        while not self.node.camera_has_frames.wait_for_service(timeout_sec=1.0):
            self.logger.info("Camera service not available, waiting again...")

    async def has_frames(self):
        result = await self.node.camera_has_frames.call_async(GetBool.Request())
        return result.data

    async def set_exposure(self, exposure_ms):
        request = SetExposure.Request(exposure_ms=exposure_ms)
        return await self.node.camera_set_exposure.call_async(request)

    async def auto_exposure(self):
        request = SetExposure.Request(exposure_ms=-1.0)
        return await self.node.camera_set_exposure.call_async(request)

    async def get_lasers(self):
        result = await self.node.camera_get_lasers.call_async(
            GetDetectionResult.Request()
        )
        return self._unpack_detection_result(result)

    async def get_runners(self):
        result = await self.node.camera_get_runners.call_async(
            GetDetectionResult.Request()
        )
        return self._unpack_detection_result(result)

    async def get_positions_for_pixels(self, pixels):
        request = GetPositionsForPixels.Request(
            pixels=[Vector2(x=float(pixel[0]), y=float(pixel[1])) for pixel in pixels]
        )
        result = await self.node.camera_get_positions_for_pixels.call_async(request)
        return [(position.x, position.y, position.z) for position in result.positions]

    def _unpack_detection_result(self, data):
        detection_result = data.result
        res = {}
        res["timestamp"] = detection_result.timestamp
        res["instances"] = [
            {
                "track_id": instance.track_id if instance.track_id > 0 else None,
                "position": [
                    instance.position.x,
                    instance.position.y,
                    instance.position.z,
                ],
                "point": [instance.point.x, instance.point.y],
            }
            for instance in detection_result.instances
        ]
        res["invalid_points"] = [
            [data.x, data.y] for data in detection_result.invalid_points
        ]
        return res
