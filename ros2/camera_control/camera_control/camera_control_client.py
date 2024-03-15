from rclpy.callback_groups import ReentrantCallbackGroup

from camera_control_interfaces.srv import (
    GetPosData,
    GetPositionsForPixels,
    SetExposure,
)
from common_interfaces.msg import Vector2
from common_interfaces.srv import GetBool


# Could make a mixin if desired
class CameraControlClient:
    def __init__(self, node, camera_node_name):
        self.node = node
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
            GetPosData,
            f"/{camera_node_name}/get_laser_detection",
            callback_group=callback_group,
        )
        node.camera_get_runners = node.create_client(
            GetPosData,
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
            self.node.logger.info("Camera service not available, waiting again...")

    async def has_frames(self):
        result = await self.node.camera_has_frames.call_async(GetBool.Request())
        return result.data

    async def set_exposure(self, exposure_ms):
        request = SetExposure.Request(exposure_ms=exposure_ms)
        return await self.node.camera_set_exposure.call_async(request)

    async def get_laser_pos(self):
        result = await self.node.camera_get_lasers.call_async(GetPosData.Request())
        return self._unpack_pos_data(result)

    async def get_runner_pos(self):
        result = await self.node.camera_get_runners.call_async(GetPosData.Request())
        return self._unpack_pos_data(result)

    async def get_positions_for_pixels(self, pixels):
        request = GetPositionsForPixels.Request(
            pixels=[Vector2(x=float(pixel[0]), y=float(pixel[1])) for pixel in pixels]
        )
        result = await self.node.camera_get_positions_for_pixels.call_async(request)
        return [(position.x, position.y, position.z) for position in result.positions]

    def _unpack_pos_data(self, res_data):
        pos_data = res_data.pos_data
        res = {}
        res["timestamp"] = pos_data.timestamp
        res["pos_list"] = [[data.x, data.y, data.z] for data in pos_data.pos_list]
        res["point_list"] = [[data.x, data.y] for data in pos_data.point_list]
        res["invalid_point_list"] = [
            [data.x, data.y] for data in pos_data.invalid_point_list
        ]
        return res
