import functools
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from camera_control_interfaces.srv import GetBool, GetPosData, SetExposure


def add_sync_option(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        sync = kwargs.pop("sync", False)
        response = func(self, *args, **kwargs)
        if sync:
            rclpy.spin_until_future_complete(self.node, response)
            return response.result()
        else:
            return response

    return wrapper


# Could make a mixin if desired
class CameraControlClient:
    def __init__(self, node, camera_node_name):
        self.node = node
        callback_group = MutuallyExclusiveCallbackGroup()
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

    def wait_active(self):
        while not self.node.laser_scaled_frame_corners.wait_for_service(
            timeout_sec=1.0
        ):
            self.node.logger.info("laser service not available, waiting again...")

    def has_frames(self):
        request = GetBool.Request()
        response = self.node.camera_has_frames.call_async(request)
        rclpy.spin_until_future_complete(self.node, response)
        return response.result().data

    @add_sync_option
    def set_exposure(self, exposure_ms):
        request = SetExposure.Request()
        request.exposure_ms = exposure_ms
        return self.node.camera_set_exposure.call_async(request)

    def get_laser_pos(self):
        request = GetPosData.Request()
        response = self.node.camera_get_lasers.call_async(request)
        rclpy.spin_until_future_complete(self.node, response)
        res_data = response.result()
        return self._unpack_pos_data(res_data)

    def get_runner_pos(self):
        request = GetPosData.Request()
        response = self.node.camera_get_runners.call_async(request)
        rclpy.spin_until_future_complete(self.node, response)
        res_data = response.result()
        return self._unpack_pos_data(res_data)

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
