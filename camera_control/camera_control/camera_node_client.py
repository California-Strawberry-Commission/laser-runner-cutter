import rclpy

from std_srvs.srv import Empty
from camera_control_interfaces.srv import GetBool, GetPosData
from laser_control_interfaces.msg import Point


# Could make a mixin if desired
class CameraNodeClient:
    def __init__(self, node):
        node.camera_get_lasers = node.create_client(
            GetPosData, "camera_control/get_laser_detection"
        )
        node.camera_get_runners = node.create_client(
            GetPosData, "camera_control/get_runner_detection"
        )
        node.camera_has_frames = node.create_client(
            GetBool,
            "camera_control/has_frames",
        )
        node.runner_point_pub = node.create_publisher(Point, "runner_point", 1)
        self.node = node

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

    def pub_runner_point(self, point):
        runner_msg = Point()
        runner_msg.x = int(point[0])
        runner_msg.y = int(point[1])
        self.node.runner_point_pub.publish(runner_msg)

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
