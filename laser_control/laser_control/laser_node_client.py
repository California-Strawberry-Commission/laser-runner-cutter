import rclpy

from laser_control_interfaces.srv import (
    GetBounds,
    SetColor,
    AddPoint,
    SetPoints,
    ConnectToDac,
)
from laser_control_interfaces.msg import Point
from std_srvs.srv import Empty


# Could make a mixin if desired
class LaserNodeClient:
    def __init__(self, node):
        node.laser_scaled_frame_corners = node.create_client(
            GetBounds, "laser_control/get_bounds"
        )
        node.laser_set_color = node.create_client(SetColor, "laser_control/set_color")
        node.laser_add_point = node.create_client(AddPoint, "laser_control/add_point")
        node.laser_clear_points = node.create_client(
            Empty, "laser_control/clear_points"
        )
        node.laser_set_points = node.create_client(
            SetPoints, "laser_control/set_points"
        )
        node.laser_connect = node.create_client(
            ConnectToDac, "laser_control/connect_to_dac"
        )
        node.laser_play = node.create_client(Empty, "laser_control/play")
        node.laser_stop = node.create_client(Empty, "laser_control/stop")
        self.node = node

    def wait_active(self):
        while not self.node.laser_scaled_frame_corners.wait_for_service(
            timeout_sec=1.0
        ):
            self.node.logger.info("laser service not available, waiting again...")

    def connect(self):
        request = ConnectToDac.Request()
        request.idx = 0
        response = self.node.laser_connect.call_async(request)
        rclpy.spin_until_future_complete(self.node, response)

    # TODO Add block option to all calls, if true, wait on spin, if false don't. Could be a decorator?
    def start_laser(self, point=None, color=None):
        if point is not None:
            self.set_point(point)

        if color is not None:
            self.set_color(color)

        request = Empty.Request()
        response = self.node.laser_play.call_async(request)
        rclpy.spin_until_future_complete(self.node, response)

    def stop_laser(self):
        request = Empty.Request()
        response = self.node.laser_stop.call_async(request)
        rclpy.spin_until_future_complete(self.node, response)

    def set_color(self, color):
        request = SetColor.Request()
        request.r = float(color[0])
        request.g = float(color[1])
        request.b = float(color[2])
        request.i = 0.0
        response = self.node.laser_set_color.call_async(request)
        rclpy.spin_until_future_complete(self.node, response)

    def set_point(self, point):
        self.clear_points()
        self.add_point(point)

    def clear_points(self):
        request = Empty.Request()
        response = self.node.laser_clear_points.call_async(request)
        rclpy.spin_until_future_complete(self.node, response)

    def add_point(self, point):
        request = AddPoint.Request()
        request.point = Point(x=int(point[0]), y=int(point[1]))
        response = self.node.laser_add_point.call_async(request)
        rclpy.spin_until_future_complete(self.node, response)
        self.node.logger.debug(f"Added laser point: {point}")

    def get_scaled_frame_corners(self, scale_list):
        lsr_pts = []
        for calibration_scale in scale_list:
            request = GetBounds.Request()
            request.scale = calibration_scale
            response = self.node.laser_scaled_frame_corners.call_async(request)
            rclpy.spin_until_future_complete(self.node, response)
            lsr_pts += [[point.x, point.y] for point in response.result().points]
        return lsr_pts
