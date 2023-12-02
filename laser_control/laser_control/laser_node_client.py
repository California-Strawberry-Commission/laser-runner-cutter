import rclpy
from std_srvs.srv import Empty

from laser_control_interfaces.msg import Point
from laser_control_interfaces.srv import AddPoint, GetBounds, SetColor, SetPoints


# Could make a mixin if desired
class LaserNodeClient:
    def __init__(self, node, laser_node_name):
        node.laser_get_bounds = node.create_client(
            GetBounds, f"/{laser_node_name}/get_bounds"
        )
        node.laser_set_color = node.create_client(
            SetColor, f"/{laser_node_name}/set_color"
        )
        node.laser_add_point = node.create_client(
            AddPoint, f"/{laser_node_name}/add_point"
        )
        node.laser_clear_points = node.create_client(
            Empty, f"/{laser_node_name}/clear_points"
        )
        node.laser_set_points = node.create_client(
            SetPoints, f"/{laser_node_name}/set_points"
        )
        node.laser_play = node.create_client(Empty, f"/{laser_node_name}/play")
        node.laser_stop = node.create_client(Empty, f"/{laser_node_name}/stop")
        self.node = node

    def wait_active(self):
        while not self.node.laser_get_bounds.wait_for_service(timeout_sec=1.0):
            self.node.logger.info("laser service not available, waiting again...")

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
        request = SetPoints.Request()
        request.points = [Point(x=int(point[0]), y=int(point[1]))]
        response = self.node.laser_set_points.call_async(request)
        rclpy.spin_until_future_complete(self.node, response)

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

    def get_bounds(self, scale=1.0):
        request = GetBounds.Request()
        request.scale = scale
        response = self.node.laser_get_bounds.call_async(request)
        rclpy.spin_until_future_complete(self.node, response)
        return [(point.x, point.y) for point in response.result().points]
