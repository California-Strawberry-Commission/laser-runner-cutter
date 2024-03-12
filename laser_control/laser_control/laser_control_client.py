import functools
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_srvs.srv import Empty

from laser_control_interfaces.msg import Point
from laser_control_interfaces.srv import AddPoint, GetBounds, SetColor, SetPoints


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
class LaserControlClient:
    def __init__(self, node, laser_node_name):
        self.node = node
        callback_group = MutuallyExclusiveCallbackGroup()
        node.laser_get_bounds = node.create_client(
            GetBounds, f"/{laser_node_name}/get_bounds", callback_group=callback_group
        )
        node.laser_set_color = node.create_client(
            SetColor, f"/{laser_node_name}/set_color", callback_group=callback_group
        )
        node.laser_add_point = node.create_client(
            AddPoint, f"/{laser_node_name}/add_point", callback_group=callback_group
        )
        node.laser_clear_points = node.create_client(
            Empty, f"/{laser_node_name}/clear_points", callback_group=callback_group
        )
        node.laser_set_points = node.create_client(
            SetPoints, f"/{laser_node_name}/set_points", callback_group=callback_group
        )
        node.laser_play = node.create_client(
            Empty, f"/{laser_node_name}/play", callback_group=callback_group
        )
        node.laser_stop = node.create_client(
            Empty, f"/{laser_node_name}/stop", callback_group=callback_group
        )

    def wait_active(self):
        while not self.node.laser_get_bounds.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("laser service not available, waiting again...")

    @add_sync_option
    def start_laser(self, point=None, color=None):
        if point is not None:
            self.set_point(point)

        if color is not None:
            self.set_color(color)

        request = Empty.Request()
        return self.node.laser_play.call_async(request)

    @add_sync_option
    def stop_laser(self):
        request = Empty.Request()
        return self.node.laser_stop.call_async(request)

    @add_sync_option
    def set_color(self, color):
        """
        Set color of laser points.

        Args:
            color: (r, g, b), where each channel is represented by a float from 0 to 1.
        """
        request = SetColor.Request()
        request.r = float(color[0])
        request.g = float(color[1])
        request.b = float(color[2])
        request.i = 0.0
        return self.node.laser_set_color.call_async(request)

    @add_sync_option
    def set_point(self, point):
        """
        Set rendered point. This will overwrite any existing set of points.

        Args:
            point: (x, y), in laser coordinates
        """
        return self.set_points([point])

    @add_sync_option
    def set_points(self, points):
        """
        Set rendered points. This will overwrite any existing set of points.

        Args:
            points: List((x, y)), in laser coordinates
        """
        request = SetPoints.Request()
        request.points = [Point(x=int(point[0]), y=int(point[1])) for point in points]
        return self.node.laser_set_points.call_async(request)

    @add_sync_option
    def clear_points(self):
        """
        Remove rendered points.
        """
        request = Empty.Request()
        return self.node.laser_clear_points.call_async(request)

    @add_sync_option
    def add_point(self, point):
        """
        Add rendered point.

        Args:
            point: (x, y), in laser coordinates
        """
        request = AddPoint.Request()
        request.point = Point(x=int(point[0]), y=int(point[1]))
        return self.node.laser_add_point.call_async(request)

    def get_bounds(self, scale=1.0):
        request = GetBounds.Request()
        request.scale = scale
        response = self.node.laser_get_bounds.call_async(request)
        rclpy.spin_until_future_complete(self.node, response)
        return [(point.x, point.y) for point in response.result().points]
