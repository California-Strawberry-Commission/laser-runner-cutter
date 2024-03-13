from rclpy.callback_groups import ReentrantCallbackGroup
from std_srvs.srv import Trigger

from common_interfaces.msg import Vector2
from laser_control_interfaces.srv import AddPoint, GetBounds, SetColor, SetPoints


class LaserControlClient:
    def __init__(self, node, laser_node_name):
        self.node = node
        callback_group = ReentrantCallbackGroup()
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
            Trigger, f"/{laser_node_name}/clear_points", callback_group=callback_group
        )
        node.laser_set_points = node.create_client(
            SetPoints, f"/{laser_node_name}/set_points", callback_group=callback_group
        )
        node.laser_play = node.create_client(
            Trigger, f"/{laser_node_name}/play", callback_group=callback_group
        )
        node.laser_stop = node.create_client(
            Trigger, f"/{laser_node_name}/stop", callback_group=callback_group
        )

    def wait_active(self):
        while not self.node.laser_play.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("laser service not available, waiting again...")

    async def start_laser(self, point=None, color=None):
        if point is not None:
            self.set_point(point)

        if color is not None:
            self.set_color(color)

        return await self.node.laser_play.call_async(Trigger.Request())

    async def stop_laser(self):
        return await self.node.laser_stop.call_async(Trigger.Request())

    async def set_color(self, color):
        """
        Set color of laser points.

        Args:
            color: (r, g, b), where each channel is represented by a float from 0 to 1.
        """
        request = SetColor.Request(
            r=float(color[0]), g=float(color[1]), b=float(color[2]), i=0.0
        )
        return await self.node.laser_set_color.call_async(request)

    async def set_point(self, point):
        """
        Set rendered point. This will overwrite any existing set of points.

        Args:
            point: (x, y), in laser coordinates
        """
        return await self.set_points([point])

    async def set_points(self, points):
        """
        Set rendered points. This will overwrite any existing set of points.

        Args:
            points: List((x, y)), in laser coordinates
        """
        request = SetPoints.Request(
            points=[Vector2(x=point[0], y=point[1]) for point in points]
        )
        return await self.node.laser_set_points.call_async(request)

    async def clear_points(self):
        """
        Remove rendered points.
        """
        return await self.node.laser_clear_points.call_async(Trigger.Request())

    async def add_point(self, point):
        """
        Add rendered point.

        Args:
            point: (x, y), in laser coordinates
        """
        request = AddPoint.Request(point=Vector2(x=point[0], y=point[1]))
        return await self.node.laser_add_point.call_async(request)

    async def get_bounds(self, scale=1.0):
        result = await self.node.laser_get_bounds.call_async(
            GetBounds.Request(scale=scale)
        )
        return [(point.x, point.y) for point in result.points]
