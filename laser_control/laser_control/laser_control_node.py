import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from laser_control.laser_dac import EtherDreamDAC, HeliosDAC
from common_interfaces.msg import Vector2
from laser_control_interfaces.msg import State
from laser_control_interfaces.srv import (
    AddPoint,
    GetBounds,
    GetState,
    SetColor,
    SetPlaybackParams,
    SetPoints,
)
from std_srvs.srv import Empty


class LaserControlNode(Node):
    def __init__(self):
        super().__init__("laser_control_node")

        # Parameters

        self.declare_parameters(
            namespace="",
            parameters=[
                ("dac_type", "helios"),  # "helios" or "ether_dream"
                ("dac_index", 0),
                ("fps", 30),
                ("pps", 30000),
                ("transition_duration_ms", 0.5),
            ],
        )

        self.dac_type = (
            self.get_parameter("dac_type").get_parameter_value().string_value
        )
        self.dac_index = (
            self.get_parameter("dac_index").get_parameter_value().integer_value
        )
        self.fps = self.get_parameter("fps").get_parameter_value().integer_value
        self.pps = self.get_parameter("pps").get_parameter_value().integer_value
        self.transition_duration_ms = (
            self.get_parameter("transition_duration_ms")
            .get_parameter_value()
            .double_value
        )

        # Services

        self.set_color_srv = self.create_service(
            SetColor, "~/set_color", self._set_color_callback
        )
        self.get_bounds_srv = self.create_service(
            GetBounds, "~/get_bounds", self._get_bounds_callback
        )
        self.add_point_srv = self.create_service(
            AddPoint, "~/add_point", self._add_point_callback
        )
        self.set_points_srv = self.create_service(
            SetPoints, "~/set_points", self._set_points_callback
        )
        self.remove_point_srv = self.create_service(
            Empty, "~/remove_point", self._remove_point_callback
        )
        self.clear_points_srv = self.create_service(
            Empty, "~/clear_points", self._clear_points_callback
        )
        self.set_playback_params_srv = self.create_service(
            SetPlaybackParams,
            "~/set_playback_params",
            self._set_playback_params_callback,
        )
        self.play_srv = self.create_service(Empty, "~/play", self._play_callback)
        self.stop_srv = self.create_service(Empty, "~/stop", self._stop_callback)
        self.state_srv = self.create_service(
            GetState, "~/get_state", self._get_state_callback
        )

        # Pub/sub

        self.state_pub = self.create_publisher(State, "~/state", 5)

        # Initialize DAC

        include_dir = os.path.join(
            get_package_share_directory("laser_control"), "include"
        )
        self.dac = None
        if self.dac_type == "helios":
            self.dac = HeliosDAC(os.path.join(include_dir, "libHeliosDacAPI.so"))
        elif self.dac_type == "ether_dream":
            self.dac = EtherDreamDAC(os.path.join(include_dir, "libEtherDream.so"))
        else:
            raise Exception(f"Unknown dac_type: {self.dac_type}")

        num_dacs = self.dac.initialize()
        self.get_logger().info(f"{num_dacs} DACs of type {self.dac_type} found")
        self.dac.connect(self.dac_index)

    def get_state(self):
        if self.dac is None:
            return State.DISCONNECTED
        elif self.dac.playing:
            return State.PLAYING
        else:
            return State.STOPPED

    def _set_color_callback(self, request, response):
        if self.dac is not None:
            self.dac.set_color(request.r, request.g, request.b, request.i)
        return response

    def _get_bounds_callback(self, request, response):
        if self.dac is not None:
            points = self.dac.get_bounds(request.scale)
            response.points = [Vector2(x=point[0], y=point[1]) for point in points]
        return response

    def _add_point_callback(self, request, response):
        if self.dac is not None:
            self.dac.add_point(request.point.x, request.point.y)
        return response

    def _set_points_callback(self, request, response):
        if self.dac is not None:
            self.dac.clear_points()
            for point in request.points:
                self.dac.add_point(point.x, point.y)
        return response

    def _remove_point_callback(self, request, response):
        if self.dac is not None:
            self.dac.remove_point()
        return response

    def _clear_points_callback(self, request, response):
        if self.dac is not None:
            self.dac.clear_points()
        return response

    def _set_playback_params_callback(self, request, response):
        self.fps = request.fps
        self.pps = request.pps
        self.transition_duration_ms = request.transition_duration_ms
        return response

    def _play_callback(self, request, response):
        if self.dac is not None:
            self.dac.play(self.fps, self.pps, self.transition_duration_ms)
            self._publish_state()
        return response

    def _stop_callback(self, request, response):
        if self.dac is not None:
            self.dac.stop()
            self._publish_state()
        return response

    def _get_state_callback(self, request, response):
        response.dac_type = self.dac_type
        response.dac_index = self.dac_index
        response.state.data = self.get_state()
        return response

    def _publish_state(self):
        msg = State()
        msg.data = self.get_state()
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaserControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
