import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from laser_control.laser_dac import EtherDreamDAC, HeliosDAC
from laser_control_interfaces.msg import Dac
from laser_control_interfaces.srv import (
    AddPoint,
    ConnectToDac,
    ListDacs,
    Play,
    SetColor,
)
from std_srvs.srv import Empty


class LaserControlNode(Node):
    def __init__(self):
        super().__init__("laser_control_node")
        self.logger = self.get_logger()
        self.list_dacs_srv = self.create_service(
            ListDacs, "laser_control/list_dacs", self.list_dacs_callback
        )
        self.connect_to_dac_srv = self.create_service(
            ConnectToDac, "laser_control/connect_to_dac", self.connect_to_dac_callback
        )
        self.set_color_srv = self.create_service(
            SetColor, "laser_control/set_color", self.set_color_callback
        )
        self.add_point_srv = self.create_service(
            AddPoint, "laser_control/add_point", self.add_point_callback
        )
        self.remove_point_srv = self.create_service(
            Empty, "laser_control/remote_point", self.remove_point_callback
        )
        self.clear_points_srv = self.create_service(
            Empty, "laser_control/clear_points", self.clear_points_callback
        )
        self.play_srv = self.create_service(
            Play, "laser_control/play", self.play_callback
        )
        self.stop_srv = self.create_service(
            Empty, "laser_control/stop", self.stop_callback
        )
        self.close_srv = self.create_service(
            Empty, "laser_control/close", self.close_callback
        )

        include_dir = os.path.join(
            get_package_share_directory("laser_control"), "include"
        )
        self.helios = HeliosDAC(os.path.join(include_dir, "libHeliosDacAPI.so"))
        self.ether_dream = EtherDreamDAC(os.path.join(include_dir, "libEtherDream.so"))
        self.connected_dac = None

        # Initialize DAC interfaces
        self.num_helios_dacs = self.helios.initialize()
        self.num_ether_dream_dacs = self.ether_dream.initialize()

        # Construct DAC list
        self.dac_list = []
        for i in range(0, self.num_helios_dacs):
            dac = Dac()
            dac.type = "helios"
            dac.name = f"Helios {i}"
            self.dac_list.append(dac)

        for i in range(0, self.num_ether_dream_dacs):
            dac = Dac()
            dac.type = "ether_dream"
            dac.name = f"Ether Dream {i}"
            self.dac_list.append(dac)

        self.logger.info(
            f"Found {self.num_ether_dream_dacs} Ether Dream and {self.num_helios_dacs} Helios DACs"
        )

    def list_dacs_callback(self, request, response):
        response.dacs = self.dac_list
        return response

    def connect_to_dac_callback(self, request, response):
        dac_idx = request.idx
        if dac_idx < 0 or dac_idx >= len(self.dac_list):
            response.success = False
            return response

        dac = self.dac_list[dac_idx]
        if dac.type == "helios":
            self.connected_dac = self.helios
            self.connected_dac.connect(dac_idx)
        elif dac.type == "ether_dream":
            self.connected_dac = self.ether_dream
            self.connected_dac.connect(dac_idx - self.num_helios_dacs)

        response.success = True
        return response

    def set_color_callback(self, request, response):
        if self.connected_dac is not None:
            self.connected_dac.set_color(request.r, request.g, request.b, request.i)
        return response

    def add_point_callback(self, request, response):
        if self.connected_dac is not None:
            self.connected_dac.add_point(request.x, request.y)
        return response

    def remove_point_callback(self, request, response):
        if self.connected_dac is not None:
            self.connected_dac.remove_point()
        return response

    def clear_points_callback(self, request, response):
        if self.connected_dac is not None:
            self.connected_dac.clear_points()
        return response

    def play_callback(self, request, response):
        if self.connected_dac is not None:
            self.connected_dac.play(
                request.fps, request.pps, request.transition_duration_ms
            )
        return response

    def stop_callback(self, request, response):
        if self.connected_dac is not None:
            self.connected_dac.stop()
        return response

    def close_callback(self, request, response):
        if self.connected_dac is not None:
            self.connected_dac.close()
        return response


def main(args=None):
    rclpy.init(args=args)
    node = LaserControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
