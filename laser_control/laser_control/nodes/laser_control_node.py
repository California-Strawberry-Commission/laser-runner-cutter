import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from laser_control.laser_dac import EtherDreamDAC, HeliosDAC
from laser_control_interfaces.srv import ListConnectedDacs
from laser_control_interfaces.msg import Dac


class LaserControlNode(Node):
    def __init__(self):
        super().__init__("laser_control_node")
        self.logger = self.get_logger()
        self.srv = self.create_service(
            ListConnectedDacs, "list_connected_dacs", self.list_connected_dacs_callback
        )

        include_dir = os.path.join(
            get_package_share_directory("laser_control"), "include"
        )
        helios = HeliosDAC(os.path.join(include_dir, "libHeliosDacAPI.so"))
        ether_dream = EtherDreamDAC(os.path.join(include_dir, "libEtherDream.so"))

        # Initialize DAC interfaces
        num_helios_dacs = helios.initialize()
        num_ether_dream_dacs = ether_dream.initialize()

        # Construct DAC list
        self.dac_list = []
        for i in range(0, num_helios_dacs):
            dac = Dac()
            dac.type = "helios"
            dac.name = f"Helios {i}"
            self.dac_list.append(dac)

        for i in range(0, num_ether_dream_dacs):
            dac = Dac()
            dac.type = "ether_dream"
            dac.name = f"Ether Dream {i}"
            self.dac_list.append(dac)

        self.logger.info(
            f"Found {num_ether_dream_dacs} Ether Dream and {num_helios_dacs} Helios DACs"
        )

    def list_connected_dacs_callback(self, request, response):
        response.dacs = self.dac_list
        return response


def main(args=None):
    rclpy.init(args=args)
    node = LaserControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
