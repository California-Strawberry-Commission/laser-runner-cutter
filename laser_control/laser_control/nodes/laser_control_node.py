import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from laser_control.laser_dac import EtherDreamDAC, HeliosDAC


class LaserControlNode(Node):
    def __init__(self):
        super().__init__("laser_control_node")
        self.logger = self.get_logger()

        include_dir = os.path.join(
            get_package_share_directory("laser_control"), "include"
        )
        ether_dream = EtherDreamDAC(os.path.join(include_dir, "libEtherDream.so"))
        helios = HeliosDAC(os.path.join(include_dir, "libHeliosDacAPI.so"))

        num_ether_dream_dacs = ether_dream.initialize()
        num_helios_dacs = helios.initialize()

        self.logger.info(
            f"Found {num_ether_dream_dacs} Ether Dream and {num_helios_dacs} Helios DACs"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LaserControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
