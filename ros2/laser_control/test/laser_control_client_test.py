import rclpy
from rclpy.node import Node
from laser_control.laser_control_client import LaserControlClient
import subprocess


class TestNode(Node):
    def __init__(self):
        super().__init__("test_node")

        self.laser_client = LaserControlClient(self, "laser_control_node")
        self.laser_client.wait_active()
        print("Laser node is active")
        self.laser_client.set_color((0.0, 1.0, 0.0))
        self.laser_client.set_point((0.0, 0.0))
        self.laser_on = False
        timer = self.create_timer(0.1, self._keyboard_input)

    def _keyboard_input(self):
        if self.laser_on:
            input("Press any key to turn laser off...")
            self.laser_client.stop_laser()
            self.laser_on = False
        else:
            input("Press any key to turn laser on...")
            self.laser_client.start_laser()
            self.laser_on = True


def main(args=None):
    laser_control_process = subprocess.Popen(
        ["ros2", "run", "laser_control", "laser_control_node"]
    )
    rclpy.init(args=args)
    node = TestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    laser_control_process.terminate()


if __name__ == "__main__":
    main()
