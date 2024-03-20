import rclpy
import rclpy.node
from amiga_control_interfaces.srv import SetTwist
from enum import Enum
import dataclasses

class AmigaControlClient(rclpy.node.Node):
    def __init__(self):
        super().__init__("amiga_control_client")

        self.cli_set_twist = self.create_client(SetTwist, "/amiga_control_node/set_twist")
        while not self.cli_set_twist.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        print("connected")

    def set_twist(self, linear_vel, angular_vel):
        req = SetTwist.Request()
        req.twist.x = 1.
        req.twist.y = 1.
        f = self.cli_set_twist.call_async(req)
        rclpy.spin_until_future_complete(self, f)
        return f.result()


def main():
    print("Starting amiga control client")
    rclpy.init()
    node = AmigaControlClient()
    node.set_twist(0, 0)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
