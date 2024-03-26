import rclpy
from rclpy.node import Node
from example_interfaces.srv import SetBool


class FurrowPerceiverNode(Node):
    def __init__(self):
        super().__init__("furrow_perceiver")
        self.activated_ = False
        self.service_ = self.create_service(
            SetBool, "activate_robot", self.callback_activate_robot)

    def callback_activate_robot(self, request, response):
        self.activated_ = request.data
        response.success = True
        if self.activated_:
            response.message = "Robot has been activated"
        else:
            response.message = "Robot has been deactivated"
        return response

    def on_shutdown(self):
        pass


def main(args=None):
    rclpy.init(args=args)

    node = FurrowPerceiverNode()
    rclpy.spin(node)

    node.on_shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
