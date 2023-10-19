import rclpy
from simple_node import Node
from yasmin import State, StateMachine
from std_srvs.srv import Empty
from laser_control_interfaces.msg import Point
from laser_control_interfaces.srv import (
    AddPoint,
    ConnectToDac,
    GetBounds,
    Play,
    SetColor,
)
import time


class InitializationState(State):
    """Initialization state for setting up resources needed by the system"""

    def __init__(self, node):
        State.__init__(self, outcomes=["initialization_success"])
        self.node = node

    def execute(self, blackboard):
        self.node.get_logger().info("State: InitializationState")

        # Connect to laser
        while not self.node.laser_connect_to_dac_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("Laser not available, waiting...")

        future = self.node.laser_connect_to_dac_cli.call_async(
            ConnectToDac.Request(idx=0)
        )
        rclpy.spin_until_future_complete(self.node, future)

        if not future.result().success:
            "initialization_failure"

        return "initialization_success"


class CalibrationState(State):
    """Calibrate the laser to camera, creating a 3D pos to laser frame transform."""

    def __init__(self, node):
        State.__init__(self, outcomes=["calibration_failure", "calibration_success"])
        self.node = node

    def execute(self, blackboard):
        self.node.get_logger().info("State: CalibrationState")

        self.node.laser_set_color_cli.call_async(
            SetColor.Request(r=1.0, g=0.0, b=0.0, i=1.0)
        )

        future = self.node.laser_get_bounds_cli.call_async(GetBounds.Request(scale=1.0))
        rclpy.spin_until_future_complete(self.node, future)
        bounds = future.result().points
        self.node.get_logger().info(f"BOUNDS: ${bounds}")

        self.node.laser_add_point_cli.call_async(
            AddPoint.Request(point=Point(x=0, y=0))
        )
        self.node.laser_play_cli.call_async(
            Play.Request(fps=30, pps=30000, transition_duration_ms=0.5)
        )
        time.sleep(5)
        self.node.laser_stop_cli.call_async(Empty.Request())

        return "calibration_success"


class TestNode(Node):
    def __init__(self):
        Node.__init__(self, "test_node")

        # Set up laser_control clients

        self.laser_connect_to_dac_cli = self.create_client(
            ConnectToDac, "laser_control/connect_to_dac"
        )
        self.laser_set_color_cli = self.create_client(
            SetColor, "laser_control/set_color"
        )
        self.laser_get_bounds_cli = self.create_client(
            GetBounds, "laser_control/get_bounds"
        )
        self.laser_add_point_cli = self.create_client(
            AddPoint, "laser_control/add_point"
        )
        self.laser_clear_points_cli = self.create_client(
            Empty, "laser_control/clear_points"
        )
        self.laser_play_cli = self.create_client(Play, "laser_control/play")
        self.laser_stop_cli = self.create_client(Empty, "laser_control/stop")

        # Set up state machine

        state_machine = StateMachine(outcomes=["finished"])
        state_machine.add_state(
            "INITIALIZATION",
            InitializationState(self),
            transitions={
                "initialization_failure": "finished",
                "initialization_success": "CALIBRATION",
            },
        )
        state_machine.add_state(
            "CALIBRATION",
            CalibrationState(self),
            transitions={
                "calibration_failure": "finished",
                "calibration_success": "finished",
            },
        )

        outcome = state_machine()
        self.get_logger().info(outcome)


def main(args=None):
    rclpy.init(args=args)
    node = TestNode()
    node.join_spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
