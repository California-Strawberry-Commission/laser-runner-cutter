import time

import numpy as np
import rclpy
from simple_node import Node
from std_srvs.srv import Empty
from yasmin import State, StateMachine

from laser_control_interfaces.msg import Point
from laser_control_interfaces.srv import (
    AddPoint,
    ConnectToDac,
    GetBounds,
    ListDacs,
    Play,
    SetColor,
)
from laser_runner_removal.ts_queue import TsQueue
from lrr_interfaces.msg import LaserOn, PosData


class InitializationState(State):
    """Initialization state for setting up resources needed by the system"""

    def __init__(self, node):
        State.__init__(self, outcomes=["initialization_success"])
        self.node = node

    def execute(self, blackboard):
        self.node.get_logger().info("State: InitializationState")

        # Connect to laser

        while not self.node.laser_list_dacs_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("Laser not available, waiting...")

        future = self.node.laser_list_dacs_cli.call_async(ListDacs.Request())
        rclpy.spin_until_future_complete(self.node, future)
        self.node.get_logger().info(f"Available laser DACs: {future.result().dacs}")

        future = self.node.laser_connect_to_dac_cli.call_async(
            ConnectToDac.Request(idx=0)
        )
        rclpy.spin_until_future_complete(self.node, future)

        # TODO: wait for camera node

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

        # Get image correspondences

        calibration_points = []
        future = self.node.laser_get_bounds_cli.call_async(GetBounds.Request(scale=1.0))
        rclpy.spin_until_future_complete(self.node, future)
        calibration_points += [(point.x, point.y) for point in future.result().points]
        future = self.node.laser_get_bounds_cli.call_async(
            GetBounds.Request(scale=0.75)
        )
        rclpy.spin_until_future_complete(self.node, future)
        calibration_points += [(point.x, point.y) for point in future.result().points]
        future = self.node.laser_get_bounds_cli.call_async(GetBounds.Request(scale=0.5))
        rclpy.spin_until_future_complete(self.node, future)
        calibration_points += [(point.x, point.y) for point in future.result().points]

        laser_points = []
        camera_points = []
        for calibration_point in calibration_points:
            res = self.get_camera_point_for_laser_point(calibration_point)
            if res is not None:
                self.node.get_logger().info(
                    f"Point correspondence found: laser = {calibration_point}, camera = {res}"
                )
                laser_points.append(calibration_point)
                camera_points.append(res)

        self.node.get_logger().info(f"{len(laser_points)} point correspondences found")

        transform = self.dlt(laser_points, camera_points)

        return "calibration_success"

    def get_camera_point_for_laser_point(self, laser_point):
        future = self.node.laser_clear_points_cli.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self.node, future)
        future = self.node.laser_add_point_cli.call_async(
            AddPoint.Request(point=Point(x=laser_point[0], y=laser_point[1]))
        )
        rclpy.spin_until_future_complete(self.node, future)
        self.node.laser_play_cli.call_async(
            Play.Request(fps=30, pps=30000, transition_duration_ms=0.5)
        )
        laser_send_ts = time.time()
        self.node.laser_state_pub.publish(LaserOn(laser_state=True))
        time.sleep(0.5)

        # TODO: with a service on the camera node to get the laser pos ad-hoc, we
        # won't need this logic
        last_ts, laser_data = self.node.laser_pos_queue.get_ts(time.time())
        found_point = None
        if last_ts and last_ts > laser_send_ts + 0.05:
            found_pos, found_point = laser_data
            found_pos = found_pos[0]
            found_point = found_point[0]
        self.node.laser_state_pub.publish(LaserOn(laser_state=False))

        future = self.node.laser_stop_cli.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self.node, future)

        return (found_point.x, found_point.y) if found_point else None

    def dlt(self, laser_points, camera_points):
        """Estimate the transformation matrix between camera and laser using the Direct Linear Transform algorithm"""

        # Normalize the points to help stabilize DLT
        laser_points, laser_points_normalization_matrix = self.normalize_points(
            np.array(laser_points)
        )
        camera_points, camera_points_normalization_matrix = self.normalize_points(
            np.array(camera_points)
        )

        # Construct list of points in homogeneous coordinates
        laser_points = np.hstack((laser_points, np.ones((laser_points.shape[0], 1))))
        camera_points = np.hstack((camera_points, np.ones((camera_points.shape[0], 1))))

        # Construct the linear equation matrix A
        A = []
        for i in range(len(camera_points)):
            x1, y1, z1 = camera_points[i]
            x2, y2, z2 = laser_points[i]
            A.append([0, 0, 0, -z2 * x1, -z2 * y1, -z2 * z1, y2 * x1, y2 * y1, y2 * z1])
            A.append([z2 * x1, z2 * y1, z2 * z1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2 * z1])
        A = np.array(A)

        # Perform Singular Value Decomposition (SVD) to obtain the transformation matrix P
        _, _, V = np.linalg.svd(A)
        P = V[-1].reshape(3, 3)

        # Denormalize the transformation matrix P
        P_denormalized = np.dot(
            np.linalg.inv(laser_points_normalization_matrix),
            np.dot(P, camera_points_normalization_matrix),
        )

        return P_denormalized

    def normalize_points(self, points):
        # Calculate the centroid of the points
        centroid = np.mean(points, axis=0)

        # Calculate the average distance from the centroid
        avg_distance = np.mean(np.linalg.norm(points - centroid, axis=1))

        # Scale and translate the points
        scale_factor = np.sqrt(2) / avg_distance
        translation_vector = -scale_factor * centroid

        # Create the transformation matrix for normalization
        T = np.array(
            [
                [scale_factor, 0, translation_vector[0]],
                [0, scale_factor, translation_vector[1]],
                [0, 0, 1],
            ]
        )

        # Apply the transformation to the points
        normalized_points = np.dot(T, np.vstack((points.T, np.ones(len(points)))))
        normalized_points = normalized_points[:2, :].T

        return normalized_points, T


class TestNode(Node):
    def __init__(self):
        Node.__init__(self, "test_node")

        # Set up laser_control clients

        self.laser_list_dacs_cli = self.create_client(
            ListDacs, "laser_control/list_dacs"
        )
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

        # Set up camera subscriptions

        # TODO: add service to get laser pos and runner pos on demand so camera
        # node isn't doing unnecessary work, and we don't need to maintain a
        # TSQueue in this node
        self.laser_state_pub = self.create_publisher(LaserOn, "laser_on", 1)
        self.laser_pos_queue = TsQueue(10)
        self.retrieve_laser_pos_sub = self.create_subscription(
            PosData, "laser_pos_data", self.laser_pos_callback, 5
        )

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

    def laser_pos_callback(self, msg):
        if not msg.point_list == [] and not msg.pos_list == []:
            self.laser_pos_queue.add(msg.timestamp, [msg.pos_list, msg.point_list])


def main(args=None):
    rclpy.init(args=args)
    node = TestNode()
    node.join_spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
