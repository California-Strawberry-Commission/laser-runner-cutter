"""
Integration test for runner_cutter_control_node.
"""

import sys
import unittest
from pathlib import Path

# launch_testing's launch_test runner does not add the test file's own directory to sys.path, so
# manually it so that we can import fake_dependency_nodes
sys.path.insert(0, str(Path(__file__).resolve().parent))

import launch_ros.actions
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import pytest
import rclpy
from fake_dependency_nodes import (
    FakeCameraControlNode,
    FakeDetectionNode,
    FakeLaserControlNode,
)
from launch import LaunchDescription
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from runner_cutter_control_interfaces.srv import GetState
from std_srvs.srv import Trigger

NODE_NAME = "runner_cutter_control_node"
SERVICE_TIMEOUT_SEC = 10.0
TASK_STOP_TIMEOUT_SEC = 15.0


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    runner_cutter_control_node = launch_ros.actions.Node(
        package="runner_cutter_control",
        executable="runner_cutter_control_node",
        name=NODE_NAME,
        output="screen",
    )
    return (
        LaunchDescription(
            [
                runner_cutter_control_node,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {"runner_cutter_control_node": runner_cutter_control_node},
    )


class TestRunnerCutterControlNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.test_node = Node("test_driver_node")
        self.fake_laser_node = FakeLaserControlNode()
        self.fake_camera_node = FakeCameraControlNode()
        self.fake_detection_node = FakeDetectionNode()
        self.nodes = [
            self.test_node,
            self.fake_laser_node,
            self.fake_camera_node,
            self.fake_detection_node,
        ]
        self.executor = MultiThreadedExecutor()
        for node in self.nodes:
            self.executor.add_node(node)

        self.get_state_client = self.test_node.create_client(
            GetState, f"/{NODE_NAME}/get_state"
        )
        self.stop_client = self.test_node.create_client(Trigger, f"/{NODE_NAME}/stop")
        self.start_runner_cutter_client = self.test_node.create_client(
            Trigger, f"/{NODE_NAME}/start_runner_cutter"
        )

        # Ensure a clean slate
        self._call_service(
            self.stop_client, Trigger.Request(), timeout_sec=TASK_STOP_TIMEOUT_SEC
        )

    def tearDown(self):
        self._call_service(
            self.stop_client, Trigger.Request(), timeout_sec=TASK_STOP_TIMEOUT_SEC
        )
        for node in self.nodes:
            self.executor.remove_node(node)
            node.destroy_node()

    def _call_service(self, client, request, timeout_sec=SERVICE_TIMEOUT_SEC):
        self.assertTrue(
            client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_SEC),
            f"Service {client.srv_name} was not available",
        )
        future = client.call_async(request)
        # Spin the shared executor directly instead of the module-level
        # rclpy.spin_until_future_complete helper, since that helper removes the node from the
        # executor once the future completes, which would undo the multi-node setup in setUp().
        self.executor.spin_until_future_complete(future, timeout_sec=timeout_sec)
        self.assertTrue(future.done(), f"Call to {client.srv_name} timed out")
        return future.result()

    def test_initial_state_is_idle_and_uncalibrated(self):
        result = self._call_service(self.get_state_client, GetState.Request())
        self.assertFalse(result.state.calibrated)
        self.assertEqual(result.state.state, "idle")
        bounds = result.state.normalized_laser_bounds
        self.assertEqual(bounds.w, 0.0)
        self.assertEqual(bounds.x, 0.0)
        self.assertEqual(bounds.y, 0.0)
        self.assertEqual(bounds.z, 0.0)

    def test_stop_with_no_task_running_fails(self):
        result = self._call_service(self.stop_client, Trigger.Request())
        self.assertFalse(result.success)

    def test_start_runner_cutter_updates_state_and_stop_returns_to_idle(self):
        start_result = self._call_service(
            self.start_runner_cutter_client, Trigger.Request()
        )
        self.assertTrue(start_result.success)

        # startTask() sets the task name and running flag synchronously, before spawning the
        # background task thread, so the state should reflect the running task immediately.
        state_result = self._call_service(self.get_state_client, GetState.Request())
        self.assertEqual(state_result.state.state, "runner_cutter")

        # A second start attempt should be rejected while a task is already running
        second_start_result = self._call_service(
            self.start_runner_cutter_client, Trigger.Request()
        )
        self.assertFalse(second_start_result.success)

        stop_result = self._call_service(
            self.stop_client, Trigger.Request(), timeout_sec=TASK_STOP_TIMEOUT_SEC
        )
        self.assertTrue(stop_result.success)

        state_result = self._call_service(self.get_state_client, GetState.Request())
        self.assertEqual(state_result.state.state, "idle")


@launch_testing.post_shutdown_test()
class TestRunnerCutterControlNodeShutdown(unittest.TestCase):
    def test_exit_code(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
