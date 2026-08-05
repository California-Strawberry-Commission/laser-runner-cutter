"""
Integration test for camera_control_node.

These tests require a LUCID Triton (color) + Helios (depth) camera pair to be connected.
"""

import unittest

import launch_ros.actions
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import pytest
import rclpy
from camera_control_interfaces.msg import CaptureMode, DeviceState
from camera_control_interfaces.srv import GetState, StartDevice
from launch import LaunchDescription
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

NODE_NAME = "camera_control_node"
SERVICE_TIMEOUT_SEC = 10.0
STREAMING_TIMEOUT_SEC = 30.0
COLOR_FRAME_WIDTH = 2048
COLOR_FRAME_HEIGHT = 1536
DEPTH_FRAME_WIDTH = 640
DEPTH_FRAME_HEIGHT = 480


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    camera_control_node = launch_ros.actions.Node(
        package="camera_control",
        executable="camera_control_node",
        name=NODE_NAME,
        output="screen",
    )
    return (
        LaunchDescription(
            [
                camera_control_node,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {"camera_control_node": camera_control_node},
    )


class TestCameraControlNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.test_node = Node("test_driver_node")
        self.start_device_client = self.test_node.create_client(
            StartDevice, f"/{NODE_NAME}/start_device"
        )
        self.close_device_client = self.test_node.create_client(
            Trigger, f"/{NODE_NAME}/close_device"
        )
        self.get_state_client = self.test_node.create_client(
            GetState, f"/{NODE_NAME}/get_state"
        )

    def tearDown(self):
        self.test_node.destroy_node()

    def _call_service(self, client, request):
        self.assertTrue(
            client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_SEC),
            f"Service {client.srv_name} was not available",
        )
        future = client.call_async(request)
        rclpy.spin_until_future_complete(
            self.test_node, future, timeout_sec=SERVICE_TIMEOUT_SEC
        )
        self.assertTrue(future.done(), f"Call to {client.srv_name} timed out")
        return future.result()

    def _wait_for_message(self, topic, msg_type, timeout_sec):
        # The node publishes images with SensorDataQoS (best-effort,
        # volatile), so the subscription QoS must match or no messages will
        # ever be delivered.
        received = {}

        def callback(msg):
            received["msg"] = msg

        subscription = self.test_node.create_subscription(
            msg_type, topic, callback, qos_profile_sensor_data
        )
        try:
            end_time = self.test_node.get_clock().now().nanoseconds + int(
                timeout_sec * 1e9
            )
            while (
                "msg" not in received
                and self.test_node.get_clock().now().nanoseconds < end_time
            ):
                rclpy.spin_once(self.test_node, timeout_sec=0.5)
        finally:
            self.test_node.destroy_subscription(subscription)
        return received.get("msg")

    def test_start_device_streams_color_and_depth(self):
        # The device should start out disconnected.
        initial_state = self._call_service(self.get_state_client, GetState.Request())
        self.assertEqual(
            initial_state.state.device_state,
            DeviceState.DISCONNECTED,
            "Expected camera to be disconnected before start_device is called",
        )

        start_result = self._call_service(
            self.start_device_client,
            StartDevice.Request(capture_mode=CaptureMode.CONTINUOUS),
        )
        self.assertTrue(start_result.success, "start_device service call failed")

        try:
            color_image = self._wait_for_message(
                f"/{NODE_NAME}/color/image_raw", Image, STREAMING_TIMEOUT_SEC
            )
            self.assertIsNotNone(
                color_image,
                "Timed out waiting for a color image. Make sure that a Triton + Helios camera "
                "pair is connected.",
            )
            self.assertEqual(color_image.width, COLOR_FRAME_WIDTH)
            self.assertEqual(color_image.height, COLOR_FRAME_HEIGHT)
            self.assertEqual(color_image.encoding, "bayer_rggb8")

            depth_xyz = self._wait_for_message(
                f"/{NODE_NAME}/depth/xyz", Image, STREAMING_TIMEOUT_SEC
            )
            self.assertIsNotNone(depth_xyz, "Timed out waiting for a depth xyz image")
            self.assertEqual(depth_xyz.width, DEPTH_FRAME_WIDTH)
            self.assertEqual(depth_xyz.height, DEPTH_FRAME_HEIGHT)
            self.assertEqual(depth_xyz.encoding, "32FC3")

            depth_intensity = self._wait_for_message(
                f"/{NODE_NAME}/depth/intensity", Image, STREAMING_TIMEOUT_SEC
            )
            self.assertIsNotNone(
                depth_intensity, "Timed out waiting for a depth intensity image"
            )
            self.assertEqual(depth_intensity.width, DEPTH_FRAME_WIDTH)
            self.assertEqual(depth_intensity.height, DEPTH_FRAME_HEIGHT)
            self.assertEqual(depth_intensity.encoding, "mono16")

            state_result = self._call_service(self.get_state_client, GetState.Request())
            self.assertEqual(state_result.state.device_state, DeviceState.STREAMING)
            self.assertEqual(state_result.state.color_width, COLOR_FRAME_WIDTH)
            self.assertEqual(state_result.state.color_height, COLOR_FRAME_HEIGHT)
            self.assertEqual(state_result.state.depth_width, DEPTH_FRAME_WIDTH)
            self.assertEqual(state_result.state.depth_height, DEPTH_FRAME_HEIGHT)
        finally:
            close_result = self._call_service(
                self.close_device_client, Trigger.Request()
            )
            self.assertTrue(close_result.success, "close_device service call failed")

        final_state = self._call_service(self.get_state_client, GetState.Request())
        self.assertEqual(final_state.state.device_state, DeviceState.DISCONNECTED)


@launch_testing.post_shutdown_test()
class TestCameraControlNodeShutdown(unittest.TestCase):
    def test_exit_code(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
