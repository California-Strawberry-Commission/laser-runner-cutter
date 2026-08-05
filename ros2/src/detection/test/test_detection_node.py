"""
Integration test for detection_node.

These tests require a GPU whose TensorRT engine plan matches the one under models/
as well as PVA hardware.
"""

import pathlib
import struct
import unittest

import cv2
import launch_ros.actions
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import numpy as np
import pytest
import rclpy
from common_interfaces.msg import Vector2
from detection_interfaces.msg import DetectionResult, DetectionType
from detection_interfaces.srv import (
    GetDetectionResult,
    GetPositions,
    GetState,
    StartDetection,
    StopDetection,
)
from geometry_msgs.msg import TransformStamped
from launch import LaunchDescription
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster

NODE_NAME = "detection_node"
SERVICE_TIMEOUT_SEC = 10.0
MESSAGE_TIMEOUT_SEC = 15.0
RUNNER_MODEL = "RunnerSegYoloV8l.engine"
TEST_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "test_images"


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    detection_node = launch_ros.actions.Node(
        package="detection",
        executable="detection_node",
        name=NODE_NAME,
        parameters=[{"runner_model": RUNNER_MODEL}],
        output="screen",
    )
    return (
        LaunchDescription(
            [
                detection_node,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {"detection_node": detection_node},
    )


def _make_bayer_image_from_rgb(rgb, stamp):
    # Mosaic an RGB image down into a single-channel RGGB Bayer pattern (the
    # inverse of cv::COLOR_BayerRG2RGB)
    height, width = rgb.shape[:2]
    bayer = np.zeros((height, width), dtype=np.uint8)
    bayer[0::2, 0::2] = rgb[0::2, 0::2, 0]  # R
    bayer[0::2, 1::2] = rgb[0::2, 1::2, 1]  # G
    bayer[1::2, 0::2] = rgb[1::2, 0::2, 1]  # G
    bayer[1::2, 1::2] = rgb[1::2, 1::2, 2]  # B

    image = Image()
    image.header.stamp = stamp
    image.height = height
    image.width = width
    image.encoding = "bayer_rggb8"
    image.is_bigendian = 0
    image.step = width
    image.data = bayer.tobytes()
    return image


def _make_camera_info(width, height, stamp):
    info = CameraInfo()
    info.header.stamp = stamp
    info.width = width
    info.height = height
    fx = fy = float(width)
    info.k = [fx, 0.0, width / 2.0, 0.0, fy, height / 2.0, 0.0, 0.0, 1.0]
    info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    return info


def _make_depth_xyz(width, height, stamp):
    image = Image()
    image.header.stamp = stamp
    image.height = height
    image.width = width
    image.encoding = "32FC3"
    image.is_bigendian = 0
    image.step = width * 3 * 4
    # A flat plane 1000mm in front of the camera at every pixel.
    values = [0.0, 0.0, 1000.0] * (width * height)
    image.data = struct.pack("<%df" % len(values), *values)
    return image


class TestDetectionNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.test_node = Node("test_driver_node")
        self.get_state_client = self.test_node.create_client(
            GetState, f"/{NODE_NAME}/get_state"
        )
        self.start_detection_client = self.test_node.create_client(
            StartDetection, f"/{NODE_NAME}/start_detection"
        )
        self.stop_detection_client = self.test_node.create_client(
            StopDetection, f"/{NODE_NAME}/stop_detection"
        )
        self.stop_all_detections_client = self.test_node.create_client(
            Trigger, f"/{NODE_NAME}/stop_all_detections"
        )
        self.get_detection_client = self.test_node.create_client(
            GetDetectionResult, f"/{NODE_NAME}/get_detection"
        )
        self.get_positions_client = self.test_node.create_client(
            GetPositions, f"/{NODE_NAME}/get_positions"
        )

        # Ensure a clean slate
        self._call_service(self.stop_all_detections_client, Trigger.Request())

    def tearDown(self):
        self._call_service(self.stop_all_detections_client, Trigger.Request())
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

    def _assert_valid_runner_detection_result(self, detection_result, width, height):
        self.assertEqual(detection_result.detection_type, DetectionType.RUNNER)

        for instance in detection_result.instances:
            self.assertGreater(instance.confidence, 0.0)
            self.assertGreaterEqual(instance.point.x, 0.0)
            self.assertLess(instance.point.x, width)
            self.assertGreaterEqual(instance.point.y, 0.0)
            self.assertLess(instance.point.y, height)

        for point in detection_result.invalid_points:
            self.assertGreaterEqual(point.x, 0.0)
            self.assertLess(point.x, width)
            self.assertGreaterEqual(point.y, 0.0)
            self.assertLess(point.y, height)

    def test_initial_state_has_no_enabled_detections(self):
        result = self._call_service(self.get_state_client, GetState.Request())
        self.assertEqual(list(result.state.enabled_detection_types), [])
        self.assertFalse(result.state.recording_video)

    def test_start_stop_detection_updates_state(self):
        start_result = self._call_service(
            self.start_detection_client,
            StartDetection.Request(detection_type=DetectionType.RUNNER),
        )
        self.assertTrue(start_result.success)

        state_result = self._call_service(self.get_state_client, GetState.Request())
        self.assertIn(
            DetectionType.RUNNER, list(state_result.state.enabled_detection_types)
        )

        stop_result = self._call_service(
            self.stop_detection_client,
            StopDetection.Request(detection_type=DetectionType.RUNNER),
        )
        self.assertTrue(stop_result.success)

        state_result = self._call_service(self.get_state_client, GetState.Request())
        self.assertNotIn(
            DetectionType.RUNNER, list(state_result.state.enabled_detection_types)
        )

    def test_start_detection_twice_fails(self):
        first = self._call_service(
            self.start_detection_client,
            StartDetection.Request(detection_type=DetectionType.RUNNER),
        )
        self.assertTrue(first.success)

        second = self._call_service(
            self.start_detection_client,
            StartDetection.Request(detection_type=DetectionType.RUNNER),
        )
        self.assertFalse(second.success)

    def test_stop_detection_not_enabled_fails(self):
        result = self._call_service(
            self.stop_detection_client,
            StopDetection.Request(detection_type=DetectionType.RUNNER),
        )
        self.assertFalse(result.success)

    def test_stop_all_detections_clears_state(self):
        self._call_service(
            self.start_detection_client,
            StartDetection.Request(detection_type=DetectionType.RUNNER),
        )
        self._call_service(
            self.start_detection_client,
            StartDetection.Request(detection_type=DetectionType.LASER),
        )

        result = self._call_service(self.stop_all_detections_client, Trigger.Request())
        self.assertTrue(result.success)

        state_result = self._call_service(self.get_state_client, GetState.Request())
        self.assertEqual(list(state_result.state.enabled_detection_types), [])

    def test_detection_with_real_image(self):
        # Publish camera->world transforms and synthetic depth data so the node can resolve
        # detected points into 3D positions via RgbdAlignment, on top of running detection on a
        # real image.
        tf_broadcaster = StaticTransformBroadcaster(self.test_node)
        now = self.test_node.get_clock().now().to_msg()
        transforms = []
        for child_frame in ("color_camera", "depth_camera"):
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = "world"
            t.child_frame_id = child_frame
            t.transform.rotation.w = 1.0
            transforms.append(t)
        tf_broadcaster.sendTransform(transforms)

        # Use a test image known to contain detectable runners
        image_path = TEST_IMAGES_DIR / "20240703102906.png"
        bgr = cv2.imread(str(image_path))
        self.assertIsNotNone(bgr, f"Failed to load test image at {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]

        color_pub = self.test_node.create_publisher(
            Image, f"color/image_raw", qos_profile_sensor_data
        )
        color_info_pub = self.test_node.create_publisher(
            CameraInfo, f"color/camera_info", qos_profile_sensor_data
        )
        depth_pub = self.test_node.create_publisher(
            Image, f"depth/xyz", qos_profile_sensor_data
        )
        depth_info_pub = self.test_node.create_publisher(
            CameraInfo, f"depth/camera_info", qos_profile_sensor_data
        )

        # Also enable RUNNER detection so the node's detection thread runs on
        # each published frame and publishes to ~/detections
        start_result = self._call_service(
            self.start_detection_client,
            StartDetection.Request(detection_type=DetectionType.RUNNER),
        )
        self.assertTrue(start_result.success)

        topic_result = None

        def on_detection(msg):
            nonlocal topic_result
            if msg.detection_type == DetectionType.RUNNER and (
                msg.instances or msg.invalid_points
            ):
                topic_result = msg

        subscription = self.test_node.create_subscription(
            DetectionResult,
            f"/{NODE_NAME}/detections",
            on_detection,
            qos_profile_sensor_data,
        )

        # Publish image repeatedly until we receive a detection result, or until
        # we time out.
        end_time = self.test_node.get_clock().now().nanoseconds + int(
            MESSAGE_TIMEOUT_SEC * 1e9
        )
        try:
            while (
                topic_result is None
                and self.test_node.get_clock().now().nanoseconds < end_time
            ):
                stamp = self.test_node.get_clock().now().to_msg()
                color_info_pub.publish(_make_camera_info(width, height, stamp))
                depth_info_pub.publish(_make_camera_info(640, 480, stamp))
                # Publish depth first since we need it to be in the depth xyz queue before
                # detection is triggered by the color frame
                depth_pub.publish(_make_depth_xyz(640, 480, stamp))
                color_pub.publish(_make_bayer_image_from_rgb(rgb, stamp))
                rclpy.spin_once(self.test_node, timeout_sec=0.5)
        finally:
            self.test_node.destroy_subscription(subscription)

        self.assertIsNotNone(
            topic_result,
            "Timed out waiting for a non-empty DetectionResult on ~/detections topic",
        )
        self._assert_valid_runner_detection_result(topic_result, width, height)

        get_detection_result = self._call_service(
            self.get_detection_client,
            GetDetectionResult.Request(detection_type=DetectionType.RUNNER),
        )
        self._assert_valid_runner_detection_result(
            get_detection_result.result, width, height
        )


@launch_testing.post_shutdown_test()
class TestDetectionNodeShutdown(unittest.TestCase):
    def test_exit_code(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
