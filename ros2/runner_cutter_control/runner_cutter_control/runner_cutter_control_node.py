"""File: runner_cutter_control_node.py

Main ROS2 control node for the Laser Runner Cutter. This 
node uses a state machine to control the general objective of the 
system. States are things like calibrating the camera laser system,
finding a specific runner to burn, and burning said runner.  
"""

import asyncio
import threading

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger
from transitions.extensions.asyncio import AsyncMachine

from camera_control.camera_control_client import CameraControlClient
from laser_control.laser_control_client import LaserControlClient
from runner_cutter_control.calibration import Calibration
from runner_cutter_control.tracker import Tracker, TrackState
from runner_cutter_control_interfaces.msg import State
from runner_cutter_control_interfaces.srv import AddCalibrationPoints, GetState


class RunnerCutterControlNode(Node):
    def __init__(self):
        super().__init__("runner_cutter_control_node")
        self.logger = self.get_logger()

        # Parameters

        self.declare_parameters(
            namespace="",
            parameters=[
                ("laser_node_name", "laser"),
                ("camera_node_name", "camera"),
                ("tracking_laser_color", [0.15, 0.0, 0.0]),
                ("burn_laser_color", [0.0, 0.0, 1.0]),
                ("burn_time_secs", 5),
            ],
        )
        self.laser_node_name = (
            self.get_parameter("laser_node_name").get_parameter_value().string_value
        )
        self.camera_node_name = (
            self.get_parameter("camera_node_name").get_parameter_value().string_value
        )
        self.tracking_laser_color = (
            self.get_parameter("tracking_laser_color")
            .get_parameter_value()
            .double_array_value
        )
        self.burn_laser_color = (
            self.get_parameter("burn_laser_color")
            .get_parameter_value()
            .double_array_value
        )
        self.burn_time_secs = (
            self.get_parameter("burn_time_secs").get_parameter_value().integer_value
        )

        # Services

        # TODO: use action instead once there's a new release of roslib. Currently
        # roslib does not support actions with ROS2
        self.calibrate_srv = self.create_service(
            Trigger,
            "~/calibrate",
            self._calibrate_callback,
        )
        self.add_calibration_points_srv = self.create_service(
            AddCalibrationPoints,
            "~/add_calibration_points",
            self._add_calibration_points_callback,
        )
        self.start_runner_cutter_srv = self.create_service(
            Trigger,
            "~/start_runner_cutter",
            self._start_runner_cutter_callback,
        )
        self.start_runner_cutter_srv = self.create_service(
            Trigger,
            "~/stop",
            self._stop_callback,
        )
        self.get_state_srv = self.create_service(
            GetState, "~/get_state", self._get_state_callback
        )

        # Pub/sub

        self.state_publisher = self.create_publisher(State, "~/state", 5)

        # Set up dependencies

        self.laser_client = LaserControlClient(self, self.laser_node_name, self.logger)
        self.camera_client = CameraControlClient(
            self, self.camera_node_name, self.logger
        )
        self.laser_client.wait_active()
        self.camera_client.wait_active()
        self.calibration = Calibration(
            self.laser_client,
            self.camera_client,
            self.tracking_laser_color,
            logger=self.logger,
        )
        self.runner_tracker = Tracker(self.logger)

        # State machine

        # We run the state machine on a separate thread that runs an asyncio event loop.
        # Note that we cannot directly call triggers on self.state_machine from service callbacks,
        # as there are long running tasks that run on certain states. So, from service callbacks,
        # we queue triggers that will be processed on the state machine thread.
        self.state_machine = StateMachine(
            self,
            self.laser_client,
            self.camera_client,
            self.calibration,
            self.runner_tracker,
            self.tracking_laser_color,
            self.burn_laser_color,
            self.burn_time_secs,
            self.logger,
        )
        self.state_machine_thread = StateMachineThread(self.state_machine, self.logger)
        self.state_machine_thread.start()

    def get_state(self):
        return State(
            calibrated=self.state_machine.is_calibrated, state=self.state_machine.state
        )

    def publish_state(self):
        self.state_publisher.publish(self.get_state())

    def _calibrate_callback(self, request, response):
        if self.state_machine.state == "idle":
            self.state_machine_thread.queue("run_calibration")
            response.success = True
        return response

    def _add_calibration_points_callback(self, request, response):
        if self.state_machine.state == "idle":
            camera_pixels = [(pixel.x, pixel.y) for pixel in request.camera_pixels]
            self.state_machine_thread.queue("run_add_calibration_points", camera_pixels)
            response.success = True
        return response

    def _start_runner_cutter_callback(self, request, response):
        if self.state_machine.state == "idle":
            self.state_machine_thread.queue("run_runner_cutter")
            response.success = True
        return response

    def _stop_callback(self, request, response):
        if self.state_machine.state != "idle":
            self.state_machine_thread.queue("stop")
            response.success = True
        return response

    def _get_state_callback(self, request, response):
        response.state = self.get_state()
        return response


class StateMachine:

    states = [
        "idle",
        "calibration",
        "add_calibration_points",
        "acquire_target",
        "aim_laser",
        "burn_target",
    ]

    def __init__(
        self,
        node,
        laser_client,
        camera_client,
        calibration,
        runner_tracker,
        tracking_laser_color,
        burn_laser_color,
        burn_time_secs,
        logger,
    ):
        self.node = node
        self.laser_client = laser_client
        self.camera_client = camera_client
        self.calibration = calibration
        self.runner_tracker = runner_tracker
        self.tracking_laser_color = tracking_laser_color
        self.burn_laser_color = burn_laser_color
        self.burn_time_secs = burn_time_secs
        self.logger = logger

        self.calibrated = False

        self.machine = AsyncMachine(
            model=self, states=StateMachine.states, initial="idle", queued=False
        )
        self.machine.add_transition("run_calibration", "idle", "calibration")
        self.machine.add_transition("calibration_complete", "calibration", "idle")
        self.machine.add_transition(
            "run_add_calibration_points",
            "idle",
            "add_calibration_points",
            conditions=["is_calibrated"],
        )
        self.machine.add_transition(
            "add_calibration_points_complete", "add_calibration_points", "idle"
        )
        self.machine.add_transition(
            "run_runner_cutter", "idle", "acquire_target", conditions=["is_calibrated"]
        )
        self.machine.add_transition("target_acquired", "acquire_target", "aim_laser")
        self.machine.add_transition(
            "no_target_found", "acquire_target", "acquire_target"
        )
        self.machine.add_transition("aim_successful", "aim_laser", "burn_target")
        self.machine.add_transition("aim_failed", "aim_laser", "acquire_target")
        self.machine.add_transition("burn_complete", "burn_target", "acquire_target")
        self.machine.add_transition("stop", "*", "idle")

    @property
    def is_calibrated(self):
        return self.calibrated

    async def on_enter_idle(self):
        self.node.publish_state()
        await self.laser_client.stop_laser()
        await self.laser_client.clear_points()
        await self.camera_client.auto_exposure()

    async def on_enter_calibration(self):
        self.node.publish_state()
        self.calibrated = False
        await self.calibration.calibrate()
        self.calibrated = True
        await self.calibration_complete()

    async def on_enter_add_calibration_points(self, camera_pixels):
        self.node.publish_state()
        # For each camera pixel, find the 3D position wrt the camera
        positions = await self.camera_client.get_positions_for_pixels(camera_pixels)
        # Filter out any invalid positions
        positions = [p for p in positions if not all(x < 0 for x in p)]
        # Convert camera positions to laser pixels
        laser_coords = [
            self.calibration.camera_point_to_laser_coord(position)
            for position in positions
        ]
        await self.calibration.add_calibration_points(
            laser_coords, update_transform=True
        )
        await self.add_calibration_points_complete()

    async def on_enter_acquire_target(self):
        self.node.publish_state()

        # Do nothing if there is already an active target
        if self.runner_tracker.has_active_tracks:
            self.logger.info("Active target already exists.")
            await self.target_acquired(self.runner_tracker.active_tracks[0])
            return

        # If there are no pending tracks, check for new runners
        if not self.runner_tracker.has_pending_tracks:
            detection_result = await self.camera_client.get_runners()
            for instance in detection_result["instances"]:
                self.runner_tracker.add_track(instance["point"], instance["position"])

        if self.runner_tracker.has_pending_tracks:
            # There are pending tracks. Set one as active.
            self.logger.info("Found a pending track. Marking as active.")
            track = self.runner_tracker.pending_tracks[0]
            track.state = TrackState.ACTIVE
            await self.target_acquired(track)
        else:
            # There are no pending targets.
            self.logger.info("No pending targets found.")
            await self.no_target_found()

    async def on_enter_aim_laser(self, target):
        self.node.publish_state()

        # TODO: set exposure automatically when detecting laser
        await self.camera_client.set_exposure(0.001)
        initial_laser_coord = self.calibration.camera_point_to_laser_coord(
            target.position
        )
        await self.laser_client.start_laser(
            point=initial_laser_coord, color=self.tracking_laser_color
        )
        corrected_laser_coord = await self._correct_laser(
            target.pixel, initial_laser_coord
        )
        await self.laser_client.stop_laser()
        await self.camera_client.auto_exposure()
        if corrected_laser_coord is not None:
            await self.aim_successful(target, corrected_laser_coord)
        else:
            self.logger.info("Failed to aim laser.")
            target.state = TrackState.FAILED
            await self.aim_failed()

    async def _get_laser_pixel_and_pos(self, max_attempts=3):
        attempt = 0
        while attempt < max_attempts:
            detection_result = await self.camera_client.get_lasers()
            instances = detection_result["instances"]
            if instances:
                if len(instances) > 1:
                    self.logger.info("Found more than 1 laser during correction")
                # TODO: better handle case where more than 1 laser detected
                instance = instances[0]
                return instance["point"], instance["position"]
            # No lasers detected. Try again.
            await asyncio.sleep(0.2)
            attempt += 1
        return None, None

    async def _correct_laser(
        self,
        target_pixel,
        original_laser_coord,
        dist_threshold=2.5,
    ):
        current_laser_coord = original_laser_coord

        while True:
            await self.laser_client.set_point(current_laser_coord)
            # Wait for galvo to settle and for camera frame capture
            await asyncio.sleep(0.1)
            laser_pixel, laser_pos = await self._get_laser_pixel_and_pos()
            if laser_pixel is None:
                self.logger.info("Could not detect laser.")
                return None

            # Calculate camera pixel distance
            laser_pixel = np.array(laser_pixel)
            target_pixel = np.array(target_pixel)
            dist = np.linalg.norm(laser_pixel - target_pixel)
            if dist <= dist_threshold:
                return current_laser_coord
            else:
                # Use this opportunity to add to calibration points since we have the laser
                # coord and associated position in camera space
                await self.calibration.add_point_correspondence(
                    current_laser_coord, laser_pos, update_transform=True
                )

                # TODO: scale correction by camera frame size
                correction = (target_pixel - laser_pixel) / 10
                # Invert Y axis as laser coord Y is flipped from camera frame Y
                correction[1] *= -1
                new_laser_coord = current_laser_coord + correction
                self.logger.debug(
                    f"Correcting laser. Dist = {dist}, correction = {correction}, current coord = {current_laser_coord}, new coord = {new_laser_coord}"
                )

                if np.any(
                    new_laser_coord[0] > 1.0
                    or new_laser_coord[1] > 1.0
                    or new_laser_coord[0] < 0.0
                    or new_laser_coord[1] < 0.0
                ):
                    self.logger.info("Laser coord is outside of renderable area.")
                    return None

                current_laser_coord = new_laser_coord

    async def on_enter_burn_target(self, target, laser_coord):
        self.node.publish_state()

        await self.laser_client.start_laser(
            point=laser_coord, color=self.burn_laser_color
        )
        await asyncio.sleep(self.burn_time_secs)
        await self.laser_client.stop_laser()
        target.state = TrackState.COMPLETED
        await self.burn_complete()


class StateMachineThread(threading.Thread):

    def __init__(self, state_machine, logger):
        super().__init__()
        self.daemon = True
        self.state_machine = state_machine
        self.logger = logger
        self._queued_trigger = None
        self._queued_args = None
        self._queued_kwargs = None
        self._lock = threading.Lock()

    def queue(self, trigger, *args, **kwargs):
        with self._lock:
            self._queued_trigger = trigger
            self._queued_args = args
            self._queued_kwargs = kwargs

    async def _state_machine_task(self):
        while True:
            trigger = None
            with self._lock:
                if self._queued_trigger is not None:
                    trigger = self._queued_trigger
                    args = self._queued_args
                    kwargs = self._queued_kwargs
                    self._queued_trigger = None
                    self._queued_args = None
                    self._queued_kwargs = None
            if trigger is not None:
                # Create a task so we don't block the queue processing loop
                task = asyncio.create_task(
                    self.state_machine.trigger(trigger, *args, **kwargs)
                )
            await asyncio.sleep(0.1)

    def run(self):
        asyncio.run(self._state_machine_task())


def main(args=None):
    rclpy.init(args=args)
    node = RunnerCutterControlNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
