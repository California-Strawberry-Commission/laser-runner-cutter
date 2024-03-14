"""File: runner_cutter_control_node.py

Main ROS2 control node for the Laser Runner Cutter. This 
node uses a state machine to control the general objective of the 
system. States are things like calibrating the camera laser system,
finding a specific runner to burn, and burning said runner.  
"""

import asyncio
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from transitions.extensions.asyncio import AsyncMachine

from camera_control.camera_control_client import CameraControlClient
from laser_control.laser_control_client import LaserControlClient
from runner_cutter_control.calibration import Calibration
from runner_cutter_control.tracker import Tracker
from runner_cutter_control_interfaces.srv import AddCalibrationPoints, GetState


class Runner:
    def __init__(self, pos=None, point=None):
        self.pos = pos
        self.point = point
        self.corrected_laser_point = None


class ServiceWaitState:
    """State to wait for other services that need to come up"""

    def __init__(self, node, logger, laser_client, camera_client):
        super().__init__(outcomes=["services_up"])
        self.node = node
        self.logger = logger
        self.laser_client = laser_client
        self.camera_client = camera_client

    def execute(self, blackboard):
        self.logger.info("Entering State ServiceWait")
        self.node.publish_state("ServiceWait")

        # TODO: Check that laser connected correctly
        time.sleep(3)
        self.laser_client.wait_active()
        # Cache laser bounds on blackboard
        bounds = self.laser_client.get_bounds(1.0)
        blackboard.laser_x_bounds = (
            min(bounds, key=lambda point: point[0]),
            max(bounds, key=lambda point: point[0]),
        )
        blackboard.laser_y_bounds = (
            min(bounds, key=lambda point: point[1]),
            max(bounds, key=lambda point: point[1]),
        )

        # Check that the camera has frames
        self.logger.info("wait for frames")
        has_frames = False
        while not has_frames:
            has_frames = self.camera_client.has_frames()

        return "services_up"


class CalibrateState:
    """Calibrate the laser to camera, creating a 3D pos to laser frame transform."""

    def __init__(self, node, logger, calibration):
        super().__init__(outcomes=["failed_to_calibrate", "calibrated"])
        self.node = node
        self.logger = logger
        self.calibration = calibration

    def execute(self, blackboard):
        self.logger.info("Entering State Calibrate")
        self.node.publish_state("Calibrate")

        result = self.calibration.calibrate()
        return "calibrated" if result else "failed_to_calibrate"


class AcquireState:
    """A transition phase, checks if calibration is needed, or if runners are present to be trimmed"""

    def __init__(self, node, logger, camera_client, calibration, runner_tracker):
        super().__init__(
            outcomes=["calibration_needed", "target_acquired", "no_target_found"]
        )
        self.node = node
        self.logger = logger
        self.camera_client = camera_client
        self.calibration = calibration
        self.runner_tracker = runner_tracker

    def execute(self, blackboard):
        self.logger.info("Entering State Acquire")
        self.node.publish_state("Acquire")

        if not self.calibration.is_calibrated:
            return "calibration_needed"

        # If there are active tracks set it as the current track
        if self._set_available_track(blackboard):
            return "target_acquired"

        # If there are no active tracks check for new runners
        pos_data = self.camera_client.get_runner_pos()
        if pos_data["pos_list"]:
            for pos, point in zip(pos_data["pos_list"], pos_data["point_list"]):
                self.runner_tracker.add_track(pos, point)
            # after adding additional tracks, check if a new track is available.
            if self._set_available_track(blackboard):
                return "target_acquired"

        return "no_target_found"

    def _set_available_track(self, blackboard):
        if self.runner_tracker.has_active_tracks:
            blackboard.curr_track = self.runner_tracker.active_tracks[0]
            return True
        else:
            return False


class CorrectState:
    def __init__(
        self,
        node,
        logger,
        laser_client,
        camera_client,
        calibration,
        tracking_laser_color,
        runner_tracker,
    ):
        """Given a runner to cut, check that a tracking laser lines up with the runner. If not correct
        until they line up correctly.
        """
        super().__init__(outcomes=["on_target", "target_not_reached"])
        self.node = node
        self.logger = logger
        self.laser_client = laser_client
        self.camera_client = camera_client
        self.calibration = calibration
        self.tracking_laser_color = tracking_laser_color
        self.runner_tracker = runner_tracker

    def execute(self, blackboard):
        self.logger.info("Entering State Correct")
        self.node.publish_state("Correct")

        # Find the position of the needed laser point based on
        self.camera_client.set_exposure(0.001)
        laser_send_point = self.calibration.camera_point_to_laser_pixel(
            blackboard.curr_track.pos_wrt_cam
        )
        self.laser_client.start_laser(
            point=laser_send_point, color=self.tracking_laser_color
        )
        self.missing_laser_count = 0
        self.logger.info(
            f"laser_send_point: {laser_send_point} tracking_laser_color{self.tracking_laser_color}"
        )
        targ_reached = self._correct_laser(laser_send_point, blackboard)
        self.laser_client.stop_laser()
        self.logger.info(f"targ_reached: {targ_reached}")
        self.camera_client.set_exposure(-1.0)
        if targ_reached:
            return "on_target"
        else:
            self.logger.info("Targets not reached")
            self.runner_tracker.deactivate(blackboard.curr_track)
            return "target_not_reached"

    def _correct_laser(self, laser_send_point, blackboard):
        """Iteratively move the tracking laser until it is at the same point as the runner."""
        blackboard.curr_track.corrected_laser_point = laser_send_point

        laser_data = self.camera_client.get_laser_pos()
        if laser_data["point_list"]:
            if len(laser_data["point_list"]) == 1:
                laser_point = np.array(laser_data["point_list"][0])
            else:
                self.logger.info("to many lasers")
                self.missing_laser_count += 1
                if self.missing_laser_count > 20:
                    self.logger.info("Laser missing during state correct")
                    return False
                time.sleep(0.05)
                return self._correct_laser(laser_send_point, blackboard)

        else:
            self.missing_laser_count += 1
            if self.missing_laser_count > 20:
                self.logger.info("Laser missing during state correct")
                return False
            time.sleep(0.05)
            return self._correct_laser(laser_send_point, blackboard)

        # Using the saved runner point, this won't work once we begin moving
        runner_point = np.array(blackboard.curr_track.point)
        dist = np.linalg.norm(laser_point - runner_point)

        if dist > 2.5:
            correction = (runner_point - laser_point) / 10
            correction[1] = correction[1] * -1
            new_point = laser_send_point + correction
            self.logger.debug(
                f"Dist:{dist}, Correction{correction}, Old Point:{laser_send_point}, New Point{new_point}"
            )
            # TODO: Normalize dac points
            if np.any(
                new_point[0] > 4095
                or new_point[1] > 4095
                or new_point[0] < 0
                or new_point[1] < 0
            ):
                self.logger.info("Failed to reach pos, outside of laser window")
                return False

            self.laser_client.set_point(laser_send_point)
            self.missing_laser_count = 0
            return self._correct_laser(new_point, blackboard)

        return True


class BurnState:
    def __init__(
        self,
        node,
        logger,
        laser_client,
        camera_client,
        burn_color,
        burn_time_secs,
        runner_tracker,
    ):
        super().__init__(outcomes=["runner_removed"])
        self.node = node
        self.logger = logger
        self.laser_client = laser_client
        self.camera_client = camera_client
        self.burn_color = burn_color
        self.burn_time_secs = burn_time_secs
        self.runner_tracker = runner_tracker

    def execute(self, blackboard):
        self.logger.info("Entering State Burn")
        self.node.publish_state("Burn")

        # Turn on burn laser
        self.laser_client.start_laser(
            point=blackboard.curr_track.corrected_laser_point,
            color=self.burn_color,
        )
        time.sleep(self.burn_time_secs)

        self.laser_client.stop_laser()
        self.runner_tracker.deactivate(blackboard.curr_track)
        return "runner_removed"


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
                ("tracking_laser_color", [0.2, 0.0, 0.0]),
                ("burn_color", [0.0, 0.0, 1.0]),
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
        self.burn_color = (
            self.get_parameter("burn_color").get_parameter_value().double_array_value
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
        self.get_state_srv = self.create_service(
            GetState, "~/get_state", self._get_state_callback
        )

        # Pub/sub

        self.state_publisher = self.create_publisher(String, "~/state", 5)

        # Set up dependencies

        self.laser_client = LaserControlClient(self, self.laser_node_name)
        self.camera_client = CameraControlClient(self, self.camera_node_name)
        self.calibration = Calibration(
            self.laser_client,
            self.camera_client,
            self.tracking_laser_color,
            self.logger,
        )
        self.runner_tracker = Tracker(self.logger)

        # State machine

        # We run the state machine on a separate thread. Note that we cannot directly call
        # triggers on self.state_machine from service callbacks, as there are long running
        # tasks that run on certain states. So, in service callbacks, we queue triggers
        # on the state machine thread.
        self.state_machine = StateMachine(self.logger, self)
        self.state_machine_thread = StateMachineThread(self.state_machine, self.logger)
        self.state_machine_thread.start()

    def publish_state(self):
        msg = String(data=self.state_machine.state)
        self.state_publisher.publish(msg)

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

    def _get_state_callback(self, request, response):
        response.state = self.state_machine.state
        return response


class StateMachine:

    states = ["idle", "calibration", "add_calibration_points"]

    def __init__(self, logger, node):
        self.logger = logger
        self.node = node
        self.machine = AsyncMachine(
            model=self, states=StateMachine.states, initial="idle"
        )
        self.machine.add_transition("run_calibration", "idle", "calibration")
        self.machine.add_transition("calibration_complete", "calibration", "idle")
        self.machine.add_transition(
            "run_add_calibration_points", "idle", "add_calibration_points"
        )
        self.machine.add_transition(
            "add_calibration_points_complete", "add_calibration_points", "idle"
        )

    async def on_enter_calibration(self):
        self.node.publish_state()
        await self.node.calibration.calibrate()
        await self.calibration_complete()

    async def on_enter_add_calibration_points(self, camera_pixels):
        self.node.publish_state()
        # For each camera pixel, find the 3D position wrt the camera
        positions = await self.camera_client.get_positions_for_pixels(camera_pixels)
        # Filter out any invalid positions
        positions = [p for p in positions if not all(x < 0 for x in p)]
        # Convert camera positions to laser pixels
        laser_pixels = [
            self.calibration.camera_point_to_laser_pixel(position)
            for position in positions
        ]
        await self.calibration.add_calibration_points(laser_pixels)
        await self.add_calibration_points_complete()

    async def on_enter_idle(self):
        self.node.publish_state()


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
                await self.state_machine.trigger(trigger, *args, **kwargs)
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
