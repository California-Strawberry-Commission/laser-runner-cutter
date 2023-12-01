"""File: control_node.py

Main ros2 control node for the laser runner removal system. This 
node uses a state machine to control the general objective of the 
system. States are things like calibrating the camera laser system,
finding a specific runner to burn, and burning said runner.  
"""
import time

import numpy as np
import rclpy
from simple_node import Node
from std_msgs.msg import String
from yasmin import Blackboard, State, StateMachine

from camera_control.camera_node_client import CameraNodeClient
from control_node.calibration import Calibration
from control_node.tracker import Tracker
from laser_control.laser_node_client import LaserNodeClient


class Runner:
    def __init__(self, pos=None, point=None):
        self.pos = pos
        self.point = point
        self.corrected_laser_point = None


class Initialize(State):
    """Initialization state for setting up resources needed by the system"""

    def __init__(self, node, logger):
        super().__init__(outcomes=["init_complete"])
        self.node = node
        self.logger = logger

    def execute(self, blackboard):
        self.logger.info("Entering State Initialize")
        self.node.publish_state("Initialize")

        return "init_complete"


class ServiceWait(State):
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

        # Check that the camera has frames
        self.logger.info("wait for frames")
        has_frames = False
        while not has_frames:
            has_frames = self.camera_client.has_frames()

        return "services_up"


class Calibrate(State):
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


class Acquire(State):
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
            # Publish the runner point we are attempting to cut for logging and image drawing
            self.camera_client.pub_runner_point(blackboard.curr_track.point)
            return True
        else:
            return False


class Correct(State):
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
        laser_send_point = self.calibration.camera_point_to_laser_pixel(
            blackboard.curr_track.pos_wrt_cam
        )

        self.laser_client.start_laser(
            point=laser_send_point, color=self.tracking_laser_color
        )
        self.missing_laser_count = 0
        targ_reached = self._correct_laser(laser_send_point, blackboard)
        self.laser_client.stop_laser()
        self.logger.info(f"targ_reached: {targ_reached}")
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
                self.logger.debug("to many lasers")
        else:
            self.missing_laser_count += 1
            if self.missing_laser_count > 20:
                self.logger.info("Laser missing during state correct")
                return False
            time.sleep(0.01)
            return self._correct_laser(laser_send_point, blackboard)

        # Using the saved runner point, this won't work once we begin moving
        runner_point = np.array(blackboard.curr_track.point)
        dist = np.linalg.norm(laser_point - runner_point)

        if dist > 10:
            correction = (runner_point - laser_point) / 10
            correction[1] = correction[1] * -1
            new_point = laser_send_point + correction
            self.logger.debug(
                f"Dist:{dist}, Correction{correction}, Old Point:{laser_send_point}, New Point{new_point}"
            )
            # Need to add min max checks to dac or otherwise account for different scales
            if (
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


class Burn(State):
    def __init__(
        self, node, logger, laser_client, burn_color, burn_time_secs, runner_tracker
    ):
        super().__init__(outcomes=["runner_removed"])
        self.node = node
        self.logger = logger
        self.laser_client = laser_client
        self.burn_color = burn_color
        self.burn_time_secs = burn_time_secs
        self.runner_tracker = runner_tracker

    def execute(self, blackboard):
        self.logger.info("Entering State Burn")
        self.node.publish_state("Burn")

        burn_start = time.time()

        # Turn on burn laser
        self.laser_client.start_laser(
            point=blackboard.curr_track.corrected_laser_point,
            color=self.burn_color,
        )

        while burn_start + self.burn_time_secs > time.time():
            time.sleep(0.1)

        self.laser_client.stop_laser()
        self.runner_tracker.deactivate(blackboard.curr_track)
        return "runner_removed"


class MainNode(Node):
    def __init__(self):
        """State Machine that controls what the system is currently doing"""
        super().__init__("main_node")
        self.logger = self.get_logger()

        # Parameters

        self.declare_parameters(
            namespace="",
            parameters=[
                ("laser_node_name", "laser"),
                ("camera_node_name", "camera"),
                ("tracking_laser_color", [0.2, 0.0, 0.0]),
                ("burn_color", [0.0, 0.0, 1.0]),
                ("burn_time", 5),
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
            self.get_parameter("burn_time").get_parameter_value().integer_value
        )

        # Pub/sub

        self.state_publisher = self.create_publisher(String, "~/state", 5)

        # Set up dependencies

        self.laser_client = LaserNodeClient(self, self.laser_node_name)
        self.camera_client = CameraNodeClient(self, self.camera_node_name)
        self.calibration = Calibration(
            self.laser_client,
            self.camera_client,
            self.tracking_laser_color,
            self.logger,
        )
        self.runner_tracker = Tracker()

        # Set up state machine

        blackboard = Blackboard()
        blackboard.curr_track = None

        main_sm = StateMachine(outcomes=["Finished"])
        main_sm.add_state(
            "INIT", Initialize(self, self.logger), transitions={"init_complete": "WAIT"}
        )
        main_sm.add_state(
            "WAIT",
            ServiceWait(self, self.logger, self.laser_client, self.camera_client),
            transitions={"services_up": "CALIB"},
        )
        main_sm.add_state(
            "CALIB",
            Calibrate(self, self.logger, self.calibration),
            transitions={"failed_to_calibrate": "Finished", "calibrated": "ACQ"},
        )
        main_sm.add_state(
            "ACQ",
            Acquire(
                self,
                self.logger,
                self.camera_client,
                self.calibration,
                self.runner_tracker,
            ),
            transitions={
                "calibration_needed": "CALIB",
                "target_acquired": "CORRECT",
                "no_target_found": "ACQ",
            },
        )
        main_sm.add_state(
            "CORRECT",
            Correct(
                self,
                self.logger,
                self.laser_client,
                self.camera_client,
                self.calibration,
                self.tracking_laser_color,
                self.runner_tracker,
            ),
            transitions={"on_target": "BURN", "target_not_reached": "ACQ"},
        )
        main_sm.add_state(
            "BURN",
            Burn(
                self,
                self.logger,
                self.laser_client,
                self.burn_color,
                self.burn_time_secs,
                self.runner_tracker,
            ),
            transitions={"runner_removed": "ACQ"},
        )

        outcome = main_sm(blackboard)
        self.logger.info(outcome)

    def publish_state(self, state_name):
        msg = String()
        msg.data = state_name
        self.state_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MainNode()
    node.join_spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
