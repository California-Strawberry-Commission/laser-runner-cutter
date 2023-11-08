"""File: main_node.py

Main ros2 control node for the laser runner removal system. This 
node uses a state machine to control the general objective of the 
system. States are things like calibrating the camera laser system,
finding a specific runner to burn, and burning said runner.  
"""
import rclpy

import numpy as np

from simple_node import Node
import time
from laser_control.laser_node_client import LaserNodeClient
from laser_runner_removal.Tracker import Tracker
from camera_control.camera_node_client import CameraNodeClient
from yasmin import State, StateMachine, Blackboard

DEBUG = 1


class Runner:
    def __init__(self, pos=None, point=None):
        self.pos = pos
        self.point = point
        self.corrected_laser_point = None


class Initialize(State):
    """Initialization state for setting up resources needed by the system"""

    def __init__(self):
        State.__init__(self, outcomes=["init_complete"])

    def execute(self, blackboard):
        blackboard.main_node.logger.info("Entering State Initialize")

        return "init_complete"


class ServiceWait(State):
    """State to wait for other services that need to come up"""

    def __init__(self):
        State.__init__(self, outcomes=["services_up"])

    def execute(self, blackboard):
        node = blackboard.main_node

        # THIS SHOULDN'T BE NEEDED
        time.sleep(3)
        node.logger.info("Entering State ServiceWait")
        node.laser_client.wait_active()

        # Currently don't support enabling multiple laser connections
        node.logger.info("connecting to laser")
        node.laser_client.connect()
        node.logger.info("laser connected")

        # ToDo add check laser connected correctly

        # Check that the camera has frames
        node.logger.info("wait for frames")
        has_frames = False
        while not has_frames:
            has_frames = node.camera_client.has_frames()

        return "services_up"


class Calibrate(State):
    """Calibrate the laser to camera, creating a 3D pos to laser frame transform."""

    def __init__(self):
        State.__init__(self, outcomes=["failed_to_calibrate", "calibrated"])

    def execute(self, blackboard):
        node = blackboard.main_node
        node.logger.info("Entering State Calibrate")
        scale_factors = [0.15, 0.25]
        lsr_pts = node.laser_client.get_scaled_frame_corners(scale_factors)
        found_lsr_pts = []
        found_img_pts = []
        pos_wrt_cam = []

        for point in lsr_pts:
            attempts = 0
            node.laser_client.stop_laser()
            node.laser_client.start_laser(point=point, color=node.tracking_laser_color)

            while attempts < 50:
                attempts += 1
                pos_data = node.camera_client.get_laser_pos()
                if pos_data["pos_list"]:
                    # can add error if len greater then 1
                    found_lsr_pts.append(point)
                    found_img_pts.append(pos_data["point_list"][0])
                    pos_wrt_cam.append(pos_data["pos_list"][0])
                    break

                # Should be a ros way to sleep
                time.sleep(0.005)

        node.laser_client.stop_laser()

        if len(found_lsr_pts) >= 3:
            pos_wrt_cam = np.array(pos_wrt_cam)
            found_lsr_pts = np.array(found_lsr_pts)

            # Solve for transform between 3D points 'pos_wrt_cam' and 2D points 'found_lsr_pts'
            # Create an augmented matrix A to solve for the transformation matrix T
            res = np.linalg.lstsq(pos_wrt_cam, found_lsr_pts, rcond=None)
            node.cam_to_laser = res[0]
            node.logger.info("----------Calibration Test----------")
            node.logger.info(f"Sent points: \n{found_lsr_pts}")
            node.logger.info(f"Found img points: \n{found_img_pts}")
            node.logger.info(f"Calculated points: \n{np.dot(pos_wrt_cam,res[0])}")
            return "calibrated"
        else:
            node.logger.info("failed to find at least 3 laser points")
            return "failed_to_calibrate"


class Acquire(State):
    """A transition phase, checks if calibration is needed, or if runners are present to be trimmed"""

    def __init__(self):
        State.__init__(
            self, outcomes=["calibration_needed", "target_acquired", "no_target_found"]
        )

    def execute(self, blackboard):
        node = blackboard.main_node
        node.logger.info("Entering State Acquire")
        if node.cam_to_laser is None:
            return "calibration_needed"

        # If there are active tracks set it as the current track
        if self._set_available_track(node):
            return "target_acquired"

        # If there are no active tracks check for new runners
        pos_data = node.camera_client.get_runner_pos()
        if pos_data["pos_list"]:
            for pos, point in zip(pos_data["pos_list"], pos_data["point_list"]):
                node.runner_tracker.add_track(pos, point)
            # after adding additional tracks, check if a new track is available.
            if self._set_available_track(node):
                return "target_acquired"

        return "no_target_found"

    def _set_available_track(self, node):
        if node.runner_tracker.has_active_tracks:
            node.curr_track = node.runner_tracker.active_tracks[0]
            # Publish the runner point we are attempting to cut for logging and image drawing
            node.camera_client.pub_runner_point(node.curr_track.point)
            return True
        else:
            return False


class Correct(State):
    def __init__(self):
        """Given a runner to cut, check that a tracking laser lines up with the runner. If not correct
        until they line up correctly.
        """
        State.__init__(self, outcomes=["on_target", "target_not_reached"])

    def execute(self, blackboard):
        node = blackboard.main_node
        node.logger.info("Entering State Correct")
        # Find the position of the needed laser point based on
        laser_send_point = node.curr_track.pos_wrt_cam @ node.cam_to_laser

        node.laser_client.start_laser(
            point=laser_send_point, color=node.tracking_laser_color
        )
        self.missing_laser_count = 0
        targ_reached = self._correct_laser(laser_send_point, blackboard)
        node.laser_client.stop_laser()
        blackboard.main_node.logger.info(f"targ_reached: {targ_reached}")
        if targ_reached:
            return "on_target"
        else:
            node.logger.info("Targets not reached")
            node.runner_tracker.deactivate(node.curr_track)
            return "target_not_reached"

    def _correct_laser(self, laser_send_point, blackboard):
        """Iteratively move the tracking laser until it is at the same point as the runner."""
        node = blackboard.main_node
        node.curr_track.corrected_laser_point = laser_send_point

        laser_data = node.camera_client.get_laser_pos()
        if laser_data["point_list"]:
            if len(laser_data["point_list"]) == 1:
                laser_point = np.array(laser_data["point_list"][0])
            else:
                node.logger.debug("to many lasers")
        else:
            self.missing_laser_count += 1
            if self.missing_laser_count > 20:
                node.logger.info("Laser missing during state correct")
                return False
            time.sleep(0.01)
            return self._correct_laser(laser_send_point, blackboard)

        # Using the saved runner point, this won't work once we begin moving
        runner_point = np.array(node.curr_track.point)
        dist = np.linalg.norm(laser_point - runner_point)

        if dist > 10:
            correction = (runner_point - laser_point) / 10
            correction[1] = correction[1] * -1
            new_point = laser_send_point + correction
            node.logger.debug(
                f"Dist:{dist}, Correction{correction}, Old Point:{laser_send_point}, New Point{new_point}"
            )
            # Need to add min max checks to dac or otherwise account for different scales
            if (
                new_point[0] > 4095
                or new_point[1] > 4095
                or new_point[0] < 0
                or new_point[1] < 0
            ):
                node.logger.info("Failed to reach pos, outside of laser window")
                return False

            node.laser_client.set_point(laser_send_point)
            self.missing_laser_count = 0
            return self._correct_laser(new_point, blackboard)

        return True


class Burn(State):
    def __init__(self):
        State.__init__(self, outcomes=["runner_removed"])

    def execute(self, blackboard):
        node = blackboard.main_node
        node.logger.info("Entering State Burn")
        burn_start = time.time()

        # Turn on burn laser
        node.laser_client.start_laser(
            point=node.curr_track.corrected_laser_point,
            color=blackboard.main_node.burn_color,
        )

        while burn_start + blackboard.main_node.burn_time > time.time():
            time.sleep(0.1)

        node.laser_client.stop_laser()
        node.runner_tracker.deactivate(node.curr_track)
        return "runner_removed"


class MainNode(Node):
    def __init__(self):
        """State Machine that controls what the system is currently doing"""
        Node.__init__(self, "main_node")
        self.logger = self.get_logger()

        # declare parameters from a ros config file, if no parameter is found, the default is used
        self.declare_parameters(
            namespace="",
            parameters=[
                ("tracking_laser_color", [0.2, 0.0, 0.0]),
                ("burn_color", [0.0, 0.0, 1.0]),
                ("burn_time", 5),
            ],
        )
        self.tracking_laser_color = (
            self.get_parameter("tracking_laser_color")
            .get_parameter_value()
            .double_array_value
        )
        self.burn_color = (
            self.get_parameter("burn_color").get_parameter_value().double_array_value
        )
        self.burn_time = (
            self.get_parameter("burn_time").get_parameter_value().integer_value
        )

        main_sm = StateMachine(outcomes=["Finished"])

        main_sm.add_state("INIT", Initialize(), transitions={"init_complete": "WAIT"})
        main_sm.add_state("WAIT", ServiceWait(), transitions={"services_up": "CALIB"})
        main_sm.add_state(
            "CALIB",
            Calibrate(),
            transitions={"failed_to_calibrate": "Finished", "calibrated": "ACQ"},
        )
        main_sm.add_state(
            "ACQ",
            Acquire(),
            transitions={
                "calibration_needed": "CALIB",
                "target_acquired": "CORRECT",
                "no_target_found": "ACQ",
            },
        )
        main_sm.add_state(
            "CORRECT",
            Correct(),
            transitions={"on_target": "BURN", "target_not_reached": "ACQ"},
        )
        main_sm.add_state("BURN", Burn(), transitions={"runner_removed": "ACQ"})

        self.cam_to_laser = None
        self.curr_track = None
        self.laser_client = LaserNodeClient(self)
        self.camera_client = CameraNodeClient(self)
        self.runner_tracker = Tracker()
        blackboard = Blackboard()
        blackboard.main_node = self

        self.cam_info_received = False

        outcome = main_sm(blackboard)
        self.logger.info(outcome)


def main(args=None):
    rclpy.init(args=args)
    node = MainNode()
    node.join_spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
