"""File: main_node.py

Main ros2 control node for the laser runner removal system. This 
node uses a state machine to control the general objective of the 
system. States are things like calibrating the camera laser system,
finding a specific runner to burn, and burning said runner.  
"""
import rclpy

import numpy as np

from simple_node import Node
from laser_runner_removal.laser import IldaLaser
from laser_runner_removal.realsense import RealSense
from laser_runner_removal.cv_utils import find_laser_point
from laser_runner_removal.ts_queue import TsQueue
from laser_runner_removal.Tracker import Tracker
from lrr_interfaces.srv import RetrieveFrame
from lrr_interfaces.msg import LaserOn, PosData, Point
import cv2
import time

from yasmin import State, StateMachine, Blackboard

DEBUG = 1


class Initialize(State):
    """Initialization state for setting up resources needed by the system"""

    def __init__(self):
        State.__init__(self, outcomes=["init_complete"])

    def execute(self, blackboard):
        blackboard.main_node.logger.info("Entering State Initialize")

        blackboard.laser = IldaLaser(logger=blackboard.main_node.logger)
        blackboard.laser.initialize()

        # Add wait for dependent states check

        return "init_complete"


class ServiceWait(State):
    """State to wait for other services that need to come up"""

    def __init__(self):
        State.__init__(self, outcomes=["services_up"])

    def execute(self, blackboard):
        while not blackboard.main_node.cam_info_received:
            time.sleep(0.005)

        return "services_up"


class Calibrate(State):
    """Calibrate the laser to camera, creating a 3D pos to laser frame transform."""

    def __init__(self):
        State.__init__(self, outcomes=["failed_to_calibrate", "calibrated"])

    def execute(self, blackboard):
        blackboard.main_node.logger.info("Entering State Calibrate")
        lsr_pts = blackboard.laser.scale_frame_corners(0.15)
        lsr_pts += blackboard.laser.scale_frame_corners(0.25)
        found_lsr_pts = []
        found_img_pts = []
        pos_wrt_cam = []

        laser_msg = LaserOn()
        laser_msg.laser_state = True
        blackboard.main_node.laser_state_pub.publish(laser_msg)

        for point in lsr_pts:
            found_pos = None
            attempts = 0
            blackboard.laser.sendEmpty(x=point[0], y=point[0])
            blackboard.main_node.laser_pos_queue.empty()
            laser_send_ts = time.time()
            blackboard.main_node.logger.debug(
                f"Point:{point} Laser Send Ts:{laser_send_ts}"
            )
            while not found_pos and attempts < 50:
                blackboard.laser.add_point(
                    point, pad=False, color=(10, 0, 0), intensity=1
                )
                blackboard.laser.sendFrame()

                attempts += 1
                last_ts, laser_data = blackboard.main_node.laser_pos_queue.get_ts(
                    time.time()
                )

                # Add offset to make sure correct laser pos is detected
                blackboard.main_node.logger.debug(f"Found laser Ts:{last_ts}")
                if last_ts and last_ts > laser_send_ts + 0.05:
                    found_pos, found_point = laser_data
                    found_pos = found_pos[0]
                    found_point = found_point[0]
                else:
                    time.sleep(0.005)

            if found_pos:
                blackboard.main_node.logger.debug(f"Send Laser Point:{point}")
                blackboard.main_node.logger.debug(f"Found Laser Pos:{found_pos}")
                found_lsr_pts.append(point)
                found_img_pts.append([found_point.x, found_point.y])
                pos_wrt_cam.append([found_pos.x, found_pos.y, found_pos.z])

        blackboard.laser.sendEmpty()
        laser_msg.laser_state = False
        blackboard.main_node.logger.info("Setting Laser Off")
        blackboard.main_node.laser_state_pub.publish(laser_msg)

        if len(found_lsr_pts) >= 3:
            pos_wrt_cam = np.array(pos_wrt_cam)
            found_lsr_pts = np.array(found_lsr_pts)

            # Solve for transform between 3D points 'pos_wrt_cam' and 2D points 'found_lsr_pts'
            # Create an augmented matrix A to solve for the transformation matrix T
            res = np.linalg.lstsq(pos_wrt_cam, found_lsr_pts, rcond=None)
            blackboard.laser.transform_to_laser = res[0]
            blackboard.main_node.logger.debug("----------Calibration Test----------")
            blackboard.main_node.logger.debug(f"Sent points: \n{found_lsr_pts}")
            blackboard.main_node.logger.debug(f"Found img points: \n{found_img_pts}")
            blackboard.main_node.logger.debug(
                f"Calculated points: \n{np.dot(pos_wrt_cam,res[0])}"
            )
            return "calibrated"
        else:
            blackboard.main_node.logger.info("failed to find at least 3 laser points")
            return "failed_to_calibrate"


class Acquire(State):
    """A transition phase, checks if calibration is needed, or if runners are present to be trimmed"""

    def __init__(self):
        State.__init__(
            self, outcomes=["calibration_needed", "target_acquired", "no_target_found"]
        )

    def execute(self, blackboard):
        blackboard.main_node.logger.info("Entering State Acquire")
        if blackboard.laser.transform_to_laser is None:
            return "calibration_needed"

        if blackboard.main_node.runner_tracker.has_active_tracks:
            # currently only do one runner at at time
            blackboard.curr_track = blackboard.main_node.runner_tracker.active_tracks[0]
            runner_msg = Point()
            runner_point = blackboard.curr_track.point
            runner_msg.x = runner_point[0]
            runner_msg.y = runner_point[1]
            blackboard.main_node.runner_point_pub.publish(runner_msg)
            return "target_acquired"

        else:
            time.sleep(0.01)
            return "no_target_found"

        return True


class Correct(State):
    def __init__(self):
        """Given a runner to cut, check that a tracking laser lines up with the runner. If not correct
        until they line up correctly.
        """
        State.__init__(self, outcomes=["on_target", "target_not_reached"])

    def execute(self, blackboard):
        blackboard.main_node.logger.info("Entering State Correct")

        # Turn on tracking laser
        laser_msg = LaserOn()
        laser_msg.laser_state = True
        blackboard.main_node.laser_pos_queue.empty()
        blackboard.main_node.logger.info("Setting Laser On")
        blackboard.main_node.laser_state_pub.publish(laser_msg)
        if np.linalg.norm(blackboard.curr_track.pos_wrt_cam) < 5:
            laser_msg = LaserOn()
            laser_msg.laser_state = False
            blackboard.main_node.logger.info("Setting Laser Off")
            blackboard.main_node.laser_state_pub.publish(laser_msg)
            blackboard.main_node.runner_tracker.deactivate(blackboard.curr_track)
            # from remote_pdb import RemotePdb
            # RemotePdb('127.0.0.1', 4444).set_trace()
            return "target_not_reached"
        laser_send_point = blackboard.laser.add_pos(
            blackboard.curr_track.pos_wrt_cam,
            pad=False,
            color=(10, 0, 0),
            intensity=255,
        )
        self.send_frame_ts = time.time()
        self.missing_laser_count = 0
        blackboard.laser.sendFrame()

        targ_reached = self._correct_laser(laser_send_point, blackboard)
        blackboard.main_node.logger.info(f"targ_reached: {targ_reached}")
        if targ_reached:
            return "on_target"
        else:
            blackboard.laser.sendEmpty()

            laser_msg = LaserOn()
            laser_msg.laser_state = False
            blackboard.main_node.logger.info("Setting Laser Off")
            blackboard.main_node.laser_state_pub.publish(laser_msg)

            blackboard.main_node.runner_tracker.deactivate(blackboard.curr_track)

            blackboard.main_node.logger.info("Targets not reached")
            return "target_not_reached"

    def _correct_laser(self, laser_send_point, blackboard):
        """Iteratively move the tracking laser until it is at the same point as the runner."""
        blackboard.curr_track.corrected_point = laser_send_point

        # Check the latest laser position and compare to the expected runner pos
        ts = time.time()
        timestamp, laser_data = blackboard.main_node.laser_pos_queue.get_ts(ts)

        # try:
        if laser_data is not None and self.send_frame_ts < timestamp:
            laser_point = np.array([laser_data[1][0].x, laser_data[1][0].y])
            blackboard.main_node.laser_pos_queue.empty()
        else:
            # except:
            # blackboard.main_node.logger.warning(f"No data for ts: {ts} in {blackboard.main_node.laser_pos_queue.datums}")
            self.missing_laser_count += 1
            if self.missing_laser_count > 20:
                blackboard.main_node.logger.info("Laser missing during state correct")
                return False
            time.sleep(0.01)
            return self._correct_laser(laser_send_point, blackboard)

        runner_point = np.array(blackboard.curr_track.point_list[-1])
        dist = np.linalg.norm(laser_point - runner_point)

        if dist > 10:
            correction = (runner_point - laser_point) / 10
            correction[1] = correction[1] * -1
            new_point = laser_send_point + correction
            blackboard.main_node.logger.debug(
                f"Dist:{dist}, Correction{correction}, Old Point:{laser_send_point}, New Point{new_point}"
            )
            if (
                new_point[0] > 4095
                or new_point[1] > 4095
                or new_point[0] < 0
                or new_point[1] < 0
            ):
                blackboard.main_node.logger.info(
                    "Failed to reach pos, outside of laser window"
                )
                return False

            blackboard.laser.add_point(
                new_point, pad=False, color=(10, 0, 0), intensity=255
            )
            blackboard.laser.sendFrame()
            self.send_frame_ts = time.time()
            self.missing_laser_count = 0
            return self._correct_laser(new_point, blackboard)

        return True


class Burn(State):
    def __init__(self):
        State.__init__(self, outcomes=["runner_removed"])

    def execute(self, blackboard):
        blackboard.main_node.logger.info("Entering State Burn")
        burn_start = time.time()

        # add config to control burn time
        burn_time = 5

        # Turn on burn laser
        laser_msg = LaserOn()
        laser_msg.laser_state = True
        blackboard.main_node.laser_state_pub.publish(laser_msg)
        blackboard.main_node.logger.info("Setting Laser On")
        laser_send_point = blackboard.laser.add_point(
            blackboard.curr_track.corrected_point,
            pad=False,
            color=(0, 0, 15),
            intensity=255,
        )
        blackboard.laser.sendFrame()

        while burn_start + burn_time > time.time():
            time.sleep(0.1)

        blackboard.laser.sendEmpty()

        laser_msg = LaserOn()
        laser_msg.laser_state = False
        blackboard.main_node.laser_state_pub.publish(laser_msg)
        blackboard.main_node.logger.info("Setting Laser Off")

        # Set inactive instead of removing
        blackboard.main_node.runner_tracker.deactivate(blackboard.curr_track)
        return "runner_removed"


class MainNode(Node):
    def __init__(self):
        """State Machine that controls what the system is currently doing"""
        Node.__init__(self, "main_node")

        self.laser_state_pub = self.create_publisher(LaserOn, "laser_on", 1)
        self.runner_point_pub = self.create_publisher(Point, "runner_point", 1)
        self.retrieve_laser_pos = self.create_subscription(
            PosData, "laser_pos_data", self.laser_pos_callback, 5
        )
        self.retrieve_runner_pos = self.create_subscription(
            PosData, "runner_pos_data", self.runner_pos_callback, 5
        )

        self.logger = self.get_logger()

        # move to a tracker implementation
        self.laser_pos_queue = TsQueue(10)
        self.runner_tracker = Tracker(logger=self.logger)

        main_sm = StateMachine(outcomes=["Finished"])

        main_sm.add_state("INIT", Initialize(), transitions={"init_complete": "WAIT"})
        main_sm.add_state("WAIT", ServiceWait(), transitions={"services_up": "ACQ"})
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

        blackboard = Blackboard()
        blackboard.main_node = self

        self.cam_info_received = False

        outcome = main_sm(blackboard)
        self.logger.info(outcome)

    def laser_pos_callback(self, msg):
        self.cam_info_received = True
        ts = msg.timestamp
        if not msg.point_list == [] and not msg.pos_list == []:
            self.laser_pos_queue.add(ts, [msg.pos_list, msg.point_list])

    def runner_pos_callback(self, msg):
        self.cam_info_received = True
        ts = msg.timestamp
        for pos, point in zip(msg.pos_list, msg.point_list):
            self.runner_tracker.add_track([pos.x, pos.y, pos.z], [point.x, point.y])


def main(args=None):
    rclpy.init(args=args)
    node = MainNode()
    node.join_spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
