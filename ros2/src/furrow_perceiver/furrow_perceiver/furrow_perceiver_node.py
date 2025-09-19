import logging
from dataclasses import dataclass
from typing import Optional

import cv2
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

import aioros2
import furrow_perceiver.realsense_node_stub as realsense_node_stub
from common_interfaces.srv import SetInt32
from furrow_perceiver.furrow_tracker import FurrowTracker
from furrow_perceiver.furrow_tracker_annotator import FurrowTrackerAnnotator
from furrow_perceiver_interfaces.msg import PositionResult, State

#                              /\                         X
#  #####________________________________________________#####
#  #####________________________________________________#####
#  #####                                                #####
#  #####                                                #####
#  #####                                                #####
#  || ||                                                || ||
#  || ||                                                || ||
#  || ||                                                || ||
#  || ||                                         |------|| ||------|
#  || ||                                                || ||
#  || ||                                                || ||
#  || ||                                                || ||
#  #####                                                #####
#  #####                                                #####
#  #####________________________________________________#####
#  #####________________________________________________#####
#  #####                                                #####


@dataclass
class PerceiverParams:
    guidance_offset: int = 0


perceiver_params = aioros2.params(PerceiverParams)
state_topic = aioros2.topic("~/state", State, aioros2.QOS_LATCHED)
debug_img_topic = aioros2.topic("~/debug_img", Image, qos=qos_profile_sensor_data)
tracker_result_topic = aioros2.topic("~/tracker_result", PositionResult, qos=5)
realsense_node: realsense_node_stub = aioros2.use(realsense_node_stub)


class SharedState:
    logger: Optional[logging.Logger] = None
    tracker: Optional[FurrowTracker] = None
    annotator: Optional[FurrowTrackerAnnotator] = None
    # For converting numpy array to image msg
    cv_bridge = CvBridge()


shared_state = SharedState()


@aioros2.start
async def start(node):
    shared_state.logger = node.get_logger()


@aioros2.subscribe(realsense_node.depth_image_topic)
async def on_depth_image(
    node, header, height, width, encoding, is_bigendian, step, data
):
    """Takes a realsense depth image, processes it, and emits a debug image"""

    cv_image = shared_state.cv_bridge.imgmsg_to_cv2(
        Image(
            header=header,
            height=height,
            width=width,
            encoding=encoding,
            is_bigendian=is_bigendian,
            step=step,
            data=data,
        )
    )

    # Initialize tracker on first image
    if not shared_state.tracker:
        shared_state.tracker = FurrowTracker()
        shared_state.tracker.init(cv_image)
        shared_state.annotator = FurrowTrackerAnnotator(shared_state.tracker)
        shared_state.tracker.guidance_offset_x = perceiver_params.guidance_offset

    # Process image for guidance
    shared_state.tracker.process(cv_image)

    err = shared_state.tracker.get_error()

    if err is None:
        tracker_result_topic.publish(linear_deviation=0.0, heading=0.0, is_valid=False)
    else:
        tracker_result_topic.publish(
            linear_deviation=float(err), heading=0.0, is_valid=True
        )

    # Create & publish debug image
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(cv_image, alpha=0.09), cv2.COLORMAP_JET
    )
    shared_state.annotator.annotate(depth_colormap)
    msg = shared_state.cv_bridge.cv2_to_imgmsg(depth_colormap, "bgr8")
    debug_img_topic.publish(msg)


@aioros2.service("~/set_guidance_offset", SetInt32)
async def set_guidance_offset(node, data):
    shared_state.tracker.guidance_offset_x = data
    _publish_state()
    return {"success": True}


def _publish_state():
    state_topic.publish(guidance_offset=shared_state.tracker.guidance_offset_x)


def main():
    aioros2.run()


if __name__ == "__main__":
    main()
