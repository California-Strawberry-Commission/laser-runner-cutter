import asyncio
from typing import AsyncGenerator
from .furrow_tracker import FurrowTracker
from common_interfaces.srv import SetInt32
from furrow_perceiver_interfaces.msg import State, PositionResult
from furrow_perceiver_interfaces.srv import GetState
from dataclasses import dataclass
import cv2

from aioros2 import (
    timer,
    service,
    action,
    serve_nodes,
    result,
    feedback,
    subscribe,
    topic,
    import_node,
    params,
    node,
    subscribe_param,
    param,
    start,
    QOS_LATCHED
)
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from common_interfaces.msg import Vector2
from . import realsense_stub
from sensor_msgs.msg import Image
from .furrow_tracker_annotator import FurrowTrackerAnnotator

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


# ros2 run furrow_perceiver furrow_perceiver --ros-args -p "realsense.name:=camera1" -p "realsense.ns:=/"
# ros2 run realsense2_camera realsense2_camera_node --ros-args -r __node:=camera1


@dataclass
class PerceiverNodeParams:
    guidance_offset: int = 0


# Executable to call to launch this node (defined in `setup.py`)
@node("furrow_perceiver")
class FurrowPerceiverNode:
    p = params(PerceiverNodeParams)
    state = topic("~/state", State, QOS_LATCHED)
    cvb = CvBridge()

    realsense: realsense_stub.RealsenseStub = import_node(realsense_stub)

    debug_img_topic = topic("~/debug_img", Image)
    tracker_result_topic = topic("~/tracker_result", PositionResult)

    tracker = None
    annotator = None

    async def emit_state(self):
        asyncio.create_task(self.state(
            guidance_offset=self.tracker.guidance_offset_x
        ))

    @subscribe(realsense.depth_image_topic)
    async def on_depth_image(
        self, header, height, width, encoding, is_bigendian, step, data
    ):
        """Takes a realsense depth image, processes it, and emits a debug image"""

        cv_image = self.cvb.imgmsg_to_cv2(
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
        if not self.tracker:
            self.tracker = FurrowTracker()
            self.tracker.init(cv_image)
            self.annotator = FurrowTrackerAnnotator(self.tracker)
            self.tracker.guidance_offset_x = self.p.guidance_offset

        # Process image for guidance
        self.tracker.process(cv_image)
        
        err = self.tracker.get_error()
        
        if err is None:
            await self.tracker_result_topic(
                linear_deviation=0.0, heading=0.0, is_valid=False
            )
        else:
            await self.tracker_result_topic(
                linear_deviation=float(err), heading=0.0, is_valid=True
            )

        # Create & publish debug image
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(cv_image, alpha=0.09), cv2.COLORMAP_JET
        )
        self.annotator.annotate(depth_colormap)
        m = self.cvb.cv2_to_imgmsg(depth_colormap, "bgr8")
        await self.debug_img_topic(m)

    @service("~/set_guidance_offset", SetInt32)
    async def set_guidance_offset(self, data):
        self.tracker.guidance_offset_x = data
        await self.emit_state()
        return{}
        
# Boilerplate below here.
def main():
    serve_nodes(FurrowPerceiverNode())


if __name__ == "__main__":
    main()
