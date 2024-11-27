import asyncio
from dataclasses import dataclass

import cv2
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from aioros2 import (
    QOS_LATCHED,
    import_node,
    node,
    params,
    result,
    serve_nodes,
    service,
    start,
    subscribe,
    topic,
)
from common_interfaces.srv import SetInt32
from furrow_perceiver_interfaces.msg import PositionResult, State

from . import realsense_stub
from .furrow_tracker import FurrowTracker
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
class PerceiverParams:
    guidance_offset: int = 0


# Executable to call to launch this node (defined in `setup.py`)
@node("furrow_perceiver_node")
class FurrowPerceiverNode:
    perceiver_params = params(PerceiverParams)
    state_topic = topic("~/state", State, QOS_LATCHED)
    debug_img_topic = topic("~/debug_img", Image, qos=qos_profile_sensor_data)
    tracker_result_topic = topic("~/tracker_result", PositionResult, qos=5)

    realsense: realsense_stub.RealsenseStub = import_node(realsense_stub)

    @start
    async def start(self):
        self._tracker = None
        self._annotator = None
        # For converting numpy array to image msg
        self._cv_bridge = CvBridge()

    @subscribe(realsense.depth_image_topic)
    async def on_depth_image(
        self, header, height, width, encoding, is_bigendian, step, data
    ):
        """Takes a realsense depth image, processes it, and emits a debug image"""

        cv_image = self._cv_bridge.imgmsg_to_cv2(
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
        if not self._tracker:
            self._tracker = FurrowTracker()
            self._tracker.init(cv_image)
            self._annotator = FurrowTrackerAnnotator(self._tracker)
            self._tracker.guidance_offset_x = self.perceiver_params.guidance_offset

        # Process image for guidance
        self._tracker.process(cv_image)

        err = self._tracker.get_error()

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
        self._annotator.annotate(depth_colormap)
        m = self._cv_bridge.cv2_to_imgmsg(depth_colormap, "bgr8")
        await self.debug_img_topic(m)

    @service("~/set_guidance_offset", SetInt32)
    async def set_guidance_offset(self, data):
        self._tracker.guidance_offset_x = data
        self._publish_state()
        return result(success=True)

    def _publish_state(self):
        asyncio.create_task(
            self.state_topic(guidance_offset=self._tracker.guidance_offset_x)
        )


def main():
    serve_nodes(FurrowPerceiverNode())


if __name__ == "__main__":
    main()
