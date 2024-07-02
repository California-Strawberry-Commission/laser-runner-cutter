import asyncio
from typing import AsyncGenerator
from .furrow_tracker import FurrowTracker
from furrow_perceiver_interfaces.msg import State
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
    pass


# Executable to call to launch this node (defined in `setup.py`)
@node("furrow_perceiver")
class FurrowPerceiverNode:
    p = params(PerceiverNodeParams)
    state = topic("~/state", State)
    cvb = CvBridge()

    realsense: realsense_stub.RealsenseStub = import_node(realsense_stub)

    debug_img_topic = topic("~/debug_img", Image)
    
    tracker = None
    annotator = None
    @start
    async def s(self):
        self.log("STARTING FURROW TRACKER")
        
    # TODO: Allow these annotations using parameters
    # @subscribe(amiga_params.depth_topic)
    # @subscribe(amiga_params.rs_name + "/image")
    @subscribe(realsense.depth_image_topic)
    async def on_depth_image(self, header, height, width, encoding, is_bigendian, step, data):
        self.log("GOT IMAGE!!")
        cv_image = self.cvb.imgmsg_to_cv2(Image(header=header, height=height, width=width, encoding=encoding, is_bigendian=is_bigendian, step=step, data=data))
        print(cv_image[100][100])
        
        if not self.tracker:
            self.tracker = FurrowTracker()
            self.tracker.init(cv_image)
            self.annotator = FurrowTrackerAnnotator(self.tracker)
        
        
        self.tracker.process(cv_image)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(cv_image, alpha=0.09), cv2.COLORMAP_JET
        )
        self.annotator.annotate(depth_colormap)
        
        print(self.tracker.get_error())

        m = self.cvb.cv2_to_imgmsg(depth_colormap)
        await self.debug_img_topic(m)
    
    @service("get_state", GetState)
    async def get_state(self):
        return result(state=State(fps=0, camera_connected=False))
    
# Boilerplate below here.
def main():
    serve_nodes(FurrowPerceiverNode())


if __name__ == "__main__":
    main()
