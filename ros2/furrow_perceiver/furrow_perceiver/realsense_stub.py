from aioros2 import topic, node
from sensor_msgs.msg import Image

@node("realsense_node")
class RealsenseStub:
    depth_image_topic = topic("~/depth/image_rect_raw", Image)
    color_image_topic = topic("~/color/image_raw", Image)
