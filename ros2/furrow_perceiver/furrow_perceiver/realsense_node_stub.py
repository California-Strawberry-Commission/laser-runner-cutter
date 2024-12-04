from sensor_msgs.msg import Image

from aioros2 import node, topic


@node("realsense_node")
class RealsenseNodeStub:
    depth_image_topic = topic("~/depth/image_rect_raw", Image)
    color_image_topic = topic("~/color/image_raw", Image)
