from sensor_msgs.msg import Image

import aioros2

depth_image_topic = aioros2.topic("~/depth/image_rect_raw", Image)
color_image_topic = aioros2.topic("~/color/image_raw", Image)
