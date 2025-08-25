import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import qos_profile_sensor_data


class VideoPublisher(Node):
    def __init__(self):
        super().__init__("video_publisher")
        self.publisher_ = self.create_publisher(
            Image, "~/test_video", qos_profile_sensor_data
        )
        self.bridge = CvBridge()

        # Path to your video file
        video_path = "/home/genki/Desktop/video.mp4"
        self.video_capture_ = cv2.VideoCapture(video_path)

        if not self.video_capture_.isOpened():
            self.get_logger().error(f"Failed to open video: {video_path}")
            return

        timer_period = 1.0 / self.video_capture_.get(
            cv2.CAP_PROP_FPS
        )  # Based on video FPS
        self.timer_ = self.create_timer(timer_period, self.publish_frame)

    def publish_frame(self):
        ret, frame = self.video_capture_.read()
        if not ret:
            self.get_logger().info("Looping video...")
            self.video_capture_.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    rclpy.spin(node)
    node.video_capture_.release()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
