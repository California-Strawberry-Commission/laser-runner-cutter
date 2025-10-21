#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

class TestImagePublisherNode : public rclcpp::Node {
 public:
  explicit TestImagePublisherNode(const rclcpp::NodeOptions& options)
      : Node("test_image_publisher_node", options),
        imageFile_(declare_parameter<std::string>("image_file", "")),
        fps_(declare_parameter<double>("fps", 10.0)),
        frameId_(declare_parameter<std::string>("frame_id", "test_frame")) {
    using namespace std::chrono_literals;

    imagePub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "image_raw", rclcpp::SensorDataQoS());
    cameraInfoPub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
        "camera_info", rclcpp::SensorDataQoS());
    timer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(1.0 / fps_)),
        std::bind(&TestImagePublisherNode::onTimer, this));

    img_ = cv::imread(imageFile_);

    RCLCPP_INFO(get_logger(),
                "Started test image publisher for %s @ %.1f FPS on topic '%s' "
                "with camera info topic '%s'",
                imageFile_.c_str(), fps_, imagePub_->get_topic_name(),
                cameraInfoPub_->get_topic_name());
  }

 private:
  void onTimer() {
    sensor_msgs::msg::Image::UniquePtr imageMsg(new sensor_msgs::msg::Image());
    imageMsg->header.frame_id = frameId_;
    imageMsg->header.stamp = now();
    imageMsg->height = static_cast<uint32_t>(img_.rows);
    imageMsg->width = static_cast<uint32_t>(img_.cols);
    imageMsg->encoding = "bgr8";
    imageMsg->is_bigendian = false;
    imageMsg->step =
        static_cast<sensor_msgs::msg::Image::_step_type>(img_.step);
    imageMsg->data.assign(img_.datastart, img_.dataend);

    sensor_msgs::msg::CameraInfo::UniquePtr cameraInfoMsg(
        new sensor_msgs::msg::CameraInfo());
    cameraInfoMsg->header =
        imageMsg->header;  // IMPORTANT: same timestamp & frame_id
    cameraInfoMsg->width = imageMsg->width;
    cameraInfoMsg->height = imageMsg->height;
    cameraInfoMsg->distortion_model = "plumb_bob";
    cameraInfoMsg->d = {0.0, 0.0, 0.0, 0.0, 0.0};
    // Simple pinhole intrinsics
    double fx = 800.0;
    double fy = 800.0;
    double cx = imageMsg->width / 2.0;
    double cy = imageMsg->height / 2.0;
    cameraInfoMsg->k = {fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
    cameraInfoMsg->r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    cameraInfoMsg->p = {fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0};

    RCLCPP_DEBUG(get_logger(), "Publishing image: %dx%d", imageMsg->width,
                 imageMsg->height);

    imagePub_->publish(std::move(imageMsg));
    cameraInfoPub_->publish(std::move(cameraInfoMsg));
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr imagePub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cameraInfoPub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::string imageFile_;
  double fps_;
  std::string frameId_;
  cv::Mat img_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(TestImagePublisherNode)
