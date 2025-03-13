#include <cv_bridge/cv_bridge.h>

#include <atomic>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "camera_control_cpp/camera/lucid_camera.hpp"
#include "camera_control_interfaces/msg/detection_result.hpp"
#include "camera_control_interfaces/msg/device_state.hpp"
#include "camera_control_interfaces/msg/state.hpp"
#include "camera_control_interfaces/srv/get_frame.hpp"
#include "camera_control_interfaces/srv/get_state.hpp"
#include "camera_control_interfaces/srv/set_exposure.hpp"
#include "camera_control_interfaces/srv/set_gain.hpp"
#include "camera_control_interfaces/srv/set_save_directory.hpp"
#include "common_interfaces/msg/vector2.hpp"
#include "common_interfaces/srv/get_bool.hpp"
#include "rcl_interfaces/msg/log.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_srvs/srv/trigger.hpp"

std::pair<int, int> millisecondsToRosTime(double milliseconds) {
  // ROS timestamps consist of two integers, one for seconds and one for
  // nanoseconds
  int seconds = static_cast<int>(milliseconds / 1000);
  int nanoseconds =
      static_cast<int>((static_cast<int>(milliseconds) % 1000) * 1e6);
  return {seconds, nanoseconds};
}

class CameraControlNode : public rclcpp::Node {
 public:
  explicit CameraControlNode() : Node("camera_control_node") {
    /////////////
    // Parameters
    /////////////
    declare_parameter<int>("camera_index", 0);
    declare_parameter<double>("exposure_us", -1.0);
    declare_parameter<double>("gain_db", -1.0);
    declare_parameter<std::string>("save_dir", "~/runner_cutter/camera");
    declare_parameter<int>("debug_frame_width", 640);
    declare_parameter<double>("debug_video_fps", 30.0);
    declare_parameter<double>("image_capture_interval_secs", 5.0);

    /////////
    // Topics
    /////////
    rclcpp::QoS latchedQos(rclcpp::KeepLast(1));
    latchedQos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
    statePublisher_ = create_publisher<camera_control_interfaces::msg::State>(
        "~/state", latchedQos);
    debugFramePublisher_ = create_publisher<sensor_msgs::msg::Image>(
        "~/debug_frame", rclcpp::SensorDataQoS());
    detectionsPublisher_ =
        create_publisher<camera_control_interfaces::msg::DetectionResult>(
            "~/detections", 5);
    notificationsPublisher_ =
        create_publisher<rcl_interfaces::msg::Log>("~/notifications", 1);

    ///////////
    // Services
    ///////////
    startDeviceService_ = create_service<std_srvs::srv::Trigger>(
        "~/start_device",
        std::bind(&CameraControlNode::onStartDevice, this,
                  std::placeholders::_1, std::placeholders::_2));
    closeDeviceService_ = create_service<std_srvs::srv::Trigger>(
        "~/close_device",
        std::bind(&CameraControlNode::onCloseDevice, this,
                  std::placeholders::_1, std::placeholders::_2));
    hasFramesService_ = create_service<common_interfaces::srv::GetBool>(
        "~/has_frames",
        std::bind(&CameraControlNode::onHasFrames, this, std::placeholders::_1,
                  std::placeholders::_2));
    getFrameService_ = create_service<camera_control_interfaces::srv::GetFrame>(
        "~/get_frame", std::bind(&CameraControlNode::onGetFrame, this,
                                 std::placeholders::_1, std::placeholders::_2));
    setExposureService_ =
        create_service<camera_control_interfaces::srv::SetExposure>(
            "~/set_exposure",
            std::bind(&CameraControlNode::onSetExposure, this,
                      std::placeholders::_1, std::placeholders::_2));
    autoExposureService_ = create_service<std_srvs::srv::Trigger>(
        "~/auto_exposure",
        std::bind(&CameraControlNode::onAutoExposure, this,
                  std::placeholders::_1, std::placeholders::_2));
    setGainService_ = create_service<camera_control_interfaces::srv::SetGain>(
        "~/set_gain", std::bind(&CameraControlNode::onSetGain, this,
                                std::placeholders::_1, std::placeholders::_2));
    autoGainService_ = create_service<std_srvs::srv::Trigger>(
        "~/auto_gain", std::bind(&CameraControlNode::onAutoGain, this,
                                 std::placeholders::_1, std::placeholders::_2));
    setSaveDirectoryService_ =
        create_service<camera_control_interfaces::srv::SetSaveDirectory>(
            "~/set_save_directory",
            std::bind(&CameraControlNode::onSetSaveDirectory, this,
                      std::placeholders::_1, std::placeholders::_2));
    getStateService_ = create_service<camera_control_interfaces::srv::GetState>(
        "~/get_state", std::bind(&CameraControlNode::onGetState, this,
                                 std::placeholders::_1, std::placeholders::_2));

    ///////////////
    // Camera Setup
    ///////////////
    cv::Mat testMat;  // TODO: replace with real mats
    LucidCamera::StateChangeCallback stateChangeCallback =
        [this](LucidCamera::State state) {
          RCLCPP_INFO(this->get_logger(), "Camera state changed");
        };
    camera_ = std::make_shared<LucidCamera>(
        testMat, testMat, testMat, testMat, testMat, testMat, std::nullopt,
        std::nullopt, std::pair<int, int>{2048, 1536}, stateChangeCallback);

    //////////////////
    // Detectors Setup
    //////////////////
  }

 private:
#pragma region Service callback definitions

  void onStartDevice(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (camera_->getState() != LucidCamera::State::DISCONNECTED) {
      response->success = false;
      return;
    }

    LucidCamera::FrameCallback frameCallback =
        [this](std::shared_ptr<LucidFrame> frame) {
          if (!cameraStarted_ ||
              camera_->getState() != LucidCamera::State::STREAMING) {
            return;
          }

          // Update shared frame and notify
          {
            std::lock_guard<std::mutex> lock(frameMutex_);
            currentFrame_ = frame;
          }
          frameEvent_.notify_all();
        };

    detectionThread_ = std::thread([this]() {
      while (true) {
        // Wait for new frame
        std::shared_ptr<LucidFrame> frame;
        {
          std::unique_lock<std::mutex> lock(frameMutex_);
          frameEvent_.wait(lock, [this] { return currentFrame_ != nullptr; });
          frame = currentFrame_;
        }

        // Run detection
        detectionCallback(frame);
      }
    });

    camera_->start(get_parameter("exposure_us").as_double(),
                   get_parameter("gain_db").as_double(), frameCallback);

    cameraStarted_ = true;
    response->success = true;
  }

  void onCloseDevice(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (camera_->getState() == LucidCamera::State::DISCONNECTED) {
      response->success = false;
      return;
    }

    if (detectionThread_.joinable()) {
      detectionThread_.join();
    }

    cameraStarted_ = false;
    camera_->stop();
    {
      std::lock_guard<std::mutex> lock(frameMutex_);
      currentFrame_ = nullptr;
    }

    response->success = true;
  }

  void onHasFrames(
      const std::shared_ptr<common_interfaces::srv::GetBool::Request>,
      std::shared_ptr<common_interfaces::srv::GetBool::Response> response) {
    std::lock_guard<std::mutex> lock(frameMutex_);
    response->data = (currentFrame_ != nullptr);
  }

  void onGetFrame(
      const std::shared_ptr<camera_control_interfaces::srv::GetFrame::Request>,
      std::shared_ptr<camera_control_interfaces::srv::GetFrame::Response>
          response) {
    std::shared_ptr<LucidFrame> frame;
    {
      std::lock_guard<std::mutex> lock(frameMutex_);
      frame = currentFrame_;
    }

    auto colorFrameMsg =
        getColorFrameMsg(frame->getColorFrame(), frame->getTimestampMillis());
    response->color_frame = *colorFrameMsg;
    auto depthFrameMsg =
        getDepthFrameMsg(frame->getDepthFrame(), frame->getTimestampMillis());
    response->depth_frame = *depthFrameMsg;
  }

  void onSetExposure(
      const std::shared_ptr<
          camera_control_interfaces::srv::SetExposure::Request>
          request,
      std::shared_ptr<camera_control_interfaces::srv::SetExposure::Response>
          response) {
    set_parameter(rclcpp::Parameter("exposure_us",
                                    static_cast<double>(request->exposure_us)));
    camera_->setExposureUs(request->exposure_us);
    publishState();
    response->success = true;
  }

  void onAutoExposure(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    set_parameter(rclcpp::Parameter("exposure_us", -1.0));
    camera_->setExposureUs(-1.0);
    publishState();
    response->success = true;
  }

  void onSetGain(
      const std::shared_ptr<camera_control_interfaces::srv::SetGain::Request>
          request,
      std::shared_ptr<camera_control_interfaces::srv::SetGain::Response>
          response) {
    set_parameter(
        rclcpp::Parameter("gain_db", static_cast<double>(request->gain_db)));
    camera_->setGainDb(request->gain_db);
    publishState();
    response->success = true;
  }

  void onAutoGain(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                  std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    set_parameter(rclcpp::Parameter("gain_db", -1.0));
    camera_->setGainDb(-1.0);
    publishState();
    response->success = true;
  }

  void onSetSaveDirectory(
      const std::shared_ptr<
          camera_control_interfaces::srv::SetSaveDirectory::Request>
          request,
      std::shared_ptr<
          camera_control_interfaces::srv::SetSaveDirectory::Response>
          response) {
    set_parameter(rclcpp::Parameter("save_dir", request->save_directory));
    publishState();
    response->success = true;
  }

  void onGetState(
      const std::shared_ptr<camera_control_interfaces::srv::GetState::Request>,
      std::shared_ptr<camera_control_interfaces::srv::GetState::Response>
          response) {
    response->state = *getState();
  }

#pragma endregion

  void detectionCallback(std::shared_ptr<const LucidFrame> frame) {
    RCLCPP_INFO(this->get_logger(), "detectionCallback");

    // Create debug frame as a copy of the color frame
    auto debugFrame{frame->getColorFrame().clone()};

    // Downscale debug_frame using INTER_NEAREST for best performance
    double aspectRatio{static_cast<double>(debugFrame.rows) / debugFrame.cols};
    int newWidth{static_cast<int>(get_parameter("debug_frame_width").as_int())};
    int newHeight{static_cast<int>(std::round(newWidth * aspectRatio))};
    cv::resize(debugFrame, debugFrame, cv::Size(newWidth, newHeight), 0, 0,
               cv::INTER_NEAREST);

    sensor_msgs::msg::Image::SharedPtr debugFrameMsg =
        getColorFrameMsg(debugFrame, frame->getTimestampMillis());
    debugFramePublisher_->publish(*debugFrameMsg);
  }

#pragma region State and notifs publishing

  uint8_t getDeviceState() {
    switch (camera_->getState()) {
      case LucidCamera::State::CONNECTING:
        return camera_control_interfaces::msg::DeviceState::CONNECTING;
      case LucidCamera::State::STREAMING:
        return camera_control_interfaces::msg::DeviceState::STREAMING;
      default:
        return camera_control_interfaces::msg::DeviceState::DISCONNECTED;
    }
  }

  camera_control_interfaces::msg::State::SharedPtr getState() {
    auto msg{std::make_shared<camera_control_interfaces::msg::State>()};
    msg->device_state = getDeviceState();
    msg->exposure_us = camera_->getExposureUs();
    auto exposureUsRange{camera_->getExposureUsRange()};
    msg->exposure_us_range.x = exposureUsRange.first;
    msg->exposure_us_range.y = exposureUsRange.second;
    msg->gain_db = camera_->getGainDb();
    auto gainDbRange{camera_->getGainDbRange()};
    msg->gain_db_range.x = gainDbRange.first;
    msg->gain_db_range.y = gainDbRange.second;
    return msg;
  }

  void publishState() { statePublisher_->publish(*getState()); }

#pragma endregion

#pragma region Message builders

  sensor_msgs::msg::Image::SharedPtr getColorFrameMsg(const cv::Mat& colorFrame,
                                                      double timestampMillis) {
    auto [sec, nanosec]{millisecondsToRosTime(timestampMillis)};
    auto header{std_msgs::msg::Header()};
    header.stamp.sec = sec;
    header.stamp.nanosec = nanosec;
    return cv_bridge::CvImage(header, "rgb8", colorFrame).toImageMsg();
  }

  sensor_msgs::msg::Image::SharedPtr getDepthFrameMsg(const cv::Mat& depthFrame,
                                                      double timestampMillis) {
    auto [sec, nanosec]{millisecondsToRosTime(timestampMillis)};
    auto header{std_msgs::msg::Header()};
    header.stamp.sec = sec;
    header.stamp.nanosec = nanosec;
    return cv_bridge::CvImage(header, "mono16", depthFrame).toImageMsg();
  }

#pragma endregion

  rclcpp::Publisher<camera_control_interfaces::msg::State>::SharedPtr
      statePublisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debugFramePublisher_;
  rclcpp::Publisher<camera_control_interfaces::msg::DetectionResult>::SharedPtr
      detectionsPublisher_;
  rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
      notificationsPublisher_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr startDeviceService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr closeDeviceService_;
  rclcpp::Service<common_interfaces::srv::GetBool>::SharedPtr hasFramesService_;
  rclcpp::Service<camera_control_interfaces::srv::GetFrame>::SharedPtr
      getFrameService_;
  rclcpp::Service<camera_control_interfaces::srv::SetExposure>::SharedPtr
      setExposureService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr autoExposureService_;
  rclcpp::Service<camera_control_interfaces::srv::SetGain>::SharedPtr
      setGainService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr autoGainService_;
  rclcpp::Service<camera_control_interfaces::srv::SetSaveDirectory>::SharedPtr
      setSaveDirectoryService_;
  rclcpp::Service<camera_control_interfaces::srv::GetState>::SharedPtr
      getStateService_;

  std::shared_ptr<LucidCamera> camera_;
  // Used to prevent frame callback from updating the current frame after the
  // camera device has been stopped
  std::atomic<bool> cameraStarted_{false};
  std::shared_ptr<LucidFrame> currentFrame_;
  std::mutex frameMutex_;
  // Used to notify when a new frame is available
  std::condition_variable frameEvent_;
  std::thread detectionThread_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  // MultiThreadedExecutor allows callbacks to run in parallel
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<CameraControlNode>();
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();

  return 0;
}