#include <cv_bridge/cv_bridge.h>
#include <fmt/core.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <atomic>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <optional>
#include <rclcpp_components/register_node_macro.hpp>

#include "camera_control_cpp/camera/lucid_camera.hpp"
#include "camera_control_interfaces/msg/device_state.hpp"
#include "camera_control_interfaces/msg/state.hpp"
#include "camera_control_interfaces/srv/acquire_single_frame.hpp"
#include "camera_control_interfaces/srv/get_state.hpp"
#include "camera_control_interfaces/srv/start_device.hpp"
#include "camera_control_interfaces/srv/start_interval_capture.hpp"
#include "rcl_interfaces/msg/log.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_srvs/srv/trigger.hpp"

std::string expandUser(const std::string& path) {
  if (path.empty() || path[0] != '~') {
    return path;
  }

  const char* home{std::getenv("HOME")};
  if (!home) {
    throw std::runtime_error("HOME environment variable not set");
  }

  return std::string(home) + path.substr(1);
}

std::string formatRosTimestamp(const builtin_interfaces::msg::Time& stamp) {
  rclcpp::Time rosTime(stamp);
  auto sec{static_cast<time_t>(rosTime.seconds())};
  auto nsec{rosTime.nanoseconds() % 1'000'000'000};

  // Format date + time
  std::tm tm;
  localtime_r(&sec, &tm);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y%m%d%H%M%S");

  // Add milliseconds
  oss << std::setw(3) << std::setfill('0') << (nsec / 1'000'000);

  return oss.str();
}

class CameraControlNode : public rclcpp::Node {
 public:
  explicit CameraControlNode(const rclcpp::NodeOptions& options)
      : Node("camera_control_node", options) {
    /////////////
    // Parameters
    /////////////
    declare_parameter<int>("camera_index", 0);
    declare_parameter<double>("exposure_us", -1.0);
    declare_parameter<double>("gain_db", -1.0);
    declare_parameter<std::string>("save_dir", "~/runner_cutter/camera");
    declare_parameter<float>("image_capture_interval_secs", 5.0f);
    paramSubscriber_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    exposureUsParamCallback_ = paramSubscriber_->add_parameter_callback(
        "exposure_us", std::bind(&CameraControlNode::onExposureUsChanged, this,
                                 std::placeholders::_1));
    gainDbParamCallback_ = paramSubscriber_->add_parameter_callback(
        "gain_db", std::bind(&CameraControlNode::onGainDbChanged, this,
                             std::placeholders::_1));

    /////////////
    // Publishers
    /////////////
    // Note: we need to explicitly disable intra-process comms on latched topics
    // as intra-process comms are only allowed with volatile durability
    rclcpp::QoS latchedQos(rclcpp::KeepLast(1));
    latchedQos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
    rclcpp::PublisherOptions intraProcessDisableOpts;
    intraProcessDisableOpts.use_intra_process_comm =
        rclcpp::IntraProcessSetting::Disable;
    statePublisher_ = create_publisher<camera_control_interfaces::msg::State>(
        "~/state", latchedQos, intraProcessDisableOpts);
    notificationsPublisher_ =
        create_publisher<rcl_interfaces::msg::Log>("/notifications", 1);
    colorImagePublisher_ = create_publisher<sensor_msgs::msg::Image>(
        "~/color_image", rclcpp::SensorDataQoS());
    depthXyzPublisher_ = create_publisher<sensor_msgs::msg::Image>(
        "~/depth_xyz", rclcpp::SensorDataQoS());
    depthIntensityPublisher_ = create_publisher<sensor_msgs::msg::Image>(
        "~/depth_intensity", rclcpp::SensorDataQoS());

    ///////////
    // Services
    ///////////
    serviceCallbackGroup_ =
        create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    startDeviceService_ =
        create_service<camera_control_interfaces::srv::StartDevice>(
            "~/start_device",
            std::bind(&CameraControlNode::onStartDevice, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, serviceCallbackGroup_);
    closeDeviceService_ = create_service<std_srvs::srv::Trigger>(
        "~/close_device",
        std::bind(&CameraControlNode::onCloseDevice, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    acquireSingleFrameService_ =
        create_service<camera_control_interfaces::srv::AcquireSingleFrame>(
            "~/acquire_single_frame",
            std::bind(&CameraControlNode::onAcquireSingleFrame, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, serviceCallbackGroup_);
    saveImageService_ = create_service<std_srvs::srv::Trigger>(
        "~/save_image",
        std::bind(&CameraControlNode::onSaveImage, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    startIntervalCaptureService_ =
        create_service<camera_control_interfaces::srv::StartIntervalCapture>(
            "~/start_interval_capture",
            std::bind(&CameraControlNode::onStartIntervalCapture, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, serviceCallbackGroup_);
    stopIntervalCaptureService_ = create_service<std_srvs::srv::Trigger>(
        "~/stop_interval_capture",
        std::bind(&CameraControlNode::onStopIntervalCapture, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    getStateService_ = create_service<camera_control_interfaces::srv::GetState>(
        "~/get_state",
        std::bind(&CameraControlNode::onGetState, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);

    ///////////////
    // Camera Setup
    ///////////////
    LucidCamera::StateChangeCallback stateChangeCallback =
        [this](LucidCamera::State) { publishState(); };
    camera_ = std::make_unique<LucidCamera>(std::nullopt, std::nullopt,
                                            std::pair<int, int>{2048, 1536},
                                            stateChangeCallback);

    // Publish initial state
    publishState();

    // Publish device temperatures at a regular interval
    deviceTemperaturePublishTimer_ = create_wall_timer(
        std::chrono::duration<double>(5.0), [this]() { publishState(); });
  }

 private:
#pragma region Param helpers

  int getParamDacIndex() {
    return static_cast<int>(get_parameter("camera_index").as_int());
  }

  double getParamExposureUs() {
    return get_parameter("exposure_us").as_double();
  }

  double getParamGainDb() { return get_parameter("gain_db").as_double(); }

  std::string getParamSaveDir() {
    return get_parameter("save_dir").as_string();
  }

  float getParamImageCaptureIntervalSecs() {
    return static_cast<float>(
        get_parameter("image_capture_interval_secs").as_double());
  }

  void onExposureUsChanged(const rclcpp::Parameter& param) {
    camera_->setExposureUs(param.as_double());
  }

  void onGainDbChanged(const rclcpp::Parameter& param) {
    camera_->setGainDb(param.as_double());
  }

#pragma endregion

#pragma region Service callback definitions

  void onStartDevice(
      const std::shared_ptr<
          camera_control_interfaces::srv::StartDevice::Request>
          request,
      std::shared_ptr<camera_control_interfaces::srv::StartDevice::Response>
          response) {
    if (camera_->getState() != LucidCamera::State::DISCONNECTED) {
      response->success = false;
      return;
    }

    LucidCamera::FrameCallback frameCallback = [this](Frame frame) {
      if (!cameraStarted_ ||
          camera_->getState() != LucidCamera::State::STREAMING) {
        return;
      }

      {
        std::lock_guard<std::mutex> lock(lastColorImageMutex_);
        if (frame.colorImage) {
          lastColorImage_ = *frame.colorImage;
        } else {
          lastColorImage_.reset();
        }
      }

      // Publish frames via zero-copy intra-process. Note that the message needs
      // to be a unique_ptr.
      colorImagePublisher_->publish(std::move(frame.colorImage));
      depthXyzPublisher_->publish(std::move(frame.depthXyz));
      depthIntensityPublisher_->publish(std::move(frame.depthIntensity));
    };

    camera_->start(static_cast<LucidCamera::CaptureMode>(request->capture_mode),
                   getParamExposureUs(), getParamGainDb(), frameCallback);

    cameraStarted_ = true;
    response->success = true;
  }

  void onCloseDevice(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (camera_->getState() == LucidCamera::State::DISCONNECTED) {
      response->success = false;
      response->message = "Camera was not started";
      return;
    }

    cameraStarted_ = false;
    camera_->stop();
    {
      std::lock_guard<std::mutex> lock(lastColorImageMutex_);
      lastColorImage_.reset();
    }

    response->success = true;
  }

  void onAcquireSingleFrame(
      const std::shared_ptr<
          camera_control_interfaces::srv::AcquireSingleFrame::Request>,
      std::shared_ptr<
          camera_control_interfaces::srv::AcquireSingleFrame::Response>
          response) {
    std::optional<Frame> frameOpt{camera_->getNextFrame()};
    if (!frameOpt) {
      publishNotification("Failed to acquire frame",
                          rclcpp::Logger::Level::Error);
      return;
    }
    Frame frame{std::move(frameOpt.value())};

    publishNotification("Successfully acquired frame");

    // Demosaic color image (which is BayerRG8)
    cv::Mat raw(frame.colorImage->height, frame.colorImage->width, CV_8UC1,
                const_cast<uint8_t*>(frame.colorImage->data.data()),
                frame.colorImage->step);
    cv::Mat rgb;
    cv::cvtColor(raw, rgb, cv::COLOR_BayerRG2RGB);

    // Compress to JPEG and write to response
    sensor_msgs::msg::CompressedImage compressedImgMsg;
    compressedImgMsg.header = frame.colorImage->header;
    compressedImgMsg.format = "jpeg";
    if (!cv::imencode(".jpg", rgb, compressedImgMsg.data,
                      {cv::IMWRITE_JPEG_QUALITY, 90})) {
      throw std::runtime_error("imencode failed (bayer->jpeg)");
    }

    response->preview_image = compressedImgMsg;
  }

  void onSaveImage(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    auto res{saveImage()};
    response->success = res.has_value();
  }

  void onStartIntervalCapture(
      const std::shared_ptr<
          camera_control_interfaces::srv::StartIntervalCapture::Request>
          request,
      std::shared_ptr<
          camera_control_interfaces::srv::StartIntervalCapture::Response>
          response) {
    set_parameter(
        rclcpp::Parameter("image_capture_interval_secs",
                          static_cast<double>(request->interval_secs)));

    if (intervalCaptureTimer_) {
      intervalCaptureTimer_.reset();
    }
    intervalCaptureTimer_ =
        create_wall_timer(std::chrono::duration<double>(request->interval_secs),
                          [this]() { saveImage(); });

    publishState();
    publishNotification("Started interval capture with " +
                        std::to_string(request->interval_secs) + "s interval");
    response->success = true;
  }

  void onStopIntervalCapture(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!intervalCaptureTimer_) {
      response->success = false;
      response->message = "Interval capture was not started";
      return;
    }

    intervalCaptureTimer_.reset();
    publishState();
    publishNotification("Stopped interval capture");
    response->success = true;
  }

  void onGetState(
      const std::shared_ptr<camera_control_interfaces::srv::GetState::Request>,
      std::shared_ptr<camera_control_interfaces::srv::GetState::Response>
          response) {
    response->state = std::move(*getStateMsg());
  }

#pragma endregion

  std::optional<std::string> saveImage() {
    std::optional<sensor_msgs::msg::Image> colorImageOpt;
    {
      std::lock_guard<std::mutex> lock(lastColorImageMutex_);
      if (!lastColorImage_) {
        return std::nullopt;
      }
      colorImageOpt = lastColorImage_;
    }
    auto colorImage{colorImageOpt.value()};

    // Create the save directory if it doesn't exist
    std::string saveDir{getParamSaveDir()};
    saveDir = expandUser(saveDir);
    std::filesystem::create_directories(saveDir);

    // Generate the image file name and path
    std::string filepath{fmt::format(
        "{}/{}.jpg", saveDir, formatRosTimestamp(colorImage.header.stamp))};

    // Demosaic color image (which is BayerRG8) and save the image
    cv::Mat raw(colorImage.height, colorImage.width, CV_8UC1,
                const_cast<uint8_t*>(colorImage.data.data()), colorImage.step);
    cv::Mat rgb;
    cv::cvtColor(raw, rgb, cv::COLOR_BayerRG2RGB);
    cv::imwrite(filepath, rgb);

    publishNotification("Saved image: " + filepath);

    return filepath;
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

  camera_control_interfaces::msg::State::UniquePtr getStateMsg() {
    // TODO: remove unused fields from State once migration to C++ is complete.
    // Note that we should use this node's params for exposure_us, gain_db,
    // save_dir, and image_capture_interval_secs instead of duplicating those in
    // State
    auto msg{std::make_unique<camera_control_interfaces::msg::State>()};
    msg->device_state = getDeviceState();
    msg->interval_capture_active = (intervalCaptureTimer_ != nullptr);
    auto exposureUsRange{camera_->getExposureUsRange()};
    msg->exposure_us_range.x = exposureUsRange.first;
    msg->exposure_us_range.y = exposureUsRange.second;
    auto gainDbRange{camera_->getGainDbRange()};
    msg->gain_db_range.x = gainDbRange.first;
    msg->gain_db_range.y = gainDbRange.second;
    msg->color_device_temperature =
        static_cast<float>(camera_->getColorDeviceTemperature());
    msg->depth_device_temperature =
        static_cast<float>(camera_->getDepthDeviceTemperature());
    return msg;
  }

  void publishState() { statePublisher_->publish(std::move(getStateMsg())); }

  void publishNotification(
      const std::string& msg,
      rclcpp::Logger::Level level = rclcpp::Logger::Level::Info) {
    uint8_t logMsgLevel{0};
    switch (level) {
      case rclcpp::Logger::Level::Debug:
        RCLCPP_DEBUG(get_logger(), msg.c_str());
        logMsgLevel = rcl_interfaces::msg::Log::DEBUG;
        break;
      case rclcpp::Logger::Level::Info:
        RCLCPP_INFO(get_logger(), msg.c_str());
        logMsgLevel = rcl_interfaces::msg::Log::INFO;
        break;
      case rclcpp::Logger::Level::Warn:
        RCLCPP_WARN(get_logger(), msg.c_str());
        logMsgLevel = rcl_interfaces::msg::Log::WARN;
        break;
      case rclcpp::Logger::Level::Error:
        RCLCPP_ERROR(get_logger(), msg.c_str());
        logMsgLevel = rcl_interfaces::msg::Log::ERROR;
        break;
      case rclcpp::Logger::Level::Fatal:
        RCLCPP_FATAL(get_logger(), msg.c_str());
        logMsgLevel = rcl_interfaces::msg::Log::FATAL;
        break;
      default:
        RCLCPP_ERROR(get_logger(), "Unknown log level: %s", msg.c_str());
        return;
    }

    auto logMsg{rcl_interfaces::msg::Log()};
    logMsg.stamp = rclcpp::Clock().now();
    logMsg.level = logMsgLevel;
    logMsg.msg = msg;
    notificationsPublisher_->publish(std::move(logMsg));
  }

#pragma endregion

  std::shared_ptr<rclcpp::ParameterEventHandler> paramSubscriber_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> exposureUsParamCallback_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> gainDbParamCallback_;
  rclcpp::Publisher<camera_control_interfaces::msg::State>::SharedPtr
      statePublisher_;
  rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
      notificationsPublisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr colorImagePublisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depthXyzPublisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr
      depthIntensityPublisher_;
  rclcpp::CallbackGroup::SharedPtr serviceCallbackGroup_;
  rclcpp::Service<camera_control_interfaces::srv::StartDevice>::SharedPtr
      startDeviceService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr closeDeviceService_;
  rclcpp::Service<camera_control_interfaces::srv::AcquireSingleFrame>::SharedPtr
      acquireSingleFrameService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr saveImageService_;
  rclcpp::Service<camera_control_interfaces::srv::StartIntervalCapture>::
      SharedPtr startIntervalCaptureService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr
      stopIntervalCaptureService_;
  rclcpp::Service<camera_control_interfaces::srv::GetState>::SharedPtr
      getStateService_;
  rclcpp::TimerBase::SharedPtr intervalCaptureTimer_;
  rclcpp::TimerBase::SharedPtr deviceTemperaturePublishTimer_;

  std::unique_ptr<LucidCamera> camera_;
  // Used to prevent frame callback from updating the current frame after the
  // camera device has been stopped
  std::atomic<bool> cameraStarted_{false};
  std::mutex lastColorImageMutex_;
  std::optional<sensor_msgs::msg::Image> lastColorImage_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(CameraControlNode)