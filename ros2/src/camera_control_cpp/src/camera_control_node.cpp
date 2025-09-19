#include <cv_bridge/cv_bridge.h>
#include <fmt/core.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <atomic>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "camera_control_cpp/camera/calibration.hpp"
#include "camera_control_cpp/camera/lucid_camera.hpp"
#include "camera_control_interfaces/msg/device_state.hpp"
#include "camera_control_interfaces/msg/state.hpp"
#include "camera_control_interfaces/srv/acquire_single_frame.hpp"
#include "camera_control_interfaces/srv/get_frame.hpp"
#include "camera_control_interfaces/srv/get_state.hpp"
#include "camera_control_interfaces/srv/start_device.hpp"
#include "camera_control_interfaces/srv/start_interval_capture.hpp"
#include "common_interfaces/srv/get_bool.hpp"
#include "rcl_interfaces/msg/log.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_srvs/srv/trigger.hpp"

std::pair<int, int> millisecondsToRosTime(double milliseconds) {
  // ROS timestamps consist of two integers, one for seconds and one for
  // nanoseconds
  int seconds{static_cast<int>(milliseconds / 1000)};
  int nanoseconds{
      static_cast<int>((static_cast<int>(milliseconds) % 1000) * 1e6)};
  return {seconds, nanoseconds};
}

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

std::string getCurrentTimeString() {
  // Get the current timestamp and format it as a string
  auto now{std::chrono::system_clock::now()};
  auto timestamp{std::chrono::system_clock::to_time_t(now)};
  std::stringstream datetimeString;
  datetimeString << std::put_time(std::localtime(&timestamp), "%Y%m%d%H%M%S");
  auto ms{std::chrono::duration_cast<std::chrono::milliseconds>(
              now.time_since_epoch()) %
          1000};
  datetimeString << std::setw(3) << std::setfill('0') << ms.count();
  return datetimeString.str();
}

class CameraControlNode : public rclcpp::Node {
 public:
  explicit CameraControlNode() : Node("camera_control_node") {
    /////////////
    // Parameters
    /////////////
    declare_parameter<int>("camera_index", 0);
    declare_parameter<std::string>(
        "calibration_id",
        "1c0faf4b115d1c0faf4d17ce");  // calibration ID is <Triton MAC><Helios
                                      // MAC>
    declare_parameter<double>("exposure_us", -1.0);
    declare_parameter<double>("gain_db", -1.0);
    declare_parameter<std::string>("save_dir", "~/runner_cutter/camera");
    declare_parameter<int>("debug_frame_width", 640);
    declare_parameter<float>("debug_video_fps", 30.0f);
    declare_parameter<float>("image_capture_interval_secs", 5.0f);
    paramSubscriber_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    exposureUsParamCallback_ = paramSubscriber_->add_parameter_callback(
        "exposure_us", std::bind(&CameraControlNode::onExposureUsChanged, this,
                                 std::placeholders::_1));
    gainDbParamCallback_ = paramSubscriber_->add_parameter_callback(
        "gain_db", std::bind(&CameraControlNode::onGainDbChanged, this,
                             std::placeholders::_1));

    /////////
    // Topics
    /////////
    rclcpp::QoS latchedQos(rclcpp::KeepLast(1));
    latchedQos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
    statePublisher_ = create_publisher<camera_control_interfaces::msg::State>(
        "~/state", latchedQos);
    // TODO: move debug frame topic publishing to detection node
    debugFramePublisher_ = create_publisher<sensor_msgs::msg::Image>(
        "~/debug_frame", rclcpp::SensorDataQoS());
    notificationsPublisher_ =
        create_publisher<rcl_interfaces::msg::Log>("/notifications", 1);

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
    hasFramesService_ = create_service<common_interfaces::srv::GetBool>(
        "~/has_frames",
        std::bind(&CameraControlNode::onHasFrames, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    getFrameService_ = create_service<camera_control_interfaces::srv::GetFrame>(
        "~/get_frame",
        std::bind(&CameraControlNode::onGetFrame, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    acquireSingleFrameService_ =
        create_service<camera_control_interfaces::srv::AcquireSingleFrame>(
            "~/acquire_single_frame",
            std::bind(&CameraControlNode::onAcquireSingleFrame, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, serviceCallbackGroup_);
    startRecordingVideoService_ = create_service<std_srvs::srv::Trigger>(
        "~/start_recording_video",
        std::bind(&CameraControlNode::onStartRecordingVideo, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    stopRecordingVideoService_ = create_service<std_srvs::srv::Trigger>(
        "~/stop_recording_video",
        std::bind(&CameraControlNode::onStopRecordingVideo, this,
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
    auto calibrationParams{readCalibrationParams(getParamCalibrationId())};
    LucidCamera::StateChangeCallback stateChangeCallback =
        [this](LucidCamera::State) { publishState(); };
    camera_ = std::make_shared<LucidCamera>(
        calibrationParams.tritonIntrinsicMatrix,
        calibrationParams.tritonDistCoeffs,
        calibrationParams.heliosIntrinsicMatrix,
        calibrationParams.heliosDistCoeffs,
        calibrationParams.xyzToTritonExtrinsicMatrix,
        calibrationParams.xyzToHeliosExtrinsicMatrix, std::nullopt,
        std::nullopt, std::pair<int, int>{2048, 1536}, stateChangeCallback);

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

  std::string getParamCalibrationId() {
    return get_parameter("calibration_id").as_string();
  }

  double getParamExposureUs() {
    return get_parameter("exposure_us").as_double();
  }

  double getParamGainDb() { return get_parameter("gain_db").as_double(); }

  std::string getParamSaveDir() {
    return get_parameter("save_dir").as_string();
  }

  int getParamDebugFrameWidth() {
    return static_cast<int>(get_parameter("debug_frame_width").as_int());
  }

  float getParamDebugVideoFps() {
    return static_cast<float>(get_parameter("debug_video_fps").as_double());
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

    LucidCamera::FrameCallback frameCallback =
        [this](std::shared_ptr<LucidFrame> frame) {
          if (!cameraStarted_ ||
              camera_->getState() != LucidCamera::State::STREAMING) {
            return;
          }

          {
            std::lock_guard<std::mutex> lock(frameMutex_);
            currentFrame_ = frame;
          }

          // Create debug frame as a copy of the color frame
          auto debugFrame{frame->getColorFrame().clone()};

          // Downscale debug_frame using INTER_NEAREST for best performance
          double aspectRatio{static_cast<double>(debugFrame.rows) /
                             debugFrame.cols};
          int newWidth{getParamDebugFrameWidth()};
          int newHeight{static_cast<int>(std::round(newWidth * aspectRatio))};
          cv::resize(debugFrame, debugFrame, cv::Size(newWidth, newHeight), 0,
                     0, cv::INTER_NEAREST);

          {
            std::lock_guard<std::mutex> lock(debugFrameMutex_);
            currentDebugFrame_ = debugFrame;
          }

          sensor_msgs::msg::Image::SharedPtr debugFrameMsg{
              getColorImageMsg(debugFrame, frame->getTimestampMillis())};
          debugFramePublisher_->publish(*debugFrameMsg);
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
      if (currentFrame_ == nullptr) {
        return;
      }
      frame = currentFrame_;
    }

    auto colorImageMsg{
        getColorImageMsg(frame->getColorFrame(), frame->getTimestampMillis())};
    response->color_frame = *colorImageMsg;
    auto depthImageMsg{
        getDepthImageMsg(frame->getDepthFrame(), frame->getTimestampMillis())};
    response->depth_frame = *depthImageMsg;
  }

  void onAcquireSingleFrame(
      const std::shared_ptr<
          camera_control_interfaces::srv::AcquireSingleFrame::Request>,
      std::shared_ptr<
          camera_control_interfaces::srv::AcquireSingleFrame::Response>
          response) {
    std::optional<LucidFrame> frame{camera_->getNextFrame()};
    if (!frame) {
      publishNotification("Failed to acquire frame",
                          rclcpp::Logger::Level::Error);
      return;
    }

    publishNotification("Successfully acquired frame");
    auto colorImageMsg{getColorCompressedImageMsg(frame->getColorFrame(),
                                                  frame->getTimestampMillis())};
    response->preview_image = *colorImageMsg;
  }

  void onStartRecordingVideo(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (videoRecordingTimer_ && videoRecordingTimer_->is_canceled() == false) {
      videoRecordingTimer_->cancel();
      videoRecordingTimer_.reset();
      std::lock_guard<std::mutex> lock(videoWriterMutex_);
      if (videoWriter_.isOpened()) {
        videoWriter_.release();
      }
    }

    float fps{getParamDebugVideoFps()};
    if (fps <= 0.0f) {
      response->success = false;
      response->message = "Invalid FPS value";
      return;
    }

    videoRecordingTimer_ =
        create_wall_timer(std::chrono::duration<double>(1.0 / fps),
                          [this]() { writeVideoFrame(); });

    publishState();
    response->success = true;
  }

  void onStopRecordingVideo(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!videoRecordingTimer_) {
      response->success = false;
      response->message = "Video recording was not active";
      return;
    }

    videoRecordingTimer_->cancel();
    videoRecordingTimer_.reset();
    {
      std::lock_guard<std::mutex> lock(videoWriterMutex_);
      if (videoWriter_.isOpened()) {
        videoWriter_.release();
      }
    }
    publishState();
    publishNotification("Stopped recording video");
    response->success = true;
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
    std::shared_ptr<LucidFrame> frame;
    {
      std::lock_guard<std::mutex> lock(frameMutex_);
      if (currentFrame_ == nullptr) {
        return std::nullopt;
      }
      frame = currentFrame_;
    }

    // Create the save directory if it doesn't exist
    std::string saveDir{getParamSaveDir()};
    saveDir = expandUser(saveDir);
    std::filesystem::create_directories(saveDir);

    // Generate the image file name and path
    std::string filepath{
        fmt::format("{}/{}.png", saveDir, getCurrentTimeString())};

    // Convert the frame color format (if needed) and save the image
    cv::Mat colorFrameBgr;
    cv::cvtColor(frame->getColorFrame(), colorFrameBgr, cv::COLOR_RGB2BGR);
    cv::imwrite(filepath, colorFrameBgr);

    publishNotification("Saved image: " + filepath);

    return filepath;
  }

  void writeVideoFrame() {
    std::lock_guard<std::mutex> lock(videoWriterMutex_);

    cv::Mat frame;
    {
      std::lock_guard<std::mutex> lock(debugFrameMutex_);
      if (currentDebugFrame_.empty()) {
        return;
      }
      frame = currentDebugFrame_;
    }

    if (!videoWriter_.isOpened()) {
      // Create the save directory if it doesn't exist
      std::string saveDir{getParamSaveDir()};
      saveDir = expandUser(saveDir);
      std::filesystem::create_directories(saveDir);

      // Generate the video file name and path
      std::string filepath{
          fmt::format("{}/{}.avi", saveDir, getCurrentTimeString())};

      int width{frame.cols};
      int height{frame.rows};
      videoWriter_.open(filepath, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                        getParamDebugVideoFps(), cv::Size(width, height));

      if (!videoWriter_.isOpened()) {
        RCLCPP_ERROR(get_logger(), "Failed to open video writer.");
        return;
      }

      publishNotification("Started recording video: " + filepath);
    }

    videoWriter_.write(frame);
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
    msg->recording_video = (videoRecordingTimer_ != nullptr);
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

    // Get current time in milliseconds
    double timestampMillis{
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count()) /
        1000.0};
    auto [sec, nanosec]{millisecondsToRosTime(timestampMillis)};
    auto logMsg{rcl_interfaces::msg::Log()};
    logMsg.stamp.sec = sec;
    logMsg.stamp.nanosec = nanosec;
    logMsg.level = logMsgLevel;
    logMsg.msg = msg;
    notificationsPublisher_->publish(std::move(logMsg));
  }

#pragma endregion

#pragma region Message builders

  sensor_msgs::msg::Image::SharedPtr getColorImageMsg(const cv::Mat& colorFrame,
                                                      double timestampMillis) {
    auto [sec, nanosec]{millisecondsToRosTime(timestampMillis)};
    auto header{std_msgs::msg::Header()};
    header.stamp.sec = sec;
    header.stamp.nanosec = nanosec;
    return cv_bridge::CvImage(header, "bgr8", colorFrame).toImageMsg();
  }

  sensor_msgs::msg::Image::SharedPtr getDepthImageMsg(const cv::Mat& depthFrame,
                                                      double timestampMillis) {
    auto [sec, nanosec]{millisecondsToRosTime(timestampMillis)};
    auto header{std_msgs::msg::Header()};
    header.stamp.sec = sec;
    header.stamp.nanosec = nanosec;
    return cv_bridge::CvImage(header, "mono16", depthFrame).toImageMsg();
  }

  sensor_msgs::msg::CompressedImage::SharedPtr getColorCompressedImageMsg(
      const cv::Mat& colorFrame, double timestampMillis) {
    auto [sec, nanosec]{millisecondsToRosTime(timestampMillis)};
    auto header{std_msgs::msg::Header()};
    header.stamp.sec = sec;
    header.stamp.nanosec = nanosec;
    return cv_bridge::CvImage(header, "bgr8", colorFrame)
        .toCompressedImageMsg(cv_bridge::JPG);
  }

#pragma endregion

  struct CalibrationParams {
    cv::Mat tritonIntrinsicMatrix;
    cv::Mat tritonDistCoeffs;
    cv::Mat heliosIntrinsicMatrix;
    cv::Mat heliosDistCoeffs;
    cv::Mat xyzToTritonExtrinsicMatrix;
    cv::Mat xyzToHeliosExtrinsicMatrix;
  };
  CalibrationParams readCalibrationParams(
      const std::string& calib_id = "1c0faf4b115d1c0faf4d17ce") {
    std::string packageShareDirectory{
        ament_index_cpp::get_package_share_directory("camera_control")};
    std::filesystem::path calibParamsDir{
        std::filesystem::path(packageShareDirectory) / "calibration_params" /
        calib_id};
    std::filesystem::path tritonIntrinsicsPath{calibParamsDir /
                                               "triton_intrinsics.yml"};
    std::filesystem::path heliosIntrinsicsPath{calibParamsDir /
                                               "helios_intrinsics.yml"};
    std::filesystem::path xyzToTritonIntrinsicsPath{
        calibParamsDir / "xyz_to_triton_extrinsics.yml"};
    std::filesystem::path xyzToHeliosIntrinsicsPath{
        calibParamsDir / "xyz_to_helios_extrinsics.yml"};
    auto tritonIntrinsicsOpt{
        calibration::readIntrinsicsFile(tritonIntrinsicsPath.string())};
    if (!tritonIntrinsicsOpt) {
      throw std::runtime_error(fmt::format("Could not read calibration file {}",
                                           tritonIntrinsicsPath.string()));
    }
    auto [tritonIntrinsicMatrix, tritonDistCoeffs]{tritonIntrinsicsOpt.value()};
    auto heliosIntrinsicsOpt{
        calibration::readIntrinsicsFile(heliosIntrinsicsPath.string())};
    if (!heliosIntrinsicsOpt) {
      throw std::runtime_error(fmt::format("Could not read calibration file {}",
                                           heliosIntrinsicsPath.string()));
    }
    auto [heliosIntrinsicMatrix, heliosDistCoeffs]{heliosIntrinsicsOpt.value()};
    auto xyzToTritonExtrinsicsOpt{
        calibration::readExtrinsicsFile(xyzToTritonIntrinsicsPath.string())};
    if (!xyzToTritonExtrinsicsOpt) {
      throw std::runtime_error(fmt::format("Could not read calibration file {}",
                                           xyzToTritonIntrinsicsPath.string()));
    }
    auto xyzToTritonExtrinsicMatrix{xyzToTritonExtrinsicsOpt.value()};
    auto xyzToHeliosExtrinsicsOpt{
        calibration::readExtrinsicsFile(xyzToHeliosIntrinsicsPath.string())};
    if (!xyzToHeliosExtrinsicsOpt) {
      throw std::runtime_error(fmt::format("Could not read calibration file {}",
                                           xyzToHeliosIntrinsicsPath.string()));
    }
    auto xyzToHeliosExtrinsicMatrix{xyzToHeliosExtrinsicsOpt.value()};

    CalibrationParams result;
    result.tritonIntrinsicMatrix = tritonIntrinsicMatrix;
    result.tritonDistCoeffs = tritonDistCoeffs;
    result.heliosIntrinsicMatrix = heliosIntrinsicMatrix;
    result.heliosDistCoeffs = heliosDistCoeffs;
    result.xyzToTritonExtrinsicMatrix = xyzToTritonExtrinsicMatrix;
    result.xyzToHeliosExtrinsicMatrix = xyzToHeliosExtrinsicMatrix;

    return result;
  }

  std::shared_ptr<rclcpp::ParameterEventHandler> paramSubscriber_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> exposureUsParamCallback_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> gainDbParamCallback_;
  rclcpp::Publisher<camera_control_interfaces::msg::State>::SharedPtr
      statePublisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debugFramePublisher_;
  rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
      notificationsPublisher_;
  rclcpp::CallbackGroup::SharedPtr serviceCallbackGroup_;
  rclcpp::Service<camera_control_interfaces::srv::StartDevice>::SharedPtr
      startDeviceService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr closeDeviceService_;
  rclcpp::Service<common_interfaces::srv::GetBool>::SharedPtr hasFramesService_;
  rclcpp::Service<camera_control_interfaces::srv::GetFrame>::SharedPtr
      getFrameService_;
  rclcpp::Service<camera_control_interfaces::srv::AcquireSingleFrame>::SharedPtr
      acquireSingleFrameService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr
      startRecordingVideoService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stopRecordingVideoService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr saveImageService_;
  rclcpp::Service<camera_control_interfaces::srv::StartIntervalCapture>::
      SharedPtr startIntervalCaptureService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr
      stopIntervalCaptureService_;
  rclcpp::Service<camera_control_interfaces::srv::GetState>::SharedPtr
      getStateService_;

  std::shared_ptr<LucidCamera> camera_;
  // Used to prevent frame callback from updating the current frame after the
  // camera device has been stopped
  std::atomic<bool> cameraStarted_{false};
  std::shared_ptr<LucidFrame> currentFrame_;
  std::mutex frameMutex_;
  cv::Mat currentDebugFrame_;
  std::mutex debugFrameMutex_;
  rclcpp::TimerBase::SharedPtr intervalCaptureTimer_;
  rclcpp::TimerBase::SharedPtr videoRecordingTimer_;
  cv::VideoWriter videoWriter_;
  std::mutex videoWriterMutex_;
  rclcpp::TimerBase::SharedPtr deviceTemperaturePublishTimer_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);

  try {
    // MultiThreadedExecutor allows callbacks to run in parallel
    rclcpp::executors::MultiThreadedExecutor executor;
    auto node{std::make_shared<CameraControlNode>()};
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception& e) {
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}