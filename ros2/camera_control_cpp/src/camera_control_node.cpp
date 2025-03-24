#include <cv_bridge/cv_bridge.h>

#include <atomic>
#include <filesystem>
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
#include "camera_control_interfaces/srv/start_detection.hpp"
#include "camera_control_interfaces/srv/start_device.hpp"
#include "camera_control_interfaces/srv/start_interval_capture.hpp"
#include "camera_control_interfaces/srv/stop_detection.hpp"
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

std::string expandUser(const std::string& path) {
  if (path.empty() || path[0] != '~') {
    return path;
  }

  const char* home = std::getenv("HOME");
  if (!home) {
    throw std::runtime_error("HOME environment variable not set");
  }

  return std::string(home) + path.substr(1);
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
        create_publisher<rcl_interfaces::msg::Log>("/notifications", 1);

    ///////////
    // Services
    ///////////
    startDeviceService_ =
        create_service<camera_control_interfaces::srv::StartDevice>(
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
    startDetectionService_ =
        create_service<camera_control_interfaces::srv::StartDetection>(
            "~/start_detection",
            std::bind(&CameraControlNode::onStartDetection, this,
                      std::placeholders::_1, std::placeholders::_2));
    stopDetectionService_ =
        create_service<camera_control_interfaces::srv::StopDetection>(
            "~/stop_detection",
            std::bind(&CameraControlNode::onStopDetection, this,
                      std::placeholders::_1, std::placeholders::_2));
    stopAllDetectionsService_ = create_service<std_srvs::srv::Trigger>(
        "~/stop_all_detections",
        std::bind(&CameraControlNode::onStopAllDetections, this,
                  std::placeholders::_1, std::placeholders::_2));
    setSaveDirectoryService_ =
        create_service<camera_control_interfaces::srv::SetSaveDirectory>(
            "~/set_save_directory",
            std::bind(&CameraControlNode::onSetSaveDirectory, this,
                      std::placeholders::_1, std::placeholders::_2));
    startRecordingVideoService_ = create_service<std_srvs::srv::Trigger>(
        "~/start_recording_video",
        std::bind(&CameraControlNode::onStartRecordingVideo, this,
                  std::placeholders::_1, std::placeholders::_2));
    stopRecordingVideoService_ = create_service<std_srvs::srv::Trigger>(
        "~/stop_recording_video",
        std::bind(&CameraControlNode::onStopRecordingVideo, this,
                  std::placeholders::_1, std::placeholders::_2));
    saveImageService_ = create_service<std_srvs::srv::Trigger>(
        "~/save_image",
        std::bind(&CameraControlNode::onSaveImage, this, std::placeholders::_1,
                  std::placeholders::_2));
    startIntervalCaptureService_ =
        create_service<camera_control_interfaces::srv::StartIntervalCapture>(
            "~/start_interval_capture",
            std::bind(&CameraControlNode::onStartIntervalCapture, this,
                      std::placeholders::_1, std::placeholders::_2));
    stopIntervalCaptureService_ = create_service<std_srvs::srv::Trigger>(
        "~/stop_interval_capture",
        std::bind(&CameraControlNode::onStopIntervalCapture, this,
                  std::placeholders::_1, std::placeholders::_2));
    getStateService_ = create_service<camera_control_interfaces::srv::GetState>(
        "~/get_state", std::bind(&CameraControlNode::onGetState, this,
                                 std::placeholders::_1, std::placeholders::_2));

    ///////////////
    // Camera Setup
    ///////////////
    cv::Mat testMat;  // TODO: replace with real mats
    LucidCamera::StateChangeCallback stateChangeCallback =
        [this](LucidCamera::State state) { publishState(); };
    camera_ = std::make_shared<LucidCamera>(
        testMat, testMat, testMat, testMat, testMat, testMat, std::nullopt,
        std::nullopt, std::pair<int, int>{2048, 1536}, stateChangeCallback);

    //////////////////
    // Detectors Setup
    //////////////////

    // Publish initial state
    publishState();
  }

  ~CameraControlNode() { stopDetectionThread(); }

 private:
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

    startDetectionThread();

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

    stopDetectionThread();

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

  void onStartDetection(
      const std::shared_ptr<
          camera_control_interfaces::srv::StartDetection::Request>
          request,
      std::shared_ptr<camera_control_interfaces::srv::StartDetection::Response>
          response) {
    double minX = request->normalized_bounds.w;
    double minY = request->normalized_bounds.x;
    double w = request->normalized_bounds.y;
    double h = request->normalized_bounds.z;
    // If normalized bounds are not defined, set to full bounds (0, 0, 1, 1)
    if (minX == 0.0 && minY == 0.0 && w == 0.0 && h == 0.0) {
      enabledDetections_[request->detection_type] =
          std::make_tuple(0.0f, 0.0f, 1.0f, 1.0f);
    } else {
      enabledDetections_[request->detection_type] =
          std::make_tuple(minX, minY, w, h);
    }
    response->success = true;
  }

  void onStopDetection(
      const std::shared_ptr<
          camera_control_interfaces::srv::StopDetection::Request>
          request,
      std::shared_ptr<camera_control_interfaces::srv::StopDetection::Response>
          response) {
    if (enabledDetections_.find(request->detection_type) ==
        enabledDetections_.end()) {
      response->success = false;
      return;
    }

    enabledDetections_.erase(request->detection_type);
    publishState();
    response->success = true;
  }

  void onStopAllDetections(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (enabledDetections_.empty()) {
      response->success = false;
      return;
    }

    enabledDetections_.clear();
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

  void onStartRecordingVideo(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (videoRecordingTimer_) {
      videoRecordingTimer_.reset();
      std::lock_guard<std::mutex> lock(videoWriterMutex_);
      videoWriter_.release();
    }

    videoRecordingTimer_ = create_wall_timer(
        std::chrono::duration<double>(
            1.0 / get_parameter("debug_video_fps").as_double()),
        [this]() { writeVideoFrame(); });

    publishState();
    response->success = true;
  }

  void onStopRecordingVideo(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!videoRecordingTimer_) {
      response->success = false;
      return;
    }

    videoRecordingTimer_.reset();
    {
      std::lock_guard<std::mutex> lock(videoWriterMutex_);
      videoWriter_.release();
    }
    publishState();
    publishNotification("Stopped recording video");
    response->success = true;
  }

  void onSaveImage(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    auto res = saveImage();
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
    response->state = *getState();
  }

#pragma endregion

  void startDetectionThread() {
    if (detectionThreadRunFlag_) {
      return;
    }

    detectionThreadRunFlag_ = true;
    detectionThread_ = std::thread([this]() {
      while (detectionThreadRunFlag_ && rclcpp::ok()) {
        // Wait for new frame
        std::shared_ptr<LucidFrame> frame;
        {
          std::unique_lock<std::mutex> lock(frameMutex_);
          frameEvent_.wait(lock);

          if (!detectionThreadRunFlag_ || !rclcpp::ok()) {
            return;
          }

          frame = currentFrame_;
        }

        detectionCallback(frame);
      }
    });
  }

  void stopDetectionThread() {
    if (!detectionThreadRunFlag_) {
      return;
    }

    detectionThreadRunFlag_ = false;
    if (detectionThread_.joinable()) {
      frameEvent_.notify_all();
      detectionThread_.join();
    }
  }

  void detectionCallback(std::shared_ptr<const LucidFrame> frame) {
    // Create debug frame as a copy of the color frame
    auto debugFrame{frame->getColorFrame().clone()};

    // Downscale debug_frame using INTER_NEAREST for best performance
    double aspectRatio{static_cast<double>(debugFrame.rows) / debugFrame.cols};
    int newWidth{static_cast<int>(get_parameter("debug_frame_width").as_int())};
    int newHeight{static_cast<int>(std::round(newWidth * aspectRatio))};
    cv::resize(debugFrame, debugFrame, cv::Size(newWidth, newHeight), 0, 0,
               cv::INTER_NEAREST);

    {
      std::lock_guard<std::mutex> lock(debugFrameMutex_);
      currentDebugFrame_ = debugFrame;
    }

    sensor_msgs::msg::Image::SharedPtr debugFrameMsg =
        getColorFrameMsg(debugFrame, frame->getTimestampMillis());
    debugFramePublisher_->publish(*debugFrameMsg);
  }

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
    std::string saveDir = get_parameter("save_dir").as_string();
    saveDir = expandUser(saveDir);
    std::filesystem::create_directories(saveDir);

    // Get the current timestamp and format it as a string
    auto ts =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm = *std::localtime(&ts);
    char datetimeString[20];
    std::strftime(datetimeString, sizeof(datetimeString), "%Y%m%d%H%M%S", &tm);

    // Generate the image name and path
    std::string filename = std::string(datetimeString) + ".png";
    std::string filepath = saveDir + "/" + filename;

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
      std::string saveDir = get_parameter("save_dir").as_string();
      saveDir = expandUser(saveDir);
      std::filesystem::create_directories(saveDir);

      // Get the current timestamp and format it as a string
      auto ts = std::chrono::system_clock::to_time_t(
          std::chrono::system_clock::now());
      std::tm tm = *std::localtime(&ts);
      char datetimeString[20];
      std::strftime(datetimeString, sizeof(datetimeString), "%Y%m%d%H%M%S",
                    &tm);

      // Generate the video file name and path
      std::string filename = std::string(datetimeString) + ".avi";
      std::string filepath = saveDir + "/" + filename;

      int width = frame.cols;
      int height = frame.rows;
      double fps = get_parameter("debug_video_fps").as_double();
      videoWriter_.open(filepath, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                        fps, cv::Size(width, height));

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

  camera_control_interfaces::msg::State::SharedPtr getState() {
    auto msg{std::make_shared<camera_control_interfaces::msg::State>()};
    msg->device_state = getDeviceState();
    for (const auto& pair : enabledDetections_) {
      msg->enabled_detection_types.push_back(pair.first);
    }
    msg->recording_video = (videoRecordingTimer_ != nullptr);
    msg->interval_capture_active = (intervalCaptureTimer_ != nullptr);
    msg->exposure_us = camera_->getExposureUs();
    auto exposureUsRange{camera_->getExposureUsRange()};
    msg->exposure_us_range.x = exposureUsRange.first;
    msg->exposure_us_range.y = exposureUsRange.second;
    msg->gain_db = camera_->getGainDb();
    auto gainDbRange{camera_->getGainDbRange()};
    msg->gain_db_range.x = gainDbRange.first;
    msg->gain_db_range.y = gainDbRange.second;
    msg->save_directory = get_parameter("save_dir").as_string();
    msg->image_capture_interval_secs = static_cast<float>(
        get_parameter("image_capture_interval_secs").as_double());
    return msg;
  }

  void publishState() { statePublisher_->publish(*getState()); }

  void publishNotification(
      const std::string& msg,
      rclcpp::Logger::Level level = rclcpp::Logger::Level::Info) {
    uint8_t logMsgLevel = 0;
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
    notificationsPublisher_->publish(logMsg);
  }

#pragma endregion

#pragma region Message builders

  sensor_msgs::msg::Image::SharedPtr getColorFrameMsg(const cv::Mat& colorFrame,
                                                      double timestampMillis) {
    auto [sec, nanosec]{millisecondsToRosTime(timestampMillis)};
    auto header{std_msgs::msg::Header()};
    header.stamp.sec = sec;
    header.stamp.nanosec = nanosec;
    return cv_bridge::CvImage(header, "bgr8", colorFrame).toImageMsg();
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
  rclcpp::Service<camera_control_interfaces::srv::StartDevice>::SharedPtr
      startDeviceService_;
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
  rclcpp::Service<camera_control_interfaces::srv::StartDetection>::SharedPtr
      startDetectionService_;
  rclcpp::Service<camera_control_interfaces::srv::StopDetection>::SharedPtr
      stopDetectionService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stopAllDetectionsService_;
  rclcpp::Service<camera_control_interfaces::srv::SetSaveDirectory>::SharedPtr
      setSaveDirectoryService_;
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
  // Used to notify when a new frame is available
  std::condition_variable frameEvent_;
  std::thread detectionThread_;
  std::atomic<bool> detectionThreadRunFlag_{false};
  // DetectionType -> normalized rect bounds (min x, min y, width, height)
  std::unordered_map<int, std::tuple<float, float, float, float>>
      enabledDetections_;
  cv::Mat currentDebugFrame_;
  std::mutex debugFrameMutex_;
  rclcpp::TimerBase::SharedPtr intervalCaptureTimer_;
  rclcpp::TimerBase::SharedPtr videoRecordingTimer_;
  cv::VideoWriter videoWriter_;
  std::mutex videoWriterMutex_;
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