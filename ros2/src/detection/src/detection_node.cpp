#include <cv_bridge/cv_bridge.h>
#include <fmt/core.h>

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "detection/detector/runner_detector.hpp"
#include "rcl_interfaces/msg/log.hpp"
#include "rclcpp/rclcpp.hpp"
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

void drawRunners(const cv::Mat& image, const std::vector<Runner>& runners,
                 const cv::Size& runnerImageSize,
                 cv::Scalar color = {255, 0, 0}, unsigned int scale = 1) {
  if (image.empty() || runnerImageSize.width <= 0 ||
      runnerImageSize.height <= 0) {
    return;
  }

  // The image we are drawing on may not be the size of the image that the
  // runner detection was run on. Thus, we'll need to scale the bbox, mask, and
  // point of each runner to the image we are drawing on.
  double xScale{static_cast<double>(image.cols) / runnerImageSize.width};
  double yScale{static_cast<double>(image.rows) / runnerImageSize.height};

  // Draw segmentation masks
  if (!runners.empty() && !runners[0].boxMask.empty()) {
    cv::Mat mask{image.clone()};
    for (const auto& runner : runners) {
      // Scale rect to current image space
      cv::Rect2f rect{runner.rect};
      rect.x = static_cast<float>(rect.x * xScale);
      rect.y = static_cast<float>(rect.y * yScale);
      rect.width = static_cast<float>(rect.width * xScale);
      rect.height = static_cast<float>(rect.height * yScale);
      cv::Rect rectInt = rect;  // implicit float -> int conversion
      cv::Rect imgRect(0, 0, image.cols, image.rows);
      cv::Rect roi{rectInt & imgRect};  // clip to image
      if (roi.width <= 0 || roi.height <= 0) {
        continue;
      }

      // Scale boxMask to current image space
      cv::Mat boxMask{runner.boxMask};
      cv::Mat resizedBoxMask;
      cv::resize(boxMask, resizedBoxMask, cv::Size(roi.width, roi.height), 0, 0,
                 cv::INTER_NEAREST);

      mask(roi).setTo(color, resizedBoxMask);
    }
    // Add all the masks to our image
    cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
  }

  // Bounding boxes and annotations
  double meanColor{cv::mean(color)[0]};
  cv::Scalar textColor{(meanColor > 128) ? cv::Scalar(0, 0, 0)
                                         : cv::Scalar(255, 255, 255)};
  cv::Scalar markerColor{255, 255, 255};
  for (const auto& runner : runners) {
    // Scale rect to current image space
    cv::Rect2f rect{runner.rect};
    rect.x = static_cast<float>(rect.x * xScale);
    rect.y = static_cast<float>(rect.y * yScale);
    rect.width = static_cast<float>(rect.width * xScale);
    rect.height = static_cast<float>(rect.height * yScale);
    cv::Rect rectInt = rect;  // implicit float -> int conversion
    cv::Rect imgRect(0, 0, image.cols, image.rows);
    cv::Rect roi{rectInt & imgRect};  // clip to image
    if (roi.width <= 0 || roi.height <= 0) {
      continue;
    }

    // Draw rectangles and text
    char text[256];
    sprintf(text, "%d: %.1f%%", runner.trackId, runner.conf * 100);
    int baseLine{0};
    cv::Size labelSize{cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                       0.5 * scale, scale, &baseLine)};
    cv::Scalar textBackgroundColor{color * 0.7};
    cv::rectangle(image, roi, color, scale + 1);
    cv::rectangle(
        image,
        cv::Rect(cv::Point(roi.x, roi.y),
                 cv::Size(labelSize.width, labelSize.height + baseLine)),
        textBackgroundColor, -1);
    cv::putText(image, text, cv::Point(roi.x, roi.y + labelSize.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5 * scale, textColor, scale);

    // Draw representative point
    if (runner.point.x >= 0 && runner.point.y >= 0) {
      int x{static_cast<int>(std::round(runner.point.x * xScale))};
      int y{static_cast<int>(std::round(runner.point.y * yScale))};
      cv::drawMarker(image, cv::Point2i(x, y), markerColor,
                     cv::MARKER_TILTED_CROSS, 20, 2);
    }
  }
}

/**
 * Concurrency primitive that provides a shared flag that can be set and waited
 * on.
 */
class Event {
 public:
  Event() : flag_(false) {}

  void set() {
    std::lock_guard<std::mutex> lock(mtx_);
    flag_ = true;
    cv_.notify_all();
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    flag_ = false;
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this] { return flag_; });
  }

  bool wait_for(float timeoutSecs) {
    std::unique_lock<std::mutex> lock(mtx_);
    return cv_.wait_for(lock, std::chrono::duration<float>(timeoutSecs),
                        [this] { return flag_; });
  }

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  bool flag_;
};

class DetectionNode : public rclcpp::Node {
 public:
  explicit DetectionNode(const rclcpp::NodeOptions& options)
      : Node("detection_node", options) {
    /////////////
    // Parameters
    /////////////
    declare_parameter<int>("debug_image_width", 640);
    declare_parameter<float>("debug_video_fps", 30.0f);
    declare_parameter<std::string>("save_dir", "~/runner_cutter/camera");

    /////////////
    // Publishers
    /////////////
    // TODO: State
    debugImagePublisher_ = create_publisher<sensor_msgs::msg::Image>(
        "debug_image", rclcpp::SensorDataQoS());

    //////////////
    // Subscribers
    //////////////
    rclcpp::SubscriptionOptions subOptions;
    subscriberCallbackGroup_ =
        create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    subOptions.callback_group = subscriberCallbackGroup_;
    colorImageSubscriber_ = create_subscription<sensor_msgs::msg::Image>(
        "color_image", rclcpp::SensorDataQoS(),
        std::bind(&DetectionNode::onColorImage, this, std::placeholders::_1),
        subOptions);

    ///////////
    // Services
    ///////////
    serviceCallbackGroup_ =
        create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    startRecordingVideoService_ = create_service<std_srvs::srv::Trigger>(
        "~/start_recording_video",
        std::bind(&DetectionNode::onStartRecordingVideo, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    stopRecordingVideoService_ = create_service<std_srvs::srv::Trigger>(
        "~/stop_recording_video",
        std::bind(&DetectionNode::onStopRecordingVideo, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);

    runnerDetector_ = std::make_unique<RunnerDetector>();

    detectionThread_ = std::thread(&DetectionNode::detectionThreadFn, this);

    // Publish initial state
    publishState();
  }

  ~DetectionNode() {
    // Signal stop and join detection thread
    detectionStopSignal_ = true;
    colorImageEvent_.set();
    if (detectionThread_.joinable()) {
      detectionThread_.join();
    }
  }

 private:
#pragma region Param Helpers

  int getParamDebugImageWidth() {
    return static_cast<int>(get_parameter("debug_image_width").as_int());
  }

  float getParamDebugVideoFps() {
    return static_cast<float>(get_parameter("debug_video_fps").as_double());
  }

  std::string getParamSaveDir() {
    return get_parameter("save_dir").as_string();
  }

#pragma endregion

#pragma region Callbacks

  void onColorImage(const sensor_msgs::msg::Image::ConstSharedPtr imgMsg) {
    std::lock_guard<std::mutex> lock(lastColorImageMutex_);
    lastColorImage_ = imgMsg;
    colorImageEvent_.set();
  }

  void detectionThreadFn() {
    // Note: we use separate CUDA streams due to branching
    // GPU upload -> demosaic ---> runner detection
    //        (stream A)       |     (stream B)
    //                         └-> resize for debug frame
    //                               (stream C)
    cv::cuda::Stream cvStream0_;
    cv::cuda::Stream cvStream1_;

    // Allocate once up front to avoid overhead and fragmentation
    cv::cuda::GpuMat gpuRaw;
    cv::cuda::GpuMat gpuRgb;
    cv::cuda::GpuMat gpuDebugImage;
    cv::Mat debugImage;

    while (!detectionStopSignal_) {
      colorImageEvent_.wait();
      sensor_msgs::msg::Image::ConstSharedPtr imgMsg{nullptr};
      {
        std::lock_guard<std::mutex> lock(lastColorImageMutex_);
        imgMsg = lastColorImage_;
      }
      colorImageEvent_.clear();

      if (!imgMsg) {
        continue;
      }

      // Wrap raw Bayer bytes with stride (no copy)
      cv::Mat raw(imgMsg->height, imgMsg->width, CV_8UC1,
                  const_cast<uint8_t*>(imgMsg->data.data()),
                  static_cast<size_t>(imgMsg->step));

      // Upload image (BayerRG8) to GPU
      // Note: GpuMat::create is a no-op if size/type matches
      gpuRaw.create(imgMsg->height, imgMsg->width, CV_8UC1);
      gpuRaw.upload(raw, cvStream0_);

      // Demosaic on GPU
      gpuRgb.create(imgMsg->height, imgMsg->width, CV_8UC3);
      cv::cuda::demosaicing(gpuRaw, gpuRgb, cv::COLOR_BayerRG2RGB, -1,
                            cvStream0_);

      cvStream0_.waitForCompletion();

      ////////////
      // Detection
      ////////////
      // TODO: Control this via service calls
      auto runners{runnerDetector_->track(gpuRgb)};

      /////////////////////
      // Create debug image
      /////////////////////
      // Downscale using INTER_NEAREST for best perf
      double aspectRatio{static_cast<double>(imgMsg->height) / imgMsg->width};
      int debugImageWidth{getParamDebugImageWidth()};
      int debugImageHeight{
          static_cast<int>(std::round(debugImageWidth * aspectRatio))};
      gpuDebugImage.create(debugImageHeight, debugImageWidth, CV_8UC3);
      cv::cuda::resize(gpuRgb, gpuDebugImage,
                       cv::Size(debugImageWidth, debugImageHeight), 0.0, 0.0,
                       cv::INTER_NEAREST, cvStream1_);

      // Copy back to host
      debugImage.create(debugImageHeight, debugImageWidth, CV_8UC3);
      gpuDebugImage.download(debugImage, cvStream1_);

      // Synchronize stream before using debugImage
      cvStream1_.waitForCompletion();

      drawRunners(debugImage, runners, cv::Size(imgMsg->width, imgMsg->height));

      {
        std::lock_guard<std::mutex> lock(debugImageMutex_);
        lastDebugImage_ = debugImage;
      }

      // Publish debug image
      auto debugImageMsg{
          cv_bridge::CvImage(imgMsg->header, "rgb8", debugImage).toImageMsg()};
      debugImagePublisher_->publish(*debugImageMsg);
    }
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

  void writeVideoFrame() {
    std::lock_guard<std::mutex> lock(videoWriterMutex_);

    cv::Mat debugImage;
    {
      std::lock_guard<std::mutex> lock(debugImageMutex_);
      if (lastDebugImage_.empty()) {
        return;
      }
      debugImage = lastDebugImage_;
    }

    if (!videoWriter_.isOpened()) {
      // Create the save directory if it doesn't exist
      std::string saveDir{getParamSaveDir()};
      saveDir = expandUser(saveDir);
      std::filesystem::create_directories(saveDir);

      // Generate the video file name and path
      auto now{std::chrono::system_clock::now()};
      auto timestamp{std::chrono::system_clock::to_time_t(now)};
      std::stringstream datetimeString;
      datetimeString << std::put_time(std::localtime(&timestamp),
                                      "%Y%m%d%H%M%S");
      auto ms{std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000};
      datetimeString << std::setw(3) << std::setfill('0') << ms.count();
      std::string filepath{
          fmt::format("{}/{}.avi", saveDir, datetimeString.str())};

      int width{debugImage.cols};
      int height{debugImage.rows};
      videoWriter_.open(filepath, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                        getParamDebugVideoFps(), cv::Size(width, height));

      if (!videoWriter_.isOpened()) {
        RCLCPP_ERROR(get_logger(), "Failed to open video writer.");
        return;
      }

      publishNotification("Started recording video: " + filepath);
    }

    videoWriter_.write(debugImage);
  }

#pragma endregion

#pragma region State and notifs publishing

  void publishState() {
    // TODO
  }

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

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debugImagePublisher_;
  rclcpp::CallbackGroup::SharedPtr subscriberCallbackGroup_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      colorImageSubscriber_;
  rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
      notificationsPublisher_;
  rclcpp::CallbackGroup::SharedPtr serviceCallbackGroup_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr
      startRecordingVideoService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stopRecordingVideoService_;

  std::unique_ptr<RunnerDetector> runnerDetector_;
  std::thread detectionThread_;
  std::atomic<bool> detectionStopSignal_{false};
  sensor_msgs::msg::Image::ConstSharedPtr lastColorImage_{nullptr};
  std::mutex lastColorImageMutex_;
  // Notifies the detection thread that a new image is available
  Event colorImageEvent_;
  rclcpp::TimerBase::SharedPtr videoRecordingTimer_;
  cv::VideoWriter videoWriter_;
  std::mutex videoWriterMutex_;
  cv::Mat lastDebugImage_;
  std::mutex debugImageMutex_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(DetectionNode)