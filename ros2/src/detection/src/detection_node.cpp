#include <cv_bridge/cv_bridge.h>
#include <fmt/core.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "camera_control/utils/rgbd_alignment.hpp"
#include "common_interfaces/msg/vector2.hpp"
#include "common_interfaces/msg/vector3.hpp"
#include "detection/detector/circle_detector.hpp"
#include "detection/detector/laser_detector.hpp"
#include "detection/detector/runner_detector.hpp"
#include "detection_interfaces/msg/detection_result.hpp"
#include "detection_interfaces/msg/detection_type.hpp"
#include "detection_interfaces/msg/object_instance.hpp"
#include "detection_interfaces/msg/state.hpp"
#include "detection_interfaces/srv/get_detection_result.hpp"
#include "detection_interfaces/srv/get_positions.hpp"
#include "detection_interfaces/srv/get_state.hpp"
#include "detection_interfaces/srv/start_detection.hpp"
#include "detection_interfaces/srv/stop_detection.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rcl_interfaces/msg/log.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace {

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

std::pair<cv::Mat, cv::Mat> getCameraMatrices(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& cameraInfo) {
  // Intrinsic matrix (3x3)
  cv::Mat K =
      (cv::Mat_<double>(3, 3) << cameraInfo->k[0], cameraInfo->k[1],
       cameraInfo->k[2], cameraInfo->k[3], cameraInfo->k[4], cameraInfo->k[5],
       cameraInfo->k[6], cameraInfo->k[7], cameraInfo->k[8]);

  // Distortion coefficients (Nx1)
  size_t numCoeffs{cameraInfo->d.size()};
  cv::Mat D(static_cast<int>(numCoeffs), 1, CV_64F);
  for (size_t i = 0; i < numCoeffs; ++i) {
    D.at<double>(static_cast<int>(i), 0) = cameraInfo->d[i];
  }

  return {K, D};
}

std::optional<cv::Mat> getTransformMatrix(
    const geometry_msgs::msg::TransformStamped& transformStamped) {
  const auto& t{transformStamped.transform.translation};
  const auto& q{transformStamped.transform.rotation};

  // Build and normalize quaternion
  tf2::Quaternion quat(q.x, q.y, q.z, q.w);
  if (quat.length2() == 0.0 || !std::isfinite(q.x) || !std::isfinite(q.y) ||
      !std::isfinite(q.z) || !std::isfinite(q.w)) {
    // Invalid quaternion
    return std::nullopt;
  }
  quat.normalize();

  // Convert to 3x3 rotation
  tf2::Matrix3x3 rotationMatrix(quat);

  cv::Mat T{cv::Mat::eye(4, 4, CV_64F)};
  // Rotation
  T.at<double>(0, 0) = rotationMatrix[0][0];
  T.at<double>(0, 1) = rotationMatrix[0][1];
  T.at<double>(0, 2) = rotationMatrix[0][2];
  T.at<double>(1, 0) = rotationMatrix[1][0];
  T.at<double>(1, 1) = rotationMatrix[1][1];
  T.at<double>(1, 2) = rotationMatrix[1][2];
  T.at<double>(2, 0) = rotationMatrix[2][0];
  T.at<double>(2, 1) = rotationMatrix[2][1];
  T.at<double>(2, 2) = rotationMatrix[2][2];
  // Translation
  T.at<double>(0, 3) = t.x;
  T.at<double>(1, 3) = t.y;
  T.at<double>(2, 3) = t.z;

  return T;
}

}  // namespace

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
    // Note: we need to explicitly disable intra-process comms on latched topics
    // as intra-process comms are only allowed with volatile durability
    rclcpp::QoS latchedQos(rclcpp::KeepLast(1));
    latchedQos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
    rclcpp::PublisherOptions intraProcessDisableOpts;
    intraProcessDisableOpts.use_intra_process_comm =
        rclcpp::IntraProcessSetting::Disable;
    statePublisher_ = create_publisher<detection_interfaces::msg::State>(
        "~/state", latchedQos, intraProcessDisableOpts);
    detectionsPublisher_ =
        create_publisher<detection_interfaces::msg::DetectionResult>(
            "~/detections", rclcpp::SensorDataQoS());
    notificationsPublisher_ =
        create_publisher<rcl_interfaces::msg::Log>("/notifications", 1);
    debugImagePublisher_ = create_publisher<sensor_msgs::msg::Image>(
        "debug/image", rclcpp::SensorDataQoS());

    //////////////
    // Subscribers
    //////////////
    rclcpp::SubscriptionOptions subOptions;
    subscriberCallbackGroup_ =
        create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    subOptions.callback_group = subscriberCallbackGroup_;
    // Note: I attempted to use message_filters to fire a callback when both
    // color image and xyz data came in with close timestamps, but it resulted
    // in a significant delay; instead, we use manual matching.
    colorImageSubscriber_ = create_subscription<sensor_msgs::msg::Image>(
        "color/image_raw", rclcpp::SensorDataQoS(),
        std::bind(&DetectionNode::onColorImage, this, std::placeholders::_1),
        subOptions);
    colorCameraInfoSubscriber_ =
        create_subscription<sensor_msgs::msg::CameraInfo>(
            "color/camera_info", rclcpp::SensorDataQoS(),
            std::bind(&DetectionNode::onColorCameraInfo, this,
                      std::placeholders::_1),
            subOptions);
    depthXyzSubscriber_ = create_subscription<sensor_msgs::msg::Image>(
        "depth/xyz", rclcpp::SensorDataQoS(),
        std::bind(&DetectionNode::onDepthXyz, this, std::placeholders::_1),
        subOptions);
    depthCameraInfoSubscriber_ =
        create_subscription<sensor_msgs::msg::CameraInfo>(
            "depth/camera_info", rclcpp::SensorDataQoS(),
            std::bind(&DetectionNode::onDepthCameraInfo, this,
                      std::placeholders::_1),
            subOptions);
    // Use tf2 to read camera extrinsic matrices
    tfBuffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tfListener_ = std::make_shared<tf2_ros::TransformListener>(*tfBuffer_);

    ///////////
    // Services
    ///////////
    serviceCallbackGroup_ =
        create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    getDetectionService_ =
        create_service<detection_interfaces::srv::GetDetectionResult>(
            "~/get_detection",
            std::bind(&DetectionNode::onGetDetection, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, serviceCallbackGroup_);
    startDetectionService_ =
        create_service<detection_interfaces::srv::StartDetection>(
            "~/start_detection",
            std::bind(&DetectionNode::onStartDetection, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, serviceCallbackGroup_);
    stopDetectionService_ =
        create_service<detection_interfaces::srv::StopDetection>(
            "~/stop_detection",
            std::bind(&DetectionNode::onStopDetection, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, serviceCallbackGroup_);
    stopAllDetectionsService_ = create_service<std_srvs::srv::Trigger>(
        "~/stop_all_detections",
        std::bind(&DetectionNode::onStopAllDetections, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
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
    getStateService_ = create_service<detection_interfaces::srv::GetState>(
        "~/get_state",
        std::bind(&DetectionNode::onGetState, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default, serviceCallbackGroup_);
    getPositionsService_ =
        create_service<detection_interfaces::srv::GetPositions>(
            "~/get_positions",
            std::bind(&DetectionNode::onGetPositions, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, serviceCallbackGroup_);

    laserDetector_ = std::make_unique<LaserDetector>();
    runnerDetector_ = std::make_unique<RunnerDetector>();
    circleDetector_ = std::make_unique<CircleDetector>();

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

  void onColorImage(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lock(lastColorImageMutex_);
    lastColorImage_ = msg;
    colorImageEvent_.set();
  }

  void onColorCameraInfo(
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lock(colorCameraInfoMutex_);
    colorCameraInfo_ = msg;
  }

  void onDepthXyz(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lock(depthXyzQueueMutex_);

    // Append and prune depth frames older than keepDuration_
    depthXyzQueue_.push_back(msg);
    const rclcpp::Time newest{depthXyzQueue_.back()->header.stamp};
    while (!depthXyzQueue_.empty()) {
      rclcpp::Time oldest{depthXyzQueue_.front()->header.stamp};
      if ((newest - oldest) > keepDuration_) {
        depthXyzQueue_.pop_front();
      } else {
        break;
      }
    }
  }

  void onDepthCameraInfo(
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lock(depthCameraInfoMutex_);
    depthCameraInfo_ = msg;
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
    cv::Mat rgbImage;
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

      ////////////
      // Detection
      ////////////
      if (enabledDetections_.find(
              detection_interfaces::msg::DetectionType::RUNNER) !=
          enabledDetections_.end()) {
        cv::Rect2d normalizedBounds{
            enabledDetections_
                [detection_interfaces::msg::DetectionType::RUNNER]};
        cv::Rect bounds{
            static_cast<int>(std::ceil(normalizedBounds.x * imgMsg->width)),
            static_cast<int>(std::ceil(normalizedBounds.y * imgMsg->height)),
            static_cast<int>(
                std::floor(normalizedBounds.width * imgMsg->width)),
            static_cast<int>(
                std::floor(normalizedBounds.height * imgMsg->height))};
        auto runners{runnerDetector_->track(gpuRgb, bounds)};

        // Create and publish DetectionResult
        auto detectionResult{createDetectionResult(runners, imgMsg)};
        detectionsPublisher_->publish(detectionResult);

        // Draw detections to debug image
        RunnerDetector::drawDetections(debugImage, runners,
                                       cv::Size(imgMsg->width, imgMsg->height));
      }

      if (enabledDetections_.find(
              detection_interfaces::msg::DetectionType::LASER) !=
          enabledDetections_.end()) {
        // Copy RGB back to host
        rgbImage.create(imgMsg->height, imgMsg->width, CV_8UC3);
        gpuRgb.download(rgbImage);

        // Run detection
        auto lasers{laserDetector_->detect(rgbImage)};

        // Create and publish DetectionResult
        auto detectionResult{createDetectionResult(lasers, imgMsg)};
        detectionsPublisher_->publish(detectionResult);

        // Draw detections to debug image
        LaserDetector::drawDetections(debugImage, lasers,
                                      cv::Size(imgMsg->width, imgMsg->height));
      }

      if (enabledDetections_.find(
              detection_interfaces::msg::DetectionType::CIRCLE) !=
          enabledDetections_.end()) {
        // Copy RGB back to host
        rgbImage.create(imgMsg->height, imgMsg->width, CV_8UC3);
        gpuRgb.download(rgbImage);

        // Run detection
        auto circles{circleDetector_->detect(rgbImage)};

        // Create and publish DetectionResult
        auto detectionResult{createDetectionResult(circles, imgMsg)};
        detectionsPublisher_->publish(detectionResult);

        // Draw detections to debug image
        CircleDetector::drawDetections(debugImage, circles,
                                       cv::Size(imgMsg->width, imgMsg->height));
      }

      //////////////////////
      // Publish debug image
      //////////////////////
      {
        std::lock_guard<std::mutex> lock(debugImageMutex_);
        lastDebugImage_ = debugImage;
      }
      auto debugImageMsg{
          cv_bridge::CvImage(imgMsg->header, "rgb8", debugImage).toImageMsg()};
      debugImagePublisher_->publish(*debugImageMsg);
    }
  }

  detection_interfaces::msg::DetectionResult createDetectionResult(
      const std::vector<RunnerDetector::Runner>& detections,
      const sensor_msgs::msg::Image::ConstSharedPtr image) {
    detection_interfaces::msg::DetectionResult detectionResult;
    detectionResult.detection_type =
        detection_interfaces::msg::DetectionType::RUNNER;
    detectionResult.timestamp =
        static_cast<double>(image->header.stamp.sec) +
        static_cast<double>(image->header.stamp.nanosec) * 1e-9;

    auto rgbdAlignment{getOrCreateRgbdAlignment()};
    auto depthXyz{getDepthXyz(image)};
    if (!rgbdAlignment || !depthXyz) {
      RCLCPP_WARN(get_logger(),
                  "[createDetectionResult] RgbdAlignment or depth XYZ is not "
                  "available");
      return detectionResult;
    }
    // Wrap the depthXyz data pointer (no copy)
    cv::Mat depthXyzMat(depthXyz->height, depthXyz->width, CV_32FC3,
                        const_cast<uint8_t*>(depthXyz->data.data()),
                        depthXyz->step);

    for (const auto& runner : detections) {
      common_interfaces::msg::Vector2 pointMsg;
      pointMsg.x = static_cast<double>(runner.point.x);
      pointMsg.y = static_cast<double>(runner.point.y);
      if (pointMsg.x < 0.0 || pointMsg.y < 0.0) {
        continue;
      }

      auto positionOpt{rgbdAlignment->getPosition(runner.point, depthXyzMat)};
      if (positionOpt) {
        const auto& pos{*positionOpt};
        common_interfaces::msg::Vector3 positionMsg;
        positionMsg.x = pos[0];
        positionMsg.y = pos[1];
        positionMsg.z = pos[2];

        detection_interfaces::msg::ObjectInstance objectInstance;
        objectInstance.confidence = runner.conf;
        objectInstance.point = pointMsg;
        objectInstance.position = positionMsg;
        objectInstance.track_id = runner.trackId > 0 ? runner.trackId : 0;
        detectionResult.instances.push_back(objectInstance);
      } else {
        detectionResult.invalid_points.push_back(pointMsg);
      }
    }

    return detectionResult;
  }

  detection_interfaces::msg::DetectionResult createDetectionResult(
      const std::vector<LaserDetector::Laser>& detections,
      const sensor_msgs::msg::Image::ConstSharedPtr image) {
    detection_interfaces::msg::DetectionResult detectionResult;
    detectionResult.detection_type =
        detection_interfaces::msg::DetectionType::LASER;
    detectionResult.timestamp =
        static_cast<double>(image->header.stamp.sec) +
        static_cast<double>(image->header.stamp.nanosec) * 1e-9;

    auto rgbdAlignment{getOrCreateRgbdAlignment()};
    auto depthXyz{getDepthXyz(image)};
    if (!rgbdAlignment || !depthXyz) {
      RCLCPP_WARN(get_logger(),
                  "[createDetectionResult] RgbdAlignment or depth XYZ is not "
                  "available");
      return detectionResult;
    }
    // Wrap the depthXyz data pointer (no copy)
    cv::Mat depthXyzMat(depthXyz->height, depthXyz->width, CV_32FC3,
                        const_cast<uint8_t*>(depthXyz->data.data()),
                        depthXyz->step);

    for (const auto& laser : detections) {
      common_interfaces::msg::Vector2 pointMsg;
      pointMsg.x = static_cast<double>(laser.point.x);
      pointMsg.y = static_cast<double>(laser.point.y);
      if (pointMsg.x < 0.0 || pointMsg.y < 0.0) {
        continue;
      }

      auto positionOpt{rgbdAlignment->getPosition(laser.point, depthXyzMat)};
      if (positionOpt) {
        const auto& pos{*positionOpt};
        common_interfaces::msg::Vector3 positionMsg;
        positionMsg.x = pos[0];
        positionMsg.y = pos[1];
        positionMsg.z = pos[2];

        detection_interfaces::msg::ObjectInstance objectInstance;
        objectInstance.confidence = laser.conf;
        objectInstance.point = pointMsg;
        objectInstance.position = positionMsg;
        detectionResult.instances.push_back(objectInstance);
      } else {
        detectionResult.invalid_points.push_back(pointMsg);
      }
    }

    return detectionResult;
  }

  detection_interfaces::msg::DetectionResult createDetectionResult(
      const std::vector<CircleDetector::Circle>& detections,
      const sensor_msgs::msg::Image::ConstSharedPtr image) {
    detection_interfaces::msg::DetectionResult detectionResult;
    detectionResult.detection_type =
        detection_interfaces::msg::DetectionType::CIRCLE;
    detectionResult.timestamp =
        static_cast<double>(image->header.stamp.sec) +
        static_cast<double>(image->header.stamp.nanosec) * 1e-9;

    auto rgbdAlignment{getOrCreateRgbdAlignment()};
    auto depthXyz{getDepthXyz(image)};
    if (!rgbdAlignment || !depthXyz) {
      RCLCPP_WARN(get_logger(),
                  "[createDetectionResult] RgbdAlignment or depth XYZ is not "
                  "available");
      return detectionResult;
    }
    // Wrap the depthXyz data pointer (no copy)
    cv::Mat depthXyzMat(depthXyz->height, depthXyz->width, CV_32FC3,
                        const_cast<uint8_t*>(depthXyz->data.data()),
                        depthXyz->step);

    for (const auto& circle : detections) {
      common_interfaces::msg::Vector2 pointMsg;
      pointMsg.x = static_cast<double>(circle.point.x);
      pointMsg.y = static_cast<double>(circle.point.y);
      if (pointMsg.x < 0.0 || pointMsg.y < 0.0) {
        continue;
      }

      auto positionOpt{rgbdAlignment->getPosition(circle.point, depthXyzMat)};
      if (positionOpt) {
        const auto& pos{*positionOpt};
        common_interfaces::msg::Vector3 positionMsg;
        positionMsg.x = pos[0];
        positionMsg.y = pos[1];
        positionMsg.z = pos[2];

        detection_interfaces::msg::ObjectInstance objectInstance;
        objectInstance.confidence = circle.conf;
        objectInstance.point = pointMsg;
        objectInstance.position = positionMsg;
        detectionResult.instances.push_back(objectInstance);
      } else {
        detectionResult.invalid_points.push_back(pointMsg);
      }
    }

    return detectionResult;
  }

  sensor_msgs::msg::Image::ConstSharedPtr getDepthXyz(
      const sensor_msgs::msg::Image::ConstSharedPtr colorImage) {
    sensor_msgs::msg::Image::ConstSharedPtr bestDepthXyz;
    rclcpp::Duration bestDelta{rclcpp::Duration::from_seconds(1e9)};

    std::lock_guard<std::mutex> lock(depthXyzQueueMutex_);

    if (depthXyzQueue_.empty()) {
      return nullptr;
    }

    const rclcpp::Time tc{colorImage->header.stamp};

    // Find closest depth by absolute time difference
    size_t bestIdx{depthXyzQueue_.size()};
    for (size_t i = 0; i < depthXyzQueue_.size(); ++i) {
      rclcpp::Time td{depthXyzQueue_[i]->header.stamp};
      rclcpp::Duration delta{(td > tc) ? (td - tc) : (tc - td)};
      if (delta < bestDelta) {
        bestDelta = delta;
        bestDepthXyz = depthXyzQueue_[i];
        bestIdx = i;
      }
    }

    if (bestIdx < depthXyzQueue_.size() && bestDelta <= maxIntervalDuration_) {
      // We have a match
      return bestDepthXyz;
    } else {
      return nullptr;
    }
  }

  void onGetDetection(
      const std::shared_ptr<
          detection_interfaces::srv::GetDetectionResult::Request>
          request,
      std::shared_ptr<detection_interfaces::srv::GetDetectionResult::Response>
          response) {
    sensor_msgs::msg::Image::ConstSharedPtr image{nullptr};
    {
      std::lock_guard<std::mutex> lock(lastColorImageMutex_);
      image = lastColorImage_;
    }

    if (!image) {
      return;
    }

    // Wrap raw Bayer bytes with stride (no copy)
    cv::Mat raw(image->height, image->width, CV_8UC1,
                const_cast<uint8_t*>(image->data.data()),
                static_cast<size_t>(image->step));

    // Demosaic
    cv::Mat rgb;
    cv::cvtColor(raw, rgb, cv::COLOR_BayerRG2RGB);

    switch (request->detection_type) {
      case detection_interfaces::msg::DetectionType::RUNNER: {
        auto runners{runnerDetector_->track(rgb)};
        response->result = createDetectionResult(runners, image);
        break;
      }
      case detection_interfaces::msg::DetectionType::LASER: {
        auto lasers{laserDetector_->detect(rgb)};
        response->result = createDetectionResult(lasers, image);
        break;
      }
      case detection_interfaces::msg::DetectionType::CIRCLE: {
        auto circles{circleDetector_->detect(rgb)};
        response->result = createDetectionResult(circles, image);
        break;
      }
    }
  }

  void onStartDetection(
      const std::shared_ptr<detection_interfaces::srv::StartDetection::Request>
          request,
      std::shared_ptr<detection_interfaces::srv::StartDetection::Response>
          response) {
    if (enabledDetections_.find(request->detection_type) !=
        enabledDetections_.end()) {
      // Detection already enabled for the requested DetectionType. Do nothing.
      response->success = false;
      return;
    }

    // If normalized bounds are all zero, set to full bounds (0, 0, 1, 1)
    if (request->normalized_bounds.w == 0.0 &&
        request->normalized_bounds.x == 0.0 &&
        request->normalized_bounds.y == 0.0 &&
        request->normalized_bounds.z == 0.0) {
      enabledDetections_[request->detection_type] =
          cv::Rect2d{0.0, 0.0, 1.0, 1.0};
    } else {
      enabledDetections_[request->detection_type] = cv::Rect2d{
          request->normalized_bounds.w, request->normalized_bounds.x,
          request->normalized_bounds.y, request->normalized_bounds.z};
    }

    publishState();
    response->success = true;
  }

  void onStopDetection(
      const std::shared_ptr<detection_interfaces::srv::StopDetection::Request>
          request,
      std::shared_ptr<detection_interfaces::srv::StopDetection::Response>
          response) {
    if (enabledDetections_.find(request->detection_type) ==
        enabledDetections_.end()) {
      // Detection not enabled for the requested DetectionType. Do nothing.
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
      // No detections are enabled. Do nothing.
      response->success = false;
      return;
    }

    enabledDetections_.clear();
    publishState();
    response->success = true;
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

  void onGetState(
      const std::shared_ptr<detection_interfaces::srv::GetState::Request>,
      std::shared_ptr<detection_interfaces::srv::GetState::Response> response) {
    response->state = std::move(*getStateMsg());
  }

  void onGetPositions(
      const std::shared_ptr<detection_interfaces::srv::GetPositions::Request>
          request,
      std::shared_ptr<detection_interfaces::srv::GetPositions::Response>
          response) {
    sensor_msgs::msg::Image::ConstSharedPtr image{nullptr};
    {
      std::lock_guard<std::mutex> lock(lastColorImageMutex_);
      image = lastColorImage_;
    }
    if (!image) {
      return;
    }

    auto rgbdAlignment{getOrCreateRgbdAlignment()};
    auto depthXyz{getDepthXyz(image)};
    if (!rgbdAlignment || !depthXyz) {
      RCLCPP_WARN(
          get_logger(),
          "[onGetPositions] RgbdAlignment or depth XYZ is not available");
      return;
    }
    // Wrap the depthXyz data pointer (no copy)
    cv::Mat depthXyzMat(depthXyz->height, depthXyz->width, CV_32FC3,
                        const_cast<uint8_t*>(depthXyz->data.data()),
                        depthXyz->step);

    for (const auto& p : request->normalized_pixel_coords) {
      // Clamp normalized coordinates to [0, 1]
      double nx{std::clamp(p.x, 0.0, 1.0)};
      double ny{std::clamp(p.y, 0.0, 1.0)};

      // Convert to pixel coordinates
      int x{static_cast<int>(std::round(nx * image->width))};
      int y{static_cast<int>(std::round(ny * image->height))};

      // Get 3D position
      auto positionOpt{
          rgbdAlignment->getPosition(cv::Point(x, y), depthXyzMat)};

      // Create position msg and add to response
      common_interfaces::msg::Vector3 positionMsg;
      if (positionOpt) {
        const auto& pos{*positionOpt};
        positionMsg.x = pos[0];
        positionMsg.y = pos[1];
        positionMsg.z = pos[2];
      } else {
        positionMsg.x = positionMsg.y = positionMsg.z = -1.0;
      }

      response->positions.push_back(positionMsg);
    }
  }

#pragma endregion

#pragma region State and notifs publishing

  detection_interfaces::msg::State::UniquePtr getStateMsg() {
    auto msg{std::make_unique<detection_interfaces::msg::State>()};
    std::vector<uint8_t> enabledDetectionTypes;
    for (const auto& [key, value] : enabledDetections_) {
      enabledDetectionTypes.push_back(key);
    }
    msg->enabled_detection_types = enabledDetectionTypes;
    msg->recording_video = videoRecordingTimer_ != nullptr;
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

  std::shared_ptr<RgbdAlignment> getOrCreateRgbdAlignment() {
    // We create RgbdAlignment lazily, since the necessary calibration params
    // come from various async sources:
    // - Intrinsics come from CameraInfo messages via topic subscription
    // - Extrinsics come from tf2

    if (rgbdAlignment_) {
      return rgbdAlignment_;
    }

    // Color camera intrinsics
    sensor_msgs::msg::CameraInfo::ConstSharedPtr colorCameraInfo;
    {
      std::lock_guard<std::mutex> lock(colorCameraInfoMutex_);
      if (!colorCameraInfo_) {
        return nullptr;
      }
      colorCameraInfo = colorCameraInfo_;
    }
    auto [colorCameraIntrinsicMatrix,
          colorCameraDistortionCoeffs]{getCameraMatrices(colorCameraInfo)};
    std::pair<int, int> colorFrameOffset{
        static_cast<int>(colorCameraInfo->roi.x_offset),
        static_cast<int>(colorCameraInfo->roi.y_offset)};

    // Depth camera intrinsics
    sensor_msgs::msg::CameraInfo::ConstSharedPtr depthCameraInfo;
    {
      std::lock_guard<std::mutex> lock(depthCameraInfoMutex_);
      if (!depthCameraInfo_) {
        return nullptr;
      }
      depthCameraInfo = depthCameraInfo_;
    }
    auto [depthCameraIntrinsicMatrix,
          depthCameraDistortionCoeffs]{getCameraMatrices(depthCameraInfo)};

    // World -> color camera extrinsics
    geometry_msgs::msg::TransformStamped xyzToColorTransform;
    geometry_msgs::msg::TransformStamped xyzToDepthTransform;
    try {
      xyzToColorTransform = tfBuffer_->lookupTransform("color_camera", "world",
                                                       tf2::TimePointZero);
      xyzToDepthTransform = tfBuffer_->lookupTransform("depth_camera", "world",
                                                       tf2::TimePointZero);
    } catch (const tf2::TransformException& ex) {
      return nullptr;
    }

    auto xyzToColorExtrinsicMatrixOpt{getTransformMatrix(xyzToColorTransform)};
    if (!xyzToColorExtrinsicMatrixOpt) {
      return nullptr;
    }
    auto xyzToColorExtrinsicMatrix{std::move(*xyzToColorExtrinsicMatrixOpt)};

    auto xyzToDepthExtrinsicMatrixOpt{getTransformMatrix(xyzToDepthTransform)};
    if (!xyzToDepthExtrinsicMatrixOpt) {
      return nullptr;
    }
    auto xyzToDepthExtrinsicMatrix{std::move(*xyzToDepthExtrinsicMatrixOpt)};

    rgbdAlignment_ = std::make_shared<RgbdAlignment>(
        colorCameraIntrinsicMatrix, colorCameraDistortionCoeffs,
        depthCameraIntrinsicMatrix, depthCameraDistortionCoeffs,
        xyzToColorExtrinsicMatrix, xyzToDepthExtrinsicMatrix, colorFrameOffset);
    return rgbdAlignment_;
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debugImagePublisher_;
  rclcpp::CallbackGroup::SharedPtr subscriberCallbackGroup_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      colorImageSubscriber_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      colorCameraInfoSubscriber_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depthXyzSubscriber_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      depthCameraInfoSubscriber_;
  std::shared_ptr<tf2_ros::TransformListener> tfListener_;
  std::unique_ptr<tf2_ros::Buffer> tfBuffer_;
  rclcpp::Publisher<detection_interfaces::msg::State>::SharedPtr
      statePublisher_;
  rclcpp::Publisher<detection_interfaces::msg::DetectionResult>::SharedPtr
      detectionsPublisher_;
  rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
      notificationsPublisher_;
  rclcpp::CallbackGroup::SharedPtr serviceCallbackGroup_;
  rclcpp::Service<detection_interfaces::srv::GetDetectionResult>::SharedPtr
      getDetectionService_;
  rclcpp::Service<detection_interfaces::srv::StartDetection>::SharedPtr
      startDetectionService_;
  rclcpp::Service<detection_interfaces::srv::StopDetection>::SharedPtr
      stopDetectionService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stopAllDetectionsService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr
      startRecordingVideoService_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stopRecordingVideoService_;
  rclcpp::Service<detection_interfaces::srv::GetState>::SharedPtr
      getStateService_;
  rclcpp::Service<detection_interfaces::srv::GetPositions>::SharedPtr
      getPositionsService_;

  // For color-depth matching: how close the stamps must be to accept a pair
  // TODO: reduce this when cameras are synced
  const rclcpp::Duration maxIntervalDuration_{
      rclcpp::Duration::from_seconds(0.030)};
  // For color-depth matching: how long to keep unmatched depth frames around
  const rclcpp::Duration keepDuration_{rclcpp::Duration::from_seconds(0.200)};

  std::unique_ptr<LaserDetector> laserDetector_;
  std::unique_ptr<RunnerDetector> runnerDetector_;
  std::unique_ptr<CircleDetector> circleDetector_;
  std::thread detectionThread_;
  std::atomic<bool> detectionStopSignal_{false};
  sensor_msgs::msg::CameraInfo::ConstSharedPtr colorCameraInfo_;
  std::mutex colorCameraInfoMutex_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr depthCameraInfo_;
  std::mutex depthCameraInfoMutex_;
  // Notifies the detection thread that a new image is available
  Event colorImageEvent_;
  sensor_msgs::msg::Image::ConstSharedPtr lastColorImage_;
  std::mutex lastColorImageMutex_;
  std::shared_ptr<RgbdAlignment> rgbdAlignment_;
  // DetectionType -> normalized [0, 1] rect bounds {min x, min y, width,
  // height}
  std::unordered_map<int, cv::Rect2d> enabledDetections_;
  std::mutex depthXyzQueueMutex_;
  std::deque<sensor_msgs::msg::Image::ConstSharedPtr> depthXyzQueue_;
  rclcpp::TimerBase::SharedPtr videoRecordingTimer_;
  cv::VideoWriter videoWriter_;
  std::mutex videoWriterMutex_;
  cv::Mat lastDebugImage_;
  std::mutex debugImageMutex_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(DetectionNode)