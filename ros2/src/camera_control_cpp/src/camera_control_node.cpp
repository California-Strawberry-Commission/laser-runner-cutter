#include <cv_bridge/cv_bridge.h>
#include <fmt/core.h>

#include <Eigen/Geometry>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <atomic>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <optional>
#include <rclcpp_components/register_node_macro.hpp>

#include "camera_control_cpp/camera/calibration.hpp"
#include "camera_control_cpp/camera/lucid_camera.hpp"
#include "camera_control_interfaces/msg/device_state.hpp"
#include "camera_control_interfaces/msg/state.hpp"
#include "camera_control_interfaces/srv/acquire_single_frame.hpp"
#include "camera_control_interfaces/srv/get_state.hpp"
#include "camera_control_interfaces/srv/start_device.hpp"
#include "camera_control_interfaces/srv/start_interval_capture.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rcl_interfaces/msg/log.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2_ros/static_transform_broadcaster.h"

namespace {

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
  auto [tritonIntrinsicMatrix,
        tritonDistCoeffs]{std::move(*tritonIntrinsicsOpt)};
  auto heliosIntrinsicsOpt{
      calibration::readIntrinsicsFile(heliosIntrinsicsPath.string())};
  if (!heliosIntrinsicsOpt) {
    throw std::runtime_error(fmt::format("Could not read calibration file {}",
                                         heliosIntrinsicsPath.string()));
  }
  auto [heliosIntrinsicMatrix,
        heliosDistCoeffs]{std::move(*heliosIntrinsicsOpt)};
  auto xyzToTritonExtrinsicsOpt{
      calibration::readExtrinsicsFile(xyzToTritonIntrinsicsPath.string())};
  if (!xyzToTritonExtrinsicsOpt) {
    throw std::runtime_error(fmt::format("Could not read calibration file {}",
                                         xyzToTritonIntrinsicsPath.string()));
  }
  auto xyzToTritonExtrinsicMatrix{std::move(*xyzToTritonExtrinsicsOpt)};
  auto xyzToHeliosExtrinsicsOpt{
      calibration::readExtrinsicsFile(xyzToHeliosIntrinsicsPath.string())};
  if (!xyzToHeliosExtrinsicsOpt) {
    throw std::runtime_error(fmt::format("Could not read calibration file {}",
                                         xyzToHeliosIntrinsicsPath.string()));
  }
  auto xyzToHeliosExtrinsicMatrix{std::move(*xyzToHeliosExtrinsicsOpt)};

  CalibrationParams result;
  result.tritonIntrinsicMatrix = tritonIntrinsicMatrix;
  result.tritonDistCoeffs = tritonDistCoeffs;
  result.heliosIntrinsicMatrix = heliosIntrinsicMatrix;
  result.heliosDistCoeffs = heliosDistCoeffs;
  result.xyzToTritonExtrinsicMatrix = xyzToTritonExtrinsicMatrix;
  result.xyzToHeliosExtrinsicMatrix = xyzToHeliosExtrinsicMatrix;

  return result;
}

sensor_msgs::msg::CameraInfo createCameraInfo(const cv::Mat& distCoeffs,
                                              const cv::Mat& intrinsicMatrix,
                                              const cv::Rect& roi) {
  if (intrinsicMatrix.rows != 3 || intrinsicMatrix.cols != 3) {
    throw std::runtime_error("Intrinsic matrix must be 3x3");
  }

  sensor_msgs::msg::CameraInfo cameraInfo;
  cameraInfo.distortion_model = "plumb_bob";

  // Distortion coeffs (D)
  cameraInfo.d.resize(distCoeffs.total());
  for (size_t i = 0; i < distCoeffs.total(); ++i) {
    cameraInfo.d[i] = distCoeffs.at<double>(static_cast<int>(i));
  }

  // Intrinsic matrix (K)
  cameraInfo.k[0] = intrinsicMatrix.at<double>(0, 0);  // fx
  cameraInfo.k[1] = intrinsicMatrix.at<double>(0, 1);  // skew
  cameraInfo.k[2] = intrinsicMatrix.at<double>(0, 2);  // cx
  cameraInfo.k[3] = intrinsicMatrix.at<double>(1, 0);
  cameraInfo.k[4] = intrinsicMatrix.at<double>(1, 1);  // fy
  cameraInfo.k[5] = intrinsicMatrix.at<double>(1, 2);  // cy
  cameraInfo.k[6] = intrinsicMatrix.at<double>(2, 0);
  cameraInfo.k[7] = intrinsicMatrix.at<double>(2, 1);
  cameraInfo.k[8] = intrinsicMatrix.at<double>(2, 2);

  cameraInfo.roi.width = std::max(0, roi.width);
  cameraInfo.roi.height = std::max(0, roi.height);
  cameraInfo.roi.x_offset = std::max(0, roi.x);
  cameraInfo.roi.y_offset = std::max(0, roi.y);

  return cameraInfo;
}

geometry_msgs::msg::TransformStamped createTransformStamped(
    const std::string& sourceFrame, const std::string& targetFrame,
    const cv::Mat& extrinsicMatrix) {
  if (extrinsicMatrix.rows != 4 || extrinsicMatrix.cols != 4) {
    throw std::runtime_error("Extrinsic matrix must be 4x4");
  }

  // Ensure double precision for safety
  cv::Matx44d T;
  extrinsicMatrix.convertTo(T, CV_64F);

  // Extract rotation (top-left 3x3) and translation (top-right 3x1)
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      R(r, c) = T(r, c);
    }
    t(r) = T(r, 3);
  }

  // Convert to quaternion
  Eigen::Quaterniond q(R);
  q.normalize();

  // Fill TransformStamped
  geometry_msgs::msg::TransformStamped transformStamped;
  transformStamped.header.frame_id = sourceFrame;
  transformStamped.child_frame_id = targetFrame;
  transformStamped.transform.translation.x = t.x();
  transformStamped.transform.translation.y = t.y();
  transformStamped.transform.translation.z = t.z();
  transformStamped.transform.rotation.x = q.x();
  transformStamped.transform.rotation.y = q.y();
  transformStamped.transform.rotation.z = q.z();
  transformStamped.transform.rotation.w = q.w();

  return transformStamped;
}

}  // namespace

class CameraControlNode : public rclcpp::Node {
 public:
  explicit CameraControlNode(const rclcpp::NodeOptions& options)
      : Node("camera_control_node", options) {
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
        "~/color/image_raw", rclcpp::SensorDataQoS());
    colorCameraInfoPublisher_ = create_publisher<sensor_msgs::msg::CameraInfo>(
        "~/color/camera_info", rclcpp::SensorDataQoS());
    depthXyzPublisher_ = create_publisher<sensor_msgs::msg::Image>(
        "~/depth/xyz", rclcpp::SensorDataQoS());
    depthIntensityPublisher_ = create_publisher<sensor_msgs::msg::Image>(
        "~/depth/intensity", rclcpp::SensorDataQoS());
    depthCameraInfoPublisher_ = create_publisher<sensor_msgs::msg::CameraInfo>(
        "~/depth/camera_info", rclcpp::SensorDataQoS());
    tfStaticBroadcaster_ =
        std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

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
    // TODO: be able to set ROI size and offset via params
    std::pair<int, int> colorRoiSize{2048, 1536};
    LucidCamera::StateChangeCallback stateChangeCallback =
        [this](LucidCamera::State) { publishState(); };
    camera_ = std::make_unique<LucidCamera>(
        /*colorCameraSerialNumber=*/std::nullopt,
        /*depthCameraSerialNumber=*/std::nullopt, colorRoiSize,
        stateChangeCallback);
    calibrationParams_ = readCalibrationParams(getParamCalibrationId());
    colorCameraInfo_ =
        createCameraInfo(calibrationParams_.tritonDistCoeffs,
                         calibrationParams_.tritonIntrinsicMatrix,
                         cv::Rect{(2048 - colorRoiSize.first) / 2,
                                  (2048 - colorRoiSize.second) / 2,
                                  colorRoiSize.first, colorRoiSize.second});
    depthCameraInfo_ = createCameraInfo(
        calibrationParams_.heliosDistCoeffs,
        calibrationParams_.heliosIntrinsicMatrix, cv::Rect{0, 0, 640, 480});

    // Publish extrinsics via tf2's StaticTransformBroadcaster once at startup
    auto worldToColorTransform{
        createTransformStamped("world", "color_camera",
                               calibrationParams_.xyzToTritonExtrinsicMatrix)};
    auto worldToDepthTransform{
        createTransformStamped("world", "depth_camera",
                               calibrationParams_.xyzToHeliosExtrinsicMatrix)};
    tfStaticBroadcaster_->sendTransform(worldToColorTransform);
    tfStaticBroadcaster_->sendTransform(worldToDepthTransform);

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

    LucidCamera::ColorCallback colorCallback{
        [this](sensor_msgs::msg::Image::UniquePtr colorImage) {
          if (!cameraStarted_ ||
              camera_->getState() != LucidCamera::State::STREAMING) {
            return;
          }

          {
            std::lock_guard<std::mutex> lock(lastColorImageMutex_);
            if (colorImage) {
              // TODO: eliminate this copy. It's only used for saving an image
              lastColorImage_ =
                  std::make_shared<sensor_msgs::msg::Image>(*colorImage);
            } else {
              lastColorImage_.reset();
            }
          }

          // Copy template
          auto cameraInfo{
              std::make_unique<sensor_msgs::msg::CameraInfo>(colorCameraInfo_)};
          // Ensure header/timing/size match the image exactly
          cameraInfo->header = colorImage->header;
          cameraInfo->width = colorImage->width;
          cameraInfo->height = colorImage->height;

          // Important: publish frames via zero-copy intra-process. Note that
          // the message needs to be a unique_ptr.
          colorImagePublisher_->publish(std::move(colorImage));
          colorCameraInfoPublisher_->publish(std::move(cameraInfo));
        }};

    LucidCamera::DepthCallback depthCallback{
        [this](sensor_msgs::msg::Image::UniquePtr depthXyz,
               sensor_msgs::msg::Image::UniquePtr depthIntensity) {
          if (!cameraStarted_ ||
              camera_->getState() != LucidCamera::State::STREAMING) {
            return;
          }

          // Copy template
          auto cameraInfo{
              std::make_unique<sensor_msgs::msg::CameraInfo>(depthCameraInfo_)};
          // Ensure header/timing/size match the image exactly
          cameraInfo->header = depthXyz->header;
          cameraInfo->width = depthXyz->width;
          cameraInfo->height = depthXyz->height;

          // Important: publish frames via zero-copy intra-process. Note that
          // the message needs to be a unique_ptr.
          depthXyzPublisher_->publish(std::move(depthXyz));
          depthIntensityPublisher_->publish(std::move(depthIntensity));
          depthCameraInfoPublisher_->publish(std::move(cameraInfo));
        }};

    camera_->start(static_cast<LucidCamera::CaptureMode>(request->capture_mode),
                   getParamExposureUs(), getParamGainDb(), colorCallback,
                   depthCallback);

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
    auto frameOpt{camera_->getNextFrame()};
    if (!frameOpt) {
      publishNotification("Failed to acquire frame",
                          rclcpp::Logger::Level::Error);
      return;
    }
    LucidCamera::Frame frame{std::move(*frameOpt)};

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
    sensor_msgs::msg::Image::SharedPtr colorImage;
    {
      std::lock_guard<std::mutex> lock(lastColorImageMutex_);
      colorImage = lastColorImage_;
    }
    if (!colorImage) {
      return std::nullopt;
    }

    // Create the save directory if it doesn't exist
    std::string saveDir{getParamSaveDir()};
    saveDir = expandUser(saveDir);
    std::filesystem::create_directories(saveDir);

    // Generate the image file name and path
    std::string filepath{fmt::format(
        "{}/{}.jpg", saveDir, formatRosTimestamp(colorImage->header.stamp))};

    // Demosaic color image (which is BayerRG8) and save the image
    cv::Mat raw(colorImage->height, colorImage->width, CV_8UC1,
                const_cast<uint8_t*>(colorImage->data.data()),
                colorImage->step);
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
    auto colorFrameSize{camera_->getColorFrameSize()};
    msg->color_width = colorFrameSize.first > 0 ? colorFrameSize.first : 0;
    msg->color_height = colorFrameSize.second > 0 ? colorFrameSize.second : 0;
    auto depthFrameSize{camera_->getDepthFrameSize()};
    msg->depth_width = depthFrameSize.first > 0 ? depthFrameSize.first : 0;
    msg->depth_height = depthFrameSize.second > 0 ? depthFrameSize.second : 0;
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
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr
      colorCameraInfoPublisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depthXyzPublisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr
      depthIntensityPublisher_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr
      depthCameraInfoPublisher_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tfStaticBroadcaster_;
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
  sensor_msgs::msg::Image::SharedPtr lastColorImage_;
  CalibrationParams calibrationParams_;
  sensor_msgs::msg::CameraInfo colorCameraInfo_;
  sensor_msgs::msg::CameraInfo depthCameraInfo_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(CameraControlNode)