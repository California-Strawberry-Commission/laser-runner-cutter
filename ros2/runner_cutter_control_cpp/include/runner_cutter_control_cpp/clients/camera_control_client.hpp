#pragma once

#include "camera_control_interfaces/msg/detection_result.hpp"
#include "camera_control_interfaces/msg/state.hpp"
#include "camera_control_interfaces/srv/acquire_single_frame.hpp"
#include "camera_control_interfaces/srv/get_detection_result.hpp"
#include "camera_control_interfaces/srv/get_frame.hpp"
#include "camera_control_interfaces/srv/get_positions.hpp"
#include "camera_control_interfaces/srv/get_state.hpp"
#include "camera_control_interfaces/srv/set_exposure.hpp"
#include "camera_control_interfaces/srv/set_gain.hpp"
#include "camera_control_interfaces/srv/set_save_directory.hpp"
#include "camera_control_interfaces/srv/start_detection.hpp"
#include "camera_control_interfaces/srv/start_device.hpp"
#include "camera_control_interfaces/srv/start_interval_capture.hpp"
#include "camera_control_interfaces/srv/stop_detection.hpp"
#include "common_interfaces/srv/get_bool.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_srvs/srv/trigger.hpp"

class CameraControlClient {
 public:
  CameraControlClient(rclcpp::Node& callerNode,
                      const std::string& clientNodeName, int timeoutSecs = 3);

  bool startDevice(uint8_t captureMode);
  bool closeDevice();
  bool hasFrames();
  struct GetFrameResult {
    sensor_msgs::msg::Image::SharedPtr colorFrame;
    sensor_msgs::msg::Image::SharedPtr depthFrame;
  };
  std::optional<GetFrameResult> getFrame();
  std::optional<sensor_msgs::msg::CompressedImage::SharedPtr>
  acquireSingleFrame();
  bool setExposure(float exposureUs);
  bool autoExposure();
  bool setGain(float gainDb);
  bool autoGain();
  camera_control_interfaces::msg::DetectionResult::SharedPtr getDetection(
      uint8_t detectionType, bool waitForNextFrame = false);
  bool startDetection(uint8_t detectionType,
                      std::tuple<float, float, float, float> normalizedBounds =
                          {0.0f, 0.0f, 1.0f, 1.0f});
  bool stopDetection(uint8_t detectionType);
  bool stopAllDetections();
  bool startRecordingVideo();
  bool stopRecordingVideo();
  bool saveImage();
  bool startIntervalCapture(float intervalSecs);
  bool stopIntervalCapture();
  bool setSaveDirectory(const std::string& saveDirectory);
  camera_control_interfaces::msg::State::SharedPtr getState();
  std::optional<std::vector<std::tuple<float, float, float>>> getPositions(
      const std::vector<std::pair<float, float>>& normalizedPixelCoords);

 private:
  rclcpp::Node& node_;
  int timeoutSecs_{0};

  rclcpp::CallbackGroup::SharedPtr clientCallbackGroup_;
  rclcpp::Client<camera_control_interfaces::srv::StartDevice>::SharedPtr
      startDeviceClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr closeDeviceClient_;
  rclcpp::Client<common_interfaces::srv::GetBool>::SharedPtr hasFramesClient_;
  rclcpp::Client<camera_control_interfaces::srv::GetFrame>::SharedPtr
      getFrameClient_;
  rclcpp::Client<camera_control_interfaces::srv::AcquireSingleFrame>::SharedPtr
      acquireSingleFrameClient_;
  rclcpp::Client<camera_control_interfaces::srv::SetExposure>::SharedPtr
      setExposureClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr autoExposureClient_;
  rclcpp::Client<camera_control_interfaces::srv::SetGain>::SharedPtr
      setGainClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr autoGainClient_;
  rclcpp::Client<camera_control_interfaces::srv::GetDetectionResult>::SharedPtr
      getDetectionClient_;
  rclcpp::Client<camera_control_interfaces::srv::StartDetection>::SharedPtr
      startDetectionClient_;
  rclcpp::Client<camera_control_interfaces::srv::StopDetection>::SharedPtr
      stopDetectionClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr stopAllDetectionsClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr startRecordingVideoClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr stopRecordingVideoClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr saveImageClient_;
  rclcpp::Client<camera_control_interfaces::srv::StartIntervalCapture>::
      SharedPtr startIntervalCaptureClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr stopIntervalCaptureClient_;
  rclcpp::Client<camera_control_interfaces::srv::SetSaveDirectory>::SharedPtr
      setSaveDirectoryClient_;
  rclcpp::Client<camera_control_interfaces::srv::GetState>::SharedPtr
      getStateClient_;
  rclcpp::Client<camera_control_interfaces::srv::GetPositions>::SharedPtr
      getPositionsClient_;
};