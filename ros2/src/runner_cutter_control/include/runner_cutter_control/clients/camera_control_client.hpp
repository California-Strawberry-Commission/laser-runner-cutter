#pragma once

#include "camera_control_interfaces/msg/state.hpp"
#include "camera_control_interfaces/srv/acquire_single_frame.hpp"
#include "camera_control_interfaces/srv/get_state.hpp"
#include "camera_control_interfaces/srv/start_device.hpp"
#include "camera_control_interfaces/srv/start_interval_capture.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "std_srvs/srv/trigger.hpp"

class CameraControlClient {
 public:
  explicit CameraControlClient(rclcpp::Node& callerNode,
                               const std::string& clientNodeName,
                               int timeoutSecs = 3);
  ~CameraControlClient() = default;

  bool startDevice(uint8_t captureMode);
  bool closeDevice();
  std::optional<sensor_msgs::msg::CompressedImage::SharedPtr>
  acquireSingleFrame();
  float getExposure();
  bool setExposure(float exposureUs);
  bool autoExposure();
  float getGain();
  bool setGain(float gainDb);
  bool autoGain();
  bool saveImage();
  bool startIntervalCapture(float intervalSecs);
  bool stopIntervalCapture();
  bool setSaveDirectory(const std::string& saveDirectory);
  camera_control_interfaces::msg::State::SharedPtr getState();

 private:
  rclcpp::Node& node_;
  int timeoutSecs_{0};

  std::shared_ptr<rclcpp::AsyncParametersClient> parametersClient_;
  rclcpp::CallbackGroup::SharedPtr clientCallbackGroup_;
  rclcpp::Client<camera_control_interfaces::srv::StartDevice>::SharedPtr
      startDeviceClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr closeDeviceClient_;
  rclcpp::Client<camera_control_interfaces::srv::AcquireSingleFrame>::SharedPtr
      acquireSingleFrameClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr saveImageClient_;
  rclcpp::Client<camera_control_interfaces::srv::StartIntervalCapture>::
      SharedPtr startIntervalCaptureClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr stopIntervalCaptureClient_;
  rclcpp::Client<camera_control_interfaces::srv::GetState>::SharedPtr
      getStateClient_;
};