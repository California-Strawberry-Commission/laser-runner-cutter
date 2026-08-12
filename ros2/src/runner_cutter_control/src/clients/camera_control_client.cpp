#include "runner_cutter_control/clients/camera_control_client.hpp"

#include "runner_cutter_control/clients/service_client_utils.hpp"

CameraControlClient::CameraControlClient(rclcpp::Node& callerNode,
                                         const std::string& clientNodeName,
                                         int timeoutSecs)
    : node_{callerNode}, timeoutSecs_{timeoutSecs} {
  parametersClient_ = std::make_shared<rclcpp::AsyncParametersClient>(
      &callerNode, clientNodeName);
  std::string servicePrefix{"/" + clientNodeName};
  clientCallbackGroup_ =
      callerNode.create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  startDeviceClient_ =
      callerNode.create_client<camera_control_interfaces::srv::StartDevice>(
          servicePrefix + "/start_device", rmw_qos_profile_services_default,
          clientCallbackGroup_);
  closeDeviceClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/close_device", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  acquireSingleFrameClient_ =
      callerNode
          .create_client<camera_control_interfaces::srv::AcquireSingleFrame>(
              servicePrefix + "/acquire_single_frame",
              rmw_qos_profile_services_default, clientCallbackGroup_);
  saveImageClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/save_image", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  startIntervalCaptureClient_ =
      callerNode
          .create_client<camera_control_interfaces::srv::StartIntervalCapture>(
              servicePrefix + "/start_interval_capture",
              rmw_qos_profile_services_default, clientCallbackGroup_);
  stopIntervalCaptureClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/stop_interval_capture",
      rmw_qos_profile_services_default, clientCallbackGroup_);
  getStateClient_ =
      callerNode.create_client<camera_control_interfaces::srv::GetState>(
          servicePrefix + "/get_state", rmw_qos_profile_services_default,
          clientCallbackGroup_);
}

bool CameraControlClient::startDevice(uint8_t captureMode) {
  auto request{
      std::make_shared<camera_control_interfaces::srv::StartDevice::Request>()};
  request->capture_mode = captureMode;
  auto result{
      client_utils::callService<camera_control_interfaces::srv::StartDevice>(
          startDeviceClient_, request, timeoutSecs_, node_.get_logger())};
  return result && result->success;
}

bool CameraControlClient::closeDevice() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto result{client_utils::callService<std_srvs::srv::Trigger>(
      closeDeviceClient_, request, timeoutSecs_, node_.get_logger())};
  return result && result->success;
}

std::optional<sensor_msgs::msg::CompressedImage::SharedPtr>
CameraControlClient::acquireSingleFrame() {
  auto request{std::make_shared<
      camera_control_interfaces::srv::AcquireSingleFrame::Request>()};
  auto result{client_utils::callService<
      camera_control_interfaces::srv::AcquireSingleFrame>(
      acquireSingleFrameClient_, request, timeoutSecs_, node_.get_logger())};
  if (!result || !result->success) {
    return std::nullopt;
  }

  return std::make_shared<sensor_msgs::msg::CompressedImage>(
      result->preview_image);
}

float CameraControlClient::getExposure() {
  auto future{parametersClient_->get_parameters({"exposure_us"})};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Get parameter timed out.");
    return 0.0f;
  }

  const auto& parameter{future.get()[0]};
  return static_cast<float>(parameter.as_double());
}

bool CameraControlClient::setExposure(float exposureUs) {
  return client_utils::setParameters(
      parametersClient_, {rclcpp::Parameter("exposure_us", exposureUs)},
      timeoutSecs_, node_.get_logger());
}

bool CameraControlClient::autoExposure() { return setExposure(-1.0f); }

float CameraControlClient::getGain() {
  auto future{parametersClient_->get_parameters({"gain_db"})};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Get parameter timed out.");
    return 0.0f;
  }

  const auto& parameter{future.get()[0]};
  return static_cast<float>(parameter.as_double());
}

bool CameraControlClient::setGain(float gainDb) {
  return client_utils::setParameters(parametersClient_,
                                     {rclcpp::Parameter("gain_db", gainDb)},
                                     timeoutSecs_, node_.get_logger());
}

bool CameraControlClient::autoGain() { return setGain(-1.0f); }

bool CameraControlClient::saveImage() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto result{client_utils::callService<std_srvs::srv::Trigger>(
      saveImageClient_, request, timeoutSecs_, node_.get_logger())};
  return result && result->success;
}

bool CameraControlClient::startIntervalCapture(float intervalSecs) {
  auto request{std::make_shared<
      camera_control_interfaces::srv::StartIntervalCapture::Request>()};
  request->interval_secs = intervalSecs;
  auto result{client_utils::callService<
      camera_control_interfaces::srv::StartIntervalCapture>(
      startIntervalCaptureClient_, request, timeoutSecs_, node_.get_logger())};
  return result && result->success;
}

bool CameraControlClient::stopIntervalCapture() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto result{client_utils::callService<std_srvs::srv::Trigger>(
      stopIntervalCaptureClient_, request, timeoutSecs_, node_.get_logger())};
  return result && result->success;
}

bool CameraControlClient::setSaveDirectory(const std::string& saveDirectory) {
  return client_utils::setParameters(
      parametersClient_, {rclcpp::Parameter("save_dir", saveDirectory)},
      timeoutSecs_, node_.get_logger());
}

camera_control_interfaces::msg::State::SharedPtr
CameraControlClient::getState() {
  auto request{
      std::make_shared<camera_control_interfaces::srv::GetState::Request>()};
  auto result{
      client_utils::callService<camera_control_interfaces::srv::GetState>(
          getStateClient_, request, timeoutSecs_, node_.get_logger())};
  if (!result) {
    return std::make_shared<camera_control_interfaces::msg::State>();
  }

  return std::make_shared<camera_control_interfaces::msg::State>(result->state);
}
