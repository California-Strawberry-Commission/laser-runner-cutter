#include "runner_cutter_control/clients/camera_control_client.hpp"

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
  auto future{startDeviceClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool CameraControlClient::closeDevice() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto future{closeDeviceClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

std::optional<sensor_msgs::msg::CompressedImage::SharedPtr>
CameraControlClient::acquireSingleFrame() {
  auto request{std::make_shared<
      camera_control_interfaces::srv::AcquireSingleFrame::Request>()};
  auto future{acquireSingleFrameClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return std::nullopt;
  }

  auto result{future.get()};
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
  auto future{parametersClient_->set_parameters(
      {rclcpp::Parameter("exposure_us", exposureUs)})};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Set parameter timed out.");
    return false;
  }

  for (const auto& result : future.get()) {
    if (!result.successful) {
      RCLCPP_ERROR(node_.get_logger(), "Failed to set parameter: %s",
                   result.reason.c_str());
      return false;
    }
  }

  return true;
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
  auto future{parametersClient_->set_parameters(
      {rclcpp::Parameter("gain_db", gainDb)})};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Set parameter timed out.");
    return false;
  }

  for (const auto& result : future.get()) {
    if (!result.successful) {
      RCLCPP_ERROR(node_.get_logger(), "Failed to set parameter: %s",
                   result.reason.c_str());
      return false;
    }
  }

  return true;
}

bool CameraControlClient::autoGain() { return setGain(-1.0f); }

bool CameraControlClient::saveImage() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto future{saveImageClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool CameraControlClient::startIntervalCapture(float intervalSecs) {
  auto request{std::make_shared<
      camera_control_interfaces::srv::StartIntervalCapture::Request>()};
  request->interval_secs = intervalSecs;
  auto future{startIntervalCaptureClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool CameraControlClient::stopIntervalCapture() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto future{stopIntervalCaptureClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool CameraControlClient::setSaveDirectory(const std::string& saveDirectory) {
  auto future{parametersClient_->set_parameters(
      {rclcpp::Parameter("save_dir", saveDirectory)})};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Set parameter timed out.");
    return false;
  }

  for (const auto& result : future.get()) {
    if (!result.successful) {
      RCLCPP_ERROR(node_.get_logger(), "Failed to set parameter: %s",
                   result.reason.c_str());
      return false;
    }
  }

  return true;
}

camera_control_interfaces::msg::State::SharedPtr
CameraControlClient::getState() {
  auto request{
      std::make_shared<camera_control_interfaces::srv::GetState::Request>()};
  auto future{getStateClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return std::make_shared<camera_control_interfaces::msg::State>();
  }

  auto result{future.get()};
  return std::make_shared<camera_control_interfaces::msg::State>(result->state);
}
