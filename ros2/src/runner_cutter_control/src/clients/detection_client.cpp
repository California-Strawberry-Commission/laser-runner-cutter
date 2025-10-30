#include "runner_cutter_control/clients/detection_client.hpp"

DetectionClient::DetectionClient(rclcpp::Node& callerNode,
                                 const std::string& clientNodeName,
                                 int timeoutSecs)
    : node_{callerNode}, timeoutSecs_{timeoutSecs} {
  parametersClient_ = std::make_shared<rclcpp::AsyncParametersClient>(
      &callerNode, clientNodeName);
  std::string servicePrefix{"/" + clientNodeName};
  clientCallbackGroup_ =
      callerNode.create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  getDetectionClient_ =
      callerNode.create_client<detection_interfaces::srv::GetDetectionResult>(
          servicePrefix + "/get_detection", rmw_qos_profile_services_default,
          clientCallbackGroup_);
  startDetectionClient_ =
      callerNode.create_client<detection_interfaces::srv::StartDetection>(
          servicePrefix + "/start_detection", rmw_qos_profile_services_default,
          clientCallbackGroup_);
  stopDetectionClient_ =
      callerNode.create_client<detection_interfaces::srv::StopDetection>(
          servicePrefix + "/stop_detection", rmw_qos_profile_services_default,
          clientCallbackGroup_);
  stopAllDetectionsClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/stop_all_detections", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  startRecordingVideoClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/start_recording_video",
      rmw_qos_profile_services_default, clientCallbackGroup_);
  stopRecordingVideoClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/stop_recording_video", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  getStateClient_ =
      callerNode.create_client<detection_interfaces::srv::GetState>(
          servicePrefix + "/get_state", rmw_qos_profile_services_default,
          clientCallbackGroup_);
}

detection_interfaces::msg::DetectionResult::SharedPtr
DetectionClient::getDetection(uint8_t detectionType) {
  auto request{std::make_shared<
      detection_interfaces::srv::GetDetectionResult::Request>()};
  request->detection_type = detectionType;
  auto future{getDetectionClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return std::make_shared<detection_interfaces::msg::DetectionResult>();
  }

  auto result{future.get()};
  return std::make_shared<detection_interfaces::msg::DetectionResult>(
      result->result);
}

bool DetectionClient::startDetection(
    uint8_t detectionType, const NormalizedPixelRect& normalizedBounds) {
  auto request{
      std::make_shared<detection_interfaces::srv::StartDetection::Request>()};
  request->detection_type = detectionType;
  common_interfaces::msg::Vector4 normalizedBoundsMsg;
  normalizedBoundsMsg.w = normalizedBounds.u;
  normalizedBoundsMsg.x = normalizedBounds.v;
  normalizedBoundsMsg.y = normalizedBounds.width;
  normalizedBoundsMsg.z = normalizedBounds.height;
  request->normalized_bounds = normalizedBoundsMsg;
  auto future{startDetectionClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool DetectionClient::stopDetection(uint8_t detectionType) {
  auto request{
      std::make_shared<detection_interfaces::srv::StopDetection::Request>()};
  request->detection_type = detectionType;
  auto future{stopDetectionClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool DetectionClient::stopAllDetections() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto future{stopAllDetectionsClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool DetectionClient::startRecordingVideo() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto future{startRecordingVideoClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool DetectionClient::stopRecordingVideo() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto future{stopRecordingVideoClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool DetectionClient::setSaveDirectory(const std::string& saveDirectory) {
  auto future{parametersClient_->set_parameters(
      {rclcpp::Parameter("save_dir", saveDirectory)})};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
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

detection_interfaces::msg::State::SharedPtr DetectionClient::getState() {
  auto request{
      std::make_shared<detection_interfaces::srv::GetState::Request>()};
  auto future{getStateClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return std::make_shared<detection_interfaces::msg::State>();
  }

  auto result{future.get()};
  return std::make_shared<detection_interfaces::msg::State>(result->state);
}
