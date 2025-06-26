#include "runner_cutter_control/clients/camera_control_client.hpp"

CameraControlClient::CameraControlClient(rclcpp::Node& callerNode,
                                         const std::string& clientNodeName,
                                         int timeoutSecs)
    : node_{callerNode}, timeoutSecs_{timeoutSecs} {
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
  hasFramesClient_ = callerNode.create_client<common_interfaces::srv::GetBool>(
      servicePrefix + "/has_frames", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  getFrameClient_ =
      callerNode.create_client<camera_control_interfaces::srv::GetFrame>(
          servicePrefix + "/get_frame", rmw_qos_profile_services_default,
          clientCallbackGroup_);
  acquireSingleFrameClient_ =
      callerNode
          .create_client<camera_control_interfaces::srv::AcquireSingleFrame>(
              servicePrefix + "/acquire_single_frame",
              rmw_qos_profile_services_default, clientCallbackGroup_);
  setExposureClient_ =
      callerNode.create_client<camera_control_interfaces::srv::SetExposure>(
          servicePrefix + "/set_exposure", rmw_qos_profile_services_default,
          clientCallbackGroup_);
  autoExposureClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/auto_exposure", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  setGainClient_ =
      callerNode.create_client<camera_control_interfaces::srv::SetGain>(
          servicePrefix + "/set_gain", rmw_qos_profile_services_default,
          clientCallbackGroup_);
  autoGainClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/auto_gain", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  getDetectionClient_ =
      callerNode
          .create_client<camera_control_interfaces::srv::GetDetectionResult>(
              servicePrefix + "/get_detection",
              rmw_qos_profile_services_default, clientCallbackGroup_);
  startDetectionClient_ =
      callerNode.create_client<camera_control_interfaces::srv::StartDetection>(
          servicePrefix + "/start_detection", rmw_qos_profile_services_default,
          clientCallbackGroup_);
  stopDetectionClient_ =
      callerNode.create_client<camera_control_interfaces::srv::StopDetection>(
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
  setSaveDirectoryClient_ =
      callerNode
          .create_client<camera_control_interfaces::srv::SetSaveDirectory>(
              servicePrefix + "/set_save_directory",
              rmw_qos_profile_services_default, clientCallbackGroup_);
  getStateClient_ =
      callerNode.create_client<camera_control_interfaces::srv::GetState>(
          servicePrefix + "/get_state", rmw_qos_profile_services_default,
          clientCallbackGroup_);
  getPositionsClient_ =
      callerNode.create_client<camera_control_interfaces::srv::GetPositions>(
          servicePrefix + "/get_positions", rmw_qos_profile_services_default,
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

bool CameraControlClient::hasFrames() {
  auto request{std::make_shared<common_interfaces::srv::GetBool::Request>()};
  auto future{hasFramesClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->data;
}

std::optional<CameraControlClient::GetFrameResult>
CameraControlClient::getFrame() {
  auto request{
      std::make_shared<camera_control_interfaces::srv::GetFrame::Request>()};
  auto future{getFrameClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return std::nullopt;
  }

  auto result{future.get()};
  auto colorFrame =
      std::make_shared<sensor_msgs::msg::Image>(result->color_frame);
  auto depthFrame =
      std::make_shared<sensor_msgs::msg::Image>(result->depth_frame);
  return GetFrameResult{colorFrame, depthFrame};
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

bool CameraControlClient::setExposure(float exposureUs) {
  auto request{
      std::make_shared<camera_control_interfaces::srv::SetExposure::Request>()};
  request->exposure_us = exposureUs;
  auto future{setExposureClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool CameraControlClient::autoExposure() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto future{autoExposureClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool CameraControlClient::setGain(float gainDb) {
  auto request{
      std::make_shared<camera_control_interfaces::srv::SetGain::Request>()};
  request->gain_db = gainDb;
  auto future{setGainClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool CameraControlClient::autoGain() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto future{autoGainClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

camera_control_interfaces::msg::DetectionResult::SharedPtr
CameraControlClient::getDetection(uint8_t detectionType,
                                  bool waitForNextFrame) {
  auto request{std::make_shared<
      camera_control_interfaces::srv::GetDetectionResult::Request>()};
  request->detection_type = detectionType;
  request->wait_for_next_frame = waitForNextFrame;
  auto future{getDetectionClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return std::make_shared<camera_control_interfaces::msg::DetectionResult>();
  }

  auto result{future.get()};
  return std::make_shared<camera_control_interfaces::msg::DetectionResult>(
      result->result);
}

bool CameraControlClient::startDetection(
    uint8_t detectionType, const NormalizedPixelRect& normalizedBounds) {
  auto request{std::make_shared<
      camera_control_interfaces::srv::StartDetection::Request>()};
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

bool CameraControlClient::stopDetection(uint8_t detectionType) {
  auto request{std::make_shared<
      camera_control_interfaces::srv::StopDetection::Request>()};
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

bool CameraControlClient::stopAllDetections() {
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

bool CameraControlClient::startRecordingVideo() {
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

bool CameraControlClient::stopRecordingVideo() {
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
  auto request{std::make_shared<
      camera_control_interfaces::srv::SetSaveDirectory::Request>()};
  request->save_directory = saveDirectory;
  auto future{setSaveDirectoryClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
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

std::optional<std::vector<Position>> CameraControlClient::getPositions(
    const std::vector<NormalizedPixelCoord>& normalizedPixelCoords) {
  auto request{std::make_shared<
      camera_control_interfaces::srv::GetPositions::Request>()};
  for (const auto& coord : normalizedPixelCoords) {
    common_interfaces::msg::Vector2 coordMsg;
    coordMsg.x = coord.u;
    coordMsg.y = coord.v;
    request->normalized_pixel_coords.push_back(coordMsg);
  }
  auto future{getPositionsClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return std::nullopt;
  }

  auto result{future.get()};
  std::vector<Position> positions;
  for (const auto& position : result->positions) {
    positions.push_back(Position{static_cast<float>(position.x),
                                 static_cast<float>(position.y),
                                 static_cast<float>(position.z)});
  }
  return positions;
}