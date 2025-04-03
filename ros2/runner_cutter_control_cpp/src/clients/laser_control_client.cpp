#include "runner_cutter_control_cpp/clients/laser_control_client.hpp"

#include "common_interfaces/msg/vector2.hpp"

LaserControlClient::LaserControlClient(rclcpp::Node& callerNode,
                                       const std::string& clientNodeName,
                                       int timeoutSecs)
    : node_{callerNode}, timeoutSecs_{timeoutSecs} {
  std::string servicePrefix{"/" + clientNodeName};
  clientCallbackGroup_ =
      callerNode.create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  startDeviceClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/start_device", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  closeDeviceClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/close_device", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  setColorClient_ =
      callerNode.create_client<laser_control_interfaces::srv::SetColor>(
          servicePrefix + "/set_color", rmw_qos_profile_services_default,
          clientCallbackGroup_);
  addPointClient_ =
      callerNode.create_client<laser_control_interfaces::srv::AddPoint>(
          servicePrefix + "/add_point", rmw_qos_profile_services_default,
          clientCallbackGroup_);
  setPointsClient_ =
      callerNode.create_client<laser_control_interfaces::srv::SetPoints>(
          servicePrefix + "/set_points", rmw_qos_profile_services_default,
          clientCallbackGroup_);
  removePointClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/remove_point", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  clearPointsClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/clear_points", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  setPlaybackParamsClient_ =
      callerNode
          .create_client<laser_control_interfaces::srv::SetPlaybackParams>(
              servicePrefix + "/set_playback_params",
              rmw_qos_profile_services_default, clientCallbackGroup_);
  playClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/play", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  stopClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/stop", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  getStateClient_ =
      callerNode.create_client<laser_control_interfaces::srv::GetState>(
          servicePrefix + "/get_state", rmw_qos_profile_services_default,
          clientCallbackGroup_);
}

bool LaserControlClient::startDevice() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto future{startDeviceClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool LaserControlClient::closeDevice() {
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

bool LaserControlClient::setColor(float r, float g, float b, float i) {
  auto request{
      std::make_shared<laser_control_interfaces::srv::SetColor::Request>()};
  request->r = r;
  request->g = g;
  request->b = b;
  request->i = i;
  auto future{setColorClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool LaserControlClient::addPoint(std::pair<float, float> point) {
  auto request{
      std::make_shared<laser_control_interfaces::srv::AddPoint::Request>()};
  request->point.x = point.first;
  request->point.y = point.second;
  auto future{addPointClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool LaserControlClient::setPoints(
    const std::vector<std::pair<float, float>>& points) {
  auto request{
      std::make_shared<laser_control_interfaces::srv::SetPoints::Request>()};
  for (const auto& point : points) {
    common_interfaces::msg::Vector2 pointMsg;
    pointMsg.x = point.first;
    pointMsg.y = point.second;
    request->points.push_back(pointMsg);
  }
  auto future{setPointsClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool LaserControlClient::removePoint() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto future{removePointClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool LaserControlClient::clearPoints() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto future{clearPointsClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool LaserControlClient::setPlaybackParams(int fps, int pps,
                                           float transitionDurationMs) {
  auto request{std::make_shared<
      laser_control_interfaces::srv::SetPlaybackParams::Request>()};
  request->fps = fps;
  request->pps = pps;
  request->transition_duration_ms = transitionDurationMs;
  auto future{setPlaybackParamsClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool LaserControlClient::play() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto future{playClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

bool LaserControlClient::stop() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto future{stopClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return false;
  }

  auto result{future.get()};
  return result->success;
}

laser_control_interfaces::msg::State::SharedPtr LaserControlClient::getState() {
  auto request{
      std::make_shared<laser_control_interfaces::srv::GetState::Request>()};
  auto future{getStateClient_->async_send_request(request)};
  if (future.wait_for(std::chrono::seconds(timeoutSecs_)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_.get_logger(), "Service call timed out.");
    return std::make_shared<laser_control_interfaces::msg::State>();
  }

  auto result{future.get()};
  return std::make_shared<laser_control_interfaces::msg::State>(result->state);
}