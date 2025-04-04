#include "runner_cutter_control_cpp/clients/laser_control_client.hpp"

#include "common_interfaces/msg/vector2.hpp"

LaserControlClient::LaserControlClient(rclcpp::Node& callerNode,
                                       const std::string& clientNodeName,
                                       int timeoutSecs)
    : node_{callerNode}, timeoutSecs_{timeoutSecs} {
  std::string servicePrefix{"/" + clientNodeName};
  pathPublisher_ =
      callerNode.create_publisher<laser_control_interfaces::msg::Path>(
          servicePrefix + "/path", 1);
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

bool LaserControlClient::setPoint(float x, float y) {
  auto msg{laser_control_interfaces::msg::Path()};
  msg.start.x = x;
  msg.start.y = y;
  msg.end.x = x;
  msg.end.y = y;
  msg.duration_ms = 0.0;
  msg.laser_on = true;
  pathPublisher_->publish(std::move(msg));

  return true;
}

bool LaserControlClient::clearPoint() {
  auto msg{laser_control_interfaces::msg::Path()};
  msg.laser_on = false;
  pathPublisher_->publish(std::move(msg));

  return true;
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