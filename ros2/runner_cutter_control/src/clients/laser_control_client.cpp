#include "runner_cutter_control/clients/laser_control_client.hpp"

#include "common_interfaces/msg/vector2.hpp"

LaserControlClient::LaserControlClient(rclcpp::Node& callerNode,
                                       const std::string& clientNodeName,
                                       int timeoutSecs)
    : node_{callerNode}, timeoutSecs_{timeoutSecs} {
  std::string servicePrefix{"/" + clientNodeName};
  parametersClient_ = std::make_shared<rclcpp::AsyncParametersClient>(
      &callerNode, clientNodeName);
  updatePathPublisher_ =
      callerNode.create_publisher<laser_control_interfaces::msg::PathUpdate>(
          servicePrefix + "/update_path", 1);
  clientCallbackGroup_ =
      callerNode.create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  startDeviceClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/start_device", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  closeDeviceClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/close_device", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  removePathClient_ =
      callerNode.create_client<laser_control_interfaces::srv::RemovePath>(
          servicePrefix + "/remove_path", rmw_qos_profile_services_default,
          clientCallbackGroup_);
  clearPathsClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/clear_paths", rmw_qos_profile_services_default,
      clientCallbackGroup_);
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

bool LaserControlClient::setColor(const LaserColor& color) {
  std::vector<double> colorVec{color.r, color.g, color.b, color.i};
  auto future{parametersClient_->set_parameters(
      {rclcpp::Parameter("color", colorVec)})};
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

bool LaserControlClient::setPoint(const LaserCoord& point) {
  auto msg{laser_control_interfaces::msg::PathUpdate()};
  msg.path_id = 1;
  msg.destination.x = point.x;
  msg.destination.y = point.y;
  updatePathPublisher_->publish(std::move(msg));

  return true;
}

bool LaserControlClient::clearPoint() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto future{clearPathsClient_->async_send_request(request)};
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